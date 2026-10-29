"""Open-set speaker identification against enrolled voiceprints.

Each person keeps a gallery of voiceprints from different conditions instead of
one average. Scores are normalized against a cohort of unrelated voices
(adaptive s-norm) so one threshold works in quiet and noisy rooms. Context
priors (calendar, who you usually talk to) and the wearer's loudness are added
on top, and anything below threshold stays unknown.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .diarize import Cluster
from .embed import l2norm
from .types import UNKNOWN


def is_unknown(label: str) -> bool:
    return label.startswith(UNKNOWN)


@dataclass
class Gallery:
    max_per_person: int = 20
    people: dict[str, list[np.ndarray]] = field(default_factory=dict)

    def add(self, person: str, emb: np.ndarray) -> None:
        g = self.people.setdefault(person, [])
        g.append(l2norm(emb))
        if len(g) > self.max_per_person:
            # Drop the most redundant sample to keep the gallery diverse.
            m = np.stack(g)
            sim = m @ m.T
            np.fill_diagonal(sim, 0)
            g.pop(int(sim.mean(axis=1).argmax()))

    def matrix(self, person: str) -> np.ndarray:
        return np.stack(self.people[person])

    def save(self, path: str) -> None:
        np.savez(path, **{p: np.stack(g) for p, g in self.people.items()})

    @classmethod
    def load(cls, path: str, max_per_person: int = 20) -> "Gallery":
        d = np.load(path)
        return cls(max_per_person, {p: list(d[p]) for p in d.files})


@dataclass
class IdentifyConfig:
    top_k: int = 3                 # gallery samples averaged per score
    cohort_top: int = 50           # cohort scores used for normalization
    use_snorm: bool = True
    threshold: float = 2.5         # on the normalized score (raw cosine if snorm is off)
    acoustic_floor: float = 1.0    # the voice alone must reach this; priors can't name a stranger
    prior_weight: float = 1.0      # per unit of log-odds of the prior
    wearer: str | None = None
    wearer_weight: float = 1.5     # bonus at +6 dB louder than the others
    merge_split_clusters: bool = True
    confidence_scale: float = 1.5  # normalized score units per logit


@dataclass
class Match:
    person: str
    score: float
    confidence: float
    runner_up: float


class Identifier:
    def __init__(self, gallery: Gallery, cohort: np.ndarray, config: IdentifyConfig | None = None):
        self.gallery = gallery
        self.cohort = l2norm(cohort)
        self.cfg = config or IdentifyConfig()
        self._sample_stats: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}

    def _stats(self, v: np.ndarray) -> tuple[float, float]:
        s = np.sort(self.cohort @ v)[-self.cfg.cohort_top:]
        return float(s.mean()), float(s.std() + 1e-6)

    def _samples(self, p: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gallery matrix plus each sample's own cohort mean and spread.

        Normalizing per sample matters once the gallery holds voiceprints from
        real conversations: a far-field sample that resembles every distant
        voice gets a high cohort mean, so it can't win on channel alone.
        """
        G = self.gallery.matrix(p)
        cached = self._sample_stats.get(p)
        if cached is None or cached[2] != len(G):
            top = np.sort(G @ self.cohort.T, axis=1)[:, -self.cfg.cohort_top:]
            self._sample_stats[p] = (top.mean(1), top.std(1) + 1e-6, len(G))
        mu, sd, _ = self._sample_stats[p]
        return G, mu, sd

    def score(self, emb: np.ndarray, person: str) -> float:
        G, mu, sd = self._samples(person)
        raw = G @ emb
        if not self.cfg.use_snorm:
            return float(np.sort(raw)[-self.cfg.top_k:].mean())
        mc, sc = self._stats(emb)
        norm = 0.5 * ((raw - mc) / sc + (raw - mu) / sd)
        return float(np.sort(norm)[-self.cfg.top_k:].mean())

    def identify(self, clusters: list[Cluster], priors: dict[str, float] | None = None,
                 base_prior: float = 0.5) -> dict[str, Match]:
        people = list(self.gallery.people)
        if not clusters:
            return {}
        if not people:
            return {c.label: Match(UNKNOWN, 0.0, 0.0, 0.0) for c in clusters}
        cfg = self.cfg
        A = np.array([[self.score(c.centroid, p) for p in people] for c in clusters])
        S = A.copy()
        if priors:
            for j, p in enumerate(people):
                if p in priors:
                    pr = min(max(priors[p], 1e-3), 1 - 1e-3)
                    S[:, j] += cfg.prior_weight * (np.log(pr / (1 - pr)) - np.log(base_prior / (1 - base_prior)))
        if cfg.wearer in people and len(clusters) > 1:
            j = people.index(cfg.wearer)
            loud = np.array([c.loudness_db for c in clusters])
            for i in range(len(clusters)):
                delta = loud[i] - np.delete(loud, i).max()
                S[i, j] += cfg.wearer_weight * np.tanh(delta / 6.0)
        # One person per cluster, best total score.
        rows, cols = linear_sum_assignment(-S)
        out: dict[str, Match] = {}
        for i, c in enumerate(clusters):
            order = np.sort(S[i])[::-1]
            runner = float(order[1]) if len(order) > 1 else -np.inf
            j = cols[list(rows).index(i)] if i in rows else None
            if j is not None and S[i, j] >= cfg.threshold and A[i, j] >= cfg.acoustic_floor:
                conf = _sigmoid((S[i, j] - cfg.threshold) / cfg.confidence_scale) * \
                    _sigmoid((S[i, j] - runner) / cfg.confidence_scale)
                out[c.label] = Match(people[j], float(S[i, j]), conf, runner)
            else:
                best = float(order[0])
                out[c.label] = Match(UNKNOWN, best, _sigmoid((cfg.threshold - best) / cfg.confidence_scale), runner)
        if cfg.merge_split_clusters:
            # Diarization can split one voice into two clusters. A cluster that
            # lost the assignment but clears the threshold for an already-taken
            # person joins that person.
            for i, c in enumerate(clusters):
                m = out[c.label]
                if m.person == UNKNOWN:
                    j = int(S[i].argmax())
                    if S[i, j] >= cfg.threshold + 1.0 and A[i, j] >= cfg.acoustic_floor:
                        out[c.label] = Match(people[j], float(S[i, j]),
                                             _sigmoid((S[i, j] - cfg.threshold) / cfg.confidence_scale), m.runner_up)
        return out


def _sigmoid(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))


class UnknownTracker:
    """Clusters unknown voices across conversations to surface recurring people."""

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self.voices: list[dict] = []

    def observe(self, emb: np.ndarray, conversation_id: str) -> int:
        emb = l2norm(emb)
        for i, v in enumerate(self.voices):
            if float(v["centroid"] @ emb) >= self.threshold:
                v["embs"].append(emb)
                v["centroid"] = l2norm(np.mean(v["embs"], axis=0))
                v["conversations"].add(conversation_id)
                return i
        self.voices.append({"centroid": emb, "embs": [emb], "conversations": {conversation_id}})
        return len(self.voices) - 1

    def recurring(self, min_conversations: int = 3) -> list[int]:
        return [i for i, v in enumerate(self.voices) if len(v["conversations"]) >= min_conversations]


def final_labels(clusters: list[Cluster], matches: dict[str, Match],
                 unknown_merge: float = 0.6) -> dict[str, str]:
    """Name every cluster: matched person, or a shared label per unknown voice.

    Diarization is tuned to over-split (a wrong merge can't be undone, a split
    can), so identification does the merging for known people. Unknown
    clusters are merged here when their pooled voiceprints agree.
    """
    labels = {c.label: matches[c.label].person for c in clusters}
    unk = sorted([c for c in clusters if labels[c.label] == UNKNOWN], key=lambda c: -c.speech_time)
    groups: list[tuple[np.ndarray, float, list[str]]] = []
    for c in unk:
        for gi, (cent, w, members) in enumerate(groups):
            if float(cent @ c.centroid) >= unknown_merge:
                new = l2norm(cent * w + c.centroid * c.speech_time)
                groups[gi] = (new, w + c.speech_time, members + [c.label])
                break
        else:
            groups.append((c.centroid, c.speech_time, [c.label]))
    for n, (_, _, members) in enumerate(groups):
        for m in members:
            labels[m] = f"{UNKNOWN}{n + 1}"
    return labels
