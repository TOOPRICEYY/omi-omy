"""Window-level re-identification inside turns.

The segmentation model can fuse two distant, similar-sounding people into one
local speaker, and no clustering can undo that. For enrolled people we can:
slide a short window through each turn, score it against the gallery, and
re-split the turn where the best-matching person changes. A switch penalty
(Viterbi smoothing) stops one noisy window from flipping the label.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import SR
from .embed import Embedder
from .identify import Identifier, is_unknown
from .types import Turn


@dataclass
class RefineConfig:
    window: float = 1.5
    hop: float = 0.75
    min_turn: float = 3.0         # shorter turns are left alone
    switch_penalty: float = 2.0   # in normalized score units
    keep_bonus: float = 0.5       # small preference for the turn's own label


class WindowRefiner:
    def __init__(self, embedder: Embedder, identifier: Identifier, config: RefineConfig | None = None):
        self.embedder = embedder
        self.identifier = identifier
        self.cfg = config or RefineConfig()

    def refine(self, x: np.ndarray, turns: list[Turn], candidates: list[str] | None = None,
               priors_bonus: dict[str, float] | None = None) -> list[Turn]:
        cfg = self.cfg
        people = candidates or list(self.identifier.gallery.people)
        thr = self.identifier.cfg.threshold
        out: list[Turn] = []
        for t in turns:
            if t.duration < cfg.min_turn or not people:
                out.append(t)
                continue
            starts = np.arange(t.start, t.end - cfg.window + 1e-6, cfg.hop)
            # Labels: each candidate person, plus "keep" (the turn's label).
            labels = people + ["__keep__"]
            E = np.zeros((len(starts), len(labels)))
            for w, s in enumerate(starts):
                v = self.embedder(x[int(s * SR): int((s + cfg.window) * SR)])
                for j, p in enumerate(people):
                    E[w, j] = self.identifier.score(v, p) + (priors_bonus or {}).get(p, 0.0)
                own = E[w, people.index(t.speaker)] if t.speaker in people else thr
                E[w, -1] = own + cfg.keep_bonus
            path = _viterbi(E, cfg.switch_penalty)
            # Windows that never beat the threshold for a person keep the label.
            seq = []
            for w, j in enumerate(path):
                lab = labels[j]
                if lab == "__keep__" or E[w, j] < thr:
                    lab = t.speaker
                seq.append(lab)
            out.extend(_spans(t, starts, seq, cfg))
        return out


def _viterbi(E: np.ndarray, penalty: float) -> list[int]:
    n, k = E.shape
    score = E[0].copy()
    back = np.zeros((n, k), int)
    for i in range(1, n):
        trans = score[:, None] - penalty * (1 - np.eye(k))
        back[i] = trans.argmax(0)
        score = trans.max(0) + E[i]
    path = [int(score.argmax())]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i, path[-1]]))
    return path[::-1]


def _spans(t: Turn, starts: np.ndarray, seq: list[str], cfg: RefineConfig) -> list[Turn]:
    out = []
    cur, s0 = seq[0], t.start
    for w in range(1, len(seq)):
        if seq[w] != cur:
            # Put the boundary in the middle of the overlap of the two windows.
            b = starts[w] + (cfg.window - cfg.hop) / 2
            out.append(Turn(s0, b, cur, t.confidence, t.cluster))
            cur, s0 = seq[w], b
    out.append(Turn(s0, t.end, cur, t.confidence, t.cluster))
    return out


__all__ = ["RefineConfig", "WindowRefiner", "is_unknown"]
