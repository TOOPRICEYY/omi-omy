"""Who spoke when from MAI Transcribe 2's own diarization, plus voiceprints for naming.

MAI labels speakers per request ("0", "1", ...), and long recordings go up in
pieces, so the transcriber tags each word "p<piece>:<label>". Here:

1. Words become turns: consecutive words of one label with gaps under `gap` s.
2. Each piece-local speaker gets a "selective" voiceprint from only its clearest
   speech: its longest turns of at least `min_turn` s, with `trim` s cut off each
   edge (away from speaker changes), up to `max_pool` s in all.
3. Piece-local speakers are linked across pieces by average-linkage clustering of
   those voiceprints, never joining two speakers of the same piece.
4. One Cluster per linked speaker (same structure the local diarizer produces), so
   naming, labeling and learning work unchanged. `clean_time` says how much clear
   speech the voiceprint came from; too little and the speaker is not named.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .audio import SR, rms_db
from .diarize import Cluster
from .embed import l2norm
from .types import Turn, Word


def turns_from_words(words: list[Word], gap: float = 1.0) -> list[Turn]:
    out: list[Turn] = []
    last: dict[str, int] = {}
    for w in sorted((w for w in words if w.speaker is not None), key=lambda w: w.start):
        t = out[-1] if out else None
        if t is not None and t.speaker == w.speaker and w.start - t.end <= gap:
            t.end = max(t.end, w.end)
        else:
            out.append(Turn(w.start, w.end, w.speaker))
            last[w.speaker] = len(out) - 1
    return out


def clear_spans(ts: list[Turn], min_turn: float = 1.5, trim: float = 0.2,
                max_pool: float = 30.0) -> list[tuple[float, float]]:
    """The speaker's clearest stretches: longest turns first, edges trimmed."""
    spans, total = [], 0.0
    for t in sorted(ts, key=lambda t: -t.duration):
        if t.duration < min_turn:
            break
        a, b = t.start + trim, t.end - trim
        b = min(b, a + max_pool - total)
        spans.append((a, b))
        total += b - a
        if total >= max_pool - 1e-6:
            break
    return spans


def _cut(x: np.ndarray, spans, piece: float = 30.0) -> list[np.ndarray]:
    step = int(piece * SR)
    out = []
    for a, b in spans:
        seg = x[max(0, int(a * SR)): int(b * SR)]
        out += [seg[i: i + step] for i in range(0, len(seg), step) if len(seg[i: i + step])]
    return out


def link_pieces(x: np.ndarray, turns: list[Turn], embedder, threshold: float = 0.5,
                **sel) -> dict[str, str]:
    """piece-local label ("p0:1") -> conversation-wide label ("S0", ...)."""
    by: dict[str, list[Turn]] = defaultdict(list)
    for t in turns:
        by[t.speaker].append(t)
    labels = sorted(by, key=lambda s: min(t.start for t in by[s]))
    piece = {s: s.split(":", 1)[0] for s in labels}
    embs: dict[str, np.ndarray] = {}
    if len(set(piece.values())) > 1:
        for s in labels:
            parts = _cut(x, clear_spans(by[s], **sel))
            if parts:
                embs[s] = embedder(np.concatenate(parts))
    # Groups start as single speakers; merge the closest pair (average linkage)
    # while it is closer than the threshold and the two share no piece.
    groups = [[s] for s in labels]

    def sim(g, h):
        a = [embs[s] for s in g if s in embs]
        b = [embs[s] for s in h if s in embs]
        if not a or not b:
            return -1.0
        return float(np.mean([u @ v for u in a for v in b]))

    while True:
        best, pair = 1.0 - threshold, None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if {piece[s] for s in groups[i]} & {piece[s] for s in groups[j]}:
                    continue
                v = sim(groups[i], groups[j])
                if v >= best:
                    best, pair = v, (i, j)
        if pair is None:
            break
        i, j = pair
        groups[i] += groups.pop(j)
    groups.sort(key=lambda g: min(t.start for s in g for t in by[s]))
    return {s: f"S{k}" for k, g in enumerate(groups) for s in g}


def mai_clusters(x: np.ndarray, words: list[Word], embedder, gap: float = 1.0,
                 threshold: float = 0.5, min_seg: float = 0.8, **sel) -> tuple[list[Turn], list[Cluster]]:
    """Turns (conversation-wide labels) and one Cluster per speaker, for packed audio `x`
    and words carrying MAI's piece-local speaker labels. Word speakers are relabeled too."""
    raw = turns_from_words(words, gap)
    if not raw:
        return [], []
    glob = link_pieces(x, raw, embedder, threshold, **sel)
    for w in words:
        if w.speaker is not None:
            w.speaker = glob.get(w.speaker, w.speaker)
    turns = [Turn(t.start, t.end, glob[t.speaker]) for t in raw]
    # Adjacent turns of one person across a piece boundary stay separate; harmless.
    by: dict[str, list[Turn]] = defaultdict(list)
    for t in turns:
        by[t.speaker].append(t)
    out = []
    for label, ts in by.items():
        spans = clear_spans(ts, **sel)
        parts = _cut(x, spans)
        clean = sum(len(p) for p in parts) / SR
        if not parts:   # no clear stretch: a voiceprint from whatever there is, never named
            parts = _cut(x, [(t.start, t.end) for t in ts])
        parts = [p for p in parts if len(p) >= SR // 10] or parts
        centroid = embedder(np.concatenate(parts))
        seg_embs = [embedder(p) for p in sorted(parts, key=len, reverse=True)[:12] if len(p) / SR >= min_seg]
        loud = float(np.median([rms_db(p) for p in parts]))
        c = Cluster(label, ts, l2norm(centroid), seg_embs or [centroid], loud,
                    sum(t.duration for t in ts))
        c.clean_time = clean
        out.append(c)
    return turns, out
