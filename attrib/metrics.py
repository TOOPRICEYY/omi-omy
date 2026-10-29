"""DER, identification error, and word-level speaker error."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .types import UNKNOWN, Turn, Word

STEP = 0.01  # 10 ms frames


def _frames(turns: list[Turn], labels: list[str], n: int) -> np.ndarray:
    m = np.zeros((n, len(labels)), bool)
    idx = {l: i for i, l in enumerate(labels)}
    for t in turns:
        m[int(round(t.start / STEP)): int(round(t.end / STEP)), idx[t.speaker]] = True
    return m


def _error(ref: list[Turn], hyp: list[Turn], mapping: dict[str, str] | None) -> dict:
    """Frame-level error. With mapping=None the best one-to-one map is used (DER);
    with a mapping (hyp label -> ref label, identity for named output) it is
    an identification error rate."""
    end = max([t.end for t in ref + hyp] + [0.0])
    n = int(np.ceil(end / STEP)) + 1
    rl = sorted({t.speaker for t in ref})
    hl = sorted({t.speaker for t in hyp})
    R, H = _frames(ref, rl, n), _frames(hyp, hl, n)
    nr, nh = R.sum(1), H.sum(1)
    total = nr.sum()
    miss = np.maximum(nr - nh, 0).sum()
    fa = np.maximum(nh - nr, 0).sum()
    co = (R.astype(np.int32).T @ H.astype(np.int32))  # ref x hyp co-occurrence
    if mapping is None:
        if len(rl) and len(hl):
            r, c = linear_sum_assignment(-co)
            correct_pairs = co[r, c].sum()
        else:
            correct_pairs = 0
    else:
        correct_pairs = sum(co[rl.index(mapping[h]), j] for j, h in enumerate(hl)
                            if mapping.get(h) in rl)
    correct = np.minimum(nr, nh).sum()
    conf = correct - correct_pairs
    return {"miss": miss / total, "fa": fa / total, "confusion": conf / total,
            "total": (miss + fa + conf) / total, "ref_seconds": total * STEP}


def der(ref: list[Turn], hyp: list[Turn]) -> dict:
    return _error(ref, hyp, None)


def identification_error(ref: list[Turn], hyp: list[Turn]) -> dict:
    """Like DER, but names must match exactly; 'unknown' never matches."""
    names = {t.speaker for t in hyp}
    return _error(ref, hyp, {h: h for h in names if not h.startswith(UNKNOWN)})


def word_speaker_error(ref_words: list[Word], hyp_words: list[Word],
                       enrolled: set[str] | None = None) -> float:
    """Share of words attributed to the wrong person.

    Enrolled people must be named correctly. A stranger's word counts as right
    when it is labeled with any unknown label, since nobody can name them yet.
    """
    pairs = list(zip(ref_words, hyp_words))
    if not pairs:
        return 0.0
    names = {h.speaker for h in hyp_words if h.speaker and not h.speaker.startswith(UNKNOWN)}
    known = {r.speaker for r in ref_words} & names if enrolled is None else enrolled

    def ok(r, h):
        if r.speaker in known:
            return h.speaker == r.speaker
        return bool(h.speaker) and h.speaker.startswith(UNKNOWN)
    return sum(not ok(r, h) for r, h in pairs) / len(pairs)


def false_accept_rate(ref: list[Turn], hyp: list[Turn], enrolled: set[str]) -> float:
    """Share of strangers' speech time that got labeled as an enrolled person."""
    end = max([t.end for t in ref + hyp] + [0.0])
    n = int(np.ceil(end / STEP)) + 1
    stranger = np.zeros(n, bool)
    named = np.zeros(n, bool)
    for t in ref:
        if t.speaker not in enrolled:
            stranger[int(t.start / STEP): int(t.end / STEP)] = True
    for t in hyp:
        if t.speaker in enrolled:
            named[int(t.start / STEP): int(t.end / STEP)] = True
    return float((stranger & named).sum() / max(stranger.sum(), 1))
