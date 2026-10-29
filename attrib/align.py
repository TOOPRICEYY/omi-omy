from __future__ import annotations

from .types import Turn, Word


def assign_words(words: list[Word], turns: list[Turn], max_snap: float = 0.5) -> list[Word]:
    """Give each word the speaker whose turn overlaps it most.

    A word that falls in no turn snaps to the nearest turn within `max_snap`
    seconds, otherwise it stays unattributed.
    """
    ts = sorted(turns, key=lambda t: t.start)
    out = []
    for w in words:
        best, best_ov = None, 0.0
        for t in ts:
            if t.start > w.end:
                break
            ov = min(w.end, t.end) - max(w.start, t.start)
            if ov > best_ov:
                best, best_ov = t, ov
        if best is None and ts:
            near = min(ts, key=lambda t: min(abs(w.mid - t.start), abs(w.mid - t.end)))
            if min(abs(w.mid - near.start), abs(w.mid - near.end)) <= max_snap:
                best = near
        out.append(Word(w.text, w.start, w.end, best.speaker if best else None))
    return out
