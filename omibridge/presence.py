"""Is the wearer's voice in this recording? Decided before any MAI call.

pyannote segmentation-3.0 (ONNX, CPU, ~0.2 s per audio minute) marks speech; the speech is cut
into 3 s pieces, the 30 most clearly speech-like are voiceprinted with TitaNet and scored against
the wearer's gallery entries with the bridge's own normalised score (the same one naming uses).
The presence score is the mean of the best three pieces. Measured 2026-11-13 on 36 pendant
conversations with his enrolled voice: at 1.5 no conversation he speaks in was skipped and 39% of
minutes were; at 2.5, 48% of minutes but two faint conversations of his were skipped.
"""
from __future__ import annotations

import logging

import numpy as np

from attrib.audio import SR
from attrib.identify import Gallery, Identifier, IdentifyConfig

log = logging.getLogger("omibridge.presence")
PYANNOTE = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"


def wearer_entries(gallery: Gallery, wearer: str | None) -> list[str]:
    """Gallery names that are the wearer (case-insensitive: "Me" and "ME" are one person)."""
    if not wearer:
        return []
    return [p for p in gallery.people if p.strip().lower() == wearer.strip().lower()]


class Presence:
    def __init__(self, models_dir, embedder, cohort: np.ndarray, window_s: float = 3.0, top: int = 30):
        from attrib.chunkdiar import SegmentationModel

        self.seg = SegmentationModel(str(models_dir / PYANNOTE), threads=2)
        self.embedder = embedder
        self.cohort = cohort
        self.window_s = window_s
        self.top = top

    def speech_prob(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """Per-frame speech probability (1 - P(no speaker)) and the frame length in seconds."""
        W = self.seg.window
        k = max(1, int(np.ceil(len(x) / W)))
        pad = np.zeros(k * W, np.float32)
        pad[: len(x)] = x
        y = np.concatenate([self.seg.scores(pad[i * W:(i + 1) * W][None]) for i in range(k)])
        p = 1 - np.exp(y[..., 0]).reshape(-1)
        return p, W / y.shape[1] / SR

    def pieces(self, x: np.ndarray) -> list[np.ndarray]:
        p, fs = self.speech_prob(x)
        on = p >= 0.5
        out, i = [], 0
        while i < len(on):
            if not on[i]:
                i += 1
                continue
            j = i
            while j < len(on) and on[j]:
                j += 1
            t, b = i * fs, min(j * fs, len(x) / SR)
            while t + 1.5 <= b:
                e = min(b, t + self.window_s)
                out.append((float(p[int(t / fs): max(int(t / fs) + 1, int(e / fs))].mean()), t, e))
                t = e
            i = j
        out.sort(key=lambda w: -w[0])
        return [x[int(a * SR): int(b * SR)] for _, a, b in out[: self.top]]

    def score(self, x: np.ndarray, gallery: Gallery, wearer: str | None) -> float | None:
        """Presence score, or None when no wearer is enrolled (then nothing may be skipped)."""
        names = wearer_entries(gallery, wearer)
        if not names:
            return None
        g = Gallery(max_per_person=64)
        for n in names:
            for e in gallery.people[n]:
                g.add("wearer", e)
        ident = Identifier(g, self.cohort, IdentifyConfig())
        scores = []
        for piece in self.pieces(x):
            e = self.embedder(piece)
            scores.append(ident.score(e / (np.linalg.norm(e) + 1e-9), "wearer"))
        if not scores:
            return float("-inf")
        return float(np.mean(sorted(scores, reverse=True)[:3]))
