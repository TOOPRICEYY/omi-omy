"""Speech detection: a cheap permissive gate (pendant) and Silero VAD (phone).

The pendant gate only has to avoid dropping speech; Silero makes the real cut.
Speech-only audio is packed into one buffer for upload, with a TimeMap that
puts timestamps from the packed buffer back on the recording's own clock.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import SR
from .types import Region


def merge_regions(regions: list[Region], max_gap: float) -> list[Region]:
    out: list[Region] = []
    for r in sorted(regions, key=lambda r: r.start):
        if out and r.start - out[-1].end <= max_gap:
            out[-1].end = max(out[-1].end, r.end)
        else:
            out.append(Region(r.start, r.end))
    return out


class EnergyGate:
    """Pendant-side gate: smoothed speech-band energy against a noise floor.

    Costs a first-order high-pass, a square and a running average per sample,
    so it fits on the pendant's chip. The floor is the 20th percentile of the
    last few seconds, which ignores short bumps. `preroll` models the ring
    buffer that keeps word onsets, `hangover` keeps the radio on through short
    pauses.
    """

    def __init__(self, margin_db: float = 6.0, frame: float = 0.02, smooth: int = 5,
                 floor_window: float = 5.0, preroll: float = 0.5, hangover: float = 1.0):
        self.margin_db = margin_db
        self.frame = frame
        self.smooth = smooth
        self.floor_window = floor_window
        self.preroll = preroll
        self.hangover = hangover

    def __call__(self, x: np.ndarray, sr: int = SR) -> list[Region]:
        from scipy.signal import lfilter

        n = int(self.frame * sr)
        frames = len(x) // n
        if frames == 0:
            return []
        # One-pole high-pass near 200 Hz: drops rumble that carries no speech.
        a = np.exp(-2 * np.pi * 200 / sr)
        hp = lfilter([1, -1], [1, -a], x)
        p = np.mean(hp[: frames * n].reshape(frames, n) ** 2, axis=1)
        k = self.smooth
        p = np.convolve(np.pad(p, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")
        e = 10 * np.log10(p + 1e-12)
        w = int(self.floor_window / self.frame)
        hist = np.lib.stride_tricks.sliding_window_view(np.pad(e, (w - 1, 0), mode="edge"), w)
        floor = np.percentile(hist, 20, axis=1)
        active = e > floor + self.margin_db
        regions = []
        i = 0
        while i < frames:
            if active[i]:
                j = i
                while j < frames and active[j]:
                    j += 1
                regions.append(Region(max(0.0, i * self.frame - self.preroll),
                                      min(len(x) / sr, j * self.frame + self.hangover)))
                i = j
            else:
                i += 1
        return merge_regions(regions, 0.0)


class SileroVAD:
    def __init__(self, threshold: float = 0.15, pad: float = 0.3, merge_gap: float = 1.0,
                 min_speech: float = 0.25):
        from silero_vad import load_silero_vad

        self.model = load_silero_vad(onnx=True)
        self.threshold = threshold
        self.pad = pad
        self.merge_gap = merge_gap
        self.min_speech = min_speech

    def __call__(self, x: np.ndarray, sr: int = SR) -> list[Region]:
        import torch
        from silero_vad import get_speech_timestamps

        ts = get_speech_timestamps(torch.from_numpy(x), self.model, sampling_rate=sr,
                                   threshold=self.threshold,
                                   min_speech_duration_ms=int(self.min_speech * 1000),
                                   return_seconds=True)
        dur = len(x) / sr
        regions = [Region(max(0.0, t["start"] - self.pad), min(dur, t["end"] + self.pad)) for t in ts]
        return merge_regions(regions, self.merge_gap)


def split_conversations(regions: list[Region], gap: float = 120.0) -> list[list[Region]]:
    """Group speech regions into conversations separated by long silences."""
    convs: list[list[Region]] = []
    for r in regions:
        if convs and r.start - convs[-1][-1].end <= gap:
            convs[-1].append(r)
        else:
            convs.append([r])
    return convs


@dataclass
class TimeMap:
    """Maps times in a packed speech-only buffer back to the original clock."""

    packed_starts: np.ndarray
    orig_starts: np.ndarray
    durations: np.ndarray

    def to_orig(self, t: float) -> float:
        i = int(np.searchsorted(self.packed_starts, t, side="right") - 1)
        i = max(0, min(i, len(self.packed_starts) - 1))
        offset = min(max(t - self.packed_starts[i], 0.0), self.durations[i])
        return float(self.orig_starts[i] + offset)


def pack(x: np.ndarray, regions: list[Region], sr: int = SR, spacer: float = 0.5):
    """Concatenate speech regions with short silences between them.

    The spacer keeps the diarizer from fusing the end of one region with the
    start of the next. Only the packed audio is uploaded, so billing covers
    speech plus spacers.
    """
    gap = np.zeros(int(spacer * sr), np.float32)
    parts, ps, os_, ds = [], [], [], []
    t = 0.0
    for r in regions:
        seg = x[int(r.start * sr): int(r.end * sr)]
        if parts:
            parts.append(gap)
            t += spacer
        ps.append(t)
        os_.append(r.start)
        ds.append(len(seg) / sr)
        parts.append(seg)
        t += len(seg) / sr
    packed = np.concatenate(parts) if parts else np.zeros(0, np.float32)
    return packed, TimeMap(np.array(ps), np.array(os_), np.array(ds))
