"""How much audio does each speech gate keep, and how much speech does it lose?

Builds a long simulated stretch of the day: cached conversations separated by
1 to 3 minutes of background noise with non-speech bumps (doors, dishes), then
scores the pendant energy gate and Silero VAD.

    python -m attrib.sim.gate_bench
"""
from __future__ import annotations

import pickle

import numpy as np

from ..audio import SR
from ..types import Region
from ..vad import EnergyGate, SileroVAD
from .pendant import pink_noise


def build_day(n_conv: int = 8, seed: int = 5):
    rng = np.random.default_rng(seed)
    parts, words, t = [], [], 0.0
    for i in range(n_conv):
        c = pickle.load(open(f"data/sim/conv0_{i}.pkl", "rb"))
        level = np.sqrt(np.mean(c.audio[: int(SR)] ** 2))  # this conversation's noise floor
        gap_n = int(rng.uniform(60, 180) * SR)
        gap = pink_noise(rng, gap_n) * level * rng.uniform(0.7, 1.5)
        for _ in range(int(rng.integers(3, 10))):
            # Non-speech bumps: short loud bursts of filtered noise.
            b0 = int(rng.integers(0, gap_n - SR))
            bl = int(rng.uniform(0.1, 0.8) * SR)
            gap[b0: b0 + bl] += pink_noise(rng, bl) * level * 10 ** (rng.uniform(10, 25) / 20) * np.hanning(bl)
        parts.append(gap)
        t += gap_n / SR
        parts.append(c.audio)
        words += [(w.start + t, w.end + t, w.speaker == "wearer") for w in c.words]
        t += len(c.audio) / SR
    return np.concatenate(parts).astype(np.float32), words


def kept(regions: list[Region], words, cover: float = 0.8):
    starts = np.array([r.start for r in regions])
    ends = np.array([r.end for r in regions])
    out = []
    for a, b, wearer in words:
        ov = np.clip(np.minimum(ends, b) - np.maximum(starts, a), 0, None).sum()
        out.append((ov >= cover * (b - a), wearer))
    return np.array(out)


def main():
    x, words = build_day()
    dur = len(x) / SR
    speech = sum(b - a for a, b, _ in words)
    print(f"day: {dur / 60:.1f} min, words {len(words)}, speech ~{speech / dur:.0%} of time")
    rows = []
    configs = [(f"energy {m} dB, Silero {th}", EnergyGate(margin_db=m), th)
               for m in (3, 6) for th in (0.4, 0.25, 0.15)]
    configs += [(f"no gate, Silero {th}", None, th) for th in (0.4, 0.25, 0.15)]
    for name, gate, th in configs:
        silero = SileroVAD(threshold=th)
        regs = gate(x) if gate else [Region(0, dur)]
        sent = sum(r.duration for r in regs)
        # Silero runs on the phone on whatever the pendant sent.
        mask = np.zeros(len(x), np.float32)
        for r in regs:
            mask[int(r.start * SR): int(r.end * SR)] = 1
        sv = silero(x * mask)
        up = sum(r.duration for r in sv)
        k1, k2 = kept(regs, words), kept(sv, words)
        rows.append((name, sent / dur, k1[k1[:, 1] == 1, 0].mean(), k1[k1[:, 1] == 0, 0].mean(),
                     up / dur, k2[k2[:, 1] == 1, 0].mean(), k2[k2[:, 1] == 0, 0].mean()))
    print("gate | radio on | wearer words kept | others kept | uploaded after Silero | wearer kept | others kept")
    for r in rows:
        print(f"{r[0]} | {r[1]:.0%} | {r[2]:.1%} | {r[3]:.1%} | {r[4]:.0%} | {r[5]:.1%} | {r[6]:.1%}")


if __name__ == "__main__":
    main()
