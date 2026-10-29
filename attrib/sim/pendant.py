"""Simulated pendant recordings with ground truth.

Voices come from a 904-speaker VITS model (LibriTTS-R voices). The wearer is
near-field and loud; everyone else is 1 to 3 m away in a reverberant room,
with background noise. This is a stand-in until real Omi recordings exist:
synthetic voices are cleaner and less varied than people, so absolute numbers
will be optimistic, but relative comparisons between pipeline variants hold
up better.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from ..audio import SR, resample
from ..types import Region, Turn, Word
from .text import BACKCHANNELS, utterance

TTS_DIR = "models/vits-piper-en_US-libritts_r-medium/"


@lru_cache(maxsize=1)
def _tts():
    import sherpa_onnx as so

    cfg = so.OfflineTtsConfig(model=so.OfflineTtsModelConfig(
        vits=so.OfflineTtsVitsModelConfig(model=TTS_DIR + "en_US-libritts_r-medium.onnx",
                                          tokens=TTS_DIR + "tokens.txt",
                                          data_dir=TTS_DIR + "espeak-ng-data"),
        num_threads=4))
    return so.OfflineTts(cfg)


def speak(text: str, sid: int, speed: float = 1.0) -> np.ndarray:
    a = _tts().generate(text, sid=sid, speed=speed)
    x = resample(np.array(a.samples, np.float32), a.sample_rate, SR)
    # Trim the model's leading/trailing silence so word timing is honest.
    e = np.abs(x) > 0.01
    if e.any():
        i, j = np.argmax(e), len(e) - np.argmax(e[::-1])
        x = x[max(0, i - 160): j + 160]
    return x / (np.abs(x).max() + 1e-9) * 0.5


@dataclass
class Acoustics:
    rir: np.ndarray
    gain_db: float


def room_rir(rng: np.random.Generator, distance: float, rt60: float) -> np.ndarray:
    import pyroomacoustics as pra

    dims = rng.uniform([4, 3.5, 2.6], [8, 7, 3.2])
    e_abs, max_order = pra.inverse_sabine(rt60, dims)
    room = pra.ShoeBox(dims, fs=SR, materials=pra.Material(e_abs), max_order=min(max_order, 12))
    mic = dims / 2 + np.array([0, 0, -0.3])
    ang = rng.uniform(0, 2 * np.pi)
    src = mic + np.array([distance * np.cos(ang), distance * np.sin(ang), 0.3])
    src = np.clip(src, 0.2, dims - 0.2)
    room.add_source(src)
    room.add_microphone(mic)
    room.compute_rir()
    h = np.array(room.rir[0][0], np.float32)
    return h / (np.abs(h).max() + 1e-9)


def pink_noise(rng: np.random.Generator, n: int) -> np.ndarray:
    w = rng.standard_normal(n)
    f = np.fft.rfft(w)
    f /= np.sqrt(np.arange(1, len(f) + 1))
    x = np.fft.irfft(f, n)
    return (x / (x.std() + 1e-9)).astype(np.float32)


def apply(x: np.ndarray, ac: Acoustics) -> np.ndarray:
    from scipy.signal import fftconvolve

    y = fftconvolve(x, ac.rir)[: len(x) + len(ac.rir) // 4]
    y = y / (np.abs(y).max() + 1e-9) * 0.5
    return (y * 10 ** (ac.gain_db / 20)).astype(np.float32)


@dataclass
class Conversation:
    audio: np.ndarray
    turns: list[Turn]
    words: list[Word]
    speakers: list[str]
    regions: list[Region] = field(default_factory=list)  # true speech regions


@dataclass
class World:
    """A fixed cast: the wearer, enrolled contacts, and strangers."""

    wearer: int
    contacts: dict[str, int]
    strangers: dict[str, int]
    cohort_sids: list[int]

    @classmethod
    def make(cls, seed: int = 0, n_contacts: int = 6, n_strangers: int = 6, n_cohort: int = 200):
        rng = random.Random(seed)
        sids = rng.sample(range(_tts().num_speakers), 1 + n_contacts + n_strangers + n_cohort)
        contacts = {f"contact{i}": sids[1 + i] for i in range(n_contacts)}
        strangers = {f"stranger{i}": sids[1 + n_contacts + i] for i in range(n_strangers)}
        return cls(sids[0], contacts, strangers, sids[1 + n_contacts + n_strangers:])

    def sid(self, name: str) -> int:
        if name == "wearer":
            return self.wearer
        return self.contacts.get(name, self.strangers.get(name))


def wearer_acoustics(rng: np.random.Generator) -> Acoustics:
    return Acoustics(room_rir(rng, 0.2, rng.uniform(0.2, 0.6)), 0.0)


PROFILES = {
    # Busy room: people 1-3 m away, echoey, noisy.
    "hard": {"dist": (1.0, 3.0), "rt60": (0.3, 0.8), "atten": (6, 14), "snr": (8, 25)},
    # Across a table in a quiet room.
    "moderate": {"dist": (0.7, 1.5), "rt60": (0.2, 0.5), "atten": (3, 8), "snr": (15, 30)},
}


def other_acoustics(rng: np.random.Generator, profile: str = "hard") -> Acoustics:
    pr = PROFILES[profile]
    d = rng.uniform(*pr["dist"])
    # Extra attenuation beyond the room model: body shadowing and mic placement.
    return Acoustics(room_rir(rng, d, rng.uniform(*pr["rt60"])), -rng.uniform(*pr["atten"]) - 4 * (d - 1))


def make_conversation(world: World, speakers: list[str], seed: int, n_turns: int = 24,
                      overlap_p: float = 0.15, backchannel_p: float = 0.2,
                      profile: str = "hard", lead_silence: float = 2.0) -> Conversation:
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    snr_db = PROFILES[profile]["snr"]
    ac = {s: (wearer_acoustics(nrng) if s == "wearer" else other_acoustics(nrng, profile)) for s in speakers}
    events = []  # (start, audio, speaker, text, rendered_dry_len)
    t = lead_silence
    prev = None
    for k in range(n_turns):
        s = rng.choice([x for x in speakers if x != prev] if len(speakers) > 1 else speakers)
        text = utterance(rng)
        dry = speak(text, world.sid(s), speed=rng.uniform(0.9, 1.15))
        y = apply(dry, ac[s])
        if events and rng.random() < overlap_p:
            start = max(lead_silence, t - rng.uniform(0.2, 1.0))
        else:
            start = t + max(0.05, rng.gauss(0.35, 0.25))
        events.append((start, y, s, text, len(dry)))
        t = start + len(dry) / SR
        if rng.random() < backchannel_p and len(speakers) > 1:
            b = rng.choice([x for x in speakers if x != s])
            bt = rng.choice(BACKCHANNELS)
            bd = speak(bt, world.sid(b))
            bstart = start + rng.uniform(0.3, max(0.4, len(dry) / SR - 0.5))
            events.append((bstart, apply(bd, ac[b]), b, bt, len(bd)))
        prev = s
    end = max(st + len(y) / SR for st, y, *_ in events) + 1.0
    mix = np.zeros(int(end * SR) + 1, np.float32)
    turns, words = [], []
    for st, y, s, text, n_dry in events:
        i = int(st * SR)
        mix[i: i + len(y)] += y[: len(mix) - i]
        dur = n_dry / SR
        turns.append(Turn(st, st + dur, s))
        toks = text.split()
        lens = np.array([len(w) + 1 for w in toks], float)
        edges = st + dur * np.concatenate([[0], np.cumsum(lens) / lens.sum()])
        words += [Word(w, float(a), float(b), s) for w, a, b in zip(toks, edges[:-1], edges[1:])]
    wear_rms = np.sqrt(np.mean(np.concatenate([y for _, y, s, *_ in events if s == "wearer"] or [mix]) ** 2))
    snr = nrng.uniform(*snr_db)
    noise = pink_noise(nrng, len(mix)) * wear_rms * 10 ** (-snr / 20)
    audio = np.clip(mix + noise, -1, 1)
    words.sort(key=lambda w: w.start)
    turns.sort(key=lambda t: t.start)
    return Conversation(audio, turns, words, speakers, [Region(t.start, t.end) for t in turns])


def enrollment_clips(world: World, name: str, seconds: float, seed: int) -> list[np.ndarray]:
    """Enrollment audio for one person, across several rooms and distances."""
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    clips, total = [], 0.0
    while total < seconds:
        ac = wearer_acoustics(nrng) if name == "wearer" else other_acoustics(nrng)
        dry = speak(utterance(rng), world.sid(name), speed=rng.uniform(0.9, 1.15))
        y = apply(dry, ac)
        y = y + pink_noise(nrng, len(y)) * np.sqrt(np.mean(y ** 2)) * 10 ** (-nrng.uniform(15, 30) / 20)
        clips.append(y.astype(np.float32))
        total += len(dry) / SR
    return clips


def cohort_clips(world: World, seed: int = 99) -> list[np.ndarray]:
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    out = []
    for sid in world.cohort_sids:
        # Spread the cohort over every condition we score in: a cohort from one
        # kind of room inflates normalized scores in any other kind.
        u = rng.random()
        ac = (other_acoustics(nrng, "hard") if u < 0.4 else
              other_acoustics(nrng, "moderate") if u < 0.8 else wearer_acoustics(nrng))
        out.append(apply(speak(utterance(rng), sid), ac))
    return out
