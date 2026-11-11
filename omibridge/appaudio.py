"""Audio from the Omi app: live socket frames and offline-sync (.bin) files.

Live: each binary WebSocket message is one raw codec packet (no header): Opus
(10 ms or 20 ms frames, "opus" / "opus_fs320") or little-endian PCM16.

Offline sync: a .bin file is repeated [uint32 little-endian length][packet],
named ..._<codec>_<rate>_..._<unix seconds or ms>.bin (Omi backend:
utils/sync/files.py, utils/request_validation.py). pcm8 files hold one byte per
sample (unsigned, like 8-bit WAV), pcm16 files two.
"""
from __future__ import annotations

import re
import struct

import numpy as np

from attrib.audio import SR

MAX_FRAME_BYTES = 64 * 1024


class OpusDecoder:
    """Raw Opus packets -> 16 kHz int16 mono."""

    def __init__(self):
        import av

        self._av = av
        self.ctx = av.CodecContext.create("opus", "r")
        self.ctx.sample_rate = 48000
        self.ctx.layout = "mono"
        self.resampler = av.AudioResampler(format="s16", layout="mono", rate=SR)

    def decode(self, packet: bytes) -> np.ndarray:
        out = []
        try:
            frames = self.ctx.decode(self._av.Packet(packet))
        except Exception:
            return np.zeros(0, np.int16)   # one corrupt packet must not end the stream
        for fr in frames:
            for r in self.resampler.resample(fr):
                out.append(r.to_ndarray().reshape(-1))
        return np.concatenate(out).astype(np.int16) if out else np.zeros(0, np.int16)


class LiveDecoder:
    """Turns live socket packets into 16 kHz int16 audio."""

    def __init__(self, codec: str, sample_rate: int = SR):
        self.codec = (codec or "opus").lower()
        self.sample_rate = sample_rate
        self.opus = OpusDecoder() if self.codec.startswith("opus") else None
        if self.opus is None and self.codec not in ("pcm8", "pcm16"):
            raise ValueError(f"unsupported codec {codec!r}")

    def decode(self, packet: bytes) -> np.ndarray:
        if self.opus is not None:
            return self.opus.decode(packet)
        x = np.frombuffer(packet[: len(packet) // 2 * 2], dtype="<i2")
        return to_16k(x, self.sample_rate)


def to_16k(x: np.ndarray, rate: int) -> np.ndarray:
    if rate == SR or len(x) == 0:
        return x.astype(np.int16)
    from attrib.audio import resample

    y = resample(x.astype(np.float32) / 32768.0, rate)
    return (np.clip(y, -1, 1) * 32767).astype(np.int16)


def wal_start_time(filename: str) -> float:
    """The capture start encoded at the end of an offline-sync file name."""
    stem = filename.rsplit("/", 1)[-1]
    if not stem.endswith(".bin") or "_" not in stem:
        raise ValueError(f"not an Omi sync file name: {filename!r}")
    t = float(stem[:-4].rsplit("_", 1)[1])
    if t <= 0:
        raise ValueError("bad timestamp")
    return t / 1000 if t > 10_000_000_000 else t


def wal_frames(data: bytes) -> list[bytes]:
    """Split a .bin file into packets. Stops at the first damaged frame and keeps what came before."""
    frames, i = [], 0
    while i + 4 <= len(data):
        (n,) = struct.unpack_from("<I", data, i)
        if n == 0 or n > MAX_FRAME_BYTES or i + 4 + n > len(data):
            break
        frames.append(data[i + 4: i + 4 + n])
        i += 4 + n
    return frames


def decode_wal(filename: str, data: bytes) -> tuple[float, np.ndarray]:
    """(start unix time, 16 kHz int16 audio) for one offline-sync file."""
    start = wal_start_time(filename)
    frames = wal_frames(data)
    if "_pcm16_" in filename or "_pcm8_" in filename:
        m = re.search(r"_pcm(?:8|16)_(\d+)_", filename)
        rate = int(m.group(1)) if m else (SR if "_pcm16_" in filename else 8000)
        raw = b"".join(frames)
        if "_pcm8_" in filename:
            x = (np.frombuffer(raw, np.uint8).astype(np.int16) - 128) * 256
        else:
            x = np.frombuffer(raw[: len(raw) // 2 * 2], "<i2")
        return start, to_16k(x, rate)
    dec = OpusDecoder()
    parts = [dec.decode(f) for f in frames]
    return start, np.concatenate(parts) if parts else np.zeros(0, np.int16)
