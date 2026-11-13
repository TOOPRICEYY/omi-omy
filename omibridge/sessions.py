"""Turns the Omi app's stream of raw audio chunks into whole conversations.

The app posts PCM16 every few seconds for as long as the pendant is connected,
silence included. Each chunk is appended to a file per user. A cheap voice
detector watches for speech: audio before the first speech is dropped, and a
conversation is closed and queued once nobody has spoken for `idle_seconds`.
"""
from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from attrib.audio import SR, resample

from .config import Config


TAIL = 2     # seconds kept after the last speech
PREROLL = 3  # seconds kept from before the first speech
PREROLL_GAP = 2  # seconds of missing audio after which the pre-roll is stale


@dataclass
class Session:
    uid: str
    sample_rate: int
    started_at: float          # wall clock of the first kept sample
    samples: int = 0
    speech_seconds: float = 0.0
    last_speech_at: float = 0.0
    samples_at_last_speech: int = 0
    last_rx_at: float = 0.0


def pcm_to_float(body: bytes) -> np.ndarray:
    return np.frombuffer(body[: len(body) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0


class SessionManager:
    def __init__(self, cfg: Config, speech_seconds: Callable[[np.ndarray], float],
                 clock: Callable[[], float] = time.time):
        self.cfg = cfg
        self.speech_seconds = speech_seconds  # 16 kHz float audio -> seconds of speech in it
        self.clock = clock
        self.lock = threading.Lock()
        self.sessions: dict[str, Session] = {}
        self.preroll: dict[str, bytes] = {}     # last few seconds heard while idle
        self.preroll_at: dict[str, float] = {}  # when the pre-roll's last audio was received
        self.seen: OrderedDict[str, None] = OrderedDict()  # recent Idempotency-Keys
        cfg.ensure_dirs()

    def _path(self, uid: str) -> Path:
        safe = "".join(c for c in uid if c.isalnum() or c in "-_") or "user"
        return self.cfg.live_dir / f"{safe}.pcm"

    def ingest(self, uid: str, sample_rate: int, body: bytes, key: str | None = None) -> dict:
        """`key` is Omi's Idempotency-Key: a retried POST must not add the audio twice."""
        now = self.clock()
        if key:
            with self.lock:
                if key in self.seen:
                    return {"state": "duplicate"}
                self.seen[key] = None
                while len(self.seen) > 5000:
                    self.seen.popitem(last=False)
        x = pcm_to_float(body)
        spoken = self.speech_seconds(resample(x, sample_rate)) if len(x) else 0.0
        with self.lock:
            s = self.sessions.get(uid)
            if s is not None and s.sample_rate != sample_rate:
                self._close(s, now)
                s = None
            if s is None:
                pre = self._lead_in(uid, now - len(body) / 2 / sample_rate)
                if spoken <= 0:
                    # Omi sends one-second pieces, so keep a little to not clip the first word.
                    self.preroll[uid] = (pre + body)[-PREROLL * sample_rate * 2:]
                    self.preroll_at[uid] = now
                    return {"state": "idle"}
                s = Session(uid, sample_rate, now - (len(pre) + len(body)) / 2 / sample_rate,
                            samples=len(pre) // 2)
                self.sessions[uid] = s
                self._path(uid).write_bytes(pre)
            with open(self._path(uid), "ab") as f:
                f.write(body)
            s.samples += len(body) // 2
            s.last_rx_at = now
            if spoken > 0:
                s.speech_seconds += spoken
                s.last_speech_at = now
                s.samples_at_last_speech = s.samples
            if s.samples / s.sample_rate >= self.cfg.max_minutes * 60:
                return {"state": "closed", "id": self._close(s, now)}
            return {"state": "recording", "seconds": round(s.samples / s.sample_rate, 1)}

    def _lead_in(self, uid: str, start: float) -> bytes:
        """Take the pre-roll if it runs up to `start`. Quiet audio from before a gap (an earlier
        recording in an offline upload, a dropped connection) is not this audio's lead-in, and
        prepending it would put the conversation's start time too early."""
        pre = self.preroll.pop(uid, b"")
        end = self.preroll_at.pop(uid, start)
        return pre if start - end <= PREROLL_GAP else b""

    def sweep(self) -> list[str]:
        """Close sessions that have gone quiet. Returns the queued conversation ids."""
        now = self.clock()
        closed = []
        with self.lock:
            for s in list(self.sessions.values()):
                if now - max(s.last_speech_at, s.started_at) >= self.cfg.idle_seconds:
                    cid = self._close(s, now)
                    if cid:
                        closed.append(cid)
        return closed

    def close(self, uid: str) -> str | None:
        """End `uid`'s conversation now (the app's "stop" button). Returns the queued id, if any."""
        with self.lock:
            s = self.sessions.get(uid)
            return self._close(s, self.clock()) if s is not None else None

    def _close(self, s: Session, now: float) -> str | None:
        self.sessions.pop(s.uid, None)
        src = self._path(s.uid)
        if s.speech_seconds < self.cfg.min_speech_seconds:
            src.unlink(missing_ok=True)
            return None
        # Drop the silence that ran out the idle timer.
        keep = min(s.samples, s.samples_at_last_speech + TAIL * s.sample_rate)
        if keep < s.samples:
            with open(src, "r+b") as f:
                f.truncate(keep * 2)
            s.samples = keep
        cid = time.strftime("%Y%m%d-%H%M%S", time.localtime(s.started_at)) + "-" + src.stem[:8]
        src.rename(self.cfg.queue_dir / f"{cid}.pcm")
        meta = asdict(s) | {"id": cid, "closed_at": now}
        (self.cfg.queue_dir / f"{cid}.json").write_text(json.dumps(meta))
        return cid

    def recover(self) -> None:
        """After a restart, queue whatever was being recorded."""
        for p in self.cfg.live_dir.glob("*.pcm"):
            n = p.stat().st_size // 2
            if n < SR * self.cfg.min_speech_seconds:
                p.unlink()
                continue
            # The sample rate isn't stored with the raw file; Omi devices send 16 kHz.
            s = Session(p.stem, SR, p.stat().st_mtime - n / SR, n, speech_seconds=n / SR,
                        samples_at_last_speech=n)
            self.sessions[p.stem] = s
            self._close(s, self.clock())
