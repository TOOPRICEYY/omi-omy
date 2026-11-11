"""Queues a recording from a file as if it had arrived over the webhook.

For audio the webhook never sees, such as what the pendant stored while the
phone was away: export it from the Omi app and upload it on the review page.
"""
from __future__ import annotations

import json
import time
from datetime import datetime

import numpy as np

from attrib.audio import SR, resample
from attrib.transcribe import split_points

from .config import Config


def parse_when(value: str | None) -> float | None:
    """'2026-11-11T09:30' (local time unless it carries an offset) or epoch seconds."""
    value = (value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def queue_recording(cfg: Config, x: np.ndarray, sr: int, started_at: float | None = None,
                    source: str = "", now: float | None = None) -> list[str]:
    """Writes the audio to the queue as one conversation per `max_minutes`,
    cut at quiet points. Returns the conversation ids, in order."""
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = resample(np.asarray(x, dtype=np.float32), sr)
    if len(x) < SR * cfg.min_speech_seconds:
        raise ValueError(f"the recording is shorter than {cfg.min_speech_seconds:g} seconds")
    now = time.time() if now is None else now
    # Without a recording time, assume it ended just now.
    t0 = started_at if started_at is not None else now - len(x) / SR
    cfg.ensure_dirs()
    ids = []
    cuts = split_points(x, cfg.max_minutes * 60)
    for a, b in zip(cuts, cuts[1:]):
        start = t0 + a / SR
        stem = time.strftime("%Y%m%d-%H%M%S", time.localtime(start)) + "-import"
        cid, n = stem, 1
        while (cfg.conv_dir / cid).exists() or (cfg.queue_dir / f"{cid}.pcm").exists():
            n += 1
            cid = f"{stem}{n}"
        pcm = (np.clip(x[a:b], -1.0, 1.0) * 32767.0).astype("<i2")
        (cfg.queue_dir / f"{cid}.pcm").write_bytes(pcm.tobytes())
        meta = {"id": cid, "uid": "import", "sample_rate": SR, "started_at": start, "samples": b - a,
                "closed_at": now, "imported": True, "source": source}
        (cfg.queue_dir / f"{cid}.json").write_text(json.dumps(meta))
        ids.append(cid)
    return ids
