"""Plays a recording into the bridge the way Omi's backend does: one-second
POSTs of raw PCM16 with an Idempotency-Key. For testing without a pendant.

    python -m omibridge.replay data/sample.wav --speed 20
"""
from __future__ import annotations

import argparse
import os
import time
import uuid
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

from attrib.audio import SR, resample

from .config import load_env_file


def main() -> None:
    load_env_file(Path(".env"))
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--url", default=None, help="webhook URL; default is the local bridge with the token from .env")
    ap.add_argument("--uid", default="replay")
    ap.add_argument("--speed", type=float, default=1.0, help="1 = real time")
    args = ap.parse_args()
    url = args.url or "http://127.0.0.1:{}/omi/audio/{}".format(
        os.environ.get("OMIBRIDGE_PORT", "8377"), os.environ.get("OMIBRIDGE_TOKEN", ""))
    x, sr = sf.read(args.file, dtype="float32", always_2d=True)
    x = resample(x.mean(axis=1), sr)
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    last = {}
    for i in range(0, len(pcm), SR):
        t0 = time.time()
        r = requests.post(url, params={"sample_rate": SR, "uid": args.uid}, data=pcm[i: i + SR].tobytes(),
                          headers={"Content-Type": "application/octet-stream", "Idempotency-Key": uuid.uuid4().hex},
                          timeout=30)
        r.raise_for_status()
        last = r.json()
        time.sleep(max(0.0, 1.0 / args.speed - (time.time() - t0)))
    print(f"sent {len(pcm) / SR:.0f} s; bridge says: {last}")


if __name__ == "__main__":
    main()
