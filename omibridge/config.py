"""Settings, read from the environment (and a .env file next to the data)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def load_env_file(path: Path) -> None:
    """KEY=VALUE lines; values already in the environment win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass
class Config:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("OMIBRIDGE_DATA", "data/bridge")))
    models_dir: Path = field(default_factory=lambda: Path(os.environ.get("OMIBRIDGE_MODELS", "models")))
    # Shared secret in the webhook path, so only your Omi app can post audio.
    token: str = field(default_factory=lambda: os.environ.get("OMIBRIDGE_TOKEN", ""))
    host: str = field(default_factory=lambda: os.environ.get("OMIBRIDGE_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.environ.get("OMIBRIDGE_PORT", "8377")))
    # The name you enroll yourself under; gets the "closest to the mic" bonus.
    wearer: str | None = field(default_factory=lambda: os.environ.get("OMIBRIDGE_WEARER") or None)

    # A conversation ends after this much time without speech.
    idle_seconds: float = field(default_factory=lambda: _f("OMIBRIDGE_IDLE_SECONDS", 120))
    max_minutes: float = field(default_factory=lambda: _f("OMIBRIDGE_MAX_MINUTES", 60))
    min_speech_seconds: float = field(default_factory=lambda: _f("OMIBRIDGE_MIN_SPEECH_SECONDS", 8))

    segmentation: str = field(default_factory=lambda: os.environ.get("OMIBRIDGE_SEGMENTATION", "diarizen"))
    # 0.5 over-splits a little; naming re-joins known people. 0.6 scored the same on
    # AMI but sometimes fuses two similar voices, which cannot be undone later.
    cluster_threshold: float = field(default_factory=lambda: _f("OMIBRIDGE_CLUSTER_THRESHOLD", 0.5))
    identify_threshold: float = field(default_factory=lambda: _f("OMIBRIDGE_IDENTIFY_THRESHOLD", 2.5))
    transcribe: bool = field(default_factory=lambda: os.environ.get("OMIBRIDGE_TRANSCRIBE", "1") != "0")

    # Omi developer API write-back (off unless a key is set).
    omi_api_key: str = field(default_factory=lambda: os.environ.get("OMI_API_KEY", ""))
    omi_api_base: str = field(default_factory=lambda: os.environ.get("OMI_API_BASE", "https://api.omi.me"))

    @property
    def live_dir(self) -> Path:
        return self.data_dir / "live"

    @property
    def queue_dir(self) -> Path:
        return self.data_dir / "queue"

    @property
    def conv_dir(self) -> Path:
        return self.data_dir / "conversations"

    @property
    def gallery_path(self) -> Path:
        return self.data_dir / "gallery.npz"

    def ensure_dirs(self) -> None:
        for d in (self.live_dir, self.queue_dir, self.conv_dir):
            d.mkdir(parents=True, exist_ok=True)
