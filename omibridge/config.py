"""Settings, read from the environment (and a .env file next to the data)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def parse_env_file(path: Path) -> dict[str, str]:
    """KEY=VALUE lines of a .env file. Quotes and trailing comments are removed."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if v[:1] in ("'", '"') and v[:1] in v[1:]:
            v = v[1:v.index(v[0], 1)]
        else:
            # "KEY=   # comment" is empty; "KEY=value  # comment" is "value".
            v = "" if v.startswith("#") else v.split(" #", 1)[0].split("\t#", 1)[0].rstrip()
        out[k.strip()] = v
    return out


def load_env_file(path: Path) -> None:
    """Values already in the environment win."""
    for k, v in parse_env_file(path).items():
        os.environ.setdefault(k, v)


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

    # Who spoke when: "mai" uses MAI Transcribe 2's diarization (names come from voiceprints
    # of each speaker's clearest speech); "local" runs DiariZen on this Mac. "mai" falls back
    # to "local" for a conversation when MAI returns no speaker labels or fails.
    diarizer: str = field(default_factory=lambda: os.environ.get("OMIBRIDGE_DIARIZER", "mai").strip().lower())
    # A MAI speaker needs this much clear speech (turns of 1.5 s or more) before it is named.
    min_name_seconds: float = field(default_factory=lambda: _f("OMIBRIDGE_MIN_NAME_SECONDS", 3.0))
    # What goes to MAI: "speech" (Silero's speech stretches, packed) or "all" (the whole
    # conversation). Measured 2026-11-12 on 18 conversations: "speech" lost ~40% of the words
    # MAI finds in the whole recording (muffled pendant speech scores near 0 in Silero) and
    # moved 14% of the remaining words to another speaker; "all" costs ~2.7x more MAI time.
    mai_audio: str = field(default_factory=lambda: os.environ.get("OMIBRIDGE_MAI_AUDIO", "speech").strip().lower())
    # What the conversation speech detector hears: "raw", or "hp150norm" (150 Hz high-pass plus
    # one gain per recording; recovers about a third of the lost words at 46% of audio sent).
    vad_front: str = field(default_factory=lambda: os.environ.get("OMIBRIDGE_VAD_FRONT", "raw").strip().lower())
    # Presence filter (off by default): "skip" leaves a conversation untranscribed (audio kept,
    # no MAI cost) when the wearer's enrolled voice is not found in it; it can be transcribed
    # later from the review page. Never skips while no wearer is enrolled.
    presence: str = field(default_factory=lambda: os.environ.get("OMIBRIDGE_PRESENCE", "off").strip().lower())
    presence_threshold: float = field(default_factory=lambda: _f("OMIBRIDGE_PRESENCE_THRESHOLD", 1.5))
    segmentation: str = field(default_factory=lambda: os.environ.get("OMIBRIDGE_SEGMENTATION", "diarizen"))
    # "mps": DiariZen's PyTorch model on the Mac's GPU (needs requirements-gpu.txt; falls back
    # to the ONNX model on the CPU when unavailable). Only applies to the diarizen segmentation.
    diarizer_device: str = field(default_factory=lambda: os.environ.get("OMIBRIDGE_DIARIZER_DEVICE", "cpu"))
    diarizer_batch: int = field(default_factory=lambda: int(os.environ.get("OMIBRIDGE_DIARIZER_BATCH", "4")))
    diarizen_checkpoint: str = field(default_factory=lambda: os.environ.get("OMIBRIDGE_DIARIZEN_CHECKPOINT", ""))
    # 0.5 over-splits a little; naming re-joins known people. 0.6 scored the same on
    # AMI but sometimes fuses two similar voices, which cannot be undone later.
    cluster_threshold: float = field(default_factory=lambda: _f("OMIBRIDGE_CLUSTER_THRESHOLD", 0.5))
    identify_threshold: float = field(default_factory=lambda: _f("OMIBRIDGE_IDENTIFY_THRESHOLD", 2.5))
    # A voice is named only if it also resembles the person's samples this much in plain cosine
    # (0 switches the check off). Measured 2026-11-13 on 318 pendant conversations: an entry
    # learned from a quiet, distant speaker was named on 104 clusters in 36 voice groups without
    # the check and on 53 in 11 with 0.4, while the wearer kept 144 of 152.
    identify_raw_floor: float = field(default_factory=lambda: _f("OMIBRIDGE_IDENTIFY_RAW_FLOOR", 0.4))
    # The same check for the wearer's own entry; negative means "the same as above". Measured on
    # the same 318 with a wearer entry learned from one recording: at 0.4 thirteen of the wearer's
    # own names went to unknown, at 0.3 ten of them stay and no confident one is lost, and the
    # other entry stays at 53. Worth raising to 0.4 once the wearer has tens of seconds enrolled.
    identify_raw_floor_wearer: float = field(default_factory=lambda: _f("OMIBRIDGE_IDENTIFY_RAW_FLOOR_WEARER", 0.3))
    transcribe: bool = field(default_factory=lambda: os.environ.get("OMIBRIDGE_TRANSCRIBE", "1") != "0")
    # Title, overview and action items for each conversation (same OpenRouter key).
    summarize: bool = field(default_factory=lambda: os.environ.get("OMIBRIDGE_SUMMARIZE", "1") != "0")

    # Posting finished conversations into Omi (developer key with conversations:write).
    # Off by default: Omi also keeps its own copy of every live conversation, so
    # this creates a second one and runs Omi's summaries and memories on it again.
    omi_writeback: bool = field(default_factory=lambda: os.environ.get("OMIBRIDGE_OMI_WRITEBACK", "0") == "1")
    omi_api_key: str = field(default_factory=lambda: os.environ.get("OMI_API_KEY", ""))
    omi_api_base: str = field(default_factory=lambda: os.environ.get("OMI_API_BASE", "https://api.omi.me"))

    # The app API: what the forked Omi app (dev flavor) talks to instead of Omi's
    # servers. Served on the public port under /app/<OMIBRIDGE_APP_SECRET>/; off while
    # the secret is empty. The Firebase sign-in emulator calls sit at the root.
    app_secret: str = field(default_factory=lambda: os.environ.get("OMIBRIDGE_APP_SECRET", ""))
    # The single account the app signs in as.
    app_uid: str = field(default_factory=lambda: os.environ.get("OMIBRIDGE_APP_UID", "owner"))

    @property
    def app_dir(self) -> Path:
        return self.data_dir / "app"

    @property
    def trash_dir(self) -> Path:
        return self.data_dir / "trash"

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
