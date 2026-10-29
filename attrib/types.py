from __future__ import annotations

from dataclasses import dataclass, field

UNKNOWN = "unknown"


@dataclass
class Turn:
    """A stretch of audio attributed to one speaker label."""

    start: float
    end: float
    speaker: str
    confidence: float = 1.0
    cluster: str | None = None  # anonymous diarization label behind the name

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Word:
    text: str
    start: float
    end: float
    speaker: str | None = None

    @property
    def mid(self) -> float:
        return 0.5 * (self.start + self.end)


@dataclass
class Region:
    """A span of the original recording, in seconds."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Attribution:
    """Final output for one conversation."""

    turns: list[Turn]
    words: list[Word]
    # cluster label -> (person or UNKNOWN, confidence in [0, 1])
    identities: dict[str, tuple[str, float]] = field(default_factory=dict)
    escalate: bool = False
    escalate_reasons: list[str] = field(default_factory=list)
    clusters: list = field(default_factory=list)   # diarize.Cluster, for gallery updates
    matches: dict = field(default_factory=dict)    # cluster label -> identify.Match
