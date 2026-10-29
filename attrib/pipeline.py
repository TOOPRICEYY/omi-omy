"""One conversation in, named speaker turns and words out."""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import soundfile as sf

from .align import assign_words
from .audio import SR
from .diarize import Diarizer, overlap_ratio
from .identify import Identifier, final_labels, is_unknown
from .types import Attribution, Region, Turn, Word
from .vad import TimeMap, pack


@dataclass
class EscalationPolicy:
    """When a conversation is worth re-running on Precision-2."""

    min_confidence: float = 0.6     # any speaker below this
    max_overlap: float = 0.15       # share of speech with crosstalk
    max_speakers: int = 3
    min_unknown_time: float = 0.25  # share of speech left unknown

    def check(self, att: Attribution, overlap: float) -> list[str]:
        reasons = []
        speech = sum(t.duration for t in att.turns) or 1.0
        if any(c < self.min_confidence for p, c in att.identities.values() if not is_unknown(p)):
            reasons.append("low-confidence match")
        if overlap > self.max_overlap:
            reasons.append(f"overlap {overlap:.0%}")
        if len(att.identities) > self.max_speakers:
            reasons.append(f"{len(att.identities)} speakers")
        unknown = sum(t.duration for t in att.turns if is_unknown(t.speaker)) / speech
        if unknown > self.min_unknown_time:
            reasons.append(f"{unknown:.0%} unknown")
        return reasons


Transcriber = Callable[[bytes, list[str] | None], list[Word]]


@dataclass
class Pipeline:
    diarizer: Diarizer
    identifier: Identifier
    transcriber: Transcriber | None = None
    escalation: EscalationPolicy = field(default_factory=EscalationPolicy)
    unknown_merge: float = 0.6

    def run(self, x: np.ndarray, regions: list[Region] | None = None,
            priors: dict[str, float] | None = None,
            words: list[Word] | None = None) -> Attribution:
        """Attribute one conversation.

        `regions` are speech regions from the VAD; without them the whole clip
        is treated as speech. `words` can be passed in directly (tests, or a
        transcript that already exists); otherwise the transcriber is called
        on the packed speech-only audio.
        """
        if regions is None:
            regions = [Region(0.0, len(x) / SR)]
        packed, tmap = pack(x, regions)
        raw_turns = self.diarizer.turns(packed)
        clusters = self.diarizer.clusters(packed, raw_turns)
        matches = self.identifier.identify(clusters, priors)
        names = final_labels(clusters, matches, self.unknown_merge)
        overlap = overlap_ratio(raw_turns)

        turns = [Turn(tmap.to_orig(t.start), tmap.to_orig(t.end),
                      names[t.speaker], matches[t.speaker].confidence, t.speaker)
                 for t in raw_turns]
        if words is None and self.transcriber is not None:
            buf = io.BytesIO()
            sf.write(buf, packed, SR, format="WAV", subtype="PCM_16")
            phrases = list(self.identifier.gallery.people)
            words = [_to_orig(w, tmap) for w in self.transcriber(buf.getvalue(), phrases)]
        att = Attribution(turns, assign_words(words or [], turns),
                          {c.label: (matches[c.label].person, matches[c.label].confidence) for c in clusters},
                          clusters=clusters, matches=matches)
        att.escalate_reasons = self.escalation.check(att, overlap)
        att.escalate = bool(att.escalate_reasons)
        return att


def _to_orig(w: Word, tmap: TimeMap) -> Word:
    return Word(w.text, tmap.to_orig(w.start), tmap.to_orig(w.end))
