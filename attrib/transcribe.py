"""MAI Transcribe 2 client, through OpenRouter's OpenAI-compatible audio API.

With diarize=True MAI also says who spoke when; its labels only hold within one
request, so each word's speaker is "p<piece>:<label>" (see attrib.maidiar).
Long recordings are cut into pieces at quiet points so each upload stays small,
and word times are shifted back onto the full recording.
"""
from __future__ import annotations

import io
import os

import numpy as np
import soundfile as sf

from .audio import SR
from .types import Word

BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "microsoft/mai-transcribe-2"


def parse_response(body: dict, offset: float = 0.0, piece: int | None = None) -> list[Word]:
    def spk(w):
        s = w.get("speaker")
        return None if piece is None or s is None else f"p{piece}:{s}"

    words = [Word(w["word"], offset + w["start"], offset + w["end"], spk(w)) for w in body.get("words") or []]
    if words:
        return words
    # No word timings: spread each segment's text evenly as a fallback.
    for seg in body.get("segments") or []:
        toks = seg.get("text", "").split()
        t0, dur = offset + seg["start"], seg["end"] - seg["start"]
        for k, tok in enumerate(toks):
            words.append(Word(tok, t0 + dur * k / len(toks), t0 + dur * (k + 1) / len(toks)))
    return words


def split_points(x: np.ndarray, max_s: float = 600.0, search_s: float = 15.0, sr: int = SR) -> list[int]:
    """Sample offsets that cut `x` into pieces of at most `max_s`, each cut at
    the quietest 0.2 s within the last `search_s` of its piece."""
    cuts, start, hop = [0], 0, int(0.2 * sr)
    while len(x) - start > max_s * sr:
        hi = start + int(max_s * sr)
        lo = hi - int(search_s * sr)
        energy = [float(np.mean(x[i: i + hop] ** 2)) for i in range(lo, hi - hop, hop)]
        start = lo + int(np.argmin(energy)) * hop + hop // 2
        cuts.append(start)
    return cuts + [len(x)]


class OpenRouterTranscriber:
    def __init__(self, key: str | None = None, model: str = MODEL, base_url: str = BASE_URL,
                 timeout: float = 600, max_piece_s: float = 600.0):
        self.key = key or os.environ["OPENROUTER_API_KEY"]
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_piece_s = max_piece_s
        self.seconds_billed = 0.0

    def __call__(self, wav_bytes: bytes, phrases: list[str] | None = None, diarize: bool = False) -> list[Word]:
        x, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        cuts = split_points(x, self.max_piece_s, sr=sr)
        words: list[Word] = []
        for k, (a, b) in enumerate(zip(cuts, cuts[1:])):
            words += self._piece(x[a:b], sr, a / sr, phrases, k if diarize else None)
        return words

    def _piece(self, x: np.ndarray, sr: int, offset: float, phrases: list[str] | None,
               piece: int | None = None) -> list[Word]:
        import requests

        buf = io.BytesIO()
        sf.write(buf, x, sr, format="FLAC", subtype="PCM_16")
        data = [("model", self.model), ("response_format", "verbose_json"),
                ("timestamp_granularities[]", "word")]
        if piece is not None:
            data.append(("diarize", "true"))
        if phrases:
            # Names of enrolled people, so they are spelled right.
            data.append(("prompt", ", ".join(phrases)))
        r = requests.post(f"{self.base_url}/audio/transcriptions",
                          headers={"Authorization": f"Bearer {self.key}"},
                          data=data, files={"file": ("audio.flac", buf.getvalue(), "audio/flac")},
                          timeout=(30, self.timeout))
        r.raise_for_status()
        body = r.json()
        self.seconds_billed += float((body.get("usage") or {}).get("seconds") or len(x) / sr)
        return parse_response(body, offset, piece)
