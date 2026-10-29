"""MAI Transcribe 2 client (Azure Speech LLM Speech API, fast transcription).

Speaker labels are turned off on purpose: native diarization fails on audio
over ~15 minutes, and we diarize locally anyway. The response parser follows
the fast-transcription schema and still needs checking against a live key.
"""
from __future__ import annotations

import json
import os

from .types import Word

API_VERSION = "2025-10-15"


def build_definition(phrases: list[str] | None = None, locale: str | None = "en-US") -> dict:
    d: dict = {
        "enhancedMode": {
            "enabled": True,
            "model": "MAI-Transcribe-2",
            "modelOptions": {"timestamps": "word", "transcribeStyle": "verbatim"},
        },
    }
    if locale:
        d["locales"] = [locale]
    if phrases:
        d["phraseList"] = {"phrases": phrases}
    return d


def parse_response(body: dict) -> list[Word]:
    words: list[Word] = []
    for ph in body.get("phrases", []):
        ws = ph.get("words") or []
        if not ws:
            # No word timings: spread the phrase evenly as a fallback.
            toks = ph.get("text", "").split()
            t0 = ph["offsetMilliseconds"] / 1000
            dur = ph["durationMilliseconds"] / 1000
            for k, tok in enumerate(toks):
                words.append(Word(tok, t0 + dur * k / len(toks), t0 + dur * (k + 1) / len(toks)))
            continue
        for w in ws:
            s = w["offsetMilliseconds"] / 1000
            words.append(Word(w["text"], s, s + w["durationMilliseconds"] / 1000))
    return words


class MAITranscriber:
    def __init__(self, endpoint: str | None = None, key: str | None = None, timeout: float = 1800):
        self.endpoint = (endpoint or os.environ["AZURE_SPEECH_ENDPOINT"]).rstrip("/")
        self.key = key or os.environ["AZURE_SPEECH_KEY"]
        self.timeout = timeout

    def __call__(self, wav_bytes: bytes, phrases: list[str] | None = None) -> list[Word]:
        import requests

        r = requests.post(
            f"{self.endpoint}/speechtotext/transcriptions:transcribe?api-version={API_VERSION}",
            headers={"Ocp-Apim-Subscription-Key": self.key},
            files={
                "audio": ("audio.wav", wav_bytes, "audio/wav"),
                "definition": (None, json.dumps(build_definition(phrases)), "application/json"),
            },
            timeout=(30, self.timeout),
        )
        r.raise_for_status()
        return parse_response(r.json())
