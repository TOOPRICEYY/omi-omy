"""Title, overview and action items for a finished conversation, written by an
LLM through OpenRouter's chat API (the same key as the transcription)."""
from __future__ import annotations

import json
import os

BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "anthropic/claude-sonnet-5.5"

# The categories the Omi app knows how to show.
CATEGORIES = ["personal", "education", "health", "finance", "legal", "philosophy", "spiritual", "science",
              "entrepreneurship", "parenting", "romantic", "travel", "inspiration", "technology", "business",
              "social", "work", "sports", "politics", "literature", "history", "architecture", "music",
              "weather", "news", "entertainment", "psychology", "real", "design", "family", "economics",
              "environment", "other"]

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "overview", "emoji", "category", "action_items"],
    "properties": {
        "title": {"type": "string", "description": "At most eight words, no trailing period."},
        "overview": {"type": "string", "description": "Two to five sentences: what was discussed and decided."},
        "emoji": {"type": "string", "description": "One emoji that fits the conversation."},
        "category": {"type": "string", "enum": CATEGORIES},
        "action_items": {"type": "array", "items": {"type": "string"},
                         "description": "Things someone said they would do. Empty when there are none."},
    },
}

SYSTEM = """You summarize a conversation recorded by a wearable microphone for the person who wore it.
The transcript is machine-made: words can be wrong and a line can be credited to the wrong speaker.
Speakers called unknown1, unknown2 and so on are people who have not been identified; refer to them
as "someone" or "another speaker", never by that label.
Only state what the transcript supports. Do not invent names, numbers, dates or decisions.
If the transcript is too short or garbled to summarize, say so plainly in the overview.
Write in the language the conversation is in."""


def transcript_text(doc: dict, max_chars: int = 400_000) -> str:
    lines = [f"[{int(s['start']) // 60}:{int(s['start']) % 60:02d}] {s['speaker']}: {s['text']}"
             for s in doc.get("segments", []) if s.get("text")]
    text = "\n".join(lines)
    if len(text) > max_chars:  # keep the start and the end of a very long recording
        half = max_chars // 2
        text = text[:half] + "\n[... middle left out ...]\n" + text[-half:]
    return text


def clean(raw: dict) -> dict:
    """Whatever the model returned, in the shape the rest of the bridge expects."""
    items = raw.get("action_items") or []
    cat = str(raw.get("category") or "other").lower()
    return {"title": str(raw.get("title") or "").strip().rstrip(".")[:120],
            "overview": str(raw.get("overview") or "").strip(),
            "emoji": str(raw.get("emoji") or "").strip()[:8],
            "category": cat if cat in CATEGORIES else "other",
            "action_items": [str(i).strip() for i in items if i is not None and str(i).strip()][:20]}


class Summarizer:
    def __init__(self, key: str | None = None, model: str | None = None, base_url: str = BASE_URL,
                 timeout: float = 120):
        self.key = key or os.environ["OPENROUTER_API_KEY"]
        self.model = model or os.environ.get("OMIBRIDGE_SUMMARY_MODEL") or MODEL
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def __call__(self, doc: dict) -> dict | None:
        """None when there is no text to summarize."""
        text = transcript_text(doc)
        if not text:
            return None
        import requests

        people = ", ".join(s["name"] for s in doc.get("speakers", []) if s.get("known"))
        user = (f"Recorded: {doc.get('started_at_text') or 'unknown time'}; "
                f"length {round(doc.get('duration', 0) / 60, 1)} minutes.\n"
                + (f"Identified speakers: {people}.\n" if people else "")
                + f"\nTranscript:\n{text}")
        r = requests.post(f"{self.base_url}/chat/completions",
                          headers={"Authorization": f"Bearer {self.key}"},
                          json={"model": self.model, "max_tokens": 1500,
                                "messages": [{"role": "system", "content": SYSTEM},
                                             {"role": "user", "content": user}],
                                "response_format": {"type": "json_schema",
                                                    "json_schema": {"name": "summary", "strict": True,
                                                                    "schema": SCHEMA}}},
                          timeout=(30, self.timeout))
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        if isinstance(content, list):  # some providers return content parts
            content = "".join(p.get("text", "") for p in content)
        out = clean(json.loads(content))
        out["model"] = self.model
        return out
