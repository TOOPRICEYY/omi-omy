"""Writes a finished conversation back into Omi through its developer API.

UNVERIFIED: the endpoint and field names below come from Omi's public docs
summary (POST /v1/dev/user/conversations/from-segments, segments with a
`speaker` label such as SPEAKER_00 and an optional numeric `speaker_id`). They
have not been checked against the live API yet, so write-back stays off until
OMI_API_KEY is set, and the first real call should be inspected.
"""
from __future__ import annotations

import datetime as dt


def build_payload(doc: dict, wearer: str | None = None) -> dict:
    ids: dict[str, int] = {}
    segs = []
    for s in doc["segments"]:
        if not s["text"]:
            continue
        i = ids.setdefault(s["speaker"], len(ids))
        known = not s["speaker"].startswith("unknown")
        segs.append({
            "text": f'{s["speaker"]}: {s["text"]}' if known and s["speaker"] != wearer else s["text"],
            "speaker": f"SPEAKER_{i:02d}",
            "speaker_id": i,
            "is_user": s["speaker"] == wearer,
            "start": s["start"],
            "end": s["end"],
        })
    start = dt.datetime.fromtimestamp(doc.get("started_at") or 0, dt.timezone.utc)
    return {
        "transcript_segments": segs,
        "started_at": start.isoformat(),
        "finished_at": (start + dt.timedelta(seconds=doc.get("duration", 0))).isoformat(),
        "source": "omi-omy",
    }


def push(doc: dict, api_key: str, base: str = "https://api.omi.me", wearer: str | None = None) -> dict:
    import requests

    r = requests.post(f"{base.rstrip('/')}/v1/dev/user/conversations/from-segments",
                      headers={"Authorization": f"Bearer {api_key}"},
                      json=build_payload(doc, wearer), timeout=60)
    r.raise_for_status()
    return r.json()
