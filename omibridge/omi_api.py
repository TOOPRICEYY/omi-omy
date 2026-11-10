"""Posts a finished conversation into Omi through its developer API.

Schema checked against Omi's source (backend/routers/developer.py at 760499bd)
on 2026-11-10; not yet exercised against the live API. A segment cannot carry
a name. It can point at an existing Omi person through `person_id`, and a
developer key can list people but not create them, so names are matched to
Omi's people by name and everyone else has their name put in front of the text.
"""
from __future__ import annotations

import datetime as dt

MAX_SEGMENTS = 500


def fit(segments: list[dict], limit: int = MAX_SEGMENTS) -> list[dict]:
    """Join neighbouring lines by the same speaker until the count fits Omi's limit."""
    segs = [dict(s) for s in segments if s["text"] and s["end"] > s["start"]]
    while len(segs) > limit:
        out: list[dict] = []
        for s in segs:
            if out and out[-1]["speaker"] == s["speaker"] and not out[-1].get("_joined"):
                out[-1] = out[-1] | {"text": out[-1]["text"] + " " + s["text"], "end": s["end"], "_joined": True}
            else:
                out.append(s)
        if len(out) == len(segs):
            return out[:limit]
        segs = [{k: v for k, v in s.items() if k != "_joined"} for s in out]
    return segs


def build_payload(doc: dict, wearer: str | None = None, people: dict[str, str] | None = None) -> dict:
    """`people` maps a name (lower case) to an Omi person id."""
    people = people or {}
    ids: dict[str, int] = {}
    segs = []
    for s in fit(doc["segments"]):
        i = ids.setdefault(s["speaker"], len(ids))
        known = not s["speaker"].startswith("unknown")
        is_user = s["speaker"] == wearer
        pid = None if is_user else people.get(s["speaker"].lower())
        seg = {
            "text": f'{s["speaker"]}: {s["text"]}' if known and not is_user and not pid else s["text"],
            "speaker": f"SPEAKER_{i:02d}",
            "speaker_id": i,
            "is_user": is_user,
            "start": s["start"],
            "end": s["end"],
        }
        if pid:
            seg["person_id"] = pid
        segs.append(seg)
    start = dt.datetime.fromtimestamp(doc.get("started_at") or 0, dt.timezone.utc)
    return {
        "transcript_segments": segs,
        "started_at": start.isoformat(),
        "finished_at": (start + dt.timedelta(seconds=doc.get("duration", 0))).isoformat(),
        "language": "en",
        "source": "external_integration",
        "client_session_id": f'omi-omy-{doc["id"]}',   # makes a retry safe
    }


def list_people(api_key: str, base: str = "https://api.omi.me") -> dict[str, str]:
    import requests

    r = requests.get(f"{base.rstrip('/')}/v1/dev/user/people",
                     headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
    r.raise_for_status()
    body = r.json()
    rows = body if isinstance(body, list) else body.get("people", [])
    return {p["name"].lower(): p["id"] for p in rows if p.get("name") and p.get("id")}


def push(doc: dict, api_key: str, base: str = "https://api.omi.me", wearer: str | None = None) -> dict:
    import requests

    try:
        people = list_people(api_key, base)
    except Exception:
        people = {}   # the key may lack people:read; names then go in the text
    r = requests.post(f"{base.rstrip('/')}/v1/dev/user/conversations/from-segments",
                      headers={"Authorization": f"Bearer {api_key}"},
                      json=build_payload(doc, wearer, people), timeout=60)
    r.raise_for_status()
    return r.json()
