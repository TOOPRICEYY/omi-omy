import numpy as np

from attrib.types import Turn, Word
from omibridge.config import Config
from omibridge.engine import segments
from omibridge.omi_api import build_payload
from omibridge.sessions import SessionManager, pcm_to_float

SR = 16000


def _pcm(seconds: float, level: float) -> bytes:
    return (np.full(int(seconds * SR), level, np.float32) * 32767).astype("<i2").tobytes()


def _manager(tmp_path, **kw):
    cfg = Config(data_dir=tmp_path, token="t", idle_seconds=60, min_speech_seconds=8, **kw)
    now = [1000.0]
    # Fake voice detector: loud chunks are all speech, quiet ones none.
    m = SessionManager(cfg, lambda x: len(x) / SR if np.abs(x).mean() > 0.1 else 0.0, clock=lambda: now[0])
    return cfg, now, m


def test_pcm_roundtrip():
    x = pcm_to_float(_pcm(0.5, 0.5) + b"\x01")  # odd trailing byte is ignored
    assert len(x) == SR // 2 and abs(x[0] - 0.5) < 1e-3


def test_session_cut_on_silence(tmp_path):
    cfg, now, m = _manager(tmp_path)
    assert m.ingest("u", SR, _pcm(5, 0.0))["state"] == "idle"       # silence before speech is not stored
    assert not list(cfg.live_dir.glob("*.pcm"))
    for _ in range(4):
        now[0] += 5
        assert m.ingest("u", SR, _pcm(5, 0.5))["state"] == "recording"
    for _ in range(11):                                              # 55 s of silence: still open
        now[0] += 5
        m.ingest("u", SR, _pcm(5, 0.0))
    assert m.sweep() == []
    now[0] += 10
    (cid,) = m.sweep()
    assert (cfg.queue_dir / f"{cid}.pcm").stat().st_size == 2 * SR * 22  # 20 s of speech + 2 s tail
    assert not m.sessions and m.sweep() == []


def test_short_blip_is_dropped(tmp_path):
    cfg, now, m = _manager(tmp_path)
    m.ingest("u", SR, _pcm(3, 0.5))
    now[0] += 100
    assert m.sweep() == []
    assert not list(cfg.queue_dir.iterdir()) and not list(cfg.live_dir.iterdir())


def test_max_length_closes(tmp_path):
    cfg, now, m = _manager(tmp_path, max_minutes=0.5)
    res = {}
    for _ in range(6):
        now[0] += 5
        res = m.ingest("u", SR, _pcm(5, 0.5))
    assert res["state"] == "closed" and (cfg.queue_dir / f"{res['id']}.json").exists()


def test_segments_and_omi_payload():
    turns = [Turn(0, 2, "Ana"), Turn(2, 4, "unknown1")]
    words = [Word("hi", 0.1, 0.4, "Ana"), Word("there", 0.5, 0.9, "Ana"), Word("hey", 2.2, 2.5, "unknown1"),
             Word("again", 9.0, 9.3, "unknown1")]
    segs = segments(words, turns)
    assert [(s["speaker"], s["text"]) for s in segs] == [("Ana", "hi there"), ("unknown1", "hey"), ("unknown1", "again")]
    assert segments([], turns)[1] == {"speaker": "unknown1", "start": 2, "end": 4, "text": ""}
    p = build_payload({"segments": segs, "started_at": 0, "duration": 10}, wearer="Ana")
    a, b = p["transcript_segments"][:2]
    assert a["is_user"] and a["text"] == "hi there" and a["speaker"] == "SPEAKER_00"
    assert not b["is_user"] and b["speaker"] == "SPEAKER_01"
