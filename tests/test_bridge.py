import numpy as np
import pytest

from attrib.types import Turn, Word
from omibridge.config import Config
from omibridge.engine import segments
from omibridge.omi_api import build_payload, fit
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
    assert not list(cfg.live_dir.glob("*.pcm"))                      # only a short pre-roll in memory
    for _ in range(4):
        now[0] += 5
        assert m.ingest("u", SR, _pcm(5, 0.5))["state"] == "recording"
    for _ in range(11):                                              # 55 s of silence: still open
        now[0] += 5
        m.ingest("u", SR, _pcm(5, 0.0))
    assert m.sweep() == []
    now[0] += 10
    (cid,) = m.sweep()
    assert (cfg.queue_dir / f"{cid}.pcm").stat().st_size == 2 * SR * 25  # 3 s from before, 20 s of speech, 2 s tail
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
    p = build_payload({"id": "c1", "segments": segs, "started_at": 0, "duration": 10}, wearer="Ana")
    a, b = p["transcript_segments"][:2]
    assert a["is_user"] and a["text"] == "hi there" and a["speaker"] == "SPEAKER_00"
    assert not b["is_user"] and b["speaker"] == "SPEAKER_01"
    assert p["client_session_id"] == "omi-omy-c1" and p["started_at"].endswith("+00:00")
    named = [{"speaker": "Bo", "start": 0, "end": 1, "text": "yo"}, {"speaker": "Cy", "start": 1, "end": 2, "text": "hm"}]
    q = build_payload({"id": "c2", "segments": named, "started_at": 0}, people={"bo": "p-1"})["transcript_segments"]
    assert q[0]["person_id"] == "p-1" and q[0]["text"] == "yo"      # an Omi person: linked by id
    assert "person_id" not in q[1] and q[1]["text"] == "Cy: hm"     # not in Omi: name goes in the text


def test_omi_segment_limit():
    segs = [{"speaker": "A" if (i // 3) % 2 else "B", "start": i, "end": i + 1, "text": str(i)} for i in range(1200)]
    out = fit(segs, 500)
    assert len(out) <= 500 and out[0]["text"].startswith("0 1")
    assert sum(len(s["text"].split()) for s in out) == 1200


def test_retry_and_preroll(tmp_path):
    cfg, now, m = _manager(tmp_path)
    m.ingest("u", SR, _pcm(1, 0.0), key="a")
    m.ingest("u", SR, _pcm(1, 0.0), key="b")
    assert m.ingest("u", SR, _pcm(1, 0.5), key="c")["state"] == "recording"
    assert m.ingest("u", SR, _pcm(1, 0.5), key="c")["state"] == "duplicate"   # Omi retried the same POST
    assert (cfg.live_dir / "u.pcm").stat().st_size == 2 * SR * 3             # 2 s kept from before + 1 s


def test_import_recording_is_queued_and_split(tmp_path):
    import json

    from omibridge.importer import parse_when, queue_recording

    cfg = Config(data_dir=tmp_path, max_minutes=1, min_speech_seconds=8)
    sr = 8000
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(150 * sr) * 0.1).astype(np.float32)   # 2.5 minutes, 8 kHz
    x[55 * sr: 56 * sr] = 0.0                                      # a quiet second to cut at
    stereo = np.stack([x, x], axis=1)
    t0 = parse_when("2026-11-11T09:30:00+00:00")
    ids = queue_recording(cfg, stereo, sr, t0, "walk.wav", now=t0 + 9999)
    assert len(ids) == 3 and len(set(ids)) == 3 and all(i.endswith("-import") for i in ids)
    metas = [json.loads((cfg.queue_dir / f"{i}.json").read_text()) for i in ids]
    assert metas[0]["started_at"] == t0 and metas[0]["sample_rate"] == 16000 and metas[0]["source"] == "walk.wav"
    assert 55 <= metas[1]["started_at"] - t0 <= 56                 # cut inside the quiet second
    total = sum((cfg.queue_dir / f"{i}.pcm").stat().st_size for i in ids)
    assert total == 150 * 16000 * 2                                # nothing dropped, resampled to 16 kHz
    # The same recording again gets new ids instead of overwriting.
    again = queue_recording(cfg, stereo, sr, t0, "walk.wav", now=t0 + 9999)
    assert not set(again) & set(ids)
    with pytest.raises(ValueError):
        queue_recording(cfg, x[: 3 * sr], sr)
    assert parse_when("") is None and parse_when("1700000000") == 1700000000.0


def test_summary_text_and_cleaning():
    from omibridge.summarize import clean, transcript_text

    doc = {"segments": [{"speaker": "Ana", "start": 65.2, "end": 70, "text": "hello there"},
                        {"speaker": "unknown1", "start": 71, "end": 72, "text": ""},
                        {"speaker": "unknown1", "start": 3725, "end": 3730, "text": "bye"}]}
    assert transcript_text(doc) == "[1:05] Ana: hello there\n[62:05] unknown1: bye"
    assert transcript_text({"segments": [{"speaker": "a", "start": 0, "text": ""}]}) == ""
    long = {"segments": [{"speaker": "a", "start": i, "text": "word " * 20} for i in range(200)]}
    cut = transcript_text(long, max_chars=2000)
    assert len(cut) < 2100 and "middle left out" in cut
    out = clean({"title": " Budget review. ", "overview": " ok ", "emoji": "📊", "category": "Sorcery",
                 "action_items": ["send the sheet", " ", None]})
    assert out == {"title": "Budget review", "overview": "ok", "emoji": "📊", "category": "other",
                   "action_items": ["send the sheet"]}


# -- optional GPU segmentation: fails safe -------------------------------------------
class _FakeOnnxSeg:
    window = 16
    def scores(self, chunks):
        return np.zeros((len(chunks), 3, 2), np.float32)
    def binarize(self, y, speech_bias=0.0):
        return y.argmax(-1)


def test_gpu_off_by_default_and_fallbacks():
    from omibridge.gpu_segmentation import maybe_gpu

    onnx = _FakeOnnxSeg()
    assert maybe_gpu(onnx, "cpu") is onnx                                   # default: untouched
    def missing(_):
        raise ImportError("No module named 'diarizen'")
    assert maybe_gpu(onnx, "mps", loader=missing) is onnx                   # packages missing -> CPU


def test_gpu_wrapper_batches_and_falls_back_on_error():
    torch = pytest.importorskip("torch")
    from omibridge.gpu_segmentation import MpsSegmentation

    class Net(torch.nn.Module):
        def forward(self, x):                      # (B, 1, W) -> (B, 3, 2)
            return torch.ones(x.shape[0], 3, 2) * x.mean()
    seg = MpsSegmentation(_FakeOnnxSeg(), Net(), device="cpu", batch=2)
    out = seg.scores(np.ones((5, 16), np.float32))
    assert out.shape == (5, 3, 2) and np.allclose(out, 1.0)               # 3 batches of <= 2
    assert seg.window == 16 and seg.device_name == "cpu"                  # delegated to the ONNX model
    class Broken(torch.nn.Module):
        def forward(self, x):
            raise RuntimeError("MPS out of memory")
    seg = MpsSegmentation(_FakeOnnxSeg(), Broken(), device="cpu")
    assert np.all(seg.scores(np.ones((2, 16), np.float32)) == 0)           # ONNX answer instead
    assert seg.failed and seg.device_name == "cpu (GPU failed)"
