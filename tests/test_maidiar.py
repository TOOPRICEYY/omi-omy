"""MAI diarization path: per-piece labels linked across pieces, naming, fallback, labeling.
Uses fake audio (one tone per speaker), a fake voiceprint model and a fake transcriber."""
import json
import threading

import numpy as np
import pytest

from attrib.audio import SR
from attrib.maidiar import clear_spans, mai_clusters, turns_from_words
from attrib.types import Region, Turn, Word
from omibridge.config import Config
from omibridge.engine import COHORT, Engine

TONES = {"A": 220.0, "B": 550.0, "C": 900.0}


def fake_embed(x: np.ndarray) -> np.ndarray:
    """192-d 'voiceprint': energy in 192 frequency bands, so each tone is its own voice."""
    spec = np.abs(np.fft.rfft(x, n=SR))[:1920]
    v = spec.reshape(192, 10).sum(1) + 1e-6
    return v / np.linalg.norm(v)


def audio(script):
    """script: [(who, seconds)] -> signal and the true turns."""
    parts, truth, t = [], [], 0.0
    for who, dur in script:
        n = int(dur * SR)
        parts.append(0.3 * np.sin(2 * np.pi * TONES[who] * np.arange(n) / SR).astype(np.float32))
        truth.append((t, t + dur, who))
        t += dur
    return np.concatenate(parts), truth


def words_for(truth, label_of, piece_of=lambda t: 0):
    out = []
    for a, b, who in truth:
        t = a
        while t + 0.4 <= b:
            out.append(Word("w", t, t + 0.35, f"p{piece_of(t)}:{label_of(who, piece_of(t))}"))
            t += 0.4
    return out


def test_turns_from_words_and_clear_spans():
    ws = [Word("a", 0, 0.5, "p0:0"), Word("b", 0.6, 1.0, "p0:0"), Word("c", 2.5, 3.0, "p0:0"),
          Word("d", 3.1, 3.5, "p0:1")]
    t = turns_from_words(ws)
    assert [(x.start, x.end, x.speaker) for x in t] == [(0, 1.0, "p0:0"), (2.5, 3.0, "p0:0"), (3.1, 3.5, "p0:1")]
    spans = clear_spans([Turn(0, 10, "x"), Turn(20, 21, "x"), Turn(30, 60, "x")], max_pool=25)
    assert spans[0] == (30.2, 55.2) and len(spans) == 1          # longest first, trimmed, capped
    assert clear_spans([Turn(0, 1.0, "x")]) == []                 # too short to be clear


def test_swapped_labels_across_pieces_are_the_same_two_people():
    x, truth = audio([("A", 6), ("B", 6), ("A", 5), ("B", 5), ("A", 6), ("B", 6)])
    # Two MAI requests (cut at 22 s); the second calls A "1" and B "0".
    piece = lambda t: 0 if t < 22 else 1
    label = lambda who, p: {"A": 0, "B": 1}[who] if p == 0 else {"A": 1, "B": 0}[who]
    words = words_for(truth, label, piece)
    turns, clusters = mai_clusters(x, words, fake_embed)
    assert len(clusters) == 2
    by_who = {}
    for a, b, who in truth:
        labs = {t.speaker for t in turns if a <= t.start < b}
        by_who.setdefault(who, set()).update(labs)
    assert len(by_who["A"]) == 1 and len(by_who["B"]) == 1 and by_who["A"] != by_who["B"]
    assert {w.speaker for w in words} == {c.label for c in clusters}   # words relabeled too


def test_same_piece_speakers_are_never_merged():
    # Two labels of one piece with the same voice stay two speakers (MAI said they differ).
    x, truth = audio([("A", 6), ("A", 6)])
    words = words_for(truth, lambda who, p: 0) [:14] + [Word("w", 6.0 + 0.4 * k, 6.35 + 0.4 * k, "p0:1") for k in range(14)]
    _, clusters = mai_clusters(x, words, fake_embed)
    assert len(clusters) == 2


class FakeTranscriber:
    def __init__(self, words):
        self.words, self.calls = words, []

    def __call__(self, wav, phrases=None, diarize=False):
        self.calls.append(diarize)
        return [Word(w.text, w.start, w.end, w.speaker if diarize else None) for w in self.words]


class FakeLocal:
    seg = None

    def __init__(self):
        self.used = 0

    def turns(self, x):
        self.used += 1
        return [Turn(0, len(x) / SR, "S0")]

    def clusters(self, x, turns):
        from attrib.diarize import Cluster
        e = fake_embed(x)
        return [Cluster("S0", turns, e, [e], -20.0, len(x) / SR)]


def engine(tmp_path, transcriber, **env):
    cfg = Config(data_dir=tmp_path, **env)
    cfg.ensure_dirs()
    e = Engine.__new__(Engine)
    from attrib.identify import Gallery
    e.cfg, e.lock, e.transcriber, e.summarizer = cfg, threading.Lock(), transcriber, None
    e.vad = lambda x: [Region(0, len(x) / SR)]
    e.embedder, e.diarizer = fake_embed, FakeLocal()
    e.cohort, e.gallery = np.load(COHORT), Gallery()
    return e


def run(e, x, cid="c1"):
    p = e.cfg.queue_dir / f"{cid}.pcm"
    p.write_bytes((x * 32767).astype("<i2").tobytes())
    return e.process(p, {"id": cid, "sample_rate": SR})


def test_mai_mode_naming_short_speaker_and_labeling(tmp_path):
    x, truth = audio([("A", 8), ("B", 7), ("C", 1.2), ("A", 8), ("B", 7)])
    words = words_for(truth, lambda who, p: {"A": 0, "B": 1, "C": 2}[who])
    e = engine(tmp_path, FakeTranscriber(words))
    for k in range(4):                              # enroll A and B from other recordings
        e.gallery.add("Alice", fake_embed(audio([("A", 4 + k)])[0]))
        e.gallery.add("Bob", fake_embed(audio([("B", 4 + k)])[0]))
    doc = run(e, x)
    assert e.transcriber.calls == [True] and e.diarizer.used == 0
    assert json.loads((tmp_path / "conversations/c1/meta.json").read_text())["diarizer"] == "mai"
    names = {s["name"] for s in doc["speakers"]}
    assert {"Alice", "Bob"} <= names
    assert any(n.startswith("unknown") for n in names)      # C: 1.2 s, never named
    short = next(s["name"] for s in doc["speakers"] if s["name"].startswith("unknown"))
    # Labeling still pins and learns.
    doc = e.label("c1", short, "Carol")
    assert "Carol" in {s["name"] for s in doc["speakers"]} and "Carol" in e.gallery.people


def test_falls_back_to_local_when_mai_gives_no_speakers(tmp_path):
    x, truth = audio([("A", 5), ("B", 5)])
    unlabeled = [Word(w.text, w.start, w.end, None) for w in words_for(truth, lambda who, p: 0)]

    class NoSpeakers(FakeTranscriber):
        def __call__(self, wav, phrases=None, diarize=False):
            self.calls.append(diarize)
            return list(self.words)

    e = engine(tmp_path, NoSpeakers(unlabeled))
    doc = run(e, x)
    assert e.diarizer.used == 1 and doc["segments"]
    assert json.loads((tmp_path / "conversations/c1/meta.json").read_text())["diarizer"] == "local"


def test_falls_back_to_local_when_mai_fails(tmp_path):
    def broken(wav, phrases=None, diarize=False):
        raise RuntimeError("boom")

    e = engine(tmp_path, broken)
    doc = run(e, audio([("A", 5)])[0])
    assert e.diarizer.used == 1 and "boom" in doc["error"]


def test_local_mode_does_not_ask_mai_for_speakers(tmp_path, monkeypatch):
    monkeypatch.setenv("OMIBRIDGE_DIARIZER", "local")
    x, truth = audio([("A", 5)])
    e = engine(tmp_path, FakeTranscriber(words_for(truth, lambda who, p: 0)))
    run(e, x)
    assert e.transcriber.calls == [False] and e.diarizer.used == 1


def test_transcriber_retries_rate_limits(monkeypatch):
    import io
    import requests
    import soundfile as sf
    from attrib.transcribe import OpenRouterTranscriber

    class R:
        def __init__(self, code, body=None):
            self.status_code, self.headers, self._b = code, {}, body or {}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(str(self.status_code))

        def json(self):
            return self._b

    replies = [R(429), R(503), R(200, {"words": [{"word": "hi", "start": 0.1, "end": 0.3, "speaker": 1}]})]
    monkeypatch.setattr(requests, "post", lambda *a, **k: replies.pop(0))
    t = OpenRouterTranscriber(key="k")
    waits = []
    t.sleep = waits.append
    buf = io.BytesIO()
    sf.write(buf, np.zeros(SR, np.float32), SR, format="WAV")
    words = t(buf.getvalue(), diarize=True)
    assert [(w.text, w.speaker) for w in words] == [("hi", "p0:1")] and waits == [5.0, 15.0]


def test_naming_from_a_stale_page_names_the_voice_that_was_shown(tmp_path):
    x, truth = audio([("A", 8), ("B", 7), ("A", 8), ("B", 7)])
    e = engine(tmp_path, FakeTranscriber(words_for(truth, lambda who, p: {"A": 0, "B": 1}[who])))
    page = lambda cid: json.loads((tmp_path / "conversations" / cid / "conversation.json").read_text())
    by_cluster = lambda doc: {l: s["name"] for s in doc["speakers"] for l in s["clusters"]}
    run(e, x, "c1"); run(e, x, "c2"); run(e, x, "c3")
    a, b = by_cluster(page("c1"))["S0"], by_cluster(page("c1"))["S1"]
    assert {a, b} == {"unknown1", "unknown2"}
    # Naming A in c2 teaches the gallery A's voice. c1 and c3 keep their stored pages (old names),
    # but with the gallery as it is now B would be the first unknown voice there: "unknown1".
    e.label("c2", by_cluster(page("c2"))["S0"], "Alice")
    assert by_cluster(page("c1")) == {"S0": a, "S1": b}
    # By name alone, the name means what the stored page shows: A, not the voice now called that.
    doc = e.label("c1", a, "Alice")
    assert by_cluster(doc) == {"S0": "Alice", "S1": "unknown1"} and len(e.gallery.people) == 1
    # By clusters (what the page and the app send), whatever the names have become since.
    stale = page("c3")
    e.label("c3", "unknown1", "Bob", clusters=[l for s in stale["speakers"] if "S1" in s["clusters"] for l in s["clusters"]])
    assert by_cluster(page("c3"))["S1"] == "Bob" and by_cluster(e.render("c3"))["S0"] == "Alice"
    with pytest.raises(KeyError):
        e.label("c3", "unknown1", "Carol", clusters=["nope"])


def test_lines_do_not_depend_on_names(tmp_path):
    # The same voice split into two clusters (two MAI speakers), talking back to back.
    x, truth = audio([("A", 8), ("A", 8), ("B", 7)])
    ws = sum((words_for(truth[i:i + 1], lambda who, p, n=n: n) for i, n in enumerate([0, 2, 1])), [])
    e = engine(tmp_path, FakeTranscriber(ws))
    before = run(e, x, "c1")
    cut = [(s["start"], s["end"], s["cluster"]) for s in before["segments"]]
    assert all(s["cluster"] for s in before["segments"])
    one, two = before["segments"][:2]
    assert one["speaker"] == two["speaker"] and one["cluster"] != two["cluster"]         # one name, two lines
    first = before["segments"][0]
    e.label("c1", None, "Alice", clusters=[l for s in before["speakers"] for l in s["clusters"]
                                           if s["name"] == first["speaker"]])
    after = e.render("c1")
    assert [(s["start"], s["end"], s["cluster"]) for s in after["segments"]] == cut      # same lines, same order
    assert not list((tmp_path / "conversations/c1").glob("*.tmp"))


def test_rerender_all_refreshes_pages(tmp_path):
    x, truth = audio([("A", 8), ("B", 7), ("A", 8), ("B", 7)])
    e = engine(tmp_path, FakeTranscriber(words_for(truth, lambda who, p: {"A": 0, "B": 1}[who])))
    run(e, x, "c1"); run(e, x, "c2")
    a_name = next(s["name"] for s in e.render("c2")["speakers"] if "S0" in s["clusters"])
    e.label("c2", a_name, "Alice")
    assert e.rerender_all(skip="c2") == 1
    page = json.loads((tmp_path / "conversations/c1/conversation.json").read_text())
    assert "Alice" in {s["name"] for s in page["speakers"]}
