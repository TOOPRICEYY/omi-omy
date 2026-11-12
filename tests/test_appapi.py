"""Contract tests for the API the forked Omi app talks to. No models: a fake engine
stands in for diarization and transcription, and Opus is encoded with PyAV."""
import base64
import json
import struct
import time
from fractions import Fraction

import numpy as np
import pytest
from fastapi.testclient import TestClient

from omibridge.appapi import AppAPI, to_omi
from omibridge.appauth import Tokens
from omibridge.config import Config
from omibridge.server import create_app

SR = 16000
SECRET = "s3cret-" + "x" * 40
P = f"/app/{SECRET}"     # every app API call goes under the secret path


class FakeGallery:
    people = {"Ana": [1], "Bo": [1]}


class FakeEngine:
    """Loud audio counts as speech; process() writes a two-speaker conversation."""

    transcriber = None
    summarizer = None

    def __init__(self, cfg):
        self.cfg = cfg
        self.gallery = FakeGallery()

    def speech_seconds(self, x):
        return len(x) / SR if np.abs(x).mean() > 0.05 else 0.0

    def process(self, pcm_path, meta):
        n = pcm_path.stat().st_size // 2
        out = self.cfg.conv_dir / meta["id"]
        out.mkdir(parents=True)
        doc = {"id": meta["id"], "started_at": meta["started_at"], "duration": n / SR, "speech_seconds": n / SR,
               "error": None, "speakers": [{"name": "Ana"}, {"name": "unknown1"}],
               "summary": {"title": "Lunch plans", "overview": "They agreed on lunch.", "emoji": "🍜",
                           "category": "social", "action_items": ["Book a table"]},
               "segments": [{"speaker": "Ana", "start": 0.0, "end": 2.0, "text": "Lunch at noon?"},
                            {"speaker": "unknown1", "start": 2.5, "end": 4.0, "text": "Sure."}]}
        (out / "meta.json").write_text(json.dumps(meta))
        (out / "conversation.json").write_text(json.dumps(doc))
        return doc

    def summarize(self, cid):
        return None

    def people(self):
        return [{"name": p, "voiceprints": 1} for p in sorted(self.gallery.people)]


@pytest.fixture
def env(tmp_path):
    now = [1_800_000_000.0]
    cfg = Config(data_dir=tmp_path, token="t", idle_seconds=10, min_speech_seconds=2, app_secret=SECRET,
                 app_uid="owner", wearer="Ana")
    bridge = create_app(cfg, engine=FakeEngine(cfg), start_workers=False, clock=lambda: now[0])
    api = AppAPI(cfg, bridge, clock=lambda: now[0])
    client = TestClient(api.public)        # the public port, as the tunnel sees it
    return cfg, now, bridge, api, client


def sign_in(cfg, client, now):
    r = client.post(f"{P}/v1/auth/local-dev/custom-token", data={"uid": "local-dev-user"})
    assert r.status_code == 200, r.text
    custom = r.json()["custom_token"]
    r = client.post("/identitytoolkit.googleapis.com/v1/accounts:signInWithCustomToken?key=x",
                    json={"token": custom, "returnSecureToken": True})
    assert r.status_code == 200, r.text
    return r.json()


def auth(idt):
    return {"Authorization": f"Bearer {idt}"}


def opus_packets(x: np.ndarray) -> list[bytes]:
    import av

    enc = av.CodecContext.create("libopus", "w")
    enc.sample_rate, enc.layout, enc.format = SR, "mono", "s16"
    enc.options = {"frame_duration": "10", "application": "voip"}
    enc.bit_rate = 32000
    enc.open()
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    out = []
    for i in range(0, len(pcm) - 159, 160):
        f = av.AudioFrame.from_ndarray(pcm[i:i + 160].reshape(1, -1), format="s16", layout="mono")
        f.sample_rate, f.pts, f.time_base = SR, i, Fraction(1, SR)
        out += [bytes(p) for p in enc.encode(f)]
    out += [bytes(p) for p in enc.encode(None)]
    return out


def tone(seconds, level=0.3):
    t = np.arange(int(seconds * SR)) / SR
    return (np.sin(2 * np.pi * 220 * t) * level).astype(np.float32)


# -- sign-in --------------------------------------------------------------------
def test_sign_in_flow_and_token_check(env):
    cfg, now, _, api, client = env
    assert client.get("/").status_code == 200                       # emulator reachability probe
    # Without the secret path there is no API at all: the same 404 as any unknown path.
    unknown = client.post("/nothing-here", data={"uid": "x"})
    for path in ("/v1/auth/local-dev/custom-token", "/app/wrong/v1/auth/local-dev/custom-token",
                 f"/app/{SECRET[:-1]}/v1/auth/local-dev/custom-token"):
        r = client.post(path, data={"uid": "x"})
        assert (r.status_code, r.content) == (unknown.status_code, unknown.content) == (404, unknown.content)
    s = sign_in(cfg, client, now)
    assert s["localId"] == "owner" and s["expiresIn"] == "3600"
    claims = json.loads(base64.urlsafe_b64decode(s["idToken"].split(".")[1] + "=="))
    assert claims["user_id"] == "owner" and claims["exp"] > claims["iat"]
    assert claims["firebase"]["sign_in_provider"] == "custom"
    r = client.post("/identitytoolkit.googleapis.com/v1/accounts:lookup?key=x", json={"idToken": s["idToken"]})
    assert r.json()["users"][0]["localId"] == "owner"
    # Refresh, JSON and form bodies.
    r = client.post("/securetoken.googleapis.com/v1/token?key=x",
                    json={"grantType": "refresh_token", "refreshToken": s["refreshToken"]})
    assert r.status_code == 200 and r.json()["user_id"] == "owner"
    r = client.post("/securetoken.googleapis.com/v1/token?key=x",
                    data={"grant_type": "refresh_token", "refresh_token": s["refreshToken"]})
    assert r.status_code == 200
    # API routes need a token this bridge made.
    assert client.get(f"{P}/v1/users/onboarding").status_code == 401
    forged = Tokens("other-secret", "owner").id_token()
    assert client.get(f"{P}/v1/users/onboarding", headers=auth(forged)).status_code == 401
    assert client.get(f"{P}/v1/users/onboarding", headers=auth(s["refreshToken"])).status_code == 401
    r = client.get(f"{P}/v1/users/onboarding", headers=auth(s["idToken"]))
    assert r.status_code == 200 and r.json()["completed"] is True
    # Expired ID tokens are refused.
    now[0] += 3601
    assert client.get(f"{P}/v1/users/onboarding", headers=auth(s["idToken"])).status_code == 401
    # A bad custom token is refused by the emulator endpoint.
    r = client.post("/identitytoolkit.googleapis.com/v1/accounts:signInWithCustomToken",
                    json={"token": forged})
    assert r.status_code == 400


def test_other_routes_are_404(env):
    cfg, now, _, _, client = env
    idt = sign_in(cfg, client, now)["idToken"]
    for path in ("/v3/memories", "/v2/messages", "/v1/apps", "/v1/users/me/subscription", "/v1/goals"):
        assert client.get(P + path, headers=auth(idt)).status_code == 404
    assert client.get(f"{P}/v1/account/cutover/control", headers=auth(idt)).json()["product_traffic_allowed"] is True


# -- conversations -------------------------------------------------------------------
def test_conversation_mapping():
    doc = {"id": "c1", "started_at": 1_800_000_000.0, "duration": 10, "speakers": [{"name": "Ana"}, {"name": "Bo"},
           {"name": "unknown1"}], "summary": {"title": "T", "overview": "O", "emoji": "🎯", "category": "work",
           "action_items": ["Do it"]},
           "segments": [{"speaker": "Ana", "start": 0, "end": 1, "text": "a"},
                        {"speaker": "Bo", "start": 1, "end": 2, "text": "b"},
                        {"speaker": "unknown1", "start": 2, "end": 3, "text": "c"}]}
    c = to_omi(doc, wearer="Ana")
    assert c["id"] == "c1" and c["created_at"].endswith("Z") and c["finished_at"] and c["status"] == "completed"
    assert c["structured"] == {"title": "T", "overview": "O", "emoji": "🎯", "category": "work",
                               "action_items": [{"description": "Do it", "completed": False}], "events": []}
    a, b, u = c["transcript_segments"]
    assert a["is_user"] and a["person_id"] is None and a["speaker"] == "SPEAKER_00"
    assert not b["is_user"] and b["person_id"] == "p-bo" and b["speaker_id"] == 1
    assert u["person_id"] is None and u["speaker_id"] == 2
    bare = to_omi({"id": "c2", "started_at": None, "segments": []}, None)
    assert bare["structured"]["title"] and bare["started_at"] is None and "finished_at" in bare


def test_list_detail_delete_people(env):
    cfg, now, bridge, api, client = env
    idt = sign_in(cfg, client, now)["idToken"]
    # Queue a conversation the way the importer does, then process it.
    from omibridge.importer import queue_recording

    (cid,) = queue_recording(cfg, tone(5), SR, started_at=now[0] - 60, now=now[0])
    listed = client.get(f"{P}/v1/conversations", headers=auth(idt)).json()
    assert [c["status"] for c in listed] == ["processing"]
    assert client.get(f"{P}/v1/conversations?statuses=in_progress", headers=auth(idt)).json() == []
    bridge.state.process_one(cid)
    (c,) = client.get(f"{P}/v1/conversations?limit=10&offset=0&statuses=", headers=auth(idt)).json()
    assert c["id"] == cid and c["structured"]["title"] == "Lunch plans" and len(c["transcript_segments"]) == 2
    assert client.get(f"{P}/v1/conversations/{cid}", headers=auth(idt)).json()["id"] == cid
    assert client.get(f"{P}/v1/conversations/nope", headers=auth(idt)).status_code == 404
    people = client.get(f"{P}/v1/users/people", headers=auth(idt)).json()
    assert [p["name"] for p in people] == ["Bo"] and people[0]["id"] == "p-bo"   # the wearer is not a "person"
    assert client.delete(f"{P}/v1/conversations/{cid}", headers=auth(idt)).status_code == 204
    assert (cfg.trash_dir / cid / "conversation.json").exists()                     # moved, not erased
    assert client.get(f"{P}/v1/conversations", headers=auth(idt)).json() == []


# -- live socket ----------------------------------------------------------------------
def test_listen_socket_to_conversation(env):
    cfg, now, bridge, api, client = env
    idt = sign_in(cfg, client, now)["idToken"]
    with pytest.raises(Exception):
        with client.websocket_connect(f"{P}/v4/listen?codec=opus&sample_rate=16000") as ws:
            ws.receive_text()                      # no token: refused
    packets = opus_packets(tone(4))
    with client.websocket_connect(f"{P}/v4/listen?codec=opus&sample_rate=16000&uid=owner&language=en",
                                  headers=auth(idt)) as ws:
        assert json.loads(ws.receive_text())["type"] == "service_status"
        ws.send_text(json.dumps({"type": "client_state", "foreground": True}))   # ignored
        for p in packets:
            ws.send_bytes(p)
        # The app's stop button ends the conversation; then the bridge processes it.
        deadline = time.time() + 5
        while not api.sessions.sessions and time.time() < deadline:
            time.sleep(0.05)
        r = client.post(f"{P}/v1/conversations", headers=auth(idt), json={})
        assert r.status_code == 200, r.text
        cid = r.json()["conversation"]["id"]
        assert r.json()["conversation"]["status"] == "processing"
        started = json.loads(ws.receive_text())
        assert started["type"] == "memory_processing_started" and started["memory"]["id"] == cid
        bridge.state.process_one(cid)
        created = json.loads(ws.receive_text())
        assert created["type"] == "memory_created" and created["memory"]["id"] == cid
        assert created["memory"]["transcript_segments"][0]["text"] == "Lunch at noon?"
    assert client.post(f"{P}/v1/conversations", headers=auth(idt), json={}).status_code == 404   # nothing open now
    assert api.coverage.covered(now[0] - 1, now[0]) > 0.9         # (frozen test clock: all audio lands at "now")


def test_listen_socket_silence_gate(env):
    """The phone's silence gate: a pause in the audio hands on the partial chunk, and the
    silence the phone left out is put back, so the recording keeps its real length."""
    cfg, now, bridge, api, client = env
    api.idle_flush = 0.1
    idt = sign_in(cfg, client, now)["idToken"]

    def seconds(at_least):
        deadline = time.time() + 5
        while time.time() < deadline:
            s = api.sessions.sessions.get(api.key("owner"))
            if s is not None and s.samples / SR >= at_least:
                return s.samples / SR
            time.sleep(0.02)
        s = api.sessions.sessions.get(api.key("owner"))
        raise AssertionError(f"have {s.samples / SR if s else None} s, want {at_least}")

    with client.websocket_connect(f"{P}/v4/listen?codec=opus&sample_rate=16000", headers=auth(idt)) as ws:
        assert json.loads(ws.receive_text())["type"] == "service_status"
        ws.send_text(json.dumps({"type": "omiomy_silence_gap", "ms": 4000}))   # before any speech: nothing to fill
        for p in opus_packets(tone(2.5)):
            ws.send_bytes(p)
        assert 2.4 <= seconds(2.4) <= 2.6           # the half-second remainder did not wait for more audio
        for bad in ('{"type": "omiomy_silence_gap", "ms": "x"}', '{"type": "omiomy_silence_gap", "ms": -5}',
                    '{"type": "omiomy_silence_gap", "ms": true}', 'omiomy_silence_gap', '["omiomy_silence_gap"]'):
            ws.send_text(bad)                        # malformed notes are ignored
        ws.send_text(json.dumps({"type": "omiomy_silence_gap", "ms": 3000}))
        assert 5.4 <= seconds(5.4) <= 5.6           # three seconds of silence put back
        ws.send_text(json.dumps({"type": "omiomy_silence_gap", "ms": 3_600_000}))
        assert 15.4 <= seconds(15.4) <= 15.6        # never more than the idle timeout (10 s here)
        for p in opus_packets(tone(1)):
            ws.send_bytes(p)
        assert 16.3 <= seconds(16.3) <= 16.6
    s = api.sessions.sessions[api.key("owner")]
    assert 3.3 <= s.speech_seconds <= 3.7           # the put-back silence is not speech


# -- offline sync ------------------------------------------------------------------------
def wal_file(packets: list[bytes]) -> bytes:
    return b"".join(struct.pack("<I", len(p)) + p for p in packets)


def test_sync_upload_decode_and_dedupe(env):
    cfg, now, bridge, api, client = env
    idt = sign_in(cfg, client, now)["idToken"]
    t0 = int(now[0] - 3600)
    # 6 s of speech recorded offline, then a later file that the bridge already heard live.
    api.coverage.add(t0 + 100, t0 + 110)
    files = [("files", (f"audio_omi_opus_16000_1_fs160_{t0}.bin", wal_file(opus_packets(tone(6))))),
             ("files", (f"audio_omi_opus_16000_1_fs160_{(t0 + 100) * 1000}.bin", wal_file(opus_packets(tone(6)))))]
    r = client.post(f"{P}/v2/sync-local-files", headers=auth(idt), files=files)
    assert r.status_code == 202, r.text
    job = r.json()
    assert {"job_id", "status", "total_files", "total_segments", "poll_after_ms"} <= set(job)
    deadline = time.time() + 10
    while True:
        st = client.get(f"{P}/v2/sync-local-files/{job['job_id']}", headers=auth(idt)).json()
        if st["status"] != "processing" or time.time() > deadline:
            break
        time.sleep(0.05)
    assert st["status"] == "completed" and st["result"]["failed_segments"] == 0
    new = st["result"]["new_memories"]
    assert len(new) == 1                                            # the live-covered file was skipped
    meta = json.loads((cfg.queue_dir / f"{new[0]}.json").read_text())
    assert abs(meta["started_at"] - t0) < 4                         # start time from the file name
    # The same upload again: everything is covered now, nothing new.
    r = client.post(f"{P}/v2/sync-local-files", headers=auth(idt), files=files)
    jid = r.json()["job_id"]
    deadline = time.time() + 10
    while (st := client.get(f"{P}/v2/sync-local-files/{jid}", headers=auth(idt)).json())["status"] == "processing":
        assert time.time() < deadline
        time.sleep(0.05)
    assert st["result"]["new_memories"] == []
    # Two recordings far apart in one upload stay two conversations.
    t1 = t0 + 7200
    files = [("files", (f"audio_omi_opus_16000_1_fs160_{t1}.bin", wal_file(opus_packets(tone(6))))),
             ("files", (f"audio_omi_opus_16000_1_fs160_{t1 + 1800}.bin", wal_file(opus_packets(tone(6)))))]
    jid = client.post(f"{P}/v2/sync-local-files", headers=auth(idt), files=files).json()["job_id"]
    deadline = time.time() + 10
    while (st := client.get(f"{P}/v2/sync-local-files/{jid}", headers=auth(idt)).json())["status"] == "processing":
        assert time.time() < deadline
        time.sleep(0.05)
    starts = sorted(json.loads((cfg.queue_dir / f"{c}.json").read_text())["started_at"]
                    for c in st["result"]["new_memories"])
    assert len(starts) == 2 and abs(starts[0] - t1) < 4 and abs(starts[1] - (t1 + 1800)) < 4
    # Bad names are refused outright (terminal for the app).
    bad = client.post(f"{P}/v2/sync-local-files", headers=auth(idt), files=[("files", ("x.wav", b"1234"))])
    assert bad.status_code == 400
    assert client.get(f"{P}/v2/sync-local-files/unknown", headers=auth(idt)).status_code == 404


# -- public-port protections ---------------------------------------------------------------
def test_custom_token_single_use_and_short_lived(env):
    cfg, now, _, api, client = env
    custom = client.post(f"{P}/v1/auth/local-dev/custom-token").json()["custom_token"]
    url = "/identitytoolkit.googleapis.com/v1/accounts:signInWithCustomToken"
    assert client.post(url, json={"token": custom}).status_code == 200
    assert client.post(url, json={"token": custom}).status_code == 400          # used up
    late = client.post(f"{P}/v1/auth/local-dev/custom-token").json()["custom_token"]
    now[0] += 61
    assert client.post(url, json={"token": late}).status_code == 400            # expired


def test_api_needs_secret_and_token(env):
    cfg, now, _, api, client = env
    idt = sign_in(cfg, client, now)["idToken"]
    assert client.get("/v1/conversations", headers=auth(idt)).status_code == 404       # token, no secret
    assert client.get("/app/nope/v1/conversations", headers=auth(idt)).status_code == 404
    assert client.get(f"{P}/v1/conversations").status_code == 401                      # secret, no token
    assert client.get(f"{P}/v1/conversations", headers=auth(idt)).status_code == 200
    assert client.head(f"{P}/v1/health").status_code == 200                           # connectivity probe
    assert client.head(f"{P}/v9/unknown").status_code == 404
    assert client.post("/omi/audio/wrong?uid=x&sample_rate=16000", content=b"").status_code == 403   # webhook kept


def test_body_caps(env):
    cfg, now, _, api, client = env
    big = b"x" * (300 * 1024)
    r = client.post("/identitytoolkit.googleapis.com/v1/accounts:lookup", content=big,
                    headers={"content-type": "application/json"})
    assert r.status_code == 413


def test_failed_checks_block_the_address(env):
    cfg, now, _, api, client = env
    hdr = {"CF-Connecting-IP": "203.0.113.9"}
    for _ in range(30):
        client.get("/app/guess/v1/conversations", headers=hdr)
    assert client.get("/app/guess/v1/conversations", headers=hdr).status_code == 429  # blocked
    assert client.get("/", headers=hdr).status_code == 429                            # sign-in calls too
    assert client.get(f"{P}/v1/health", headers=hdr).status_code == 200               # never with the right secret
    assert client.get("/app/guess/v1/health", headers={"CF-Connecting-IP": "198.51.100.1"}).status_code == 404
    now[0] += 901
    assert client.get("/app/guess/v1/conversations", headers=hdr).status_code == 404  # block expires


def test_right_secret_is_never_locked_out(env):
    """A phone with a stale token must not block itself (it may be recording)."""
    cfg, now, _, api, client = env
    hdr = {"CF-Connecting-IP": "203.0.113.10"}
    for _ in range(40):
        assert client.get(f"{P}/v1/conversations", headers=hdr | auth("stale")).status_code == 401
    assert client.get("/", headers=hdr).status_code == 200
    idt = sign_in(cfg, client, now)["idToken"]
    assert client.get(f"{P}/v1/conversations", headers=hdr | auth(idt)).status_code == 200
    r = client.post("/identitytoolkit.googleapis.com/v1/accounts:lookup", content=b"{not json",
                    headers={"content-type": "application/json"})
    assert r.status_code == 400


def test_log_redaction():
    import logging

    from omibridge.__main__ import _RedactToken

    rec = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s"', ("1.2.3.4", "GET", f"{P}/v1/x"),
                            None)
    _RedactToken("tok", SECRET).filter(rec)
    assert SECRET not in rec.getMessage() and "/app/<secret>/v1/x" in rec.getMessage()


def test_ios_sdk_v3_endpoints(env):
    """The Firebase iOS SDK (11.x) calls the v3 relyingparty names."""
    cfg, now, _, api, client = env
    custom = client.post(f"{P}/v1/auth/local-dev/custom-token").json()["custom_token"]
    base = "/www.googleapis.com/identitytoolkit/v3/relyingparty"
    r = client.post(f"{base}/verifyCustomToken?key=k", json={"token": custom, "returnSecureToken": True})
    assert r.status_code == 200 and r.json()["idToken"]
    r = client.post(f"{base}/getAccountInfo?key=k", json={"idToken": r.json()["idToken"]})
    assert r.json()["users"][0]["localId"] == "owner"
    assert client.post(f"{base}/signupNewUser?key=k", json={}).status_code == 400


def test_language_round_trip(env):
    """Onboarding's language step: PATCH must answer status ok, GET reads it back."""
    cfg, now, _, api, client = env
    idt = sign_in(cfg, client, now)["idToken"]
    assert client.get(f"{P}/v1/users/language", headers=auth(idt)).json() == {"language": "en"}
    r = client.patch(f"{P}/v1/users/language", headers=auth(idt), json={"language": "en-US"})
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["single_language_mode"] is False
    assert client.get(f"{P}/v1/users/language", headers=auth(idt)).json() == {"language": "en-US"}
    assert client.patch(f"{P}/v1/users/language", headers=auth(idt), json={}).status_code == 400
    assert client.patch(f"{P}/v1/users/language", json={"language": "de"}).status_code == 401
    assert client.put(f"{P}/v1/users/time-zone", headers=auth(idt), json={"time_zone": "America/Chicago"}).status_code == 200


def test_display_name_kept_without_new_session(env):
    """Onboarding's name step: updateProfile must not hand out new tokens (the app would
    discard the onboarding save made under the old session), and lookup returns the name."""
    cfg, now, _, api, client = env
    s = sign_in(cfg, client, now)
    base = "/www.googleapis.com/identitytoolkit/v3/relyingparty"
    r = client.post(f"{base}/setAccountInfo?key=k", json={"idToken": s["idToken"], "displayName": "Gabriel Price",
                                                          "returnSecureToken": True})
    assert r.status_code == 200 and r.json()["displayName"] == "Gabriel Price"
    assert "idToken" not in r.json() and "refreshToken" not in r.json()
    u = client.post(f"{base}/getAccountInfo?key=k", json={"idToken": s["idToken"]}).json()["users"][0]
    assert u["displayName"] == "Gabriel Price"
    assert client.post(f"{base}/setAccountInfo?key=k", json={"idToken": "bad"}).status_code == 400


# -- naming a speaker from the app (real Engine on fake audio and voiceprints) ---------
def naming_env(tmp_path, wearer=None):
    from tests.test_maidiar import FakeTranscriber, audio, engine, run, words_for
    x, truth = audio([("A", 8), ("B", 7), ("A", 8), ("B", 7)])
    e = engine(tmp_path, FakeTranscriber(words_for(truth, lambda who, p: {"A": 0, "B": 1}[who])))
    e.cfg.app_secret, e.cfg.app_uid, e.cfg.wearer, e.cfg.token = SECRET, "owner", wearer, "t"
    run(e, x, "20261112-100000-conv")
    now = [1_800_000_000.0]
    bridge = create_app(e.cfg, engine=e, start_workers=False, clock=lambda: now[0])
    api = AppAPI(e.cfg, bridge, clock=lambda: now[0])
    client = TestClient(api.public)
    return e, api, client, auth(sign_in(e.cfg, client, now)["idToken"]), x


def test_app_names_a_voice_and_bridge_learns_it(tmp_path):
    e, api, client, h, _ = naming_env(tmp_path)
    cid = "20261112-100000-conv"
    conv = client.get(f"{P}/v1/conversations/{cid}", headers=h).json()
    lines = conv["transcript_segments"]
    assert all(not l["is_user"] and l["person_id"] is None for l in lines)
    # People sheet: "add person", then assign a line's voice to them (Omi's legacy route,
    # which the app uses when the recording-speakers route answers 404).
    assert client.patch(f"{P}/v1/conversations/{cid}/recording-speakers", headers=h, json={}).status_code == 404
    r = client.post(f"{P}/v1/users/people", headers=h, json={"name": "Bob"})
    assert r.status_code == 200 and r.json()["name"] == "Bob"
    assert client.post(f"{P}/v1/users/people", headers=h, json={"name": "bob"}).json()["id"] == r.json()["id"]
    pid = r.json()["id"]
    b_line = next(l for l in lines if l["start"] >= 8 and l["start"] < 15)
    r = client.patch(f"{P}/v1/conversations/{cid}/assign-speaker/{b_line['speaker_id']}",
                     params={"assign_type": "person_id", "value": pid}, headers=h,
                     json={"segment_ids": [b_line["id"]], "assign_type": "person_id", "value": pid})
    assert r.status_code == 200, r.text
    named = [l for l in r.json()["transcript_segments"] if l["person_id"] == pid]
    assert named and all(8 <= l["start"] < 15 or 23 <= l["start"] < 30 for l in named)   # every line of B's voice
    assert "Bob" in e.gallery.people                                                     # voice learned
    assert pid in [p["id"] for p in client.get(f"{P}/v1/users/people", headers=h).json()]
    # "This is me" through the bulk route: the wearer name comes from the app profile.
    (e.cfg.app_dir / "profile.json").write_text(json.dumps({"displayName": "Gabriel"}))
    a_line = next(l for l in r.json()["transcript_segments"] if l["start"] < 8)
    r = client.patch(f"{P}/v1/conversations/{cid}/segments/assign-bulk", headers=h,
                     json={"segment_ids": [a_line["id"]], "assign_type": "is_user", "value": "true"})
    assert r.status_code == 200, r.text
    mine = [l for l in r.json()["transcript_segments"] if l["is_user"]]
    assert mine and all(l["start"] < 8 or 15 <= l["start"] < 23 for l in mine)
    assert e.cfg.wearer == "Gabriel" and "Gabriel" in e.gallery.people
    assert "Gabriel" not in [p["name"] for p in client.get(f"{P}/v1/users/people", headers=h).json()]
    assert json.loads((e.cfg.app_dir / "wearer.json").read_text())["name"] == "Gabriel"


def test_app_naming_rejects_bad_input(tmp_path):
    e, api, client, h, _ = naming_env(tmp_path, wearer="Gabriel")
    cid = "20261112-100000-conv"
    base = f"{P}/v1/conversations/{cid}"
    assert client.patch(f"{base}/assign-speaker/0", params={"assign_type": "person_id", "value": "p-nobody"},
                        headers=h).status_code == 404
    assert client.patch(f"{base}/assign-speaker/9", params={"assign_type": "is_user", "value": "true"},
                        headers=h).status_code == 409
    assert client.patch(f"{base}/assign-speaker/0", params={"assign_type": "bogus"}, headers=h).status_code == 400
    assert client.patch(f"{base}/segments/assign-bulk", headers=h,
                        json={"segment_ids": ["other-0"], "assign_type": "is_user", "value": "true"}).status_code == 404
    assert client.patch(f"{P}/v1/conversations/nope/assign-speaker/0", params={"assign_type": "is_user", "value": "true"},
                        headers=h).status_code == 404
    assert client.post(f"{P}/v1/users/people", headers=h, json={"name": "unknown2"}).status_code == 400
    assert client.patch(f"{base}/assign-speaker/0", params={"assign_type": "is_user", "value": "true"}).status_code == 401
    # The wearer set in the environment is used as is; single-line route works too.
    r = client.patch(f"{base}/segments/0/assign", params={"assign_type": "is_user", "value": "true"}, headers=h)
    assert r.status_code == 200 and any(l["is_user"] for l in r.json()["transcript_segments"])
    assert not e.gallery.people.keys() - {"Gabriel"}
