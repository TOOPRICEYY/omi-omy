"""The API the forked Omi app (dev flavor) talks to instead of Omi's servers.

Only what a transcribe-and-summarize app needs: sign-in (appauth), onboarding
status, the conversation list and detail, delete, "stop recording", the live
listen socket and offline sync. Every other route answers 404, which the app
treats as "nothing here" (empty memories, chat, apps, people lists ...).

Served on the bridge's public port (the one the Cloudflare tunnel exposes) under
/app/<OMIBRIDGE_APP_SECRET>/: without the secret every path is the same plain 404
as any unknown address. Shares the bridge's session manager, job queue and
engine, so live audio from the app is cut, transcribed, named and summarized
exactly like audio from the webhook.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import re
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response

from attrib.audio import SR

from . import appaudio
from .appauth import Tokens, custom_token_router, emulator_router
from .config import Config
from .sessions import SessionManager

log = logging.getLogger("omibridge.appapi")

CHUNK_SAMPLES = SR            # feed the session manager about one second at a time
COVERAGE_DAYS = 14

CUTOVER_CONTROL = {"state": "legacy", "client_action": "none", "offline_queue_instruction": "none",
                   "account_generation": 0, "product_traffic_allowed": True, "legacy_writes_allowed": True,
                   "auth_bootstrap_reachable": True}


def iso(ts: float | None) -> str | None:
    return None if ts is None else datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def person_id(name: str) -> str:
    return "p-" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def is_unknown(name: str) -> bool:
    return name == "unknown" or bool(re.fullmatch(r"unknown\d*", name))


# -- conversation.json -> the app's conversation shape ----------------------------
def to_omi(doc: dict, wearer: str | None) -> dict:
    """Map the bridge's conversation.json to the shape the Omi app parses
    (app/lib/backend/schema/gen/conversation_wire.g.dart)."""
    started = doc.get("started_at")
    duration = doc.get("duration") or 0
    order = [s["name"] for s in doc.get("speakers", [])]
    segs = []
    for i, s in enumerate(doc.get("segments", [])):
        name = s.get("speaker") or "unknown"
        known = not is_unknown(name)
        is_user = bool(wearer) and name == wearer
        sid = order.index(name) if name in order else len(order)
        segs.append({"id": f"{doc['id']}-{i}", "text": s.get("text", ""), "start": float(s["start"]),
                     "end": float(s["end"]), "is_user": is_user, "speaker": f"SPEAKER_{sid:02d}",
                     "speaker_id": sid, "person_id": person_id(name) if known and not is_user else None,
                     "speech_profile_processed": True})
    summary = doc.get("summary") or {}
    first = " ".join(s["text"] for s in segs[:3] if s["text"]).strip()
    structured = {
        "title": summary.get("title") or ("Conversation" if not first else first[:48]),
        "overview": summary.get("overview") or first[:300],
        "emoji": summary.get("emoji") or "💬",
        "category": summary.get("category") or "other",
        "action_items": [{"description": a, "completed": False} if isinstance(a, str) else a
                         for a in summary.get("action_items") or []],
        "events": [],
    }
    return {"id": doc["id"], "created_at": iso(started) or iso(time.time()), "started_at": iso(started),
            "finished_at": iso(started + duration) if started is not None else None,
            "structured": structured, "transcript_segments": segs, "status": "completed",
            "discarded": False, "source": "omi", "language": summary.get("language"),
            "apps_results": [], "plugins_results": [], "photos": [], "audio_files": []}


def processing_stub(cid: str, meta: dict) -> dict:
    """A queued conversation that has not been processed yet."""
    started = meta.get("started_at")
    rate = meta.get("sample_rate") or SR
    finished = started + meta.get("samples", 0) / rate if started is not None else None
    return {"id": cid, "created_at": iso(started) or iso(time.time()), "started_at": iso(started),
            "finished_at": iso(finished), "structured": {"title": "", "overview": "", "emoji": "⏳"},
            "transcript_segments": [], "status": "processing", "discarded": False, "source": "omi"}


class Coverage:
    """Wall-clock spans of audio the bridge already has, so offline sync can
    skip what was also heard live (and what an earlier upload already sent)."""

    def __init__(self, path: Path, clock=time.time):
        self.path, self.clock, self.lock = path, clock, threading.Lock()
        try:
            self.spans = [tuple(s) for s in json.loads(path.read_text())]
        except (OSError, ValueError):
            self.spans = []

    def add(self, a: float, b: float) -> None:
        with self.lock:
            spans = sorted(self.spans + [(a, b)])
            merged: list[tuple[float, float]] = []
            for s, e in spans:
                if merged and s <= merged[-1][1] + 1.0:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                else:
                    merged.append((s, e))
            cutoff = self.clock() - COVERAGE_DAYS * 86400
            self.spans = [s for s in merged if s[1] >= cutoff]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.spans))
            tmp.replace(self.path)

    def covered(self, a: float, b: float) -> float:
        """Fraction of [a, b) already covered."""
        if b <= a:
            return 1.0
        with self.lock:
            got = sum(max(0.0, min(b, e) - max(a, s)) for s, e in self.spans)
        return min(1.0, got / (b - a))


class AppAPI:
    def __init__(self, cfg: Config, bridge: FastAPI, clock=time.time):
        self.cfg, self.clock = cfg, clock
        self.sessions: SessionManager = bridge.state.sessions
        self.engine = bridge.state.engine
        self.enqueue = bridge.state.enqueue
        self.tokens = Tokens(cfg.app_secret, cfg.app_uid, clock)
        self.coverage = Coverage(cfg.app_dir / "coverage.json", clock)
        self.sockets: dict[str, set] = {}       # session key -> {(websocket, loop)}
        self.undelivered: dict[str, str] = {}   # session key -> last finished conversation not yet shown
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()
        bridge.state.listeners.append(self)
        self.app = self._build()
        # The public port: the webhook, the sign-in emulator calls at the root, and this API
        # under the secret path, behind size caps and a limiter on failed checks.
        self.limiter = FailureLimiter(clock)
        bridge.state.public.include_router(emulator_router(self.tokens))
        self.public = PublicGate(bridge.state.public, self.app, cfg.app_secret, self.limiter)

    # A session key per app account, separate from webhook users.
    def key(self, uid: str) -> str:
        return f"app-{uid}"

    # -- bridge events -> connected phones ------------------------------------
    def _meta_uid(self, cid: str) -> str | None:
        for p in (self.cfg.queue_dir / f"{cid}.json", self.cfg.conv_dir / cid / "meta.json"):
            try:
                return json.loads(p.read_text()).get("uid")
            except (OSError, ValueError):
                continue
        return None

    def _send(self, key: str, message: dict) -> bool:
        sent = False
        for ws, loop in list(self.sockets.get(key, ())):
            asyncio.run_coroutine_threadsafe(ws.send_text(json.dumps(message)), loop)
            sent = True
        return sent

    def queued(self, cid: str) -> None:
        key = self._meta_uid(cid)
        if key and key.startswith("app-"):
            meta = json.loads((self.cfg.queue_dir / f"{cid}.json").read_text())
            self._send(key, {"type": "memory_processing_started", "memory": processing_stub(cid, meta)})

    def processed(self, cid: str, doc: dict) -> None:
        key = self._meta_uid(cid)
        if not (key and key.startswith("app-")):
            return
        if not self._send(key, {"type": "memory_created", "memory": to_omi(doc, self.cfg.wearer), "messages": []}):
            self.undelivered[key] = cid

    # -- helpers ----------------------------------------------------------------
    def conv_dir(self, cid: str) -> Path | None:
        p = (self.cfg.conv_dir / cid).resolve()
        if p.parent == self.cfg.conv_dir.resolve() and (p / "conversation.json").exists():
            return p
        return None

    def get(self, cid: str) -> dict | None:
        p = self.conv_dir(cid)
        if p is not None:
            return to_omi(json.loads((p / "conversation.json").read_text()), self.cfg.wearer)
        q = (self.cfg.queue_dir / f"{cid}.json").resolve()
        if q.parent == self.cfg.queue_dir.resolve() and q.exists():
            return processing_stub(cid, json.loads(q.read_text()))
        return None

    def all(self) -> list[dict]:
        out = [to_omi(json.loads(p.read_text()), self.cfg.wearer)
               for p in self.cfg.conv_dir.glob("*/conversation.json")]
        for q in self.cfg.queue_dir.glob("*.json"):
            if not (self.cfg.conv_dir / q.stem / "conversation.json").exists():
                out.append(processing_stub(q.stem, json.loads(q.read_text())))
        return sorted(out, key=lambda c: c["started_at"] or c["created_at"], reverse=True)

    # -- offline sync -------------------------------------------------------------
    def run_sync(self, job_id: str, files: list[tuple[str, bytes]]) -> dict:
        """Decode the uploaded files, drop what the bridge already has, and cut the
        rest into conversations the same way live audio is cut."""
        decoded = []
        errors = []
        for name, data in files:
            try:
                start, pcm = appaudio.decode_wal(name, data)
            except Exception as e:
                errors.append(f"{name}: {e}")
                continue
            if len(pcm):
                decoded.append((start, pcm))
        decoded.sort(key=lambda t: t[0])
        clock = {"now": 0.0}
        mgr = SessionManager(self.cfg, self.engine.speech_seconds, clock=lambda: clock["now"])
        key = self.key(self.cfg.app_uid).replace("app-", "sync-") + "-" + job_id[:8]
        new = []
        for start, pcm in decoded:
            for i in range(0, len(pcm), CHUNK_SAMPLES):
                chunk = pcm[i: i + CHUNK_SAMPLES]
                a = start + i / SR
                b = a + len(chunk) / SR
                clock["now"] = b
                if self.coverage.covered(a, b) < 0.5:
                    res = mgr.ingest(key, SR, chunk.astype("<i2").tobytes())
                    if res.get("id"):
                        new.append(res["id"])
                new += mgr.sweep()
            self.coverage.add(start, start + len(pcm) / SR)
        clock["now"] += self.cfg.idle_seconds + 1
        new += mgr.sweep()
        for cid in new:
            self.enqueue(cid)
        total = sum(len(p) for _, p in decoded) // CHUNK_SAMPLES + 1
        log.info("sync %s: %d files, %d conversations, %d errors", job_id, len(files), len(new), len(errors))
        # Undecodable files are terminal for the app (it would retry forever otherwise),
        # so they count as handled.
        return {"new_memories": new, "updated_memories": [], "failed_segments": 0,
                "total_segments": total, "errors": errors}

    # -- the FastAPI app -------------------------------------------------------------
    def _build(self) -> FastAPI:
        cfg, tokens = self.cfg, self.tokens
        app = FastAPI(title="omi-omy app API", docs_url=None, redoc_url=None, openapi_url=None)
        app.include_router(custom_token_router(tokens))

        def user(authorization: str | None) -> str:
            try:
                return tokens.check_header(authorization)
            except PermissionError as e:
                raise HTTPException(401, str(e))

        @app.api_route("/v1/health", methods=["GET", "HEAD"])   # the app's connectivity probe uses HEAD
        def health():
            return {"status": "ok"}

        @app.get("/v1/users/onboarding")
        def onboarding(authorization: str | None = Header(None)):
            user(authorization)
            return {"completed": True, "acquisition_source": "", "device_onboarding_completed": True}

        @app.patch("/v1/users/onboarding")
        def onboarding_update(authorization: str | None = Header(None)):
            user(authorization)
            return {"status": "ok"}

        @app.get("/v1/account/cutover/control")
        def cutover(authorization: str | None = Header(None)):
            user(authorization)
            return CUTOVER_CONTROL

        @app.get("/v1/conversations")
        def conversations(authorization: str | None = Header(None), limit: int = 50, offset: int = 0,
                          statuses: str = "", include_discarded: bool = False, starred: bool = False,
                          start_date: str | None = None, end_date: str | None = None,
                          folder_id: str | None = None):
            user(authorization)
            if starred or folder_id:
                return []
            wanted = {s for s in statuses.split(",") if s}
            items = [c for c in self.all() if not wanted or c["status"] in wanted]

            def parse(text: str) -> datetime:
                d = datetime.fromisoformat(text.replace("Z", "+00:00"))
                return d if d.tzinfo else d.replace(tzinfo=timezone.utc)

            def when(c):
                return parse(c["started_at"] or c["created_at"])

            try:
                lo = parse(start_date) if start_date else None
                hi = parse(end_date) if end_date else None
            except ValueError:
                raise HTTPException(400, "bad date")
            if lo:
                items = [c for c in items if when(c) >= lo]
            if hi:
                items = [c for c in items if when(c) <= hi]
            return items[offset: offset + max(0, min(limit, 500))]

        @app.get("/v1/conversations/{cid}")
        def conversation(cid: str, authorization: str | None = Header(None)):
            user(authorization)
            c = self.get(cid)
            if c is None:
                raise HTTPException(404, "no such conversation")
            return c

        @app.delete("/v1/conversations/{cid}")
        def delete(cid: str, authorization: str | None = Header(None)):
            user(authorization)
            p = self.conv_dir(cid)
            if p is None:
                raise HTTPException(404, "no such conversation")
            cfg.trash_dir.mkdir(parents=True, exist_ok=True)
            dest = cfg.trash_dir / cid
            if dest.exists():
                dest = cfg.trash_dir / f"{cid}-{int(self.clock())}"
            shutil.move(str(p), str(dest))
            log.info("moved %s to the trash", cid)
            return Response(status_code=204)

        @app.post("/v1/conversations")
        def stop(authorization: str | None = Header(None)):
            """The app's "stop" button: end the conversation being recorded now."""
            uid = user(authorization)
            cid = self.sessions.close(self.key(uid))
            if cid is None:
                raise HTTPException(404, "nothing is being recorded")
            self.enqueue(cid)
            return {"conversation": self.get(cid), "messages": []}

        @app.get("/v1/users/people")
        def people(authorization: str | None = Header(None)):
            # Named voices from the bridge's gallery, so transcript lines can show real names.
            user(authorization)
            now = iso(self.clock())
            return [{"id": person_id(p["name"]), "name": p["name"], "created_at": now, "updated_at": now,
                     "speech_samples": []}
                    for p in self.engine.people() if p["name"] != cfg.wearer]

        # -- offline sync ----------------------------------------------------------
        @app.post("/v2/sync-local-files")
        async def sync_local_files(request: Request, files: list[UploadFile] = File(...),
                                   authorization: str | None = Header(None)):
            user(authorization)
            payload = []
            for f in files:
                name = f.filename or ""
                if not name.endswith(".bin") or "/" in name or "\\" in name:
                    raise HTTPException(400, f"invalid file name {name!r}")
                try:
                    appaudio.wal_start_time(name)
                except ValueError:
                    raise HTTPException(400, f"invalid file name {name!r}")
                payload.append((name, await f.read()))
            job_id = uuid.uuid4().hex
            job = {"job_id": job_id, "status": "processing", "total_files": len(payload),
                   "total_segments": len(payload), "processed_segments": 0, "successful_segments": 0,
                   "failed_segments": 0, "result": None, "error": None, "poll_after_ms": 2000}
            with self.lock:
                self.jobs[job_id] = job

            def work():
                try:
                    res = self.run_sync(job_id, payload)
                    job.update(status="completed", result=res, total_segments=res["total_segments"],
                               processed_segments=res["total_segments"],
                               successful_segments=res["total_segments"])
                except Exception as e:
                    log.exception("sync %s failed", job_id)
                    job.update(status="failed", error=str(e))

            threading.Thread(target=work, daemon=True).start()
            return JSONResponse({k: job[k] for k in ("job_id", "status", "total_files", "total_segments",
                                                     "poll_after_ms")} | {"lane": "fresh"}, status_code=202)

        @app.get("/v2/sync-local-files/{job_id}")
        def sync_job(job_id: str, authorization: str | None = Header(None)):
            user(authorization)
            job = self.jobs.get(job_id)
            if job is None:
                raise HTTPException(404, "no such job")   # the app uploads again
            return job

        # -- live audio --------------------------------------------------------------
        @app.websocket("/v4/listen")
        async def listen(ws: WebSocket, codec: str = "opus", sample_rate: int = SR):
            try:
                uid = tokens.check_header(ws.headers.get("authorization"))
            except PermissionError as e:
                log.warning("listen refused: %s", e)
                await ws.close(code=1008)
                return
            try:
                decoder = appaudio.LiveDecoder(codec, sample_rate)
            except ValueError as e:
                await ws.close(code=1003, reason=str(e))
                return
            await ws.accept()
            key = self.key(uid)
            entry = (ws, asyncio.get_running_loop())
            self.sockets.setdefault(key, set()).add(entry)
            log.info("listen socket open (codec %s)", codec)
            await ws.send_text(json.dumps({"type": "service_status", "status": "ready"}))
            if key in self.undelivered:
                await ws.send_text(json.dumps({"type": "last_memory", "memory_id": self.undelivered.pop(key)}))
            buf: list[np.ndarray] = []
            have = 0

            async def flush():
                nonlocal buf, have
                if not have:
                    return
                pcm = np.concatenate(buf)
                buf, have = [], 0
                end = self.clock()
                self.coverage.add(end - len(pcm) / SR, end)
                res = await run_in_threadpool(self.sessions.ingest, key, SR, pcm.astype("<i2").tobytes())
                if res.get("id"):
                    self.enqueue(res["id"])

            try:
                while True:
                    msg = await ws.receive()
                    if msg["type"] == "websocket.disconnect":
                        break
                    if msg.get("bytes") is not None:
                        x = decoder.decode(msg["bytes"])
                        if len(x):
                            buf.append(x)
                            have += len(x)
                        if have >= CHUNK_SAMPLES:
                            await flush()
                    elif msg.get("text"):
                        text = msg["text"]
                        if text == "ping":
                            await ws.send_text("pong")
                        # JSON control messages (client_state, capture_evidence_frame,
                        # suggested_transcript ...) are not needed here.
            except WebSocketDisconnect:
                pass
            finally:
                await flush()
                self.sockets.get(key, set()).discard(entry)
                log.info("listen socket closed")

        @app.api_route("/{path:path}", methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"])
        def everything_else(path: str):
            # Memories, chat, apps, subscriptions ...: switched off. 404 reads as "none" in the app.
            raise HTTPException(404, "not available on this backend")

        return app


# -- the public port's front door ----------------------------------------------------
FAIL_WINDOW = 600      # seconds
FAIL_LIMIT = 30        # failed checks per address in the window before it is blocked
BLOCK_SECONDS = 900
MAX_BODY = {"/v2/sync-local-files": 128 * 2**20, "/omi/audio/": 2**20}
DEFAULT_MAX_BODY = 256 * 2**10


class FailureLimiter:
    """Counts failed secret/token checks per client address; too many blocks it for a while."""

    def __init__(self, clock=time.time):
        self.clock, self.lock = clock, threading.Lock()
        self.fails: dict[str, list[float]] = {}
        self.blocked_until: dict[str, float] = {}

    def fail(self, ip: str) -> None:
        now = self.clock()
        with self.lock:
            recent = [t for t in self.fails.get(ip, []) if t > now - FAIL_WINDOW] + [now]
            self.fails[ip] = recent
            if len(recent) >= FAIL_LIMIT:
                self.blocked_until[ip] = now + BLOCK_SECONDS
                self.fails.pop(ip, None)
                log.warning("blocked an address for %d minutes after repeated failed checks", BLOCK_SECONDS // 60)
            if len(self.fails) > 10000:          # bound memory under a flood of addresses
                self.fails.clear()

    def blocked(self, ip: str) -> bool:
        until = self.blocked_until.get(ip)
        if until is None:
            return False
        if until <= self.clock():
            self.blocked_until.pop(ip, None)
            return False
        return True


def client_ip(scope) -> str:
    for k, v in scope.get("headers", []):
        if k == b"cf-connecting-ip":
            return v.decode("latin-1").strip()
    c = scope.get("client")
    return c[0] if c else "?"


def body_cap(path: str) -> int:
    for prefix, cap in MAX_BODY.items():
        if path.startswith(prefix):
            return cap
    return DEFAULT_MAX_BODY


class PublicGate:
    """ASGI front for the public port.

    /app/<secret>/<rest> goes to the app API as /<rest> (constant-time secret check).
    Anything else, including /app/<wrong>/..., goes to the public app unchanged, so a
    wrong secret gets exactly the 404 an unknown path gets. Failed checks without the
    secret (a wrong secret, a wrong webhook token, refused sign-in tokens) count against
    the client's address. A request that carries the right secret is never counted and
    never blocked: the phone must not be locked out of its own backend, least of all
    while it is recording, because of a stale token or of someone else on its network.
    """

    def __init__(self, public, api, secret: str, limiter: FailureLimiter):
        self.public, self.api, self.limiter = public, api, limiter
        self.secret = secret.encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.public(scope, receive, send)
        ip = client_ip(scope)
        path = scope["path"]
        target = self.public
        wrong_secret = False
        if path.startswith("/app/"):
            seg, _, tail = path[5:].partition("/")
            if hmac.compare_digest(seg.encode(), self.secret):
                tail = "/" + tail
                scope = dict(scope, path=tail, raw_path=tail.encode(), root_path="")
                target = self.api
            else:
                wrong_secret = True
        trusted = target is self.api
        if not trusted and self.limiter.blocked(ip):
            if scope["type"] == "http":
                await send({"type": "http.response.start", "status": 429,
                            "headers": [(b"content-type", b"text/plain")]})
                await send({"type": "http.response.body", "body": b"Too many failed attempts"})
            else:
                await send({"type": "websocket.close", "code": 1008})
            return
        if wrong_secret:
            self.limiter.fail(ip)
        inner_path = scope["path"]

        if scope["type"] == "http":
            cap = body_cap(inner_path)
            length = next((v for k, v in scope.get("headers", []) if k == b"content-length"), None)
            if length is not None and length.isdigit() and int(length) > cap:
                await send({"type": "http.response.start", "status": 413, "headers": []})
                await send({"type": "http.response.body", "body": b""})
                return
            seen = 0

            async def capped_receive():
                nonlocal seen
                msg = await receive()
                if msg["type"] == "http.request":
                    seen += len(msg.get("body", b""))
                    if seen > cap:
                        raise ValueError("request body too large")
                return msg

            emulator = inner_path.startswith(("/identitytoolkit.googleapis.com/", "/securetoken.googleapis.com/",
                                              "/www.googleapis.com/"))

            async def watching_send(msg):
                if msg["type"] == "http.response.start":
                    st = msg["status"]
                    if not trusted and (st in (401, 403) or (emulator and st == 400)):
                        self.limiter.fail(ip)
                    if trusted or emulator:
                        # One line per app request; the secret is already stripped from the path.
                        log.info("app %s %s -> %d", scope["method"], inner_path, st)
                await send(msg)

            return await target(scope, capped_receive, watching_send)

        async def watching_ws_send(msg):
            if msg["type"] in ("websocket.accept", "websocket.close") and trusted:
                log.info("app websocket %s %s", inner_path, msg["type"].split(".")[1])
            await send(msg)

        return await target(scope, receive, watching_ws_send)
