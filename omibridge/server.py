"""HTTP service: Omi's "Realtime audio bytes" webhook in, named transcripts out.

    python -m omibridge            # webhook on OMIBRIDGE_PORT, review page on the next port
    python -m omibridge enroll NAME file.wav

Point the Omi app (Settings > Developer Mode > Realtime audio bytes) at
    https://<your-host>/omi/audio/<OMIBRIDGE_TOKEN>
"""
from __future__ import annotations

import hmac
import io
import json
import logging
import queue
import threading
import time
from pathlib import Path

import soundfile as sf
from fastapi import Body, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, Response

from attrib.audio import SR

from .config import Config
from .engine import Engine
from .importer import parse_when, queue_recording
from .sessions import SessionManager

log = logging.getLogger("omibridge")
STATIC = Path(__file__).with_name("static")


def create_app(cfg: Config | None = None, engine: Engine | None = None, start_workers: bool = True,
               clock=time.time) -> FastAPI:
    cfg = cfg or Config()
    cfg.ensure_dirs()
    if engine is None:
        transcriber = None
        if cfg.transcribe:
            import os

            if os.environ.get("OPENROUTER_API_KEY"):
                from attrib.transcribe import OpenRouterTranscriber

                transcriber = OpenRouterTranscriber()
            else:
                log.warning("OPENROUTER_API_KEY is not set: conversations will have speakers but no text")
        engine = Engine(cfg, transcriber)
    sessions = SessionManager(cfg, engine.speech_seconds, clock)
    jobs: queue.Queue[str] = queue.Queue()
    status = {"processing": None, "last_error": None, "processed": 0}
    app = FastAPI(title="omi-omy bridge")
    app.state.cfg, app.state.engine, app.state.sessions, app.state.jobs = cfg, engine, sessions, jobs

    def process_one(cid: str) -> None:
        pcm, meta_p = cfg.queue_dir / f"{cid}.pcm", cfg.queue_dir / f"{cid}.json"
        status["processing"] = cid
        try:
            doc = engine.process(pcm, json.loads(meta_p.read_text()))
            if cfg.omi_writeback and cfg.omi_api_key and any(s["text"] for s in doc["segments"]):
                from . import omi_api

                try:
                    res = omi_api.push(doc, cfg.omi_api_key, cfg.omi_api_base, cfg.wearer)
                    (cfg.conv_dir / cid / "omi_response.json").write_text(json.dumps(res))
                except Exception as e:
                    status["last_error"] = f"{cid}: Omi write-back failed: {e}"
                    log.exception("Omi write-back failed for %s", cid)
            pcm.unlink(missing_ok=True)
            meta_p.unlink(missing_ok=True)
            status["processed"] += 1
        except Exception as e:
            status["last_error"] = f"{cid}: {e}"
            log.exception("processing %s failed", cid)
            pcm.rename(pcm.with_suffix(".failed"))
        finally:
            status["processing"] = None

    def worker() -> None:
        while True:
            process_one(jobs.get())

    def sweeper() -> None:
        while True:
            time.sleep(5)
            for cid in sessions.sweep():
                jobs.put(cid)

    app.state.process_one = process_one
    if start_workers:
        sessions.recover()
        for p in sorted(cfg.queue_dir.glob("*.json")):
            jobs.put(p.stem)
        threading.Thread(target=worker, daemon=True).start()
        threading.Thread(target=sweeper, daemon=True).start()

    def conv_path(cid: str) -> Path:
        p = (cfg.conv_dir / cid).resolve()
        if p.parent != cfg.conv_dir.resolve() or not p.is_dir():
            raise HTTPException(404, "no such conversation")
        return p

    # -- Omi webhook -----------------------------------------------------------
    @app.post("/omi/audio/{token}")
    async def omi_audio(token: str, request: Request, uid: str = "user", sample_rate: int = SR):
        if not cfg.token or not hmac.compare_digest(token, cfg.token):
            raise HTTPException(403, "bad token")
        if not 8000 <= sample_rate <= 48000:
            raise HTTPException(400, "bad sample_rate")
        body = await request.body()
        # Omi turns the webhook off after 100 failed deliveries in a row, so this
        # must answer fast and never fail because of our own processing.
        try:
            res = await run_in_threadpool(sessions.ingest, uid, sample_rate, body,
                                          request.headers.get("idempotency-key"))
        except Exception:
            log.exception("ingest failed")
            return {"state": "error"}
        if res.get("id"):
            jobs.put(res["id"])
        return res

    # Only the webhook goes on the port that the tunnel exposes.
    public = FastAPI(title="omi-omy bridge (public)", docs_url=None, redoc_url=None, openapi_url=None)
    public.add_api_route("/omi/audio/{token}", omi_audio, methods=["POST"])
    app.state.public = public

    # -- review and training: served on the private port only ------------------
    @app.get("/health")
    def health():
        return {"ok": True, "recording": [s.uid for s in sessions.sessions.values()],
                "queued": jobs.qsize(), **status, "people": len(engine.gallery.people),
                "transcription": engine.transcriber is not None, "omi_writeback": bool(cfg.omi_writeback and cfg.omi_api_key)}

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text()

    @app.get("/api/conversations")
    def conversations():
        out = []
        for p in sorted(cfg.conv_dir.glob("*/conversation.json"), reverse=True)[:200]:
            d = json.loads(p.read_text())
            out.append({k: d[k] for k in ("id", "started_at", "duration", "speech_seconds", "speakers", "error")}
                       | {"preview": " ".join(s["text"] for s in d["segments"][:3])[:160]})
        return out

    @app.post("/api/import")
    async def import_recording(file: UploadFile, started_at: str | None = Form(None)):
        """A recording the webhook never saw (for example exported from the Omi app)."""
        data = await file.read()

        def work() -> list[str]:
            x, sr = sf.read(io.BytesIO(data), dtype="float32")
            return queue_recording(cfg, x, sr, parse_when(started_at), file.filename or "")

        try:
            ids = await run_in_threadpool(work)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception:
            raise HTTPException(400, "could not read that file as audio (WAV, FLAC, MP3 and OGG work)")
        for cid in ids:
            jobs.put(cid)
        return {"queued": ids}

    @app.get("/api/conversations/{cid}")
    def conversation(cid: str):
        return json.loads((conv_path(cid) / "conversation.json").read_text())

    @app.get("/api/conversations/{cid}/audio")
    def audio(cid: str, start: float | None = None, end: float | None = None):
        p = conv_path(cid) / "audio.flac"
        if start is None:
            return FileResponse(p, media_type="audio/flac")
        info = sf.info(p)
        a = max(0, int(start * info.samplerate))
        b = min(info.frames, int((end or start + 10) * info.samplerate))
        x, sr = sf.read(p, start=a, stop=max(b, a + 1), dtype="float32")
        buf = io.BytesIO()
        sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
        return Response(buf.getvalue(), media_type="audio/wav")

    @app.post("/api/conversations/{cid}/label")
    def label(cid: str, speaker: str = Body(...), person: str = Body(...), learn: bool = Body(True)):
        conv_path(cid)
        try:
            return engine.label(cid, speaker, person, learn)
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e))

    @app.post("/api/conversations/{cid}/rerun")
    def rerun(cid: str):
        conv_path(cid)
        return engine.render(cid)

    @app.get("/api/people")
    def people():
        return engine.people()

    @app.post("/api/people/{name}/enroll")
    async def enroll(name: str, file: UploadFile):
        x, sr = sf.read(io.BytesIO(await file.read()), dtype="float32")
        n = engine.enroll(name, x, sr)
        if not n:
            raise HTTPException(400, "no usable speech found in that file")
        return {"name": name, "added": n}

    @app.delete("/api/people/{name}")
    def forget(name: str):
        if not engine.forget(name):
            raise HTTPException(404, "no such person")
        return {"ok": True}

    return app
