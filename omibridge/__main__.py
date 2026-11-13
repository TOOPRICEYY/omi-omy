from __future__ import annotations

import logging
import sys
from pathlib import Path

from .config import Config, load_env_file


class _RedactToken(logging.Filter):
    """Keeps the webhook token and the app secret (both part of URL paths) out of uvicorn's logs."""

    def __init__(self, *secrets: str) -> None:
        super().__init__()
        self.secrets = [s for s in secrets if s]

    def _clean(self, v):
        if isinstance(v, str):
            for s in self.secrets:
                v = v.replace(s, "<secret>")
        return v

    def filter(self, record: logging.LogRecord) -> bool:
        if self.secrets:
            if isinstance(record.args, tuple):
                record.args = tuple(self._clean(a) for a in record.args)
            elif isinstance(record.args, dict):
                record.args = {k: self._clean(v) for k, v in record.args.items()}
            record.msg = self._clean(record.msg)
        return True


def main() -> None:
    load_env_file(Path(".env"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = Config()
    args = sys.argv[1:]
    if args[:1] == ["enroll"]:
        import soundfile as sf

        from .engine import Engine

        if len(args) != 3:
            sys.exit("usage: python -m omibridge enroll NAME file.wav")
        cfg.ensure_dirs()
        x, sr = sf.read(args[2], dtype="float32")
        n = Engine(cfg).enroll(args[1], x, sr)
        print(f"added {n} voiceprints for {args[1]}")
        return
    if not cfg.token:
        sys.exit("Set OMIBRIDGE_TOKEN (a long random string) in .env first.")
    import uvicorn

    from .server import create_app

    import threading

    redact = _RedactToken(cfg.token, cfg.app_secret)
    for name in ("uvicorn.access", "uvicorn.error"):
        logging.getLogger(name).addFilter(redact)
    app = create_app(cfg)
    public = app.state.public
    if cfg.app_secret:
        # The forked Omi app's API, under /app/<secret>/ on the same public port.
        from .appapi import AppAPI

        public = AppAPI(cfg, app).public
    # Public port: the Omi webhook (and the app API); this is the one behind the tunnel.
    threading.Thread(target=uvicorn.run, args=(public,), daemon=True,
                     kwargs={"host": cfg.host, "port": cfg.port, "log_level": "warning",
                             "ws_ping_interval": 20}).start()
    print(f"webhook:  http://{cfg.host}:{cfg.port}/omi/audio/<token>")
    if cfg.app_secret:
        print(f"app API:  http://{cfg.host}:{cfg.port}/app/<secret>/")
    # Private port: review page and training API, loopback only.
    print(f"review:   http://127.0.0.1:{cfg.port + 1}/")
    uvicorn.run(app, host="127.0.0.1", port=cfg.port + 1, log_level="info")


if __name__ == "__main__":
    main()
