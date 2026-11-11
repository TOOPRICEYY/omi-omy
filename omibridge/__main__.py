from __future__ import annotations

import logging
import sys
from pathlib import Path

from .config import Config, load_env_file


class _RedactToken(logging.Filter):
    """Keeps the webhook token (part of the URL path) out of uvicorn's access log."""

    def __init__(self, token: str) -> None:
        super().__init__()
        self.token = token

    def filter(self, record: logging.LogRecord) -> bool:
        if self.token and isinstance(record.args, tuple):
            record.args = tuple(a.replace(self.token, "<token>") if isinstance(a, str) else a
                                for a in record.args)
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
    if args[:1] == ["pair"]:
        # Lets the forked app sign in during the next ten minutes.
        import time

        from .appauth import open_pairing

        until = open_pairing(cfg)
        print("Sign-in is open until " + time.strftime("%H:%M", time.localtime(until))
              + ". In the app, tap the local sign-in button.")
        return
    if not cfg.token:
        sys.exit("Set OMIBRIDGE_TOKEN (a long random string) in .env first.")
    import uvicorn

    from .server import create_app

    import threading

    logging.getLogger("uvicorn.access").addFilter(_RedactToken(cfg.token))
    app = create_app(cfg)
    # Public port: the Omi webhook only (this is the one to put behind a tunnel).
    threading.Thread(target=uvicorn.run, args=(app.state.public,), daemon=True,
                     kwargs={"host": cfg.host, "port": cfg.port, "log_level": "warning"}).start()
    if cfg.app_secret:
        # App API for the forked Omi app. Loopback or the Tailscale address, never the tunnel.
        from .appapi import AppAPI

        api = AppAPI(cfg, app)
        threading.Thread(target=uvicorn.run, args=(api.app,), daemon=True,
                         kwargs={"host": cfg.app_host, "port": cfg.app_port, "log_level": "info",
                                 "ws_ping_interval": 20}).start()
        print(f"app API:  http://{cfg.app_host}:{cfg.app_port}/")
    # Private port: review page and training API, loopback only.
    print(f"webhook:  http://{cfg.host}:{cfg.port}/omi/audio/<token>")
    print(f"review:   http://127.0.0.1:{cfg.port + 1}/")
    uvicorn.run(app, host="127.0.0.1", port=cfg.port + 1, log_level="info")


if __name__ == "__main__":
    main()
