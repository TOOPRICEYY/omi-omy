from __future__ import annotations

import logging
import sys
from pathlib import Path

from .config import Config, load_env_file


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

    app = create_app(cfg)
    # Public port: the Omi webhook only (this is the one to put behind a tunnel).
    threading.Thread(target=uvicorn.run, args=(app.state.public,), daemon=True,
                     kwargs={"host": cfg.host, "port": cfg.port, "log_level": "warning"}).start()
    # Private port: review page and training API, loopback only.
    print(f"webhook:  http://{cfg.host}:{cfg.port}/omi/audio/<token>")
    print(f"review:   http://127.0.0.1:{cfg.port + 1}/")
    uvicorn.run(app, host="127.0.0.1", port=cfg.port + 1, log_level="info")


if __name__ == "__main__":
    main()
