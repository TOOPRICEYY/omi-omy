"""Sign-in for the forked Omi app without Firebase or Omi's servers.

The app's dev flavor signs in through the Firebase Auth *emulator*: it asks its
backend for a custom token (POST /v1/auth/local-dev/custom-token), hands that
to the Firebase SDK, and the SDK exchanges it at the emulator's REST endpoints
for an ID token and a refresh token. This module answers those few emulator
calls itself, so no emulator, Node or Java is needed.

Every token is an HS256 JWT signed with OMIBRIDGE_APP_SECRET. The Firebase SDK
only decodes the payload (it never checks signatures), and the bridge checks the
signature on every request, so a token is only good if this bridge made it.
There is one account (OMIBRIDGE_APP_UID). Rotating the secret signs out every
device.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import JSONResponse

from .config import Config

log = logging.getLogger("omibridge.appauth")

PROJECT_ID = "demo-omi-local"   # the dev flavor's Firebase project (firebase_options_local.dart)
ID_TOKEN_SECONDS = 3600
CUSTOM_TOKEN_SECONDS = 300
PAIRING_SECONDS = 600


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


class Tokens:
    def __init__(self, secret: str, uid: str, clock=time.time):
        if not secret:
            raise ValueError("OMIBRIDGE_APP_SECRET is empty")
        self.key = hashlib.sha256(("omiomy-app:" + secret).encode()).digest()
        self.uid = uid
        self.clock = clock

    def _sign(self, payload: dict) -> str:
        head = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
        body = _b64(json.dumps(payload, separators=(",", ":")).encode())
        sig = hmac.new(self.key, f"{head}.{body}".encode(), hashlib.sha256).digest()
        return f"{head}.{body}.{_b64(sig)}"

    def _verify(self, token: str, kind: str) -> dict:
        try:
            head, body, sig = token.split(".")
            want = hmac.new(self.key, f"{head}.{body}".encode(), hashlib.sha256).digest()
            if not hmac.compare_digest(want, _unb64(sig)):
                raise ValueError("bad signature")
            payload = json.loads(_unb64(body))
        except Exception as e:
            raise PermissionError(f"invalid {kind} token") from e
        if payload.get("omiomy") != kind:
            raise PermissionError(f"not a {kind} token")
        if payload.get("exp", 0) < self.clock():
            raise PermissionError(f"{kind} token expired")
        if payload.get("user_id", payload.get("uid")) != self.uid:
            raise PermissionError("unknown account")
        return payload

    def custom_token(self) -> str:
        now = int(self.clock())
        return self._sign({"omiomy": "custom", "uid": self.uid, "iat": now, "exp": now + CUSTOM_TOKEN_SECONDS})

    def id_token(self, auth_time: int | None = None) -> str:
        now = int(self.clock())
        # The claims the Firebase emulator puts in its ID tokens; the SDK reads exp, iat and auth_time.
        return self._sign({
            "omiomy": "id", "iss": f"https://securetoken.google.com/{PROJECT_ID}", "aud": PROJECT_ID,
            "auth_time": auth_time or now, "user_id": self.uid, "sub": self.uid, "iat": now,
            "exp": now + ID_TOKEN_SECONDS, "firebase": {"identities": {}, "sign_in_provider": "custom"},
        })

    def refresh_token(self, auth_time: int) -> str:
        # Never expires; rotate OMIBRIDGE_APP_SECRET to revoke.
        return self._sign({"omiomy": "refresh", "uid": self.uid, "auth_time": auth_time, "exp": 2**40})

    def check_id(self, token: str) -> dict:
        return self._verify(token, "id")

    def check_custom(self, token: str) -> dict:
        return self._verify(token, "custom")

    def check_refresh(self, token: str) -> dict:
        return self._verify(token, "refresh")

    def check_header(self, authorization: str | None) -> str:
        """`Authorization: Bearer <id token>` -> uid, or PermissionError."""
        if not authorization or not authorization.lower().startswith("bearer "):
            raise PermissionError("missing bearer token")
        return self.check_id(authorization.split(" ", 1)[1].strip())["user_id"]


def pairing_file(cfg: Config) -> Path:
    return cfg.app_dir / "pairing_until"


def open_pairing(cfg: Config, seconds: int = PAIRING_SECONDS, clock=time.time) -> float:
    cfg.app_dir.mkdir(parents=True, exist_ok=True)
    until = clock() + seconds
    pairing_file(cfg).write_text(str(until))
    return until


def pairing_open(cfg: Config, clock=time.time) -> bool:
    if cfg.app_pairing == "open":
        return True
    try:
        return float(pairing_file(cfg).read_text()) > clock()
    except (OSError, ValueError):
        return False


def router(cfg: Config, tokens: Tokens) -> APIRouter:
    r = APIRouter()

    @r.post("/v1/auth/local-dev/custom-token")
    def custom_token(uid: str = Form("local-dev-user")):
        # The app asks for "local-dev-user"; there is only one account here, whatever it asks for.
        if not pairing_open(cfg, tokens.clock):
            log.warning("sign-in refused: pairing is closed (run `python -m omibridge pair`)")
            raise HTTPException(403, "Sign-in is closed. On the Mac, run: python -m omibridge pair")
        log.info("issued a custom token")
        return {"custom_token": tokens.custom_token()}

    # -- the Firebase Auth emulator's REST surface, as the Firebase SDK calls it ----
    @r.get("/")
    def emulator_root():
        # The app checks the emulator is reachable before signing in.
        return {"authEmulator": {"ready": True}, "service": "omi-omy bridge"}

    def error(message: str, code: int = 400) -> JSONResponse:
        return JSONResponse({"error": {"code": code, "message": message, "errors": [{"message": message}]}},
                            status_code=code)

    async def body_of(request: Request) -> dict:
        raw = await request.body()
        if not raw:
            return {}
        if "json" in request.headers.get("content-type", "") or raw[:1] == b"{":
            return json.loads(raw)
        from urllib.parse import parse_qsl

        return dict(parse_qsl(raw.decode()))

    def session(auth_time: int) -> dict:
        return {"idToken": tokens.id_token(auth_time), "refreshToken": tokens.refresh_token(auth_time),
                "expiresIn": str(ID_TOKEN_SECONDS), "localId": tokens.uid}

    @r.post("/identitytoolkit.googleapis.com/v1/{action}")
    async def identitytoolkit(action: str, request: Request):
        body = await body_of(request)
        if action == "accounts:signInWithCustomToken":
            try:
                tokens.check_custom(body.get("token", ""))
            except PermissionError as e:
                log.warning("custom token refused: %s", e)
                return error("INVALID_CUSTOM_TOKEN")
            log.info("signed in with a custom token")
            return {"kind": "identitytoolkit#VerifyCustomTokenResponse", "isNewUser": False,
                    **session(int(tokens.clock()))}
        if action == "accounts:lookup":
            try:
                claims = tokens.check_id(body.get("idToken", ""))
            except PermissionError:
                return error("INVALID_ID_TOKEN")
            ms = str(int(claims["auth_time"]) * 1000)
            return {"kind": "identitytoolkit#GetAccountInfoResponse", "users": [{
                "localId": tokens.uid, "lastLoginAt": ms, "createdAt": ms, "customAuth": True,
                "emailVerified": False, "disabled": False, "providerUserInfo": [],
                "lastRefreshAt": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(tokens.clock())),
            }]}
        if action == "accounts:update":
            # Profile edits (display name and so on) are accepted and dropped.
            try:
                tokens.check_id(body.get("idToken", ""))
            except PermissionError:
                return error("INVALID_ID_TOKEN")
            return {"kind": "identitytoolkit#SetAccountInfoResponse", "localId": tokens.uid,
                    "providerUserInfo": [], **session(int(tokens.clock()))}
        log.warning("unsupported emulator call %s (fields: %s)", action, sorted(body))
        return error("OPERATION_NOT_ALLOWED")

    @r.post("/securetoken.googleapis.com/v1/token")
    async def securetoken(request: Request):
        body = await body_of(request)
        grant = body.get("grant_type") or body.get("grantType")
        refresh = body.get("refresh_token") or body.get("refreshToken") or ""
        if grant != "refresh_token":
            return error("INVALID_GRANT_TYPE")
        try:
            claims = tokens.check_refresh(refresh)
        except PermissionError:
            return error("INVALID_REFRESH_TOKEN")
        idt = tokens.id_token(int(claims["auth_time"]))
        return {"access_token": idt, "id_token": idt, "expires_in": str(ID_TOKEN_SECONDS), "token_type": "Bearer",
                "refresh_token": refresh, "user_id": tokens.uid, "project_id": PROJECT_ID}

    return r
