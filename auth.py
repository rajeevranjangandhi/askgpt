"""
Signup, login and logout.

- Passwords are stored only as salted scrypt hashes.
- A login creates a random session token. The browser keeps it in an HttpOnly
  cookie; the database keeps only its SHA-256 hash.
- require_user() is applied to the whole app, so every route needs a login
  unless its path is listed in PUBLIC_PATHS.
"""
import hashlib
import hmac
import os
import re
import secrets
import threading
import time
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from database import (
    create_session,
    create_user,
    delete_session,
    get_session_user,
    get_user_by_email,
)

router = APIRouter()
templates = Jinja2Templates(directory="templates")

PUBLIC_PATHS = {"/login", "/signup", "/logout"}

SESSION_COOKIE = "session"
SESSION_DAYS = 7
# Set COOKIE_SECURE=true once the site is served over HTTPS.
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").strip().lower() == "true"

MIN_PASSWORD, MAX_PASSWORD = 8, 128
EMAIL_PATTERN = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------
def _scrypt(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=2**14, r=8, p=5, maxmem=64 * 1024 * 1024
    )


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    return f"{salt.hex()}${_scrypt(password, salt).hex()}"


def verify_password(password: str, stored: str) -> bool:
    salt, expected = stored.split("$")
    return hmac.compare_digest(_scrypt(password, bytes.fromhex(salt)).hex(), expected)


# Checked when the email is unknown, so both cases take the same time.
_DUMMY_HASH = hash_password(secrets.token_hex(16))


# ---------------------------------------------------------------------------
# Rate limiting (in memory, per server process)
# ---------------------------------------------------------------------------
class RateLimiter:
    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str) -> bool:
        """Count one attempt. Returns False when the key is over its limit."""
        now = time.monotonic()
        with self._lock:
            if len(self._hits) > 5000:
                self._hits = {
                    k: v for k, v in self._hits.items() if now - v[-1] < self.window
                }
            recent = [t for t in self._hits.get(key, []) if now - t < self.window]
            if len(recent) >= self.limit:
                return False
            self._hits[key] = recent + [now]
            return True

    def reset(self, key: str):
        with self._lock:
            self._hits.pop(key, None)


_login_by_account = RateLimiter(limit=8, window_seconds=15 * 60)
_login_by_ip = RateLimiter(limit=50, window_seconds=15 * 60)
_signup_by_ip = RateLimiter(limit=10, window_seconds=60 * 60)


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _error(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


_TOO_MANY = "Too many attempts. Please wait a while and try again."


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _start_session(user_id: int) -> JSONResponse:
    token = secrets.token_urlsafe(32)
    create_session(
        _token_hash(token), user_id, datetime.utcnow() + timedelta(days=SESSION_DAYS)
    )
    response = JSONResponse({"success": True})
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=SESSION_DAYS * 24 * 60 * 60,
        httponly=True,
        samesite="lax",
        secure=COOKIE_SECURE,
    )
    return response


def current_user(request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    return get_session_user(_token_hash(token)) if token else None


def require_user(request: Request):
    """App-wide dependency: puts the logged-in user on request.state.user."""
    # Not request.url.path: that is built using the client-supplied Host header.
    path = request.scope["path"]
    if path in PUBLIC_PATHS:
        return

    user = current_user(request)
    if user is None:
        if path == "/":
            raise HTTPException(status_code=303, headers={"Location": "/login"})
        raise HTTPException(status_code=401, detail="Please log in.")

    request.state.user = user


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
class Credentials(BaseModel):
    email: str
    password: str


def _auth_page(request: Request, mode: str):
    if current_user(request):
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"mode": mode, "min_password": MIN_PASSWORD, "max_password": MAX_PASSWORD},
    )


@router.get("/login")
def login_page(request: Request):
    return _auth_page(request, "login")


@router.get("/signup")
def signup_page(request: Request):
    return _auth_page(request, "signup")


@router.post("/signup")
def signup(request: Request, data: Credentials):
    email = data.email.strip().lower()

    if len(email) > 254 or not EMAIL_PATTERN.fullmatch(email):
        return _error("Please enter a valid email address.", 400)
    if not MIN_PASSWORD <= len(data.password) <= MAX_PASSWORD:
        return _error(
            f"Password must be {MIN_PASSWORD} to {MAX_PASSWORD} characters long.", 400
        )
    if not _signup_by_ip.hit(_client_ip(request)):
        return _error(_TOO_MANY, 429)

    user_id = create_user(email, hash_password(data.password))
    if user_id is None:
        return _error("An account with this email already exists.", 409)

    return _start_session(user_id)


@router.post("/login")
def login(request: Request, data: Credentials):
    email = data.email.strip().lower()
    ip = _client_ip(request)
    account_key = f"{ip}|{email[:254]}"

    if not _login_by_ip.hit(ip) or not _login_by_account.hit(account_key):
        return _error(_TOO_MANY, 429)

    user = get_user_by_email(email)
    password_ok = len(data.password) <= MAX_PASSWORD and verify_password(
        data.password, user.password_hash if user else _DUMMY_HASH
    )
    if not user or not password_ok:
        return _error("Incorrect email or password.", 401)

    _login_by_account.reset(account_key)
    return _start_session(user.id)


@router.post("/logout")
def logout(request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        delete_session(_token_hash(token))
    response = JSONResponse({"success": True})
    response.delete_cookie(SESSION_COOKIE)
    return response
