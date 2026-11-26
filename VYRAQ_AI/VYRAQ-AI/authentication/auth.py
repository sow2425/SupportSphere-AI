"""User authentication, password hashing, and Streamlit session management."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from database.db import create_user, get_user_by_email, get_user_by_id
from database.models import User

EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
PBKDF2_ITERATIONS = 200_000


@dataclass
class AuthResult:
    """Outcome of a registration or login operation."""

    success: bool
    message: str
    user: User | None = None


def hash_password(password: str) -> str:
    """Hash a plaintext password using PBKDF2-HMAC-SHA256 with a random salt."""
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt}${dk.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a plaintext password against a stored PBKDF2-HMAC-SHA256 hash."""
    try:
        algorithm, iter_str, salt, expected_hex = stored_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iter_str)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations)
        return hmac.compare_digest(dk.hex(), expected_hex)
    except (ValueError, TypeError):
        return False


def validate_email(email: str) -> bool:
    """Return True if email matches standard email syntax."""
    return bool(EMAIL_REGEX.match(email.strip()))


def register_user(
    db_path: str | Path,
    full_name: str,
    email: str,
    password: str,
    confirm_password: str | None = None,
) -> AuthResult:
    """Validate inputs, enforce unique email, hash password, and create user."""
    clean_name = full_name.strip()
    clean_email = email.strip().lower()

    if len(clean_name) < 2:
        return AuthResult(False, "Full name must be at least 2 characters.")
    if not validate_email(clean_email):
        return AuthResult(False, "Please enter a valid email address.")
    if len(password) < 8:
        return AuthResult(False, "Password must be at least 8 characters long.")
    if confirm_password is not None and password != confirm_password:
        return AuthResult(False, "Passwords do not match.")

    existing = get_user_by_email(db_path, clean_email)
    if existing is not None:
        return AuthResult(False, "An account with this email already exists.")

    pw_hash = hash_password(password)
    user = create_user(db_path, clean_name, clean_email, pw_hash)
    return AuthResult(True, "Account created successfully. You can now sign in.", user=user)


def login_user(db_path: str | Path, email: str, password: str) -> AuthResult:
    """Authenticate an existing user by email and password."""
    clean_email = email.strip().lower()
    if not validate_email(clean_email):
        return AuthResult(False, "Please enter a valid email address.")
    if not password:
        return AuthResult(False, "Password is required.")

    user = get_user_by_email(db_path, clean_email)
    if user is None or not verify_password(password, user.password_hash):
        return AuthResult(False, "Invalid email or password.")

    return AuthResult(True, f"Welcome back, {user.full_name}!", user=user)


def init_session_state(session_state: Any) -> None:
    """Initialize required authentication and navigation keys in Streamlit session_state."""
    defaults: dict[str, Any] = {
        "authenticated": False,
        "user_id": None,
        "user_email": None,
        "user_name": None,
        "auth_page": "login",
        "current_page": "Dashboard",
        "selected_analysis_id": None,
        "session_gemini_api_key": "",
        "session_groq_api_key": "",
    }
    for key, val in defaults.items():
        if key not in session_state:
            session_state[key] = val


def set_authenticated_user(session_state: Any, user: User) -> None:
    """Store logged-in user identity in session state."""
    session_state["authenticated"] = True
    session_state["user_id"] = user.id
    session_state["user_email"] = user.email
    session_state["user_name"] = user.full_name
    session_state["current_page"] = "Dashboard"


def logout_user(session_state: Any) -> None:
    """Clear authentication state and active analysis selection."""
    session_state["authenticated"] = False
    session_state["user_id"] = None
    session_state["user_email"] = None
    session_state["user_name"] = None
    session_state["selected_analysis_id"] = None
    session_state["auth_page"] = "login"
    session_state["current_page"] = "Dashboard"


def is_authenticated(session_state: Any) -> bool:
    """Return True if a valid authenticated user is active in session state."""
    return bool(session_state.get("authenticated") and session_state.get("user_id"))


def get_current_user(db_path: str | Path, session_state: Any) -> User | None:
    """Resolve and verify the active session user from SQLite."""
    if not is_authenticated(session_state):
        return None
    user_id = session_state.get("user_id")
    if user_id is None:
        return None
    return get_user_by_id(db_path, int(user_id))
