"""Application configuration for VYRAQ AI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=False)


def _get_setting(key: str, default: str = "") -> str:
    """Read a setting from environment variables, falling back to Streamlit secrets if present."""
    env_val = os.getenv(key, "").strip()
    if env_val:
        return env_val
    try:
        import streamlit as st  # local import to avoid hard dependency during non-UI imports

        if key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    return default


def _parse_bool(value: str, default: bool) -> bool:
    if not value:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AppConfig:
    """Central configuration loaded from environment variables or Streamlit secrets."""

    app_name: str = "VYRAQ AI"
    app_subtitle: str = "AI-Powered Resume & Job Compatibility Analyzer"
    base_dir: Path = BASE_DIR
    db_path: Path = BASE_DIR / "vyraq.db"
    upload_dir: Path = BASE_DIR / "uploads"
    vector_store_dir: Path = BASE_DIR / "vector_store"

    # Deployment UX mode:
    # public_mode=True hides API keys, Ollama setup, and Settings from regular end users.
    # Set VYRAQ_PUBLIC_MODE=false for local developer mode, or add owner emails to VYRAQ_ADMIN_EMAILS.
    public_mode: bool = True
    admin_emails: tuple[str, ...] = ()

    # LLM defaults (configured by site owner on the server)
    default_provider: str = "groq"
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    gemini_embedding_model: str = "models/gemini-embedding-001"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    ollama_embedding_model: str = "nomic-embed-text"

    # RAG settings
    rag_chunk_size: int = 700
    rag_chunk_overlap: int = 120
    rag_top_k: int = 5


def get_config() -> AppConfig:
    """Return an AppConfig instance reflecting current environment variables and secrets."""
    db_env = _get_setting("VYRAQ_DB_PATH")
    upload_env = _get_setting("VYRAQ_UPLOAD_DIR")
    vector_env = _get_setting("VYRAQ_VECTOR_STORE_DIR")

    db_path = Path(db_env) if db_env else BASE_DIR / "vyraq.db"
    if not db_path.is_absolute():
        db_path = BASE_DIR / db_path

    upload_dir = Path(upload_env) if upload_env else BASE_DIR / "uploads"
    if not upload_dir.is_absolute():
        upload_dir = BASE_DIR / upload_dir

    vector_dir = Path(vector_env) if vector_env else BASE_DIR / "vector_store"
    if not vector_dir.is_absolute():
        vector_dir = BASE_DIR / vector_dir

    public_mode = _parse_bool(_get_setting("VYRAQ_PUBLIC_MODE", "true"), default=True)
    raw_admins = _get_setting("VYRAQ_ADMIN_EMAILS")
    admin_emails = tuple(e.strip().lower() for e in raw_admins.split(",") if e.strip())

    groq_key = _get_setting("GROQ_API_KEY")
    gemini_key = _get_setting("GEMINI_API_KEY") or _get_setting("GOOGLE_API_KEY")
    default_provider = _get_setting("VYRAQ_DEFAULT_PROVIDER", "groq").lower() or "groq"

    return AppConfig(
        db_path=db_path,
        upload_dir=upload_dir,
        vector_store_dir=vector_dir,
        public_mode=public_mode,
        admin_emails=admin_emails,
        default_provider=default_provider,
        groq_api_key=groq_key,
        groq_model=_get_setting("GROQ_MODEL", "openai/gpt-oss-120b") or "openai/gpt-oss-120b",
        gemini_api_key=gemini_key,
        gemini_model=_get_setting("GEMINI_MODEL", "gemini-3.8-flash") or "gemini-3.8-flash",
        gemini_embedding_model=_get_setting("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
        or "models/gemini-embedding-001",
        ollama_base_url=(_get_setting("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        or "http://localhost:11434",
        ollama_model=_get_setting("OLLAMA_MODEL", "llama3.2") or "llama3.2",
        ollama_embedding_model=_get_setting("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        or "nomic-embed-text",
    )


def can_access_admin_settings(user_email: str | None, cfg: AppConfig | None = None) -> bool:
    """Return True if the active user should see provider/model Settings."""
    config = cfg or get_config()
    if not config.public_mode:
        return True
    if not user_email:
        return False
    return user_email.strip().lower() in config.admin_emails
