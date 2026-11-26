"""SQLite database initialization and user-scoped repository helpers."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from database.models import (
    AnalysisRecord,
    JobRecord,
    ResumeRecord,
    User,
    UserSettings,
)


@contextmanager
def get_connection(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection with foreign keys and dict-like row access."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str | Path) -> None:
    """Initialize all SQLite tables and indexes if they do not already exist."""
    with get_connection(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_users_email ON users (email);

            CREATE TABLE IF NOT EXISTS resumes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_path TEXT NOT NULL DEFAULT '',
                extracted_text TEXT NOT NULL,
                word_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_resumes_user_id ON resumes (user_id);

            CREATE TABLE IF NOT EXISTS job_descriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                source_name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                file_path TEXT NOT NULL DEFAULT '',
                extracted_text TEXT NOT NULL,
                word_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_jobs_user_id ON job_descriptions (user_id);

            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                resume_id INTEGER NOT NULL,
                job_id INTEGER NOT NULL,
                resume_filename TEXT NOT NULL,
                job_title TEXT NOT NULL,
                provider TEXT NOT NULL,
                model_name TEXT NOT NULL,
                ats_score REAL NOT NULL DEFAULT 0.0,
                compatibility_score REAL NOT NULL DEFAULT 0.0,
                report_json TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                FOREIGN KEY (resume_id) REFERENCES resumes (id) ON DELETE CASCADE,
                FOREIGN KEY (job_id) REFERENCES job_descriptions (id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_analyses_user_created
                ON analyses (user_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS user_settings (
                user_id INTEGER PRIMARY KEY,
                preferred_provider TEXT NOT NULL DEFAULT 'ollama',
                preferred_model TEXT NOT NULL DEFAULT 'llama3.2',
                ollama_base_url TEXT NOT NULL DEFAULT 'http://localhost:11434',
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS vector_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                collection_name TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                source_name TEXT NOT NULL DEFAULT '',
                chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                embedding_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_vector_user_collection
                ON vector_chunks (user_id, collection_name);
            """
        )


def _row_to_user(row: sqlite3.Row) -> User:
    return User(
        id=int(row["id"]),
        full_name=str(row["full_name"]),
        email=str(row["email"]),
        password_hash=str(row["password_hash"]),
        created_at=str(row["created_at"]),
    )


def create_user(db_path: str | Path, full_name: str, email: str, password_hash: str) -> User:
    """Insert a new user and return the created User model."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO users (full_name, email, password_hash)
            VALUES (?, ?, ?)
            """,
            (full_name.strip(), email.strip().lower(), password_hash),
        )
        user_id = int(cursor.lastrowid)
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return _row_to_user(row)


def get_user_by_email(db_path: str | Path, email: str) -> User | None:
    """Fetch a user record by normalized email."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?",
            (email.strip().lower(),),
        ).fetchone()
        return _row_to_user(row) if row else None


def get_user_by_id(db_path: str | Path, user_id: int) -> User | None:
    """Fetch a user record by ID."""
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return _row_to_user(row) if row else None


def save_resume_record(
    db_path: str | Path,
    user_id: int,
    filename: str,
    file_type: str,
    file_path: str,
    extracted_text: str,
) -> ResumeRecord:
    """Persist an uploaded resume for a user."""
    word_count = len(extracted_text.split())
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO resumes (user_id, filename, file_type, file_path, extracted_text, word_count)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, filename, file_type, file_path, extracted_text, word_count),
        )
        row = conn.execute("SELECT * FROM resumes WHERE id = ?", (int(cursor.lastrowid),)).fetchone()
        return ResumeRecord(
            id=int(row["id"]),
            user_id=int(row["user_id"]),
            filename=str(row["filename"]),
            file_type=str(row["file_type"]),
            file_path=str(row["file_path"]),
            extracted_text=str(row["extracted_text"]),
            word_count=int(row["word_count"]),
            created_at=str(row["created_at"]),
        )


def save_job_record(
    db_path: str | Path,
    user_id: int,
    title: str,
    source_name: str,
    source_type: str,
    file_path: str,
    extracted_text: str,
) -> JobRecord:
    """Persist a job description for a user."""
    word_count = len(extracted_text.split())
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO job_descriptions (user_id, title, source_name, source_type, file_path, extracted_text, word_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, title.strip() or "Target Role", source_name, source_type, file_path, extracted_text, word_count),
        )
        row = conn.execute("SELECT * FROM job_descriptions WHERE id = ?", (int(cursor.lastrowid),)).fetchone()
        return JobRecord(
            id=int(row["id"]),
            user_id=int(row["user_id"]),
            title=str(row["title"]),
            source_name=str(row["source_name"]),
            source_type=str(row["source_type"]),
            file_path=str(row["file_path"]),
            extracted_text=str(row["extracted_text"]),
            word_count=int(row["word_count"]),
            created_at=str(row["created_at"]),
        )


def _row_to_analysis(row: sqlite3.Row) -> AnalysisRecord:
    return AnalysisRecord(
        id=int(row["id"]),
        user_id=int(row["user_id"]),
        resume_id=int(row["resume_id"]),
        job_id=int(row["job_id"]),
        resume_filename=str(row["resume_filename"]),
        job_title=str(row["job_title"]),
        provider=str(row["provider"]),
        model_name=str(row["model_name"]),
        ats_score=float(row["ats_score"]),
        compatibility_score=float(row["compatibility_score"]),
        report_data=json.loads(row["report_json"]),
        summary=str(row["summary"]),
        created_at=str(row["created_at"]),
    )


def save_analysis_record(
    db_path: str | Path,
    user_id: int,
    resume_id: int,
    job_id: int,
    resume_filename: str,
    job_title: str,
    provider: str,
    model_name: str,
    ats_score: float,
    compatibility_score: float,
    report_data: dict[str, Any],
    summary: str,
) -> AnalysisRecord:
    """Persist a completed analysis report for the logged-in user."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO analyses (
                user_id, resume_id, job_id, resume_filename, job_title,
                provider, model_name, ats_score, compatibility_score, report_json, summary
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                resume_id,
                job_id,
                resume_filename,
                job_title,
                provider,
                model_name,
                float(ats_score),
                float(compatibility_score),
                json.dumps(report_data, ensure_ascii=False),
                summary,
            ),
        )
        row = conn.execute("SELECT * FROM analyses WHERE id = ?", (int(cursor.lastrowid),)).fetchone()
        return _row_to_analysis(row)


def list_user_analyses(db_path: str | Path, user_id: int, limit: int = 50) -> list[AnalysisRecord]:
    """Return analysis history belonging solely to user_id, newest first."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM analyses
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        return [_row_to_analysis(row) for row in rows]


def get_user_analysis(db_path: str | Path, user_id: int, analysis_id: int) -> AnalysisRecord | None:
    """Fetch a specific analysis only if owned by user_id."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM analyses WHERE id = ? AND user_id = ?",
            (analysis_id, user_id),
        ).fetchone()
        return _row_to_analysis(row) if row else None


def delete_user_analysis(db_path: str | Path, user_id: int, analysis_id: int) -> bool:
    """Delete a specific analysis record only if owned by user_id."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "DELETE FROM analyses WHERE id = ? AND user_id = ?",
            (analysis_id, user_id),
        )
        return cursor.rowcount > 0


def get_user_settings(
    db_path: str | Path,
    user_id: int,
    default_provider: str = "ollama",
    default_model: str = "llama3.2",
    default_ollama_url: str = "http://localhost:11434",
) -> UserSettings:
    """Fetch user model settings or initialize defaults on first read."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM user_settings WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            conn.execute(
                """
                INSERT INTO user_settings (user_id, preferred_provider, preferred_model, ollama_base_url)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, default_provider, default_model, default_ollama_url),
            )
            row = conn.execute(
                "SELECT * FROM user_settings WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return UserSettings(
            user_id=int(row["user_id"]),
            preferred_provider=str(row["preferred_provider"]),
            preferred_model=str(row["preferred_model"]),
            ollama_base_url=str(row["ollama_base_url"]),
            updated_at=str(row["updated_at"]),
        )


def upsert_user_settings(
    db_path: str | Path,
    user_id: int,
    preferred_provider: str,
    preferred_model: str,
    ollama_base_url: str,
) -> UserSettings:
    """Create or update a user's preferred LLM provider and model settings."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO user_settings (user_id, preferred_provider, preferred_model, ollama_base_url, updated_at)
            VALUES (?, ?, ?, ?, datetime('now'))
            ON CONFLICT(user_id) DO UPDATE SET
                preferred_provider = excluded.preferred_provider,
                preferred_model = excluded.preferred_model,
                ollama_base_url = excluded.ollama_base_url,
                updated_at = datetime('now')
            """,
            (user_id, preferred_provider, preferred_model, ollama_base_url.strip().rstrip("/")),
        )
    return get_user_settings(db_path, user_id)


def list_user_resumes(db_path: str | Path, user_id: int, limit: int = 20) -> list[ResumeRecord]:
    """List recent resumes uploaded by user_id."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM resumes WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [
            ResumeRecord(
                id=int(r["id"]),
                user_id=int(r["user_id"]),
                filename=str(r["filename"]),
                file_type=str(r["file_type"]),
                file_path=str(r["file_path"]),
                extracted_text=str(r["extracted_text"]),
                word_count=int(r["word_count"]),
                created_at=str(r["created_at"]),
            )
            for r in rows
        ]
