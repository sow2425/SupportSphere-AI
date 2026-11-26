"""Data models for VYRAQ AI SQLite persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class User:
    """Registered user account record."""

    id: int
    full_name: str
    email: str
    password_hash: str
    created_at: str


@dataclass
class ResumeRecord:
    """Uploaded resume metadata and extracted text."""

    id: int
    user_id: int
    filename: str
    file_type: str
    file_path: str
    extracted_text: str
    word_count: int
    created_at: str


@dataclass
class JobRecord:
    """Uploaded or pasted job description record."""

    id: int
    user_id: int
    title: str
    source_name: str
    source_type: str
    file_path: str
    extracted_text: str
    word_count: int
    created_at: str


@dataclass
class AnalysisRecord:
    """Saved resume-to-job compatibility analysis report."""

    id: int
    user_id: int
    resume_id: int
    job_id: int
    resume_filename: str
    job_title: str
    provider: str
    model_name: str
    ats_score: float
    compatibility_score: float
    report_data: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    created_at: str = ""


@dataclass
class UserSettings:
    """Per-user model provider preferences."""

    user_id: int
    preferred_provider: str
    preferred_model: str
    ollama_base_url: str
    updated_at: str
