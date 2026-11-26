"""Automated tests for authentication, password hashing, session state, and user isolation."""

from __future__ import annotations

from pathlib import Path

from authentication.auth import (
    get_current_user,
    hash_password,
    init_session_state,
    is_authenticated,
    login_user,
    logout_user,
    register_user,
    set_authenticated_user,
    verify_password,
)
from config import can_access_admin_settings, get_config
from database.db import (
    delete_user_analysis,
    get_user_analysis,
    list_user_analyses,
    save_analysis_record,
    save_job_record,
    save_resume_record,
)


def test_password_hashing_and_verification() -> None:
    hashed = hash_password("SecurePass123!")
    assert hashed.startswith("pbkdf2_sha256$")
    assert verify_password("SecurePass123!", hashed) is True
    assert verify_password("WrongPassword!", hashed) is False


def test_register_duplicate_email_and_login(tmp_db: Path) -> None:
    res1 = register_user(tmp_db, "Alex Rivera", "Alex@Example.com", "StrongPass99", "StrongPass99")
    assert res1.success is True
    assert res1.user is not None
    assert res1.user.email == "alex@example.com"

    # Duplicate email (case-insensitive)
    dup = register_user(tmp_db, "Alex Two", "alex@example.com", "StrongPass99", "StrongPass99")
    assert dup.success is False
    assert "already exists" in dup.message.lower()

    # Weak / mismatched validation
    bad_pw = register_user(tmp_db, "Bob", "bob@example.com", "short", "short")
    assert bad_pw.success is False

    # Login success & failure
    ok_login = login_user(tmp_db, "alex@example.com", "StrongPass99")
    assert ok_login.success is True
    assert ok_login.user is not None and ok_login.user.id == res1.user.id

    fail_login = login_user(tmp_db, "alex@example.com", "Incorrect99")
    assert fail_login.success is False


def test_session_management_and_user_history_isolation(tmp_db: Path) -> None:
    u1 = register_user(tmp_db, "User One", "u1@example.com", "Password123").user
    u2 = register_user(tmp_db, "User Two", "u2@example.com", "Password123").user
    assert u1 is not None and u2 is not None

    session: dict = {}
    init_session_state(session)
    assert is_authenticated(session) is False

    set_authenticated_user(session, u1)
    assert is_authenticated(session) is True
    resolved = get_current_user(tmp_db, session)
    assert resolved is not None and resolved.email == "u1@example.com"

    # Save an analysis owned by User One
    r1 = save_resume_record(tmp_db, u1.id, "r1.pdf", "pdf", "", "Python developer resume text")
    j1 = save_job_record(tmp_db, u1.id, "Python Engineer", "jd.txt", "txt", "", "Python engineer job description")
    a1 = save_analysis_record(
        tmp_db,
        user_id=u1.id,
        resume_id=r1.id,
        job_id=j1.id,
        resume_filename="r1.pdf",
        job_title="Python Engineer",
        provider="ollama",
        model_name="llama3.2",
        ats_score=85.0,
        compatibility_score=80.0,
        report_data={"job_title": "Python Engineer"},
        summary="Test summary",
    )

    assert len(list_user_analyses(tmp_db, u1.id)) == 1
    assert len(list_user_analyses(tmp_db, u2.id)) == 0
    assert get_user_analysis(tmp_db, u2.id, a1.id) is None
    assert delete_user_analysis(tmp_db, u2.id, a1.id) is False
    assert get_user_analysis(tmp_db, u1.id, a1.id) is not None

    logout_user(session)
    assert is_authenticated(session) is False
    assert get_current_user(tmp_db, session) is None


def test_public_mode_hides_admin_settings_for_regular_users(monkeypatch) -> None:
    monkeypatch.setenv("VYRAQ_PUBLIC_MODE", "true")
    monkeypatch.setenv("VYRAQ_ADMIN_EMAILS", "owner@example.com")
    cfg = get_config()
    assert can_access_admin_settings("visitor@example.com", cfg) is False
    assert can_access_admin_settings("owner@example.com", cfg) is True

    monkeypatch.setenv("VYRAQ_PUBLIC_MODE", "false")
    dev_cfg = get_config()
    assert can_access_admin_settings("visitor@example.com", dev_cfg) is True
