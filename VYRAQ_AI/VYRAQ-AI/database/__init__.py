"""Database package exports."""

from database.db import (
    create_user,
    delete_user_analysis,
    get_connection,
    get_user_analysis,
    get_user_by_email,
    get_user_by_id,
    get_user_settings,
    init_db,
    list_user_analyses,
    list_user_resumes,
    save_analysis_record,
    save_job_record,
    save_resume_record,
    upsert_user_settings,
)
from database.models import (
    AnalysisRecord,
    JobRecord,
    ResumeRecord,
    User,
    UserSettings,
)

__all__ = [
    "AnalysisRecord",
    "JobRecord",
    "ResumeRecord",
    "User",
    "UserSettings",
    "create_user",
    "delete_user_analysis",
    "get_connection",
    "get_user_analysis",
    "get_user_by_email",
    "get_user_by_id",
    "get_user_settings",
    "init_db",
    "list_user_analyses",
    "list_user_resumes",
    "save_analysis_record",
    "save_job_record",
    "save_resume_record",
    "upsert_user_settings",
]
