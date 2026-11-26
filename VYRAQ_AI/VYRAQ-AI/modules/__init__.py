"""Domain service modules for VYRAQ AI."""

from modules.compatibility import CompatibilityResult, calculate_compatibility
from modules.document_parser import (
    SUPPORTED_JOB_EXTENSIONS,
    SUPPORTED_RESUME_EXTENSIONS,
    DocumentParseError,
    extract_document_text,
    extract_text_from_docx,
    extract_text_from_pdf,
    extract_text_from_txt,
    save_uploaded_file,
)
from modules.interview_questions import InterviewPrepResult, generate_interview_questions
from modules.job_analyzer import JobAnalysisResult, analyze_job_description
from modules.resume_analyzer import ResumeAnalysisResult, analyze_resume, compute_ats_signals
from modules.skill_gap import SkillGapResult, identify_missing_skills

__all__ = [
    "CompatibilityResult",
    "DocumentParseError",
    "InterviewPrepResult",
    "JobAnalysisResult",
    "ResumeAnalysisResult",
    "SUPPORTED_JOB_EXTENSIONS",
    "SUPPORTED_RESUME_EXTENSIONS",
    "SkillGapResult",
    "analyze_job_description",
    "analyze_resume",
    "calculate_compatibility",
    "compute_ats_signals",
    "extract_document_text",
    "extract_text_from_docx",
    "extract_text_from_pdf",
    "extract_text_from_txt",
    "generate_interview_questions",
    "identify_missing_skills",
    "save_uploaded_file",
]
