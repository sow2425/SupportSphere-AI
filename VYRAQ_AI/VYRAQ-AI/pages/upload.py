"""Resume and Job Description upload & LangGraph analysis execution page."""

from __future__ import annotations

import logging
import traceback

import streamlit as st

logger = logging.getLogger(__name__)

from ai.gemini_provider import is_gemini_configured
from ai.groq_provider import is_groq_configured
from ai.langgraph_workflow import run_compatibility_workflow
from ai.llm_factory import LLMFactoryError
from ai.ollama_provider import check_ollama_health
from config import can_access_admin_settings, get_config
from database.db import (
    get_user_settings,
    save_analysis_record,
    save_job_record,
    save_resume_record,
)
from modules.document_parser import (
    SUPPORTED_JOB_EXTENSIONS,
    SUPPORTED_RESUME_EXTENSIONS,
    DocumentParseError,
    extract_document_text,
    normalize_extracted_text,
    save_uploaded_file,
)


def render_upload_page() -> None:
    """Render the Resume & Job Description upload and analysis runner."""
    cfg = get_config()
    user_id = int(st.session_state["user_id"])
    user_email = st.session_state.get("user_email", "")
    is_admin = can_access_admin_settings(user_email, cfg)

    default_model = (
        cfg.groq_model
        if cfg.default_provider == "groq"
        else (cfg.gemini_model if cfg.default_provider == "gemini" else cfg.ollama_model)
    )
    settings = get_user_settings(
        cfg.db_path,
        user_id=user_id,
        default_provider=cfg.default_provider,
        default_model=default_model,
        default_ollama_url=cfg.ollama_base_url,
    )

    st.title("Upload & Analyze")
    st.caption(
        "Upload your resume and target job description to generate your ATS score, compatibility breakdown, skill gaps, and interview prep."
    )

    if is_admin:
        provider = settings.preferred_provider
        model_name = settings.preferred_model
        ollama_url = settings.ollama_base_url
        if provider == "groq":
            active_api_key = st.session_state.get("session_groq_api_key", "") or cfg.groq_api_key
        elif provider == "gemini":
            active_api_key = st.session_state.get("session_gemini_api_key", "") or cfg.gemini_api_key
        else:
            active_api_key = ""
    else:
        # In public mode, always use the server owner's configured provider & model
        provider = cfg.default_provider
        if provider == "groq":
            model_name = cfg.groq_model
            active_api_key = cfg.groq_api_key
        elif provider == "gemini":
            model_name = cfg.gemini_model
            active_api_key = cfg.gemini_api_key
        else:
            model_name = cfg.ollama_model
            active_api_key = ""
        ollama_url = cfg.ollama_base_url

    # Provider readiness banner (detailed for admins; clean UX for regular users)
    if provider == "groq":
        groq_ready = is_groq_configured(active_api_key)
        if is_admin:
            if groq_ready:
                st.success(f"Active Model: **Groq** (`{model_name}`) — API key configured.")
            else:
                st.warning(
                    "Active provider is **Groq**, but `GROQ_API_KEY` is not configured. "
                    "Add it in `.env` or enter a session key on the **Settings** page."
                )
        elif not groq_ready:
            st.warning("AI analysis service is temporarily unavailable while server configuration is pending.")
    elif provider == "gemini":
        gemini_ready = is_gemini_configured(active_api_key)
        if is_admin:
            if gemini_ready:
                st.success(f"Active Model: **Google Gemini** (`{model_name}`) — API key configured.")
            else:
                st.warning(
                    "Active provider is **Google Gemini**, but `GEMINI_API_KEY` is not configured. "
                    "Add it in `.env` or enter a session key on the **Settings** page, or switch to Groq/Ollama."
                )
        elif not gemini_ready:
            st.warning("AI analysis service is temporarily unavailable while server configuration is pending.")
    else:
        ollama_status = check_ollama_health(ollama_url, timeout=1.5)
        if is_admin:
            if ollama_status.available:
                st.success(f"Active Model: **Ollama** (`{model_name}`) — {ollama_status.message}")
            else:
                st.warning(
                    f"Active provider is **Ollama** (`{model_name}`), but server at `{ollama_url}` is unreachable. "
                    "Start Ollama (`ollama serve`) or switch to Groq/Gemini in **Settings**."
                )
        elif not ollama_status.available:
            st.warning("AI analysis service is temporarily unavailable. Please try again shortly.")

    col_resume, col_job = st.columns(2)

    with col_resume:
        st.subheader("1. Upload Resume")
        resume_file = st.file_uploader(
            "Choose Resume (PDF or DOCX)",
            type=["pdf", "docx"],
            key="vyraq_resume_uploader",
        )
        if resume_file is not None:
            st.caption(f"Selected: `{resume_file.name}` ({resume_file.size / 1024:.1f} KB)")

    with col_job:
        st.subheader("2. Job Description")
        jd_mode = st.radio(
            "Job Description Input Method",
            options=["Paste Text", "Upload File (PDF, DOCX, TXT)"],
            horizontal=True,
        )
        job_title_hint = st.text_input(
            "Target Job Title (Optional)",
            placeholder="e.g., Senior Python AI Engineer",
        )
        job_file = None
        pasted_jd = ""
        if jd_mode == "Upload File (PDF, DOCX, TXT)":
            job_file = st.file_uploader(
                "Choose Job Description",
                type=["pdf", "docx", "txt"],
                key="vyraq_jd_uploader",
            )
        else:
            pasted_jd = st.text_area(
                "Paste Job Description Text",
                height=210,
                placeholder="Paste the full job posting including responsibilities and required skills...",
            )

    st.write("")
    run_clicked = st.button("Run Compatibility Analysis", type="primary", use_container_width=True)

    if not run_clicked:
        return

    if resume_file is None:
        st.error("Please upload your resume in PDF or DOCX format.")
        return

    resume_bytes = resume_file.getvalue()
    try:
        resume_text, resume_ext = extract_document_text(
            resume_file.name,
            resume_bytes,
            allowed_extensions=SUPPORTED_RESUME_EXTENSIONS,
        )
    except DocumentParseError as exc:
        st.error(f"Resume parsing failed: {exc}")
        return

    job_bytes: bytes | None = None
    job_filename = "pasted_job_description.txt"
    job_ext = "txt"
    if jd_mode == "Upload File (PDF, DOCX, TXT)":
        if job_file is None:
            st.error("Please upload a job description file or switch to 'Paste Text'.")
            return
        job_bytes = job_file.getvalue()
        job_filename = job_file.name
        try:
            job_text, job_ext = extract_document_text(
                job_filename,
                job_bytes,
                allowed_extensions=SUPPORTED_JOB_EXTENSIONS,
            )
        except DocumentParseError as exc:
            st.error(f"Job description parsing failed: {exc}")
            return
    else:
        job_text = normalize_extracted_text(pasted_jd)
        if len(job_text) < 30:
            st.error("Please paste a complete job description (at least 30 characters).")
            return

    progress_bar = st.progress(0, text="Initializing analysis...")
    status_box = st.status("Analyzing your resume and job description...", expanded=True)

    def _on_step(step_num: int, total: int, label: str) -> None:
        pct = int((step_num / total) * 100)
        progress_bar.progress(pct, text=f"Step {step_num}/{total}: {label}")
        status_box.write(f"Completed Step {step_num}/{total}: **{label}**")

    try:
        final_state = run_compatibility_workflow(
            user_id=user_id,
            resume_text=resume_text,
            job_text=job_text,
            resume_filename=resume_file.name,
            job_filename=job_filename,
            job_title_hint=job_title_hint,
            provider=provider,
            model_name=model_name,
            api_key=active_api_key,
            ollama_base_url=ollama_url,
            db_path=str(cfg.db_path),
            vector_store_dir=str(cfg.vector_store_dir),
            progress_callback=_on_step,
        )
    except (LLMFactoryError, DocumentParseError, ValueError) as exc:
        logger.exception("Analysis workflow failed")
        traceback.print_exc()
        status_box.update(label="Analysis could not complete", state="error")
        if is_admin:
            st.error(f"Analysis failed: {exc}")
        else:
            st.error("We couldn't complete the AI analysis right now. Please try again in a moment.")
        return
    except Exception as exc:
        logger.exception("Unexpected error during analysis workflow")
        traceback.print_exc()
        status_box.update(label="Unexpected error during analysis", state="error")
        if is_admin:
            st.error(f"An unexpected error occurred: {exc}")
        else:
            st.error("Something went wrong while analyzing your documents. Please try again.")
        return

    saved_resume_path = save_uploaded_file(cfg.upload_dir, user_id, resume_file.name, resume_bytes)
    resume_rec = save_resume_record(
        cfg.db_path,
        user_id=user_id,
        filename=resume_file.name,
        file_type=resume_ext,
        file_path=str(saved_resume_path),
        extracted_text=resume_text,
    )

    saved_job_path = ""
    if job_bytes is not None:
        saved_job_path = str(save_uploaded_file(cfg.upload_dir, user_id, job_filename, job_bytes))

    report = final_state["final_report"]
    resolved_job_title = job_title_hint.strip() or report.get("job_title") or "Target Role"
    job_rec = save_job_record(
        cfg.db_path,
        user_id=user_id,
        title=resolved_job_title,
        source_name=job_filename,
        source_type=job_ext,
        file_path=saved_job_path,
        extracted_text=job_text,
    )

    ats_score = float(report["ats_evaluation"]["ats_score"])
    comp_score = float(report["compatibility"]["compatibility_score"])
    summary = str(final_state.get("executive_summary", ""))

    analysis_rec = save_analysis_record(
        cfg.db_path,
        user_id=user_id,
        resume_id=resume_rec.id,
        job_id=job_rec.id,
        resume_filename=resume_file.name,
        job_title=resolved_job_title,
        provider=provider,
        model_name=model_name,
        ats_score=ats_score,
        compatibility_score=comp_score,
        report_data=report,
        summary=summary,
    )

    status_box.update(label="Analysis complete and saved!", state="complete")
    st.session_state["selected_analysis_id"] = analysis_rec.id
    st.success("Report generated and saved to your history.")
    st.session_state["current_page"] = "Analysis Results"
    st.rerun()
