"""Dashboard page showing welcome summary, metrics, and recent analyses."""

from __future__ import annotations

import streamlit as st

from ai.gemini_provider import is_gemini_configured
from ai.groq_provider import is_groq_configured
from ai.ollama_provider import check_ollama_health
from config import can_access_admin_settings, get_config
from database.db import get_user_settings, list_user_analyses


def render_dashboard() -> None:
    """Render the main user dashboard."""
    cfg = get_config()
    user_id = int(st.session_state["user_id"])
    user_name = st.session_state.get("user_name", "User")
    user_email = st.session_state.get("user_email", "")
    is_admin = can_access_admin_settings(user_email, cfg)

    analyses = list_user_analyses(cfg.db_path, user_id=user_id, limit=25)
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

    st.markdown(
        f"""
        <div class="vyraq-hero">
            <div class="vyraq-badge">Welcome back</div>
            <h2 style="margin:0.35rem 0 0.25rem 0;">Hello, {user_name}</h2>
            <p style="margin:0; opacity:0.85;">
                Analyze your resume against target job descriptions with RAG-grounded insights,
                transparent ATS scoring, skill gap roadmaps, and tailored interview prep.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    c1, c2, c3, c4 = st.columns(4)
    total_count = len(analyses)
    avg_ats = round(sum(a.ats_score for a in analyses) / total_count, 1) if total_count else 0.0
    avg_comp = round(sum(a.compatibility_score for a in analyses) / total_count, 1) if total_count else 0.0

    c1.metric("Total Analyses", str(total_count))
    c2.metric("Avg ATS Score", f"{avg_ats:.1f} / 100" if total_count else "—")
    c3.metric("Avg Compatibility", f"{avg_comp:.1f}%" if total_count else "—")

    if is_admin:
        c4.metric("Active AI Model", f"{settings.preferred_provider.upper()} • {settings.preferred_model}")
    else:
        if cfg.default_provider == "groq":
            server_ready = is_groq_configured(cfg.groq_api_key)
        elif cfg.default_provider == "gemini":
            server_ready = is_gemini_configured(cfg.gemini_api_key)
        else:
            server_ready = check_ollama_health(cfg.ollama_base_url, timeout=1.0).available
        c4.metric("AI Engine", "Online" if server_ready else "Standby")

    st.write("")
    if is_admin:
        action_col1, action_col2, action_col3 = st.columns([1.2, 1.2, 1])
        with action_col1:
            if st.button("Start New Resume Analysis", type="primary", use_container_width=True):
                st.session_state["current_page"] = "Upload & Analyze"
                st.rerun()
        with action_col2:
            if st.button("Configure LLM Settings", use_container_width=True):
                st.session_state["current_page"] = "Settings"
                st.rerun()
        with action_col3:
            if st.button("View Full History", use_container_width=True):
                st.session_state["current_page"] = "History"
                st.rerun()
    else:
        action_col1, action_col2 = st.columns([1.4, 1.0])
        with action_col1:
            if st.button("Start New Resume Analysis", type="primary", use_container_width=True):
                st.session_state["current_page"] = "Upload & Analyze"
                st.rerun()
        with action_col2:
            if st.button("View Full History", use_container_width=True):
                st.session_state["current_page"] = "History"
                st.rerun()

    st.divider()
    st.subheader("Recent Analyses")

    if not analyses:
        st.info(
            "You haven't run an analysis yet. Click **Start New Resume Analysis** above to upload "
            "your resume and a job description."
        )
        return

    for item in analyses[:5]:
        with st.container(border=True):
            left, mid, right = st.columns([2.5, 1.4, 0.9])
            with left:
                st.markdown(f"**{item.job_title}**")
                meta_text = f"Resume: `{item.resume_filename}` • {item.created_at}"
                if is_admin:
                    meta_text = f"Resume: `{item.resume_filename}` • Model: `{item.provider}/{item.model_name}` • {item.created_at}"
                st.caption(meta_text)
                if item.summary:
                    st.write(item.summary)
            with mid:
                st.markdown(f"**ATS Score:** {item.ats_score:.1f}/100")
                st.progress(min(1.0, max(0.0, item.ats_score / 100.0)))
                st.markdown(f"**Job Match:** {item.compatibility_score:.1f}%")
                st.progress(min(1.0, max(0.0, item.compatibility_score / 100.0)))
            with right:
                st.write("")
                if st.button("Open Report", key=f"dash_open_{item.id}", use_container_width=True):
                    st.session_state["selected_analysis_id"] = item.id
                    st.session_state["current_page"] = "Analysis Results"
                    st.rerun()
