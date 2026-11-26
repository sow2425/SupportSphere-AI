"""Analysis History page showing user-scoped saved reports."""

from __future__ import annotations

import streamlit as st

from config import can_access_admin_settings, get_config
from database.db import delete_user_analysis, list_user_analyses


def render_history_page() -> None:
    """Render searchable, user-specific analysis history."""
    cfg = get_config()
    user_id = int(st.session_state["user_id"])

    st.title("Analysis History")
    st.caption("All saved resume-to-job reports linked to your account.")

    analyses = list_user_analyses(cfg.db_path, user_id=user_id, limit=100)
    if not analyses:
        st.info("No saved analyses yet. Start by uploading a resume and job description.")
        return

    search_term = st.text_input(
        "Filter by Job Title or Resume Filename",
        placeholder="Type to filter history...",
    ).strip().lower()

    filtered = [
        a
        for a in analyses
        if not search_term
        or search_term in a.job_title.lower()
        or search_term in a.resume_filename.lower()
    ]

    if not filtered:
        st.warning("No reports match your search filter.")
        return

    is_admin = can_access_admin_settings(st.session_state.get("user_email", ""), cfg)
    for item in filtered:
        with st.container(border=True):
            c1, c2, c3 = st.columns([3.0, 1.2, 1.0])
            with c1:
                st.markdown(f"### {item.job_title}")
                if is_admin:
                    st.caption(
                        f"Resume: `{item.resume_filename}` • Model: `{item.provider}/{item.model_name}` • {item.created_at}"
                    )
                else:
                    st.caption(f"Resume: `{item.resume_filename}` • {item.created_at}")
                if item.summary:
                    st.write(item.summary)
            with c2:
                st.metric("ATS Score", f"{item.ats_score:.1f}/100")
                st.metric("Compatibility", f"{item.compatibility_score:.1f}%")
            with c3:
                st.write("")
                if st.button("View Report", key=f"hist_view_{item.id}", type="primary", use_container_width=True):
                    st.session_state["selected_analysis_id"] = item.id
                    st.session_state["current_page"] = "Analysis Results"
                    st.rerun()
                if st.button("Delete", key=f"hist_del_{item.id}", use_container_width=True):
                    delete_user_analysis(cfg.db_path, user_id=user_id, analysis_id=item.id)
                    if st.session_state.get("selected_analysis_id") == item.id:
                        st.session_state["selected_analysis_id"] = None
                    st.success("Report deleted.")
                    st.rerun()
