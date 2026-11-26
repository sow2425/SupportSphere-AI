"""VYRAQ AI – AI-Powered Resume and Job Compatibility Analyzer (Streamlit entrypoint)."""

from __future__ import annotations

import streamlit as st

from authentication.auth import (
    get_current_user,
    init_session_state,
    is_authenticated,
    logout_user,
)
from config import can_access_admin_settings, get_config
from database.db import init_db
from pages.auth_pages import render_auth_pages
from pages.dashboard import render_dashboard
from pages.history import render_history_page
from pages.results import render_results_page
from pages.settings import render_settings_page
from pages.upload import render_upload_page

USER_NAV_PAGES = [
    "Dashboard",
    "Upload & Analyze",
    "Analysis Results",
    "History",
]
ADMIN_NAV_PAGES = USER_NAV_PAGES + ["Settings"]


def _inject_custom_styles() -> None:
    """Apply clean modern styling inside Streamlit."""
    st.markdown(
        """
        <style>
        .vyraq-hero {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.22), rgba(16, 185, 129, 0.14));
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 16px;
            padding: 1.4rem 1.6rem;
        }
        .vyraq-badge {
            display: inline-block;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            padding: 0.22rem 0.65rem;
            border-radius: 999px;
            background: rgba(99, 102, 241, 0.28);
            border: 1px solid rgba(129, 140, 248, 0.45);
        }
        div[data-testid="stMetric"] {
            background: rgba(30, 41, 59, 0.55);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 12px;
            padding: 0.85rem 1rem;
        }
        [data-testid="stAppDeployButton"],
        .stAppDeployButton {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Configure Streamlit, enforce authentication, and route to the active page."""
    st.set_page_config(
        page_title="VYRAQ AI – Resume & Job Compatibility Analyzer",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    cfg = get_config()
    cfg.upload_dir.mkdir(parents=True, exist_ok=True)
    init_db(cfg.db_path)
    init_session_state(st.session_state)
    _inject_custom_styles()

    # Mandatory Authentication Gate
    current_user = get_current_user(cfg.db_path, st.session_state) if is_authenticated(st.session_state) else None
    if current_user is None:
        if is_authenticated(st.session_state):
            logout_user(st.session_state)
        render_auth_pages()
        return

    # Authenticated Sidebar Navigation
    show_settings = can_access_admin_settings(current_user.email, cfg)
    nav_pages = ADMIN_NAV_PAGES if show_settings else USER_NAV_PAGES

    with st.sidebar:
        st.markdown("## VYRAQ AI")
        st.caption(f"Signed in as **{current_user.full_name}**")
        st.caption(current_user.email)
        st.divider()

        current_page = st.session_state.get("current_page", "Dashboard")
        if current_page not in nav_pages:
            current_page = "Dashboard"
            st.session_state["current_page"] = "Dashboard"

        for page_name in nav_pages:
            btn_type = "primary" if current_page == page_name else "secondary"
            if st.button(page_name, key=f"nav_{page_name}", type=btn_type, use_container_width=True):
                st.session_state["current_page"] = page_name
                st.rerun()

        st.divider()
        if st.button("Logout", use_container_width=True):
            logout_user(st.session_state)
            st.rerun()

    # Route to selected page
    active = st.session_state.get("current_page", "Dashboard")
    if active == "Dashboard":
        render_dashboard()
    elif active == "Upload & Analyze":
        render_upload_page()
    elif active == "Analysis Results":
        render_results_page()
    elif active == "History":
        render_history_page()
    elif active == "Settings" and show_settings:
        render_settings_page()
    else:
        render_dashboard()


if __name__ == "__main__":
    main()
