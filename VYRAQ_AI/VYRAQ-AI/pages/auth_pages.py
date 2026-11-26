"""Login and Registration pages rendered in Streamlit."""

from __future__ import annotations

import streamlit as st

from authentication.auth import login_user, register_user, set_authenticated_user
from config import get_config


def render_auth_pages() -> None:
    """Render mandatory Login and Registration pages before app access."""
    cfg = get_config()

    col_left, col_center, col_right = st.columns([1, 1.4, 1])
    with col_center:
        st.markdown(
            """
            <div class="vyraq-hero" style="text-align:center; margin-bottom:1.25rem;">
                <div class="vyraq-badge">VYRAQ AI • Career Intelligence</div>
                <h1 style="margin:0.4rem 0 0.25rem 0; font-size:2.1rem;">VYRAQ AI</h1>
                <p style="margin:0; opacity:0.82; font-size:0.98rem;">
                    AI-Powered Resume &amp; Job Compatibility Analyzer
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        active_mode = st.session_state.get("auth_page", "login")
        tab_login, tab_register = st.tabs(["Sign In", "Create Account"])

        with tab_login:
            if active_mode == "register":
                st.caption("Already registered? Sign in below.")
            with st.form("vyraq_login_form", clear_on_submit=False):
                email = st.text_input("Email Address", placeholder="you@example.com")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submitted = st.form_submit_button("Sign In", type="primary", use_container_width=True)

            if submitted:
                res = login_user(cfg.db_path, email=email, password=password)
                if res.success and res.user is not None:
                    set_authenticated_user(st.session_state, res.user)
                    st.success(res.message)
                    st.rerun()
                else:
                    st.error(res.message)

        with tab_register:
            with st.form("vyraq_register_form", clear_on_submit=False):
                full_name = st.text_input("Full Name", placeholder="Alex Rivera")
                reg_email = st.text_input("Email Address", placeholder="alex@example.com")
                reg_password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Minimum 8 characters",
                )
                reg_confirm = st.text_input(
                    "Confirm Password",
                    type="password",
                    placeholder="Re-enter your password",
                )
                reg_submitted = st.form_submit_button(
                    "Create Account",
                    type="primary",
                    use_container_width=True,
                )

            if reg_submitted:
                res = register_user(
                    cfg.db_path,
                    full_name=full_name,
                    email=reg_email,
                    password=reg_password,
                    confirm_password=reg_confirm,
                )
                if res.success and res.user is not None:
                    st.session_state["auth_page"] = "login"
                    st.success(res.message)
                else:
                    st.error(res.message)

        st.info(
            "Authentication is required to keep your resumes, job descriptions, vector embeddings, "
            "and analysis history private to your account."
        )
