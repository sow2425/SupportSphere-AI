"""Settings and LLM model selection page (Groq, Ollama local, and Google Gemini API)."""

from __future__ import annotations

import streamlit as st

from ai.gemini_provider import (
    DEFAULT_GEMINI_MODELS,
    is_gemini_configured,
    test_gemini_connection,
)
from ai.groq_provider import (
    DEFAULT_GROQ_MODELS,
    is_groq_configured,
    test_groq_connection,
)
from ai.ollama_provider import (
    SUGGESTED_OLLAMA_MODELS,
    check_ollama_health,
    test_ollama_connection,
)
from config import get_config
from database.db import get_user_settings, upsert_user_settings


def render_settings_page() -> None:
    """Render model provider selection and connection testing UI."""
    cfg = get_config()
    user_id = int(st.session_state["user_id"])

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

    st.title("Settings & Model Selection")
    st.caption("Choose between Groq (fast cloud Llama/Qwen), Ollama (local open-source), or Google Gemini API.")

    provider_options = ["groq", "ollama", "gemini"]
    default_idx = provider_options.index(settings.preferred_provider) if settings.preferred_provider in provider_options else 0

    selected_provider = st.radio(
        "LLM Provider",
        options=provider_options,
        index=default_idx,
        format_func=lambda p: (
            "Groq Cloud (Llama 3.3 / Fast Inference)"
            if p == "groq"
            else ("Ollama (Open-Source Local Model)" if p == "ollama" else "Google Gemini API (Cloud)")
        ),
        horizontal=True,
    )

    st.divider()

    if selected_provider == "groq":
        st.subheader("Groq Configuration")
        if is_groq_configured(cfg.groq_api_key):
            st.success("`GROQ_API_KEY` detected from `.env` / environment.")
        else:
            st.info(
                "No `GROQ_API_KEY` found in `.env`. Add it in `.env` or enter a session-only API key below."
            )

        session_groq_key = st.text_input(
            "Session Groq API Key Override (Optional, kept in memory only)",
            value=st.session_state.get("session_groq_api_key", ""),
            type="password",
            placeholder="gsk_...",
        )
        st.session_state["session_groq_api_key"] = session_groq_key.strip()

        groq_models = DEFAULT_GROQ_MODELS
        if settings.preferred_provider == "groq" and settings.preferred_model in groq_models:
            gr_idx = groq_models.index(settings.preferred_model)
        else:
            gr_idx = 0
        chosen_groq = st.selectbox("Select Groq Model", options=groq_models, index=gr_idx)
        custom_groq = st.text_input(
            "Or enter custom Groq model ID (optional)",
            value="" if settings.preferred_model in groq_models else (
                settings.preferred_model if settings.preferred_provider == "groq" else ""
            ),
            placeholder="e.g., llama-3.3-70b-versatile",
        ).strip()
        final_model = custom_groq or chosen_groq

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Save Groq Settings", type="primary", use_container_width=True):
                upsert_user_settings(
                    cfg.db_path,
                    user_id=user_id,
                    preferred_provider="groq",
                    preferred_model=final_model,
                    ollama_base_url=settings.ollama_base_url,
                )
                st.success(f"Saved Groq model `{final_model}`.")
        with c2:
            if st.button("Test Groq Connection", use_container_width=True):
                with st.spinner("Testing Groq API..."):
                    ok, msg = test_groq_connection(
                        model_name=final_model,
                        api_key=session_groq_key.strip() or cfg.groq_api_key,
                    )
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    elif selected_provider == "ollama":
        st.subheader("Ollama Configuration")
        ollama_url = st.text_input("Ollama Server URL", value=settings.ollama_base_url or cfg.ollama_base_url)
        status = check_ollama_health(ollama_url)
        if status.available:
            st.success(status.message)
        else:
            st.warning(status.message)

        available_models = status.models if status.models else SUGGESTED_OLLAMA_MODELS
        if settings.preferred_provider == "ollama" and settings.preferred_model in available_models:
            model_index = available_models.index(settings.preferred_model)
        else:
            model_index = 0

        chosen_model = st.selectbox(
            "Select Local Ollama Model",
            options=available_models,
            index=model_index,
            help="Discovered from Ollama /api/tags when running, or choose a standard model name.",
        )
        custom_model = st.text_input(
            "Or enter custom Ollama model tag (optional)",
            value="",
            placeholder="e.g., llama3.2:latest",
        ).strip()
        final_model = custom_model or chosen_model

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Save Ollama Settings", type="primary", use_container_width=True):
                upsert_user_settings(
                    cfg.db_path,
                    user_id=user_id,
                    preferred_provider="ollama",
                    preferred_model=final_model,
                    ollama_base_url=ollama_url,
                )
                st.success(f"Saved Ollama model `{final_model}`.")
        with c2:
            if st.button("Test Ollama Connection", use_container_width=True):
                with st.spinner("Testing Ollama inference..."):
                    ok, msg = test_ollama_connection(model_name=final_model, base_url=ollama_url)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    else:
        st.subheader("Google Gemini Configuration")
        env_configured = is_gemini_configured(cfg.gemini_api_key)
        if env_configured:
            st.success("`GEMINI_API_KEY` detected from `.env` / environment.")
        else:
            st.info(
                "No `GEMINI_API_KEY` found in `.env`. You can configure it in `.env` or provide a "
                "session-only API key below (never written to disk)."
            )

        session_key = st.text_input(
            "Session Gemini API Key Override (Optional, kept in memory only)",
            value=st.session_state.get("session_gemini_api_key", ""),
            type="password",
            placeholder="AIza...",
        )
        st.session_state["session_gemini_api_key"] = session_key.strip()

        gemini_models = DEFAULT_GEMINI_MODELS
        if settings.preferred_provider == "gemini" and settings.preferred_model in gemini_models:
            g_idx = gemini_models.index(settings.preferred_model)
        else:
            g_idx = 0
        chosen_gemini = st.selectbox("Select Gemini Model", options=gemini_models, index=g_idx)
        custom_gemini = st.text_input(
            "Or enter custom Gemini model ID (optional)",
            value="" if settings.preferred_model in gemini_models else (
                settings.preferred_model if settings.preferred_provider == "gemini" else ""
            ),
            placeholder="e.g., gemini-3.8-flash",
        ).strip()
        final_model = custom_gemini or chosen_gemini

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Save Gemini Settings", type="primary", use_container_width=True):
                upsert_user_settings(
                    cfg.db_path,
                    user_id=user_id,
                    preferred_provider="gemini",
                    preferred_model=final_model,
                    ollama_base_url=settings.ollama_base_url,
                )
                st.success(f"Saved Gemini model `{final_model}`.")
        with c2:
            if st.button("Test Gemini Connection", use_container_width=True):
                with st.spinner("Testing Gemini API..."):
                    ok, msg = test_gemini_connection(
                        model_name=final_model,
                        api_key=session_key.strip() or cfg.gemini_api_key,
                    )
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    st.divider()
    with st.expander("Optional FastMCP Server Instructions"):
        st.markdown(
            """
            VYRAQ AI includes an optional standalone MCP server (`mcp_server/server.py`) exposing:
            `analyze_resume`, `analyze_job_description`, `calculate_compatibility`,
            `identify_missing_skills`, and `generate_interview_questions`.

            Start it separately from a terminal when connecting an MCP client:
            ```bash
            python -m mcp_server.server
            ```
            """
        )
