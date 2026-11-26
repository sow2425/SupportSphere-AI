"""Streamlit UI page modules."""

from pages.auth_pages import render_auth_pages
from pages.dashboard import render_dashboard
from pages.history import render_history_page
from pages.results import render_results_page
from pages.settings import render_settings_page
from pages.upload import render_upload_page

__all__ = [
    "render_auth_pages",
    "render_dashboard",
    "render_history_page",
    "render_results_page",
    "render_settings_page",
    "render_upload_page",
]
