"""Optional FastMCP server package."""

from mcp_server.server import (
    analyze_job_description,
    analyze_resume,
    calculate_compatibility,
    generate_interview_questions,
    identify_missing_skills,
    mcp,
)

__all__ = [
    "analyze_job_description",
    "analyze_resume",
    "calculate_compatibility",
    "generate_interview_questions",
    "identify_missing_skills",
    "mcp",
]
