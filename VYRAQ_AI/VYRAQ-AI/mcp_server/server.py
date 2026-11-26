"""Optional FastMCP / MCPServer exposing VYRAQ AI analysis tools.

Reuses the core service functions in `modules/` and `ai/llm_factory.py`.
Compatible with both `mcp < 2.0` (`FastMCP`), standalone `fastmcp`, and
`mcp >= 2.0` (`MCPServer`, the v2 rename of FastMCP).
"""

from __future__ import annotations

import json
from typing import Any

from ai.llm_factory import LLMFactoryError, get_llm
from modules.compatibility import calculate_compatibility as svc_calculate_compatibility
from modules.interview_questions import generate_interview_questions as svc_generate_interview_questions
from modules.job_analyzer import analyze_job_description as svc_analyze_job_description
from modules.resume_analyzer import analyze_resume as svc_analyze_resume
from modules.skill_gap import identify_missing_skills as svc_identify_missing_skills

try:
    from mcp.server.fastmcp import FastMCP  # type: ignore
except ImportError:
    try:
        from fastmcp import FastMCP  # type: ignore
    except ImportError:
        from mcp.server.mcpserver import MCPServer as FastMCP  # type: ignore


mcp = FastMCP("VYRAQ AI Career Analyzer")


def _resolve_model(provider: str | None = None, model_name: str | None = None):
    return get_llm(provider=provider, model_name=model_name)


@mcp.tool()
def analyze_resume(
    resume_text: str,
    job_context: str = "",
    provider: str = "ollama",
    model_name: str = "",
) -> str:
    """Analyze resume text and compute a transparent ATS score and keyword report."""
    try:
        llm = _resolve_model(provider=provider, model_name=model_name or None)
        result = svc_analyze_resume(resume_text=resume_text, llm=llm, job_context=job_context)
        return json.dumps(result.model_dump(), indent=2)
    except (LLMFactoryError, ValueError) as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
def analyze_job_description(
    job_text: str,
    provider: str = "ollama",
    model_name: str = "",
) -> str:
    """Extract required skills, preferred skills, and responsibilities from a job description."""
    try:
        llm = _resolve_model(provider=provider, model_name=model_name or None)
        result = svc_analyze_job_description(job_text=job_text, llm=llm)
        return json.dumps(result.model_dump(), indent=2)
    except (LLMFactoryError, ValueError) as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
def calculate_compatibility(
    resume_text: str,
    job_text: str,
    provider: str = "ollama",
    model_name: str = "",
) -> str:
    """Calculate transparent compatibility percentage and skill alignment between resume and JD."""
    try:
        llm = _resolve_model(provider=provider, model_name=model_name or None)
        resume_res = svc_analyze_resume(resume_text=resume_text, llm=llm, job_context=job_text)
        job_res = svc_analyze_job_description(job_text=job_text, llm=llm)
        comp_res = svc_calculate_compatibility(resume_analysis=resume_res, job_analysis=job_res, llm=llm)
        return json.dumps(comp_res.model_dump(), indent=2)
    except (LLMFactoryError, ValueError) as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
def identify_missing_skills(
    resume_text: str,
    job_text: str,
    provider: str = "ollama",
    model_name: str = "",
) -> str:
    """Identify required, available, and missing skills plus recommended learning steps."""
    try:
        llm = _resolve_model(provider=provider, model_name=model_name or None)
        resume_res = svc_analyze_resume(resume_text=resume_text, llm=llm, job_context=job_text)
        job_res = svc_analyze_job_description(job_text=job_text, llm=llm)
        comp_res = svc_calculate_compatibility(resume_analysis=resume_res, job_analysis=job_res, llm=llm)
        gap_res = svc_identify_missing_skills(
            resume_analysis=resume_res,
            job_analysis=job_res,
            compatibility=comp_res,
            llm=llm,
        )
        return json.dumps(gap_res.model_dump(), indent=2)
    except (LLMFactoryError, ValueError) as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
def generate_interview_questions(
    resume_text: str,
    job_text: str,
    provider: str = "ollama",
    model_name: str = "",
) -> str:
    """Generate categorized interview questions and sample answers grounded in resume and JD."""
    try:
        llm = _resolve_model(provider=provider, model_name=model_name or None)
        resume_res = svc_analyze_resume(resume_text=resume_text, llm=llm, job_context=job_text)
        job_res = svc_analyze_job_description(job_text=job_text, llm=llm)
        comp_res = svc_calculate_compatibility(resume_analysis=resume_res, job_analysis=job_res, llm=llm)
        gap_res = svc_identify_missing_skills(
            resume_analysis=resume_res,
            job_analysis=job_res,
            compatibility=comp_res,
            llm=llm,
        )
        iv_res = svc_generate_interview_questions(
            resume_analysis=resume_res,
            job_analysis=job_res,
            skill_gap=gap_res,
            llm=llm,
        )
        return json.dumps(iv_res.model_dump(), indent=2)
    except (LLMFactoryError, ValueError) as exc:
        return json.dumps({"error": str(exc)})


def main(transport: str = "stdio") -> None:
    """Run the VYRAQ AI FastMCP server."""
    mcp.run(transport=transport)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
