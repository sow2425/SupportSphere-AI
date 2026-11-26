"""Job description analysis service."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from ai.llm_factory import invoke_structured_json


class JobAnalysisResult(BaseModel):
    """Structured extraction of job requirements and skills."""

    job_title: str = "Target Role"
    seniority_level: str = "Not specified"
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    key_responsibilities: list[str] = Field(default_factory=list)
    experience_requirements: str = "Not specified"
    education_requirements: str = "Not specified"
    domain_keywords: list[str] = Field(default_factory=list)
    role_summary: str = ""


def analyze_job_description(
    job_text: str,
    llm: BaseChatModel,
    rag_context: str = "",
) -> JobAnalysisResult:
    """Extract structured role requirements, skills, and keywords from a job description."""
    if not job_text or not job_text.strip():
        raise ValueError("Job description text cannot be empty.")

    system_prompt = (
        "You are an expert talent acquisition analyst. Extract structured job requirements "
        "strictly from the provided job description. Distinguish mandatory required skills from "
        "nice-to-have preferred skills. Return ONLY valid JSON."
    )

    rag_section = f"\nRETRIEVED RAG EVIDENCE:\n{rag_context}\n" if rag_context else ""

    user_prompt = f"""Analyze the following job description and extract structured requirements.
{rag_section}
JOB DESCRIPTION TEXT:
{job_text[:8000]}

Return JSON with EXACT keys:
{{
  "job_title": "extracted role title",
  "seniority_level": "Intern / Junior / Mid / Senior / Lead / Not specified",
  "required_skills": ["mandatory technical and domain skills"],
  "preferred_skills": ["optional or nice-to-have skills"],
  "key_responsibilities": ["core day-to-day duties"],
  "experience_requirements": "years and type of experience requested",
  "education_requirements": "degree or certification requirements",
  "domain_keywords": ["important domain, methodology, and ATS keywords"],
  "role_summary": "concise 2-sentence summary of what the employer is looking for"
}}"""

    return invoke_structured_json(llm, system_prompt, user_prompt, JobAnalysisResult)
