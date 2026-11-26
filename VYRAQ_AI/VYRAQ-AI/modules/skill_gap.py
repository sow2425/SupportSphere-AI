"""Skill gap identification and learning recommendation service."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from ai.llm_factory import invoke_structured_json
from modules.compatibility import CompatibilityResult
from modules.job_analyzer import JobAnalysisResult
from modules.resume_analyzer import ResumeAnalysisResult


class RecommendedSkill(BaseModel):
    """Actionable skill recommendation to close a resume-to-job gap."""

    skill: str
    priority: str = "Medium"  # High, Medium, Low
    reason: str = ""
    learning_focus: str = ""


class SkillGapResult(BaseModel):
    """Comprehensive skill gap breakdown."""

    required_skills: list[str] = Field(default_factory=list)
    available_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    recommended_skills_to_learn: list[RecommendedSkill] = Field(default_factory=list)
    gap_summary: str = ""


def identify_missing_skills(
    resume_analysis: ResumeAnalysisResult | dict,
    job_analysis: JobAnalysisResult | dict,
    compatibility: CompatibilityResult | dict | None,
    llm: BaseChatModel,
    rag_context: str = "",
) -> SkillGapResult:
    """Identify missing skills and build a prioritized upskilling roadmap."""
    r_obj = (
        resume_analysis
        if isinstance(resume_analysis, ResumeAnalysisResult)
        else ResumeAnalysisResult.model_validate(resume_analysis)
    )
    j_obj = (
        job_analysis
        if isinstance(job_analysis, JobAnalysisResult)
        else JobAnalysisResult.model_validate(job_analysis)
    )
    c_obj = (
        compatibility
        if isinstance(compatibility, CompatibilityResult)
        else CompatibilityResult.model_validate(compatibility)
        if compatibility is not None
        else None
    )

    system_prompt = (
        "You are a career coach and technical skills strategist. Based strictly on the job "
        "requirements and candidate resume, map required skills, available skills, missing skills, "
        "and practical recommendations to close the gap. Return ONLY valid JSON."
    )

    comp_block = (
        f"- Matching skills: {c_obj.matching_skills}\n"
        f"- Partial skills: {c_obj.partially_matching_skills}\n"
        f"- Missing skills: {c_obj.missing_skills}\n"
        if c_obj
        else ""
    )
    rag_section = f"\nRETRIEVED RAG EVIDENCE:\n{rag_context}\n" if rag_context else ""

    user_prompt = f"""Produce a structured Skill Gap Analysis.

JOB REQUIRED SKILLS: {j_obj.required_skills}
JOB PREFERRED SKILLS: {j_obj.preferred_skills}
CANDIDATE DETECTED SKILLS: {r_obj.detected_skills}
{comp_block}{rag_section}
Return JSON with EXACT keys:
{{
  "required_skills": ["all skills required/expected by the job"],
  "available_skills": ["job-relevant skills currently present in the resume"],
  "missing_skills": ["required or key skills missing from the resume"],
  "recommended_skills_to_learn": [
    {{
      "skill": "skill name",
      "priority": "High | Medium | Low",
      "reason": "why this matters for the target role",
      "learning_focus": "concrete project, certification, or practice focus"
    }}
  ],
  "gap_summary": "concise 2-sentence overview of skill readiness and top priorities"
}}"""

    return invoke_structured_json(llm, system_prompt, user_prompt, SkillGapResult)
