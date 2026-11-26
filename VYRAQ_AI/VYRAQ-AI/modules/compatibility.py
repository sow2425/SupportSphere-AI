"""Job compatibility scoring and skill comparison service."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from ai.llm_factory import invoke_structured_json
from modules.job_analyzer import JobAnalysisResult
from modules.resume_analyzer import ResumeAnalysisResult


class CompatibilityBreakdown(BaseModel):
    """Transparent weighted criteria contributing to overall compatibility percentage."""

    required_skills_score: float = Field(default=0.0, ge=0.0, le=50.0)
    preferred_skills_score: float = Field(default=0.0, ge=0.0, le=15.0)
    experience_alignment_score: float = Field(default=0.0, ge=0.0, le=25.0)
    domain_keyword_score: float = Field(default=0.0, ge=0.0, le=10.0)


class CompatibilityResult(BaseModel):
    """Structured resume-to-job compatibility report."""

    compatibility_score: float = Field(default=0.0, ge=0.0, le=100.0)
    breakdown: CompatibilityBreakdown = Field(default_factory=CompatibilityBreakdown)
    matching_skills: list[str] = Field(default_factory=list)
    partially_matching_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    experience_comparison: str = ""
    match_explanation: str = ""
    strengths: list[str] = Field(default_factory=list)
    risks_or_gaps: list[str] = Field(default_factory=list)


def calculate_compatibility(
    resume_analysis: ResumeAnalysisResult | dict,
    job_analysis: JobAnalysisResult | dict,
    llm: BaseChatModel,
    rag_context: str = "",
) -> CompatibilityResult:
    """Compare analyzed resume and job requirements to produce a transparent compatibility score."""
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

    system_prompt = (
        "You are an objective technical hiring evaluator. Compare the candidate's verified resume "
        "skills and experience against the job requirements using transparent criteria. "
        "Do not exaggerate or fabricate matches. Return ONLY valid JSON."
    )

    rag_section = f"\nRETRIEVED RAG CROSS-DOCUMENT EVIDENCE:\n{rag_context}\n" if rag_context else ""

    user_prompt = f"""Evaluate compatibility between the candidate resume and target job description.

RESUME PROFILE:
- Summary: {r_obj.candidate_summary}
- Technical Skills: {r_obj.technical_skills}
- Soft Skills: {r_obj.soft_skills}
- Experience: {r_obj.experience_highlights}
- Estimated Experience: {r_obj.estimated_years_experience}
- Education: {r_obj.education}

JOB PROFILE:
- Title: {j_obj.job_title} ({j_obj.seniority_level})
- Required Skills: {j_obj.required_skills}
- Preferred Skills: {j_obj.preferred_skills}
- Responsibilities: {j_obj.key_responsibilities}
- Experience Requirements: {j_obj.experience_requirements}
- Domain Keywords: {j_obj.domain_keywords}
{rag_section}
Scoring Rubric (sum to 100 max):
1. required_skills_score (0 to 50): proportion & depth of mandatory skills evidenced
2. preferred_skills_score (0 to 15): proportion of nice-to-have skills evidenced
3. experience_alignment_score (0 to 25): alignment of past roles/projects/years with responsibilities
4. domain_keyword_score (0 to 10): domain & methodology terminology alignment

Return JSON with EXACT keys:
{{
  "breakdown": {{
    "required_skills_score": 0.0,
    "preferred_skills_score": 0.0,
    "experience_alignment_score": 0.0,
    "domain_keyword_score": 0.0
  }},
  "compatibility_score": 0.0,
  "matching_skills": ["skills clearly demonstrated in resume"],
  "partially_matching_skills": ["related or adjacent skills needing deeper proof"],
  "missing_skills": ["required or important job skills absent from resume"],
  "experience_comparison": "detailed comparison of candidate experience vs job expectations",
  "match_explanation": "clear explanation of why the candidate matches or does not match",
  "strengths": ["top alignment highlights"],
  "risks_or_gaps": ["key gaps a hiring manager would notice"]
}}"""

    result = invoke_structured_json(llm, system_prompt, user_prompt, CompatibilityResult)
    total = round(
        min(50.0, max(0.0, result.breakdown.required_skills_score))
        + min(15.0, max(0.0, result.breakdown.preferred_skills_score))
        + min(25.0, max(0.0, result.breakdown.experience_alignment_score))
        + min(10.0, max(0.0, result.breakdown.domain_keyword_score)),
        1,
    )
    result.compatibility_score = round(min(100.0, max(0.0, total)), 1)
    return result
