"""Resume analysis and transparent ATS scoring service."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from ai.llm_factory import invoke_structured_json

ATS_DISCLAIMER = (
    "Note: This ATS Resume Score reflects transparent VYRAQ AI rubric criteria "
    "(keywords, quantified impact, section completeness, and formatting signals) "
    "and does not guarantee results on any employer's proprietary ATS."
)

ACTION_VERBS = {
    "built", "designed", "implemented", "developed", "architected", "led", "managed",
    "optimized", "reduced", "increased", "improved", "deployed", "automated",
    "created", "launched", "delivered", "engineered", "integrated", "analyzed", "mentored",
}

EXPECTED_SECTIONS = {
    "summary": ("summary", "objective", "profile", "about"),
    "experience": ("experience", "employment", "work history", "career"),
    "skills": ("skills", "technologies", "competencies", "tech stack"),
    "education": ("education", "academic", "degree", "university", "bachelor", "master"),
    "projects": ("projects", "portfolio"),
}


class ATSBreakdown(BaseModel):
    """Criterion-by-criterion ATS score breakdown summing to 100."""

    keyword_relevance: float = Field(default=0.0, ge=0.0, le=35.0)
    quantified_impact: float = Field(default=0.0, ge=0.0, le=25.0)
    section_structure: float = Field(default=0.0, ge=0.0, le=20.0)
    clarity_formatting: float = Field(default=0.0, ge=0.0, le=20.0)


class ResumeAnalysisResult(BaseModel):
    """Structured resume analysis and ATS evaluation report."""

    candidate_summary: str = ""
    detected_skills: list[str] = Field(default_factory=list)
    technical_skills: list[str] = Field(default_factory=list)
    soft_skills: list[str] = Field(default_factory=list)
    experience_highlights: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    estimated_years_experience: str = "Not explicitly stated"
    ats_score: float = Field(default=0.0, ge=0.0, le=100.0)
    ats_breakdown: ATSBreakdown = Field(default_factory=ATSBreakdown)
    resume_quality_assessment: str = ""
    important_keywords_detected: list[str] = Field(default_factory=list)
    missing_keywords: list[str] = Field(default_factory=list)
    ats_suggestions: list[str] = Field(default_factory=list)
    observable_signals: dict[str, Any] = Field(default_factory=dict)
    ats_disclaimer: str = ATS_DISCLAIMER


def compute_ats_signals(resume_text: str) -> dict[str, Any]:
    """Compute observable, deterministic structural and formatting metrics from resume text."""
    lower = resume_text.lower()
    words = resume_text.split()
    word_count = len(words)

    detected_sections = [
        sec for sec, kws in EXPECTED_SECTIONS.items() if any(kw in lower for kw in kws)
    ]
    missing_sections = [sec for sec in EXPECTED_SECTIONS if sec not in detected_sections]

    quantified_matches = re.findall(
        r"\b\d[\d,]*(?:\.\d+)?\s*(?:%|\+|(?:x|k|m|b|ms|sec|hrs|years?|yrs?)\b)",
        lower,
    )
    verb_hits = sorted({v for v in ACTION_VERBS if re.search(rf"\b{v}\b", lower)})

    has_email = bool(re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", resume_text))
    has_phone_or_link = bool(
        re.search(r"(?:\+?\d[\d\s\-()]{7,}\d|linkedin\.com|github\.com)", lower)
    )

    bullet_lines = sum(1 for ln in resume_text.splitlines() if ln.strip().startswith(("-", "*", "•")))

    # Deterministic baseline sub-scores (used as transparent anchor for LLM calibration)
    section_score = round((len(detected_sections) / len(EXPECTED_SECTIONS)) * 20.0, 1)
    impact_score = round(min(25.0, len(quantified_matches) * 3.0 + min(len(verb_hits), 8) * 1.5), 1)
    length_ok = 1.0 if 250 <= word_count <= 1400 else 0.65
    clarity_score = round(
        min(
            20.0,
            (5.0 if has_email else 0.0)
            + (4.0 if has_phone_or_link else 0.0)
            + (5.0 if bullet_lines >= 4 else 2.5 if bullet_lines >= 1 else 0.0)
            + (6.0 * length_ok),
        ),
        1,
    )

    return {
        "word_count": word_count,
        "detected_sections": detected_sections,
        "missing_sections": missing_sections,
        "quantified_metrics_count": len(quantified_matches),
        "action_verbs_found": verb_hits,
        "has_email": has_email,
        "has_contact_or_profile_link": has_phone_or_link,
        "bullet_line_count": bullet_lines,
        "baseline_section_structure": section_score,
        "baseline_quantified_impact": impact_score,
        "baseline_clarity_formatting": clarity_score,
    }


def analyze_resume(
    resume_text: str,
    llm: BaseChatModel,
    rag_context: str = "",
    job_context: str = "",
) -> ResumeAnalysisResult:
    """Analyze a resume and compute a transparent ATS compatibility score using LangChain LLM."""
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text cannot be empty.")

    signals = compute_ats_signals(resume_text)

    system_prompt = (
        "You are a senior technical recruiter and ATS evaluation specialist. "
        "Analyze the candidate resume strictly from its actual content and observable metrics. "
        "Do NOT invent skills, companies, or metrics that do not appear in the resume. "
        "Return ONLY a single valid JSON object matching the requested schema."
    )

    job_hint = f"\nTARGET JOB CONTEXT (for keyword relevance calibration):\n{job_context[:2500]}\n" if job_context else ""
    rag_section = f"\nRETRIEVED RAG EVIDENCE:\n{rag_context}\n" if rag_context else ""

    user_prompt = f"""Analyze the following resume and produce a transparent ATS evaluation.

OBSERVABLE RESUME SIGNALS (measured directly from document):
- Word count: {signals['word_count']}
- Detected sections: {signals['detected_sections']}
- Missing standard sections: {signals['missing_sections']}
- Quantified metric expressions count: {signals['quantified_metrics_count']}
- Action verbs detected: {signals['action_verbs_found']}
- Contact email present: {signals['has_email']}
- Phone/LinkedIn/GitHub present: {signals['has_contact_or_profile_link']}
- Bullet lines count: {signals['bullet_line_count']}
- Baseline structural scores: section_structure={signals['baseline_section_structure']}/20, quantified_impact={signals['baseline_quantified_impact']}/25, clarity_formatting={signals['baseline_clarity_formatting']}/20
{job_hint}{rag_section}
FULL RESUME TEXT:
{resume_text[:8000]}

Return JSON with EXACT keys:
{{
  "candidate_summary": "concise 2-3 sentence grounded summary",
  "detected_skills": ["all detected hard and soft skills"],
  "technical_skills": ["programming languages, frameworks, tools, cloud, databases"],
  "soft_skills": ["communication, leadership, collaboration, etc."],
  "experience_highlights": ["key verified roles/projects/achievements"],
  "education": ["degrees, institutions, graduation info found"],
  "certifications": ["certifications found, empty list if none"],
  "estimated_years_experience": "e.g. 3+ years or Entry-level / Student",
  "ats_breakdown": {{
    "keyword_relevance": 0.0 to 35.0,
    "quantified_impact": 0.0 to 25.0,
    "section_structure": 0.0 to 20.0,
    "clarity_formatting": 0.0 to 20.0
  }},
  "ats_score": "sum of the 4 breakdown scores (0 to 100)",
  "resume_quality_assessment": "clear evidence-based assessment explaining strengths and weaknesses",
  "important_keywords_detected": ["key ATS keywords present in resume"],
  "missing_keywords": ["important ATS keywords missing relative to target role or industry norms"],
  "ats_suggestions": ["specific, actionable suggestions to improve ATS readability and impact"]
}}"""

    result = invoke_structured_json(llm, system_prompt, user_prompt, ResumeAnalysisResult)
    # Enforce exact mathematical consistency between ats_score and breakdown sum
    breakdown_sum = round(
        min(35.0, max(0.0, result.ats_breakdown.keyword_relevance))
        + min(25.0, max(0.0, result.ats_breakdown.quantified_impact))
        + min(20.0, max(0.0, result.ats_breakdown.section_structure))
        + min(20.0, max(0.0, result.ats_breakdown.clarity_formatting)),
        1,
    )
    result.ats_score = round(min(100.0, max(0.0, breakdown_sum)), 1)
    result.observable_signals = signals
    result.ats_disclaimer = ATS_DISCLAIMER
    if not result.detected_skills:
        result.detected_skills = list(dict.fromkeys(result.technical_skills + result.soft_skills))
    return result
