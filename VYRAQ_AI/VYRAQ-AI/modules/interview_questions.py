"""Grounded interview question and answer preparation service."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field, field_validator, model_validator

from ai.llm_factory import invoke_structured_json
from modules.job_analyzer import JobAnalysisResult
from modules.resume_analyzer import ResumeAnalysisResult
from modules.skill_gap import SkillGapResult


class InterviewQuestionItem(BaseModel):
    """Single interview question with rationale and sample answer points."""

    category: str = "technical"  # technical | role_specific | resume_based | hr
    question: str
    difficulty: str = "Medium"
    target_skill_or_topic: str = ""
    rationale: str = ""
    key_answer_points: list[str] = Field(default_factory=list)
    sample_answer: str = ""

    @field_validator("key_answer_points", mode="before")
    @classmethod
    def _coerce_points(cls, val: Any) -> list[str]:
        if val is None:
            return []
        if isinstance(val, str):
            return [line.strip("-• ").strip() for line in val.splitlines() if line.strip()]
        if isinstance(val, list):
            return [str(x).strip() for x in val if x is not None and str(x).strip()]
        return [str(val)]


def _normalize_items(raw_items: Any, default_category: str) -> list[dict[str, Any]]:
    """Coerce list items (dicts or strings) into InterviewQuestionItem-compatible dicts."""
    if not isinstance(raw_items, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in raw_items:
        if isinstance(item, str) and item.strip():
            normalized.append({"category": default_category, "question": item.strip()})
        elif isinstance(item, dict):
            copy_item = dict(item)
            copy_item.setdefault("category", default_category)
            if "question" not in copy_item:
                for alt_q in ("prompt", "title", "text"):
                    if alt_q in copy_item:
                        copy_item["question"] = str(copy_item[alt_q])
                        break
            if "sample_answer" not in copy_item:
                for alt_a in ("answer", "model_answer", "example_answer"):
                    if alt_a in copy_item:
                        copy_item["sample_answer"] = str(copy_item[alt_a])
                        break
            if "key_answer_points" not in copy_item:
                for alt_k in ("answer_points", "key_points", "talking_points", "bullets"):
                    if alt_k in copy_item:
                        copy_item["key_answer_points"] = copy_item[alt_k]
                        break
            if copy_item.get("question"):
                normalized.append(copy_item)
    return normalized


class InterviewPrepResult(BaseModel):
    """Grouped interview preparation questions grounded in resume, JD, and skill gaps."""

    technical_questions: list[InterviewQuestionItem] = Field(default_factory=list)
    role_specific_questions: list[InterviewQuestionItem] = Field(default_factory=list)
    resume_based_questions: list[InterviewQuestionItem] = Field(default_factory=list)
    hr_questions: list[InterviewQuestionItem] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        # Unwrap common outer container keys if present
        for wrapper in ("interview_questions", "interview_prep", "data", "result"):
            if wrapper in data and isinstance(data[wrapper], dict):
                data = data[wrapper]
                break

        alias_map = {
            "technical_questions": ("technical_questions", "technical", "tech_questions"),
            "role_specific_questions": (
                "role_specific_questions",
                "role_specific",
                "role_questions",
                "domain_questions",
            ),
            "resume_based_questions": (
                "resume_based_questions",
                "resume_based",
                "resume_questions",
                "experience_questions",
            ),
            "hr_questions": (
                "hr_questions",
                "hr",
                "behavioral_questions",
                "hr_behavioral_questions",
                "hr_and_behavioral_questions",
            ),
        }

        out: dict[str, Any] = {}
        for target_key, aliases in alias_map.items():
            default_cat = target_key.replace("_questions", "")
            for alias in aliases:
                if alias in data and data[alias]:
                    out[target_key] = _normalize_items(data[alias], default_cat)
                    break
            out.setdefault(target_key, [])

        # Handle flat {"questions": [...]} outputs by bucketing by item["category"]
        if not any(out.values()) and isinstance(data.get("questions"), list):
            for raw_item in data["questions"]:
                if not isinstance(raw_item, dict):
                    continue
                cat = str(raw_item.get("category", "technical")).strip().lower().replace("-", "_").replace(" ", "_")
                if "role" in cat:
                    bucket = "role_specific_questions"
                elif "resume" in cat or "experience" in cat:
                    bucket = "resume_based_questions"
                elif "hr" in cat or "behav" in cat:
                    bucket = "hr_questions"
                else:
                    bucket = "technical_questions"
                out[bucket].extend(_normalize_items([raw_item], bucket.replace("_questions", "")))

        return out

    def total_count(self) -> int:
        return (
            len(self.technical_questions)
            + len(self.role_specific_questions)
            + len(self.resume_based_questions)
            + len(self.hr_questions)
        )


def generate_interview_questions(
    resume_analysis: ResumeAnalysisResult | dict,
    job_analysis: JobAnalysisResult | dict,
    skill_gap: SkillGapResult | dict,
    llm: BaseChatModel,
    rag_context: str = "",
) -> InterviewPrepResult:
    """Generate categorized interview questions grounded in resume, JD, existing and missing skills."""
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
    g_obj = (
        skill_gap
        if isinstance(skill_gap, SkillGapResult)
        else SkillGapResult.model_validate(skill_gap)
    )

    system_prompt = (
        "You are a senior hiring manager and interview coach. Generate realistic, insightful "
        "interview questions grounded specifically in the candidate's resume, the target job "
        "description, existing skills, and missing skills. Include clear key answer points and a "
        "strong sample answer for every question. Return ONLY a single valid JSON object."
    )

    rag_section = f"\nRETRIEVED RAG EVIDENCE:\n{rag_context}\n" if rag_context else ""

    item_example = """{{
      "category": "{cat}",
      "question": "specific interview question",
      "difficulty": "Easy | Medium | Hard",
      "target_skill_or_topic": "specific existing or missing skill",
      "rationale": "why this question is asked based on JD/resume/gaps",
      "key_answer_points": ["point 1", "point 2", "point 3"],
      "sample_answer": "concise strong sample answer"
    }}"""

    user_prompt = f"""Generate tailored interview questions across ALL 4 categories below.
CRITICAL REQUIREMENT: Every one of the 4 JSON arrays (`technical_questions`, `role_specific_questions`, `resume_based_questions`, `hr_questions`) MUST contain 2 to 3 question objects and MUST NOT be empty.

CANDIDATE PROFILE:
- Summary: {r_obj.candidate_summary}
- Existing Skills: {r_obj.detected_skills}
- Experience Highlights: {r_obj.experience_highlights}

TARGET JOB:
- Role: {j_obj.job_title} ({j_obj.seniority_level})
- Responsibilities: {j_obj.key_responsibilities}
- Required Skills: {j_obj.required_skills}

SKILL GAPS:
- Available Skills: {g_obj.available_skills}
- Missing Skills: {g_obj.missing_skills}
{rag_section}
Return a single JSON object with EXACTLY these 4 top-level keys:
{{
  "technical_questions": [
    {item_example.format(cat="technical")}
  ],
  "role_specific_questions": [
    {item_example.format(cat="role_specific")}
  ],
  "resume_based_questions": [
    {item_example.format(cat="resume_based")}
  ],
  "hr_questions": [
    {item_example.format(cat="hr")}
  ]
}}"""

    result = invoke_structured_json(llm, system_prompt, user_prompt, InterviewPrepResult)
    if result.total_count() == 0:
        retry_prompt = (
            user_prompt
            + "\n\nYour previous output returned empty arrays. Populate at least 2 questions inside each of the 4 arrays."
        )
        result = invoke_structured_json(llm, system_prompt, retry_prompt, InterviewPrepResult)
    return result
