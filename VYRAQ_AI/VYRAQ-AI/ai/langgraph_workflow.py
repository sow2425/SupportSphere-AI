"""LangGraph 10-step orchestration workflow for VYRAQ AI."""

from __future__ import annotations

import uuid
from typing import Any, Callable, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from ai.llm_factory import get_llm
from modules.compatibility import CompatibilityResult, calculate_compatibility
from modules.document_parser import DocumentParseError, extract_document_text, normalize_extracted_text
from modules.interview_questions import InterviewPrepResult, generate_interview_questions
from modules.job_analyzer import JobAnalysisResult, analyze_job_description
from modules.resume_analyzer import ATS_DISCLAIMER, ResumeAnalysisResult, analyze_resume, compute_ats_signals
from modules.skill_gap import SkillGapResult, identify_missing_skills
from rag.rag_pipeline import VyraqRAGPipeline


class AnalysisState(TypedDict, total=False):
    """Shared state passed across the 10 LangGraph workflow nodes."""

    # Inputs
    user_id: int
    resume_filename: str
    resume_bytes: bytes | None
    resume_text: str
    job_filename: str
    job_bytes: bytes | None
    job_text: str
    job_title_hint: str

    # LLM & storage configuration
    provider: str
    model_name: str
    api_key: str
    ollama_base_url: str
    db_path: str
    vector_store_dir: str

    # Runtime services (non-serialized helpers injected via closure or state)
    collection_name: str
    rag_stats: dict[str, Any]
    rag_resume_context: str
    rag_job_context: str
    rag_comparison_context: str
    rag_interview_context: str

    # Intermediate domain outputs
    resume_analysis: dict[str, Any]
    job_analysis: dict[str, Any]
    skill_comparison_summary: dict[str, Any]
    ats_evaluation: dict[str, Any]
    compatibility: dict[str, Any]
    skill_gap: dict[str, Any]
    interview_questions: dict[str, Any]

    # Final report
    final_report: dict[str, Any]
    executive_summary: str
    steps_completed: list[str]


def build_analysis_workflow(
    llm: BaseChatModel | None = None,
    rag_pipeline: VyraqRAGPipeline | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
):
    """Compile the 10-step LangGraph StateGraph for resume-to-job compatibility analysis."""
    total_steps = 10

    def _notify(step_num: int, label: str, state: AnalysisState) -> list[str]:
        completed = list(state.get("steps_completed", []))
        completed.append(label)
        if progress_callback is not None:
            progress_callback(step_num, total_steps, label)
        return completed

    def _resolve_llm(state: AnalysisState) -> BaseChatModel:
        if llm is not None:
            return llm
        return get_llm(
            provider=state.get("provider"),
            model_name=state.get("model_name"),
            api_key=state.get("api_key") or None,
            ollama_base_url=state.get("ollama_base_url") or None,
        )

    def _resolve_rag(state: AnalysisState) -> VyraqRAGPipeline:
        if rag_pipeline is not None:
            return rag_pipeline
        collection_name = state.get("collection_name") or f"analysis_{uuid.uuid4().hex[:12]}"
        return VyraqRAGPipeline(
            user_id=int(state.get("user_id", 1)),
            collection_name=collection_name,
            db_path=state.get("db_path") or None,
            vector_store_dir=state.get("vector_store_dir") or None,
            provider=state.get("provider", "ollama"),
            api_key=state.get("api_key") or None,
            ollama_base_url=state.get("ollama_base_url") or None,
        )

    # 1. Resume text extraction
    def extract_resume_node(state: AnalysisState) -> AnalysisState:
        raw_text = state.get("resume_text", "")
        if not raw_text.strip() and state.get("resume_bytes"):
            raw_text, _ = extract_document_text(
                state.get("resume_filename", "resume.pdf"),
                state["resume_bytes"] or b"",
            )
        cleaned = normalize_extracted_text(raw_text)
        if not cleaned:
            raise DocumentParseError("Resume text extraction yielded empty content.")
        return {
            "resume_text": cleaned,
            "steps_completed": _notify(1, "Resume text extraction", state),
        }

    # 2. Job description extraction
    def extract_job_node(state: AnalysisState) -> AnalysisState:
        raw_text = state.get("job_text", "")
        if not raw_text.strip() and state.get("job_bytes"):
            raw_text, _ = extract_document_text(
                state.get("job_filename", "job_description.txt"),
                state["job_bytes"] or b"",
            )
        cleaned = normalize_extracted_text(raw_text)
        if not cleaned:
            raise DocumentParseError("Job description extraction yielded empty content.")
        return {
            "job_text": cleaned,
            "steps_completed": _notify(2, "Job description extraction", state),
        }

    # 3. Resume analysis (with RAG indexing & retrieval)
    def analyze_resume_node(state: AnalysisState) -> AnalysisState:
        active_llm = _resolve_llm(state)
        rag = _resolve_rag(state)
        index_counts = rag.index_documents(
            resume_text=state["resume_text"],
            job_text=state["job_text"],
            resume_name=state.get("resume_filename", "resume"),
            job_name=state.get("job_filename", "job_description"),
        )
        rag_resume_ctx = rag.build_grounded_context(
            "candidate skills experience achievements education certifications projects",
            top_k_resume=5,
            top_k_job=2,
        )
        res_analysis = analyze_resume(
            resume_text=state["resume_text"],
            llm=active_llm,
            rag_context=rag_resume_ctx,
            job_context=state["job_text"],
        )
        return {
            "collection_name": rag.collection_name,
            "rag_stats": {
                **index_counts,
                "vector_backend": rag.vector_backend_name,
                "embedding_backend": rag.embedding_backend_name,
            },
            "rag_resume_context": rag_resume_ctx,
            "resume_analysis": res_analysis.model_dump(),
            "steps_completed": _notify(3, "Resume analysis", state),
        }

    # 4. Job requirement analysis
    def analyze_job_node(state: AnalysisState) -> AnalysisState:
        active_llm = _resolve_llm(state)
        rag = _resolve_rag(state)
        rag_job_ctx = rag.build_grounded_context(
            "required skills preferred skills responsibilities experience qualifications",
            top_k_resume=1,
            top_k_job=5,
        )
        job_analysis = analyze_job_description(
            job_text=state["job_text"],
            llm=active_llm,
            rag_context=rag_job_ctx,
        )
        if state.get("job_title_hint") and job_analysis.job_title in ("", "Target Role"):
            job_analysis.job_title = state["job_title_hint"].strip()
        return {
            "rag_job_context": rag_job_ctx,
            "job_analysis": job_analysis.model_dump(),
            "steps_completed": _notify(4, "Job requirement analysis", state),
        }

    # 5. Skill comparison (RAG cross-document retrieval + lexical comparison)
    def compare_skills_node(state: AnalysisState) -> AnalysisState:
        rag = _resolve_rag(state)
        r_obj = ResumeAnalysisResult.model_validate(state["resume_analysis"])
        j_obj = JobAnalysisResult.model_validate(state["job_analysis"])

        skill_query = " ".join(j_obj.required_skills + j_obj.preferred_skills + j_obj.domain_keywords)
        rag_comp_ctx = rag.build_grounded_context(
            skill_query or "technical skills responsibilities experience",
            top_k_resume=5,
            top_k_job=5,
        )

        resume_skills_lower = {s.lower(): s for s in r_obj.detected_skills}
        exact_overlap = [s for s in j_obj.required_skills if s.lower() in resume_skills_lower]
        missing_exact = [s for s in j_obj.required_skills if s.lower() not in resume_skills_lower]

        return {
            "rag_comparison_context": rag_comp_ctx,
            "skill_comparison_summary": {
                "resume_skill_count": len(r_obj.detected_skills),
                "job_required_count": len(j_obj.required_skills),
                "job_preferred_count": len(j_obj.preferred_skills),
                "exact_required_overlap": exact_overlap,
                "unmatched_required_candidates": missing_exact,
            },
            "steps_completed": _notify(5, "Skill comparison", state),
        }

    # 6. ATS scoring verification & breakdown
    def ats_scoring_node(state: AnalysisState) -> AnalysisState:
        r_obj = ResumeAnalysisResult.model_validate(state["resume_analysis"])
        signals = r_obj.observable_signals or compute_ats_signals(state["resume_text"])
        ats_eval = {
            "ats_score": r_obj.ats_score,
            "ats_breakdown": r_obj.ats_breakdown.model_dump(),
            "resume_quality_assessment": r_obj.resume_quality_assessment,
            "important_keywords_detected": r_obj.important_keywords_detected,
            "missing_keywords": r_obj.missing_keywords,
            "ats_suggestions": r_obj.ats_suggestions,
            "observable_signals": signals,
            "ats_disclaimer": r_obj.ats_disclaimer or ATS_DISCLAIMER,
        }
        return {
            "ats_evaluation": ats_eval,
            "steps_completed": _notify(6, "ATS scoring", state),
        }

    # 7. Compatibility scoring
    def compatibility_scoring_node(state: AnalysisState) -> AnalysisState:
        active_llm = _resolve_llm(state)
        comp = calculate_compatibility(
            resume_analysis=state["resume_analysis"],
            job_analysis=state["job_analysis"],
            llm=active_llm,
            rag_context=state.get("rag_comparison_context", ""),
        )
        return {
            "compatibility": comp.model_dump(),
            "steps_completed": _notify(7, "Compatibility scoring", state),
        }

    # 8. Missing skill identification
    def missing_skills_node(state: AnalysisState) -> AnalysisState:
        active_llm = _resolve_llm(state)
        gap = identify_missing_skills(
            resume_analysis=state["resume_analysis"],
            job_analysis=state["job_analysis"],
            compatibility=state["compatibility"],
            llm=active_llm,
            rag_context=state.get("rag_comparison_context", ""),
        )
        return {
            "skill_gap": gap.model_dump(),
            "steps_completed": _notify(8, "Missing skill identification", state),
        }

    # 9. Interview question generation
    def interview_questions_node(state: AnalysisState) -> AnalysisState:
        active_llm = _resolve_llm(state)
        rag = _resolve_rag(state)
        gap_obj = SkillGapResult.model_validate(state["skill_gap"])
        interview_query = " ".join(gap_obj.available_skills + gap_obj.missing_skills)
        rag_iv_ctx = rag.build_grounded_context(
            interview_query or "projects architecture experience problem solving",
            top_k_resume=4,
            top_k_job=4,
        )
        iv = generate_interview_questions(
            resume_analysis=state["resume_analysis"],
            job_analysis=state["job_analysis"],
            skill_gap=state["skill_gap"],
            llm=active_llm,
            rag_context=rag_iv_ctx,
        )
        return {
            "rag_interview_context": rag_iv_ctx,
            "interview_questions": iv.model_dump(),
            "steps_completed": _notify(9, "Interview question generation", state),
        }

    # 10. Final report generation
    def final_report_node(state: AnalysisState) -> AnalysisState:
        r_obj = ResumeAnalysisResult.model_validate(state["resume_analysis"])
        j_obj = JobAnalysisResult.model_validate(state["job_analysis"])
        c_obj = CompatibilityResult.model_validate(state["compatibility"])
        g_obj = SkillGapResult.model_validate(state["skill_gap"])
        iv_obj = InterviewPrepResult.model_validate(state["interview_questions"])

        completed = _notify(10, "Final report generation", state)
        executive_summary = (
            f"Candidate scored {r_obj.ats_score:.1f}/100 on ATS readiness and "
            f"{c_obj.compatibility_score:.1f}% compatibility for {j_obj.job_title}. "
            f"Matched {len(c_obj.matching_skills)} key skills with {len(g_obj.missing_skills)} skill gap(s) identified."
        )

        final_report = {
            "job_title": j_obj.job_title,
            "seniority_level": j_obj.seniority_level,
            "executive_summary": executive_summary,
            "resume_analysis": state["resume_analysis"],
            "job_analysis": state["job_analysis"],
            "ats_evaluation": state["ats_evaluation"],
            "compatibility": state["compatibility"],
            "skill_gap": state["skill_gap"],
            "interview_questions": state["interview_questions"],
            "skill_comparison_summary": state.get("skill_comparison_summary", {}),
            "rag_metadata": state.get("rag_stats", {}),
            "workflow_steps": completed,
        }
        return {
            "final_report": final_report,
            "executive_summary": executive_summary,
            "steps_completed": completed,
        }

    graph = StateGraph(AnalysisState)
    graph.add_node("extract_resume", extract_resume_node)
    graph.add_node("extract_job", extract_job_node)
    graph.add_node("analyze_resume", analyze_resume_node)
    graph.add_node("analyze_job", analyze_job_node)
    graph.add_node("compare_skills", compare_skills_node)
    graph.add_node("ats_scoring", ats_scoring_node)
    graph.add_node("compatibility_scoring", compatibility_scoring_node)
    graph.add_node("missing_skills", missing_skills_node)
    graph.add_node("interview_questions", interview_questions_node)
    graph.add_node("final_report", final_report_node)

    graph.add_edge(START, "extract_resume")
    graph.add_edge("extract_resume", "extract_job")
    graph.add_edge("extract_job", "analyze_resume")
    graph.add_edge("analyze_resume", "analyze_job")
    graph.add_edge("analyze_job", "compare_skills")
    graph.add_edge("compare_skills", "ats_scoring")
    graph.add_edge("ats_scoring", "compatibility_scoring")
    graph.add_edge("compatibility_scoring", "missing_skills")
    graph.add_edge("missing_skills", "interview_questions")
    graph.add_edge("interview_questions", "final_report")
    graph.add_edge("final_report", END)

    return graph.compile()


def run_compatibility_workflow(
    user_id: int,
    resume_text: str = "",
    job_text: str = "",
    resume_filename: str = "resume.pdf",
    resume_bytes: bytes | None = None,
    job_filename: str = "job_description.txt",
    job_bytes: bytes | None = None,
    job_title_hint: str = "",
    provider: str = "ollama",
    model_name: str = "llama3.2",
    api_key: str = "",
    ollama_base_url: str = "http://localhost:11434",
    db_path: str = "",
    vector_store_dir: str = "",
    llm: BaseChatModel | None = None,
    rag_pipeline: VyraqRAGPipeline | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Execute the full 10-step VYRAQ AI LangGraph workflow and return final state."""
    compiled = build_analysis_workflow(
        llm=llm,
        rag_pipeline=rag_pipeline,
        progress_callback=progress_callback,
    )
    initial_state: AnalysisState = {
        "user_id": user_id,
        "resume_filename": resume_filename,
        "resume_bytes": resume_bytes,
        "resume_text": resume_text,
        "job_filename": job_filename,
        "job_bytes": job_bytes,
        "job_text": job_text,
        "job_title_hint": job_title_hint,
        "provider": provider,
        "model_name": model_name,
        "api_key": api_key,
        "ollama_base_url": ollama_base_url,
        "db_path": db_path,
        "vector_store_dir": vector_store_dir,
        "steps_completed": [],
    }
    return compiled.invoke(initial_state)
