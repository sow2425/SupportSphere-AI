"""Automated tests for RAG pipeline, job analysis, compatibility, skill gaps, and interview prep."""

from __future__ import annotations

from pathlib import Path

import pytest

from authentication.auth import register_user
from modules.compatibility import calculate_compatibility
from modules.interview_questions import generate_interview_questions
from modules.job_analyzer import analyze_job_description
from modules.resume_analyzer import analyze_resume
from modules.skill_gap import identify_missing_skills
from rag.rag_pipeline import LocalTfidfEmbeddings, VyraqRAGPipeline


def test_rag_pipeline_chunking_storage_and_user_scoped_retrieval(
    tmp_db: Path,
    tmp_path: Path,
    sample_resume_text: str,
    sample_job_text: str,
) -> None:
    u1 = register_user(tmp_db, "User One", "rag1@example.com", "Password123").user
    u2 = register_user(tmp_db, "User Two", "rag2@example.com", "Password123").user
    assert u1 is not None and u2 is not None

    rag_u1 = VyraqRAGPipeline(
        user_id=u1.id,
        collection_name="col_1",
        db_path=tmp_db,
        vector_store_dir=tmp_path / "vec",
        embeddings=LocalTfidfEmbeddings(),
    )
    counts = rag_u1.index_documents(
        resume_text=sample_resume_text,
        job_text=sample_job_text,
        resume_name="alex_resume.pdf",
        job_name="senior_ai.txt",
        chunk_size=260,
        chunk_overlap=40,
    )
    assert counts["resume_chunks"] >= 1
    assert counts["job_chunks"] >= 1

    hits = rag_u1.retrieve("LangGraph RAG retrieval pipeline", doc_type="resume", top_k=3)
    assert len(hits) >= 1
    assert any("LangGraph" in h.text or "RAG" in h.text for h in hits)

    context = rag_u1.build_grounded_context("LangChain LangGraph Pytest")
    assert "Retrieved Resume Evidence" in context
    assert "Retrieved Job Description Evidence" in context

    # Another user querying the same collection name must see zero chunks
    rag_u2 = VyraqRAGPipeline(
        user_id=u2.id,
        collection_name="col_1",
        db_path=tmp_db,
        vector_store_dir=tmp_path / "vec",
        embeddings=LocalTfidfEmbeddings(),
    )
    assert rag_u2.retrieve("LangGraph", doc_type="resume", top_k=3) == []


def test_compatibility_skill_gap_and_interview_questions(
    sample_resume_text: str,
    sample_job_text: str,
    mock_llm,
) -> None:
    res_analysis = analyze_resume(sample_resume_text, llm=mock_llm, job_context=sample_job_text)
    job_analysis = analyze_job_description(sample_job_text, llm=mock_llm)
    assert job_analysis.job_title == "Senior Python AI Engineer"
    assert "LangGraph" in job_analysis.required_skills

    comp = calculate_compatibility(res_analysis, job_analysis, llm=mock_llm)
    expected = (
        comp.breakdown.required_skills_score
        + comp.breakdown.preferred_skills_score
        + comp.breakdown.experience_alignment_score
        + comp.breakdown.domain_keyword_score
    )
    assert comp.compatibility_score == pytest.approx(expected, abs=0.1)
    assert "Python" in comp.matching_skills
    assert "Docker" in comp.missing_skills

    gap = identify_missing_skills(res_analysis, job_analysis, comp, llm=mock_llm)
    assert "FastMCP" in gap.missing_skills
    assert len(gap.recommended_skills_to_learn) >= 1

    iv = generate_interview_questions(res_analysis, job_analysis, gap, llm=mock_llm)
    assert len(iv.technical_questions) >= 1
    assert len(iv.role_specific_questions) >= 1
    assert len(iv.resume_based_questions) >= 1
    assert len(iv.hr_questions) >= 1
    assert iv.technical_questions[0].sample_answer


def test_interview_prep_result_normalizes_alternate_llm_keys() -> None:
    from modules.interview_questions import InterviewPrepResult

    # Alternate short keys wrapped inside "interview_questions"
    wrapped = InterviewPrepResult.model_validate(
        {
            "interview_questions": {
                "technical": [{"question": "Explain RAG chunking.", "sample_answer": "Split with overlap."}],
                "role_specific": [{"question": "Design an ATS scorer.", "answer": "Use weighted criteria."}],
                "resume_based": [{"question": "Describe NovaTech project.", "key_points": ["45% faster"]}],
                "behavioral_questions": [{"question": "Tell me about a conflict.", "sample_answer": "Aligned on goals."}],
            }
        }
    )
    assert wrapped.total_count() == 4
    assert wrapped.role_specific_questions[0].sample_answer == "Use weighted criteria."
    assert wrapped.resume_based_questions[0].key_answer_points == ["45% faster"]
    assert len(wrapped.hr_questions) == 1

    # Flat "questions" list format
    flat = InterviewPrepResult.model_validate(
        {
            "questions": [
                {"category": "technical", "question": "What is Python GIL?"},
                {"category": "hr", "question": "Why this role?"},
            ]
        }
    )
    assert len(flat.technical_questions) == 1
    assert len(flat.hr_questions) == 1
