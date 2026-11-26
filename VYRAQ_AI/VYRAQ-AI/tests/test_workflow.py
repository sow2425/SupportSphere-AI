"""End-to-end tests for the 10-step LangGraph workflow, provider error handling, and FastMCP tools."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ai.langgraph_workflow import run_compatibility_workflow
from ai.llm_factory import LLMFactoryError, get_llm
from authentication.auth import register_user
from database.db import get_user_analysis, save_analysis_record, save_job_record, save_resume_record
from mcp_server import server as mcp_srv
from rag.rag_pipeline import LocalTfidfEmbeddings, VyraqRAGPipeline


def test_full_10_step_langgraph_workflow_and_persistence(
    tmp_db: Path,
    tmp_path: Path,
    sample_resume_text: str,
    sample_job_text: str,
    mock_llm,
) -> None:
    user = register_user(tmp_db, "Test Candidate", "candidate@example.com", "Password123").user
    assert user is not None

    progress_events: list[tuple[int, int, str]] = []
    rag = VyraqRAGPipeline(
        user_id=user.id,
        collection_name="wf_test",
        db_path=tmp_db,
        vector_store_dir=tmp_path / "vec",
        embeddings=LocalTfidfEmbeddings(),
    )

    state = run_compatibility_workflow(
        user_id=user.id,
        resume_text=sample_resume_text,
        job_text=sample_job_text,
        resume_filename="sample_resume.pdf",
        job_filename="sample_job_description.txt",
        provider="ollama",
        model_name="llama3.2",
        db_path=str(tmp_db),
        vector_store_dir=str(tmp_path / "vec"),
        llm=mock_llm,
        rag_pipeline=rag,
        progress_callback=lambda s, t, lbl: progress_events.append((s, t, lbl)),
    )

    assert len(state["steps_completed"]) == 10
    assert len(progress_events) == 10
    assert state["steps_completed"] == [
        "Resume text extraction",
        "Job description extraction",
        "Resume analysis",
        "Job requirement analysis",
        "Skill comparison",
        "ATS scoring",
        "Compatibility scoring",
        "Missing skill identification",
        "Interview question generation",
        "Final report generation",
    ]

    report = state["final_report"]
    assert report["ats_evaluation"]["ats_score"] == 88.0
    assert report["compatibility"]["compatibility_score"] == 83.0
    assert report["rag_metadata"]["resume_chunks"] >= 1

    # Verify SQLite persistence of final report
    r_rec = save_resume_record(tmp_db, user.id, "sample_resume.pdf", "pdf", "", state["resume_text"])
    j_rec = save_job_record(tmp_db, user.id, report["job_title"], "sample_job_description.txt", "txt", "", state["job_text"])
    saved = save_analysis_record(
        tmp_db,
        user_id=user.id,
        resume_id=r_rec.id,
        job_id=j_rec.id,
        resume_filename="sample_resume.pdf",
        job_title=report["job_title"],
        provider="ollama",
        model_name="llama3.2",
        ats_score=report["ats_evaluation"]["ats_score"],
        compatibility_score=report["compatibility"]["compatibility_score"],
        report_data=report,
        summary=state["executive_summary"],
    )
    loaded = get_user_analysis(tmp_db, user.id, saved.id)
    assert loaded is not None
    assert loaded.report_data["compatibility"]["compatibility_score"] == 83.0


def test_graceful_errors_when_providers_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(LLMFactoryError) as exc_groq:
        get_llm(provider="groq", model_name="llama-3.3-70b-versatile", api_key="")
    assert "api key" in str(exc_groq.value).lower()

    with pytest.raises(LLMFactoryError) as exc_gemini:
        get_llm(provider="gemini", model_name="gemini-3.8-flash", api_key="")
    assert "api key" in str(exc_gemini.value).lower()

    with pytest.raises(LLMFactoryError) as exc_ollama:
        get_llm(provider="ollama", model_name="llama3.2", ollama_base_url="http://127.0.0.1:19999", verify_ollama=True)
    assert "ollama" in str(exc_ollama.value).lower()


def test_groq_chat_model_rest_invocation(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai.groq_provider import GroqChatModel

    class FakeResponse:
        status_code = 200
        text = "{}"

        def json(self):
            return {"choices": [{"message": {"content": "OK."}}]}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            assert "api.groq.com" in url
            assert json["model"] == "llama-3.3-70b-versatile"
            return FakeResponse()

    monkeypatch.setattr("ai.groq_provider.httpx.Client", FakeClient)
    llm = GroqChatModel(model="llama-3.3-70b-versatile", api_key="gsk_test")
    reply = llm.invoke("Reply with OK.")
    assert reply.content == "OK."


def test_mcp_server_tools_reuse_services(
    sample_resume_text: str,
    sample_job_text: str,
    mock_llm,
) -> None:
    with patch.object(mcp_srv, "_resolve_model", return_value=mock_llm):
        r_json = json.loads(mcp_srv.analyze_resume(sample_resume_text, job_context=sample_job_text))
        assert r_json["ats_score"] == 88.0

        j_json = json.loads(mcp_srv.analyze_job_description(sample_job_text))
        assert j_json["job_title"] == "Senior Python AI Engineer"

        c_json = json.loads(mcp_srv.calculate_compatibility(sample_resume_text, sample_job_text))
        assert c_json["compatibility_score"] == 83.0

        g_json = json.loads(mcp_srv.identify_missing_skills(sample_resume_text, sample_job_text))
        assert "FastMCP" in g_json["missing_skills"]

        iv_json = json.loads(mcp_srv.generate_interview_questions(sample_resume_text, sample_job_text))
        assert len(iv_json["technical_questions"]) >= 1


def test_extract_message_text_handles_gemini_thinking_and_text_blocks() -> None:
    from langchain_core.messages import AIMessage
    from ai.llm_factory import _extract_message_text, extract_json_payload

    msg = AIMessage(
        content=[
            {"type": "thinking", "thinking": "Let me evaluate {skills} first."},
            {
                "type": "text",
                "text": '```json\n{"job_title": "AI Engineer", "required_skills": ["Python"]}\n```',
                "extras": {"signature": "Eu123"},
            },
        ]
    )
    text = _extract_message_text(msg)
    assert "thinking" not in text
    parsed = extract_json_payload(text)
    assert parsed["job_title"] == "AI Engineer"
