"""Shared Pytest fixtures and mocked LangChain chat model for VYRAQ AI tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.db import init_db  # noqa: E402


class MockVyraqChatModel(BaseChatModel):
    """Deterministic LangChain BaseChatModel mock returning structured JSON per prompt stage."""

    calls: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "mock-vyraq-llm"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt_text = "\n".join(str(m.content) for m in messages)
        self.calls.append(prompt_text[:120])

        if "transparent ATS evaluation" in prompt_text:
            payload = {
                "candidate_summary": "Python AI Engineer with 4+ years building LangGraph, RAG, and Streamlit pipelines.",
                "detected_skills": ["Python", "LangChain", "LangGraph", "RAG", "SQLite", "Streamlit", "Pytest", "Ollama"],
                "technical_skills": ["Python", "LangChain", "LangGraph", "RAG", "SQLite", "Streamlit", "Pytest", "Ollama"],
                "soft_skills": ["Collaboration", "Technical Communication"],
                "experience_highlights": [
                    "Built multi-step LangGraph workflow reducing screening time by 45%",
                    "Implemented RAG pipeline indexing 12,000+ documents",
                ],
                "education": ["B.S. in Computer Science, State University (2021)"],
                "certifications": [],
                "estimated_years_experience": "4+ years",
                "ats_breakdown": {
                    "keyword_relevance": 30.0,
                    "quantified_impact": 21.0,
                    "section_structure": 19.0,
                    "clarity_formatting": 18.0,
                },
                "ats_score": 88.0,
                "resume_quality_assessment": "Strong quantified achievements, clear sections, and relevant AI stack keywords.",
                "important_keywords_detected": ["Python", "LangChain", "LangGraph", "RAG", "Pytest", "SQLite"],
                "missing_keywords": ["Docker", "CI/CD", "FastMCP"],
                "ats_suggestions": [
                    "Add explicit containerization (Docker) or CI/CD pipeline metrics if applicable.",
                    "Mention MCP / FastMCP exposure if experienced with tool servers.",
                ],
            }
        elif "Analyze the following job description" in prompt_text:
            payload = {
                "job_title": "Senior Python AI Engineer",
                "seniority_level": "Senior",
                "required_skills": ["Python", "LangChain", "LangGraph", "RAG", "SQLite", "Pytest", "Google Gemini", "Ollama"],
                "preferred_skills": ["Streamlit", "FastMCP", "Docker", "CI/CD"],
                "key_responsibilities": [
                    "Architect modular AI workflows for document parsing and structured scoring",
                    "Write automated tests with mocked LLM responses",
                ],
                "experience_requirements": "3+ years of software engineering and applied AI experience",
                "education_requirements": "Bachelor's or Master's degree in Computer Science",
                "domain_keywords": ["RAG", "LangGraph", "Vector Search", "Automated Testing"],
                "role_summary": "Build production Python LLM and RAG orchestration systems with strong testing.",
            }
        elif "Evaluate compatibility between the candidate resume and target job description" in prompt_text:
            payload = {
                "breakdown": {
                    "required_skills_score": 44.0,
                    "preferred_skills_score": 8.0,
                    "experience_alignment_score": 22.0,
                    "domain_keyword_score": 9.0,
                },
                "compatibility_score": 83.0,
                "matching_skills": ["Python", "LangChain", "LangGraph", "RAG", "SQLite", "Pytest", "Ollama", "Streamlit"],
                "partially_matching_skills": ["Google Gemini"],
                "missing_skills": ["FastMCP", "Docker", "CI/CD"],
                "experience_comparison": "Candidate has 4+ years in Python/AI matching the 3+ year requirement.",
                "match_explanation": "Strong alignment across core Python, LangChain, LangGraph, RAG, and Pytest requirements; missing Docker/CI/CD and FastMCP.",
                "strengths": ["Direct LangGraph and RAG production metrics", "Mocked Pytest experience"],
                "risks_or_gaps": ["No explicit Docker or CI/CD pipeline evidence"],
            }
        elif "Produce a structured Skill Gap Analysis" in prompt_text:
            payload = {
                "required_skills": ["Python", "LangChain", "LangGraph", "RAG", "SQLite", "Pytest", "Google Gemini", "Ollama", "Docker"],
                "available_skills": ["Python", "LangChain", "LangGraph", "RAG", "SQLite", "Pytest", "Ollama"],
                "missing_skills": ["FastMCP", "Docker", "CI/CD"],
                "recommended_skills_to_learn": [
                    {
                        "skill": "FastMCP",
                        "priority": "High",
                        "reason": "Requested as preferred MCP tool server framework.",
                        "learning_focus": "Build and test a small stdio MCP server exposing existing Python tools.",
                    },
                    {
                        "skill": "Docker",
                        "priority": "Medium",
                        "reason": "Needed for reproducible deployment pipelines.",
                        "learning_focus": "Containerize a Streamlit + SQLite Python app with multi-stage Dockerfile.",
                    },
                ],
                "gap_summary": "Candidate meets almost all core AI requirements; closing FastMCP and Docker gaps will maximize fit.",
            }
        else:
            q_item = lambda cat, q, topic: {
                "category": cat,
                "question": q,
                "difficulty": "Medium",
                "target_skill_or_topic": topic,
                "rationale": f"Evaluates candidate readiness on {topic}.",
                "key_answer_points": ["Explain architecture", "Discuss trade-offs", "Cite measurable outcome"],
                "sample_answer": f"I approach {topic} by isolating interfaces, validating inputs, and testing with mocks.",
            }
            payload = {
                "technical_questions": [
                    q_item("technical", "How do you chunk documents and evaluate retrieval quality in a RAG pipeline?", "RAG"),
                    q_item("technical", "How would you structure a 10-node LangGraph workflow for deterministic failure handling?", "LangGraph"),
                ],
                "role_specific_questions": [
                    q_item("role_specific", "How would you expose existing Python analysis services through FastMCP?", "FastMCP"),
                ],
                "resume_based_questions": [
                    q_item("resume_based", "Walk through how your LangGraph workflow reduced screening time by 45%.", "NovaTech Experience"),
                ],
                "hr_questions": [
                    q_item("hr", "How do you communicate transparent AI limitations to non-technical stakeholders?", "Communication"),
                ],
            }

        message = AIMessage(content=json.dumps(payload))
        return ChatResult(generations=[ChatGeneration(message=message)])


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Create and initialize an isolated temporary SQLite database."""
    db_file = tmp_path / "test_vyraq.db"
    init_db(db_file)
    return db_file


@pytest.fixture
def mock_llm() -> MockVyraqChatModel:
    """Provide a fresh mocked LangChain chat model."""
    return MockVyraqChatModel()


@pytest.fixture
def sample_resume_text() -> str:
    return (
        "Alex Rivera\n"
        "alex.rivera@example.com | github.com/alexrivera\n\n"
        "SUMMARY\n"
        "Python AI Engineer with 4+ years building LangChain, LangGraph, and RAG pipelines.\n\n"
        "SKILLS\n"
        "Python, LangChain, LangGraph, RAG, SQLite, Streamlit, Pytest, Ollama, NumPy, scikit-learn\n\n"
        "EXPERIENCE\n"
        "- Built a multi-step LangGraph workflow reducing screening time by 45%.\n"
        "- Implemented a RAG pipeline indexing 12,000+ technical documents.\n\n"
        "EDUCATION\n"
        "B.S. in Computer Science, State University (2021)\n"
    )


@pytest.fixture
def sample_job_text() -> str:
    jd_path = PROJECT_ROOT / "sample_data" / "sample_job_description.txt"
    return jd_path.read_text(encoding="utf-8")
