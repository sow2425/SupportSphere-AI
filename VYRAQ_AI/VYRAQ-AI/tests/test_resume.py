"""Automated tests for PDF/DOCX/TXT document parsing and resume ATS analysis."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from docx import Document

from modules.document_parser import (
    SUPPORTED_JOB_EXTENSIONS,
    SUPPORTED_RESUME_EXTENSIONS,
    DocumentParseError,
    extract_document_text,
)
from modules.resume_analyzer import ATS_DISCLAIMER, analyze_resume, compute_ats_signals

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_extract_sample_pdf_and_docx() -> None:
    pdf_bytes = (PROJECT_ROOT / "sample_data" / "sample_resume.pdf").read_bytes()
    pdf_text, pdf_ext = extract_document_text("sample_resume.pdf", pdf_bytes, SUPPORTED_RESUME_EXTENSIONS)
    assert pdf_ext == "pdf"
    assert "ALEX RIVERA" in pdf_text
    assert "LangGraph" in pdf_text

    docx_bytes = (PROJECT_ROOT / "sample_data" / "sample_resume.docx").read_bytes()
    docx_text, docx_ext = extract_document_text("sample_resume.docx", docx_bytes, SUPPORTED_RESUME_EXTENSIONS)
    assert docx_ext == "docx"
    assert "NovaTech Labs" in docx_text


def test_extract_docx_with_tables_and_txt_job() -> None:
    doc = Document()
    doc.add_paragraph("Candidate Profile")
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Python"
    table.cell(0, 1).text = "SQLite"
    buf = io.BytesIO()
    doc.save(buf)

    text, ext = extract_document_text("profile.docx", buf.getvalue(), SUPPORTED_RESUME_EXTENSIONS)
    assert ext == "docx"
    assert "Python | SQLite" in text

    jd_bytes = (PROJECT_ROOT / "sample_data" / "sample_job_description.txt").read_bytes()
    jd_text, jd_ext = extract_document_text("jd.txt", jd_bytes, SUPPORTED_JOB_EXTENSIONS)
    assert jd_ext == "txt"
    assert "Senior Python AI Engineer" in jd_text


def test_document_parser_rejects_invalid_extensions_and_empty_text() -> None:
    with pytest.raises(DocumentParseError):
        extract_document_text("resume.txt", b"hello", SUPPORTED_RESUME_EXTENSIONS)

    with pytest.raises(DocumentParseError):
        extract_document_text("empty.txt", b"   \n  ", SUPPORTED_JOB_EXTENSIONS)


def test_compute_ats_signals_and_analyze_resume(sample_resume_text: str, mock_llm) -> None:
    signals = compute_ats_signals(sample_resume_text)
    assert signals["has_email"] is True
    assert signals["quantified_metrics_count"] >= 2
    assert "experience" in signals["detected_sections"]

    result = analyze_resume(sample_resume_text, llm=mock_llm)
    expected_sum = (
        result.ats_breakdown.keyword_relevance
        + result.ats_breakdown.quantified_impact
        + result.ats_breakdown.section_structure
        + result.ats_breakdown.clarity_formatting
    )
    assert result.ats_score == pytest.approx(expected_sum, abs=0.1)
    assert 0.0 <= result.ats_score <= 100.0
    assert "Python" in result.important_keywords_detected
    assert result.ats_disclaimer == ATS_DISCLAIMER
