"""Document parsing utilities for PDF, DOCX, and plain text files."""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import BinaryIO

from docx import Document as DocxDocument
from pypdf import PdfReader


class DocumentParseError(Exception):
    """Raised when a document cannot be parsed or contains no extractable text."""


SUPPORTED_RESUME_EXTENSIONS = {".pdf", ".docx"}
SUPPORTED_JOB_EXTENSIONS = {".pdf", ".docx", ".txt"}


def normalize_extracted_text(text: str) -> str:
    """Clean excessive whitespace and control characters while preserving structure."""
    text = text.replace("\x00", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    cleaned_lines: list[str] = []
    blank_streak = 0
    for line in lines:
        if not line:
            blank_streak += 1
            if blank_streak <= 1:
                cleaned_lines.append("")
        else:
            blank_streak = 0
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def extract_text_from_pdf(source: bytes | BinaryIO | str | Path) -> str:
    """Extract readable text from a PDF file or byte stream using PyPDF."""
    try:
        if isinstance(source, (str, Path)):
            reader = PdfReader(str(source))
        elif isinstance(source, (bytes, bytearray)):
            reader = PdfReader(io.BytesIO(bytes(source)))
        else:
            source.seek(0)
            reader = PdfReader(source)

        if reader.is_encrypted:
            try:
                decrypt_result = reader.decrypt("")
                if not decrypt_result:
                    raise DocumentParseError("The uploaded PDF is password-protected and cannot be read.")
            except Exception as exc:
                raise DocumentParseError("The uploaded PDF is encrypted and cannot be parsed.") from exc

        pages_text: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages_text.append(page_text)

        combined = normalize_extracted_text("\n\n".join(pages_text))
        if not combined:
            raise DocumentParseError(
                "No selectable text was found in this PDF. Please upload a text-based PDF or DOCX."
            )
        return combined
    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError(f"Failed to parse PDF document: {exc}") from exc


def extract_text_from_docx(source: bytes | BinaryIO | str | Path) -> str:
    """Extract text from paragraphs and tables of a DOCX file using python-docx."""
    try:
        if isinstance(source, (str, Path)):
            doc = DocxDocument(str(source))
        elif isinstance(source, (bytes, bytearray)):
            doc = DocxDocument(io.BytesIO(bytes(source)))
        else:
            source.seek(0)
            doc = DocxDocument(source)

        chunks: list[str] = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                chunks.append(text)

        for table in doc.tables:
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_cells:
                    deduped = list(dict.fromkeys(row_cells))
                    chunks.append(" | ".join(deduped))

        combined = normalize_extracted_text("\n".join(chunks))
        if not combined:
            raise DocumentParseError("The uploaded DOCX document appears to be empty.")
        return combined
    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError(f"Failed to parse DOCX document: {exc}") from exc


def extract_text_from_txt(source: bytes | BinaryIO | str | Path) -> str:
    """Extract text from a plain text file or byte stream."""
    try:
        if isinstance(source, (str, Path)):
            raw = Path(source).read_bytes()
        elif isinstance(source, (bytes, bytearray)):
            raw = bytes(source)
        else:
            source.seek(0)
            raw = source.read()

        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError:
            decoded = raw.decode("latin-1", errors="replace")

        combined = normalize_extracted_text(decoded)
        if not combined:
            raise DocumentParseError("The uploaded text file is empty.")
        return combined
    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError(f"Failed to parse text document: {exc}") from exc


def extract_document_text(
    filename: str,
    content: bytes,
    allowed_extensions: set[str] | None = None,
) -> tuple[str, str]:
    """Extract text from PDF, DOCX, or TXT bytes. Returns (cleaned_text, ext)."""
    ext = Path(filename).suffix.lower()
    if allowed_extensions is not None and ext not in allowed_extensions:
        allowed_str = ", ".join(sorted(allowed_extensions))
        raise DocumentParseError(f"Unsupported file format '{ext}'. Allowed formats: {allowed_str}")

    if ext == ".pdf":
        return extract_text_from_pdf(content), "pdf"
    if ext == ".docx":
        return extract_text_from_docx(content), "docx"
    if ext == ".txt":
        return extract_text_from_txt(content), "txt"

    raise DocumentParseError(f"Unsupported file extension '{ext}'.")


def save_uploaded_file(
    upload_dir: str | Path,
    user_id: int,
    filename: str,
    content: bytes,
) -> Path:
    """Persist uploaded document bytes inside a user-isolated upload directory."""
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename).name) or "document"
    user_dir = Path(upload_dir) / f"user_{user_id}"
    user_dir.mkdir(parents=True, exist_ok=True)

    target = user_dir / safe_name
    counter = 1
    while target.exists():
        stem = Path(safe_name).stem
        suffix = Path(safe_name).suffix
        target = user_dir / f"{stem}_{counter}{suffix}"
        counter += 1

    target.write_bytes(content)
    return target
