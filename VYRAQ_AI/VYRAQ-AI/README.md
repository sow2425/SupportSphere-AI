# VYRAQ AI – AI-Powered Resume and Job Compatibility Analyzer

VYRAQ AI helps students and job seekers evaluate how well their resume aligns with a target job description using **LangChain**, **LangGraph**, **RAG**, **Ollama**, **Google Gemini**, **Streamlit**, and **SQLite**.

## Features

- **Mandatory User Authentication**: Secure registration and login with PBKDF2-SHA256 password hashing, duplicate email checks, session management, and user-isolated records in `vyraq.db`.
- **Multi-Format Document Parsing**: Extracts text from PDF (`PyPDF`) and DOCX (`python-docx`) resumes, plus PDF, DOCX, or TXT/pasted job descriptions.
- **10-Step LangGraph Workflow**:
  1. Resume text extraction
  2. Job description extraction
  3. Resume analysis (with RAG indexing & retrieval)
  4. Job requirement analysis
  5. Skill comparison
  6. Transparent ATS scoring
  7. Weighted compatibility scoring
  8. Missing skill identification
  9. Grounded interview question generation
  10. Final report compilation
- **Meaningful RAG Pipeline**: Chunks resumes and job descriptions, generates embeddings (Gemini, Ollama, or local deterministic `LocalTfidfEmbeddings`), stores vectors per user in SQLite (`vector_chunks`) and ChromaDB (when installed), and feeds retrieved evidence into LLM prompts.
- **Dual LLM Providers**:
  - **Open-Source Local**: Ollama (`llama3.2`, `mistral`, `qwen2.5`, etc.) with live local model discovery.
  - **Closed-Source Cloud**: Google Gemini (`gemini-3.8-flash`, `gemini-3.8-pro`, etc.) via `.env` or session key.
- **Optional FastMCP Server**: Standalone MCP server (`mcp_server/server.py`) exposing 5 tools reusing the core services.

## Project Structure

```text
VYRAQ-AI/
├── app.py
├── config.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── run.bat
├── database/
│   ├── db.py
│   └── models.py
├── authentication/
│   └── auth.py
├── ai/
│   ├── llm_factory.py
│   ├── gemini_provider.py
│   ├── ollama_provider.py
│   └── langgraph_workflow.py
├── modules/
│   ├── document_parser.py
│   ├── resume_analyzer.py
│   ├── job_analyzer.py
│   ├── compatibility.py
│   ├── skill_gap.py
│   └── interview_questions.py
├── rag/
│   └── rag_pipeline.py
├── mcp_server/
│   └── server.py
├── pages/
│   ├── auth_pages.py
│   ├── dashboard.py
│   ├── upload.py
│   ├── results.py
│   ├── history.py
│   └── settings.py
├── sample_data/
├── tests/
└── uploads/
```

## Quick Start (Windows)

1. Open the `VYRAQ-AI` folder in VS Code.
2. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Configure environment variables on the server:
   ```powershell
   copy .env.example .env
   ```
   - Set `GEMINI_API_KEY=your_key` in `.env` (or in **Streamlit Cloud Secrets**) and `VYRAQ_DEFAULT_PROVIDER=gemini`.
   - `VYRAQ_PUBLIC_MODE=true` (default) hides API keys, Ollama setup, and Settings from regular website visitors so they can sign up, upload resumes, and view reports seamlessly.
   - Set `VYRAQ_PUBLIC_MODE=false` (for local developer mode) or add your email to `VYRAQ_ADMIN_EMAILS=you@example.com` to access the **Settings / Model Selection** page.
5. Start the Streamlit app:
   ```powershell
   streamlit run app.py
   ```
   Or double-click `run.bat`.

## Optional: Running the FastMCP Server

The MCP server is completely optional and not required to run the Streamlit UI.
To launch it over stdio for MCP clients:

```powershell
python -m mcp_server.server
```

## Running Automated Tests

All tests use isolated temporary SQLite databases and mocked LangChain LLM responses:

```powershell
python -m pytest -v
```
