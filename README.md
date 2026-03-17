# Parking Chatbot (RAG)

## Project Overview
Parking Chatbot is a production-ready Python scaffold for a Retrieval-Augmented Generation (RAG) assistant focused on parking information and reservation support. It combines LangChain and LangGraph orchestration with OpenAI GPT-4o, Weaviate vector search, SQLAlchemy persistence, and Presidio guardrails.

## Architecture Diagram (ASCII)
```
+--------------------+        +---------------------+
| Streamlit UI       | -----> | LangGraph Workflow  |
| app/main.py        |        | guardrail -> intent |
+--------------------+        | -> rag/reservation  |
          |                   +----------+----------+
          |                              |
          v                              v
+--------------------+        +---------------------+
| Guardrails         |        | RAG Layer           |
| Presidio + regex   |        | Weaviate retriever  |
+--------------------+        +---------------------+
          |                              |
          +---------------+--------------+
                          v
                 +------------------+
                 | OpenAI GPT-4o    |
                 +------------------+
                          |
                          v
                 +------------------+
                 | SQL Database     |
                 | dynamic config & |
                 | reservations     |
                 +------------------+
```

## Tech Stack
- Python 3.11
- Streamlit (chat interface)
- LangChain + LangGraph
- OpenAI GPT-4o + OpenAI embeddings
- Weaviate vector database
- SQLAlchemy ORM (+ Alembic-ready)
- Presidio Analyzer/Anonymizer
- Pytest + GitHub Actions CI
- Docker + Docker Compose

## Setup Instructions
### Local
1. Create and activate a Python 3.11 environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy environment variables:
   ```bash
   cp .env.example .env
   ```
4. (Optional, recommended for Presidio quality):
   ```bash
   python -m spacy download en_core_web_lg
   ```
5. Run the app:
   ```bash
   streamlit run app/main.py
   ```

> **Upgrading from Stage 1?** If you have an existing `parking.db` from a
> pre-Stage-2 version it may be missing the `thread_id` column on the
> `reservations` table.  The application auto-migrates the schema on first
> run — no manual action is required.

## Running the Application

Run each step in order for a full local setup:

**Step 1 — Start Weaviate:**
```bash
docker compose up -d weaviate
```

**Step 2 — Initialise the database:**
```bash
python -c "from app.database.sql_client import init_db; init_db()"
```

**Step 3 — Ingest parking data into Weaviate:**
```bash
python -m app.rag.ingestion
```

**Step 4 — Start the chatbot:**
```bash
streamlit run app/main.py
```

> **MCP Server:** The MCP server is launched automatically as a subprocess when a reservation is confirmed. You do not need to start it manually.

**Step 5 — Run the admin console (when a reservation awaits approval):**
```bash
python scripts/admin_review.py
```
> The admin console must be run in a **separate terminal** while the chatbot is running. Both processes share `checkpoints.db` and `parking.db`.

## Important: Python Path Setup
This project requires the repository root to be on the Python path.
This is handled automatically if you copy .env.example to .env:
```bash
cp .env.example .env
```

For manual terminal launches on Windows PowerShell:
```powershell
$env:PYTHONPATH = "."
streamlit run app/main.py
```

For manual terminal launches on Mac/Linux:
```bash
export PYTHONPATH=.
streamlit run app/main.py
```

When using Docker Compose, PYTHONPATH is set automatically.

### Docker
1. Create `.env` from `.env.example`.
2. Start services:
   ```bash
   docker compose up --build
   ```
3. Open `http://localhost:8501`.

## Environment Variables Reference
- `OPENAI_API_KEY`: OpenAI API key for GPT-4o and embeddings
- `WEAVIATE_URL`: Weaviate endpoint (default local: `http://localhost:8080`)
- `WEAVIATE_API_KEY`: Optional API key for secured Weaviate deployments
- `DATABASE_URL`: SQLAlchemy URL (default sqlite file)
- `ENVIRONMENT`: Environment name (development/production)
- `LOG_LEVEL`: Logging verbosity (e.g., INFO, DEBUG)
- `MCP_API_KEY`: API key used by the MCP server/client reservation tools
- `RESERVATIONS_FILE_PATH`: Output file path for confirmed reservation entries (default `data/reservations.txt`)

```env
MCP_API_KEY=your_mcp_api_key_here
RESERVATIONS_FILE_PATH=data/reservations.txt
```

## Project Structure
```
parking-chatbot/
├── app/
│   ├── main.py
│   ├── chatbot/
│   ├── mcp_client/
│   ├── rag/
│   ├── database/
│   ├── guardrails/
│   └── evaluation/
├── mcp_server/
│   ├── __init__.py
│   ├── server.py
│   ├── file_writer.py
│   └── security.py
├── data/
│   ├── static/
│   ├── dynamic/
│   └── reservations.example.txt
├── docs/
│   └── SOLUTION.md
├── scripts/
│   ├── admin_review.py
│   └── run_evaluation.py
├── tests/
│   ├── conftest.py
│   ├── test_rag.py
│   ├── test_rag_db.py
│   ├── test_chatbot.py
│   ├── test_guardrails.py
│   ├── test_functional.py
│   ├── test_evaluation.py
│   ├── test_checkpointer.py
│   ├── test_hitl.py
│   └── test_admin_cli.py
├── .github/workflows/
├── .env.example
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
└── README.md
```

## Running Tests
```bash
pytest tests/ -v --tb=short
```

## Running Evaluation
Run the standalone evaluation pipeline with:
```bash
python scripts/run_evaluation.py
```

The evaluation measures RAG quality using RAGAS metrics:
- faithfulness
- answer_relevancy
- context_recall
- context_precision

The generated Markdown report is saved in the reports directory as:
- reports/eval_report_{timestamp}.md

The latest machine-readable results are also saved to:
- reports/latest_results.json

Before running evaluation, ensure:
- Weaviate is running and reachable
- Parking data has already been ingested into Weaviate

## Admin Review Console

The admin CLI allows an administrator to approve or reject pending parking reservations.

Prerequisites:
- The chatbot app must be running (`streamlit run app/main.py`)
- A user must have completed the reservation flow

How to run in a separate terminal:
```bash
python scripts/admin_review.py
```

What it does:
- Lists all pending reservations waiting for admin approval
- Allows admin to approve or reject each reservation
- Resumes the paused LangGraph graph with the admin decision
- User sees the result on their next message in the chat UI

> **Note:** The admin console and the chatbot share the same `checkpoints.db` and `parking.db` files.

## Stage Completion Status

| Stage | Description | Status |
|---|---|---|
| 1 | RAG Chatbot + Guardrails + Evaluation | ✅ Complete |
| 2 | Human-in-the-Loop Agent | ✅ Complete |
| 3 | MCP Server | ✅ Complete |
| 4 | Full Orchestration | 🔄 Planned |
