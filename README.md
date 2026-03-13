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

## Project Structure
```
parking-chatbot/
├── app/
│   ├── main.py
│   ├── chatbot/
│   ├── rag/
│   ├── database/
│   ├── guardrails/
│   └── evaluation/
├── data/
│   ├── static/
│   └── dynamic/
├── tests/
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

## Stages Overview
- **Stage 1:** Core backend setup (RAG wiring, graph flow, and baseline modules).
- **Stage 2:** Reservation workflow hardening (validation, persistence, and admin confirmations).
- **Stage 3:** Safety and observability expansion (guardrails, monitoring, and failure handling).
- **Stage 4:** Evaluation and optimization (RAGAS metrics, reporting, and production tuning).
