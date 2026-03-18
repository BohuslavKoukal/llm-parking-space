# Parking Chatbot: RAG + HITL + MCP

A production-ready parking assistant that combines RAG retrieval, stateful reservation orchestration, admin approval via LangGraph interrupt and resume, secure MCP-based reservation recording, layered guardrails, evaluation, and load testing.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-enabled-green)
![LangGraph](https://img.shields.io/badge/LangGraph-stateful-orange)

Detailed technical documentation: docs/SOLUTION.md

## Quick Start

1. Install dependencies: pip install -r requirements.txt
2. Copy environment file: copy .env.example .env (Windows) or cp .env.example .env (macOS/Linux)
3. Start Weaviate: docker compose up -d weaviate
4. Initialize DB and ingest data:
   - python -c "from app.database.sql_client import init_db; init_db()"
   - python -m app.rag.ingestion
5. Start chatbot: streamlit run app/main.py

For complete setup, environment variable details, architecture, and operations guidance, see docs/SOLUTION.md.

## Project Structure

```text
llm-parking-space/
├── app/                                # Application package
│   ├── __init__.py                     # Package marker (if present)
│   ├── main.py                         # Streamlit UI, state handling, admin decision notification
│   ├── chatbot/                        # LangGraph orchestration and prompt/chain definitions
│   │   ├── __init__.py                 # Chatbot package marker
│   │   ├── chains.py                   # LangChain chain builders (RAG, intent, reservation, guardrails)
│   │   ├── graph.py                    # LangGraph state graph, interrupt/resume flow, checkpoint wiring
│   │   └── prompts.py                  # System, routing, extraction, and guardrail prompts
│   ├── database/                       # SQLAlchemy models and DB helpers
│   │   ├── __init__.py                 # Database package marker
│   │   ├── models.py                   # Parking and reservation ORM models
│   │   └── sql_client.py               # Engine/session lifecycle, reservation CRUD, status updates
│   ├── evaluation/                     # RAGAS evaluation logic and reporting
│   │   ├── __init__.py                 # Evaluation package marker
│   │   ├── ragas_eval.py               # Evaluation pipeline and metric computation
│   │   └── report.py                   # Markdown and JSON report generation
│   ├── guardrails/                     # Safety checks and blocking logic
│   │   ├── __init__.py                 # Guardrails package marker
│   │   └── filters.py                  # Presidio and regex/LLM guardrail implementation
│   ├── mcp_client/                     # MCP client boundary used by graph node
│   │   ├── __init__.py                 # MCP client package marker
│   │   └── tools.py                    # MCP server process params and write tool invocation
│   └── rag/                            # RAG indexing and retrieval utilities
│       ├── __init__.py                 # RAG package marker
│       ├── embeddings.py               # Embedding model builder
│       ├── ingestion.py                # Static data ingestion into Weaviate
│       └── weaviate_client.py          # Weaviate client and retriever helpers
├── mcp_server/                         # MCP server implementation
│   ├── __init__.py                     # MCP server package marker
│   ├── server.py                       # MCP tool server (write/read/stats tools)
│   ├── file_writer.py                  # Reservation file write/read/stat helpers
│   └── security.py                     # API-key validation and payload validation
├── scripts/                            # Operational scripts
│   ├── admin_review.py                 # Admin console for approve/reject and graph resume
│   ├── run_evaluation.py               # Evaluation runner script
│   └── load_test.py                    # Load testing and latency report generation
├── tests/                              # Full automated test suite
│   ├── conftest.py                     # Shared fixtures and test DB lifecycle
│   ├── test_admin_cli.py               # Admin CLI tests
│   ├── test_chatbot.py                 # Routing and intent tests
│   ├── test_checkpointer.py            # Checkpointer and state persistence tests
│   ├── test_evaluation.py              # Evaluation/report tests
│   ├── test_functional.py              # Functional graph flow tests
│   ├── test_guardrails.py              # Guardrail tests
│   ├── test_hitl.py                    # HITL interrupt/resume tests
│   ├── test_integration.py             # End-to-end integration pipeline tests
│   ├── test_load_test.py               # Load test utility/report tests
│   ├── test_mcp_client.py              # MCP client tests
│   ├── test_mcp_server.py              # MCP server tests
│   ├── test_notification.py            # Admin decision notification tests
│   └── test_rag.py / test_rag_db.py    # RAG and SQL-context tests
├── data/                               # Static and dynamic data files
│   ├── static/parking_info.json        # Source static parking knowledge
│   ├── dynamic/seed_data.csv           # Seed dynamic operational values
│   └── evaluation/eval_questions.json  # Evaluation dataset
├── docs/                               # Technical documentation and generated diagrams
│   ├── SOLUTION.md                     # Unified architecture and implementation documentation
│   └── graph_diagram.png               # Generated graph diagram artifact
├── reports/                            # Evaluation and load-test outputs
├── docker-compose.yml                  # Local services orchestration (Weaviate)
├── Dockerfile                          # Container build recipe
├── requirements.txt                    # Python dependencies
├── .env.example                        # Environment template
└── check_schema.py                     # Local schema check helper
```

## Stage Completion

| Stage | Description | Status |
|-------|-------------|--------|
| 1 | RAG Chatbot + Guardrails + Evaluation | ✅ Complete |
| 2 | Human-in-the-Loop Admin Approval | ✅ Complete |
| 3 | MCP Server for Reservation Recording | ✅ Complete |
| 4 | Full Orchestration + Load Testing | ✅ Complete |

## Running Tests

Run full suite:

```bash
pytest tests/ -v
```

Run a specific file:

```bash
pytest tests/test_notification.py -v
```

Run only integration tests:

```bash
pytest tests/test_integration.py -v
```

Current expected count: 100+ tests (currently 106 collected).

## Key Design Decisions

- LangGraph interrupt for HITL:
  - interrupt() allows reservation submission to pause at submit_to_admin and resume on the same thread with Command(resume=...).
  - This gives deterministic, auditable approval control without losing conversation state.

- Weaviate plus SQL split:
  - Weaviate handles semantic retrieval over descriptive parking content.
  - SQLite handles transactional and dynamic data (reservation records, status transitions, live values).

- Presidio-first guardrails:
  - Presidio and regex checks provide low-latency deterministic safety filtering.
  - LLM guardrails run as a second semantic safety layer for subtle abuse patterns.

- MCP Python library over FastAPI:
  - MCP provides a tool-oriented protocol boundary aligned with agent workflows.
  - It keeps file-writing capability isolated behind authenticated tools without coupling graph nodes to direct file I/O.

## Additional Operational Commands

Run admin review console in a separate terminal:

```bash
python scripts/admin_review.py
```

Run evaluation:

```bash
python scripts/run_evaluation.py
```

Run load test:

```bash
python scripts/load_test.py
```
