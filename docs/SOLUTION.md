# 1. Project Overview

This project is a production-ready parking chatbot that combines Retrieval-Augmented Generation (RAG), stateful reservation orchestration, human-in-the-loop approval, and secure MCP tool integration.

Current end-to-end capabilities:
- RAG-based information retrieval from static parking knowledge.
- Interactive reservation collection with one-field-at-a-time completion.
- Human-in-the-loop admin approval using LangGraph interrupt().
- MCP server writes confirmed reservations to file.
- User notification after admin approval or rejection.
- Two-layer guardrails (Presidio first, then LLM guardrail chain).
- RAGAS evaluation framework with report generation.
- Load testing with component latency reporting.

The system manages five parking spaces:
- CityPark Central (parking_001)
- AirportPark Express (parking_002)
- ShoppingMall Park (parking_003)
- Stadium Park (parking_004)
- OldTown Garage (parking_005)

Roadmap status:

| Stage | Scope | Status |
| --- | --- | --- |
| 1 | RAG chatbot + guardrails + evaluation baseline | ✅ Complete |
| 2 | Human-in-the-loop admin approval flow | ✅ Complete |
| 3 | MCP server integration for reservation recording | ✅ Complete |
| 4 | Full orchestration, integration coverage, load testing | ✅ Complete |

Current System Status: All 4 stages complete.

# 2. Technology Stack

| Component | Technology | Version | Purpose |
| --- | --- | --- | --- |
| LLM | OpenAI GPT-4o via LangChain OpenAI | openai>=1.30.0, langchain-openai>=0.1.0 | Answer generation, intent classification, extraction, guardrail checks |
| Orchestration | LangGraph + LangChain | langgraph>=0.1.0, langchain>=0.2.0 | Stateful graph runtime and decision routing |
| Checkpointing | LangGraph SqliteSaver checkpointer | langgraph-checkpoint-sqlite | Pause/resume state persistence across processes |
| Vector Store | Weaviate | weaviate-client>=4.5.0 | Semantic retrieval over static parking corpus |
| Relational DB | SQLite + SQLAlchemy ORM | sqlalchemy>=2.0.0 | Dynamic values, reservation records, status lifecycle |
| Embeddings | text-embedding-3-small | openai>=1.30.0 | Vector embedding generation for retrieval |
| Guardrails | Presidio analyzer and anonymizer + LLM chain | presidio-analyzer>=2.2.0, presidio-anonymizer>=2.2.0 | Deterministic and semantic safety checks |
| MCP Server | Python mcp library | mcp>=1.0.0 | Tool server for reservation file writing |
| UI | Streamlit | streamlit>=1.35.0 | Chat UI and admin decision notification UX |
| Evaluation | RAGAS evaluation framework | ragas>=0.1.0 | Faithfulness and retrieval quality benchmarking |
| Testing | pytest + pytest-asyncio | pytest>=8.0.0, pytest-asyncio>=0.23.0 | Unit, functional, integration, and script-level tests |
| Operations | Docker Compose | Docker | Local Weaviate service orchestration |
| CI/CD | GitHub Actions | actions/checkout v4, setup-python v5 | Automated test execution on pull request and push pipelines |
| Admin Operations | Admin CLI (scripts/admin_review.py) | Python script | Lists pending reservations, validates paused threads, resumes graph with admin decision |

# 3. High-Level Architecture

```text
+-----------------+       +-----------------------------+
| User            | ----> | Streamlit UI (app/main.py) |
+-----------------+       +-------------+---------------+
                                      |
                                      v
                         +------------+-------------------------+
                         | LangGraph Chatbot Graph              |
                         | (SqliteSaver checkpointer enabled)   |
                         +-----+--------+--------+-------+------+
                               |        |        |       |
                               |        |        |       +------------------------------+
                               |        |        |                                      |
                               v        v        v                                      v
                      +--------+--+ +---+-----+ +-------------------+          +-------+--------+
                      | Weaviate  | | parking | | Presidio Layer 1  |          | OpenAI GPT-4o  |
                      | static RAG| | .db     | | guardrails         |          | LLM calls      |
                      +-----------+ | SQLite  | +-------------------+          +----------------+
                                    | dynamic |
                                    +----+----+
                                         |
                                         v
                                  +------+-------+
                                  | checkpoints.db|
                                  | SQLite state  |
                                  +------+-------+
                                         |
                      interrupt()        |        Command(resume=decision)
+------------------+   pause            v                   +------------------+
| submit_to_admin  | -----------------> Admin CLI <-------- | scripts/admin_   |
| node             |                    review.py           | review.py         |
+------------------+                                        +------------------+
         |
         v
+------------------+      +------------------+      +------------------------+
| MCP Client       | ---> | MCP Server       | ---> | data/reservations.txt  |
| app/mcp_client   |      | mcp_server       |      | confirmed reservations |
+------------------+      +------------------+      +------------------------+
```

Each user turn enters the LangGraph runtime through Streamlit. The graph performs guardrail checks, intent routing, and either informational answer generation or reservation handling. During approval waiting points, interrupt() pauses execution and serializes full state through SqliteSaver into checkpoints.db; the admin CLI later resumes the same thread with Command(resume=...), allowing status-finalization nodes to complete the flow. On approval, record_reservation_node updates DB status and then calls the MCP client, which starts the MCP server subprocess and writes the normalized reservation line to the reservations file.

# 4. LangGraph Decision Graph

Graph image artifact (generated):
- docs/graph_diagram.png

If image rendering is not available, use the Mermaid fallback artifact:
- docs/graph_diagram.mmd

Complete node structure:

```text
START
  |
  v
guardrail_node
  |-- [guardrail_triggered=true] ------------------------------> response_node
  |
  '-- [guardrail_triggered=false] -----------------------------> intent_node
                                                                  |-- [intent=info] ---------> rag_node ----------> response_node
                                                                  |-- [intent=unknown] ------> unknown_node ------> response_node
                                                                  '-- [intent=reservation] --> reservation_node
                                                                                                |-- [awaiting_admin=false] ---> response_node
                                                                                                '-- [awaiting_admin=true] ----> submit_to_admin_node
                                                                                                                                [interrupt() occurs here]
                                                                                                                                          |
                                                                                                                                          '-- [admin_decision=approved] --> record_reservation_node
                                                                                                                                                                             [MCP called here]
                                                                                                                                                                             --> response_node
                                                                                                                                          '-- [admin_decision=rejected] --> notify_rejection_node
                                                                                                                                                                             --> response_node
                                                                                                                                                                                  |
                                                                                                                                                                                  v
                                                                                                                                                                                 END
```

# 5. RAG Pipeline Design

## 5.1 Data split
- Static semantic content is stored in Weaviate.
- Dynamic operational values and reservation records are stored in SQLite.

Static data is indexed once at ingestion time and queried through semantic nearest-neighbor search. Dynamic data is queried at runtime with exact SQL lookups per parking_id. This split avoids embedding volatile values that would require constant re-indexing and allows precise numerical comparisons for prices and availability.

## 5.2 Chunking strategy
Static parking documents are chunked by semantic section per parking space:
- general
- location
- booking
- contact

This avoids mixed-topic chunks and improves retrieval precision.
Each parking space produces exactly 4 chunks: general, location, booking, contact. Total corpus size is 20 chunks across 5 spaces.
Chunking by semantic section rather than fixed character windows ensures each retrieved chunk is topically coherent and avoids splitting related facts across chunk boundaries.

## 5.3 Retrieval strategy
- Retriever uses semantic nearest-neighbor search in Weaviate.
- Runtime injects dynamic SQL context for live values such as price, hours, and availability.
- Embedding model: text-embedding-3-small, 1536 dimensions.
- Why text-embedding-3-small: cost-efficient with strong semantic performance for domain-specific retrieval.
- Distance measure: cosine similarity. Cosine similarity measures the angle between vectors rather than their magnitude, making it robust to variations in text length. This is standard for semantic search over dense embeddings.
- k=20 retrieval window: set high relative to corpus size (20 chunks total) to ensure all relevant chunks for broad queries like list all parking spaces are included in context.
- Dynamic SQL enrichment: after Weaviate retrieval, prices, availability, and working hours are fetched from SQLite and injected into the LLM context alongside retrieved chunks.

## 5.4 Prompt strategy
Prompt set includes:
- System policy prompt.
- Intent classification prompt.
- RAG answer prompt with grounding constraints.
- Reservation extraction prompt for next missing field only.

Prompt rationale by component:
- SYSTEM_PROMPT: defines assistant role, hard rules against revealing internal data, and scope boundaries. Placed in system role to establish policy before any user content is processed.
- INTENT_CLASSIFICATION_PROMPT: zero-shot classification into exactly three categories (info, reservation, unknown). Three categories were chosen to minimize ambiguity and keep routing logic deterministic.
- RAG_PROMPT: injects retrieved context with explicit grounding instruction to never fabricate facts. Includes dynamic parking ID list to prevent omission of spaces from list responses.
- RESERVATION_EXTRACTION_PROMPT: requests one missing field at a time to keep user interaction natural and avoid overwhelming the user with a multi-field form.

# 6. Guardrails Design

Two-layer guardrails:
1. Presidio plus deterministic regex checks.
2. LLM guardrail chain for nuanced misuse patterns.

Why this order:
- Presidio is fast, local, deterministic, and cost-free.
- LLM guardrail catches semantic attacks that lexical checks may miss.

Guardrail continuity rule:
- During active reservation collection, normal free-form guardrails are bypassed to avoid interrupting legitimate required field capture.

## 6.1 Presidio layer

PII entity types detected:
- PERSON
- PHONE_NUMBER
- EMAIL_ADDRESS
- CREDIT_CARD
- IBAN_CODE
- US_SSN
- UK_NHS
- NRP

Each type was included because it represents a category of personally identifiable information that must not be extracted from the system or used to probe user records.

Forbidden pattern categories:
- Internal system keywords: weaviate, vector, embedding, database, schema, sql, table, column
- Security probing: api_key, secret, password, token
- Prompt injection attempts: ignore.*instructions, forget.*instructions, you are now, act as if, pretend you are
- Data extraction attempts: show.*all.*reservations, list.*all.*users, dump.*data, show.*database

## 6.2 LLM guardrail layer

The LLM guardrail prompt instructs the model to detect subtle semantic attacks that lexical patterns may miss, such as indirect role-reassignment attempts, fictional framing used to bypass restrictions, and multi-step social engineering sequences. It returns only safe or blocked to minimize output variability.

# 7. Reservation Flow

End-to-end reservation flow:
1. User expresses intent to reserve.
2. Intent is classified as reservation.
3. Missing fields are collected one by one via reservation_node.
4. User confirms with yes.
5. submit_to_admin saves reservation to DB with status pending.
6. interrupt() pauses graph and user sees pending message.
7. Admin runs scripts/admin_review.py.
8. Admin approves or rejects.
9. Command(resume=decision) resumes graph on same thread.
10. Approved branch: record_reservation_node updates DB to confirmed, calls MCP server, writes to reservations.txt.
10. Rejected branch: notify_rejection_node updates DB to rejected.
11. User sends any message; UI detects resumed state and delivers approval or rejection notification.

# 8. Human-in-the-Loop Design

Why HITL is used:
- Reservation confirmation is an operational side effect and requires controlled approval.

LangGraph semantics used:
- interrupt(payload) pauses execution and persists full graph state in checkpoints.db.
- Admin CLI resumes same thread with Command(resume=approved|rejected).
- Node idempotency in submit_to_admin prevents duplicate pending inserts on resume replay.

Thread ID strategy:
- Each Streamlit session generates a uuid4 thread_id at startup.
- Thread ID is stored in st.session_state and passed as configurable.thread_id in every graph invocation config.
- The same thread_id is written to the Reservation DB record so the admin CLI can look up which checkpointed thread to resume.

Checkpointer design:
- SqliteSaver persists the full ChatState snapshot at every node transition, not just at interrupt points.
- This means any graph execution can be inspected or replayed from any node transition.
- checkpoints.db grows proportionally to the number of conversation turns across all sessions.

Admin CLI flow:
- CLI calls get_pending_reservations() to find DB rows with status=pending and a non-null thread_id.
- For each candidate, it calls chatbot_graph.get_state(thread_config) to verify the thread is genuinely interrupted before offering the resume action.
- This guard prevents resume attempts on threads that are not actually paused, which would cause a LangGraph error.

# 9. MCP Server Design

The MCP boundary isolates file writing from graph business logic and enforces authentication and payload validation at the tool layer.

MCP invocation from graph:
- record_reservation_node reads canonical reservation data from DB after approval.
- It invokes write_reservation_via_mcp from app/mcp_client/tools.py.
- The MCP client launches mcp_server.server via stdio transport and calls write_parking_reservation.

Failure resilience design:
- DB status update to confirmed is authoritative and occurs before MCP write.
- If MCP write fails, reservation remains confirmed and user receives a non-blocking note about file recording failure.
- This prevents user-visible approval rollback due to downstream file tool issues.

# 10. Evaluation Framework

## 10.1 RAGAS Metrics

RAGAS metrics used:
- faithfulness
- answer_relevancy
- context_recall
- context_precision

Metric details:
- faithfulness: measures whether the generated answer is supported by the retrieved context. Prevents hallucination by checking factual grounding. Chosen because parking information accuracy is critical.
- answer_relevancy: measures whether the answer addresses the question asked. Catches responses that are factually grounded but topically evasive.
- context_recall: measures whether all relevant information needed to answer the question was retrieved. Maps to Recall@K in traditional IR evaluation.
- context_precision: measures whether retrieved chunks are relevant to the question. Maps to Precision in traditional IR evaluation. High precision means low retrieval noise.

## 10.2 Evaluation Dataset

The dataset contains 10 hand-crafted question and ground-truth entries covering all five parking spaces with at least two entries per parking_id.

Covered question families:
- pricing (hourly, daily, monthly)
- working hours per day type
- metro proximity and location details
- features and amenities
- booking process behavior

## 10.3 Score Thresholds

| Metric | Threshold | Status | Rationale |
| --- | ---: | --- | --- |
| faithfulness | 0.70 | failed if below | Hallucination is critical |
| answer_relevancy | 0.70 | warning if below | Important but tunable |
| context_recall | 0.60 | warning if below | Moderate-high expected |
| context_precision | 0.60 | warning if below | Some noise tolerable |

## 10.4 Performance Metrics

Latency is captured at two levels:
- per-question latency_ms measured around answer generation
- total_latency_ms for full benchmark execution

Average latency is computed across all questions to provide a stable signal trackable over repeated benchmark runs.

Evaluation artifacts:
- reports/eval_report_{timestamp}.md
- reports/latest_results.json

Evaluation purpose:
- Track grounding and retrieval quality over iterative changes.

# 11. Testing Strategy

Current suite coverage: 106 tests collected.

Test files and purpose:
- tests/test_chatbot.py: routing and intent parsing behavior (4 tests).
- tests/test_guardrails.py: PII checks, forbidden patterns, block reason logic, anonymization (10 tests).
- tests/test_rag.py: chunking correctness (2 tests).
- tests/test_rag_db.py: SQL dynamic context lookup (2 tests).
- tests/test_evaluation.py: dataset loading, report generation, save behavior, sample latency capture (6 tests).
- tests/test_functional.py: multi-node functional flows, reservation collection, guardrail outcomes (15 tests).
- tests/test_checkpointer.py: SqliteSaver wiring, thread isolation, persistence across invocations (6 tests).
- tests/test_hitl.py: interrupt payload, approval/rejection status nodes, routing, interrupted-state detection (11 tests).
- tests/test_admin_cli.py: pending list filtering, status updates, interrupted-state guard, CLI empty-list handling (7 tests).
- tests/test_mcp_server.py: auth, validation, file writes, concurrency safety, tool listing (12 tests).
- tests/test_mcp_client.py: server process params, call success/failure, timestamp behavior, LangChain tool wrapper (9 tests).
- tests/test_notification.py: user notification after admin decision and fallback behavior (7 tests).
- tests/test_integration.py: full end-to-end pipeline scenarios (9 tests).
- tests/test_load_test.py: load-test stats and report logic (6 tests).

Integration test approach:
- Uses real graph orchestration with isolated test SQLite databases.
- Mocks external services (LLM, Weaviate, MCP subprocess I/O) for deterministic CI behavior.
- Verifies full approved and rejected reservation lifecycles and state persistence.

Load test approach:
- Measures component latencies for RAG, MCP write, graph invocation, and concurrent graph invocations.
- Generates a markdown performance report under reports/.

Mocking strategy summary:
- OpenAI and Weaviate calls are mocked in unit and integration tests where external variability is not desired.
- MCP calls are mocked for reliability-focused logic tests, while server/client unit tests validate payload contract and behavior.
- SQLite is used directly for persistence-path correctness.

Testing layers:
- Unit tests validate focused logic in isolation.
- Functional tests validate composed graph behavior across multi-node flows.
- Integration tests validate full pipeline with real graph orchestration and isolated test databases.
- Evaluation smoke tests validate pipeline interfaces and artifact generation contract.

# 12. Performance

Load testing summary (current baseline):
- RAG Query: approximately 4.6s average (GPT-4o + Weaviate retrieval path).
- MCP Write: approximately 928ms average (subprocess startup overhead dominates).
- Graph Invocation: approximately 8.2s average (roughly 3 LLM calls in path).
- Concurrent execution: good parallelism observed with ThreadPoolExecutor.

Production optimization opportunities:
- Keep MCP server running persistently to avoid per-request process startup costs.
- Cache repeated LLM responses for recurring informational queries.
- Use GPT-4o-mini for guardrail and intent classification paths while reserving GPT-4o for answer generation.

# 13. Setup and Running

Step-by-step full setup:
1. Clone repository.
2. Create and activate virtual environment.
3. Install dependencies: pip install -r requirements.txt
4. Download spaCy model: python -m spacy download en_core_web_lg
5. Copy .env.example to .env and fill values.
6. Start Weaviate: docker compose up -d weaviate
7. Initialize DB: python -c "from app.database.sql_client import init_db; init_db()"
8. Ingest static data: python -m app.rag.ingestion
9. Start chatbot: streamlit run app/main.py
10. In separate terminal when needed: python scripts/admin_review.py

Docker-based run:
1. Create .env from .env.example.
2. Start all services: docker compose up --build.
3. Open http://localhost:8501.

Run evaluation:
- python scripts/run_evaluation.py
- Performs Weaviate reachability check, runs RAGAS, writes reports
- Generated artifacts: reports/eval_report_{timestamp}.md and reports/latest_results.json

Run load tests:
- python scripts/load_test.py
- Reports saved to reports/load_test_report_{timestamp}.md

Operational notes:
- The chatbot process and admin CLI must share the same checkpoints.db and parking.db files.
- Re-run ingestion after static data changes to keep vector retrieval aligned with parking_info.json.
- reservations.txt is auto-created on first confirmed approval; see data/reservations.example.txt for format reference.

Environment variable reference:

| Variable | Description | Default | Required |
| --- | --- | --- | --- |
| OPENAI_API_KEY | OpenAI API key for GPT and embeddings | none | Yes |
| WEAVIATE_URL | Weaviate endpoint URL | http://localhost:8080 | Yes |
| WEAVIATE_API_KEY | Weaviate cloud API key (local can stay empty) | empty | Optional |
| DATABASE_URL | SQLAlchemy database URL | sqlite:///./parking.db | Yes |
| CHECKPOINT_DB_PATH | LangGraph checkpoint sqlite file path | checkpoints.db | Yes |
| MCP_API_KEY | Shared secret for MCP auth | none | Yes |
| RESERVATIONS_FILE_PATH | Output file for confirmed reservations | data/reservations.txt | Yes |
| MCP_ALLOWED_ORIGINS | Allowed origins for MCP deployments over HTTP transports | * | Optional |
| ENVIRONMENT | Runtime environment name | development | Optional |
| LOG_LEVEL | Logging verbosity | INFO | Optional |
| LOAD_TEST_N_REQUESTS | Number of load-test iterations per component | 10 | Optional |
| PYTHONPATH | Module import root for local execution | . | Yes |
