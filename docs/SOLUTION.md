# 1. Project Overview

The system is a domain-specific parking chatbot that combines Retrieval-Augmented Generation (RAG), a stateful reservation workflow, human-in-the-loop (HITL) approval, and layered guardrails for safe operation. It answers parking information questions from indexed static knowledge, enriches responses with live dynamic values from SQL, supports multi-turn reservation creation with persistence and confirmation, and routes reservation completion through an admin approval process before final status is committed.

The system manages these five parking spaces:
- CityPark Central (parking_001)
- AirportPark Express (parking_002)
- ShoppingMall Park (parking_003)
- Stadium Park (parking_004)
- OldTown Garage (parking_005)

Current end-to-end capabilities:
- Information retrieval over static parking knowledge with dynamic SQL enrichment for operational values.
- Multi-turn reservation field collection with structured extraction and explicit user confirmation.
- Human approval workflow using graph pause/resume semantics and an admin CLI review loop.
- MCP server writes confirmed reservations to file through a tool-based boundary.
- Layered guardrails for PII and abuse patterns with deterministic and model-based checks.
- Quality evaluation with RAGAS metrics, benchmark reporting, and latency tracking.

Roadmap status:

| Stage | Scope | Status |
| --- | --- | --- |
| Stage 1 | RAG chatbot, guardrails, evaluation foundation | Complete |
| Stage 2 | HITL reservation approval, checkpointer persistence, admin review flow | Complete |
| Stage 3 | MCP server integration for reservation file writing and tool transport | Complete |
| Stage 4 | Evaluation optimization and iterative quality tuning | Planned |

# 2. Technology Stack

| Component | Technology | Version | Purpose |
| --- | --- | --- | --- |
| LLM | OpenAI GPT-4o via LangChain OpenAI | openai>=1.30.0, langchain-openai>=0.1.0 | Answer generation, intent classification, reservation field extraction, LLM-layer guardrails |
| Orchestration | LangGraph + LangChain | langgraph>=0.1.0, langchain>=0.2.0 | Stateful decision graph, chain composition, interrupt/resume control flow |
| Checkpointing | LangGraph SqliteSaver | langgraph-checkpoint-sqlite | Persists full graph state for paused HITL threads and cross-process resume |
| Vector Database | Weaviate | weaviate-client>=4.5.0 | Semantic retrieval over static parking knowledge |
| SQL Database | SQLite via SQLAlchemy ORM | sqlalchemy>=2.0.0 | Dynamic operational data, reservation persistence, status transitions |
| Embeddings | OpenAI text-embedding-3-small | openai>=1.30.0 | Dense vector generation for retrieval |
| Guardrails | Microsoft Presidio + regex + LLM guardrail chain | presidio-analyzer>=2.2.0, presidio-anonymizer>=2.2.0 | PII/system-abuse filtering with layered security checks |
| MCP Server | Python mcp library | mcp>=1.0.0 | Exposes file-writing tools to LangGraph agent |
| Admin Operations | Admin CLI (`scripts/admin_review.py`) | Python script in repository | Lists pending reservations, validates paused threads, resumes graph with admin decision |
| Evaluation | RAGAS + Hugging Face Datasets | ragas>=0.1.0 | Faithfulness and retrieval quality measurement |
| UI | Streamlit | streamlit>=1.35.0 | Interactive chat interface and system status display |
| Testing | Pytest | pytest>=8.0.0, pytest-asyncio>=0.23.0 | Unit and functional validation including HITL and evaluation smoke tests |
| CI/CD | GitHub Actions | actions/checkout v4, setup-python v5 workflow setup | Automated test execution in pull request and push pipelines |

# 3. High-Level Architecture

```text
+------------------+      +----------------------+      +------------------------------+
| User             | ---> | Streamlit UI         | ---> | LangGraph Chatbot Graph      |
| Browser Client   |      | app/main.py          |      | (state machine + checkpointer)|
+------------------+      +----------------------+      +-----------+------------------+
                                                                    / | | \
                                                                   /  | |  \
                                                                  v   v v   v
                                                    +----------------+ +----------------+
                                                    | Presidio Layer | | OpenAI GPT-4o  |
                                                    | PII + regex    | | LLM reasoning  |
                                                    +----------------+ +----------------+
                                                                    \      /
                                                                     \    /
                                                                      v  v
                                                    +-----------------------------------+
                                                    | Runtime Data Integrations         |
                                                    +-----------------------------------+
                                                    | Weaviate: static semantic content |
                                                    | SQLite: dynamic config + bookings |
                                                    +-----------------------------------+

                                  +------------------------------+
                                  | SqliteSaver Checkpointer     |
                                  | checkpoints.db               |
                                  +---------------+--------------+
                                                  ^
                                                  |
                              interrupt() from submit_to_admin
                                                  |
                                                  v
                                  +------------------------------+
                                  | Admin CLI                    |
                                  | scripts/admin_review.py      |
                                  +---------------+--------------+
                                                  |
                                      Command(resume=decision)
                                                  |
                                                  v
                                  +------------------------------+
                                  | LangGraph resumes same thread|
                                  +------------------------------+

record_reservation_node
         |
         v
+------------------------------+      +------------------------------+      +---------------------+
| MCP Client                   | ---> | MCP Server                   | ---> | reservations.txt    |
| app/mcp_client/tools.py      |      | python -m mcp_server.server  |      | file append entries |
+------------------------------+      +------------------------------+      +---------------------+
```

Each user turn enters the LangGraph runtime through Streamlit. The graph performs guardrail checks, intent routing, and either informational answer generation or reservation handling. During approval waiting points, `interrupt()` pauses execution and serializes full state through `SqliteSaver` into `checkpoints.db`; the admin CLI later resumes the same thread with `Command(resume=...)`, allowing status-finalization nodes to complete the flow. On approval, `record_reservation_node` updates DB status and then calls the MCP client, which starts the MCP server subprocess and writes the normalized reservation line to the reservations file.

# 4. LangGraph Decision Graph

```text
START
  |
  v
guardrail_node
  |-- [guardrail_triggered = true] -----------------------------------------------> response_node
  |
  '-- [guardrail_triggered = false] ----------------------------------------------> intent_node
                                                                                     |-- [intent = info] --------> rag_node ---------------------------> response_node
                                                                                     |-- [intent = unknown] -----> unknown_node ----------------------> response_node
                                                                                     '-- [intent = reservation] -> reservation_node
                                                                                                                  |-- [awaiting_admin = false] ------> response_node
                                                                                                                  '-- [awaiting_admin = true] -------> submit_to_admin_node [interrupt]
                                                                                                                                                         |
                                                                                                                                                         '-- [admin_decision = approved] --> record_reservation_node [DB confirm + MCP write to reservations.txt] --> response_node
                                                                                                                                                         '-- [admin_decision = rejected] --> notify_rejection_node --> response_node
                                                                                                                                                                                             |
                                                                                                                                                                                             v
                                                                                                                                                                                            END
```

Node descriptions:
- START: Entry point that receives the current conversation state and latest user input.
- guardrail_node: Applies layered safety checks and marks the turn as blocked or safe.
- intent_node: Determines whether safe input is informational, reservation-related, or unknown.
- rag_node: Produces grounded informational answers using retrieval plus dynamic SQL enrichment.
- reservation_node: Collects reservation fields one step at a time, handles user confirmation, and marks the flow as awaiting admin when ready.
- submit_to_admin_node: Persists pending reservation data (idempotently) and pauses execution via `interrupt()` until an admin decision is provided.
- record_reservation_node: On admin approval, updates reservation status to confirmed, loads canonical reservation data from DB, calls MCP file-write tooling, and prepares final confirmation messaging.
- notify_rejection_node: On admin rejection, updates reservation status to rejected and prepares rejection messaging.
- unknown_node: Returns a fallback message that redirects the conversation to supported parking topics.
- response_node: Generates final user-facing output and appends turn messages to chat history.
- END: Terminates graph execution for the current turn.

Edge conditions:
- START -> guardrail_node: Always.
- guardrail_node -> response_node: When `guardrail_triggered` is true.
- guardrail_node -> intent_node: When `guardrail_triggered` is false.
- intent_node -> rag_node: When `intent` is `info`.
- intent_node -> reservation_node: When `intent` is `reservation`.
- intent_node -> unknown_node: When `intent` is `unknown` or classification is not usable.
- rag_node -> response_node: Always after RAG answer generation.
- unknown_node -> response_node: Always after fallback generation.
- reservation_node -> response_node: When reservation flow is still collecting data or user has not finalized confirmation.
- reservation_node -> submit_to_admin_node: When reservation fields are complete, user confirms, and `awaiting_admin` is true.
- submit_to_admin_node -> record_reservation_node: On resume with `admin_decision = approved`.
- submit_to_admin_node -> notify_rejection_node: On resume with `admin_decision = rejected`.
- record_reservation_node -> response_node: Always after approval status update.
- notify_rejection_node -> response_node: Always after rejection status update.
- response_node -> END: Always.

# 5. RAG Pipeline Design

## 5.1 Data Architecture

The design uses a split-data approach:
- Static data is stored in Weaviate for semantic retrieval of descriptive parking knowledge.
- Dynamic data is stored in SQLite for exact operational values such as prices, working hours, availability, and reservation records.

This split was chosen because semantic and operational workloads have different optimal storage patterns. Weaviate is best for meaning-based retrieval across rich text, while SQLite is best for exact, mutable fields that must remain authoritative and consistent.

Data placement:
- Weaviate stores sectioned textual chunks derived from parking descriptions, location details, booking instructions, and contact details.
- SQLite stores dynamic configuration tuples by parking_id and key (for example hourly price, sunday hours, available spaces) and persisted reservation rows.

## 5.2 Document Chunking Strategy

The source parking knowledge file is transformed into section-level chunks per parking space rather than arbitrary token windows. Each parking object is split into four semantic chunks:
- general
- location
- booking
- contact

Semantic section chunking improves retrieval precision by aligning each vector with a coherent intent region, reducing mixed-topic chunks that confuse nearest-neighbor search. The total static chunk count is 20, derived from 5 parking spaces multiplied by 4 chunk types.

## 5.3 Embedding Model

The embedding model is text-embedding-3-small. It was selected because it provides a practical cost-to-quality balance for medium-sized domain corpora while maintaining strong semantic recall. The vector dimensionality is 1536, which is well-suited for semantic similarity operations in vector databases.

## 5.4 Retrieval Strategy

The retriever uses vector similarity search in Weaviate over pre-computed embeddings. Cosine similarity is the standard distance family for semantic search because it emphasizes directional similarity in high-dimensional embedding space and is less sensitive to raw magnitude than Euclidean distance.

The retrieval depth is set to k=20 to maximize recall in a small-domain corpus where total chunk count is limited. A higher k helps prevent omission of relevant parking sections while downstream prompting constrains answer grounding.

In addition to retrieved static chunks, the runtime injects dynamic SQL context containing known parking identifiers and per-parking operational summaries. This enrichment ensures answers can include live prices, availability, and working hours that are not explicitly represented as static narrative text.

## 5.5 Prompt Design

The prompt suite is intentionally specialized by task:
- SYSTEM_PROMPT: Defines assistant scope and hard rules, including non-disclosure constraints and reservation-field policy.
- INTENT_CLASSIFICATION_PROMPT: Uses zero-shot constrained classification for exactly three categories to keep routing deterministic and auditable.
- RAG_PROMPT: Injects retrieved context and explicit grounding requirements, including completeness guidance to avoid selective omission.
- RESERVATION_PROMPT: Drives one-field-at-a-time collection to reduce extraction ambiguity and improve conversational control.

Design rationale:
- Category compression to three intents minimizes routing entropy and simplifies downstream node logic.
- Grounding instructions prioritize factual discipline and explicit uncertainty statements when context is insufficient.
- Stepwise reservation prompting lowers user error rates and reduces multi-field parsing failures.

# 6. Guardrails Design

## 6.1 Two-Layer Architecture

The guardrails are implemented as Presidio-first and LLM-second:
- Layer 1 is local Presidio plus deterministic regex checks.
- Layer 2 is a dedicated LLM guardrail chain that runs only if Layer 1 passes.

Presidio executes first because it is fast, deterministic, and incurs no external API cost. The LLM layer is reserved for nuanced unsafe patterns that deterministic detection can miss, such as subtle intent-manipulation phrasing that remains semantically malicious without obvious keywords.

During active reservation collection, guardrails are intentionally bypassed for continuity. This prevents false positives from interrupting the collection of personal fields that are required for a legitimate booking workflow.

The same continuity principle applies through the admin approval branch: once a reservation is in-progress and routed to `submit_to_admin_node`, the flow remains reservation-controlled until an approval decision is resumed and finalized by status nodes, rather than being re-routed through normal free-form intent and guardrail handling.

## 6.2 Presidio Layer

PII entities currently targeted include:
- PERSON
- PHONE_NUMBER
- EMAIL_ADDRESS
- CREDIT_CARD
- IBAN_CODE
- US_SSN
- UK_NHS
- NRP

Each entity matters because the assistant handles user-entered booking data and must prevent unintended leakage or unsafe transformation of sensitive details. Regex enforcement adds deterministic blocks for three pattern families:
- system probing: attempts to expose internals such as database structure, embeddings, or schema terms
- prompt injection: attempts to override policy through instruction manipulation phrases
- data extraction: attempts to bulk-extract reservation or user records

## 6.3 LLM Guardrail Layer

The LLM guardrail prompt asks for a strict safe or blocked decision and focuses on categories that can be semantically implicit rather than lexical. This layer catches sophisticated misuse patterns where wording avoids direct forbidden tokens but intent remains policy-violating.

# 7. Reservation Flow

The reservation flow is stateful and multi-turn:
1. The initial reservation request routes to reservation intent via intent classification.
2. `parking_id` is treated as the first required field and extracted from natural language if present in the user turn.
3. Remaining fields are collected one at a time: `name`, `surname`, `car_number`, `start_date`, `end_date`.
4. Field extraction uses GPT-4o in a constrained JSON-extraction step that targets only the next missing field.
5. Once all fields exist, the flow enters confirmation mode and asks for explicit `yes` or `no` confirmation.
6. On negative confirmation, reservation state is reset and the flow restarts.
7. On positive confirmation, reservation data is stored with `status = pending`, associated with `thread_id`, and routed to `submit_to_admin_node`.
8. `submit_to_admin_node` issues `interrupt(payload)`, which pauses graph execution and returns control to the caller while preserving full graph state in `checkpoints.db`.
9. An administrator reviews pending items in `scripts/admin_review.py`, selects approve (`a`) or reject (`r`), and resumes the same thread with `Command(resume=decision)`.
10. On resume, `record_reservation_node` sets status to `confirmed` for approvals, or `notify_rejection_node` sets status to `rejected` for rejections.
11. For approvals, `record_reservation_node` reads canonical reservation data by `thread_id` from SQL and validates required fields before any MCP call.
12. `write_reservation_via_mcp()` invokes the MCP server write tool, which validates input and appends the formatted line to `reservations.txt`.
13. If MCP write fails, confirmation still stands because DB status is already authoritative; the user receives a confirmation message with a non-critical file-recording note.
14. `response_node` produces the final user-facing outcome and updates message history consistently for both outcomes.

State continuity is governed by `is_reservation_in_progress`, which checks whether `reservation_data` exists and is not finalized. This flag forces routing to reservation handling, preserves user progress across turns, and ensures the flow remains deterministic through the approval handoff.

# 8. Human-in-the-Loop Design

## 8.1 Why HITL Approval Is Used

Parking reservations involve resource allocation and potential payment. A fully automated path that confirms every request without human oversight carries operational risk. The HITL pattern introduces a human gate before irreversible side effects, so reservation outcomes are finalized only after explicit admin review.

## 8.2 `interrupt()` Semantics in LangGraph

When LangGraph encounters `interrupt(payload)` inside `submit_to_admin_node`, it:
1. Freezes execution at that node and serializes full `ChatState` into the configured checkpointer.
2. Returns control with an interrupt payload (`{"__interrupt__": [...], ...}`) instead of continuing downstream nodes.
3. Waits for `invoke(Command(resume=value), config=config)` on the same `thread_id`.
4. Re-runs the interrupted node from the top, with `interrupt()` returning the supplied resume value instead of pausing again.

`submit_to_admin_node` is implemented with an idempotency guard (`get_reservation_by_thread_id`) to avoid duplicate inserts when the node re-runs on resume.

## 8.3 Checkpointer Persistence Model

`SqliteSaver` persists one checkpoint per node execution, including full conversation history, reservation fields, routing flags, and admin decision context. Because both Streamlit and the admin CLI use the same `checkpoints.db`, resume can happen in a separate process from where interruption occurred.

SQLite checkpointer infrastructure keeps local development and CI simple because it requires no separate service. For production deployments, `PostgresSaver` from `langgraph-checkpoint-postgres` can be used as an API-compatible replacement.

Checkpoint volume is expected: a reservation that passes through guardrail, intent, reservation, submit, resume, status update, and response typically generates around 8-10 checkpoint rows for that thread.

Old checkpoints can be pruned periodically in production operations to control storage growth.

## 8.4 Thread ID Strategy

Each Streamlit session creates a `uuid4` thread identifier (`st.session_state.thread_id`). Every graph invocation uses:

```python
{"configurable": {"thread_id": thread_id}}
```

The same `thread_id` is stored on the reservation row in SQLite. This linkage allows the admin CLI to reconstruct the exact graph config required to resume the paused conversation state safely.

Because Streamlit re-runs the script on each interaction, persisted checkpoints on the same `thread_id` are what preserve graph continuity across reruns.

## 8.5 Admin CLI Design and Resume Flow

The admin CLI (`scripts/admin_review.py`) performs a strict resume workflow:
1. `get_pending_reservations()` returns pending rows with non-null `thread_id`.
2. Admin selects a reservation index and decision (`a` approve / `r` reject).
3. CLI checks paused-state validity using `chatbot_graph.get_state(config)` and requires non-empty `state.next`.
4. If the thread is genuinely interrupted, CLI calls `chatbot_graph.invoke(Command(resume=decision), config=config)`.
5. Graph resumes, executes decision branch nodes, updates DB status, and prepares final user response.

This state check prevents accidental resume calls against already completed threads.

# 9. MCP Server Design

## 9.1 Why MCP Protocol Is Used

The file write path is intentionally separated from graph business logic by using MCP tools instead of direct file I/O in node code. This protocol boundary improves encapsulation, allows strict authentication/validation at the server edge, and keeps the graph node focused on workflow state rather than storage mechanics.

## 9.2 Exposed MCP Tools

The MCP server exposes three tools:
- `write_parking_reservation`: Authenticated write of a confirmed reservation entry.
- `read_parking_reservations`: Authenticated read of current file entries.
- `get_reservations_file_stats`: Authenticated file metadata and line-count reporting.

## 9.3 Security and Concurrency Model

Security and input controls are enforced server-side:
- API-key style authentication using an `api_key` field (equivalent to an `X-API-Key` guard in API deployments).
- Timing-safe key verification via `hmac.compare_digest`.
- Input validation for names, car number, date formats, and date ordering.
- Thread-safe file appends with `threading.Lock` to prevent concurrent write interleaving.

## 9.4 Client Transport and Runtime Behavior

LangChain connects to the MCP server via stdio transport (`stdio_client` + `ClientSession`). The MCP subprocess command uses `sys.executable` so the same Python interpreter/environment is used as the caller, avoiding virtual-environment mismatch. The sync wrapper uses asyncio loop detection so calls are safe from both synchronous code paths and already-running async contexts.

## 9.5 Reservation File Entry Format

Each confirmed entry is stored as one line:

`Name Surname | Car Number | Start Date to End Date | Approval Time`

# 10. Evaluation Framework

## 9.1 RAGAS Metrics

- faithfulness: Measures whether generated answers are grounded in provided context and is the primary anti-hallucination signal.
- answer_relevancy: Measures whether the answer directly addresses the user’s question intent and scope.
- context_recall: Measures whether retrieval captured relevant evidence needed to answer, analogous to Recall at K behavior.
- context_precision: Measures whether retrieved evidence is mostly relevant rather than noisy, analogous to Precision at K behavior.

These metrics were selected because together they cover grounding quality, topical alignment, retrieval completeness, and retrieval cleanliness.

## 9.2 Evaluation Dataset

The evaluation dataset contains 10 hand-crafted question and ground-truth entries. Coverage spans all five parking spaces with at least two entries per parking_id.

Covered question families include:
- pricing (hourly, daily, monthly)
- working hours
- metro proximity and location details
- features and amenities
- booking process behavior

## 9.3 Score Thresholds

| Metric | Threshold | Status Levels | Rationale |
| --- | ---: | --- | --- |
| faithfulness | 0.70 | passed if >=0.70, failed otherwise | Hallucination risk is treated as critical, so this metric uses a hard fail boundary |
| answer_relevancy | 0.70 | passed if >=0.70, warning otherwise | Relevancy drops are important but can be improved incrementally without immediate hard block |
| context_recall | 0.60 | passed if >=0.60, warning otherwise | Retrieval completeness is expected to be moderate-high and tuned over time |
| context_precision | 0.60 | passed if >=0.60, warning otherwise | Some retrieval noise is tolerable early, with optimization planned in later stages |

## 9.4 Performance Metrics

Latency is captured at two levels:
- per-question latency_ms, measured around answer generation in build_eval_sample
- total_latency_ms for full benchmark execution in run_evaluation

Average latency is computed across all evaluated questions to provide a stable runtime signal that can be tracked over repeated benchmark runs.

# 11. Testing Strategy

Test coverage is split into unit-focused and functional graph-focused suites.

Current test inventory:
- tests/test_chatbot.py: 4 tests focused on routing and intent parsing behavior.
- tests/test_evaluation.py: 6 tests for dataset loading, report generation, save behavior, and sample latency capture.
- tests/test_functional.py: 15 end-to-end functional tests over graph pathways, reservation flow, and guardrail outcomes.
- tests/test_guardrails.py: 10 tests covering PII detection, forbidden patterns, block reasons, anonymization, and guardrail node behavior.
- tests/test_rag.py: 2 tests for chunking correctness.
- tests/test_rag_db.py: 2 tests for dynamic SQL aggregation and key lookup.
- tests/test_checkpointer.py: 6 tests for SqliteSaver wiring, thread isolation, persistence across invocations, and DB helper behavior.
- tests/test_hitl.py: 11 tests for submit interrupt payload, approval/rejection status nodes, routing, interrupted-state detection, and full approved/rejected HITL flows.
- tests/test_admin_cli.py: 7 tests for pending-reservation filtering, status update decisions, interrupted-state guard checks, CLI empty-list handling, and input validation.
- tests/test_mcp_server.py: 12 tests for API-key verification, input validation, reservation append semantics, file stats, concurrency safety, and tool listing.
- tests/test_mcp_client.py: 9 tests for server process parameters, call success/failure handling, timestamp behavior, LangChain tool wrapper metadata, and record-node MCP invocation behavior.

Total test count: 84 tests.

Mocking strategy:
- OpenAI and Weaviate dependencies are mocked in unit and functional tests to avoid network cost, flakiness, and nondeterministic model variance.
- Real SQLite is used selectively where persistence semantics must be validated, especially reservation save behavior and status transitions.
- HITL interrupt behavior is tested with controlled interruption/resume conditions: tests assert `interrupt()` is invoked at submit time, verify paused-state detection through `get_state().next`, and exercise both resume branches deterministically.
- MCP tests use an autouse `temp_reservations_file` fixture to force `RESERVATIONS_FILE_PATH` into per-test temporary directories and patch module-level path constants, ensuring tests never write to real runtime files.

Testing layers:
- Unit tests validate focused logic in isolation, such as classification parsing, report formatting, guardrail functions, DB helpers, and CLI decision handling.
- Functional tests validate composed graph behavior across multi-node flows, including reservation collection, pause/resume transitions, and final response generation.

Evaluation smoke testing:
- tests/test_evaluation.py serves as a smoke layer for the evaluation pipeline interfaces and artifact generation contract.

# 12. Setup and Running

Prerequisites:
- Python 3.11+ runtime and virtual environment.
- OpenAI API key configured in environment.
- Docker and Docker Compose available for local Weaviate/service orchestration.
- Weaviate running locally or reachable remotely.

Environment setup:
1. Create and activate a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Create `.env` from `.env.example` and configure required variables.
4. Download spaCy model `en_core_web_lg` for Presidio NLP support.
5. Ensure repository root is available on Python path (handled by `.env` defaults in this project).

Environment variables reference:

| Variable | Purpose | Example |
| --- | --- | --- |
| `OPENAI_API_KEY` | LLM and embedding API authentication | `sk-...` |
| `WEAVIATE_URL` | Weaviate endpoint | `http://localhost:8080` |
| `WEAVIATE_API_KEY` | Optional key for secured Weaviate deployments | `` |
| `DATABASE_URL` | SQLAlchemy database URL | `sqlite:///./parking.db` |
| `CHECKPOINT_DB_PATH` | LangGraph checkpoint SQLite file path | `checkpoints.db` |
| `MCP_API_KEY` | MCP server authentication key | `your_mcp_api_key_here` |
| `RESERVATIONS_FILE_PATH` | Target file for confirmed reservation entries | `data/reservations.txt` |

Full local runtime sequence:
1. Start Weaviate:
   - `docker compose up -d weaviate`
2. Initialize SQL database schema:
   - `python -c "from app.database.sql_client import init_db; init_db()"`
3. Ingest static parking data into Weaviate:
   - `python -m app.rag.ingestion`
4. Start chat UI:
   - `streamlit run app/main.py`
5. Run admin review console in a separate terminal when reservations are awaiting approval:
   - `python scripts/admin_review.py`

Docker-based application run:
1. Create `.env` from `.env.example`.
2. Start all services:
   - `docker compose up --build`
3. Open `http://localhost:8501`.

Run tests:
- Execute full tests with `pytest -q` or `pytest tests -v`.

Run evaluation:
- Execute `python scripts/run_evaluation.py`.
- The command performs Weaviate reachability checks, runs RAGAS, and writes report artifacts.

Generated evaluation artifacts:
- `reports/eval_report_{timestamp}.md`
- `reports/latest_results.json`

Operational notes:
- The chatbot process and admin CLI must share the same `checkpoints.db` and `parking.db` files.
- Re-run ingestion after static data changes to keep vector retrieval aligned with `data/static/parking_info.json`.
- `reservations.txt` is auto-created on first confirmed approval write; see `data/reservations.example.txt` for the entry format reference.
