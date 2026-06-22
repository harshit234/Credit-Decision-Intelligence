# Technical Requirements Document (TRD)
## Halcyon Credit — Agentic Underwriting Copilot

**Team Jamun** · Harshit · Ayush · Aditya · Himkar  
**Program:** Futurense × IIT Gandhinagar · PG Diploma in AI-ML & Agentic AI Engineering · Cohort 1  
**Capstone Project 02** · Version 1.0 · June 2025  
**Status:** Draft — Sprint 1

---

## 1. Document Purpose & Scope

This Technical Requirements Document (TRD) specifies the engineering contracts, interface definitions, data schemas, infrastructure requirements, and implementation constraints for the Halcyon Credit Agentic Underwriting Copilot. It is the authoritative reference for developers building any component of the system. The companion PRD defines *what* the system must do; this TRD defines *how* it must be built.

**In scope:** LangGraph orchestration, agent contracts, tool specifications, LiteLLM gateway, FastAPI layer, ChromaDB RAG, risk model integration, evaluation pipeline, persistence layer, CI/CD, and observability.  
**Out of scope:** Frontend beyond the minimal loan application UI, payment/disbursement systems, applicant identity verification infrastructure.

---

## 2. System Architecture

### 2.1 Architectural Patterns

The system implements two nested patterns:

**Orchestrator-Worker** — A central Orchestrator Agent coordinates all worker agents (Income Verification, Credit History, Policy Compliant). Workers are dispatched in parallel and write results into a single shared `ApplicationState`. No worker communicates directly with another worker.

**Evaluator-Optimizer** — After the Decision Synthesizer produces a draft decision, the Evaluation Agent scores it. If faithfulness falls below threshold, LangGraph routes back to the synthesizer with judge feedback appended to state. This loop is capped at `N=2` retries; on exhaustion, the orchestrator writes an escalation flag and routes to the human review queue.

### 2.2 Component Topology

```
┌──────────────────────────────────────────────────────────────────────┐
│  Client Layer                                                        │
│  Loan Application UI (minimal HTML/React form)                       │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ HTTPS POST /applications
┌───────────────────────────────▼──────────────────────────────────────┐
│  API Layer — FastAPI (async)                                         │
│  • POST /applications    • GET /applications/{id}                    │
│  • GET /health           • GET /metrics                              │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ invoke LangGraph graph
┌───────────────────────────────▼──────────────────────────────────────┐
│  Orchestrator Agent (LangGraph state machine)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐             │
│  │ Income      │  │ Credit      │  │ Policy Compliant  │  ← parallel │
│  │ Verification│  │ History     │  │ Agent             │    fan-out  │
│  │ Agent       │  │ Agent       │  │                   │             │
│  └──────┬──────┘  └──────┬──────┘  └────────┬─────────┘             │
│         │                │                  │                        │
│  Income DB Tool   Credit Bureau Tool   Policy Retrieval Tool         │
│  (mock → real)    (mock → real)        (ChromaDB + RAG)             │
│                                                                      │
│         └────────────────┴──────────────────┘                       │
│                      merge_state (await all)                         │
│                            │                                         │
│                  ┌─────────▼──────────┐                             │
│                  │ Risk Scoring Agent │                              │
│                  │ XGBoost/LightGBM   │                              │
│                  └─────────┬──────────┘                             │
│                            │                                         │
│                  ┌─────────▼──────────────┐                         │
│                  │ Decision Synthesizer   │◄──── retry (max N=2)    │
│                  │ Agent                  │                          │
│                  └─────────┬──────────────┘                         │
│                            │                                         │
│                  ┌─────────▼──────────┐                             │
│                  │ Evaluation Agent   │──── pass ──►  Decision      │
│                  │ (LLM-as-Judge)     │              Record Writer   │
│                  └────────────────────┘                              │
└──────────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────▼────────────┐
                    │   LiteLLM Gateway      │
                    │   Routing · Fallback   │
                    │   Retries · Cost log   │
                    └────────────────────────┘
                                │
                    ┌───────────▼────────────┐
                    │   Persistence Layer    │
                    │   Postgres / SQLite    │
                    │   Decision DB          │
                    └────────────────────────┘
```

### 2.3 LangGraph Graph Definition

The LangGraph graph is a directed graph with conditional edges. Key structural rules:

- Worker nodes (Income, Credit, Policy) are grouped in a `parallel` fan-out; the synchronization barrier (`merge_state`) is the single downstream edge.
- `score_risk` is a conditional entry node — it only executes when `income_verified`, `credit_report`, and `policy_findings` are all non-None in state.
- The edge from `evaluate` back to `synthesize` is a conditional retry edge; it fires when `eval_result.faithfulness < FAITHFULNESS_THRESHOLD` and `retry_count < MAX_RETRIES`.
- The edge from `evaluate` to `write_record` fires when `eval_result.faithfulness >= FAITHFULNESS_THRESHOLD` or `retry_count >= MAX_RETRIES`.

```python
# Pseudocode — graph construction
graph = StateGraph(ApplicationState)
graph.add_node("init_state",   init_state_node)
graph.add_node("verify_income", income_agent_node)
graph.add_node("fetch_credit",  credit_agent_node)
graph.add_node("check_policy",  policy_agent_node)
graph.add_node("merge_state",   merge_node)
graph.add_node("score_risk",    risk_scoring_node)
graph.add_node("synthesize",    synthesizer_node)
graph.add_node("evaluate",      evaluation_node)
graph.add_node("write_record",  record_writer_node)

graph.add_edge("init_state", ["verify_income", "fetch_credit", "check_policy"])  # parallel
graph.add_edge(["verify_income", "fetch_credit", "check_policy"], "merge_state")
graph.add_edge("merge_state", "score_risk")
graph.add_edge("score_risk",  "synthesize")
graph.add_edge("synthesize",  "evaluate")
graph.add_conditional_edges("evaluate", retry_or_persist_router)
```

---

## 3. Shared State Schema

### 3.1 ApplicationState (TypedDict)

```python
from typing import TypedDict, Optional
from pydantic import BaseModel

class ApplicationState(TypedDict):
    application_id:  str
    applicant_file:  ApplicantFile          # immutable after init
    income_verified: Optional[IncomeResult]
    credit_report:   Optional[CreditResult]
    policy_findings: Optional[PolicyResult]
    risk_score:      Optional[RiskResult]
    draft_decision:  Optional[Decision]
    eval_result:     Optional[EvalResult]
    retry_count:     int                    # default 0
    final_record:    Optional[DecisionRecord]
    trace:           list[NodeEvent]
    errors:          list[AgentError]
```

State is checkpointed by LangGraph after every node transition using the configured checkpointer (SQLite in dev, Postgres in prod). **No agent may mutate a state key it does not own.** This is enforced at the LangGraph node wrapper level via Pydantic validators.

### 3.2 Sub-Schema Definitions

```python
class ApplicantFile(BaseModel):
    applicant_id:    str
    name:            str
    annual_income:   float
    loan_amount:     float
    loan_purpose:    str
    employment_type: str                    # salaried | self-employed | gig
    existing_debts:  float
    months_employed: int
    prior_applications: list[PriorApp]

class IncomeResult(BaseModel):
    verified_income: float
    confidence:      float                  # 0.0–1.0
    source_refs:     list[str]              # DB row IDs used
    verified_at:     datetime

class CreditResult(BaseModel):
    credit_score:    int
    delinquencies:   int
    credit_age_months: int
    open_accounts:   int
    utilization_pct: float
    thin_file:       bool
    source_refs:     list[str]              # bureau field IDs

class PolicyResult(BaseModel):
    applicable_clauses: list[PolicyClause]
    hard_stops:         list[str]           # clause IDs that mandate decline
    flags:              list[str]           # advisory flags
    source_refs:        list[str]           # ChromaDB chunk IDs

class RiskResult(BaseModel):
    risk_score:   float                     # 0.0–1.0 (higher = riskier)
    risk_band:    str                       # Low | Medium | High
    top_features: list[SHAPFeature]         # SHAP attribution, top 5
    model_version: str

class Decision(BaseModel):
    recommendation:  str                    # APPROVE | DECLINE | REFER
    reasons:         list[str]              # human-readable, source-traced
    conditions:      list[str]              # if APPROVE with conditions
    draft_version:   int

class EvalResult(BaseModel):
    faithfulness:    float                  # 0.0–1.0
    relevancy:       float
    unsupported_claims: list[str]           # claims not grounded in state
    pass_flag:       bool

class DecisionRecord(BaseModel):
    application_id:  str
    final_decision:  Decision
    eval_result:     EvalResult
    risk_result:     RiskResult
    policy_refs:     list[str]
    full_trace:      list[NodeEvent]
    audit_id:        str                    # UUID, returned to client
    created_at:      datetime
```

---

## 4. Agent Technical Contracts

### 4.1 Income Verification Agent

| Property | Value |
|----------|-------|
| Node name | `verify_income` |
| Reads from state | `applicant_file` |
| Writes to state | `income_verified` |
| Tool | Income DB Tool |
| LLM required | No (deterministic lookup in dev; LLM-assisted extraction in prod) |
| Failure behaviour | Writes `AgentError` to `errors`; downstream agents receive `income_verified=None`; orchestrator triggers human escalation if critical |

**Income DB Tool spec:**
```
Function:  lookup_income(applicant_id: str) -> IncomeRecord
Transport: HTTP GET /income/{applicant_id}   (mock: in-memory dict)
Response:  { applicant_id, reported_income, verified_income, source, fetched_at }
Timeout:   3s
Retry:     2x with 500ms backoff
```

### 4.2 Credit History Agent

| Property | Value |
|----------|-------|
| Node name | `fetch_credit` |
| Reads from state | `applicant_file` |
| Writes to state | `credit_report` |
| Tool | Credit Bureau Tool |
| LLM required | No |
| Failure behaviour | Writes `AgentError`; thin-file flag set to True if bureau returns no record |

**Credit Bureau Tool spec:**
```
Function:  fetch_bureau(applicant_id: str) -> BureauRecord
Transport: HTTP GET /bureau/{applicant_id}   (mock: synthetic data generator)
Response:  { credit_score, delinquencies, credit_age_months, open_accounts,
             utilization_pct, thin_file, bureau_ref_id }
Timeout:   5s
Retry:     2x with 1s backoff
```

### 4.3 Policy Compliant Agent

| Property | Value |
|----------|-------|
| Node name | `check_policy` |
| Reads from state | `applicant_file` |
| Writes to state | `policy_findings` |
| Tool | Policy Retrieval Tool (ChromaDB + RAG) |
| LLM required | Yes (clause relevance ranking and hard-stop extraction) |
| Model path | Cheap model (Gemini Flash) |

**Policy Retrieval Tool spec:**
```
Function:  retrieve_policy(query: str, top_k: int = 5) -> list[PolicyChunk]
Backend:   ChromaDB vector store (sentence-transformers embeddings)
Chunk size: 512 tokens, 64-token overlap
Metadata:  { clause_id, section, effective_date, jurisdiction }
Fallback:  If ChromaDB unavailable → return hardcoded conservative defaults
           + alert logged to monitoring
```

The agent LLM prompt must instruct the model to output only clause IDs, hard-stop flags, and advisory flags — no free-text synthesis at this stage. Output is validated against `PolicyResult` schema before writing to state.

### 4.4 Risk Scoring Agent

| Property | Value |
|----------|-------|
| Node name | `score_risk` |
| Reads from state | `income_verified`, `credit_report`, `policy_findings` |
| Writes to state | `risk_score` |
| Tool | ML model (XGBoost / LightGBM) |
| LLM required | No — feature vector assembled from typed state fields only |
| Guard | Node only executes if all three upstream fields are non-None |

**Feature vector (v1):**

```python
features = {
    "verified_income":        state["income_verified"].verified_income,
    "income_confidence":      state["income_verified"].confidence,
    "loan_to_income_ratio":   applicant_file.loan_amount / verified_income,
    "credit_score":           state["credit_report"].credit_score,
    "delinquencies":          state["credit_report"].delinquencies,
    "utilization_pct":        state["credit_report"].utilization_pct,
    "credit_age_months":      state["credit_report"].credit_age_months,
    "thin_file":              int(state["credit_report"].thin_file),
    "existing_debt_ratio":    applicant_file.existing_debts / verified_income,
    "months_employed":        applicant_file.months_employed,
    "hard_stop_count":        len(state["policy_findings"].hard_stops),
}
```

Output: `risk_score` (float 0–1), `risk_band`, SHAP top-5 feature attributions, `model_version`. No LLM text ever reaches the model input vector.

**Model training requirements:**
- Train on real lending dataset (selected in Sprint 0)
- Evaluate on: AUC-ROC, precision/recall at operating threshold, calibration curve
- Fairness evaluation: demographic parity and equalized odds across applicant segments
- Artefact: serialised model + feature schema version locked together

### 4.5 Decision Synthesizer Agent

| Property | Value |
|----------|-------|
| Node name | `synthesize` |
| Reads from state | `risk_score`, `credit_report`, `policy_findings`, `income_verified`, `eval_result` (on retry) |
| Writes to state | `draft_decision`, increments `draft_version` |
| LLM required | Yes — strong model path |

**Prompt contract (abridged):**
- Input context: structured JSON of all upstream state fields
- Required output: `{ recommendation, reasons: list[str], conditions: list[str] }`
- Each reason string must cite at least one source field (income_verified, credit_report, policy clause ID, or risk band)
- On retry: judge feedback from `eval_result.unsupported_claims` is appended to the user turn
- Output validated against `Decision` schema; malformed output triggers agent error

**DSPy optimization target (Sprint 3):** Optimise the synthesizer prompt signature for faithfulness score improvement. Before/after RAGAS comparison to be reported.

### 4.6 Evaluation Agent

| Property | Value |
|----------|-------|
| Node name | `evaluate` |
| Reads from state | `draft_decision`, `credit_report`, `policy_findings`, `income_verified`, `risk_score` |
| Writes to state | `eval_result`, increments `retry_count` |
| LLM required | Yes — strong model as judge |

**Evaluation rubric (LLM-as-Judge):**

The judge is provided the `draft_decision` and the full set of source state fields. It must output:

```json
{
  "faithfulness": 0.0–1.0,
  "relevancy": 0.0–1.0,
  "unsupported_claims": ["claim text not grounded in source data"],
  "pass_flag": true | false
}
```

`pass_flag = true` when `faithfulness >= FAITHFULNESS_THRESHOLD` (default 0.75).

**RAGAS integration (Sprint 3):** RAGAS faithfulness, answer relevancy, context precision, and context recall scores computed on the golden set and logged per evaluation run.

**Retry router logic:**
```python
def retry_or_persist_router(state: ApplicationState) -> str:
    if state["eval_result"].pass_flag:
        return "write_record"
    if state["retry_count"] >= MAX_RETRIES:
        state["errors"].append(AgentError(type="escalation", ...))
        return "write_record"   # writes with escalation flag
    return "synthesize"         # retry loop
```

### 4.7 Decision Record Writer

| Property | Value |
|----------|-------|
| Node name | `write_record` |
| Reads from state | `draft_decision`, `eval_result`, `risk_score`, `policy_findings`, `trace`, `errors` |
| Writes to state | `final_record` |
| LLM required | No |

Persists the full `ApplicationState` trace (not just the final decision) as a `DecisionRecord` to Postgres (prod) or SQLite (dev). Generates a UUID `audit_id`. The record is immutable after write — no update operations permitted on existing records. Soft-delete only.

---

## 5. LiteLLM Gateway

### 5.1 Routing Policy

```yaml
model_routing:
  cheap_path:                         # Income, Credit, Policy agents
    primary:   gemini/gemini-flash
    fallback:  openai/gpt-4o-mini
    max_tokens: 512
    timeout_s:  10

  strong_path:                        # Synthesizer, Evaluation agents
    primary:   gemini/gemini-pro
    fallback:  openai/gpt-4o
    max_tokens: 2048
    timeout_s:  30
```

### 5.2 Retry & Fallback Chain

```
Primary model call
    → Timeout / 5xx → retry once (same model, 500ms delay)
    → Still failing  → promote to fallback model
    → Fallback fails → write AgentError + route to human escalation
```

### 5.3 Cost Logging

Every LiteLLM completion call logs: `{ model, prompt_tokens, completion_tokens, cost_usd, latency_ms, agent_name, application_id }`. Aggregated per-application cost is written to the DecisionRecord. Alert fires if `cost_usd_per_application > COST_CEILING` (configurable, default $0.10).

### 5.4 Semantic Caching

Policy Retrieval Tool responses are cached by query embedding similarity (cosine > 0.95) with a TTL of 24 hours. Implemented via LiteLLM's built-in semantic cache (Redis-backed in prod, in-memory in dev). Cache hits are logged; cost is $0 for cached responses.

---

## 6. API Layer — FastAPI

### 6.1 Endpoints

```
POST   /applications
       Body:   ApplicationInput (JSON)
       Returns: 200 { decision, reasons, audit_id, risk_band, cost_usd }
                202 { audit_id, status: "escalated" }  (human escalation)
                422 Validation error
                500 Internal error with error_id

GET    /applications/{audit_id}
       Returns: Full DecisionRecord (for audit / review UI)

GET    /health
       Returns: { status: "ok", version, uptime_s }

GET    /metrics
       Returns: Prometheus-format metrics (requests, latency, cost, error rate)
```

### 6.2 ApplicationInput Schema

```python
class ApplicationInput(BaseModel):
    applicant_id:       str
    name:               str
    annual_income:      float = Field(gt=0)
    loan_amount:        float = Field(gt=0)
    loan_purpose:       str
    employment_type:    Literal["salaried", "self-employed", "gig"]
    existing_debts:     float = Field(ge=0)
    months_employed:    int   = Field(ge=0)
    prior_applications: list[PriorApp] = []
```

### 6.3 Async Execution

The FastAPI endpoint invokes the LangGraph graph asynchronously via `await graph.ainvoke(state)`. The server is configured with `uvicorn` workers and `asyncio` event loop. Concurrency target: 10 simultaneous applications without queue degradation (validated in Sprint 3 load test).

---

## 7. Data & Storage

### 7.1 ChromaDB (Policy Vector Store)

- **Collection:** `halcyon_policy_v1`
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- **Chunk strategy:** 512 tokens, 64-token overlap, paragraph-respecting splitter
- **Metadata per chunk:** `clause_id`, `section`, `effective_date`, `jurisdiction`, `is_hard_stop`
- **Persistence:** local disk in dev; managed ChromaDB cloud in prod

### 7.2 Decision Database (Postgres / SQLite)

```sql
CREATE TABLE decision_records (
    audit_id          UUID PRIMARY KEY,
    application_id    VARCHAR(128) NOT NULL,
    recommendation    VARCHAR(16)  NOT NULL,   -- APPROVE | DECLINE | REFER | ESCALATED
    risk_score        FLOAT,
    risk_band         VARCHAR(16),
    faithfulness      FLOAT,
    retry_count       INT,
    cost_usd          FLOAT,
    full_state_trace  JSONB,                   -- complete ApplicationState at write time
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    escalated         BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_application_id ON decision_records (application_id);
CREATE INDEX idx_created_at     ON decision_records (created_at DESC);
```

### 7.3 Synthetic Data Pipeline

Sprint 1 delivers a synthetic data generator producing:
- Standard applicants (varied income, credit score, loan purpose)
- Thin-file profiles (no bureau record, short credit history)
- Adversarial cases (inconsistent stated vs. verified income, circular loan purpose)
- Fraud patterns (multiple simultaneous applications, velocity flags)

All synthetic records include a ground-truth label (`approve` / `decline` / `refer`) for golden-set evaluation.

---

## 8. Evaluation & Quality Pipeline

### 8.1 Golden Set

A curated set of ≥ 100 synthetic application cases with ground-truth decisions. Every merge to `main` runs the full agent pipeline against the golden set and asserts:
- Decision Quality ≥ 85% agreement with ground truth
- RAGAS Faithfulness ≥ 0.80
- Fairness gap < 5 percentage points across segments
- Zero policy hard-stop violations

Failures block merge (enforced by GitHub Actions gate).

### 8.2 RAGAS Metrics (Sprint 3)

| Metric | Description | Target |
|--------|-------------|--------|
| Faithfulness | Are decision reasons supported by source context? | ≥ 0.80 |
| Answer Relevancy | Are reasons relevant to the decision made? | ≥ 0.75 |
| Context Precision | Are retrieved policy chunks relevant? | ≥ 0.70 |
| Context Recall | Are all relevant policy clauses retrieved? | ≥ 0.70 |

### 8.3 DSPy Optimization (Sprint 3)

Two prompt signatures to be optimized via DSPy BootstrapFewShot:
- `SynthesizerSignature`: applicant context → structured decision with cited reasons
- `EvaluatorSignature`: decision + context → faithfulness score + unsupported claims

Before/after RAGAS scores reported in the Sprint 3 evaluation report.

### 8.4 Fairness Testing

Approval rate and error rate measured across: employment type, income band, loan purpose category, and thin-file status. A fairness gap report is generated on every golden-set run. If any segment gap exceeds 5 percentage points, a GitHub Actions warning annotation is emitted and the issue is flagged for Sprint 3 remediation.

---

## 9. Infrastructure & Deployment

### 9.1 Repository Structure

```
halcyon-underwriting-copilot/
├── agents/
│   ├── orchestrator.py
│   ├── income_agent.py
│   ├── credit_agent.py
│   ├── policy_agent.py
│   ├── risk_agent.py
│   ├── synthesizer_agent.py
│   ├── evaluation_agent.py
│   └── record_writer.py
├── tools/
│   ├── income_db_tool.py
│   ├── credit_bureau_tool.py
│   ├── policy_retrieval_tool.py
│   └── decision_record_tool.py
├── gateway/
│   ├── litellm_config.yaml
│   └── router.py
├── api/
│   ├── main.py
│   ├── schemas.py
│   └── routes.py
├── state/
│   └── application_state.py        # TypedDict + Pydantic models
├── eval/
│   ├── golden_set/
│   ├── ragas_runner.py
│   ├── dspy_optimizer.py
│   ├── fairness_report.py
│   └── regression_gate.py
├── data/
│   ├── synthetic_generator.py
│   └── dataset_card.md
├── docs/
│   ├── PRD.md
│   ├── TRD.md                      # this document
│   ├── risk_register.md
│   └── runbook.md
└── tests/
    ├── unit/
    └── integration/
```

### 9.2 Environment Configuration

All secrets via environment variables — zero secrets in source code. Required vars:

```
LITELLM_API_KEY_GEMINI
LITELLM_API_KEY_OPENAI        # fallback
CHROMA_PERSIST_PATH
DATABASE_URL                  # postgres://... or sqlite:///...
FAITHFULNESS_THRESHOLD        # default 0.75
MAX_RETRIES                   # default 2
COST_CEILING_USD              # default 0.10
LOG_LEVEL                     # INFO | DEBUG
```

### 9.3 CI/CD (GitHub Actions)

```yaml
# Triggers: push to main, PR to main
jobs:
  lint:        ruff + mypy type checks
  unit_tests:  pytest tests/unit/ (mocked tools, mocked LLM)
  integration: pytest tests/integration/ (real LangGraph, mock tools)
  golden_gate: python eval/regression_gate.py  (blocks merge on failure)
  cost_check:  assert baseline cost per application within ceiling
```

No secrets committed. `.env` in `.gitignore`. CI uses GitHub Actions secrets for all API keys.

### 9.4 Observability

- **Structured logging:** JSON logs via `structlog`; every log line includes `application_id`, `agent_name`, `node`, `trace_id`
- **Metrics:** Prometheus-format `/metrics` endpoint; key metrics: `applications_total`, `latency_seconds` (histogram), `cost_usd_total`, `retry_count_total`, `escalation_total`, `eval_faithfulness` (gauge)
- **Alerting:** Cost-per-application alarm if > `COST_CEILING_USD`; escalation rate alarm if > 5% of applications in a 1-hour window

---

## 10. Security & Compliance

| Requirement | Implementation |
|-------------|---------------|
| PII in LLM prompts | Applicant name stripped from all LLM prompt text in prod; `applicant_id` used as reference |
| Secrets management | All credentials via environment variables; GitHub Actions secrets for CI |
| Audit trail | Full `ApplicationState` trace persisted immutably; `audit_id` returned to client |
| State immutability | DecisionRecord is append-only; no UPDATE on existing records |
| Agent write isolation | Pydantic validators on LangGraph node wrappers reject writes to unowned state keys |
| Dependency scanning | `pip-audit` and `npm audit` run in CI |

---

## 11. SLOs & Operating Constraints

| SLO | Target | Measurement |
|-----|--------|-------------|
| End-to-end latency (p95) | ≤ 30 seconds | `/metrics` histogram |
| Availability | ≥ 99% | Uptime monitor |
| Cost per application | ≤ $0.10 | LiteLLM cost log |
| Escalation rate | ≤ 5% | `escalation_total / applications_total` |
| Golden-set decision quality | ≥ 85% | Regression gate |
| RAGAS faithfulness | ≥ 0.80 | RAGAS runner |

---

## 12. Open Technical Decisions

| Decision | Options | Owner | Sprint |
|----------|---------|-------|--------|
| Risk model dataset | LendingClub, HMEQ, Kaggle lending sets | Harshit | Sprint 0 |
| Postgres vs. SQLite for prod | Depends on cloud tier constraints | Aditya | Sprint 2 |
| ChromaDB cloud vs. self-hosted | Cost and latency tradeoff | Ayush | Sprint 2 |
| Retry cap N tuning | Calibrate against golden-set faithfulness distribution | Himkar | Sprint 3 |
| DSPy optimizer (BootstrapFewShot vs. MIPROv2) | RAGAS score comparison | All | Sprint 3 |
| Fairness segment proxy variables | Must not use protected attributes directly | All | Sprint 1 |

---

*This document is a living specification. All schema versions, threshold values, and model routing configs are subject to revision as sprint work reveals new constraints. Breaking changes to agent contracts require TRD version bump and team sign-off.*

---

*Halcyon Credit is a fictional persona created for the Futurense AI Clinic Capstone Program. This document is for academic and portfolio purposes.*
