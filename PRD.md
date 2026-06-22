# Product Requirements Document
## Halcyon Credit — Agentic Underwriting Copilot

**Team Jamun** · Harshit · Ayush · Aditya · Himkar  
**Program:** Futurense × IIT Gandhinagar · PG Diploma in AI-ML & Agentic AI Engineering · Cohort 1  
**Capstone Project 02** · Version 1.0 · June 2025

---

## 1. Problem Statement & Context

### 1.1 Business Problem

Halcyon Credit is a digital consumer lender offering personal loans, with a significant portion of its applicant pool consisting of individuals with thin or non-traditional credit histories. As application volumes grow, human underwriters face an increasingly unsustainable queue. Manual review — reading files, cross-checking income, weighing risk, applying policy, and writing reasoned decisions — is careful, slow, and does not scale linearly.

**The critical constraint:** Halcyon cannot adopt a black-box scoring system. Regulatory and ethical obligations require that every decline be explainable to the applicant and defensible before a regulator. Fairness across applicant segments is non-negotiable.

### 1.2 Opportunity

An agentic AI system can mirror the full human underwriting workflow — gathering data, checking policy, scoring risk, synthesizing a decision, and writing a reasoned output — at a fraction of the cost and time per application, while maintaining the traceability and auditability that compliance demands.

### 1.3 Scope

This PRD covers the **Agentic Underwriting Copilot**: a multi-agent AI pipeline that processes a structured loan application and produces an auditable, evidence-backed underwriting recommendation. Out of scope: direct applicant-facing communication, loan disbursement, and collections.

---

## 2. Goals, Success Metrics & Baselines

### 2.1 Primary Goals

- Produce underwriting recommendations that are accurate, explainable, and policy-compliant.
- Reduce average human review time per application.
- Ensure no protected-class bias is introduced or amplified by the system.
- Generate a complete, replayable audit trail for every decision.

### 2.2 Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Decision Quality | Agreement with ground-truth outcome on held-out loans | ≥ 85% alignment |
| Explanation Faithfulness | Written reasons match actual decision factors (RAGAS) | Faithfulness score ≥ 0.80 |
| Fairness Gap | Approval / error rate differential across applicant segments | < 5 percentage points |
| Policy Adherence | Every decision respects stated lending policy | 100% (gated by Evaluation Agent) |
| Cost per Application | Tokens + infrastructure cost per processed loan | To be baselined in Sprint 2 |
| Latency | End-to-end pipeline time per application | ≤ 30 seconds (p95) |

### 2.3 Baseline

A single-LLM baseline (one prompt, no agent orchestration) will be built in Sprint 4 alongside the full agentic system. All delta metrics will be reported honestly against this baseline.

---

## 3. Solution Architecture

### 3.1 High-Level Design

The system follows an **Orchestrator-Worker pattern** combined with an **Evaluator-Optimizer loop**, implemented using **LangGraph** for stateful multi-agent orchestration and **FastAPI** as the API gateway. All LLM calls route through a **LiteLLM gateway** (routing, fallback, retries).

```
Loan Application UI
        ↓
  API Gateway (FastAPI)
        ↓
  Orchestrator Agent  ←──── LiteLLM Gateway (routing · fallback · retries)
   ├──→ Income Verification Agent   ──→ Income DB Tool (mock → real)
   ├──→ Credit History Agent        ──→ Credit Bureau Tool (mock → real)
   ├──→ Policy Compliant Agent      ──→ Policy Retrieval Tool (ChromaDB / RAG)
   ├──→ Risk Scoring Agent          ──→ XGBoost / LightGBM model
   ├──→ Decision Synthesizer Agent
   ├──→ Evaluation Agent (LLM-as-Judge + RAGAS)  ←── retry loop (max N)
   └──→ Decision Record Writer
        ↓
  Output: Auditable Decision (JSON + DB)
```

### 3.2 Agent Responsibilities

| Agent | Responsibility | Tool / Model |
|-------|---------------|--------------|
| Orchestrator Agent | Coordinates pipeline; manages shared LangGraph state | LangGraph state machine |
| Income Verification Agent | Verifies applicant income from source DB | Income DB Tool |
| Credit History Agent | Fetches bureau records and credit profile | Credit Bureau Tool |
| Policy Compliant Agent | Retrieves and checks applicable lending policies | Policy Retrieval Tool (ChromaDB + RAG) |
| Risk Scoring Agent | Computes repayment risk score from structured features | XGBoost / LightGBM |
| Decision Synthesizer Agent | Combines all signals into a reasoned recommendation | LLM (strong model path) |
| Evaluation Agent | Judges decision quality and explanation faithfulness | LLM-as-Judge + RAGAS |
| Decision Record Writer | Persists full state trace as auditable JSON/DB record | Postgres / Decision DB |

### 3.3 Shared State Schema

A single typed `ApplicationState` object threads through every LangGraph node. Agents read their designated inputs and write only their assigned keys — enforced by Pydantic validators on every node wrapper.

**Core state fields:**

```
application_id: str
applicant_file: ApplicantFile
income_verified: IncomeResult | None
credit_report:   CreditResult | None
policy_findings: PolicyResult | None
risk_score:      RiskResult   | None
draft_decision:  Decision     | None
eval_result:     EvalResult   | None
retry_count:     int = 0
final_record:    DecisionRecord | None
trace:           list[NodeEvent]
errors:          list[AgentError]
```

State is **checkpointed after every node transition** by LangGraph persistence — enabling replay, resume, and complete audit history.

### 3.4 Execution Flow

Worker agents (Income, Credit, Policy) run **in parallel** as LangGraph parallel nodes; a synchronization barrier blocks until all three write their results. Risk Scoring fires only once all three upstream fields are present in state (state-gated, never speculative). If the Evaluation Agent's faithfulness score falls below the configured threshold, LangGraph routes back to the Decision Synthesizer with judge feedback appended to state — capped at N retries, after which human escalation is forced.

### 3.5 Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Orchestration | LangGraph |
| LLM Gateway | LiteLLM (routing, fallback chain, rate limits, retries) |
| API Layer | FastAPI (async, concurrency support) |
| Vector Store / RAG | ChromaDB |
| LLM Models | Gemini Flash (dev / cheap path); stronger model (eval / synthesis path) |
| Risk Model | XGBoost / LightGBM (trained on real lending data) |
| Evaluation | RAGAS, DSPy, LLM-as-Judge |
| Deployment | Free cloud tier (TBD) |
| CI/CD | GitHub Actions |

---

## 4. Functional Requirements

### 4.1 Core Pipeline

**FR-01** The system shall accept a structured loan application via a POST endpoint and return a decision with written reasons and an `audit_id` in a single synchronous response (200 OK).

**FR-02** The Orchestrator shall dispatch Income Verification, Credit History, and Policy Compliant agents concurrently and merge their results before invoking Risk Scoring.

**FR-03** The Risk Scoring Agent shall construct its feature vector strictly from typed upstream state fields — no free-text LLM output shall reach the ML model directly.

**FR-04** The Risk Scoring Agent output shall include: `risk_score` (0–1), `risk_band` (Low / Medium / High), and `top_features` (SHAP values for explainability).

**FR-05** The Decision Synthesizer shall produce a `draft_decision` that references evidence from `income_verified`, `credit_report`, `policy_findings`, and `risk_score` — every claim must be traceable to a source field.

**FR-06** The Evaluation Agent shall score `draft_decision` for faithfulness and quality. Scores below the configured threshold shall trigger a retry loop (max N=2 retries); on exhaustion, the orchestrator shall escalate to a human reviewer queue.

**FR-07** The Decision Record Writer shall persist the full state trace (not just the final answer) to the Decision DB, enabling replay and regulatory review.

### 4.2 Policy & Fairness

**FR-08** The Policy Compliant Agent shall retrieve relevant policy clauses via RAG (ChromaDB) and attach `source_refs` (clause IDs) to its `policy_findings` output. Every decision record shall reference the policy clauses that governed it.

**FR-09** The system shall include a fairness testing suite that measures approval and error rate gaps across defined applicant segments. Gaps exceeding 5 percentage points shall be flagged for human review before promotion.

**FR-10** Every decline decision shall include written reasons sufficient to satisfy adverse action notice requirements.

### 4.3 Observability & Operations

**FR-11** Every tool call and state transition shall be logged with timestamp, agent identity, input keys read, and output keys written.

**FR-12** The system shall expose cost-per-application telemetry (token counts and estimated USD cost) from the LiteLLM gateway.

**FR-13** LiteLLM shall implement a fallback chain: if the primary model is unavailable or exceeds latency SLO, routing shall automatically promote to the next model in the chain without dropping the request.

---

## 5. Non-Functional Requirements

| Category | Requirement |
|----------|-------------|
| Latency | End-to-end p95 ≤ 30 seconds per application under nominal load |
| Availability | 99% uptime target on free cloud tier; degraded-mode fallback to human queue |
| Cost | Cost-per-application ceiling defined and enforced via LiteLLM rate limits; alarms if exceeded |
| Auditability | Every decision replayable from persisted state trace; no state mutation after record is written |
| Explainability | Every output includes written reasons and SHAP-attributed top risk features |
| Security | No applicant PII in prompt logs; no secrets committed to repo; all credentials via environment variables |
| Fairness | Fairness gap report generated on every golden-set evaluation run; gates promotion to production |

---

## 6. Risk Register (v1)

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Hallucinated income or credit facts | Medium | High | `source_refs` required on all worker outputs; Evaluation Agent checks every factual claim against state |
| Unexplainable decline | Low | High | SHAP values mandatory in Risk Scoring output; faithfulness gate in Evaluation Agent |
| Protected-class bias amplification | Medium | High | Fairness testing suite on every eval run; segment gap alert threshold |
| Policy retrieval failure (RAG miss) | Medium | Medium | Fallback to conservative policy defaults; alert + human escalation |
| Over-automation (removing human judgment) | Low | High | Human escalation path mandatory on retry exhaustion; copilot framing (recommendation, not binding decision) |
| Data exposure via LLM prompt logs | Low | High | PII scrubbing before LiteLLM call; log audit in Sprint 3 |
| Cost overrun | Medium | Medium | Token budget per application; LiteLLM cost alarms; cheap-path routing for non-critical agents |

---

## 7. Sprint Plan Summary

| Sprint | Theme | Key Deliverables |
|--------|-------|-----------------|
| Sprint 0 | Discover & Define | Dataset profiling, EDA, user personas, PRD v1, Risk Register v1, evaluation plan |
| Sprint 1 | Design & De-risk | Architecture spec, agent contracts (JSON schemas), LiteLLM routing design, synthetic data pipeline v1, golden set, Policy Retrieval spike |
| Sprint 2 | Build Core | FastAPI entry point, all agents + tools (mock), LiteLLM gateway live, first end-to-end test, cost baseline |
| Sprint 3 | Harden & Optimize | Evaluation Agent, Decision Record Writer, async concurrency, RAGAS scores, DSPy prompt optimization, fairness testing, adversarial red-team pass |
| Sprint 4 | Verify & Operate | End-to-end benchmark vs. single-LLM baseline, deployment, runbook, SLOs, 20-minute recorded presentation, viva |

---

## 8. Repository Structure

```
halcyon-underwriting-copilot/
├── agents/         # LangGraph agent definitions
├── tools/          # Income DB, Credit Bureau, Policy Retrieval, Record Writer
├── gateway/        # LiteLLM gateway + routing policy
├── api/            # FastAPI endpoints
├── ui/             # Minimal loan application UI
├── eval/           # RAGAS, DSPy, golden set, regression suite
├── data/           # Dataset cards, synthetic data, EDA notebooks
├── docs/           # PRD, architecture spec, risk register, runbook
└── tests/          # Unit + integration tests
```

---

## 9. Open Questions

- **Dataset selection:** Which real-world lending dataset will the risk model train on? License and class-balance gaps TBD (Sprint 0 exit criterion).
- **Retry cap N:** Default is 2 retries before human escalation. Needs calibration against golden-set faithfulness score distribution.
- **Deployment target:** Free cloud tier options (Render, Railway, Fly.io) to be evaluated in Sprint 3 against latency and cold-start constraints.
- **Fairness segment definitions:** Protected-class proxies must be defined without using protected attributes as direct model inputs; methodology to be documented in Risk Register v2.

---

*Halcyon Credit is a fictional persona created for the Futurense AI Clinic Capstone Program. This document is for academic and portfolio purposes.*
