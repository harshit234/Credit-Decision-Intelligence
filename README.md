# 🏦 Halcyon Credit — Agentic Underwriting Copilot

> **Team Jamun** · Harshit · Himkar · Aditya · Ayush
> Futurense × IIT Gandhinagar · PG Diploma in AI-ML & Agentic AI Engineering · Cohort 1

---

## 📌 Problem Statement

Halcyon Credit is a digital consumer lender offering personal loans — many to applicants with thin or non-traditional credit histories. Applications arrive faster than human underwriters can process them manually.

A human underwriter reads the application file, cross-checks the numbers, weighs the risk, applies policy, and writes a decision with reasons. This work is careful, slow, and the queue keeps growing.

**The core challenge:** Halcyon does not want a black-box score. A lender must:
- Explain every decline to the applicant
- Defend it to a regulator
- Treat all applicants fairly

---

## 💡 Our Solution

An **Agentic Underwriting Copilot** — a multi-agent AI system that:

1. **Verifies** income and pulls bureau credit data via specialized agents
2. **Assesses** repayment risk using a trained LightGBM model (ROC-AUC 0.7166)
3. **Checks** every decision against 7 institutional lending policies via ChromaDB
4. **Synthesizes** an evidence-backed APPROVE / DECLINE / REFER recommendation using GPT-4.1
5. **Evaluates** the decision for faithfulness before output (LLM-as-Judge, threshold 0.75)
6. **Records** a fully auditable decision with written reasons and full pipeline trace

---

## 🏗️ System Architecture

```
Loan Application UI (http://localhost:3000)
         ↓
   FastAPI API Gateway (http://localhost:8000)
         ↓
   LangGraph Orchestrator (8-node StateGraph)
    ├──[parallel]──► Income Verification Agent  →  Income DB Tool
    ├──[parallel]──► Credit Bureau Agent        →  Credit Bureau Tool
    └──[parallel]──► Policy Compliance Agent    →  ChromaDB (7 policies)
                          ↓
                    Risk Scoring Agent           →  LightGBM (41 features)
                          ↓
                    Decision Synthesizer         →  GPT-4.1 via OpenRouter
                          ↓
                    Evaluation Agent             →  GPT-4.1 (LLM-as-Judge)
                       ┌──┴──┐
                  [PASS]    [RETRY → max 2]
                       ↓
                  Record Writer                 →  SQLite Audit Trail
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Orchestration | LangGraph |
| LLM Gateway | Direct HTTP → OpenRouter (GPT-4.1) |
| API Layer | FastAPI (v1 + v2 endpoints) |
| Vector Store / RAG | ChromaDB + all-MiniLM-L6-v2 |
| LLM Models | GPT-4.1 (synthesis + evaluation) |
| Risk Model | LightGBM (trained on 1.3M LendingClub rows) |
| Evaluation | LLM-as-Judge + RAGAS + Deterministic regression gate |
| Frontend | Vanilla HTML/JS/CSS (3-stage SPA) |
| Database | SQLite (audit trail) |

---

## 📊 Key Metrics

| Metric | Value |
|--------|-------|
| Decision Accuracy (golden set) | **100%** (10/10) |
| LLM Faithfulness | **1.000** |
| LLM Relevancy | **1.000** |
| Cost per Application | **~$0.005** |
| End-to-End Latency | **~12 seconds** |
| ML Model ROC-AUC | **0.7166** |
| ML Model PR-AUC | **0.3854** |
| Training Rows | **1,302,850** |
| Model Features | **41** |
| Policy Rules | **7** |
| Integration Tests | **13 / 13 pass** |
| Adversarial Tests | **8 / 8 pass** |
| CI Regression Gate | **10 / 10 pass** |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- OpenRouter API key (get one at [openrouter.ai](https://openrouter.ai))

### 1. Clone & Install
```bash
git clone https://github.com/harshit234/Credit-Decision-Intelligence.git
cd Credit-Decision-Intelligence
pip install -r requirements_stage3.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

### 3. Run Backend API
```bash
uvicorn api.main:app --reload --port 8000
```

### 4. Run Frontend Dashboard
```bash
python -m http.server 3000 --directory ui
```

### 5. Run Tests
```bash
# CI regression gate (deterministic, no LLM cost)
python tests/run_regression.py

# Full integration tests
python tests/integration/test_pipeline_integration.py

# Adversarial / red-team tests
python tests/adversarial/test_adversarial.py

# Full RAGAS golden set eval (uses API key)
python eval/ragas_runner.py

# Fairness segment analysis
python eval/fairness_test.py
```

---

## 📁 Repository Structure

```
Credit-Decision-Intelligence/
├── agents/                        # 7 LangGraph agent nodes
│   ├── income_agent.py            # Income verification
│   ├── credit_agent.py            # Credit bureau pull
│   ├── policy_agent.py            # Policy compliance (deterministic + RAG)
│   ├── risk_agent.py              # LightGBM risk scoring
│   ├── synthesizer_agent.py       # GPT-4.1 decision synthesis
│   ├── evaluation_agent.py        # LLM-as-Judge faithfulness gate
│   └── record_writer.py           # SQLite audit writer
├── api/
│   ├── main.py                    # FastAPI entry point
│   ├── routes.py                  # v1 endpoints
│   ├── routes_v2.py               # v2 endpoints (metrics, health, records)
│   └── schemas.py                 # Pydantic input/output schemas
├── data/                          # EDA and ML feasibility scripts
├── docs/
│   └── RUNBOOK.md                 # Operational runbook (SLOs, incidents, rollback)
├── eval/
│   ├── golden_set/test_cases.json # 10 curated golden test cases
│   ├── ragas_runner.py            # Full RAGAS evaluation runner
│   ├── fairness_test.py           # 4-cohort fairness disparity analysis
│   ├── dspy_optimizer.py          # Prompt optimization before/after comparison
│   └── final_evaluation_report.md # Sprint 3 full evaluation report
├── gateway/
│   ├── router.py                  # Direct OpenRouter HTTP gateway
│   └── prompts.py                 # Centralised LLM prompt library (v2)
├── graph/
│   └── pipeline.py                # LangGraph StateGraph (8 nodes)
├── models/
│   └── lgbm_halcyon_v2_lc.txt    # Production LightGBM model
├── state/
│   └── application_state.py       # ApplicationState TypedDict + all Pydantic models
├── tests/
│   ├── integration/               # 13 deterministic integration tests (Aditya)
│   └── adversarial/               # 8 red-team tests (Aditya)
├── tools/
│   ├── income_db_tool.py          # Mock income bureau
│   ├── credit_bureau_tool.py      # Mock credit bureau
│   ├── policy_retrieval_tool.py   # ChromaDB semantic search wrapper
│   ├── credit_score_bridge.py     # FICO → 41-feature vector bridge (Harshit)
│   └── decision_record_tool.py    # SQLite persistence
├── ui/
│   └── index.html                 # Full-stack SPA dashboard (Ayush)
├── .env.example                   # Environment config template
├── run_live_test.py               # End-to-end smoke test
├── PRD.md                         # Product Requirements Document
├── TRD.md                         # Technical Requirements Document
├── dataset_card.md                # LendingClub dataset documentation
└── Project_Report.md              # Comprehensive project report
```

---

## 🔌 API Reference

### POST `/applications`
Submit a loan application for underwriting.

**Request body** (all fields required except `applicant_id`):
```json
{
  "name": "Rohan Mehta",
  "loan_amount": 250000,
  "loan_purpose": "home_improvement",
  "loan_term_months": 36,
  "annual_income": 900000,
  "employment_type": "salaried",
  "months_employed": 48,
  "existing_debts": 5000,
  "credit_score": 720,
  "delinquencies_2yr": 0,
  "open_accounts": 8,
  "revolving_utilisation": 22.0,
  "credit_age_months": 84,
  "public_records": 0,
  "inquiries_6mo": 1,
  "home_ownership": "MORTGAGE"
}
```

**Response:**
```json
{
  "audit_id": "uuid",
  "recommendation": "APPROVE",
  "reasons": ["Credit score of 720 [credit_report.credit_score=720] is strong..."],
  "risk_score": 0.1821,
  "risk_band": "Low",
  "faithfulness_score": 1.0,
  "relevancy_score": 1.0,
  "eval_pass": true,
  "cost_usd_total": 0.00464,
  "top_risk_features": [...],
  "applicant": {...},
  "trace": [...]
}
```

### Other Endpoints
| Endpoint | Description |
|----------|-------------|
| `GET /health` | Basic health check |
| `GET /v2/health/detailed` | Subsystem health (DB, Model, ChromaDB, API key) |
| `GET /v2/model/info` | LightGBM version and performance metrics |
| `GET /v2/metrics/operational` | Aggregated dashboard KPIs |
| `GET /v2/records/{audit_id}` | Retrieve full decision record |
| `GET /v2/records` | List recent decisions (paginated) |

---

## 👥 Team Jamun

| Name | Domain |
|------|--------|
| **Harshit Gautam** | ML Model Training, Risk Scoring, API Integration |
| **Himkar Vashistha** | Policy KB, Prompt Engineering, LLM Gateway, Documentation |
| **Aditya Arora** | State Schema, Data Pipelines, Integration & Adversarial Testing |
| **Ayush Kumar** | LLM Agents, Evaluation, UI Dashboard |

---

## 📜 License

This project is built for academic and portfolio purposes as part of the Futurense AI Clinic Capstone Program.

---

*Halcyon Credit is a fictional lender created for this capstone engagement.*
