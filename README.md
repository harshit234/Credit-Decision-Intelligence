# 🏦 Halcyon Credit — Agentic Underwriting Copilot

> **Team Jamun** · Harshit · Ayush · Aditya · Himkar
> Futurense AI Clinic · Capstone Project 02

---

## 📌 Problem Statement

Halcyon Credit is a digital consumer lender offering personal loans — many to applicants with thin or non-traditional credit histories. Applications arrive faster than human underwriters can process them manually.

Each application file contains structured data: income, existing debts, bureau records, prior applications, and a stated loan purpose. A human underwriter reads the file, cross-checks the numbers, weighs the risk, applies policy, and writes a decision with reasons. This work is careful, slow, and the queue keeps growing.

**The core challenge:** Halcyon does not want a black-box score that simply outputs *approve* or *decline*. A lender must:
- Explain every decline to the applicant
- Defend it to a regulator
- Treat all applicants fairly

---

## 💡 Our Solution

We are building an **Agentic Underwriting Copilot** — a multi-agent AI system that:

1. **Assembles** the full applicant file from structured data sources
2. **Verifies** income and credit history via specialized agents and tools
3. **Assesses** repayment risk using a trained risk scoring model
4. **Checks** every decision against lending policy and fairness requirements
5. **Synthesizes** an evidence-backed recommendation
6. **Evaluates** the decision for quality and faithfulness before output
7. **Records** a fully auditable decision with written reasons

The output is a recommendation a human underwriter can trust and a regulator could review — not a black box, but a traceable reasoning chain.

---

## 🏗️ System Architecture

The system follows an **Orchestrator-Worker** pattern combined with an **Evaluator-Optimizer** layer, built using **LangGraph** for stateful multi-agent orchestration and **FastAPI** as the API gateway.

```
Loan Application UI
        ↓
  API Gateway / FastAPI
        ↓
  Orchestrator Agent
   ├──→ Income Verification Agent ──→ Income DB Tool
   ├──→ Credit History Agent ──────→ Credit Bureau Tool
   ├──→ Policy Complaint Agent ────→ Policy Retrieval Tool
   ├──→ Risk Scoring Agent (receives from Credit History + Policy agents)
   ├──→ Decision Synthesizer Agent
   ├──→ Evaluation Agent
   └──→ Decision Record Writer
        ↓
     Output
```

### Agent Responsibilities

| Agent | Responsibility | Tool Used |
|---|---|---|
| Orchestrator Agent | Coordinates the full pipeline, manages state | — |
| Income Verification Agent | Verifies applicant income from DB | Income DB Tool |
| Credit History Agent | Fetches bureau records and credit data | Credit Bureau Tool |
| Policy Complaint Agent | Retrieves and checks relevant lending policies | Policy Retrieval Tool |
| Risk Scoring Agent | Computes repayment risk score using ML model | — |
| Decision Synthesizer Agent | Combines all signals into a recommendation | — |
| Evaluation Agent | Judges decision quality and explanation faithfulness | — |
| Decision Record Writer | Writes the final auditable decision record | — |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Agent Orchestration | LangGraph |
| LLM Gateway | LiteLLM |
| API Layer | FastAPI |
| Vector Store / RAG | ChromaDB |
| LLM Models | Gemini Flash (dev), stronger model (eval) |
| Risk Model | XGBoost / LightGBM (trained on real data) |
| Evaluation | RAGAS, DSPy, LLM-as-Judge |
| Deployment | Free cloud tier (TBD) |
| CI/CD | GitHub Actions |

---

## 📋 Sprint Plan

### Sprint 0 — Discover & Define
**Goal:** Understand the problem, profile the data, write PRD v1

- Dataset selection and profiling (TBD)
- Exploratory Data Analysis (EDA)
- Dataset card: source, license, schema, class balance, gaps
- 3–5 user personas (underwriter, applicant, ops lead, compliance officer, regulator)
- PRD v1: problem, scope, metrics with baselines
- Risk Register v1: bias/fairness, unexplainable decline, hallucinated facts, data exposure, over-automation
- Evaluation plan

**Exit criteria:** Profiled data, personas, PRD v1 with metrics and baselines, Risk Register v1

---

### Sprint 1 — Design & De-risk
**Goal:** Finalize architecture, write the build-ready spec, spike riskiest assumption

- Finalize system architecture diagram
- Orchestration Decision Record: justify LangGraph orchestrator-worker + evaluator-optimizer pattern
- Agent contracts as JSON schemas (input/output per agent)
- LLM Gateway design (LiteLLM): routing policy, fallback chain, rate limits, retries
- Model routing table: cheap model path vs. strong model path
- Tool specifications: Income DB, Credit Bureau, Policy Retrieval, Decision Record Writer
- Agent Registry design
- Memory & state design
- Synthetic data pipeline v1: applicant narratives, thin-file profiles, adversarial cases, fraud patterns
- Evaluation harness and golden set
- Spike: validate Policy Retrieval Tool faithfully grounds decisions

**Exit criteria:** Architecture spec, agent contracts, synthetic pipeline v1, golden set, spike done

---

### Sprint 2 — Build the Core
**Goal:** End-to-end agent pipeline working, gateway live, first cost baseline

- Project structure setup: `agents/`, `tools/`, `gateway/`, `api/`, `ui/`, `eval/`, `data/`
- FastAPI entry point + minimal Loan Application UI
- Orchestrator Agent (LangGraph)
- Income Verification Agent + Income DB Tool (mock)
- Credit History Agent + Credit Bureau Tool (mock)
- Policy Complaint Agent + Policy Retrieval Tool (ChromaDB + RAG)
- Risk Scoring Agent (XGBoost/LightGBM trained on dataset)
- Decision Synthesizer Agent
- LiteLLM gateway with routing live
- First end-to-end test: one application through full pipeline
- Cost baseline: tokens and cost per application

**Exit criteria:** Core agent path working end-to-end, gateway live, cost baseline recorded

---

### Sprint 3 — Harden, Scale & Optimize
**Goal:** Full agent set, concurrency, evaluation pipeline, DSPy optimization

- Evaluation Agent (LLM-as-judge with written rubric)
- Decision Record Writer (structured JSON/DB output)
- Async FastAPI + concurrency support
- Semantic caching for repeated policy lookups
- RAGAS evaluation: faithfulness, answer relevancy, context precision/recall
- DSPy: optimize prompts for Risk Scoring + Decision Synthesizer — before/after comparison
- A/B comparison: routing policies and prompt variants
- Fairness testing: approval/error rate gaps across applicant segments
- Adversarial/red-team pass: inconsistent applications, misleading statements
- Risk Register updated
- Regression suite: golden set gates every merge

**Exit criteria:** Full agent set, RAGAS scores, DSPy results, A/B winners, regression suite live

---

### Sprint 4 — Verify, Operate & Present
**Goal:** Final benchmark, deployment, runbook, 20-minute video, viva

- End-to-end benchmark: agentic system vs. single-LLM baseline (report all deltas honestly)
- Final evaluation report: RAGAS, calibrated judge, DSPy results, A/B outcomes, safety pass, gap analysis vs PRD
- Deployment on free cloud tier
- Operate Runbook: monitoring, human-fallback path, rollback, cost alarms, incident response
- SLOs: latency, availability, cost ceiling + alerts
- Final repo cleanup: tests, CI check, no secrets committed
- Synthetic data card + augmentation benchmark table
- 20-minute recorded presentation
- Viva preparation

**Exit criteria:** Deployed system, runbook, final evaluation report, video submitted, viva ready

---

## 📊 Key Metrics

| Metric | What It Measures |
|---|---|
| Decision Quality | Agreement with ground-truth outcome on held-out loans |
| Explanation Faithfulness | Whether written reasons match actual decision factors |
| Fairness Gap | Approval/error rate difference across applicant segments |
| Policy Adherence | Whether every decision respects stated lending policy |
| Cost per Application | Unit economics — the guardrail that decides if this ships |

---

## 📁 Repository Structure *(planned)*

```
halcyon-underwriting-copilot/
├── agents/               # LangGraph agent definitions
├── tools/                # Income DB, Credit Bureau, Policy Retrieval, Record Writer
├── gateway/              # LiteLLM gateway + routing policy
├── api/                  # FastAPI endpoints
├── ui/                   # Minimal loan application UI
├── eval/                 # RAGAS, DSPy, golden set, regression suite
├── data/                 # Dataset cards, synthetic data, EDA notebooks
├── docs/                 # PRD, architecture spec, risk register, runbook
├── tests/                # Unit + integration tests
├── Agentic_Architecture.png
└── README.md
```

---

## 👥 Team Jamun

| Name | Role |
|---|---|
| Harshit | AI/GenAI Engineer |
| Ayush | AI/GenAI Engineer |
| Aditya | AI/GenAI Engineer |
| Himkar | AI/GenAI Engineer |

**Program:** Futurense × IIT Gandhinagar · PG Diploma in AI-ML & Agentic AI Engineering · Cohort 1

---

## 📜 License

This project is built for academic and portfolio purposes as part of the Futurense AI Clinic Capstone Program.

---

*Halcyon Credit is a fictional persona created for this engagement.*
