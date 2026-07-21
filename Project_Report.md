# Comprehensive Project Report: Halcyon Credit — Agentic Underwriting Copilot

**Project Title**: Halcyon Credit — Agentic Underwriting Copilot
**Team**: Team Jamun (Futurense × IIT Gandhinagar PG Diploma in AI-ML — Capstone Project 02)
**Current Status**: 🟢 Completed up to Stage 4 (Local Production Ready)

---

## 1. Complete Stage-by-Stage Breakdown & Individual Contributions

### Stage 1: MVP Scaffolding
**Goal:** Establish the repository, backend structures, and baseline StateGraph architecture.
* **Aditya Arora (sedoCdA):** 
  * Designed the core LangGraph `ApplicationState` schema (`ApplicantFile`, `Decision`, `AgentErrors`).
  * Scaffolded the project structure (`agents/`, `tools/`, `gateway/`, `state/`).
* **Himkar Vashistha (Himkar001):** 
  * Set up the base FastAPI entry points (`api/main.py`) and standard Pydantic schemas.
* **Harshit Gautam (harshit234):** 
  * Defined the SQLite persistence schema (`decision_record_tool.py`) to audit all pipeline runs.
* **Ayush Kumar:** 
  * Created the initial end-to-end `run_pipeline` test logic to ensure basic node traversal.

### Stage 2: Core ML & Policy Knowledge Base (KB)
**Goal:** Train the quantitative ML model and vectorize the qualitative regulatory policies.
* **Harshit Gautam:** 
  * Trained the LightGBM predictive risk model on 1.3M rows of LendingClub data. 
  * Created the `credit_score_bridge.py` tool to transform standard application features into the strict 41-feature vector required by the model.
* **Himkar Vashistha:** 
  * Processed the 7 Halcyon underwriting policies into chunked text.
  * Populated and configured the ChromaDB semantic search vector store (`policy_retrieval_tool.py`).
* **Aditya Arora:** 
  * Built mock data connectors for Income DB and Credit Bureau tools to simulate third-party data pulling.
* **Ayush Kumar:** 
  * Designed the golden set evaluation dataset structure (`eval/golden_set`).

### Stage 3: LangGraph Multi-Agent Orchestration
**Goal:** Wire all 8 independent nodes together and introduce LLM generation.
* **Himkar Vashistha:** 
  * Solved the critical `LiteLLM` routing bug by writing a direct HTTP OpenRouter wrapper (`gateway/router.py`).
  * Wired the `PolicyComplianceAgent` to extract hard stops and advisory flags from ChromaDB chunks.
* **Aditya Arora:** 
  * Built the parallel execution routing logic (Income, Credit, Policy, and Risk running concurrently) converging into the Synthesizer.
* **Harshit Gautam:** 
  * Built the `RiskScoringAgent` node, integrating the LightGBM model and SHAP feature importance in real-time.
* **Ayush Kumar:** 
  * Engineered the `DecisionSynthesizerAgent` and `EvaluationAgent` (LLM-as-a-judge). 
  * Wrote the RAGAS evaluation runner (`eval/ragas_runner.py`).

### Stage 4: UI, Evaluation, and Hardening
**Goal:** Build the frontend, ensure 100% determinism on rules, and harden the API.
* **Ayush Kumar:** 
  * Built the premium full-stack HTML/JS/CSS dashboard (`ui/index.html`).
  * Implemented real-time LangGraph step animations, Risk Gauges, and SHAP visualizations.
* **Aditya Arora:** 
  * Wrote the comprehensive integration test suite (`tests/integration/test_pipeline_integration.py`) covering all 10 golden scenarios and validating the 7 policy rules.
* **Himkar Vashistha:** 
  * Centralized all LLM instructions into a Prompt Library (`gateway/prompts.py`).
  * Enforced strict output formats and `[field.subfield=value]` grounding citations.
* **Harshit Gautam:** 
  * Developed the `/v2` production FastAPI middleware. 
  * Added `/model/info`, `/records`, and system health endpoints, along with X-Request-ID tracking and JSON structured logging.

---

## 2. Problems Encountered and Solutions

| Problem | Root Cause | Solution Implemented |
|---|---|---|
| **LiteLLM Routing Failures** | `litellm==1.40.20` threw an `UnboundLocalError: exception_provider` on any API exception, breaking the pipeline. | **(Himkar)** Bypassed LiteLLM completely. Wrote a direct `requests`-based router hitting OpenRouter's OpenAI-compatible endpoint. |
| **Model 404/400 Errors** | Attempting to use `google/gemini-2.0-flash-001` failed due to OpenRouter tier limits on the provided API key. | Switched standard routing to `openai/gpt-4.1-mini` (cheap) and `openai/gpt-4.1` (strong) which successfully returned 200 OKs. |
| **SHAP Explainer Crashes** | The `transformers` package was incompatible with the local version of `Keras 3`, crashing the Risk Node. | **(Harshit)** Wrote a graceful fallback using LightGBM's native `.feature_importance()` to guarantee the pipeline never halts. |
| **Windows Unicode Exceptions** | Printing LLM responses containing characters like `→` caused `cp1252` encoding crashes on Windows CLI. | Added `sys.stdout.reconfigure(encoding='utf-8', errors='replace')` to all entry points (`run_live_test.py` and `api/main.py`). |
| **LLM Hallucinated Data** | The Synthesizer occasionally invented facts to justify borderline recommendations. | **(Himkar & Ayush)** Overhauled the Prompts. Required strict `[field=value]` brackets citing exact source data. The Evaluator node penalizes scores heavily for ungrounded text. |

---

## 3. Core Project Metrics

* **Machine Learning Performance (LightGBM on LendingClub v2)**: 
  * ROC-AUC: `0.7166`
  * PR-AUC: `0.3854`
  * Default Threshold: `0.2687` (Top 25% marked as High Risk)
  * Feature Count: `41` (Trained on 1.3M rows)
* **LLM Pipeline (GPT-4.1 via OpenRouter)**:
  * Latency per decision: `~2.5 - 4.5 seconds` (Parallelized)
  * Cost per application: `~$0.004`
  * Faithfulness Evaluation: Consistently hitting `1.0` (PASS) on standard applications.
* **System Integration**:
  * Pass rate on 13 integration test assertions: `100%`

---

## 4. What Is Left From The Current Plan

While the core functionality (Stage 1-4) is 100% complete and working beautifully in a local setup, the following items remain to turn this into a cloud-hosted production system:

1. **Docker Containerization**: 
   * Writing a `Dockerfile` and `docker-compose.yml` to package the FastAPI backend, UI, and ChromaDB instance together so it can run reliably on any OS without local Python environment setups.
2. **Cloud Deployment (CI/CD)**: 
   * Deploying the FastAPI backend to an AWS EC2 instance, GCP Cloud Run, or Render.
   * Hosting the `ui/index.html` frontend on Vercel or GitHub Pages.
3. **Mass Scale RAGAS Evaluation**: 
   * Currently, we run integration tests on the 10 currated *Golden Set* cases. We need to run the `ragas_runner.py` against a larger subset (e.g., 500 cases) to get statistically significant metrics on LLM Relevancy and Faithfulness.
4. **Real Database Migrations**: 
   * Replacing the local SQLite database used by `decision_record_tool.py` with a production PostgreSQL database and using Alembic for schema migrations.
