# DECISIONS.md — Architecture & Project Decision Log (ADRs)

> Every consequential choice — tooling, schema, metric, scope, ownership — gets a short ADR here.
> This is the durable, team-visible record that survives the viva. If you decide something in a chat
> and don't log it here, it didn't happen. Format below; newest at the bottom.

**ADR template**
```
## ADR-NNN: <title>
- Date: YYYY-MM-DD · Owner: <name> · Status: Proposed | Accepted | Superseded
- Context: <why this came up>
- Decision: <what we chose>
- Consequences: <what this implies, good and bad>
- Alternatives considered: <options + why rejected>
```

---

## ADR-001: Orchestration framework = LangGraph (orchestrator-worker + evaluator-optimizer)
- Date: 2026-06 · Owner: Team · Status: Accepted
- Context: Need stateful multi-agent orchestration with parallel fan-out, a typed shared state, checkpointing for audit replay, and a conditional retry loop.
- Decision: LangGraph state machine; central Orchestrator; parallel workers (Income, Credit, Policy); state-gated Risk Scoring; Synthesizer→Evaluator retry loop capped at N=2 then human escalation.
- Consequences: Native checkpointing gives a replayable audit trail (a grading differentiator). Adds LangGraph as a core dependency the team must learn deeply for the viva.
- Alternatives considered: CrewAI (less explicit state control), raw Python orchestration (more code, weaker audit story), Autogen.

## ADR-002: LLM access via LiteLLM gateway
- Date: 2026-06 · Owner: Himkar · Status: Accepted
- Context: Need routing, fallback, retries, and per-call cost logging across cheap/strong model paths under a cost budget.
- Decision: All LLM calls route through LiteLLM. Cheap path (Gemini Flash) for retrieval/extraction; strong path for synthesis + judging; fallback chain on timeout/5xx; cost logged per call.
- Consequences: Centralised cost telemetry → measurable cost-per-application. One more component to operate.
- Alternatives considered: Direct SDK calls (no central cost/fallback control).

## ADR-003: Policy compliance via RAG over ChromaDB
- Date: 2026-06 · Owner: Ayush (proposed) · Status: Accepted
- Context: Decisions must be grounded in retrievable, citable policy clauses for adverse-action defensibility.
- Decision: ChromaDB vector store (`all-MiniLM-L6-v2`, 512-token chunks/64 overlap, clause metadata). Policy agent outputs clause IDs + hard-stops + flags with `source_refs`. Fallback to conservative defaults + alert on retrieval failure.
- Consequences: Grounded, auditable policy checks — a capability Upstart/Zest lack (see research review). Requires a curated policy corpus (to be authored, since Halcyon is fictional).
- Alternatives considered: Hardcoded rules (not auditable/flexible), full-text LLM policy reasoning (ungrounded, hallucination risk).

## ADR-004: Risk model is a commodity, not the deliverable
- Date: 2026-06 · Owner: Team · Status: Accepted
- Context: The brief explicitly states the risk model is a commodity tool; the graded deliverable is the agentic reasoning/verification/policy/eval layer.
- Decision: Use XGBoost/LightGBM with SHAP attribution; spend effort on calibration + fairness + explainability, not exotic tuning. Cap time spent on raw model AUC chasing.
- Consequences: Effort concentrates where marks are. Risk model still must be defensible (calibration curve, SHAP, fairness).
- Alternatives considered: Deep tabular nets (overkill, worse explainability).

---

## Proposed / Open decisions (ratify and move up as ADRs)

## ADR-005: North Star = Trusted Decision Rate (TDR) — PROPOSED
- Date: 2026-06-23 · Owner: Ayush · Status: Proposed
- Context: PRD v1 (docx) used "Trusted Decision Rate ≥70%"; repo PRD.md leads with "Decision Quality ≥85%". The two must be reconciled or evaluators will see inconsistency.
- Decision (proposed): Adopt **TDR** as the composite north star = % of applications that are correct **and** faithful **and** policy-compliant **and** within cost budget; target ≥70%. Decision Quality, Faithfulness, Fairness gap, Policy adherence, Cost become *supporting* metrics retaining their PRD/TRD targets.
- Consequences: One headline metric that maps to all four grading concerns; forces joint optimisation rather than cherry-picking. Update PRD.md + README to match once ratified.
- Alternatives: Keep Decision Quality as sole north star (ignores faithfulness/fairness/cost — weaker story).

## ADR-006: Dataset = Home Credit Default Risk — PROPOSED
- Date: 2026-06-23 · Owner: Harshit · Status: Proposed
- Context: Sprint 0 exit criterion (profiled data + dataset card) is unmet; PRD/TRD still say "TBD". Risk model and golden set are both blocked on this.
- Decision (proposed): Commit to **Home Credit Default Risk** (Kaggle; ~307k rows × 122 cols; ~8% default; multi-table; thin-file relevant). Download, EDA, write dataset card with license, class balance, gaps, fairness-relevant fields (note `CODE_GENDER` for fairness — never a direct model input).
- Consequences: Unblocks risk model + golden set + fairness work. Class imbalance (~8%) needs handling (resampling/threshold). Multi-table joins add data-prep effort.
- Alternatives: LendingClub, HMEQ — smaller / less thin-file representative.

## Open (need an owner + date to become ADRs)
- **Deployment target:** Render vs Railway vs Fly.io (latency, cold-start, free-tier limits).
- **Retry cap N tuning:** calibrate against golden-set faithfulness distribution (Himkar).
- **DSPy optimizer:** BootstrapFewShot vs MIPROv2 (compare RAGAS deltas).
- **Fairness proxy methodology:** how to measure disparate impact without using protected attributes as model inputs.
- **Persistence:** Postgres vs SQLite for the deployed tier (Aditya).
- **ChromaDB:** managed cloud vs self-hosted (Ayush).
- **LLM-as-judge calibration:** size + labelling protocol for the human-labelled calibration set.
