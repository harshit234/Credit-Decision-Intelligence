# ROADMAP.md — Sequenced Path to a Top-Grade Submission

> Ordered by **grade-per-effort and dependency**, not by the original sprint calendar. Each item names the
> rubric bucket it serves and a proposed owner. Keep this in sync with `PROGRESS.md`. Re-prioritise freely —
> but log scope changes in `DECISIONS.md`.

Legend — Buckets: **E**=Engineering · **Q**=Quality/Eval · **A**=Architecture · **P**=Product · **D**=Demo · **V**=Viva

---

## Phase 0 — Unblock & align (this week)

| # | Item | Bucket | Owner | Done when |
|---|---|---|---|---|
| 0.1 | Ratify North Star = **TDR**; reconcile PRD.md/README metrics | P | Ayush | PRD + README updated; ADR-005 Accepted |
| 0.2 | Commit to **Home Credit**; download + EDA + dataset card | Q/E | Harshit | `data/dataset_card.md` + EDA notebook committed; ADR-006 Accepted |
| 0.3 | Repo scaffolding: folder tree (TRD §9.1), `.gitignore`, `LICENSE`, `requirements`/`pyproject`, `.env.example`, `Makefile`, `CODEOWNERS` | E | Aditya | `make setup` works on a clean clone |
| 0.4 | Move root docs → `docs/`; add ADR log + failure-mode table | A | Ayush | `docs/` populated; CLAUDE/PROGRESS/DECISIONS at root |
| 0.5 | CI stub (lint + unit + placeholder golden gate) | E | Himkar | GitHub Actions green on PR |
| 0.6 | Assign individual verticals; **Ayush + Aditya first commits** | V | All | `PROGRESS.md` §9 + `CODEOWNERS` reflect ownership |

## Phase 1 — Vertical slice + measurement spine (de-risks everything)

| # | Item | Bucket | Owner | Done when |
|---|---|---|---|---|
| 1.1 | `state/application_state.py` — Pydantic models from TRD §3 | E/A | Aditya | Models + validators + unit tests |
| 1.2 | Mock tools (income, credit, policy, record) | E | each owner | Deterministic mocks + tests |
| 1.3 | LangGraph graph: init → parallel workers → merge → risk → synthesize → evaluate → write | E/A | Aditya | One application runs end-to-end locally |
| 1.4 | LiteLLM gateway live with cheap/strong routing + cost log | E | Himkar | Cost-per-call logged |
| 1.5 | **Single-LLM baseline** harness (same inputs, one prompt) | Q | Himkar | Baseline decisions produced for the golden set |
| 1.6 | **Cost-per-application baseline** recorded | E/P | Himkar | Number in PROGRESS + DecisionRecord |
| 1.7 | Capture a screen-recording of the first end-to-end run | D | Ayush | Clip saved to `docs/demo/` |

## Phase 2 — Real grounding, risk model, golden set

| # | Item | Bucket | Owner | Done when |
|---|---|---|---|---|
| 2.1 | Author Halcyon **policy corpus**; ingest to ChromaDB | A/E | Ayush | Collection built; chunks have clause metadata |
| 2.2 | **Policy-RAG spike**: prove retrieval faithfully grounds decisions | Q/A | Ayush | Spike report in `docs/`; source_refs verified |
| 2.3 | Train risk model (XGBoost/LightGBM) + SHAP + calibration curve | E/Q | Harshit | Model artifact + model card + AUC/calibration/fairness |
| 2.4 | Synthetic generator (standard, thin-file, adversarial, fraud) | Q | Himkar | Seeded generator + dataset card |
| 2.5 | **Golden set** ≥100 cases w/ ground-truth labels | Q | All | `eval/golden_set/` committed |

## Phase 3 — Evaluation rigor (the bucket that wins)

| # | Item | Bucket | Owner | Done when |
|---|---|---|---|---|
| 3.1 | RAGAS runner (faithfulness, relevancy, ctx precision/recall) | Q | Ayush | Scores logged per run |
| 3.2 | **Calibrate the LLM-judge** vs human-labelled set (report κ/agreement) | Q | Ayush | Calibration report in `docs/` |
| 3.3 | **Fairness report** — segment gaps + proxy methodology + mitigation | Q/P | shared | Report + gate on golden-set run |
| 3.4 | Adversarial / red-team pass + results | Q | Himkar | Red-team set + findings |
| 3.5 | DSPy optimize synthesizer + evaluator; before/after RAGAS | Q | Himkar | Delta table committed |
| 3.6 | Regression gate enforced in CI (blocks merge on golden-set fail) | E/Q | Aditya | CI blocks on threshold breach |
| 3.7 | **Agentic-vs-baseline benchmark report** (honest deltas) | Q/D | All | `docs/benchmark.md` + chart |

## Phase 4 — Operate, deploy, present

| # | Item | Bucket | Owner | Done when |
|---|---|---|---|---|
| 4.1 | FastAPI endpoints (POST /applications, GET /{id}, /health, /metrics) | E | Aditya | API live + tested |
| 4.2 | Minimal UI: application form + **reasoning-chain + audit-replay view** | E/D | Ayush | Underwriter can see the trace |
| 4.3 | Deploy to free tier (Render/Railway/Fly.io) | E/D | Aditya | Public URL up |
| 4.4 | Runbook: monitoring, human-fallback, rollback, cost alarms, SLOs | A/E | Himkar | `docs/runbook.md` |
| 4.5 | Final evaluation report vs PRD (honest gap analysis) | Q/P | All | `docs/final_report.md` |
| 4.6 | 20-min recorded presentation (each owner presents their vertical) | D/V | All | Video submitted |
| 4.7 | Viva prep: each member rehearses their vertical + whole-system Q&A | V | All | Dry-run done |

---

## Critical path & parallelism

- **Critical path:** 0.2/0.3 → 1.1→1.3 → 2.5 → 3.1/3.7 → 4.3/4.6.
- **Parallelisable now:** dataset (Harshit), scaffolding+CI (Aditya/Himkar), policy corpus (Ayush), North-star reconciliation (Ayush).
- **Front-load the baseline (1.5) and golden set (2.5)** — every later eval number depends on them, and the agentic-vs-baseline delta is the project's headline result.

## "Minimum viable top grade" if time gets tight
Vertical slice (1.1–1.4) + golden set (2.5) + single-LLM baseline (1.5) + RAGAS + calibrated judge (3.1–3.2) + fairness (3.3) + deploy (4.3) + honest benchmark (3.7) + each person owning a vertical for the viva. That set alone fills both 25% buckets and clears the gate.
