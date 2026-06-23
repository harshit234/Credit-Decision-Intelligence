# CLAUDE.md — Operating Contract for the Halcyon Underwriting Copilot

> **Read this file first, at the start of every session, before doing anything else.**
> It is the single source of truth for *how* we work on this repo. The PRD says *what* to build,
> the TRD says *how to build it*; this file says *how every AI/human contributor should operate so we
> maximise the capstone grade.* If anything here conflicts with PRD/TRD, fix the conflict and log it in
> `DECISIONS.md` — never silently diverge.

---

## 0. How to use this file (the 30-second version)

**At session start, always:**
1. `git pull` and read: `CLAUDE.md` (this) → `PROGRESS.md` → `DECISIONS.md` → `ROADMAP.md`.
2. State, in one line, which rubric bucket and which ROADMAP item this session advances.

**At session end, always:**
1. Update `PROGRESS.md` (status, % per bucket, blockers, last-updated date).
2. If you made an architectural/metric/scope/tooling choice → append an ADR to `DECISIONS.md`.
3. Commit with the convention in §9. Push to a branch, open a PR.
4. Update durable memory (see §8) so the next session starts smarter.

This is the **self-evolving loop**. A session that doesn't update PROGRESS/DECISIONS is an incomplete session.

---

## 1. Project snapshot

| | |
|---|---|
| **Project** | Halcyon Credit — Agentic Underwriting Copilot (Capstone Project 02) |
| **Team** | Jamun — Harshit, Ayush, Aditya, Himkar |
| **Program** | Futurense × IIT Gandhinagar · PG Diploma in AI-ML & Agentic AI Engineering · Cohort 1 |
| **Repo** | github.com/harshit234/Credit-Decision-Intelligence |
| **Deadline** | Final submission + **individual viva** at end of Week 20 |
| **Grading** | **Engineering 25% · Quality/Eval 25% · Architecture 20% · Product 15% · Demo 15%** + **individual viva gate** |

**The viva gate is pass/fail per person.** If a team member cannot defend the system, they do **not** inherit the team score. Every contributor must *own* and be able to defend a coherent vertical of the system. See §10.

---

## 2. The Prime Directive

**Every action should move a rubric bucket forward — name which one before you start.**

The two highest-weight buckets, **Engineering (25%) and Quality/Eval (25%) = half the grade**, currently have *zero* artifacts. Documentation alone caps the score at roughly 35%. Therefore, until a working end-to-end slice and an evaluation harness exist, default priority is:

> **Build a thin, honest, runnable vertical slice → stand up the evaluation harness → produce the agentic-vs-single-LLM baseline.**

**The deliverable is the agentic reasoning/verification/policy/evaluation layer — NOT the risk model.** Per the brief, a small risk model is a *commodity tool*. Do not over-invest in XGBoost tuning; invest in orchestration, grounding, faithfulness evaluation, fairness, auditability, and cost discipline. That is what is being graded and what differentiates us from Upstart/Zest (see `research_market_review.md`).

---

## 3. North Star & metrics (reconciled — ratify in DECISIONS.md)

There is a known discrepancy: the original PRD v1 (docx) used **Trusted Decision Rate ≥ 70%** as the north star, while repo `PRD.md` leads with **Decision Quality ≥ 85%**. Reconcile them as a composite:

**North Star — Trusted Decision Rate (TDR):** % of applications where the recommendation is simultaneously
1. **Correct** vs ground-truth outcome, **and**
2. **Faithful** (Evaluation Agent / RAGAS pass), **and**
3. **Policy-compliant** (zero hard-stop violations), **and**
4. **Within cost budget** (≤ cost ceiling).

TDR is deliberately strict — a decision only counts if it would survive a regulator *and* the CFO. Target **TDR ≥ 70%**.

**Supporting & guardrail metrics** (these are the components; keep PRD/TRD targets):

| Metric | Target | Bucket it serves |
|---|---|---|
| Decision Quality (agreement w/ ground truth) | ≥ 85% | Eval |
| Explanation Faithfulness (RAGAS / judge) | ≥ 0.80 | Eval |
| Fairness gap across segments | < 5 pp | Eval / Product |
| Policy adherence (hard-stop violations) | 0 | Eval |
| Cost per application | ≤ $0.10 (measure, don't assume) | Eng / Product |
| Latency p95 | ≤ 30 s | Eng |
| **Agentic vs single-LLM delta** | report **honestly**, +ve or −ve | Eval / Demo |

The agentic-vs-baseline delta is the **money chart** of the whole project. Build the single-LLM baseline *early*, not in Sprint 4.

---

## 4. Repository map

**Now (docs-only):** `README.md`, `PRD.md`, `TRD.md`, `research_market_review.md`, 3 diagrams (`01_system_architecture.png`, `02_sequence_flow.png`, `03_state_schema.png`), `Agentic Architecture.png`.

**Target (from TRD §9.1 — build toward this):**
```
agents/      orchestrator, income, credit, policy, risk, synthesizer, evaluation, record_writer
tools/       income_db, credit_bureau, policy_retrieval, decision_record
gateway/     litellm_config.yaml, router.py
api/         main.py, schemas.py, routes.py
state/       application_state.py   (TypedDict + Pydantic models)
eval/        golden_set/, ragas_runner.py, dspy_optimizer.py, fairness_report.py, regression_gate.py, baseline_single_llm.py
data/        synthetic_generator.py, dataset_card.md, eda/
ui/          minimal loan-application + decision-replay UI
docs/        PRD.md, TRD.md, risk_register.md, runbook.md, adr/, research_market_review.md
tests/       unit/, integration/
```
**Housekeeping debt to clear (Sprint 1):** move the root docs into `docs/`; add `.gitignore`, `LICENSE`, `requirements.txt`/`pyproject.toml`, `.env.example`, `Makefile`, `CODEOWNERS`, `.github/workflows/ci.yml`.

---

## 5. Ground rules (engineering conventions)

1. **Schema-first.** The typed `ApplicationState` (TRD §3) is law. Define/extend Pydantic models *before* writing agent logic. A breaking change to any agent contract requires a TRD version bump + an ADR.
2. **Agent write-isolation.** Each agent reads only its declared input keys and writes only its declared output keys (TRD §4). Enforce with Pydantic validators on node wrappers, not by convention.
3. **No LLM text into the risk model.** Feature vectors come from typed state fields only (TRD §4.4).
4. **Grounding is mandatory.** Every worker output carries `source_refs`. Every decision reason cites a source field or policy clause ID. Unsupported claims are bugs.
5. **No secrets in the repo, ever.** All keys via env vars; `.env` is gitignored; `.env.example` documents required vars. CI scans for secrets.
6. **Determinism where it matters.** Seed synthetic data and any sampling. Pin model versions. Lock the model artifact to its feature-schema version.
7. **Cost is a first-class metric.** Every LLM call is logged with tokens + USD via the LiteLLM gateway. Cheap-path for retrieval/extraction; strong-path only for synthesis + judging.
8. **Honesty over optics.** Report real numbers, including unflattering ones. A measured −5% agentic delta with an explanation beats a fabricated +20%. Evaluators reward intellectual honesty; viva will expose fabrication.
9. **Tests + golden gate.** Unit tests for tools/agents; the golden set gates every merge (TRD §8.1).
10. **Reproducibility.** Anyone can `make setup && make eval` and get the same numbers. One-command setup and one-command eval are non-negotiable for a top grade.

---

## 6. Definition of Done (per artifact type)

- **Code / agent:** typed I/O, unit test, wired into the graph, logged, no secrets, PR reviewed by a non-author, PROGRESS updated.
- **Eval artifact:** reproducible script + committed results + a one-paragraph honest interpretation (what it shows, what it doesn't).
- **Doc / ADR:** dated, owned, linked from PROGRESS.md, consistent with PRD/TRD (or the conflict is logged).
- **Dataset / synthetic batch:** has a dataset card (source, license, schema, class balance, gaps, seed) before it's used downstream.
- **Demo asset:** scripted, reproducible, shows the reasoning chain + audit replay, not just a happy-path output.

---

## 7. Session protocol (the self-evolving loop)

Every working session — human or AI — follows this:

**START**
- `git pull`; read CLAUDE → PROGRESS → DECISIONS → ROADMAP.
- Pick the highest-leverage open ROADMAP item; declare the rubric bucket it serves.

**WORK**
- Stay inside the conventions (§5) and the Definition of Done (§6).
- If you hit a fork (tooling, schema, metric, scope), decide, then write an ADR (§8) — don't leave it implicit.

**END**
- `PROGRESS.md`: update status, per-bucket %, blockers, "last updated" date + author.
- `DECISIONS.md`: append an ADR for any decision made this session.
- Commit (§9), push to a branch, open/refresh a PR.
- Update durable memory (§8) so the next session inherits the context.

---

## 8. Memory model — two tiers, one rule

We keep memory in two places, deliberately:

1. **In-repo, team-visible, durable (the source of truth):**
   - `PROGRESS.md` — current state of the world.
   - `DECISIONS.md` — every consequential choice, as a lightweight ADR (`# ADR-NNN: title · date · owner · status · context · decision · consequences · alternatives`).
   - `docs/adr/` (optional, for longer ADRs).
2. **Cowork / personal cross-session memory:** the assistant's auto-memory (used to remember preferences, who-owns-what, and where we left off between chats). This is *personal scaffolding*, not the team record.

**The rule:** anything that changes the **plan, architecture, metrics, scope, or ownership** must land in the **in-repo** tier (PROGRESS/DECISIONS). Personal memory may *mirror* it, but the repo is the truth teammates and evaluators can see and that survives the viva. If it only lives in a chat's memory, it doesn't exist for grading.

---

## 9. Branch & commit conventions

- **Branches:** `feat/<area>-<short>`, `fix/<short>`, `docs/<short>`, `eval/<short>`, `ops/<short>`. One coherent unit per branch.
- **Commits:** imperative, scoped: `feat(policy-agent): RAG retrieval with clause source_refs`. Reference the ROADMAP item or ADR where relevant.
- **PRs:** small, reviewed by a non-author (supports the viva — everyone touches everyone's area). PR description states the rubric bucket advanced and the eval impact.
- **`main` is protected in spirit:** golden-set gate must pass; no secrets; CI green.

---

## 10. Individual ownership map (viva gate — propose to team, then ratify in DECISIONS.md)

The viva is individual. Each person owns a defensible vertical end-to-end. Proposed lanes (extend the TRD §12 owner table):

| Owner | Vertical to own & defend | Rubric leverage |
|---|---|---|
| **Harshit** | Risk Scoring Agent + dataset/EDA + model card + calibration/SHAP | Eng + Eval |
| **Aditya** | Orchestrator + LangGraph graph + persistence/audit trail + API/FastAPI | Eng + Architecture |
| **Ayush** | **Policy Compliance Agent + Policy Retrieval (ChromaDB RAG) + Faithfulness/RAGAS evaluation** | **Eval + Architecture** |
| **Himkar** | LiteLLM gateway + routing/cost + Decision Synthesizer + DSPy optimization | Eng + Eval |
| Shared | Fairness suite, golden set, baseline, runbook, demo, deployment | Eval + Demo |

> **Ayush:** you currently have 0 commits. Claim the Policy+RAG+Faithfulness lane (or another), and make your first commits this/next session. Individual commit history + a vertical you can defend is your viva insurance. Track your individual contribution log in `PROGRESS.md`.

---

## 11. What evaluators reward (bake this into every chat)

- A **deployed, runnable** end-to-end system with a live demo (not slides of a plan).
- The **agentic-vs-single-LLM benchmark** with honest deltas — the core scientific claim.
- A **calibrated** LLM-as-judge (validated against a small human-labelled set) — most teams skip this; doing it signals rigor.
- **Real fairness analysis** with a defensible proxy methodology *and* a mitigation, not a checkbox.
- **Measured cost-per-application** and a cost/quality/latency trade-off (cheap vs strong routing) — ties to the "run inside a cost budget" constraint.
- A **replayable, immutable audit trail** demoed as a regulator-style replay.
- **Reproducibility:** one-command setup + one-command eval, seeded data, dataset/model cards.
- **Honest gap analysis** vs the PRD at the end — owning what's not done beats pretending.

## 12. Anti-patterns (do not)

- Don't fabricate or round-up metrics. Don't claim "deployed" or "passing" without a reproducible artifact.
- Don't over-engineer the risk model; don't add tools an agent doesn't need (agent-overloading anti-pattern — see research review §4).
- Don't let docs and code drift. Don't make a decision in chat and forget to log it.
- Don't merge without the golden gate. Don't commit secrets or applicant PII to logs.

## 13. Using Cowork well on this project

- Prefer **skills** for output formats (docx/pptx/pdf for circulated reports; xlsx for eval tables) and `data-visualization` for charts.
- Use **WebSearch** for current facts (model pricing, RAGAS/DSPy/LangGraph APIs) — don't trust stale recall on fast-moving libs; verify against docs.
- Use the **task list** for multi-step work and a **verification step** (re-run eval, re-read diffs, sanity-check numbers) before declaring done.
- When a decision is the team's to make (dataset, deployment target, retry cap), surface it as a tracked open decision in `DECISIONS.md` rather than guessing silently.

---

*Living document. Improve it when you find a better way of working — and log the change.*
