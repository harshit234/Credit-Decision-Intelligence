# EVALUATION_GAP_ANALYSIS.md — Where We Stand vs. Top Marks

> An evaluator's-eye review of the repo as of **2026-06-23**, mapped to the grading rubric
> (**Engineering 25% · Quality/Eval 25% · Architecture 20% · Product 15% · Demo 15% + individual viva gate**).
> Goal: a brutally honest read of what would score today, and the specific moves that convert
> "good plan" into a top-grade, defensible, deployed system.

---

## 0. Executive read

The **planning and design work is genuinely strong** — the TRD is build-ready, the research review is sharp and honestly framed, and the diagrams are professional. If this were graded on *design intent*, it would score well.

But the rubric weights **execution**: 50% of the grade (Engineering + Quality/Eval) sits in buckets that currently contain **zero artifacts**. Documentation alone realistically caps the total around **35%**. The plan is not the product. **The single biggest risk is treating the design phase as the deliverable.**

Two structural risks compound this:
1. **Schedule risk** — half the grade is unstarted with finite weeks to the Week-20 deadline.
2. **Individual viva risk** — the viva is pass/fail per person; **Ayush has 0 commits** and no owned vertical yet.

The good news: because the design is solid, execution can move fast. The path to a top grade is narrow but clear — see §7 and `ROADMAP.md`.

---

## 1. Architecture — 20% · Estimated ~80%

**What scores today**
- Clean orchestrator-worker + evaluator-optimizer pattern, justified against real prior art.
- Build-ready TRD: typed `ApplicationState`, per-agent read/write contracts, tool specs, LangGraph graph definition, LiteLLM routing, persistence schema, SLOs.
- Three professional diagrams (HLD, LLD sequence, state schema).

**Gaps to close for full marks**
- **No explicit ADRs / Orchestration Decision Record.** The brief calls for a decision record justifying the orchestration choice. The *reasoning* exists in the research review but isn't captured as decisions with alternatives + consequences. → `DECISIONS.md` (now seeded) closes this; expand ADR-001.
- **No failure-mode / resilience analysis.** What happens on partial worker failure, RAG miss, judge disagreement loop, gateway outage? Some is implied in the TRD; make it an explicit failure-mode table.
- **No scalability / concurrency reasoning beyond a target number.** Back the "10 concurrent" claim with the async design rationale.
- **C4 / level-consistency:** the diagrams are great but ad hoc. A short "why this topology" narrative tying HLD↔LLD↔state would tighten it.

**Top-grade move:** a 1-page "Architecture Decisions & Failure Modes" doc + the ADR log. Cheap, high-rubric-yield.

---

## 2. Product — 15% · Estimated ~75%

**What scores today**
- PRD with personas, scope/non-scope, a risk register, and metrics with targets.
- Clear regulatory framing (explainability, fairness, adverse-action).

**Gaps to close**
- **North-star inconsistency:** PRD v1 docx = "Trusted Decision Rate ≥70%"; repo PRD.md = "Decision Quality ≥85%". Evaluators *will* notice. → Reconcile via the **TDR composite** (ADR-005); update PRD.md + README.
- **Cost baseline is "TBD".** "Run inside a cost budget" is a core constraint — an unquantified budget is a weak spot. Measure it as soon as the slice runs.
- **Personas → requirements traceability.** Maya (underwriter), David (escalation), Priya (ops-risk), Regulator exist, but the link from each persona's need → a specific FR → a specific metric isn't drawn. A small traceability table makes the product story airtight.
- **Success-criteria tie to the brief lifecycle** could be more explicit.

**Top-grade move:** reconcile the north star, add a persona→requirement→metric traceability table, and replace every "TBD" with a number or a dated owner.

---

## 3. Engineering — 25% · Estimated 0% · ⚠️ HIGHEST PRIORITY

**What scores today:** nothing is built.

**Everything is a gap:**
- No code: no `agents/`, `tools/`, `gateway/`, `api/`, `state/`, `ui/`.
- No repo hygiene: no `.gitignore`, `LICENSE`, `requirements.txt`/`pyproject.toml`, `.env.example`, `Makefile`, `CODEOWNERS`.
- No CI: the TRD specifies a GitHub Actions pipeline + golden gate; none exists.
- No tests.
- No dataset downloaded/profiled; no synthetic generator.
- Docs sit at repo root, not in `docs/` as the TRD prescribes.

**Top-grade moves (in order):**
1. **Repo scaffolding** (folders + hygiene + CI stub) — half a day, unblocks everyone.
2. **Vertical slice:** one application → mock tools → LangGraph graph → typed decision → persisted record. Prove the spine works before widening it.
3. Widen to the **full agent set** with real Policy-RAG and the trained risk model.
4. **FastAPI** endpoints + a **minimal UI** that shows the *reasoning chain and audit replay* (the demo's wow factor, not just an approve/decline).
5. **Tests + CI golden gate** wired so every merge is checked.
6. **Deploy** to a free tier with a public URL.

Engineering is where the most marks are currently unclaimed and where the team is weakest — **start here, today.**

---

## 4. Quality / Evaluation — 25% · Estimated 0% · ⚠️ CO-HIGHEST PRIORITY

This is the bucket that, done well, **separates a top capstone from an average one** — and it's empty.

**Gaps (all of it):**
- **No golden set.** Need ≥100 cases with ground-truth decisions (synthetic + a slice of real data).
- **No single-LLM baseline.** This is *the* scientific control — the agentic-vs-baseline delta is the project's central claim. Build it early; report the delta honestly even if it's small or negative.
- **No RAGAS** (faithfulness, answer relevancy, context precision/recall).
- **No DSPy** before/after optimization with measured RAGAS deltas.
- **No fairness report.** Segment approval/error-rate gaps with a defensible proxy methodology (Home Credit has `CODE_GENDER`; treat as a fairness *measurement* variable, never a model input).
- **Uncalibrated LLM-as-judge.** A judge you haven't validated against human labels is itself a risk. **Calibrate it** against a small human-labelled set and report agreement (e.g., Cohen's κ). *Most teams skip this — doing it is a standout signal of rigor.*
- **No adversarial / red-team set.** Inconsistent income, circular loan purpose, velocity/fraud patterns.
- **No calibration curve / regression gate** results.

**Top-grade moves:**
1. Synthetic generator + golden set with ground truth.
2. Single-LLM baseline harness alongside the agentic pipeline.
3. RAGAS runner + fairness report + adversarial set, all reproducible.
4. **Calibrate the judge** (human-labelled set + agreement metric) — headline rigor.
5. DSPy before/after with real numbers.
6. Regression gate in CI.

---

## 5. Demo — 15% · Estimated 0% (expected — Sprint 4)

**What top marks need (start collecting assets early):**
- A **live, deployed** system, not a localhost screen-share.
- A demo that walks the **reasoning chain** and an **audit replay** of a real decision (regulator framing).
- The **agentic-vs-baseline** chart shown on screen.
- A tight 20-minute recorded presentation; every team member presents the vertical they own (feeds the viva).
- Honest "what we'd do next" gap slide.

**Risk:** leaving the demo to the final week. Capture screen-recordings of each milestone as you go.

---

## 6. Individual viva gate — pass/fail per person · ⚠️ AT RISK

The viva does not inherit the team score automatically — each person must defend a vertical.

- **Ayush: 0 commits, no owned vertical.** Highest personal risk. → Claim **Policy Compliance + ChromaDB RAG + Faithfulness/RAGAS evaluation** (a coherent Eval+Architecture vertical) and start committing now.
- **Aditya: 0 commits** so far too — claim Orchestrator + persistence + API.
- Ensure cross-review on PRs so each person can speak to the whole system, not only their slice.
- Maintain the **individual contribution log** in `PROGRESS.md` §9 — it's your viva evidence.

---

## 7. If you only do the next 8 things (highest grade-per-effort)

1. **Ratify the North Star (TDR)** and fix the PRD/repo metric inconsistency. *(Product, 1 hr)*
2. **Commit to Home Credit**, download, EDA, dataset card. *(Eng/Eval, unblocks two buckets)*
3. **Scaffold the repo** (folders, hygiene, CI stub, `docs/` reorg). *(Eng, half day)*
4. **Vertical slice** end-to-end with mock tools. *(Eng, the spine)*
5. **Single-LLM baseline** harness. *(Eval, the money chart)*
6. **Golden set** (≥100 cases, ground truth). *(Eval)*
7. **Calibrate the LLM-judge** vs a small human-labelled set. *(Eval, standout rigor)*
8. **Assign + start individual verticals**; Ayush + Aditya make first commits. *(Viva)*

Do these eight and the project moves from ~35%-capped to genuinely top-tier, because they directly fill the two empty 25% buckets and de-risk the viva. Full sequencing in `ROADMAP.md`.

---

## 8. Differentiators that win (what pushes "good" → "world-class")

- The **honest agentic-vs-single-LLM** comparison (with negative results owned, not hidden).
- A **calibrated judge** with reported human-agreement.
- **Fairness done properly** — proxy methodology + measured gaps + a mitigation, not a checkbox.
- **Measured cost-per-application** and a cost/quality/latency Pareto across routing strategies.
- A **regulator-style audit replay** in the demo.
- **One-command reproducibility** (`make setup && make eval`) with seeded data and dataset/model cards.

These are exactly the gaps the research review identified in Upstart/Zest/APA — delivering them is the thesis of the project made real.
