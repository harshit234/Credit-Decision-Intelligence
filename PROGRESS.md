# PROGRESS.md — Living Status Board

> Update at the end of **every** session (see `CLAUDE.md` §7). This is the team-visible source of truth
> for where we are. Keep it honest and current.

**Last updated:** 2026-06-23 · by: Ayush (via Cowork analysis session) · branch: `capstone-ops-setup`

---

## 1. One-line status

Documentation phase essentially complete (PRD, TRD, research review, architecture diagrams). **Engineering and Evaluation = 0% — both must start now.** Repo housekeeping not yet done.

## 2. Rubric scoreboard (self-assessed — be honest)

| Bucket | Weight | Est. now | Target | Gap to close |
|---|---|---|---|---|
| Architecture | 20% | ~80% | 95%+ | ADRs, trade-off justification, failure-mode analysis |
| Product | 15% | ~75% | 90%+ | Reconcile north-star (TDR), set cost baseline, PRD↔repo consistency |
| **Engineering** | **25%** | **0%** | 80%+ | Vertical slice → full pipeline → API/UI → tests → CI → deploy |
| **Quality/Eval** | **25%** | **0%** | 80%+ | Golden set, RAGAS, DSPy, fairness, **single-LLM baseline**, calibrated judge |
| Demo | 15% | 0% | 90%+ | Deployed system + 20-min video + audit-replay demo (Sprint 4) |
| Individual viva | gate | ⚠️ at risk | pass×4 | Each member owns + defends a vertical; **Ayush has 0 commits** |

## 3. Sprint tracker

| Sprint | Theme | State | Notes |
|---|---|---|---|
| Sprint 0 | Discover & Define | 🟡 ~90% | PRD/personas/risk register/research done. **Open:** dataset not yet downloaded/profiled; dataset card missing (Sprint 0 exit criterion). |
| Sprint 1 | Design & De-risk | 🟡 ~60% | TRD + agent contracts + diagrams done. **Open:** ADRs, synthetic pipeline, golden set, LiteLLM routing impl, Policy-RAG spike, repo scaffolding. |
| Sprint 2 | Build Core | 🔴 0% | Nothing built. |
| Sprint 3 | Harden & Optimize | 🔴 0% | — |
| Sprint 4 | Verify & Operate | 🔴 0% | — |

## 4. Done

- ✅ Read full brief; principal-engineer strategy briefing.
- ✅ README.md, PRD.md, TRD.md (build-ready depth), research_market_review.md.
- ✅ HLD, LLD sequence, state-schema diagrams (professional quality).
- ✅ Original `Agentic Architecture.png`.
- ✅ This ops kit (CLAUDE.md, PROGRESS.md, DECISIONS.md, EVALUATION_GAP_ANALYSIS.md, ROADMAP.md).

## 5. In progress

- 🟡 Ops kit being committed on `capstone-ops-setup` branch (this session).

## 6. Next (top of queue — see ROADMAP.md)

1. Ratify North Star = Trusted Decision Rate; reconcile PRD/repo metrics (ADR).
2. Commit to **Home Credit Default Risk** dataset; download, profile (EDA), write dataset card.
3. Repo scaffolding: folders, `.gitignore`, `LICENSE`, `requirements`, `.env.example`, `Makefile`, `CODEOWNERS`, CI stub.
4. Synthetic data generator v1 + golden set (≥100 cases w/ ground truth).
5. **Vertical slice:** mock tools → LangGraph state graph → decision, one application end-to-end.
6. **Single-LLM baseline** harness (so the agentic delta can be measured from day one).

## 7. Blockers / risks

- ⚠️ **Ayush has 0 commits** — individual viva risk. Claim a vertical + start committing.
- ⚠️ 50% of grade (Eng+Eval) unstarted with finite weeks left — schedule risk.
- ⚠️ Dataset undecided/unprofiled — blocks risk model *and* golden set.
- ⚠️ LLM-as-judge not yet calibrated — an uncalibrated judge is itself a quality risk.

## 8. Open decisions (mirror of DECISIONS.md "Proposed/Open")

- North Star definition (TDR) — **propose → ratify**.
- Dataset = Home Credit — **propose → ratify**.
- Deployment target (Render / Railway / Fly.io).
- Retry cap N tuning; DSPy optimizer choice; fairness proxy methodology; Postgres vs SQLite; Chroma cloud vs self-host.

## 9. Individual contribution log (for the viva)

| Member | Owns | First commit | Recent work |
|---|---|---|---|
| Harshit | Risk model + dataset | ✅ | Uploaded PRD/TRD/research, diagrams |
| Himkar | Gateway + synthesizer + DSPy | ✅ | Uploaded HLD/LLD/schema diagrams |
| Aditya | Orchestrator + persistence + API | ❌ | — |
| **Ayush** | **Policy RAG + Faithfulness eval (proposed)** | ❌ | This ops kit |

> Keep this table updated — it is the evidence each person brings to their viva.
