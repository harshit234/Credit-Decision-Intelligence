# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/); this project uses sprint-based milestones.

## [Unreleased]

### Added — Repository foundation (Sprint 1)
- World-class `README.md` (problem, solution, differentiator matrix, architecture, evaluation philosophy, responsible-AI, quickstart, roadmap).
- Project governance kit: `CLAUDE.md`, `PROGRESS.md`, `DECISIONS.md`, `ROADMAP.md`, `EVALUATION_GAP_ANALYSIS.md`, `COWORK_PROJECT_INSTRUCTIONS.md`.
- Repo hygiene: `.gitignore`, `LICENSE` (academic), `CONTRIBUTING.md`, `SECURITY.md`, `CODEOWNERS`, this `CHANGELOG.md`.
- Python project config: `requirements.txt`, `pyproject.toml` (ruff + mypy + pytest), `Makefile`, `.env.example`.
- CI: GitHub Actions workflow (`lint` → `test` → golden-set gate placeholder).
- Documentation: `docs/risk_register.md`, `docs/runbook.md`, `data/dataset_card.md`.

### Decided
- North Star = **Trusted Decision Rate (TDR)** composite (ADR-005, proposed).
- Risk-model dataset = **Home Credit Default Risk** (ADR-006, proposed).
- See `DECISIONS.md` for the full ADR log.

## [Sprint 0] — Discover & Define
### Added
- `PRD.md` — problem, personas, metrics with targets, risk register.
- `research_market_review.md` — Upstart / Zest AI / Agentic APA prior-art teardown.
- Architecture diagrams: `01_system_architecture.png`, `02_sequence_flow.png`, `03_state_schema.png`, `Agentic Architecture.png`.
- `TRD.md` — agent contracts, state schema, tool specs, LiteLLM routing, CI plan, SLOs.
- Initial `README.md`.

---

> Upcoming (Sprint 2+): end-to-end pipeline, LiteLLM gateway, golden set, single-LLM baseline, RAGAS, DSPy, fairness suite, deployment. Tracked in `ROADMAP.md`.
