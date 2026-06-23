# Contributing to Halcyon Credit — Agentic Underwriting Copilot

Welcome, Team Jamun. This guide keeps our work consistent, reviewable, and viva-ready.
The authoritative operating contract is [`CLAUDE.md`](./CLAUDE.md) — read it first.

## Golden rules

1. **Read before you write.** Start every session with `CLAUDE.md` → `PROGRESS.md` → `DECISIONS.md` → `ROADMAP.md`.
2. **Name the rubric bucket** your change advances (Engineering / Quality-Eval / Architecture / Product / Demo).
3. **Schema-first.** Define/extend the typed `ApplicationState` (TRD §3) before agent logic. Breaking a contract = TRD version bump + an ADR.
4. **Grounding is mandatory.** Every worker output carries `source_refs`; every decision reason cites a source. Unsupported claims are bugs.
5. **No secrets, ever.** Env vars only; `.env` is gitignored; `.env.example` documents required vars.
6. **Honesty over optics.** Report real numbers — including unflattering ones. Fabrication fails the viva.

## Workflow

```bash
git checkout -b feat/<area>-<short>     # or fix/, docs/, eval/, ops/, ci/
make setup
# ... make changes ...
make lint && make test                  # must pass locally
git commit -m "feat(policy-agent): RAG retrieval with clause source_refs"
git push -u origin feat/<area>-<short>
# open a PR; request review from a non-author
```

### Branch naming
`feat/…` `fix/…` `docs/…` `eval/…` `ops/…` `ci/…` `data/…` — one coherent unit per branch.

### Commit messages (Conventional Commits)
`type(scope): imperative summary` — e.g. `feat(gateway): add LiteLLM fallback chain`.
Types: `feat` `fix` `docs` `test` `refactor` `chore` `ci` `eval` `data` `perf`.

### Pull requests
- Small and focused; reviewed by a **non-author** (so everyone can speak to the whole system at the viva).
- PR description states the **rubric bucket** advanced and any **eval impact**.
- CI must be green; the **golden-set gate** must pass; no secrets; no large data/artifacts.
- Prefer a **merge commit** (not squash) when preserving a meaningful multi-commit sequence.

## Definition of Done
- Typed I/O, unit test, wired into the graph, structured logging, no secrets.
- For eval work: a reproducible script + committed results + an honest one-paragraph interpretation.
- `PROGRESS.md` updated; an ADR appended to `DECISIONS.md` if a decision was made.

## Code style
- Python 3.11+, fully type-annotated. `ruff` for lint/format, `mypy` for types (`make lint`).
- Pydantic models for all agent/tool I/O. Deterministic seeds for any sampling or synthetic data.

## Decisions & memory
Anything that changes the plan, architecture, metrics, scope, or ownership goes in the **repo**
(`PROGRESS.md` / `DECISIONS.md`) — not just chat or personal notes. The repo is the source of truth.

## Individual ownership (viva gate)
Each member owns and must defend a vertical (see [`CLAUDE.md` §10](./CLAUDE.md)). Keep your contribution
log in [`PROGRESS.md`](./PROGRESS.md) §9 current — it's your viva evidence.

Thanks for keeping Halcyon world-class. 🏦
