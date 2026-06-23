# Security Policy

This is an academic research prototype. **Halcyon Credit is fictional** and the system must not be
used for real lending decisions. Even so, we hold it to production-grade security hygiene because the
domain (consumer credit) is high-stakes and the discipline is part of what's being graded.

## Reporting a vulnerability or exposure

Open a **private** report by emailing the team or opening a GitHub security advisory. Do **not** open a
public issue for anything involving leaked credentials, PII, or an exploit. Include steps to reproduce
and the affected file/commit.

## Secrets

- **Zero secrets in source control.** All credentials are supplied via environment variables; see
  [`.env.example`](./.env.example). `.env` is gitignored.
- CI runs dependency and secret scanning. A pushed secret must be **rotated immediately**, not just removed.
- API keys for the LLM gateway live in GitHub Actions secrets for CI, never in code or logs.

## Personal / applicant data (PII)

- No applicant PII (e.g. names) is placed in LLM prompt text in production paths; `applicant_id` is used as a reference.
- The chosen dataset (see [`data/dataset_card.md`](./data/dataset_card.md)) is pseudonymised; we add no re-identifying joins.
- Protected attributes are used **only** for fairness measurement, **never** as direct model inputs.
- Decision records persist a full state trace for audit; access is restricted and records are immutable (append-only).

## Dependencies

- Pinned in [`requirements.txt`](./requirements.txt); `pip-audit` runs in CI.
- Upgrade promptly on advisories; note material changes in [`CHANGELOG.md`](./CHANGELOG.md).

## Supported versions

This project is pre-release (active development). Security fixes target the `main` branch only.
