# Operations Runbook — Halcyon Credit Agentic Underwriting Copilot

> **Status: draft / target.** Fleshed out as the system is built and deployed (Sprint 4). Captures how we
> operate, monitor, and recover the service so it's defensible at the viva and usable by an on-call owner.

## 1. Service overview

- **What:** FastAPI service wrapping a LangGraph agent pipeline that returns auditable underwriting recommendations.
- **Entry points:** `POST /applications`, `GET /applications/{audit_id}`, `GET /health`, `GET /metrics`.
- **Dependencies:** LiteLLM gateway (LLM providers), ChromaDB (policy RAG), decision DB (Postgres/SQLite).
- **Framing:** copilot — outputs are recommendations for a human underwriter, never binding decisions.

## 2. Service Level Objectives (SLOs)

| SLO | Target | Source |
|---|---|---|
| End-to-end latency (p95) | ≤ 30 s | `/metrics` histogram |
| Availability | ≥ 99% | uptime monitor |
| Cost per application | ≤ $0.10 | LiteLLM cost log |
| Escalation rate | ≤ 5% | `escalation_total / applications_total` |
| Golden-set decision quality | ≥ 85% | regression gate |
| RAGAS faithfulness | ≥ 0.80 | RAGAS runner |

## 3. Monitoring & alerting

- **Structured logs** (`structlog`, JSON): every line carries `application_id`, `agent_name`, `node`, `trace_id`.
- **Metrics** (`/metrics`, Prometheus): `applications_total`, `latency_seconds`, `cost_usd_total`, `retry_count_total`, `escalation_total`, `eval_faithfulness`.
- **Alarms:**
  - Cost-per-application > `COST_CEILING_USD` → page owner; inspect routing (strong-path overuse?).
  - Escalation rate > 5% in a 1-hour window → investigate judge threshold / upstream tool failures.
  - Faithfulness gauge drop → check recent prompt/model changes; consider rollback.

## 4. Common incidents & response

| Symptom | Likely cause | First actions |
|---|---|---|
| Latency spike | Strong-path overuse, provider slowness | Check LiteLLM routing/cost log; confirm cheap-path for workers; enable/inspect semantic cache |
| Cost spike | Retry storms, wrong routing | Inspect `retry_count_total`; verify `MAX_RETRIES`; check prompt sizes |
| Many escalations | Judge too strict, RAG misses, tool failures | Review `eval_faithfulness`; check policy retrieval recall; check tool error logs |
| Provider outage | LLM API down/rate-limited | LiteLLM fallback should engage; if both fail, requests degrade to human queue |
| RAG returns nothing | ChromaDB down / empty collection | Conservative-default fallback should trigger; rebuild/reconnect collection |

## 5. Human-fallback path

On retry exhaustion or critical tool failure, the orchestrator writes an `escalation` flag and routes the
case to the human review queue (API returns `202 { audit_id, status: "escalated" }`). No silent auto-decline.

## 6. Rollback

- Code: revert the offending merge on `main`; redeploy previous known-good build.
- Prompts/models: prompt + model-version configs are versioned; pin to the last config that passed the golden gate.
- Risk model: model artifact is locked to its feature-schema version; redeploy the prior artifact pair.
- Decision records are **immutable** — rollback never rewrites history; it changes forward behaviour only.

## 7. Deployment

- Target: free cloud tier (Render / Railway / Fly.io — TBD, see DECISIONS.md).
- Secrets via platform env vars / GitHub Actions secrets; never in source.
- Promotion gate: golden-set regression must pass before deploy.

## 8. Audit & compliance

- Every decision persists the full `ApplicationState` trace + `audit_id`; retrievable via `GET /applications/{audit_id}`.
- Records are append-only (soft-delete only) to preserve the audit trail for regulatory replay.

## 9. On-call checklist

1. Check `/health` and `/metrics`.
2. Scan recent structured logs by `trace_id` for the affected applications.
3. Confirm dependency health (LiteLLM, ChromaDB, DB).
4. Apply the matching playbook in §4; if unresolved, roll back (§6) and open an incident note.
