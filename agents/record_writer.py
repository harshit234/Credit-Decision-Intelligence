"""
================================================================================
   HALCYON CREDIT — Decision Record Writer
   Stage 3 | Author: Harshit
   LangGraph node: write_record
   Reads:  draft_decision, eval_result, risk_score, policy_findings, trace, errors
   Writes: state["final_record"]
   Persists full DecisionRecord to SQLite (dev) / Postgres (prod).
   Records are IMMUTABLE after write — no UPDATE operations permitted.
================================================================================
"""
from __future__ import annotations
import sys, os, time, uuid, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.application_state import (
    ApplicationState, DecisionRecord, AgentError, log_event
)
from tools.decision_record_tool import persist_record

AGENT = "RecordWriterAgent"


def write_record_node(state: ApplicationState) -> dict:
    """
    LangGraph node: write_record
    Assembles the full DecisionRecord from state and persists it.
    Generates a UUID audit_id returned to the client.
    This node is always reached — even on escalation paths.
    """
    t0      = time.time()
    af      = state["applicant_file"]
    decision = state.get("draft_decision")
    ev       = state.get("eval_result")
    rs       = state.get("risk_score")
    pf       = state.get("policy_findings")
    errors   = state.get("errors", [])

    print(f"  [{AGENT}] Writing decision record for {af.applicant_id}...")

    # Determine escalation status
    # Escalated if: eval failed AND retry_count >= MAX_RETRIES, or critical errors
    max_retries   = int(os.getenv("MAX_RETRIES", "2"))
    retry_count   = state.get("retry_count", 0)
    escalated     = (ev is not None and not ev.pass_flag and retry_count >= max_retries) or \
                    any(e.error_type == "escalation" for e in errors)

    # Total cost across all LLM calls
    cost_total = sum(e.cost_usd or 0.0 for e in state.get("trace", []))

    try:
        record = DecisionRecord(
            audit_id       = str(uuid.uuid4()),
            application_id = af.applicant_id,
            final_decision = decision,
            eval_result    = ev,
            risk_result    = rs,
            policy_refs    = pf.source_refs if pf else [],
            full_trace     = state.get("trace", []),
            errors         = errors,
            escalated      = escalated,
            cost_usd_total = round(cost_total, 6),
        )

        # Persist to database
        persist_record(record)

        latency = (time.time() - t0) * 1000
        trace   = log_event(state, AGENT, "record_written", latency_ms=round(latency, 1))

        status = "ESCALATED" if escalated else decision.recommendation if decision else "UNKNOWN"
        print(f"  [{AGENT}] Record written. audit_id={record.audit_id} "
              f"decision={status} cost=${cost_total:.5f}")

        return {"final_record": record, "trace": trace}

    except Exception as e:
        error = AgentError(agent=AGENT, error_type="persistence_error", message=str(e))
        trace = log_event(state, AGENT, f"ERROR: {e}")
        print(f"  [{AGENT}] ERROR writing record: {e}")
        return {"final_record": None, "errors": errors + [error], "trace": trace}
