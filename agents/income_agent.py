"""
================================================================================
   HALCYON CREDIT — Income Verification Agent
   Stage 3 | Author: Aditya
   LangGraph node: verify_income
   Reads:  state["applicant_file"]
   Writes: state["income_verified"]
================================================================================
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.application_state import ApplicationState, IncomeResult, AgentError, log_event
from tools.income_db_tool import lookup_income

AGENT = "IncomeVerificationAgent"


def verify_income_node(state: ApplicationState) -> dict:
    """
    LangGraph node: verify_income
    Calls the Income DB tool and writes IncomeResult to state.
    Returns ONLY the keys this agent owns.
    """
    t0 = time.time()
    af = state["applicant_file"]
    print(f"  [{AGENT}] Verifying income for applicant {af.applicant_id}...")

    try:
        record = lookup_income(
            applicant_id        = af.applicant_id,
            stated_income       = af.annual_income,
            verification_status = af.verification_status,
        )

        result = IncomeResult(
            verified_income = record.verified_income,
            confidence      = record.confidence,
            source_refs     = [f"{record.source}::{record.applicant_id}"],
        )

        latency = (time.time() - t0) * 1000
        trace   = log_event(state, AGENT, "income_verified", latency_ms=round(latency, 1))

        print(f"  [{AGENT}] Verified: {record.verified_income:,.0f} "
              f"(confidence: {record.confidence:.0%}, source: {record.source})")
        return {"income_verified": result, "trace": trace}

    except Exception as e:
        error = AgentError(agent=AGENT, error_type="tool_error", message=str(e))
        trace = log_event(state, AGENT, f"ERROR: {e}")
        print(f"  [{AGENT}] ERROR: {e}")
        return {"income_verified": None, "errors": state["errors"] + [error], "trace": trace}
