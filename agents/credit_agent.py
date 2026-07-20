"""
================================================================================
   HALCYON CREDIT — Credit History Agent
   Stage 3 | Author: Aditya
   LangGraph node: fetch_credit
   Reads:  state["applicant_file"]
   Writes: state["credit_report"]
================================================================================
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.application_state import ApplicationState, CreditResult, AgentError, log_event
from tools.credit_bureau_tool import fetch_bureau

AGENT = "CreditHistoryAgent"


def fetch_credit_node(state: ApplicationState) -> dict:
    """
    LangGraph node: fetch_credit
    Calls Credit Bureau tool and writes CreditResult to state.
    Returns ONLY the keys this agent owns.
    """
    t0 = time.time()
    af = state["applicant_file"]
    print(f"  [{AGENT}] Fetching bureau data for {af.applicant_id}...")

    try:
        bureau_input = {
            "credit_score":          af.credit_score,
            "delinquencies_2yr":     af.delinquencies_2yr,
            "credit_age_months":     af.credit_age_months,
            "open_accounts":         af.open_accounts,
            "revolving_utilisation": af.revolving_utilisation,
            "verification_status":   af.verification_status,
        }

        record = fetch_bureau(af.applicant_id, bureau_input)

        result = CreditResult(
            credit_score      = record.credit_score,
            delinquencies     = record.delinquencies,
            credit_age_months = record.credit_age_months,
            open_accounts     = record.open_accounts,
            utilization_pct   = record.utilization_pct,
            thin_file         = record.thin_file,
            source_refs       = [record.bureau_ref_id],
        )

        latency = (time.time() - t0) * 1000
        trace   = log_event(state, AGENT, "credit_fetched", latency_ms=round(latency, 1))

        tf_flag = " [THIN FILE]" if record.thin_file else ""
        print(f"  [{AGENT}] Score={record.credit_score} Delinq={record.delinquencies} "
              f"Age={record.credit_age_months}mo Util={record.utilization_pct:.1f}%{tf_flag}")
        return {"credit_report": result, "trace": trace}

    except Exception as e:
        error = AgentError(agent=AGENT, error_type="tool_error", message=str(e))
        trace = log_event(state, AGENT, f"ERROR: {e}")
        print(f"  [{AGENT}] ERROR: {e}")
        return {"credit_report": None, "errors": state["errors"] + [error], "trace": trace}
