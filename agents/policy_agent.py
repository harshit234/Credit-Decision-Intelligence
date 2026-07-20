"""
================================================================================
   HALCYON CREDIT — Policy Compliance Agent
   Stage 3 | Author: Aditya
   LangGraph node: check_policy
   Reads:  state["applicant_file"]
   Writes: state["policy_findings"]
   Tools:  ChromaDB policy_retrieval_tool + deterministic rule engine
================================================================================
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.application_state import (
    ApplicationState, PolicyResult, PolicyClause, AgentError, log_event
)
from tools.policy_retrieval_tool import retrieve_policy

AGENT = "PolicyComplianceAgent"


# ─────────────────────────────────────────────────────────────────────────────
# DETERMINISTIC RULE ENGINE — no LLM needed for clear violations
# ─────────────────────────────────────────────────────────────────────────────
def _check_hard_stops(af) -> list[str]:
    """
    Apply hard-stop policy rules deterministically.
    These CANNOT be overridden by the LLM synthesizer.
    Returns list of violated clause IDs.
    """
    violations = []
    annual_inc  = max(af.annual_income, 1)
    dti         = (af.existing_debts * 12) / annual_inc * 100
    thin_file   = (af.credit_age_months < 24) or (af.open_accounts < 3)

    if dti > 40.0:
        violations.append("POL-001")   # DTI ceiling

    if af.public_records >= 1:
        violations.append("POL-002")   # Public record

    if af.credit_score < 580 and not thin_file:
        violations.append("POL-007")   # Low score (non-thin)

    return violations


def _check_advisory_flags(af) -> list[str]:
    """
    Apply advisory (non-hard-stop) policy flags.
    These inform the synthesizer's reasoning.
    """
    flags      = []
    annual_inc = max(af.annual_income, 1)
    dti        = (af.existing_debts * 12) / annual_inc * 100
    lti        = af.loan_amount / annual_inc

    if lti > 3.0:
        flags.append("POL-003")   # LTI high

    if af.loan_purpose == "debt_consolidation" and dti > 35.0:
        flags.append("POL-004")   # Debt consolidation + high DTI

    if (af.credit_age_months < 24) or (af.open_accounts < 3):
        flags.append("POL-005")   # Thin file

    if af.delinquencies_2yr >= 2:
        flags.append("POL-006")   # Multiple delinquencies

    return flags


# ─────────────────────────────────────────────────────────────────────────────
# AGENT NODE
# ─────────────────────────────────────────────────────────────────────────────
def check_policy_node(state: ApplicationState) -> dict:
    """
    LangGraph node: check_policy
    1. Runs deterministic hard-stop + advisory flag rules
    2. Retrieves relevant policy text from ChromaDB for synthesizer context
    3. Writes PolicyResult to state
    """
    t0 = time.time()
    af = state["applicant_file"]
    print(f"  [{AGENT}] Checking policies for {af.applicant_id}...")

    try:
        # Step 1: Deterministic rule checks
        hard_stops = _check_hard_stops(af)
        flags      = _check_advisory_flags(af)

        # Step 2: Semantic retrieval from ChromaDB for context richness
        dti   = (af.existing_debts * 12) / max(af.annual_income, 1) * 100
        query = (
            f"Loan application: purpose={af.loan_purpose}, "
            f"annual_income={af.annual_income:.0f}, loan_amount={af.loan_amount:.0f}, "
            f"credit_score={af.credit_score}, DTI={dti:.1f}%, "
            f"delinquencies={af.delinquencies_2yr}, public_records={af.public_records}, "
            f"employment_type={af.employment_type}"
        )
        raw_chunks = retrieve_policy(query, top_k=5)

        # Step 3: Build PolicyClause objects
        clauses = [
            PolicyClause(
                clause_id    = c.clause_id,
                text         = c.text,
                is_hard_stop = c.is_hard_stop,
                section      = c.section,
                source_ref   = c.chunk_id,
            )
            for c in raw_chunks
        ]

        result = PolicyResult(
            applicable_clauses = clauses,
            hard_stops         = hard_stops,
            flags              = flags,
            source_refs        = [c.chunk_id for c in raw_chunks],
        )

        latency = (time.time() - t0) * 1000
        trace   = log_event(state, AGENT, "policy_checked", latency_ms=round(latency, 1))

        print(f"  [{AGENT}] Hard stops: {hard_stops or 'None'} | Flags: {flags or 'None'} "
              f"| Clauses retrieved: {len(clauses)}")
        return {"policy_findings": result, "trace": trace}

    except Exception as e:
        error = AgentError(agent=AGENT, error_type="tool_error", message=str(e))
        trace = log_event(state, AGENT, f"ERROR: {e}")
        print(f"  [{AGENT}] ERROR: {e}")
        return {"policy_findings": None, "errors": state["errors"] + [error], "trace": trace}
