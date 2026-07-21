"""
================================================================================
   HALCYON CREDIT — Decision Synthesizer Agent
   Stage 3 | Author: Ayush
   LangGraph node: synthesize
   Reads:  risk_score, credit_report, policy_findings, income_verified,
           eval_result (on retry — judge feedback appended)
   Writes: state["draft_decision"], increments draft_version
   LLM:    Strong model path (Gemini via OpenRouter)
================================================================================
"""
from __future__ import annotations
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.application_state import (
    ApplicationState, Decision, AgentError, log_event
)
from gateway.router import strong_llm_call, parse_json_response
from gateway.prompts import SYNTHESIZER_SYSTEM_PROMPT_V2

AGENT       = "DecisionSynthesizerAgent"
SYSTEM_PROMPT = SYNTHESIZER_SYSTEM_PROMPT_V2


def _build_context(state: ApplicationState) -> str:
    """Build the structured JSON context for the LLM from current state."""
    af  = state["applicant_file"]
    inc = state.get("income_verified")
    cr  = state.get("credit_report")
    pf  = state.get("policy_findings")
    rs  = state.get("risk_score")
    ev  = state.get("eval_result")

    ctx = {
        "applicant": {
            "id":              af.applicant_id,
            "loan_amount":     af.loan_amount,
            "loan_purpose":    af.loan_purpose,
            "loan_term_months":af.loan_term_months,
            "employment_type": af.employment_type,
            "months_employed": af.months_employed,
            "home_ownership":  af.home_ownership,
            "annual_income":   af.annual_income,
            "existing_monthly_debts": af.existing_debts,
            "dti_pct":         round((af.existing_debts * 12) / max(af.annual_income, 1) * 100, 1),
            "loan_to_income":  round(af.loan_amount / max(af.annual_income, 1), 2),
        },
        "income_verified": {
            "verified_income": inc.verified_income if inc else None,
            "confidence":      inc.confidence if inc else None,
        } if inc else None,
        "credit_report": {
            "credit_score":      cr.credit_score if cr else None,
            "delinquencies":     cr.delinquencies if cr else None,
            "credit_age_months": cr.credit_age_months if cr else None,
            "open_accounts":     cr.open_accounts if cr else None,
            "utilization_pct":   cr.utilization_pct if cr else None,
            "thin_file":         cr.thin_file if cr else None,
        } if cr else None,
        "policy_findings": {
            "hard_stops": pf.hard_stops if pf else [],
            "flags":      pf.flags if pf else [],
            "clauses":    [{"id": c.clause_id, "text": c.text[:100]} for c in (pf.applicable_clauses[:3] if pf else [])],
        },
        "risk_score": {
            "score":     rs.risk_score if rs else None,
            "risk_band": rs.risk_band if rs else None,
            "top_features": [
                {"feature": f.feature, "value": f.value, "direction": f.direction}
                for f in (rs.top_features[:3] if rs else [])
            ],
        } if rs else None,
        "retry_context": None,
    }

    # On retry: append judge feedback so LLM can correct itself
    if ev and not ev.pass_flag and state.get("retry_count", 0) > 0:
        ctx["retry_context"] = {
            "previous_faithfulness": ev.faithfulness,
            "unsupported_claims":    ev.unsupported_claims,
            "instruction":           "Correct the above unsupported claims. Cite data from the source fields.",
        }

    return json.dumps(ctx, indent=2)


def synthesize_node(state: ApplicationState) -> dict:
    """
    LangGraph node: synthesize
    Calls strong LLM to produce a structured APPROVE/DECLINE/REFER decision.
    On retry: appends judge feedback from eval_result to the prompt.
    """
    t0          = time.time()
    af          = state["applicant_file"]
    retry_count = state.get("retry_count", 0)
    draft_ver   = (state.get("draft_decision").draft_version + 1
                   if state.get("draft_decision") else 1)

    print(f"  [{AGENT}] Synthesizing decision (attempt {draft_ver}) for {af.applicant_id}...")

    try:
        context  = _build_context(state)
        user_msg = f"Applicant data:\n{context}\n\nProduce the underwriting decision JSON now."

        text, cost, latency = strong_llm_call(
            system_prompt  = SYSTEM_PROMPT,
            user_prompt    = user_msg,
            max_tokens     = 1024,
            agent_name     = AGENT,
            application_id = af.applicant_id,
        )

        parsed = parse_json_response(text, AGENT)

        # Validate recommendation value
        rec = parsed.get("recommendation", "REFER").upper()
        if rec not in ("APPROVE", "DECLINE", "REFER"):
            rec = "REFER"

        # Enforce hard-stop override
        pf = state.get("policy_findings")
        if pf and pf.hard_stops:
            rec = "DECLINE"

        # Enforce thin-file override (cannot DECLINE thin-file)
        cr = state.get("credit_report")
        if cr and cr.thin_file and rec == "DECLINE":
            pf_flags = pf.flags if pf else []
            if "POL-005" in pf_flags or (not pf):
                rec = "REFER"

        # Ensure at least 3 auditable reasons — supplement with grounded
        # data-cited reasons if the LLM returned fewer.
        reasons = parsed.get("reasons", ["Insufficient data for reasoning."])
        if len(reasons) < 3:
            rs  = state.get("risk_score")
            inc = state.get("income_verified")
            supplemental = []
            if rs:
                supplemental.append(
                    f"Risk model assessed a default probability of {rs.risk_score:.3f}, "
                    f"placing the applicant in the {rs.risk_band} risk band "
                    f"[risk_score.risk_band={rs.risk_band}]."
                )
            if cr:
                supplemental.append(
                    f"Credit profile: score {cr.credit_score} with {cr.delinquencies} "
                    f"delinquencies in 24 months, revolving utilisation "
                    f"{cr.utilization_pct}%, and {cr.credit_age_months} months of credit history "
                    f"[credit_report.credit_score={cr.credit_score}]."
                )
            if inc:
                supplemental.append(
                    f"Verified income of {inc.verified_income:,.0f} "
                    f"(confidence {inc.confidence:.0%}) was used to assess affordability "
                    f"[income_verified.verified_income={inc.verified_income:,.0f}]."
                )
            if pf is not None:
                stops_txt = ", ".join(pf.hard_stops) if pf.hard_stops else "none"
                flags_txt = ", ".join(pf.flags) if pf.flags else "none"
                supplemental.append(
                    f"Policy compliance check: hard stops — {stops_txt}; advisory flags — {flags_txt} "
                    f"[policy_findings.hard_stops={stops_txt}]."
                )
            for s in supplemental:
                if len(reasons) >= 3:
                    break
                reasons.append(s)

        decision = Decision(
            recommendation = rec,
            reasons        = reasons,
            conditions     = parsed.get("conditions", []),
            draft_version  = draft_ver,
        )

        trace = log_event(state, AGENT, f"decision_drafted_v{draft_ver}",
                          latency_ms=round(latency, 1), cost_usd=cost)

        print(f"  [{AGENT}] Decision: {rec} | Reasons: {len(decision.reasons)} | v{draft_ver}")
        return {"draft_decision": decision, "trace": trace}

    except Exception as e:
        error = AgentError(agent=AGENT, error_type="llm_error", message=str(e))
        trace = log_event(state, AGENT, f"ERROR: {e}")
        print(f"  [{AGENT}] ERROR: {e}")
        # Fallback decision on LLM failure
        fallback = Decision(
            recommendation = "REFER",
            reasons        = [f"LLM synthesis failed: {str(e)[:100]}. Routed to human review."],
            conditions     = [],
            draft_version  = draft_ver,
        )
        return {"draft_decision": fallback, "errors": state["errors"] + [error], "trace": trace}
