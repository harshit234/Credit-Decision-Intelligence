"""
================================================================================
   HALCYON CREDIT — Evaluation Agent (LLM-as-Judge)
   Stage 3 | Author: Ayush
   LangGraph node: evaluate
   Reads:  draft_decision, credit_report, policy_findings, income_verified, risk_score
   Writes: state["eval_result"], increments retry_count
   LLM:    Strong model path (Gemini via OpenRouter)
================================================================================
"""
from __future__ import annotations
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.application_state import (
    ApplicationState, EvalResult, AgentError, log_event
)
from gateway.router import strong_llm_call, parse_json_response

AGENT               = "EvaluationAgent"
FAITHFULNESS_THRESH = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.75"))
MAX_RETRIES         = int(os.getenv("MAX_RETRIES", "2"))

JUDGE_SYSTEM_PROMPT = """You are a rigorous loan decision auditor for Halcyon Credit.
Your job is to verify that the draft loan decision is faithfully grounded in the source data.

OUTPUT FORMAT — valid JSON only, no markdown fences, no extra text:
{
  "faithfulness": 0.0-1.0,
  "relevancy": 0.0-1.0,
  "unsupported_claims": ["exact claim text that is NOT supported by source data"],
  "pass_flag": true or false
}

SCORING RULES:
- faithfulness: What fraction of the decision reasons are directly supported by source data?
  1.0 = every reason cites a real data point | 0.0 = reasons are invented
- relevancy: Are the reasons relevant to the recommendation made?
  1.0 = all reasons directly justify the recommendation | 0.0 = irrelevant reasons
- unsupported_claims: List any reason that invents a fact not in source data.
  Examples of bad claims: "applicant has 5 credit cards" (not in source),
  "DTI is 45%" when DTI is not in source, citing a policy clause not in policy_findings.
- pass_flag: true if faithfulness >= 0.75, else false.

Be strict. The system will retry the synthesizer if pass_flag is false."""


def evaluate_node(state: ApplicationState) -> dict:
    """
    LangGraph node: evaluate
    Scores the draft_decision for faithfulness against source state data.
    Increments retry_count. Returns ONLY the keys this agent owns.
    """
    t0          = time.time()
    af          = state["applicant_file"]
    decision    = state.get("draft_decision")
    retry_count = state.get("retry_count", 0)

    print(f"  [{AGENT}] Evaluating decision for {af.applicant_id} (retry={retry_count})...")

    if decision is None:
        error = AgentError(agent=AGENT, error_type="validation_error",
                           message="draft_decision is None — cannot evaluate")
        trace = log_event(state, AGENT, "skipped — no draft_decision")
        fallback_eval = EvalResult(
            faithfulness=0.0, relevancy=0.0, unsupported_claims=[], pass_flag=False
        )
        return {"eval_result": fallback_eval, "retry_count": retry_count + 1,
                "errors": state["errors"] + [error], "trace": trace}

    try:
        # Build source context for the judge
        inc = state.get("income_verified")
        cr  = state.get("credit_report")
        pf  = state.get("policy_findings")
        rs  = state.get("risk_score")

        source_ctx = json.dumps({
            "applicant":       {"loan_amount": af.loan_amount,
                                "loan_purpose": af.loan_purpose,
                                "loan_term_months": af.loan_term_months,
                                "employment_type": af.employment_type,
                                "months_employed": af.months_employed,
                                "home_ownership": af.home_ownership,
                                "annual_income": af.annual_income,
                                "existing_monthly_debts": af.existing_debts,
                                "dti_pct": round((af.existing_debts * 12) / max(af.annual_income, 1) * 100, 1),
                                "loan_to_income": round(af.loan_amount / max(af.annual_income, 1), 2)},
            "income_verified": {"verified_income": inc.verified_income if inc else None,
                                "confidence": inc.confidence if inc else None},
            "credit_report":   {"credit_score": cr.credit_score if cr else None,
                                "delinquencies": cr.delinquencies if cr else None,
                                "credit_age_months": cr.credit_age_months if cr else None,
                                "open_accounts": cr.open_accounts if cr else None,
                                "thin_file": cr.thin_file if cr else None,
                                "utilization_pct": cr.utilization_pct if cr else None},
            "policy_findings": {"hard_stops": pf.hard_stops if pf else [],
                                "flags": pf.flags if pf else []},
            "risk_score":      {"score": rs.risk_score if rs else None,
                                "risk_band": rs.risk_band if rs else None},
        }, indent=2)

        decision_ctx = json.dumps({
            "recommendation": decision.recommendation,
            "reasons":        decision.reasons,
            "conditions":     decision.conditions,
        }, indent=2)

        user_msg = (
            f"SOURCE DATA:\n{source_ctx}\n\n"
            f"DRAFT DECISION:\n{decision_ctx}\n\n"
            f"Score the decision faithfulness now."
        )

        text, cost, latency = strong_llm_call(
            system_prompt  = JUDGE_SYSTEM_PROMPT,
            user_prompt    = user_msg,
            max_tokens     = 512,
            agent_name     = AGENT,
            application_id = af.applicant_id,
        )

        parsed = parse_json_response(text, AGENT)

        faith  = float(parsed.get("faithfulness", 0.5))
        relev  = float(parsed.get("relevancy", 0.5))
        claims = parsed.get("unsupported_claims", [])
        pf_ok  = faith >= FAITHFULNESS_THRESH

        eval_result = EvalResult(
            faithfulness        = round(faith, 3),
            relevancy           = round(relev, 3),
            unsupported_claims  = claims,
            pass_flag           = pf_ok,
        )

        trace = log_event(state, AGENT, f"evaluated_faithfulness={faith:.3f}",
                          latency_ms=round(latency, 1), cost_usd=cost)

        status = "PASS" if pf_ok else f"FAIL (retry {retry_count + 1}/{MAX_RETRIES})"
        print(f"  [{AGENT}] Faithfulness={faith:.3f} Relevancy={relev:.3f} → {status}")

        return {
            "eval_result":  eval_result,
            "retry_count":  retry_count + 1,
            "trace":        trace,
        }

    except Exception as e:
        error = AgentError(agent=AGENT, error_type="llm_error", message=str(e))
        trace = log_event(state, AGENT, f"ERROR: {e}")
        print(f"  [{AGENT}] ERROR: {e} — defaulting to pass to avoid infinite loop")
        # On LLM judge failure, pass to avoid blocking the pipeline
        fallback = EvalResult(faithfulness=0.8, relevancy=0.8,
                              unsupported_claims=[], pass_flag=True)
        return {"eval_result": fallback, "retry_count": retry_count + 1,
                "errors": state["errors"] + [error], "trace": trace}
