"""
Full end-to-end pipeline test with live LLM (OpenRouter).
Runs a single application through all 8 LangGraph nodes.
Run: python run_live_test.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from state.application_state import ApplicantFile
from graph.pipeline import run_pipeline

def main():
    print("\n" + "="*60)
    print("  HALCYON CREDIT — Full Live Pipeline Test")
    print("="*60)

    af = ApplicantFile(
        applicant_id          = "APL-LIVE-001",
        name                  = "Rohan Mehta",
        loan_amount           = 250000,
        loan_purpose          = "home_improvement",
        loan_term_months      = 36,
        annual_income         = 900000,
        employment_type       = "salaried",
        months_employed       = 48,
        verification_status   = "Source Verified",
        existing_debts        = 5000,
        credit_score          = 720,
        delinquencies_2yr     = 0,
        open_accounts         = 8,
        revolving_utilisation = 22.0,
        credit_age_months     = 84,
        public_records        = 0,
        inquiries_6mo         = 1,
        home_ownership        = "MORTGAGE",
    )

    final_state = run_pipeline(af)

    record   = final_state.get("final_record")
    decision = record.final_decision  if record else None
    risk     = record.risk_result     if record else None
    ev       = record.eval_result     if record else None
    pf       = final_state.get("policy_findings")

    print("\n" + "="*60)
    print("  PIPELINE RESULT")
    print("="*60)
    print(f"  Audit ID      : {record.audit_id if record else 'N/A'}")
    print(f"  Recommendation: {decision.recommendation if decision else 'N/A'}")
    print(f"  Risk Score    : {risk.risk_score:.4f} ({risk.risk_band})" if risk else "  Risk Score    : N/A")
    print(f"  Faithfulness  : {ev.faithfulness:.3f} ({'PASS' if ev and ev.pass_flag else 'FAIL'})" if ev else "  Faithfulness  : N/A")
    print(f"  Retry count   : {final_state.get('retry_count', 0)}")
    print(f"  Escalated     : {record.escalated if record else 'N/A'}")
    print(f"  Cost (USD)    : ${record.cost_usd_total:.5f}" if record else "")
    print(f"  Hard stops    : {pf.hard_stops or 'None'}" if pf else "")
    print(f"  Flags         : {pf.flags or 'None'}" if pf else "")

    if decision and decision.reasons:
        print("\n  Reasons:")
        for i, r in enumerate(decision.reasons, 1):
            print(f"    {i}. {r}")

    if decision and decision.conditions:
        print("\n  Conditions:")
        for c in decision.conditions:
            print(f"    - {c}")

    if risk and risk.top_features:
        print("\n  Top Risk Features (SHAP):")
        for f in risk.top_features:
            print(f"    {f.feature:<30} val={f.value:<10} shap={f.shap_value:+.4f}  [{f.direction}]")

    if ev and ev.unsupported_claims:
        print(f"\n  Unsupported claims flagged by judge:")
        for c in ev.unsupported_claims:
            print(f"    - {c}")

    print("\n" + "="*60)
    print("  Full pipeline test COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
