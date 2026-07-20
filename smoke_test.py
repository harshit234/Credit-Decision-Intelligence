"""
Quick smoke test for the Halcyon pipeline (no LLM required).
Tests: income agent, credit agent, policy agent, risk scoring.
Run: python smoke_test.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from state.application_state import ApplicantFile, make_initial_state
from agents.income_agent import verify_income_node
from agents.credit_agent import fetch_credit_node
from agents.policy_agent import check_policy_node
from agents.risk_agent import score_risk_node

def run():
    af = ApplicantFile(
        applicant_id="TEST-001", name="Test User",
        loan_amount=200000, loan_purpose="home_improvement", loan_term_months=36,
        annual_income=800000, employment_type="salaried", months_employed=48,
        verification_status="Source Verified", existing_debts=5000,
        credit_score=720, delinquencies_2yr=0, open_accounts=8,
        revolving_utilisation=25.0, credit_age_months=72,
        public_records=0, inquiries_6mo=1, home_ownership="MORTGAGE"
    )

    state = make_initial_state(af)

    # ── Data agents ─────────────────────────────────────────────────────
    inc_res = verify_income_node(state)
    state.update(inc_res)

    cred_res = fetch_credit_node(state)
    state.update(cred_res)

    pol_res = check_policy_node(state)
    state.update(pol_res)

    inc  = state["income_verified"]
    cr   = state["credit_report"]
    pf   = state["policy_findings"]

    print("=== Data Agents ===")
    print(f"Verified income  : {inc.verified_income:,.0f}  (confidence: {inc.confidence:.0%})")
    print(f"Credit score     : {cr.credit_score}")
    print(f"Thin file        : {cr.thin_file}")
    print(f"Hard stops       : {pf.hard_stops or 'None'}")
    print(f"Advisory flags   : {pf.flags or 'None'}")

    # ── Risk scoring ─────────────────────────────────────────────────────
    risk_res = score_risk_node(state)
    state.update(risk_res)

    rs = state["risk_score"]
    if rs:
        print("\n=== Risk Scoring ===")
        print(f"Risk score   : {rs.risk_score:.4f}")
        print(f"Risk band    : {rs.risk_band}")
        print(f"Model ver    : {rs.model_version}")
        if rs.top_features:
            print(f"Top feature  : {rs.top_features[0].feature} = {rs.top_features[0].value} (shap={rs.top_features[0].shap_value:.4f})")
    else:
        print("\n[Risk agent] No score returned (model may not be loaded)")

    print("\nSmoke test PASSED")

if __name__ == "__main__":
    run()
