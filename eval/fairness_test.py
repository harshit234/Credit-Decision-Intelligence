"""
================================================================================
   HALCYON CREDIT — Fairness & Segment Disparity Testing
   Sprint 3 | Author: Harshit
   Domain: ML Model Training, RiskScoring Node, API Integration

   Tests approval/decline/refer rates across 4 demographic-proxy cohorts.
   Flags any segment with > 15% approval gap vs the prime cohort.
   Uses only deterministic agents — no LLM cost.

   Segments:
     1. Prime         — High income, high credit, salaried, verified
     2. Underserved   — Low income, thin file (short credit age, few accounts)
     3. Non-Trad      — Self-employed / gig, moderate income
     4. Impaired      — Prior delinquencies, lower score

   Usage: python eval/fairness_test.py
================================================================================
"""
from __future__ import annotations
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from state.application_state import ApplicantFile, make_initial_state
from agents.income_agent  import verify_income_node
from agents.credit_agent  import fetch_credit_node
from agents.policy_agent  import check_policy_node
from agents.risk_agent    import score_risk_node


# ── Cohort definitions (10 applicants each) ──────────────────────────────────

def _make(base_id: str, overrides: dict) -> dict:
    base = dict(
        loan_amount=300000, loan_purpose="home_improvement",
        loan_term_months=36, existing_debts=5000,
        open_accounts=8, inquiries_6mo=1, home_ownership="RENT"
    )
    base.update(overrides)
    base["applicant_id"] = base_id
    return base


COHORTS = {
    "Prime (High income, high credit)": [
        _make(f"PR-{i:02d}", dict(
            name=f"Prime User {i}", annual_income=1200000 + i*50000,
            employment_type="salaried", months_employed=60,
            verification_status="Source Verified", credit_score=750 + i,
            delinquencies_2yr=0, revolving_utilisation=15.0 + i,
            credit_age_months=120 + i*6, public_records=0
        )) for i in range(1, 11)
    ],

    "Underserved (Low income, thin file)": [
        _make(f"US-{i:02d}", dict(
            name=f"Underserved User {i}", annual_income=250000 + i*10000,
            employment_type="salaried", months_employed=12,
            verification_status="Not Verified", credit_score=640 + i,
            delinquencies_2yr=0, revolving_utilisation=30.0 + i,
            credit_age_months=16 + i,  # thin file (< 24 months)
            open_accounts=2, public_records=0
        )) for i in range(1, 11)
    ],

    "Non-Traditional (Self-employed/gig)": [
        _make(f"NT-{i:02d}", dict(
            name=f"NonTrad User {i}", annual_income=700000 + i*30000,
            employment_type="self-employed", months_employed=30,
            verification_status="Verified", credit_score=690 + i,
            delinquencies_2yr=0, revolving_utilisation=28.0 + i,
            credit_age_months=72 + i*3, public_records=0
        )) for i in range(1, 11)
    ],

    "Impaired (Prior delinquencies)": [
        _make(f"IM-{i:02d}", dict(
            name=f"Impaired User {i}", annual_income=500000 + i*20000,
            employment_type="salaried", months_employed=48,
            verification_status="Verified", credit_score=620 + i,
            delinquencies_2yr=min(i, 3), revolving_utilisation=50.0 + i,
            credit_age_months=60 + i*4, public_records=0
        )) for i in range(1, 11)
    ],
}

FAIRNESS_THRESHOLD = 0.15  # max allowed approval gap vs prime


def _score_applicant(inp: dict) -> dict:
    af    = ApplicantFile(**inp)
    state = make_initial_state(af)
    state.update(verify_income_node(state))
    state.update(fetch_credit_node(state))
    state.update(check_policy_node(state))
    state.update(score_risk_node(state))

    pf = state["policy_findings"]
    rs = state["risk_score"]

    if pf.hard_stops:
        outcome = "DECLINE"
    elif pf.flags:
        outcome = "REFER"
    else:
        outcome = "APPROVE" if (rs and rs.risk_band in ("Low", "Medium")) else "REFER"

    return {
        "applicant_id": inp["applicant_id"],
        "outcome":      outcome,
        "hard_stops":   pf.hard_stops,
        "flags":        pf.flags,
        "risk_score":   rs.risk_score if rs else None,
        "risk_band":    rs.risk_band  if rs else None,
    }


def run_fairness_test():
    print("\n" + "=" * 65)
    print("  HALCYON CREDIT — Fairness & Segment Disparity Test")
    print("=" * 65)

    segment_stats = {}
    all_results   = {}

    for segment, cohort in COHORTS.items():
        results = []
        for inp in cohort:
            try:
                r = _score_applicant(inp)
                results.append(r)
            except Exception as e:
                results.append({"applicant_id": inp["applicant_id"],
                                 "outcome": "ERROR", "error": str(e)})

        approve  = sum(1 for r in results if r["outcome"] == "APPROVE")
        decline  = sum(1 for r in results if r["outcome"] == "DECLINE")
        refer    = sum(1 for r in results if r["outcome"] == "REFER")
        n        = len(results)
        risks    = [r["risk_score"] for r in results if r.get("risk_score")]

        segment_stats[segment] = {
            "n":            n,
            "approve":      approve,
            "decline":      decline,
            "refer":        refer,
            "approve_rate": round(approve / n, 3),
            "decline_rate": round(decline / n, 3),
            "refer_rate":   round(refer / n, 3),
            "avg_risk":     round(sum(risks) / len(risks), 4) if risks else None,
        }
        all_results[segment] = results

    # ── Print table ──────────────────────────────────────────────────────────
    prime_rate = segment_stats["Prime (High income, high credit)"]["approve_rate"]

    print(f"\n  {'Segment':<40} {'N':>4} {'Approve%':>9} {'Decline%':>9} {'Refer%':>7} {'AvgRisk':>8} {'Gap':>6} {'Flag':>5}")
    print("  " + "-" * 93)

    fairness_pass = True
    for seg, s in segment_stats.items():
        gap  = prime_rate - s["approve_rate"]
        flag = "WARN" if gap > FAIRNESS_THRESHOLD else "OK"
        if flag == "WARN":
            fairness_pass = False
        risk_str = f"{s['avg_risk']:.4f}" if s["avg_risk"] else "  N/A"
        print(f"  {seg:<40} {s['n']:>4} {s['approve_rate']:>8.1%} "
              f"{s['decline_rate']:>8.1%} {s['refer_rate']:>6.1%} "
              f"{risk_str:>8} {gap:>+6.1%} {flag:>5}")

    print("\n" + "=" * 65)
    print(f"  Fairness Threshold : <= {FAIRNESS_THRESHOLD:.0%} approval gap vs Prime")
    print(f"  Fairness Gate      : {'PASS' if fairness_pass else 'WARN — gap exceeds threshold'}")
    print("=" * 65)

    if not fairness_pass:
        print("\n  NOTE: Disparity detected — review may be needed before production.")
        print("  Thin-file (underserved) applicants are correctly routed to REFER,")
        print("  not auto-declined. This is by policy design (POL-005).")

    return {"segments": segment_stats, "fairness_pass": fairness_pass}


if __name__ == "__main__":
    run_fairness_test()
