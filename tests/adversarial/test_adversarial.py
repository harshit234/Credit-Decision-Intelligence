"""
================================================================================
   HALCYON CREDIT — Adversarial & Red-Team Test Suite
   Sprint 3 | Author: Aditya
   Domain: Project Scaffolding, Data Pipelines, Integration Testing

   Tests 6 adversarial input patterns to ensure the pipeline:
   - Never crashes on edge-case inputs
   - Always returns a valid state with a recommendation
   - Correctly enforces hard boundaries (DTI=40%, FICO=580, pub_rec)
   - Rejects structurally invalid inputs at the schema layer

   Usage: python tests/adversarial/test_adversarial.py
================================================================================
"""
from __future__ import annotations
import sys, os, unittest
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from state.application_state import ApplicantFile, make_initial_state
from agents.income_agent  import verify_income_node
from agents.credit_agent  import fetch_credit_node
from agents.policy_agent  import check_policy_node
from agents.risk_agent    import score_risk_node


def _run_det(inp: dict) -> dict:
    """Run deterministic agents only (no LLM cost)."""
    af    = ApplicantFile(**inp)
    state = make_initial_state(af)
    state.update(verify_income_node(state))
    state.update(fetch_credit_node(state))
    state.update(check_policy_node(state))
    state.update(score_risk_node(state))
    return state


BASE = dict(
    applicant_id="ADV-TEST", name="Test User",
    loan_amount=200000, loan_purpose="car",
    loan_term_months=36, annual_income=600000,
    employment_type="salaried", months_employed=24,
    verification_status="Verified", existing_debts=3000,
    credit_score=700, delinquencies_2yr=0, open_accounts=6,
    revolving_utilisation=25.0, credit_age_months=60,
    public_records=0, inquiries_6mo=1, home_ownership="RENT"
)


class TestAdversarialInputs(unittest.TestCase):

    def test_adv01_boundary_dti_exactly_40pct(self):
        """DTI above 40% (40.1%) must trigger POL-001. Exactly 40% does NOT (strict > 40)."""
        # monthly_debt * 12 / annual_income > 40% -> use 40.1%
        # 20050 * 12 / 600000 = 40.1%
        inp = {**BASE,
               "applicant_id": "ADV-001",
               "annual_income": 600000,
               "existing_debts": 20050}  # 20050*12/600000 = 40.1% > 40
        state = _run_det(inp)
        pf = state["policy_findings"]
        self.assertIn("POL-001", pf.hard_stops,
                      "DTI=40.1% must trigger POL-001 hard stop")

        # Also verify exactly 40% does NOT trigger (policy is strict >)
        inp2 = {**BASE, "applicant_id": "ADV-001b",
                "annual_income": 600000, "existing_debts": 20000}  # exactly 40%
        state2 = _run_det(inp2)
        pf2 = state2["policy_findings"]
        self.assertNotIn("POL-001", pf2.hard_stops,
                         "DTI exactly 40% must NOT trigger POL-001 (policy is DTI > 40)")

    def test_adv02_fico_exactly_580_boundary(self):
        """FICO exactly 579 (non-thin-file) must trigger POL-007. Exactly 580 does NOT (policy is < 580)."""
        inp = {**BASE,
               "applicant_id": "ADV-002",
               "credit_score": 579,  # one below the cutoff
               "credit_age_months": 72,
               "open_accounts": 8}
        state = _run_det(inp)
        pf = state["policy_findings"]
        self.assertIn("POL-007", pf.hard_stops,
                      "FICO=579 non-thin-file must trigger POL-007")

        # Also verify FICO=580 does NOT trigger (policy is strict <)
        inp2 = {**BASE, "applicant_id": "ADV-002b",
                "credit_score": 580, "credit_age_months": 72, "open_accounts": 8}
        state2 = _run_det(inp2)
        pf2 = state2["policy_findings"]
        self.assertNotIn("POL-007", pf2.hard_stops,
                         "FICO exactly 580 must NOT trigger POL-007 (policy is credit_score < 580)")

    def test_adv03_public_record_always_hard_stop(self):
        """Even a perfect applicant with 1 public record must get POL-002."""
        inp = {**BASE,
               "applicant_id": "ADV-003",
               "credit_score": 800,
               "annual_income": 2000000,
               "existing_debts": 0,
               "public_records": 1}
        state = _run_det(inp)
        pf = state["policy_findings"]
        self.assertIn("POL-002", pf.hard_stops,
                      "Public record must always produce POL-002 regardless of other metrics")

    def test_adv04_extreme_lti_over_10(self):
        """LTI > 10 (loan=10x annual income) should flag POL-003."""
        inp = {**BASE,
               "applicant_id": "ADV-004",
               "loan_amount": 6000000,
               "annual_income": 600000}   # LTI = 10.0
        state = _run_det(inp)
        pf = state["policy_findings"]
        self.assertIn("POL-003", pf.flags,
                      "LTI=10 must flag POL-003 high LTI advisory")

    def test_adv05_zero_income_pipeline_survives(self):
        """Pipeline must not crash even with annual_income close to zero."""
        # ApplicantFile has gt=0 on annual_income, so use very small value
        inp = {**BASE,
               "applicant_id": "ADV-005",
               "annual_income": 1.0,
               "loan_amount": 500000}
        try:
            state = _run_det(inp)
            # Must complete — decision doesn't matter, no crash
            self.assertIsNotNone(state)
        except Exception as e:
            self.fail(f"Pipeline crashed on near-zero income: {e}")

    def test_adv06_invalid_schema_rejected(self):
        """Negative loan_amount must be rejected by Pydantic schema."""
        from pydantic import ValidationError
        with self.assertRaises((ValidationError, Exception),
                               msg="Negative loan amount must raise a validation error"):
            ApplicantFile(**{**BASE, "loan_amount": -50000})

    def test_adv07_risk_score_always_in_bounds(self):
        """Risk score must always be 0.0 to 1.0 regardless of input extremes."""
        extreme_cases = [
            {**BASE, "applicant_id": "ADV-007a", "credit_score": 850,
             "annual_income": 5000000, "existing_debts": 0},
            {**BASE, "applicant_id": "ADV-007b", "credit_score": 300,
             "annual_income": 60000, "existing_debts": 15000,
             "revolving_utilisation": 99.0, "delinquencies_2yr": 5},
        ]
        for inp in extreme_cases:
            with self.subTest(aid=inp["applicant_id"]):
                state = _run_det(inp)
                rs = state.get("risk_score")
                if rs:
                    self.assertGreaterEqual(rs.risk_score, 0.0)
                    self.assertLessEqual(rs.risk_score, 1.0)

    def test_adv08_no_state_key_is_missing_after_agents(self):
        """All required state keys must be populated after the 4 det agents."""
        state = _run_det({**BASE, "applicant_id": "ADV-008"})
        for key in ["income_verified", "credit_report", "policy_findings", "risk_score"]:
            self.assertIn(key, state, f"Key '{key}' missing after deterministic agents")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  HALCYON — Adversarial / Red-Team Test Suite")
    print("=" * 60 + "\n")
    unittest.main(verbosity=2)
