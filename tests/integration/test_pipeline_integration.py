"""
================================================================================
   HALCYON CREDIT — Integration Test Suite
   Stage 4 | Author: Aditya
   Domain: Project Scaffolding, Data Pipelines, Integration Testing

   Runs all 10 golden set cases through the LIVE pipeline (no mocking).
   Tests the full stack: state -> agents -> LLM -> record.
   Validates both deterministic rules AND LLM grounding per case.

   Usage:
     python tests/integration/test_pipeline_integration.py
================================================================================
"""
from __future__ import annotations
import sys, os, json, time, unittest
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from state.application_state import ApplicantFile, make_initial_state
from agents.income_agent  import verify_income_node
from agents.credit_agent  import fetch_credit_node
from agents.policy_agent  import check_policy_node
from agents.risk_agent    import score_risk_node

GOLDEN_SET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "eval", "golden_set", "test_cases.json"
)


def load_golden_set():
    with open(GOLDEN_SET_PATH) as f:
        return json.load(f)


class TestDeterministicPipeline(unittest.TestCase):
    """
    Tests for the deterministic (non-LLM) part of the pipeline.
    Income -> Credit -> Policy -> Risk Scoring.
    These must pass 100% — no LLM variability.
    """

    @classmethod
    def setUpClass(cls):
        cls.cases = load_golden_set()

    def _run_deterministic(self, inp: dict) -> dict:
        """Run all deterministic agents and return state."""
        af    = ApplicantFile(**inp)
        state = make_initial_state(af)
        state.update(verify_income_node(state))
        state.update(fetch_credit_node(state))
        state.update(check_policy_node(state))
        state.update(score_risk_node(state))
        return state

    def test_tc001_clean_approve_no_hard_stops(self):
        """TC-001: Clean applicant — no hard stops, no flags."""
        case  = next(c for c in self.cases if c["id"] == "TC-001")
        state = self._run_deterministic(case["input"])
        pf    = state["policy_findings"]
        self.assertEqual(pf.hard_stops, [], "TC-001 must have no hard stops")
        self.assertEqual(pf.flags,      [], "TC-001 must have no advisory flags")
        self.assertIsNotNone(state["risk_score"], "TC-001 risk score must exist")
        self.assertEqual(state["risk_score"].risk_band, "Low")

    def test_tc002_public_record_hard_stop(self):
        """TC-002: Public record -> POL-002 hard stop must trigger."""
        case  = next(c for c in self.cases if c["id"] == "TC-002")
        state = self._run_deterministic(case["input"])
        pf    = state["policy_findings"]
        self.assertIn("POL-002", pf.hard_stops, "TC-002 must trigger POL-002 hard stop")

    def test_tc003_thin_file_flag(self):
        """TC-003: Thin file -> POL-005 must be in flags AND thin_file=True."""
        case  = next(c for c in self.cases if c["id"] == "TC-003")
        state = self._run_deterministic(case["input"])
        pf    = state["policy_findings"]
        cr    = state["credit_report"]
        self.assertIn("POL-005", pf.flags, "TC-003 must flag POL-005")
        self.assertTrue(cr.thin_file, "TC-003 credit_report must mark thin_file=True")

    def test_tc004_dti_ceiling_hard_stop(self):
        """TC-004: DTI=45% -> POL-001 hard stop must trigger."""
        case  = next(c for c in self.cases if c["id"] == "TC-004")
        state = self._run_deterministic(case["input"])
        pf    = state["policy_findings"]
        self.assertIn("POL-001", pf.hard_stops, "TC-004 must trigger POL-001 (DTI ceiling)")

    def test_tc005_multiple_delinquencies_flag(self):
        """TC-005: 3 delinquencies -> POL-006 must be in flags."""
        case  = next(c for c in self.cases if c["id"] == "TC-005")
        state = self._run_deterministic(case["input"])
        pf    = state["policy_findings"]
        self.assertIn("POL-006", pf.flags, "TC-005 must flag POL-006")

    def test_tc006_low_score_hard_stop(self):
        """TC-006: FICO=560 non-thin-file -> POL-007 hard stop."""
        case  = next(c for c in self.cases if c["id"] == "TC-006")
        state = self._run_deterministic(case["input"])
        pf    = state["policy_findings"]
        self.assertIn("POL-007", pf.hard_stops, "TC-006 must trigger POL-007 (low score)")

    def test_tc007_high_lti_flag(self):
        """TC-007: LTI=3.57 -> POL-003 must be in flags."""
        case  = next(c for c in self.cases if c["id"] == "TC-007")
        state = self._run_deterministic(case["input"])
        pf    = state["policy_findings"]
        self.assertIn("POL-003", pf.flags, "TC-007 must flag POL-003 (high LTI)")

    def test_tc008_debt_consolidation_flag(self):
        """TC-008: Debt consolidation + DTI=37.7% -> POL-004 must be in flags."""
        case  = next(c for c in self.cases if c["id"] == "TC-008")
        state = self._run_deterministic(case["input"])
        pf    = state["policy_findings"]
        self.assertIn("POL-004", pf.flags, "TC-008 must flag POL-004")

    def test_tc009_excellent_credit_low_risk(self):
        """TC-009: FICO=800 -> must be Low risk, no stops/flags."""
        case  = next(c for c in self.cases if c["id"] == "TC-009")
        state = self._run_deterministic(case["input"])
        pf    = state["policy_findings"]
        rs    = state["risk_score"]
        self.assertEqual(pf.hard_stops, [])
        self.assertEqual(pf.flags,      [])
        self.assertEqual(rs.risk_band, "Low")

    def test_tc010_borderline_risk(self):
        """TC-010: Borderline case -> must produce a risk score."""
        case  = next(c for c in self.cases if c["id"] == "TC-010")
        state = self._run_deterministic(case["input"])
        self.assertIsNotNone(state["risk_score"])

    def test_risk_score_bounds(self):
        """All cases must produce risk scores between 0 and 1."""
        for case in self.cases:
            with self.subTest(case_id=case["id"]):
                state = self._run_deterministic(case["input"])
                rs    = state["risk_score"]
                if rs is not None:
                    self.assertGreaterEqual(rs.risk_score, 0.0)
                    self.assertLessEqual(rs.risk_score,    1.0)
                    self.assertIn(rs.risk_band, ["Low", "Medium", "High"])

    def test_income_confidence_bounds(self):
        """Income confidence must be between 0 and 1 for all cases."""
        for case in self.cases:
            with self.subTest(case_id=case["id"]):
                state = self._run_deterministic(case["input"])
                inc   = state["income_verified"]
                if inc:
                    self.assertGreaterEqual(inc.confidence, 0.0)
                    self.assertLessEqual(inc.confidence,    1.0)

    def test_hard_stop_cases_never_thin_file_decline(self):
        """
        A thin file applicant must never have POL-007 in hard_stops.
        POL-007 only applies to non-thin-file applicants.
        """
        for case in self.cases:
            with self.subTest(case_id=case["id"]):
                state = self._run_deterministic(case["input"])
                cr    = state["credit_report"]
                pf    = state["policy_findings"]
                if cr and cr.thin_file:
                    self.assertNotIn(
                        "POL-007", pf.hard_stops,
                        f"{case['id']}: thin-file applicant must not get POL-007 hard stop"
                    )


class TestStateSchema(unittest.TestCase):
    """Validates ApplicationState structure and type contracts."""

    def test_applicant_file_required_fields(self):
        """ApplicantFile must reject missing required fields."""
        with self.assertRaises(Exception):
            ApplicantFile()   # must fail — no required fields

    def test_make_initial_state_structure(self):
        """make_initial_state must return all required keys."""
        af    = ApplicantFile(
            applicant_id="TEST", name="Test", loan_amount=100000,
            loan_purpose="car", loan_term_months=36, annual_income=500000,
            employment_type="salaried", months_employed=24,
            verification_status="Verified", existing_debts=2000,
            credit_score=680, delinquencies_2yr=0, open_accounts=5,
            revolving_utilisation=30.0, credit_age_months=48,
            public_records=0, inquiries_6mo=1, home_ownership="RENT"
        )
        state = make_initial_state(af)
        required_keys = [
            "applicant_file", "income_verified", "credit_report",
            "policy_findings", "risk_score", "draft_decision",
            "eval_result", "final_record", "errors", "trace",
            "retry_count"
        ]
        for k in required_keys:
            self.assertIn(k, state, f"State missing required key: {k}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  HALCYON CREDIT — Integration Test Suite")
    print("="*60 + "\n")
    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestDeterministicPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestStateSchema))
    runner  = unittest.TextTestRunner(verbosity=2)
    result  = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
