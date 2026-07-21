"""
================================================================================
   HALCYON CREDIT — CI Regression Gate
   Sprint 3 | Author: Aditya
   Domain: Project Scaffolding, Data Pipelines, Regression Suite

   Lightweight deterministic-only regression runner.
   Designed to run in CI/CD before every merge to dev.
   No LLM calls — zero API cost.

   Exit code 0 = all pass. Exit code 1 = regression detected.

   Usage: python tests/run_regression.py
================================================================================
"""
from __future__ import annotations
import sys, os, json, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from state.application_state import ApplicantFile, make_initial_state
from agents.income_agent  import verify_income_node
from agents.credit_agent  import fetch_credit_node
from agents.policy_agent  import check_policy_node
from agents.risk_agent    import score_risk_node

GOLDEN_SET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "eval", "golden_set", "test_cases.json"
)

# ── What we assert deterministically (policy rule outcomes) ──────────────────
DETERMINISTIC_ASSERTIONS = {
    "TC-001": {"hard_stops": [],       "flags_include": []},
    "TC-002": {"hard_stops": ["POL-002"], "flags_include": []},
    "TC-003": {"hard_stops": [],       "flags_include": ["POL-005"]},
    "TC-004": {"hard_stops": ["POL-001"], "flags_include": []},
    "TC-005": {"hard_stops": [],       "flags_include": ["POL-006"]},
    "TC-006": {"hard_stops": ["POL-007"], "flags_include": []},
    "TC-007": {"hard_stops": [],       "flags_include": ["POL-003"]},
    "TC-008": {"hard_stops": [],       "flags_include": ["POL-004"]},
    "TC-009": {"hard_stops": [],       "flags_include": []},
    "TC-010": {"hard_stops": [],       "flags_include": []},
}


def run_regression() -> bool:
    with open(GOLDEN_SET_PATH) as f:
        cases = json.load(f)

    passed = 0
    failed = 0
    failures = []

    print("\n" + "=" * 60)
    print("  HALCYON — CI Regression Gate (deterministic only)")
    print("=" * 60)

    for case in cases:
        cid = case["id"]
        if cid not in DETERMINISTIC_ASSERTIONS:
            continue

        expected = DETERMINISTIC_ASSERTIONS[cid]
        t0 = time.time()

        try:
            af    = ApplicantFile(**case["input"])
            state = make_initial_state(af)
            state.update(verify_income_node(state))
            state.update(fetch_credit_node(state))
            state.update(check_policy_node(state))
            state.update(score_risk_node(state))

            pf = state["policy_findings"]
            elapsed = round(time.time() - t0, 2)

            errs = []
            for stop in expected["hard_stops"]:
                if stop not in pf.hard_stops:
                    errs.append(f"Expected hard stop {stop} not found")
            for flag in expected["flags_include"]:
                if flag not in pf.flags:
                    errs.append(f"Expected flag {flag} not found")
            # Ensure no unexpected hard stops on clean cases
            if not expected["hard_stops"] and pf.hard_stops:
                errs.append(f"Unexpected hard stops: {pf.hard_stops}")

            if errs:
                failed += 1
                failures.append({"id": cid, "errors": errs})
                print(f"  [FAIL] {cid} ({elapsed}s) — {'; '.join(errs)}")
            else:
                passed += 1
                print(f"  [PASS] {cid} ({elapsed}s)")

        except Exception as e:
            failed += 1
            failures.append({"id": cid, "errors": [str(e)]})
            print(f"  [ERR]  {cid} — {e}")

    total = passed + failed
    print("\n" + "=" * 60)
    print(f"  Result: {passed}/{total} passed")
    if failures:
        print(f"  Regressions detected: {len(failures)}")
        for f in failures:
            print(f"    [{f['id']}] {' | '.join(f['errors'])}")
        print("\n  REGRESSION GATE: FAIL — do not merge")
    else:
        print("\n  REGRESSION GATE: PASS — safe to merge")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    ok = run_regression()
    sys.exit(0 if ok else 1)
