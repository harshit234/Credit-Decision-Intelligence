"""
================================================================================
   HALCYON CREDIT — Prompt Optimization: Before vs After Comparison
   Sprint 3 | Author: Himkar
   Domain: Prompt Engineering, Policy KB

   Performs a manual before/after comparison of two synthesizer prompts
   on 4 held-out golden set cases WITHOUT calling DSPy training.

   Measures:
   - Faithfulness score (from the Evaluator node)
   - Number of grounded citations per decision
   - Recommendation accuracy vs ground truth

   Usage: python eval/dspy_optimizer.py
================================================================================
"""
from __future__ import annotations
import sys, os, json, re, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from state.application_state import ApplicantFile, make_initial_state
from agents.income_agent    import verify_income_node
from agents.credit_agent    import fetch_credit_node
from agents.policy_agent    import check_policy_node
from agents.risk_agent      import score_risk_node
from agents.synthesizer_agent import synthesize_node
from agents.evaluation_agent  import evaluate_node

GOLDEN_SET_PATH = os.path.join(
    os.path.dirname(__file__), "golden_set", "test_cases.json"
)

# 4 held-out cases for comparison
HELD_OUT_IDS = ["TC-001", "TC-003", "TC-006", "TC-009"]

# ── Prompt A: Baseline (minimal instructions) ────────────────────────────────
PROMPT_A = """You are a loan underwriting decision engine.
Output JSON only: {"recommendation": "APPROVE"|"DECLINE"|"REFER", "reasons": [...], "conditions": [...]}
Base your decision on the input data."""

# ── Prompt B: Optimised (current production v2) ──────────────────────────────
from gateway.prompts import SYNTHESIZER_SYSTEM_PROMPT_V2
PROMPT_B = SYNTHESIZER_SYSTEM_PROMPT_V2


def _count_citations(reasons: list[str]) -> int:
    """Count [field=value] citations across all reasons."""
    text = " ".join(reasons)
    return len(re.findall(r"\[[^\]]+\]", text))


def _run_with_prompt(af: ApplicantFile, prompt: str) -> dict:
    """Run the full pipeline using a specific system prompt."""
    import agents.synthesizer_agent as sa
    original = sa.SYSTEM_PROMPT
    sa.SYSTEM_PROMPT = prompt

    try:
        state = make_initial_state(af)
        state.update(verify_income_node(state))
        state.update(fetch_credit_node(state))
        state.update(check_policy_node(state))
        state.update(score_risk_node(state))
        state.update(synthesize_node(state))
        state.update(evaluate_node(state))
    finally:
        sa.SYSTEM_PROMPT = original

    decision = state.get("draft_decision")
    ev       = state.get("eval_result")

    return {
        "recommendation": decision.recommendation if decision else "ERROR",
        "reasons":        decision.reasons if decision else [],
        "faithfulness":   ev.faithfulness if ev else 0.0,
        "citations":      _count_citations(decision.reasons if decision else []),
    }


def run_comparison():
    with open(GOLDEN_SET_PATH) as f:
        all_cases = json.load(f)

    cases = [c for c in all_cases if c["id"] in HELD_OUT_IDS]

    print("\n" + "=" * 70)
    print("  HALCYON — Prompt Optimization: Before vs After")
    print(f"  Held-out cases: {HELD_OUT_IDS}")
    print("=" * 70)

    results = []

    for case in cases:
        cid      = case["id"]
        expected = case["expected_recommendation"]
        af       = ApplicantFile(**case["input"])

        print(f"\n  [{cid}] {case['description']}")
        print(f"  Expected: {expected}")

        # Prompt A (baseline)
        t0 = time.time()
        ra = _run_with_prompt(af, PROMPT_A)
        ta = round(time.time() - t0, 1)

        # Prompt B (optimized)
        t0 = time.time()
        rb = _run_with_prompt(af, PROMPT_B)
        tb = round(time.time() - t0, 1)

        correct_a = ra["recommendation"] == expected
        correct_b = rb["recommendation"] == expected

        print(f"  Prompt A (baseline):   rec={ra['recommendation']:8s} "
              f"faith={ra['faithfulness']:.3f}  citations={ra['citations']}  "
              f"{'CORRECT' if correct_a else 'WRONG'}  ({ta}s)")
        print(f"  Prompt B (optimized):  rec={rb['recommendation']:8s} "
              f"faith={rb['faithfulness']:.3f}  citations={rb['citations']}  "
              f"{'CORRECT' if correct_b else 'WRONG'}  ({tb}s)")

        results.append({
            "case_id":  cid,
            "expected": expected,
            "prompt_a": ra, "correct_a": correct_a,
            "prompt_b": rb, "correct_b": correct_b,
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    n              = len(results)
    acc_a          = sum(1 for r in results if r["correct_a"]) / n
    acc_b          = sum(1 for r in results if r["correct_b"]) / n
    avg_faith_a    = sum(r["prompt_a"]["faithfulness"] for r in results) / n
    avg_faith_b    = sum(r["prompt_b"]["faithfulness"] for r in results) / n
    avg_cite_a     = sum(r["prompt_a"]["citations"] for r in results) / n
    avg_cite_b     = sum(r["prompt_b"]["citations"] for r in results) / n

    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print(f"  {'Metric':<28} {'Prompt A (baseline)':>20} {'Prompt B (optimized)':>20}")
    print("  " + "-" * 68)
    print(f"  {'Accuracy':<28} {acc_a:>19.1%} {acc_b:>19.1%} {'  <-- BETTER' if acc_b > acc_a else ''}")
    print(f"  {'Avg Faithfulness':<28} {avg_faith_a:>20.3f} {avg_faith_b:>20.3f} {'  <-- BETTER' if avg_faith_b > avg_faith_a else ''}")
    print(f"  {'Avg Citations/Decision':<28} {avg_cite_a:>20.1f} {avg_cite_b:>20.1f} {'  <-- BETTER' if avg_cite_b > avg_cite_a else ''}")
    print("=" * 70)
    print(f"\n  Verdict: Prompt B {'OUTPERFORMS' if acc_b >= acc_a and avg_faith_b >= avg_faith_a else 'does not clearly outperform'} Prompt A")
    print("  Production system uses Prompt B (SYNTHESIZER_SYSTEM_PROMPT_V2)\n")

    # Save
    out = os.path.join(os.path.dirname(__file__), "prompt_comparison_results.json")
    with open(out, "w") as f:
        json.dump({
            "summary": {
                "accuracy_a": acc_a, "accuracy_b": acc_b,
                "avg_faithfulness_a": avg_faith_a, "avg_faithfulness_b": avg_faith_b,
                "avg_citations_a": avg_cite_a, "avg_citations_b": avg_cite_b,
            },
            "cases": results
        }, f, indent=2)
    print(f"  Results saved -> {out}")

    return results


if __name__ == "__main__":
    run_comparison()
