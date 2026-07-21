"""
================================================================================
   HALCYON CREDIT — RAGAS Evaluation Runner
   Stage 3 | Author: Ayush
   Runs all golden set test cases through the live pipeline and reports:
   - Decision accuracy vs. ground truth
   - RAGAS faithfulness and relevancy scores
   - Per-case breakdown

   Usage:
     python eval/ragas_runner.py
================================================================================
"""
from __future__ import annotations
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from state.application_state import ApplicantFile
from graph.pipeline import run_pipeline

GOLDEN_SET_PATH = os.path.join(os.path.dirname(__file__), "golden_set", "test_cases.json")


def load_golden_set() -> list[dict]:
    with open(GOLDEN_SET_PATH, "r") as f:
        return json.load(f)


def run_golden_set_eval(verbose: bool = True) -> dict:
    """
    Run all golden set cases through the live pipeline.
    Returns a results summary dict.
    """
    cases   = load_golden_set()
    results = []

    print("\n" + "=" * 70)
    print("  HALCYON CREDIT — Golden Set Evaluation")
    print(f"  Cases: {len(cases)}")
    print("=" * 70)

    for case in cases:
        inp      = case["input"]
        expected = case["expected_recommendation"]
        case_id  = case["id"]

        print(f"\n  [{case_id}] {case['description']}")
        t0 = time.time()

        try:
            af = ApplicantFile(**inp)
            final_state = run_pipeline(af)

            record   = final_state.get("final_record")
            decision = record.final_decision if record else None
            ev       = record.eval_result if record else None
            risk     = record.risk_result if record else None

            actual_rec   = "ESCALATED" if (record and record.escalated) else (
                decision.recommendation if decision else "ERROR"
            )
            correct      = actual_rec == expected
            faithfulness = ev.faithfulness if ev else None
            relevancy    = ev.relevancy    if ev else None
            risk_score   = risk.risk_score if risk else None
            cost_usd     = record.cost_usd_total if record else None
            elapsed      = time.time() - t0

            result = {
                "case_id":       case_id,
                "description":   case["description"],
                "expected":      expected,
                "actual":        actual_rec,
                "correct":       correct,
                "faithfulness":  round(faithfulness, 3) if faithfulness is not None else None,
                "relevancy":     round(relevancy,    3) if relevancy    is not None else None,
                "risk_score":    round(risk_score, 4)   if risk_score   is not None else None,
                "cost_usd":      round(cost_usd, 5)     if cost_usd     is not None else None,
                "elapsed_s":     round(elapsed, 1),
                "reasons":       decision.reasons if decision else [],
                "hard_stops":    final_state.get("policy_findings").hard_stops
                                 if final_state.get("policy_findings") else [],
                "flags":         final_state.get("policy_findings").flags
                                 if final_state.get("policy_findings") else [],
                "error":         None,
            }

            status = "PASS" if correct else "FAIL"
            print(f"    Expected: {expected:8s} | Actual: {actual_rec:8s} | "
                  f"[{status}] | Faith={faithfulness:.3f if faithfulness else 'N/A':>5} "
                  f"Relev={relevancy:.3f if relevancy else 'N/A':>5} "
                  f"| Risk={risk_score:.4f if risk_score else 'N/A'} | {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - t0
            result  = {
                "case_id": case_id, "description": case["description"],
                "expected": expected, "actual": "ERROR", "correct": False,
                "faithfulness": None, "risk_score": None, "elapsed_s": round(elapsed, 1),
                "reasons": [], "hard_stops": [], "flags": [], "error": str(e),
            }
            print(f"    [ERROR] {e}")

        results.append(result)

    # ── Summary ──
    total     = len(results)
    correct   = sum(1 for r in results if r["correct"])
    accuracy  = correct / total

    faith_vals = [r["faithfulness"] for r in results if r["faithfulness"] is not None]
    relev_vals = [r["relevancy"]    for r in results if r.get("relevancy") is not None]
    cost_vals  = [r["cost_usd"]     for r in results if r.get("cost_usd")  is not None]
    avg_faith  = sum(faith_vals) / len(faith_vals) if faith_vals else 0.0
    avg_relev  = sum(relev_vals) / len(relev_vals) if relev_vals else 0.0
    total_cost = sum(cost_vals)

    approve_n = sum(1 for r in results if r["actual"] == "APPROVE")
    decline_n = sum(1 for r in results if r["actual"] == "DECLINE")
    refer_n   = sum(1 for r in results if r["actual"] == "REFER")

    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Total cases          : {total}")
    print(f"  Correct              : {correct} / {total} ({accuracy:.0%})")
    print(f"  Avg faithfulness     : {avg_faith:.3f}")
    print(f"  Avg relevancy        : {avg_relev:.3f}")
    print(f"  Total LLM cost       : ${total_cost:.4f}")
    print(f"  Cost per application : ${total_cost/total:.5f}")
    print(f"  APPROVE / DECLINE / REFER : {approve_n} / {decline_n} / {refer_n}")
    print(f"  Target accuracy  : >= 85%  ->  {'PASS' if accuracy >= 0.85 else 'FAIL'}")
    print(f"  Target faith     : >= 0.80 ->  {'PASS' if avg_faith >= 0.80 else 'FAIL'}")
    print("=" * 70)

    # Per-case failure report
    failures = [r for r in results if not r["correct"]]
    if failures:
        print(f"\n  FAILURES ({len(failures)}):")
        for f in failures:
            print(f"    [{f['case_id']}] Expected {f['expected']} → Got {f['actual']}")
            if f["error"]:
                print(f"           Error: {f['error']}")
            if f["reasons"]:
                print(f"           Reasons: {f['reasons'][0][:80]}...")

    summary = {
        "total":            total,
        "correct":          correct,
        "accuracy":         round(accuracy, 4),
        "avg_faithfulness": round(avg_faith, 4),
        "avg_relevancy":    round(avg_relev, 4),
        "total_cost_usd":   round(total_cost, 4),
        "cost_per_app":     round(total_cost / total, 5),
        "approve_count":    approve_n,
        "decline_count":    decline_n,
        "refer_count":      refer_n,
        "accuracy_pass":    accuracy  >= 0.85,
        "faith_pass":       avg_faith >= 0.80,
        "results":          results,
    }

    return summary


if __name__ == "__main__":
    summary = run_golden_set_eval(verbose=True)

    # Write results to file
    out_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved → {out_path}")
