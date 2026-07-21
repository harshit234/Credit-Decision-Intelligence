"""
================================================================================
   HALCYON CREDIT — LangGraph Pipeline
   Stage 3 | Author: Himkar
   Wires all 8 agent nodes into a directed LangGraph StateGraph.

   Graph topology:
     init_state
       │
       ├──[concurrent]── verify_income  (Aditya)
       ├──[concurrent]── fetch_credit   (Aditya)
       └──[concurrent]── check_policy   (Aditya)
                              │
                          gather_data   (fan-in — collects all 3 results)
                              │
                          score_risk    (Harshit)
                              │
                          synthesize    (Ayush)
                              │
                          evaluate      (Ayush)
                         ┌────┴────┐
                      [pass]  [retry -> synthesize, max 2x]
                         │
                     write_record  (Harshit)
================================================================================
"""
from __future__ import annotations
import sys, os, uuid, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, START, END

from state.application_state import (
    ApplicationState, ApplicantFile, make_initial_state, log_event
)
from agents.income_agent       import verify_income_node
from agents.credit_agent       import fetch_credit_node
from agents.policy_agent       import check_policy_node
from agents.risk_agent         import score_risk_node
from agents.synthesizer_agent  import synthesize_node
from agents.evaluation_agent   import evaluate_node
from agents.record_writer      import write_record_node

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))


# ─────────────────────────────────────────────────────────────────────────────
# INIT NODE
# ─────────────────────────────────────────────────────────────────────────────
def init_state_node(state: ApplicationState) -> dict:
    """First node — validates state is properly initialized."""
    af = state["applicant_file"]
    print(f"\n{'='*60}")
    print(f"  [Pipeline] START — Application: {af.applicant_id}")
    print(f"  Applicant: {af.name} | Loan: {af.loan_amount:,.0f} | Purpose: {af.loan_purpose}")
    print(f"{'='*60}")
    trace = log_event(state, "Orchestrator", "pipeline_started")
    return {"trace": trace}


# ─────────────────────────────────────────────────────────────────────────────
# GATHER NODE — concurrent fan-out + fan-in for data agents
# ─────────────────────────────────────────────────────────────────────────────
def gather_data_node(state: ApplicationState) -> dict:
    """
    Runs Income, Credit, and Policy agents concurrently using asyncio.
    This implements the parallel fan-out pattern from the TRD.
    All three results are merged into a single state update.
    """
    print(f"\n  [Orchestrator] Running data agents in parallel...")

    async def _run_all():
        loop = asyncio.get_event_loop()
        income_fut = loop.run_in_executor(None, verify_income_node, state)
        credit_fut = loop.run_in_executor(None, fetch_credit_node,  state)
        policy_fut = loop.run_in_executor(None, check_policy_node,  state)
        return await asyncio.gather(income_fut, credit_fut, policy_fut)

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        income_res, credit_res, policy_res = loop.run_until_complete(_run_all())
        loop.close()
    except Exception:
        # Synchronous fallback if async fails
        income_res = verify_income_node(state)
        credit_res = fetch_credit_node(state)
        policy_res = check_policy_node(state)

    # Merge all errors and trace events
    all_errors = (
        income_res.get("errors", []) +
        credit_res.get("errors", []) +
        policy_res.get("errors", [])
    )
    all_trace = (
        state["trace"] +
        income_res.get("trace", [])[-1:] +
        credit_res.get("trace", [])[-1:] +
        policy_res.get("trace", [])[-1:]
    )

    return {
        "income_verified": income_res.get("income_verified"),
        "credit_report":   credit_res.get("credit_report"),
        "policy_findings": policy_res.get("policy_findings"),
        "errors":          all_errors,
        "trace":           all_trace,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RETRY ROUTER — conditional edge from evaluate
# ─────────────────────────────────────────────────────────────────────────────
def retry_or_persist_router(state: ApplicationState) -> str:
    """
    Conditional edge from evaluate node.
    Returns:
        "synthesize"   -> retry (eval failed and retries remaining)
        "write_record" -> proceed (eval passed or retries exhausted)
    """
    ev          = state.get("eval_result")
    retry_count = state.get("retry_count", 0)

    if ev is None:
        print(f"  [Router] No eval_result -> write_record")
        return "write_record"

    if ev.pass_flag:
        print(f"  [Router] Faithfulness={ev.faithfulness:.3f} >= threshold -> write_record")
        return "write_record"

    if retry_count >= MAX_RETRIES:
        print(f"  [Router] Retries exhausted ({retry_count}/{MAX_RETRIES}) -> write_record (ESCALATED)")
        return "write_record"

    print(f"  [Router] Faithfulness={ev.faithfulness:.3f} < threshold "
          f"-> synthesize (retry {retry_count}/{MAX_RETRIES})")
    return "synthesize"


# ─────────────────────────────────────────────────────────────────────────────
# BUILD GRAPH
# ─────────────────────────────────────────────────────────────────────────────
def build_pipeline() -> StateGraph:
    """
    Builds and compiles the full Halcyon LangGraph pipeline.
    Returns a compiled graph ready for .invoke() or .ainvoke().
    """
    graph = StateGraph(ApplicationState)

    # Add all 8 nodes (+ gather orchestration node)
    graph.add_node("init_state",    init_state_node)
    graph.add_node("gather_data",   gather_data_node)   # parallel fan-out inside
    graph.add_node("score_risk",    score_risk_node)
    graph.add_node("synthesize",    synthesize_node)
    graph.add_node("evaluate",      evaluate_node)
    graph.add_node("write_record",  write_record_node)

    # Linear edges
    graph.add_edge(START,          "init_state")
    graph.add_edge("init_state",   "gather_data")
    graph.add_edge("gather_data",  "score_risk")
    graph.add_edge("score_risk",   "synthesize")
    graph.add_edge("synthesize",   "evaluate")

    # Conditional retry edge from evaluate
    graph.add_conditional_edges(
        "evaluate",
        retry_or_persist_router,
        {
            "synthesize":   "synthesize",
            "write_record": "write_record",
        }
    )

    graph.add_edge("write_record", END)

    return graph.compile()


# Singleton compiled pipeline — loaded once at import time
pipeline = build_pipeline()


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(applicant_file: ApplicantFile) -> ApplicationState:
    """
    Run the full underwriting pipeline for a single loan application.

    Args:
        applicant_file: Populated ApplicantFile from the API layer

    Returns:
        Final ApplicationState with all fields populated
    """
    initial_state = make_initial_state(applicant_file)
    final_state   = pipeline.invoke(initial_state)

    rec = final_state.get("final_record")
    if rec:
        print(f"\n{'='*60}")
        print(f"  [Pipeline] COMPLETE — audit_id: {rec.audit_id}")
        print(f"  Decision:     {rec.final_decision.recommendation if rec.final_decision else 'UNKNOWN'}")
        print(f"  Risk score:   {rec.risk_result.risk_score if rec.risk_result else 'N/A'}")
        print(f"  Faithfulness: {rec.eval_result.faithfulness if rec.eval_result else 'N/A'}")
        print(f"  Cost:         ${rec.cost_usd_total:.5f}")
        print(f"  Escalated:    {rec.escalated}")
        print(f"{'='*60}\n")

    return final_state
