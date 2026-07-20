"""
================================================================================
   HALCYON CREDIT — FastAPI Routes
   Stage 3 | Author: Himkar
================================================================================
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, HTTPException
from api.schemas import ApplicationInput, DecisionResponse, HealthResponse, SHAPFeatureOut
from state.application_state import ApplicantFile, PriorApp
from graph.pipeline import run_pipeline
from tools.decision_record_tool import fetch_record, list_records

router = APIRouter()


@router.post("/applications", response_model=DecisionResponse, summary="Submit loan application")
def submit_application(body: ApplicationInput):
    """
    Submit a loan application and receive an underwriting decision.
    Runs the full LangGraph pipeline synchronously.
    Returns the decision, reasons, risk score, and audit_id.
    """
    # Map API input → ApplicantFile
    prior_apps = [PriorApp(**p.model_dump()) for p in body.prior_applications]
    applicant  = ApplicantFile(
        applicant_id           = body.applicant_id,
        name                   = body.name,
        loan_amount            = body.loan_amount,
        loan_purpose           = body.loan_purpose,
        loan_term_months       = body.loan_term_months,
        annual_income          = body.annual_income,
        employment_type        = body.employment_type,
        months_employed        = body.months_employed,
        verification_status    = body.verification_status,
        existing_debts         = body.existing_debts,
        credit_score           = body.credit_score,
        delinquencies_2yr      = body.delinquencies_2yr,
        open_accounts          = body.open_accounts,
        revolving_utilisation  = body.revolving_utilisation,
        credit_age_months      = body.credit_age_months,
        public_records         = body.public_records,
        inquiries_6mo          = body.inquiries_6mo,
        home_ownership         = body.home_ownership,
        prior_applications     = prior_apps,
    )

    # Run pipeline
    final_state = run_pipeline(applicant)
    record      = final_state.get("final_record")

    if record is None:
        raise HTTPException(status_code=500, detail="Pipeline failed to produce a decision record")

    decision   = record.final_decision
    risk       = record.risk_result
    ev         = record.eval_result
    pf         = final_state.get("policy_findings")

    recommendation = "ESCALATED" if record.escalated else (
        decision.recommendation if decision else "REFER"
    )

    shap_features = []
    if risk and risk.top_features:
        shap_features = [
            SHAPFeatureOut(
                feature    = f.feature,
                value      = f.value,
                shap_value = f.shap_value,
                direction  = f.direction,
            )
            for f in risk.top_features
        ]

    return DecisionResponse(
        audit_id           = record.audit_id,
        application_id     = body.applicant_id,
        recommendation     = recommendation,
        reasons            = decision.reasons if decision else ["Decision unavailable"],
        conditions         = decision.conditions if decision else [],
        risk_score         = risk.risk_score if risk else 0.0,
        risk_band          = risk.risk_band if risk else "Unknown",
        top_risk_features  = shap_features,
        faithfulness_score = ev.faithfulness if ev else 0.0,
        policy_flags       = pf.flags if pf else [],
        hard_stops         = pf.hard_stops if pf else [],
        retry_count        = record.retry_count if hasattr(record, "retry_count") else final_state.get("retry_count", 0),
        escalated          = record.escalated,
        cost_usd_total     = record.cost_usd_total,
        created_at         = record.created_at,
    )


@router.get("/applications/{audit_id}", summary="Retrieve decision by audit ID")
def get_application(audit_id: str):
    """Retrieve the full decision record for a given audit_id."""
    record = fetch_record(audit_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No record found for audit_id: {audit_id}")
    return record


@router.get("/applications", summary="List recent decisions")
def list_applications(limit: int = 20):
    """List the most recent loan decisions."""
    return list_records(limit=limit)


@router.get("/health", response_model=HealthResponse, summary="Health check")
def health():
    """Health check endpoint."""
    return HealthResponse(status="ok")


@router.get("/metrics", summary="Basic metrics")
def metrics():
    """Return basic operational metrics."""
    records = list_records(limit=1000)
    total   = len(records)
    if total == 0:
        return {"total_applications": 0}

    approvals  = sum(1 for r in records if r.get("recommendation") == "APPROVE")
    declines   = sum(1 for r in records if r.get("recommendation") == "DECLINE")
    refers     = sum(1 for r in records if r.get("recommendation") == "REFER")
    escalated  = sum(1 for r in records if r.get("escalated"))
    avg_cost   = sum(r.get("cost_usd", 0) for r in records) / total
    avg_faith  = sum(r.get("faithfulness", 0) or 0 for r in records) / total

    return {
        "total_applications":    total,
        "approve_rate":          round(approvals / total, 3),
        "decline_rate":          round(declines / total, 3),
        "refer_rate":            round(refers / total, 3),
        "escalation_rate":       round(escalated / total, 3),
        "avg_cost_per_app_usd":  round(avg_cost, 5),
        "avg_faithfulness":      round(avg_faith, 3),
    }
