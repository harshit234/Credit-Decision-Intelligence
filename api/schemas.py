"""
================================================================================
   HALCYON CREDIT — FastAPI Schemas (API Layer)
   Stage 3 | Author: Himkar
   Request / Response schemas for POST /applications
================================================================================
"""
from __future__ import annotations
from typing import Optional, Literal
from pydantic import BaseModel, Field


class PriorAppInput(BaseModel):
    application_id: str
    date:           str
    outcome:        str


class ApplicationInput(BaseModel):
    """
    Loan application form submitted by the borrower.
    This is the ONLY external-facing input to the pipeline.
    Maps directly to ApplicantFile inside the graph.

    NOTE: applicant_id and verification_status are OPTIONAL —
    the system generates/determines them automatically when omitted.
    """
    applicant_id:           Optional[str] = Field(default=None, description="Auto-generated if omitted")
    name:                   str

    # ── Loan request ──────────────────────────────────────────────────────
    loan_amount:            float = Field(gt=0, description="Loan amount requested in INR/USD")
    loan_purpose:           str   = Field(description="e.g. debt_consolidation, home_improvement, medical, car, other")
    loan_term_months:       int   = Field(default=36, description="36 or 60 months")

    # ── Income & employment ───────────────────────────────────────────────
    annual_income:          float = Field(gt=0, description="Gross annual income")
    employment_type:        Literal["salaried", "self-employed", "gig"]
    months_employed:        int   = Field(ge=0, description="Months at current employer")
    verification_status:    Optional[str] = Field(default=None, description="System-determined if omitted")
    existing_debts:         float = Field(ge=0, description="Total existing monthly debt payments")

    # ── Credit bureau (applicant-declared; validated by CreditAgent) ──────
    credit_score:           int   = Field(ge=300, le=850, description="FICO score 300-850")
    delinquencies_2yr:      int   = Field(ge=0, default=0)
    open_accounts:          int   = Field(ge=0, default=5)
    revolving_utilisation:  float = Field(ge=0.0, le=100.0, default=30.0)
    credit_age_months:      int   = Field(ge=0, default=60)
    public_records:         int   = Field(ge=0, default=0)
    inquiries_6mo:          int   = Field(ge=0, default=0)

    # ── Context ───────────────────────────────────────────────────────────
    home_ownership:         str   = Field(default="RENT", description="OWN | MORTGAGE | RENT")

    # ── Prior applications (optional) ─────────────────────────────────────
    prior_applications:     list[PriorAppInput] = []


class SHAPFeatureOut(BaseModel):
    feature:    str
    value:      float
    shap_value: float
    direction:  str


class TraceEventOut(BaseModel):
    """One audit-trace event from the pipeline run."""
    agent:      str
    action:     str
    timestamp:  str
    latency_ms: Optional[float] = None
    cost_usd:   Optional[float] = None


class ApplicantProfileOut(BaseModel):
    """Echo of the applicant file, enriched with verified data for the report."""
    applicant_id:           str
    name:                   str
    annual_income:          float
    verified_income:        Optional[float] = None
    income_confidence:      Optional[float] = None
    loan_amount:            float
    loan_purpose:           str
    loan_term_months:       int
    employment_type:        str
    months_employed:        int
    verification_status:    str
    credit_score:           int
    dti_pct:                float
    revolving_utilisation:  float
    delinquencies_2yr:      int
    credit_age_months:      int
    open_accounts:          int
    thin_file:              Optional[bool] = None
    public_records:         int


class DecisionResponse(BaseModel):
    """
    Response returned to the client after POST /applications.
    Contains the final decision + key supporting evidence.
    """
    audit_id:           str
    application_id:     str
    recommendation:     Literal["APPROVE", "DECLINE", "REFER", "ESCALATED"]
    reasons:            list[str]
    conditions:         list[str]
    risk_score:         float
    risk_band:          str
    top_risk_features:  list[SHAPFeatureOut]
    model_version:      str = ""
    faithfulness_score: float
    relevancy_score:    float = 0.0
    eval_pass:          bool = False
    policy_flags:       list[str]
    hard_stops:         list[str]
    retry_count:        int
    escalated:          bool
    cost_usd_total:     float
    created_at:         str
    applicant:          Optional[ApplicantProfileOut] = None
    trace:              list[TraceEventOut] = []


class HealthResponse(BaseModel):
    status:  str
    version: str = "3.0.0"
    uptime_s: Optional[float] = None


class ErrorResponse(BaseModel):
    error_id:   str
    message:    str
    detail:     Optional[str] = None
