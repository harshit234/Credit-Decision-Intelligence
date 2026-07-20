"""
================================================================================
   HALCYON CREDIT — Application State Schema
   Stage 3 | Author: Himkar
   Shared Pydantic models + TypedDict used by every agent in the pipeline.
   ALL agents import from this file. Do not modify without team sign-off.
================================================================================
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ─────────────────────────────────────────────────────────────────────────────
# LEAF MODELS — composites built from these
# ─────────────────────────────────────────────────────────────────────────────

class SHAPFeature(BaseModel):
    """One entry in the SHAP top-5 attribution list."""
    feature:    str
    value:      float   # actual feature value at inference
    shap_value: float   # contribution to model output (positive = riskier)
    direction:  str     # "increases_risk" | "decreases_risk"


class PolicyClause(BaseModel):
    """A single retrieved policy clause from ChromaDB."""
    clause_id:      str
    text:           str
    is_hard_stop:   bool
    section:        str
    source_ref:     str     # ChromaDB chunk ID


class PriorApp(BaseModel):
    """A prior loan application linked to the same applicant."""
    application_id: str
    date:           str
    outcome:        str     # APPROVE | DECLINE | REFER


class NodeEvent(BaseModel):
    """Immutable trace event written by every agent node."""
    agent:      str
    action:     str
    timestamp:  str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    retry:      int = 0
    latency_ms: Optional[float] = None
    cost_usd:   Optional[float] = None


class AgentError(BaseModel):
    """Error written to state when an agent fails."""
    agent:      str
    error_type: str     # "tool_timeout" | "llm_error" | "validation_error" | "escalation"
    message:    str
    timestamp:  str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ─────────────────────────────────────────────────────────────────────────────
# CORE STATE SCHEMAS — one per agent
# ─────────────────────────────────────────────────────────────────────────────

class ApplicantFile(BaseModel):
    """
    Immutable applicant record populated from the loan application form.
    Set once at init_state, never mutated by downstream agents.
    Maps to ApplicationInput from api/schemas.py.
    """
    applicant_id:           str
    name:                   str

    # ── Loan request ──────────────────────────────────────────────────────
    loan_amount:            float = Field(gt=0)
    loan_purpose:           str   # debt_consolidation | home_improvement | medical | car | other
    loan_term_months:       int   # 36 | 60

    # ── Income & employment ───────────────────────────────────────────────
    annual_income:          float = Field(gt=0)
    employment_type:        str   # salaried | self-employed | gig
    months_employed:        int   = Field(ge=0)
    verification_status:    str   # Source Verified | Verified | Not Verified
    existing_debts:         float = Field(ge=0)   # total monthly debt payments

    # ── Credit bureau (applicant-provided; cross-checked by CreditAgent) ─
    credit_score:           int   = Field(ge=300, le=850)
    delinquencies_2yr:      int   = Field(ge=0)
    open_accounts:          int   = Field(ge=0)
    revolving_utilisation:  float = Field(ge=0.0, le=100.0)
    credit_age_months:      int   = Field(ge=0)
    public_records:         int   = Field(ge=0)
    inquiries_6mo:          int   = Field(ge=0)

    # ── Context ───────────────────────────────────────────────────────────
    home_ownership:         str   # OWN | MORTGAGE | RENT

    # ── Prior applications ────────────────────────────────────────────────
    prior_applications:     list[PriorApp] = []


class IncomeResult(BaseModel):
    """Written by IncomeVerificationAgent → verify_income node."""
    verified_income:    float
    confidence:         float = Field(ge=0.0, le=1.0)
    source_refs:        list[str] = []
    verified_at:        str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class CreditResult(BaseModel):
    """Written by CreditHistoryAgent → fetch_credit node."""
    credit_score:       int
    delinquencies:      int
    credit_age_months:  int
    open_accounts:      int
    utilization_pct:    float
    thin_file:          bool    # True if credit_age < 24mo OR open_acc < 3
    source_refs:        list[str] = []


class PolicyResult(BaseModel):
    """Written by PolicyComplianceAgent → check_policy node."""
    applicable_clauses: list[PolicyClause] = []
    hard_stops:         list[str] = []      # clause IDs that mandate DECLINE
    flags:              list[str] = []      # advisory flags (e.g. POL-004)
    source_refs:        list[str] = []      # ChromaDB chunk IDs


class RiskResult(BaseModel):
    """Written by RiskScoringAgent → score_risk node."""
    risk_score:     float = Field(ge=0.0, le=1.0)   # higher = riskier
    risk_band:      Literal["Low", "Medium", "High"]
    top_features:   list[SHAPFeature] = []           # SHAP top-5
    model_version:  str = "lgbm_halcyon_v2_lc"


class Decision(BaseModel):
    """Written by DecisionSynthesizerAgent → synthesize node."""
    recommendation: Literal["APPROVE", "DECLINE", "REFER"]
    reasons:        list[str]   # each must cite a source field or clause ID
    conditions:     list[str] = []  # empty unless APPROVE with conditions
    draft_version:  int = 1


class EvalResult(BaseModel):
    """Written by EvaluationAgent → evaluate node."""
    faithfulness:           float = Field(ge=0.0, le=1.0)
    relevancy:              float = Field(ge=0.0, le=1.0)
    unsupported_claims:     list[str] = []  # claims not grounded in source data
    pass_flag:              bool = False    # True when faithfulness >= threshold


class DecisionRecord(BaseModel):
    """Written by RecordWriterAgent → write_record node. Immutable after write."""
    audit_id:           str = Field(default_factory=lambda: str(uuid.uuid4()))
    application_id:     str
    final_decision:     Decision
    eval_result:        EvalResult
    risk_result:        RiskResult
    policy_refs:        list[str] = []
    full_trace:         list[NodeEvent] = []
    errors:             list[AgentError] = []
    escalated:          bool = False
    cost_usd_total:     float = 0.0
    created_at:         str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION STATE — the single object threaded through the entire graph
# ─────────────────────────────────────────────────────────────────────────────

class ApplicationState(TypedDict):
    """
    LangGraph state TypedDict. One instance per application run.
    Checkpointed after every node transition (SQLite in dev).

    OWNERSHIP RULE: No agent may mutate a key it does not own.
    Each agent returns a dict with ONLY the keys it owns.

    Key ownership:
      init_state     → application_id, applicant_file, retry_count, trace, errors
      verify_income  → income_verified
      fetch_credit   → credit_report
      check_policy   → policy_findings
      score_risk     → risk_score
      synthesize     → draft_decision
      evaluate       → eval_result, retry_count
      write_record   → final_record
    """
    application_id:     str
    applicant_file:     ApplicantFile

    # Agent outputs (None until that node executes)
    income_verified:    Optional[IncomeResult]
    credit_report:      Optional[CreditResult]
    policy_findings:    Optional[PolicyResult]
    risk_score:         Optional[RiskResult]
    draft_decision:     Optional[Decision]
    eval_result:        Optional[EvalResult]
    final_record:       Optional[DecisionRecord]

    # Pipeline control
    retry_count:        int
    trace:              list[NodeEvent]
    errors:             list[AgentError]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_initial_state(applicant_file: ApplicantFile) -> ApplicationState:
    """Create a fresh ApplicationState for a new application."""
    return ApplicationState(
        application_id  = applicant_file.applicant_id,
        applicant_file  = applicant_file,
        income_verified = None,
        credit_report   = None,
        policy_findings = None,
        risk_score      = None,
        draft_decision  = None,
        eval_result     = None,
        final_record    = None,
        retry_count     = 0,
        trace           = [],
        errors          = [],
    )


def log_event(state: ApplicationState, agent: str, action: str,
              latency_ms: float = None, cost_usd: float = None) -> list[NodeEvent]:
    """Return updated trace list with new event appended."""
    event = NodeEvent(
        agent      = agent,
        action     = action,
        retry      = state["retry_count"],
        latency_ms = latency_ms,
        cost_usd   = cost_usd,
    )
    return state["trace"] + [event]
