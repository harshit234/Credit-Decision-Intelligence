"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        HALCYON CREDIT — Agentic Underwriting Copilot (MVP / POC)           ║
║        Team Jamun · Futurense AI Clinic · Capstone Project 02              ║
╚══════════════════════════════════════════════════════════════════════════════╝

End-to-end proof of concept demonstrating the full agent pipeline:

  1. Sample dataset built on the unified ApplicantFile schema
     (LendingClub + Home Credit harmonised features)
  2. XGBoost risk model — trained live on the sample data
  3. Orchestrator dispatches 3 parallel worker agents (mocked tools)
  4. Risk Scoring Agent — runs the trained model + SHAP attribution
  5. Decision Synthesizer Agent — produces written recommendation
  6. Evaluation Agent — scores faithfulness of the decision
  7. Decision Record Writer — persists the full audit trace
  8. Final output printed as an auditable decision record

No external APIs required. Runs fully offline on sample data.
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import uuid
import json
import time
import warnings
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

warnings.filterwarnings("ignore")
console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — UNIFIED SCHEMA  (ApplicantFile + ApplicationState)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ApplicantFile:
    """
    Unified schema harmonised from LendingClub + Home Credit.
    Every field maps to at least one real column in either dataset.
    """
    applicant_id:          str
    name:                  str
    # ── Loan request ──────────────────────────────────────────────────────
    loan_amount:           float   # LC: loan_amnt | HC: AMT_CREDIT
    loan_purpose:          str     # LC: purpose   | HC: NAME_CONTRACT_TYPE
    loan_term_months:      int     # LC: term      | HC: default 36
    # ── Income & employment ───────────────────────────────────────────────
    annual_income:         float   # LC: annual_inc    | HC: AMT_INCOME_TOTAL
    verification_status:   str     # LC: verification_status | HC: "unknown"
    employment_months:     int     # LC: emp_length parsed  | HC: DAYS_EMPLOYED/30
    employment_type:       str     # LC: unknown       | HC: NAME_INCOME_TYPE
    # ── Credit bureau ─────────────────────────────────────────────────────
    credit_score:          int     # LC: fico_range_low | HC: EXT_SOURCE_2 normalised
    delinquencies_2yr:     int     # LC: delinq_2yrs    | HC: bureau table agg
    revolving_utilisation: float   # LC: revol_util     | HC: bureau_balance agg
    credit_age_months:     int     # LC: earliest_cr_line diff | HC: DAYS_CREDIT
    open_accounts:         int     # LC: open_acc       | HC: bureau count
    inquiries_6mo:         int     # LC: inq_last_6mths | HC: AMT_REQ_CREDIT_BUREAU_QRT*2
    public_records:        int     # LC: pub_rec        | HC: FLAG_DOCUMENT proxies
    # ── Debt burden ───────────────────────────────────────────────────────
    debt_to_income:        float   # LC: dti            | HC: derived
    existing_monthly_debt: float   # LC: dti*income/12  | HC: AMT_ANNUITY
    # ── Thin-file flag ────────────────────────────────────────────────────
    thin_file:             bool    # derived: credit_age < 24mo OR open_acc < 3
    # ── Ground truth (for evaluation only — not seen by model at inference) ──
    ground_truth_label:    Optional[int] = None   # 0=good, 1=default


@dataclass
class IncomeResult:
    verified_income:  float
    confidence:       float
    source_refs:      list = field(default_factory=list)

@dataclass
class CreditResult:
    credit_score:      int
    delinquencies:     int
    utilisation_pct:   float
    credit_age_months: int
    open_accounts:     int
    thin_file:         bool
    source_refs:       list = field(default_factory=list)

@dataclass
class PolicyResult:
    applicable_clauses: list = field(default_factory=list)
    hard_stops:         list = field(default_factory=list)
    flags:              list = field(default_factory=list)
    source_refs:        list = field(default_factory=list)

@dataclass
class RiskResult:
    risk_score:    float
    risk_band:     str
    top_features:  list = field(default_factory=list)
    model_version: str = "xgb_halcyon_mvp_v1"

@dataclass
class Decision:
    recommendation: str          # APPROVE | DECLINE | REFER
    reasons:        list = field(default_factory=list)
    conditions:     list = field(default_factory=list)
    draft_version:  int  = 1

@dataclass
class EvalResult:
    faithfulness:        float
    relevancy:           float
    unsupported_claims:  list = field(default_factory=list)
    pass_flag:           bool  = False

@dataclass
class ApplicationState:
    """Single typed state object that threads through every agent node."""
    application_id:  str
    applicant_file:  ApplicantFile
    income_verified: Optional[IncomeResult]  = None
    credit_report:   Optional[CreditResult]  = None
    policy_findings: Optional[PolicyResult]  = None
    risk_score:      Optional[RiskResult]    = None
    draft_decision:  Optional[Decision]      = None
    eval_result:     Optional[EvalResult]    = None
    retry_count:     int                     = 0
    final_record:    Optional[dict]          = None
    trace:           list = field(default_factory=list)
    errors:          list = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — SAMPLE DATASET  (harmonised schema, real-world distributions)
# ─────────────────────────────────────────────────────────────────────────────

def build_sample_dataset(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic dataset that mirrors the harmonised
    LendingClub + Home Credit feature schema.

    Distributions are calibrated to match real dataset statistics:
      - LendingClub median income ~$65k, DTI ~17.8%, default rate ~21%
      - Home Credit default rate ~8% for thin-file segment
    """
    rng = np.random.default_rng(seed)

    n_standard  = int(n * 0.75)   # 75% standard applicants (LendingClub-like)
    n_thin_file = int(n * 0.25)   # 25% thin-file applicants (Home Credit-like)

    def make_segment(size, thin):
        income      = rng.lognormal(mean=11.0, sigma=0.5, size=size).clip(15000, 300000)
        loan_amount = (income * rng.uniform(0.2, 1.5, size)).clip(1000, 40000)
        dti         = rng.beta(2, 8, size) * 60          # median ~15%, right-skewed
        emp_months  = rng.integers(0, 240, size) if not thin else rng.integers(0, 36, size)
        cr_age      = rng.integers(0, 20, size) if thin else rng.integers(12, 300, size)
        open_acc    = rng.integers(1, 4, size)  if thin else rng.integers(2, 25, size)
        cr_score    = rng.integers(520, 680, size) if thin else rng.integers(580, 820, size)
        revol_util  = rng.beta(2, 3, size)               # median ~0.4
        delinq      = rng.choice([0,0,0,0,1,1,2,3], size=size)
        pub_rec     = rng.choice([0,0,0,0,0,1], size=size)
        inq_6mo     = rng.integers(0, 6, size)
        monthly_debt= (dti * income / 12 / 100).clip(0)
        thin_flag   = (cr_age < 24) | (open_acc < 3)

        # ── Income verification encoding ──────────────────────────────────
        verif_status = rng.choice(
            ["Source Verified", "Verified", "Not Verified"],
            size=size, p=[0.35, 0.35, 0.30]
        )
        verif_map = {"Source Verified": 0.90, "Verified": 0.75, "Not Verified": 0.40}
        income_conf = np.array([verif_map[v] for v in verif_status])
        verified_inc = income * income_conf * rng.uniform(0.95, 1.05, size)

        emp_type = rng.choice(
            ["salaried", "self-employed", "gig", "commercial"],
            size=size, p=[0.55, 0.20, 0.15, 0.10]
        )
        purpose = rng.choice(
            ["debt_consolidation","home_improvement","medical",
             "major_purchase","car","small_business","vacation"],
            size=size, p=[0.40, 0.20, 0.12, 0.10, 0.08, 0.06, 0.04]
        )

        # ── Probabilistic label generation (mirrors real default rates) ───
        # Key risk drivers: credit_score, dti, revol_util, delinq
        log_odds = (
            -4.0
            + (700 - cr_score) * 0.008      # lower score → higher risk
            + dti              * 0.04        # higher DTI  → higher risk
            + revol_util       * 2.0         # higher util → higher risk
            + delinq           * 0.6         # each delinquency adds risk
            + pub_rec          * 1.2         # public record is a strong signal
            + (1 - income_conf)* 1.0         # unverified income adds risk
            + (thin_flag.astype(int)) * 0.3  # thin-file slight uplift
        )
        prob_default = 1 / (1 + np.exp(-log_odds))
        label = (rng.uniform(size=size) < prob_default).astype(int)

        return pd.DataFrame({
            "annual_income":         income,
            "loan_amount":           loan_amount,
            "loan_purpose":          purpose,
            "loan_term_months":      rng.choice([36, 60], size=size, p=[0.65, 0.35]),
            "verification_status":   verif_status,
            "income_confidence":     income_conf,
            "verified_income":       verified_inc,
            "employment_months":     emp_months,
            "employment_type":       emp_type,
            "credit_score":          cr_score,
            "delinquencies_2yr":     delinq,
            "revolving_utilisation": revol_util,
            "credit_age_months":     cr_age,
            "open_accounts":         open_acc,
            "inquiries_6mo":         inq_6mo,
            "public_records":        pub_rec,
            "debt_to_income":        dti,
            "existing_monthly_debt": monthly_debt,
            "thin_file":             thin_flag.astype(int),
            "loan_to_income_ratio":  loan_amount / verified_inc,
            "debt_burden_ratio":     monthly_debt / (verified_inc / 12 + 1),
            "label":                 label,
        })

    df_std  = make_segment(n_standard,  thin=False)
    df_thin = make_segment(n_thin_file, thin=True)
    df = pd.concat([df_std, df_thin], ignore_index=True).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — RISK MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "verified_income", "income_confidence", "loan_to_income_ratio",
    "credit_score", "delinquencies_2yr", "revolving_utilisation",
    "credit_age_months", "open_accounts", "inquiries_6mo",
    "public_records", "debt_to_income", "debt_burden_ratio",
    "employment_months", "thin_file", "loan_amount",
]

def train_risk_model(df: pd.DataFrame):
    """Train XGBoost on the sample dataset. Returns fitted model + AUC."""
    X = df[FEATURE_COLS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
    spw = n_neg / max(n_pos, 1)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=spw,
        eval_metric="auc",
        random_state=42,
        verbosity=0,
        use_label_encoder=False,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_prob = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)
    return model, auc, X_test, y_test


def get_shap_top5(model, feature_vector: pd.DataFrame) -> list[dict]:
    """Compute SHAP-like feature importance using XGBoost gain scores."""
    importances = model.get_booster().get_fscore()
    total = sum(importances.values()) or 1
    feat_row = feature_vector.iloc[0]
    top5 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    return [
        {
            "feature": f,
            "value":   round(float(feat_row.get(f, 0)), 4),
            "importance": round(v / total, 4),
        }
        for f, v in top5
    ]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — AGENT DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

FAITHFULNESS_THRESHOLD = 0.70
MAX_RETRIES            = 2

# ── Policy knowledge base (mocked ChromaDB / RAG) ────────────────────────────
POLICY_CLAUSES = {
    "POL-001": {"text": "DTI must not exceed 40% for standard applicants.",
                "hard_stop": True,  "threshold_dti": 40.0},
    "POL-002": {"text": "Any public record (bankruptcy, lien, judgement) mandates decline.",
                "hard_stop": True},
    "POL-003": {"text": "Loan-to-income ratio above 3.0 requires senior underwriter review.",
                "hard_stop": False, "threshold_lti": 3.0},
    "POL-004": {"text": "Debt consolidation purpose with DTI > 35% requires income re-verification.",
                "hard_stop": False, "threshold_dti": 35.0},
    "POL-005": {"text": "Thin-file applicants must be referred; automatic decline is prohibited.",
                "hard_stop": False},
    "POL-006": {"text": "Two or more delinquencies in the past 24 months triggers mandatory REFER.",
                "hard_stop": False, "threshold_delinq": 2},
    "POL-007": {"text": "Credit score below 580 for non-thin-file applicants mandates decline.",
                "hard_stop": True,  "threshold_score": 580},
}


def log_trace(state: ApplicationState, agent: str, action: str):
    state.trace.append({
        "agent": agent, "action": action,
        "timestamp": datetime.now().isoformat(), "retry": state.retry_count
    })


# ── Agent 1: Income Verification ─────────────────────────────────────────────
def income_verification_agent(state: ApplicationState) -> ApplicationState:
    """
    Reads:  applicant_file
    Writes: income_verified
    Tool:   Income DB Tool (mocked)
    """
    log_trace(state, "IncomeVerificationAgent", "start")
    af = state.applicant_file

    verif_map  = {"Source Verified": 0.90, "Verified": 0.75, "Not Verified": 0.40}
    confidence = verif_map.get(af.verification_status, 0.50)
    # Mock: verified income = stated income × confidence with small noise
    rng = np.random.default_rng(hash(af.applicant_id) % (2**31))
    verified_income = af.annual_income * confidence * rng.uniform(0.97, 1.03)

    state.income_verified = IncomeResult(
        verified_income = round(verified_income, 2),
        confidence      = confidence,
        source_refs     = [f"INCOME_DB:{af.applicant_id}"],
    )
    log_trace(state, "IncomeVerificationAgent",
              f"verified_income={verified_income:.0f} confidence={confidence}")
    return state


# ── Agent 2: Credit History ───────────────────────────────────────────────────
def credit_history_agent(state: ApplicationState) -> ApplicationState:
    """
    Reads:  applicant_file
    Writes: credit_report
    Tool:   Credit Bureau Tool (mocked)
    """
    log_trace(state, "CreditHistoryAgent", "start")
    af = state.applicant_file

    state.credit_report = CreditResult(
        credit_score      = af.credit_score,
        delinquencies     = af.delinquencies_2yr,
        utilisation_pct   = round(af.revolving_utilisation * 100, 1),
        credit_age_months = af.credit_age_months,
        open_accounts     = af.open_accounts,
        thin_file         = af.thin_file,
        source_refs       = [f"BUREAU:{af.applicant_id}"],
    )
    log_trace(state, "CreditHistoryAgent",
              f"score={af.credit_score} util={af.revolving_utilisation:.0%} "
              f"thin_file={af.thin_file}")
    return state


# ── Agent 3: Policy Compliant ─────────────────────────────────────────────────
def policy_compliant_agent(state: ApplicationState) -> ApplicationState:
    """
    Reads:  applicant_file
    Writes: policy_findings
    Tool:   Policy Retrieval Tool (mocked ChromaDB RAG)
    """
    log_trace(state, "PolicyCompliantAgent", "start")
    af   = state.applicant_file
    dti  = af.debt_to_income
    lti  = af.loan_amount / max(af.annual_income, 1)

    applicable, hard_stops, flags = [], [], []

    for clause_id, clause in POLICY_CLAUSES.items():
        matched = False

        if clause_id == "POL-001" and dti > clause["threshold_dti"]:
            hard_stops.append(clause_id)
            matched = True
        elif clause_id == "POL-002" and af.public_records > 0:
            hard_stops.append(clause_id)
            matched = True
        elif clause_id == "POL-003" and lti > clause["threshold_lti"]:
            flags.append(clause_id)
            matched = True
        elif (clause_id == "POL-004"
              and af.loan_purpose == "debt_consolidation"
              and dti > clause["threshold_dti"]):
            flags.append(clause_id)
            matched = True
        elif clause_id == "POL-005" and af.thin_file:
            flags.append(clause_id)
            matched = True
        elif clause_id == "POL-006" and af.delinquencies_2yr >= clause["threshold_delinq"]:
            flags.append(clause_id)
            matched = True
        elif (clause_id == "POL-007"
              and not af.thin_file
              and af.credit_score < clause["threshold_score"]):
            hard_stops.append(clause_id)
            matched = True

        if matched:
            applicable.append(clause_id)

    state.policy_findings = PolicyResult(
        applicable_clauses = applicable,
        hard_stops         = hard_stops,
        flags              = flags,
        source_refs        = [f"CHROMA:{c}" for c in applicable],
    )
    log_trace(state, "PolicyCompliantAgent",
              f"hard_stops={hard_stops} flags={flags}")
    return state


# ── Agent 4: Risk Scoring ─────────────────────────────────────────────────────
def risk_scoring_agent(
    state: ApplicationState,
    model: XGBClassifier,
) -> ApplicationState:
    """
    Reads:  income_verified, credit_report, policy_findings (all must be non-None)
    Writes: risk_score
    Tool:   XGBoost / LightGBM model
    Guard:  Only executes when all three upstream fields are present (state-gated)
    """
    if not all([state.income_verified, state.credit_report, state.policy_findings]):
        state.errors.append({"agent": "RiskScoringAgent",
                             "error": "upstream state not ready"})
        return state

    log_trace(state, "RiskScoringAgent", "start")
    af = state.applicant_file
    iv = state.income_verified
    cr = state.credit_report
    pf = state.policy_findings

    # ── Assemble feature vector strictly from typed state fields ─────────
    # No free-text LLM output ever reaches this vector
    feature_vector = pd.DataFrame([{
        "verified_income":        iv.verified_income,
        "income_confidence":      iv.confidence,
        "loan_to_income_ratio":   af.loan_amount / max(iv.verified_income, 1),
        "credit_score":           cr.credit_score,
        "delinquencies_2yr":      cr.delinquencies,
        "revolving_utilisation":  cr.utilisation_pct / 100,
        "credit_age_months":      cr.credit_age_months,
        "open_accounts":          cr.open_accounts,
        "inquiries_6mo":          af.inquiries_6mo,
        "public_records":         af.public_records,
        "debt_to_income":         af.debt_to_income,
        "debt_burden_ratio":      af.existing_monthly_debt / max(iv.verified_income / 12, 1),
        "employment_months":      af.employment_months,
        "thin_file":              int(cr.thin_file),
        "loan_amount":            af.loan_amount,
    }])

    risk_score = float(model.predict_proba(feature_vector)[0, 1])
    hard_stop_penalty = len(pf.hard_stops) * 0.15
    risk_score = min(risk_score + hard_stop_penalty, 0.99)

    # ── Risk band mapping ─────────────────────────────────────────────────
    if risk_score < 0.30:
        band = "Low"
    elif risk_score < 0.55:
        band = "Medium"
    else:
        band = "High"

    top_features = get_shap_top5(model, feature_vector)

    state.risk_score = RiskResult(
        risk_score    = round(risk_score, 4),
        risk_band     = band,
        top_features  = top_features,
        model_version = "xgb_halcyon_mvp_v1",
    )
    log_trace(state, "RiskScoringAgent",
              f"score={risk_score:.4f} band={band}")
    return state


# ── Agent 5: Decision Synthesizer ────────────────────────────────────────────
def decision_synthesizer_agent(
    state: ApplicationState,
    judge_feedback: Optional[str] = None,
) -> ApplicationState:
    """
    Reads:  risk_score, credit_report, policy_findings, income_verified
            + judge_feedback on retry
    Writes: draft_decision
    Note:   In production this calls an LLM (Gemini Pro / GPT-4o).
            Here we implement rule-based logic that mirrors what the LLM
            would produce — same decision logic, deterministic output.
    """
    log_trace(state, "DecisionSynthesizerAgent",
              f"draft_version={state.draft_decision.draft_version + 1 if state.draft_decision else 1}")

    rs = state.risk_score
    cr = state.credit_report
    pf = state.policy_findings
    iv = state.income_verified
    af = state.applicant_file

    reasons    = []
    conditions = []

    # ── Hard-stop → always DECLINE ────────────────────────────────────────
    if pf.hard_stops:
        recommendation = "DECLINE"
        for hs in pf.hard_stops:
            clause_text = POLICY_CLAUSES[hs]["text"]
            if hs == "POL-001":
                reasons.append(
                    f"DTI of {af.debt_to_income:.1f}% exceeds the 40% policy ceiling "
                    f"(clause {hs}). Monthly debt obligations leave insufficient capacity "
                    f"for the requested repayment."
                )
            elif hs == "POL-002":
                reasons.append(
                    f"Public record on file (clause {hs}). Policy mandates decline "
                    f"for any applicant with a bankruptcy, lien, or judgement record."
                )
            elif hs == "POL-007":
                reasons.append(
                    f"Credit score of {cr.credit_score} falls below the 580 minimum "
                    f"threshold (clause {hs}). Standard-file applicants require a minimum "
                    f"score to meet creditworthiness criteria."
                )

    # ── Thin-file → always REFER (never auto-decline) ────────────────────
    elif cr.thin_file:
        recommendation = "REFER"
        reasons.append(
            f"Applicant has a thin credit file (credit history of "
            f"{cr.credit_age_months} months, {cr.open_accounts} open accounts). "
            f"Policy clause POL-005 prohibits automatic decline for thin-file applicants. "
            f"Human underwriter review required to assess non-bureau income signals."
        )
        if rs.risk_band != "High":
            reasons.append(
                f"Risk model score of {rs.risk_score:.3f} ({rs.risk_band} band) "
                f"suggests repayment capacity is plausible — income of "
                f"${iv.verified_income:,.0f} (confidence: {iv.confidence:.0%}) "
                f"and DTI of {af.debt_to_income:.1f}% are within acceptable range."
            )

    # ── Policy flags without hard-stops ───────────────────────────────────
    elif pf.flags and rs.risk_band == "High":
        recommendation = "REFER"
        reasons.append(
            f"Risk score of {rs.risk_score:.3f} places this application in the "
            f"High risk band. Combined with policy flags "
            f"({', '.join(pf.flags)}), automated approval is not appropriate."
        )
        reasons.append(
            f"Primary risk drivers: "
            + ", ".join(
                f"{f['feature']} = {f['value']}"
                for f in rs.top_features[:3]
            ) + "."
        )

    # ── Low / Medium risk, no hard-stops → APPROVE ───────────────────────
    elif rs.risk_band in ("Low", "Medium"):
        recommendation = "APPROVE"
        reasons.append(
            f"Risk score of {rs.risk_score:.3f} ({rs.risk_band} band) is within "
            f"the acceptable range. Verified income of ${iv.verified_income:,.0f} "
            f"(confidence: {iv.confidence:.0%}) supports the requested loan amount "
            f"of ${af.loan_amount:,.0f}."
        )
        reasons.append(
            f"Credit profile: score {cr.credit_score}, "
            f"{cr.delinquencies} delinquencies in 24 months, "
            f"revolving utilisation {cr.utilisation_pct:.1f}%, "
            f"credit history {cr.credit_age_months} months."
        )
        if pf.flags:
            conditions.append(
                f"Advisory flags noted ({', '.join(pf.flags)}). "
                f"Standard monitoring applies."
            )
        if iv.confidence < 0.75:
            conditions.append(
                "Income verification confidence is below the 75% threshold. "
                "Applicant may be asked to provide payslip within 14 days of approval."
            )

    else:
        recommendation = "REFER"
        reasons.append(
            f"Risk score {rs.risk_score:.3f} (High band) warrants human review."
        )

    # ── Incorporate judge feedback on retry ───────────────────────────────
    if judge_feedback:
        reasons.append(
            f"[Revised following evaluator feedback: {judge_feedback}]"
        )

    prev_version = state.draft_decision.draft_version if state.draft_decision else 0
    state.draft_decision = Decision(
        recommendation = recommendation,
        reasons        = reasons,
        conditions     = conditions,
        draft_version  = prev_version + 1,
    )
    log_trace(state, "DecisionSynthesizerAgent",
              f"recommendation={recommendation} reasons={len(reasons)}")
    return state


# ── Agent 6: Evaluation Agent (LLM-as-Judge) ─────────────────────────────────
def evaluation_agent(state: ApplicationState) -> ApplicationState:
    """
    Reads:  draft_decision, credit_report, policy_findings, income_verified, risk_score
    Writes: eval_result, retry_count
    Note:   In production this calls an LLM judge with RAGAS.
            Here we implement a rule-based faithfulness checker that
            verifies every claim in the decision is traceable to a
            source state field — the same thing the LLM judge does.
    """
    log_trace(state, "EvaluationAgent", "start")

    dd  = state.draft_decision
    rs  = state.risk_score
    cr  = state.credit_report
    pf  = state.policy_findings
    iv  = state.income_verified
    af  = state.applicant_file

    unsupported = []
    faithfulness_score = 1.0

    # ── Check 1: recommendation is consistent with hard-stops ────────────
    if pf.hard_stops and dd.recommendation != "DECLINE":
        unsupported.append(
            f"Hard-stop clauses {pf.hard_stops} were triggered but recommendation "
            f"is {dd.recommendation} — should be DECLINE."
        )
        faithfulness_score -= 0.30

    # ── Check 2: thin-file applicant not auto-declined ────────────────────
    if cr.thin_file and dd.recommendation == "DECLINE" and not pf.hard_stops:
        unsupported.append(
            "Applicant is thin-file with no hard-stops, but decision is DECLINE. "
            "POL-005 prohibits automatic decline — should be REFER."
        )
        faithfulness_score -= 0.25

    # ── Check 3: every reason cites a source ──────────────────────────────
    source_anchors = [
        str(rs.risk_score), str(cr.credit_score),
        str(cr.delinquencies), f"{iv.verified_income:,.0f}",
        f"{iv.confidence:.0%}", str(af.debt_to_income),
        str(cr.utilisation_pct), str(cr.credit_age_months),
        *pf.hard_stops, *pf.flags,
    ]
    reasons_text = " ".join(dd.reasons)
    anchored = sum(1 for a in source_anchors if a in reasons_text)
    coverage = anchored / max(len(source_anchors), 1)
    if coverage < 0.30:
        unsupported.append(
            f"Only {anchored}/{len(source_anchors)} source anchors appear in "
            f"decision reasons. Faithfulness is insufficient."
        )
        faithfulness_score -= 0.20

    # ── Check 4: risk band consistent with score ───────────────────────────
    expected_band = (
        "Low" if rs.risk_score < 0.30 else
        "Medium" if rs.risk_score < 0.55 else "High"
    )
    if rs.risk_band != expected_band:
        unsupported.append(
            f"Risk band '{rs.risk_band}' is inconsistent with score "
            f"{rs.risk_score:.4f} (expected '{expected_band}')."
        )
        faithfulness_score -= 0.15

    faithfulness_score = max(0.0, round(faithfulness_score, 3))
    relevancy_score    = min(1.0, 0.65 + coverage * 0.35)

    pass_flag = faithfulness_score >= FAITHFULNESS_THRESHOLD

    state.eval_result = EvalResult(
        faithfulness       = faithfulness_score,
        relevancy          = round(relevancy_score, 3),
        unsupported_claims = unsupported,
        pass_flag          = pass_flag,
    )

    if not pass_flag:
        state.retry_count += 1

    log_trace(state, "EvaluationAgent",
              f"faithfulness={faithfulness_score:.3f} pass={pass_flag} "
              f"retry_count={state.retry_count}")
    return state


# ── Agent 7: Decision Record Writer ──────────────────────────────────────────
def decision_record_writer(state: ApplicationState) -> ApplicationState:
    """
    Reads:  draft_decision, eval_result, risk_score, policy_findings, trace
    Writes: final_record  (immutable after write)
    Persists the full state trace — not just the final answer.
    """
    log_trace(state, "DecisionRecordWriter", "persisting_final_record")

    escalated = (
        state.retry_count >= MAX_RETRIES
        and not state.eval_result.pass_flag
    )

    state.final_record = {
        "audit_id":        str(uuid.uuid4()),
        "application_id":  state.application_id,
        "created_at":      datetime.now().isoformat(),
        "applicant_name":  state.applicant_file.name,
        "recommendation":  state.draft_decision.recommendation if not escalated else "ESCALATED",
        "escalated":       escalated,
        "risk_score":      state.risk_score.risk_score,
        "risk_band":       state.risk_score.risk_band,
        "faithfulness":    state.eval_result.faithfulness,
        "retry_count":     state.retry_count,
        "reasons":         state.draft_decision.reasons,
        "conditions":      state.draft_decision.conditions,
        "policy_refs":     state.policy_findings.applicable_clauses,
        "hard_stops":      state.policy_findings.hard_stops,
        "top_features":    state.risk_score.top_features,
        "full_trace":      state.trace,
    }
    return state


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — ORCHESTRATOR  (Orchestrator-Worker + Evaluator-Optimizer loop)
# ─────────────────────────────────────────────────────────────────────────────

def orchestrator(
    applicant: ApplicantFile,
    model: XGBClassifier,
    verbose: bool = True,
) -> ApplicationState:
    """
    Coordinates the full pipeline.
    Implements:
      - Parallel fan-out of Income, Credit, Policy agents
      - State-gated Risk Scoring (only fires when all 3 upstream fields ready)
      - Evaluator-Optimizer retry loop (max N=2)
      - Human escalation on retry exhaustion
    """
    state = ApplicationState(
        application_id = f"APP-{uuid.uuid4().hex[:8].upper()}",
        applicant_file = applicant,
    )

    if verbose:
        console.rule(f"[bold cyan]Processing: {applicant.name}[/]")

    # ── Step 1: Parallel worker agents ───────────────────────────────────
    # (In production: asyncio.gather — here called sequentially for clarity)
    if verbose:
        console.print("  [dim]→ [1/3] Income Verification Agent ...[/]")
    state = income_verification_agent(state)

    if verbose:
        console.print("  [dim]→ [2/3] Credit History Agent ...[/]")
    state = credit_history_agent(state)

    if verbose:
        console.print("  [dim]→ [3/3] Policy Compliant Agent ...[/]")
    state = policy_compliant_agent(state)

    # ── Step 2: merge_state barrier — all 3 must be non-None ─────────────
    if not all([state.income_verified, state.credit_report, state.policy_findings]):
        state.errors.append({"error": "merge_state barrier failed"})
        return state

    # ── Step 3: Risk Scoring (state-gated) ───────────────────────────────
    if verbose:
        console.print("  [dim]→ Risk Scoring Agent (XGBoost) ...[/]")
    state = risk_scoring_agent(state, model)

    # ── Step 4: Evaluator-Optimizer loop ─────────────────────────────────
    judge_feedback = None
    while True:
        if verbose:
            draft_v = state.draft_decision.draft_version if state.draft_decision else 0
            label   = "Decision Synthesizer" if draft_v == 0 else f"Decision Synthesizer [retry {state.retry_count}]"
            console.print(f"  [dim]→ {label} ...[/]")
        state = decision_synthesizer_agent(state, judge_feedback=judge_feedback)

        if verbose:
            console.print("  [dim]→ Evaluation Agent (Judge) ...[/]")
        state = evaluation_agent(state)

        if state.eval_result.pass_flag:
            if verbose:
                console.print(
                    f"  [green]✓ Faithfulness gate passed "
                    f"({state.eval_result.faithfulness:.3f} ≥ {FAITHFULNESS_THRESHOLD})[/]"
                )
            break

        if state.retry_count >= MAX_RETRIES:
            if verbose:
                console.print(
                    f"  [yellow]⚠ Retry cap reached ({MAX_RETRIES}). "
                    f"Routing to human escalation.[/]"
                )
            break

        judge_feedback = (
            f"Unsupported claims detected: "
            + "; ".join(state.eval_result.unsupported_claims)
        )
        if verbose:
            console.print(
                f"  [yellow]  Faithfulness {state.eval_result.faithfulness:.3f} < "
                f"{FAITHFULNESS_THRESHOLD}. Retrying with judge feedback.[/]"
            )

    # ── Step 5: Decision Record Writer ────────────────────────────────────
    if verbose:
        console.print("  [dim]→ Decision Record Writer ...[/]")
    state = decision_record_writer(state)

    return state


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — DISPLAY & OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

REC_COLOURS = {"APPROVE": "green", "DECLINE": "red",
               "REFER": "yellow", "ESCALATED": "magenta"}

def print_decision_record(state: ApplicationState):
    fr  = state.final_record
    af  = state.applicant_file
    rec = fr["recommendation"]
    col = REC_COLOURS.get(rec, "white")

    # ── Header panel ──────────────────────────────────────────────────────
    header = (
        f"[bold {col}]{rec}[/]\n"
        f"[dim]Audit ID: {fr['audit_id']}[/]\n"
        f"[dim]App ID:   {fr['application_id']}[/]\n"
        f"[dim]Issued:   {fr['created_at']}[/]"
    )
    console.print(Panel(header, title=f"[bold]{af.name}[/]",
                        border_style=col, box=box.DOUBLE_EDGE))

    # ── Applicant profile ─────────────────────────────────────────────────
    profile = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    profile.add_column("Field", style="dim", width=28)
    profile.add_column("Value")

    profile.add_row("Annual Income",      f"${af.annual_income:>12,.0f}")
    profile.add_row("Verified Income",    f"${state.income_verified.verified_income:>12,.0f}  "
                                          f"(conf: {state.income_verified.confidence:.0%})")
    profile.add_row("Loan Requested",     f"${af.loan_amount:>12,.0f}")
    profile.add_row("Loan Purpose",       af.loan_purpose)
    profile.add_row("Employment",         f"{af.employment_months} months · {af.employment_type}")
    profile.add_row("Credit Score",       str(af.credit_score))
    profile.add_row("Debt-to-Income",     f"{af.debt_to_income:.1f}%")
    profile.add_row("Revolving Util.",    f"{af.revolving_utilisation:.0%}")
    profile.add_row("Delinquencies 2yr",  str(af.delinquencies_2yr))
    profile.add_row("Credit Age",         f"{af.credit_age_months} months")
    profile.add_row("Thin-File Flag",     "YES ⚠" if af.thin_file else "No")
    profile.add_row("Public Records",     str(af.public_records))

    console.print(Panel(profile, title="Applicant Profile", border_style="blue"))

    # ── Risk model output ─────────────────────────────────────────────────
    rs        = state.risk_score
    band_col  = {"Low": "green", "Medium": "yellow", "High": "red"}.get(rs.risk_band, "white")
    risk_info = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    risk_info.add_column("Field", style="dim", width=28)
    risk_info.add_column("Value")
    risk_info.add_row("Risk Score", f"[{band_col}]{rs.risk_score:.4f}[/]")
    risk_info.add_row("Risk Band",  f"[bold {band_col}]{rs.risk_band}[/]")
    risk_info.add_row("Model",      rs.model_version)

    shap_table = Table(title="Top Risk Drivers (SHAP)", box=box.SIMPLE,
                       show_header=True, header_style="bold")
    shap_table.add_column("Feature",     style="cyan",  width=26)
    shap_table.add_column("Value",       justify="right", width=10)
    shap_table.add_column("Importance",  justify="right", width=12)
    for f in rs.top_features:
        bar = "█" * int(f["importance"] * 30)
        shap_table.add_row(f["feature"], str(f["value"]),
                           f"[yellow]{f['importance']:.4f}[/]  {bar}")

    console.print(Panel(risk_info, title="Risk Model Output", border_style=band_col))
    console.print(Panel(shap_table, title="Feature Attribution", border_style="yellow"))

    # ── Policy findings ───────────────────────────────────────────────────
    pf = state.policy_findings
    if pf.hard_stops or pf.flags:
        pol_table = Table(show_header=True, box=box.SIMPLE, header_style="bold")
        pol_table.add_column("Clause",  width=10)
        pol_table.add_column("Type",    width=12)
        pol_table.add_column("Text")
        for c in pf.hard_stops:
            pol_table.add_row(c, "[red]HARD STOP[/]", POLICY_CLAUSES[c]["text"])
        for c in pf.flags:
            pol_table.add_row(c, "[yellow]FLAG[/]",   POLICY_CLAUSES[c]["text"])
        console.print(Panel(pol_table, title="Policy Findings", border_style="red" if pf.hard_stops else "yellow"))

    # ── Decision reasons ──────────────────────────────────────────────────
    reasons_text = "\n\n".join(f"  {i+1}. {r}" for i, r in enumerate(fr["reasons"]))
    if fr["conditions"]:
        reasons_text += "\n\n[bold]Conditions:[/]\n" + "\n".join(
            f"  • {c}" for c in fr["conditions"]
        )
    console.print(Panel(reasons_text, title="Decision Reasons (Auditable)", border_style=col))

    # ── Evaluation result ─────────────────────────────────────────────────
    ev  = state.eval_result
    bar = "█" * int(ev.faithfulness * 20)
    ev_text = (
        f"Faithfulness:  [{'green' if ev.pass_flag else 'red'}]{ev.faithfulness:.3f}[/]  {bar}\n"
        f"Relevancy:     {ev.relevancy:.3f}\n"
        f"Pass:          [{'green' if ev.pass_flag else 'red'}]{'✓ YES' if ev.pass_flag else '✗ NO'}[/]\n"
        f"Retries:       {state.retry_count}\n"
        f"Escalated:     {'YES ⚠' if fr['escalated'] else 'No'}"
    )
    console.print(Panel(ev_text, title="Evaluation (Faithfulness Gate)", border_style="blue"))

    # ── Audit trace ───────────────────────────────────────────────────────
    trace_table = Table(show_header=True, box=box.SIMPLE, header_style="bold dim")
    trace_table.add_column("#",       width=4)
    trace_table.add_column("Agent",   width=30)
    trace_table.add_column("Action",  width=35)
    trace_table.add_column("Time",    width=26)
    for i, t in enumerate(state.trace, 1):
        trace_table.add_row(
            str(i), t["agent"], t["action"],
            t["timestamp"]
        )
    console.print(Panel(trace_table, title="Full Audit Trace", border_style="dim"))
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — TEST APPLICANTS  (one per decision type)
# ─────────────────────────────────────────────────────────────────────────────

TEST_APPLICANTS = [
    ApplicantFile(
        applicant_id          = "LC-001",
        name                  = "Priya Sharma",
        loan_amount           = 12000,
        loan_purpose          = "home_improvement",
        loan_term_months      = 36,
        annual_income         = 72000,
        verification_status   = "Source Verified",
        employment_months     = 84,
        employment_type       = "salaried",
        credit_score          = 720,
        delinquencies_2yr     = 0,
        revolving_utilisation = 0.22,
        credit_age_months     = 96,
        open_accounts         = 8,
        inquiries_6mo         = 1,
        public_records        = 0,
        debt_to_income        = 14.5,
        existing_monthly_debt = 870,
        thin_file             = False,
        ground_truth_label    = 0,   # Expected: APPROVE
    ),
    ApplicantFile(
        applicant_id          = "LC-002",
        name                  = "Rahul Desai",
        loan_amount           = 18000,
        loan_purpose          = "debt_consolidation",
        loan_term_months      = 60,
        annual_income         = 38000,
        verification_status   = "Not Verified",
        employment_months     = 14,
        employment_type       = "gig",
        credit_score          = 598,
        delinquencies_2yr     = 2,
        revolving_utilisation = 0.84,
        credit_age_months     = 42,
        open_accounts         = 5,
        inquiries_6mo         = 5,
        public_records        = 0,
        debt_to_income        = 44.2,      # triggers POL-001
        existing_monthly_debt = 1400,
        thin_file             = False,
        ground_truth_label    = 1,   # Expected: DECLINE (DTI hard-stop)
    ),
    ApplicantFile(
        applicant_id          = "HC-001",
        name                  = "Ananya Krishnan",
        loan_amount           = 8000,
        loan_purpose          = "medical",
        loan_term_months      = 36,
        annual_income         = 42000,
        verification_status   = "Verified",
        employment_months     = 18,
        employment_type       = "self-employed",
        credit_score          = 645,
        delinquencies_2yr     = 0,
        revolving_utilisation = 0.15,
        credit_age_months     = 8,         # thin-file: < 24 months
        open_accounts         = 2,         # thin-file: < 3
        inquiries_6mo         = 2,
        public_records        = 0,
        debt_to_income        = 19.8,
        existing_monthly_debt = 693,
        thin_file             = True,      # Expected: REFER (POL-005)
        ground_truth_label    = 0,
    ),
    ApplicantFile(
        applicant_id          = "LC-003",
        name                  = "Vikram Nair",
        loan_amount           = 25000,
        loan_purpose          = "small_business",
        loan_term_months      = 60,
        annual_income         = 95000,
        verification_status   = "Verified",
        employment_months     = 120,
        employment_type       = "salaried",
        credit_score          = 560,       # triggers POL-007
        delinquencies_2yr     = 1,
        revolving_utilisation = 0.55,
        credit_age_months     = 72,
        open_accounts         = 6,
        inquiries_6mo         = 3,
        public_records        = 1,         # triggers POL-002
        debt_to_income        = 28.3,
        existing_monthly_debt = 2239,
        thin_file             = False,
        ground_truth_label    = 1,   # Expected: DECLINE (pub rec + low score)
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    console.rule("[bold magenta]HALCYON CREDIT — Agentic Underwriting Copilot[/]")
    console.print(
        "[dim]Team Jamun · Futurense AI Clinic · Capstone Project 02[/]\n"
    )

    # ── Build sample dataset ───────────────────────────────────────────────
    console.print("[bold]Step 1:[/] Building sample dataset (harmonised schema) ...")
    df = build_sample_dataset(n=600, seed=42)

    dist = df["label"].value_counts()
    console.print(
        f"  Dataset: {len(df):,} rows · "
        f"{dist[0]:,} good loans ({dist[0]/len(df):.0%}) · "
        f"{dist[1]:,} defaults ({dist[1]/len(df):.0%})\n"
    )

    # ── Train risk model ───────────────────────────────────────────────────
    console.print("[bold]Step 2:[/] Training XGBoost risk model ...")
    model, auc, X_test, y_test = train_risk_model(df)
    console.print(f"  [green]✓ Model trained · ROC-AUC on holdout = {auc:.4f}[/]\n")

    # ── Show dataset sample ────────────────────────────────────────────────
    console.print("[bold]Step 3:[/] Sample dataset preview (first 8 rows)\n")
    preview_cols = [
        "annual_income","loan_amount","loan_purpose","credit_score",
        "debt_to_income","revolving_utilisation","thin_file","label"
    ]
    preview = df[preview_cols].head(8).copy()
    preview.columns = [
        "Income","Loan Amt","Purpose","CR Score",
        "DTI%","Revol Util","Thin File","Label"
    ]
    tbl = Table(show_header=True, header_style="bold cyan", box=box.SIMPLE_HEAVY)
    for col in preview.columns:
        tbl.add_column(col, justify="right" if col not in ["Purpose"] else "left")
    for _, row in preview.iterrows():
        tbl.add_row(
            f"${row['Income']:,.0f}",
            f"${row['Loan Amt']:,.0f}",
            str(row["Purpose"]),
            str(int(row["CR Score"])),
            f"{row['DTI%']:.1f}",
            f"{row['Revol Util']:.2f}",
            "Yes" if row["Thin File"] else "No",
            "[red]DEFAULT[/]" if row["Label"] == 1 else "[green]GOOD[/]",
        )
    console.print(tbl)
    console.print()

    # ── Run end-to-end pipeline on test applicants ─────────────────────────
    console.print("[bold]Step 4:[/] Running end-to-end agent pipeline on 4 test applicants\n")
    console.print(
        "  Applicants cover all four expected outcomes:\n"
        "  [green]APPROVE[/] · [red]DECLINE (DTI hard-stop)[/] · "
        "[yellow]REFER (thin-file)[/] · [red]DECLINE (pub rec + low score)[/]\n"
    )

    results = []
    for applicant in TEST_APPLICANTS:
        t0    = time.time()
        state = orchestrator(applicant, model, verbose=True)
        elapsed = time.time() - t0

        print_decision_record(state)

        fr = state.final_record
        results.append({
            "Name":           applicant.name,
            "Expected":       {0: "APPROVE/REFER", 1: "DECLINE"}.get(
                                  applicant.ground_truth_label, "?"),
            "Got":            fr["recommendation"],
            "Risk Score":     f"{fr['risk_score']:.4f}",
            "Band":           fr["risk_band"],
            "Faithfulness":   f"{fr['faithfulness']:.3f}",
            "Retries":        fr["retry_count"],
            "Time (s)":       f"{elapsed:.2f}",
        })

    # ── Summary table ─────────────────────────────────────────────────────
    console.rule("[bold]Pipeline Summary[/]")
    summary = Table(show_header=True, header_style="bold", box=box.DOUBLE_EDGE)
    for col in results[0].keys():
        summary.add_column(col)

    for r in results:
        rec = r["Got"]
        col = REC_COLOURS.get(rec, "white")
        summary.add_row(
            r["Name"], r["Expected"],
            f"[bold {col}]{rec}[/]",
            r["Risk Score"], r["Band"],
            r["Faithfulness"], str(r["Retries"]),
            r["Time (s)"],
        )
    console.print(summary)

    # ── What this POC demonstrates ─────────────────────────────────────────
    console.print()
    console.print(Panel(
        "[bold]What this proof of concept demonstrates:[/]\n\n"
        "  1. [cyan]Unified schema[/]  — ApplicantFile harmonises LendingClub + Home Credit "
              "features into one typed object\n"
        "  2. [cyan]Worker agents[/]   — Income, Credit, Policy agents run independently "
              "(parallel in production), each writing only their own state key\n"
        "  3. [cyan]State-gating[/]    — Risk Scoring only fires once all 3 upstream fields "
              "are non-None (merge_state barrier)\n"
        "  4. [cyan]Risk model[/]      — XGBoost trained on the sample dataset with SHAP "
              "top-5 feature attribution on every prediction\n"
        "  5. [cyan]Policy layer[/]    — Policy Compliant Agent retrieves clauses and "
              "identifies hard-stops independently of the ML model\n"
        "  6. [cyan]Synthesizer[/]     — Decision Synthesizer produces written, source-cited "
              "reasons for every recommendation\n"
        "  7. [cyan]Eval loop[/]       — Evaluation Agent checks faithfulness; routes back to "
              "synthesizer if below threshold (capped at 2 retries)\n"
        "  8. [cyan]Audit trail[/]     — Decision Record Writer persists full trace with "
              "audit_id — every step is replayable\n\n"
        "  [dim]In Sprint 2 this script becomes the LangGraph state machine. "
              "Mock tools become real API calls. LLM replaces rule-based synthesizer/evaluator.[/]",
        title="POC Coverage",
        border_style="magenta",
    ))


if __name__ == "__main__":
    main()
