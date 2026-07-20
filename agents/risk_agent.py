"""
================================================================================
   HALCYON CREDIT — Risk Scoring Agent
   Stage 3 | Author: Harshit
   LangGraph node: score_risk
   Reads:  state["income_verified"], state["credit_report"], state["policy_findings"]
   Writes: state["risk_score"]
   Model:  lgbm_halcyon_v2_lc.txt (1.3M LendingClub rows, 41 features)
   SHAP:   Top-5 feature attributions for decision explainability
================================================================================
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import lightgbm as lgb

from state.application_state import (
    ApplicationState, RiskResult, SHAPFeature, AgentError, log_event
)
from tools.credit_score_bridge import build_feature_vector, FEATURE_COLS

AGENT       = "RiskScoringAgent"
MODEL_PATH  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "models", "lgbm_halcyon_v2_lc.txt")
MODEL_VER   = "lgbm_halcyon_v2_lc"
THRESHOLD   = 0.2687   # optimal threshold from training evaluation


# Load model once at import time (not per-request)
print(f"  [{AGENT}] Loading LightGBM model from {MODEL_PATH}...")
try:
    _MODEL = lgb.Booster(model_file=MODEL_PATH)
    # Validate model loaded correctly by checking feature count
    n_feat = len(_MODEL.feature_name())
    if n_feat != 41:
        print(f"  [{AGENT}] WARNING: Expected 41 features, got {n_feat}. Model may be corrupted.")
        _MODEL = None
    else:
        print(f"  [{AGENT}] Model loaded. Features: {n_feat}")
except Exception as e:
    print(f"  [{AGENT}] WARNING: Could not load model ({type(e).__name__}). "
          f"Run models/train_lc_v2.py to regenerate it.")
    _MODEL = None


def _get_top5_shap(shap_values: np.ndarray, feature_names: list[str],
                   feature_values: list) -> list[SHAPFeature]:
    """Extract top-5 most impactful SHAP features."""
    abs_shap = np.abs(shap_values)
    top5_idx = np.argsort(abs_shap)[::-1][:5]

    features = []
    for i in top5_idx:
        sv = float(shap_values[i])
        features.append(SHAPFeature(
            feature    = feature_names[i],
            value      = round(float(feature_values[i]), 4),
            shap_value = round(sv, 4),
            direction  = "increases_risk" if sv > 0 else "decreases_risk",
        ))
    return features


def score_risk_node(state: ApplicationState) -> dict:
    """
    LangGraph node: score_risk
    Guard: Only executes when all three upstream fields are non-None.
    Builds the 41-feature vector from state, scores it, and computes SHAP.
    Returns ONLY the keys this agent owns.
    """
    t0  = time.time()
    af  = state["applicant_file"]
    inc = state.get("income_verified")
    cr  = state.get("credit_report")
    pf  = state.get("policy_findings")

    print(f"  [{AGENT}] Scoring risk for {af.applicant_id}...")

    # Guard — all upstream agents must have succeeded
    if inc is None or cr is None or pf is None:
        missing = [k for k, v in {"income": inc, "credit": cr, "policy": pf}.items() if v is None]
        error = AgentError(agent=AGENT, error_type="validation_error",
                           message=f"Upstream agents failed: {missing}")
        trace = log_event(state, AGENT, f"skipped — missing: {missing}")
        print(f"  [{AGENT}] Skipped — missing upstream data: {missing}")
        return {"risk_score": None, "errors": state["errors"] + [error], "trace": trace}

    # If model not loaded, return a rule-based score
    if _MODEL is None:
        error = AgentError(agent=AGENT, error_type="model_load_error",
                           message="LightGBM model not loaded")
        trace = log_event(state, AGENT, "model_not_loaded")
        print(f"  [{AGENT}] ERROR: Model not loaded")
        return {"risk_score": None, "errors": state["errors"] + [error], "trace": trace}

    try:
        # Build 41-feature vector via the credit score bridge
        X = build_feature_vector(
            annual_income         = af.annual_income,
            loan_amount           = af.loan_amount,
            loan_purpose          = af.loan_purpose,
            loan_term_months      = af.loan_term_months,
            employment_type       = af.employment_type,
            months_employed       = af.months_employed,
            existing_debts        = af.existing_debts,
            verification_status   = af.verification_status,
            credit_score          = cr.credit_score,
            delinquencies_2yr     = cr.delinquencies,
            open_accounts         = cr.open_accounts,
            revolving_utilisation = cr.utilization_pct,
            credit_age_months     = cr.credit_age_months,
            public_records        = af.public_records,
            inquiries_6mo         = af.inquiries_6mo,
            home_ownership        = af.home_ownership,
            verified_income       = inc.verified_income,
            income_confidence     = inc.confidence,
        )

        # Ensure column order matches training
        X = X[FEATURE_COLS]

        # Predict
        risk_score = float(_MODEL.predict(X, num_iteration=_MODEL.best_iteration)[0])

        # Risk band
        if risk_score >= THRESHOLD:
            risk_band = "High"
        elif risk_score >= 0.25:
            risk_band = "Medium"
        else:
            risk_band = "Low"

        # SHAP attributions
        try:
            import shap
            explainer  = shap.TreeExplainer(_MODEL)
            shap_vals  = explainer.shap_values(X)[0]
            top5       = _get_top5_shap(shap_vals, FEATURE_COLS, X.iloc[0].tolist())
        except Exception as shap_err:
            print(f"  [{AGENT}] SHAP failed ({shap_err}), using feature importance fallback")
            imp    = _MODEL.feature_importance(importance_type="gain")
            top5   = [
                SHAPFeature(feature=FEATURE_COLS[i], value=float(X.iloc[0, i]),
                            shap_value=float(imp[i]) / 1e6,
                            direction="increases_risk" if imp[i] > 0 else "decreases_risk")
                for i in np.argsort(imp)[::-1][:5]
            ]

        result = RiskResult(
            risk_score    = round(risk_score, 4),
            risk_band     = risk_band,
            top_features  = top5,
            model_version = MODEL_VER,
        )

        latency = (time.time() - t0) * 1000
        trace   = log_event(state, AGENT, "risk_scored", latency_ms=round(latency, 1))

        print(f"  [{AGENT}] Score={risk_score:.4f} Band={risk_band} "
              f"Top feature: {top5[0].feature if top5 else 'N/A'}")
        return {"risk_score": result, "trace": trace}

    except Exception as e:
        error = AgentError(agent=AGENT, error_type="model_error", message=str(e))
        trace = log_event(state, AGENT, f"ERROR: {e}")
        print(f"  [{AGENT}] ERROR: {e}")
        return {"risk_score": None, "errors": state["errors"] + [error], "trace": trace}
