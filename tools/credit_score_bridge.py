"""
================================================================================
   HALCYON CREDIT — Credit Score → sub_grade Bridge
   Stage 3 | Author: Harshit
   Maps applicant-provided credit data to the 41-feature vector
   expected by lgbm_halcyon_v2_lc.txt.

   WHY THIS EXISTS:
   The LendingClub training data contains 'sub_grade' (A1-G5, 35 levels)
   and 'int_rate' which are assigned by LC's underwriters AFTER assessment.
   At loan application time, WE are the lender — we have no sub_grade yet.
   This bridge estimates sub_grade from the applicant's credit_score + DTI,
   and estimates int_rate from historical LC medians per sub_grade.
================================================================================
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# GRADE / SUB-GRADE MAPPING
# ─────────────────────────────────────────────────────────────────────────────

# Credit score → LC grade (based on LC's historical published thresholds)
_SCORE_TO_GRADE: list[tuple[int, str]] = [
    (720, "A"),   # 720+ → A
    (680, "B"),   # 680–719 → B
    (640, "C"),   # 640–679 → C
    (600, "D"),   # 600–639 → D
    (560, "E"),   # 560–599 → E
    (520, "F"),   # 520–559 → F
    (0,   "G"),   # < 520 → G
]

# Grade → encoded integer (A=1 … G=7)
_GRADE_ENCODED: dict[str, int] = {
    "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7
}

# sub_grade encoded: A1=1 … G5=35
_SUB_GRADE_ENCODED: dict[str, int] = {
    f"{g}{n}": (gi * 5 + n)
    for gi, g in enumerate("ABCDEFG")
    for n in range(1, 6)
}

# Historical LC median interest rates per sub_grade (approximate)
_SUB_GRADE_INT_RATE: dict[str, float] = {
    "A1": 6.00, "A2": 6.49, "A3": 6.99, "A4": 7.49, "A5": 7.97,
    "B1": 8.50, "B2": 9.49, "B3":10.49, "B4":11.49, "B5":12.49,
    "C1":13.32, "C2":14.09, "C3":14.85, "C4":15.61, "C5":16.36,
    "D1":17.09, "D2":17.77, "D3":18.45, "D4":19.13, "D5":19.79,
    "E1":20.49, "E2":21.18, "E3":21.86, "E4":22.55, "E5":23.22,
    "F1":23.99, "F2":24.59, "F3":25.19, "F4":25.79, "F5":26.30,
    "G1":26.77, "G2":27.31, "G3":27.88, "G4":28.49, "G5":28.99,
}

# DTI → sub_grade number within a grade (1=best/lowest DTI … 5=worst/highest)
_DTI_THRESHOLDS: list[float] = [10.0, 20.0, 30.0, 37.0]  # breakpoints for 1/2/3/4/5


def _credit_score_to_grade(credit_score: int) -> str:
    """Map FICO score to LC grade A–G."""
    for threshold, grade in _SCORE_TO_GRADE:
        if credit_score >= threshold:
            return grade
    return "G"


def _dti_to_sub_number(dti: float) -> int:
    """Map DTI (0–100) to sub-grade number 1–5 within a grade."""
    for i, threshold in enumerate(_DTI_THRESHOLDS):
        if dti < threshold:
            return i + 1
    return 5


def estimate_sub_grade(credit_score: int, dti: float) -> str:
    """
    Estimate LC sub_grade from credit_score and DTI.
    Returns e.g. 'B3', 'A1', 'G5'.
    """
    grade      = _credit_score_to_grade(credit_score)
    sub_number = _dti_to_sub_number(dti)
    return f"{grade}{sub_number}"


# ─────────────────────────────────────────────────────────────────────────────
# INSTALLMENT CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────
def _compute_installment(loan_amount: float, annual_rate_pct: float, term_months: int) -> float:
    """Compute monthly installment using standard amortisation formula."""
    if annual_rate_pct <= 0:
        return loan_amount / max(term_months, 1)
    r = (annual_rate_pct / 100) / 12   # monthly rate
    return loan_amount * r * (1 + r) ** term_months / ((1 + r) ** term_months - 1)


# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE RISK ENCODING
# ─────────────────────────────────────────────────────────────────────────────
_PURPOSE_RISK: dict[str, int] = {
    "debt_consolidation": 3,
    "credit_card":        2,
    "home_improvement":   1,
    "medical":            2,
    "car":                1,
    "small_business":     3,
    "vacation":           2,
    "moving":             2,
    "house":              1,
    "wedding":            2,
    "renewable_energy":   1,
    "educational":        2,
    "other":              2,
}

_HOME_OWN: dict[str, int] = {
    "OWN": 1, "MORTGAGE": 2, "RENT": 3, "OTHER": 3, "NONE": 3
}

_CONFIDENCE_MAP: dict[str, float] = {
    "Source Verified": 0.90,
    "Verified":        0.75,
    "Not Verified":    0.40,
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BRIDGE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_vector(
    annual_income:        float,
    loan_amount:          float,
    loan_purpose:         str,
    loan_term_months:     int,
    employment_type:      str,
    months_employed:      int,
    existing_debts:       float,
    verification_status:  str,
    credit_score:         int,
    delinquencies_2yr:    int,
    open_accounts:        int,
    revolving_utilisation: float,
    credit_age_months:    int,
    public_records:       int,
    inquiries_6mo:        int,
    home_ownership:       str,
    verified_income:      float,
    income_confidence:    float,
) -> pd.DataFrame:
    """
    Builds the full 41-feature vector expected by lgbm_halcyon_v2_lc.txt.
    All features are derived from applicant form fields + bureau data.

    Returns a single-row pd.DataFrame with columns matching FEATURE_COLS
    from models/train_lc_v2.py.
    """
    # ── Compute DTI ────────────────────────────────────────────────────────
    dti = (existing_debts * 12) / max(annual_income, 1) * 100
    dti = min(dti, 100.0)

    # ── sub_grade bridge ───────────────────────────────────────────────────
    sub_grade      = estimate_sub_grade(credit_score, dti)
    grade          = sub_grade[0]
    sub_grade_enc  = _SUB_GRADE_ENCODED.get(sub_grade, 18)
    grade_enc      = _GRADE_ENCODED.get(grade, 4)
    int_rate       = _SUB_GRADE_INT_RATE.get(sub_grade, 15.0)

    # ── Loan computations ──────────────────────────────────────────────────
    installment            = _compute_installment(loan_amount, int_rate, loan_term_months)
    loan_to_income         = loan_amount / max(annual_income, 1)
    installment_to_income  = installment / max(annual_income / 12, 1)
    revol_bal              = (revolving_utilisation / 100) * max(annual_income * 0.3, 1000)
    revol_bal_to_income    = revol_bal / max(annual_income, 1)

    # ── Balance / utilisation proxies ─────────────────────────────────────
    all_util        = revolving_utilisation
    bc_util         = revolving_utilisation
    il_util         = min(dti * 0.6, 100.0)   # installment debt fraction of credit
    total_rev_hi    = max(annual_income * 0.5, revol_bal * 2)
    tot_cur_bal     = existing_debts * 12 + revol_bal
    tot_hi_cred_lim = max(annual_income * 0.8, tot_cur_bal * 1.5)
    avg_cur_bal     = tot_cur_bal / max(open_accounts, 1)

    # ── Account depth proxies ─────────────────────────────────────────────
    total_acc        = max(open_accounts + int(credit_age_months / 24), open_accounts)
    mort_acc         = 1 if home_ownership == "MORTGAGE" else 0
    num_bc_tl        = max(1, open_accounts // 3)   # ~30% of accounts are credit cards
    num_il_tl        = max(1, open_accounts // 4)   # installment accounts
    acc_open_past_24 = max(0, open_accounts - max(0, open_accounts - int(credit_age_months / 12)))
    num_actv_rev_tl  = max(1, open_accounts // 2)

    # ── Delinquency proxies ───────────────────────────────────────────────
    pct_tl_nvr_dlq    = max(0.0, 100.0 - delinquencies_2yr * 15.0)   # rough: each delinq ≈ 15%
    num_tl_90g_dpd24  = 1 if delinquencies_2yr >= 2 else 0
    num_tl_30dpd      = delinquencies_2yr

    # ── Public record ─────────────────────────────────────────────────────
    has_public_record = 1 if public_records >= 1 else 0

    # ── Thin file flag ────────────────────────────────────────────────────
    thin_file = 1 if (credit_age_months < 24 or open_accounts < 3) else 0

    # ── Purpose + home encoding ───────────────────────────────────────────
    purpose_risk   = _PURPOSE_RISK.get(loan_purpose.lower(), 2)
    home_own_enc   = _HOME_OWN.get(home_ownership.upper(), 3)

    # ── Assemble into DataFrame ───────────────────────────────────────────
    row = {
        "sub_grade_encoded":    sub_grade_enc,
        "grade_encoded":        grade_enc,
        "dti":                  round(dti, 4),
        "revol_util":           revolving_utilisation,
        "revol_bal_to_income":  round(revol_bal_to_income, 6),
        "loan_to_income":       round(loan_to_income, 6),
        "installment_to_income":round(installment_to_income, 6),
        "delinq_2yrs":          delinquencies_2yr,
        "pct_tl_nvr_dlq":       round(pct_tl_nvr_dlq, 2),
        "num_tl_90g_dpd_24m":   num_tl_90g_dpd24,
        "num_tl_30dpd":         num_tl_30dpd,
        "num_actv_rev_tl":      num_actv_rev_tl,
        "has_public_record":    has_public_record,
        "pub_rec":              public_records,
        "pub_rec_bankruptcies": min(public_records, 1),
        "credit_age_months":    credit_age_months,
        "open_acc":             open_accounts,
        "total_acc":            total_acc,
        "mort_acc":             mort_acc,
        "num_bc_tl":            num_bc_tl,
        "num_il_tl":            num_il_tl,
        "acc_open_past_24mths": acc_open_past_24,
        "tot_cur_bal":          round(tot_cur_bal, 2),
        "tot_hi_cred_lim":      round(tot_hi_cred_lim, 2),
        "avg_cur_bal":          round(avg_cur_bal, 2),
        "bc_util":              round(bc_util, 2),
        "il_util":              round(il_util, 2),
        "all_util":             round(all_util, 2),
        "total_rev_hi_lim":     round(total_rev_hi, 2),
        "annual_inc":           annual_income,
        "loan_amnt":            loan_amount,
        "int_rate":             int_rate,
        "installment":          round(installment, 2),
        "term_months":          loan_term_months,
        "employment_months":    months_employed,
        "income_confidence":    income_confidence,
        "verified_income":      verified_income,
        "inq_last_6mths":       inquiries_6mo,
        "purpose_risk":         purpose_risk,
        "home_own_encoded":     home_own_enc,
        "thin_file":            thin_file,
    }

    return pd.DataFrame([row])


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE NAMES — must match train_lc_v2.py FEATURE_COLS exactly
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "sub_grade_encoded", "grade_encoded", "dti", "revol_util",
    "revol_bal_to_income", "loan_to_income", "installment_to_income",
    "delinq_2yrs", "pct_tl_nvr_dlq", "num_tl_90g_dpd_24m",
    "num_tl_30dpd", "num_actv_rev_tl", "has_public_record",
    "pub_rec", "pub_rec_bankruptcies", "credit_age_months",
    "open_acc", "total_acc", "mort_acc", "num_bc_tl", "num_il_tl",
    "acc_open_past_24mths", "tot_cur_bal", "tot_hi_cred_lim",
    "avg_cur_bal", "bc_util", "il_util", "all_util", "total_rev_hi_lim",
    "annual_inc", "loan_amnt", "int_rate", "installment", "term_months",
    "employment_months", "income_confidence", "verified_income",
    "inq_last_6mths", "purpose_risk", "home_own_encoded", "thin_file",
]
