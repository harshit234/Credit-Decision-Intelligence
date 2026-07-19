# Dataset Card
## Halcyon Credit — Agentic Underwriting Copilot

**Team Jamun** · Harshit · Ayush · Aditya · Himkar  
**Program:** Futurense × IIT Gandhinagar · PG Diploma in AI-ML & Agentic AI Engineering · Cohort 1  
**Capstone Project 02** · Version 3.0 · July 2025

---

## 1. Dataset Strategy — Full History & Final Decision

### 1.1 Attempt 1 — LendingClub + Home Credit Unified Dataset (❌ Failed)

Our initial strategy unified LendingClub and Home Credit into a single 657K-row schema (`unified_training_data.csv`). While the feature schema aligned perfectly with a loan application form, **PR-AUC was capped at ~0.36** due to:
- Static, point-in-time features with limited predictive signal
- `credit_score` derived from LC's grade (A-G → 7 discrete values only), not a real FICO score
- LightGBM early stopping at just **4 iterations** — no residual signal to learn

### 1.2 Attempt 2 — American Express Default Prediction Dataset (⚠️ Incompatible)

We pivoted to the AmEx Kaggle 2022 dataset to access time-series behavioral signals. The LightGBM model achieved outstanding metrics:

| Metric | Value |
|--------|-------|
| PR-AUC | **0.9378** ✅ |
| ROC-AUC | **0.9610** ✅ |
| Default Recall | 91.69% ✅ |

However, this model **cannot serve the live pipeline**. It requires 1,136 time-series features derived from 13 months of credit card statement history (`P_2_mean`, `D_39_trend`, etc.) — features that do not exist at loan application time. The AmEx model is retained in the repository as a **ML capability showcase artifact** (`models/lgbm_halcyon_v1.txt`).

**See:** `model_dataset_progress_report.md` for the full wall analysis.

### 1.3 Final Decision — LendingClub (Raw, Direct) ✅

After identifying the root cause of Attempt 1's failure (fake grade-derived credit score, insufficient features), we re-trained directly on the **raw `loan.csv`** using:
- `sub_grade` encoded as a 35-level ordinal (A1=1 → G5=35) — far richer than 7-level grade
- 41 features that are **all collectable at loan application time**
- 1.3M completed loans (3.7× more than Attempt 1)

---

## 2. Final Dataset — LendingClub (Raw)

### 2.1 Source

**Dataset:** LendingClub Loan Data  
**Source:** Kaggle (`lending-club-loan-data`) / LendingClub public release  
**Raw Size:** 2,260,668 rows · 145 features  
**After filtering (completed loans only):** 1,303,638 rows  
**Default rate:** 20.07% (Fully Paid=0, Charged Off/Default=1)  
**Scope:** US personal loans issued 2007–2018

### 2.2 Why Only Completed Loans

Only `loan_status` values of `Fully Paid`, `Charged Off`, or `Default` are retained. Loans with status `Current`, `Late`, or `In Grace Period` have **unknown outcomes** and would introduce label noise.

---

## 3. Feature Engineering Pipeline (train_lc_v2.py)

### 3.1 Key Fix: sub_grade Instead of Grade

The critical error in Attempt 1 was mapping grade A–G to 7 fake FICO scores (750, 700, 650…). This gave LightGBM a discrete 7-value ordinal with massive information loss.

**V2 Fix:** Encode `sub_grade` (A1 through G5) as a continuous 1–35 ordinal:
```
A1=1, A2=2, A3=3, A4=4, A5=5,
B1=6 ... G5=35
```
This gives a **35-level continuous risk gradient** the model can meaningfully interpolate between.

### 3.2 41 Features Used (All Pipeline-Compatible)

| Group | Features |
|-------|----------|
| **Credit quality** | `sub_grade_encoded` (35 levels), `grade_encoded` (7 levels) |
| **Debt burden** | `dti`, `revol_util`, `revol_bal_to_income`, `loan_to_income`, `installment_to_income` |
| **Delinquency depth** | `delinq_2yrs`, `pct_tl_nvr_dlq`, `num_tl_90g_dpd_24m`, `num_tl_30dpd`, `num_actv_rev_tl` |
| **Public records** | `has_public_record`, `pub_rec`, `pub_rec_bankruptcies` |
| **Credit history** | `credit_age_months`, `open_acc`, `total_acc`, `mort_acc`, `num_bc_tl`, `num_il_tl`, `acc_open_past_24mths` |
| **Balance signals** | `tot_cur_bal`, `tot_hi_cred_lim`, `avg_cur_bal`, `bc_util`, `il_util`, `all_util`, `total_rev_hi_lim` |
| **Loan details** | `loan_amnt`, `int_rate`, `installment`, `term_months` |
| **Income & employment** | `annual_inc`, `verified_income`, `income_confidence`, `employment_months` |
| **Behavior / context** | `inq_last_6mths`, `purpose_risk`, `home_own_encoded`, `thin_file` |

### 3.3 Derived Features

| Feature | Formula | Maps To |
|---------|---------|---------|
| `credit_age_months` | `(issue_d - earliest_cr_line) / 30.44` | POL-005 thin-file check |
| `has_public_record` | `pub_rec >= 1 OR pub_rec_bankruptcies >= 1` | POL-002 hard stop |
| `thin_file` | `credit_age < 24 months OR open_acc < 3` | POL-005 thin-file flag |
| `income_confidence` | `verification_status → {0.90, 0.75, 0.40}` | Income verification weight |
| `verified_income` | `annual_inc × income_confidence` | Adjusted income signal |
| `loan_to_income` | `loan_amnt / annual_inc` | Affordability check |
| `installment_to_income` | `installment / (annual_inc / 12)` | Monthly burden ratio |
| `revol_bal_to_income` | `revol_bal / annual_inc` | Revolving debt load |
| `purpose_risk` | Purpose category → {1=low, 2=medium, 3=high} | POL-004 purpose risk |

---

## 4. Train / Test Split

| Split | Rows | Default Rate |
|-------|------|-------------|
| **Train** | 1,042,280 (80%) | 20.07% |
| **Test** | 260,570 (20%) | 20.07% |
| **Stratified** | Yes | Both splits balanced |
| **scale_pos_weight** | 3.98 | Handles class imbalance |

---

## 5. Final Model Performance — LightGBM V2 (Production Pipeline Model)

Model artifact: `models/lgbm_halcyon_v2_lc.txt`  
Schema: `models/feature_schema_v2_lc.json`  
Training script: `models/train_lc_v2.py`

| Metric | Value |
|--------|-------|
| **ROC-AUC** | **0.7166** |
| **PR-AUC** | **0.3854** |
| **Accuracy** | 66.26% |
| **F1 (Default class)** | 0.4346 |
| **F1 (Good Loan)** | 0.7595 |
| **Default Recall** | 64.62% |
| **Optimal Threshold** | 0.2687 |
| **Best Iteration** | 12 (early stopping at 50) |
| **Training time** | 42 seconds |

### Confusion Matrix (Test Set, 260,570 rows)

|  | Predicted Good | Predicted Default |
|--|---|---|
| **Actual Good** | 138,855 (TN) | 69,416 (FP) |
| **Actual Default** | 18,503 (FN) | 33,796 (TP) |

### Top 10 Features by Gain

| Rank | Feature | Gain |
|------|---------|------|
| 1 | `sub_grade_encoded` | 1,128,536 |
| 2 | `int_rate` | 339,610 |
| 3 | `grade_encoded` | 337,649 |
| 4 | `term_months` | 88,146 |
| 5 | `all_util` | 84,625 |
| 6 | `avg_cur_bal` | 69,181 |
| 7 | `loan_to_income` | 68,961 |
| 8 | `acc_open_past_24mths` | 37,174 |
| 9 | `dti` | 31,052 |
| 10 | `employment_months` | 26,157 |

### Risk Bands

| Band | Score Range | Interpretation |
|------|-------------|----------------|
| **Low Risk** | score < 0.25 | Strong approval candidate |
| **Medium Risk** | 0.25 ≤ score < 0.27 | Review required |
| **High Risk** | score ≥ 0.27 (optimal threshold) | Decline / REFER |

---

## 6. Historical Showcase Model — AmEx LightGBM V1

This model is retained to demonstrate the team's ML capability on large-scale time-series data.

| Attribute | Value |
|-----------|-------|
| **Artifact** | `models/lgbm_halcyon_v1.txt` |
| **Dataset** | AmEx Default Prediction (Kaggle 2022) |
| **PR-AUC** | **0.9378** |
| **ROC-AUC** | **0.9610** |
| **Features** | 1,136 (time-series aggregates) |
| **Training rows** | 239,062 |
| **Pipeline compatible** | ❌ No — requires 13-month behavioral history |

---

## 7. Both Models in the Repository

| Artifact | Role | Status |
|---------|------|--------|
| `models/lgbm_halcyon_v2_lc.txt` | **Production pipeline model** — used by RiskScoringAgent in Stage 3 | ✅ Active |
| `models/feature_schema_v2_lc.json` | Feature schema + metrics for V2 | ✅ Active |
| `models/train_lc_v2.py` | Training script for V2 | ✅ Active |
| `models/lgbm_halcyon_v1.txt` | AmEx showcase model | 📦 Archived |
| `models/feature_schema_v1.json` | Feature schema for AmEx model | 📦 Archived |

---

*This dataset card is part of the Futurense AI Clinic Capstone Program academic portfolio. Halcyon Credit is a fictional persona.*