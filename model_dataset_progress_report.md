# Halcyon Credit — ML Model & Dataset Progress Report

**Team Jamun** · Harshit · Ayush · Aditya · Himkar  
**Program:** Futurense × IIT Gandhinagar · PG Diploma in AI-ML & Agentic AI Engineering · Cohort 1  
**Capstone Project 02** · Version 1.0 · July 2025

---

## 1. Executive Summary — The Wall We've Hit

We have successfully built a high-performance LightGBM risk model (PR-AUC: 0.9378, ROC-AUC: 0.9610) trained on the American Express Default Prediction dataset. However, **this model cannot be used in our live agentic underwriting pipeline**. The reason is a fundamental schema mismatch:

- **The model needs:** 1,136 time-series behavioral features derived from 13 months of credit card statement history (e.g., `P_2_mean`, `D_39_trend`, `balance_stress`)
- **Our pipeline receives:** ~15 static fields from a loan application form (e.g., `credit_score`, `annual_income`, `loan_amount`, `employment_type`)

There is no honest way to bridge this gap. The model literally does not understand any of the fields a loan applicant submits. Before we built the AmEx model, we attempted a LendingClub + Home Credit unified dataset, which *did* have the right feature schema — but it failed to achieve the target PR-AUC threshold. Both paths are blocked. We need to resolve this before Stage 3 can be fully functional.

This document explains both failures in detail, documents all metrics, and specifies exactly what an ideal dataset must look like to unblock the team.

---

## 2. Timeline of Dataset & Model Attempts

```
Attempt 1 (Stage 1)                   Attempt 2 (Stage 2)
┌─────────────────────────┐           ┌─────────────────────────┐
│ LendingClub + HomeCredit│           │ American Express (AmEx) │
│ Unified Dataset         │           │ Default Prediction      │
│ ~657K rows, 21 features │           │ 298K customers, 1,136   │
│                         │           │ features                │
│ Result: PR-AUC < 0.75   │           │ Result: PR-AUC = 0.9378 │
│ Status: ❌ FAILED       │           │ Status: ⚠️ INCOMPATIBLE │
│ (below minimum target)  │           │ (wrong feature schema)  │
└─────────────────────────┘           └─────────────────────────┘
```

---

## 3. Attempt 1 — LendingClub + Home Credit Unified Dataset

### 3.1 What We Built

We merged two public lending datasets into a single training corpus:

- **LendingClub** (~350K sampled rows) — US peer-to-peer personal loans, mature credit profiles
- **Home Credit** (~307K rows) — Global consumer lending, includes thin-file / unbanked applicants

The unification was handled by `data/build_unified_dataset.py`, which mapped both datasets onto a common 21-feature schema.

### 3.2 Unified Feature Schema (21 features)

These are the features that **match what a loan applicant would submit**:

| # | Feature | Type | Source (LC) | Source (HC) |
|---|---------|------|-------------|-------------|
| 1 | `loan_amount` | float | `loan_amnt` | `AMT_CREDIT` |
| 2 | `annual_income` | float | `annual_inc` | `AMT_INCOME_TOTAL` |
| 3 | `employment_months` | int | `emp_length` (parsed) | `DAYS_EMPLOYED / 30` |
| 4 | `credit_score` | int | Grade → score mapping (A=750...G=450) | `EXT_SOURCE_2` normalised to 300–850 |
| 5 | `delinquencies_2yr` | int | `delinq_2yrs` | Bureau table aggregate |
| 6 | `credit_age_months` | int | `earliest_cr_line` diff | `-DAYS_CREDIT / 30` |
| 7 | `open_accounts` | int | `open_acc` | Active bureau accounts |
| 8 | `debt_to_income` | float | `dti` | `(AMT_ANNUITY × 12) / AMT_INCOME_TOTAL × 100` |
| 9 | `revolving_utilisation` | float | `revol_util` | Imputed at 50.0 (no HC equivalent) |
| 10 | `income_verified` | str | `verification_status` | Imputed as `Not Verified` |
| 11 | `employment_type` | str | Imputed as `unknown` | `NAME_INCOME_TYPE` |
| 12 | `interest_rate` | float | `int_rate` | Estimated from annuity/credit ratio |
| 13 | `monthly_installment` | float | `installment` | `AMT_ANNUITY` |
| 14 | `loan_grade_encoded` | int | Grade A–G → 1–7 | Derived from credit score bands |
| 15 | `total_credit_lines` | int | `total_acc` | Total bureau records |
| 16 | `income_confidence` | float | Verification status → 0.4–0.9 | Default 0.4 |
| 17 | `verified_income` | float | `annual_income × income_confidence` | Derived |
| 18 | `loan_to_income_ratio` | float | Derived | Derived |
| 19 | `debt_burden_ratio` | float | Derived | Derived |
| 20 | `thin_file` | int | `credit_age < 24 OR open_accounts < 3` | Same logic |
| 21 | `installment_to_income` | float | Derived | Derived |

**Label:** Binary default (`0` = repaid, `1` = default)

### 3.3 Why It Failed

The unified dataset is **cross-sectional / static** — each row is a snapshot at the time of loan application. The fundamental mathematical limitation:

> **Static loan application features have a theoretical PR-AUC ceiling of approximately 0.75 for default prediction.**

This is because:

1. **Information asymmetry at application time**: At the moment someone applies for a loan, only ~30% of the signal that determines whether they will default has actually materialised. The remaining ~70% depends on *future events* (job loss, medical emergency, interest rate changes, behavioural drift).

2. **Feature overlap between default and non-default populations**: In the LC+HC dataset, the distributions of `credit_score`, `dti`, and `income` overlap heavily between defaulters and non-defaulters. There is no clean separating hyperplane in static feature space.

3. **Imputation noise**: Home Credit lacked direct equivalents for `revolving_utilisation` (imputed at 50.0), `verification_status` (imputed as "Not Verified"), and `employment_type` (from a different taxonomy). These imputed values added noise that degraded model performance.

4. **Label inconsistency**: LendingClub defines default as "Charged Off" on completed loans. Home Credit defines it as a binary flag on loan applications. The populations and observation windows are structurally different.

### 3.4 Estimated Performance (Attempt 1)

While we did not persist a formal evaluation report from the unified dataset training, the dataset card documents the limitation:

> *"Our mathematical and empirical feasibility analysis revealed a critical limitation: the theoretical maximum PR-AUC for standard, static loan application datasets capped around 0.75."*

The MVP script (`halcyon_mvp.py`) trains an XGBoost model on a synthetic 500-row dataset mimicking the unified schema and uses 15 of these features. This establishes that the schema *works mechanically* with the pipeline, but the real-world unified dataset could not break the 0.75 barrier.

---

## 4. Attempt 2 — American Express Default Prediction Dataset

### 4.1 What We Built

After the unified dataset hit its ceiling, we pivoted to the **AmEx Default Prediction** dataset (Kaggle 2022) to capture temporal behavioral signals.

- **Source**: Kaggle `amex-default-prediction`
- **Raw Data**: 5.5M rows × 190 features × 15.27 GB
- **Customers**: 458,913 unique credit card customers
- **Observation Window**: Up to 13 monthly statements per customer
- **Target**: Binary default (120 days past due) within 18 months of last statement

### 4.2 Aggregation Pipeline

Built by `data/build_amex_dataset.py`:
- Streamed 5.5M rows in 100K chunks (memory-safe, ~3-4GB peak RAM)
- Sampled: **all 118,828 defaulters** + **180,000 non-defaulters** = **298,828 customers**
- For each of 188 raw features, computed **6 temporal statistics**: `mean`, `std`, `min`, `max`, `last`, `trend`
- Result: **1,128 base temporal features** + 1 meta feature (`n_statements`)

### 4.3 The 7 Engineered Composite Features

These domain features were designed to capture the "default spiral":

| Feature | Computation | Captures |
|---------|-------------|----------|
| `delinquency_escalation` | Mean trend of all 96 `D_*` features | Are missed payments worsening? |
| `spend_collapse` | Negative mean trend of all 21 `S_*` features | Is spending suddenly dropping? |
| `balance_stress` | Mean trend of all 40 `B_*` features | Is balance rising month-over-month? |
| `risk_composite_last` | Mean of all latest `R_*` values | Current risk indicator level |
| `payment_to_balance` | `P_2_last / B_2_last` (clipped ±10) | Payment discipline vs. balance |
| `delinquency_volatility` | Mean std of all 96 `D_*` features | How erratic is their delinquency pattern? |
| `composite_stress_score` | Mean of above 4 stress composites | Combined financial distress signal |

**Final Feature Count: 1,136** features per customer.

### 4.4 Model Architecture — LightGBM

Trained by `models/train_risk_model.py`:

```
Objective        : binary (log loss)
Class balance    : scale_pos_weight = n_neg / n_pos
Num Leaves       : 127
Max Depth        : unlimited
Learning Rate    : 0.02
Estimators       : 2,000 (early stopping at 50)
Feature Fraction : 0.20 (regularization for 1,136 features)
Bagging Fraction : 0.85
Regularization   : reg_alpha=0.1, reg_lambda=0.1
Train/Test Split : 80/20, stratified
```

### 4.5 Model Performance — Full Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **PR-AUC** | **0.9378** | > 0.90 | ✅ Exceeded |
| **ROC-AUC** | **0.9610** | > 0.90 | ✅ Exceeded |
| **Accuracy** | **89.47%** | — | ✅ Strong |
| **F1 (Default class)** | **0.8738** | — | ✅ Strong |
| **Default Recall** | **91.69%** | > 85% | ✅ Exceeded |
| **Optimal Threshold** | **0.5021** | — | Calibrated via PR curve |

**Risk Bands (defined in feature schema):**

| Band | Score Range |
|------|-------------|
| Low Risk | score < 0.25 |
| Medium Risk | 0.25 ≤ score < 0.50 |
| High Risk | score ≥ 0.50 |

**Model Artifacts Saved:**
- `models/lgbm_halcyon_v1.txt` — Trained LightGBM model (13.5 MB)
- `models/feature_schema_v1.json` — Full 1,136 feature names + metadata
- `models/xgb_halcyon_v1.json` — XGBoost variant (5.7 MB)

### 4.6 Why It Cannot Serve the Live Pipeline

The AmEx model is **excellent at what it does** — predicting default from 13 months of behavioral history. But our agentic pipeline receives a **one-time loan application**, not 13 months of credit card statements. Here is the exact mismatch:

| What the Pipeline Sends | What the Model Expects |
|--------------------------|------------------------|
| `credit_score` (int, 300–850) | `D_39_mean`, `D_39_std`, `D_39_last`, `D_39_trend` ... |
| `annual_income` (float) | `P_2_mean`, `P_2_last`, `S_3_trend` ... |
| `loan_amount` (float) | `B_1_mean`, `B_2_last`, `balance_stress` ... |
| `delinquencies` (int, 0–5) | `delinquency_escalation`, `delinquency_volatility` ... |
| `employment_type` (str) | *(no equivalent — all features are numerical)* |
| `debt_to_income` (float) | *(no equivalent)* |

**There is no feature mapping possible.** These are fundamentally different data domains:
- The pipeline data describes a **loan application event** (point-in-time snapshot)
- The AmEx data describes **behavioral trajectories** (13-month time series)

Zero-padding 1,125 features with 0s would produce garbage predictions because the model learned feature interactions between temporal statistics, not static inputs.

---

## 5. The MVP Script — Proof the Pipeline Works

The `halcyon_mvp.py` script demonstrates that the pipeline architecture works end-to-end:
- Generates a 500-row synthetic dataset matching the unified schema
- Trains an XGBoost on **15 applicant-level features**
- Runs the full agent pipeline: Income → Credit → Policy → Risk → Synthesizer → Evaluator → Record Writer

**Key Features Used in MVP (the 15 features that map to a loan application):**

```
verified_income, income_confidence, loan_to_income_ratio,
credit_score, delinquencies_2yr, revolving_utilisation,
credit_age_months, open_accounts, inquiries_6mo,
public_records, debt_to_income, debt_burden_ratio,
employment_months, thin_file, loan_amount
```

This proves the architecture is sound. The missing piece is a **real-world model** trained on these exact features with enough data volume to achieve strong performance.

---

## 6. What We Need — Ideal Dataset Specification

### 6.1 Core Requirements

For the Halcyon Credit Agentic Underwriting Copilot, the ideal training dataset must satisfy ALL of the following:

| Requirement | Why |
|---|---|
| **Per-application record structure** | Each row = one loan application → one outcome. This matches the pipeline's input (a loan officer submitting one applicant's data). |
| **Standard credit/financial features** | Features that exist on a real loan application form: income, credit score, DTI, employment, delinquencies, etc. |
| **Binary default/repay label** | Clear default outcome observed over a fixed window (e.g., 2–3 years) |
| **Size: ≥100K rows** | Statistical power for gradient-boosted models to learn complex interactions |
| **Default rate: 5–25%** | Too low (<3%) creates severe class imbalance that degrades recall; too high (>30%) suggests sampling bias |
| **Thick-file AND thin-file applicants** | Halcyon specifically serves thin-file / unbanked borrowers — the dataset must include them |
| **No feature anonymization** | We need interpretable feature names for SHAP explanations (unlike AmEx's `P_2`, `D_39`) |
| **Freely available / open license** | Academic capstone project — no licensing barriers |

### 6.2 Ideal Feature Schema (18–25 features)

The dataset MUST contain most of these features (directly or derivable):

#### Tier 1 — Must Have (core pipeline features)

| Feature | Type | Maps to TRD Field | Used By Agent |
|---------|------|--------------------|---------------|
| `annual_income` or `gross_income` | float | `applicant_file.annual_income` | Income Agent |
| `loan_amount` or `credit_amount` | float | `applicant_file.loan_amount` | Risk Agent |
| `interest_rate` | float | — | Risk Agent |
| `credit_score` or `fico_score` or external score | int/float | `credit_report.credit_score` | Credit Agent, Risk Agent |
| `debt_to_income` or DTI | float | Derived or direct | Policy Agent (POL-001) |
| `delinquencies` or `delinq_2yrs` | int | `credit_report.delinquencies` | Policy Agent (POL-006) |
| `employment_length` or employment duration | int | `applicant_file.months_employed` | Income Agent |
| `loan_purpose` | categorical | `applicant_file.loan_purpose` | Policy Agent (POL-004) |
| `default` / `charged_off` / `target` | binary | Ground truth label | Model training |

#### Tier 2 — Should Have (policy triggers + explainability)

| Feature | Type | Maps to TRD Field | Used By Agent |
|---------|------|--------------------|---------------|
| `revolving_utilisation` | float (0–100%) | `credit_report.utilization_pct` | Credit Agent, Risk Agent |
| `open_accounts` or `num_open_accounts` | int | `credit_report.open_accounts` | Credit Agent |
| `credit_age` or `earliest_credit_line` | int/date | `credit_report.credit_age_months` | Policy Agent (POL-005 thin-file) |
| `public_records` or `pub_rec` | int | Hard-stop check | Policy Agent (POL-002) |
| `employment_type` or `income_type` | categorical | `applicant_file.employment_type` | Fairness testing |
| `existing_debts` or `total_monthly_debt` | float | `applicant_file.existing_debts` | Risk Agent |
| `installment` or `monthly_payment` | float | — | Risk Agent |
| `total_credit_lines` | int | — | Credit Agent |

#### Tier 3 — Nice to Have (fairness + enrichment)

| Feature | Type | Purpose |
|---------|------|---------|
| `home_ownership` | categorical | Risk signal |
| `verification_status` | categorical | Income confidence |
| `loan_grade` or equivalent | ordinal | Risk band calibration |
| `inquiries_last_6_months` | int | Credit-seeking behavior |
| `annual_income_joint` (if joint application) | float | Dual-income applicants |

### 6.3 Must NOT Have (deal-breakers)

| Anti-Pattern | Why |
|---|---|
| ❌ Anonymized feature names (`P_2`, `D_39`, `S_3`) | Cannot produce interpretable SHAP explanations |
| ❌ Time-series / multi-row per applicant | Pipeline sends one row per application |
| ❌ Credit card statement data only | Doesn't match personal loan underwriting |
| ❌ Image/document data | Pipeline uses structured numerical features |
| ❌ Missing default labels | Cannot train supervised model |

### 6.4 Expected Model Performance on Ideal Dataset

Based on published benchmarks and academic literature for static personal loan datasets with the feature richness specified above:

| Metric | Realistic Target | Stretch Target |
|--------|------------------|----------------|
| **ROC-AUC** | 0.78–0.85 | 0.85–0.90 |
| **PR-AUC** | 0.60–0.75 | 0.75–0.82 |
| **Default Recall** | 75–85% | 85–90% |
| **Accuracy** | 75–82% | 82–88% |

> **Important**: PR-AUC > 0.90 is **not realistically achievable** with static application-time features. The AmEx model achieved 0.9378 specifically because it had 13 months of behavioral data. A static dataset target of **ROC-AUC ≥ 0.80 and PR-AUC ≥ 0.65** is both honest and competitive.

---

## 7. Candidate Datasets for Research

The following publicly available datasets should be evaluated against the requirements in Section 6:

### 7.1 LendingClub (Revisited — Solo, Not Merged)

| Attribute | Details |
|---|---|
| **Source** | Kaggle: `lending-club` or `lending-club-loan-data` |
| **Size** | ~2.26M rows (all years combined) |
| **Features Available** | `loan_amnt`, `annual_inc`, `int_rate`, `grade`, `dti`, `delinq_2yrs`, `revol_util`, `emp_length`, `open_acc`, `pub_rec`, `earliest_cr_line`, `installment`, `purpose`, `total_acc`, `verification_status`, `home_ownership` |
| **Label** | `loan_status` → Fully Paid / Charged Off / Default |
| **Thin-File Coverage** | ❌ Weak (LC borrowers are mostly thick-file) |
| **Feature Anonymization** | ✅ None — all features are interpretable |
| **Schema Match** | ✅ Excellent — almost 1:1 with TRD feature vector |
| **Known PR-AUC** | ~0.55–0.70 (published benchmarks) |
| **Risk** | No thin-file segment; class imbalance (~21% default rate with careful filtering) |

### 7.2 Home Credit Default Risk (Revisited — Solo with Bureau Tables)

| Attribute | Details |
|---|---|
| **Source** | Kaggle: `home-credit-default-risk` |
| **Size** | ~307K applications + 1.7M bureau records + 13.6M installment payment records |
| **Features Available** | `AMT_CREDIT`, `AMT_INCOME_TOTAL`, `AMT_ANNUITY`, `DAYS_EMPLOYED`, `EXT_SOURCE_1/2/3`, `NAME_INCOME_TYPE`, bureau aggregates |
| **Label** | `TARGET` (binary default) |
| **Thin-File Coverage** | ✅ Strong — many unbanked/thin-file applicants |
| **Feature Anonymization** | ❌ Partial — `EXT_SOURCE_*` are opaque external scores |
| **Schema Match** | ⚠️ Partial — requires heavy feature mapping |
| **Known PR-AUC** | ~0.65–0.80 (Kaggle leaderboard top solutions used all tables) |
| **Risk** | Low default rate (~8%); `EXT_SOURCE_*` dominates feature importance but is unexplainable |

### 7.3 Give Me Some Credit (Kaggle)

| Attribute | Details |
|---|---|
| **Source** | Kaggle: `GiveMeSomeCredit` |
| **Size** | 150K rows |
| **Features Available** | `RevolvingUtilizationOfUnsecuredLines`, `age`, `NumberOfTime30-59DaysPastDueNotWorse`, `DebtRatio`, `MonthlyIncome`, `NumberOfOpenCreditLinesAndLoans`, `NumberOfTimes90DaysLate`, `NumberRealEstateLoansOrLines`, `NumberOfTime60-89DaysPastDueNotWorse`, `NumberOfDependents` |
| **Label** | `SeriousDlqin2yrs` (binary) |
| **Thin-File Coverage** | ❌ No |
| **Feature Anonymization** | ✅ None — fully interpretable |
| **Schema Match** | ⚠️ Moderate — has delinquency detail but lacks loan amount, purpose, employment type |
| **Known PR-AUC** | ~0.55–0.65 |
| **Risk** | Missing key features (`loan_amount`, `loan_purpose`, `employment_type`) |

### 7.4 Taiwan Credit Card Default (UCI)

| Attribute | Details |
|---|---|
| **Source** | UCI ML Repository |
| **Size** | 30K rows |
| **Features Available** | `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`, `PAY_0–PAY_6` (repayment status), `BILL_AMT1–6`, `PAY_AMT1–6` |
| **Label** | `default.payment.next.month` |
| **Thin-File Coverage** | ❌ No |
| **Feature Anonymization** | ✅ None |
| **Schema Match** | ❌ Poor — credit card repayment data, not loan application data |
| **Risk** | Too small (30K); credit card context not personal loans; contains protected attributes (SEX) |

### 7.5 Recommendation Matrix

| Dataset | Schema Match | Size | Thin-File | Interpretability | Estimated PR-AUC | Overall Fit |
|---|---|---|---|---|---|---|
| **LendingClub (solo)** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | 0.55–0.70 | ⭐⭐⭐⭐ |
| **Home Credit (with bureau)** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 0.65–0.80 | ⭐⭐⭐⭐ |
| **GiveMeSomeCredit** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | 0.55–0.65 | ⭐⭐⭐ |
| **Taiwan Credit Card** | ⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐ | 0.50–0.60 | ⭐ |

---

## 8. Recommended Path Forward

### Option A — LendingClub Solo (Best Schema Match)

Train LightGBM/XGBoost exclusively on the full 2.26M-row LendingClub dataset using the features that directly map to the TRD `ApplicantFile` schema. Accept a PR-AUC of ~0.60–0.70. This model will work perfectly in the live pipeline with zero feature bridging needed.

**Pros:** Perfect schema alignment, largest volume, fully interpretable features, proven to work with our pipeline (MVP uses the same schema)
**Cons:** No thin-file coverage, PR-AUC won't hit 0.90

### Option B — Home Credit with Deep Bureau Feature Engineering

Use the full Home Credit dataset with all auxiliary tables (bureau, previous applications, installment payments, credit card balance). Engineer aggregate features similar to the AmEx approach but from the bureau and payment tables. The `EXT_SOURCE_*` features provide strong signal.

**Pros:** Best thin-file coverage, strong achievable AUC when using all tables
**Cons:** Feature interpretability issues with `EXT_SOURCE_*`, significant engineering effort, feature names need mapping

### Option C — LendingClub + Home Credit (Revised Merge Strategy)

Re-attempt the merge with improved methodology:
1. Drop imputed features that added noise (`revolving_utilisation` imputed at 50% for HC)
2. Only use features that exist natively in BOTH datasets
3. Accept the smaller common feature set but gain the thin-file segment

**Pros:** Addresses both thick-file and thin-file; uses real data for all features
**Cons:** Smallest common feature set; still capped by static-data ceiling

### Option D — Search for a New Dataset

Research and find a public consumer lending dataset that has:
- Per-application rows with standard loan features
- 100K+ rows
- Thin-file applicants included
- Non-anonymized features
- Clear default labels

---

## 9. What Stays Regardless of Decision

No matter which dataset option the team picks:

| Asset | Status | Role in Final System |
|---|---|---|
| `lgbm_halcyon_v1.txt` (AmEx model) | ✅ Stays in repo | Showcase ML artifact — demonstrates team's ability to train high-performance models on complex data |
| `feature_schema_v1.json` | ✅ Stays in repo | Documentation of AmEx feature engineering |
| `build_amex_dataset.py` | ✅ Stays in repo | Demonstrates memory-safe streaming pipeline |
| `build_unified_dataset.py` | ✅ Stays in repo | Reference for dataset merging methodology |
| `chroma_db/` (Policy KB) | ✅ Stays and is USED | Consumed by the PolicyCompliantAgent in Stage 3 |
| `halcyon_mvp.py` | ✅ Stays in repo | Demonstrates the full pipeline architecture works |

The **new model** trained on the chosen dataset will produce a separate artifact (e.g., `lgbm_halcyon_v2.txt`) specifically designed for the live pipeline.

---

## 10. Summary of All Model Metrics

### LightGBM v1 (AmEx — Current)

| Metric | Value |
|--------|-------|
| Dataset | American Express Default Prediction (Kaggle 2022) |
| Customers (train) | 239,062 |
| Customers (test) | 59,766 |
| Features | 1,136 |
| PR-AUC | **0.9378** |
| ROC-AUC | **0.9610** |
| Accuracy | 89.47% |
| F1 (Default) | 0.8738 |
| Default Recall | 91.69% |
| Optimal Threshold | 0.5021 |
| Usable in Pipeline? | **❌ No** — feature schema mismatch |

### XGBoost v1 (AmEx — Variant)

| Metric | Value |
|--------|-------|
| Artifact | `models/xgb_halcyon_v1.json` (5.7 MB) |
| Same dataset/features as LightGBM v1 |
| Usable in Pipeline? | **❌ No** — same mismatch |

### Unified Dataset Model (LendingClub + Home Credit — Attempt 1)

| Metric | Value |
|--------|-------|
| Dataset | LendingClub + Home Credit unified |
| Rows | ~657,000 |
| Features | 21 |
| PR-AUC | **< 0.75** (below target) |
| Usable in Pipeline? | **✅ Yes** — correct schema, but accuracy insufficient |

### XGBoost MVP (halcyon_mvp.py — Synthetic)

| Metric | Value |
|--------|-------|
| Dataset | 500-row synthetic (matches unified schema) |
| Features | 15 |
| Purpose | Architecture proof-of-concept only |
| Usable in Pipeline? | ✅ Architecturally yes, but not production-quality |

---

*This document is part of the Futurense AI Clinic Capstone Program academic portfolio. Halcyon Credit is a fictional persona.*
