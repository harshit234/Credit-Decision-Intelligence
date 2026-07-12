# Dataset Card
## Halcyon Credit — Agentic Underwriting Copilot

**Team Jamun** · Harshit · Ayush · Aditya · Himkar  
**Program:** Futurense × IIT Gandhinagar · PG Diploma in AI-ML & Agentic AI Engineering · Cohort 1  
**Capstone Project 02** · Version 1.1 · June 2025

---

## 1. The Decision — Which Datasets We Chose and Why

### 1.1 The Selection Problem

Finding data for an underwriting system is not the same as finding data for a generic classification problem. We needed datasets that satisfied four criteria simultaneously:

1. **Real loan outcomes** — an actual repaid/defaulted label, not a synthetic proxy
2. **Application-time features only** — no post-origination data (payment history, outstanding balance evolution) that would not be available at the moment a new applicant walks in
3. **Coverage of thin-file applicants** — our core business problem is lending to people FICO ignores
4. **Enough volume** for XGBoost/LightGBM to generalise reliably, with enough diversity for fairness testing across segments

We evaluated five candidate datasets before settling on two. The eliminated candidates were:

- **HMEQ (Home Equity)** — 5,960 records, mortgage-only, far too small for production-grade training
- **Give Me Some Credit (Kaggle)** — 150,000 records, only 11 features, no income verification, no loan purpose, no bureau detail
- **Prosper Loan Data** — 113,000 records, good features but older vintage (pre-2015), peer-to-peer mechanics differ from direct consumer lending

What remained were two datasets that together cover the full spectrum of Halcyon Credit's applicant population. **We have chosen to merge both datasets into a single Unified Training Dataset** to ensure the risk model learns from both thick-file (standard) and thin-file/international applicants.

---

### 1.2 Dataset 1 — LendingClub Loan Data (2007–2020)

**Source:** Kaggle (`wordsforthewise/lending-club`) — originally released by LendingClub Corporation  
**License:** Public domain / CC0  
**Size:** 2.9 million rows · 141 columns · ~3.5 GB raw CSV  
**Scope:** All accepted personal loan applications issued through LendingClub's marketplace from January 2007 through Q3 2020  
**Loan type:** Unsecured personal loans, $1,000–$40,000, 36 or 60 month terms  
**Geography:** United States (50 states + DC)

**Why this dataset matters:**
LendingClub provides high-volume, definitive resolved outcomes (repaid or charged off). It provides critical features like income verification status, FICO ranges, revolving credit utilization, and delinquency history. The sheer volume ensures the XGBoost model has enough data to learn standard applicant behavior.

---

### 1.3 Dataset 2 — Home Credit Default Risk (2018)

**Source:** Kaggle competition (`home-credit-default-risk`), released by Home Credit Group  
**License:** CC BY 4.0  
**Size:** 307,511 rows (main application table) · 122 columns in main table · 7 related tables (bureau, bureau_balance, etc.)  
**Scope:** Loan applications submitted to Home Credit Group across Eastern Europe and Asia  
**Loan type:** Consumer loans, cash loans, revolving credit  
**Geography:** International (primarily Czech Republic, Russia, Kazakhstan, Philippines)

**Why this dataset matters:**
Home Credit was designed specifically to serve the unbanked and underbanked — people with little or no formal credit history. This directly addresses our thin-file evaluation requirements. It also provides `NAME_INCOME_TYPE` (distinguishing salaried, self-employed, pensioners), which is vital for fairness testing across gig and informal workers, a feature absent from LendingClub entirely.

---

## 2. How We Use the Datasets Together: The Unified Schema

We have unified both datasets into a single training corpus. The mental model is a **feature harmonization and merge operation**:

```
LendingClub (Filtered to ~1.2M rows)
    │
    ├──► Parse and derive features (thin_file, emp_length, etc.)
    │
Home Credit (307K rows + aggregated bureau tables)
    │
    ├──► Aggregate bureau features at SK_ID_CURR level
    │
    ▼
[ Unified Schema Mapping & Imputation ]
    │
    ├──► Handle missing columns (e.g., impute verification status for HC)
    ├──► Engineer Derived Features (loan_to_income, debt_burden)
    │
    ▼
Unified Training Dataset (Parquet) -> Used for XGBoost/LightGBM
```

---

## 3. The Target Label

Both datasets are mapped to a binary outcome: `0` (Good Loan / Repaid) and `1` (Bad Loan / Default).
- **LendingClub:** `Fully Paid` -> 0, `Charged Off` / `Default` -> 1. (Current/Late dropped).
- **Home Credit:** `TARGET` == 0 -> 0, `TARGET` == 1 -> 1.

Class imbalance is handled natively via XGBoost's `scale_pos_weight`.

---

## 4. Feature Reference & Unified Schema Mapping

Below is the mapping used to create the unified feature space from both datasets:

| Unified Feature | LendingClub Source | Home Credit Source | Rationale |
|-----------------|--------------------|--------------------|-----------|
| `loan_amount` | `loan_amnt` | `AMT_CREDIT` | Core affordability input |
| `annual_income` | `annual_inc` | `AMT_INCOME_TOTAL` | Denominator for DTI ratios |
| `employment_months` | `emp_length` (parsed) | `DAYS_EMPLOYED` (converted) | Stability metric |
| `credit_score` | `fico_range_low` | `EXT_SOURCE_2` (scaled 300-850) | Bureau risk proxy |
| `delinquencies_2yr` | `delinq_2yrs` | Extracted from `bureau` | Past due history |
| `credit_age_months` | `earliest_cr_line` (diff) | `DAYS_CREDIT` (min) | Thin-file proxy |
| `open_accounts` | `open_acc` | `STATUS=Active` in `bureau` | Credit history breadth |
| `debt_to_income` | `dti` | Derived (`AMT_ANNUITY * 12 / INCOME`) | Affordability threshold |
| `revolving_utilisation`| `revol_util` | Extracted from balances | Strongest stress predictor |
| `income_verified` | `verification_status` | Imputed (`Not Verified`) | Determines confidence |
| `employment_type` | Imputed (`unknown`) | `NAME_INCOME_TYPE` | Fairness proxy |
| `thin_file` (Flag) | Derived (`age < 24` or `acc < 3`) | Derived (`age < 24` or `acc < 3`)| Routing & evaluation |
| `dataset_source` | Hardcoded `"lending_club"` | Hardcoded `"home_credit"` | For segmentation analysis |
| `label` | `loan_status` (0/1) | `TARGET` (0/1) | Target variable |

### 4.1 Feature Engineering for the Risk Model

The `RiskScoringAgent` uses the following derived features generated during dataset unification:
- `loan_to_income_ratio`: `loan_amount / max(annual_income, 1)`
- `income_confidence`: Encoded from `income_verified` (Source Verified=0.9, Verified=0.75, Not Verified=0.4)
- `verified_income`: `annual_income * income_confidence`
- `debt_burden_ratio`: Existing monthly debt over verified monthly income.

---

## 5. Data Quality Issues & Cleaning Decisions

- **LendingClub Pre-2013:** Dropped due to high missing values.
- **LendingClub Income Outliers:** Winsorized at the 99th percentile (~$300,000).
- **Home Credit DAYS_EMPLOYED:** Fixed the `365243` anomaly (changed to 0, representing non-employed).
- **Imputation across datasets:** LendingClub lacks `employment_type`, so it's imputed as "unknown". Home Credit lacks `verification_status`, so we conservatively impute it as "Not Verified".

---

*This dataset card is part of the Futurense AI Clinic Capstone Program academic portfolio.*
