# Dataset Card
## Halcyon Credit — Agentic Underwriting Copilot

**Team Jamun** · Harshit · Ayush · Aditya · Himkar  
**Program:** Futurense × IIT Gandhinagar · PG Diploma in AI-ML & Agentic AI Engineering · Cohort 1  
**Capstone Project 02** · Version 1.0 · June 2025

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

What remained were two datasets that together cover the full spectrum of Halcyon Credit's applicant population:

---

### 1.2 Dataset 1 — LendingClub Loan Data (2007–2020)

**Source:** Kaggle (`wordsforthewise/lending-club`) — originally released by LendingClub Corporation  
**License:** Public domain / CC0  
**Size:** 2.9 million rows · 141 columns · ~3.5 GB raw CSV  
**Scope:** All accepted personal loan applications issued through LendingClub's marketplace from January 2007 through Q3 2020  
**Loan type:** Unsecured personal loans, $1,000–$40,000, 36 or 60 month terms  
**Geography:** United States (50 states + DC)

**Why this dataset is the primary training source:**

LendingClub is the largest public repository of real, resolved consumer loan outcomes in existence. Every row is a loan that has fully matured — either repaid in full or charged off — so the label is definitive, not probabilistic. The feature set was built to support lending decisions, meaning almost everything in it was known at application time. It has income verification status (a column no other public dataset includes), FICO ranges, revolving credit utilisation, delinquency history, loan purpose, debt-to-income ratio, employment length, and state-level geography. This is exactly the data a human underwriter reads before making a decision, which means it is exactly the data our XGBoost risk model should learn from.

The dataset also spans 13 years and multiple economic cycles — the 2008–2009 financial crisis, the post-crisis recovery, the low-rate expansion of 2013–2019, and the early COVID period in 2020. A model trained across this range learns that the same applicant profile carries different risk in different macro environments, which is a more honest representation of credit risk than a single-vintage snapshot.

---

### 1.3 Dataset 2 — Home Credit Default Risk (2018)

**Source:** Kaggle competition (`home-credit-default-risk`), released by Home Credit Group  
**License:** CC BY 4.0  
**Size:** 307,511 rows (main application table) · 122 columns in main table · 7 related tables (bureau, bureau_balance, previous_application, POS_CASH_balance, installments_payments, credit_card_balance)  
**Scope:** Loan applications submitted to Home Credit Group across Eastern Europe and Asia  
**Loan type:** Consumer loans, cash loans, revolving credit  
**Geography:** International (primarily Czech Republic, Russia, Kazakhstan, Philippines)

**Why this dataset is the thin-file evaluation source:**

Home Credit was designed specifically to serve the unbanked and underbanked — people who have little or no formal credit history with any bureau. This is the segment that traditional FICO-based underwriting locks out, and it is the segment that Halcyon Credit explicitly wants to serve. The dataset's class distribution (8% defaulters, 92% repaid) is more extreme than LendingClub, which means it better represents the realistic base rate of thin-file lending: most people, even those who look risky on paper, do repay.

Home Credit also provides `NAME_INCOME_TYPE` — a categorical field distinguishing salaried workers, self-employed, pensioners, students, and commercial associates. This is the employment-type signal our pipeline needs for fairness testing across gig and informal workers, and it is absent from LendingClub entirely. The bureau and bureau_balance supplementary tables contain each applicant's complete external credit history, including accounts with zero records (the true thin-file cases). These cases are used to populate our golden evaluation set with realistic profiles of applicants that LendingClub would never have approved in the first place.

**Role distinction:** Home Credit does not contribute rows to XGBoost training. Its role is to provide thin-file applicant profiles and employment-type diversity for the evaluation harness and golden set. This distinction matters: the two datasets come from different credit markets, different currencies, and different macro environments. Mixing them as training rows would introduce distribution shift that would invalidate model calibration.

---

## 2. How We Use the Datasets Together

The mental model is a division of labour, not a merge:

```
LendingClub (2.9M rows)
    │
    ├──► Filter: keep only Fully Paid + Charged Off
    │    (drops ~40% of rows in current/late/grace period)
    │
    ├──► Clean: parse emp_length, earliest_cr_line, revol_util
    │
    ├──► Derive: thin_file flag, loan_to_income, income_confidence,
    │            employment_type (default=0 where absent)
    │
    └──► XGBoost / LightGBM training + validation (1.1–1.3M usable rows)

Home Credit (307K rows, main table + bureau join)
    │
    ├──► Aggregate bureau tables: delinquency count, credit age,
    │    max credit active, days overdue
    │
    ├──► Map to ApplicantFile schema (feature harmonisation)
    │
    └──► Golden evaluation set: thin-file cases, employment-type
         diversity, cross-segment fairness test cases
```

The feature harmonisation step is where the two datasets speak the same language. Every applicant — regardless of source — enters the Risk Scoring Agent as an `ApplicantFile` with the same typed fields. The mapping for non-identical columns is documented in Section 4.

---

## 3. The Target Label

### LendingClub

The raw `loan_status` column has seven values. We collapse it to a binary outcome:

| Raw value | Our label | Reasoning |
|-----------|-----------|-----------|
| `Fully Paid` | 0 (good loan) | Definitively resolved — repaid in full |
| `Charged Off` | 1 (bad loan) | Definitively resolved — written off as loss |
| `Default` | 1 (bad loan) | Treated as charged off for modelling |
| `Current` | **Dropped** | Outcome unknown — cannot label |
| `Late (31–120 days)` | **Dropped** | Outcome unknown — could still recover |
| `Late (16–30 days)` | **Dropped** | Outcome unknown |
| `In Grace Period` | **Dropped** | Outcome unknown |

After filtering, the usable dataset is approximately **1.1–1.3 million rows** with a class distribution of roughly **78–80% good loans (0) and 20–22% defaults (1)**. This imbalance is handled in XGBoost via `scale_pos_weight = n_negatives / n_positives ≈ 4.0`.

### Home Credit

The `TARGET` column is already binary: 0 = loan repaid, 1 = payment difficulties. Class distribution is **91.9% non-default, 8.1% default** (282,686 vs 24,825 records). This more extreme imbalance is appropriate for thin-file lending evaluation — it reflects that even among high-risk applicants, most people repay.

---

## 4. Feature Reference — Every Column We Use

### 4.1 Loan Request Features

| Feature name (pipeline) | LendingClub column | Home Credit column | What it tells us | Why it matters |
|------------------------|-------------------|-------------------|-----------------|----------------|
| `loan_amount` | `loan_amnt` | `AMT_CREDIT` | The principal amount requested | Core affordability input; higher loan relative to income signals higher default risk |
| `loan_purpose` | `purpose` | `NAME_CONTRACT_TYPE` (partial) | Why the borrower needs the money | Debt consolidation loans historically default more than home improvement; purpose affects policy compliance check |
| `loan_term_months` | `term` (parse "36 months" → 36) | Not available → default 36 | Duration of repayment obligation | Longer terms mean lower monthly payment but more exposure time |

**Discovery:** In LendingClub, `debt_consolidation` is the most common purpose (~60% of loans) and carries a default rate approximately 3–4 percentage points higher than `home_improvement`. This is counterintuitive — borrowers consolidating existing debt are already under financial stress at application time.

---

### 4.2 Income & Employment Features

| Feature name (pipeline) | LendingClub column | Home Credit column | What it tells us | Why it matters |
|------------------------|-------------------|-------------------|-----------------|----------------|
| `annual_income` | `annual_inc` | `AMT_INCOME_TOTAL` | Gross annual income (self-reported) | Denominator for all ratio features; must be verified before trusting |
| `income_verified` | `verification_status` (3-level) | Not available | Whether income was independently verified | Verified income borrowers default at lower rates than unverified; gap is ~6 percentage points in LendingClub |
| `income_confidence` | Derived from `verification_status` | Not derivable → 0.5 default | Float encoding of verification strength | Passed to Risk Scoring Agent as model feature and to Decision Synthesizer as explanation anchor |
| `employment_length_months` | `emp_length` (parse "10+ years" → 120) | `DAYS_EMPLOYED` (convert to months) | How long the borrower has held their current job | Employment stability is a strong default predictor — borrowers employed less than 1 year default at 2x the rate of 10+ year employees |
| `employment_type` | Not available → 0 (unknown) | `NAME_INCOME_TYPE` (encode) | Salaried / self-employed / gig / pensioner | Critical for fairness testing across informal workers; gig workers have irregular income that DTI ratios miss |

**Income verification encoding:**

```python
verification_map = {
    "Source Verified": 0.90,   # income source confirmed with employer/payroll
    "Verified":        0.75,   # bureau or third-party match
    "Not Verified":    0.40    # self-reported only
}
```

**Discovery:** The `annual_inc` column in LendingClub has extreme right-skew — a small number of borrowers report incomes over $1,000,000, which are almost certainly data entry errors or outliers that would distort the model. Income is capped at the 99th percentile (~$300,000) before feature construction. After capping, the median verified income for a LendingClub borrower is approximately $65,000 with a standard deviation of ~$45,000.

---

### 4.3 Credit Bureau Signal Features

| Feature name (pipeline) | LendingClub column | Home Credit column | What it tells us | Why it matters |
|------------------------|-------------------|-------------------|-----------------|----------------|
| `credit_score` | `fico_range_low` (conservative) | `EXT_SOURCE_2` (normalise to 300–850 range) | Bureau risk score | Single strongest individual predictor; below 580 is the high-risk threshold |
| `delinquencies_2yr` | `delinq_2yrs` | Aggregated from `bureau` table | Number of times 30+ days past due in last 2 years | Even a single delinquency roughly doubles default probability; two or more is a strong policy flag |
| `revolving_utilisation` | `revol_util` (parse %, cap at 100) | Approximated from `bureau_balance` aggregation | Credit card balance / total credit limit | Above 60% utilisation is a stress signal; above 90% is treated as a policy flag in our system |
| `credit_age_months` | `earliest_cr_line` (diff from issue date in months) | `DAYS_CREDIT` (convert, take min across bureau records) | Age of oldest credit account | Thin-file proxy — under 12 months triggers the thin_file flag |
| `open_accounts` | `open_acc` | Aggregated count from `bureau` (STATUS != 'C' and != 'X') | Number of currently open credit lines | Too few = thin file; too many = credit-seeking behaviour |
| `total_accounts` | `total_acc` | Total bureau records per applicant | All accounts ever opened | Combined with open_acc gives a picture of credit history breadth |
| `inquiries_6mo` | `inq_last_6mths` | `AMT_REQ_CREDIT_BUREAU_QRT` × 2 (approximate) | Hard credit pulls in last 6 months | High inquiry count signals someone shopping aggressively for credit — often a distress indicator |
| `public_records` | `pub_rec` | Proxied via `FLAG_DOCUMENT_*` presence | Bankruptcies, judgements, liens | Any public record is a hard-stop trigger in Halcyon policy |

**Discovery — the thin-file signal:**  
In LendingClub, 11.3% of approved borrowers have a credit history under 24 months, and 4.2% have fewer than 3 total accounts. These are not high-risk borrowers as a group — their default rate is within 2 percentage points of borrowers with 5-year histories, once income and DTI are controlled for. This confirms that rejecting thin-file applicants on bureau age alone is a false signal, not a genuine risk discriminator. Our model treats thin-file as a feature to learn from, not a rule to filter on.

---

### 4.4 Debt Burden Features

| Feature name (pipeline) | LendingClub column | Home Credit column | What it tells us | Why it matters |
|------------------------|-------------------|-------------------|-----------------|----------------|
| `debt_to_income` | `dti` (direct column, %) | Derived: total existing payments / income | Monthly debt obligations as a fraction of income | Most important single ratio for affordability; above 40% DTI is the primary policy threshold |
| `existing_monthly_debt` | Derived: `dti × annual_inc / 12` | `AMT_ANNUITY` (existing annuity payments) | Absolute monthly debt burden in dollars | Used alongside loan_amount to compute total payment stress |
| `mortgage_accounts` | `mort_acc` | `FLAG_OWN_REALTY` (binary) | Whether the borrower carries mortgage debt | Signals stability (owns property) but also higher existing obligations |

**Discovery — DTI distribution:**  
The median DTI in LendingClub is 17.8%. Above 35% DTI the default rate climbs steeply — from 18% at the median to 31% at DTI=40 and 38% at DTI=50. This non-linearity is exactly why XGBoost outperforms logistic regression here: tree-based models naturally capture the kink at the policy threshold.

---

### 4.5 Derived Features (Computed by the Pipeline, Not Sourced from Raw Data)

These are not columns in either dataset — they are computed inside the Risk Scoring Agent from upstream state fields before the feature vector is assembled:

| Derived feature | Formula | Rationale |
|----------------|---------|-----------|
| `loan_to_income_ratio` | `loan_amount / verified_income` | More informative than loan amount alone; normalises across income levels |
| `debt_burden_ratio` | `existing_monthly_debt / (verified_income / 12)` | Measures how much of take-home pay is already committed before this loan |
| `income_confidence` | Encoded from verification status (see Section 4.2) | Tells the synthesizer how much to trust the income figure |
| `thin_file` (flag) | `credit_age_months < 24 OR total_accounts < 3` | Triggers thin-file routing in the Orchestrator |
| `hard_stop_count` | Count of policy clauses flagged by Policy Agent | Policy-compliance input to the risk model |
| `payment_stress_index` | `(existing_monthly_debt + estimated_new_payment) / (verified_income / 12)` | Total post-loan payment burden as fraction of income |

---

### 4.6 Fairness & Segmentation Features

These columns are **never used as model inputs** — Fair Lending law prohibits the use of protected-class attributes in credit decisions. They are collected and held separately, used only during fairness evaluation runs:

| Feature | Source | Fairness use |
|---------|--------|-------------|
| `addr_state` | LendingClub | Geographic concentration and redlining checks |
| `employment_type` | Home Credit `NAME_INCOME_TYPE` | Approval rate gap testing across gig vs salaried |
| `thin_file` flag | Derived | Approval and error rate comparison for credit-invisible segment |
| `income_band` | Bucketed from `annual_income` (Low / Mid / High) | Checks that the model does not systematically disadvantage lower-income applicants |

---

## 5. Class Imbalance & How We Handle It

Both datasets have significant class imbalance, but in different directions relevant to different uses:

| Dataset | Majority class | Minority class | Ratio | Handling |
|---------|---------------|----------------|-------|---------|
| LendingClub (training) | Fully Paid: ~79% | Charged Off: ~21% | ~4:1 | `scale_pos_weight = 4.0` in XGBoost |
| Home Credit (evaluation) | Repaid: ~92% | Payment difficulties: ~8% | ~11.4:1 | Class-weighted scoring; ROC-AUC as primary metric |

We do not use SMOTE or synthetic oversampling on the training data. XGBoost's built-in class weighting handles the imbalance without introducing distribution artifacts. The evaluation metric is **ROC-AUC** (insensitive to class imbalance) alongside **precision-recall AUC** (which rewards correctly identifying the rare positive — the defaulter).

**Discovery — why accuracy is the wrong metric:**  
A model that predicts "Fully Paid" for every LendingClub application would achieve 79% accuracy while being completely useless for underwriting. We caught one team member using accuracy as a validation metric early in EDA. This is logged in the risk register under model evaluation risk. All subsequent evaluation uses AUC-ROC as the headline metric, with calibration curves checked to ensure the risk score (0–1 output) is a reliable probability estimate, not just a rank order.

---

## 6. Data Quality Issues & Cleaning Decisions

### 6.1 LendingClub — Known Issues

**Structural inconsistency before 2012:** LendingClub added new columns progressively over time. Loans issued before 2012 are missing many of the columns added after 2012, resulting in >40% null rates on some important features for the early period. **Decision: drop all loans issued before January 2013.** This leaves 2.4M rows with complete column coverage.

**The `emp_length` column is coarse:** Values are bucketed as `< 1 year`, `1 year`, `2 years`, ..., `10+ years`. The `10+` bucket is by far the largest and hides real variation between someone with 10 years and someone with 30 years experience. **Decision:** encode `10+` as 120 months and treat it as a lower bound — the model will learn that 120 months is a strong positive signal relative to shorter tenures.

**`revol_util` has entries above 100%:** These represent genuine cases of balance exceeding credit limit (due to interest and fees). **Decision:** cap at 100 before normalisation. Values above 100 are themselves a risk signal — they indicate the borrower is already in a revolving credit trap.

**`annual_inc` outliers:** The top 0.1% of income values (above ~$800,000) are implausible for unsecured personal loans of max $40,000. **Decision:** winsorise at the 99th percentile (~$300,000).

**`mths_since_last_delinq` and similar "time since" columns:** These have null values for borrowers who have never had the relevant event (never delinquent, never bankrupt). A null does not mean missing — it means "never happened." **Decision:** impute with a large sentinel value (999 months) and add a binary indicator column (`ever_delinquent`, `ever_bankrupt`) so the model sees both the absence and the duration.

### 6.2 Home Credit — Known Issues

**Multi-table join required:** The main `application_train.csv` is a single row per applicant. Credit history, payment records, and prior application data live in six supplementary tables with one-to-many relationships. Simple concatenation is wrong — one applicant may have 15 bureau records. **Decision:** aggregate each supplementary table by `SK_ID_CURR` (the applicant key) before joining, using statistics: count, mean, max, min, sum. This converts variable-length histories into fixed-width feature vectors.

**`DAYS_EMPLOYED` anomaly:** 55,374 records have `DAYS_EMPLOYED = 365243`, which is clearly a data encoding error (approximately 1,000 years of employment). These correspond to pensioners and other non-employed applicants. **Decision:** replace with 0 (zero months employed) and set `employment_type = "non-employed"`.

**`EXT_SOURCE` is not a FICO score:** The three external source scores (0–1 float) are bureau scores from institutions that did not disclose their methodology. We normalise `EXT_SOURCE_2` (the most complete, with <0.1% nulls) to a 300–850 range to make it interoperable with the LendingClub credit score field in the unified feature schema.

---

## 7. Key Discoveries from EDA

These are findings from exploratory analysis of both datasets that directly shaped architecture or model decisions:

**Discovery 1 — Income verification has stronger signal than income level.**  
In LendingClub, two borrowers with identical annual income but different verification status default at materially different rates. A borrower with $60,000 verified income defaults at roughly the same rate as one with $90,000 unverified income. This justified building the Income Verification Agent as a separate pipeline step with its own confidence score, rather than passing raw stated income directly to the risk model.

**Discovery 2 — Revolving utilisation above 60% is the clearest single-variable default predictor.**  
In a simple univariate analysis, revolving utilisation above 60% is more predictive of default than credit score in the subprime (FICO < 640) segment. This is because utilisation captures current financial stress, whereas credit score reflects historical behaviour. This finding shaped the SHAP attribution output requirement — the top-5 features reported by the Risk Scoring Agent will frequently surface utilisation as the lead explanation, which directly drives the written reasons the Decision Synthesizer produces.

**Discovery 3 — DTI has a non-linear threshold effect, not a linear relationship.**  
Default rates are roughly flat between DTI = 5% and DTI = 25%, then rise steeply above 35%. This confirmed that a tree-based model (XGBoost) is the right architecture choice over logistic regression — the kink cannot be captured by a linear coefficient. It also confirmed that Halcyon's policy threshold of DTI > 40% as a hard-stop is empirically grounded.

**Discovery 4 — Thin-file applicants are not high-risk as a class; they are high-uncertainty.**  
Controlling for income and employment length, borrowers with credit history under 24 months default at a rate only 2–3 percentage points higher than borrowers with 5+ year histories. The real risk is uncertainty, not default propensity. This is why our system routes thin-file applicants to the REFER band rather than automatic decline — a human underwriter can resolve uncertainty with a short interview or additional document check that the model cannot perform.

**Discovery 5 — Loan purpose encodes latent financial stress.**  
Debt consolidation loans are the highest-default-rate category in LendingClub (~24% charge-off rate versus ~15% for home improvement). This is not because debt consolidation is inherently risky — it is because borrowers in financial distress seek consolidation as a last resort. Our policy knowledge base includes a clause that requires additional income verification for consolidation loans above a DTI threshold, directly informed by this finding.

**Discovery 6 — The pre-2013 data in LendingClub is structurally different.**  
Before 2013, LendingClub had far fewer applicants, operated with manual underwriting, and had a substantially different approval rate. Including pre-2013 data in training creates a distribution shift — the model learns patterns from a different lending regime. Restricting to 2013–2020 produces a more homogeneous training distribution that better represents the automated, high-volume lending Halcyon Credit operates.

**Discovery 7 — Home Credit's `NAME_INCOME_TYPE` reveals a gig-worker gap.**  
In the Home Credit dataset, applicants categorised as `Working` (formal salaried) default at 7.2% while those categorised as `Commercial associate` (broadly self-employed and gig) default at 9.8% — a gap of 2.6 percentage points. However, when income is controlled for, the gap narrows to less than 1 percentage point. This means that a model trained only on LendingClub (which lacks employment type) would implicitly disadvantage gig workers by treating their income as less reliable, without having the feature to correct for it. This is the primary reason Home Credit is included in the evaluation harness: to surface and quantify this fairness gap in our system's outputs.

---

## 8. Dataset Card Summary

| Property | LendingClub | Home Credit |
|----------|-------------|-------------|
| Rows (usable) | ~1.1–1.3M (after filtering) | 307,511 (main table) |
| Columns used | 28 raw → 17 model features | ~40 raw (after aggregation) → 12 model features |
| Label | `loan_status` → binary 0/1 | `TARGET` 0/1 |
| Class balance | 79% / 21% | 92% / 8% |
| Role in system | XGBoost / LightGBM training | Thin-file golden set evaluation |
| Data vintage | 2013–2020 (post-filter) | 2016–2018 |
| Geography | United States | Eastern Europe / Asia |
| Income verification | ✅ Direct column | ❌ Not available |
| Employment type | ❌ Not available | ✅ Direct column |
| Thin-file cases | ✅ Derived flag | ✅ Core purpose of dataset |
| FICO / bureau score | ✅ `fico_range_low/high` | ✅ `EXT_SOURCE_2` (proxy) |
| Known gaps | No employment type; pre-2013 unusable | No verification status; 7-table join required |
| Primary known issue | 25%+ missing values before cleaning | `DAYS_EMPLOYED` encoding error (365243) |

---

## 9. What the Datasets Cannot Tell Us

Honest acknowledgement of limitations:

- **Neither dataset contains the policy retrieval context** the Policy Compliant Agent uses. The ChromaDB knowledge base is built from regulatory documents (CFPB guidelines, Halcyon's fictional policy doc), not from these datasets.
- **LendingClub ended retail lending in 2020.** The most recent data is now 5 years old. Macro conditions (interest rate environment, inflation, post-COVID labour market) have changed significantly. The model will need retraining with more recent data for genuine production deployment. This is documented as a known gap.
- **Neither dataset reflects Halcyon Credit's actual approval policy.** LendingClub's approval decisions are the ground truth, and they reflect LendingClub's policy, not Halcyon's. Our model learns what LendingClub approved and defaulted on — then Halcyon's policy layer (the Policy Compliant Agent) overlays Halcyon-specific rules that may differ.
- **Income confidence cannot be validated end-to-end.** We derive income confidence from verification status, but we have no dataset that tells us how often verified income was subsequently found to be falsified. This remains a model assumption documented in the risk register.

---

*This dataset card is part of the Futurense AI Clinic Capstone Program academic portfolio. LendingClub and Home Credit datasets are used for research and educational purposes under their respective open licences. Halcyon Credit is a fictional persona.*
