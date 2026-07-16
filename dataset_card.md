# Dataset Card
## Halcyon Credit — Agentic Underwriting Copilot

**Team Jamun** · Harshit · Ayush · Aditya · Himkar  
**Program:** Futurense × IIT Gandhinagar · PG Diploma in AI-ML & Agentic AI Engineering · Cohort 1  
**Capstone Project 02** · Version 2.0 · June 2025

---

## 1. The Decision — Why We Pivoted to the American Express Dataset

### 1.1 The Selection Problem & Pivot

Our initial strategy utilized a unified dataset combining LendingClub and Home Credit data. While that provided a good foundation, our mathematical and empirical feasibility analysis revealed a critical limitation: **the theoretical maximum PR-AUC for standard, static loan application datasets capped around 0.75**. Because our goal is to build a highly precise risk engine capable of targeting PR-AUC > 0.90, a flat cross-sectional dataset was insufficient.

To accurately predict defaults with precision, we needed **time-series behavioral data**. We pivoted to the **American Express Default Prediction** dataset.

**Why this dataset matters:**
Instead of a single snapshot at application time, the AmEx dataset provides up to 13 months of financial behavior leading up to the default event. This allows us to capture the "default spiral"—escalating delinquencies, collapsing spend, and rising balance stress—which are the strongest predictors of credit risk.

---

### 1.2 American Express Dataset (Kaggle 2022)

**Source:** Kaggle (`amex-default-prediction`) — released by American Express  
**Size:** 5.5 million raw rows · 190 raw features · 15.27 GB raw CSV  
**Scope:** 13-month time-series profile for 458,913 unique credit card customers  
**Target:** Binary default outcome (120 days past due) observed 18 months after the latest credit card statement  
**Feature Anonymization:** Features are normalized and anonymized into five categories:
- `D_*`: Delinquency variables
- `S_*`: Spend variables
- `P_*`: Payment variables
- `B_*`: Balance variables
- `R_*`: Risk variables

---

## 2. Dataset Engineering & Aggregation Strategy

Because the raw data is a time-series (multiple rows per customer), we built a heavy-duty aggregation pipeline to flatten the data for our LightGBM model.

### 2.1 Customer Sampling Strategy
To optimize for training speed without losing minority class information:
- We kept **all 118,828 default customers**.
- We sampled **180,000 non-default customers**.
- **Final Training Size:** 298,828 customers (39.76% default rate). This near-balanced ratio removes the need for SMOTE or extreme class weighting.

### 2.2 Temporal Feature Aggregation
For each of the 188 raw features, we computed up to **6 temporal statistics** per customer over their 13-month history:
1. `mean` (average baseline)
2. `std` (volatility)
3. `min` (floor behavior)
4. `max` (ceiling behavior)
5. `last` (most recent state - the most critical signal)
6. `trend` (`last - first` - the direction of movement)

This expanded the feature space to over 1,128 baseline temporal features.

---

## 3. Cross-Feature Engineering (Domain Signals)

To push the PR-AUC past 0.90, we engineered 8 composite domain features that capture the multidimensional default spiral:

1. **Delinquency Escalation (`delinquency_escalation`)**: The average trend across 96 `D_` features.
2. **Spend Collapse (`spend_collapse`)**: The inverted trend across 21 `S_` features.
3. **Balance Stress (`balance_stress`)**: The average trend across 40 `B_` features.
4. **Risk Composite (`risk_composite_last`)**: The average of the most recent `R_` features.
5. **Payment-to-Balance Ratio (`payment_to_balance`)**: `P_2_last / B_2_last`.
6. **Delinquency Volatility (`delinquency_volatility`)**: The average standard deviation across 96 `D_` features.
7. **Composite Stress Score**: An unweighted average of the delinquency, spend, balance, and risk composites.

**Final Feature Count:** 1,136 features.

---

## 4. The Target Label

- **`0` (Good Loan / Non-Default):** The customer did not default within the 18-month observation window.
- **`1` (Default):** The customer reached 120 days past due within the 18-month observation window.

---

## 5. Final Model Performance (LightGBM)

By leveraging the rich temporal features and our engineered composites, the LightGBM model achieved the following metrics on the hold-out test set (59,766 customers):

- **PR-AUC:** 0.9378 (Target >0.90 ✅)
- **ROC-AUC:** 0.9610
- **Accuracy:** 89.47%
- **Default Recall:** 91.69%

*See `eda_report.md` artifact for detailed calibration curves, PR curves, and SHAP feature importance plots.*

---

*This dataset card is part of the Futurense AI Clinic Capstone Program academic portfolio.*
