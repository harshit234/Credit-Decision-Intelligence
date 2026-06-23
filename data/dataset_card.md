# Dataset Card — Home Credit Default Risk

> Status: **proposed** primary dataset (ADR-006, awaiting team ratification). This card is the Sprint-0
> exit artifact; numbers marked *(verify)* must be confirmed during EDA before they're cited anywhere.

## 1. Overview

| | |
|---|---|
| **Name** | Home Credit Default Risk |
| **Source** | Kaggle competition (Home Credit Group) |
| **Why this one** | Large, multi-table, rich in **thin-file / non-traditional** applicants — exactly Halcyon's population |
| **Scale** | ~307,511 rows × 122 columns (main `application_train`) *(verify)* |
| **Target** | `TARGET` — 1 = client with payment difficulties, 0 = otherwise |
| **Class balance** | ~8% positive (default) — **imbalanced** *(verify)* |
| **License** | Kaggle competition rules — research/educational use; review terms before redistribution |

## 2. Structure

Multi-table relational dataset (joined on `SK_ID_CURR` / `SK_ID_BUREAU` / `SK_ID_PREV`):

- `application_{train,test}` — main applicant features (income, credit amount, employment, housing, etc.)
- `bureau` + `bureau_balance` — prior credits from other institutions
- `previous_application` — prior Home Credit applications
- `POS_CASH_balance`, `installments_payments`, `credit_card_balance` — repayment behaviour

## 3. Mapping to our pipeline

| Pipeline field | Source columns (illustrative — confirm in EDA) |
|---|---|
| verified income | `AMT_INCOME_TOTAL` |
| loan amount / annuity | `AMT_CREDIT`, `AMT_ANNUITY` |
| employment tenure | `DAYS_EMPLOYED` |
| credit history / delinquencies | `bureau.*`, `bureau_balance.STATUS` |
| utilization / open accounts | `credit_card_balance.*`, `bureau.CREDIT_ACTIVE` |
| thin-file flag | absence of `bureau` records |
| ground-truth label | `TARGET` |

## 4. Fairness-relevant fields ⚠️

- `CODE_GENDER` — used **only** to measure fairness gaps; **never** a model input.
- Age proxy `DAYS_BIRTH`, region/housing fields — audit for proxy leakage of protected attributes.
- Document the proxy methodology in the Risk Register (R-06) before reporting fairness results.

## 5. Known limitations & gaps

- **Imbalance (~8%)** → use class weights / resampling and tune the decision threshold; report PR-AUC, not just AUC-ROC.
- `DAYS_EMPLOYED` has a known sentinel anomaly (e.g. 365243) → clean during EDA.
- Missing values are widespread across bureau tables → document imputation strategy.
- It is a single-lender snapshot in time → no macroeconomic drift; note as an external-validity caveat.

## 6. Intended use & synthetic augmentation

- **Use:** train/validate the commodity risk model; seed the evaluation **golden set** with real-labelled cases.
- **Augmentation (TRD §7.3):** generate synthetic applicant *narratives* and edge cases on top of the structured
  data — thin-file profiles, adversarial inconsistencies (stated vs verified income), and fraud/velocity patterns —
  each with a ground-truth `approve/decline/refer` label. Seeded for reproducibility; documented in its own card.

## 7. Provenance & reproducibility

- Raw data is **not** committed (see `.gitignore`); document the exact download steps + checksum here once fetched.
- Record the train/val/test split seed and methodology so every downstream metric is reproducible.

## 8. Alternatives considered

LendingClub, HMEQ — smaller and less representative of thin-file borrowers. See ADR-006 in `DECISIONS.md`.
