"""
================================================================================
   HALCYON CREDIT -- Unified Dataset Builder v2
   Adds 4 high-signal features to the schema:
     interest_rate      : LC int_rate / HC estimated from annuity ratio
     monthly_installment: LC installment / HC AMT_ANNUITY/12
     loan_grade_encoded : LC grade A-G -> 1-7 / HC from credit_score bands
     total_credit_lines : LC total_acc / HC active_accounts + imputed median
================================================================================
"""

import os
import gc
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# LendingClub
# ─────────────────────────────────────────────────────────────────────────────
def process_lending_club(file_path):
    print("Processing LendingClub Data...")

    usecols = [
        'loan_amnt', 'annual_inc', 'emp_length', 'grade',
        'delinq_2yrs', 'earliest_cr_line', 'open_acc', 'dti',
        'revol_util', 'verification_status', 'loan_status', 'issue_d',
        # NEW v2 features
        'int_rate', 'installment', 'total_acc',
    ]

    df = pd.read_csv(file_path, usecols=usecols, low_memory=False)

    # ── Labels ────────────────────────────────────────────────────────────────
    valid_statuses = ['Fully Paid', 'Charged Off', 'Default']
    df = df[df['loan_status'].isin(valid_statuses)].copy()
    df['label'] = df['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1, 'Default': 1})

    # ── Credit age ────────────────────────────────────────────────────────────
    df['issue_d']          = pd.to_datetime(df['issue_d'],          format='%b-%Y', errors='coerce')
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y', errors='coerce')
    df['credit_age_months'] = (
        (df['issue_d'] - df['earliest_cr_line']) / np.timedelta64(1, 'D') / 30.4368
    ).fillna(0).astype(int)

    # ── Employment length ─────────────────────────────────────────────────────
    emp_map = {
        '< 1 year': 6, '1 year': 12, '2 years': 24, '3 years': 36,
        '4 years': 48, '5 years': 60, '6 years': 72, '7 years': 84,
        '8 years': 96, '9 years': 108, '10+ years': 120
    }
    df['employment_months'] = df['emp_length'].map(emp_map).fillna(0)

    # ── Core schema ───────────────────────────────────────────────────────────
    df['loan_amount']   = df['loan_amnt']
    df['annual_income'] = df['annual_inc']
    p99 = df['annual_income'].quantile(0.99)
    df['annual_income'] = df['annual_income'].clip(upper=p99)

    grade_score_map = {'A': 750, 'B': 700, 'C': 650, 'D': 600, 'E': 550, 'F': 500, 'G': 450}
    df['credit_score']         = df['grade'].map(grade_score_map).fillna(600)
    df['delinquencies_2yr']    = df['delinq_2yrs']
    df['open_accounts']        = df['open_acc']
    df['debt_to_income']       = df['dti']

    if df['revol_util'].dtype == 'O':
        df['revol_util'] = df['revol_util'].str.rstrip('%').astype(float)
    df['revolving_utilisation'] = df['revol_util'].clip(upper=100)

    df['income_verified']  = df['verification_status']
    df['employment_type']  = 'unknown'
    df['dataset_source']   = 'lending_club'

    # ── NEW v2 Features ───────────────────────────────────────────────────────
    # 1. interest_rate — direct from LC (already float)
    df['interest_rate'] = df['int_rate'].fillna(df['int_rate'].median())

    # 2. monthly_installment — direct from LC
    df['monthly_installment'] = df['installment'].fillna(df['installment'].median())

    # 3. loan_grade_encoded — A=1 (safest) to G=7 (riskiest)
    grade_num_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df['loan_grade_encoded'] = df['grade'].map(grade_num_map).fillna(4)

    # 4. total_credit_lines — direct from LC
    df['total_credit_lines'] = df['total_acc'].fillna(df['total_acc'].median())

    unified_cols = [
        'loan_amount', 'annual_income', 'employment_months', 'credit_score',
        'delinquencies_2yr', 'credit_age_months', 'open_accounts',
        'debt_to_income', 'revolving_utilisation', 'income_verified',
        'employment_type', 'dataset_source', 'label',
        # NEW
        'interest_rate', 'monthly_installment', 'loan_grade_encoded', 'total_credit_lines',
    ]

    df = df[unified_cols].dropna()
    print(f"  LendingClub processed: {len(df):,} rows.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Home Credit
# ─────────────────────────────────────────────────────────────────────────────
def process_home_credit(app_path, bureau_path):
    print("Processing Home Credit Data...")

    app_cols = [
        'SK_ID_CURR', 'TARGET', 'AMT_CREDIT', 'AMT_INCOME_TOTAL',
        'DAYS_EMPLOYED', 'EXT_SOURCE_2', 'AMT_ANNUITY', 'NAME_INCOME_TYPE',
        'AMT_GOODS_PRICE',
    ]
    df_app = pd.read_csv(app_path, usecols=app_cols)

    # Employment
    df_app['DAYS_EMPLOYED'].replace(365243, 0, inplace=True)
    df_app['employment_months'] = (-df_app['DAYS_EMPLOYED'] / 30).clip(lower=0).fillna(0).astype(int)

    # DTI approximation
    df_app['debt_to_income'] = (
        (df_app['AMT_ANNUITY'] * 12) / df_app['AMT_INCOME_TOTAL'] * 100
    ).fillna(0)

    # Credit score from EXT_SOURCE_2
    ext_min = df_app['EXT_SOURCE_2'].min()
    ext_max = df_app['EXT_SOURCE_2'].max()
    df_app['credit_score'] = 300 + (
        (df_app['EXT_SOURCE_2'] - ext_min) / (ext_max - ext_min + 1e-8)
    ) * (850 - 300)
    df_app['credit_score'] = df_app['credit_score'].fillna(600)

    # Bureau aggregation
    print("  Aggregating Bureau data...")
    bur_cols = ['SK_ID_CURR', 'DAYS_CREDIT', 'CREDIT_DAY_OVERDUE', 'CREDIT_ACTIVE']
    df_bureau = pd.read_csv(bureau_path, usecols=bur_cols)
    bureau_agg = df_bureau.groupby('SK_ID_CURR').agg(
        min_days_credit  = ('DAYS_CREDIT', 'min'),
        total_overdue    = ('CREDIT_DAY_OVERDUE', 'sum'),
        active_accounts  = ('CREDIT_ACTIVE', lambda x: (x == 'Active').sum()),
        total_bureau_acc = ('DAYS_CREDIT', 'count'),        # proxy for total_credit_lines
    ).reset_index()

    bureau_agg['credit_age_months'] = (-bureau_agg['min_days_credit'] / 30).fillna(0).astype(int)
    bureau_agg['delinquencies_2yr'] = (bureau_agg['total_overdue'] > 30).astype(int)

    df = df_app.merge(bureau_agg, on='SK_ID_CURR', how='left')
    df['credit_age_months']  = df['credit_age_months'].fillna(0)
    df['active_accounts']    = df['active_accounts'].fillna(0)
    df['delinquencies_2yr']  = df['delinquencies_2yr'].fillna(0)
    df['total_bureau_acc']   = df['total_bureau_acc'].fillna(0)

    # Core schema
    df['loan_amount']           = df['AMT_CREDIT']
    df['annual_income']         = df['AMT_INCOME_TOTAL']
    df['open_accounts']         = df['active_accounts']
    df['revolving_utilisation'] = 50.0    # HC has no direct equivalent; impute median
    df['income_verified']       = 'Not Verified'
    df['employment_type']       = df['NAME_INCOME_TYPE']
    df['dataset_source']        = 'home_credit'
    df['label']                 = df['TARGET']

    # ── NEW v2 Features (HC equivalents) ──────────────────────────────────────
    # 1. interest_rate — estimated: annuity / credit ratio (scaled to typical range 6-30%)
    #    HC loan annuity / credit amount * 12 gives annual payment rate
    df['interest_rate'] = (
        (df['AMT_ANNUITY'] * 12) / df['AMT_CREDIT'].clip(lower=1) * 100
    ).clip(lower=5, upper=40).fillna(15.0)

    # 2. monthly_installment — AMT_ANNUITY is the monthly payment in HC
    df['monthly_installment'] = df['AMT_ANNUITY'].fillna(df['AMT_ANNUITY'].median())

    # 3. loan_grade_encoded — derived from credit_score bands (matches LC grade logic)
    def score_to_grade(score):
        if score >= 720: return 1   # A
        elif score >= 680: return 2 # B
        elif score >= 640: return 3 # C
        elif score >= 600: return 4 # D
        elif score >= 560: return 5 # E
        elif score >= 520: return 6 # F
        else: return 7              # G
    df['loan_grade_encoded'] = df['credit_score'].apply(score_to_grade)

    # 4. total_credit_lines — total bureau records (best HC proxy for total_acc)
    df['total_credit_lines'] = df['total_bureau_acc'].fillna(0)

    unified_cols = [
        'loan_amount', 'annual_income', 'employment_months', 'credit_score',
        'delinquencies_2yr', 'credit_age_months', 'open_accounts',
        'debt_to_income', 'revolving_utilisation', 'income_verified',
        'employment_type', 'dataset_source', 'label',
        # NEW
        'interest_rate', 'monthly_installment', 'loan_grade_encoded', 'total_credit_lines',
    ]

    df = df[unified_cols]
    print(f"  Home Credit processed: {len(df):,} rows.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
def feature_engineer(df):
    print("Performing Feature Engineering...")

    # 1. Income confidence encoding
    verif_map = {'Source Verified': 0.90, 'Verified': 0.75, 'Not Verified': 0.40}
    df['income_confidence'] = df['income_verified'].map(verif_map).fillna(0.40)
    df['verified_income']   = df['annual_income'] * df['income_confidence']

    # 2. Loan-to-Income Ratio
    df['loan_to_income_ratio'] = df['loan_amount'] / df['verified_income'].clip(lower=1)

    # 3. Debt Burden Ratio
    existing_monthly_debt  = (df['debt_to_income'] / 100) * (df['annual_income'] / 12)
    df['debt_burden_ratio'] = existing_monthly_debt / (df['verified_income'] / 12).clip(lower=1)

    # 4. Thin File Flag
    df['thin_file'] = ((df['credit_age_months'] < 24) | (df['open_accounts'] < 3)).astype(int)

    # 5. NEW — installment-to-income ratio (how heavy is the monthly payment vs income)
    df['installment_to_income'] = df['monthly_installment'] / (df['annual_income'] / 12).clip(lower=1)

    df = df.fillna(0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    lc_path        = "../dataset/lending_club/loan.csv"
    hc_app_path    = "../dataset/home-credit-default-risk/application_train.csv"
    hc_bureau_path = "../dataset/home-credit-default-risk/bureau.csv"
    output_path    = "../dataset/unified_training_data.csv"

    try:
        df_lc = process_lending_club(lc_path)
        df_lc = df_lc.sample(n=min(350000, len(df_lc)), random_state=42)

        df_hc = process_home_credit(hc_app_path, hc_bureau_path)

        print("Concatenating datasets...")
        df_unified = pd.concat([df_lc, df_hc], ignore_index=True)

        del df_lc, df_hc
        gc.collect()

        df_unified = feature_engineer(df_unified)

        print(f"Final Dataset Shape: {df_unified.shape}")
        print(f"Columns: {list(df_unified.columns)}")
        print(f"Default rate: {df_unified['label'].mean():.2%}")

        df_unified.to_csv(output_path, index=False)
        print(f"Saved -> {output_path}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")
