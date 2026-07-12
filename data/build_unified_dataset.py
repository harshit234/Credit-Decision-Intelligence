import os
import gc
import pandas as pd
import numpy as np

def process_lending_club(file_path):
    print("Processing LendingClub Data...")
    
    # We only read the necessary columns to save memory
    usecols = [
        'loan_amnt', 'annual_inc', 'emp_length', 'grade', 
        'delinq_2yrs', 'earliest_cr_line', 'open_acc', 'dti', 
        'revol_util', 'verification_status', 'loan_status', 'issue_d'
    ]
    
    # Load dataset
    df = pd.read_csv(file_path, usecols=usecols, low_memory=False)
    
    # 1. Filter Outcomes
    valid_statuses = ['Fully Paid', 'Charged Off', 'Default']
    df = df[df['loan_status'].isin(valid_statuses)].copy()
    
    # Map label (0 = Good, 1 = Bad)
    df['label'] = df['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1, 'Default': 1})
    
    # 2. Date parsing (Credit Age)
    df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y')
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y')
    
    # Calculate credit age in months
    df['credit_age_months'] = ((df['issue_d'] - df['earliest_cr_line']) / np.timedelta64(1, 'D') / 30.4368).fillna(0).astype(int)
    
    # 3. Employment length parsing
    emp_map = {
        '< 1 year': 6, '1 year': 12, '2 years': 24, '3 years': 36,
        '4 years': 48, '5 years': 60, '6 years': 72, '7 years': 84,
        '8 years': 96, '9 years': 108, '10+ years': 120
    }
    df['employment_months'] = df['emp_length'].map(emp_map).fillna(0)
    
    # 4. Map to Unified Schema
    df['loan_amount'] = df['loan_amnt']
    df['annual_income'] = df['annual_inc']
    
    # Winsorize income at 99th percentile (~300k)
    p99 = df['annual_income'].quantile(0.99)
    df['annual_income'] = df['annual_income'].clip(upper=p99)
    
    # Map grade to proxy credit score since fico_range_low is missing
    grade_map = {'A': 750, 'B': 700, 'C': 650, 'D': 600, 'E': 550, 'F': 500, 'G': 450}
    df['credit_score'] = df['grade'].map(grade_map).fillna(600)
    
    df['delinquencies_2yr'] = df['delinq_2yrs']
    df['open_accounts'] = df['open_acc']
    df['debt_to_income'] = df['dti']
    
    # Parse revol_util (string with % to float)
    if df['revol_util'].dtype == 'O':
        df['revol_util'] = df['revol_util'].str.rstrip('%').astype(float)
    df['revolving_utilisation'] = df['revol_util'].clip(upper=100)
    
    df['income_verified'] = df['verification_status']
    
    # Impute missing columns for LC
    df['employment_type'] = 'unknown'
    df['dataset_source'] = 'lending_club'
    
    # Keep only unified schema columns
    unified_cols = [
        'loan_amount', 'annual_income', 'employment_months', 'credit_score',
        'delinquencies_2yr', 'credit_age_months', 'open_accounts',
        'debt_to_income', 'revolving_utilisation', 'income_verified',
        'employment_type', 'dataset_source', 'label'
    ]
    
    df = df[unified_cols]
    
    # Drop NAs to ensure clean training data
    df = df.dropna()
    
    print(f"LendingClub processed: {len(df)} rows.")
    return df

def process_home_credit(app_path, bureau_path):
    print("Processing Home Credit Data...")
    
    # Load Main Application Table
    app_cols = [
        'SK_ID_CURR', 'TARGET', 'AMT_CREDIT', 'AMT_INCOME_TOTAL', 
        'DAYS_EMPLOYED', 'EXT_SOURCE_2', 'AMT_ANNUITY', 'NAME_INCOME_TYPE'
    ]
    df_app = pd.read_csv(app_path, usecols=app_cols)
    
    # Fix DAYS_EMPLOYED anomaly
    df_app['DAYS_EMPLOYED'].replace(365243, 0, inplace=True)
    df_app['employment_months'] = (-df_app['DAYS_EMPLOYED'] / 30).clip(lower=0).fillna(0).astype(int)
    
    # DTI Approximation
    df_app['debt_to_income'] = ((df_app['AMT_ANNUITY'] * 12) / df_app['AMT_INCOME_TOTAL'] * 100).fillna(0)
    
    # Scale EXT_SOURCE_2 to FICO Range (300 - 850)
    # Handle NaNs first
    ext_min = df_app['EXT_SOURCE_2'].min()
    ext_max = df_app['EXT_SOURCE_2'].max()
    df_app['credit_score'] = 300 + ((df_app['EXT_SOURCE_2'] - ext_min) / (ext_max - ext_min)) * (850 - 300)
    df_app['credit_score'] = df_app['credit_score'].fillna(600) # conservative imputation for missing
    
    # Load Bureau Table
    print("Aggregating Home Credit Bureau...")
    bur_cols = ['SK_ID_CURR', 'DAYS_CREDIT', 'CREDIT_DAY_OVERDUE', 'CREDIT_ACTIVE']
    df_bureau = pd.read_csv(bureau_path, usecols=bur_cols)
    
    # Aggregate bureau features
    # credit_age_months = -min(DAYS_CREDIT)/30
    bureau_agg = df_bureau.groupby('SK_ID_CURR').agg(
        min_days_credit=('DAYS_CREDIT', 'min'),
        total_overdue=('CREDIT_DAY_OVERDUE', 'sum'),
        active_accounts=('CREDIT_ACTIVE', lambda x: (x == 'Active').sum())
    ).reset_index()
    
    bureau_agg['credit_age_months'] = (-bureau_agg['min_days_credit'] / 30).fillna(0).astype(int)
    bureau_agg['delinquencies_2yr'] = (bureau_agg['total_overdue'] > 30).astype(int) # proxy
    
    # Merge
    df = df_app.merge(bureau_agg, on='SK_ID_CURR', how='left')
    
    # Fill missing bureau info with 0s (Thin file)
    df['credit_age_months'] = df['credit_age_months'].fillna(0)
    df['active_accounts'] = df['active_accounts'].fillna(0)
    df['delinquencies_2yr'] = df['delinquencies_2yr'].fillna(0)
    
    # Map to Unified Schema
    df['loan_amount'] = df['AMT_CREDIT']
    df['annual_income'] = df['AMT_INCOME_TOTAL']
    df['open_accounts'] = df['active_accounts']
    df['revolving_utilisation'] = 50.0 # HC doesn't explicitly have this without complex joins, imputing median
    df['income_verified'] = 'Not Verified' # Conservative imputation
    df['employment_type'] = df['NAME_INCOME_TYPE']
    df['dataset_source'] = 'home_credit'
    df['label'] = df['TARGET']
    
    unified_cols = [
        'loan_amount', 'annual_income', 'employment_months', 'credit_score',
        'delinquencies_2yr', 'credit_age_months', 'open_accounts',
        'debt_to_income', 'revolving_utilisation', 'income_verified',
        'employment_type', 'dataset_source', 'label'
    ]
    
    df = df[unified_cols]
    
    print(f"Home Credit processed: {len(df)} rows.")
    return df

def feature_engineer(df):
    print("Performing Feature Engineering...")
    # 1. Income Confidence Encoding
    verif_map = {
        'Source Verified': 0.90,
        'Verified': 0.75,
        'Not Verified': 0.40
    }
    df['income_confidence'] = df['income_verified'].map(verif_map).fillna(0.40)
    df['verified_income'] = df['annual_income'] * df['income_confidence']
    
    # 2. Loan to Income Ratio
    df['loan_to_income_ratio'] = df['loan_amount'] / df['verified_income'].clip(lower=1)
    
    # 3. Debt Burden Ratio (Existing Monthly Debt / Monthly Verified Income)
    existing_monthly_debt = (df['debt_to_income'] / 100) * (df['annual_income'] / 12)
    df['debt_burden_ratio'] = existing_monthly_debt / (df['verified_income'] / 12).clip(lower=1)
    
    # 4. Thin File Flag (Age < 24mo or Acc < 3)
    df['thin_file'] = ((df['credit_age_months'] < 24) | (df['open_accounts'] < 3)).astype(int)
    
    # 5. Missing values
    df = df.fillna(0)
    
    return df

if __name__ == "__main__":
    lc_path = "../dataset/lending_club/loan.csv"
    hc_app_path = "../dataset/home-credit-default-risk/application_train.csv"
    hc_bureau_path = "../dataset/home-credit-default-risk/bureau.csv"
    
    output_path = "../dataset/unified_training_data.csv"
    
    # Optional: We can sample to prevent memory issues and balance the dataset
    # E.g., Use 300k from LC and 300k from HC.
    
    try:
        df_lc = process_lending_club(lc_path)
        # Sample LC to roughly match HC volume to prevent class/source overwhelming
        df_lc = df_lc.sample(n=350000, random_state=42)
        
        df_hc = process_home_credit(hc_app_path, hc_bureau_path)
        
        # Merge
        print("Concatenating datasets...")
        df_unified = pd.concat([df_lc, df_hc], ignore_index=True)
        
        # Free up memory
        del df_lc
        del df_hc
        gc.collect()
        
        df_unified = feature_engineer(df_unified)
        
        print(f"Final Dataset Shape: {df_unified.shape}")
        
        print("Saving to CSV...")
        df_unified.to_csv(output_path, index=False)
        print(f"Unified dataset saved successfully at {output_path}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
