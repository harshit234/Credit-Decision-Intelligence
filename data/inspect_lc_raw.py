"""Quick column inspector for raw loan.csv"""
import pandas as pd

print("Reading first 500 rows of loan.csv to inspect columns...")
df = pd.read_csv('dataset/lending_club/loan.csv', nrows=500, low_memory=False)

print(f"\nShape (500 rows sample): {df.shape}")
print(f"\nAll columns ({len(df.columns)}):")
for i, col in enumerate(df.columns):
    sample_val = df[col].dropna().iloc[0] if df[col].notna().any() else "NULL"
    null_pct = df[col].isnull().mean() * 100
    print(f"  {i+1:3d}. {col:<45s} dtype={str(df[col].dtype):<12s} null%={null_pct:5.1f}%  sample={str(sample_val)[:30]}")

print("\n\nChecking key columns of interest:")
key_cols = ['fico_range_low','fico_range_high','last_fico_range_low','last_fico_range_high',
            'loan_status','pub_rec','inq_last_6mths','revol_bal','revol_util',
            'annual_inc','loan_amnt','int_rate','dti','delinq_2yrs','earliest_cr_line',
            'open_acc','total_acc','emp_length','grade','sub_grade','purpose',
            'mort_acc','num_bc_tl','num_il_tl','num_op_rev_tl','pub_rec_bankruptcies',
            'installment','term','home_ownership','verification_status',
            'tot_cur_bal','tot_hi_cred_lim','total_rev_hi_lim','bc_util',
            'pct_tl_nvr_dlq','num_sats','acc_open_past_24mths']

for col in key_cols:
    if col in df.columns:
        null_pct = df[col].isnull().mean() * 100
        sample = str(df[col].dropna().iloc[0] if df[col].notna().any() else "NULL")[:30]
        print(f"  ✅ {col:<40s} null%={null_pct:5.1f}%  sample={sample}")
    else:
        print(f"  ❌ {col} — NOT FOUND")

print("\nLoan status distribution (500 rows sample):")
print(df['loan_status'].value_counts().to_string())
