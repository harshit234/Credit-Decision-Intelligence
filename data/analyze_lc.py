import pandas as pd
import numpy as np

# Load the unified dataset to get LC-only stats
df = pd.read_csv('dataset/unified_training_data.csv')

print("=== FULL UNIFIED DATASET ===")
print(f"Total rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")
print(f"Overall default rate: {df['label'].mean():.4f} ({df['label'].mean()*100:.2f}%)")
print()

# Split by source
lc = df[df['dataset_source'] == 'lending_club']
hc = df[df['dataset_source'] == 'home_credit']

print("=== LENDING CLUB (LC) ONLY ===")
print(f"  Rows: {len(lc):,}")
print(f"  Default rate: {lc['label'].mean():.4f} ({lc['label'].mean()*100:.2f}%)")
print(f"  Defaults: {int(lc['label'].sum()):,}  |  Non-defaults: {int((lc['label']==0).sum()):,}")
print()

print("=== HOME CREDIT (HC) ONLY ===")
print(f"  Rows: {len(hc):,}")
print(f"  Default rate: {hc['label'].mean():.4f} ({hc['label'].mean()*100:.2f}%)")
print(f"  Defaults: {int(hc['label'].sum()):,}  |  Non-defaults: {int((hc['label']==0).sum()):,}")
print()

print("=== LC - KEY FEATURE STATISTICS ===")
feat_cols = ['credit_score','debt_to_income','annual_income','loan_amount',
             'revolving_utilisation','delinquencies_2yr','credit_age_months',
             'open_accounts','loan_to_income_ratio','thin_file']
lc_num = lc[feat_cols].describe().round(2)
print(lc_num.to_string())

print()
print("=== LC - FEATURE VALUE RANGES ===")
for col in feat_cols:
    print(f"  {col}: min={lc[col].min():.2f} | median={lc[col].median():.2f} | max={lc[col].max():.2f} | null%={lc[col].isnull().mean()*100:.1f}%")

print()
print("=== LC - DEFAULT RATES BY KEY SEGMENTS ===")

# By credit score bucket
lc['score_bucket'] = pd.cut(lc['credit_score'], bins=[0,580,620,660,700,750,850], labels=['<580','580-620','620-660','660-700','700-750','750+'])
print("\nDefault rate by credit score bucket:")
print(lc.groupby('score_bucket', observed=True)['label'].agg(['mean','count']).round(3).to_string())

# By DTI bucket
lc['dti_bucket'] = pd.cut(lc['debt_to_income'], bins=[0,15,25,35,40,100], labels=['<15','15-25','25-35','35-40','40+'])
print("\nDefault rate by DTI bucket:")
print(lc.groupby('dti_bucket', observed=True)['label'].agg(['mean','count']).round(3).to_string())

# Thin file
print("\nDefault rate by thin file status:")
print(lc.groupby('thin_file', observed=True)['label'].agg(['mean','count']).round(3).to_string())

print()
print("=== LC - LOAN PURPOSE DISTRIBUTION ===")
if 'loan_purpose' in lc.columns:
    print(lc.groupby('loan_purpose')['label'].agg(['mean','count']).sort_values('count', ascending=False).round(3).to_string())

print()
print("=== UNIFIED DATASET - FEATURE LIST ===")
for c in df.columns:
    print(f"  {c}: dtype={df[c].dtype}")
