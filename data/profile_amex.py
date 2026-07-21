import pandas as pd
import os

size_gb = os.path.getsize('../dataset/train_data.csv/train_data.csv') / (1024**3)
print(f"File size: {size_gb:.2f} GB")

df = pd.read_csv('../dataset/train_data.csv/train_data.csv', nrows=200)
print(f"Total columns: {len(df.columns)}")
print(f"First 10 columns: {list(df.columns[:10])}")

# Check key identifying columns
key_cols = ['customer_ID', 'S_2', 'target']
for col in key_cols:
    print(f"  {col} exists: {col in df.columns}")

# Column prefix groups (D=delinquency, S=spend, P=payment, B=balance, R=risk)
prefixes = {}
for col in df.columns:
    if col not in ['customer_ID', 'S_2', 'target']:
        prefix = col.split('_')[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
print(f"\nColumn groups: {prefixes}")

# Null analysis
null_pct = df.isnull().mean()
high_null = null_pct[null_pct > 0.3]
print(f"Columns with >30% nulls: {len(high_null)}")
print(f"Columns with 0% nulls: {(null_pct == 0).sum()}")

# Time series check
if 'customer_ID' in df.columns:
    print(f"\nUnique customers in 200 rows: {df['customer_ID'].nunique()}")
if 'S_2' in df.columns:
    print(f"Sample dates: {df['S_2'].unique()[:5]}")
if 'target' in df.columns:
    print(f"Target values: {df['target'].unique()}")
    print(f"Target type: {df['target'].dtype}")
