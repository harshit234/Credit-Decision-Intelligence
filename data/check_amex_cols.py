import pandas as pd
import numpy as np

df = pd.read_csv('../dataset/train_data.csv/train_data.csv', nrows=1000, low_memory=False)
feat_cols = [c for c in df.columns if c not in ['customer_ID', 'S_2']]

numeric_cols = []
categorical_cols = []

for col in feat_cols:
    try:
        df[col].astype(np.float32)
        numeric_cols.append(col)
    except (ValueError, TypeError):
        categorical_cols.append((col, df[col].dtype, df[col].unique()[:10]))

print(f"Numeric cols  : {len(numeric_cols)}")
print(f"Categorical cols: {len(categorical_cols)}")
for col, dtype, vals in categorical_cols:
    print(f"  {col:15s} | dtype={dtype} | values={vals}")
