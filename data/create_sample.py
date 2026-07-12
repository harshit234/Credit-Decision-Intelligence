import pandas as pd

df = pd.read_csv('../dataset/unified_training_data.csv')

lc_sample = df[df['dataset_source'] == 'lending_club'].sample(n=25, random_state=42)
hc_sample = df[df['dataset_source'] == 'home_credit'].sample(n=25, random_state=42)

sample = pd.concat([lc_sample, hc_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
sample.to_csv('../dataset/sample_unified_dataset.csv', index=False)

print(f"Sample shape: {sample.shape}")
print(f"LC rows: {(sample['dataset_source']=='lending_club').sum()}")
print(f"HC rows: {(sample['dataset_source']=='home_credit').sum()}")
print(f"Default rate: {sample['label'].mean():.2%}")
print(f"Columns: {list(sample.columns)}")
