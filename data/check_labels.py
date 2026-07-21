import pandas as pd

labels = pd.read_csv('../dataset/train_labels.csv/train_labels.csv')
print(f"Total label rows: {len(labels):,}")
print(f"Columns: {list(labels.columns)}")
print(f"Default rate: {labels['target'].mean():.2%}")
print(f"Class counts: {labels['target'].value_counts().to_dict()}")
print(f"Sample:")
print(labels.head())
