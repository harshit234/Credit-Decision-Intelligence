import os
import shutil
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

PLOT_DIR = "amex_eda_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

print("Loading AmEx data sample for EDA...")
# Load a sample for EDA to save memory
df = pd.read_parquet('../dataset/amex_training_data.parquet')
df_sample = df.sample(min(50000, len(df)), random_state=42)

# 1. Target Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df_sample, x='target', palette='Set2')
plt.title("Target Distribution (0 = Good, 1 = Default)")
plt.savefig(f"{PLOT_DIR}/target_dist.png", dpi=150)
plt.close()

# 2. Distribution of a top feature (e.g., P_2_last)
if 'P_2_last' in df_sample.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df_sample, x='P_2_last', hue='target', bins=50, kde=True, palette='Set2', stat='density', common_norm=False)
    plt.title("Distribution of P_2_last by Target")
    plt.savefig(f"{PLOT_DIR}/p2_last_dist.png", dpi=150)
    plt.close()

# 3. Delinquency Escalation
if 'delinquency_escalation' in df_sample.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df_sample, x='target', y='delinquency_escalation', palette='Set2')
    plt.title("Delinquency Escalation by Target")
    plt.ylim(-0.5, 0.5) # clip outliers for better view
    plt.savefig(f"{PLOT_DIR}/delinq_esc_dist.png", dpi=150)
    plt.close()

# Move existing plots from dataset/eda_plots to data/amex_eda_plots
existing_plots = ['shap_summary.png', 'calibration_curve.png', 'pr_curve.png']
for p in existing_plots:
    src = f"../dataset/eda_plots/{p}"
    dst = f"{PLOT_DIR}/{p}"
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"Copied {src} to {dst}")

print("EDA plots generated successfully.")
