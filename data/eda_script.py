import os
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_eda(csv_path, output_dir):
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Generating Univariate Distribution of Risk Score Proxy...")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='credit_score', hue='label', bins=30, kde=True, palette='coolwarm')
    plt.title("Distribution of Credit Score by Default Status")
    plt.xlabel("Credit Score Proxy")
    plt.ylabel("Count")
    plt.savefig(os.path.join(output_dir, 'credit_score_dist.png'))
    plt.close()
    
    print("Generating Bivariate Plot (Loan to Income Ratio vs Default)...")
    plt.figure(figsize=(10, 6))
    # Cap ratio at 95th percentile for better visualization
    cap = df['loan_to_income_ratio'].quantile(0.95)
    sns.boxplot(data=df[df['loan_to_income_ratio'] <= cap], x='label', y='loan_to_income_ratio', palette='Set2')
    plt.title("Loan to Income Ratio vs Default")
    plt.xlabel("Default Status (0=Repaid, 1=Default)")
    plt.ylabel("Loan to Verified Income Ratio")
    plt.savefig(os.path.join(output_dir, 'lti_vs_default.png'))
    plt.close()

    print("Generating Thin File Impact Plot...")
    plt.figure(figsize=(8, 5))
    default_rates = df.groupby('thin_file')['label'].mean().reset_index()
    sns.barplot(data=default_rates, x='thin_file', y='label', palette='viridis')
    plt.title("Default Rate by Thin File Status")
    plt.xlabel("Thin File (1=Yes, 0=No)")
    plt.ylabel("Default Rate")
    plt.savefig(os.path.join(output_dir, 'thin_file_default_rate.png'))
    plt.close()
    
    print("Generating Correlation Heatmap...")
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=False, cmap='RdBu_r', center=0, vmin=-1, vmax=1)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()
    
    print("EDA generation complete.")

if __name__ == "__main__":
    csv_path = "../dataset/unified_training_data.csv"
    output_dir = "../dataset/eda_plots"
    generate_eda(csv_path, output_dir)
