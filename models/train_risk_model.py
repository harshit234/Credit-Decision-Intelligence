"""
================================================================================
   HALCYON CREDIT -- Risk Model Training Pipeline v3 (LightGBM / AmEx)
   Stage 2 | Author: Harshit
   Trains LightGBM on aggregated AmEx dataset targeting PR-AUC > 0.90
================================================================================

Outputs:
  models/lgbm_halcyon_v1.txt          trained LightGBM model
  models/feature_schema_v1.json       updated schema + optimal threshold
  dataset/eda_plots/shap_summary.png  SHAP feature importance
  dataset/eda_plots/pr_curve.png      Precision-Recall curve
  dataset/eda_plots/calibration_curve.png
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, classification_report, f1_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, accuracy_score
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

# ── Graceful rich console ─────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    console = Console(highlight=False)
    def rprint(msg): 
        try: console.print(msg)
        except: print(msg)
    def rule(t): 
        try: console.rule(f"[bold cyan]{t}[/]")
        except: print(f"\n{'─'*60}\n  {t}\n{'─'*60}")
except:
    def rprint(msg): print(msg)
    def rule(t): print(f"\n{'─'*60}\n  {t}\n{'─'*60}")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH     = "../dataset/amex_training_data.parquet"
MODEL_DIR     = "../models"
PLOT_DIR      = "../dataset/eda_plots"
MODEL_PATH    = f"{MODEL_DIR}/lgbm_halcyon_v1.txt"
SCHEMA_PATH   = f"{MODEL_DIR}/feature_schema_v1.json"
SHAP_PLOT     = f"{PLOT_DIR}/shap_summary.png"
PR_PLOT       = f"{PLOT_DIR}/pr_curve.png"
CALIB_PLOT    = f"{PLOT_DIR}/calibration_curve.png"
MODEL_VERSION = "lgbm_halcyon_v1_amex"
TARGET_COL    = "target"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 -- LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    rule("Section 1 -- Load Data")
    rprint(f"  Loading: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    rprint(f"  Shape        : {df.shape}")
    rprint(f"  Default rate : {df[TARGET_COL].mean():.2%}")
    rprint(f"  Class counts : {df[TARGET_COL].value_counts().to_dict()}")

    # Separate features and target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Drop any remaining non-numeric columns
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        rprint(f"  Dropping non-numeric cols: {non_numeric}")
        X = X.drop(columns=non_numeric)

    # Fill remaining NaNs with median
    X = X.fillna(X.median())
    rprint(f"  Feature count: {X.shape[1]:,}")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 -- TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
def split_data(X, y):
    rule("Section 2 -- Train / Test Split")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    rprint(f"  Train : {len(X_tr):,} rows | Default rate: {y_tr.mean():.2%}")
    rprint(f"  Test  : {len(X_te):,} rows | Default rate: {y_te.mean():.2%}")
    return X_tr, X_te, y_tr, y_te


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 -- TRAIN LIGHTGBM
# ─────────────────────────────────────────────────────────────────────────────
def train_model(X_tr, y_tr, X_te, y_te):
    rule("Section 3 -- Training LightGBM")
    try:
        import lightgbm as lgb
    except ImportError:
        rprint("  LightGBM not found. Installing...")
        os.system("pip install lightgbm -q")
        import lightgbm as lgb

    n_pos = int(y_tr.sum())
    n_neg = int((y_tr == 0).sum())
    scale = n_neg / n_pos
    rprint(f"  Class balance  : {n_neg:,} good / {n_pos:,} default")
    rprint(f"  scale_pos_weight: {scale:.2f}")

    # LightGBM datasets
    dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
    dval   = lgb.Dataset(X_te, label=y_te, reference=dtrain, free_raw_data=False)

    params = {
        # Objective
        "objective"         : "binary",
        "metric"            : ["binary_logloss", "auc", "average_precision"],
        # Class imbalance
        "scale_pos_weight"  : scale,
        "is_unbalance"      : False,   # using scale_pos_weight instead
        # Tree structure
        "num_leaves"        : 127,     # deep trees for complex patterns
        "max_depth"         : -1,      # unlimited depth
        "min_child_samples" : 20,      # minimum samples per leaf
        "min_child_weight"  : 1e-3,
        # Learning
        "learning_rate"     : 0.02,    # slow + more trees = better generalization
        "n_estimators"      : 2000,
        "feature_fraction"  : 0.20,    # use 20% features per tree (regularize ~1,900 features)
        "bagging_fraction"  : 0.85,
        "bagging_freq"      : 1,
        # Regularization
        "reg_alpha"         : 0.1,
        "reg_lambda"        : 0.1,
        "min_split_gain"    : 0.0,
        # Speed
        "n_jobs"            : -1,
        "verbose"           : -1,
        "random_state"      : 42,
        "device_type"       : "cpu",
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=100),
    ]

    rprint("  Training LightGBM (early stopping at 50 rounds)...")
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=callbacks,
    )
    rprint(f"  Best iteration : {model.best_iteration}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 -- OPTIMAL THRESHOLD VIA PR CURVE
# ─────────────────────────────────────────────────────────────────────────────
def find_optimal_threshold(model, X_te, y_te):
    rule("Section 4 -- Optimal Threshold via PR Curve")
    y_prob = model.predict(X_te, num_iteration=model.best_iteration)
    precisions, recalls, thresholds = precision_recall_curve(y_te, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

    best_idx    = np.argmax(f1_scores[:-1])
    best_thresh = float(thresholds[best_idx])
    best_f1     = float(f1_scores[best_idx])

    rprint(f"  Optimal threshold : {best_thresh:.4f}")
    rprint(f"  Best F1 (Default) : {best_f1:.4f}")

    # PR Curve plot
    pr_auc = average_precision_score(y_te, y_prob)
    plt.figure(figsize=(9, 6))
    plt.plot(recalls, precisions, color="steelblue", lw=2.5,
             label=f"PR Curve (AUC = {pr_auc:.4f})")
    plt.scatter(recalls[best_idx], precisions[best_idx],
                color="red", s=150, zorder=5,
                label=f"Optimal (thresh={best_thresh:.3f}, F1={best_f1:.3f})")
    plt.axhline(y=0.85, color="orange", linestyle="--", alpha=0.7, label="PR=0.85 reference")
    plt.xlabel("Recall (Default)", fontsize=12)
    plt.ylabel("Precision (Default)", fontsize=12)
    plt.title("Precision-Recall Curve -- Halcyon Risk Model v3 (LightGBM/AmEx)", fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PR_PLOT, dpi=150)
    plt.close()
    rprint(f"  PR curve saved -> {PR_PLOT}")

    return y_prob, best_thresh, pr_auc


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 -- FULL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(y_te, y_prob, threshold):
    rule("Section 5 -- Full Model Evaluation")

    y_pred = (y_prob >= threshold).astype(int)
    auc    = roc_auc_score(y_te, y_prob)
    prauc  = average_precision_score(y_te, y_prob)
    acc    = accuracy_score(y_te, y_pred)
    f1_def = f1_score(y_te, y_pred, pos_label=1)
    f1_good= f1_score(y_te, y_pred, pos_label=0)

    rprint(f"\n  [bold green]ROC-AUC          : {auc:.4f}[/]")
    rprint(f"  [bold green]PR-AUC           : {prauc:.4f}[/]")
    rprint(f"  Accuracy         : {acc:.4f}")
    rprint(f"  F1 (Default)     : {f1_def:.4f}")
    rprint(f"  F1 (Good Loan)   : {f1_good:.4f}")
    rprint(f"  Threshold        : {threshold:.4f}")

    report = classification_report(y_te, y_pred, target_names=["Good Loan", "Default"])
    rprint(f"\n{report}")

    cm = confusion_matrix(y_te, y_pred)
    rprint(f"  Confusion Matrix:")
    rprint(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    rprint(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    recall_def = cm[1,1] / (cm[1,1] + cm[1,0] + 1e-8)
    rprint(f"  Default Recall   : {recall_def:.2%}  (catching {cm[1,1]:,} of {cm[1,1]+cm[1,0]:,} real defaults)")

    # Balanced evaluation
    rprint("\n  [B] BALANCED TEST SET (50/50)")
    idx_good = np.where(y_te == 0)[0]
    idx_bad  = np.where(y_te == 1)[0]
    n_bal = min(len(idx_good), len(idx_bad))
    np.random.seed(42)
    bal_idx = np.concatenate([
        np.random.choice(idx_good, n_bal, replace=False),
        idx_bad
    ])
    y_bal    = y_te.iloc[bal_idx] if hasattr(y_te, 'iloc') else y_te[bal_idx]
    prob_bal = y_prob[bal_idx]
    pred_bal = (prob_bal >= threshold).astype(int)
    bal_acc  = accuracy_score(y_bal, pred_bal)
    bal_auc  = roc_auc_score(y_bal, prob_bal)
    rprint(f"  Balanced Accuracy : {bal_acc:.4f}")
    rprint(f"  Balanced ROC-AUC  : {bal_auc:.4f}")
    rprint(f"\n{classification_report(y_bal, pred_bal, target_names=['Good Loan', 'Default'])}")

    # Calibration
    prob_true, prob_pred_cal = calibration_curve(y_te, y_prob, n_bins=10)
    plt.figure(figsize=(7, 5))
    plt.plot(prob_pred_cal, prob_true, marker="o", label="LightGBM v3")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve -- Halcyon Risk Model v3")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CALIB_PLOT, dpi=150)
    plt.close()
    rprint(f"\n  Calibration curve -> {CALIB_PLOT}")

    return auc, prauc, acc, f1_def


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 -- SHAP FEATURE IMPORTANCE (TOP 30)
# ─────────────────────────────────────────────────────────────────────────────
def compute_shap(model, X_te):
    rule("Section 6 -- SHAP Feature Importance")
    try:
        import shap
        sample = X_te.sample(min(1000, len(X_te)), random_state=42)
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, sample, plot_type="bar",
                          show=False, max_display=30)
        plt.title("Top 30 SHAP Features -- Halcyon Risk Model v3 (LightGBM/AmEx)")
        plt.tight_layout()
        plt.savefig(SHAP_PLOT, dpi=150)
        plt.close()
        rprint(f"  SHAP plot saved -> {SHAP_PLOT}")
    except Exception as e:
        rprint(f"  SHAP fallback (reason: {e}) -- using LightGBM importance")
        imp = model.feature_importance(importance_type="gain")
        feat_names = model.feature_name()
        top_idx = np.argsort(imp)[-30:]
        plt.figure(figsize=(12, 8))
        plt.barh([feat_names[i] for i in top_idx],
                 [imp[i] for i in top_idx], color="steelblue")
        plt.xlabel("Gain Importance")
        plt.title("Top 30 Features (Gain) -- Halcyon Risk Model v3")
        plt.tight_layout()
        plt.savefig(SHAP_PLOT, dpi=150)
        plt.close()
        rprint(f"  Feature importance -> {SHAP_PLOT}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 -- SAVE MODEL & SCHEMA
# ─────────────────────────────────────────────────────────────────────────────
def save_artifacts(model, X_tr, auc, prauc, acc, f1_def, threshold):
    rule("Section 7 -- Saving Artifacts")

    model.save_model(MODEL_PATH)
    rprint(f"  Model saved    -> {MODEL_PATH}")

    schema = {
        "model_version"      : MODEL_VERSION,
        "model_type"         : "LightGBM",
        "trained_at"         : datetime.now().isoformat(),
        "dataset"            : "American Express Default Prediction (Kaggle 2022)",
        "n_customers_train"  : int(len(X_tr)),
        "feature_count"      : int(X_tr.shape[1]),
        "feature_names"      : list(X_tr.columns),
        "target_column"      : TARGET_COL,
        "optimal_threshold"  : round(threshold, 4),
        "aggregation_stats"  : ["mean","std","min","max","last","first",
                                 "trend","recent_trend","cv","nonzero_count"],
        "risk_bands"         : {
            "Low"    : "score < 0.25",
            "Medium" : f"0.25 <= score < {round(threshold, 2)}",
            "High"   : f"score >= {round(threshold, 2)}"
        },
        "evaluation_metrics" : {
            "roc_auc"    : round(auc, 4),
            "pr_auc"     : round(prauc, 4),
            "accuracy"   : round(acc, 4),
            "f1_default" : round(f1_def, 4),
        },
    }
    with open(SCHEMA_PATH, "w") as f:
        json.dump(schema, f, indent=2)
    rprint(f"  Schema saved   -> {SCHEMA_PATH}")

    print("\n" + "=" * 65)
    print("  HALCYON RISK MODEL v3 (LightGBM/AmEx) -- COMPLETE")
    print("=" * 65)
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  PR-AUC    : {prauc:.4f}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1(Default): {f1_def:.4f}")
    print(f"  Threshold : {threshold:.4f}")
    print(f"  Features  : {X_tr.shape[1]:,}")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X, y             = load_data()
    X_tr, X_te, y_tr, y_te = split_data(X, y)
    model            = train_model(X_tr, y_tr, X_te, y_te)
    y_prob, threshold, prauc_check = find_optimal_threshold(model, X_te, y_te)
    auc, prauc, acc, f1_def = evaluate_model(y_te, y_prob, threshold)
    compute_shap(model, X_te)
    save_artifacts(model, X_tr, auc, prauc, acc, f1_def, threshold)
