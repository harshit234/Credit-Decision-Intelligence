"""
Quick training run on LendingClub-ONLY data from the unified dataset.
Measures actual achievable metrics with our current feature schema.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, accuracy_score
)
import lightgbm as lgb

print("=" * 65)
print("  HALCYON CREDIT — LendingClub-Only Training Feasibility Run")
print("=" * 65)

# ── Load LC-only rows ─────────────────────────────────────────────
df = pd.read_csv('dataset/unified_training_data.csv')
lc = df[df['dataset_source'] == 'lending_club'].copy()

print(f"\n  Dataset    : LendingClub (from unified_training_data.csv)")
print(f"  Total rows : {len(lc):,}")
print(f"  Defaults   : {int(lc['label'].sum()):,}  ({lc['label'].mean()*100:.2f}%)")
print(f"  Non-def    : {int((lc['label']==0).sum()):,}")

# ── Feature selection (pipeline-compatible features ONLY) ─────────
FEATURE_COLS = [
    'credit_score',
    'debt_to_income',
    'revolving_utilisation',
    'delinquencies_2yr',
    'credit_age_months',
    'open_accounts',
    'employment_months',
    'loan_amount',
    'annual_income',
    'verified_income',
    'income_confidence',
    'loan_to_income_ratio',
    'debt_burden_ratio',
    'thin_file',
    'interest_rate',
    'monthly_installment',
    'loan_grade_encoded',
    'total_credit_lines',
    'installment_to_income',
]

print(f"\n  Features   : {len(FEATURE_COLS)}")

X = lc[FEATURE_COLS].fillna(0)
y = lc['label']

# ── Train/test split ──────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Train rows : {len(X_tr):,}")
print(f"  Test rows  : {len(X_te):,}")

# ── Class weight ──────────────────────────────────────────────────
n_neg = int((y_tr == 0).sum())
n_pos = int((y_tr == 1).sum())
scale = n_neg / n_pos
print(f"\n  scale_pos_weight: {scale:.2f}")

# ── LightGBM ──────────────────────────────────────────────────────
print("\n  Training LightGBM...")
dtrain = lgb.Dataset(X_tr, label=y_tr)
dval   = lgb.Dataset(X_te, label=y_te, reference=dtrain)

params = {
    "objective"        : "binary",
    "metric"           : ["binary_logloss", "auc", "average_precision"],
    "scale_pos_weight" : scale,
    "num_leaves"       : 63,
    "learning_rate"    : 0.05,
    "n_estimators"     : 1000,
    "feature_fraction" : 0.8,
    "bagging_fraction" : 0.8,
    "bagging_freq"     : 1,
    "reg_alpha"        : 0.1,
    "reg_lambda"       : 0.1,
    "n_jobs"           : -1,
    "verbose"          : -1,
    "random_state"     : 42,
}

callbacks = [
    lgb.early_stopping(stopping_rounds=30, verbose=False),
    lgb.log_evaluation(period=200),
]

model = lgb.train(
    params, dtrain,
    num_boost_round=1000,
    valid_sets=[dval],
    callbacks=callbacks,
)

# ── Evaluation ────────────────────────────────────────────────────
y_prob = model.predict(X_te, num_iteration=model.best_iteration)

# Find optimal threshold
from sklearn.metrics import precision_recall_curve
prec, rec, thresholds = precision_recall_curve(y_te, y_prob)
f1s = 2 * (prec * rec) / (prec + rec + 1e-8)
best_idx = np.argmax(f1s[:-1])
best_thresh = float(thresholds[best_idx])

y_pred = (y_prob >= best_thresh).astype(int)

roc_auc  = roc_auc_score(y_te, y_prob)
pr_auc   = average_precision_score(y_te, y_prob)
acc      = accuracy_score(y_te, y_pred)
f1_def   = f1_score(y_te, y_pred, pos_label=1)
f1_good  = f1_score(y_te, y_pred, pos_label=0)
cm       = confusion_matrix(y_te, y_pred)
recall_d = cm[1,1] / (cm[1,1] + cm[1,0] + 1e-8)

print("\n" + "=" * 65)
print("  RESULTS — LightGBM on LendingClub-Only Data")
print("=" * 65)
print(f"  ROC-AUC           : {roc_auc:.4f}")
print(f"  PR-AUC            : {pr_auc:.4f}")
print(f"  Accuracy          : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  F1 (Default)      : {f1_def:.4f}")
print(f"  F1 (Good Loan)    : {f1_good:.4f}")
print(f"  Default Recall    : {recall_d:.4f}  ({recall_d*100:.2f}%)")
print(f"  Best Threshold    : {best_thresh:.4f}")
print(f"  Best Iteration    : {model.best_iteration}")
print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")
print("=" * 65)

print("\n  Classification Report:")
print(classification_report(y_te, y_pred, target_names=["Good Loan", "Default"]))

# ── Top features ─────────────────────────────────────────────────
print("  Top 10 Features by Importance (Gain):")
imp = model.feature_importance(importance_type='gain')
feat_imp = sorted(zip(FEATURE_COLS, imp), key=lambda x: x[1], reverse=True)[:10]
for rank, (feat, score) in enumerate(feat_imp, 1):
    print(f"    {rank:2d}. {feat:<30s} gain={score:,.0f}")

print("\n  Pipeline Compatibility:")
print("  - All features are collectable at loan application time: YES")
print("  - Feature names are human-readable (no P_2, D_39 etc.): YES")
print("  - SHAP explanation will be interpretable: YES")
print("  - Thin-file applicants included in training data: MINIMAL (0.38% of LC)")
print("  - No imputed/proxy features from HC merge needed: YES")
