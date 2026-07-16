"""
================================================================================
   HALCYON CREDIT -- AmEx Dataset Aggregation Pipeline v2 (Memory-Safe)
   Stage 2 | Author: Aditya
   Architecture: Stream → Batch-Aggregate → Temp Parquet → Concat
================================================================================

Memory-safe design:
  - Stores raw rows as numpy float32 arrays (3x lighter than DataFrames)
  - Aggregates and writes in batches of 10K customers → temp parquet files
  - Concatenates at the end → single amex_training_data.parquet
  - Max RAM at any point: ~3-4GB

Output:
  dataset/amex_training_data.parquet   <- final training file
  dataset/amex_sample_50rows.csv       <- 50-row human-readable sample
"""

import os
import gc
import glob
import numpy as np
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# PATHS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH     = "../dataset/train_data.csv/train_data.csv"
LABELS_PATH   = "../dataset/train_labels.csv/train_labels.csv"
OUTPUT_DIR    = "../dataset"
TEMP_DIR      = "../dataset/temp_batches"
OUTPUT_PKL    = f"{OUTPUT_DIR}/amex_training_data.parquet"
OUTPUT_SAMPLE = f"{OUTPUT_DIR}/amex_sample_50rows.csv"

CHUNK_SIZE    = 100_000   # rows per read chunk (larger = faster streaming)
BATCH_SIZE    = 10_000    # customers per aggregation batch
N_NONDEFAULT  = 180_000   # non-defaults to keep (balance dataset)
RANDOM_SEED   = 42

os.makedirs(TEMP_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 -- LOAD LABELS & SELECT CUSTOMERS
# ─────────────────────────────────────────────────────────────────────────────
def select_customers():
    print("\n--- Section 1: Loading Labels & Selecting Customers ---")
    labels = pd.read_csv(LABELS_PATH)
    print(f"  Total customers : {len(labels):,}")
    print(f"  Default rate    : {labels['target'].mean():.2%}")

    defaults    = labels[labels['target'] == 1]['customer_ID'].tolist()
    non_def     = labels[labels['target'] == 0]['customer_ID'].tolist()

    rng = np.random.RandomState(RANDOM_SEED)
    sampled_nd  = rng.choice(non_def, size=min(N_NONDEFAULT, len(non_def)),
                              replace=False).tolist()

    selected    = set(defaults + sampled_nd)
    label_map   = dict(zip(labels['customer_ID'], labels['target']))

    print(f"  Defaults kept   : {len(defaults):,}  (all)")
    print(f"  Non-defaults    : {len(sampled_nd):,}  (sampled)")
    print(f"  Total selected  : {len(selected):,}")
    print(f"  New default rate: {len(defaults)/len(selected):.2%}")
    return selected, label_map


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 -- STREAM, FILTER & ACCUMULATE (numpy float32 arrays)
# ─────────────────────────────────────────────────────────────────────────────
# Categorical encodings for D_63 and D_64
D63_MAP = {'CR': 0, 'CO': 1, 'CL': 2, 'XZ': 3}
D64_MAP = {'O': 0, 'R': 1, 'U': 2, '-1': 3}


def stream_and_accumulate(selected_ids):
    """
    Streams 5.5M rows in 100K chunks.
    Encodes categorical cols (D_63, D_64) before converting to float32.
    Stores per-customer rows as numpy float32 arrays (memory-efficient).
    """
    print("\n--- Section 2: Streaming 5.5M Rows ---")

    customer_data = {}
    feat_cols     = None
    total_read    = 0

    for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE, low_memory=False):
        total_read += len(chunk)

        # Identify and fix feature columns once
        if feat_cols is None:
            feat_cols = [c for c in chunk.columns
                         if c not in ['customer_ID', 'S_2']]
            print(f"  Feature columns  : {len(feat_cols)}")

        # Filter to selected customers
        mask  = chunk['customer_ID'].isin(selected_ids)
        chunk = chunk[mask].copy()
        if len(chunk) == 0:
            continue

        # Encode categorical columns
        if 'D_63' in chunk.columns:
            chunk['D_63'] = chunk['D_63'].map(D63_MAP).fillna(-1)
        if 'D_64' in chunk.columns:
            chunk['D_64'] = chunk['D_64'].map(D64_MAP).fillna(-1)

        # Convert features to float32 numpy (memory-efficient)
        vals_np  = chunk[feat_cols].astype(np.float32).values
        dates_np = chunk['S_2'].values
        cids_arr = chunk['customer_ID'].values

        for i, cid in enumerate(cids_arr):
            if cid not in customer_data:
                customer_data[cid] = {'dates': [], 'vals': []}
            customer_data[cid]['dates'].append(dates_np[i])
            customer_data[cid]['vals'].append(vals_np[i])

        if total_read % 500_000 == 0:
            print(f"  Read {total_read:,} rows | "
                  f"customers loaded: {len(customer_data):,}")

    print(f"\n  Done. Total rows read     : {total_read:,}")
    print(f"  Customers accumulated     : {len(customer_data):,}")
    return customer_data, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 -- BATCH AGGREGATE (10K customers at a time → temp parquet)
# ─────────────────────────────────────────────────────────────────────────────
def compute_customer_stats(cid, data, feat_cols):
    """Compute 6 temporal stats per feature for one customer."""
    dates  = np.array(data['dates'])
    vals   = np.vstack(data['vals'])  # shape: (n_statements, n_features)

    # Sort by date
    order  = np.argsort(dates)
    vals   = vals[order]
    n      = len(vals)

    row    = {'customer_ID': cid, 'n_statements': n}

    for j, col in enumerate(feat_cols):
        v = vals[:, j].astype(np.float64)
        valid = v[~np.isnan(v)]

        if len(valid) == 0:
            for s in ['mean','std','min','max','last','trend']:
                row[f"{col}_{s}"] = 0.0
            continue

        v_last  = float(valid[-1])
        v_first = float(valid[0])

        row[f"{col}_mean"]  = float(np.mean(valid))
        row[f"{col}_std"]   = float(np.std(valid))
        row[f"{col}_min"]   = float(np.min(valid))
        row[f"{col}_max"]   = float(np.max(valid))
        row[f"{col}_last"]  = v_last
        row[f"{col}_trend"] = v_last - v_first   # direction over full history

    return row


def aggregate_in_batches(customer_data, feat_cols):
    """Process customers in batches of BATCH_SIZE, write temp parquet per batch."""
    print(f"\n--- Section 3: Batch Aggregation ({BATCH_SIZE:,} customers/batch) ---")

    cids       = list(customer_data.keys())
    n_batches  = (len(cids) + BATCH_SIZE - 1) // BATCH_SIZE
    batch_files = []

    for b_idx in range(n_batches):
        start = b_idx * BATCH_SIZE
        end   = min(start + BATCH_SIZE, len(cids))
        batch_cids = cids[start:end]

        rows = []
        for cid in batch_cids:
            row = compute_customer_stats(cid, customer_data[cid], feat_cols)
            rows.append(row)
            del customer_data[cid]   # free RAM as we go

        # Save batch
        batch_df   = pd.DataFrame(rows)
        batch_path = f"{TEMP_DIR}/batch_{b_idx:04d}.parquet"
        batch_df.to_parquet(batch_path, index=False, compression='snappy')
        batch_files.append(batch_path)
        del rows, batch_df
        gc.collect()

        print(f"  Batch {b_idx+1:3d}/{n_batches} | "
              f"customers {start:,}-{end:,} | saved: {batch_path}")

    return batch_files


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 -- CONCAT ALL TEMP FILES
# ─────────────────────────────────────────────────────────────────────────────
def concat_batches(batch_files):
    print(f"\n--- Section 4: Concatenating {len(batch_files)} batch files ---")
    dfs = []
    for f in batch_files:
        dfs.append(pd.read_parquet(f))
    df = pd.concat(dfs, ignore_index=True)
    print(f"  Combined shape: {df.shape}")

    # Clean up temp files
    for f in batch_files:
        os.remove(f)
    print(f"  Temp files cleaned up.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 -- CROSS-FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def engineer_cross_features(df):
    print("\n--- Section 5: Cross-Feature Engineering ---")

    # 1. Delinquency escalation (avg trend across all D_ features)
    d_trend = [c for c in df.columns if c.startswith('D_') and c.endswith('_trend')]
    if d_trend:
        df['delinquency_escalation'] = df[d_trend].mean(axis=1)
        print(f"  delinquency_escalation  from {len(d_trend)} D-trend cols")

    # 2. Spend collapse (negative spend trend = stress)
    s_trend = [c for c in df.columns if c.startswith('S_') and c.endswith('_trend')]
    if s_trend:
        df['spend_collapse'] = -df[s_trend].mean(axis=1)
        print(f"  spend_collapse          from {len(s_trend)} S-trend cols")

    # 3. Balance stress (rising balance trend)
    b_trend = [c for c in df.columns if c.startswith('B_') and c.endswith('_trend')]
    if b_trend:
        df['balance_stress'] = df[b_trend].mean(axis=1)
        print(f"  balance_stress          from {len(b_trend)} B-trend cols")

    # 4. Risk composite (avg of latest R_ features)
    r_last = [c for c in df.columns if c.startswith('R_') and c.endswith('_last')]
    if r_last:
        df['risk_composite_last'] = df[r_last].mean(axis=1)
        print(f"  risk_composite_last     from {len(r_last)} R-last cols")

    # 5. Payment-to-balance ratio
    if 'P_2_last' in df.columns and 'B_2_last' in df.columns:
        df['payment_to_balance'] = (
            df['P_2_last'] / (df['B_2_last'].abs() + 1e-8)
        ).clip(-10, 10)
        print(f"  payment_to_balance      (P_2_last / B_2_last)")

    # 6. Delinquency volatility (std of D_ features)
    d_std = [c for c in df.columns if c.startswith('D_') and c.endswith('_std')]
    if d_std:
        df['delinquency_volatility'] = df[d_std].mean(axis=1)
        print(f"  delinquency_volatility  from {len(d_std)} D-std cols")

    # 7. Combined stress score
    stress_cols = [c for c in ['delinquency_escalation','spend_collapse',
                                'balance_stress','risk_composite_last']
                   if c in df.columns]
    if stress_cols:
        df['composite_stress_score'] = df[stress_cols].mean(axis=1)
        print(f"  composite_stress_score  from {len(stress_cols)} stress cols")

    print(f"  Final shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 -- MERGE LABELS & SAVE
# ─────────────────────────────────────────────────────────────────────────────
def merge_and_save(df, label_map):
    print("\n--- Section 6: Merging Labels & Saving ---")
    df['target'] = df['customer_ID'].map(label_map)
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].astype(int)

    print(f"  Final shape  : {df.shape}")
    print(f"  Default rate : {df['target'].mean():.2%}")

    df_save = df.drop(columns=['customer_ID'])
    df_save.to_parquet(OUTPUT_PKL, index=False, compression='snappy')
    print(f"  Saved parquet -> {OUTPUT_PKL}")

    sample = df.sample(50, random_state=42)
    sample.to_csv(OUTPUT_SAMPLE, index=False)
    print(f"  Saved sample  -> {OUTPUT_SAMPLE}")

    return df_save


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    start = datetime.now()
    print("=" * 65)
    print("  HALCYON CREDIT -- AmEx Aggregation Pipeline v2")
    print(f"  Started: {start.strftime('%H:%M:%S')}")
    print("=" * 65)

    selected_ids, label_map = select_customers()
    customer_data, feat_cols = stream_and_accumulate(selected_ids)

    batch_files = aggregate_in_batches(customer_data, feat_cols)
    del customer_data
    gc.collect()

    df = concat_batches(batch_files)
    df = engineer_cross_features(df)
    df_final = merge_and_save(df, label_map)

    elapsed = int((datetime.now() - start).total_seconds() // 60)
    print(f"\n  Total time : ~{elapsed} minutes")
    print(f"  Features   : {df_final.shape[1] - 1:,}")
    print("  Ready for LightGBM training!")
    print("=" * 65)
