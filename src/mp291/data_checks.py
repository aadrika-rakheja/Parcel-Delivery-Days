"""
Data Checks Module for MP291 - Parcel Delivery Days (Dehradun).
Checks train.csv for the seven kinds of data problems described in the Student Handbook.
"""

import os
import yaml
import numpy as np
import pandas as pd


def check_seven_data_problems(train_path="data/raw/train.csv",
                               later_path="data/raw/later.csv",
                               holdout_path="data/raw/holdout_features.csv"):
    """
    Run diagnostic checks on train.csv across the 7 categories.
    Returns a dictionary matching the data_issues schema in findings.yaml.
    """
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}")

    train_df = pd.read_csv(train_path)
    later_df = pd.read_csv(later_path) if os.path.exists(later_path) else None
    holdout_df = pd.read_csv(holdout_path) if os.path.exists(holdout_path) else None

    findings = {}

    # 1. Leaking column
    leaking_col = None
    if holdout_df is not None:
        missing_in_holdout = [c for c in train_df.columns if c not in holdout_df.columns and c != "delivery_days"]
        for c in missing_in_holdout:
            if c in train_df.select_dtypes(include=[np.number]).columns:
                # Compare correlation with target after ignoring extreme target typos
                clean_target = train_df[train_df["delivery_days"] < 20]
                corr = clean_target[c].corr(clean_target["delivery_days"])
                if corr > 0.95:
                    leaking_col = c
                    break

    findings["leaking_column"] = {
        "present": leaking_col is not None,
        "column": leaking_col
    }

    # 2. Unit mixup
    # Check if any numeric column grouped by destination has a mean ~10x, 100x, 1000x different
    unit_mixup_found = False
    unit_col, unit_group, unit_factor = None, None, None
    if "destination" in train_df.columns:
        for c in train_df.select_dtypes(include=[np.number]).columns:
            if c in ["record_id", "date", "delivery_days", "days_from_tracking_page"]:
                continue
            group_means = train_df.groupby("destination")[c].mean()
            if len(group_means) > 1:
                ratio = group_means.max() / (group_means.min() + 1e-9)
                for f in [10, 12, 60, 100, 1000]:
                    if 0.8 * f <= ratio <= 1.2 * f:
                        unit_mixup_found = True
                        unit_col = c
                        unit_group = group_means.idxmax()
                        unit_factor = f
                        break
    findings["unit_mixup"] = {
        "present": unit_mixup_found,
        "column": unit_col,
        "group_value": unit_group,
        "factor": unit_factor
    }

    # 3. Sentinel value
    sentinel_found = False
    sentinel_col, sentinel_val, sentinel_count = None, None, None
    for c in train_df.select_dtypes(include=[np.number]).columns:
        if c in ["record_id", "delivery_days"]:
            continue
        vc = train_df[c].value_counts()
        for s in [-999, -1, 999, 9999]:
            if s in vc.index and vc[s] > 5:
                sentinel_found = True
                sentinel_col = c
                sentinel_val = int(s)
                sentinel_count = int(vc[s])
                break
        if sentinel_found:
            break

    findings["sentinel_value"] = {
        "present": sentinel_found,
        "column": sentinel_col,
        "value": sentinel_val,
        "count": sentinel_count
    }

    # 4. Duplicate rows
    dup_count = int(train_df.duplicated().sum())
    findings["duplicate_rows"] = {
        "present": dup_count > 0,
        "count": dup_count if dup_count > 0 else None
    }

    # 5. Stuck sensor
    stuck_found = False
    stuck_col, stuck_start, stuck_end = None, None, None
    train_sorted = train_df.sort_values("date").copy()
    for c in train_sorted.select_dtypes(include=[np.number]).columns:
        if c in ["pin_code_tier", "transfers", "delivery_days"]:
            continue
        streak = (train_sorted[c] != train_sorted[c].shift(1)).cumsum()
        streak_lens = train_sorted.groupby(streak).size()
        if (streak_lens > 10).any():
            stuck_found = True
            stuck_col = c
            # find streak
            break

    findings["stuck_sensor"] = {
        "present": stuck_found,
        "column": stuck_col,
        "start_date": stuck_start,
        "end_date": stuck_end
    }

    # 6. Target typos (values typed 10 times too big)
    target_series = train_df["delivery_days"].dropna()
    typos = target_series[target_series > 20]
    typo_count = len(typos)
    findings["target_typos"] = {
        "present": typo_count > 0,
        "count": typo_count if typo_count > 0 else None
    }

    # 7. Spelling variants
    spelling_found = False
    spelling_col, true_levels = None, None
    for c in train_df.select_dtypes(include=["object"]).columns:
        if c in ["record_id", "date"]:
            continue
        train_unique = train_df[c].dropna().unique()
        if later_df is not None and c in later_df.columns:
            later_unique = later_df[c].dropna().unique()
            if len(train_unique) > len(later_unique):
                spelling_found = True
                spelling_col = c
                true_levels = int(len(later_unique))
                break
    findings["spelling_variants"] = {
        "present": spelling_found,
        "column": spelling_col,
        "true_levels": true_levels
    }

    return findings


if __name__ == "__main__":
    results = check_seven_data_problems()
    print("Detected Data Issues:")
    print(yaml.dump({"data_issues": results}, sort_keys=False))
