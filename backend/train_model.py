"""
Machine Learning Training Pipeline
----------------------------------
Trains a Random Forest Classifier for Fake Social Media Account Detection.
Updated to use the simplified, user-friendly 8-feature schema:
  - followers_count
  - following_count
  - posts_count
  - account_age_bracket (0: <1 mo, 1: 1-6 mo, 2: 6-12 mo, 3: 1-2 yr, 4: >2 yr)
  - has_bio (0 or 1)
  - has_external_link (0 or 1)
  - has_profile_pic (0 or 1)
  - is_private (0 or 1)
Evaluates the model using Accuracy, Precision, Recall, F1-Score, ROC-AUC,
and a Confusion Matrix.
Saves:
  - models/fake_account_model.pkl
  - models/model_metrics.json
"""

import os
import sys
import csv
import json
import argparse
from datetime import datetime, timezone
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

RAW_FEATURE_NAMES = [
    "followers_count",
    "following_count",
    "posts_count",
    "account_age_bracket",
    "has_bio",
    "has_external_link",
    "has_profile_pic",
    "is_private"
]

ENGINEERED_FEATURE_NAMES = [
    "follower_following_ratio",
    "follow_asymmetry",
    "post_rate_per_age"
]

FEATURE_NAMES = RAW_FEATURE_NAMES + ENGINEERED_FEATURE_NAMES
TARGET_NAME = "is_fake"

# Flexible column aliases for mapping real-world datasets
COLUMN_ALIASES = {
    "followers_count": ["followers", "followers_count", "nb_followers", "follower_count", "n_followers"],
    "following_count": ["following", "following_count", "nb_following", "friends_count", "followees", "n_following"],
    "posts_count": ["posts", "posts_count", "media_count", "statuses_count", "nb_posts", "num_posts"],
    "account_age_bracket": ["account_age_bracket", "age_bracket", "account_age_group", "age_group"],
    "has_bio": ["has_bio", "has_description", "is_bio", "biography_present"],
    "has_external_link": ["has_external_link", "has_link", "external_url_present", "external_link"],
    "has_profile_pic": ["has_profile_pic", "profile_pic", "has_avatar", "default_profile_image"],
    "is_private": ["is_private", "private", "account_type_private", "is_protected"],
    "follower_following_ratio": ["follower_following_ratio", "follower_ratio", "follow_ratio"],
    "follow_asymmetry": ["follow_asymmetry", "asymmetry"],
    "post_rate_per_age": ["post_rate_per_age", "post_rate"],
    "is_fake": ["is_fake", "fake", "is_bot", "label", "class", "target", "spammer"]
}

def resolve_dataset_path(custom_path: str = None) -> tuple[str, str]:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, "dataset")
    
    if custom_path and os.path.exists(custom_path):
        return custom_path, "user_provided_dataset"
        
    real_csv = os.path.join(dataset_dir, "real_accounts_dataset.csv")
    if os.path.exists(real_csv):
        return real_csv, "real_world_dataset"
        
    synth_csv = os.path.join(dataset_dir, "synthetic_accounts_dataset.csv")
    if os.path.exists(synth_csv):
        return synth_csv, "synthetic_demonstration_dataset"
        
    # Auto-generate if missing
    print("[!] No dataset file found. Generating synthetic demonstration dataset...")
    sys.path.append(dataset_dir)
    from generate_dataset import generate_synthetic_data, save_to_csv
    data_mat = generate_synthetic_data(num_samples=3000, random_state=42)
    save_to_csv(data_mat, synth_csv)
    return synth_csv, "synthetic_demonstration_dataset"

def load_and_preprocess_dataset(csv_path: str):
    """
    Loads dataset using CSV module, maps column aliases, and converts to float arrays.
    """
    with open(csv_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        headers = [h.strip() for h in next(reader)]
        
        header_lower = [h.lower() for h in headers]
        col_indices = {}
        
        for std_col, aliases in COLUMN_ALIASES.items():
            matched_idx = -1
            for alias in aliases:
                if alias.lower() in header_lower:
                    matched_idx = header_lower.index(alias.lower())
                    break
            if matched_idx == -1 and std_col in headers:
                matched_idx = headers.index(std_col)
            col_indices[std_col] = matched_idx

        # Check raw features first
        missing_raw = [col for col in RAW_FEATURE_NAMES if col_indices.get(col, -1) == -1]
        if missing_raw:
            raise ValueError(f"Dataset is missing required columns or aliases for: {missing_raw}. Available headers: {headers}")

        X_rows = []
        y_rows = []
        skipped = 0
        
        for row in reader:
            if not row or len(row) < len(headers):
                continue
            try:
                followers = float(row[col_indices["followers_count"]])
                following = float(row[col_indices["following_count"]])
                posts = float(row[col_indices["posts_count"]])
                age_bracket = float(row[col_indices["account_age_bracket"]])
                has_bio = float(row[col_indices["has_bio"]])
                has_link = float(row[col_indices["has_external_link"]])
                has_pic = float(row[col_indices["has_profile_pic"]])
                is_private = float(row[col_indices["is_private"]])

                # Engineered features (read if in file, else compute dynamically)
                if col_indices.get("follower_following_ratio", -1) != -1:
                    ratio = float(row[col_indices["follower_following_ratio"]])
                else:
                    ratio = followers / (following + 1.0)

                if col_indices.get("follow_asymmetry", -1) != -1:
                    asym = float(row[col_indices["follow_asymmetry"]])
                else:
                    asym = (following - followers) / (following + followers + 1.0)

                if col_indices.get("post_rate_per_age", -1) != -1:
                    rate = float(row[col_indices["post_rate_per_age"]])
                else:
                    rate = posts / (age_bracket + 1.0)

                features = [
                    followers, following, posts, age_bracket,
                    has_bio, has_link, has_pic, is_private,
                    ratio, asym, rate
                ]
                label = int(float(row[col_indices[TARGET_NAME]]))
                X_rows.append(features)
                y_rows.append(label)
            except (ValueError, TypeError):
                skipped += 1
                continue
                
    if skipped > 0:
        print(f"[!] Warning: Skipped {skipped} rows due to formatting/missing values.")

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int32)
    
    return X, y

def train_and_evaluate(dataset_path: str, dataset_type: str):
    print("=" * 60)
    print("  AI-BASED FAKE SOCIAL MEDIA ACCOUNT DETECTION - MODEL TRAINER")
    print("=" * 60)
    print(f"[*] Dataset: {dataset_path}")
    print(f"[*] Dataset Type: {dataset_type}")
    
    X, y = load_and_preprocess_dataset(dataset_path)
    print(f"[*] Total valid samples loaded: {len(X)}")
    print(f"[*] Class distribution: Genuine (0) = {np.sum(y == 0)}, Suspicious (1) = {np.sum(y == 1)}")
    
    # 80/20 train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[*] Training samples: {len(X_train)} | Test samples: {len(X_test)}")
    
    # Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    print("[*] Training Random Forest Classifier on simplified features...")
    rf_model.fit(X_train, y_train)
    
    # Predictions & probabilities on test set
    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)[:, 1]
    
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_test, y_proba))
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    # Feature importances
    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    feature_imp_list = [
        {"feature": FEATURE_NAMES[i], "importance": round(float(importances[i]), 4)}
        for i in sorted_idx
    ]
    
    print("\n" + "-" * 50)
    print("  MODEL EVALUATION RESULTS (Test Set)")
    print("-" * 50)
    print(f"  Accuracy:         {acc * 100:.2f}%")
    print(f"  Precision:        {prec * 100:.2f}%")
    print(f"  Recall:           {rec * 100:.2f}%")
    print(f"  F1-Score:         {f1 * 100:.2f}%")
    print(f"  ROC-AUC:          {auc * 100:.2f}%")
    print(f"  Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")
    print("-" * 50)
    print("\nTop Feature Importances:")
    for item in feature_imp_list:
        print(f"  - {item['feature']:<22}: {item['importance'] * 100:.2f}%")
        
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Genuine", "Suspicious"]))
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_save_path = os.path.join(models_dir, "fake_account_model.pkl")
    metrics_save_path = os.path.join(models_dir, "model_metrics.json")
    
    model_payload = {
        "model": rf_model,
        "feature_names": FEATURE_NAMES,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_type": dataset_type,
        "total_samples": len(X),
        "test_accuracy": acc
    }
    
    joblib.dump(model_payload, model_save_path)
    print(f"\n[+] Model artifact saved to: {model_save_path}")
    
    metrics_data = {
        "algorithm": "Random Forest Classifier",
        "dataset_type": dataset_type,
        "dataset_path": os.path.basename(dataset_path),
        "total_samples": len(X),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": {
            "true_negative": cm[0][0],
            "false_positive": cm[0][1],
            "false_negative": cm[1][0],
            "true_positive": cm[1][1]
        },
        "feature_importances": feature_imp_list,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "is_demonstration": (dataset_type == "synthetic_demonstration_dataset")
    }
    
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"[+] Metrics JSON saved to: {metrics_save_path}")
    print("=" * 60)
    
    return rf_model, metrics_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fake Social Media Account Detection Model")
    parser.add_argument("--dataset", type=str, default=None, help="Path to custom CSV dataset")
    args = parser.parse_args()
    
    dataset_file, dataset_kind = resolve_dataset_path(args.dataset)
    train_and_evaluate(dataset_file, dataset_kind)
