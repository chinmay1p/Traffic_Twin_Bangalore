from __future__ import annotations

import os
import sys
import logging
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

DATA_DIR = BASE_DIR / "data"
EVENTS_PATH = DATA_DIR / "astram_mapped_events.csv"
MODELS_DIR = BASE_DIR / "models"

def load_and_preprocess_events() -> tuple[pd.DataFrame, csr_matrix, list[str], dict[str, LabelEncoder], TfidfVectorizer]:
    logging.info("Loading and preprocessing Astram events data...")
    df = pd.read_csv(EVENTS_PATH)
    
    # Fill missing values
    cat_cols = ["event_cause", "road_type", "corridor", "junction", "zone", "police_station", "veh_type"]
    num_cols = ["latitude", "longitude", "capacity", "hour", "day_of_week"]
    
    for col in cat_cols:
        df[col] = df[col].fillna("unknown").astype(str)
    
    # Estimate or parse start hour and day of week if not present
    df["start_datetime"] = pd.to_datetime(df["start_datetime"], format="mixed")
    df["hour"] = df["start_datetime"].dt.hour
    df["day_of_week"] = df["start_datetime"].dt.dayofweek
    df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce").fillna(1800)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce").fillna(12.9716)
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce").fillna(77.5946)
    
    # Label encoding
    encoders = {}
    encoded_cats = []
    for col in cat_cols:
        le = LabelEncoder()
        df[f"{col}_encoded"] = le.fit_transform(df[col])
        encoders[col] = le
        encoded_cats.append(f"{col}_encoded")
        
    # Text processing: description
    df["description"] = df["description"].fillna("").astype(str)
    tfidf = TfidfVectorizer(max_features=500)
    X_text = tfidf.fit_transform(df["description"])
    
    # Save preprocessing assets
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(encoders, MODELS_DIR / "event_encoders.pkl")
    joblib.dump(tfidf, MODELS_DIR / "event_tfidf.pkl")
    logging.info("Saved LabelEncoders and TfidfVectorizer to models/")
    
    # Numeric and Categorical features as sparse matrix
    X_num_cat = csr_matrix(df[num_cols + encoded_cats].values)
    
    # Combine sparse matrices
    X = hstack([X_num_cat, X_text]).tocsr()
    
    feature_names = num_cols + encoded_cats + [f"tfidf_{i}" for i in range(X_text.shape[1])]
    
    return df, X, feature_names, encoders, tfidf

def calculate_targets(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Calculating target metrics for training...")
    
    # 1. priority_score
    df["priority_score"] = df["priority"].str.lower().apply(lambda x: 1.0 if x == "high" else 0.0)
    
    # 2. closure_score
    df["closure_score"] = df["requires_road_closure"].astype(str).str.lower().apply(
        lambda x: 1.0 if x in ("true", "1", "yes") else 0.0
    )
    
    # 3. normalized_duration
    df["end_datetime"] = pd.to_datetime(df["end_datetime"], format="mixed")
    
    # Calculate duration in minutes
    durations = []
    for _, row in df.iterrows():
        start = row["start_datetime"]
        end = row["end_datetime"]
        if pd.isna(end):
            cause = str(row["event_cause"]).lower()
            if "accident" in cause:
                dur = 120.0
            elif "breakdown" in cause:
                dur = 60.0
            elif "water" in cause:
                dur = 180.0
            elif "construction" in cause:
                dur = 1440.0
            else:
                dur = 120.0
        else:
            dur = (end - start).total_seconds() / 60.0
        durations.append(dur)
        
    df["event_duration_minutes"] = durations
    
    # Min-max scaling for duration score
    d_min, d_max = df["event_duration_minutes"].min(), df["event_duration_minutes"].max()
    if d_max > d_min:
        df["normalized_duration"] = (df["event_duration_minutes"] - d_min) / (d_max - d_min)
    else:
        df["normalized_duration"] = 0.0
        
    # 4. hotspot_score
    station_counts = df["police_station"].value_counts()
    max_count = station_counts.max()
    df["hotspot_score"] = df["police_station"].map(lambda x: station_counts.get(x, 0) / max_count)
    
    # Calculate impact_score
    df["impact_score"] = (
        0.4 * df["priority_score"] +
        0.3 * df["closure_score"] +
        0.2 * df["normalized_duration"] +
        0.1 * df["hotspot_score"]
    )
    
    # Convert impact_score to categories: 0=LOW, 1=MEDIUM, 2=HIGH
    df["impact_class"] = pd.cut(
        df["impact_score"],
        bins=[-0.1, 0.3, 0.7, 1.1],
        labels=[0, 1, 2]
    ).astype(int)
    
    return df

def train_models():
    df, X, feature_names, encoders, tfidf = load_and_preprocess_events()
    df = calculate_targets(df)
    
    # Train-test split (80-20 random since events are independent incidents)
    np.random.seed(42)
    indices = np.random.permutation(len(df))
    split_idx = int(len(df) * 0.8)
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    # Save split indices for evaluation consistency
    joblib.dump({"train": train_indices, "test": test_indices}, MODELS_DIR / "event_split.pkl")
    
    # 1. Train Event Impact Classifier (LightGBM)
    logging.info("Training Event Impact Classifier...")
    y_impact = df["impact_class"].values
    
    clf_impact = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=32,
        random_state=42,
        verbose=-1
    )
    clf_impact.fit(X[train_indices], y_impact[train_indices])
    
    preds_impact = clf_impact.predict(X[test_indices])
    print("\n--- Event Impact Model Evaluation ---")
    print(classification_report(y_impact[test_indices], preds_impact, target_names=["LOW", "MEDIUM", "HIGH"]))
    
    joblib.dump(clf_impact, MODELS_DIR / "event_impact_model.pkl")
    joblib.dump(clf_impact, MODELS_DIR / "event_impact.pkl")
    
    # 2. Train Closure Classifier (XGBoost)
    logging.info("Training Closure Predictor...")
    y_closure = df["closure_score"].astype(int).values
    
    # Handle class imbalance
    pos_count = np.sum(y_closure[train_indices] == 1)
    neg_count = np.sum(y_closure[train_indices] == 0)
    scale_pos = neg_count / max(1, pos_count)
    
    clf_closure = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=scale_pos,
        random_state=42,
        eval_metric="logloss"
    )
    clf_closure.fit(X[train_indices], y_closure[train_indices])
    
    preds_closure = clf_closure.predict(X[test_indices])
    print("\n--- Closure Predictor Evaluation ---")
    print(classification_report(y_closure[test_indices], preds_closure, target_names=["NO CLOSURE", "REQUIRES CLOSURE"]))
    
    joblib.dump(clf_closure, MODELS_DIR / "closure_predictor.pkl")
    
    # 3. Train Duration Regressor (XGBoost)
    logging.info("Training Duration Regressor...")
    # Remove outliers: duration > 99th percentile
    dur_limit = np.percentile(df["event_duration_minutes"], 99)
    valid_dur_mask = df["event_duration_minutes"] <= dur_limit
    
    df_dur = df[valid_dur_mask].reset_index(drop=True)
    X_dur = X[valid_dur_mask.values]
    
    indices_dur = np.random.permutation(len(df_dur))
    split_dur = int(len(df_dur) * 0.8)
    
    train_dur = indices_dur[:split_dur]
    test_dur = indices_dur[split_dur:]
    
    y_dur = df_dur["event_duration_minutes"].values
    
    reg_duration = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    reg_duration.fit(X_dur[train_dur], y_dur[train_dur])
    
    preds_dur = reg_duration.predict(X_dur[test_dur])
    mae = mean_absolute_error(y_dur[test_dur], preds_dur)
    r2 = r2_score(y_dur[test_dur], preds_dur)
    
    print("\n--- Duration Regressor Evaluation ---")
    print(f" MAE: {mae:.2f} minutes")
    print(f" R2:  {r2:.4f}")
    
    joblib.dump(reg_duration, MODELS_DIR / "duration_predictor.pkl")
    logging.info("Saved all event-level prediction models successfully.")

if __name__ == "__main__":
    train_models()
