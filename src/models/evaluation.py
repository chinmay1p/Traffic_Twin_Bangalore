from __future__ import annotations

import os
import sys
import logging
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

def evaluate_traffic_models():
    print("=" * 60)
    print("      TRAFFIC FORECASTING MODEL EVALUATION REPORT")
    print("=" * 60)
    
    features_path = DATA_DIR / "ml_traffic_features.csv"
    if not features_path.exists():
        logging.error("ML features file not found. Run train_traffic_model.py first.")
        return
        
    df = pd.read_csv(features_path)
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    
    # Sequential Split (80-20)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:]
    
    feature_cols = [
        "road_type_code", "capacity", "lanes",
        "hour", "day_of_week", "month", "is_weekend",
        "speed", "density", "flow", "congestion_score",
        "speed_15min_before", "speed_30min_before", "speed_1hr_before",
        "congestion_15min_before", "congestion_30min_before",
        "avg_speed_last_hour", "avg_congestion_last_hour",
        "neighbor_average_congestion", "nearby_event_count", "max_event_impact"
    ]
    
    X_test = test_df[feature_cols]
    
    targets = {
        "15min": "congestion_after_15min",
        "30min": "congestion_after_30min",
        "60min": "congestion_after_60min"
    }
    
    for name, target_col in targets.items():
        model_path = MODELS_DIR / f"traffic_{name}.pkl"
        if not model_path.exists():
            logging.warning("Model file %s not found. Skipping.", model_path)
            continue
            
        model = joblib.load(model_path)
        y_test = test_df[target_col]
        preds = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        print(f"\nModel: {name} (Target: {target_col})")
        print(f"  Mean Absolute Error (MAE) : {mae:.4f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"  R2 Coefficient of Det. (R2): {r2:.4f}")

def evaluate_event_models():
    print("\n" + "=" * 60)
    print("      EVENT-LEVEL ML MODELS EVALUATION REPORT")
    print("=" * 60)
    
    split_path = MODELS_DIR / "event_split.pkl"
    if not split_path.exists():
        logging.error("Event split index file not found. Run train_event_models.py first.")
        return
        
    split_indices = joblib.load(split_path)
    test_indices = split_indices["test"]
    
    # Reload processed data and target vectors
    from src.models.train_event_models import load_and_preprocess_events, calculate_targets
    
    df, X, _, _, _ = load_and_preprocess_events()
    df = calculate_targets(df)
    
    X_test = X[test_indices]
    
    # 1. Event Impact Model
    impact_model_path = MODELS_DIR / "event_impact_model.pkl"
    if impact_model_path.exists():
        model = joblib.load(impact_model_path)
        y_test = df["impact_class"].values[test_indices]
        preds = model.predict(X_test)
        
        print("\n1. Event Impact Classifier Report:")
        print(classification_report(y_test, preds, target_names=["LOW", "MEDIUM", "HIGH"]))
        
    # 2. Closure Predictor
    closure_model_path = MODELS_DIR / "closure_predictor.pkl"
    if closure_model_path.exists():
        model = joblib.load(closure_model_path)
        y_test = df["closure_score"].astype(int).values[test_indices]
        preds = model.predict(X_test)
        
        print("2. Road Closure Predictor Report:")
        print(classification_report(y_test, preds, target_names=["NO CLOSURE", "REQUIRES CLOSURE"]))
        
    # 3. Duration Predictor
    duration_model_path = MODELS_DIR / "duration_predictor.pkl"
    if duration_model_path.exists():
        model = joblib.load(duration_model_path)
        
        # Filter outliers
        dur_limit = np.percentile(df["event_duration_minutes"], 99)
        valid_dur_mask = df["event_duration_minutes"] <= dur_limit
        
        df_dur = df[valid_dur_mask].reset_index(drop=True)
        X_dur = X[valid_dur_mask.values]
        
        # Re-apply random split logic to match training set split
        np.random.seed(42)
        indices_dur = np.random.permutation(len(df_dur))
        split_dur = int(len(df_dur) * 0.8)
        test_dur = indices_dur[split_dur:]
        
        y_test = df_dur["event_duration_minutes"].values[test_dur]
        preds = model.predict(X_dur[test_dur])
        
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        print("3. Event Duration Regressor Metrics:")
        print(f"  Mean Absolute Error (MAE) : {mae:.2f} minutes")
        print(f"  R2 Coefficient of Det. (R2): {r2:.4f}")
        
    print("\n" + "=" * 60)

if __name__ == "__main__":
    evaluate_traffic_models()
    evaluate_event_models()
