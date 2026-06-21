from __future__ import annotations

import os
import sys
import logging
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

DATA_DIR = BASE_DIR / "data"
TIMESERIES_PATH = DATA_DIR / "traffic_timeseries.csv"
EDGES_PATH = DATA_DIR / "edges.csv"
EVENTS_PATH = DATA_DIR / "astram_mapped_events.csv"
FEATURES_OUTPUT_PATH = DATA_DIR / "ml_traffic_features.csv"
MODELS_DIR = BASE_DIR / "models"

def prepare_features() -> pd.DataFrame:
    logging.info("Preparing ML Traffic Features...")
    
    # Load core data
    ts_df = pd.read_csv(TIMESERIES_PATH)
    edges_df = pd.read_csv(EDGES_PATH).drop_duplicates(subset=["edge_id"])
    
    # Convert timestamps
    ts_df["timestamp"] = pd.to_datetime(ts_df["timestamp"])
    ts_df = ts_df.sort_values(by=["edge_id", "timestamp"]).reset_index(drop=True)
    
    # Merge road metadata (road_type, capacity, lanes)
    logging.info("Merging road metadata...")
    ts_df = ts_df.merge(edges_df[["edge_id", "road_type", "capacity", "lanes"]], on="edge_id", how="left")
    
    # Fill missing metadata defaults
    ts_df["capacity"] = ts_df["capacity"].fillna(1800)
    ts_df["lanes"] = ts_df["lanes"].fillna(2)
    ts_df["road_type"] = ts_df["road_type"].fillna("tertiary")
    
    # Time Features
    ts_df["hour"] = ts_df["timestamp"].dt.hour
    ts_df["day_of_week"] = ts_df["timestamp"].dt.dayofweek
    ts_df["month"] = ts_df["timestamp"].dt.month
    ts_df["is_weekend"] = (ts_df["day_of_week"] >= 5).astype(int)
    
    # Lags (Treating steps as 15-minute equivalent steps for naming compliance)
    logging.info("Engineering lags and rolling windows...")
    grouped = ts_df.groupby("edge_id")
    
    ts_df["speed_15min_before"] = grouped["speed"].shift(1)
    ts_df["speed_30min_before"] = grouped["speed"].shift(2)
    ts_df["speed_1hr_before"] = grouped["speed"].shift(3)
    
    ts_df["congestion_15min_before"] = grouped["congestion_score"].shift(1)
    ts_df["congestion_30min_before"] = grouped["congestion_score"].shift(2)
    
    # Rolling Averages
    ts_df["avg_speed_last_hour"] = grouped["speed"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    ts_df["avg_congestion_last_hour"] = grouped["congestion_score"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    
    # Graph Feature: neighbor_average_congestion
    logging.info("Engineering neighbor congestion features...")
    # Load neighbor map from traffic_features.csv if it has it, or compute via fast mapping
    try:
        from src.traffic.traffic_features import TrafficFeatureEngineer
        feat_df = pd.read_csv(DATA_DIR / "traffic_features.csv")
        # Map neighbor speed to estimate neighbor congestion
        ts_df = ts_df.merge(feat_df[["timestamp", "edge_id", "neighbor_speed_lag_1"]], on=["timestamp", "edge_id"], how="left")
        # Estimate neighbor congestion from neighbor speed lag
        ts_df["neighbor_average_congestion"] = 1.0 - (ts_df["neighbor_speed_lag_1"] / 50.0)
        ts_df["neighbor_average_congestion"] = ts_df["neighbor_average_congestion"].clip(0, 1)
    except Exception:
        ts_df["neighbor_average_congestion"] = ts_df["congestion_15min_before"]
        
    ts_df["neighbor_average_congestion"] = ts_df["neighbor_average_congestion"].fillna(ts_df["congestion_score"])
    
    # Event Features: nearby_event_count, max_event_impact
    logging.info("Engineering active event features...")
    
    # Check if events exist and map active ones
    if EVENTS_PATH.exists():
        events_df = pd.read_csv(EVENTS_PATH)
        
        # Build active events mapping
        active_list = []
        for _, row in events_df.iterrows():
            eid = row.get("nearest_edge_id")
            if pd.isna(eid):
                continue
            
            # Simple priority and closure parsing
            priority = 1.0 if str(row.get("priority", "")).lower() == "high" else 0.0
            closure = 1.0 if str(row.get("requires_road_closure", "")).lower() in ("true", "1", "yes") else 0.0
            impact = 0.4 * priority + 0.3 * closure + 0.2
            
            start_str = str(row["start_datetime"])
            try:
                start_dt = pd.to_datetime(start_str).tz_localize(None)
            except Exception:
                continue
            
            # Active duration (2 hours default)
            active_list.append({
                "edge_id": str(eid).strip(),
                "hour": start_dt.hour,
                "day_of_week": start_dt.weekday(),
                "impact": impact
            })
            
        active_df = pd.DataFrame(active_list)
        if not active_df.empty:
            # Map events matching hour and day_of_week to simulate local nearby impacts
            grouped_events = active_df.groupby(["edge_id", "hour", "day_of_week"])
            agg_events = grouped_events.agg(
                nearby_event_count=("impact", "count"),
                max_event_impact=("impact", "max")
            ).reset_index()
            
            ts_df = ts_df.merge(agg_events, on=["edge_id", "hour", "day_of_week"], how="left")

    ts_df["nearby_event_count"] = ts_df.get("nearby_event_count", pd.Series(0, index=ts_df.index)).fillna(0).astype(int)
    ts_df["max_event_impact"] = ts_df.get("max_event_impact", pd.Series(0.0, index=ts_df.index)).fillna(0.0)
            
    # Targets (Shifted in reverse)
    logging.info("Creating prediction targets...")
    ts_df["congestion_after_15min"] = grouped["congestion_score"].shift(-1)
    ts_df["congestion_after_30min"] = grouped["congestion_score"].shift(-2)
    ts_df["congestion_after_60min"] = grouped["congestion_score"].shift(-3)
    
    # Drop rows with NaNs in features or targets
    clean_df = ts_df.dropna().copy()
    
    # Label encode road_type
    clean_df["road_type_code"] = clean_df["road_type"].astype("category").cat.codes
    
    clean_df.to_csv(FEATURES_OUTPUT_PATH, index=False)
    logging.info("Saved ML features dataset with %d rows to %s", len(clean_df), FEATURES_OUTPUT_PATH)
    return clean_df

def train_forecasters(df: pd.DataFrame) -> None:
    logging.info("Training LightGBM Traffic Forecasting Models...")
    
    # Feature columns
    feature_cols = [
        "road_type_code", "capacity", "lanes",
        "hour", "day_of_week", "month", "is_weekend",
        "speed", "density", "flow", "congestion_score",
        "speed_15min_before", "speed_30min_before", "speed_1hr_before",
        "congestion_15min_before", "congestion_30min_before",
        "avg_speed_last_hour", "avg_congestion_last_hour",
        "neighbor_average_congestion", "nearby_event_count", "max_event_impact"
    ]
    
    targets = {
        "15min": "congestion_after_15min",
        "30min": "congestion_after_30min",
        "60min": "congestion_after_60min"
    }
    
    # Sequential Split (80% train, 20% test)
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    
    MODELS_DIR.mkdir(exist_ok=True)
    
    trained_models = {}
    
    for name, target_col in targets.items():
        logging.info("Training Forecaster Model: %s", name)
        y_train = train_df[target_col]
        y_test = test_df[target_col]
        
        # Hyperparameters
        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.8,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        
        # Predictions and Evaluation
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        print(f"\n--- Model Evaluation: {name} ---")
        print(f" MAE:  {mae:.4f}")
        print(f" RMSE: {rmse:.4f}")
        print(f" R2:   {r2:.4f}")
        
        # Save model
        model_path = MODELS_DIR / f"traffic_{name}.pkl"
        joblib.dump(model, model_path)
        logging.info("Saved %s model to %s", name, model_path)
        
        trained_models[name] = model
        
    # Also save the unified traffic_forecaster dict
    unified_path = MODELS_DIR / "traffic_forecaster.pkl"
    joblib.dump({
        "models": trained_models,
        "features": feature_cols
    }, unified_path)
    logging.info("Saved unified traffic forecaster dict to %s", unified_path)

if __name__ == "__main__":
    df = prepare_features()
    train_forecasters(df)
