from __future__ import annotations

import os
import sys
import logging
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Global cache for lazy model loading
_MODELS_CACHE = {}

def load_prediction_assets():
    """
    Helper to lazy-load models, encoders, and feature structures.
    """
    if _MODELS_CACHE:
        return _MODELS_CACHE

    logging.info("Loading predictive models and vectorizers from %s...", MODELS_DIR)
    
    try:
        # Event models
        _MODELS_CACHE["event_encoders"] = joblib.load(MODELS_DIR / "event_encoders.pkl")
        _MODELS_CACHE["event_tfidf"] = joblib.load(MODELS_DIR / "event_tfidf.pkl")
        _MODELS_CACHE["event_impact"] = joblib.load(MODELS_DIR / "event_impact_model.pkl")
        _MODELS_CACHE["closure_predictor"] = joblib.load(MODELS_DIR / "closure_predictor.pkl")
        _MODELS_CACHE["duration_predictor"] = joblib.load(MODELS_DIR / "duration_predictor.pkl")
        
        # Traffic models
        _MODELS_CACHE["traffic_15min"] = joblib.load(MODELS_DIR / "traffic_15min.pkl")
        _MODELS_CACHE["traffic_30min"] = joblib.load(MODELS_DIR / "traffic_30min.pkl")
        _MODELS_CACHE["traffic_60min"] = joblib.load(MODELS_DIR / "traffic_60min.pkl")
        _MODELS_CACHE["is_fallback"] = False
    except Exception as e:
        logging.error("Failed to load ML models, entering simulation fallback: %s", e)
        _MODELS_CACHE["is_fallback"] = True
    
    # Load ML feature columns reference
    try:
        forecaster_meta = joblib.load(MODELS_DIR / "traffic_forecaster.pkl")
        _MODELS_CACHE["traffic_features"] = forecaster_meta.get("features")
    except Exception:
        _MODELS_CACHE["traffic_features"] = [
            "road_type_code", "capacity", "lanes",
            "hour", "day_of_week", "month", "is_weekend",
            "speed", "density", "flow", "congestion_score",
            "speed_15min_before", "speed_30min_before", "speed_1hr_before",
            "congestion_15min_before", "congestion_30min_before",
            "avg_speed_last_hour", "avg_congestion_last_hour",
            "neighbor_average_congestion", "nearby_event_count", "max_event_impact"
        ]
        
    return _MODELS_CACHE

def predict_event_effect(event: dict) -> dict:
    """
    Predicts impact score, closure probability, expected duration, and affected radius for an incident.
    """
    assets = load_prediction_assets()
    
    event_cause = str(event.get("event_cause", "vehicle_breakdown"))
    if assets.get("is_fallback"):
        # Return fallback mock predictions
        duration_min = 60
        if "accident" in event_cause:
            impact_label = "HIGH"
            impact_score = 0.8
            closure_prob = 0.6
            duration_min = 90
        elif "breakdown" in event_cause:
            impact_label = "MEDIUM"
            impact_score = 0.5
            closure_prob = 0.4
            duration_min = 45
        else:
            impact_label = "LOW"
            impact_score = 0.3
            closure_prob = 0.1
            
        return {
            "impact": impact_label,
            "impact_score": impact_score,
            "closure_probability": closure_prob,
            "expected_duration": duration_min,
            "duration": f"{duration_min} minutes",
            "affected_radius": "1.5km",
            "using_fallback": True
        }
    
    # Parse event time
    time_str = str(event.get("time", "8:00")).lower()
    hour = 8 # default
    try:
        if "am" in time_str:
            hour = int(time_str.replace("am", "").split(":")[0])
        elif "pm" in time_str:
            hour = int(time_str.replace("pm", "").split(":")[0]) + 12
        else:
            hour = int(time_str.split(":")[0])
    except Exception:
        hour = 8
        
    # Check day_of_week
    day_of_week = 0 # Monday default
    
    # Map Silk Board or custom location to typical high-congestion coordinate attributes
    location = str(event.get("location", "")).lower()
    if "silk board" in location:
        latitude = 12.9176
        longitude = 77.6244
        capacity = 2400
        road_type = "trunk"
        corridor = "Hosur Road"
        junction = "Silk Board Flyover"
        zone = "South"
        police_station = "Madiwala"
    else:
        latitude = float(event.get("latitude", 12.9716))
        longitude = float(event.get("longitude", 77.5946))
        capacity = float(event.get("capacity", 1800))
        road_type = str(event.get("road_type", "primary"))
        corridor = str(event.get("corridor", "unknown"))
        junction = str(event.get("junction", "unknown"))
        zone = str(event.get("zone", "unknown"))
        police_station = str(event.get("police_station", "unknown"))
        
    event_cause = str(event.get("event_cause", "vehicle_breakdown"))
    vehicle = str(event.get("vehicle", "truck"))
    veh_type = "HGV" if "truck" in vehicle or "bus" in vehicle else "four_wheeler"
    
    description = str(event.get("description", f"A {vehicle} {event_cause.replace('_', ' ')} occurred at {location}"))
    
    # Encode categorical columns
    cat_cols = ["event_cause", "road_type", "corridor", "junction", "zone", "police_station", "veh_type"]
    encoded_vals = {}
    encoders = assets["event_encoders"]
    
    vals_to_encode = {
        "event_cause": event_cause,
        "road_type": road_type,
        "corridor": corridor,
        "junction": junction,
        "zone": zone,
        "police_station": police_station,
        "veh_type": veh_type
    }
    
    encoded_cats_list = []
    for col in cat_cols:
        val = vals_to_encode[col]
        le = encoders[col]
        if val in le.classes_:
            encoded_val = le.transform([val])[0]
        elif "unknown" in le.classes_:
            encoded_val = le.transform(["unknown"])[0]
        else:
            encoded_val = 0
        encoded_vals[f"{col}_encoded"] = encoded_val
        encoded_cats_list.append(encoded_val)
        
    # Build text features
    tfidf = assets["event_tfidf"]
    X_text = tfidf.transform([description])
    
    # Combine features: order of numeric columns was:
    # ["latitude", "longitude", "capacity", "hour", "day_of_week"] followed by encoded cats
    num_vals = [latitude, longitude, capacity, hour, day_of_week]
    X_num_cat = csr_matrix([num_vals + encoded_cats_list])
    
    X = hstack([X_num_cat, X_text]).tocsr()
    
    # Run predictions
    clf_impact = assets["event_impact"]
    clf_closure = assets["closure_predictor"]
    reg_duration = assets["duration_predictor"]
    
    # 1. Impact Class and Impact Score
    probs = clf_impact.predict_proba(X)[0]
    impact_class = int(np.argmax(probs))
    impact_label = ["LOW", "MEDIUM", "HIGH"][impact_class]
    
    # Synthesize impact score based on probability weighting
    impact_score = float(np.dot(probs, [0.15, 0.5, 0.85]))
    
    # 2. Closure Probability
    closure_prob = float(clf_closure.predict_proba(X)[0][1])
    
    # 3. Expected Duration in minutes
    expected_duration = int(np.clip(reg_duration.predict(X)[0], 15, 1440))
    
    # 4. Affected radius
    affected_radius = float(0.5 + 2.0 * impact_score)
    
    return {
        "impact": impact_label,
        "impact_score": round(impact_score, 2),
        "closure_probability": round(closure_prob, 2),
        "expected_duration": expected_duration,
        "duration": f"{expected_duration} minutes",
        "affected_radius": f"{affected_radius:.1f}km"
    }

def predict_future_traffic(edge_id: str, time=None) -> dict:
    """
    Predicts the future congestion score for a road edge at multiple temporal horizons.
    """
    assets = load_prediction_assets()
    
    if assets.get("is_fallback"):
        return {
            "current": 0.35,
            "now": 0.35,
            "15_min": 0.45,
            "15min": 0.45,
            "30_min": 0.6,
            "30min": 0.6,
            "60_min": 0.5,
            "60min": 0.5,
            "using_fallback": True
        }
    
    # Load ML features file to find matching state
    features_path = DATA_DIR / "ml_traffic_features.csv"
    if not features_path.exists():
        features_path = DATA_DIR / "traffic_features.csv"
        
    df = pd.read_csv(features_path)
    
    # Search for edge state
    edge_state = df[df["edge_id"] == edge_id]
    if edge_state.empty:
        # Fallback to general mean state
        logging.warning("Edge %s not found in ML features database. Using average network state.", edge_id)
        numeric_cols = df.select_dtypes(include=np.number).columns
        mean_vals = df[numeric_cols].mean()
        state_row = mean_vals.to_dict()
        state_row["road_type_code"] = 0
    else:
        state_row = edge_state.iloc[-1].to_dict()
        
    # Prepare features vector matching the exact feature column list
    feature_cols = assets["traffic_features"]
    
    X_df = pd.DataFrame([state_row])[feature_cols]
    
    # Predict future congestion levels
    m15 = assets["traffic_15min"]
    m30 = assets["traffic_30min"]
    m60 = assets["traffic_60min"]
    
    pred_15 = float(np.clip(m15.predict(X_df)[0], 0.0, 1.0))
    pred_30 = float(np.clip(m30.predict(X_df)[0], 0.0, 1.0))
    pred_60 = float(np.clip(m60.predict(X_df)[0], 0.0, 1.0))
    
    current_val = float(state_row.get("congestion_score", 0.35))
    
    return {
        "current": round(current_val, 2),
        "now": round(current_val, 2),
        "15_min": round(pred_15, 2),
        "15min": round(pred_15, 2),
        "30_min": round(pred_30, 2),
        "30min": round(pred_30, 2),
        "60_min": round(pred_60, 2),
        "60min": round(pred_60, 2)
    }

if __name__ == "__main__":
    # Test execution
    res = predict_event_effect({
        "event_cause": "vehicle_breakdown",
        "location": "Silk Board",
        "time": "8:30"
    })
    print("Sample Event Prediction Output:")
    print(res)
    
    # Try with first edge ID
    features_path = DATA_DIR / "ml_traffic_features.csv"
    if features_path.exists():
        df = pd.read_csv(features_path)
        first_edge = df["edge_id"].iloc[0]
        print(f"\nSample Traffic Prediction for edge {first_edge}:")
        print(predict_future_traffic(first_edge))
