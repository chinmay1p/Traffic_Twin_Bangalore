from __future__ import annotations

import os
import sys
import logging
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
import shap

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

MODELS_DIR = BASE_DIR / "models"

def explain_event_impact(event: dict) -> str:
    """
    Computes SHAP feature contributions for the given event and formats an explanation.
    """
    # Lazy load prediction assets
    from src.models.predict import load_prediction_assets
    assets = load_prediction_assets()
    
    # Parse event details
    time_str = str(event.get("time", "8:30")).lower()
    hour = 8
    try:
        if "am" in time_str:
            hour = int(time_str.replace("am", "").split(":")[0])
        elif "pm" in time_str:
            hour = int(time_str.replace("pm", "").split(":")[0]) + 12
        else:
            hour = int(time_str.split(":")[0])
    except Exception:
        hour = 8
        
    day_of_week = 0 # Monday
    
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
    
    # Encode categoricals
    cat_cols = ["event_cause", "road_type", "corridor", "junction", "zone", "police_station", "veh_type"]
    encoded_cats_list = []
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
    
    for col in cat_cols:
        val = vals_to_encode[col]
        le = encoders[col]
        if val in le.classes_:
            encoded_val = le.transform([val])[0]
        elif "unknown" in le.classes_:
            encoded_val = le.transform(["unknown"])[0]
        else:
            encoded_val = 0
        encoded_cats_list.append(encoded_val)
        
    # Vectorize text
    tfidf = assets["event_tfidf"]
    X_text = tfidf.transform([description])
    
    # Combine feature matrices
    num_vals = [latitude, longitude, capacity, hour, day_of_week]
    X_num_cat = csr_matrix([num_vals + encoded_cats_list])
    X = hstack([X_num_cat, X_text]).tocsr()
    
    # Feature names
    num_names = ["latitude", "longitude", "capacity", "hour", "day_of_week"]
    cat_names = cat_cols
    tfidf_names = [f"text_token_{w}" for w in tfidf.get_feature_names_out()]
    feature_names = num_names + cat_names + tfidf_names
    
    # Load model and run explainer
    model = assets["event_impact"]
    
    # TreeExplainer
    # Use dense format for shap computation consistency
    X_dense = X.toarray()
    explainer = shap.TreeExplainer(model)
    
    # Compute shap values
    shap_values = explainer.shap_values(X_dense)
    
    # Predict probabilities and class
    probs = model.predict_proba(X_dense)[0]
    pred_class = int(np.argmax(probs))
    class_label = ["LOW", "MEDIUM", "HIGH"][pred_class]
    
    # For LightGBM multi-class, shap_values is a list of arrays (one per class), or a 3D array
    # Let's extract the SHAP values for the predicted class
    if isinstance(shap_values, list):
        class_shap = shap_values[pred_class][0]
    else:
        # Check shape: (n_samples, n_features, n_classes) or similar
        if len(shap_values.shape) == 3:
            class_shap = shap_values[0, :, pred_class]
        else:
            class_shap = shap_values[0]
            
    # Pair feature name with SHAP value
    contributions = list(zip(feature_names, class_shap))
    
    # Sort contributions by magnitude of positive impact
    positive_contribs = sorted([c for c in contributions if c[1] > 0.001], key=lambda x: x[1], reverse=True)
    
    # Format the explanation text
    explanation_lines = [f"{class_label} IMPACT because:"]
    
    # Map raw features to human-friendly terms
    mapped_explanations = []
    
    # 1. Check if Silk Board or high capacity corridor was key
    for name, val in positive_contribs:
        if name == "capacity" and capacity >= 2000:
            mapped_explanations.append("+ heavy congestion bottleneck (high capacity junction)")
        elif name == "hour" and (7 <= hour <= 10 or 17 <= hour <= 20):
            mapped_explanations.append("+ peak traffic hour")
        elif name == "road_type" and road_type in ("trunk", "motorway"):
            mapped_explanations.append(f"+ major arterial highway ({road_type} road)")
        elif name == "corridor" and corridor != "unknown":
            mapped_explanations.append(f"+ active transport corridor ({corridor})")
        elif name == "event_cause" and "breakdown" in event_cause:
            mapped_explanations.append("+ vehicle breakdown blocking lanes")
        elif name == "event_cause" and "accident" in event_cause:
            mapped_explanations.append("+ collision incident blocking flow")
        elif name == "veh_type" and veh_type == "HGV":
            mapped_explanations.append("+ heavy commercial vehicle (truck/bus) involved")
            
    # Check text tokens from tfidf
    text_tokens = [name for name, val in positive_contribs if name.startswith("text_token_")]
    if text_tokens:
        words = [t.replace("text_token_", "") for t in text_tokens[:2]]
        mapped_explanations.append(f"+ keywords detected in description: '{', '.join(words)}'")
        
    # Default fallback rules if no specific rules matched
    if not mapped_explanations:
        for name, val in positive_contribs[:4]:
            mapped_explanations.append(f"+ feature '{name}' contributed positively to prediction")
            
    # Always check road closure likelihood as part of the Unified API or impact
    closure_prob = float(assets["closure_predictor"].predict_proba(X)[0][1])
    if closure_prob > 0.5:
        mapped_explanations.insert(0, "+ road closure likely")
        
    explanation_lines.extend(mapped_explanations[:5])
    raw_str = "\n".join(explanation_lines)
    return raw_str.encode("ascii", "ignore").decode("ascii")

if __name__ == "__main__":
    test_event = {
        "event_cause": "vehicle_breakdown",
        "location": "Silk Board",
        "time": "8:30 AM",
        "vehicle": "truck"
    }
    print("SHAP Model Explanation Example:\n")
    print(explain_event_impact(test_event))
