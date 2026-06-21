from __future__ import annotations

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.stgnn.model import STGNN

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "stgnn_model.pt"
ADJACENCY_PATH = DATA_DIR / "adjacency_matrix.npy"
SEQUENCES_PATH = DATA_DIR / "temporal_sequences.npy"
EDGES_PATH = DATA_DIR / "edges.csv"

def load_prediction_assets():
    """
    Loads GNN model, edge mapping, and adjacency matrix.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train GNN first.")
    if not ADJACENCY_PATH.exists():
        raise FileNotFoundError(f"Adjacency matrix not found at {ADJACENCY_PATH}.")
        
    # Load model checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=False)
    
    num_nodes = checkpoint["num_nodes"]
    in_features = checkpoint["in_features"]
    
    model = STGNN(num_nodes=num_nodes, in_features=in_features)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Load edge mapping
    unique_edges = joblib.load(DATA_DIR / "gnn_edge_mapping.pkl")
    
    # Load adjacency
    adj_matrix = np.load(ADJACENCY_PATH)
    edge_index = torch.tensor(np.argwhere(adj_matrix == 1.0).T, dtype=torch.long)
    
    return model, unique_edges, edge_index

def simulate_event_spread(event_edge_id: str, impact_score: float) -> list[dict]:
    """
    Runs autoregressive traffic propagation simulation starting with an accident impact.
    
    Returns:
        A list of dictionaries containing congestion metrics over time for each road segment.
    """
    logging.info("Running event propagation simulation for %s with impact %.2f", event_edge_id, impact_score)
    
    model, unique_edges, edge_index = load_prediction_assets()
    edge_to_idx = {eid: idx for idx, eid in enumerate(unique_edges)}
    
    if event_edge_id not in edge_to_idx:
        logging.warning("Edge %s is not in GNN mapped roads. Defaulting to first mapped edge.", event_edge_id)
        event_edge_id = unique_edges[0]
        
    event_idx = edge_to_idx[event_edge_id]
    
    # Load sequences to get initial baseline state (use the most recent sequence)
    data = np.load(SEQUENCES_PATH, allow_pickle=True).item()
    X = data["X"]
    
    # Take the last sample as our baseline starting window
    X_init = torch.tensor(X[-1:], dtype=torch.float32) # Shape: (1, 4, 100, 5)
    
    # Extract baseline/current congestion for return output
    current_congestion = X_init[0, -1, :, 3].clone().numpy() # 3 is congestion_score index
    
    # Inject accident impact at the most recent history timestep (t)
    X_init[0, -1, event_idx, 3] = np.clip(X_init[0, -1, event_idx, 3].item() + impact_score, 0.0, 1.0)
    X_init[0, -1, event_idx, 4] = impact_score # 4 is event_impact index
    
    # Autoregressive rollout: T+15, T+30, T+45, T+60
    predictions = {
        "15min": None,
        "30min": None,
        "45min": None,
        "60min": None
    }
    
    steps = ["15min", "30min", "45min", "60min"]
    
    current_state = X_init.clone()
    
    with torch.no_grad():
        for step in steps:
            # Predict next congestion scores
            pred = model(current_state, edge_index) # Shape: (1, 100)
            pred = torch.clamp(pred, 0.0, 1.0)
            
            predictions[step] = pred[0].numpy()
            
            # Construct the next state input window (autoregressive shift)
            next_state = torch.zeros_like(current_state)
            next_state[0, 0:3] = current_state[0, 1:4] # Shift oldest history out
            
            # Populate the latest step (step 3)
            # We copy baseline values from step 3 and overlay predicted congestion and dependencies
            next_state[0, 3] = current_state[0, 3].clone()
            
            # Update congestion score feature (index 3)
            next_state[0, 3, :, 3] = pred[0]
            
            # Update event impact: let it decay slightly or persist
            next_state[0, 3, :, 4] = current_state[0, 3, :, 4] * 0.95
            
            # Map standard traffic equations to update speed (idx 0), density (idx 1), and flow (idx 2)
            # Speed drops as congestion increases
            next_state[0, 3, :, 0] = torch.clamp(1.0 - pred[0], 0.0, 1.0)
            # Density rises with congestion
            next_state[0, 3, :, 1] = pred[0]
            # Parabolic flow relationship
            next_state[0, 3, :, 2] = torch.clamp(pred[0] * (1.0 - pred[0]) * 4.0, 0.0, 1.0)
            
            current_state = next_state
            
    # Format return list
    output_list = []
    for idx, eid in enumerate(unique_edges):
        output_list.append({
            "edge_id": eid,
            "current": float(current_congestion[idx]),
            "15min": float(predictions["15min"][idx]),
            "30min": float(predictions["30min"][idx]),
            "45min": float(predictions["45min"][idx]),
            "60min": float(predictions["60min"][idx])
        })
        
    return output_list

def get_affected_roads(event_edge_id: str, impact_score: float, threshold: float = 0.05) -> dict:
    """
    Simulates event spread and identifies road names that experience significant congestion increase.
    """
    simulation_results = simulate_event_spread(event_edge_id, impact_score)
    
    # Load road names mapping from edges.csv
    road_names = {}
    if EDGES_PATH.exists():
        edges_df = pd.read_csv(EDGES_PATH)
        for _, row in edges_df.iterrows():
            eid = row["edge_id"]
            name = str(row.get("road_name", "Unknown"))
            if name and name != "nan" and name != "Unknown":
                road_names[eid] = name
                
    # Find names of the target event road
    main_road_name = road_names.get(event_edge_id, "Silk Board Area")
    
    affected_list = []
    for row in simulation_results:
        eid = row["edge_id"]
        if eid == event_edge_id:
            continue
            
        current = row["current"]
        future = row["60min"] # Measure long-term propagation at 60 mins
        
        increase = future - current
        if increase > threshold:
            name = road_names.get(eid, f"Unidentified Road (Segment {eid[:8]})")
            # Calculate absolute percentage points increase
            impact_pct = round(increase * 100, 1)
            affected_list.append({
                "road_name": name,
                "impact_pct": impact_pct
            })
            
    # Group by road name to avoid segment duplicates and take the max impact
    grouped_affected = {}
    for aff in affected_list:
        name = aff["road_name"]
        pct = aff["impact_pct"]
        if name not in grouped_affected or pct > grouped_affected[name]:
            grouped_affected[name] = pct
            
    sorted_affected = [{"road_name": k, "impact_pct": v} for k, v in sorted(grouped_affected.items(), key=lambda x: x[1], reverse=True)]
    
    return {
        "main_incident": main_road_name,
        "affected": sorted_affected
    }

if __name__ == "__main__":
    # Self-test if run directly
    try:
        res = get_affected_roads("32261256___", 0.8)
        print("\n=== ST-GNN Propagation Simulation Result ===")
        print(f"Main Incident location: {res['main_incident']}")
        print("Affected Roads:")
        for r in res["affected"][:5]:
            print(f" - {r['road_name']}: +{r['impact_pct']}% congestion")
    except Exception as e:
        print("Model assets not loaded yet or test failed:", e)
