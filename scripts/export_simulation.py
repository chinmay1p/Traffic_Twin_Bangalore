from __future__ import annotations

import os
import sys
import json
import logging
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.stgnn.predict_stgnn import simulate_event_spread

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

DATA_DIR = BASE_DIR / "data"
EDGES_PATH = DATA_DIR / "edges.csv"
OUTPUT_PATH = DATA_DIR / "simulation_output.json"

def export_simulation():
    logging.info("Starting simulation data export for visualization...")
    
    # 1. Load road geometries and details
    if not EDGES_PATH.exists():
        logging.error("edges.csv file not found at %s", EDGES_PATH)
        return
        
    logging.info("Reading road segment geometries from edges.csv...")
    edges_df = pd.read_csv(EDGES_PATH)
    
    # Create mapping: edge_id -> geometry string
    geom_mapping = {}
    for _, row in edges_df.iterrows():
        eid = row["edge_id"]
        geom = row.get("geometry")
        if pd.notna(geom):
            geom_mapping[eid] = str(geom)
            
    # 2. Run simulation on a priority edge (e.g. Silk Board or first unique edge)
    # Let's read one of the 100 GNN edges
    import joblib
    unique_edges = joblib.load(DATA_DIR / "gnn_edge_mapping.pkl")
    test_edge = unique_edges[0] # Pick the first priority edge
    
    logging.info("Simulating accident spread on test edge: %s", test_edge)
    sim_results = simulate_event_spread(test_edge, impact_score=0.8)
    
    # 3. Format visualization outputs
    roads_list = []
    for row in sim_results:
        eid = row["edge_id"]
        roads_list.append({
            "edge_id": eid,
            "geometry": geom_mapping.get(eid, "LINESTRING EMPTY"),
            "congestion": round(row["current"], 3),
            "predicted_congestion": round(row["60min"], 3) # Congestion propagation at T+60
        })
        
    export_data = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "roads": roads_list
    }
    
    # Save to JSON
    with open(OUTPUT_PATH, "w") as f:
        json.dump(export_data, f, indent=2)
        
    logging.info("Successfully exported simulation to %s", OUTPUT_PATH)

if __name__ == "__main__":
    export_simulation()
