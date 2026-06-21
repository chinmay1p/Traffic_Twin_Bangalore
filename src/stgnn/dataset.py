from __future__ import annotations

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

DATA_DIR = BASE_DIR / "data"
TIMESERIES_PATH = DATA_DIR / "traffic_timeseries.csv"
EVENTS_PATH = DATA_DIR / "astram_mapped_events.csv"
SEQUENCES_OUTPUT_PATH = DATA_DIR / "temporal_sequences.npy"

def create_temporal_dataset():
    logging.info("Starting temporal dataset generation...")
    
    if not TIMESERIES_PATH.exists():
        logging.error("Traffic timeseries file not found at %s", TIMESERIES_PATH)
        return
        
    # 1. Load Timeseries
    ts_df = pd.read_csv(TIMESERIES_PATH)
    ts_df["timestamp"] = pd.to_datetime(ts_df["timestamp"])
    
    # Get sorted unique edge list (consistent mapping from node index to edge_id)
    unique_edges = sorted(ts_df["edge_id"].unique())
    num_roads = len(unique_edges)
    edge_to_idx = {eid: idx for idx, eid in enumerate(unique_edges)}
    
    logging.info("Mapping active event impacts...")
    # Initialize event_impact
    ts_df["event_impact"] = 0.0
    
    # 2. Parse Events and calculate overlap
    if EVENTS_PATH.exists():
        events_df = pd.read_csv(EVENTS_PATH)
        events_df = events_df[events_df["nearest_edge_id"].isin(unique_edges)]
        
        # Calculate impact for each event
        event_list = []
        for _, row in events_df.iterrows():
            eid = row["nearest_edge_id"]
            if pd.isna(eid):
                continue
            priority_val = 1.0 if str(row.get("priority", "")).lower() == "high" else 0.0
            closure_val = 1.0 if str(row.get("requires_road_closure", "")).lower() in ("true", "1", "yes") else 0.0
            impact = 0.4 * priority_val + 0.3 * closure_val + 0.2
            
            try:
                start_dt = pd.to_datetime(row["start_datetime"]).tz_localize(None)
                # Assume active for 2 hours
                end_dt = start_dt + pd.Timedelta(hours=2)
                event_list.append({
                    "edge_id": eid,
                    "start": start_dt,
                    "end": end_dt,
                    "impact": impact
                })
            except Exception:
                continue
                
        # Apply impacts to timeseries
        for eid in unique_edges:
            edge_events = [ev for ev in event_list if ev["edge_id"] == eid]
            if not edge_events:
                continue
            
            # Filter timeseries for this road
            mask = ts_df["edge_id"] == eid
            road_ts = ts_df.loc[mask, ["timestamp"]].copy()
            road_impacts = np.zeros(len(road_ts))
            
            for ev in edge_events:
                # Find indices where timestamp is within event duration
                in_event = (road_ts["timestamp"] >= ev["start"]) & (road_ts["timestamp"] < ev["end"])
                road_impacts[in_event] = np.maximum(road_impacts[in_event], ev["impact"])
                
            ts_df.loc[mask, "event_impact"] = road_impacts

    # 3. Scale Features
    logging.info("Normalizing features...")
    features = ["speed", "density", "flow", "congestion_score", "event_impact"]
    
    # Scale each feature independently
    scalers = {}
    for feat in features:
        scaler = MinMaxScaler()
        ts_df[feat] = scaler.fit_transform(ts_df[[feat]].fillna(0.0))
        scalers[feat] = scaler
        
    joblib.dump(scalers, DATA_DIR / "dataset_scalers.pkl")

    # 4. Construct 3D Grid: (timestamps, roads, features)
    logging.info("Pivoting data to align roads...")
    ts_df = ts_df.sort_values(by=["timestamp", "edge_id"])
    
    timestamps = sorted(ts_df["timestamp"].unique())
    num_timestamps = len(timestamps)
    
    grid = np.zeros((num_timestamps, num_roads, len(features)), dtype=np.float32)
    
    for t_idx, t in enumerate(timestamps):
        t_data = ts_df[ts_df["timestamp"] == t].sort_values("edge_id")
        # Ensure perfect alignment
        if len(t_data) != num_roads:
            # Fallback alignment if any road is missing for a timestamp
            road_features = {row["edge_id"]: [row[f] for f in features] for _, row in t_data.iterrows()}
            for r_idx, eid in enumerate(unique_edges):
                if eid in road_features:
                    grid[t_idx, r_idx] = road_features[eid]
                else:
                    grid[t_idx, r_idx] = [0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            grid[t_idx] = t_data[features].values

    # 5. Generate sliding window sequences
    # Input: 4 timestamps (1 hour)
    # Output: 1 timestamp (next 15 mins congestion_score)
    logging.info("Generating sequences...")
    window_size = 4
    num_samples = num_timestamps - window_size
    
    X = np.zeros((num_samples, window_size, num_roads, len(features)), dtype=np.float32)
    Y = np.zeros((num_samples, num_roads), dtype=np.float32)
    
    # Congestion score is the 4th feature (index 3)
    congestion_feat_idx = features.index("congestion_score")
    
    for i in range(num_samples):
        X[i] = grid[i : i + window_size]
        Y[i] = grid[i + window_size, :, congestion_feat_idx]
        
    # Save datasets
    np.save(SEQUENCES_OUTPUT_PATH, {"X": X, "Y": Y})
    logging.info("Saved temporal sequences: X shape %s, Y shape %s", X.shape, Y.shape)
    logging.info("Temporal dataset creation successfully finished.")

if __name__ == "__main__":
    create_temporal_dataset()
