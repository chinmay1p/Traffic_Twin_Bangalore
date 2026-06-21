from __future__ import annotations

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

DATA_DIR = BASE_DIR / "data"
GRAPH_PATH = DATA_DIR / "bangalore_graph.graphml"
EDGES_PATH = DATA_DIR / "edges.csv"
TIMESERIES_PATH = DATA_DIR / "traffic_timeseries.csv"

def prepare_graph():
    logging.info("Preparing Spatio-Temporal GNN graph representation...")
    
    # 1. Identify the 100 priority edges from timeseries data
    if not TIMESERIES_PATH.exists():
        logging.error("Traffic timeseries file not found at %s", TIMESERIES_PATH)
        return
        
    ts_df = pd.read_csv(TIMESERIES_PATH)
    unique_edges = sorted(ts_df["edge_id"].unique())
    num_nodes = len(unique_edges)
    logging.info("Identified %d unique road segments (GNN nodes).", num_nodes)
    
    # Map edge_id -> integer index [0, ..., 99]
    edge_to_idx = {edge_id: idx for idx, edge_id in enumerate(unique_edges)}
    
    # 2. Extract edge properties and connectivity
    edge_data = []
    connectivity_dict = {} # edge_id -> (u, v)
    
    # Load from edges.csv if available for 100x speedup
    if EDGES_PATH.exists():
        logging.info("Loading edge properties from edges.csv...")
        edges_df = pd.read_csv(EDGES_PATH)
        filtered_edges = edges_df[edges_df["edge_id"].isin(unique_edges)].drop_duplicates(subset=["edge_id"])
        
        for _, row in filtered_edges.iterrows():
            eid = row["edge_id"]
            u, v = str(row["u"]), str(row["v"])
            connectivity_dict[eid] = (u, v)
            edge_data.append({
                "edge_id": eid,
                "length": float(row.get("length", 100.0)),
                "speed": float(row.get("speed", 40.0)),
                "lanes": float(row.get("lanes", 2)),
                "capacity": float(row.get("capacity", 3600)),
                "road_type": str(row.get("road_type", "primary"))
            })
    else:
        # Fallback to loading the large GraphML file
        logging.info("edges.csv not found. Loading bangalore_graph.graphml fallback...")
        G = nx.read_graphml(GRAPH_PATH)
        for u, v, k, data in G.edges(keys=True, data=True):
            eid = data.get("edge_id")
            if eid in unique_edges and eid not in connectivity_dict:
                connectivity_dict[eid] = (str(u), str(v))
                edge_data.append({
                    "edge_id": eid,
                    "length": float(data.get("length", 100.0)),
                    "speed": float(data.get("speed", 40.0)),
                    "lanes": float(data.get("lanes", 2)),
                    "capacity": float(data.get("capacity", 3600)),
                    "road_type": str(data.get("road_type", "primary"))
                })
                
    # Sort edge_data to align with unique_edges index ordering
    edge_data_df = pd.DataFrame(edge_data)
    edge_data_df["node_idx"] = edge_data_df["edge_id"].map(edge_to_idx)
    edge_data_df = edge_data_df.sort_values(by="node_idx").reset_index(drop=True)
    
    # Verify we mapped all 100 edges
    if len(edge_data_df) < num_nodes:
        logging.warning("Only mapped %d out of %d edges. Filling missing with defaults...", len(edge_data_df), num_nodes)
        missing_ids = [eid for eid in unique_edges if eid not in edge_data_df["edge_id"].values]
        for eid in missing_ids:
            connectivity_dict[eid] = ("unknown_u", "unknown_v")
            new_row = pd.DataFrame([{
                "edge_id": eid, "length": 100.0, "speed": 40.0, "lanes": 2.0, "capacity": 3600.0,
                "road_type": "primary", "node_idx": edge_to_idx[eid]
            }])
            edge_data_df = pd.concat([edge_data_df, new_row], ignore_index=True)
        edge_data_df = edge_data_df.sort_values(by="node_idx").reset_index(drop=True)

    # 3. Create line-graph adjacency matrix (100, 100)
    logging.info("Constructing adjacency matrix based on junction sharing...")
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    
    for i, eid_i in enumerate(unique_edges):
        u_i, v_i = connectivity_dict.get(eid_i, ("i_u", "i_v"))
        for j, eid_j in enumerate(unique_edges):
            if i == j:
                continue
            u_j, v_j = connectivity_dict.get(eid_j, ("j_u", "j_v"))
            
            # Check if they share any junction node in the original graph
            if len({u_i, v_i} & {u_j, v_j}) > 0:
                adj_matrix[i, j] = 1.0
                adj_matrix[j, i] = 1.0
                
    # 4. Generate normalized static node features
    logging.info("Engineering static node features...")
    le = LabelEncoder()
    edge_data_df["road_type_code"] = le.fit_transform(edge_data_df["road_type"].astype(str))
    
    feature_cols = ["length", "speed", "lanes", "capacity", "road_type_code"]
    X_static = edge_data_df[feature_cols].values
    
    scaler = StandardScaler()
    X_static_scaled = scaler.fit_transform(X_static)
    
    # 5. Save assets
    DATA_DIR.mkdir(exist_ok=True)
    np.save(DATA_DIR / "adjacency_matrix.npy", adj_matrix)
    np.save(DATA_DIR / "node_features.npy", X_static_scaled)
    joblib.dump(unique_edges, DATA_DIR / "gnn_edge_mapping.pkl")
    
    logging.info("Adjacency matrix saved with shape: %s", adj_matrix.shape)
    logging.info("Static node features saved with shape: %s", X_static_scaled.shape)
    logging.info("Line graph construction successfully finished.")

if __name__ == "__main__":
    prepare_graph()
