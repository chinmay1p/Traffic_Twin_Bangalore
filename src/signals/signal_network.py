from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
import networkx as nx
import osmnx as ox
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
GRAPH_PATH = DATA_DIR / "bangalore_graph.graphml"
SIGNALS_CSV_PATH = DATA_DIR / "signals.csv"

def build_signal_network():
    logging.info("Loading Bangalore graph for signal network construction...")
    if not GRAPH_PATH.exists():
        logging.error("Graph file not found at %s", GRAPH_PATH)
        return
        
    G = ox.load_graphml(GRAPH_PATH)
    logging.info("Graph loaded successfully: %d nodes, %d edges", len(G.nodes), len(G.edges))
    
    # Convert to undirected to easily compute junction degrees
    G_undirected = G.to_undirected()
    
    signals = []
    major_types = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary", "secondary_link"}
    
    for node_id, node_data in G.nodes(data=True):
        lat = node_data.get("latitude") or node_data.get("y")
        lon = node_data.get("longitude") or node_data.get("x")
        
        # Calculate undirected degree
        undir_deg = G_undirected.degree(node_id)
        
        # Check if connected to major road type
        is_major_connected = False
        connected_edge_ids = []
        in_edges_count = 0
        
        # Scan incoming edges
        if node_id in G:
            for u, v, k, data in G.in_edges(node_id, keys=True, data=True):
                eid = data.get("edge_id")
                if eid:
                    connected_edge_ids.append(eid)
                    in_edges_count += 1
                rtype = data.get("road_type") or data.get("highway")
                if isinstance(rtype, list):
                    rtype = rtype[0]
                if rtype in major_types:
                    is_major_connected = True
                    
            # Scan outgoing edges
            for u, v, k, data in G.out_edges(node_id, keys=True, data=True):
                eid = data.get("edge_id")
                if eid and eid not in connected_edge_ids:
                    connected_edge_ids.append(eid)
                rtype = data.get("road_type") or data.get("highway")
                if isinstance(rtype, list):
                    rtype = rtype[0]
                if rtype in major_types:
                    is_major_connected = True
        
        # Criteria: Undirected degree >= 3 or connected to a major road corridor
        # AND it must have at least 2 incoming approaches to have signal conflicts.
        if (undir_deg >= 3 or is_major_connected) and in_edges_count >= 2:
            signal_id = f"signal_{node_id}"
            signals.append({
                "signal_id": signal_id,
                "node_id": node_id,
                "latitude": float(lat) if lat is not None else 0.0,
                "longitude": float(lon) if lon is not None else 0.0,
                "connected_edges": ";".join(connected_edge_ids),
                "number_of_approaches": in_edges_count
            })
            
    df_signals = pd.DataFrame(signals)
    df_signals.to_csv(SIGNALS_CSV_PATH, index=False)
    logging.info("Saved %d detected signals to %s", len(df_signals), SIGNALS_CSV_PATH)
    return df_signals

if __name__ == "__main__":
    build_signal_network()
