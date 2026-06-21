from __future__ import annotations

import logging
import math
import networkx as nx
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple

from src.signals.queue_estimator import estimate_queue

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
SIGNALS_CSV_PATH = DATA_DIR / "signals.csv"

_graph_cache = None
_signals_df = None

def get_shared_resources() -> Tuple[nx.MultiDiGraph, pd.DataFrame]:
    """Loads and caches the graph and signals dataframe for standalone calls."""
    global _graph_cache, _signals_df
    if _graph_cache is None:
        from src.simulator.scenario_engine import get_base_graph
        _graph_cache = get_base_graph()
    if _signals_df is None:
        if SIGNALS_CSV_PATH.exists():
            _signals_df = pd.read_csv(SIGNALS_CSV_PATH)
        else:
            from src.signals.signal_network import build_signal_network
            _signals_df = build_signal_network()
    return _graph_cache, _signals_df

def get_approach_direction(u_lat: float, u_lon: float, node_lat: float, node_lon: float) -> str:
    """Classifies an incoming approach direction based on geographic coordinates."""
    dy = u_lat - node_lat
    dx = u_lon - node_lon
    angle = math.atan2(dy, dx) * 180 / math.pi
    if angle < 0:
        angle += 360
        
    # Sector classification
    if 45 <= angle < 135:
        return "North"
    elif 135 <= angle < 225:
        return "West"
    elif 225 <= angle < 315:
        return "South"
    else:
        return "East"

def calculate_pressure(signal_id: str, G: nx.MultiDiGraph = None) -> Dict[str, Any]:
    """
    Calculates traffic pressure for all directions of a signalized junction.
    
    Formula:
      pressure = incoming_queue - outgoing_available_space
      
      where:
        incoming_queue = density * road_length_km * lanes
        outgoing_available_space = max(0, capacity - current_vehicles_downstream)
        
    Args:
      signal_id: The ID of the signal (e.g. 'signal_node_id').
      G: Optional graph structure. If None, uses cached base graph.
      
    Returns:
      A dictionary mapping direction names (North, East, South, West) to their pressure details.
    """
    cached_G, df_signals = get_shared_resources()
    if G is None:
        G = cached_G
        
    # Extract node ID from signal_id
    try:
        node_str = signal_id.replace("signal_", "")
        node_id = int(node_str)
    except ValueError:
        # Fallback search in dataframe
        row = df_signals[df_signals["signal_id"] == signal_id]
        if row.empty:
            logging.warning("Signal ID %s not found in signals.csv", signal_id)
            return {}
        node_id = int(row.iloc[0]["node_id"])
        
    if node_id not in G:
        logging.warning("Node %d (from %s) not found in Graph.", node_id, signal_id)
        return {}
        
    node_lat = float(G.nodes[node_id].get("latitude") or G.nodes[node_id].get("y"))
    node_lon = float(G.nodes[node_id].get("longitude") or G.nodes[node_id].get("x"))
    
    # 1. Group incoming edges by approach direction
    incoming_by_direction = {}
    for u, v, k, data in G.in_edges(node_id, keys=True, data=True):
        u_lat = float(G.nodes[u].get("latitude") or G.nodes[u].get("y"))
        u_lon = float(G.nodes[u].get("longitude") or G.nodes[u].get("x"))
        dir_name = get_approach_direction(u_lat, u_lon, node_lat, node_lon)
        
        density = data.get("current_density", 0.0) or 0.0
        length_m = data.get("length_meter", data.get("length", 100.0)) or 100.0
        lanes = data.get("lanes", 1) or 1
        
        queue = estimate_queue(density, length_m, lanes)
        incoming_by_direction[dir_name] = incoming_by_direction.get(dir_name, 0.0) + queue
        
    # 2. Group outgoing edges by departure direction (where the road leads)
    outgoing_by_direction = {}
    for u, v, k, data in G.out_edges(node_id, keys=True, data=True):
        v_lat = float(G.nodes[v].get("latitude") or G.nodes[v].get("y"))
        v_lon = float(G.nodes[v].get("longitude") or G.nodes[v].get("x"))
        dir_name = get_approach_direction(v_lat, v_lon, node_lat, node_lon)
        
        capacity = data.get("capacity", 1800.0) or 1800.0
        density = data.get("current_density", 0.0) or 0.0
        length_m = data.get("length_meter", data.get("length", 100.0)) or 100.0
        lanes = data.get("lanes", 1) or 1
        
        current_vehicles = estimate_queue(density, length_m, lanes)
        available_space = max(0.0, capacity - current_vehicles)
        outgoing_by_direction[dir_name] = outgoing_by_direction.get(dir_name, 0.0) + available_space

    # Map opposite directions: straight-through flows go from approach to departure
    # e.g., incoming North (heading South) matches outgoing South (departure leads South)
    opposite_map = {
        "North": "South",
        "South": "North",
        "East": "West",
        "West": "East"
    }
    
    pressures = {}
    for direction in incoming_by_direction.keys():
        in_queue = incoming_by_direction[direction]
        out_dir = opposite_map.get(direction, "South")
        
        # If no specific opposite outgoing exists, default available space to 1000
        out_space = outgoing_by_direction.get(out_dir, 1000.0)
        
        # Calculate pressure
        pressure_val = in_queue - out_space
        pressures[direction] = {
            "incoming_queue": float(round(in_queue, 2)),
            "outgoing_space": float(round(out_space, 2)),
            "pressure": float(round(pressure_val, 2))
        }
        
    return pressures
