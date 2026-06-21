from __future__ import annotations

import logging
import networkx as nx
import pandas as pd
from typing import Dict, Any, List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def get_signal_neighbors(
    G: nx.MultiDiGraph,
    signals_df: pd.DataFrame,
    max_distance_m: float = 1000.0
) -> List[Tuple[str, str, float, str]]:
    """
    Identifies neighboring signals (pairs connected by a short path).
    
    Args:
      G: The NetworkX road network.
      signals_df: DataFrame of signal locations.
      max_distance_m: Maximum road distance to consider as neighbors.
      
    Returns:
      A list of tuples: (signal_A, signal_B, travel_time_sec, direction_A_to_B)
    """
    neighbors = []
    node_to_signal = {int(row["node_id"]): row["signal_id"] for _, row in signals_df.iterrows()}
    signal_nodes = set(node_to_signal.keys())
    
    for u in signal_nodes:
        # Find neighbors within a small radius or direct connections
        # We can look at outgoing edges from u
        for v in G.successors(u):
            # Check direct edge
            if v in signal_nodes:
                # Direct neighbor signal!
                edges_data = G.get_edge_data(u, v)
                if not edges_data:
                    continue
                # Take first key
                key = list(edges_data.keys())[0]
                edge_data = edges_data[key]
                
                length = edge_data.get("length_meter", edge_data.get("length", 200.0))
                speed = edge_data.get("speed_kmph", 40.0)
                travel_time = (length / 1000.0) / speed * 3600.0 if speed > 0 else 30.0
                
                # Determine direction from u to v
                # u coordinates
                u_lat = float(G.nodes[u].get("latitude") or G.nodes[u].get("y"))
                u_lon = float(G.nodes[u].get("longitude") or G.nodes[u].get("x"))
                v_lat = float(G.nodes[v].get("latitude") or G.nodes[v].get("y"))
                v_lon = float(G.nodes[v].get("longitude") or G.nodes[v].get("x"))
                
                from src.signals.pressure_control import get_approach_direction
                # The direction of the flow is from u to v, which is departures for u, and approach for v.
                # Let's get the approach direction at v
                direction = get_approach_direction(u_lat, u_lon, v_lat, v_lon)
                
                neighbors.append((node_to_signal[u], node_to_signal[v], float(travel_time), direction))
                
    return neighbors

def coordinate_signal_offsets(
    neighbors: List[Tuple[str, str, float, str]],
    total_cycle: float = 180.0
) -> Dict[str, Dict[str, float]]:
    """
    Calculates green wave coordination offsets for neighboring signalized intersections.
    
    Formula:
      offset_B = (offset_A + travel_time) % total_cycle
      
    Returns:
      A dictionary mapping signal_id pair to coordination offset.
    """
    offsets = {}
    for sig_A, sig_B, travel_time, direction in neighbors:
        # Calculate offset in seconds
        offset_val = travel_time % total_cycle
        key = f"{sig_A}->{sig_B}"
        offsets[key] = {
            "travel_time_sec": float(round(travel_time, 1)),
            "recommended_offset_sec": float(round(offset_val, 1)),
            "coordinated_direction": direction
        }
    return offsets

def propagate_coordination_flows(
    G: nx.MultiDiGraph,
    signals_df: pd.DataFrame,
    base_pressures: Dict[str, Dict[str, Dict[str, float]]],
    neighbors: List[Tuple[str, str, float, str]],
    coordination_factor: float = 0.35
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Propagates upstream release queues to downstream signals to prepare green times (cooperative pressure).
    
    If Signal A has a large queue in direction North (heading South to B), B will receive
    an anticipatory queue increase in its North approach to trigger larger green times.
    """
    # Deep copy base pressures
    coordinated_pressures = {}
    for sig_id, directions in base_pressures.items():
        coordinated_pressures[sig_id] = {
            d: dict(metrics) for d, metrics in directions.items()
        }
        
    for sig_A, sig_B, travel_time, direction in neighbors:
        if sig_A not in coordinated_pressures or sig_B not in coordinated_pressures:
            continue
            
        # Upstream queue released from A towards B
        # Let's map direction: if B's approach is 'direction' (e.g. North), then the flow from A is also North
        # We look at A's opposite direction or B's approach direction
        # Let's check A's queue in B's approach direction
        upstream_metrics = coordinated_pressures[sig_A].get(direction)
        if not upstream_metrics:
            continue
            
        upstream_queue = upstream_metrics.get("incoming_queue", 0.0)
        
        # If upstream queue is substantial, propagate a virtual queue to B
        if upstream_queue > 5.0:
            anticipated_arrival = upstream_queue * coordination_factor
            
            # Increase downstream queue in B for 'direction'
            if direction in coordinated_pressures[sig_B]:
                coordinated_pressures[sig_B][direction]["incoming_queue"] += anticipated_arrival
                # Recalculate pressure
                in_q = coordinated_pressures[sig_B][direction]["incoming_queue"]
                out_s = coordinated_pressures[sig_B][direction]["outgoing_space"]
                coordinated_pressures[sig_B][direction]["pressure"] = in_q - out_s
                
    return coordinated_pressures
