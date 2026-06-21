from __future__ import annotations

import logging
import networkx as nx
from typing import Dict, Any, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def estimate_queue(density: float, length_m: float, lanes: int) -> float:
    """
    Estimates the number of vehicles waiting in a queue on a road segment.
    
    Formula:
      queue_length = density * road_length_km * lanes
      
    where density is in vehicles/km/lane, and road_length_km is in kilometers.
    """
    length_km = length_m / 1000.0
    queue = density * length_km * lanes
    return float(max(0.0, queue))

def get_signal_queues_and_capacities(
    G: nx.MultiDiGraph,
    node_id: int
) -> Tuple[Dict[str, float], float]:
    """
    Calculates incoming queues (per edge approach) and total outgoing capacity for a signal node.
    
    Args:
      G: The NetworkX MultiDiGraph.
      node_id: The junction node ID.
      
    Returns:
      A tuple (incoming_queues, total_outgoing_capacity) where:
        incoming_queues: Dict mapping edge_id to estimated queue length
        total_outgoing_capacity: Float representing total downstream capacity
    """
    incoming_queues = {}
    total_outgoing_capacity = 0.0
    
    if node_id not in G:
        return incoming_queues, total_outgoing_capacity
        
    # 1. Calculate incoming queues
    for u, v, k, data in G.in_edges(node_id, keys=True, data=True):
        edge_id = data.get("edge_id")
        if not edge_id:
            continue
            
        density = data.get("current_density", 0.0) or 0.0
        length_m = data.get("length_meter", data.get("length", 100.0)) or 100.0
        lanes = data.get("lanes", 1) or 1
        
        queue = estimate_queue(density, length_m, lanes)
        incoming_queues[edge_id] = queue
        
    # 2. Calculate outgoing capacity
    for u, v, k, data in G.out_edges(node_id, keys=True, data=True):
        # Capacity of downstream road
        cap = data.get("capacity", 1800.0) or 1800.0
        total_outgoing_capacity += float(cap)
        
    return incoming_queues, total_outgoing_capacity
