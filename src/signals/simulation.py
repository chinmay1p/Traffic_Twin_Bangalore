from __future__ import annotations

import json
import logging
import math
import copy
from pathlib import Path
from typing import Dict, Any, List, Tuple
import networkx as nx
import numpy as np
import pandas as pd

from src.signals.pressure_control import calculate_pressure, get_shared_resources, get_approach_direction
from src.signals.signal_optimizer import optimize_signal_timings
from src.signals.coordinate_signals import get_signal_neighbors, coordinate_signal_offsets, propagate_coordination_flows
from src.simulator.scenario_engine import get_base_graph
from src.simulator.road_closure import close_road, partial_closure
from src.simulator.traffic_assignment import redistribute_traffic
from src.simulator.impact_analyzer import run_gnn_propagation

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "outputs"
TIMINGS_JSON_PATH = OUTPUT_DIR / "signal_timings.json"
RECOMMENDATIONS_JSON_PATH = OUTPUT_DIR / "signal_recommendations.json"

def resolve_location_to_edge(G: nx.MultiDiGraph, location: str) -> str | None:
    """Resolves a road name or edge ID string to an edge_id in the graph."""
    # Check if direct match
    for u, v, k, data in G.edges(keys=True, data=True):
        eid = data.get("edge_id")
        if eid == location:
            return eid
            
    # Check by road name
    highest_flow = -1.0
    best_eid = None
    for u, v, k, data in G.edges(keys=True, data=True):
        rname = data.get("road_name", "")
        if rname and location.lower() in str(rname).lower():
            flow = float(data.get("current_flow", 0.0) or 0.0)
            if flow > highest_flow:
                highest_flow = flow
                best_eid = data.get("edge_id")
                
    return best_eid

def run_signal_queue_simulation(
    G: nx.MultiDiGraph,
    node_id: int,
    is_ai: bool = False,
    simulation_duration_sec: int = 3600,
    cycle_time_sec: int = 180
) -> Dict[str, float]:
    """
    Simulates queue dynamics at a signalized junction.
    
    In fixed-time mode, green times are equally divided (e.g. 45s each).
    In AI mode, timings are recalculated every cycle using Max Pressure optimization.
    
    Formula:
      Queue_t+1 = Queue_t + ArrivalRate - DepartureRate (if green)
      WaitTime = sum(Queue_t)
    """
    node_lat = float(G.nodes[node_id].get("latitude") or G.nodes[node_id].get("y"))
    node_lon = float(G.nodes[node_id].get("longitude") or G.nodes[node_id].get("x"))
    
    # 1. Identify approaches
    approaches = []
    for u, v, k, data in G.in_edges(node_id, keys=True, data=True):
        u_lat = float(G.nodes[u].get("latitude") or G.nodes[u].get("y"))
        u_lon = float(G.nodes[u].get("longitude") or G.nodes[u].get("x"))
        dir_name = get_approach_direction(u_lat, u_lon, node_lat, node_lon)
        
        flow = data.get("current_flow", 1000.0) or 1000.0
        lanes = data.get("lanes", 1) or 1
        length_m = data.get("length_meter", data.get("length", 100.0)) or 100.0
        
        approaches.append({
            "direction": dir_name,
            "edge_id": data.get("edge_id"),
            "arrival_rate": flow / 3600.0, # vehicles/second
            "saturation_flow": lanes * 0.5, # vehicles/second capacity during green
            "queue": 5.0, # initial baseline queue
            "length_m": length_m,
            "lanes": lanes
        })
        
    if not approaches:
        return {"avg_waiting_time_sec": 0.0, "queue_reduction_pct": 0.0, "throughput": 0.0}
        
    num_cycles = math.ceil(simulation_duration_sec / cycle_time_sec)
    total_wait_time = 0.0
    total_vehicles_arrived = 0.0
    total_vehicles_cleared = 0.0
    queue_history = []
    
    # Run cycle-by-cycle simulation
    for cycle in range(num_cycles):
        # 1. Determine green timings for this cycle
        timings = {}
        active_dirs = list(set([a["direction"] for a in approaches]))
        
        if is_ai:
            # Calculate pressures dynamically from current queues (acting as local density)
            pressures = {}
            for d in active_dirs:
                # Sum queues for this direction
                q_sum = sum(a["queue"] for a in approaches if a["direction"] == d)
                # Map to density (vehicles/km/lane)
                # density = queue / (length_km * lanes)
                # For simplicity in pressure calculation, we map directly:
                pressures[d] = {
                    "incoming_queue": q_sum,
                    "outgoing_space": 100.0, # assumed downstream space
                    "pressure": q_sum - 100.0
                }
            timings = optimize_signal_timings(pressures, total_cycle=cycle_time_sec)
        else:
            # Fixed timings: equal distribution
            share = cycle_time_sec / len(active_dirs)
            timings = {d: share for d in active_dirs}
            
        # 2. Simulate second-by-second traffic flow during the cycle
        for sec in range(cycle_time_sec):
            # Find which direction has green light at this second
            # Active directions serve sequentially
            active_dirs_sorted = sorted(active_dirs)
            green_dir = None
            elapsed = 0.0
            
            for d in active_dirs_sorted:
                green_dur = timings.get(d, 0.0)
                if elapsed <= sec < elapsed + green_dur:
                    green_dir = d
                    break
                elapsed += green_dur
                
            for app in approaches:
                # Arrival
                app["queue"] += app["arrival_rate"]
                total_vehicles_arrived += app["arrival_rate"]
                
                # Departure
                if app["direction"] == green_dir:
                    dep = min(app["queue"], app["saturation_flow"])
                    app["queue"] -= dep
                    total_vehicles_cleared += dep
                    
                total_wait_time += app["queue"]
                queue_history.append(app["queue"])
                
    avg_wait = total_wait_time / max(1.0, total_vehicles_arrived)
    avg_queue = sum(queue_history) / len(queue_history) if queue_history else 0.0
    
    return {
        "avg_waiting_time_sec": float(round(avg_wait, 1)),
        "avg_queue_size": float(round(avg_queue, 1)),
        "throughput": float(round(total_vehicles_cleared, 1))
    }

def evaluate_fixed_vs_ai(
    G: nx.MultiDiGraph,
    node_id: int
) -> Dict[str, Any]:
    """Compares Fixed signal performance against AI Signal Control."""
    fixed_metrics = run_signal_queue_simulation(G, node_id, is_ai=False)
    ai_metrics = run_signal_queue_simulation(G, node_id, is_ai=True)
    
    wait_reduction = 0.0
    f_wait = fixed_metrics["avg_waiting_time_sec"]
    a_wait = ai_metrics["avg_waiting_time_sec"]
    if f_wait > 0.0:
        wait_reduction = ((f_wait - a_wait) / f_wait) * 100.0
        
    throughput_improvement = 0.0
    f_thru = fixed_metrics["throughput"]
    a_thru = ai_metrics["throughput"]
    if f_thru > 0.0:
        throughput_improvement = ((a_thru - f_thru) / f_thru) * 100.0
        
    return {
        "fixed": {
            "average_wait_sec": f_wait,
            "average_queue": fixed_metrics["avg_queue_size"],
            "throughput": f_thru
        },
        "ai": {
            "average_wait_sec": a_wait,
            "average_queue": ai_metrics["avg_queue_size"],
            "throughput": a_thru
        },
        "metrics": {
            "waiting_time_reduction_pct": float(round(wait_reduction, 1)),
            "throughput_improvement_pct": float(round(throughput_improvement, 1))
        }
    }

def optimize_after_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Triggers the signal optimization and coordination workflow following a road incident.
    
    Pipeline:
      1. Resolve incident location to road edge.
      2. Simulate closure/redistribution on graph G.
      3. Run ST-GNN prediction to propagate congestion states.
      4. Calculate pre-event and post-event pressures and optimal timings for local signals.
      5. Apply green wave coordination offsets.
      6. Formulate police brief recommendations.
    """
    loc = event.get("accident_location")
    ctype = event.get("type", "full")
    pct = float(event.get("closure_percentage", 100.0))
    
    if not loc:
        raise ValueError("Missing parameter: 'accident_location'")
        
    G = get_base_graph()
    target_eid = resolve_location_to_edge(G, loc)
    if not target_eid:
        raise ValueError(f"Could not find road matching name or ID: {loc}")
        
    # Find epicenter coordinates
    u_closed, v_closed = None, None
    for u, v, k, data in G.edges(keys=True, data=True):
        if data.get("edge_id") == target_eid:
            u_closed, v_closed = u, v
            break
            
    # Apply closure & redistribute
    orig_flow = float(G[u_closed][v_closed][0].get("current_flow", 1000.0))
    if ctype == "full":
        close_road(G, target_eid)
        blocked_flow = orig_flow
        impact_score = 1.0
    else:
        partial_closure(G, target_eid, pct)
        blocked_flow = orig_flow * (pct / 100.0)
        impact_score = pct / 100.0
        
    redistribute_traffic(G, u_closed, v_closed, blocked_flow, K=5)
    
    # Run ST-GNN propagation
    timeline_results = run_gnn_propagation(G, target_eid, impact_score)
    
    # Update G densities based on GNN predicted congestion values (60min step)
    edge_to_fut_cong = {row["edge_id"]: row["60min"] for row in timeline_results}
    for u, v, k, data in G.edges(keys=True, data=True):
        eid = data.get("edge_id")
        if eid in edge_to_fut_cong:
            fut_cong = edge_to_fut_cong[eid]
            # Map congestion back to density (fraction of 100 vehicles/km/lane)
            data["current_density"] = float(fut_cong * 100.0)
            data["congestion_score"] = float(fut_cong)
            
    # Find affected signals near the incident epicenter
    epi_lat = float(G.nodes[u_closed].get("latitude") or G.nodes[u_closed].get("y"))
    epi_lon = float(G.nodes[u_closed].get("longitude") or G.nodes[u_closed].get("x"))
    
    _, df_signals = get_shared_resources()
    
    # Find closest signal node
    closest_sig_row = None
    min_dist = float("inf")
    
    from src.simulator.impact_analyzer import haversine_distance
    
    affected_signals = []
    for _, row in df_signals.iterrows():
        sig_lat = float(row["latitude"])
        sig_lon = float(row["longitude"])
        # Calculate Euclidean distance in meters, convert to km
        dist_m = math.sqrt((epi_lat - sig_lat)**2 + (epi_lon - sig_lon)**2)
        dist = dist_m / 1000.0
        
        # Consider signals within 3.0 km
        if dist < 3.0:
            affected_signals.append((row["signal_id"], int(row["node_id"]), dist))
        if dist < min_dist:
            min_dist = dist
            closest_sig_row = row
            
    if not affected_signals and closest_sig_row is not None:
        affected_signals.append((closest_sig_row["signal_id"], int(closest_sig_row["node_id"]), min_dist))
        
    # Sort by distance
    affected_signals.sort(key=lambda x: x[2])
    
    # Generate recommendations for primary signal
    primary_sig_id, primary_node_id, _ = affected_signals[0]
    
    # Calculate baseline (pre-event) pressure on original cached graph
    G_base = get_base_graph()
    pre_pressures = calculate_pressure(primary_sig_id, G_base)
    pre_timings = optimize_signal_timings(pre_pressures)
    
    # Calculate post-event pressures on G_post
    post_pressures = calculate_pressure(primary_sig_id, G)
    post_timings = optimize_signal_timings(post_pressures)
    
    # Format changes list
    changes = []
    for direction in ["North", "South", "East", "West"]:
        old_g = pre_timings.get(direction, 0.0)
        new_g = post_timings.get(direction, 0.0)
        if old_g > 0.0 or new_g > 0.0:
            changes.append({
                "direction": direction,
                "old_green": int(old_g),
                "new_green": int(new_g)
            })
            
    # Green Wave coordination for neighbors
    neighbors = get_signal_neighbors(G, df_signals)
    coor_offsets = coordinate_signal_offsets(neighbors)
    
    # Calculate evaluation statistics
    eval_stats = evaluate_fixed_vs_ai(G, primary_node_id)
    
    # Prepare reports structure
    primary_signal_name = G.nodes[primary_node_id].get("name") or f"Junction {primary_node_id}"
    if "silk board" in loc.lower() or "silk board" in primary_signal_name.lower():
        primary_signal_name = "Silk Board Junction"
        
    output_report = {
        "signal": primary_signal_name,
        "signal_id": primary_sig_id,
        "changes": changes,
        "evaluation": eval_stats,
        "coordination_offsets": coor_offsets
    }
    
    # Save results to outputs files
    timings_data = {
        "signal_id": primary_sig_id,
        "pre_timings": pre_timings,
        "post_timings": post_timings
    }
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(TIMINGS_JSON_PATH, "w") as f:
        json.dump(timings_data, f, indent=2)
        
    with open(RECOMMENDATIONS_JSON_PATH, "w") as f:
        json.dump(output_report, f, indent=2)
        
    logging.info("Saved signal timings to %s", TIMINGS_JSON_PATH)
    logging.info("Saved signal recommendations to %s", RECOMMENDATIONS_JSON_PATH)
    
    return output_report
