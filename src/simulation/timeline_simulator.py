"""
Timeline Simulator — Generates animation-ready traffic snapshots.

Produces a ``timeline.json`` containing the traffic state of every road
at T, T+15, T+30, T+45, and T+60 minutes.  Each snapshot stores:
  edge_id, road_name, geometry (WKT), congestion, speed, status.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "outputs"


def _congestion_to_speed(congestion: float, speed_limit: float) -> float:
    """BPR speed-congestion relationship."""
    return max(1.0, speed_limit / (1.0 + 0.15 * (congestion ** 4)))


def _congestion_to_status(congestion: float) -> str:
    if congestion >= 0.8:
        return "gridlock"
    elif congestion >= 0.6:
        return "heavy"
    elif congestion >= 0.35:
        return "moderate"
    else:
        return "free_flow"


def build_timeline(
    road_states: Dict[str, Dict[str, Any]],
    gnn_timeline: List[Dict[str, Any]],
    event_edge_id: str,
) -> Dict[str, Any]:
    """
    Build a five-step timeline dictionary.

    Args:
        road_states:   Current city state dict (edge_id -> road info).
        gnn_timeline:  ST-GNN output list of dicts with keys
                       ``edge_id``, ``current``, ``15min``, ``30min``, ``60min``.
        event_edge_id: The epicenter edge id.

    Returns:
        Dict with keys ``timestamps`` and ``snapshots``.
    """
    # Index GNN predictions by edge_id
    gnn_map: Dict[str, Dict[str, float]] = {}
    for row in gnn_timeline:
        gnn_map[row["edge_id"]] = {
            "current": float(row.get("current", 0.0)),
            "15min": float(row.get("15min", 0.0)),
            "30min": float(row.get("30min", 0.0)),
            "60min": float(row.get("60min", 0.0)),
        }

    # Interpolate 45min from 30min and 60min
    for eid, preds in gnn_map.items():
        preds["45min"] = (preds["30min"] + preds["60min"]) / 2.0

    timestamps = ["T+0", "T+15", "T+30", "T+45", "T+60"]
    horizon_keys = ["current", "15min", "30min", "45min", "60min"]

    snapshots: Dict[str, List[Dict[str, Any]]] = {}
    for ts, hk in zip(timestamps, horizon_keys):
        snap: List[Dict[str, Any]] = []
        for eid, rd in road_states.items():
            cong = gnn_map.get(eid, {}).get(hk, rd.get("congestion", 0.0))
            speed_limit = rd.get("speed_limit", 30.0)
            snap.append({
                "edge_id": eid,
                "road_name": rd.get("road_name", "Unknown"),
                "congestion": round(float(cong), 4),
                "speed": round(_congestion_to_speed(cong, speed_limit), 1),
                "status": _congestion_to_status(cong),
            })
        snapshots[ts] = snap

    timeline = {
        "event_edge_id": event_edge_id,
        "timestamps": timestamps,
        "snapshots": snapshots,
    }

    # Persist
    import os
    if "VERCEL" in os.environ:
        out_path = Path("/tmp/timeline.json")
    else:
        OUTPUT_DIR.mkdir(exist_ok=True)
        out_path = OUTPUT_DIR / "timeline.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(timeline, f, indent=2, default=str)
    logging.info("Timeline saved to %s (%d roads x %d steps).", out_path, len(road_states), len(timestamps))

    return timeline
