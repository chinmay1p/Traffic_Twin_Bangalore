from __future__ import annotations

import csv
import logging
import re
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
EDGES_CSV = DATA_DIR / "edges.csv"
ASTRAM_EVENTS_CSV = DATA_DIR / "astram_mapped_events.csv"
COMMAND_CENTER_ROADS_CSV = DATA_DIR / "command_center_roads.csv"

ROAD_TYPE_BASE_SCORES = {
    "motorway": 100,
    "motorway_link": 100,
    "trunk": 90,
    "trunk_link": 90,
    "primary": 80,
    "primary_link": 80,
    "secondary": 60,
    "secondary_link": 60,
    "tertiary": 25,
    "tertiary_link": 25,
    "residential": 5,
}

CONGESTION_HOTSPOTS = {
    "silk board": ["silk", "hosur", "btm"],
    "orr": ["outer ring", "orr", "marathahalli", "bellandur", "ibblur"],
    "hebbal": ["hebbal", "mekhri"],
    "kr puram": ["kr puram", "tin factory", "hanging bridge", "old madras"],
    "marathahalli": ["marathahalli", "kalamandir", "multiplex"],
    "whitefield": ["whitefield", "itpl", "hope farm"],
    "electronic city": ["electronic"],
    "mg road": ["mg road", "m g road"],
    "majestic": ["majestic", "k g road", "mysore road"],
    "koramangala": ["koramangala", "sony world", "80 feet"]
}

def _normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()

def _safe_float(value, default=0.0) -> float:
    try:
        if value in (None, "", "nan"):
            return default
        return float(value)
    except Exception:
        return default

def _safe_int(value, default=0) -> int:
    try:
        if value in (None, "", "nan"):
            return default
        return int(float(value))
    except Exception:
        return default

def _get_hotspot_boost(road_name: str) -> float:
    name = _normalized(road_name)
    if not name:
        return 0.0
    for area, keywords in CONGESTION_HOTSPOTS.items():
        if any(kw in name for kw in keywords):
            return 30.0
    return 0.0

def _load_event_frequency() -> Counter:
    frequency = Counter()
    if not ASTRAM_EVENTS_CSV.exists():
        return frequency

    try:
        with open(ASTRAM_EVENTS_CSV, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                candidates = [
                    row.get("road_name"),
                    row.get("location"),
                    row.get("junction_name"),
                    row.get("address"),
                ]
                for candidate in candidates:
                    key = _normalized(candidate)
                    if key:
                        frequency[key] += 1
                        break
    except Exception as exc:
        logging.warning("Could not read Astram event frequency: %s", exc)

    return frequency

def build_command_center_roads(limit: int = 3500) -> Path:
    if not EDGES_CSV.exists():
        raise FileNotFoundError(f"Missing edges file: {EDGES_CSV}")

    logging.info("Building closed graph network of ASTRAM priority roads...")
    
    # 1. Load all ASTRAM nearest_edge_ids
    astram_edge_ids = set()
    if ASTRAM_EVENTS_CSV.exists():
        try:
            with open(ASTRAM_EVENTS_CSV, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    eid = row.get("nearest_edge_id")
                    if eid:
                        astram_edge_ids.add(str(eid).strip())
        except Exception as exc:
            logging.warning("Could not read ASTRAM events: %s", exc)

    # 2. Read all edges from edges.csv
    edges = []
    with open(EDGES_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            edges.append(row)

    # 3. Find the priority direct matching edges and collect their endpoint nodes
    priority_types = {
        "motorway", "motorway_link",
        "trunk", "trunk_link",
        "primary", "primary_link",
        "secondary", "secondary_link",
        "tertiary", "tertiary_link"
    }

    direct_nodes = set()
    for edge in edges:
        eid = str(edge.get("edge_id", "")).strip()
        rtype = str(edge.get("road_type", "")).lower().strip()
        if eid in astram_edge_ids and rtype in priority_types:
            direct_nodes.add(str(edge.get("u", "")))
            direct_nodes.add(str(edge.get("v", "")))

    # 4. Build adjacency on the induced priority edges to find connected components
    from collections import defaultdict
    adj = defaultdict(list)
    induced_edges = []
    for edge in edges:
        u = str(edge.get("u", ""))
        v = str(edge.get("v", ""))
        rtype = str(edge.get("road_type", "")).lower().strip()
        if u in direct_nodes and v in direct_nodes and rtype in priority_types:
            induced_edges.append(edge)
            adj[u].append(v)
            adj[v].append(u)

    # Find connected components of nodes using BFS
    visited = set()
    components = []
    for node in direct_nodes:
        if node not in visited:
            comp = []
            queue = [node]
            visited.add(node)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(comp)

    if not components:
        logging.warning("No connected components found! Falling back to raw selection.")
        selected_nodes = direct_nodes
    else:
        # Sort by size descending and take the largest component
        components.sort(key=len, reverse=True)
        selected_nodes = set(components[0])
        logging.info("Selected largest connected component with %d nodes of %d total nodes.", len(selected_nodes), len(direct_nodes))

    # 5. Extract only the edges belonging to this largest component
    selected_edges = []
    for edge in induced_edges:
        u = str(edge.get("u", ""))
        v = str(edge.get("v", ""))
        if u in selected_nodes and v in selected_nodes:
            selected_edges.append(edge)

    logging.info("Formed closed graph network: %d edges.", len(selected_edges))

    COMMAND_CENTER_ROADS_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    # Map to requested columns: edge_id, road_name, geometry, road_type, speed, capacity, traffic_density
    output_rows = []
    for edge in selected_edges:
        speed_val = _safe_float(edge.get("speed"), _safe_float(edge.get("current_speed"), 30.0))
        density_val = _safe_float(edge.get("congestion_score"), _safe_float(edge.get("current_density"), 0.0))
        output_rows.append({
            "edge_id": edge.get("edge_id"),
            "road_name": edge.get("road_name"),
            "geometry": edge.get("geometry"),
            "road_type": edge.get("road_type"),
            "speed": speed_val,
            "capacity": edge.get("capacity"),
            "traffic_density": density_val
        })

    fieldnames = ["edge_id", "road_name", "geometry", "road_type", "speed", "capacity", "traffic_density"]
    with open(COMMAND_CENTER_ROADS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    logging.info("Command Center roads written to %s with %d rows.", COMMAND_CENTER_ROADS_CSV, len(output_rows))
    return COMMAND_CENTER_ROADS_CSV

def load_or_create_command_center_roads(limit: int = 3500) -> Path:
    if COMMAND_CENTER_ROADS_CSV.exists():
        return COMMAND_CENTER_ROADS_CSV
    return build_command_center_roads(limit=limit)
