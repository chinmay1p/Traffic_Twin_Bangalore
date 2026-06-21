from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
GRAPH_PATH = DATA_DIR / "bangalore_graph.graphml"
NODES_PATH = DATA_DIR / "nodes.csv"
EDGES_PATH = DATA_DIR / "edges.csv"
ROAD_SEGMENTS_PATH = DATA_DIR / "road_segments.csv"

PLACE_NAME = "Bengaluru, Karnataka, India"
NETWORK_TYPE = "drive"

SPEED_BY_HIGHWAY = {
    "motorway": 70,
    "motorway_link": 70,
    "trunk": 70,
    "trunk_link": 70,
    "primary": 50,
    "primary_link": 50,
    "secondary": 40,
    "secondary_link": 40,
    "tertiary": 30,
    "tertiary_link": 30,
    "residential": 20,
    "living_street": 20,
    "service": 20,
    "unclassified": 30,
    "road": 30,
}

LANES_BY_HIGHWAY = {
    "motorway": 4,
    "motorway_link": 4,
    "trunk": 4,
    "trunk_link": 4,
    "primary": 4,
    "primary_link": 4,
    "secondary": 3,
    "secondary_link": 3,
    "tertiary": 2,
    "tertiary_link": 2,
    "residential": 1,
    "living_street": 1,
    "service": 1,
    "unclassified": 1,
    "road": 1,
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def normalize_highway(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unclassified"
    if isinstance(value, (list, tuple, set)):
        return str(next(iter(value), "unclassified"))
    return str(value)


def estimate_speed(highway: Any, maxspeed: Any = None) -> float:
    if maxspeed not in (None, "", np.nan):
        try:
            if isinstance(maxspeed, (list, tuple)) and maxspeed:
                maxspeed = maxspeed[0]
            maxspeed_text = str(maxspeed).lower().replace("km/h", "").replace("kph", "").strip()
            if maxspeed_text.isdigit():
                return float(maxspeed_text)
            parsed = float(maxspeed_text.split(";")[0].split(",")[0])
            return parsed
        except Exception:
            pass
    highway_key = normalize_highway(highway)
    return float(SPEED_BY_HIGHWAY.get(highway_key, 30))


def estimate_lanes(highway: Any, lanes: Any = None) -> int:
    if lanes not in (None, "", np.nan):
        try:
            if isinstance(lanes, (list, tuple)) and lanes:
                lanes = lanes[0]
            lanes_text = str(lanes).strip()
            if ";" in lanes_text:
                lanes_text = lanes_text.split(";")[0]
            return max(1, int(float(lanes_text)))
        except Exception:
            pass
    highway_key = normalize_highway(highway)
    return int(LANES_BY_HIGHWAY.get(highway_key, 1))


def ensure_directories() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)


def download_graph() -> nx.MultiDiGraph:
    if GRAPH_PATH.exists():
        logging.info("Loading cached graph from %s", GRAPH_PATH)
        return ox.load_graphml(GRAPH_PATH)

    logging.info("Downloading road network for %s", PLACE_NAME)
    ox.settings.log_console = False
    ox.settings.use_cache = True

    graph = ox.graph_from_place(
        PLACE_NAME,
        network_type=NETWORK_TYPE,
        simplify=False,
        retain_all=True,
        truncate_by_edge=True,
    )
    graph = ox.simplify_graph(graph)
    graph = ox.project_graph(graph)

    for node_id, data in graph.nodes(data=True):
        data["latitude"] = data.get("y")
        data["longitude"] = data.get("x")

    for u, v, key, data in graph.edges(keys=True, data=True):
        geometry = data.get("geometry")
        if geometry is not None:
            data["length"] = float(geometry.length)
        else:
            x1, y1 = graph.nodes[u]["x"], graph.nodes[u]["y"]
            x2, y2 = graph.nodes[v]["x"], graph.nodes[v]["y"]
            data["length"] = float(math.hypot(x2 - x1, y2 - y1))

    logging.info("Graph downloaded and projected: %s nodes / %s edges", len(graph.nodes), len(graph.edges))
    return graph


def enrich_edges(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    logging.info("Enriching edge attributes")

    for _, _, _, data in tqdm(graph.edges(keys=True, data=True), total=graph.number_of_edges(), desc="Edge enrichment"):
        highway = data.get("highway")
        if isinstance(highway, list) and highway:
            highway = highway[0]
        road_type = normalize_highway(highway)
        speed = estimate_speed(highway, data.get("maxspeed"))
        lanes = estimate_lanes(highway, data.get("lanes"))
        length_meter = float(data.get("length", 0.0) or 0.0)
        travel_time_seconds = (length_meter / 1000.0) / speed * 3600.0 if speed > 0 else math.inf
        capacity = int(lanes * 1800)

        data["road_name"] = data.get("name") if data.get("name") is not None else "Unknown"
        data["road_type"] = road_type
        data["length_meter"] = length_meter
        data["speed_kmph"] = float(speed)
        data["travel_time_seconds"] = float(travel_time_seconds)
        data["lanes"] = int(lanes)
        data["capacity"] = capacity
        data["current_speed"] = float(speed)
        data["current_density"] = 0.0
        data["current_flow"] = 0.0
        data["congestion_score"] = 0.0
        data["edge_id"] = f"{data.get('osmid', '')}_{data.get('u', '')}_{data.get('v', '')}_{data.get('key', '')}"

    return graph


def project_edge_geometry(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    for u, v, key, data in graph.edges(keys=True, data=True):
        if "geometry" not in data or data["geometry"] is None:
            from shapely.geometry import LineString

            x1, y1 = graph.nodes[u]["x"], graph.nodes[u]["y"]
            x2, y2 = graph.nodes[v]["x"], graph.nodes[v]["y"]
            data["geometry"] = LineString([(x1, y1), (x2, y2)])
    return graph


def save_graph(graph: nx.MultiDiGraph) -> None:
    logging.info("Saving graph to %s", GRAPH_PATH)
    ox.save_graphml(graph, GRAPH_PATH)


def export_nodes_edges(graph: nx.MultiDiGraph) -> None:
    logging.info("Exporting nodes and edges CSV files")
    nodes = []
    for node_id, data in graph.nodes(data=True):
        latitude = data.get("latitude", data.get("y"))
        longitude = data.get("longitude", data.get("x"))
        nodes.append({"node_id": node_id, "latitude": latitude, "longitude": longitude})
    pd.DataFrame(nodes).to_csv(NODES_PATH, index=False)

    edges = []
    for u, v, key, data in graph.edges(keys=True, data=True):
        geometry = data.get("geometry")
        geometry_text = geometry.wkt if geometry is not None else ""
        edges.append({
            "edge_id": data.get("edge_id"),
            "u": u,
            "v": v,
            "road_name": data.get("road_name", "Unknown"),
            "road_type": data.get("road_type", "unclassified"),
            "length": data.get("length_meter", data.get("length", 0.0)),
            "speed": data.get("speed_kmph", 0.0),
            "travel_time": data.get("travel_time_seconds", 0.0),
            "lanes": data.get("lanes", 1),
            "capacity": data.get("capacity", 0),
            "current_speed": data.get("current_speed", data.get("speed_kmph", 0.0)),
            "current_density": data.get("current_density", 0.0),
            "current_flow": data.get("current_flow", 0.0),
            "congestion_score": data.get("congestion_score", 0.0),
            "geometry": geometry_text,
        })

    edges_df = pd.DataFrame(edges)
    edges_df.to_csv(EDGES_PATH, index=False)
    edges_df.to_csv(ROAD_SEGMENTS_PATH, index=False)


def build_bangalore_graph() -> nx.MultiDiGraph:
    ensure_directories()
    graph = download_graph()
    graph = enrich_edges(graph)
    graph = project_edge_geometry(graph)
    save_graph(graph)
    export_nodes_edges(graph)
    return graph


def load_bangalore_graph() -> nx.MultiDiGraph:
    if not GRAPH_PATH.exists():
        return build_bangalore_graph()
    graph = ox.load_graphml(GRAPH_PATH)
    for _, _, _, data in graph.edges(keys=True, data=True):
        for field in ["length_meter", "speed_kmph", "travel_time_seconds", "lanes", "capacity", "current_speed", "current_density", "current_flow", "congestion_score"]:
            if field in data and data[field] is not None and data[field] != "":
                try:
                    if field in {"lanes", "capacity"}:
                        data[field] = int(float(data[field]))
                    else:
                        data[field] = float(data[field])
                except Exception:
                    pass
    return graph


if __name__ == "__main__":
    setup_logging()
    build_bangalore_graph()