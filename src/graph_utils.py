from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import networkx as nx
import geopandas as gpd
import osmnx as ox

try:
    from .graph_builder import GRAPH_PATH, build_bangalore_graph, load_bangalore_graph
except ImportError:  # pragma: no cover - direct script execution fallback
    from graph_builder import GRAPH_PATH, build_bangalore_graph, load_bangalore_graph


GRAPH = None
ORIGINAL_EDGE_VALUES: dict[tuple[int, int, int], dict[str, float]] = {}


def get_graph() -> nx.MultiDiGraph:
    global GRAPH
    if GRAPH is None:
        GRAPH = load_bangalore_graph() if GRAPH_PATH.exists() else build_bangalore_graph()
    return GRAPH


def _project_point(lat: float, lng: float, graph: nx.MultiDiGraph):
    point = gpd.GeoSeries([gpd.points_from_xy([lng], [lat])[0]], crs="EPSG:4326")
    projected = point.to_crs(graph.graph["crs"])
    return projected.iloc[0]


def get_nearest_edge(lat: float, lng: float):
    graph = get_graph()
    projected_point = _project_point(lat, lng, graph)
    edge = ox.distance.nearest_edges(graph, projected_point.x, projected_point.y)
    if isinstance(edge, tuple) and len(edge) == 3:
        u, v, key = edge
    else:
        u, v, key = edge[0], edge[1], edge[2] if len(edge) > 2 else 0
    data = graph.get_edge_data(u, v, key)
    return {
        "edge_id": data.get("edge_id"),
        "u": u,
        "v": v,
        "key": key,
        "road_name": data.get("road_name", "Unknown"),
        "road_type": data.get("road_type", "unclassified"),
        "capacity": data.get("capacity", 0),
        "distance": None,
    }


def block_road(edge_id: str) -> bool:
    graph = get_graph()
    for u, v, key, data in graph.edges(keys=True, data=True):
        if data.get("edge_id") == edge_id:
            ORIGINAL_EDGE_VALUES[(u, v, key)] = {
                "capacity": data.get("capacity", 0),
                "travel_time_seconds": data.get("travel_time_seconds", math.inf),
            }
            data["capacity"] = 0
            data["travel_time_seconds"] = math.inf
            data["current_flow"] = 0.0
            data["congestion_score"] = 1.0
            return True
    return False


def restore_road(edge_id: str) -> bool:
    graph = get_graph()
    for u, v, key, data in graph.edges(keys=True, data=True):
        if data.get("edge_id") == edge_id:
            original = ORIGINAL_EDGE_VALUES.get((u, v, key))
            if original:
                data["capacity"] = original["capacity"]
                data["travel_time_seconds"] = original["travel_time_seconds"]
                data["congestion_score"] = 0.0
                return True
    return False


def get_neighbors(edge_id: str, radius: int):
    graph = get_graph()
    target = None
    for u, v, key, data in graph.edges(keys=True, data=True):
        if data.get("edge_id") == edge_id:
            target = (u, v, key)
            break
    if target is None:
        return []

    u0, v0, _ = target
    neighbor_edges = []
    visited_nodes = {u0, v0}
    frontier = {u0, v0}
    for _ in range(max(1, radius)):
        next_frontier = set()
        for node in frontier:
            next_frontier.update(graph.predecessors(node))
            next_frontier.update(graph.successors(node))
        next_frontier -= visited_nodes
        visited_nodes.update(next_frontier)
        frontier = next_frontier
    for u, v, key, data in graph.edges(keys=True, data=True):
        if u in visited_nodes or v in visited_nodes:
            neighbor_edges.append({
                "edge_id": data.get("edge_id"),
                "u": u,
                "v": v,
                "road_name": data.get("road_name", "Unknown"),
                "road_type": data.get("road_type", "unclassified"),
                "capacity": data.get("capacity", 0),
            })
    return neighbor_edges[: max(0, radius)]


def calculate_shortest_route(source: int, destination: int):
    graph = get_graph()
    return nx.shortest_path(graph, source=source, target=destination, weight="travel_time_seconds")


def get_nearby_roads(lat: float, lng: float, radius_m: float = 500.0) -> list[dict[str, Any]]:
    """
    Finds all road segments within a given geographic radius of a coordinate.
    Uses a fast bounding box pre-filtering of nodes to handle performance on the Bangalore graph.
    """
    graph = get_graph()
    projected_point = _project_point(lat, lng, graph)
    
    # 0.001 degrees is ~111 meters. 500m is ~0.0045 degrees.
    lat_delta = radius_m / 111000.0
    lng_delta = radius_m / (111000.0 * math.cos(math.radians(lat)))
    
    nearby_nodes = set()
    for node, ndata in graph.nodes(data=True):
        nlat = ndata.get("y", ndata.get("latitude"))
        nlng = ndata.get("x", ndata.get("longitude"))
        if nlat is not None and nlng is not None:
            if abs(nlat - lat) <= lat_delta and abs(nlng - lng) <= lng_delta:
                nearby_nodes.add(node)
                
    nearby_edges = []
    seen_edges = set()
    for u, v, key, data in graph.edges(keys=True, data=True):
        if u in nearby_nodes or v in nearby_nodes:
            eid = data.get("edge_id")
            if eid in seen_edges:
                continue
            geom = data.get("geometry")
            if geom:
                dist = projected_point.distance(geom)
                if dist <= radius_m:
                    seen_edges.add(eid)
                    nearby_edges.append({
                        "edge_id": eid,
                        "u": u,
                        "v": v,
                        "key": key,
                        "road_name": data.get("road_name", "Unknown"),
                        "road_type": data.get("road_type", "unclassified"),
                        "capacity": data.get("capacity", 0),
                        "distance": float(dist),
                    })
                    
    nearby_edges.sort(key=lambda x: x["distance"])
    return nearby_edges

