from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm

try:
    from .graph_builder import GRAPH_PATH, build_bangalore_graph, load_bangalore_graph
except ImportError:  # pragma: no cover - direct script execution fallback
    from graph_builder import GRAPH_PATH, build_bangalore_graph, load_bangalore_graph


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
INPUT_PATH = DATA_DIR / "astram.csv"
OUTPUT_PATH = DATA_DIR / "astram_mapped_events.csv"


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_events() -> pd.DataFrame:
    if INPUT_PATH.exists():
        return pd.read_csv(INPUT_PATH, dtype="string", encoding="utf-8-sig")
    fallback = BASE_DIR / "astram.csv"
    if fallback.exists():
        return pd.read_csv(fallback, dtype="string", encoding="utf-8-sig")
    raise FileNotFoundError("Could not find astram.csv in data/ or workspace root")


def map_events() -> pd.DataFrame:
    setup_logging()
    graph = load_bangalore_graph() if GRAPH_PATH.exists() else build_bangalore_graph()
    events = load_events()
    events["latitude"] = pd.to_numeric(events["latitude"], errors="coerce")
    events["longitude"] = pd.to_numeric(events["longitude"], errors="coerce")

    valid_mask = events["latitude"].notna() & events["longitude"].notna()
    valid_events = events.loc[valid_mask].copy()
    invalid_events = events.loc[~valid_mask].copy()

    if not valid_events.empty:
        projected_points = gpd.GeoSeries(
            [Point(lon, lat) for lat, lon in zip(valid_events["latitude"], valid_events["longitude"])],
            crs="EPSG:4326",
        ).to_crs(graph.graph["crs"])
        xs = projected_points.x.to_numpy()
        ys = projected_points.y.to_numpy()

        nearest_edges = ox.distance.nearest_edges(graph, xs, ys)
        nearest_nodes = ox.distance.nearest_nodes(graph, xs, ys)

        mapped_rows = []
        for idx, row in tqdm(list(valid_events.iterrows()), total=len(valid_events), desc="Mapping Astram events"):
            local_index = valid_events.index.get_loc(idx)
            edge_lookup = nearest_edges[local_index]
            if len(edge_lookup) == 3:
                u, v, key = edge_lookup
            else:
                u, v, key = edge_lookup[0], edge_lookup[1], 0
            edge_data = graph.get_edge_data(u, v, key)
            point = projected_points.iloc[local_index]
            edge_geom = edge_data.get("geometry")
            distance_m = float(point.distance(edge_geom)) if edge_geom is not None else 0.0

            mapped_rows.append({
                **row.to_dict(),
                "nearest_edge_id": edge_data.get("edge_id"),
                "nearest_node_id": nearest_nodes[local_index],
                "distance_from_road_meter": distance_m,
                "road_name": edge_data.get("road_name", "Unknown"),
                "road_type": edge_data.get("road_type", "unclassified"),
                "capacity": edge_data.get("capacity", 0),
            })
    else:
        mapped_rows = []

    if not invalid_events.empty:
        for _, row in invalid_events.iterrows():
            mapped_rows.append({**row.to_dict(), "nearest_edge_id": None, "nearest_node_id": None, "distance_from_road_meter": None, "road_name": None, "road_type": None, "capacity": None})

    mapped_df = pd.DataFrame(mapped_rows)
    mapped_df.to_csv(OUTPUT_PATH, index=False)
    logging.info("Saved mapped Astram events to %s", OUTPUT_PATH)
    return mapped_df


if __name__ == "__main__":
    map_events()
