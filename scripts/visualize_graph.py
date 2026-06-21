from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import folium
from folium.plugins import MarkerCluster
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from src.graph_builder import GRAPH_PATH, build_bangalore_graph, load_bangalore_graph
except ImportError:  # pragma: no cover - direct script execution fallback
    from graph_builder import GRAPH_PATH, build_bangalore_graph, load_bangalore_graph


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = BASE_DIR / "outputs" / "bangalore_graph.html"
MAPPED_EVENTS_PATH = BASE_DIR / "data" / "astram_mapped_events.csv"

# Vibrant neon color palette for dark mode
ROAD_COLORS = {
    "motorway": "#ff0055",        # Neon Pink
    "motorway_link": "#ff0055",
    "trunk": "#ff5500",           # Neon Orange/Red
    "trunk_link": "#ff5500",
    "primary": "#ffcc00",         # Glowing Yellow
    "primary_link": "#ffcc00",
    "secondary": "#00ffcc",       # Neon Teal
    "secondary_link": "#00ffcc",
    "tertiary": "#0099ff",        # Bright Cyan-Blue
    "tertiary_link": "#0099ff",
    "residential": "#4a5a6a",     # Muted Slate Blue/Gray
    "living_street": "#5a6a7a",
    "service": "#7b2cbf",         # Deep Electric Purple
    "unclassified": "#5a5a5a",    # Mid Gray
    "road": "#5a5a5a",
}

EVENT_COLORS = {
    "accident": "#d73027",          # Red
    "vehicle_breakdown": "#fc8d59",  # Orange
    "water_logging": "#4575b4",     # Blue
    "pot_holes": "#fee08b",         # Yellow
    "construction": "#984ea3",      # Purple
    "others": "#999999",            # Gray
}


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def escape_js_html(val: Any) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    s = str(val).strip()
    s = s.replace("<", "&lt;").replace(">", "&gt;")
    s = s.replace("\\", "\\\\")
    s = s.replace("'", "&#39;")
    s = s.replace('"', "&quot;")
    s = s.replace("\n", " ").replace("\r", " ")
    return s


def build_map() -> folium.Map:
    setup_logging()
    logging.info("Starting visualization builder")

    # Load Bangalore Graph
    graph = load_bangalore_graph() if GRAPH_PATH.exists() else build_bangalore_graph()
    
    # Project to EPSG:4326 (WGS84) for mapping
    logging.info("Projecting graph to EPSG:4326 (WGS84)...")
    web_graph = graph if str(graph.graph.get("crs", "")).lower() in {"epsg:4326", "crs84"} else __import__("osmnx").project_graph(graph, to_crs="EPSG:4326")

    total_nodes = len(web_graph.nodes)
    total_edges = len(web_graph.edges)
    logging.info("Graph loaded: %s nodes, %s edges", total_nodes, total_edges)

    # Calculate map center
    node_points = [(data["y"], data["x"]) for _, data in web_graph.nodes(data=True)]
    center_lat = sum(lat for lat, _ in node_points) / len(node_points)
    center_lng = sum(lng for _, lng in node_points) / len(node_points)
    
    # Initialize Folium Map
    fmap = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=12,
        tiles="CartoDB dark_matter",
        prefer_canvas=True
    )

    # Initialize Layers & Groups
    logging.info("Preparing road layers...")
    
    # Define groups by significance to balance performance and detail
    road_groups = {
        "Highways & Trunks": {
            "types": {"motorway", "motorway_link", "trunk", "trunk_link"},
            "show": True,
            "weight": 3.5,
            "opacity": 0.9,
            "features": []
        },
        "Primary Roads": {
            "types": {"primary", "primary_link"},
            "show": True,
            "weight": 2.5,
            "opacity": 0.85,
            "features": []
        },
        "Secondary Roads": {
            "types": {"secondary", "secondary_link"},
            "show": True,
            "weight": 2.0,
            "opacity": 0.8,
            "features": []
        },
        "Tertiary Roads": {
            "types": {"tertiary", "tertiary_link"},
            "show": False,  # Off by default, toggleable
            "weight": 1.5,
            "opacity": 0.7,
            "features": []
        }
    }

    road_centers = {}  # To store search query metadata {road_name: {coords, speeds, edge_ids}}

    # Build GeoJSON collections for roads
    # First pass: collect ALL named roads into search index regardless of type
    for u, v, key, data in tqdm(web_graph.edges(keys=True, data=True), desc="Indexing roads for search"):
        geom = data.get("geometry")
        if geom is None:
            continue
        road_name = data.get("road_name", "Unknown")
        if isinstance(road_name, list):
            road_name = " / ".join(str(n) for n in road_name)
        elif not isinstance(road_name, str):
            road_name = str(road_name)
        if road_name and road_name != "Unknown" and road_name.lower() != "nan":
            coords = list(geom.coords)
            if coords:
                mid_idx = len(coords) // 2
                mid_coord = coords[mid_idx]
                speed = float(data.get("speed_kmph", data.get("speed", 30.0)))
                edge_id = str(data.get("edge_id", ""))
                # mid_coord is (lon, lat) in WGS84
                if road_name not in road_centers:
                    road_centers[road_name] = {
                        "coords": [],
                        "speeds": [],
                        "edge_ids": []
                    }
                road_centers[road_name]["coords"].append((mid_coord[1], mid_coord[0]))
                road_centers[road_name]["speeds"].append(speed)
                if edge_id:
                    road_centers[road_name]["edge_ids"].append(edge_id)


    # Second pass: build GeoJSON layers for major roads only (for performance)
    for u, v, key, data in tqdm(web_graph.edges(keys=True, data=True), desc="Processing major road layers"):
        road_type = data.get("road_type", "unclassified")
        geom = data.get("geometry")
        if geom is None:
            continue
        
        # Determine which group it belongs to
        assigned_group = None
        for group_name, config in road_groups.items():
            if road_type in config["types"]:
                assigned_group = config
                break
        
        if assigned_group is None:
            continue  # Skip local and residential roads to optimize rendering performance

        coords = list(geom.coords)
        
        # Prepare properties for tooltip
        road_name = data.get("road_name", "Unknown")
        if isinstance(road_name, list):
            road_name = " / ".join(str(n) for n in road_name)
        elif not isinstance(road_name, str):
            road_name = str(road_name)

        properties = {
            "road_name": road_name,
            "road_type": road_type,
            "length_meter": round(float(data.get("length_meter", data.get("length", 0.0))), 1),
            "speed_kmph": round(float(data.get("speed_kmph", 30.0)), 1),
            "travel_time_seconds": round(float(data.get("travel_time_seconds", 0.0)), 1),
            "lanes": int(data.get("lanes", 1)),
            "capacity": int(data.get("capacity", 1800)),
            "congestion_score": round(float(data.get("congestion_score", 0.0)), 2),
            "edge_id": data.get("edge_id", ""),
        }
        
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords
            },
            "properties": properties
        }
        
        assigned_group["features"].append(feature)

    # Add Road Layers to Map
    for group_name, config in road_groups.items():
        if not config["features"]:
            continue
            
        fc = {
            "type": "FeatureCollection",
            "features": config["features"]
        }
        
        fg = folium.FeatureGroup(name=group_name, show=config["show"])
        
        # Style function for this group
        def style_fn(feat, w=config["weight"], op=config["opacity"]):
            rt = feat["properties"]["road_type"]
            col = ROAD_COLORS.get(rt, "#999999")
            return {
                "color": col,
                "weight": w,
                "opacity": op
            }

        # Add GeoJson to FeatureGroup
        folium.GeoJson(
            fc,
            style_function=style_fn,
            tooltip=folium.GeoJsonTooltip(
                fields=["road_name", "road_type", "speed_kmph", "capacity", "congestion_score"],
                aliases=["Road Name:", "Road Type:", "Speed Limit (km/h):", "Capacity (veh/h):", "Congestion Score:"],
                localize=True,
                sticky=True
            )
        ).add_to(fg)
        
        fg.add_to(fmap)

    # Process Junction Nodes (deg >= 3 in the major network)
    logging.info("Processing network nodes...")
    node_cluster = MarkerCluster(name="Network Junctions (Nodes)", show=False)
    
    # Calculate degrees on sub-graph of major roads to locate real intersections
    major_types = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link"}
    major_edges = [(u, v, k) for u, v, k, d in web_graph.edges(keys=True, data=True) if d.get("road_type") in major_types]
    
    if major_edges:
        sub_g = web_graph.edge_subgraph(major_edges)
        degrees = dict(sub_g.degree())
        junctions = [n for n, deg in degrees.items() if deg >= 3]
        
        for node_id in tqdm(junctions[:8000], desc="Adding junctions"):
            data = web_graph.nodes[node_id]
            lat, lng = data["y"], data["x"]
            folium.CircleMarker(
                location=[lat, lng],
                radius=2.0,
                color="#00ffcc",
                fill=True,
                fill_color="#00ffcc",
                fill_opacity=0.8,
                opacity=0.9,
                tooltip=f"Junction Node: {node_id}"
            ).add_to(node_cluster)
            
        node_cluster.add_to(fmap)

    # Map Astram Events
    total_events_mapped = 0
    if MAPPED_EVENTS_PATH.exists():
        logging.info("Mapping Astram events from %s...", MAPPED_EVENTS_PATH)
        events_df = pd.read_csv(MAPPED_EVENTS_PATH)
        events_df["latitude"] = pd.to_numeric(events_df["latitude"], errors="coerce")
        events_df["longitude"] = pd.to_numeric(events_df["longitude"], errors="coerce")
        events_df = events_df.dropna(subset=["latitude", "longitude"])
        
        event_cluster = MarkerCluster(name="Astram Incidents (Events)", show=True)
        total_events_mapped = len(events_df)

        for _, row in tqdm(events_df.iterrows(), total=len(events_df), desc="Adding events"):
            lat = float(row["latitude"])
            lng = float(row["longitude"])
            cause = str(row.get("event_cause", "others"))
            color = EVENT_COLORS.get(cause, "#999999")
            
            # Formulate detailed popup HTML
            road_name = row.get("road_name")
            road_name = road_name if pd.notna(road_name) else "Unknown"
            distance_m = row.get("distance_from_road_meter")
            distance_str = f"{round(float(distance_m), 1)} m" if pd.notna(distance_m) else "N/A"
            status = str(row.get("status", "Active")).upper()
            status_color = "#4caf50" if "RESOLV" in status or "CLOSE" in status else "#f44336"
            
            # Escape dynamic strings to prevent syntax break in Folium's JS engine
            safe_cause = escape_js_html(cause.replace('_', ' ').title())
            safe_address = escape_js_html(row.get('address', 'N/A'))
            safe_road_name = escape_js_html(road_name)
            safe_distance = escape_js_html(distance_str)
            safe_status = escape_js_html(status)
            safe_id = escape_js_html(row.get('id', 'N/A'))

            popup_html = f"""
            <div style="font-family: 'Inter', sans-serif; font-size: 12px; color: #e0e0e0; background-color: #1a1a24; border-radius: 8px; padding: 12px; width: 220px; box-shadow: 0 4px 15px rgba(0,0,0,0.5);">
                <h4 style="margin: 0 0 8px 0; color: {color}; font-weight: 700; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 4px;">
                    {safe_cause}
                </h4>
                <div style="margin-bottom: 6px;"><b>Location:</b> <span style="color: #bbb;">{safe_address}</span></div>
                <div style="margin-bottom: 6px;"><b>Nearest Road:</b> <span style="color: #bbb;">{safe_road_name}</span></div>
                <div style="margin-bottom: 6px;"><b>Distance:</b> <span style="color: #bbb;">{safe_distance}</span></div>
                <div style="margin-bottom: 6px;"><b>Status:</b> <span style="color: {status_color}; font-weight: 700;">{safe_status}</span></div>
                <div style="font-size: 10px; color: #888; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 4px; margin-top: 6px;">
                    ID: {safe_id}
                </div>
            </div>
            """
            
            folium.CircleMarker(
                location=[lat, lng],
                radius=4.5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                opacity=0.9,
                popup=folium.Popup(popup_html, max_width=250)
            ).add_to(event_cluster)
            
        event_cluster.add_to(fmap)

    # Standard Folium Layer Control
    folium.LayerControl(collapsed=False).add_to(fmap)

    # Build the search index JS array (Top 3000 roads by segment count)
    sorted_roads = sorted(road_centers.items(), key=lambda x: len(x[1]["coords"]), reverse=True)[:3000]
    js_road_data = []
    for name, info in sorted_roads:
        coords = info["coords"]
        avg_lat = sum(c[0] for c in coords) / len(coords)
        avg_lng = sum(c[1] for c in coords) / len(coords)
        avg_speed = sum(info["speeds"]) / len(info["speeds"]) if info["speeds"] else 30.0
        edge_id = info["edge_ids"][0] if info["edge_ids"] else ""
        js_road_data.append({
            "name": name,
            "lat": round(avg_lat, 6),
            "lng": round(avg_lng, 6),
            "speed": round(avg_speed, 1),
            "edge_id": edge_id
        })

    # Save search index
    search_index_path = BASE_DIR / "data" / "road_search_index.json"
    search_index_path.parent.mkdir(exist_ok=True)
    with open(search_index_path, "w", encoding="utf-8") as f:
        json.dump(js_road_data, f, ensure_ascii=False, indent=2)


    # Prepare stats for HTML sidebar
    stats = {
        "nodes": f"{total_nodes:,}",
        "edges": f"{total_edges:,}",
        "events": f"{total_events_mapped:,}"
    }

    # Inject CSS / HTML Glassmorphism Sidebar & Google Fonts
    css_injection = """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        /* Force full-viewport layout — critical for iframe embedding */
        html, body {
            margin: 0 !important;
            padding: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            overflow: hidden !important;
            background: #0a0a12;
        }

        /* Make the Folium map container fill the entire iframe absolutely */
        div.folium-map {
            position: absolute !important;
            top: 0 !important;
            left: 0 !important;
            right: 0 !important;
            bottom: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
        }

        /* Completely remove sidebar from layout flow */
        #sidebar {
            position: absolute !important;
            top: -9999px !important;
            left: -9999px !important;
            width: 0 !important;
            height: 0 !important;
            overflow: hidden !important;
            visibility: hidden !important;
            pointer-events: none !important;
        }
        
        #sidebar::-webkit-scrollbar {
            width: 6px;
        }
        #sidebar::-webkit-scrollbar-track {
            background: rgba(0,0,0,0.1);
        }
        #sidebar::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 3px;
        }
        
        .sidebar-header h2 {
            margin: 0;
            font-size: 20px;
            font-weight: 700;
            background: linear-gradient(45deg, #ff007f, #00ffcc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }
        .tagline {
            margin: 6px 0 0 0;
            font-size: 10px;
            color: #718096;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-weight: 600;
        }
        
        .stats-panel {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            background: rgba(255, 255, 255, 0.04);
            padding: 12px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.06);
        }
        .stat-card {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }
        .stat-value {
            font-size: 14px;
            font-weight: 700;
            color: #ffffff;
        }
        .stat-label {
            font-size: 9px;
            color: #a0aec0;
            margin-top: 3px;
            text-transform: uppercase;
        }
        
        .sidebar-section h3 {
            margin: 0 0 10px 0;
            font-size: 13px;
            font-weight: 600;
            color: #ffffff;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding-bottom: 6px;
        }
        
        .legend-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
            font-size: 11px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .legend-color {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 8px currentColor;
        }
        
        .event-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }
        .event-badge {
            font-size: 9px;
            padding: 4px 8px;
            border-radius: 6px;
            font-weight: 700;
            letter-spacing: 0.2px;
        }
        .badge-accident { background: rgba(215, 48, 39, 0.15); color: #ff5252; border: 1px solid rgba(215, 48, 39, 0.4); }
        .badge-breakdown { background: rgba(252, 141, 89, 0.15); color: #ff9f43; border: 1px solid rgba(252, 141, 89, 0.4); }
        .badge-water { background: rgba(69, 117, 180, 0.15); color: #54a0ff; border: 1px solid rgba(69, 117, 180, 0.4); }
        .badge-pothole { background: rgba(254, 224, 139, 0.15); color: #ffd35c; border: 1px solid rgba(254, 224, 139, 0.4); }
        .badge-construction { background: rgba(152, 78, 163, 0.15); color: #c582ff; border: 1px solid rgba(152, 78, 163, 0.4); }
        
        .search-box {
            position: relative;
        }
        #road-search {
            width: 100%;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 10px 14px;
            border-radius: 8px;
            color: #ffffff;
            font-size: 12px;
            box-sizing: border-box;
            outline: none;
            transition: all 0.3s;
        }
        #road-search:focus {
            border-color: #00ffcc;
            background: rgba(255, 255, 255, 0.08);
            box-shadow: 0 0 12px rgba(0, 255, 204, 0.25);
        }
        #search-results {
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background: #13131c;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-top: none;
            border-radius: 0 0 8px 8px;
            margin: 0;
            padding: 0;
            list-style: none;
            max-height: 180px;
            overflow-y: auto;
            z-index: 1001;
            display: none;
            box-shadow: 0 8px 24px rgba(0,0,0,0.5);
        }
        #search-results li {
            padding: 10px 14px;
            font-size: 11px;
            cursor: pointer;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            color: #cbd5e0;
            transition: all 0.2s;
        }
        #search-results li:hover {
            background: rgba(0, 255, 204, 0.15);
            color: #00ffcc;
            padding-left: 18px;
        }
        
        /* Reposition default Leaflet controls to avoid sidebar overlay */
        .leaflet-top.leaflet-left {
            /* left: 350px !important; */
        }
        .leaflet-bottom.leaflet-left {
            /* left: 350px !important; */
        }
    </style>
    """

    html_injection = """
    <div id="sidebar">
        <div class="sidebar-header">
            <h2>Bangalore Traffic Twin</h2>
            <div class="tagline">Foundation Module Dashboard</div>
        </div>
        
        <div class="stats-panel">
            <div class="stat-card">
                <span class="stat-value">__NODES__</span>
                <span class="stat-label">Nodes</span>
            </div>
            <div class="stat-card">
                <span class="stat-value">__EDGES__</span>
                <span class="stat-label">Edges</span>
            </div>
            <div class="stat-card">
                <span class="stat-value">__EVENTS__</span>
                <span class="stat-label">Events</span>
            </div>
        </div>
        
        <div class="sidebar-section">
            <h3>Network Layers</h3>
            <div class="legend-list">
                <div class="legend-item"><span class="legend-color" style="background: #ff0055; color: #ff0055;"></span>Motorways (70 km/h)</div>
                <div class="legend-item"><span class="legend-color" style="background: #ffcc00; color: #ffcc00;"></span>Primary Roads (50 km/h)</div>
                <div class="legend-item"><span class="legend-color" style="background: #00ffcc; color: #00ffcc;"></span>Secondary Roads (40 km/h)</div>
                <div class="legend-item"><span class="legend-color" style="background: #0099ff; color: #0099ff;"></span>Tertiary Roads (30 km/h)</div>
                <div class="legend-item" style="font-size: 9px; color: #888; font-style: italic; margin-top: 4px;">* Note: Local/Residential roads (330k segments) are omitted from this map to ensure smooth loading.</div>
            </div>
        </div>
        
        <div class="sidebar-section">
            <h3>Astram Incident Key</h3>
            <div class="event-legend">
                <div class="event-badge badge-accident">Accident</div>
                <div class="event-badge badge-breakdown">Breakdown</div>
                <div class="event-badge badge-water">Flood</div>
                <div class="event-badge badge-pothole">Pothole</div>
                <div class="event-badge badge-construction">Work</div>
            </div>
        </div>
        
        <div class="sidebar-section">
            <h3>Query Road Network</h3>
            <div class="search-box">
                <input type="text" id="road-search" placeholder="Type road name... (e.g. Outer Ring Road)" onkeyup="searchRoads()">
                <ul id="search-results"></ul>
            </div>
        </div>

        <div class="sidebar-section">
            <h3>Twin Simulator & Query</h3>
            <div id="sim-status">
                <p style="font-size: 11px; color: #a0aec0; margin: 0 0 10px 0;">Select a road from search or click any road segment on the map to query speeds or simulate accidents.</p>
            </div>
            <div id="sim-controls" style="display: none; flex-direction: column; gap: 8px;">
                <div style="font-size: 12px; font-weight: 700; color: #00ffcc;" id="selected-road-name">Road: None</div>
                <div style="font-size: 11px; color: #cbd5e0;">Normal Speed: <span id="selected-road-speed">30</span> km/h</div>
                
                <!-- ML Forecasts Card -->
                <div id="ml-forecasts-panel" style="display: none; margin-top: 4px; font-size: 11px; background: rgba(0, 255, 204, 0.05); padding: 8px; border-radius: 6px; border: 1px solid rgba(0, 255, 204, 0.15); width: 100%; box-sizing: border-box;">
                    <b style="color: #00ffcc; font-size: 11px; display: block; margin-bottom: 4px;">ML Traffic Forecast (Congestion):</b>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                        <span>Current:</span>
                        <span id="fc-current" style="font-weight: bold; color: #fff;">-</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                        <span>In 15 Min:</span>
                        <span id="fc-15min" style="font-weight: bold; color: #ff9f43;">-</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                        <span>In 30 Min:</span>
                        <span id="fc-30min" style="font-weight: bold; color: #ff5252;">-</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span>In 60 Min:</span>
                        <span id="fc-60min" style="font-weight: bold; color: #d73027;">-</span>
                    </div>
                </div>

                <div style="display: flex; gap: 6px; margin-top: 6px;">
                    <button onclick="runFrontEndSimulation(0.4)" style="flex: 1; background: #ff9f43; border: none; padding: 6px; border-radius: 4px; color: white; font-weight: bold; cursor: pointer; font-size: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.3);">Accident</button>
                    <button onclick="runFrontEndSimulation(0.7)" style="flex: 1; background: #ff5252; border: none; padding: 6px; border-radius: 4px; color: white; font-weight: bold; cursor: pointer; font-size: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.3);">Severe Crash</button>
                    <button onclick="runFrontEndSimulation(1.0)" style="flex: 1; background: #d73027; border: none; padding: 6px; border-radius: 4px; color: white; font-weight: bold; cursor: pointer; font-size: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.3);">Closure</button>
                </div>
                <button onclick="resetFrontEndSimulation()" style="width: 100%; background: #4a5568; border: none; padding: 6px; border-radius: 4px; color: white; font-weight: bold; cursor: pointer; font-size: 10px; margin-top: 4px;">Reset Map</button>
            </div>
            <div id="sim-results" style="margin-top: 10px; display: none; font-size: 11px; background: rgba(0,0,0,0.3); padding: 10px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.05);">
                <b style="color: #ffcc00; font-size: 12px; display: block; margin-bottom: 4px;">Simulation & ML Outputs:</b>
                <div id="target-impact" style="margin-bottom: 6px; color: #ff5252; font-weight: 600;"></div>
                <div id="prop-impact" style="color: #cbd5e0; line-height: 1.4; margin-bottom: 8px;"></div>
                <div id="gnn-timeline" style="border-top: 1px solid rgba(255,255,255,0.08); padding-top: 8px;">
                    <b style="color: #00ffcc; font-size: 12px; display: block; margin-bottom: 6px;">ST-GNN Propagation Timeline:</b>
                    <div id="gnn-timeline-content" style="color: #cbd5e0; font-family: monospace; line-height: 1.5; white-space: pre-line;"></div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        var roadData = __ROAD_DATA__;
        var activeHighlight = null;
        var selectedRoadName = null;
        var selectedRoadSpeed = 30;
        var roadLayers = [];

        // Collect road layers and register click events after page load
        window.addEventListener('load', function() {
            setTimeout(function() {
                var mapKeys = Object.keys(window).filter(function(k) { return k.indexOf('map_') === 0; });
                if (mapKeys.length > 0) {
                    var mapInstance = window[mapKeys[0]];
                    mapInstance.eachLayer(function(layer) {
                        if (layer.feature && layer.feature.properties && layer.feature.properties.road_name) {
                            // Save original style
                            layer.originalColor = layer.options.color;
                            layer.originalWeight = layer.options.weight;
                            layer.originalOpacity = layer.options.opacity;
                            roadLayers.push(layer);
                            
                            // Bind click handler
                            layer.on('click', function(e) {
                                selectRoadForSimulation(layer.feature.properties.road_name, layer.feature.properties.speed_kmph, layer.feature.properties.edge_id);
                            });
                        }
                    });
                    console.log("Registered click handlers for " + roadLayers.length + " road segments.");
                }
            }, 1000);
        });

        function findEdgeIdByRoadName(name) {
            for (var i = 0; i < roadLayers.length; i++) {
                if (roadLayers[i].feature.properties.road_name.toLowerCase() === name.toLowerCase()) {
                    return roadLayers[i].feature.properties.edge_id;
                }
            }
            return "";
        }

        function selectRoadForSimulation(name, speed, edgeId) {
            selectedRoadName = name;
            selectedRoadSpeed = speed || 30;
            if (!edgeId) {
                edgeId = findEdgeIdByRoadName(name);
            }
            
            // Notify parent dashboard if embedded in iframe
            if (window.parent && typeof window.parent.selectRoad === 'function') {
                window.parent.selectRoad(name, selectedRoadSpeed, edgeId);
            }
            
            var nameEl = document.getElementById('selected-road-name');
            if (nameEl) nameEl.textContent = "Road: " + name;
            
            var speedEl = document.getElementById('selected-road-speed');
            if (speedEl) speedEl.textContent = Math.round(selectedRoadSpeed);
            
            var statusEl = document.getElementById('sim-status');
            if (statusEl) statusEl.style.display = 'none';
            
            var controlsEl = document.getElementById('sim-controls');
            if (controlsEl) controlsEl.style.display = 'flex';
            
            var resultsEl = document.getElementById('sim-results');
            if (resultsEl) resultsEl.style.display = 'none';

            // Query ML traffic forecasts from server API
            if (edgeId) {
                document.getElementById('ml-forecasts-panel').style.display = 'block';
                document.getElementById('fc-current').textContent = 'Loading...';
                document.getElementById('fc-15min').textContent = 'Loading...';
                document.getElementById('fc-30min').textContent = 'Loading...';
                document.getElementById('fc-60min').textContent = 'Loading...';
                
                fetch('/api/predict_traffic?edge_id=' + encodeURIComponent(edgeId))
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            console.error(data.error);
                            return;
                        }
                        document.getElementById('fc-current').textContent = data.current;
                        document.getElementById('fc-15min').textContent = data["15_min"];
                        document.getElementById('fc-30min').textContent = data["30_min"];
                        document.getElementById('fc-60min').textContent = data["60_min"];
                    })
                    .catch(err => {
                        console.warn('Backend server offline. Hiding ML forecast panel.', err);
                        document.getElementById('ml-forecasts-panel').style.display = 'none';
                    });
            } else {
                document.getElementById('ml-forecasts-panel').style.display = 'none';
            }
        }

        function runFrontEndSimulation(reductionFactor) {
            if (!selectedRoadName) return;
            
            // Reset colors first
            resetLayerStyles();
            
            var targetCoords = [];
            // Find target layer endpoints for connectivity
            roadLayers.forEach(function(layer) {
                if (layer.feature.properties.road_name.toLowerCase() === selectedRoadName.toLowerCase()) {
                    var geom = layer.feature.geometry;
                    if (geom && geom.coordinates) {
                        targetCoords.push(geom.coordinates[0]);
                        targetCoords.push(geom.coordinates[geom.coordinates.length - 1]);
                        
                        // Style target road RED
                        layer.setStyle({
                            color: '#ff5252',
                            weight: 6,
                            opacity: 1.0
                        });
                    }
                }
            });

            // Helper to check coord intersection (within ~15 meters)
            function matchCoords(coordsList, targetList) {
                for (var i = 0; i < coordsList.length; i++) {
                    var c = coordsList[i];
                    for (var j = 0; j < targetList.length; j++) {
                        var tc = targetList[j];
                        if (Math.abs(c[0] - tc[0]) < 0.00015 && Math.abs(c[1] - tc[1]) < 0.00015) {
                            return true;
                        }
                    }
                }
                return false;
            }

            var d1_layers = [];
            var d1_names = new Set();
            var d1_coords = [];

            // Find Depth 1 neighbors
            roadLayers.forEach(function(layer) {
                var name = layer.feature.properties.road_name;
                if (name.toLowerCase() === selectedRoadName.toLowerCase()) return;
                
                var geom = layer.feature.geometry;
                if (geom && geom.coordinates) {
                    if (matchCoords(geom.coordinates, targetCoords)) {
                        d1_layers.push(layer);
                        d1_names.add(name);
                        d1_coords.push(geom.coordinates[0]);
                        d1_coords.push(geom.coordinates[geom.coordinates.length - 1]);
                        
                        // Style Depth 1 ORANGE
                        layer.setStyle({
                            color: '#ff9f43',
                            weight: 5,
                            opacity: 0.95
                        });
                    }
                }
            });

            var d2_layers = [];
            var d2_names = new Set();

            // Find Depth 2 neighbors
            roadLayers.forEach(function(layer) {
                var name = layer.feature.properties.road_name;
                if (name.toLowerCase() === selectedRoadName.toLowerCase()) return;
                if (d1_names.has(name)) return;
                
                var geom = layer.feature.geometry;
                if (geom && geom.coordinates) {
                    if (matchCoords(geom.coordinates, d1_coords)) {
                        d2_layers.push(layer);
                        d2_names.add(name);
                        
                        // Style Depth 2 YELLOW
                        layer.setStyle({
                            color: '#ffd35c',
                            weight: 4,
                            opacity: 0.85
                        });
                    }
                }
            });

            // Display Local Propagation Results
            var newTargetSpeed = Math.max(1, selectedRoadSpeed * (1 - reductionFactor));
            var reductionPct = Math.round(reductionFactor * 100);
            
            // Trigger backend prediction
            var cause = "vehicle_breakdown";
            if (reductionFactor === 0.7) {
                cause = "accident";
            } else if (reductionFactor === 1.0) {
                cause = "road_closure";
            }
            
            document.getElementById('target-impact').innerHTML = "Calculating ML predictions...";
            
            fetch('/api/predict_event', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    event_cause: cause,
                    location: selectedRoadName,
                    time: "8:30 AM",
                    vehicle: "truck"
                })
            })
            .then(res => res.json())
            .then(data => {
                if (data.error) throw new Error(data.error);
                
                var resultsHtml = 
                    "<div style='border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 6px; margin-bottom: 6px; line-height: 1.4;'>" +
                    "• <b>ML Impact Severity:</b> <span style='color: " + (data.impact === 'HIGH' ? '#ff5252' : '#ff9f43') + "; font-weight: bold;'>" + data.impact + "</span> (Score: " + data.impact_score + ")<br>" +
                    "• <b>Expected Duration:</b> " + data.duration + "<br>" +
                    "• <b>Closure Probability:</b> " + Math.round(data.closure_probability * 100) + "%" +
                    "</div>" +
                    "<div style='margin-bottom: 4px; color: #ffcc00; font-family: monospace; font-size: 10px; line-height: 1.35; white-space: pre-line; background: rgba(0,0,0,0.2); padding: 6px; border-radius: 4px;'>" +
                    data.explanation +
                    "</div>";
                    
                document.getElementById('target-impact').innerHTML = resultsHtml;
            })
            .catch(err => {
                console.warn("Backend error or offline fallback:", err);
                // Fallback to client-side formulas
                document.getElementById('target-impact').innerHTML = 
                    "• Speed Drop: " + Math.round(selectedRoadSpeed) + " → " + Math.round(newTargetSpeed) + " km/h (-" + reductionPct + "%)";
            });
            
            var d1_list = Array.from(d1_names);
            var d2_list = Array.from(d2_names);
            
            var propHtml = "";
            if (d1_list.length > 0) {
                propHtml += "<b>Level 1 Spatial Impact (-" + Math.round(reductionPct*0.5) + "%):</b><br>" + 
                            d1_list.slice(0, 4).join(", ") + (d1_list.length > 4 ? "..." : "") + "<br>";
            }
            if (d2_list.length > 0) {
                propHtml += "<b style='margin-top: 4px; display: inline-block;'>Level 2 Spatial Impact (-" + Math.round(reductionPct*0.25) + "%):</b><br>" + 
                            d2_list.slice(0, 4).join(", ") + (d2_list.length > 4 ? "..." : "");
            }
            if (d1_list.length === 0 && d2_list.length === 0) {
                propHtml += "No nearby roads directly affected spatially.";
            }
            
            document.getElementById('prop-impact').innerHTML = propHtml;
            document.getElementById('sim-results').style.display = 'block';

            // Trigger ST-GNN Propagation API
            document.getElementById('gnn-timeline-content').innerHTML = "Running ST-GNN propagation model...";
            
            fetch('/api/predict_stgnn', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    location: selectedRoadName,
                    impact_score: reductionFactor === 0.7 ? 0.9 : reductionFactor
                })
            })
            .then(res => res.json())
            .then(data => {
                if (data.error) throw new Error(data.error);
                var t = data.timeline;
                
                var timelineHtml = "";
                if (t["0"] && t["0"].length > 0) {
                    timelineHtml += "<b>TIME 0</b>\n\n";
                    t["0"].forEach(function(item) {
                        timelineHtml += item.road_name + ":\n" + item.congestion_pct + "%\n\n";
                    });
                    timelineHtml += "\n";
                }
                
                if (t["15min"] && t["15min"].length > 0) {
                    timelineHtml += "<b>+15 min:</b>\n\n";
                    t["15min"].forEach(function(item) {
                        timelineHtml += item.road_name + ":\n" + item.congestion_pct + "%\n\n";
                    });
                    timelineHtml += "\n";
                }
                
                if (t["30min"] && t["30min"].length > 0) {
                    timelineHtml += "<b>+30 min:</b>\n\n";
                    t["30min"].forEach(function(item) {
                        timelineHtml += item.road_name + ":\n" + item.congestion_pct + "%\n\n";
                    });
                }
                
                document.getElementById('gnn-timeline-content').innerHTML = timelineHtml;
            })
            .catch(err => {
                console.error("GNN Simulation API failed:", err);
                document.getElementById('gnn-timeline-content').innerHTML = "ST-GNN model offline or not trained.";
            });
        }

        function resetLayerStyles() {
            roadLayers.forEach(function(layer) {
                layer.setStyle({
                    color: layer.originalColor,
                    weight: layer.originalWeight,
                    opacity: layer.originalOpacity
                });
            });
        }

        function resetFrontEndSimulation() {
            resetLayerStyles();
            document.getElementById('sim-status').style.display = 'block';
            document.getElementById('sim-controls').style.display = 'none';
            document.getElementById('sim-results').style.display = 'none';
        }

        function searchRoads() {
            var query = document.getElementById('road-search').value.toLowerCase();
            var resultsUl = document.getElementById('search-results');
            resultsUl.innerHTML = '';
            if (!query || query.length < 2) {
                resultsUl.style.display = 'none';
                return;
            }
            // Normalize query by removing spaces for flexible matching (e.g., "silk board" matches "Silkboard")
            var normalizedQuery = query.replace(/\s+/g, '');
            var filtered = roadData.filter(function(r) {
                var normalizedName = r.name.toLowerCase().replace(/\s+/g, '');
                return normalizedName.includes(normalizedQuery);
            }).slice(0, 10);
            if (filtered.length === 0) {
                resultsUl.style.display = 'none';
                return;
            }
            filtered.forEach(function(r) {
                var li = document.createElement('li');
                li.textContent = r.name;
                li.onclick = function() {
                    var mapKeys = Object.keys(window).filter(function(k) { return k.indexOf('map_') === 0; });
                    if (mapKeys.length > 0) {
                        var mapInstance = window[mapKeys[0]];
                        mapInstance.setView([r.lat, r.lng], 16);
                        if (activeHighlight) {
                            mapInstance.removeLayer(activeHighlight);
                        }
                        activeHighlight = L.circle([r.lat, r.lng], {
                            color: '#00ffcc',
                            fillColor: '#00ffcc',
                            fillOpacity: 0.15,
                            radius: 200,
                            weight: 2,
                            dashArray: '4, 4'
                        }).addTo(mapInstance);
                        activeHighlight.bindPopup("<b>" + r.name + "</b><br>Centroid Coordinate: " + r.lat.toFixed(4) + ", " + r.lng.toFixed(4)).openPopup();
                        // Select for simulation
                        selectRoadForSimulation(r.name, 40, "");
                    }
                    document.getElementById('road-search').value = r.name;
                    resultsUl.style.display = 'none';
                };
                resultsUl.appendChild(li);
            });
            resultsUl.style.display = 'block';
        }

        // API exposed to parent dashboard frame
        window.highlightRoadOnMap = function(name, lat, lng) {
            selectedRoadName = name;
            var mapKeys = Object.keys(window).filter(function(k) { return k.indexOf('map_') === 0; });
            if (mapKeys.length > 0) {
                var mapInstance = window[mapKeys[0]];
                mapInstance.setView([lat, lng], 16);
                if (activeHighlight) {
                    mapInstance.removeLayer(activeHighlight);
                }
                activeHighlight = L.circle([lat, lng], {
                    color: '#00ffcc',
                    fillColor: '#00ffcc',
                    fillOpacity: 0.15,
                    radius: 200,
                    weight: 2,
                    dashArray: '4, 4'
                }).addTo(mapInstance);
                activeHighlight.bindPopup("<b>" + name + "</b><br>Centroid: " + lat.toFixed(4) + ", " + lng.toFixed(4)).openPopup();
                
                // Select locally
                var speed = 30;
                for (var i = 0; i < roadLayers.length; i++) {
                    if (roadLayers[i].feature.properties.road_name.toLowerCase() === name.toLowerCase()) {
                        speed = roadLayers[i].feature.properties.speed_kmph;
                        break;
                    }
                }
                selectRoadForSimulation(name, speed, "");
            }
        };

        window.runSimulationOnMap = function(reductionFactor) {
            runFrontEndSimulation(reductionFactor);
        };

        window.resetMapSimulation = function() {
            resetFrontEndSimulation();
        };

        // Close search results when clicking outside
        document.addEventListener('click', function(e) {
            if (e.target.id !== 'road-search') {
                var resultsUl = document.getElementById('search-results');
                if (resultsUl) resultsUl.style.display = 'none';
            }
        });
    </script>
    """

    # Perform placeholder replacements
    html_injection = html_injection.replace("__NODES__", stats["nodes"])
    html_injection = html_injection.replace("__EDGES__", stats["edges"])
    html_injection = html_injection.replace("__EVENTS__", stats["events"])
    html_injection = html_injection.replace("__ROAD_DATA__", json.dumps(js_road_data))

    fmap.get_root().header.add_child(folium.Element(css_injection))
    fmap.get_root().html.add_child(folium.Element(html_injection))

    # Save visualization HTML
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    fmap.save(str(OUTPUT_PATH))
    logging.info("Saved complete digital twin visualization to %s", OUTPUT_PATH)
    return fmap


if __name__ == "__main__":
    build_map()
