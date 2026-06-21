import os
import logging
import pandas as pd
import numpy as np
import pyproj

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def extract():
    logging.info("Starting major junctions extraction...")
    
    edges_path = "data/edges.csv"
    nodes_path = "data/nodes.csv"
    signals_path = "data/signals.csv"
    output_path = "data/major_junctions.csv"
    
    if not os.path.exists(edges_path) or not os.path.exists(nodes_path):
        logging.error("Required data files edges.csv or nodes.csv missing.")
        return
        
    logging.info("Loading edges and nodes...")
    edges_df = pd.read_csv(edges_path)
    nodes_df = pd.read_csv(nodes_path)
    
    signal_node_ids = set()
    if os.path.exists(signals_path):
        logging.info("Loading signals...")
        try:
            signals_df = pd.read_csv(signals_path)
            # Some node_ids might be float or contain NaNs
            signal_node_ids = set(signals_df["node_id"].dropna().astype(int))
            logging.info(f"Loaded {len(signal_node_ids)} traffic signal node IDs.")
        except Exception as e:
            logging.warning(f"Could not load signals.csv: {e}")
            
    # Index nodes for fast coordinate lookup
    logging.info("Indexing nodes...")
    # latitude is UTM y, longitude is UTM x
    nodes_lookup = nodes_df.set_index("node_id")[["latitude", "longitude"]].to_dict(orient="index")
    
    # Analyze node degrees and road names
    logging.info("Analyzing edge connections...")
    node_edges = {}
    
    for _, row in edges_df.iterrows():
        u = int(row['u'])
        v = int(row['v'])
        name = str(row.get('road_name', 'Unknown'))
        if pd.isna(row.get('road_name')) or name == 'nan' or name.strip() == '':
            name = 'Unknown'
        rtype = str(row.get('road_type', 'unclassified'))
        if pd.isna(row.get('road_type')):
            rtype = 'unclassified'
            
        edge_info = {"name": name, "type": rtype}
        
        if u not in node_edges:
            node_edges[u] = []
        if v not in node_edges:
            node_edges[v] = []
            
        node_edges[u].append(edge_info)
        node_edges[v].append(edge_info)

    # Filter junctions based on criteria
    # Node degree >= 3 and connected to primary, secondary, trunk or motorway
    important_types = {
        "primary", "primary_link",
        "secondary", "secondary_link",
        "trunk", "trunk_link",
        "motorway", "motorway_link"
    }
    
    # Set up projection transformer: UTM 43N (EPSG:32643) to WGS84 (EPSG:4326)
    transformer = pyproj.Transformer.from_crs('EPSG:32643', 'EPSG:4326', always_xy=True)
    
    junctions = []
    logging.info("Filtering and projecting major junctions...")
    
    for node, edges in node_edges.items():
        # Check degree
        if len(edges) >= 3:
            # Check road importance
            has_important = any(e["type"] in important_types for e in edges)
            # Or is it a known major corridor intersection? (e.g. name contains major road names)
            has_major_name = any(
                "outer ring" in e["name"].lower() or 
                "silk board" in e["name"].lower() or
                "richmond" in e["name"].lower() or
                "hosur" in e["name"].lower() or
                "old madras" in e["name"].lower() or
                "mg road" in e["name"].lower()
                for e in edges
            )
            
            if has_important or has_major_name:
                if node in nodes_lookup:
                    utm_y = nodes_lookup[node]["latitude"]
                    utm_x = nodes_lookup[node]["longitude"]
                    
                    # Convert to WGS84 lat/lng
                    lng, lat = transformer.transform(utm_x, utm_y)
                    
                    # Gather unique connected road names
                    names = sorted(list(set(e["name"] for e in edges if e["name"] not in ("Unknown", "nan", None))))
                    connected_roads_str = ", ".join(names) if names else "Unknown Road"
                    
                    # Check signal availability
                    has_signal = 1 if node in signal_node_ids else 0
                    
                    # Classify importance and compute score
                    types = set(e["type"] for e in edges)
                    score = len(edges) * 10
                    
                    if any(t in {"motorway", "motorway_link", "trunk", "trunk_link"} for t in types) or "silk board" in connected_roads_str.lower():
                        importance = "high"
                        score += 50
                    elif any(t in {"primary", "primary_link", "secondary", "secondary_link"} for t in types):
                        importance = "medium"
                        score += 25
                    else:
                        importance = "low"
                        
                    if has_signal:
                        score += 15
                    
                    junctions.append({
                        "junction_id": f"junction_{node}",
                        "lat": lat,
                        "lng": lng,
                        "connected_roads": connected_roads_str,
                        "traffic_signal_available": has_signal,
                        "importance": importance,
                        "score": score
                    })
                    
    # Sort and take top 120
    if junctions:
        junctions_df = pd.DataFrame(junctions)
        # Sort by score desc, ensure Silk Board is prioritized
        junctions_df = junctions_df.sort_values(by="score", ascending=False)
        
        # Take top 120
        top_120 = junctions_df.head(120).copy()
        
        # Drop the temporary score column before saving
        top_120 = top_120.drop(columns=["score"])
        top_120.to_csv(output_path, index=False)
        logging.info(f"Extracted top {len(top_120)} major junctions. Saved to {output_path}")
    else:
        logging.error("No junctions found to save.")

if __name__ == "__main__":
    extract()
