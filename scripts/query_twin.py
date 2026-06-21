from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add base dir to system path for imports
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.traffic.tomtom_collector import TomTomCollector
from src.traffic.traffic_processor import TrafficProcessor

# Suppress verbose logging for clean CLI output
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

DATA_DIR = BASE_DIR / "data"
EDGES_PATH = DATA_DIR / "edges.csv"
TIMESERIES_PATH = DATA_DIR / "traffic_timeseries.csv"
LIVE_TRAFFIC_DIR = DATA_DIR / "live_traffic"

def load_edges_metadata() -> pd.DataFrame:
    if not EDGES_PATH.exists():
        print("Error: edges.csv not found. Please run graph builder first.")
        sys.exit(1)
    return pd.read_csv(EDGES_PATH).drop_duplicates(subset=["edge_id"])

def get_latest_live_file() -> Path | None:
    if not LIVE_TRAFFIC_DIR.exists():
        return None
    files = list(LIVE_TRAFFIC_DIR.glob("live_traffic_*.csv"))
    if not files:
        return None
    return max(files, key=lambda f: f.stat().st_mtime)

def get_congestion_label(score: float) -> str:
    if score < 0.15:
        return "Free Flow (Green)"
    elif score < 0.35:
        return "Slow Traffic (Yellow)"
    elif score < 0.6:
        return "Congested (Orange)"
    else:
        return "Severe Gridlock / Blocked (Red)"

def answer_q1():
    """
    Q1: What does Bangalore traffic look like right now?
    """
    print("\n=== Q1: Bangalore Traffic Right Now ===")
    live_file = get_latest_live_file()
    
    if not live_file:
        print("No live traffic logs found. Collecting fresh sample from TomTom (Mock fallback)...")
        collector = TomTomCollector()
        live_df = collector.collect(limit=30)
    else:
        print(f"Loading latest traffic snapshot: {live_file.name}")
        live_df = pd.read_csv(live_file)
        
    avg_congestion = live_df["congestion_score"].mean()
    status = get_congestion_label(avg_congestion)
    
    print(f"\nOverall Network State: {status} (Avg Congestion: {avg_congestion:.2f})")
    print(f"Sampled Segments:      {len(live_df)}")
    
    print("\nTop 5 Most Congested Segments:")
    top_congested = live_df.sort_values(by="congestion_score", ascending=False).head(5)
    for _, row in top_congested.iterrows():
        road = row["road_name"] if pd.notna(row["road_name"]) else "Arterial Road"
        print(f" - {road} ({row['road_type']}): Speed {row['current_speed']} km/h / Free-Flow {row['free_flow_speed']} km/h [Congestion: {row['congestion_score']:.2f}]")

    print("\nAverage Speeds by Road Type:")
    type_speeds = live_df.groupby("road_type")["current_speed"].mean().reset_index()
    for _, row in type_speeds.iterrows():
        print(f" - {row['road_type']}: {row['current_speed']:.1f} km/h")

def answer_q2(road_name: str | None = None):
    """
    Q2: How congested is every road?
    """
    print("\n=== Q2: Congestion Levels ===")
    live_file = get_latest_live_file()
    if not live_file:
        collector = TomTomCollector()
        live_df = collector.collect(limit=50)
    else:
        live_df = pd.read_csv(live_file)
        
    if road_name:
        # Search for specific road
        matches = live_df[live_df["road_name"].str.contains(road_name, case=False, na=False)]
        if matches.empty:
            print(f"No active live data for road containing '{road_name}'. Searching historical database...")
            # Fallback to timeseries
            if TIMESERIES_PATH.exists():
                ts_df = pd.read_csv(TIMESERIES_PATH)
                edges_df = load_edges_metadata()
                ts_df = ts_df.merge(edges_df[["edge_id", "road_name"]], on="edge_id", how="left")
                matches = ts_df[ts_df["road_name"].str.contains(road_name, case=False, na=False)].tail(5)
                
        if matches.empty:
            print(f"Road '{road_name}' not found in current logs or timeseries.")
            return
            
        print(f"\nFound {len(matches)} matching segment(s) for '{road_name}':")
        for _, row in matches.head(10).iterrows():
            road = row["road_name"]
            congestion = row["congestion_score"]
            lbl = get_congestion_label(congestion)
            speed = row.get("current_speed", row.get("speed", 0.0))
            ff_speed = row.get("free_flow_speed", 50.0)
            print(f" - {road}: Speed {speed:.1f} km/h / Limit {ff_speed:.1f} km/h | Status: {lbl} [Score: {congestion:.2f}]")
    else:
        # Show general summary
        print(f"Displaying sample of current road network status (Total segments: {len(live_df)}):")
        sample_df = live_df.dropna(subset=["road_name"]).sample(min(15, len(live_df)), random_state=42)
        for _, row in sample_df.iterrows():
            lbl = get_congestion_label(row["congestion_score"])
            print(f" - {row['road_name']}: {row['current_speed']} km/h (Limit: {row['free_flow_speed']} km/h) | {lbl}")

def answer_q3():
    """
    Q3: What is normal traffic at this time?
    """
    print("\n=== Q3: Normal (Historical Baseline) Traffic ===")
    now = datetime.now()
    hour = now.hour
    day_name = now.strftime("%A")
    is_weekend = now.weekday() >= 5
    
    print(f"Current Local Time: {now.strftime('%I:%M %p')} on {day_name} ({'Weekend' if is_weekend else 'Weekday'})")
    
    if not TIMESERIES_PATH.exists():
        print("Historical timeseries not generated yet. Running baseline formula...")
        # Use generator formula directly
        from src.traffic.synthetic_generator import SyntheticTrafficGenerator
        gen = SyntheticTrafficGenerator()
        mult = gen.generate_baseline_speed_multiplier(now)
        expected_congestion = 1.0 - mult
        print(f"\nExpected Baseline Congestion Multiplier: {mult:.2f}")
        print(f"Expected Network Congestion Level:      {expected_congestion:.2f} ({get_congestion_label(expected_congestion)})")
        return
        
    print("Loading historical timeseries database...")
    ts_df = pd.read_csv(TIMESERIES_PATH)
    ts_df["timestamp"] = pd.to_datetime(ts_df["timestamp"])
    
    # Filter by hour and weekday/weekend type
    ts_df["hour"] = ts_df["timestamp"].dt.hour
    ts_df["is_weekend"] = ts_df["timestamp"].dt.dayofweek >= 5
    
    subset = ts_df[(ts_df["hour"] == hour) & (ts_df["is_weekend"] == is_weekend)]
    if subset.empty:
        subset = ts_df[ts_df["hour"] == hour] # fallback
        
    avg_speed = subset["speed"].mean()
    avg_congestion = subset["congestion_score"].mean()
    lbl = get_congestion_label(avg_congestion)
    
    print(f"\nNormal Traffic Parameters for {hour}:00 on {'Weekends' if is_weekend else 'Weekdays'}:")
    print(f" - Average Speed:            {avg_speed:.2f} km/h")
    print(f" - Average Congestion Score: {avg_congestion:.2f} | Status: {lbl}")
    
    # Group by road type if available
    edges_df = load_edges_metadata()
    merged = subset.merge(edges_df[["edge_id", "road_type"]], on="edge_id", how="inner")
    if not merged.empty:
        print("\nNormal Speed by Road Category:")
        type_means = merged.groupby("road_type")[["speed", "congestion_score"]].mean().reset_index()
        for _, row in type_means.iterrows():
            print(f" - {row['road_type']}: {row['speed']:.1f} km/h [Congestion: {row['congestion_score']:.2f}]")

def answer_q4_q5(road_query: str, severity: str = "high"):
    """
    Q4 & Q5: Simulation of accident and neighboring impact propagation.
    """
    print(f"\n=== Q4 & Q5: Incident Simulation ({severity.upper()} severity accident) ===")
    
    edges_df = load_edges_metadata()
    
    # Find a matching edge ID
    matches = edges_df[edges_df["road_name"].str.contains(road_query, case=False, na=False)]
    if matches.empty:
        print(f"Road containing '{road_query}' not found in database.")
        return
        
    target_row = matches.iloc[0]
    target_edge_id = target_row["edge_id"]
    road_name = target_row["road_name"]
    free_flow_speed = target_row["speed"]
    
    print(f"Simulating Incident on Target Road: {road_name} ({target_row['road_type']})")
    print(f"Free-Flow Speed Limit:             {free_flow_speed} km/h")
    
    # Initialize processor to query neighbors
    processor = TrafficProcessor()
    
    # Q4: Determine speed reduction factor
    # High: 70%, Medium: 40%, Complete Closure: 100%
    if severity.lower() == "closure":
        reduction = 1.0
    elif severity.lower() == "high":
        reduction = 0.70
    else:
        reduction = 0.40
        
    # Get neighbors
    neighbors = processor.get_neighbors_by_depth(target_edge_id, max_depth=2)
    
    # Print Q4 result (traffic drops)
    print("\n--- Q4: Simulated Speed Drops ---")
    new_speed = max(1.0, free_flow_speed * (1.0 - reduction))
    congestion_score = 1.0 - (new_speed / free_flow_speed)
    print(f"Target Road speed drops by {reduction*100:.0f}%:")
    print(f" - {road_name}: {free_flow_speed} km/h ---> {new_speed:.1f} km/h [Congestion: {congestion_score:.2f} - {get_congestion_label(congestion_score)}]")
    
    # Print Q5 result (neighbor road names)
    print("\n--- Q5: Affected Nearby Roads (Propagation Map) ---")
    
    def print_neighbors(level: int, factor: float):
        n_edges = list(neighbors[level])
        if not n_edges:
            print(f" Level {level} Neighbors: None found")
            return
            
        print(f" Level {level} Impact Area (Speed drops by {factor*100:.1f}%):")
        # Map edge IDs back to road names
        n_rows = edges_df[edges_df["edge_id"].isin(n_edges)]
        unique_roads = n_rows.dropna(subset=["road_name"])["road_name"].unique()
        
        for r in unique_roads:
            # Look up standard speed
            r_limit = n_rows[n_rows["road_name"] == r]["speed"].values[0]
            r_new = max(1.0, r_limit * (1.0 - factor))
            print(f"  - {r}: Limit {r_limit} km/h ---> {r_new:.1f} km/h")
            
    # Depth 1: 50% of the primary reduction
    print_neighbors(1, reduction * 0.5)
    # Depth 2: 25% of the primary reduction
    print_neighbors(2, reduction * 0.25)

def main():
    print("==========================================")
    print("   BANGALORE TRAFFIC TWIN QUERY ENGINE    ")
    print("==========================================")
    
    if len(sys.argv) > 1:
        # Command line arg handling
        mode = sys.argv[1]
        if mode == "--q1":
            answer_q1()
        elif mode == "--q2":
            road = sys.argv[2] if len(sys.argv) > 2 else None
            answer_q2(road)
        elif mode == "--q3":
            answer_q3()
        elif mode == "--q45":
            road = sys.argv[2] if len(sys.argv) > 2 else "Outer Ring Road"
            sev = sys.argv[3] if len(sys.argv) > 3 else "high"
            answer_q4_q5(road, sev)
        else:
            print("Invalid args. Running interactive menu...")
            interactive_menu()
    else:
        interactive_menu()

def interactive_menu():
    while True:
        print("\nSelect a query to answer:")
        print("1. What does Bangalore traffic look like right now?")
        print("2. How congested is every road? (Query specific road)")
        print("3. What is normal traffic at this time?")
        print("4. If an accident happened, how much does traffic drop nearby?")
        print("5. Which nearby roads start getting affected?")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == "1":
            answer_q1()
        elif choice == "2":
            road = input("Enter road name to search (or press Enter for sample): ").strip()
            answer_q2(road if road else None)
        elif choice == "3":
            answer_q3()
        elif choice == "4" or choice == "5":
            road = input("Enter target road name (e.g. Outer Ring Road, Richmond Road): ").strip()
            if not road:
                road = "Outer Ring Road"
            sev = input("Enter accident severity (low, high, closure) [default: high]: ").strip()
            if not sev:
                sev = "high"
            answer_q4_q5(road, sev)
        elif choice == "6":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1-6.")
            
        print("\n" + "-"*40)

if __name__ == "__main__":
    main()
