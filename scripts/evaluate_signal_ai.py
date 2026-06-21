from __future__ import annotations

import sys
import logging
from pathlib import Path
import networkx as nx
import pandas as pd

from src.signals.simulation import evaluate_fixed_vs_ai
from src.simulator.scenario_engine import get_base_graph

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def main():
    print("==================================================")
    print(" EVALUATING FIXED VS ADAPTIVE AI TRAFFIC SIGNALS  ")
    print("==================================================")
    
    # Load graph and signals list
    G = get_base_graph()
    signals_csv = Path(__file__).resolve().parent.parent / "data" / "signals.csv"
    
    if not signals_csv.exists():
        print(f"Signals list not found at {signals_csv}. Please run signal_network.py first.")
        sys.exit(1)
        
    df_signals = pd.read_csv(signals_csv)
    if df_signals.empty:
        print("No signalized junctions found in signals.csv.")
        sys.exit(1)
        
    # Find a major junction (node with high degree or named Silk Board if possible)
    # Search for Silk Board in node name
    silk_board_node = None
    for node_id, data in G.nodes(data=True):
        name = data.get("name", "")
        if name and "silk board" in str(name).lower():
            silk_board_node = node_id
            break
            
    # Fallback to the signal node with the maximum approaches in signals.csv
    if silk_board_node is None:
        max_row = df_signals.sort_values(by="number_of_approaches", ascending=False).iloc[0]
        silk_board_node = int(max_row["node_id"])
        
    print(f"Selected Junction Node: {silk_board_node}")
    
    # Run evaluation
    results = evaluate_fixed_vs_ai(G, silk_board_node)
    
    print("\n--- Queue Simulation Results ---")
    print(f"Fixed: Average wait: {results['fixed']['average_wait_sec']} seconds")
    print(f"AI:    Average wait: {results['ai']['average_wait_sec']} seconds")
    print(f"Improvement: {results['metrics']['waiting_time_reduction_pct']}%")
    print("--------------------------------")
    print(f"Fixed Throughput: {results['fixed']['throughput']} vehicles/hour")
    print(f"AI Throughput:    {results['ai']['throughput']} vehicles/hour")
    print(f"Throughput Improvement: {results['metrics']['throughput_improvement_pct']}%")
    print("==================================================")

if __name__ == "__main__":
    main()
