"""
run_demo.py — Demonstration of the Unified City Simulation Engine (Task 7).

Runs four hardcoded scenarios and prints Before/After state, timeline, and recommendations.

Scenarios:
  1. Truck breakdown on Outer Ring Road
  2. Public event at Chinnaswamy Stadium
  3. Heavy rain flooding
  4. Full road closure
"""
from __future__ import annotations

import json
import sys
import logging
import time
from pathlib import Path

# Ensure project root is on path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.simulation.scenario import TrafficScenario
from src.simulation.city_engine import CitySimulationEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

SCENARIOS = [
    {
        "name": "Scenario 1: Truck Breakdown on ORR",
        "input": {
            "type": "vehicle_breakdown",
            "location": {"lat": 12.917, "lng": 77.623},
            "location_name": "Outer Ring Road",
            "time": "2026-06-20 09:00",
            "vehicle": "truck",
            "description": "A heavy truck broke down blocking one lane on the Outer Ring Road near Silk Board.",
            "road_action": {"closure": False},
        },
    },
    {
        "name": "Scenario 2: Public Event at Chinnaswamy Stadium",
        "input": {
            "type": "public_event",
            "location": {"lat": 12.9788, "lng": 77.5996},
            "location_name": "MG Road",
            "time": "2026-06-20 18:00",
            "vehicle": "car",
            "description": "IPL cricket match at M. Chinnaswamy Stadium causing massive crowd and traffic.",
            "road_action": {"closure": True, "percentage": 50, "closure_type": "partial"},
        },
    },
    {
        "name": "Scenario 3: Heavy Rain Flooding",
        "input": {
            "type": "heavy_rain",
            "location": {"lat": 12.934, "lng": 77.612},
            "location_name": "Hosur Road",
            "time": "2026-06-20 16:30",
            "vehicle": "car",
            "description": "Heavy monsoon rain causing waterlogging and flooding on Hosur Road near Madiwala.",
            "road_action": {"closure": True, "percentage": 70, "closure_type": "partial"},
        },
    },
    {
        "name": "Scenario 4: Full Road Closure for Construction",
        "input": {
            "type": "construction",
            "location": {"lat": 12.960, "lng": 77.641},
            "location_name": "Old Madras Road",
            "time": "2026-06-20 10:00",
            "vehicle": "car",
            "description": "Metro construction work requiring full closure of Old Madras Road near Indiranagar.",
            "road_action": {"closure": True, "percentage": 100, "closure_type": "full"},
        },
    },
]


def print_section(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_single_scenario(engine: CitySimulationEngine, scenario_def: dict) -> dict:
    """Run one scenario through the engine and print results."""
    print_section(scenario_def["name"])

    # Build scenario
    scenario = TrafficScenario.from_user_input(scenario_def["input"])
    scenario.validate()

    print(f"\n  Event Type  : {scenario.event_type}")
    print(f"  Location    : {scenario.location_name} ({scenario.latitude}, {scenario.longitude})")
    print(f"  Time        : {scenario.time}")
    print(f"  Vehicle     : {scenario.vehicle}")
    print(f"  Closure     : {'Yes (' + str(scenario.road_action.percentage) + '%)' if scenario.road_action.closure else 'No'}")
    print()

    t0 = time.time()
    result = engine.run_simulation(scenario)
    elapsed = time.time() - t0

    # --- Print results ---
    print(f"\n  [Completed in {elapsed:.1f}s]")

    # Impact
    city_impact = result.get("city_impact", {})
    print(f"\n  CITY IMPACT: {city_impact.get('impact_category', 'N/A')} "
          f"(score: {city_impact.get('city_impact_score', 0):.3f})")

    # Event prediction
    ep = result.get("event_prediction", {})
    print(f"  ML Impact       : {ep.get('impact', 'N/A')} (score {ep.get('impact_score', 0)})")
    print(f"  Duration        : {ep.get('expected_duration', 'N/A')} min")
    print(f"  Closure Prob    : {int(ep.get('closure_probability', 0) * 100)}%")

    # Future congestion
    fc = result.get("future_congestion", {})
    print(f"\n  CONGESTION FORECAST (epicenter):")
    for k, v in fc.items():
        if isinstance(v, float):
            print(f"    {k:>8s}: {v:.3f}")
        else:
            print(f"    {k:>8s}: {v}")

    # Affected roads
    affected = result.get("affected_roads", [])
    if affected:
        print(f"\n  AFFECTED ROADS ({len(affected)}):")
        for i, rd in enumerate(affected[:5], 1):
            if isinstance(rd, dict):
                name = rd.get("road", rd.get("road_name", "Unknown"))
                inc = rd.get("congestion_increase", "")
                print(f"    {i}. {name} {inc}")
            else:
                print(f"    {i}. {rd}")
        if len(affected) > 5:
            print(f"    ... and {len(affected) - 5} more")

    # Timeline summary
    tl = result.get("timeline", {})
    print(f"\n  TIMELINE: {tl.get('total_roads', 0)} roads x {len(tl.get('timestamps', []))} steps")

    # Recommendations
    recs = result.get("police_action_plan", {})
    actions = recs.get("priority_actions", [])
    if actions:
        print(f"\n  RECOMMENDED ACTIONS:")
        for i, act in enumerate(actions[:5], 1):
            print(f"    {i}. {act}")

    print()
    return result


def main():
    print_section("BANGALORE TRAFFIC DIGITAL TWIN — TASK 7 DEMO")
    print("  Unified City Simulation Engine")
    print("  Running 4 hardcoded scenarios...\n")

    engine = CitySimulationEngine()
    results = []

    for scenario_def in SCENARIOS:
        try:
            result = run_single_scenario(engine, scenario_def)
            results.append({"name": scenario_def["name"], "status": "OK", "impact": result.get("impact", "N/A")})
        except Exception as e:
            logging.error("Scenario '%s' failed: %s", scenario_def["name"], e)
            import traceback
            traceback.print_exc()
            results.append({"name": scenario_def["name"], "status": "FAILED", "error": str(e)})

    # Summary
    print_section("DEMO SUMMARY")
    for r in results:
        status_icon = "[OK]" if r["status"] == "OK" else "[FAIL]"
        impact = r.get("impact", r.get("error", ""))
        print(f"  {status_icon} {r['name']} — {impact}")

    print()
    print("  All outputs saved to: outputs/")
    print()


if __name__ == "__main__":
    main()
