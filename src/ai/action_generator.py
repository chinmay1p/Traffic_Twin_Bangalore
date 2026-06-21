# [ignoring loop detection]
"""
Traffic Twin Bengaluru — AI Action Generator
"""
import math

def recommend_manpower(event_type, severity, crowd_size=0, road_importance="medium"):
    """
    Determines traffic police manpower requirements based on event severity, crowd, and corridor type.
    """
    # Base officer count rules
    if severity == "LOW":
        base_officers = 3
    elif severity == "MEDIUM":
        base_officers = 7
    elif severity == "HIGH":
        base_officers = 14
    else: # CRITICAL
        base_officers = 22

    # Boost based on public events crowd size
    crowd_boost = 0
    if event_type == "public_event" and crowd_size > 0:
        crowd_boost = int(math.ceil(crowd_size / 5000)) # 1 officer per 5000 people

    total_officers = base_officers + crowd_boost

    # Allocation of officers
    near_entry = max(1, int(total_officers * 0.35))
    near_junction = max(1, int(total_officers * 0.4))
    managing_diversion = max(1, total_officers - (near_entry + near_junction))

    return {
        "total_officers": total_officers,
        "allocation": {
            "entry_gates_exits": near_entry,
            "main_junction_control": near_junction,
            "diversion_points": managing_diversion
        },
        "description": f"Deploy {total_officers} officers: {near_entry} near access points, {near_junction} at main junction, {managing_diversion} managing diversions."
    }

def recommend_diversion(road_name, event_type="accident"):
    """
    Recommends diversion routes using city corridors and estimates travel delay vs capacity.
    """
    # Mapping of main corridors to logical diversion routes with mock capacity calculations
    diversion_map = {
        "Silk Board": {
            "route": "Outer Ring Road → HSR Layout Sector 4 → BTM 2nd Stage",
            "spare_capacity": 42,
            "delay_reduction_min": 18
        },
        "MG Road": {
            "route": "Residency Road → Richmond Road → Trinity Circle",
            "spare_capacity": 38,
            "delay_reduction_min": 15
        },
        "Outer Ring Road": {
            "route": "Bellandur Service Road → Sarjapur Road → Haralur Road",
            "spare_capacity": 45,
            "delay_reduction_min": 25
        },
        "Cubbon Road": {
            "route": "Infantry Road → Kasturba Road → MG Road",
            "spare_capacity": 30,
            "delay_reduction_min": 12
        },
        "Hosur Road": {
            "route": "Electronic City Flyover → Begur Road → Kudlu Gate",
            "spare_capacity": 50,
            "delay_reduction_min": 22
        }
    }

    # Fallback/dynamic match if not exact
    selected = None
    for key, data in diversion_map.items():
        if key.lower() in road_name.lower():
            selected = data
            break

    if not selected:
        # Generate a dynamic fallback diversion based on the road name
        selected = {
            "route": f"{road_name} Service Road → Parallel Bypass Road",
            "spare_capacity": 35,
            "delay_reduction_min": 10
        }

    return {
        "route": selected["route"],
        "spare_capacity_pct": selected["spare_capacity"],
        "expected_delay_reduction_min": selected["delay_reduction_min"],
        "reason": f"Route has {selected['spare_capacity']}% spare capacity and avoids the primary congested intersection."
    }

def recommend_signal_strategy(road_name, event_type="accident"):
    """
    Calculates adaptive signal timing adjustments (Task 5 hook) based on queue pressure.
    """
    # Example signal adjustments for major corridors
    if "orr" in road_name.lower() or "outer ring road" in road_name.lower():
        pressure_increase = 85
        current_green = 45
        recommended_green = 90
    elif "silk board" in road_name.lower():
        pressure_increase = 110
        current_green = 45
        recommended_green = 100
    elif "mg road" in road_name.lower() or "cubbon" in road_name.lower():
        pressure_increase = 60
        current_green = 35
        recommended_green = 75
    else:
        pressure_increase = 40
        current_green = 45
        recommended_green = 60

    return {
        "corridor": road_name,
        "pressure_increase_pct": pressure_increase,
        "current_green_sec": current_green,
        "recommended_green_sec": recommended_green,
        "reason": f"Incoming vehicle queue pressure increased by {pressure_increase}%. Extending green phase to prevent gridlock."
    }
