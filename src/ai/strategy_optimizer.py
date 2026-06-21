# [ignoring loop detection]
"""
Traffic Twin Bengaluru — Strategy Optimizer
"""

def generate_comparison_plans(event_type, road_name, severity, duration_min):
    """
    Creates multiple strategic action plans and estimates outcomes for each.
    """
    # Baseline clearance time without any intervention
    unmanaged_clear_time = duration_min + 45

    # Plan A: Full closure (high restriction, fast clearance for work but redirects load)
    plan_a_clearance = int(unmanaged_clear_time * 0.8)
    # Plan B: Diversion + Manpower (moderated, balanced)
    plan_b_clearance = int(unmanaged_clear_time * 0.5)
    # Plan C: Signal Override Only (low restriction, slowest to clear)
    plan_c_clearance = int(unmanaged_clear_time * 0.7)

    # Determine recommended plan
    # For accidents & breakdowns: Plan B (Diversion + Manpower) is typically best
    # For massive events: Plan A or B depending on scale
    recommended_plan = "Plan B"

    plans = {
        "Plan A": {
            "name": "Full Road Closure & Redirect",
            "clearance_time_min": plan_a_clearance,
            "avg_speed_kph": 15,
            "congestion_reduction_pct": 12,
            "complexity": "High",
            "actions": [
                "Deploy barricades at entry junctions",
                "Force full detour via Outer Ring Road",
                "Apply fixed detour phase timers at major intersections"
            ]
        },
        "Plan B": {
            "name": "Partial Lane Closure + Strategic Diversions & Manpower",
            "clearance_time_min": plan_b_clearance,
            "avg_speed_kph": 28,
            "congestion_reduction_pct": 35,
            "complexity": "Medium",
            "actions": [
                "Deploy 12 traffic police officers at high-pressure merge lines",
                "Initiate diversion route warning signs 1km upstream",
                "Activate adaptive max-pressure signal timing at Silk Board Junction"
            ]
        },
        "Plan C": {
            "name": "Adaptive Signal Optimization Only",
            "clearance_time_min": plan_c_clearance,
            "avg_speed_kph": 22,
            "congestion_reduction_pct": 20,
            "complexity": "Low",
            "actions": [
                "Request AI adaptive signal overrides at adjacent nodes",
                "No physical detour or manpower deployment"
            ]
        }
    }

    return {
        "plans": plans,
        "recommended": recommended_plan,
        "expected_delay_reduction_pct": plans[recommended_plan]["congestion_reduction_pct"],
        "unmanaged_clearance_time_min": unmanaged_clear_time
    }
