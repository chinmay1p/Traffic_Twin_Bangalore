# [ignoring loop detection]
"""
Traffic Twin Bengaluru — AI Recommendation Engine
"""

from src.ai.impact_explainer import calculate_severity, explain_reasoning
from src.ai.action_generator import recommend_manpower, recommend_diversion, recommend_signal_strategy
from src.ai.strategy_optimizer import generate_comparison_plans
from src.ai.llm_summary import get_natural_language_summary

def analyze_incident(event_type, road_name, duration_min, parameters=None):
    """
    Main analysis pipeline returning a unified AI assistant recommendation payload.
    """
    parameters = parameters or {}
    duration_min = int(duration_min)

    # 1. Severity Evaluation
    sev_data = calculate_severity(event_type, road_name, duration_min, parameters)
    severity = sev_data["severity"]

    # 2. Natural Language Summary
    summary = get_natural_language_summary(event_type, road_name, duration_min, severity, parameters)

    # 3. Action Recommendations
    crowd_size = int(parameters.get("crowd_size", 0))
    manpower = recommend_manpower(event_type, severity, crowd_size, road_importance="high")
    diversion = recommend_diversion(road_name, event_type)
    signals = recommend_signal_strategy(road_name, event_type)

    # 4. Explanation and Reasoning
    explanations = explain_reasoning(event_type, road_name, severity, duration_min)

    # 5. Plan Comparisons & Strategic Optimization
    optimizer_data = generate_comparison_plans(event_type, road_name, severity, duration_min)

    return {
        "event_type": event_type,
        "location_name": road_name,
        "duration_min": duration_min,
        "severity": severity,
        "severity_score": sev_data["score"],
        "closure_probability_pct": sev_data["closure_probability_pct"],
        "severity_factors": sev_data["factors"],
        "summary": summary,
        "recommendations": {
            "manpower": manpower,
            "diversion": diversion,
            "signal_strategy": signals
        },
        "explanations": explanations,
        "plans": optimizer_data["plans"],
        "recommended_plan_id": optimizer_data["recommended"],
        "expected_delay_reduction_pct": optimizer_data["expected_delay_reduction_pct"],
        "unmanaged_clearance_time_min": optimizer_data["unmanaged_clearance_time_min"]
    }
