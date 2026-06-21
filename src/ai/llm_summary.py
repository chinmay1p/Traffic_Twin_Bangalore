# [ignoring loop detection]
"""
Traffic Twin Bengaluru — Optional LLM Summary Converter
"""
import os
import requests

def get_natural_language_summary(event_type, road_name, duration_min, severity, parameters=None):
    """
    Generates a natural language summary of the incident state.
    Uses an LLM if API key is present in environment, else falls back to robust rules.
    """
    parameters = parameters or {}
    
    # ── Rule-Based Generation (Always available / Fallback) ──
    event_label = event_type.replace("_", " ").title()
    time_str = "morning peak hours" if severity in ["HIGH", "CRITICAL"] else "off-peak hours"
    
    # Bengaluru-specific flavor
    summary_text = (
        f"{severity} impact disruption alert: A {event_label} detected on {road_name}. "
        f"Under current {time_str} traffic load, the incident is predicted to block lane flow, "
        f"causing significant queue spillover and gridlock risks extending toward nearest intersections. "
        f"Active response planning is recommended."
    )
    
    # Detailed dynamic text based on event type
    if event_type == "public_event":
        venue = parameters.get("venue", "Cubbon Road Stadium")
        crowd = parameters.get("crowd_size", 35000)
        summary_text = (
            f"Major crowd congestion shock: Public event ending at {venue} with an expected crowd exit "
            f"of {crowd:,} people. High pedestrian density and vehicle surge will overload {road_name} and "
            f"adjacent corridors. Critical outbound routing interventions required."
        )
    elif event_type == "vehicle_breakdown":
        v_type = parameters.get("vehicle_type", "Heavy Truck")
        summary_text = (
            f"Corridor obstruction: A {v_type} breakdown on {road_name} is blocking traffic. "
            f"During the current high volume period, this blockage will trigger vehicle queuing "
            f"stretching back to major feeder points. Manpower and towing support required."
        )
    elif event_type == "water_logging":
        summary_text = (
            f"Critical hazard: Severe waterlogging reported on {road_name}. "
            f"Road capacity is reduced by over 60%, slowing vehicle movement to a crawl. "
            f"Officers should divert vehicles to parallel drainage-clear corridors immediately."
        )

    # ── Optional LLM Call (If API key exists) ──
    # Check for GEMINI_API_KEY or other keys
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        try:
            # Construct a clear prompt directing the LLM to only rephrase and format
            prompt = (
                f"You are an AI Police Dispatch Assistant for Bangalore Traffic Control.\n"
                f"Rephrase this raw traffic incident data into a concise, professional civic-tech command center report.\n"
                f"Keep it to exactly 2-3 sentences. No chat filler. No emojis.\n\n"
                f"Raw Data:\n"
                f"- Event: {event_label}\n"
                f"- Corridor: {road_name}\n"
                f"- Duration: {duration_min} minutes\n"
                f"- Severity: {severity}\n"
                f"- Factors: {', '.join(parameters.keys())}\n"
            )
            
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={gemini_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": prompt}]}]
            }
            
            res = requests.post(url, json=payload, headers=headers, timeout=4)
            if res.status_code == 200:
                result_json = res.json()
                text = result_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                if text:
                    return text
        except Exception:
            # Silently fall back to standard text on any API error or timeout
            pass

    return summary_text
