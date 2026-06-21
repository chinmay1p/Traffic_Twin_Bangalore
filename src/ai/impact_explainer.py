# [ignoring loop detection]
"""
Traffic Twin Bengaluru — AI Impact Explainer
"""

def calculate_severity(event_type, road_name, duration_min, parameters=None):
    """
    Computes incident severity score and factors.
    """
    parameters = parameters or {}
    score = 10 # base score

    factors = []

    # 1. Road importance check
    is_major = any(keyword in road_name.lower() for keyword in ["outer ring road", "orr", "silk board", "mg road", "hosur road", "nh-", "highway", "flyover"])
    if is_major:
        score += 30
        factors.append("Major corridor route")
    else:
        factors.append("Secondary local road")

    # 2. Event type check
    if event_type in ["public_event", "vip_movement", "procession"]:
        score += 25
        factors.append(f"Public movement/event: high pedestrian overlap")
    elif event_type in ["accident", "construction"]:
        score += 20
        factors.append("Active road lane physical blockage")
    else:
        score += 10
        factors.append("Minor flow obstruction")

    # 3. Duration check
    if duration_min >= 120:
        score += 20
        factors.append("Extended event duration (> 2 hours)")
    elif duration_min >= 60:
        score += 10
        factors.append("Moderate duration (1-2 hours)")

    # 4. Parameters check (e.g. crowd size, blocked lanes)
    crowd_size = int(parameters.get("crowd_size", 0))
    if crowd_size > 20000:
        score += 20
        factors.append("Mass crowd exit shock")
    
    affected_lanes = str(parameters.get("affected_lanes", ""))
    if "full" in affected_lanes.lower() or "all" in affected_lanes.lower():
        score += 15
        factors.append("Complete corridor closure")

    # Map score to severity levels
    if score >= 75:
        severity = "CRITICAL"
        closure_prob = 85
    elif score >= 50:
        severity = "HIGH"
        closure_prob = 62
    elif score >= 30:
        severity = "MEDIUM"
        closure_prob = 35
    else:
        severity = "LOW"
        closure_prob = 10

    return {
        "severity": severity,
        "score": score,
        "closure_probability_pct": closure_prob,
        "factors": factors
    }

def explain_reasoning(event_type, road_name, severity, duration_min):
    """
    Generates bulleted reasoning explanations for recommending specific police actions.
    """
    reasons = []

    if severity in ["CRITICAL", "HIGH"]:
        reasons.append(f"Predicted spillover congestion is expected to exceed 85% on adjacent segments if left unmanaged.")
    else:
        reasons.append(f"Local traffic speed is expected to drop by 20% on the surrounding block.")

    if duration_min > 60:
        reasons.append(f"Event duration ({duration_min} minutes) is longer than typical buffer queues can absorb without active diversions.")
    else:
        reasons.append("Short incident duration suggests localized staging area control is sufficient.")

    if any(k in road_name.lower() for k in ["silk board", "orr", "outer ring road", "mg road"]):
        reasons.append(f"Corridor '{road_name}' carries a high daily vehicle load; single lane blockages propagate rapidly into gridlock.")
    else:
        reasons.append("Secondary road category allows for simple signal-guided local bypasses.")

    return reasons
