# [ignoring loop detection]
import json
import logging
from datetime import datetime
from pathlib import Path

from flask import Blueprint, jsonify, request

from routes.dashboard_routes import _major_roads, load_roads

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

event_bp = Blueprint("event_routes", __name__)

BASE_DIR = Path(__file__).resolve().parents[1]

def get_timeline_path(write=False):
    import os
    tmp_path = Path("/tmp/timeline.json")
    bundle_path = BASE_DIR / "outputs" / "timeline.json"
    if "VERCEL" in os.environ:
        if write:
            return tmp_path
        else:
            return tmp_path if tmp_path.exists() else bundle_path
    return bundle_path
SIMULATION_RESULTS = {}

EVENT_LIBRARY = {
    "ipl_match": {
        "event_type": "public_event",
        "road_name": "Cubbon Road",
        "location_name": "Chinnaswamy Stadium",
        "location": {"lat": 12.9788, "lng": 77.5996},
        "keywords": ["cubbon", "m g road", "kasturba", "queen", "brigade", "shivajinagar"],
        "crowd_size": 35000,
        "duration": 180,
        "impact": "HIGH",
    },
    "concert": {
        "event_type": "public_event",
        "road_name": "Jayamahal Road",
        "location_name": "Palace Grounds (Tripura Vasini)",
        "location": {"lat": 12.9985, "lng": 77.5921},
        "keywords": ["jayamahal", "bellary", "cunningham", "palace", "mekhri"],
        "crowd_size": 15000,
        "duration": 150,
        "impact": "MEDIUM",
    },
    "public_gathering": {
        "event_type": "public_event",
        "road_name": "Seshadri Road",
        "location_name": "Gilly's Redefined, Koramangala",
        "location": {"lat": 12.9798, "lng": 77.5734},
        "keywords": ["seshadri", "freedom", "race course", "gandhi nagar", "k g road"],
        "crowd_size": 25000,
        "duration": 120,
        "impact": "MEDIUM",
    },
    "truck_breakdown": {
        "event_type": "vehicle_breakdown",
        "road_name": "Hosur Road",
        "location_name": "Silk Board Junction",
        "location": {"lat": 12.9176, "lng": 77.6244},
        "keywords": ["silk", "hosur", "electronic", "btm", "hsr", "ring road"],
        "duration": 90,
        "impact": "HIGH",
    },
    "tree_fall": {
        "event_type": "tree_fall",
        "road_name": "100 Feet Road",
        "location_name": "Indiranagar 100 Feet Road",
        "location": {"lat": 12.9784, "lng": 77.6408},
        "keywords": ["indiranagar", "100 feet", "cmh", "old airport"],
        "duration": 70,
        "impact": "MEDIUM",
    },
    "water_logging": {
        "event_type": "water_logging",
        "road_name": "Outer Ring Road",
        "location_name": "Outer Ring Road",
        "location": {"lat": 12.9352, "lng": 77.6762},
        "keywords": ["outer ring", "orr", "marathahalli", "bellandur", "ibblur"],
        "duration": 120,
        "impact": "HIGH",
    },
}


def _normalized(text):
    return "".join(c.lower() if c.isalnum() else " " for c in str(text or "")).strip()


def _clamp(value, low, high):
    return max(low, min(high, value))


def _select_impacted_roads(primary_road_name, location_name, keywords, limit=90):
    load_roads()
    primary_name = _normalized(primary_road_name)
    location_key = _normalized(location_name)
    matched = []
    secondary = []

    for road in _major_roads:
        road_name = _normalized(road["road_name"])
        if not road_name:
            continue
        if primary_name and primary_name in road_name:
            matched.append(road)
            continue
        if location_key and location_key in road_name:
            matched.append(road)
            continue
        if any(keyword in road_name for keyword in keywords):
            secondary.append(road)

    seen = set()
    ordered = []
    for group in (matched, secondary):
        for road in group:
            if road["edge_id"] not in seen:
                seen.add(road["edge_id"])
                ordered.append(road)
            if len(ordered) >= limit:
                return ordered
    return ordered[:limit]


def _baseline_from_road(road):
    return {
        "edge_id": road["edge_id"],
        "road_name": road["road_name"],
        "road_type": road["road_type"],
        "congestion": float(road.get("fallback_congestion", road.get("congestion_score", 0.1))),
        "speed": float(road.get("fallback_speed", road.get("current_speed", 30.0))),
    }


def _event_intensity(event_type, impact, crowd_size):
    base = 0.25
    if event_type == "public_event":
        base += min(0.35, crowd_size / 100000.0)
    elif event_type in {"water_logging", "vehicle_breakdown"}:
        base += 0.25
    else:
        base += 0.18
    if impact == "HIGH":
        base += 0.08
    return _clamp(base, 0.18, 0.72)


def _timeline_curve(event_type):
    if event_type == "public_event":
        return {0: 0.55, 15: 0.82, 30: 1.0, 45: 0.92, 60: 0.74, 120: 0.32}
    return {0: 0.68, 15: 1.0, 30: 0.84, 45: 0.62, 60: 0.42, 120: 0.14}


def _build_timeline(event_type, impact, primary_road_name, location_name, keywords, duration, crowd_size=0):
    impacted = _select_impacted_roads(primary_road_name, location_name, keywords)
    if not impacted:
        impacted = _major_roads[:30]

    intensity = _event_intensity(event_type, impact, crowd_size)
    curve = _timeline_curve(event_type)
    snapshots = {}

    for minute, curve_value in curve.items():
        rows = []
        for idx, road in enumerate(impacted):
            baseline = _baseline_from_road(road)
            falloff = 1.0 if idx < 8 else (0.72 if idx < 24 else 0.45)
            delta = intensity * curve_value * falloff
            congestion = _clamp(baseline["congestion"] + delta, 0.04, 0.98)
            speed = max(4.0, baseline["speed"] * (1.0 - (delta * 0.78)))
            rows.append(
                {
                    "edge_id": road["edge_id"],
                    "road_name": road["road_name"],
                    "road_type": road["road_type"],
                    "congestion": round(congestion, 3),
                    "speed": round(speed, 1),
                    "status": "heavy" if congestion > 0.72 else ("moderate" if congestion > 0.38 else "normal"),
                }
            )
        snapshots[f"T+{minute}"] = rows

    delay = int(duration * (0.45 if impact == "HIGH" else 0.3))
    return snapshots, impacted, delay


def _build_recommendations(location_name, road_name, event_type, impact):
    location_label = location_name or road_name or "the affected corridor"
    common = [
        f"Deploy 12 officers around {location_label}",
        f"Barricade the upstream entry feeding {road_name}",
        f"Divert through parallel corridors until flow stabilizes",
    ]
    if event_type == "public_event":
        common[0] = f"Deploy 15 officers around {location_label}"
        common.append("Stage outbound traffic release in waves after the event")
    elif event_type == "water_logging":
        common[1] = f"Close the flooded stretch approaching {location_label}"
        common.append("Hold heavy vehicles upstream until water clears")
    elif event_type == "vehicle_breakdown":
        common.append("Clear the disabled vehicle with tow support immediately")
    return common[:4] if impact == "HIGH" else common[:3]


def _write_timeline(meta, snapshots):
    timeline_path = get_timeline_path(write=True)
    timeline_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "event_meta": meta,
        "timestamps": list(snapshots.keys()),
        "snapshots": snapshots,
    }
    with open(timeline_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


@event_bp.route("/api/events/types", methods=["GET"])
def get_event_types():
    types = [
        {"id": "vehicle_breakdown", "name": "Vehicle Breakdown"},
        {"id": "accident", "name": "Accident"},
        {"id": "tree_fall", "name": "Tree Fall"},
        {"id": "water_logging", "name": "Water Logging"},
        {"id": "public_event", "name": "Public Event"},
    ]
    return jsonify(types)


@event_bp.route("/api/roads/nearest", methods=["GET"])
def api_nearest_road():
    lat = request.args.get("lat", type=float)
    lng = request.args.get("lng", type=float)
    if lat is None or lng is None:
        return jsonify({"error": "Missing lat/lng coordinate parameter"}), 400

    load_roads()
    best = None
    best_score = None
    for road in _major_roads:
        if not road.get("geometry"):
            continue
        mid = road["geometry"][len(road["geometry"]) // 2]
        score = (mid[0] - lat) ** 2 + (mid[1] - lng) ** 2
        if best_score is None or score < best_score:
            best = road
            best_score = score
    if not best:
        return jsonify({"error": "No roads available"}), 404
    return jsonify(
        {
            "edge_id": best["edge_id"],
            "road_name": best["road_name"],
            "road_type": best["road_type"],
            "capacity": best["capacity"],
            "distance": round(best_score ** 0.5, 6) if best_score is not None else None,
        }
    )


@event_bp.route("/api/events/simulate", methods=["POST"])
def simulate_event():
    data = request.json or {}
    logger.info("Simulating event: %s", data)

    scenario_key = data.get("scenario_key")
    preset = EVENT_LIBRARY.get(scenario_key, {})

    event_type = data.get("event_type") or preset.get("event_type", "vehicle_breakdown")
    road_name = data.get("road_name") or preset.get("road_name", "Unknown Road")
    location_name = data.get("location_name") or preset.get("location_name", road_name)
    location = data.get("location") or preset.get("location") or {"lat": 12.9716, "lng": 77.5946}
    duration = int(data.get("duration_min") or data.get("duration") or preset.get("duration", 60))
    params = data.get("parameters") or {}
    crowd_size = int(params.get("crowd_size") or preset.get("crowd_size", 0))
    impact = data.get("impact") or preset.get("impact", "MEDIUM")
    keywords = data.get("keywords") or preset.get("keywords") or [road_name.lower()]

    sim_id = f"sim_{int(datetime.now().timestamp())}"

    snapshots, impacted, expected_delay = _build_timeline(
        event_type=event_type,
        impact=impact,
        primary_road_name=road_name,
        location_name=location_name,
        keywords=keywords,
        duration=duration,
        crowd_size=crowd_size,
    )
    recommendations = _build_recommendations(location_name, road_name, event_type, impact)
    _write_timeline(
        {
            "active_events": 1,
            "scenario_key": scenario_key,
            "event_type": event_type,
            "location_name": location_name,
            "road_name": road_name,
        },
        snapshots,
    )

    result = {
        "simulation_id": sim_id,
        "impact": impact,
        "confidence": 91 if impact == "HIGH" else 86,
        "expected_duration": duration,
        "affected_roads_count": len(impacted),
        "expected_delay": expected_delay,
        "top_affected_roads": [
            {"name": road["road_name"], "congestion": int(snapshots["T+30"][idx]["congestion"] * 100)}
            for idx, road in enumerate(impacted[:5])
        ],
        "time_to_clear": f"{max(30, duration // 2)} min",
        "recommendations": recommendations,
        "location": location,
        "road_name": road_name,
        "location_name": location_name,
        "edge_id": impacted[0]["edge_id"] if impacted else None,
        "scenario_key": scenario_key,
    }
    SIMULATION_RESULTS[sim_id] = result
    return jsonify(result)


@event_bp.route("/api/events/result/<simulation_id>", methods=["GET"])
def get_simulation_result(simulation_id):
    result = SIMULATION_RESULTS.get(simulation_id)
    if not result:
        return jsonify({"error": "Simulation ID not found"}), 404
    return jsonify(result)


from src.simulation.ipl_scenario_engine import ipl_engine

@event_bp.route("/api/ipl/load", methods=["GET"])
def ipl_load():
    return jsonify(ipl_engine.load_ipl_scenario())

@event_bp.route("/api/ipl/simulate/baseline", methods=["POST"])
def ipl_simulate_baseline():
    timeline = ipl_engine.simulate_without_action()
    _write_timeline(
        {
            "active_events": 1,
            "scenario_key": "ipl_match",
            "event_type": "public_event",
            "location_name": "M. Chinnaswamy Stadium",
            "road_name": "Cubbon Road",
        },
        timeline["snapshots"]
    )
    return jsonify(timeline)

@event_bp.route("/api/ipl/suggestions", methods=["GET"])
def ipl_suggestions():
    return jsonify(ipl_engine.generate_response_plan())

@event_bp.route("/api/ipl/apply", methods=["POST"])
def ipl_apply():
    res = ipl_engine.apply_response_plan()
    _write_timeline(
        {
            "active_events": 1,
            "scenario_key": "ipl_match",
            "event_type": "public_event",
            "location_name": "M. Chinnaswamy Stadium",
            "road_name": "Cubbon Road",
        },
        res["timeline"]["snapshots"]
    )
    return jsonify(res)

@event_bp.route("/api/ipl/simulate/custom", methods=["POST"])
def ipl_simulate_custom():
    data = request.json or {}
    barricades = data.get("barricades", [])
    diversions = data.get("diversions", [])
    manpower = int(data.get("manpower", 0))
    res = ipl_engine.simulate_custom_action(barricades, diversions, manpower)
    _write_timeline(
        {
            "active_events": 1,
            "scenario_key": "ipl_match",
            "event_type": "public_event",
            "location_name": "M. Chinnaswamy Stadium",
            "road_name": "Cubbon Road",
        },
        res["timeline"]["snapshots"]
    )
    return jsonify(res)


@event_bp.route("/api/events/reset", methods=["POST"])
def reset_city_state():
    try:
        timeline_path = get_timeline_path(write=True)
        if timeline_path.exists():
            timeline_path.unlink()
        SIMULATION_RESULTS.clear()
        return jsonify({"status": "success", "message": "City flow state restored to baseline."})
    except Exception as exc:
        logger.error("Error resetting city state: %s", exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500
