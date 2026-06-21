import csv
import json
import logging
import math
import re
from pathlib import Path

from flask import Blueprint, jsonify, request
from src.utils.road_filter import load_or_create_command_center_roads
from src.traffic.local_traffic_engine import traffic_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

dashboard_bp = Blueprint("dashboard", __name__)

BASE_DIR = Path(__file__).resolve().parents[1]
EDGES_CSV = BASE_DIR / "data" / "edges.csv"
COMMAND_CENTER_ROADS_CSV = BASE_DIR / "data" / "command_center_roads.csv"
JUNCTIONS_CSV = BASE_DIR / "data" / "major_junctions.csv"

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

_major_roads = []
_junctions = []
_roads_loaded = False
_junctions_loaded = False
_timeline_cache = {"mtime": None, "data": None}

transformer = None
try:
    import pyproj

    transformer = pyproj.Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)
except Exception:
    transformer = None


def utm_to_latlon(easting, northing, zone_number=43, northern_hemisphere=True):
    if not northern_hemisphere:
        northing = 10000000 - northing
    a = 6378137
    e = 0.081819191
    e1sq = 0.006739497
    k0 = 0.9996
    arc = northing / k0
    mu = arc / (a * (1 - math.pow(e, 2) / 4.0 - 3 * math.pow(e, 4) / 64.0 - 5 * math.pow(e, 6) / 256.0))
    ei = (1 - math.pow((1 - e * e), (1 / 2.0))) / (1 + math.pow((1 - e * e), (1 / 2.0)))
    ca = 3 * ei / 2 - 27 * math.pow(ei, 3) / 32.0
    cb = 21 * math.pow(ei, 2) / 16 - 55 * math.pow(ei, 4) / 32
    cc = 151 * math.pow(ei, 3) / 96
    cd = 1097 * math.pow(ei, 4) / 512
    phi1 = mu + ca * math.sin(2 * mu) + cb * math.sin(4 * mu) + cc * math.sin(6 * mu) + cd * math.sin(8 * mu)
    n0 = a / math.pow((1 - math.pow((e * math.sin(phi1)), 2)), (1 / 2.0))
    r0 = a * (1 - e * e) / math.pow((1 - math.pow((e * math.sin(phi1)), 2)), (3 / 2.0))
    fact1 = n0 * math.tan(phi1) / r0
    _a1 = easting - 500000
    dd0 = _a1 / (n0 * k0)
    fact2 = dd0 * dd0 / 2
    t0 = math.pow(math.tan(phi1), 2)
    Q0 = e1sq * math.pow(math.cos(phi1), 2)
    fact3 = (5 + 3 * t0 + 10 * Q0 - 4 * Q0 * Q0 - 9 * e1sq) * math.pow(dd0, 4) / 24
    lat = (phi1 - fact1 * (fact2 + fact3))
    fact6 = (1 + 2 * t0 + Q0) * math.pow(dd0, 3) / 6
    longitude = (zone_number - 1) * 6 - 180 + 3
    lon = (dd0 - fact6) / math.cos(phi1)
    return math.degrees(lat), longitude + math.degrees(lon)

MAJOR_TYPES = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
}

HOTSPOT_PROFILES = {
    "silk board": {"keywords": ["silk", "hosur", "btm", "electronic", "hsr"], "boost": 0.45},
    "orr": {"keywords": ["outer ring", "orr", "marathahalli", "bellandur", "ibblur"], "boost": 0.34},
    "marathahalli": {"keywords": ["marathahalli", "kalamandir", "multiplex"], "boost": 0.28},
    "kr puram": {"keywords": ["kr puram", "tin factory", "hanging bridge", "old madras"], "boost": 0.3},
    "hebbal": {"keywords": ["hebbal", "mekhri", "airport"], "boost": 0.28},
    "whitefield": {"keywords": ["whitefield", "itpl", "hope farm"], "boost": 0.25},
    "majestic": {"keywords": ["majestic", "k g road", "mysore road"], "boost": 0.25},
    "koramangala": {"keywords": ["koramangala", "sony world", "80 feet"], "boost": 0.2},
    "indiranagar": {"keywords": ["indiranagar", "100 feet", "cmh"], "boost": 0.18},
}


def _safe_float(value, default=0.0):
    try:
        if value in (None, "", "nan"):
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value, default=0):
    try:
        if value in (None, "", "nan"):
            return default
        return int(float(value))
    except Exception:
        return default


def _clamp(value, low, high):
    return max(low, min(high, value))


def parse_wkt(wkt_str):
    if not wkt_str or not isinstance(wkt_str, str):
        return []
    match = re.search(r"LINESTRING\s*\((.*?)\)", wkt_str, re.IGNORECASE)
    if not match:
        return []
    coords = []
    for pt in match.group(1).split(","):
        parts = pt.strip().split()
        if len(parts) < 2:
            continue
        try:
            x, y = float(parts[0]), float(parts[1])
            if transformer is not None:
                lng, lat = transformer.transform(x, y)
            else:
                lat, lng = utm_to_latlon(x, y, zone_number=43)
            coords.append([lat, lng])
        except Exception:
            continue
    return coords


def _normalized(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _hotspot_boost(road_name):
    name = _normalized(road_name)
    if not name:
        return 0.0
    boost = 0.0
    for profile in HOTSPOT_PROFILES.values():
        if any(keyword in name for keyword in profile["keywords"]):
            boost = max(boost, profile["boost"])
    return boost


def _baseline_state_for_road(road):
    current_congestion = _safe_float(road.get("congestion_score"), 0.0)
    road_type = str(road.get("road_type", "unclassified")).lower()
    if "motorway" in road_type or "trunk" in road_type:
        type_floor = 0.18
        speed_limit = 58.0
    elif "primary" in road_type:
        type_floor = 0.13
        speed_limit = 44.0
    elif "secondary" in road_type:
        type_floor = 0.09
        speed_limit = 36.0
    elif "tertiary" in road_type:
        type_floor = 0.06
        speed_limit = 30.0
    else:
        type_floor = 0.04
        speed_limit = 26.0

    jitter_seed = sum(ord(c) for c in str(road.get("edge_id", ""))) % 17
    jitter = (jitter_seed / 100.0) - 0.04
    congestion = max(current_congestion, type_floor + _hotspot_boost(road.get("road_name")) + jitter)
    congestion = _clamp(congestion, 0.03, 0.96)

    base_speed = _safe_float(road.get("current_speed"), _safe_float(road.get("speed"), speed_limit))
    speed = max(6.0, min(speed_limit, base_speed if base_speed > 0 else speed_limit))
    speed = max(6.0, speed * (1.0 - (congestion * 0.52)))
    return round(congestion, 3), round(speed, 1)


def load_roads():
    global _roads_loaded
    if _roads_loaded:
        return
    source_csv = COMMAND_CENTER_ROADS_CSV if COMMAND_CENTER_ROADS_CSV.exists() else None
    if source_csv is None:
        try:
            source_csv = load_or_create_command_center_roads()
        except Exception as exc:
            logging.warning("Command center road filter unavailable, falling back to edges.csv: %s", exc)
            source_csv = EDGES_CSV

    if not source_csv.exists():
        logging.error("Road source csv not found: %s", source_csv)
        return

    logging.info("Loading roads from %s", source_csv)
    major_list = []

    with open(source_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            geom_wkt = row.get("geometry")
            if not geom_wkt:
                continue

            capacity_val = row.get("capacity")
            speed_val = row.get("speed") or row.get("current_speed")
            congestion_val = row.get("traffic_density") or row.get("congestion_score")

            road_item = {
                "edge_id": str(row.get("edge_id", "")),
                "road_name": str(row.get("road_name") or "Unknown"),
                "road_type": str(row.get("road_type") or "unclassified"),
                "capacity": _safe_int(capacity_val, 1800),
                "current_speed": _safe_float(speed_val, 30.0),
                "congestion_score": _safe_float(congestion_val, 0.0),
                "geometry_wkt": geom_wkt,
            }
            fallback_congestion, fallback_speed = _baseline_state_for_road(road_item)
            road_item["fallback_congestion"] = fallback_congestion
            road_item["fallback_speed"] = fallback_speed

            coords = parse_wkt(geom_wkt)
            if not coords:
                continue
            road_item["geometry"] = coords
            road_item["min_lat"] = min(c[0] for c in coords)
            road_item["max_lat"] = max(c[0] for c in coords)
            road_item["min_lng"] = min(c[1] for c in coords)
            road_item["max_lng"] = max(c[1] for c in coords)
            major_list.append(road_item)

    _major_roads.clear()
    _major_roads.extend(major_list)
    _roads_loaded = True
    logging.info("Loaded %d operational roads for command center view.", len(_major_roads))


def load_junctions():
    global _junctions_loaded
    if _junctions_loaded:
        return
    if not JUNCTIONS_CSV.exists():
        logging.error("major_junctions.csv not found")
        return

    junctions_list = []
    with open(JUNCTIONS_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            junctions_list.append(
                {
                    "junction_id": str(row.get("junction_id", "")),
                    "lat": _safe_float(row.get("lat")),
                    "lng": _safe_float(row.get("lng")),
                    "connected_roads": str(row.get("connected_roads", "")),
                    "traffic_signal_available": _safe_int(row.get("traffic_signal_available"), 0),
                    "importance": str(row.get("importance", "medium")),
                }
            )
    _junctions.clear()
    _junctions.extend(junctions_list)
    _junctions_loaded = True


def _read_timeline():
    timeline_path = get_timeline_path(write=False)
    if not timeline_path.exists():
        _timeline_cache["mtime"] = None
        _timeline_cache["data"] = None
        return None
    mtime = timeline_path.stat().st_mtime
    if _timeline_cache["mtime"] == mtime:
        return _timeline_cache["data"]
    try:
        with open(timeline_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _timeline_cache["mtime"] = mtime
        _timeline_cache["data"] = data
        return data
    except Exception as exc:
        logging.error("Error reading timeline json: %s", exc)
        return None


def _timeline_snapshot(minutes):
    data = _read_timeline()
    if not data:
        return None
    snapshots = data.get("snapshots", {})
    if not snapshots:
        return None
    # Direct match first
    if f"T+{minutes}" in snapshots:
        return snapshots[f"T+{minutes}"]
    # Fallback to key index or name
    keys = list(snapshots.keys())
    # If the request matches index, return it
    if 0 <= minutes < len(keys):
        return snapshots[keys[minutes]]
    # Else fallback to closest numeric
    try:
        valid_steps = []
        for k in keys:
            m = re.findall(r'\d+', k)
            if m:
                valid_steps.append(int(m[0]))
        if valid_steps:
            closest = min(valid_steps, key=lambda x: abs(x - minutes))
            for k in keys:
                if str(closest) in k:
                    return snapshots[k]
    except Exception:
        pass
    return snapshots[keys[0]]


def _snapshot_state_map(snapshot):
    state = {}
    if not snapshot:
        return state
    for row in snapshot:
        state[str(row.get("edge_id"))] = {
            "congestion_score": _safe_float(row.get("congestion"), _safe_float(row.get("congestion_score"), 0.0)),
            "current_speed": _safe_float(row.get("speed"), _safe_float(row.get("current_speed"), 30.0)),
        }
    return state


def _aggregate_metrics(snapshot_state=None):
    load_roads()
    total_congestion = 0.0
    total_speed = 0.0
    critical = 0
    count = 0
    snapshot_state = snapshot_state or {}
    for road in _major_roads:
        state = snapshot_state.get(road["edge_id"], None)
        congestion = state["congestion_score"] if state else road["fallback_congestion"]
        speed = state["current_speed"] if state else road["fallback_speed"]
        total_congestion += congestion
        total_speed += speed
        critical += 1 if congestion >= 0.7 else 0
        count += 1
    if count == 0:
        return {"avg_congestion": 0.0, "avg_speed": 0.0, "critical_roads": 0}
    return {
        "avg_congestion": round((total_congestion / count) * 100, 1),
        "avg_speed": round(total_speed / count, 1),
        "critical_roads": critical,
    }


def _update_road_fallbacks_with_current_time():
    load_roads()
    hour, day_type, _ = traffic_engine.get_current_time_info()
    for road in _major_roads:
        state = traffic_engine.get_traffic_state(road["edge_id"], hour, day_type)
        road["fallback_congestion"] = state["congestion_score"]
        road["fallback_speed"] = state["expected_speed"]


@dashboard_bp.route("/api/city/state", methods=["GET"])
def api_city_state():
    _update_road_fallbacks_with_current_time()
    snapshot = _timeline_snapshot(request.args.get("time", default=0, type=int))
    snapshot_state = _snapshot_state_map(snapshot)
    metrics = _aggregate_metrics(snapshot_state)
    timeline_data = _read_timeline() or {}
    event_meta = timeline_data.get("event_meta", {})
    hour, day_type, time_desc = traffic_engine.get_current_time_info()
    return jsonify(
        {
            **metrics,
            "active_events": event_meta.get("active_events", 0),
            "city_flow": max(0, round(100 - metrics["avg_congestion"], 1)),
            "time_desc": time_desc,
        }
    )


@dashboard_bp.route("/api/roads", methods=["GET"])
def api_roads():
    _update_road_fallbacks_with_current_time()
    zoom = request.args.get("zoom", type=int)
    min_lat = request.args.get("min_lat", type=float)
    max_lat = request.args.get("max_lat", type=float)
    min_lng = request.args.get("min_lng", type=float)
    max_lng = request.args.get("max_lng", type=float)
    timeline_state = _snapshot_state_map(_timeline_snapshot(0))

    if zoom is not None and zoom <= 11:
        active_types = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link"}
    else:
        active_types = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link"}

    payload = []
    for road in _major_roads:
        if road["road_type"] not in active_types:
            continue
        if min_lat is not None and max_lat is not None and min_lng is not None and max_lng is not None:
            if not (road["max_lat"] >= min_lat and road["min_lat"] <= max_lat and road["max_lng"] >= min_lng and road["min_lng"] <= max_lng):
                continue
        state = timeline_state.get(road["edge_id"], {})
        payload.append(
            {
                "edge_id": road["edge_id"],
                "road_name": road["road_name"],
                "road_type": road["road_type"],
                "capacity": road["capacity"],
                "current_speed": round(state.get("current_speed", road["fallback_speed"]), 1),
                "congestion_score": round(state.get("congestion_score", road["fallback_congestion"]), 3),
                "geometry": road["geometry"],
            }
        )

    return jsonify(payload)


@dashboard_bp.route("/api/junctions", methods=["GET"])
def api_junctions():
    load_junctions()
    zoom = request.args.get("zoom", type=int)
    if zoom is not None:
        if zoom < 12:
            filtered = [j for j in _junctions if j["importance"] == "high"]
        elif zoom < 14:
            filtered = [j for j in _junctions if j["importance"] in {"high", "medium"}]
        else:
            filtered = _junctions
    else:
        filtered = _junctions
    return jsonify(filtered)


@dashboard_bp.route("/api/traffic/timeline", methods=["GET"])
def api_timeline():
    minutes = request.args.get("time", default=0, type=int)

    snapshot = _timeline_snapshot(minutes)
    if snapshot is not None:
        road_states = _snapshot_state_map(snapshot)
        metrics = _aggregate_metrics(road_states)
        return jsonify({**metrics, "roads": road_states})

    valid_steps = [0, 15, 30, 45, 60, 120]
    minutes = min(valid_steps, key=lambda x: abs(x - minutes))

    _update_road_fallbacks_with_current_time()
    road_states = {}
    for road in _major_roads:
        base_cong = road["fallback_congestion"]
        phase = (sum(ord(c) for c in road["edge_id"]) % 11) / 10.0
        wave = math.sin((minutes / 20.0) + phase) * 0.06
        congestion = _clamp(base_cong + wave, 0.03, 0.96)
        speed = max(6.0, road["fallback_speed"] * (1.0 - (congestion - road["fallback_congestion"]) * 0.35))
        road_states[road["edge_id"]] = {"congestion_score": round(congestion, 3), "current_speed": round(speed, 1)}

    metrics = _aggregate_metrics(road_states)
    return jsonify({**metrics, "roads": road_states})


@dashboard_bp.route("/api/traffic/current", methods=["GET"])
def api_traffic_current():
    _update_road_fallbacks_with_current_time()
    hour, day_type, time_desc = traffic_engine.get_current_time_info()
    road_states = {}
    for road in _major_roads:
        road_states[road["edge_id"]] = {
            "congestion_score": road["fallback_congestion"],
            "current_speed": road["fallback_speed"]
        }
    return jsonify({
        "time_desc": time_desc,
        "hour": hour,
        "day_type": day_type,
        "roads": road_states
    })
