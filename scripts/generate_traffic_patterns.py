import csv
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
COMMAND_CENTER_ROADS_CSV = DATA_DIR / "command_center_roads.csv"
TRAFFIC_PATTERNS_CSV = DATA_DIR / "bangalore_traffic_patterns.csv"

def classify_road(road_name, road_type):
    name = str(road_name or "").lower()
    rtype = str(road_type or "").lower()
    
    if any(k in name for k in ["silk", "hosur", "btm"]):
        return "Silk Board", "HIGH"
    elif any(k in name for k in ["outer ring", "orr", "bellandur", "sarjapur", "ibblur", "marathahalli"]):
        return "Outer Ring Road", "HIGH"
    elif any(k in name for k in ["whitefield", "itpl", "hope farm", "varthur"]):
        return "Whitefield", "HIGH"
    elif any(k in name for k in ["electronic", "e-city"]):
        return "Electronic City", "HIGH"
    elif any(k in name for k in ["hebbal", "manyata", "bellary", "mekhri"]):
        return "Hebbal", "HIGH"
    elif any(k in name for k in ["mg road", "m.g. road", "majestic", "indiranagar", "koramangala", "church"]):
        return "MG Road Area", "MEDIUM_HIGH"
    elif any(k in rtype for k in ["residential", "service", "living_street"]):
        return "Residential", "LOW"
    else:
        return "Central Bengaluru", "MEDIUM"

def get_base_congestion(hour, day_type, area_tier, road_name, road_type):
    name = str(road_name or "").lower()
    rtype = str(road_type or "").lower()
    
    # 1. Night: 00:00 - 05:00
    if 0 <= hour <= 5:
        # Exceptions: airport road or major highways
        if "bellary" in name or "airport" in name or rtype in ["motorway", "trunk"]:
            return random.uniform(0.30, 0.45)
        return random.uniform(0.10, 0.28)
        
    # 2. Morning Peak: 07:30 - 11:00 (Hours 8, 9, 10)
    elif 8 <= hour <= 10:
        if day_type == "weekday":
            if area_tier == "HIGH":
                return random.uniform(0.78, 0.95)
            elif area_tier == "MEDIUM_HIGH":
                return random.uniform(0.65, 0.82)
            elif area_tier == "MEDIUM":
                return random.uniform(0.48, 0.68)
            else:
                return random.uniform(0.25, 0.45)
        else: # weekend morning peak is light
            if area_tier == "HIGH":
                return random.uniform(0.35, 0.55)
            return random.uniform(0.20, 0.38)
            
    # 3. Afternoon: 11:00 - 16:30 (Hours 11 to 16)
    elif 11 <= hour <= 16:
        if day_type == "weekday":
            if area_tier == "HIGH":
                return random.uniform(0.48, 0.65)
            elif area_tier == "MEDIUM_HIGH":
                return random.uniform(0.40, 0.58)
            return random.uniform(0.30, 0.48)
        else: # weekend afternoon (leisure and shopping areas)
            if area_tier == "MEDIUM_HIGH": # MG Road, Koramangala
                return random.uniform(0.55, 0.72)
            if area_tier == "HIGH":
                return random.uniform(0.40, 0.58)
            return random.uniform(0.30, 0.48)
            
    # 4. Evening Peak: 17:00 - 21:00 (Hours 17 to 20)
    elif 17 <= hour <= 20:
        if day_type == "weekday":
            if area_tier == "HIGH":
                return random.uniform(0.80, 0.98)
            elif area_tier == "MEDIUM_HIGH":
                return random.uniform(0.72, 0.88)
            elif area_tier == "MEDIUM":
                return random.uniform(0.55, 0.75)
            else:
                return random.uniform(0.32, 0.52)
        else: # weekend evening (huge surge in MG Road, Indiranagar, Koramangala)
            if area_tier == "MEDIUM_HIGH":
                return random.uniform(0.72, 0.92)
            elif area_tier == "HIGH": # ORR, Whitefield are quieter
                return random.uniform(0.38, 0.58)
            return random.uniform(0.35, 0.55)
            
    # 5. Late Evening: 21:00 - 00:00 (Hours 21 to 23)
    else:
        if day_type == "weekday":
            if area_tier == "MEDIUM_HIGH": # some action in Koramangala/Indiranagar
                return random.uniform(0.38, 0.55)
            return random.uniform(0.20, 0.38)
        else: # weekend late night
            if area_tier == "MEDIUM_HIGH":
                return random.uniform(0.55, 0.75)
            return random.uniform(0.25, 0.45)

def main():
    if not COMMAND_CENTER_ROADS_CSV.exists():
        print(f"Error: {COMMAND_CENTER_ROADS_CSV} does not exist.")
        return

    random.seed(42) # For reproducibility
    print("Generating synthetic Bengaluru traffic patterns...")

    roads = []
    with open(COMMAND_CENTER_ROADS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            roads.append(row)

    output_rows = []
    for road in roads:
        edge_id = road["edge_id"]
        road_name = road["road_name"]
        road_type = road["road_type"]
        
        area, area_tier = classify_road(road_name, road_type)
        
        # Base speed limit based on road type
        rtype = road_type.lower()
        if "motorway" in rtype or "trunk" in rtype:
            normal_speed = 60.0
        elif "primary" in rtype:
            normal_speed = 45.0
        elif "secondary" in rtype:
            normal_speed = 35.0
            if not normal_speed: normal_speed = 35.0
        elif "tertiary" in rtype:
            normal_speed = 30.0
        else:
            normal_speed = 25.0

        for day_type in ["weekday", "weekend"]:
            for hour in range(24):
                cong = get_base_congestion(hour, day_type, area_tier, road_name, road_type)
                cong = max(0.02, min(0.98, cong))
                
                # Expected speed depends on congestion score
                expected_speed = max(5.0, normal_speed * (1.0 - (cong * 0.78)))
                
                # Traffic density strongly correlates with congestion score
                traffic_density = max(0.02, min(0.98, cong + random.uniform(-0.05, 0.05)))
                
                # Vehicle flow scales with density and road capacity
                capacity = float(road.get("capacity") or 1800.0)
                vehicle_flow = int(capacity * traffic_density * random.uniform(0.8, 1.2))
                
                if cong < 0.35:
                    traffic_level = "LOW"
                elif cong <= 0.7:
                    traffic_level = "MEDIUM"
                else:
                    traffic_level = "HIGH"

                output_rows.append({
                    "edge_id": edge_id,
                    "road_name": road_name,
                    "area": area,
                    "road_type": road_type,
                    "hour": hour,
                    "day_type": day_type,
                    "expected_speed": round(expected_speed, 1),
                    "normal_speed": round(normal_speed, 1),
                    "traffic_density": round(traffic_density, 3),
                    "congestion_score": round(cong, 3),
                    "vehicle_flow": vehicle_flow,
                    "traffic_level": traffic_level
                })

    fieldnames = [
        "edge_id", "road_name", "area", "road_type", "hour", "day_type",
        "expected_speed", "normal_speed", "traffic_density", "congestion_score",
        "vehicle_flow", "traffic_level"
    ]

    with open(TRAFFIC_PATTERNS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Generated {len(output_rows)} traffic pattern entries in {TRAFFIC_PATTERNS_CSV}")

if __name__ == "__main__":
    main()
