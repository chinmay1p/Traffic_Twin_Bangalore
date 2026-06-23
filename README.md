# ✨ Traffic Twin Bengaluru

**AI-powered urban traffic digital twin for Bengaluru Traffic Police — predictive simulation, real-time decision support, and a citizen-facing live traffic dashboard.**

---

## 🌆 What It Does

Two completely separate experiences on one platform:

- 🛡️ **Police Command Center** — Run ML-backed impact simulations for any event or incident, get tactical deployment plans (named locations, officer counts, barricade points, diversion routes), and simulate multi-event city scenarios.
- 🧭 **Citizen Dashboard** — Live city traffic status, upcoming events, active incidents, weather alerts, and a smart route planner that dynamically avoids disruptions.

Both share the same SQLite database, so police-entered events appear on the citizen dashboard instantly, and citizen-submitted incident reports surface in the Command Center immediately.

---

## ✨ Features

### 🛡️ Police Command Center (`/command-center`)
- **Live event queue** — all active and upcoming events sorted by severity, selectable for simulation
- **Single-event ML simulation** — 6 LightGBM models run in under 2 seconds: clearance time, impact score (0–100), barricade intensity (0–100%), road closure decision, officer count, diversion requirement
- **Tactical deployment planner** — translates ML outputs into named deployment locations (e.g., "17 officers — MG Road Junction"), specific barricade points with control percentages, closure segments (from/to junction, duration), and zone-appropriate diversion routes
- **Multi-event city simulation** — saturation-based congestion combination across simultaneous events, spatial overlap detection (auto-upgrades severity and boosts manpower for co-located events), 30-min interval city timeline
- **Response plan** — per-event officer distribution, priority zone ranking, total resource summary
- **GIS map** — color-coded event markers with severity-based impact radius circles, deployment point overlays
- **Weather alert management** — convert IMD/BBMP alerts to trackable police incidents in one click
- **Event CRUD** — add, update, and resolve events including citizen-submitted reports

### 📅 Public Events Page (`/events`)
- Summary strip with live counts (events, incidents, weather alerts)
- Filter by All / Public Events / Incidents / Weather Alerts
- Real-time cards with severity, status, crowd count, location, and timeline

### 🧭 Citizen Dashboard (`/citizen`)
- Dynamic city status chip (CLEAR → LOW → MEDIUM → HIGH) derived from live incident and weather data
- Three-column live feed: upcoming events, active road incidents, weather risks
- **Report a Problem** — citizen incident submissions flow directly to the police Command Center

### 🗺️ Citizen Route Planner (`/citizen/map`)
- Leaflet map pre-loaded with incident, event, and weather markers
- Dijkstra pathfinding on a 41-node / 70-edge Bengaluru road graph
- **Disruption-penalized edge weights**: incidents (up to 1.9× travel time), public events (1.6×), weather alerts (1.7×) within proximity thresholds
- Step-by-step route breakdown with per-segment congestion label
- "Avoided roads" list — shows which disruptions the algorithm bypassed and why

---

## 🤖 ML Pipeline

Six LightGBM models trained on **8,173 real ASTRAM events** from Bengaluru Traffic Police:

| Model | Type | Output |
|---|---|---|
| `clearance_model` | Regressor | Minutes until road clears |
| `impact_model` | Regressor | Road impact score 0–100 |
| `barricade_model` | Regressor | Barricading intensity 0–100% |
| `closure_model` | Classifier | Road closure required YES/NO |
| `manpower_model` | Regressor | Officers to deploy (5–150) |
| `diversion_model` | Classifier | Diversion required YES/NO |

**Feature matrix (21 features):** event cause & category (label-encoded), priority, vehicle type, is-heavy-vehicle flag, GPS coordinates, zone, corridor, junction, police station, hour, day-of-week, peak-hour flags, weekend flag, and an NLP severity score extracted from free-text descriptions via keyword matching.

**Target engineering:** Four of the six targets are engineered from domain knowledge (cause-specific base values + priority multipliers + NLP score) rather than direct labels, since ASTRAM doesn't have explicit columns for barricade%, manpower, or impact score.

**Post-prediction hard rules:** `impact ≥ 88 AND barricade ≥ 85` forces closure; `closure = True` always forces diversion; cause-specific clearance floors prevent operationally implausible predictions.

**Multi-event combination:** Saturation model — `density = 1 − ∏(1 − δᵢ)` — ensures combined congestion is physically bounded. Events within 2.5 km are flagged as overlapping (severity upgraded, manpower boosted 50%).

Retrain models:
```bash
python train_ml_engine.py
```
> Requires `data/astram.csv`.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Flask (Python 3.10+), SQLite |
| ML | LightGBM, scikit-learn, pandas, numpy, joblib |
| Frontend | Vanilla JS, CSS custom properties |
| Maps | Leaflet.js 1.9.4 |
| Icons | Lucide (citizen), Material Icons (command center) |
| Fonts | Outfit, Plus Jakarta Sans |
| Routing | Dijkstra (heap-based, citizen route planner) |
| Spatial math | Haversine distance (deployment planner, route weighting) |

---

## 🚀 Running Locally

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
python app.py
```

Open `http://localhost:5000` — the landing page links to both the Citizen Dashboard and the Police Command Center.

> The SQLite database (`traffic_twin.db`) is created automatically on first run with seed events and weather alerts.

---

## ☁️ Deployment

Deployed on **Vercel** using the `@vercel/python` serverless builder.

- `vercel.json` routes all traffic through `app.py`
- SQLite writes to `/tmp/` on Vercel (detected via `VERCEL` env var)
- ML inference uses **ONNX Runtime** instead of LightGBM directly — avoids the `libgomp.so.1` dependency missing from Vercel's runtime
- Database is seeded fresh on each cold start (in-memory for the function lifetime)
