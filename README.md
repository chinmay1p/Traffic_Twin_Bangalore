# 🚦 Traffic Twin Bengaluru
### *Predictive Traffic Simulation & AI Decision Support Platform*

Traffic Twin Bengaluru is a smart traffic management platform that creates a digital representation of Bengaluru’s road network. It predicts congestion, simulates traffic disruptions, and assists authorities in selecting the optimal response strategy in real-time.

The system combines historical traffic patterns, Bengaluru incident data, road network intelligence, and machine learning models to transition traffic management from reactive control to predictive scheduling.

---

## 🌟 Key Features

### 🗺️ Bengaluru Traffic Digital Twin
A virtual city traffic environment modeled using Bengaluru's major road networks and transport corridors.
* **Interactive City Map**: Visualized through a customized Leaflet web interface.
* **Real-time Metrics**: Tracks density, average speed, vehicle flow, and capacity per segment.
* **Density States**: Dynamic congestion mapping (Low, Medium, High) represented with standard color codes.


### ⚠️ Event Monitoring & NLP Severity Extraction
Ingests and predicts impacts from public gatherings, sports matches (e.g., IPL), breakdowns, and water logging:
* **Feature Vectorization**: Encodes locations, junctions, corridors, police zones, timings, and priority levels.
* **NLP Parse pipeline**: Processes citizen and officer descriptions to identify crowd patterns, lane blockage levels, and emergency severity automatically.

### ⌛ Clearance Time Prediction Model
Predicts how long an event will impact the road network using LightGBM regressor models:
* **Recovery Estimates**: Calculates event duration and network recovery window.
* **Impact Levels**: Classifies overall severity to determine deployment scales.

### 🕸️ Graph-Based Congestion Propagation (ST-GNN)
 Bengaluru roads are modeled as a connected graph where junctions are nodes and roads are edges:
* **Spatial Diffusion**: Simulates how a local bottleneck propagates to surrounding roads.
* **Impact Radius**: Numerically predicts affected zones using Spatio-Temporal Graph Neural Networks (ST-GNN).

---

## 🛠️ Response Planning System

The platform generates recommended action plans to mitigate congestion spikes:

| Strategy | Action Item | Target Parameter |
| :--- | :--- | :--- |
| **🚧 Barricading** | Cones, partial barricades, or full closure | Lane availability |
| **🔀 Diversions** | Alternate route generation using weighted shortest paths | Vehicle redistribution |
| **👮 Police Deployment** | Estimating officers needed per zone | Field enforcement |

---

## 💻 Tech Stack

* **Backend**: Flask (Python)
* **Frontend**: HTML5, Vanilla CSS (Premium glassmorphic styling), JavaScript
* **Mapping**: Leaflet.js
* **ML Core**: LightGBM, Scikit-learn, joblib, PyTorch (ST-GNN implementation)

---

## 🚀 How to Run & Use the Project

### 1. Prerequisites
Ensure you have **Python 3.8+** installed.

### 2. Setup Virtual Environment
Clone the repository, navigate to the folder, and activate the virtual environment:
```bash
# Create environment
python -m venv .venv

# Activate environment (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
*(If `requirements.txt` is not present, install core dependencies: `pip install flask numpy pandas scikit-learn joblib torch scipy shap`)*

### 3. Initialize & Visualize the Road Network
Generate the primary city graph HTML visualization:
```bash
python scripts/visualize_graph.py
```

### 4. Start the Server
Launch the Flask development server:
```bash
python app.py
```
Open **`http://127.0.0.1:5000/`** in your browser to view the Command Center.


---

## 📊 Workflow Overview
```
Traffic Data + Road network + Event History
            ↓
    Digital Traffic Twin
            ↓
  Event Impact Prediction
            ↓
  Congestion Simulation (ST-GNN)
            ↓
    Response Planning
            ↓
 Optimized Traffic Management
```
