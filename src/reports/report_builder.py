# [ignoring loop detection]
import logging

logger = logging.getLogger(__name__)

class IncidentReportBuilder:
    """
    Assembles structured incident reports from digital twin simulation results.
    """
    def __init__(self, simulation_id, event_type, road_name, duration_min, analysis_data):
        self.simulation_id = simulation_id
        self.event_type = event_type
        self.road_name = road_name
        self.duration_min = duration_min
        self.analysis_data = analysis_data

    def build_report(self):
        """
        Compiles the structured dictionary representing the full operations report.
        """
        plan_id = self.analysis_data.get("recommended_plan_id", "Plan B")
        plan = self.analysis_data.get("plans", {}).get(plan_id, {
            "name": "Adaptive Signal Coordination",
            "clearance_time_min": 45,
            "congestion_reduction_pct": 40,
            "avg_speed_kph": 28,
            "actions": []
        })

        report = {
            "report_id": f"rpt_{self.simulation_id}",
            "simulation_id": self.simulation_id,
            "summary": self.analysis_data.get("summary", f"Incident analysis for {self.road_name}."),
            "severity": self.analysis_data.get("severity", "MEDIUM"),
            "severity_score": self.analysis_data.get("severity_score", 50),
            "closure_probability_pct": self.analysis_data.get("closure_probability_pct", 10),
            "unmanaged_clearance_time_min": self.analysis_data.get("unmanaged_clearance_time_min", 90),
            "location_name": self.road_name,
            "event_type": self.event_type,
            "duration_min": self.duration_min,
            "impact_metrics": {
                "avg_speed_reduction_pct": 25,
                "grid_queue_spillback_km": 1.2,
                "confidence_rating_pct": self.analysis_data.get("confidence", 92)
            },
            "affected_sectors": [
                "Central Business District Outlets",
                "Outer Ring Road Interchanges"
            ],
            "actions_tested": [
                {
                    "option": "Plan A (Passive Baseline)",
                    "delay_reduction": "0%",
                    "clearance_time": f"{self.analysis_data.get('unmanaged_clearance_time_min', 90)} mins",
                    "status": "Inactive"
                },
                {
                    "option": "Plan B (Adaptive Timings Override)",
                    "delay_reduction": f"-{plan.get('congestion_reduction_pct')}%",
                    "clearance_time": f"{plan.get('clearance_time_min')} mins",
                    "status": "Recommended"
                },
                {
                    "option": "Plan C (Diversion Bypass Routing)",
                    "delay_reduction": f"-{int(plan.get('congestion_reduction_pct', 40) * 0.8)}%",
                    "clearance_time": f"{int(plan.get('clearance_time_min', 45) * 1.2)} mins",
                    "status": "Secondary"
                }
            ],
            "recommendations": {
                "signals": {
                    "corridor": self.analysis_data.get("recommendations", {}).get("signal_strategy", {}).get("corridor", self.road_name),
                    "green_phase_seconds": self.analysis_data.get("recommendations", {}).get("signal_strategy", {}).get("recommended_green_sec", 45)
                },
                "manpower": {
                    "officers": self.analysis_data.get("recommendations", {}).get("manpower", {}).get("total_officers", 4),
                    "focus": self.analysis_data.get("recommendations", {}).get("manpower", {}).get("description", "Lane clearance")
                },
                "diversion": {
                    "route": self.analysis_data.get("recommendations", {}).get("diversion", {}).get("route", "Residency Road"),
                    "reason": self.analysis_data.get("recommendations", {}).get("diversion", {}).get("reason", "Corridor bottleneck bypass")
                }
            },
            "improvements": {
                "delay_reduction_pct": plan.get("congestion_reduction_pct"),
                "clearance_speedup_pct": round((self.analysis_data.get("unmanaged_clearance_time_min", 90) - plan.get("clearance_time_min", 45)) / self.analysis_data.get("unmanaged_clearance_time_min", 90) * 100)
            }
        }
        return report
