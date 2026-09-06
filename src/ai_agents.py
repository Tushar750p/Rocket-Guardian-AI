from __future__ import annotations

import pandas as pd


class MissionAgent:
    """Deterministic safety-oriented agents over the existing AI/risk outputs."""

    @staticmethod
    def anomaly_agent(data: pd.DataFrame) -> dict:
        row = data.loc[data["overall_risk"].idxmax()]
        score = float(row["overall_risk"])
        detections = int(data["ai_anomaly"].sum()) if "ai_anomaly" in data else 0
        return {
            "agent": "Anomaly Agent",
            "status": str(row["risk_level"]),
            "score": round(score, 1),
            "detections": detections,
            "primary_sensor": str(row["primary_risk_sensor"]),
        }

    @staticmethod
    def sensor_agent(data: pd.DataFrame) -> dict:
        row = data.loc[data["overall_risk"].idxmax()]
        sensors = {
            "Pressure": float(row["pressure_risk"]),
            "Temperature": float(row["temperature_risk"]),
            "Vibration": float(row["vibration_risk"]),
            "Thrust": float(row["thrust_risk"]),
        }
        ordered = sorted(sensors.items(), key=lambda item: item[1], reverse=True)
        return {
            "agent": "Sensor Agent",
            "top_sensor": ordered[0][0],
            "top_risk": round(ordered[0][1], 1),
            "sensor_risks": {k: round(v, 1) for k, v in sensors.items()},
        }

    @staticmethod
    def mission_agent(data: pd.DataFrame) -> dict:
        row = data.loc[data["overall_risk"].idxmax()]
        risk = float(row["overall_risk"])
        if risk >= 75:
            recommendation = "ESCALATE: inspect affected subsystem and verify redundant telemetry."
        elif risk >= 45:
            recommendation = "WATCH: continue high-rate monitoring and verify sensor agreement."
        else:
            recommendation = "NOMINAL: continue monitoring."
        return {
            "agent": "Mission Agent",
            "mission_state": str(row["risk_level"]),
            "peak_time_s": round(float(row["time_s"]), 2),
            "recommendation": recommendation,
        }

    @classmethod
    def run(cls, data: pd.DataFrame) -> dict:
        return {
            "anomaly": cls.anomaly_agent(data),
            "sensor": cls.sensor_agent(data),
            "mission": cls.mission_agent(data),
        }
