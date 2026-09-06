from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd


class MissionAgent:
    """Deterministic multi-agent decision layer over telemetry + risk outputs."""

    @staticmethod
    def _latest(data: pd.DataFrame) -> pd.Series:
        if data.empty:
            raise ValueError("AI agent received an empty telemetry frame.")
        return data.loc[data["time_s"].idxmax()]

    @staticmethod
    def anomaly_agent(data: pd.DataFrame) -> dict:
        row = MissionAgent._latest(data)
        score = float(row["overall_risk"])
        detections = int(data["ai_anomaly"].sum()) if "ai_anomaly" in data else 0
        return {"agent": "Anomaly Agent", "status": str(row["risk_level"]), "score": round(score, 1), "detections": detections, "primary_sensor": str(row["primary_risk_sensor"])}

    @staticmethod
    def sensor_agent(data: pd.DataFrame) -> dict:
        row = MissionAgent._latest(data)
        sensors = {"Pressure": float(row["pressure_risk"]), "Temperature": float(row["temperature_risk"]), "Vibration": float(row["vibration_risk"]), "Thrust": float(row["thrust_risk"])}
        ordered = sorted(sensors.items(), key=lambda item: item[1], reverse=True)
        return {"agent": "Sensor Agent", "top_sensor": ordered[0][0], "top_risk": round(ordered[0][1], 1), "high_risk_sensors": [name for name, risk in ordered if risk >= 75], "sensor_risks": {k: round(v, 1) for k, v in sensors.items()}}

    @staticmethod
    def trend_agent(data: pd.DataFrame) -> dict:
        risks = pd.to_numeric(data["overall_risk"].tail(min(60, len(data))), errors="coerce").fillna(0.0)
        if len(risks) >= 6:
            slope = float(np.polyfit(np.arange(len(risks), dtype=float), risks.to_numpy(), 1)[0])
        else:
            slope = 0.0
        direction = "RISING" if slope >= 0.5 else "FALLING" if slope <= -0.5 else "STABLE"
        return {"agent": "Trend Agent", "direction": direction, "slope": round(slope, 3), "recent_risk": round(float(risks.iloc[-1]), 1)}

    @staticmethod
    def telemetry_health_agent(data: pd.DataFrame) -> dict:
        required = ["time_s", "pressure_kpa", "temperature_k", "vibration_g", "thrust_n"]
        missing = [c for c in required if c not in data.columns]
        if missing:
            return {"agent": "Telemetry Health Agent", "status": "INVALID", "quality": 0, "issues": missing}
        numeric_bad = int(data[required].isna().sum().sum())
        duplicate_times = int(data["time_s"].duplicated().sum())
        finite_bad = int((~np.isfinite(data[required].to_numpy(dtype=float))).sum())
        issues = []
        if numeric_bad:
            issues.append(f"{numeric_bad} missing values")
        if duplicate_times:
            issues.append(f"{duplicate_times} duplicate timestamps")
        if finite_bad:
            issues.append(f"{finite_bad} non-finite values")
        quality = max(0, int(100 - min(100, (numeric_bad + duplicate_times + finite_bad) * 2)))
        status = "GOOD" if quality >= 95 else "DEGRADED" if quality >= 80 else "POOR"
        return {"agent": "Telemetry Health Agent", "status": status, "quality": quality, "issues": issues}

    @staticmethod
    def mission_agent(data: pd.DataFrame) -> dict:
        row = data.loc[data["overall_risk"].idxmax()]
        risk = float(row["overall_risk"])
        trend = MissionAgent.trend_agent(data)
        sensor = MissionAgent.sensor_agent(data)
        if risk >= 75:
            recommendation = "ESCALATE: inspect affected subsystem, verify redundant telemetry, and review the event timeline."
        elif risk >= 45:
            recommendation = "WATCH: maintain high-rate monitoring and verify sensor agreement before escalation."
        else:
            recommendation = "NOMINAL: continue monitoring and watch for trend changes."
        if trend["direction"] == "RISING" and risk >= 45:
            recommendation += " Risk trend is rising."
        return {"agent": "Mission Agent", "mission_state": str(row["risk_level"]), "peak_time_s": round(float(row["time_s"]), 2), "primary_sensor": sensor["top_sensor"], "recommendation": recommendation}

    @staticmethod
    def incident_agent(data: pd.DataFrame) -> dict:
        if "overall_risk" not in data:
            return {"agent": "Incident Agent", "active": False, "event_count": 0, "events": []}
        critical = data[data["overall_risk"] >= 75].copy()
        events = []
        if not critical.empty:
            groups = (critical["time_s"].diff().fillna(0) > 2.0).cumsum()
            for _, group in critical.groupby(groups):
                peak_index = group["overall_risk"].idxmax()
                events.append({"start_s": round(float(group.iloc[0]["time_s"]), 2), "end_s": round(float(group.iloc[-1]["time_s"]), 2), "peak_risk": round(float(group["overall_risk"].max()), 1), "sensor": str(group.loc[peak_index, "primary_risk_sensor"])})
        return {"agent": "Incident Agent", "active": bool(events), "event_count": len(events), "events": events[-10:]}

    @classmethod
    def run(cls, data: pd.DataFrame) -> dict:
        result = {"anomaly": cls.anomaly_agent(data), "sensor": cls.sensor_agent(data), "trend": cls.trend_agent(data), "telemetry": cls.telemetry_health_agent(data), "mission": cls.mission_agent(data), "incident": cls.incident_agent(data)}
        result["generated_at"] = datetime.now(timezone.utc).isoformat()
        return result
