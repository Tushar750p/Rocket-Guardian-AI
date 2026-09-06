from __future__ import annotations

from io import BytesIO

import pandas as pd


def build_alerts(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    alert_columns = {
        "Pressure": "pressure_kpa_alert",
        "Temperature": "temperature_k_alert",
        "Vibration": "vibration_g_alert",
        "Thrust": "thrust_n_alert",
    }
    for _, row in data.iterrows():
        active = []
        for sensor, column in alert_columns.items():
            if column in data.columns and bool(row.get(column, False)):
                active.append(sensor)
        risk = float(row.get("overall_risk", 0.0)) if "overall_risk" in data.columns else 0.0
        if active or risk >= 45:
            if risk >= 75:
                severity = "CRITICAL"
            elif risk >= 45 or active:
                severity = "WARNING"
            else:
                severity = "NORMAL"
            rows.append({
                "Time (s)": float(row.get("time_s", 0.0)),
                "Severity": severity,
                "Risk": round(risk, 1),
                "Sensors": ", ".join(active) if active else str(row.get("primary_risk_sensor", "System")),
                "Phase": str(row.get("phase", "-")),
            })
    return pd.DataFrame(rows).sort_values("Time (s)", ascending=False) if rows else pd.DataFrame(columns=["Time (s)", "Severity", "Risk", "Sensors", "Phase"])


def alerts_csv(alerts: pd.DataFrame) -> bytes:
    return alerts.to_csv(index=False).encode("utf-8")


def history_csv(rows: list[dict]) -> bytes:
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def excel_bytes(data: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        data.to_excel(writer, index=False, sheet_name="Telemetry")
    return output.getvalue()
