import pandas as pd

from src.risk_engine_v14 import analyze


def analyze_risk(data):
    """
    Run V14 risk analysis on V11-analyzed telemetry.

    Input:
        DataFrame containing V11 z-score columns.

    Output:
        DataFrame containing V14 risk metrics.
    """

    required_columns = [
        "pressure_kpa_z",
        "temperature_k_z",
        "vibration_g_z",
        "thrust_n_z",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:

        missing = ", ".join(
            missing_columns
        )

        raise ValueError(
            "V14 risk analysis requires "
            f"these columns: {missing}"
        )

    return analyze(data)


def summarize_risk(data):
    """
    Create a product-friendly risk summary.
    """

    peak_index = data[
        "overall_risk"
    ].idxmax()

    peak = data.loc[
        peak_index
    ]

    return {
        "overall_risk": float(
            peak["overall_risk"]
        ),

        "risk_level": str(
            peak["risk_level"]
        ),

        "primary_sensor": str(
            peak["primary_risk_sensor"]
        ),

        "peak_time_s": float(
            peak["time_s"]
        ),

        "pressure_risk": float(
            data["pressure_risk"].max()
        ),

        "temperature_risk": float(
            data["temperature_risk"].max()
        ),

        "vibration_risk": float(
            data["vibration_risk"].max()
        ),

        "thrust_risk": float(
            data["thrust_risk"].max()
        ),

        "elevated_sensor_count": int(
            peak["elevated_sensor_count"]
        ),

        "explanation": str(
            peak["risk_explanation"]
        ),
    }