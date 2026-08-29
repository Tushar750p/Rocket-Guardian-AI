import pandas as pd
import numpy as np
from pathlib import Path


DATA_DIR = Path("data/processed/v11")
OUTPUT_DIR = Path("data/processed/v14")

SENSORS = {
    "Pressure": "pressure_kpa_z",
    "Temperature": "temperature_k_z",
    "Vibration": "vibration_g_z",
    "Thrust": "thrust_n_z",
}


def sensor_risk(z):
    """
    Convert deviation into a smoother 0-100 risk score.

    The exponential curve prevents moderate deviations
    from immediately saturating at 100.
    """
    z = np.abs(z)

    risk = 100 * (
        1 - np.exp(-z / 5.0)
    )

    return np.clip(risk, 0, 100)


def risk_level(score):

    if score >= 75:
        return "CRITICAL"

    if score >= 45:
        return "WARNING"

    return "NORMAL"


def analyze(data):

    result = data.copy()

    risk_columns = []

    for sensor, z_column in SENSORS.items():

        column = (
            sensor.lower()
            + "_risk"
        )

        result[column] = sensor_risk(
            result[z_column]
        )

        risk_columns.append(column)

    # Highest sensor risk represents the dominant signal.
    max_risk = result[
        risk_columns
    ].max(axis=1)

    # Average risk represents system-wide stress.
    mean_risk = result[
        risk_columns
    ].mean(axis=1)

    # Fusion:
    # 65% strongest signal
    # 35% overall sensor condition
    result["overall_risk"] = (
        0.65 * max_risk
        + 0.35 * mean_risk
    )

    result["overall_risk"] = (
        result["overall_risk"]
        .clip(0, 100)
    )

    result["risk_level"] = (
        result["overall_risk"]
        .apply(risk_level)
    )

    # Identify primary sensor.
    risk_frame = result[
        risk_columns
    ].copy()

    result["primary_risk_sensor"] = (
        risk_frame.idxmax(axis=1)
        .str.replace(
            "_risk",
            "",
            regex=False,
        )
        .str.title()
    )

    # Count how many sensors are elevated.
    result["elevated_sensor_count"] = (
        risk_frame.ge(45).sum(axis=1)
    )

    explanations = []

    for _, row in result.iterrows():

        score = row["overall_risk"]
        primary = row[
            "primary_risk_sensor"
        ]

        elevated = int(
            row["elevated_sensor_count"]
        )

        if score >= 75:

            if elevated >= 2:

                text = (
                    f"Critical multi-sensor deviation. "
                    f"{primary} is the strongest "
                    f"contributor, with "
                    f"{elevated} sensors elevated."
                )

            else:

                text = (
                    f"Critical telemetry deviation "
                    f"primarily associated with "
                    f"{primary}."
                )

        elif score >= 45:

            text = (
                f"Elevated telemetry deviation "
                f"primarily associated with "
                f"{primary}."
            )

        else:

            text = (
                "Telemetry remains within the "
                "expected prototype operating envelope."
            )

        explanations.append(text)

    result["risk_explanation"] = explanations

    return result


def summarize(data, scenario):

    risk_columns = [
        "pressure_risk",
        "temperature_risk",
        "vibration_risk",
        "thrust_risk",
    ]

    peak_index = (
        data["overall_risk"].idxmax()
    )

    peak = data.loc[peak_index]

    return {
        "scenario": scenario,
        "samples": len(data),
        "actual_anomalies": int(
            data["anomaly"].sum()
        ),
        "ai_detections": int(
            data["ai_anomaly"].sum()
        ),
        "peak_overall_risk": round(
            peak["overall_risk"],
            2,
        ),
        "peak_risk_level":
            peak["risk_level"],
        "primary_risk_sensor":
            peak["primary_risk_sensor"],
        "peak_time_s": round(
            peak["time_s"],
            2,
        ),
        "peak_pressure_risk": round(
            data["pressure_risk"].max(),
            2,
        ),
        "peak_temperature_risk": round(
            data["temperature_risk"].max(),
            2,
        ),
        "peak_vibration_risk": round(
            data["vibration_risk"].max(),
            2,
        ),
        "peak_thrust_risk": round(
            data["thrust_risk"].max(),
            2,
        ),
        "explanation":
            peak["risk_explanation"],
    }


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nRocket Guardian AI — V14"
    )
    print(
        "========================="
    )

    files = sorted(
        DATA_DIR.glob(
            "test_*_results.csv"
        )
    )

    if not files:

        print(
            "No V11 result files found."
        )

        return

    summaries = []

    for file in files:

        scenario = (
            file.stem
            .replace("test_", "")
            .replace("_results", "")
        )

        data = pd.read_csv(file)

        analyzed = analyze(data)

        output_file = (
            OUTPUT_DIR
            / f"{scenario}_v14.csv"
        )

        analyzed.to_csv(
            output_file,
            index=False,
        )

        summary = summarize(
            analyzed,
            scenario,
        )

        summaries.append(summary)

        print(
            f"\n{scenario.upper()}"
        )

        print(
            f"Peak Risk       : "
            f"{summary['peak_overall_risk']:.1f}/100"
        )

        print(
            f"Risk Level      : "
            f"{summary['peak_risk_level']}"
        )

        print(
            f"Primary Sensor  : "
            f"{summary['primary_risk_sensor']}"
        )

        print(
            f"Peak Time       : "
            f"{summary['peak_time_s']:.1f}s"
        )

        print(
            f"Pressure Risk   : "
            f"{summary['peak_pressure_risk']:.1f}"
        )

        print(
            f"Temperature Risk: "
            f"{summary['peak_temperature_risk']:.1f}"
        )

        print(
            f"Vibration Risk  : "
            f"{summary['peak_vibration_risk']:.1f}"
        )

        print(
            f"Thrust Risk     : "
            f"{summary['peak_thrust_risk']:.1f}"
        )

        print(
            f"Explanation     : "
            f"{summary['explanation']}"
        )

    summary_df = pd.DataFrame(
        summaries
    )

    summary_file = (
        OUTPUT_DIR
        / "v14_risk_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    print(
        "\nV14 Summary"
    )
    print(
        "------------"
    )

    print(
        summary_df[
            [
                "scenario",
                "peak_overall_risk",
                "peak_risk_level",
                "primary_risk_sensor",
                "peak_time_s",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        f"\nResults saved to: "
        f"{summary_file}"
    )


if __name__ == "__main__":
    main()