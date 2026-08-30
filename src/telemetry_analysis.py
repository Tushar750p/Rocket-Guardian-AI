import pandas as pd

from src.telemetry_input import load_telemetry_csv
from src.phase_aware_v11 import (
    load_training,
    build_phase_baseline,
    detect,
)


def analyze_telemetry(file_source):
    """
    Run Rocket Guardian AI telemetry analysis.

    Pipeline:
        CSV
        ↓
        Validation
        ↓
        V11 phase-aware baseline
        ↓
        V11 anomaly detection

    Returns:
        pandas.DataFrame
    """

    # --------------------------------------------------------
    # Load and validate uploaded telemetry
    # --------------------------------------------------------

    data = load_telemetry_csv(
        file_source
    )

    # --------------------------------------------------------
    # Load normal training telemetry
    # --------------------------------------------------------

    training = load_training()

    # --------------------------------------------------------
    # Build phase-specific baseline
    # --------------------------------------------------------

    baseline = build_phase_baseline(
        training
    )

    # --------------------------------------------------------
    # Run V11 anomaly detection
    # --------------------------------------------------------

    analyzed = detect(
        data,
        baseline,
    )

    return analyzed


def summarize_analysis(data):
    """
    Create basic analysis summary.

    Works with both:
        - Research/test telemetry containing 'anomaly'
        - Customer telemetry without 'anomaly'
    """

    summary = {
        "samples": len(data),
        "ai_detections": int(
            data["ai_anomaly"].sum()
        ),
    }

    # --------------------------------------------------------
    # Actual anomaly metrics
    # --------------------------------------------------------

    if "anomaly" in data.columns:

        actual_anomalies = int(
            data["anomaly"].sum()
        )

        summary["actual_anomalies"] = (
            actual_anomalies
        )

        if actual_anomalies > 0:

            anomaly_times = data.loc[
                data["anomaly"] == 1,
                "time_s",
            ]

            detection_times = data.loc[
                data["ai_anomaly"] == 1,
                "time_s",
            ]

            if not anomaly_times.empty:

                start_time = (
                    anomaly_times.iloc[0]
                )

                valid_detection = (
                    detection_times[
                        detection_times >= start_time
                    ]
                )

                if not valid_detection.empty:

                    summary["detection_delay_s"] = (
                        valid_detection.iloc[0]
                        - start_time
                    )

    return summary