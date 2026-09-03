import pandas as pd
import numpy as np
from pathlib import Path


DATA_DIR = Path("data/v11")
OUTPUT_DIR = Path("data/processed/v11")

SENSORS = [
    "pressure_kpa",
    "temperature_k",
    "vibration_g",
    "thrust_n",
]


def load_training():
    files = sorted(
        DATA_DIR.glob("training_normal_*.csv")
    )

    return pd.concat(
        [pd.read_csv(f) for f in files],
        ignore_index=True,
    )


def build_phase_baseline(training):
    baseline = {}

    for phase in training["phase"].unique():

        phase_data = training[
            training["phase"] == phase
        ]

        baseline[phase] = {}

        for sensor in SENSORS:

            baseline[phase][sensor] = {
                "mean": phase_data[sensor].mean(),
                "std": phase_data[sensor].std(),
            }

    return baseline


def detect(data, baseline):
    """
    Run phase-aware anomaly detection.

    Adds:
        - Per-sensor z-scores
        - Persistent sensor alerts
        - Active sensor count
        - Mission status
        - AI anomaly flag
        - Overall anomaly score
        - Confidence score
        - Primary risk sensor
    """

    result = data.copy()

    sensor_alerts = []
    sensor_z_scores = {}

    for sensor in SENSORS:

        z_scores = np.zeros(len(data))

        for phase in data["phase"].unique():

            mask = data["phase"] == phase

            if phase not in baseline:
                raise ValueError(
                    f"Unsupported phase '{phase}' in telemetry data."
                )

            mean = baseline[phase][sensor]["mean"]
            std = baseline[phase][sensor]["std"]

            z_scores[mask] = (
                (
                    data.loc[mask, sensor]
                    - mean
                )
                / (std + 1e-9)
            ).abs()

        sensor_z_scores[sensor] = z_scores

        result[f"{sensor}_z"] = z_scores

        # Persistent sensor alert.
        raw_alert = z_scores >= 4.0

        persistent = (
            pd.Series(
                raw_alert.astype(int)
            )
            .rolling(10)
            .sum()
            .fillna(0)
            .ge(10)
            .to_numpy()
        )

        result[f"{sensor}_alert"] = persistent

        sensor_alerts.append(persistent)

    alert_matrix = np.column_stack(
        sensor_alerts
    )

    active_sensors = alert_matrix.sum(
        axis=1
    )

    result["active_sensors"] = (
        active_sensors
    )

    result["status"] = np.select(
        [
            active_sensors >= 2,
            active_sensors >= 1,
        ],
        [
            "CRITICAL",
            "WARNING",
        ],
        default="NORMAL",
    )

    result["ai_anomaly"] = (
        result["status"] != "NORMAL"
    ).astype(int)

    # --------------------------------------------------------
    # Advanced anomaly score
    # --------------------------------------------------------

    z_matrix = np.column_stack(
        [
            sensor_z_scores[sensor]
            for sensor in SENSORS
        ]
    )

    max_z = z_matrix.max(axis=1)
    anomaly_score = np.where(
        result["status"] == "CRITICAL",
        np.clip(
            (max_z / 8.0) * 100.0,
            0.0,
            100.0,
        ),
        np.where(
            result["status"] == "WARNING",
            np.clip(
                (max_z / 8.0) * 70.0,
                0.0,
                100.0,
            ),
            np.clip(
                (max_z / 8.0) * 20.0,
                0.0,
                100.0,
            ),
        ),
    )

    result["anomaly_score"] = anomaly_score

    # --------------------------------------------------------
    # Confidence score
    # --------------------------------------------------------

    confidence = np.where(
        result["ai_anomaly"] == 1,
        np.clip(
            60.0
            + (active_sensors * 10.0)
            + (max_z * 4.0),
            0.0,
            99.0,
        ),
        np.clip(
            100.0 - (max_z * 4.0),
            1.0,
            95.0,
        ),
    )

    result["confidence"] = confidence

    # --------------------------------------------------------
    # Primary risk sensor
    # --------------------------------------------------------

    primary_sensor_index = (
        z_matrix.argmax(axis=1)
    )

    primary_sensors = [
        SENSORS[index]
        for index in primary_sensor_index
    ]

    result["primary_risk_sensor"] = (
        primary_sensors
    )

    return result


def metrics(data):

    actual = data["anomaly"]
    predicted = data["ai_anomaly"]

    tp = (
        (actual == 1)
        & (predicted == 1)
    ).sum()

    fp = (
        (actual == 0)
        & (predicted == 1)
    ).sum()

    fn = (
        (actual == 1)
        & (predicted == 0)
    ).sum()

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0
    )

    false_positive_rate = (
        fp / (actual == 0).sum()
        if (actual == 0).sum()
        else 0
    )

    # Detection delay.
    anomaly_times = data.loc[
        actual == 1,
        "time_s",
    ]

    detection_times = data.loc[
        predicted == 1,
        "time_s",
    ]

    delay = None

    if not anomaly_times.empty:

        start = anomaly_times.iloc[0]

        after = detection_times[
            detection_times >= start
        ]

        if not after.empty:
            delay = (
                after.iloc[0] - start
            )

    return {
        "samples": len(data),
        "actual_anomalies": int(
            actual.sum()
        ),
        "detected_anomalies": int(
            predicted.sum()
        ),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate":
            false_positive_rate,
        "detection_delay_s":
            delay,
    }


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nRocket Guardian AI â€” V11"
    )
    print(
        "========================="
    )

    training = load_training()

    print(
        f"Training samples: "
        f"{len(training)}"
    )

    baseline = build_phase_baseline(
        training
    )

    results = []

    files = sorted(
        list(
            DATA_DIR.glob(
                "validation_normal_*.csv"
            )
        )
        + list(
            DATA_DIR.glob(
                "test_*.csv"
            )
        )
    )

    for file in files:

        data = pd.read_csv(file)

        analyzed = detect(
            data,
            baseline
        )

        result = metrics(
            analyzed
        )

        result["anomaly_score"] = float(
            analyzed["anomaly_score"].max()
        )

        result["confidence"] = float(
            analyzed["confidence"].max()
        )

        primary_sensor_counts = (
            analyzed.loc[
                analyzed["ai_anomaly"] == 1,
                "primary_risk_sensor"
            ]
            .value_counts()
        )

        if not primary_sensor_counts.empty:
            result["primary_risk_sensor"] = (
                primary_sensor_counts.index[0]
            )
        else:
            result["primary_risk_sensor"] = "None"

        result["dataset"] = file.stem
        results.append(result)

        analyzed.to_csv(
            OUTPUT_DIR
            / f"{file.stem}_results.csv",
            index=False,
        )

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df[
        [
            "dataset",
            "samples",
            "actual_anomalies",
            "detected_anomalies",
            "anomaly_score",
            "confidence",
            "primary_risk_sensor",
            "precision",
            "recall",
            "f1_score",
            "false_positive_rate",
            "detection_delay_s",
        ]
    ]

    print("\nResults:")
    print(
        results_df.to_string(
            index=False,
            formatters={
                "anomaly_score":
                    "{:.1f}".format,
                "confidence":
                    "{:.1f}".format,
                "precision":
                    "{:.3f}".format,
                "recall":
                    "{:.3f}".format,
                "f1_score":
                    "{:.3f}".format,
                "false_positive_rate":
                    "{:.3f}".format,
                "detection_delay_s":
                    lambda x:
                    "N/A"
                    if pd.isna(x)
                    else f"{x:.1f}",
            },
        )
    )

    output = (
        OUTPUT_DIR / "v11_results.csv"
    )

    results_df.to_csv(
        output,
        index=False,
    )

    print(
        f"\nResults saved to: {output}"
    )


if __name__ == "__main__":
    main()




