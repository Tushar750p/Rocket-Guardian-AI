import pandas as pd
import numpy as np
from pathlib import Path


DATA_DIR = Path("data/v10")
OUTPUT_DIR = Path("data/processed/v10")

SENSORS = [
    "pressure_kpa",
    "temperature_k",
    "vibration_g",
    "thrust_n",
]


def load_training_data():
    files = sorted(
        DATA_DIR.glob("training_normal_*.csv")
    )

    frames = []

    for file in files:
        frames.append(
            pd.read_csv(file)
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def build_baseline(training):
    """
    Build baseline from multiple normal runs.
    """

    stable = training[
        training["time_s"] >= 30
    ]

    baseline = {}

    for sensor in SENSORS:
        baseline[sensor] = {
            "mean": stable[sensor].mean(),
            "std": stable[sensor].std(),
        }

    return baseline


def analyze(data, baseline):
    """
    Sensor-level adaptive anomaly detection.
    """

    result = data.copy()

    sensor_alerts = []

    for sensor in SENSORS:

        mean = baseline[sensor]["mean"]
        std = baseline[sensor]["std"]

        z_score = (
            (data[sensor] - mean)
            / (std + 1e-9)
        ).abs()

        # Change over approximately one second.
        rate = (
            data[sensor]
            .diff(10)
            .abs()
            / (std + 1e-9)
        )

        # Two independent signals:
        # absolute deviation + rate of change.
        raw_alert = (
            (z_score >= 4.0)
            | (rate >= 3.0)
        )

        # Require 1 second of persistence.
        confirmed = (
            pd.Series(
                raw_alert.astype(int)
            )
            .rolling(10)
            .sum()
            .fillna(0)
            .ge(10)
            .to_numpy()
        )

        result[f"{sensor}_z"] = z_score
        result[f"{sensor}_rate"] = rate
        result[f"{sensor}_alert"] = confirmed

        sensor_alerts.append(
            confirmed
        )

    alert_matrix = np.column_stack(
        sensor_alerts
    )

    active_sensors = alert_matrix.sum(
        axis=1
    )

    result["active_sensors"] = (
        active_sensors
    )

    # One persistent sensor = warning.
    # Two or more = critical.
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

    return result


def calculate_metrics(data):
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
        if tp + fp > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0
    )

    normal_samples = (
        actual == 0
    ).sum()

    false_positive_rate = (
        fp / normal_samples
        if normal_samples > 0
        else 0
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
    }


def evaluate_training_normal(
    baseline,
):
    """
    Check that training data itself is
    not producing excessive alerts.
    """

    training = load_training_data()

    analyzed = analyze(
        training,
        baseline,
    )

    metrics = calculate_metrics(
        analyzed
    )

    metrics["dataset"] = (
        "training_normal"
    )

    return metrics


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nRocket Guardian AI — V10"
    )
    print(
        "========================="
    )

    # 1. Load five independent normal runs.
    training = load_training_data()

    print(
        f"Training samples: "
        f"{len(training)}"
    )

    # 2. Build baseline.
    baseline = build_baseline(
        training
    )

    results = []

    # 3. Training sanity check.
    results.append(
        evaluate_training_normal(
            baseline
        )
    )

    # 4. Unseen normal validation.
    validation_files = sorted(
        DATA_DIR.glob(
            "validation_normal_*.csv"
        )
    )

    for file in validation_files:

        data = pd.read_csv(file)

        analyzed = analyze(
            data,
            baseline,
        )

        metrics = calculate_metrics(
            analyzed
        )

        metrics["dataset"] = (
            file.stem
        )

        results.append(metrics)

        analyzed.to_csv(
            OUTPUT_DIR
            / f"{file.stem}_results.csv",
            index=False,
        )

    # 5. Unseen failure scenarios.
    test_files = sorted(
        DATA_DIR.glob("test_*.csv")
    )

    for file in test_files:

        data = pd.read_csv(file)

        analyzed = analyze(
            data,
            baseline,
        )

        metrics = calculate_metrics(
            analyzed
        )

        metrics["dataset"] = (
            file.stem
        )

        results.append(metrics)

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
            "precision",
            "recall",
            "f1_score",
            "false_positive_rate",
        ]
    ]

    print("\nResults:")
    print(
        results_df.to_string(
            index=False,
            formatters={
                "precision":
                    "{:.3f}".format,
                "recall":
                    "{:.3f}".format,
                "f1_score":
                    "{:.3f}".format,
                "false_positive_rate":
                    "{:.3f}".format,
            },
        )
    )

    output_file = (
        OUTPUT_DIR
        / "v10_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False,
    )

    print(
        f"\nResults saved to: "
        f"{output_file}"
    )


if __name__ == "__main__":
    main()