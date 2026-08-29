import pandas as pd
import numpy as np
from pathlib import Path


DATA_DIR = Path("data/v5")
OUTPUT_DIR = Path("data/processed/v7")

SENSORS = [
    "pressure_kpa",
    "temperature_k",
    "vibration_g",
    "thrust_n",
]


def calculate_baseline(training):
    baseline = {}

    for sensor in SENSORS:
        baseline[sensor] = {
            "mean": training[sensor].mean(),
            "std": training[sensor].std(),
        }

    return baseline


def calculate_sensor_scores(data, baseline):
    scores = pd.DataFrame(index=data.index)

    for sensor in SENSORS:
        mean = baseline[sensor]["mean"]
        std = baseline[sensor]["std"]

        z_score = (
            (data[sensor] - mean)
            / (std + 1e-6)
        ).abs()

        rate = (
            data[sensor]
            .diff(10)
            .abs()
            / (std + 1e-6)
        )

        scores[sensor] = (
            0.7 * (z_score / 4.0)
            + 0.3 * (rate / 4.0)
        )

    return scores.clip(
        lower=0,
        upper=10,
    )


def consecutive_true(values, window):
    """
    Return True only when the condition has
    remained true for the required number of samples.
    """

    series = pd.Series(values)

    return (
        series
        .rolling(window=window)
        .sum()
        .fillna(0)
        .eq(window)
        .to_numpy()
    )


def detect(data, baseline):
    scores = calculate_sensor_scores(
        data,
        baseline,
    )

    sensor_alerts = (
        scores[SENSORS] >= 1.0
    )

    sensor_count = sensor_alerts.sum(axis=1)

    # Raw system alert.
    raw_warning = (
        scores[SENSORS].max(axis=1) >= 1.0
    )

    raw_critical = (
        (sensor_count >= 2)
        & (scores[SENSORS].max(axis=1) >= 2.0)
    )

    # Require persistence.
    warning_confirmed = consecutive_true(
        raw_warning,
        window=5,
    )

    critical_confirmed = consecutive_true(
        raw_critical,
        window=5,
    )

    status = np.select(
        [
            critical_confirmed,
            warning_confirmed,
        ],
        [
            "CRITICAL",
            "WARNING",
        ],
        default="NORMAL",
    )

    result = data.copy()

    for sensor in SENSORS:
        result[f"{sensor}_score"] = scores[
            sensor
        ]

    result["active_sensors"] = sensor_count
    result["status"] = status
    result["ai_anomaly"] = (
        status != "NORMAL"
    ).astype(int)

    return result


def evaluate(data):
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

    # False alarms before actual anomaly starts.
    normal_period = data[
        data["anomaly"] == 0
    ]

    false_alarms = (
        normal_period["ai_anomaly"] == 1
    ).sum()

    normal_duration_minutes = (
        len(normal_period) / 10 / 60
    )

    false_alarms_per_minute = (
        false_alarms
        / normal_duration_minutes
        if normal_duration_minutes > 0
        else 0
    )

    # Detection delay.
    anomaly_times = data.loc[
        data["anomaly"] == 1,
        "time_s",
    ]

    detected_times = data.loc[
        data["ai_anomaly"] == 1,
        "time_s",
    ]

    detection_delay = None

    if not anomaly_times.empty:
        anomaly_start = anomaly_times.iloc[0]

        after_start = detected_times[
            detected_times >= anomaly_start
        ]

        if not after_start.empty:
            detection_delay = (
                after_start.iloc[0]
                - anomaly_start
            )

    return {
        "actual_anomalies": int(
            actual.sum()
        ),
        "detected_anomalies": int(
            predicted.sum()
        ),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_alarms_per_minute":
            false_alarms_per_minute,
        "detection_delay_s":
            detection_delay,
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    training = pd.read_csv(
        DATA_DIR / "training_normal.csv"
    )

    baseline = calculate_baseline(
        training
    )

    test_files = sorted(
        DATA_DIR.glob("test_*.csv")
    )

    results = []

    for file_path in test_files:

        data = pd.read_csv(
            file_path
        )

        analyzed = detect(
            data,
            baseline,
        )

        metrics = evaluate(
            analyzed
        )

        metrics["scenario"] = (
            file_path.stem
            .replace("test_", "")
        )

        results.append(metrics)

        analyzed.to_csv(
            OUTPUT_DIR
            / f"{file_path.stem}_v7.csv",
            index=False,
        )

    results_df = pd.DataFrame(
        results
    )

    columns = [
        "scenario",
        "actual_anomalies",
        "detected_anomalies",
        "precision",
        "recall",
        "f1_score",
        "false_alarms_per_minute",
        "detection_delay_s",
    ]

    results_df = results_df[columns]

    print(
        "\nRocket Guardian AI — V7"
    )
    print(
        "========================"
    )

    print(
        results_df.to_string(
            index=False,
            formatters={
                "precision": "{:.2f}".format,
                "recall": "{:.2f}".format,
                "f1_score": "{:.2f}".format,
                "false_alarms_per_minute":
                    "{:.2f}".format,
                "detection_delay_s":
                    lambda x:
                    "N/A"
                    if pd.isna(x)
                    else f"{x:.1f}",
            },
        )
    )

    results_file = (
        OUTPUT_DIR / "v7_results.csv"
    )

    results_df.to_csv(
        results_file,
        index=False,
    )

    print(
        f"\nResults saved to: "
        f"{results_file}"
    )


if __name__ == "__main__":
    main()