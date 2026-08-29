import pandas as pd
import numpy as np
from pathlib import Path


DATA_DIR = Path("data/v5")
OUTPUT_DIR = Path("data/processed/v8")

# Each sensor gets its own sensitivity.
CONFIG = {
    "pressure_kpa": {
        "z_threshold": 4.0,
        "rate_threshold": 2.5,
        "persistence": 8,
    },
    "temperature_k": {
        "z_threshold": 4.0,
        "rate_threshold": 2.0,
        "persistence": 8,
    },
    "vibration_g": {
        "z_threshold": 3.5,
        "rate_threshold": 2.0,
        "persistence": 5,
    },
    "thrust_n": {
        "z_threshold": 4.0,
        "rate_threshold": 2.5,
        "persistence": 8,
    },
}


def build_baseline(training):
    baseline = {}

    # Use only the stable part of normal training data.
    stable = training[
        training["time_s"] >= 30
    ].copy()

    for sensor in CONFIG:
        baseline[sensor] = {
            "mean": stable[sensor].mean(),
            "std": stable[sensor].std(),
        }

    return baseline


def persistent_condition(condition, window):
    series = pd.Series(
        condition,
        dtype=int,
    )

    return (
        series
        .rolling(window=window)
        .sum()
        .fillna(0)
        .ge(window)
        .to_numpy()
    )


def analyze(data, baseline):
    result = data.copy()

    alerts = {}

    for sensor, config in CONFIG.items():

        mean = baseline[sensor]["mean"]
        std = baseline[sensor]["std"]

        z_score = (
            (data[sensor] - mean)
            / (std + 1e-9)
        ).abs()

        rate = (
            data[sensor]
            .diff(10)
            .abs()
            / (std + 1e-9)
        )

        value_alert = (
            z_score >= config["z_threshold"]
        )

        rate_alert = (
            rate >= config["rate_threshold"]
        )

        raw_alert = (
            value_alert
            | rate_alert
        )

        confirmed = persistent_condition(
            raw_alert,
            config["persistence"],
        )

        result[f"{sensor}_z"] = z_score
        result[f"{sensor}_rate"] = rate
        result[f"{sensor}_alert"] = confirmed

        alerts[sensor] = confirmed

    alert_table = pd.DataFrame(
        alerts,
        index=data.index,
    )

    active_sensors = alert_table.sum(
        axis=1
    )

    result["active_sensors"] = active_sensors

    # Adaptive system decision.
    critical = (
        active_sensors >= 2
    )

    warning = (
        active_sensors >= 1
    )

    result["status"] = np.select(
        [
            critical,
            warning,
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

    normal_data = data[
        data["anomaly"] == 0
    ]

    false_alarms = (
        normal_data["ai_anomaly"] == 1
    ).sum()

    normal_minutes = (
        len(normal_data)
        / 10
        / 60
    )

    false_alarms_per_minute = (
        false_alarms / normal_minutes
        if normal_minutes > 0
        else 0
    )

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
        start = anomaly_times.iloc[0]

        valid = detected_times[
            detected_times >= start
        ]

        if not valid.empty:
            detection_delay = (
                valid.iloc[0] - start
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

    baseline = build_baseline(
        training
    )

    test_files = sorted(
        DATA_DIR.glob("test_*.csv")
    )

    results = []

    for file_path in test_files:

        scenario = (
            file_path.stem
            .replace("test_", "")
        )

        data = pd.read_csv(
            file_path
        )

        analyzed = analyze(
            data,
            baseline,
        )

        metrics = calculate_metrics(
            analyzed
        )

        metrics["scenario"] = scenario

        results.append(metrics)

        analyzed.to_csv(
            OUTPUT_DIR
            / f"{scenario}_v8.csv",
            index=False,
        )

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df[
        [
            "scenario",
            "actual_anomalies",
            "detected_anomalies",
            "precision",
            "recall",
            "f1_score",
            "false_alarms_per_minute",
            "detection_delay_s",
        ]
    ]

    print(
        "\nRocket Guardian AI — V8"
    )
    print(
        "========================"
    )

    print(
        results_df.to_string(
            index=False,
            formatters={
                "precision":
                    "{:.2f}".format,
                "recall":
                    "{:.2f}".format,
                "f1_score":
                    "{:.2f}".format,
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

    results_df.to_csv(
        OUTPUT_DIR / "v8_results.csv",
        index=False,
    )

    print(
        "\nResults saved to:"
        " data/processed/v8/v8_results.csv"
    )


if __name__ == "__main__":
    main()