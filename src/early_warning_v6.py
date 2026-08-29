import pandas as pd
import numpy as np
from pathlib import Path


DATA_DIR = Path("data/v5")
OUTPUT_DIR = Path("data/processed/v6")

SENSORS = [
    "pressure_kpa",
    "temperature_k",
    "vibration_g",
    "thrust_n",
]


def calculate_baseline(training):
    """
    Calculate normal operating statistics.
    """

    baseline = {}

    for sensor in SENSORS:
        baseline[sensor] = {
            "mean": training[sensor].mean(),
            "std": training[sensor].std(),
        }

    return baseline


def analyze_sensor(series, mean, std):
    """
    Calculate rolling deviation and rate of change.
    """

    rolling_mean = series.rolling(
        window=30,
        min_periods=30,
    ).mean()

    rate = series.diff(10) / 1.0

    deviation = (
        (series - mean)
        / (std + 1e-6)
    )

    rolling_deviation = (
        (series - rolling_mean)
        / (rolling_mean.abs() + 1e-6)
    )

    return (
        deviation.abs(),
        rate.abs(),
        rolling_deviation.abs(),
    )


def detect_scenario(
    training,
    test,
):
    baseline = calculate_baseline(training)

    scores = pd.DataFrame(
        index=test.index
    )

    for sensor in SENSORS:

        mean = baseline[sensor]["mean"]
        std = baseline[sensor]["std"]

        deviation, rate, trend = analyze_sensor(
            test[sensor],
            mean,
            std,
        )

        # Sensor-specific normalized risk.
        value_score = deviation / 4.0

        rate_score = (
            rate
            / (std + 1e-6)
        )

        trend_score = (
            trend * 20.0
        )

        sensor_score = (
            value_score
            + rate_score
            + trend_score
        )

        scores[sensor] = sensor_score.clip(
            lower=0,
            upper=10,
        )

    # Overall system risk.
    scores["system_risk"] = scores[SENSORS].max(
        axis=1
    )

    # Number of sensors showing elevated risk.
    scores["active_sensors"] = (
        scores[SENSORS] >= 1.0
    ).sum(axis=1)

    # Decision levels.
    scores["status"] = np.select(
        [
            (
                (scores["system_risk"] >= 2.0)
                & (scores["active_sensors"] >= 2)
            ),
            (
                scores["system_risk"] >= 1.0
            ),
        ],
        [
            "CRITICAL",
            "WARNING",
        ],
        default="NORMAL",
    )

    return scores


def evaluate(
    training,
    test,
    scenario_name,
):
    scores = detect_scenario(
        training,
        test,
    )

    result = test.copy()

    result["system_risk"] = scores[
        "system_risk"
    ]

    result["active_sensors"] = scores[
        "active_sensors"
    ]

    result["status"] = scores[
        "status"
    ]

    result["ai_anomaly"] = (
        result["status"] != "NORMAL"
    ).astype(int)

    actual = result["anomaly"]
    predicted = result["ai_anomaly"]

    true_positive = (
        (actual == 1)
        & (predicted == 1)
    ).sum()

    false_positive = (
        (actual == 0)
        & (predicted == 1)
    ).sum()

    false_negative = (
        (actual == 1)
        & (predicted == 0)
    ).sum()

    precision = (
        true_positive
        / (true_positive + false_positive)
        if (true_positive + false_positive)
        else 0
    )

    recall = (
        true_positive
        / (true_positive + false_negative)
        if (true_positive + false_negative)
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall)
        else 0
    )

    # First detected anomaly after the true anomaly begins.
    actual_times = test.loc[
        test["anomaly"] == 1,
        "time_s",
    ]

    detected_times = result.loc[
        result["ai_anomaly"] == 1,
        "time_s",
    ]

    detection_delay = None
    early_warning = None

    if not actual_times.empty:
        anomaly_start = actual_times.iloc[0]

        after_start = detected_times[
            detected_times >= anomaly_start
        ]

        if not after_start.empty:
            first_detection = after_start.iloc[0]
            detection_delay = (
                first_detection
                - anomaly_start
            )

    output = {
        "scenario": scenario_name,
        "actual_anomalies": int(
            actual.sum()
        ),
        "detected_anomalies": int(
            predicted.sum()
        ),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "detection_delay_s": detection_delay,
    }

    output_file = (
        OUTPUT_DIR
        / f"{scenario_name}_v6_results.csv"
    )

    result.to_csv(
        output_file,
        index=False,
    )

    return output


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    training = pd.read_csv(
        DATA_DIR / "training_normal.csv"
    )

    test_files = sorted(
        DATA_DIR.glob("test_*.csv")
    )

    results = []

    for file_path in test_files:

        scenario_name = (
            file_path.stem
            .replace("test_", "")
        )

        test = pd.read_csv(
            file_path
        )

        result = evaluate(
            training,
            test,
            scenario_name,
        )

        results.append(result)

    results_df = pd.DataFrame(
        results
    )

    print(
        "\nRocket Guardian AI — V6"
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
                "detection_delay_s": (
                    lambda x:
                    "N/A"
                    if pd.isna(x)
                    else f"{x:.1f}"
                ),
            },
        )
    )

    results_file = (
        OUTPUT_DIR / "v6_results.csv"
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