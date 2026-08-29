import pandas as pd
import numpy as np
from pathlib import Path


DATA_DIR = Path("data/v5")
OUTPUT_DIR = Path("data/processed/v9")

SENSORS = [
    "pressure_kpa",
    "temperature_k",
    "vibration_g",
    "thrust_n",
]


def calculate_baseline(training):
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


def evaluate_configuration(
    training,
    test_files,
    baseline,
    z_threshold,
    persistence,
):
    all_results = []

    for file_path in test_files:

        data = pd.read_csv(file_path)

        alert_matrix = []

        for sensor in SENSORS:

            mean = baseline[sensor]["mean"]
            std = baseline[sensor]["std"]

            z = (
                (data[sensor] - mean)
                / (std + 1e-9)
            ).abs()

            raw_alert = z >= z_threshold

            persistent = (
                pd.Series(
                    raw_alert.astype(int)
                )
                .rolling(
                    persistence
                )
                .sum()
                .fillna(0)
                .ge(persistence)
                .to_numpy()
            )

            alert_matrix.append(persistent)

        alerts = np.column_stack(
            alert_matrix
        )

        active_sensors = alerts.sum(
            axis=1
        )

        # One persistent sensor = warning.
        # Two or more = critical.
        predicted = (
            active_sensors >= 1
        ).astype(int)

        actual = data["anomaly"].to_numpy()

        tp = np.sum(
            (actual == 1)
            & (predicted == 1)
        )

        fp = np.sum(
            (actual == 0)
            & (predicted == 1)
        )

        fn = np.sum(
            (actual == 1)
            & (predicted == 0)
        )

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

        normal = actual == 0

        false_alarms = np.sum(
            predicted[normal] == 1
        )

        normal_minutes = (
            np.sum(normal)
            / 10
            / 60
        )

        false_alarm_rate = (
            false_alarms / normal_minutes
            if normal_minutes > 0
            else 0
        )

        all_results.append(
            {
                "scenario": file_path.stem.replace(
                    "test_", ""
                ),
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "false_alarms_per_minute":
                    false_alarm_rate,
            }
        )

    result_df = pd.DataFrame(
        all_results
    )

    return {
        "z_threshold": z_threshold,
        "persistence": persistence,
        "mean_precision":
            result_df["precision"].mean(),
        "mean_recall":
            result_df["recall"].mean(),
        "mean_f1":
            result_df["f1_score"].mean(),
        "mean_false_alarms_per_minute":
            result_df[
                "false_alarms_per_minute"
            ].mean(),
    }


def calculate_objective(row):
    """
    Reward recall and precision while heavily
    penalizing false alarms.
    """

    return (
        row["mean_f1"]
        - 0.02
        * row[
            "mean_false_alarms_per_minute"
        ]
    )


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

    z_thresholds = [
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
        5.5,
        6.0,
    ]

    persistence_values = [
        3,
        5,
        8,
        10,
        15,
        20,
        30,
    ]

    print(
        "\nRocket Guardian AI — V9"
    )
    print(
        "========================"
    )

    print(
        "Testing configurations..."
    )

    for z in z_thresholds:

        for persistence in persistence_values:

            result = evaluate_configuration(
                training,
                test_files,
                baseline,
                z,
                persistence,
            )

            results.append(result)

    results_df = pd.DataFrame(
        results
    )

    results_df["objective"] = (
        results_df.apply(
            calculate_objective,
            axis=1,
        )
    )

    results_df = results_df.sort_values(
        "objective",
        ascending=False,
    )

    output_file = (
        OUTPUT_DIR
        / "threshold_search_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False,
    )

    print("\nTop 10 configurations:")
    print(
        results_df.head(10).to_string(
            index=False,
            formatters={
                "mean_precision":
                    "{:.3f}".format,
                "mean_recall":
                    "{:.3f}".format,
                "mean_f1":
                    "{:.3f}".format,
                "mean_false_alarms_per_minute":
                    "{:.2f}".format,
                "objective":
                    "{:.3f}".format,
            },
        )
    )

    best = results_df.iloc[0]

    print(
        "\nBEST CONFIGURATION"
    )
    print(
        "-------------------"
    )
    print(
        f"Z threshold : "
        f"{best['z_threshold']}"
    )
    print(
        f"Persistence : "
        f"{best['persistence']} samples"
    )
    print(
        f"Mean precision : "
        f"{best['mean_precision']:.3f}"
    )
    print(
        f"Mean recall : "
        f"{best['mean_recall']:.3f}"
    )
    print(
        f"Mean F1 : "
        f"{best['mean_f1']:.3f}"
    )
    print(
        f"False alarms/min : "
        f"{best['mean_false_alarms_per_minute']:.2f}"
    )

    print(
        f"\nAll results saved to: "
        f"{output_file}"
    )


if __name__ == "__main__":
    main()