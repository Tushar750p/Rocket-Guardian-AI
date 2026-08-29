import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score


DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

FEATURES = [
    "pressure_kpa",
    "temperature_k",
    "vibration_g",
    "thrust_n",
]


def train_baseline():
    """
    Learn normal rocket behavior from normal samples only.
    """

    baseline_file = DATA_DIR / "rocket_telemetry.csv"

    data = pd.read_csv(baseline_file)

    normal_data = data[data["anomaly"] == 0]

    scaler = StandardScaler()

    X_normal = scaler.fit_transform(
        normal_data[FEATURES]
    )

    model = IsolationForest(
        n_estimators=300,
        contamination=0.05,
        random_state=42,
    )

    model.fit(X_normal)

    return model, scaler


def calculate_sensor_deviation(data, normal_data):
    """
    Calculate standardized deviation for each sensor.
    """

    means = normal_data[FEATURES].mean()
    stds = normal_data[FEATURES].std()

    deviations = pd.DataFrame(index=data.index)

    for feature in FEATURES:
        deviations[feature] = (
            (data[feature] - means[feature])
            / stds[feature]
        ).abs()

    return deviations


def calculate_risk(deviations):
    """
    Convert sensor deviations into a simple risk score.
    """

    sensor_flags = deviations >= 3.0

    number_of_flags = sensor_flags.sum(axis=1)

    risk = np.select(
        [
            number_of_flags >= 3,
            number_of_flags >= 2,
            number_of_flags >= 1,
        ],
        [
            3,
            2,
            1,
        ],
        default=0,
    )

    return risk


def evaluate_scenario(
    model,
    scaler,
    scenario_file,
    normal_data,
):
    data = pd.read_csv(scenario_file)

    # ML prediction
    X = scaler.transform(data[FEATURES])

    ml_prediction = model.predict(X)

    data["ml_anomaly"] = (
        ml_prediction == -1
    ).astype(int)

    # Sensor-level deviation
    deviations = calculate_sensor_deviation(
        data,
        normal_data,
    )

    for feature in FEATURES:
        data[f"{feature}_deviation"] = deviations[
            feature
        ]

    # Hybrid risk
    data["risk_score"] = calculate_risk(
        deviations
    )

    # Final AI decision
    data["ai_anomaly"] = (
        (data["ml_anomaly"] == 1)
        | (data["risk_score"] >= 2)
    ).astype(int)

    # Status
    data["status"] = np.select(
        [
            data["risk_score"] >= 3,
            data["risk_score"] >= 1,
        ],
        [
            "CRITICAL",
            "WARNING",
        ],
        default="NORMAL",
    )

    actual = data["anomaly"]
    predicted = data["ai_anomaly"]

    result = {
        "scenario": data["scenario"].iloc[0],
        "samples": len(data),
        "actual_anomalies": int(actual.sum()),
        "detected_anomalies": int(predicted.sum()),
        "precision": precision_score(
            actual,
            predicted,
            zero_division=0,
        ),
        "recall": recall_score(
            actual,
            predicted,
            zero_division=0,
        ),
        "f1_score": f1_score(
            actual,
            predicted,
            zero_division=0,
        ),
        "critical_alerts": int(
            (data["status"] == "CRITICAL").sum()
        ),
    }

    output_file = (
        OUTPUT_DIR
        / f"{data['scenario'].iloc[0]}_v3_results.csv"
    )

    data.to_csv(
        output_file,
        index=False,
    )

    return result


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model, scaler = train_baseline()

    baseline_data = pd.read_csv(
        DATA_DIR / "rocket_telemetry.csv"
    )

    normal_data = baseline_data[
        baseline_data["anomaly"] == 0
    ]

    scenario_files = sorted(
        DATA_DIR.glob("*_telemetry.csv")
    )

    results = []

    for file in scenario_files:

        if file.name == "rocket_telemetry.csv":
            continue

        result = evaluate_scenario(
            model,
            scaler,
            file,
            normal_data,
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    print("\nRocket Guardian AI — V3")
    print("========================")

    print(
        results_df.to_string(
            index=False,
            formatters={
                "precision": "{:.2f}".format,
                "recall": "{:.2f}".format,
                "f1_score": "{:.2f}".format,
            },
        )
    )

    results_file = (
        OUTPUT_DIR / "v3_scenario_results.csv"
    )

    results_df.to_csv(
        results_file,
        index=False,
    )

    print(
        f"\nResults saved to: {results_file}"
    )


if __name__ == "__main__":
    main()