import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score


DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

SENSORS = [
    "pressure_kpa",
    "temperature_k",
    "vibration_g",
    "thrust_n",
]


def build_features(data):
    """
    Build time-series features from rocket telemetry.
    """

    features = pd.DataFrame(index=data.index)

    for sensor in SENSORS:
        series = data[sensor]

        # Rolling statistics
        rolling_mean = series.rolling(
            window=20,
            min_periods=1,
        ).mean()

        rolling_std = series.rolling(
            window=20,
            min_periods=1,
        ).std().fillna(0)

        # Rate of change
        rate = series.diff().fillna(0)

        # Distance from rolling baseline
        deviation = (
            series - rolling_mean
        ) / (rolling_std + 1e-6)

        features[f"{sensor}_value"] = series
        features[f"{sensor}_rate"] = rate
        features[f"{sensor}_deviation"] = deviation

    return features.replace(
        [np.inf, -np.inf],
        0,
    ).fillna(0)


def train_model():
    """
    Train only on normal baseline telemetry.
    """

    baseline = pd.read_csv(
        DATA_DIR / "rocket_telemetry.csv"
    )

    normal = baseline[
        baseline["anomaly"] == 0
    ].copy()

    X_normal = build_features(normal)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_normal
    )

    model = IsolationForest(
        n_estimators=500,
        contamination=0.05,
        random_state=42,
    )

    model.fit(X_scaled)

    return model, scaler


def evaluate_scenario(
    model,
    scaler,
    file_path,
):
    data = pd.read_csv(file_path)

    X = build_features(data)

    X_scaled = scaler.transform(X)

    predictions = model.predict(
        X_scaled
    )

    predicted = (
        predictions == -1
    ).astype(int)

    actual = data["anomaly"]

    precision = precision_score(
        actual,
        predicted,
        zero_division=0,
    )

    recall = recall_score(
        actual,
        predicted,
        zero_division=0,
    )

    f1 = f1_score(
        actual,
        predicted,
        zero_division=0,
    )

    data["ai_anomaly"] = predicted

    data["status"] = np.select(
        [
            data["ai_anomaly"] == 1,
        ],
        [
            "WARNING",
        ],
        default="NORMAL",
    )

    output_file = (
        OUTPUT_DIR
        / f"{data['scenario'].iloc[0]}_v4_results.csv"
    )

    data.to_csv(
        output_file,
        index=False,
    )

    return {
        "scenario": data["scenario"].iloc[0],
        "actual_anomalies": int(
            actual.sum()
        ),
        "detected_anomalies": int(
            predicted.sum()
        ),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model, scaler = train_model()

    scenario_files = sorted(
        DATA_DIR.glob("*_telemetry.csv")
    )

    results = []

    for file_path in scenario_files:

        if file_path.name == "rocket_telemetry.csv":
            continue

        result = evaluate_scenario(
            model,
            scaler,
            file_path,
        )

        results.append(result)

    results_df = pd.DataFrame(
        results
    )

    print(
        "\nRocket Guardian AI — V4"
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
            },
        )
    )

    results_file = (
        OUTPUT_DIR
        / "v4_scenario_results.csv"
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