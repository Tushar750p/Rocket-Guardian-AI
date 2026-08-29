import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score


DATA_DIR = Path("data/v5")
OUTPUT_DIR = Path("data/processed/v5")

SENSORS = [
    "pressure_kpa",
    "temperature_k",
    "vibration_g",
    "thrust_n",
]


def train_detector():
    training = pd.read_csv(
        DATA_DIR / "training_normal.csv"
    )

    X_train = training[SENSORS]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = IsolationForest(
        n_estimators=500,
        contamination=0.05,
        random_state=42,
    )

    model.fit(X_scaled)

    return model, scaler


def evaluate_file(model, scaler, file_path):
    data = pd.read_csv(file_path)

    X = data[SENSORS]
    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)

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
            data["ai_anomaly"] == 1
        ],
        [
            "WARNING"
        ],
        default="NORMAL",
    )

    output_file = (
        OUTPUT_DIR
        / f"{file_path.stem}_results.csv"
    )

    data.to_csv(
        output_file,
        index=False,
    )

    return {
       "scenario": file_path.stem.replace("test_", ""),
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

    print(
        "Training on NORMAL data only..."
    )

    model, scaler = train_detector()

    test_files = sorted(
        DATA_DIR.glob("test_*.csv")
    )

    results = []

    for file_path in test_files:
        result = evaluate_file(
            model,
            scaler,
            file_path,
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    print(
        "\nRocket Guardian AI — V5"
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
        OUTPUT_DIR / "v5_results.csv"
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