import pandas as pd
from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score


DATA_DIR = Path("data/raw")

FEATURES = [
    "pressure_kpa",
    "temperature_k",
    "vibration_g",
    "thrust_n",
]


def train_model():
    base_data = pd.read_csv(DATA_DIR / "rocket_telemetry.csv")

    normal_data = base_data[base_data["anomaly"] == 0]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(normal_data[FEATURES])

    model = IsolationForest(
        n_estimators=300,
        contamination=0.05,
        random_state=42,
    )

    model.fit(X_train)

    return model, scaler


def evaluate_scenario(model, scaler, file_path):
    data = pd.read_csv(file_path)

    X = scaler.transform(data[FEATURES])

    predictions = model.predict(X)

    predicted = (predictions == -1).astype(int)
    actual = data["anomaly"]

    return {
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
    }


def main():
    model, scaler = train_model()

    files = sorted(
        DATA_DIR.glob("*_telemetry.csv")
    )

    results = []

    for file_path in files:
        if file_path.name == "rocket_telemetry.csv":
            continue

        result = evaluate_scenario(
            model,
            scaler,
            file_path,
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    print("\nRocket Guardian AI — Scenario Evaluation")
    print("========================================")

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

    output_path = Path(
        "data/processed/scenario_results.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nResults saved to: {output_path}"
    )


if __name__ == "__main__":
    main()