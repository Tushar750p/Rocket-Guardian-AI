import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


DATA_PATH = "data/raw/rocket_telemetry.csv"
OUTPUT_PATH = "data/processed/detected_anomalies_v2.csv"

FEATURES = [
    "pressure_kpa",
    "temperature_k",
    "vibration_g",
    "thrust_n",
]


def main():
    data = pd.read_csv(DATA_PATH)

    # Separate known normal data for training.
    # The anomaly column is used ONLY to create the training set.
    normal_data = data[data["anomaly"] == 0]

    X_train = normal_data[FEATURES]
    X_test = data[FEATURES]

    # Scale sensor values.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train only on normal behavior.
    model = IsolationForest(
        n_estimators=300,
        contamination=0.05,
        random_state=42,
    )

    model.fit(X_train_scaled)

    # Predict all telemetry.
    predictions = model.predict(X_test_scaled)

    data["ai_prediction"] = predictions
    data["ai_anomaly"] = (predictions == -1).astype(int)

    # Save predictions.
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(output_path, index=False)

    # Evaluate against simulated ground truth.
    actual = data["anomaly"]
    predicted = data["ai_anomaly"]

    print("\nRocket Guardian AI — Version 2")
    print("--------------------------------")

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")
    print(f"AI anomalies:     {predicted.sum()}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(actual, predicted))

    print("\nClassification Report:")
    print(
        classification_report(
            actual,
            predicted,
            target_names=["Normal", "Anomaly"],
            zero_division=0,
        )
    )

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()