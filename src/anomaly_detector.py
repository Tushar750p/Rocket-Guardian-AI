import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


DATA_PATH = "data/raw/rocket_telemetry.csv"
OUTPUT_PATH = "data/processed/detected_anomalies.csv"

FEATURES = [
    "pressure_kpa",
    "temperature_k",
    "vibration_g",
    "thrust_n",
]


def main():
    # Load telemetry
    data = pd.read_csv(DATA_PATH)

    # Select sensor data only
    X = data[FEATURES].copy()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train unsupervised anomaly detector
    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
    )

    model.fit(X_scaled)

    # Model output:
    # 1  = normal
    # -1 = anomaly
    predictions = model.predict(X_scaled)

    data["ai_prediction"] = predictions
    data["ai_anomaly"] = (predictions == -1).astype(int)

    # Save results
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(output_path, index=False)

    detected = data["ai_anomaly"].sum()

    print("Rocket Guardian AI")
    print("------------------")
    print(f"Total samples: {len(data)}")
    print(f"AI detected anomalies: {detected}")
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()