import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


DATA_PATH = "data/processed/detected_anomalies.csv"


def main():
    data = pd.read_csv(DATA_PATH)

    actual = data["anomaly"]
    predicted = data["ai_anomaly"]

    print("\nRocket Guardian AI — Evaluation")
    print("--------------------------------")

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


if __name__ == "__main__":
    main()