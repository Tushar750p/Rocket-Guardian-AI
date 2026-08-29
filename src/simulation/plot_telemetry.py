import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = "data/raw/rocket_telemetry.csv"


def main():
    data = pd.read_csv(DATA_PATH)

    sensors = [
        ("pressure_kpa", "Pressure (kPa)"),
        ("temperature_k", "Temperature (K)"),
        ("vibration_g", "Vibration (g)"),
        ("thrust_n", "Thrust (N)"),
    ]

    for column, label in sensors:
        plt.figure(figsize=(12, 5))

        plt.plot(
            data["time_s"],
            data[column],
            label=label,
        )

        anomaly_data = data[data["anomaly"] == 1]

        if not anomaly_data.empty:
            plt.axvline(
                anomaly_data["time_s"].iloc[0],
                linestyle="--",
                label="Anomaly starts",
            )

        plt.title(f"Rocket Telemetry — {label}")
        plt.xlabel("Time (seconds)")
        plt.ylabel(label)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.show()


if __name__ == "__main__":
    main()