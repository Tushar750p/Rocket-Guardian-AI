import numpy as np
import pandas as pd
from pathlib import Path


def generate_telemetry(
    duration_seconds=60,
    sample_rate=10,
    anomaly_start=40,
    seed=42,
):
    """
    Generate simulated rocket-engine telemetry.

    This is a research/simulation dataset, not real flight data.
    """

    rng = np.random.default_rng(seed)

    total_samples = duration_seconds * sample_rate
    time = np.arange(total_samples) / sample_rate

    # Engine ramp-up and steady-state behavior
    ramp = np.clip(time / 10.0, 0, 1)

    pressure = (
        40
        + 80 * ramp
        + rng.normal(0, 1.5, total_samples)
    )

    temperature = (
        300
        + 450 * ramp
        + rng.normal(0, 4, total_samples)
    )

    vibration = (
        0.15
        + 0.05 * ramp
        + rng.normal(0, 0.015, total_samples)
    )

    thrust = (
        500 * ramp
        + rng.normal(0, 8, total_samples)
    )

    # Inject an artificial anomaly after anomaly_start.
    anomaly_mask = time >= anomaly_start

    pressure[anomaly_mask] += np.linspace(
        0, 25, anomaly_mask.sum()
    )

    temperature[anomaly_mask] += np.linspace(
        0, 120, anomaly_mask.sum()
    )

    vibration[anomaly_mask] += np.linspace(
        0, 0.45, anomaly_mask.sum()
    )

    thrust[anomaly_mask] -= np.linspace(
        0, 100, anomaly_mask.sum()
    )

    data = pd.DataFrame(
        {
            "time_s": time,
            "pressure_kpa": pressure,
            "temperature_k": temperature,
            "vibration_g": vibration,
            "thrust_n": thrust,
            "anomaly": anomaly_mask.astype(int),
        }
    )

    return data


def main():
    output_path = Path("data/raw/rocket_telemetry.csv")

    data = generate_telemetry()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(output_path, index=False)

    print(f"Generated {len(data)} telemetry samples.")
    print(f"Saved to: {output_path}")
    print()
    print(data.head())
    print()
    print("Anomaly samples:", data["anomaly"].sum())


if __name__ == "__main__":
    main()