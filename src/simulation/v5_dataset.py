import numpy as np
import pandas as pd
from pathlib import Path


OUTPUT_DIR = Path("data/v5")


def generate_normal(duration=120, sample_rate=10, seed=42):
    rng = np.random.default_rng(seed)

    n = duration * sample_rate
    time = np.arange(n) / sample_rate

    ramp = np.clip(time / 15.0, 0, 1)

    pressure = (
        40
        + 80 * ramp
        + rng.normal(0, 1.2, n)
    )

    temperature = (
        300
        + 450 * ramp
        + rng.normal(0, 3.0, n)
    )

    vibration = (
        0.15
        + 0.05 * ramp
        + rng.normal(0, 0.012, n)
    )

    thrust = (
        500 * ramp
        + rng.normal(0, 6.0, n)
    )

    return pd.DataFrame({
        "time_s": time,
        "pressure_kpa": pressure,
        "temperature_k": temperature,
        "vibration_g": vibration,
        "thrust_n": thrust,
        "anomaly": 0,
        "scenario": "normal",
    })


def inject_pressure(data, start=80):
    mask = data["time_s"] >= start
    progress = np.clip(
        (data.loc[mask, "time_s"] - start) / 30,
        0,
        1,
    )

    data.loc[mask, "pressure_kpa"] += (
        35 * progress**2
    )

    data.loc[mask, "anomaly"] = 1
    data.loc[mask, "scenario"] = "pressure"

    return data


def inject_temperature(data, start=80):
    mask = data["time_s"] >= start
    progress = np.clip(
        (data.loc[mask, "time_s"] - start) / 30,
        0,
        1,
    )

    data.loc[mask, "temperature_k"] += (
        220 * progress**2
    )

    data.loc[mask, "anomaly"] = 1
    data.loc[mask, "scenario"] = "temperature"

    return data


def inject_vibration(data, start=80):
    mask = data["time_s"] >= start
    progress = np.clip(
        (data.loc[mask, "time_s"] - start) / 30,
        0,
        1,
    )

    data.loc[mask, "vibration_g"] += (
        0.8 * progress**2
    )

    data.loc[mask, "anomaly"] = 1
    data.loc[mask, "scenario"] = "vibration"

    return data


def inject_thrust(data, start=80):
    mask = data["time_s"] >= start
    progress = np.clip(
        (data.loc[mask, "time_s"] - start) / 30,
        0,
        1,
    )

    data.loc[mask, "thrust_n"] -= (
        220 * progress**2
    )

    data.loc[mask, "anomaly"] = 1
    data.loc[mask, "scenario"] = "thrust"

    return data


def inject_combined(data, start=80):
    mask = data["time_s"] >= start
    progress = np.clip(
        (data.loc[mask, "time_s"] - start) / 30,
        0,
        1,
    )

    data.loc[mask, "pressure_kpa"] += (
        30 * progress**2
    )

    data.loc[mask, "temperature_k"] += (
        180 * progress**2
    )

    data.loc[mask, "vibration_g"] += (
        0.6 * progress**2
    )

    data.loc[mask, "thrust_n"] -= (
        180 * progress**2
    )

    data.loc[mask, "anomaly"] = 1
    data.loc[mask, "scenario"] = "combined"

    return data


def save_dataset(name, injector=None, seed=42):
    data = generate_normal(seed=seed)

    if injector is not None:
        data = injector(data)

    path = OUTPUT_DIR / f"{name}.csv"

    data.to_csv(path, index=False)

    print(
        f"{name:15s} "
        f"samples={len(data):4d} "
        f"anomalies={data['anomaly'].sum():4d}"
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Generating V5 datasets")
    print("-----------------------")

    # Training data: ONLY normal behavior.
    save_dataset(
        "training_normal",
        seed=500,
    )

    # Completely separate unseen test datasets.
    scenarios = {
        "test_pressure": inject_pressure,
        "test_temperature": inject_temperature,
        "test_vibration": inject_vibration,
        "test_thrust": inject_thrust,
        "test_combined": inject_combined,
    }

    for i, (name, injector) in enumerate(
        scenarios.items()
    ):
        save_dataset(
            name,
            injector=injector,
            seed=600 + i,
        )

    print("\nV5 dataset generation complete.")


if __name__ == "__main__":
    main()