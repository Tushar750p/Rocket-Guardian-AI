import numpy as np
import pandas as pd
from pathlib import Path


OUTPUT_DIR = Path("data/v10")

SENSORS = [
    "pressure_kpa",
    "temperature_k",
    "vibration_g",
    "thrust_n",
]


def generate_run(
    duration=120,
    sample_rate=10,
    seed=42,
):
    rng = np.random.default_rng(seed)

    n = duration * sample_rate
    time = np.arange(n) / sample_rate

    # Slight run-to-run variation.
    pressure_scale = rng.normal(1.0, 0.015)
    temperature_scale = rng.normal(1.0, 0.01)
    vibration_scale = rng.normal(1.0, 0.03)
    thrust_scale = rng.normal(1.0, 0.015)

    ramp = np.clip(time / 15.0, 0, 1)

    pressure = (
        40
        + 80 * ramp
    ) * pressure_scale

    temperature = (
        300
        + 450 * ramp
    ) * temperature_scale

    vibration = (
        0.15
        + 0.05 * ramp
    ) * vibration_scale

    thrust = (
        500 * ramp
    ) * thrust_scale

    # Small correlated noise.
    pressure += rng.normal(0, 1.2, n)
    temperature += rng.normal(0, 3.0, n)
    vibration += rng.normal(0, 0.012, n)
    thrust += rng.normal(0, 6.0, n)

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
        35 * progress ** 2
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
        220 * progress ** 2
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
        0.8 * progress ** 2
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
        220 * progress ** 2
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
        30 * progress ** 2
    )

    data.loc[mask, "temperature_k"] += (
        180 * progress ** 2
    )

    data.loc[mask, "vibration_g"] += (
        0.6 * progress ** 2
    )

    data.loc[mask, "thrust_n"] -= (
        180 * progress ** 2
    )

    data.loc[mask, "anomaly"] = 1
    data.loc[mask, "scenario"] = "combined"

    return data


def save_run(name, data):
    path = OUTPUT_DIR / f"{name}.csv"

    data.to_csv(
        path,
        index=False,
    )

    print(
        f"{name:25s}"
        f"samples={len(data):4d} "
        f"anomalies={data['anomaly'].sum():4d}"
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Rocket Guardian AI — V10 Dataset"
    )
    print(
        "================================"
    )

    # Multiple independent normal training runs.
    for i in range(1, 6):

        data = generate_run(
            seed=1000 + i
        )

        save_run(
            f"training_normal_{i}",
            data,
        )

    # Unseen normal validation runs.
    for i in range(1, 3):

        data = generate_run(
            seed=2000 + i
        )

        save_run(
            f"validation_normal_{i}",
            data,
        )

    # Unseen failure scenarios.
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

        data = generate_run(
            seed=3000 + i
        )

        data = injector(data)

        save_run(
            name,
            data,
        )

    print(
        "\nV10 dataset generation complete."
    )


if __name__ == "__main__":
    main()