import numpy as np
import pandas as pd
from pathlib import Path


OUTPUT_DIR = Path("data/v11")


def generate_run(
    duration=120,
    sample_rate=10,
    seed=42,
):
    rng = np.random.default_rng(seed)

    n = duration * sample_rate
    time = np.arange(n) / sample_rate

    pressure = np.zeros(n)
    temperature = np.zeros(n)
    vibration = np.zeros(n)
    thrust = np.zeros(n)

    phase = np.empty(n, dtype=object)

    for i, t in enumerate(time):

        # STARTUP
        if t < 10:
            phase[i] = "startup"

            progress = t / 10

            pressure[i] = 40 + 20 * progress
            temperature[i] = 300 + 100 * progress
            vibration[i] = 0.15 + 0.02 * progress
            thrust[i] = 100 + 200 * progress

        # RAMP
        elif t < 30:
            phase[i] = "ramp"

            progress = (t - 10) / 20

            pressure[i] = 60 + 60 * progress
            temperature[i] = 400 + 350 * progress
            vibration[i] = 0.17 + 0.03 * progress
            thrust[i] = 300 + 200 * progress

        # STEADY
        elif t < 100:
            phase[i] = "steady"

            pressure[i] = 120
            temperature[i] = 750
            vibration[i] = 0.20
            thrust[i] = 500

        # SHUTDOWN
        else:
            phase[i] = "shutdown"

            progress = (t - 100) / 20

            pressure[i] = 120 - 80 * progress
            temperature[i] = 750 - 450 * progress
            vibration[i] = 0.20 - 0.08 * progress
            thrust[i] = 500 - 400 * progress

    # Run-to-run variation
    pressure *= rng.normal(1.0, 0.01)
    temperature *= rng.normal(1.0, 0.008)
    vibration *= rng.normal(1.0, 0.02)
    thrust *= rng.normal(1.0, 0.01)

    # Sensor noise
    pressure += rng.normal(0, 1.0, n)
    temperature += rng.normal(0, 2.5, n)
    vibration += rng.normal(0, 0.01, n)
    thrust += rng.normal(0, 5.0, n)

    return pd.DataFrame({
        "time_s": time,
        "phase": phase,
        "pressure_kpa": pressure,
        "temperature_k": temperature,
        "vibration_g": vibration,
        "thrust_n": thrust,
        "anomaly": 0,
        "scenario": "normal",
    })


def inject_pressure(data, start=70):
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


def inject_temperature(data, start=70):
    mask = data["time_s"] >= start

    progress = np.clip(
        (data.loc[mask, "time_s"] - start) / 30,
        0,
        1,
    )

    data.loc[mask, "temperature_k"] += (
        200 * progress ** 2
    )

    data.loc[mask, "anomaly"] = 1
    data.loc[mask, "scenario"] = "temperature"

    return data


def inject_vibration(data, start=70):
    mask = data["time_s"] >= start

    progress = np.clip(
        (data.loc[mask, "time_s"] - start) / 30,
        0,
        1,
    )

    data.loc[mask, "vibration_g"] += (
        0.7 * progress ** 2
    )

    data.loc[mask, "anomaly"] = 1
    data.loc[mask, "scenario"] = "vibration"

    return data


def inject_thrust(data, start=70):
    mask = data["time_s"] >= start

    progress = np.clip(
        (data.loc[mask, "time_s"] - start) / 30,
        0,
        1,
    )

    data.loc[mask, "thrust_n"] -= (
        200 * progress ** 2
    )

    data.loc[mask, "anomaly"] = 1
    data.loc[mask, "scenario"] = "thrust"

    return data


def inject_combined(data, start=70):
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
        160 * progress ** 2
    )

    data.loc[mask, "vibration_g"] += (
        0.5 * progress ** 2
    )

    data.loc[mask, "thrust_n"] -= (
        170 * progress ** 2
    )

    data.loc[mask, "anomaly"] = 1
    data.loc[mask, "scenario"] = "combined"

    return data


def save(name, data):
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
        "Rocket Guardian AI — V11 Dataset"
    )
    print(
        "================================"
    )

    # Multiple normal training runs.
    for i in range(1, 6):

        data = generate_run(
            seed=5000 + i
        )

        save(
            f"training_normal_{i}",
            data,
        )

    # Unseen normal validation runs.
    for i in range(1, 3):

        data = generate_run(
            seed=6000 + i
        )

        save(
            f"validation_normal_{i}",
            data,
        )

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
            seed=7000 + i
        )

        data = injector(data)

        save(
            name,
            data,
        )

    print(
        "\nV11 dataset generation complete."
    )


if __name__ == "__main__":
    main()