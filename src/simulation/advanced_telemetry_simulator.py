import numpy as np
import pandas as pd
from pathlib import Path


OUTPUT_DIR = Path("data/raw")

FEATURES = [
    "pressure_kpa",
    "temperature_k",
    "vibration_g",
    "thrust_n",
]


def generate_base_telemetry(duration_seconds=60, sample_rate=10, seed=42):
    rng = np.random.default_rng(seed)

    total_samples = duration_seconds * sample_rate
    time = np.arange(total_samples) / sample_rate

    ramp = np.clip(time / 10.0, 0, 1)

    pressure = 40 + 80 * ramp + rng.normal(0, 1.5, total_samples)
    temperature = 300 + 450 * ramp + rng.normal(0, 4, total_samples)
    vibration = 0.15 + 0.05 * ramp + rng.normal(0, 0.015, total_samples)
    thrust = 500 * ramp + rng.normal(0, 8, total_samples)

    return pd.DataFrame(
        {
            "time_s": time,
            "pressure_kpa": pressure,
            "temperature_k": temperature,
            "vibration_g": vibration,
            "thrust_n": thrust,
            "anomaly": 0,
        }
    )


def mark_anomaly(data, start_time):
    data.loc[data["time_s"] >= start_time, "anomaly"] = 1
    return data


def inject_pressure_anomaly(data, start_time=40):
    mask = data["time_s"] >= start_time
    progress = np.linspace(0, 1, mask.sum())

    data.loc[mask, "pressure_kpa"] += 25 * progress
    data.loc[mask, "temperature_k"] += 20 * progress

    return mark_anomaly(data, start_time)


def inject_temperature_anomaly(data, start_time=40):
    mask = data["time_s"] >= start_time
    progress = np.linspace(0, 1, mask.sum())

    data.loc[mask, "temperature_k"] += 180 * progress

    return mark_anomaly(data, start_time)


def inject_vibration_anomaly(data, start_time=40):
    mask = data["time_s"] >= start_time
    progress = np.linspace(0, 1, mask.sum())

    data.loc[mask, "vibration_g"] += 0.6 * progress

    return mark_anomaly(data, start_time)


def inject_thrust_degradation(data, start_time=40):
    mask = data["time_s"] >= start_time
    progress = np.linspace(0, 1, mask.sum())

    data.loc[mask, "thrust_n"] -= 180 * progress

    return mark_anomaly(data, start_time)


def inject_sensor_drift(data, start_time=35):
    mask = data["time_s"] >= start_time
    progress = np.linspace(0, 1, mask.sum())

    data.loc[mask, "pressure_kpa"] += 15 * progress

    return mark_anomaly(data, start_time)


def inject_combined_anomaly(data, start_time=40):
    mask = data["time_s"] >= start_time
    progress = np.linspace(0, 1, mask.sum())

    data.loc[mask, "pressure_kpa"] += 25 * progress
    data.loc[mask, "temperature_k"] += 150 * progress
    data.loc[mask, "vibration_g"] += 0.5 * progress
    data.loc[mask, "thrust_n"] -= 150 * progress

    return mark_anomaly(data, start_time)


def create_scenario(name, injector, seed):
    data = generate_base_telemetry(seed=seed)

    data["scenario"] = name

    return injector(data)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = {
        "pressure": inject_pressure_anomaly,
        "temperature": inject_temperature_anomaly,
        "vibration": inject_vibration_anomaly,
        "thrust": inject_thrust_degradation,
        "sensor_drift": inject_sensor_drift,
        "combined": inject_combined_anomaly,
    }

    for index, (name, injector) in enumerate(scenarios.items()):
        data = create_scenario(
            name=name,
            injector=injector,
            seed=100 + index,
        )

        output_path = OUTPUT_DIR / f"{name}_telemetry.csv"

        data.to_csv(output_path, index=False)

        print(
            f"Generated {name:12s} "
            f"→ {len(data)} samples "
            f"→ anomalies: {data['anomaly'].sum()} "
            f"→ {output_path}"
        )

    print("\nAdvanced telemetry generation complete.")


if __name__ == "__main__":
    main()