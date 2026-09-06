import time

import numpy as np
import pandas as pd


def generate_live_telemetry(now: float | None = None, samples: int = 600) -> pd.DataFrame:
    """Generate a live, rolling rocket telemetry stream for dashboard use.

    This is a software simulator, not a physical sensor feed. The returned
    window always ends at the current wall-clock time and includes a repeatable
    anomaly event so the existing AI/risk pipeline can run continuously.
    """
    now = time.time() if now is None else float(now)
    dt = 0.25
    end_t = now % 180.0
    time_s = end_t - (np.arange(samples - 1, -1, -1) * dt)
    time_s = np.mod(time_s, 180.0)

    phase = np.select(
        [time_s < 15.0, time_s < 40.0, time_s < 155.0],
        ["startup", "ramp", "steady"],
        default="shutdown",
    )

    p_base = np.select(
        [phase == "startup", phase == "ramp", phase == "steady"],
        [101.0, 108.0 + 0.15 * time_s, 112.0],
        default=108.0,
    )
    t_base = np.select(
        [phase == "startup", phase == "ramp", phase == "steady"],
        [295.0, 300.0 + 0.08 * time_s, 312.0],
        default=305.0,
    )
    thrust_base = np.select(
        [phase == "startup", phase == "ramp", phase == "steady"],
        [120.0 + 4.0 * time_s, 220.0 + 8.0 * time_s, 540.0],
        default=350.0,
    )

    vibration = 0.22 + 0.02 * np.sin(time_s / 2.5)
    rng = np.random.default_rng(int(now // 2))
    noise = rng.normal(0.0, 1.0, (samples, 4))

    # Rolling anomaly window: vibration rises sharply, with smaller coupled
    # pressure/thrust deviations to exercise the multi-sensor risk engine.
    event = ((time_s >= 90.0) & (time_s <= 108.0)).astype(float)

    pressure = p_base + 0.45 * noise[:, 0] - 7.5 * event
    temperature = t_base + 0.35 * noise[:, 1] + 6.0 * event
    vibration = vibration + 0.015 * noise[:, 2] + 0.85 * event
    thrust = thrust_base + 2.0 * noise[:, 3] - 65.0 * event

    return pd.DataFrame(
        {
            "time_s": time_s,
            "phase": phase,
            "pressure_kpa": pressure,
            "temperature_k": temperature,
            "vibration_g": vibration,
            "thrust_n": thrust,
        }
    ).sort_values("time_s", ignore_index=True)
