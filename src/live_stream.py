from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
import requests

REQUIRED_COLUMNS = [
    "time_s",
    "phase",
    "pressure_kpa",
    "temperature_k",
    "vibration_g",
    "thrust_n",
]


def _normalize(payload: Any) -> pd.DataFrame:
    if isinstance(payload, dict):
        payload = payload.get("telemetry", payload.get("data", payload))
    if not isinstance(payload, list):
        raise ValueError("Live endpoint must return a telemetry list or {'telemetry': [...]}.")

    frame = pd.DataFrame(payload)
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError("Live telemetry missing columns: " + ", ".join(missing))

    for column in [c for c in REQUIRED_COLUMNS if c != "phase"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["phase"] = frame["phase"].astype(str).str.strip().str.lower()

    if frame[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Live telemetry contains missing or invalid values.")
    if not frame["phase"].isin({"startup", "ramp", "steady", "shutdown"}).all():
        raise ValueError("Live telemetry contains an unsupported phase.")
    if frame["time_s"].duplicated().any():
        frame = frame.drop_duplicates(subset=["time_s"], keep="last")

    return frame[REQUIRED_COLUMNS].sort_values("time_s", ignore_index=True)


def fetch_http_telemetry(url: str, timeout: float = 3.0) -> pd.DataFrame:
    headers = {"Accept": "application/json"}
    token = os.getenv("ROCKET_GUARDIAN_LIVE_API_KEY", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.get(url, timeout=timeout, headers=headers)
    response.raise_for_status()
    return _normalize(response.json())


def get_live_source() -> str:
    return os.getenv("ROCKET_GUARDIAN_LIVE_URL", "").strip()


def load_live_telemetry(url: str | None = None) -> tuple[pd.DataFrame | None, str]:
    endpoint = (url or get_live_source()).strip()
    if not endpoint:
        return None, "simulator"
    try:
        frame = fetch_http_telemetry(endpoint)
        return frame, "http"
    except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
        return None, f"http_error: {exc}"
