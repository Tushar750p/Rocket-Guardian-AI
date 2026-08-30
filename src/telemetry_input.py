import pandas as pd


# ============================================================
# REQUIRED TELEMETRY COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "time_s",
    "phase",
    "pressure_kpa",
    "temperature_k",
    "vibration_g",
    "thrust_n",
]


# ============================================================
# OPTIONAL COLUMNS
# ============================================================

OPTIONAL_COLUMNS = [
    "anomaly",
    "scenario",
]


# ============================================================
# LOAD TELEMETRY CSV
# ============================================================

def load_telemetry_csv(file_source):
    """
    Load and validate a Rocket Guardian AI telemetry CSV.

    Required columns:
        time_s
        phase
        pressure_kpa
        temperature_k
        vibration_g
        thrust_n

    Optional columns:
        anomaly
        scenario

    Returns:
        pandas.DataFrame
    """

    try:
        data = pd.read_csv(file_source)

    except Exception as exc:
        raise ValueError(
            f"Unable to read telemetry CSV: {exc}"
        )


    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:

        missing = ", ".join(missing_columns)

        raise ValueError(
            "Invalid telemetry file. "
            f"Missing required columns: {missing}"
        )


    # --------------------------------------------------------
    # Check that telemetry is not empty
    # --------------------------------------------------------

    if data.empty:

        raise ValueError(
            "Telemetry file is empty."
        )


    # --------------------------------------------------------
    # Validate numeric sensor columns
    # --------------------------------------------------------

    numeric_columns = [
        "time_s",
        "pressure_kpa",
        "temperature_k",
        "vibration_g",
        "thrust_n",
    ]

    for column in numeric_columns:

        converted = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        if converted.isna().any():

            raise ValueError(
                f"Column '{column}' contains "
                "invalid or non-numeric values."
            )

        data[column] = converted


    # --------------------------------------------------------
    # Validate phase
    # --------------------------------------------------------

    if data["phase"].isna().any():

        raise ValueError(
            "Column 'phase' contains missing values."
        )


    # --------------------------------------------------------
    # Sort telemetry by time
    # --------------------------------------------------------

    data = data.sort_values(
        "time_s"
    ).reset_index(
        drop=True
    )


    # --------------------------------------------------------
    # Validate time
    # --------------------------------------------------------

    if not data["time_s"].is_monotonic_increasing:

        raise ValueError(
            "Telemetry time values must be "
            "in increasing order."
        )


    # --------------------------------------------------------
    # Remove completely empty optional columns
    # --------------------------------------------------------

    for column in OPTIONAL_COLUMNS:

        if column in data.columns:

            if data[column].isna().all():

                data = data.drop(
                    columns=[column]
                )


    return data