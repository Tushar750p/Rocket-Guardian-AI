import sqlite3
from pathlib import Path


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

DATABASE_PATH = DATA_DIR / "rocket_guardian.db"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    """
    Create a connection to the Rocket Guardian database.
    """

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def initialize_database():
    """
    Create the core application tables if they do not exist.
    """

    connection = get_connection()

    try:

        cursor = connection.cursor()

        # ----------------------------------------------------
        # Customers
        # ----------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS customers (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                name TEXT NOT NULL,

                email TEXT UNIQUE NOT NULL,

                created_at TEXT NOT NULL
                    DEFAULT CURRENT_TIMESTAMP

            )
            """
        )

        # ----------------------------------------------------
        # Missions
        # ----------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS missions (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                customer_id INTEGER,

                name TEXT NOT NULL,

                description TEXT,

                created_at TEXT NOT NULL
                    DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (
                    customer_id
                )
                REFERENCES customers(id)

            )
            """
        )

        # ----------------------------------------------------
        # Telemetry Runs
        # ----------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS telemetry_runs (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                mission_id INTEGER,

                source_filename TEXT,

                sample_count INTEGER,

                ai_detection_count INTEGER,

                overall_risk REAL,

                risk_level TEXT,

                primary_risk_sensor TEXT,

                peak_time_s REAL,

                created_at TEXT NOT NULL
                    DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (
                    mission_id
                )
                REFERENCES missions(id)

            )
            """
        )

        connection.commit()

    finally:

        connection.close()


# ============================================================
# CREATE CUSTOMER
# ============================================================

def create_customer(
    name,
    email,
):
    """
    Create a customer and return its database ID.
    """

    connection = get_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO customers (
                name,
                email
            )
            VALUES (?, ?)
            """,
            (
                name,
                email,
            ),
        )

        connection.commit()

        return cursor.lastrowid

    finally:

        connection.close()


# ============================================================
# CREATE MISSION
# ============================================================

def create_mission(
    customer_id,
    name,
    description=None,
):
    """
    Create a mission for a customer and return its ID.
    """

    connection = get_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO missions (
                customer_id,
                name,
                description
            )
            VALUES (?, ?, ?)
            """,
            (
                customer_id,
                name,
                description,
            ),
        )

        connection.commit()

        return cursor.lastrowid

    finally:

        connection.close()


# ============================================================
# SAVE TELEMETRY RUN
# ============================================================

def save_telemetry_run(
    mission_id,
    source_filename,
    sample_count,
    ai_detection_count,
    overall_risk,
    risk_level,
    primary_risk_sensor,
    peak_time_s,
):
    """
    Save a completed telemetry analysis run.
    """

    connection = get_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO telemetry_runs (
                mission_id,
                source_filename,
                sample_count,
                ai_detection_count,
                overall_risk,
                risk_level,
                primary_risk_sensor,
                peak_time_s
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mission_id,
                source_filename,
                sample_count,
                ai_detection_count,
                overall_risk,
                risk_level,
                primary_risk_sensor,
                peak_time_s,
            ),
        )

        connection.commit()

        return cursor.lastrowid

    finally:

        connection.close()


# ============================================================
# LIST CUSTOMERS
# ============================================================

def list_customers():
    """
    Return all customers, newest first.
    """

    connection = get_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                name,
                email,
                created_at
            FROM customers
            ORDER BY created_at DESC
            """
        )

        return cursor.fetchall()

    finally:

        connection.close()


# ============================================================
# LIST MISSIONS
# ============================================================

def list_missions(
    customer_id=None,
):
    """
    Return missions.

    If customer_id is provided, only that customer's
    missions are returned.
    """

    connection = get_connection()

    try:

        cursor = connection.cursor()

        if customer_id is None:

            cursor.execute(
                """
                SELECT
                    id,
                    customer_id,
                    name,
                    description,
                    created_at
                FROM missions
                ORDER BY created_at DESC
                """
            )

        else:

            cursor.execute(
                """
                SELECT
                    id,
                    customer_id,
                    name,
                    description,
                    created_at
                FROM missions
                WHERE customer_id = ?
                ORDER BY created_at DESC
                """,
                (
                    customer_id,
                ),
            )

        return cursor.fetchall()

    finally:

        connection.close()


# ============================================================
# LIST TELEMETRY RUNS
# ============================================================

def list_telemetry_runs(
    mission_id=None,
):
    """
    Return saved telemetry analysis runs.

    If mission_id is provided, only runs for that mission
    are returned.
    """

    connection = get_connection()

    try:

        cursor = connection.cursor()

        if mission_id is None:

            cursor.execute(
                """
                SELECT
                    id,
                    mission_id,
                    source_filename,
                    sample_count,
                    ai_detection_count,
                    overall_risk,
                    risk_level,
                    primary_risk_sensor,
                    peak_time_s,
                    created_at
                FROM telemetry_runs
                ORDER BY created_at DESC
                """
            )

        else:

            cursor.execute(
                """
                SELECT
                    id,
                    mission_id,
                    source_filename,
                    sample_count,
                    ai_detection_count,
                    overall_risk,
                    risk_level,
                    primary_risk_sensor,
                    peak_time_s,
                    created_at
                FROM telemetry_runs
                WHERE mission_id = ?
                ORDER BY created_at DESC
                """,
                (
                    mission_id,
                ),
            )

        return cursor.fetchall()

    finally:

        connection.close()


# ============================================================
# INITIALIZE DATABASE
# ============================================================
# ============================================================
# FIND OR CREATE CUSTOMER
# ============================================================

def get_or_create_customer(
    name,
    email,
):
    """
    Return an existing customer ID by email,
    or create a new customer.
    """

    name = str(name).strip()
    email = str(email).strip().lower()

    connection = get_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT id
            FROM customers
            WHERE email = ?
            """,
            (
                email,
            ),
        )

        row = cursor.fetchone()

        if row is not None:

            return int(
                row["id"]
            )

        cursor.execute(
            """
            INSERT INTO customers (
                name,
                email
            )
            VALUES (?, ?)
            """,
            (
                name,
                email,
            ),
        )

        connection.commit()

        return int(
            cursor.lastrowid
        )

    finally:

        connection.close()


# ============================================================
# FIND TELEMETRY RUN
# ============================================================

def find_telemetry_run(
    source_filename,
    customer_id,
    mission_name,
):
    """
    Find an existing telemetry run for a customer,
    mission, and source filename.
    """

    connection = get_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                tr.id
            FROM telemetry_runs tr
            JOIN missions m
                ON tr.mission_id = m.id
            WHERE tr.source_filename = ?
            AND m.customer_id = ?
            AND m.name = ?
            ORDER BY tr.id DESC
            LIMIT 1
            """,
            (
                source_filename,
                customer_id,
                mission_name,
            ),
        )

        row = cursor.fetchone()

        return row

    finally:

        connection.close()

initialize_database()