import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import os
import sys
import tempfile
import time

from streamlit_autorefresh import st_autorefresh

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.telemetry_analysis import analyze_telemetry
from src.live_telemetry import analyze_live_telemetry, score_live_frame
from src.live_stream import load_live_telemetry
from src.ai_agents import MissionAgent
from src.risk_analysis import analyze_risk, summarize_risk
from src.database import (
    create_customer,
    create_mission,
    get_connection,
    initialize_database,
    save_telemetry_run,
    get_customer_by_auth_user_id,
    link_customer_to_auth_user,
    get_customer_mission_history,
)
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from src.auth import (
    get_current_user,
    render_login,
    sign_out,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Rocket Guardian AI",
    page_icon="[ROCKET]",
    layout="wide",
)

# ============================================================
# AUTHENTICATION
# ============================================================

current_user = get_current_user()

if current_user is None:
    render_login()
    st.stop()

current_user_email = (
    getattr(current_user, "email", None) or ""
).strip().lower()

if not current_user_email:
    st.error("Authenticated user email is unavailable.")
    st.stop()

# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("data/v11")
RESULT_DIR = Path("data/processed/v11")
RISK_DIR = Path("data/processed/v14")


# ============================================================
# V14 RISK FILES
# ============================================================

RISK_FILES = {
    "Combined Failure": "combined_v14.csv",
    "Pressure Failure": "pressure_v14.csv",
    "Temperature Failure": "temperature_v14.csv",
    "Thrust Failure": "thrust_v14.csv",
    "Vibration Failure": "vibration_v14.csv",
}


# ============================================================
# SENSOR CONFIGURATION
# ============================================================

# ============================================================
# PERFORMANCE CACHE HELPERS
# ============================================================

@st.cache_resource(show_spinner=False)
def _initialize_database_cached():
    initialize_database()


@st.cache_data(show_spinner=False)
def _read_csv_cached(file_path: str):
    return pd.read_csv(file_path)


@st.cache_data(show_spinner=False)
def _analyze_telemetry_cached(uploaded_bytes: bytes):
    fd, temp_path = tempfile.mkstemp(suffix=".csv")
    try:
        with os.fdopen(fd, "wb") as temp_file:
            temp_file.write(uploaded_bytes)
        return analyze_telemetry(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@st.cache_data(ttl=1, show_spinner=False)
def _analyze_live_telemetry_cached(tick: int):
    return analyze_live_telemetry(float(tick))


@st.cache_data(show_spinner=False)
def _analyze_risk_cached(frame):
    return analyze_risk(frame)


_initialize_database_cached()

SENSORS = {
    "Pressure": {
        "column": "pressure_kpa",
        "unit": "kPa",
        "alert_column": "pressure_kpa_alert",
        "risk_column": "pressure_risk",
    },
    "Temperature": {
        "column": "temperature_k",
        "unit": "K",
        "alert_column": "temperature_k_alert",
        "risk_column": "temperature_risk",
    },
    "Vibration": {
        "column": "vibration_g",
        "unit": "g",
        "alert_column": "vibration_g_alert",
        "risk_column": "vibration_risk",
    },
    "Thrust": {
        "column": "thrust_n",
        "unit": "N",
        "alert_column": "thrust_n_alert",
        "risk_column": "thrust_risk",
    },
}


# ============================================================
# SCENARIOS
# ============================================================

SCENARIOS = {
    "Combined Failure": "test_combined_results.csv",
    "Pressure Failure": "test_pressure_results.csv",
    "Temperature Failure": "test_temperature_results.csv",
    "Thrust Failure": "test_thrust_results.csv",
    "Vibration Failure": "test_vibration_results.csv",
}


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 44px;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 2px;
    }

    .subtitle {
        font-size: 16px;
        opacity: 0.65;
        margin-bottom: 28px;
    }

    .status-card {
        padding: 20px 14px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.22);
        background: rgba(128,128,128,0.04);
        text-align: center;
        margin-bottom: 15px;
    }

    .sensor-name {
        font-size: 15px;
        font-weight: 700;
        letter-spacing: 0.2px;
    }

    .sensor-status {
        font-size: 20px;
        font-weight: 750;
        margin-top: 8px;
    }

    .normal {
        color: #16803c;
    }

    .warning {
        color: #b77900;
    }

    .critical {
        color: #c62828;
    }

    .risk-box {
        padding: 22px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.22);
        background: rgba(128,128,128,0.04);
        margin-bottom: 18px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Rocket Guardian AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "AI-Powered Rocket Telemetry Risk Monitoring"
    "</div>",
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Mission Controls")

analysis_mode = st.sidebar.radio(
    "Analysis Mode",
    [
        "Demo Mission",
        "Live Mission",
        "Customer Upload",
    ],
)

st.sidebar.divider()


# ============================================================
# DEMO MODE
# ============================================================

if analysis_mode == "Demo Mission":

    st.sidebar.subheader("Demo Mission")

    selected = st.sidebar.selectbox(
        "Test Scenario",
        list(SCENARIOS.keys()),
    )

    uploaded_file = None

elif analysis_mode == "Live Mission":

    st.sidebar.subheader("Live Mission")
    refresh_seconds = st.sidebar.slider(
        "Update interval (seconds)",
        min_value=1,
        max_value=5,
        value=2,
    )
    st_autorefresh(
        interval=refresh_seconds * 1000,
        key="rocket_guardian_live_refresh",
    )
    live_source = st.sidebar.selectbox(
        "Live telemetry source",
        ["Simulator", "HTTP endpoint"],
    )

    live_endpoint = ""
    if live_source == "HTTP endpoint":
        live_endpoint = st.sidebar.text_input(
            "Telemetry endpoint URL",
            value=os.getenv("ROCKET_GUARDIAN_LIVE_URL", ""),
            placeholder="https://your-server.example/api/telemetry",
        ).strip()
        st.sidebar.caption("Endpoint must return JSON telemetry rows.")

    st.sidebar.success("LIVE telemetry stream active")
    st.sidebar.caption(
        "Real HTTP stream" if live_source == "HTTP endpoint" else "Software simulator"
    )
    selected = "Combined Failure"
    uploaded_file = None

# ============================================================
# CUSTOMER UPLOAD MODE
# ============================================================

else:

    st.sidebar.subheader("Customer Information")

    customer_name = st.sidebar.text_input(
        "Customer Name",
        placeholder="Enter customer name",
    )

    st.sidebar.text_input(
        "Account Email",
        value=current_user_email,
        disabled=True,
    )

    customer_email = current_user_email

    mission_name = st.sidebar.text_input(
        "Mission Name",
        placeholder="Enter mission name",
    )

    st.sidebar.subheader("Telemetry Input")

    uploaded_file = st.sidebar.file_uploader(
        "Upload Telemetry CSV",
        type=["csv"],
        help=(
            "Upload a telemetry CSV containing "
            "time_s, phase, pressure_kpa, "
            "temperature_k, vibration_g, "
            "and thrust_n."
        ),
    )

    selected = "Combined Failure"
# ============================================================
# LOAD TELEMETRY DATA
# ============================================================
customer_mode = analysis_mode == "Customer Upload"
live_mode = analysis_mode == "Live Mission"

if live_mode:

    if live_source == "HTTP endpoint":
        raw_live_data, live_source_status = load_live_telemetry(live_endpoint)
        if raw_live_data is None:
            st.error(f"Live endpoint unavailable: {live_source_status}")
            st.info("Switch Live telemetry source to Simulator, or fix the endpoint.")
            st.stop()
        data = score_live_frame(raw_live_data)
    else:
        live_tick = int(time.time())
        data = _analyze_live_telemetry_cached(live_tick)

elif customer_mode:

    if uploaded_file is None:

        st.info(
            "Please upload a telemetry CSV to begin analysis."
        )

        st.stop()

    if not customer_name.strip():

        st.error(
            "Please enter Customer Name."
        )

        st.stop()

    if not customer_email.strip():

        st.error(
            "Please enter Customer Email."
        )

        st.stop()

    if not mission_name.strip():

        st.error(
            "Please enter Mission Name."
        )

        st.stop()

    try:

        uploaded_bytes = uploaded_file.getvalue()

        required_columns = [
            "time_s",
            "phase",
            "pressure_kpa",
            "temperature_k",
            "vibration_g",
            "thrust_n",
        ]

        preview = pd.read_csv(
            pd.io.common.BytesIO(uploaded_bytes)
        )
        missing_columns = [
            column
            for column in required_columns
            if column not in preview.columns
        ]

        if missing_columns:

            st.error(
                "Invalid telemetry CSV. Missing columns: "
                + ", ".join(missing_columns)
            )

            st.stop()

        if preview.empty:

            st.error(
                "Invalid telemetry CSV. The file contains no data rows."
            )

            st.stop()

        numeric_columns = [
            "time_s",
            "pressure_kpa",
            "temperature_k",
            "vibration_g",
            "thrust_n",
        ]

        for column in numeric_columns:

            converted = pd.to_numeric(
                preview[column],
                errors="coerce",
            )

            invalid_count = int(
                converted.isna().sum()
            )

            if invalid_count > 0:

                st.error(
                    f"Invalid telemetry CSV. Column '{column}' "
                    f"contains {invalid_count} missing or non-numeric value(s)."
                )

                st.stop()

            preview[column] = converted

        if preview["phase"].isna().any():

            st.error(
                "Invalid telemetry CSV. "
                "The 'phase' column contains missing values."
            )

            st.stop()

        supported_phases = {
            "startup",
            "ramp",
            "steady",
            "shutdown",
        }

        phase_values = (
            preview["phase"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        unsupported_phases = sorted(
            set(phase_values.unique()) - supported_phases
        )

        if unsupported_phases:

            st.error(
                "Invalid telemetry CSV. Unsupported phase value(s): "
                + ", ".join(unsupported_phases)
                + ". Supported phases: startup, ramp, steady, shutdown."
            )

            st.stop()

        preview["phase"] = phase_values
        if len(preview) < 10:

            st.error(
                "Invalid telemetry CSV. "
                "At least 10 telemetry rows are required."
            )

            st.stop()

        fd, temp_path = tempfile.mkstemp(
            suffix=".csv"
        )
              
       
        try:

            with os.fdopen(fd, "wb") as temp_file:

                temp_file.write(
                    uploaded_bytes
                )

            data = _analyze_telemetry_cached(uploaded_bytes)

        finally:

            if os.path.exists(temp_path):

                os.remove(temp_path)

        st.sidebar.success(
            "Telemetry analyzed successfully."
        )

    except ValueError as exc:

        st.error(
            f"Telemetry validation failed: {exc}"
        )

        st.stop()

    except Exception as exc:

        st.error(
            f"Telemetry analysis failed: {exc}"
        )

        st.stop()

else:

    file_path = (
        RESULT_DIR
        / SCENARIOS[selected]
    )

    if not file_path.exists():

        st.error(
            f"Result file not found:\n{file_path}"
        )

        st.stop()

    data = _read_csv_cached(str(file_path))
   

# ============================================================
# LOAD V14 RISK DATA
# ============================================================

if live_mode:

    risk_data = _analyze_risk_cached(data)
    live_agents = MissionAgent.run(risk_data)

    st.markdown("### Live AI Agent Console")

    agent_cols = st.columns(3)
    with agent_cols[0]:
        st.metric(
            "Anomaly Agent",
            live_agents["anomaly"]["status"],
            f'{live_agents["anomaly"]["detections"]} detections',
        )
    with agent_cols[1]:
        st.metric(
            "Sensor Agent",
            live_agents["sensor"]["top_sensor"],
            f'{live_agents["sensor"]["top_risk"]:.1f} risk',
        )
    with agent_cols[2]:
        st.metric(
            "Trend Agent",
            live_agents["trend"]["direction"],
            f'{live_agents["trend"]["recent_risk"]:.1f} recent risk',
        )

    agent_cols_2 = st.columns(3)
    with agent_cols_2[0]:
        st.metric(
            "Telemetry Health",
            live_agents["telemetry"]["status"],
            f'{live_agents["telemetry"]["quality"]}% quality',
        )
    with agent_cols_2[1]:
        st.metric(
            "Mission Agent",
            live_agents["mission"]["mission_state"],
            f'Peak {live_agents["mission"]["peak_time_s"]:.2f}s',
        )
    with agent_cols_2[2]:
        incident_state = "ACTIVE" if live_agents["incident"]["active"] else "CLEAR"
        st.metric(
            "Incident Agent",
            incident_state,
            f'{live_agents["incident"]["event_count"]} event(s)',
        )

    st.info(live_agents["mission"]["recommendation"])

    with st.expander("Agent details", expanded=False):
        st.write({
            "Anomaly": live_agents["anomaly"],
            "Sensor": live_agents["sensor"],
            "Trend": live_agents["trend"],
            "Telemetry": live_agents["telemetry"],
            "Mission": live_agents["mission"],
            "Incident": live_agents["incident"],
            "Generated at": live_agents["generated_at"],
        })

    # Live Mission PDF report
    from io import BytesIO
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import ParagraphStyle

    live_pdf = BytesIO()
    live_doc = SimpleDocTemplate(
        live_pdf,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=18 * mm,
        title="Rocket Guardian AI Live Mission Report",
        author="Rocket Guardian AI",
    )
    live_styles = getSampleStyleSheet()
    live_title = ParagraphStyle(
        "LiveTitle", parent=live_styles["Title"], fontName="Helvetica-Bold",
        fontSize=21, leading=25, alignment=1, textColor=colors.HexColor("#102A43")
    )
    live_section = ParagraphStyle(
        "LiveSection", parent=live_styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=11, leading=14, textColor=colors.HexColor("#102A43"), spaceBefore=7, spaceAfter=6
    )
    live_body = ParagraphStyle(
        "LiveBody", parent=live_styles["BodyText"], fontName="Helvetica",
        fontSize=9.2, leading=12.5, textColor=colors.HexColor("#172B4D")
    )
    live_small = ParagraphStyle(
        "LiveSmall", parent=live_styles["BodyText"], fontName="Helvetica",
        fontSize=7.5, leading=9, textColor=colors.HexColor("#667085")
    )

    peak = risk_data.loc[risk_data["overall_risk"].idxmax()]
    live_story = [
        Paragraph("Rocket Guardian AI", live_title),
        Paragraph("LIVE MISSION TELEMETRY & MULTI-AGENT ASSESSMENT", live_small),
        Spacer(1, 8),
    ]
    live_summary = Table(
        [
            ["Samples", "Peak Risk", "Status", "Primary Sensor"],
            [str(len(risk_data)), f'{float(peak["overall_risk"]):.1f}/100', str(peak["risk_level"]), str(peak["primary_risk_sensor"])],
        ],
        colWidths=[42 * mm, 42 * mm, 42 * mm, 42 * mm],
    )
    live_summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F5F8FA")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#D0D5DD")),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D0D5DD")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    live_story.append(live_summary)

    live_story.append(Paragraph("Multi-Agent Assessment", live_section))
    agent_rows = [["Agent", "State / Finding", "Detail"]]
    agent_rows.extend([
        ["Anomaly Agent", live_agents["anomaly"]["status"], f'{live_agents["anomaly"]["detections"]} detections; primary {live_agents["anomaly"]["primary_sensor"]}'],
        ["Sensor Agent", live_agents["sensor"]["top_sensor"], f'{live_agents["sensor"]["top_risk"]:.1f}/100 top sensor risk'],
        ["Trend Agent", live_agents["trend"]["direction"], f'{live_agents["trend"]["recent_risk"]:.1f} recent risk; slope {live_agents["trend"]["slope"]}'],
        ["Telemetry Health", live_agents["telemetry"]["status"], f'{live_agents["telemetry"]["quality"]}% quality'],
        ["Mission Agent", live_agents["mission"]["mission_state"], live_agents["mission"]["recommendation"]],
        ["Incident Agent", "ACTIVE" if live_agents["incident"]["active"] else "CLEAR", f'{live_agents["incident"]["event_count"]} incident event(s)'],
    ])
    agent_table = Table(agent_rows, colWidths=[40 * mm, 40 * mm, 88 * mm], repeatRows=1)
    agent_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F5F8FA")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D0D5DD")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    live_story.append(agent_table)
    live_story.append(Spacer(1, 7))

    live_story.append(Paragraph("Peak Risk Event", live_section))
    live_story.append(Paragraph(
        f'Peak system risk was <b>{float(peak["overall_risk"]):.1f}/100</b> at <b>{float(peak["time_s"]):.2f} s</b>. '<
        f'<b>{peak["primary_risk_sensor"]}</b> was the primary risk sensor.',
        live_body,
    ))

    incident_events = live_agents["incident"].get("events", [])
    if incident_events:
        live_story.append(Paragraph("Recent Critical Incidents", live_section))
        incident_rows = [["Start (s)", "End (s)", "Peak Risk", "Sensor"]]
        for event in incident_events:
            incident_rows.append([str(event["start_s"]), str(event["end_s"]), f'{event["peak_risk"]:.1f}/100', str(event["sensor"])])
        incident_table = Table(incident_rows, colWidths=[36 * mm, 36 * mm, 44 * mm, 52 * mm], repeatRows=1)
        incident_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#FEF3F2")),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D0D5DD")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        live_story.append(incident_table)

    live_story.append(Spacer(1, 7))
    live_story.append(Paragraph("Prototype limitation: risk and agent outputs are analytical indicators and are not flight-safety guarantees.", live_small))
    live_doc.build(live_story)
    st.download_button(
        label="Download Live PDF Report",
        data=live_pdf.getvalue(),
        file_name="rocket_guardian_live_mission_report.pdf",
        mime="application/pdf",
    )

elif customer_mode:

    try:

        risk_data = _analyze_risk_cached(data)

        # ----------------------------------------------------
        # Save customer analysis to database
        # ----------------------------------------------------

        source_filename = uploaded_file.name

        # ----------------------------------------------------
        # Link logged-in Auth user to customer
        # ----------------------------------------------------

        customer_id = None

        # First: find customer already linked to this Auth user
        customer_row = get_customer_by_auth_user_id(
            current_user.id
        )

        if customer_row is not None:

            customer_id = int(
                customer_row["id"]
            )

        else:

            # Second: try existing customer with same login email
            connection = get_connection()

            try:

                cursor = connection.cursor()

                cursor.execute(
                    """
                    SELECT id
                    FROM customers
                    WHERE LOWER(email) = LOWER(?)
                    ORDER BY id ASC
                    LIMIT 1
                    """,
                    (
                        current_user_email,
                    ),
                )

                customer_row = cursor.fetchone()

            finally:

                connection.close()

            if customer_row is not None:

                customer_id = int(
                    customer_row["id"]
                )

                link_customer_to_auth_user(
                    customer_id,
                    current_user.id,
                )

            else:

                customer_id = int(
                    create_customer(
                        customer_name,
                        current_user_email,
                    )
                )

                link_customer_to_auth_user(
                    customer_id,
                    current_user.id,
                )

        # ----------------------------------------------------
        # Check whether this file was already saved
        # ----------------------------------------------------

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

            existing_run = cursor.fetchone()

        finally:

            connection.close()

        # ----------------------------------------------------
        # Only create a new mission/run for a new file
        # ----------------------------------------------------

        if existing_run is None:

            mission_id = create_mission(
                customer_id=customer_id,
                name=mission_name,
                description=(
                    "Customer telemetry analysis run."
                ),
            )

            # ------------------------------------------------
            # Find peak risk
            # ------------------------------------------------

            peak_index = risk_data[
                "overall_risk"
            ].idxmax()

            peak_row = risk_data.loc[
                peak_index
            ]

            # ------------------------------------------------
            # Save telemetry run
            # ------------------------------------------------

            save_telemetry_run(
                mission_id=mission_id,
                source_filename=source_filename,
                sample_count=len(data),
                ai_detection_count=int(
                    data["ai_anomaly"].sum()
                ),
                overall_risk=float(
                    peak_row["overall_risk"]
                ),
                risk_level=str(
                    peak_row["risk_level"]
                ),
                primary_risk_sensor=str(
                    peak_row["primary_risk_sensor"]
                ),
                peak_time_s=float(
                    peak_row["time_s"]
                ),
            )

            st.sidebar.success(
                "Analysis saved to mission history."
            )

        else:

            st.sidebar.info(
                "This telemetry file is already "
                "saved in mission history."
            )


    except ValueError as exc:

        st.error(
            f"Risk analysis failed: {exc}"
        )

        st.stop()

    except Exception as exc:

        st.warning(
            "Risk analysis completed, but "
            f"database save failed: {exc}"
        )

else:

    risk_file = (
        RISK_DIR
        / RISK_FILES[selected]
    )

    if risk_file.exists():

        risk_data = _read_csv_cached(str(risk_file))

    else:

        risk_data = None


# ============================================================
# OVERALL SYSTEM STATUS
# ============================================================

if risk_data is not None and "overall_risk" in risk_data.columns:

    peak_overall_risk = float(
        risk_data["overall_risk"].max()
    )

    if peak_overall_risk >= 75:

        highest_status = "CRITICAL"

    elif peak_overall_risk >= 45:

        highest_status = "WARNING"

    else:

        highest_status = "NORMAL"

else:

    severity = {
        "NORMAL": 0,
        "WARNING": 1,
        "CRITICAL": 2,
    }

    highest_status = max(
        data["status"],
        key=lambda x: severity.get(x, 0),
    )

# ============================================================
# MISSION STATUS
# ============================================================

if highest_status == "CRITICAL":

    st.error(
        "MISSION STATUS: CRITICAL\n\n"
        "Significant telemetry anomaly detected."
    )

elif highest_status == "WARNING":

    st.warning(
        "MISSION STATUS: WARNING\n\n"
        "Potential telemetry anomaly detected."
    )

else:

    st.success(
        "MISSION STATUS: NORMAL\n\n"
        "Telemetry within the expected operating envelope."
    )


# ============================================================
# CORE METRICS
# ============================================================

ai_detections = int(
    data["ai_anomaly"].sum()
)

total_samples = len(data)

actual_anomalies = None
detection_delay = None
start_time = None


# ------------------------------------------------------------
# Research / Demo telemetry with ground-truth labels
# ------------------------------------------------------------

if not customer_mode:

    if "anomaly" in data.columns:

        actual_anomalies = int(
            data["anomaly"].sum()
        )

        anomaly_times = data.loc[
            data["anomaly"] == 1,
            "time_s",
        ]

        detection_times = data.loc[
            data["ai_anomaly"] == 1,
            "time_s",
        ]

        if not anomaly_times.empty:

            start_time = anomaly_times.iloc[0]

            valid_detection = detection_times[
                detection_times >= start_time
            ]

            if not valid_detection.empty:

                detection_delay = (
                    valid_detection.iloc[0]
                    - start_time
                )


# ============================================================
# MISSION OVERVIEW
# ============================================================

st.subheader("Mission Overview")

col1, col2, col3, col4 = st.columns(4)


with col1:

    if customer_mode:

        detection_rate = (
            ai_detections / total_samples * 100
            if total_samples > 0
            else 0
        )

        st.metric(
            "AI Detection Rate",
            f"{detection_rate:.1f}%",
        )
with col2:

    st.metric(
        "AI Detections",
        ai_detections,
    )


with col3:

    if customer_mode:

        st.metric(
            "Detection Delay",
            "Ground Truth N/A",
        )

    elif detection_delay is None:

        st.metric(
            "Detection Delay",
            "N/A",
        )

    else:

        st.metric(
            "Detection Delay",
            f"{detection_delay:.1f} s",
        )
with col4:

    st.metric(
        "Telemetry Samples",
        total_samples,
    )


st.divider()

# ============================================================
# SENSOR HEALTH
# ============================================================

st.subheader("Sensor Health")

sensor_columns = st.columns(4)

for ui_column, (name, config) in zip(
    sensor_columns,
    SENSORS.items(),
):

    alert_column = config["alert_column"]
    risk_column = config["risk_column"]

    # Check alert state
    if alert_column in data.columns:
        alert_active = bool(
            data[alert_column].astype(bool).any()
        )
    else:
        alert_active = False

    # Get maximum sensor risk
    sensor_risk = None

    if (
        risk_data is not None
        and risk_column in risk_data.columns
    ):
        sensor_risk = float(
            risk_data[risk_column].max()
        )

    with ui_column:

        st.markdown(f"### {name}")

        if alert_active:
            st.error("ALERT")
        else:
            st.success("NORMAL")

        if sensor_risk is not None:
            st.metric(
                "Risk",
                f"{sensor_risk:.1f}/100",
            )


st.divider()

# ============================================================
# TELEMETRY CHART
# ============================================================

def create_chart(
    chart_data,
    sensor_name,
    config,
):

    column = config["column"]
    unit = config["unit"]
    alert_column = config["alert_column"]

    fig = go.Figure()

    # Main telemetry
    fig.add_trace(
        go.Scatter(
            x=chart_data["time_s"],
            y=chart_data[column],
            mode="lines",
            name=sensor_name,
        )
    )

    # Actual anomaly â€” available only in research/test data
    if "anomaly" in chart_data.columns:
        actual = chart_data[
            chart_data["anomaly"] == 1
        ]
    else:
        actual = pd.DataFrame(columns=chart_data.columns)

    if not actual.empty:
        fig.add_trace(
            go.Scatter(
                x=actual["time_s"],
                y=actual[column],
                mode="markers",
                name="Actual Anomaly",
                marker=dict(size=5),
            )
        )

    # AI alert
    if alert_column in chart_data.columns:
        ai_alerts = chart_data[
            chart_data[alert_column] == True
        ]

        if not ai_alerts.empty:
            fig.add_trace(
                go.Scatter(
                    x=ai_alerts["time_s"],
                    y=ai_alerts[column],
                    mode="markers",
                    name="AI Alert",
                    marker=dict(
                        size=7,
                        symbol="x",
                    ),
                )
            )

    # Anomaly start â€” research/test data only
    if not actual.empty:
        anomaly_start_time = actual[
            "time_s"
        ].iloc[0]

        fig.add_vline(
            x=anomaly_start_time,
            line_dash="dash",
            annotation_text="Anomaly Start",
            annotation_position="top",
        )

    fig.update_layout(
        title=sensor_name,
        xaxis_title="Time (seconds)",
        yaxis_title=unit,
        height=350,
        hovermode="x unified",
        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    return fig


# ============================================================
# ROCKET TELEMETRY
# ============================================================

st.subheader("Rocket Telemetry")


left, right = st.columns(2)


with left:

    st.plotly_chart(
        create_chart(
            data,
            "Pressure",
            SENSORS["Pressure"],
        ),
        width="stretch",
    )


with right:

    st.plotly_chart(
        create_chart(
            data,
            "Temperature",
            SENSORS["Temperature"],
        ),
        width="stretch",
    )


left, right = st.columns(2)


with left:

    st.plotly_chart(
        create_chart(
            data,
            "Vibration",
            SENSORS["Vibration"],
        ),
        width="stretch",
    )


with right:

    st.plotly_chart(
        create_chart(
            data,
            "Thrust",
            SENSORS["Thrust"],
        ),
        width="stretch",
    )


# ============================================================
# ADVANCED AI INSIGHTS
# ============================================================

st.subheader("Advanced AI Insights")

insight_cols = st.columns(4)

with insight_cols[0]:
    st.metric(
        "Anomaly Score",
        f"{float(data['anomaly_score'].max()):.1f}/100"
        if "anomaly_score" in data.columns else "N/A"
    )

with insight_cols[1]:
    st.metric(
        "AI Confidence",
        f"{float(data['confidence'].max()):.1f}%"
        if "confidence" in data.columns else "N/A"
    )

with insight_cols[2]:
    st.metric(
        "Active Sensors",
        int(data["active_sensors"].max())
        if "active_sensors" in data.columns else 0
    )

with insight_cols[3]:
    st.metric(
        "Primary Risk Sensor",
        {"vibration_g":"Vibration","pressure_kpa":"Pressure","temperature_k":"Temperature","thrust_n":"Thrust"}.get(str(data["primary_risk_sensor"].iloc[-1]), str(data["primary_risk_sensor"].iloc[-1]))
        if "primary_risk_sensor" in data.columns else "N/A"
    )


# ============================================================
# AI ANOMALY EXPLANATION
# ============================================================

st.subheader("AI Anomaly Explanation")

if "anomaly_score" in data.columns:

    peak_index = data["anomaly_score"].idxmax()
    peak_row = data.loc[peak_index]

    sensor_values = {
        "Pressure": float(peak_row["pressure_kpa_z"]),
        "Temperature": float(peak_row["temperature_k_z"]),
        "Vibration": float(peak_row["vibration_g_z"]),
        "Thrust": float(peak_row["thrust_n_z"]),
    }

    top_sensor = max(
        sensor_values,
        key=sensor_values.get,
    )

    top_z = sensor_values[top_sensor]

    if top_z >= 8:
        explanation_level = "Severe"
    elif top_z >= 4:
        explanation_level = "High"
    elif top_z >= 2:
        explanation_level = "Moderate"
    else:
        explanation_level = "Low"

    st.write(
        f"**Primary Driver:** {top_sensor}"
    )

    st.write(
        f"**Anomaly Severity:** {explanation_level}"
    )

    st.write(
        f"**Peak Sensor Deviation:** {top_z:.2f}σ from phase baseline"
    )

    st.write(
        f"**Detection Time:** {float(peak_row['time_s']):.2f} seconds"
    )

    st.info(
        f"AI detected the strongest abnormal behavior in the "
        f"{top_sensor} sensor during the {peak_row['phase']} phase. "
        f"The sensor deviation reached {top_z:.2f}σ from its "
        f"phase-specific baseline."
    )

else:

    st.info("Advanced anomaly explanation is not available.")

# ============================================================
# SENSOR CONTRIBUTION
# ============================================================

st.subheader("Sensor Contribution")

contribution = {
    "Pressure": float(data["pressure_kpa_z"].abs().max()),
    "Temperature": float(data["temperature_k_z"].abs().max()),
    "Vibration": float(data["vibration_g_z"].abs().max()),
    "Thrust": float(data["thrust_n_z"].abs().max()),
}

contribution_df = (
    pd.DataFrame(
        list(contribution.items()),
        columns=["Sensor", "Deviation"],
    )
    .sort_values("Deviation", ascending=True)
)

fig = go.Figure(
    go.Bar(
        x=contribution_df["Deviation"],
        y=contribution_df["Sensor"],
        orientation="h",
        text=contribution_df["Deviation"].round(2),
        textposition="auto",
    )
)

fig.update_layout(
    title="Maximum Sensor Deviation from Phase Baseline",
    xaxis_title="Deviation (s)",
    yaxis_title="Sensor",
    height=320,
)

st.plotly_chart(
    fig,
    width="stretch",
)

# ============================================================
# AI ALERT TIMELINE
# ============================================================

st.subheader("AI Alert Timeline")


status_map = {
    "NORMAL": 0,
    "WARNING": 1,
    "CRITICAL": 2,
}


timeline = data["status"].map(
    status_map
)


fig = go.Figure()


fig.add_trace(
    go.Scatter(
        x=data["time_s"],
        y=timeline,
        mode="lines",
        name="System Status",
    )
)


fig.update_yaxes(
    tickmode="array",
    tickvals=[0, 1, 2],
    ticktext=[
        "NORMAL",
        "WARNING",
        "CRITICAL",
    ],
)


fig.update_layout(
    title="System Status Over Time",
    xaxis_title="Time (seconds)",
    yaxis_title="Status",
    height=320,
    margin=dict(
        l=20,
        r=20,
        t=50,
        b=20,
    ),
)


st.plotly_chart(
    fig,
    width="stretch",
)


# ============================================================
# FLIGHT PHASE
# ============================================================

if "phase" in data.columns:

    st.subheader("Rocket Flight Phase")


    phase_map = {
        "startup": 0,
        "ramp": 1,
        "steady": 2,
        "shutdown": 3,
    }


    phase_values = data[
        "phase"
    ].map(phase_map)


    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=data["time_s"],
            y=phase_values,
            mode="lines",
            name="Flight Phase",
        )
    )


    fig.update_yaxes(
        tickmode="array",
        tickvals=[0, 1, 2, 3],
        ticktext=[
            "STARTUP",
            "RAMP",
            "STEADY",
            "SHUTDOWN",
        ],
    )


    fig.update_layout(
        title="Flight Phase Timeline",
        xaxis_title="Time (seconds)",
        yaxis_title="Phase",
        height=300,
    )


    st.plotly_chart(
        fig,
        width="stretch",
    )


# ============================================================
# DETECTION SUMMARY
# ============================================================

st.subheader("Detection Summary")


if detection_delay is not None:

    st.info(
        f"Anomaly began at "
        f"{start_time:.1f}s and the first confirmed "
        f"AI detection occurred after "
        f"{detection_delay:.1f}s."
    )


col1, col2, col3 = st.columns(3)

with col1:

    if actual_anomalies is not None and actual_anomalies > 0:

        coverage = (
            ai_detections / actual_anomalies * 100
        )

        st.metric(
            "Detection Coverage",
            f"{coverage:.1f}%",
        )

    else:

        st.metric(
            "Detection Coverage",
            "N/A",
        )
with col2:

    critical_count = int(
        (
            data["status"]
            == "CRITICAL"
        ).sum()
    )

    st.metric(
        "Critical Samples",
        critical_count,
    )

with col3:

    precision = None
    recall = None
    f1_score = None
    false_positive_rate = None

    if (
        "anomaly" in data.columns
        and "ai_anomaly" in data.columns
    ):

        actual = data["anomaly"].astype(int)
        predicted = data["ai_anomaly"].astype(int)

        true_positive = int(
            ((actual == 1) & (predicted == 1)).sum()
        )

        false_positive = int(
            ((actual == 0) & (predicted == 1)).sum()
        )

        false_negative = int(
            ((actual == 1) & (predicted == 0)).sum()
        )

        true_negative = int(
            ((actual == 0) & (predicted == 0)).sum()
        )

        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive) > 0
            else 0
        )

        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative) > 0
            else 0
        )

        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        false_positive_rate = (
            false_positive / (false_positive + true_negative)
            if (false_positive + true_negative) > 0
            else 0
        )

col4, col5, col6 = st.columns(3)

with col4:

    st.metric(
        "Precision",
        f"{precision * 100:.1f}%"
        if precision is not None
        else "N/A",
    )

with col5:

    st.metric(
        "Recall",
        f"{recall * 100:.1f}%"
        if recall is not None
        else "N/A",
    )

with col6:

    st.metric(
        "F1 Score",
        f"{f1_score * 100:.1f}%"
        if f1_score is not None
        else "N/A",
    )

if false_positive_rate is not None:

    st.caption(
        f"False Positive Rate: "
        f"{false_positive_rate * 100:.1f}%"
    )


# ============================================================
# V14 INTELLIGENT RISK ASSESSMENT
# ============================================================

st.divider()

st.subheader(
    "Intelligent Risk Assessment"
)
if risk_data is not None:

    # --------------------------------------------------------
    # Peak risk row
    # --------------------------------------------------------

    peak_index = risk_data[
        "overall_risk"
    ].idxmax()

    peak_row = risk_data.loc[
        peak_index
    ]


    overall_risk = float(
        peak_row["overall_risk"]
    )

    primary_sensor = str(peak_row["primary_risk_sensor"]).replace("vibration_g","Vibration").replace("pressure_kpa","Pressure").replace("temperature_k","Temperature").replace("thrust_n","Thrust")

    risk_level = (
        peak_row["risk_level"]
    )

    explanation = (
        peak_row["risk_explanation"]
    )


    # --------------------------------------------------------
    # Main risk metrics
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Overall Risk",
            f"{overall_risk:.1f}/100",
        )


    with col2:

        st.metric(
            "Primary Risk Sensor",
            primary_sensor,
        )


    with col3:

        st.metric(
            "Risk Level",
            risk_level,
        )


    st.info(
        f"**Why?** {explanation}"
    )


    # --------------------------------------------------------
    # Sensor risk
    # --------------------------------------------------------

    st.markdown("### Sensor Risk")


    risk_cols = st.columns(4)


    sensor_risks = [
        (
            "Pressure",
            "pressure_risk",
        ),
        (
            "Temperature",
            "temperature_risk",
        ),
        (
            "Vibration",
            "vibration_risk",
        ),
        (
            "Thrust",
            "thrust_risk",
        ),
    ]


    for column, (
        name,
        risk_column,
    ) in zip(
        risk_cols,
        sensor_risks,
    ):

        with column:

            value = float(
                risk_data[
                    risk_column
                ].max()
            )

            st.metric(
                name,
                f"{value:.1f}/100",
            )


    # --------------------------------------------------------
    # Risk over time
    # --------------------------------------------------------

    st.markdown("### Risk Over Time")


    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=risk_data["time_s"],
            y=risk_data["overall_risk"],
            mode="lines",
            name="Overall Risk",
        )
    )


    # Warning threshold
    fig.add_hline(
        y=45,
        line_dash="dash",
        annotation_text="Warning",
        annotation_position="top left",
    )


    # Critical threshold
    fig.add_hline(
        y=75,
        line_dash="dash",
        annotation_text="Critical",
        annotation_position="top left",
    )


    # Actual anomalies
    if "anomaly" in risk_data.columns:

        anomaly_data = risk_data[
            risk_data["anomaly"] == 1
        ]


        if not anomaly_data.empty:

            fig.add_trace(
                go.Scatter(
                    x=anomaly_data["time_s"],
                    y=anomaly_data["overall_risk"],
                    mode="markers",
                    name="Actual Anomaly",
                    marker=dict(
                        size=5,
                        symbol="circle",
                    ),
                )
            )


    fig.update_layout(
        title="Overall Risk Progression",
        xaxis_title="Time (seconds)",
        yaxis_title="Risk Score",
        yaxis=dict(
            range=[0, 105]
        ),
        height=400,
        hovermode="x unified",
        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20,
        ),
    )


    st.plotly_chart(
        fig,
        width="stretch",
    )


    # --------------------------------------------------------
    # Peak risk details
    # --------------------------------------------------------

    st.markdown("### Peak Risk Event")


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Peak Time",
            f"{peak_row['time_s']:.1f} s",
        )


    with col2:

        st.metric(
            "Primary Sensor",
            primary_sensor,
        )


    with col3:

        elevated = int(
            peak_row[
                "elevated_sensor_count"
            ]
        )

        st.metric(
            "Elevated Sensors",
            elevated,
        )


else:

    st.warning(
        "V14 risk results are not available. "
        "Run risk_engine_v14.py first."
    )

# ============================================================
# CUSTOMER ANALYSIS REPORT
# ============================================================

if customer_mode and risk_data is not None:

    st.divider()

    st.subheader("Customer Analysis Report")

    report_lines = []

    report_lines.append(
        "ROCKET GUARDIAN AI - TELEMETRY ANALYSIS REPORT"
    )
    report_lines.append(
        "=" * 55
    )
    report_lines.append("")

    report_lines.append("MISSION SUMMARY")
    report_lines.append("-" * 30)
    report_lines.append(
        f"Telemetry Samples: {total_samples}"
    )
    report_lines.append(
        f"AI Detections: {ai_detections}"
    )
    report_lines.append(
        f"System Status: {highest_status}"
    )
    report_lines.append("")

    report_lines.append("RISK ASSESSMENT")
    report_lines.append("-" * 30)
    report_lines.append(
        f"Overall Risk: {overall_risk:.1f}/100"
    )
    report_lines.append(
        f"Risk Level: {risk_level}"
    )
    report_lines.append(
        f"Primary Risk Sensor: {primary_sensor}"
    )
    report_lines.append(
        f"Elevated Sensors: {int(peak_row['elevated_sensor_count'])}"
    )
    report_lines.append("")

    report_lines.append("SENSOR RISK")
    report_lines.append("-" * 30)
    report_lines.append(
        f"Pressure: {float(risk_data['pressure_risk'].max()):.1f}/100"
    )
    report_lines.append(
        f"Temperature: {float(risk_data['temperature_risk'].max()):.1f}/100"
    )
    report_lines.append(
        f"Vibration: {float(risk_data['vibration_risk'].max()):.1f}/100"
    )
    report_lines.append(
        f"Thrust: {float(risk_data['thrust_risk'].max()):.1f}/100"
    )
    report_lines.append("")

    report_lines.append("PEAK RISK EVENT")
    report_lines.append("-" * 30)
    report_lines.append(
        f"Peak Time: {float(peak_row['time_s']):.1f} s"
    )
    report_lines.append(
        f"Primary Sensor: {primary_sensor}"
    )
    report_lines.append(
        f"Elevated Sensors: {int(peak_row['elevated_sensor_count'])}"
    )
    report_lines.append("")

    report_lines.append("EXPLANATION")
    report_lines.append("-" * 30)
    report_lines.append(
        str(explanation)
    )
    report_lines.append("")

    report_lines.append(
        "Ground-truth anomaly labels were not provided "
        "with this telemetry file."
    )
    report_lines.append(
        "AI detections are based on the learned "
        "phase-aware telemetry baseline."
    )
    report_lines.append("")

    report_lines.append(
        "Rocket Guardian AI - Research Prototype"
    )
    report_lines.append(
        "Synthetic telemetry / customer-provided telemetry"
    )
    report_lines.append(
        "Not for flight-critical use"
    )

    report_text = "\n".join(
        report_lines
    )
    # --------------------------------------------------------
    # Generate PDF analysis report
    # --------------------------------------------------------

    pdf_buffer = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf",
    )

    pdf_path = pdf_buffer.name

    pdf_buffer.close()

    try:

        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            KeepTogether,
            Table,
            TableStyle,
        )
        from xml.sax.saxutils import escape

        styles = getSampleStyleSheet()

        primary = colors.HexColor("#102A43")
        accent = colors.HexColor("#1F7A8C")
        critical = colors.HexColor("#B42318")
        warning = colors.HexColor("#B54708")
        normal = colors.HexColor("#067647")
        light_bg = colors.HexColor("#F5F8FA")
        border = colors.HexColor("#D0D5DD")
        text_dark = colors.HexColor("#172B4D")
        muted = colors.HexColor("#667085")

        title_style = ParagraphStyle(
            "RGTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=28,
            alignment=TA_LEFT,
            textColor=colors.white,
            spaceAfter=0,
        )

        subtitle_style = ParagraphStyle(
            "RGSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            textColor=colors.white,
        )

        section_style = ParagraphStyle(
            "RGSection",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=primary,
            spaceBefore=8,
            spaceAfter=8,
        )

        body_style = ParagraphStyle(
            "RGBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=text_dark,
        )

        small_style = ParagraphStyle(
            "RGSmall",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=muted,
        )

        metric_label = ParagraphStyle(
            "RGMetricLabel",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=9,
            textColor=muted,
            alignment=TA_CENTER,
        )

        metric_value = ParagraphStyle(
            "RGMetricValue",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=17,
            textColor=primary,
            alignment=TA_CENTER,
        )

        risk_color = {
            "CRITICAL": critical,
            "WARNING": warning,
            "NORMAL": normal,
        }.get(str(risk_level).upper(), primary)

        def safe(value):
            return escape(str(value))

        def metric_card(label, value):
            return [
                Paragraph(safe(label), metric_label),
                Paragraph(safe(value), metric_value),
            ]

        def footer(canvas, document):
            canvas.saveState()
            width, height = A4
            canvas.setStrokeColor(border)
            canvas.line(18 * mm, 15 * mm, width - 18 * mm, 15 * mm)
            canvas.setFont("Helvetica", 7)
            canvas.setFillColor(muted)
            canvas.drawString(18 * mm, 10 * mm, "Rocket Guardian AI - Research Prototype")
            canvas.drawRightString(width - 18 * mm, 10 * mm, f"Page {document.page}")
            canvas.restoreState()

        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=A4,
            rightMargin=18 * mm,
            leftMargin=18 * mm,
            topMargin=18 * mm,
            bottomMargin=22 * mm,
            title="Rocket Guardian AI Telemetry Analysis Report",
            author="Rocket Guardian AI",
        )

        story = []

        header = Table(
            [[
                Paragraph("Rocket Guardian AI", title_style),
                Paragraph("TELEMETRY ANALYSIS REPORT<br/>Customer Mission Assessment", subtitle_style),
            ]],
            colWidths=[105 * mm, 67 * mm],
        )
        header.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), primary),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ]))
        story.append(header)
        story.append(Spacer(1, 10))

        meta = Table([
            [Paragraph("Customer", small_style), Paragraph("Mission", small_style), Paragraph("Account", small_style)],
            [Paragraph(safe(customer_name), body_style), Paragraph(safe(mission_name), body_style), Paragraph(safe(customer_email), body_style)],
        ], colWidths=[57 * mm, 57 * mm, 58 * mm])
        meta.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), light_bg),
            ("BOX", (0, 0), (-1, -1), 0.6, border),
            ("INNERGRID", (0, 0), (-1, -1), 0.35, border),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(meta)
        story.append(Spacer(1, 10))

        kpis = Table([
            metric_card("TELEMETRY SAMPLES", total_samples),
            [
                Paragraph("AI DETECTIONS", metric_label),
                Paragraph("SYSTEM STATUS", metric_label),
                Paragraph("OVERALL RISK", metric_label),
                Paragraph("MISSION HEALTH", metric_label),
            ],
            [
                Paragraph(safe(ai_detections), metric_value),
                Paragraph(safe(highest_status), ParagraphStyle("RGS", parent=metric_value, textColor=risk_color)),
                Paragraph(f"{overall_risk:.1f}/100", ParagraphStyle("RGR", parent=metric_value, textColor=risk_color)),
                Paragraph(f"{max(0.0, 100.0 - overall_risk):.1f}/100", metric_value),
            ],
        ], colWidths=[44 * mm, 44 * mm, 44 * mm, 44 * mm])
        kpis.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), light_bg),
            ("BACKGROUND", (1, 1), (-1, -1), colors.white),
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("BOX", (0, 0), (-1, -1), 0.7, border),
            ("INNERGRID", (0, 1), (-1, -1), 0.35, border),
            ("SPAN", (0, 0), (0, 0)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        story.append(kpis)
        story.append(Spacer(1, 8))

        story.append(Paragraph("Risk Assessment", section_style))
        risk_summary = Table([
            [Paragraph("Risk Level", small_style), Paragraph("Primary Risk Sensor", small_style), Paragraph("Elevated Sensors", small_style), Paragraph("Peak Time", small_style)],
            [Paragraph(safe(risk_level), ParagraphStyle("RGL", parent=body_style, textColor=risk_color, fontName="Helvetica-Bold")), Paragraph(safe(primary_sensor), body_style), Paragraph(safe(int(peak_row["elevated_sensor_count"])), body_style), Paragraph(f"{float(peak_row['time_s']):.1f} s", body_style)],
        ], colWidths=[44 * mm, 50 * mm, 44 * mm, 38 * mm])
        risk_summary.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), light_bg),
            ("BOX", (0, 0), (-1, -1), 0.6, border),
            ("INNERGRID", (0, 0), (-1, -1), 0.35, border),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(risk_summary)

        story.append(Paragraph("Sensor Risk Profile", section_style))
        sensor_rows = [
            [Paragraph("Sensor", small_style), Paragraph("Peak Risk", small_style), Paragraph("Status", small_style)],
        ]
        for sensor_name, risk_column in [
            ("Pressure", "pressure_risk"),
            ("Temperature", "temperature_risk"),
            ("Vibration", "vibration_risk"),
            ("Thrust", "thrust_risk"),
        ]:
            value = float(risk_data[risk_column].max())
            status = "CRITICAL" if value >= 75 else "WARNING" if value >= 45 else "NORMAL"
            status_color = {"CRITICAL": critical, "WARNING": warning, "NORMAL": normal}[status]
            sensor_rows.append([
                Paragraph(sensor_name, body_style),
                Paragraph(f"{value:.1f}/100", body_style),
                Paragraph(status, ParagraphStyle("RGSensor", parent=body_style, textColor=status_color, fontName="Helvetica-Bold")),
            ])
        sensor_table = Table(sensor_rows, colWidths=[70 * mm, 50 * mm, 56 * mm])
        sensor_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), light_bg),
            ("BOX", (0, 0), (-1, -1), 0.6, border),
            ("INNERGRID", (0, 0), (-1, -1), 0.35, border),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(sensor_table)

        story.append(Paragraph("Peak Risk Event", section_style))
        peak_text = (
            f"The highest system risk occurred at <b>{float(peak_row['time_s']):.1f} seconds</b>. "
            f"<b>{safe(primary_sensor)}</b> was the dominant sensor with "
            f"<b>{int(peak_row['elevated_sensor_count'])}</b> elevated sensor(s)."
        )
        story.append(Paragraph(peak_text, body_style))
        story.append(Spacer(1, 6))

        story.append(Paragraph("AI Assessment", section_style))
        explanation_box = Table([[Paragraph(safe(explanation), body_style)]], colWidths=[176 * mm])
        explanation_box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), light_bg),
            ("BOX", (0, 0), (-1, -1), 0.8, accent),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ]))
        story.append(explanation_box)

        story.append(Paragraph("Methodology & Limitations", section_style))
        notes = [
            "AI detections are based on the learned phase-aware telemetry baseline.",
            "Ground-truth anomaly labels were not provided with this telemetry file; classification metrics are therefore not reported.",
            "Risk scores are prototype analytical indicators and are not flight-safety guarantees.",
        ]
        for note in notes:
            story.append(Paragraph("• " + safe(note), body_style))
            story.append(Spacer(1, 2))

        story.append(Spacer(1, 7))
        disclaimer = Table([[Paragraph("NOT FOR FLIGHT-CRITICAL USE", ParagraphStyle("RGDisclaimer", parent=body_style, fontName="Helvetica-Bold", textColor=critical, alignment=TA_CENTER))]], colWidths=[176 * mm])
        disclaimer.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEF3F2")),
            ("BOX", (0, 0), (-1, -1), 0.8, critical),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        story.append(disclaimer)

        doc.build(story, onFirstPage=footer, onLaterPages=footer)

        with open(
            pdf_path,
            "rb",
        ) as pdf_file:

            pdf_data = pdf_file.read()

        st.download_button(
            label="Download PDF Report",
            data=pdf_data,
            file_name="rocket_guardian_analysis_report.pdf",
            mime="application/pdf",
        )

    finally:

        if os.path.exists(pdf_path):

            os.remove(pdf_path)


    # --------------------------------------------------------
    # Download processed AI telemetry
    # --------------------------------------------------------

    processed_csv = risk_data.to_csv(
        index=False
    )

    st.download_button(
        label="Download Processed Telemetry CSV",
        data=processed_csv,
        file_name="rocket_guardian_analysis.csv",
        mime="text/csv",
    )


    st.text_area(
        "Report Preview",
        report_text,
        height=420,
    )

    st.download_button(
        label="Download Analysis Report",
        data=report_text,
        file_name="rocket_guardian_analysis_report.txt",
        mime="text/plain",
    )

# ============================================================
# MISSION HISTORY
# ============================================================

st.subheader("Mission History")

history_customer = get_customer_by_auth_user_id(
    current_user.id
)

if history_customer is None:

    st.info(
        "No customer profile is linked to this account yet."
    )

else:

    history_rows = get_customer_mission_history(
        int(history_customer["id"])
    )

    if not history_rows:

        st.info(
            "No mission history available yet."
        )

    else:

        history_data = pd.DataFrame(
            history_rows
        )

        history_display = history_data[
            [
                "mission_name",
                "source_filename",
                "sample_count",
                "ai_detection_count",
                "overall_risk",
                "risk_level",
                "primary_risk_sensor",
                "peak_time_s",
                "run_created_at",
            ]
        ].copy()

        history_display.columns = [
            "Mission",
            "Telemetry File",
            "Samples",
            "AI Detections",
            "Overall Risk",
            "Risk Level",
            "Primary Risk Sensor",
            "Peak Time (s)",
            "Created At",
        ]

        st.dataframe(
            history_display,
            width="stretch",
            hide_index=True,
        )

        #============================================================
        # RUN-TO-RUN MISSION COMPARISON
        #============================================================

        st.subheader("Run-to-Run Mission Comparison")

        if len(history_data) >= 2:

            comparison_data = history_data[
                [
                    "mission_name",
                    "ai_detection_count",
                    "overall_risk",
                    "risk_level",
                    "primary_risk_sensor",
                ]
            ].copy()

            comparison_data = comparison_data.head(10)

            comparison_display = comparison_data.copy()

            comparison_display.columns = [
                "Mission",
                "AI Detections",
                "Overall Risk",
                "Risk Level",
                "Primary Sensor",
            ]

            st.dataframe(
                comparison_display,
                width="stretch",
                hide_index=True,
            )

            latest = comparison_data.iloc[0]
            previous = comparison_data.iloc[1]

            risk_change = (
                float(latest["overall_risk"])
                - float(previous["overall_risk"])
            )

            detection_change = (
                int(latest["ai_detection_count"])
                - int(previous["ai_detection_count"])
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Risk Change vs Previous",
                    f"{risk_change:+.1f}",
                )

            with col2:

                st.metric(
                    "AI Detection Change",
                    f"{detection_change:+d}",
                )

            if risk_change > 0:

                st.warning(
                    "Latest mission shows increased overall risk "
                    "compared with the previous mission."
                )

            elif risk_change < 0:

                st.success(
                    "Latest mission shows reduced overall risk "
                    "compared with the previous mission."
                )

            else:

                st.info(
                    "Overall risk is unchanged from the previous mission."
                )

        else:

            st.info(
                "Run-to-run comparison will be available "
                "after at least two missions are recorded."
            )

        #============================================================
# MISSION HEALTH SCORE
#============================================================

        st.subheader("Mission Health Score")

        if len(history_data) > 0:

            health_data = history_data.copy()

            health_data["health_score"] = (
                100.0
                - health_data["overall_risk"].astype(float)
            ).clip(0, 100)

            latest_health = float(
                health_data.iloc[0]["health_score"]
            )

            st.metric(
                "Current Mission Health",
                f"{latest_health:.1f}/100",
            )

            if len(health_data) >= 2:

                previous_health = float(
                    health_data.iloc[1]["health_score"]
                )

                health_change = (
                    latest_health
                    - previous_health
                )

                st.metric(
                    "Health Change vs Previous",
                    f"{health_change:+.1f}",
                )

            trend_data = health_data.head(10).copy()
            trend_data = trend_data.iloc[::-1]

            trend_fig = go.Figure()

            trend_fig.add_trace(
                go.Scatter(
                    x=trend_data["mission_name"],
                    y=trend_data["health_score"],
                    mode="lines+markers",
                    name="Mission Health",
                )
            )

            trend_fig.update_layout(
                title="Mission Health Trend",
                xaxis_title="Mission",
                yaxis_title="Health Score",
                yaxis=dict(range=[0, 100]),
                height=320,
            )

            st.plotly_chart(
                trend_fig,
                width="stretch",
            )

        else:

            st.info(
                "Mission health will be available "
                "after a mission is recorded."
            )
        #============================================================
# RAW DATA
# ============================================================

with st.expander(
    "View Raw Telemetry & AI Results"
):

    st.dataframe(
        data,
        width="stretch",
        height=400,
    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Rocket Guardian AI - Research Prototype\n"
"Synthetic telemetry / customer-provided telemetry\n"
"Not for flight-critical use"
)











