import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import os
import tempfile

from src.telemetry_analysis import analyze_telemetry
from src.risk_analysis import analyze_risk, summarize_risk


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Rocket Guardian AI",
    page_icon="[ROCKET]",
    layout="wide",
)


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
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 17px;
        opacity: 0.7;
        margin-bottom: 25px;
    }

    .status-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        text-align: center;
        margin-bottom: 15px;
    }

    .sensor-name {
        font-size: 15px;
        font-weight: 600;
    }

    .sensor-status {
        font-size: 22px;
        font-weight: 700;
        margin-top: 6px;
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
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 15px;
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
    "Phase-Aware Rocket Telemetry Monitoring Prototype"
    "</div>",
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Mission Controls")

selected = st.sidebar.selectbox(
    "Test Scenario",
    list(SCENARIOS.keys()),
)

st.sidebar.divider()

st.sidebar.subheader("Telemetry Input")

uploaded_file = st.sidebar.file_uploader(
    "Upload Telemetry CSV",
    type=["csv"],
    help=(
        "Upload a telemetry CSV containing time_s, phase, "
        "pressure_kpa, temperature_k, vibration_g, and thrust_n."
    ),
)

st.sidebar.markdown(
    """
    **Monitoring System**

    NORMAL — Normal  
    WARNING — Warning  
    CRITICAL — Critical
    """
)

# ============================================================
# LOAD TELEMETRY DATA
# ============================================================

customer_mode = uploaded_file is not None

if customer_mode:

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

        fd, temp_path = tempfile.mkstemp(
            suffix=".csv"
        )

        try:

            with os.fdopen(fd, "wb") as temp_file:

                temp_file.write(
                    uploaded_bytes
                )

            data = analyze_telemetry(
                temp_path
            )

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

    data = pd.read_csv(
        file_path
    )


# ============================================================
# LOAD V14 RISK DATA
# ============================================================

if customer_mode:

    try:

        risk_data = analyze_risk(
            data
        )

    except ValueError as exc:

        st.error(
            f"Risk analysis failed: {exc}"
        )

        st.stop()

else:

    risk_file = (
        RISK_DIR
        / RISK_FILES[selected]
    )

    if risk_file.exists():

        risk_data = pd.read_csv(
            risk_file
        )

    else:

        risk_data = None


# ============================================================
# OVERALL SYSTEM STATUS
# ============================================================

severity = {
    "NORMAL": 0,
    "WARNING": 1,
    "CRITICAL": 2,
}


highest_status = max(
    data["status"],
    key=lambda x: severity.get(x, 0),
)


if highest_status == "CRITICAL":

    st.error(
        "[CRITICAL] CRITICAL - Significant telemetry anomaly detected"
    )

elif highest_status == "WARNING":

    st.warning(
        "[WARNING] WARNING - Potential telemetry anomaly detected"
    )

else:

    st.success(
        "[NORMAL] NORMAL - Telemetry within expected operating envelope"
    )


# ============================================================
# CORE METRICS
# ============================================================

customer_mode = "anomaly" not in data.columns

ai_detections = int(
    data["ai_anomaly"].sum()
)

total_samples = len(data)

actual_anomalies = None
anomaly_start = pd.Series(dtype=float)
detection_delay = None
start_time = None

ai_detections = int(
    data["ai_anomaly"].sum()
)

total_samples = len(data)

actual_anomalies = None
anomaly_start = pd.Series(dtype=float)
detection_delay = None
start_time = None

if not customer_mode:

    actual_anomalies = int(
        data["anomaly"].sum()
    )

    anomaly_start = data.loc[
        data["anomaly"] == 1,
        "time_s",
    ]

    detection_times = data.loc[
        data["ai_anomaly"] == 1,
        "time_s",
    ]

    if not anomaly_start.empty:

        start_time = anomaly_start.iloc[0]

        valid_detection = detection_times[
            detection_times >= start_time
        ]

        if not valid_detection.empty:

            detection_delay = (
                valid_detection.iloc[0]
                - start_time
            )


col1, col2, col3, col4 = st.columns(4)


with col1:

    if customer_mode:

        st.metric(
            "Actual Anomalies",
            "N/A",
        )

    else:

        st.metric(
            "Actual Anomalies",
            actual_anomalies,
        )


with col2:

    st.metric(
        "AI Detections",
        ai_detections,
    )


with col3:

    if detection_delay is None:

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

    if alert_column in data.columns:
        alert_active = bool(
            data[alert_column].astype(bool).any()
        )
    else:
        alert_active = False

    with ui_column:

        if alert_active:

            st.markdown(
                f"""
                <div class="status-card">
                    <div class="sensor-name">{name}</div>
                    <div class="sensor-status critical">
                        ALERT
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                f"""
                <div class="status-card">
                    <div class="sensor-name">{name}</div>
                    <div class="sensor-status normal">
                        [NORMAL] NORMAL
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
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

    # Actual anomaly — available only in research/test data
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

    # Anomaly start — research/test data only
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


col1, col2 = st.columns(2)


with col1:

    if customer_mode:

        st.metric(
            "Detection Coverage",
            "N/A",
        )

    else:

        coverage = (
            ai_detections / actual_anomalies * 100
            if actual_anomalies > 0
            else 0
        )

        st.metric(
            "Detection Coverage",
            f"{coverage:.1f}%",
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

    primary_sensor = (
        peak_row["primary_risk_sensor"]
    )

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
    "Rocket Guardian AI - Research Prototype | "
    "Synthetic telemetry data | Not for flight-critical use"
)
