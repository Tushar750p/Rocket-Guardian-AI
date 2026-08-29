import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Rocket Guardian AI",
    page_icon="🚀",
    layout="wide",
)


DATA_DIR = Path("data/v11")
RESULT_DIR = Path("data/processed/v11")


SENSORS = {
    "Pressure": {
        "column": "pressure_kpa",
        "unit": "kPa",
        "alert_column": "pressure_kpa_alert",
    },
    "Temperature": {
        "column": "temperature_k",
        "unit": "K",
        "alert_column": "temperature_k_alert",
    },
    "Vibration": {
        "column": "vibration_g",
        "unit": "g",
        "alert_column": "vibration_g_alert",
    },
    "Thrust": {
        "column": "thrust_n",
        "unit": "N",
        "alert_column": "thrust_n_alert",
    },
}


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

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🚀 Rocket Guardian AI</div>',
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

st.sidebar.markdown(
    """
    **Monitoring System**

    🟢 Normal  
    🟡 Warning  
    🔴 Critical
    """
)


file_path = RESULT_DIR / SCENARIOS[selected]


if not file_path.exists():

    st.error(
        f"Result file not found:\n{file_path}"
    )

    st.stop()


data = pd.read_csv(file_path)


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
        "🔴 CRITICAL — Significant telemetry anomaly detected"
    )

elif highest_status == "WARNING":

    st.warning(
        "🟡 WARNING — Potential telemetry anomaly detected"
    )

else:

    st.success(
        "🟢 NORMAL — Telemetry within expected operating envelope"
    )


# ============================================================
# CORE METRICS
# ============================================================

actual_anomalies = int(
    data["anomaly"].sum()
)

ai_detections = int(
    data["ai_anomaly"].sum()
)

total_samples = len(data)

anomaly_start = data.loc[
    data["anomaly"] == 1,
    "time_s",
]

detection_times = data.loc[
    data["ai_anomaly"] == 1,
    "time_s",
]


detection_delay = None

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
# SENSOR STATUS
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
            data[alert_column].any()
        )

    else:

        alert_active = False


    with ui_column:

        if alert_active:

            st.markdown(
                f"""
                <div class="status-card">
                    <div class="sensor-name">
                        {name}
                    </div>
                    <div class="sensor-status critical">
                        🔴 ALERT
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                f"""
                <div class="status-card">
                    <div class="sensor-name">
                        {name}
                    </div>
                    <div class="sensor-status normal">
                        🟢 NORMAL
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


st.divider()


# ============================================================
# TELEMETRY CHART FUNCTION
# ============================================================

def create_chart(
    data,
    sensor_name,
    config,
):

    column = config["column"]
    unit = config["unit"]
    alert_column = config["alert_column"]

    fig = go.Figure()


    # Main telemetry line
    fig.add_trace(
        go.Scatter(
            x=data["time_s"],
            y=data[column],
            mode="lines",
            name=sensor_name,
        )
    )


    # Actual anomaly points
    actual = data[
        data["anomaly"] == 1
    ]

    if not actual.empty:

        fig.add_trace(
            go.Scatter(
                x=actual["time_s"],
                y=actual[column],
                mode="markers",
                name="Actual Anomaly",
                marker=dict(
                    size=5,
                ),
            )
        )


    # AI alert points
    if alert_column in data.columns:

        ai_alerts = data[
            data[alert_column] == True
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


    # Anomaly start line
    if not actual.empty:

        anomaly_start = actual[
            "time_s"
        ].iloc[0]

        fig.add_vline(
            x=anomaly_start,
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
# TELEMETRY GRAPHS
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
# PHASE TIMELINE
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

    st.metric(
        "Detection Coverage",
        (
            f"{ai_detections / actual_anomalies * 100:.1f}%"
            if actual_anomalies > 0
            else "N/A"
        ),
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


st.caption(
    "Rocket Guardian AI — Research Prototype | "
    "Synthetic telemetry data | Not for flight-critical use"
)