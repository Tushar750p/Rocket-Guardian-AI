# 🚀 Rocket Guardian AI

### Phase-Aware Rocket Telemetry Monitoring & Anomaly Detection

Rocket Guardian AI is a research prototype for monitoring rocket telemetry and detecting abnormal sensor behavior using synthetic telemetry data.

The system combines **phase-aware anomaly detection**, **early-warning detection**, **intelligent risk assessment**, and an interactive **Streamlit monitoring dashboard**.

> ⚠️ **Research Prototype:** This project uses synthetic telemetry data and is not intended for flight-critical or operational aerospace use.

---

## 🎯 Project Overview

Rocket systems generate large amounts of telemetry from sensors such as:

- Pressure
- Temperature
- Vibration
- Thrust

A monitoring system needs to identify abnormal behavior quickly while reducing unnecessary alerts.

Rocket Guardian AI explores a telemetry monitoring pipeline that:

1. Generates synthetic rocket telemetry
2. Models normal and anomalous operating conditions
3. Detects telemetry anomalies
4. Accounts for rocket flight phases
5. Measures detection performance
6. Estimates system-level risk
7. Identifies the primary contributing sensor
8. Presents results through an interactive dashboard

---

## 🎯 Problem Statement

Rocket telemetry contains multiple sensor signals that can change significantly during different stages of flight.

A fixed threshold may incorrectly classify normal flight-phase changes as anomalies.

The goal of Rocket Guardian AI is to explore a more intelligent monitoring approach that considers:

- Sensor behavior
- Flight phase
- Anomaly persistence
- Detection timing
- Multiple sensor conditions
- System-level risk

---

# 🧠 Key Features

- 🚀 Synthetic rocket telemetry simulation
- 📡 Multi-sensor telemetry monitoring
- 🧠 Phase-aware anomaly detection
- ⚡ Early-warning detection
- 📊 Precision / Recall / F1 evaluation
- ⏱️ Detection-delay measurement
- 🚨 False-alarm evaluation
- 🧠 Intelligent risk assessment
- 🎯 Primary sensor identification
- 📈 Risk-over-time analysis
- 🖥️ Interactive Streamlit dashboard
- 📊 Plotly telemetry visualizations

---

# 🏗️ System Architecture

```text
                    Rocket Telemetry
                          │
                          ▼
              ┌───────────────────────┐
              │ Synthetic Telemetry   │
              │ Data Generation       │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ Flight-Phase Modeling │
              │ & Normal Baseline      │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ Anomaly Detection     │
              │                       │
              │ Pressure              │
              │ Temperature           │
              │ Vibration             │
              │ Thrust                │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ Early Warning Logic   │
              │ Persistence / Timing  │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ Intelligent Risk      │
              │ Assessment             │
              │ V14 Risk Engine       │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ Streamlit Dashboard   │
              │                       │
              │ • Sensor Health       │
              │ • Telemetry           │
              │ • Alerts              │
              │ • Flight Phase        │
              │ • Risk Assessment     │
              └───────────────────────┘
````

---

# 🔬 Development History

The project was developed through multiple iterations to evaluate different anomaly-detection and risk-assessment approaches.

## V5 — Normal-Only Training

V5 introduced training using normal telemetry data only.

The model was evaluated against:

* Pressure
* Temperature
* Vibration
* Thrust
* Combined anomalies

This established the foundation for normal-behavior-based anomaly detection.

---

## V6–V8 — Early Warning

The next iterations focused on detecting anomalies earlier.

The system was evaluated using:

* Precision
* Recall
* F1 Score
* Detection Delay
* False Alarms per Minute

This demonstrated the trade-off between early detection and excessive alerts.

---

## V9 — Threshold Optimization

V9 tested different anomaly-detection configurations using:

* Z-score threshold
* Persistence duration
* Precision
* Recall
* F1 Score
* False alarms per minute

The selected configuration was:

```text
Z threshold : 2.0
Persistence : 30 samples
Mean F1     : 0.723
Mean Recall : 0.736
```

---

# 🛰️ V11 — Phase-Aware Anomaly Detection

V11 introduced multiple normal training datasets and validation datasets.

### Training datasets

```text
training_normal_1
training_normal_2
training_normal_3
training_normal_4
training_normal_5
```

### Validation datasets

```text
validation_normal_1
validation_normal_2
```

### Test scenarios

```text
test_pressure
test_temperature
test_vibration
test_thrust
test_combined
```

Each test scenario contains:

```text
1200 samples
500 anomalous samples
```

## V11 Results

| Scenario    | Precision | Recall | F1 Score | Detection Delay |
| ----------- | --------: | -----: | -------: | --------------: |
| Combined    |     1.000 |  0.802 |    0.890 |           9.9 s |
| Pressure    |     1.000 |  0.276 |    0.433 |          16.2 s |
| Temperature |     1.000 |  0.418 |    0.590 |           9.1 s |
| Thrust      |     1.000 |  0.352 |    0.521 |          12.4 s |
| Vibration   |     1.000 |  0.822 |    0.902 |           8.9 s |

Validation normal datasets produced:

```text
validation_normal_1
False Positive Rate: 0.000

validation_normal_2
False Positive Rate: 0.000
```

---

# 🧠 V14 — Intelligent Risk Engine

V14 extends anomaly detection into a system-level risk assessment layer.

Instead of only asking:

> Is the telemetry anomalous?

the risk engine attempts to answer:

> How severe is the current telemetry condition, and which sensor is the primary contributor?

The V14 risk engine provides:

* Overall risk score
* Risk level
* Primary risk sensor
* Sensor-specific risk
* Peak risk time
* Elevated sensor count
* Human-readable explanation

## V14 Results

| Scenario    |   Peak Risk | Risk Level | Primary Sensor |
| ----------- | ----------: | ---------- | -------------- |
| Combined    | 99.58 / 100 | CRITICAL   | Vibration      |
| Pressure    | 79.42 / 100 | CRITICAL   | Pressure       |
| Temperature | 82.03 / 100 | CRITICAL   | Temperature    |
| Thrust      | 80.49 / 100 | CRITICAL   | Thrust         |
| Vibration   | 81.19 / 100 | CRITICAL   | Vibration      |

### Example

```text
Overall Risk        : 99.6 / 100
Risk Level          : CRITICAL
Primary Risk Sensor : Vibration
Peak Time           : 99.8 s
```

> The risk score is a prototype telemetry risk index and should not be interpreted as a probability of failure.

---

# 📊 Interactive Dashboard

Rocket Guardian AI includes an interactive Streamlit monitoring dashboard.

## Mission Controls

The dashboard provides test-scenario selection and monitoring controls.

## Monitoring System

The dashboard displays:

```text
🟢 Normal
🟡 Warning
🔴 Critical
```

## Sensor Health

Monitoring cards for:

* Pressure
* Temperature
* Vibration
* Thrust

## Rocket Telemetry

Interactive telemetry visualizations for the major rocket sensors.

## AI Alert Timeline

Shows the timing of detected anomaly conditions.

## Rocket Flight Phase

Displays telemetry relative to the simulated rocket flight phase.

## Detection Summary

Displays:

* Actual anomalies
* AI detections
* Detection delay
* Detection coverage
* Critical samples

## Intelligent Risk Assessment

Displays:

* Overall risk
* Primary risk sensor
* Risk level
* Explanation
* Sensor-specific risk
* Risk over time
* Peak risk event

---

# 🛠️ Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Plotly
* Git
* GitHub

---

# 📁 Project Structure

```text
Rocket-Guardian-AI/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── v5/
│   ├── v10/
│   └── v11/
│
├── src/
│   ├── anomaly_detector.py
│   ├── anomaly_detector_v2.py
│   ├── anomaly_detector_v3.py
│   ├── anomaly_detector_v4.py
│   ├── anomaly_detector_v5.py
│   │
│   ├── early_warning_v6.py
│   ├── early_warning_v7.py
│   ├── early_warning_v8.py
│   ├── v9_threshold_optimizer.py
│   ├── v10_detector.py
│   ├── phase_aware_v11.py
│   │
│   ├── risk_engine_v14.py
│   ├── dashboard.py
│   │
│   └── simulation/
│       ├── telemetry_simulator.py
│       ├── advanced_telemetry_simulator.py
│       ├── plot_telemetry.py
│       ├── v5_dataset.py
│       ├── v10_dataset.py
│       └── v11_dataset.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/Tushar750p/Rocket-Guardian-AI.git
cd Rocket-Guardian-AI
```

Create a virtual environment:

### Windows

```powershell
python -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell execution policy prevents activation, use the virtual environment's Python directly:

```powershell
.\.venv\Scripts\python.exe
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

---

# ▶️ Running the Project

## Run V11 Phase-Aware Detector

```powershell
.\.venv\Scripts\python.exe src\phase_aware_v11.py
```

## Run V14 Risk Engine

```powershell
.\.venv\Scripts\python.exe src\risk_engine_v14.py
```

## Run Dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run src\dashboard.py
```

The Streamlit dashboard will be available locally.

---

# 📈 Evaluation Metrics

The project uses several evaluation metrics.

### Precision

Measures how many detected anomalies were actually anomalous.

### Recall

Measures how many actual anomalies were detected.

### F1 Score

Balances precision and recall.

### Detection Delay

Measures how long the system takes to confirm an anomaly after the anomaly begins.

### False Positive Rate

Measures incorrect anomaly detections during normal conditions.

### False Alarms per Minute

Measures the frequency of unnecessary alerts during monitoring.

---

# 🧪 Synthetic Telemetry Dataset

This project uses simulated telemetry data rather than real rocket-flight data.

The simulator generates:

* Normal telemetry
* Pressure anomalies
* Temperature anomalies
* Vibration anomalies
* Thrust anomalies
* Combined anomalies

Synthetic data enables controlled experiments while avoiding dependence on operational aerospace telemetry.

---

# ⚠️ Limitations

This project is a research and portfolio prototype.

Important limitations include:

* Telemetry data is synthetic.
* The simulated flight environment is simplified.
* The anomaly detector has not been validated against real flight data.
* Risk scores are not probabilities of failure.
* The system is not aerospace certified.
* Dashboard alerts are not flight-control commands.
* Results should not be used for operational flight decisions.

> **This system must not be used for flight-critical decision making.**

---

# 🔮 Future Improvements

Potential future improvements include:

* Real aerospace telemetry datasets
* More sophisticated flight-phase modeling
* Adaptive phase-specific baselines
* Sensor correlation analysis
* Sensor fault isolation
* Sequence-based anomaly detection
* Predictive remaining-useful-life estimation
* Explainable AI
* Alert prioritization
* Historical mission comparison
* Real-time telemetry streaming
* Automated experiment tracking
* Production monitoring

---

# 🎓 Project Goal

Rocket Guardian AI demonstrates how telemetry monitoring can evolve from basic anomaly detection into an intelligent monitoring pipeline.

The overall workflow is:

```text
Synthetic Data Generation
          ↓
Normal Behaviour Modeling
          ↓
Phase-Aware Detection
          ↓
Early Warning
          ↓
Risk Assessment
          ↓
Sensor Attribution
          ↓
Interactive Dashboard
```

The goal is to provide a practical research prototype for exploring **AI-assisted rocket telemetry monitoring**.

---

# 📌 Current Status

### Stable Prototype: V14

Current stable components:

* Phase-aware anomaly detection
* Intelligent risk assessment
* Sensor-level risk analysis
* Synthetic telemetry simulation
* Interactive Streamlit dashboard

Experimental model versions were developed during the research process and are not part of the current stable dashboard.

---

# ⚖️ Disclaimer

Rocket Guardian AI is an experimental research prototype.

It is **not an aerospace-certified system** and must not be used for real-world flight-critical monitoring, control, safety decisions, or mission operations.

---

# 👨‍💻 Author

 Tushar Patil

GitHub Repository:

[https://github.com/Tushar750p/Rocket-Guardian-AI](https://github.com/Tushar750p/Rocket-Guardian-AI)


