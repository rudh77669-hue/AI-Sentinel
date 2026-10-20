# AI Sentinel: AI-Powered Cybersecurity Threat Detection & Investigation Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-black.svg)](https://flask.palletsprojects.com/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Model Accuracy](https://img.shields.io/badge/Accuracy-99.70%25-brightgreen.svg)]()
[![Status](https://img.shields.io/badge/Platform-Production%20Ready-success.svg)]()

**AI Sentinel** is an enterprise-grade Network Intrusion Detection System (NIDS) and Security Operations Center (SOC) investigation platform. It combines machine learning inference on 78 bidirectional network flow features with an analyst-centric dashboard for live incident triage, alert management, and real-time threat simulation.

---

## Architecture Overview

```
                      +---------------------------------------+
                      |       Raw Network Traffic / PCAP       |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |      Flow Extractor (Scapy / CIC)      |
                      |   (78 Bidirectional Flow Metrics)     |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |     AI Sentinel Detection Engine      |
                      |    (RandomForest + SimpleImputer)     |
                      +---------------------------------------+
                                     /         \
                      (Benign Traffic)         (Malicious Attack)
                            |                          |
                            v                          v
                      [Passed Safely]          +---------------+
                                               | Alert Manager |
                                               +---------------+
                                                       |
                                                       v
                                            +---------------------+
                                            | SQLite Audit Store  |
                                            | (data/ai_sentinel)  |
                                            +---------------------+
                                                       |
                                                       v
                                            +---------------------+
                                            | SOC Web Dashboard   |
                                            | (Flask + Real-time) |
                                            +---------------------+
```

---

## Key Features

1. **High-Accuracy ML Detection Engine**
   - Trained on the authoritative **CIC-IDS2017** benchmark dataset.
   - Evaluates **78 bidirectional network flow features** (inter-arrival times, packet length variance, TCP header flags, subflow metrics, bulk rates).
   - Achieves **99.70% test accuracy**, **99.67% precision**, and **99.52% recall** across 14 major attack families.

2. **Interactive Threat Simulation Sandbox**
   - Built-in simulation feed to test and demonstrate detection in real time without needing an external cyber range.
   - One-click testing for **DDoS**, **DoS Hulk**, **PortScan**, **Web Attacks (Brute Force / XSS / SQLi)**, and **Benign traffic**.

3. **SOC Investigation & Incident Triage**
   - Full incident lifecycle tracking: `New` &rarr; `Investigating` &rarr; `Resolved`.
   - Analyst investigation notes and severity overrides.
   - SQLite persistent storage (`data/ai_sentinel.db`).

4. **Live Network Packet Sniffing**
   - Scapy-based network sniffer (`src/flow_extractor.py`) reconstructing bidirectional IP flows from local network interfaces.
   - CICFlowMeter adapter module for real-time feature alignment.

---

## Model Evaluation Metrics

Evaluated on a stratified holdout test set (12,172 network flows across all attack families):

| Metric | Result |
| :--- | :--- |
| **Overall Accuracy** | **99.70%** |
| **Attack Precision** | **99.67%** |
| **Attack Recall** | **99.52%** |
| **Attack F1-Score** | **99.60%** |
| **False-Positive Rate on Normal Traffic** | **0.20%** (15 / 7,600) |

### Detection Rate by Threat Category

| Threat Family | Category | Detection Rate |
| :--- | :--- | :--- |
| **DDoS** | High-Rate Denial of Service | **99.87%** |
| **DoS Hulk** | HTTP Flood DoS | **99.10%** |
| **DoS GoldenEye** | Keep-Alive HTTP DoS | **99.74%** |
| **DoS slowloris** | Low-Rate Starvation DoS | **100.00%** |
| **DoS Slowhttptest** | Slow Request Flood | **100.00%** |
| **PortScan** | Network Reconnaissance | **100.00%** |
| **FTP-Patator** | Credential Brute Force | **100.00%** |
| **SSH-Patator** | SSH Credential Brute Force | **99.71%** |
| **Web Attack - Brute Force** | Web App Password Spraying | **99.29%** |
| **Web Attack - SQLi & XSS** | Web Vulnerability Exploitation | **99.40%** |
| **Botnet Traffic** | Command & Control | **98.07%** |

---

## Directory Structure

```
AI-Sentinel/
├── app.py                      # Flask web application & REST API
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── data/
│   ├── ai_sentinel.db          # SQLite incident database
│   ├── sample_flows.json       # Representative real flows for live simulation
│   └── raw/MachineLearningCVE/ # CIC-IDS2017 raw CSV captures
├── models/
│   └── classifier.joblib       # Serialized Random Forest + Imputer bundle
├── reports/
│   ├── evaluation.json         # Comprehensive evaluation metrics payload
│   └── model_evaluation.json   # Benchmark report
├── src/
│   ├── alert_manager.py        # Security alert creation & severity validation
│   ├── cicflowmeter_adapter.py # Adapter for CICFlowMeter live CSV schema
│   ├── database.py             # SQLite data access & triage operations
│   ├── detection_service.py    # Pipeline orchestrator between ML & alerts
│   ├── evaluate_model.py       # Standalone evaluation benchmark script
│   ├── flow_extractor.py       # Scapy live bidirectional packet sniffer
│   ├── live_monitor.py         # Network traffic monitoring service
│   ├── predict.py              # ML inference service
│   └── train_model.py          # Balanced model training script
├── templates/
│   └── index.html              # Modern dark-theme SOC investigation dashboard
└── tests/
    ├── test_alert_manager.py    # Unit tests for alert creation
    ├── test_api.py              # Unit tests for REST API & simulation
    └── test_detection_service.py# Unit tests for detection pipeline
```

---

## Installation & Quickstart

### 1. Prerequisites
- Python 3.10+ (Python 3.11, 3.12, 3.13 supported)
- Virtual environment recommended

### 2. Setup Environment
```bash
# Clone or navigate to the project directory
cd AI-Sentinel

# Activate your virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install required packages
pip install -r requirements.txt
```

### 3. Launch the SOC Dashboard
```bash
python app.py
```
Open your web browser and navigate to:
```
http://127.0.0.1:5000/
```

---

## How to Test and Demonstrate

1. Open the dashboard at `http://127.0.0.1:5000/`.
2. Locate the **Real-Time Threat Detection & Simulation Sandbox** bar.
3. Click any attack simulation button (e.g. **🔴 Test DDoS**, **🟠 Test DoS Hulk**, **🟡 Test PortScan**, or **🟢 Test Normal Flow**).
4. Watch the detection engine execute live inference in milliseconds:
   - A notification banner appears with the classification and confidence percentage.
   - For malicious flows, a new alert is generated with timestamp, source IP, target IP, and severity.
   - Metric cards update dynamically.
5. Click **🔎 Investigate** on any alert to open the Analyst Investigation modal:
   - Change status (`New` &rarr; `Investigating` &rarr; `Resolved`).
   - Add notes (e.g. *"Blocked IP on edge gateway"*).
   - Save updates directly to the database.

---

## Running Unit Tests

Run the test suite with Python's built-in `unittest`:
```bash
python -m unittest discover tests
```
All tests should execute and report `OK`.

---

## Re-training the Model

To re-train the model across all raw capture files:
```bash
python -m src.train_model
```
This will:
1. Load samples from all 8 CIC-IDS2017 files.
2. Fit the `SimpleImputer` and `RandomForestClassifier`.
3. Save the model bundle to `models/classifier.joblib`.
4. Extract test flows to `data/sample_flows.json`.
5. Output updated metrics to `reports/evaluation.json`.

---

## REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Web investigation dashboard UI |
| `GET` | `/api/status` | System health, model readiness, and monitoring state |
| `GET` | `/api/alerts` | Retrieve incident alert records (`?limit=50`) |
| `GET` | `/api/alert-stats` | Get count totals (Total, Critical, Unresolved) |
| `POST` | `/api/alerts/<id>` | Update alert status (`New`, `Investigating`, `Resolved`) and notes |
| `GET` | `/api/sample-threats` | List available threat categories for testing |
| `POST` | `/api/simulate-threat` | Inject a simulated attack flow into the live ML pipeline |
| `POST` | `/api/clear-alerts` | Reset and clear incident alerts for a fresh demo |
| `POST` | `/api/predict` | Predict classification for custom flow features |

---

## License & Attribution

AI Sentinel is developed for cybersecurity threat detection research and education. Dataset features based on the University of New Brunswick (UNB) CIC-IDS2017 benchmark.

