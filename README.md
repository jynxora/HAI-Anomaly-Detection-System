# HAI-Based ICS Anomaly Detection System

## 1. Project Overview
This project is an advanced, ensemble-based anomaly detection pipeline engineered for **Industrial Control Systems (ICS)**. It is built upon the **HAI (Hidden Anomaly Indicator)** dataset, which simulates a complex industrial environment with 80+ real-time IoT sensors across multiple process control loops.

Modern ICS environments generate high-volume, high-frequency telemetry. Detecting threats in these systems requires more than simple thresholds; it requires a system capable of:
* **Temporal Analysis:** Understanding how sensor values change over time.
* **Cross-Sensor Correlation:** Identifying when one subsystem affects another.
* **Ensemble Intelligence:** Combining statistical outliers with deep-learning reconstructions.

**Pipeline Flow:**
`Raw Sensor Data` → `Feature Engineering` → `Ensemble Models` → `Explainability (SHAP)` → `LLM Advisory`

---

## 2. Technical Architecture
The system employs a multi-stage detection engine to ensure robustness against "low-and-slow" cyber-physical attacks.

### Feature Engineering
Raw telemetry is expanded into **221 engineered features**, including:
* **Rate of Change (RoC):** Capturing sudden velocity shifts in sensor data.
* **Rolling Statistics:** Mean, Std Dev, and Max values over 10s, 30s, and 60s windows.
* **Control Loop Errors:** Measuring deviations between Setpoints (SP) and Process Variables (PV).

### The Ensemble Engine
1.  **Isolation Forest (Statistical):** Efficiently isolates global outliers and point anomalies using tree-based partitioning.
2.  **LSTM Autoencoder (Deep Learning):** Learns the "normal" temporal sequences of the plant; reconstruction errors indicate sequential anomalies.
3.  **Transformer Autoencoder (Deep Learning):** Utilizes self-attention mechanisms to capture long-range dependencies across the 80+ IoT sensors.

### Explainability & Advisory
* **SHAP (Shapley Additive Explanations):** Deconstructs the model's decision to show exactly which sensors contributed to the alarm.
* **AI Advisory:** Technical scores are processed by an LLM (Groq/Anthropic) to provide human-readable guidance for operators.

---

## 3. Visual Proof of Concept (PoC)

### Anomaly Timeline & Correlation
*The dashboard provides a real-time view of anomaly clusters across different control loops.*
![Anomaly Timeline](https://github.com/jynxora/HAI-Anomaly-Detection-System/blob/main/screenshots/Screenshot%202026-04-14%20184955.png)
![Anomaly Timeline](https://github.com/jynxora/HAI-Anomaly-Detection-System/blob/main/screenshots/Screenshot%202026-04-14%20185007.png)

### Sensor Contribution (SHAP)
*SHAP values identify the "Root Cause" sensors during a detected attack.*
![Anomaly Timeline](https://github.com/jynxora/HAI-Anomaly-Detection-System/blob/main/screenshots/Screenshot%202026-04-14%20192929.png)
![Anomaly Timeline](https://github.com/jynxora/HAI-Anomaly-Detection-System/blob/main/screenshots/Screenshot%202026-04-14%20192857.png)

---

## 4. System Requirements
* **OS:** Linux (Ubuntu/Kali recommended), Windows 10/11, or macOS.
* **Python Version:** 3.9+
* **CPU:** Quad-core processor (optimized for CPU inference).
* **RAM:** 8GB Minimum (16GB recommended for large datasets).
* **Disk Space:** 500MB for weights and environment.

---

## 5. Project Structure
Ensure your directory looks like this so the backend can locate the models:

```text
project_root/
├── app.py                  # Flask dashboard backend
├── main.py                 # FastAPI detection engine
├── .env                    # API keys (Groq/Anthropic)
├── requirements.txt        # Project dependencies
├── feature_cols.json       # Feature definitions
├── hai_models/             # Directory for trained weights
│   ├── lstm_ae_full.pt     # PyTorch LSTM weights
│   ├── transformer_ae_full.pt # PyTorch Transformer weights
│   ├── iso_forest.pkl      # Isolation Forest model
│   ├── scalers.pkl         # Fitted data scalers
│   └── metadata.json       # Hyperparameter metadata
└── dashboard.html          # UI Visualization
```
## 6. Installation & Setup

Step 1: Clone and Prepare Environment
```
git clone https://github.com/jynxora/HAI-Anomaly-Detection-System/tree/main/HAI-Dashboard
cd HAI-Dashboard
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

Step 2: Install Dependencies
```
pip install -r requirements.txt
```

Step 3: Launching the System
You can run either the core engine or the full dashboard:

To run the FastAPI Engine:
```
uvicorn main:app --reload
```
To run the Flask Dashboard:
```
python app.py
```
7. How to Use
Health Check: Visit `http://127.0.0.1:5000/health` to confirm models are loaded.

Upload Data: Send a POST request with a HAI dataset CSV to the /upload endpoint.

Analyze: View the /anomalies list to see severity rankings (Low to Critical).

Explain: Use the /explain endpoint to generate an AI advisory for a specific time window.

Note on Model Weights: The system is designed to auto-detect architecture shapes from the files in hai_models/. Ensure all file names match the structure in Section 5.
