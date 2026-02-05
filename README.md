# 🛡️ AeroGuardian


**Automated Pre-Flight UAV Safety Analysis System**

Transform FAA UAS sighting reports into actionable pre-flight safety recommendations through automated simulation, deterministic physics-based analysis, and LLM-driven scenario translation.

**Author:** AeroGuardian Team (Tiny Coders)  
**Version:** 1.0  
**Date:** 2026-02-06

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PX4](https://img.shields.io/badge/PX4-v1.14.3-orange.svg)](https://px4.io/)
[![Gazebo](https://img.shields.io/badge/Gazebo-Harmonic-green.svg)](https://gazebosim.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Data Flow: Phase-by-Phase I/O](#-data-flow-phase-by-phase-inputoutput-specification)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Web UI](#-web-ui)
- [Command Reference](#-command-reference)
- [Testing and Validation](#-testing-and-validation)
- [Evaluation Framework (ESRI)](#-evaluation-framework-esri)
- [Simulation Approach](#-simulation-approach)
- [Why Our Analysis is Grounded](#-why-our-analysis-is-grounded)
- [Project Structure](#-project-structure)
- [Output Structure](#-output-structure)
- [Troubleshooting](#-troubleshooting)
- [Regulatory References](#-regulatory-references)
- [Limitations & Scope](#-limitations--scope)

---

## 🎯 Overview

AeroGuardian is an **automated pre-flight safety analysis system** that transforms FAA UAS sighting reports into testable simulation scenarios for proactive hazard identification.

> ⚠️ **Important:** FAA UAS sighting reports are observational records, not accident investigations. This system generates safety hypotheses based on limited data and should not be used as the sole basis for airworthiness decisions. See [Limitations & Scope](#-limitations--scope) for details.

The system:

1. **Ingests real FAA sighting reports** (8,031 testable UAS sightings from 2019-2025)
2. **Translates to simulation** using LLM-driven parameter extraction (GPT-4o + DSPy)
3. **Runs PX4 SITL simulation** with native PX4 fault injection (parameter-based and shell command)
4. **Captures full telemetry** at 10-50 Hz sampling rate
5. **Analyzes telemetry** with deterministic, physics-based anomaly detection (no LLM)
6. **Generates structured safety reports** (JSON + PDF) with Go/Caution/No-Go recommendations
7. **Evaluates scenario trustworthiness** using the ESRI framework (SFS × BRR × ECC)

### What Makes AeroGuardian Unique?

| Strength | Description |
|:---------|:------------|
| 🔬 **Physics-Grounded** | All anomaly detection uses deterministic, non-LLM thresholds |
| 🔗 **Causal Chain Validation** | Temporal ordering validates that failures propagate physically |
| 📊 **ESRI Trust Scoring** | Multiplicative scoring ensures all components must pass |
| 🔄 **Two-Phase LLM Pipeline** | LLM #1 for scenario translation, LLM #2 for report generation |
| 📈 **Full Telemetry Logging** | Every flight captures 50Hz IMU, GPS, motor outputs for audit |

---

## ✨ Key Features

| Feature | Description |
|:--------|:------------|
| 🤖 **2-LLM Pipeline** | DSPy-constrained structured output with GPT-4o |
| 🎮 **PX4 SITL Integration** | Real flight simulation with Gazebo Harmonic or Classic |
| 🔧 **Native PX4 Fault Injection** | Parameter-based and shell command fault emulation (motor, GPS, baro, etc.) |
| 📊 **31-Parameter LLM Config** | Comprehensive scenario configuration from FAA report |
| 📡 **High-Fidelity Telemetry** | 50Hz IMU (Accel/Gyro), NED Velocity, GPS Metadata |
| 📈 **Physics-Based Analysis** | Deterministic, non-LLM anomaly detection before LLM processing |
| 📑 **Safety Reports** | JSON + PDF with executive summary |
| 📊 **ESRI Framework** | Scientific evaluation: SFS × BRR × ECC |
| 🌐 **Web UI** | Streamlit interface for file upload, analysis, and result download |
| 📦 **Headless Mode** | Run without GUI for batch processing |

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                    AEROGUARDIAN PIPELINE                           │
└────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────┐
    │  📥 FAA UAS Sightings   │
    │      (8,031 cases)      │
    └───────────┬─────────────┘
                │
                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  🤖 LLM #1: SCENARIO TRANSLATION (GPT-4o + DSPy)            │
    │  OUTPUT: 31-parameter PX4 simulation config                 │
    │  - Fault type inference from narrative                      │
    │  - Waypoint generation from location                        │
    │  - PX4 fault injection command selection                    │
    └───────────────────────────┬─────────────────────────────────┘
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  🎮 PX4 SITL + GAZEBO SIMULATION (WSL2)                     │
    │  • Simulators: gz_x500 (Harmonic), gazebo-classic_iris      │
    │  • Failure emulation via PX4 native fault injection (parameter and shell command) │
    │  • Telemetry capture @ 10-50Hz                              │
    └───────────────────────────┬─────────────────────────────────┘
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  📊 PHYSICS-BASED TELEMETRY ANALYSIS (NO LLM)               │
    │  • Deterministic, non-LLM anomaly detection with thresholds │
    │  • Subsystem failure identification                         │
    │  • Causal chain analysis with temporal ordering             │
    └───────────────────────────┬─────────────────────────────────┘
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  🤖 LLM #2: PRE-FLIGHT SAFETY REPORT (GPT-4o + DSPy)        │
    │  INPUT: Verified telemetry analysis (not raw telemetry)     │
    │  OUTPUT: Structured Safety Report (JSON + PDF) + Go/Caution/No-Go |
    └───────────────────────────┬─────────────────────────────────┘
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  📈 ESRI EVALUATION FRAMEWORK                               │
    │  • SFS: Scenario Fidelity Score (LLM translation accuracy)  │
    │  • BRR: Behavioral Reproduction Rate (telemetry validation, deterministic) │
    │  • ECC: Evidence-Conclusion Consistency (claim grounding)   │
    │  • ESRI = SFS × BRR × ECC (multiplicative trust score)      │
    └─────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow: Phase-by-Phase Input/Output Specification

The following tables define the exact data contracts between each pipeline phase, enabling end-to-end traceability and validation.

### Phase 1: FAA Data Ingestion

| Attribute | Specification |
|:----------|:--------------|
| **Module** | `src/faa/sighting_filter.py` |
| **Input** | `data/processed/faa_reports/faa_simulatable.json` (8,031 records) |
| **Output** | Python `Dict` with standardized incident record |

**Input Schema (FAA JSON):**
```json
{
  "report_id": "FAA_Apr2020-Jun2020_182",
  "date": "2020-04-15",
  "time": "14:30",
  "city": "MINNEAPOLIS",
  "state": "MINNESOTA",
  "description": "UAS observed at 3,300ft near airport approach...",
  "incident_type": "airspace_violation"
}
```

**Output Schema (Standardized Dict):**
```python
{
    "report_id": str,      # Unique identifier
    "date": str,           # ISO date (YYYY-MM-DD)
    "city": str,           # City name for geocoding
    "state": str,          # State/region name
    "description": str,    # Full narrative text
    "incident_type": str   # Category: propulsion|navigation|power|control|sensor|other
}
```

---

### Phase 2: Location Geocoding

| Attribute | Specification |
|:----------|:--------------|
| **Module** | `src/core/geocoder.py` |
| **Input** | Incident `Dict` with `city` and `state` fields |
| **Output** | Tuple `(latitude: float, longitude: float)` in WGS84 |

**Input:** `{"city": "MINNEAPOLIS", "state": "MINNESOTA"}`

**Output:** `(44.9778, -93.2650)` — GPS coordinates for PX4 home location

**Validation:** Coordinates verified against US continental bounds (24°-50°N, 66°-125°W)

---

### Phase 3: LLM Scenario Translation (LLM #1)

| Attribute | Specification |
|:----------|:--------------|
| **Module** | `src/llm/scenario_generator.py` |
| **LLM** | GPT-4o via DSPy `FAA_To_PX4_Complete` signature |
| **Input** | FAA narrative text + geocoded location |
| **Output** | 31-parameter PX4 simulation configuration |

**Input Schema:**
```python
{
    "incident_description": str,   # FAA narrative
    "incident_location": str,      # "City, State"
    "incident_type": str,          # Failure category
    "report_id": str               # For tracing
}
```

**Output Schema (31-Parameter Config):**
```json
{
  "faa_source": {
    "report_id": "FAA_xxx",
    "description": "...",
    "outcome": "crash|landing|flyaway|unknown"
  },
  "fault_injection": {
    "fault_type": "motor_failure|gps_loss|battery_failure|...",
    "fault_category": "propulsion|navigation|power|control|sensor",
    "severity": 0.0-1.0,
    "onset_sec": 60
  },
  "mission": {
    "takeoff_altitude_m": 50.0,
    "cruise_altitude_m": 50.0,
    "speed_m_s": 8.0,
    "duration_sec": 120
  },
  "waypoints": [
    {"lat": 44.9778, "lon": -93.2650, "alt": 50, "action": "takeoff"},
    {"lat": 44.9787, "lon": -93.2650, "alt": 50, "action": "waypoint"},
    {"lat": 44.9787, "lon": -93.2641, "alt": 50, "action": "waypoint"},
    {"lat": 44.9778, "lon": -93.2641, "alt": 50, "action": "waypoint"},
    {"lat": 44.9778, "lon": -93.2650, "alt": 50, "action": "land"}
  ],
  "environment": {
    "wind_speed_mps": 5.0,
    "wind_direction_deg": 270,
    "temperature_c": 20
  },
  "px4_commands": {
    "fault": "failure motor off -i 1"
  },
  "proxy_modeling": {
    "aircraft_class": "quadcopter",
    "parachute_modeled": false
  }
}
```

**Key Parameters:**
| Parameter | Description | Constraints |
|:----------|:------------|:------------|
| `fault_type` | PX4-compatible failure | motor, gps, battery, gyro, etc. |
| `severity` | Failure intensity | 0.0 (minor) to 1.0 (complete) |
| `onset_sec` | Injection timing | Default: 60s after takeoff |
| `waypoints` | GPS flight path | 4-5 waypoints, ~100m spacing |

---

### Phase 4: PX4 SITL Simulation

| Attribute | Specification |
|:----------|:--------------|
| **Module** | `scripts/run_automated_pipeline.py` → `MissionExecutor` |
| **Simulator** | PX4 SITL + Gazebo (Harmonic/Classic) via WSL2 |
| **Input** | 31-parameter config from Phase 3 |
| **Output** | Raw telemetry stream (10-50Hz) |

**Input:** Flight config with waypoints, fault injection parameters, mission settings

**Output Schema (Telemetry Point):**
```json
{
  "timestamp": 1706789012.345,
  "gyro_x": 0.012,
  "gyro_y": -0.005,
  "gyro_z": 0.003,
  "acc_x": 0.15,
  "acc_y": -0.08,
  "acc_z": -9.72,
  "gps_lat": 44.97782,
  "gps_lon": -93.26498,
  "gps_alt": 52.3,
  "velocity_x": 5.2,
  "velocity_y": 0.8,
  "velocity_z": -0.3,
  "roll_deg": 5.2,
  "pitch_deg": -2.1,
  "yaw_deg": 135.5,
  "motor_1": 0.72,
  "motor_2": 0.75,
  "motor_3": 0.73,
  "motor_4": 0.74,
  "battery_v": 16.2,
  "gps_satellites": 12,
  "flight_mode": "Mission"
}
```

**Telemetry Channels:**
| Channel | Rate | Description |
|:--------|:-----|:------------|
| IMU (Gyro/Accel) | 50 Hz | Angular rates, linear acceleration |
| Position (GPS) | 10 Hz | Lat/lon/alt coordinates |
| Velocity (NED) | 10 Hz | North/East/Down velocity |
| Attitude (RPY) | 50 Hz | Roll/pitch/yaw angles |
| Motors | 10 Hz | PWM output per motor |
| Battery | 1 Hz | Voltage, current |

---

### Phase 5: Telemetry Analysis

| Attribute | Specification |
|:----------|:--------------|
| **Module** | `src/analysis/telemetry_analyzer.py` |
| **Input** | Raw telemetry list (500-5000 points) |
| **Output** | `TelemetryStats` dataclass with 30+ metrics |

**Input:** List of telemetry dicts from Phase 4

**Output Schema (TelemetryStats):**
```python
@dataclass
class TelemetryStats:
    # Flight Metrics
    duration_s: float              # Total flight time
    data_points: int               # Number of samples
    
    # Altitude Analysis
    max_alt_m: float               # Maximum altitude AGL
    min_alt_m: float               # Minimum altitude
    alt_std_dev_m: float           # Altitude stability
    alt_deviation_m: float         # Max - Min altitude
    
    # Speed Analysis
    max_speed_mps: float           # Peak horizontal speed
    avg_speed_mps: float           # Average cruise speed
    
    # Attitude Stability
    max_roll_deg: float            # Peak roll angle
    max_pitch_deg: float           # Peak pitch angle
    roll_std_dev: float            # Roll stability
    pitch_std_dev: float           # Pitch stability
    
    # Position Analysis
    position_drift_m: float        # Max drift from start
    lateral_drift_m: float         # Horizontal drift
    
    # GPS Quality
    gps_satellite_min: int         # Minimum satellites
    gps_satellite_avg: float       # Average satellites
    gps_variance_m: float          # Position scatter
    
    # Vibration
    vibration_avg: float           # Average vibration
    vibration_max: float           # Peak vibration
    
    # Battery
    battery_start_v: float         # Initial voltage
    battery_end_v: float           # Final voltage
    battery_sag_rate_vps: float    # Discharge rate
    
    # Anomalies (Physics-Based Detection)
    anomalies: List[str]           # Detected issues
    anomaly_severity: str          # NONE|LOW|MEDIUM|HIGH|CRITICAL
    failsafe_events: List[str]     # Triggered failsafes
```

**Key Analysis Methods:**
- **Anomaly Detection**: Threshold-based (no LLM)
- **Stability Metrics**: Standard deviation of angular rates
- **Drift Analysis**: Euclidean distance from home position

---

### Phase 6: Safety Report Generation (LLM #2)

| Attribute | Specification |
|:----------|:--------------|
| **Module** | `src/llm/report_generator.py` |
| **LLM** | GPT-4o via DSPy `GeneratePreFlightReport` signature |
| **Input** | Telemetry summary + incident context |
| **Output** | Structured safety report with verdict |

**Input Schema:**
```python
{
    "incident_description": str,      # Original FAA narrative
    "report_id": str,                 # Incident ID
    "incident_location": str,         # "City, State"
    "fault_type": str,                # Simulated failure
    "expected_outcome": str,          # crash|landing|flyaway
    "telemetry_summary": str          # TelemetryStats.to_summary_text()
}
```

**Output Schema (SafetyReport):**
```json
{
  "report_id": "FAA_xxx",
  "incident_location": "Minneapolis, Minnesota",
  "fault_type": "motor_failure",
  "expected_outcome": "crash",
  
  "safety_level": "HIGH",
  "primary_hazard": "Asymmetric thrust causing uncontrolled descent",
  "observed_effect": "Roll instability detected at T+15s",
  
  "design_constraints": [
    "Pre-flight motor verification required",
    "Redundant motor configuration recommended"
  ],
  "recommendations": [
    "Inspect motor bearings before flight",
    "Install motor failure detection system",
    "Configure automatic RTL on motor anomaly"
  ],
  
  "explanation": "The simulation demonstrated motor failure effects...",
  
  "verdict": "NO-GO"
}
```

**Verdict Categories:**
| Verdict | Risk Level | Action |
|:--------|:-----------|:-------|
| **GO** | Low | Mission can proceed with standard precautions |
| **CAUTION** | Medium | Additional checks required before flight |
| **NO-GO** | High | Mission should not proceed without mitigation |

---

### Phase 7: ESRI Evaluation

| Attribute | Specification |
|:----------|:--------------|
| **Module** | `src/evaluation/evaluate_case.py` |
| **Input** | Flight config + Telemetry + Safety report |
| **Output** | Trust scores (SFS, BRR, ECC, ESRI) |

**Input:** Combined data from Phases 3, 5, 6

**Output Schema (CaseEvaluationResult):**
```json
{
  "incident_id": "FAA_xxx",
  "evaluation_timestamp": "2026-02-06T05:30:00Z",
  
  "scores": {
    "ESRI": 0.72,
    "SFS": 0.85,
    "BRR": 0.92,
    "ECC": 0.92
  },
  
  "consistency_level": "HIGH",
  "consistency_justification": "All claims grounded in telemetry",
  
  "sfs_details": {
    "fault_type_match": true,
    "location_accuracy": 0.95,
    "parameter_validity": 0.80
  },
  
  "brr_details": {
    "anomalies_detected": ["roll_instability", "altitude_loss"],
    "expected_anomalies": ["motor_asymmetry", "attitude_deviation"],
    "match_rate": 0.92
  },
  
  "ecc_details": {
    "claims_verified": 5,
    "claims_unsupported": 0,
    "evidence_coverage": 1.0
  },
  
  "confidence_ceilings_applied": {
    "esri_ceiling": 0.85,
    "sfs_ceiling": 0.80,
    "brr_ceiling": 0.95
  }
}
```

**Trust Level Interpretation:**
| ESRI Score | Trust Level | Reliability |
|:-----------|:------------|:------------|
| ≥70% | HIGH | Output suitable for decision support |
| 40-69% | MEDIUM | Manual verification recommended |
| <40% | LOW | Output should not be relied upon |

---

### Phase 8: Report Output

| Attribute | Specification |
|:----------|:--------------|
| **Module** | `src/reporting/unified_reporter.py` |
| **Input** | All phase outputs (incident, config, telemetry, report, eval) |
| **Output** | JSON, PDF, and Excel files |

**Output Files:**
| File | Format | Content |
|:-----|:-------|:--------|
| `report.json` | JSON | Complete structured report with all data |
| `report.pdf` | PDF | Executive summary (single page) |
| `evaluation.json` | JSON | ESRI scores and breakdowns |
| `evaluation_{id}.xlsx` | Excel | Multi-sheet analysis workbook |
| `full_configuration_output_from_llm.json` | JSON | LLM #1 output (31 params) |
| `full_telemetry_of_each_flight.json` | JSON | Raw telemetry array |

---

## 📊 End-to-End Data Traceability

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA FLOW DIAGRAM                                  │
└─────────────────────────────────────────────────────────────────────────────┘

FAA Sighting Report (JSON)
         │
         ▼
┌─────────────────────┐
│ Phase 1: Ingestion  │──► Dict: {report_id, city, state, description}
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Phase 2: Geocoding  │──► Tuple: (latitude, longitude)
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Phase 3: LLM #1     │──► 31-Param Config: {fault, waypoints, mission}
│ (Scenario Gen)      │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Phase 4: PX4 SITL   │──► Telemetry[]: 500-5000 points @ 10-50Hz
│ (Simulation)        │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Phase 5: Analysis   │──► TelemetryStats: 30+ metrics + anomalies
│ (Physics-based)     │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Phase 6: LLM #2     │──► SafetyReport: {verdict, recommendations}
│ (Report Gen)        │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Phase 7: ESRI       │──► Trust Scores: {ESRI, SFS, BRR, ECC}
│ (Evaluation)        │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Phase 8: Output     │──► Files: JSON + PDF + Excel
│ (Reporter)          │
└─────────────────────┘
```

---

## 💻 Requirements

### System Requirements

| Component | Requirement |
|:----------|:------------|
| **OS** | Windows 10/11 with WSL2 (Ubuntu 22.04/24.04) |
| **Python** | 3.10+ (Windows) |
| **RAM** | 8GB minimum (16GB recommended) |
| **Disk** | 20GB (with PX4 and Gazebo) |
| **API** | OpenAI API key (GPT-4o access) |

### WSL2 Requirements

| Component | Requirement |
|:----------|:------------|
| **Ubuntu** | 22.04 or 24.04 LTS |
| **PX4** | v1.14.3 or v1.15+ |
| **Gazebo** | Harmonic (gz_x500) or Classic (iris) |
| **MAVSDK** | v3.10+ |

---

## 🔧 Installation

### Step 1: Clone Repository (Windows PowerShell)

```powershell
# Clone the repository
git clone https://github.com/rak-junior/aero-guardian.git
cd aero-guardian

# Run the setup script
.\setup.bat
```

### Step 2: Configure API Key

Create a `.env` file in the project root:

```env
OPENAI_API_KEY={API_KEY}
OPENAI_MODEL=gpt-4o
```

### Step 3: Setup PX4 in WSL2

```bash
# Open WSL terminal
wsl

# Navigate to project directory
cd /mnt/c/path/to/aero-guardian/scripts

# Make setup script executable and run
chmod +x setup_px4_gui.sh
./setup_px4_gui.sh --install-deps --install-px4
```

### Step 4: Verify Installation

```powershell
# Activate virtual environment
.\venv\Scripts\activate

# Check Python environment
python --version

# Check MAVSDK
python -c "import mavsdk; print('MAVSDK:', mavsdk.__version__)"

# Check DSPy
python -c "import dspy; print('DSPy OK')"

# Check OpenAI connection
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('API Key:', 'Set' if os.getenv('OPENAI_API_KEY') else 'NOT SET')"
```

---

## ⚡ Quick Start

### 1. Get WSL IP Address

```powershell
# From Windows PowerShell (automatic):
$wsl_ip = (wsl -- hostname -I).Trim().Split()[0]; Write-Host "WSL IP: $wsl_ip"
```

Or manually in WSL:
```bash
ip addr show eth0 | grep inet | head -1 | awk '{print $2}' | cut -d'/' -f1
```

### 2. Run Single FAA Report Analysis

```powershell
# Activate virtual environment
.\venv\Scripts\activate

# Get WSL IP and run pipeline (Gazebo Harmonic, headless)
$wsl_ip = (wsl -- hostname -I).Trim().Split()[0]
python scripts/run_automated_pipeline.py --report 0 --wsl-ip $wsl_ip --headless --simulator gz_x500
```

### 3. View Results

```powershell
# Results are saved to outputs/{report_id}_{timestamp}/
# List recent outputs
Get-ChildItem outputs -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 5

# Open the latest PDF report
$latest = Get-ChildItem outputs -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Start-Process "$($latest.FullName)\report\report.pdf"
```

---

## 🌐 Web UI

AeroGuardian includes a Streamlit web interface for easy file upload and analysis.

### Launch Web UI

```powershell
# Activate environment
.\venv\Scripts\activate

# Start Streamlit server
streamlit run scripts/app.py
```

The UI will open at `http://localhost:8501`

### Web UI Features

| Feature | Description |
|:--------|:------------|
| **File Upload** | Upload JSON, CSV, or Excel files (max 1MB) |
| **WSL IP Config** | Configure WSL IP address in sidebar |
| **Headless Toggle** | Enable/disable Gazebo GUI |
| **Progress Tracking** | Real-time pipeline status updates |
| **Download Results** | Download config, telemetry, report, and evaluation |
| **Output Path Display** | Shows where files are saved for manual inspection |

### Input Data Format

**Required Fields:**
- `report_id` or `incident_id` - Unique identifier
- `description` or `summary` - The sighting narrative text

**Optional Fields:**
- `date` - Incident date (YYYY-MM-DD)
- `city` - City name for location geocoding
- `state` - State/region

**Example JSON:**
```json
{
  "report_id": "FAA_Apr2020-Jun2020_1",
  "date": "2020-04-01",
  "city": "MINNEAPOLIS",
  "state": "MINNESOTA",
  "description": "UAS sighting reported at 3,300ft..."
}
```

### Output Files

All outputs are saved to disk at `outputs/{report_id}_{timestamp}/` whether using the Web UI or command line. The UI displays the output folder path and allows downloading individual files.

---

## 📖 Command Reference

### Main Pipeline Script

```powershell
python scripts/run_automated_pipeline.py [OPTIONS]
```

### Options

| Flag | Description | Default |
|:-----|:------------|:--------|
| `--report`, `-r` | FAA report index (0-8030) | 0 |
| `--batch`, `-b` | JSON file for batch processing | None |
| `--wsl-ip` | WSL2 IP address (required) | None |
| `--headless` | Run without Gazebo GUI | False |
| `--skip-px4` | Assume PX4 already running | False |
| `--qgc-port` | QGroundControl UDP port | 18570 |
| `--vehicle` | PX4 vehicle type | iris |
| `--simulator`, `-s` | Simulator target | auto |

### Simulator Options

| Value | Description |
|:------|:------------|
| `auto` | Auto-select (sihsim_quadx for headless, gz_x500 for GUI) |
| `gz_x500` | Gazebo Harmonic X500 quadcopter (recommended) |
| `gazebo-classic_iris` | Gazebo Classic Iris quadcopter |
| `sihsim_quadx` | Software-In-Hardware simulator (no physics) |

### Example Commands

```powershell
# Activate environment first
.\venv\Scripts\activate

# Get WSL IP
$wsl_ip = (wsl -- hostname -I).Trim().Split()[0]

# Process FAA report #5 with Gazebo Harmonic (headless)
python scripts/run_automated_pipeline.py -r 5 --wsl-ip $wsl_ip --headless -s gz_x500

# Process report #10 with Gazebo Classic (requires GUI)
python scripts/run_automated_pipeline.py -r 10 --wsl-ip $wsl_ip -s gazebo-classic_iris

# Skip PX4 startup (if already running manually in WSL)
python scripts/run_automated_pipeline.py -r 0 --wsl-ip $wsl_ip --skip-px4
```

---

## 🧪 Testing and Validation

### Benchmark Validation (ALFA & RflyMAD Datasets)

AeroGuardian's anomaly detection has been validated against academic benchmark datasets:

| Dataset | Source | Samples | Flights | Fault Types |
|:--------|:-------|--------:|--------:|:------------|
| **ALFA** | CMU Air Lab | 377,585 | 35 | Engine failure |
| **RflyMAD** | Beihang University | 1,418,960 | 1,424 | Motor, sensor, wind |

**Validation Results (Full Dataset):**

| Fault Type | Precision | Recall | F1-Score | Detection Latency |
|:-----------|----------:|-------:|---------:|------------------:|
| Motor Fault | 100.0% | 78.1% | 87.7% | 4.9s |
| Sensor Fault | 100.0% | 75.6% | 86.1% | 6.6s |
| Wind Fault | 100.0% | 36.4% | 53.4% | 40.4s |
| **Overall** | **90.9%** | **59.2%** | **71.7%** | - |

**Run Benchmark Validation:**
```powershell
.\venv\Scripts\activate
python scripts/run_benchmark_validation.py --sample 0.1   # 10% sample (fast)
python scripts/run_benchmark_validation.py --sample 1.0   # Full validation
python scripts/run_benchmark_validation.py --calibrate    # Threshold calibration
```

Validation reports are saved to `outputs/verification/`.

---

### Quick Validation Tests

```powershell
# Activate environment
.\venv\Scripts\activate
$wsl_ip = (wsl -- hostname -I).Trim().Split()[0]

# Test 1: Minneapolis altitude violation (Report #0)
python scripts/run_automated_pipeline.py -r 0 --wsl-ip $wsl_ip --headless -s gz_x500

# Test 2: Pittsburgh airport approach (Report #2)
python scripts/run_automated_pipeline.py -r 2 --wsl-ip $wsl_ip --headless -s gz_x500

# Test 3: Custom test scenario (motor failure)
python scripts/run_automated_pipeline.py --batch data/test/test_propulsion.json --wsl-ip $wsl_ip --headless -s gz_x500
```

### Verify Test Output

After each test, check the evaluation results:

```powershell
# Find latest output directory
$latest = Get-ChildItem outputs -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Write-Host "Latest output: $($latest.Name)"

# View ESRI scores
Get-Content "$($latest.FullName)\evaluation\evaluation.json" | python -c "import sys,json; d=json.load(sys.stdin); print(f'ESRI: {d[\"scores\"][\"ESRI\"]:.1%}, SFS: {d[\"scores\"][\"SFS\"]:.1%}, BRR: {d[\"scores\"][\"BRR\"]:.1%}, ECC: {d[\"scores\"][\"ECC\"]:.1%}')"

# View telemetry metrics (verify bug fixes)
Get-Content "$($latest.FullName)\report\report.json" | python -c "import sys,json; d=json.load(sys.stdin); ts=d.get('telemetry_summary',{}).get('statistics',{}); print(f'max_roll_deg: {ts.get(\"max_roll_deg\", \"N/A\")} (should be <=180)'); print(f'gps_variance: {ts.get(\"flight_summary\",{}).get(\"gps_variance\", \"N/A\")}m (should be <1000)')"
```

### Expected Results

| Metric | Acceptable Range | Notes |
|:-------|:-----------------|:------|
| **ESRI** | ≥70% | HIGH trust level |
| **SFS** | ≥60% | Scenario fidelity |
| **BRR** | ≥80% | Behavioral reproduction |
| **ECC** | ≥80% | Evidence consistency |
| **max_roll_deg** | ≤180° | Normalized angle |
| **gps_variance** | <1000m | Reasonable GPS drift |

---

## 📊 Evaluation Framework (ESRI)

### Components

| Score | Name | What It Measures |
|-------|------|------------------|
| **SFS** | Scenario Fidelity Score | LLM translation accuracy (fault type, location, parameters) |
| **BRR** | Behavior Reproduction Rate | Telemetry shows expected anomalies for the fault type |
| **ECC** | Evidence-Conclusion Consistency | All claims in report are grounded in telemetry evidence |

### Trust Calculation

```
ESRI = SFS × BRR × ECC
```

This multiplicative formula ensures:
- Any component at 0 → ESRI = 0 (system output untrusted)
- All components ≥0.7 → HIGH trust (ESRI ≥0.343)

### Trust Levels

| ESRI | Level | Action |
|------|-------|--------|
| ≥70% | **HIGH** | Output reliable for decision-making |
| ≥40% | **MEDIUM** | Manual review recommended |
| <40% | **LOW** | Do not rely on output |

---

## ⚙️ Simulation Approach

### Failure Emulation via PX4 Native Fault Injection

AeroGuardian uses PX4's native fault injection system (`SYS_FAILURE_EN=1`) with both parameter-based and shell command injection:

| Failure Category | PX4 Command | Expected Effect |
|------------------|-------------|-----------------|
| **Motor Failure** | `failure motor off -i 1` | Asymmetric thrust, roll/yaw |
| **GPS Loss** | `failure gps off` | Position drift, EKF fallback |
| **Barometer Failure** | `failure baro off` | Altitude hold loss |
| **Magnetometer** | `failure mag stuck` | Heading drift |
| **Gyroscope** | `failure gyro off` | Attitude instability |
| **Accelerometer** | `failure accel garbage` | Severe control issues |

Parameter-based injection is used where possible for reliability; shell commands are used for direct PX4 fault emulation.

### Why Native Fault Injection?

1. **Realistic Behavior**: PX4 failsafe logic responds naturally
2. **Telemetry Signatures**: Observable anomalies match real failures
3. **Reproducibility**: Same command produces consistent results
4. **Validation**: Anomalies can be validated against expected physics

---

## ✅ Why Our Analysis is Grounded

> **Note:** We use "grounded" rather than "accurate" because accuracy implies comparison to ground truth. FAA sighting reports lack verified ground truth for technical failure modes. Our analysis is grounded in physics-based simulation and deterministic thresholds.

### 1. Physics-Grounded Anomaly Detection (No LLM)

All anomaly detection uses **deterministic, physics-based thresholds** (see `src/evaluation/behavior_validation.py`):

```python
class AnomalyThresholds:
  POSITION_DRIFT_M = 10.0      # Exceeded = GPS anomaly
  ALTITUDE_DEVIATION_M = 5.0   # Exceeded = altitude instability
  ROLL_MAX_DEG = 30.0          # Exceeded = attitude anomaly
  GPS_HDOP_MAX = 3.0           # Exceeded = GPS quality issue
  MOTOR_ASYMMETRY_DIFF = 0.3   # Exceeded = propulsion imbalance
```

### 2. LLM Cannot Hallucinate Anomalies

Every anomaly in the safety report is:
- **Detected deterministically**: No LLM involvement in anomaly detection
- **Timestamped**: When it first exceeded threshold
- **Measured**: Actual telemetry value
- **Threshold-based**: Industry-standard limit exceeded
- **Subsystem-attributed**: Failure component identified

### 3. Causal Chain Validation

The system traces anomalies through time:

```
propulsion (motor_asymmetry @ t=12.5s) → control (roll_instability @ t=15.2s) → navigation (position_drift @ t=18.7s)
✅ PLAUSIBLE: Motor failure → thrust imbalance → attitude deviation → position error
```

### 4. Multiplicative ESRI Prevents Partial Trust

```
ESRI = SFS × BRR × ECC
```

If any component is 0 (e.g., no anomalies detected), ESRI = 0 → output untrusted.

---

## 📁 Project Structure

```
aero-guardian/
├── scripts/
│   ├── run_automated_pipeline.py   # Main pipeline entry point
│   ├── run_batch_pipeline.py       # Batch processing script
│   ├── run_benchmark_validation.py # ALFA/RflyMAD benchmark validation
│   ├── setup_px4_gui.sh            # WSL2 PX4/Gazebo setup
│   ├── app.py                      # Streamlit web interface
│   └── process_faa_data.py         # FAA data preprocessing
│
├── src/
│   ├── llm/                        # 2-LLM Pipeline
│   │   ├── scenario_generator.py   # LLM #1: FAA → PX4 config (31 params)
│   │   ├── report_generator.py     # LLM #2: Telemetry → Safety Report
│   │   ├── signatures.py           # DSPy signatures (structured output)
│   │   ├── dspy_fewshot.py         # Few-shot learning examples
│   │   ├── client.py               # LLM client wrapper
│   │   └── llm_logger.py           # Request/Response logging
│   │
│   ├── simulation/
│   │   └── failure_emulator.py     # PX4 fault injection manager
│   │
│   ├── analysis/
│   │   └── telemetry_analyzer.py   # Physics-based telemetry analysis
│   │
│   ├── evaluation/
│   │   ├── evaluate_case.py        # Unified case evaluator (ESRI)
│   │   ├── scenario_fidelity.py    # SFS: Scenario Fidelity Score
│   │   ├── behavior_validation.py  # BRR: Behavior Reproduction Rate
│   │   ├── evidence_consistency.py # ECC: Evidence-Conclusion Consistency
│   │   ├── esri.py                 # ESRI framework calculator
│   │   ├── subsystem_analysis.py   # Causal chain analysis
│   │   └── regulatory_standards.py # FAA/industry threshold references
│   │
│   ├── reporting/
│   │   └── unified_reporter.py     # JSON/PDF/Excel report generation
│   │
│   ├── faa/
│   │   └── sighting_filter.py      # FAA data loading and filtering
│   │
│   ├── validation/
│   │   └── scenario_validator.py   # Config validation
│   │
│   ├── core/
│   │   ├── config.py               # Configuration management
│   │   ├── geocoder.py             # Location geocoding (city → GPS)
│   │   ├── pdf_report_generator.py # PDF generation (ReportLab)
│   │   ├── openai_connector.py     # OpenAI API wrapper
│   │   └── logging_config.py       # Centralized logging setup
│   │
│   └── ui/
│       └── styles.py               # Streamlit UI styling
│
├── data/
│   ├── processed/
│   │   └── faa_reports/
│   │       ├── faa_reports.json        # Full FAA dataset
│   │       └── faa_simulatable.json    # 8,031 simulatable sightings
│   ├── raw/faa/                        # Raw FAA source files
│   └── test/                           # Test scenarios
│       ├── scenarios1.json
│       ├── test_control.json
│       ├── test_navigation.json
│       ├── test_power.json
│       ├── test_propulsion.json
│       └── test_sensor.json
│
├── outputs/                            # Per-run output folders
│   ├── {report_id}_{timestamp}/        # Individual run outputs
│   │   ├── input/                      # Original input data
│   │   ├── generated/                  # LLM config + telemetry
│   │   ├── report/                     # JSON + PDF reports
│   │   ├── evaluation/                 # ESRI scores (JSON + Excel)
│   │   └── llm_logs/                   # LLM interaction traces
│   └── verification/                   # Benchmark validation results
│       ├── VALIDATION_REPORT.md        # Human-readable validation
│       ├── benchmark_results.json      # Full metrics
│       └── threshold_calibration.json  # Threshold analysis
│
├── docs/                               # Documentation
│   ├── EVALUATION.md                   # ESRI framework details
│   ├── evaluation_strategy_analysis.md # Validation methodology
│   └── related_work_resources.md       # Academic references
│
├── logs/                               # Application logs (daily)
│
├── setup.bat                           # Windows setup script
├── run_demo.bat                        # Demo launcher
├── requirements.txt                    # Python dependencies
└── .env                                # API keys (create this)
```

---

## 📂 Output Structure

Each pipeline run creates:

```
outputs/{report_id}_{timestamp}/
├── input/                                       # Original input data
│   └── original_input.json                      # Uploaded/source incident data
│
├── generated/
│   ├── full_configuration_output_from_llm.json  # LLM #1: 31-param config
│   └── full_telemetry_of_each_flight.json       # Raw telemetry (10-50Hz)
│
├── report/
│   ├── report.json                # Safety report (structured)
│   └── report.pdf                 # PDF summary (human-readable)
│
├── evaluation/
│   ├── evaluation.json            # ESRI + component scores
│   └── evaluation_{id}.xlsx       # Multi-sheet Excel analysis
│
└── llm_logs/                      # LLM interaction logs
    ├── phase1_scenario_*.json     # LLM #1 request/response
    └── phase2_report_*.json       # LLM #2 request/response
```

**All outputs are saved to disk** whether running via command line or Web UI.

---

## 🔧 Troubleshooting

### Common Issues

| Issue | Solution |
|:------|:---------|
| **WSL IP not found** | Run `wsl -- hostname -I` in PowerShell |
| **PX4 SITL timeout** | First build takes 10-15 min; subsequent runs are faster |
| **Gazebo not starting** | Use `--headless` mode or install VcXsrv |
| **MAVSDK connection failed** | Check WSL IP and firewall settings |
| **OpenAI API error** | Verify `.env` file has valid `OPENAI_API_KEY` |
| **"Event loop is closed"** | Normal cleanup message from gRPC, can be ignored |

### Checking WSL IP (PowerShell)

```powershell
$wsl_ip = (wsl -- hostname -I).Trim().Split()[0]
Write-Host "WSL IP: $wsl_ip"
```

### Checking PX4 Status (WSL)

```bash
ps aux | grep px4 | grep -v grep
```

### Viewing Logs

```powershell
# View today's log
$today = Get-Date -Format "yyyyMMdd"
Get-Content "logs\aeroguardian_$today.log" -Tail 50
```

### Killing Stuck PX4 Process

```bash
# In WSL
pkill -f px4
pkill -f gz
```

---

## 📚 Regulatory References

### Altitude (120m / 400ft Cap)

> **14 CFR Part 107.51(b):** "The altitude of the small unmanned aircraft cannot be higher than 400 feet above ground level..."
>
> — [Electronic Code of Federal Regulations](https://www.ecfr.gov/current/title-14/chapter-I/subchapter-F/part-107/subpart-B/section-107.51)

### Attitude Thresholds (30° Roll/Pitch)

> **FAA AC 25-7D (Flight Test Guide):** Bank angles of 30° are typical limits for transport category aircraft stability assessment.
>
> — [FAA Advisory Circular 25-7D](https://www.faa.gov/regulations_policies/advisory_circulars)

### GPS Quality (HDOP Thresholds)

> **RTCA DO-316:** HDOP values of 2.0-3.0 represent "Good" horizontal accuracy suitable for general navigation.
>
> — RTCA Inc., DO-316

### PX4 Control Saturation (80%)

> **PX4 Autopilot Documentation:** Actuator output exceeding 80% of range indicates potential loss of control authority.
>
> — [PX4 User Guide - Failsafe](https://docs.px4.io/main/en/config/safety.html)

---

## ⚠️ Limitations & Scope

### What This System DOES:
- Transforms FAA UAS sighting narratives into **testable simulation scenarios**
- Generates **safety hypotheses** for pre-flight risk awareness
- Provides **physics-grounded analysis** of simulated failure modes
- Supports **proactive hazard identification** in mission planning

### What This System DOES NOT DO:
- ❌ **Reconstruct real accidents** — FAA sighting reports are observational, not investigative
- ❌ **Predict future failures** — Past sightings do not predict specific aircraft failures
- ❌ **Certify aircraft safety** — This is not an airworthiness assessment tool
- ❌ **Replace human judgment** — All outputs require operator review

### FAA UAS Sighting Report Limitations

FAA UAS sighting reports are:
1. **Incomplete** — Many details are missing or approximated
2. **Non-investigative** — No root cause analysis is performed
3. **Observer-biased** — Reports reflect what observers *believed* they saw
4. **Operator-stated** — Technical claims are unverified self-reports

The system treats these reports as **hazard signals**, not ground truth.

### Simulation Fidelity Constraints

| Constraint | Description |
|:-----------|:------------|
| **Aircraft Class** | System simulates X500 quadcopter. Fixed-wing and other aircraft types cannot be accurately represented. Confidence penalties are applied automatically via guardrails. |
| **Failure Modes** | PX4 fault injection approximates real failures but cannot replicate all real-world failure dynamics. |
| **Environmental Factors** | Wind, weather, and terrain are simulated with defaults unless specified in the source report. |

### Risk Reduction ≠ Safety Guarantee

This system **reduces risk** through proactive hazard identification. It does NOT:
- Guarantee flight safety
- Replace pre-flight inspections
- Substitute for operator training
- Provide regulatory compliance certification

> **All outputs should be treated as decision support tools, not authoritative safety determinations.**

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

## Acknowledgments

- **FAA** - UAS Sighting Reports (2019-2025)
- **PX4 Autopilot** - SITL simulation framework
- **OpenAI** - GPT-4o language model
- **DSPy (Stanford NLP)** - Structured LLM output framework
- **MAVSDK** - Drone SDK for mission execution
- **Gazebo** - Physics simulation (Harmonic & Classic)

---

*AeroGuardian - Decision Support for Pre-Flight Risk Awareness*

> **Disclaimer:** This system provides simulation-based hazard analysis for educational and research purposes. Outputs are decision support tools, not safety certifications. All operational decisions remain the responsibility of the operator.
