# 🛡️ AeroGuardian

**Pre-Flight UAV Safety Analysis System using LLM-Driven Scenario Translation**

Transform FAA UAS sighting reports into actionable pre-flight safety recommendations through automated simulation and AI analysis.

**Author:** AeroGuardian Member  
**Version:** 1.0 
**Date:** 2026-01-31

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PX4](https://img.shields.io/badge/PX4-v1.14.3-orange.svg)](https://px4.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Data Flow](#-data-flow)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Configuration Formats](#-configuration-formats)
- [Output Structure](#-output-structure)
- [Logging System](#-logging-system)

---

## 🎯 Overview

AeroGuardian is an **automated pre-flight safety analysis system** that learns from historical FAA UAS sighting reports to prevent future accidents. The system:

1. **Ingests real FAA sighting reports** (8,031 testable UAS sightings)
2. **Translates to simulation** using LLM-driven parameter extraction
3. **Runs PX4 SITL simulation** with realistic fault injection
4. **Captures full telemetry** at 10 Hz sampling rate
5. **Generates safety reports** with Go/No-Go recommendations

### Key Features

| Feature | Description |
|:--------|:------------|
| 🤖 **2-LLM Pipeline** | DSPy-constrained structured output with GPT-4o |
| 🎮 **PX4 SITL Integration** | Real flight simulation with Gazebo GUI |
| 🔧 **Multi-Stage Failure Emulation** | 5-category failure models (propulsion, navigation, battery, control, sensor) |
| 📊 **31-Parameter Config** | Comprehensive LLM-generated simulation configuration |
| 📈 **Full Telemetry Capture** | GPS, IMU, battery, attitude at 10 Hz |
| 📑 **Safety Reports** | JSON + PDF with executive summary |
| 📊 **ESRI Framework** | Scientific evaluation: SFS × BRR × ECC (Excel output) |
| 🔗 **QGroundControl** | Real-time visualization at {WSL_IP}:18570 |
| 📝 **Comprehensive Logging** | Single daily log with full LLM I/O tracking |

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                    AEROGUARDIAN PIPELINE                           │
└────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────┐
    │  📥 FAA UAS Sightings   │
    │      (8,918+ cases)     │
    └───────────┬─────────────┘
                │
                ▼
    ┌─────────────────────────┐     ┌─────────────────────────┐
    │   Sighting Filter       │────▶│      Geocoder           │
    │   (Simulatable Only)    │     │   (Nominatim API)       │
    └───────────┬─────────────┘     └───────────┬─────────────┘
                │                               │
                └───────────────┬───────────────┘
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  🤖 LLM #1: SCENARIO TRANSLATION (GPT-4o + DSPy)            │
    │  ─────────────────────────────────────────────────────────  │
    │  INPUT:  FAA sighting description + location                │
    │  OUTPUT: 31-parameter PX4 simulation config                 │
    │          • Mission profile  • Fault injection               │
    │          • Environment      • Waypoints                     │
    └───────────────────────────┬─────────────────────────────────┘
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  🎮 PX4 SITL + GAZEBO SIMULATION                            │
    │  ─────────────────────────────────────────────────────────  │
    │  • WSL2 Ubuntu + X11 Display                                │
    │  • QGroundControl @ {WSL_IP}:18570                          │
    │  • MAVSDK Mission Execution                                 │
    │  • Telemetry Capture @ 10Hz                                 │
    └───────────────────────────┬─────────────────────────────────┘
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  📊 TELEMETRY ANALYZER                                      │
    │  ─────────────────────────────────────────────────────────  │
    │  • Position drift    • Altitude variance                    │
    │  • IMU vibration     • Anomaly detection                    │
    └───────────────────────────┬─────────────────────────────────┘
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  🤖 LLM #2: PRE-FLIGHT SAFETY REPORT (GPT-4o + DSPy)         │
    │  ─────────────────────────────────────────────────────────  │
    │  INPUT:  Sighting + fault_type + telemetry summary           │
    │  OUTPUT: 3-Section Pre-Flight Safety Report                  │
    │                                                              │
    │  SECTION 1: Hazard & Root Cause                              │
    │    • Safety level (CRITICAL/HIGH/MEDIUM/LOW)                 │
    │    • Pre-Flight Decision: GO / CAUTION / NO-GO               │
    │    • Primary hazard aligned with fault_type                  │
    │    • Observed effects from telemetry                         │
    │                                                              │
    │  SECTION 2: Design Constraints & Recommendations             │
    │    • 2-4 operational constraints                             │
    │    • 3-5 actionable engineering mitigations                  │
    │                                                              │
    │  SECTION 3: Evidence-Based Explanation                       │
    │    • Analysis chain: fault_type → telemetry → verdict        │
    │    • Honest about simulation limitations if applicable       │
    │                                                              │
    └───────────────────────────┬─────────────────────────────────┘
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  📤 UNIFIED REPORTER                                        │
    │  ─────────────────────────────────────────────────────────  │
    │  outputs/{sighting_id}_{timestamp}/                          │
    │  ├── report.json         ← Machine-readable report           │
    │  └── report.pdf          ← Executive summary PDF             │
    └─────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow

### LLM #1: Scenario Translation

```
FAA Sighting Text ──▶ DSPy FAA_To_PX4_Complete ──▶ 31-Parameter Config
                                                   ├── Mission Profile
                                                   ├── Fault Injection
                                                   ├── Environment
                                                   └── Waypoints
```

### Simulation & Analysis

```
PX4 Config ──▶ MAVSDK Mission ──▶ Telemetry @ 10Hz ──▶ Anomaly Detection ──▶ Feature Summary
```

### LLM #2: Pre-Flight Safety Report (3-Section Structure)

```
Sighting + fault_type + Telemetry ──▶ DSPy GeneratePreFlightReport ──▶

  ─────────────────────────────────────────────────────────────────────
  SECTION 1: HAZARD & ROOT CAUSE
  ─────────────────────────────────────────────────────────────────────
  • Safety Level: CRITICAL / HIGH / MEDIUM / LOW
  • Pre-Flight Decision: GO / CAUTION / NO-GO
  • Primary Hazard: Aligned with simulated fault_type
  • Observed Effect: From telemetry analysis

  ─────────────────────────────────────────────────────────────────────
  SECTION 2: DESIGN CONSTRAINTS & RECOMMENDATIONS
  ─────────────────────────────────────────────────────────────────────
  • Design Constraints: 2-4 operational limitations
  • Recommendations: 3-5 actionable engineering mitigations

  ─────────────────────────────────────────────────────────────────────
  SECTION 3: EVIDENCE-BASED EXPLANATION
  ─────────────────────────────────────────────────────────────────────
  • Analysis chain: fault_type → telemetry evidence → safety level
  • Honest acknowledgment if simulation didn't reproduce expected failure
```

---

## 🔧 Failure Emulation Methodology

AeroGuardian uses a **multi-stage failure emulation** approach when native PX4 fault injection is unavailable:

### Failure Categories

| Category | Emulation Method | Telemetry Signature |
|----------|------------------|---------------------|
| **Propulsion** | Asymmetric thrust via PWM reduction | Yaw-roll coupling, spiral tendency |
| **Navigation** | GPS noise injection, EKF stress | Position drift, mode transitions |
| **Battery** | Failsafe threshold triggering | RTL/land behavior, controlled descent |
| **Control** | Control gain degradation | Oscillations, settling time increase |
| **Sensor** | EKF noise parameter injection | Attitude variance, compensation effort |

### 5-Phase Progression Model

Each failure follows a realistic temporal progression:

```
NOMINAL → INCIPIENT → PROPAGATION → CRITICAL → RESOLUTION
   │          │            │            │           │
   └── Normal └── Early    └── Growing  └── Severe  └── Controlled
       flight     warning      symptoms     failure     landing
```

### Scientific Rigor

- **Temporal Randomization**: Prevents LLM script-learning (onset: 5-20s, ±30% phase durations)
- **Parameter Restoration**: Cleans up after emulation
- **Graceful Fallback**: Uses controlled landing if emulation fails

---

## 📏 Simulation Scope & Limitations

AeroGuardian prioritizes **failure mode behavior analysis** over exact scenario replication. This design choice is intentional and supported by established UAV safety research principles:

### Altitude Capping (120m Maximum)

| Aspect | Explanation |
|--------|-------------|
| **Why 120m?** | Standard consumer UAS operational ceiling under FAA Part 107 (400 ft AGL) |
| **Physical Justification** | Motor failure dynamics (asymmetric thrust, yaw-roll coupling) are physics-invariant above ~30m - the aerodynamic forces scale proportionally regardless of absolute altitude |
| **Research Goal** | Characterize **failure mode behavior** (e.g., spiral descent rate, recovery time), not replicate exact crash location |
| **Limitation Acknowledgment** | Some sightings describe aircraft at 1000+ ft AGL - these high-altitude scenarios have longer descent times but identical failure physics |

> [!IMPORTANT]
> The altitude cap affects **time-to-impact**, not **failure mode signature**. A motor failure at 120m produces the same telemetry patterns (attitude variance, yaw rate deviation) as at 500m - the physics are identical.

### Location Fidelity

| Aspect | Approach |
|--------|----------|
| **Current Method** | Geocode to city/state center, generate waypoints around that position |
| **Limitation** | Does not replicate precise location (e.g., "500 feet from airport runway") |
| **Justification** | GPS coordinates affect QGC visualization, not failure mode physics |
| **Future Enhancement** | For airport proximity scenarios, could add airspace awareness constraints |

> [!NOTE]  
> The primary purpose is to **study how specific failure modes manifest in telemetry** and generate pre-flight safety intelligence. Geographic precision is secondary to failure behavior fidelity.


### Limitations Acknowledgment

The LLM #2 safety report includes **evidence-based explanation** with explicit acknowledgment when:
- Simulation altitude differs significantly from reported altitude
- Location is approximated to city center
- Failure mode was inferred from behavior description (not explicit in FAA report)

This transparency ensures the safety report maintains scientific integrity.

---

## ⚡ Quick Start

### Prerequisites

- Windows 10/11 with WSL2
- Python 3.10+
- OpenAI API key
- (Optional) PX4-Autopilot in WSL

### 1. Clone & Setup

```bash
git clone https://github.com/your-org/aero-guardian.git
cd aero-guardian

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file:

```env
OPENAI_API_KEY={API_KEY}
OPENAI_MODEL=gpt-4o
```

### 3. Run Automated Pipeline

```bash
# Full automation (requires PX4 in WSL)
python scripts/run_automated_pipeline.py --incident 0

# Skip PX4 if already running
python scripts/run_automated_pipeline.py --incident 0 --skip-px4

# Headless mode (no Gazebo GUI)
python scripts/run_automated_pipeline.py --incident 0 --headless
```

### 4. View Results

```
outputs/FAA_xxxxx_20260119_124500/
├── generated/
│   ├── full_configuration_output_from_llm.json  ← 31-parameter LLM #1 config
│   └── full_telemetry_of_each_flight.json       ← Raw telemetry @ 10Hz
├── report/
│   ├── report.json                              ← Machine-readable report
│   └── report.pdf                               ← Executive summary PDF
└── evaluation/
    ├── evaluation.json                          ← ESRI metrics
    └── evaluation_*.xlsx                        ← Detailed ESRI spreadsheet
```

---

## 🔧 Installation

### System Requirements

| Component | Requirement |
|:----------|:------------|
| OS | Windows 10/11 with WSL2 |
| Python | 3.10+ |
| RAM | 8GB minimum |
| Disk | 20GB (with PX4) |
| GPU | Not required |

### Python Dependencies

```bash
# Core
pip install numpy pandas dspy-ai openai python-dotenv

# Reporting
pip install openpyxl reportlab

# Validation (for semantic similarity)
pip install sentence-transformers

# PX4 Integration
pip install mavsdk pymavlink
```

### PX4 + Gazebo Setup (WSL)

```bash
# In WSL terminal:
cd /mnt/c/VIRAK/Python\ Code/aero-guardian/scripts
chmod +x setup_px4_gui.sh

# Configure only (if PX4 already installed):
./setup_px4_gui.sh --configure-only

# Full installation:
./setup_px4_gui.sh --install-deps --install-px4
```

This creates launcher scripts:
- `~/launch_px4_gazebo.sh` - GUI mode
- `~/launch_px4_headless.sh` - Headless mode

### QGroundControl Connection

Configure QGroundControl to listen on:
- **IP:** `{WSL_IP}` (your WSL2 IP address)
- **Port:** 18570

---

## 📖 Usage

### Automated Pipeline

```bash
# Process sighting by index
python scripts/run_automated_pipeline.py -i 5

# Specify QGC connection
python scripts/run_automated_pipeline.py --qgc-ip {WSL_IP} --qgc-port 18570

# Different vehicle type
python scripts/run_automated_pipeline.py --vehicle typhoon_h480
```

### Command Line Options

| Flag | Description | Default |
|:-----|:------------|:--------|
| `--incident`, `-i` | FAA incident index | 0 |
| `--headless` | No Gazebo GUI | false |
| `--skip-px4` | Assume PX4 running | false |
| `--qgc-ip` | QGroundControl IP | {WSL_IP} |
| `--qgc-port` | QGroundControl port | 18570 |
| `--vehicle` | PX4 vehicle type | iris |

### Run Evaluation

```bash
# Generate research metrics
python scripts/run_evaluation.py
```

---

## 📁 Project Structure

```
aero-guardian/
├── scripts/
│   ├── run_automated_pipeline.py   # Main automation script
│   ├── run_evaluation.py           # Research evaluation metrics
│   ├── setup_px4_gui.sh            # PX4 + Gazebo WSL setup
│   ├── execute_mission_mavsdk.py   # MAVSDK mission executor
│   └── archive/                    # Legacy scripts
│
├── src/
│   ├── core/                       # Core utilities
│   │   ├── __init__.py             # Clean exports
│   │   ├── logging_config.py       # get_logger, log_exception
│   │   ├── openai_connector.py     # OpenAI API wrapper
│   │   ├── geocoder.py             # geocode, geocode_incident
│   │   ├── pdf_report_generator.py # PDFGenerator
│   │   └── config.py               # Config, get_config
│   │
│   ├── llm/                        # 2-LLM Pipeline
│   │   ├── __init__.py             # Main exports: LLMClient
│   │   ├── signatures.py           # DSPy signatures (FAA_To_PX4, Report)
│   │   ├── scenario_generator.py   # LLM #1: FAA → PX4 config
│   │   ├── report_generator.py     # LLM #2: Telemetry → Report
│   │   ├── client.py               # Main entry point ⭐
│   │   └── dspy_fewshot.py         # Few-shot examples
│   │
│   ├── simulation/                 # PX4 SITL integration & failure emulation
│   │   ├── __init__.py             # Module exports
│   │   └── failure_emulator.py     # Multi-stage failure emulation ⭐
│   │
│   ├── analysis/
│   │   └── telemetry_analyzer.py   # Telemetry feature extraction
│   │
│   ├── validation/
│   │   └── scenario_validator.py   # Semantic similarity validation
│   │
│   ├── evaluation/                 # ESRI Research Framework
│   │   ├── __init__.py             # CaseEvaluator ⭐ entry point
│   │   ├── scenario_fidelity.py    # SFS scorer
│   │   ├── behavior_validation.py  # BRR validator
│   │   ├── evidence_consistency.py # ECC checker
│   │   ├── esri.py                 # ESRI = SFS × BRR × ECC
│   │   └── evaluate_case.py        # Unified evaluator
│   │
│   ├── reporting/
│   │   ├── __init__.py             # UnifiedReporter ⭐
│   │   └── unified_reporter.py     # Multi-format report generation
│   │
│   └── faa/
│       └── sighting_filter.py       # Simulatable sighting filter
│
├── data/
│   └── processed/
│       └── faa_reports/
│           └── faa_reports.json     # 8,031 FAA UAS sighting reports
│
├── outputs/
│   └── {sighting_id}_{timestamp}/   # Per-sighting output folders
│
├── logs/
│   └── 2026-01-30.log              # Daily consolidated log
│
├── .env                            # Environment configuration
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 📄 Configuration Formats

### LLM Configuration Output (31 Parameters)

`generated/full_configuration_output_from_llm.json`:

```json
{
  "metadata": {
    "file_type": "full_configuration_output_from_llm",
    "generated_at": "2026-01-19T12:45:00",
    "incident_id": "FAA_Apr2020-Jun2020_0",
    "incident_location": "Pecos, TX"
  },
  "source_incident": {
    "id": "FAA_Apr2020-Jun2020_0",
    "city": "Pecos",
    "state": "TX",
    "summary": "Flyaway during survey operation..."
  },
  "llm_configuration": {
    "mission": {
      "start_lat": 31.4229,
      "start_lon": -103.4932,
      "takeoff_altitude_m": 30.0,
      "max_altitude_m": 120.0,
      "flight_mode": "MISSION",
      "duration_sec": 180,
      "cruise_speed_ms": 5.0
    },
    "waypoints": [
      {"lat": 31.4229, "lon": -103.4932, "alt": 30, "action": "takeoff"},
      {"lat": 31.4232, "lon": -103.4929, "alt": 50, "action": "waypoint"},
      {"lat": 31.4229, "lon": -103.4932, "alt": 30, "action": "land"}
    ],
    "fault_injection": {
      "fault_type": "gps_dropout",
      "severity": 0.7,
      "onset_sec": 45,
      "duration_sec": 30,
      "affected_components": ["gps", "navigation"]
    },
    "environment": {
      "wind_speed_ms": 8.5,
      "wind_direction_deg": 225,
      "turbulence_intensity": 0.4,
      "temperature_c": 35.0,
      "visibility_m": 8000
    },
    "gps": {
      "satellite_count": 6,
      "hdop": 2.5,
      "noise_m": 3.0
    },
    "battery": {
      "cells": 4,
      "capacity_mah": 5000,
      "start_pct": 100,
      "sag_rate": 0.15
    },
    "failsafe": {
      "action": "RTL",
      "rtl_altitude_m": 50,
      "geofence_radius_m": 500
    },
    "reasoning": "FAA report describes GPS signal loss during survey..."
  },
  "parameter_count": 31
}
```

### Telemetry Output

`generated/full_telemetry_of_each_flight.json`:

```json
{
  "metadata": {
    "file_type": "full_telemetry_of_each_flight",
    "generated_at": "2026-01-19T12:47:00",
    "incident_id": "FAA_Apr2020-Jun2020_0"
  },
  "flight_summary": {
    "total_data_points": 1200,
    "flight_duration_sec": 120.0,
    "max_altitude_m": 85.5,
    "sampling_rate_hz": 10
  },
  "telemetry": [
    {
      "timestamp": 0.0,
      "lat": 31.4229,
      "lon": -103.4932,
      "alt": 0.0,
      "relative_alt": 0.0,
      "roll": 0.0,
      "pitch": 0.0,
      "yaw": 45.0,
      "battery_v": 16.8,
      "battery_pct": 100.0
    },
    // ... 1200 data points
  ]
}
```

### Safety Report Output

`report/report.json`:

```json
{
  "report_type": "PRE-FLIGHT SAFETY REPORT",
  "version": "1.0",
  "generated_at": "2026-01-19T12:48:00",
  
  "incident_source": {
    "original_faa_narrative": "During a survey operation, the drone lost GPS signal...",
    "report_id": "FAA_Apr2020-Jun2020_0",
    "date_time": "2020-04-15",
    "location": "Pecos, TX"
  },
  
  "section_1_safety_level_and_cause": {
    "safety_level": "HIGH",
    "primary_hazard": "GPS signal loss causing position drift",
    "observed_effect": "Uncontrolled lateral drift exceeding safe margins"
  },
  
  "section_2_design_constraints_and_recommendations": {
    "design_constraints": [
      "Require GPS satellite count >= 8 before flight",
      "Maximum wind speed limit: 10 m/s"
    ],
    "recommendations": [
      "Install secondary GPS module",
      "Enable automatic RTL on GPS degradation",
      "Pre-flight GPS signal quality check"
    ]
  },
  
  "section_3_explanation": {
    "reasoning": "The GPS signal loss caused the drone to drift laterally. Based on FAA incident analysis and simulation telemetry, this failure mode requires redundant positioning systems to prevent recurrence."
  },
  
  "verdict": {
    "decision": "CAUTION",
    "go_nogo": "CAUTION"
  },
  
  "supporting_data": {
    "simulation_config": {
      "waypoints_count": 4,
      "fault_type": "gps_dropout",
      "altitude_m": 50,
      "speed_ms": 5.0
    },
    "telemetry_summary": {
      "data_points": 1200,
      "duration_sec": 120.0,
      "max_altitude_m": 85.5,
      "max_roll_deg": 15.2
    }
  }
}
```

---

## 📂 Output Structure

Each sighting generates a structured output folder:

```
outputs/{sighting_id}_{timestamp}/
│
├── generated/                              # Raw LLM & simulation outputs
│   ├── full_configuration_output_from_llm.json   # 31-parameter config
│   └── full_telemetry_of_each_flight.json        # Complete telemetry
│
├── report/                                 # Final safety reports
│   ├── report.json                         # Structured report data
│   └── report.pdf                          # Professional PDF report
│
└── evaluation/                             # Research metrics (Excel here only)
    ├── evaluation.json                     # Per-sighting evaluation
    └── evaluation_*.xlsx                   # ESRI metrics spreadsheet
```

### Report Excel Sheets

1. **Summary** - Executive overview
2. **Sighting** - FAA sighting details
3. **Configuration** - LLM-generated config
4. **Telemetry** - Flight data summary
5. **Evaluation** - Research metrics

---

## 📝 Logging System

AeroGuardian uses a **centralized daily logging system** that captures:

### Log File Location

```
logs/2026-01-19.log    # Single daily log file
```

### Log Levels

| Level | Description |
|:------|:------------|
| INFO | Pipeline progress, step completion |
| DEBUG | Detailed function entry/exit |
| WARNING | Non-critical issues |
| ERROR | Failures with full traceback |

### LLM Request/Response Logging

```
====================================================================================================
[LLM REQUEST #1]
====================================================================================================
Timestamp:      2026-01-19T12:45:00
Model:          openai/gpt-4o
Signature:      GenerateFullPX4Config

INPUT FIELDS:
{
  "incident_description": "Drone lost control during climb...",
  "incident_location": "Pecos, TX",
  "incident_type": "flyaway"
}
====================================================================================================
```

### DSPy Optimization Tracking

```
====================================================================================================
[DSPY SIGNATURE] GenerateFullPX4Config
====================================================================================================
DOCSTRING (System Prompt):
Generate a complete PX4 SITL simulation configuration...

INPUT FIELDS (5):
- incident_description: str
- incident_location: str
...

OUTPUT FIELDS (30):
- fault_type: PX4 fault type to inject
- waypoints_csv: Mission waypoints
...
====================================================================================================
```

### Evaluation Metrics

| Category | Metrics |
|:---------|:--------|
| **Input Fidelity** | NLP extraction accuracy, geocoding success |
| **Simulation Validity** | Fault injection accuracy, telemetry quality |
| **Output Utility** | Hazard classification, recommendation quality |

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **FAA** - UAS Sighting Reports (2019-2025)
- **PX4 Autopilot** - SITL simulation framework
- **OpenAI** - GPT-4o language model
- **DSPy** - Structured LLM output framework
- **Stanford NLP** - DSPy research team

---

## Support

For issues or questions:
1. Check `logs/YYYY-MM-DD.log` for detailed error information
2. Review the [Troubleshooting Guide](docs/troubleshooting.md)
3. Open an issue on GitHub

---

*AeroGuardian - Preventing UAV Incidents Before They Happen*
