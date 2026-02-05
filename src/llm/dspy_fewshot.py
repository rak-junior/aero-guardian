"""
DSPy Few-Shot Examples for AeroGuardian
========================================
Author: AeroGuardian Member
Date: 2026-01-21
Updated: 2026-01-28

Pre-defined few-shot examples to improve LLM accuracy through in-context learning.
These examples are automatically injected into DSPy prompts.

IMPORTANT: Field names must EXACTLY match the DSPy signature OutputField names!

COVERAGE (5 fault types):
1. motor_failure - Propulsion system issue
2. gps_loss - Navigation/position failure  
3. control_loss - RC link or control system failure
4. battery_failure - Power system issue
5. sensor_fault - IMU/compass/barometer failure

NEW REPORT STRUCTURE (3 Sections):
1. Safety Level & Cause
2. Design Constraints & Recommendations
3. Explanation (Why)
"""

import dspy
from typing import List


# =============================================================================
# Few-Shot Examples for FAA -> PX4 Config Generation (GenerateFullPX4Config)
# COVERAGE: 5 fault types, diverse scenarios
# =============================================================================

FAA_TO_PX4_EXAMPLES = [
    {
        # Example 1: Motor Failure - Crash scenario
        "input": {
            "faa_report_text": """PRELIM INFO FROM FAA OPS: PHOENIX, AZ/UAS INCIDENT/1045M
A DJI Inspire 2 drone conducting aerial survey at 400 feet AGL reported sudden 
loss of power to motor 3. Aircraft began spinning and descended rapidly, crashing 
into an open field. No injuries. Weather clear, winds SW at 8 mph.""",
            "faa_report_id": "FAA_PHOENIX_AZ_001",
        },
        "output": {
            "start_lat": 33.4484,
            "start_lon": -112.0740,
            "takeoff_altitude_m": 30.0,
            "max_altitude_m": 121.92,
            "flight_mode": "CRUISE",
            "mission_duration_sec": 180,
            "cruise_speed_ms": 10.0,
            "waypoints_csv": "33.4484,-112.0740,10,takeoff;33.4494,-112.0750,122,waypoint;33.4504,-112.0740,122,waypoint;33.4484,-112.0740,10,land",
            "fault_type": "MOTOR_FAILURE",
            "fault_onset_sec": 120,
            "fault_duration_sec": -1,
            "affected_motor": "motor_3",
            "affected_components": "motor,propeller,esc",
            "wind_speed_ms": 3.6,
            "wind_direction_deg": 225.0,
            "wind_gust_ms": 5.4,
            "turbulence_intensity": 0.2,
            "temperature_c": 30.0,
            "visibility_m": 10000.0,
            "gps_satellite_count": 10,
            "gps_hdop": 1.2,
            "gps_noise_m": 1.0,
            "battery_cells": 4,
            "battery_capacity_mah": 5000,
            "battery_start_pct": 90.0,
            "battery_sag_rate": 0.1,
            "failsafe_action": "LAND",
            "rtl_altitude_m": 50.0,
            "geofence_radius_m": 500.0,
            "incident_phase": "cruise",
            "estimated_flight_time_sec": 120,
            "parameter_sources": "altitude=400ft <- 'at 400 feet AGL'; motor=3 <- 'loss of power to motor 3'; wind=8mph SW <- 'winds SW at 8 mph'",
            "px4_fault_cmd": "failure motor off -i 3",
            "reasoning": "Altitude 400ft = 121.92m. Motor 3 failure caused spinning and crash. Weather clear, SW winds at 3.6 m/s. Crash in open field indicates rural environment."
        }
    },
    {
        # Example 2: GPS Loss - Urban canyon
        "input": {
            "faa_report_text": """PRELIM INFO FROM FAA OPS: DENVER, CO/UAS INCIDENT/1400M
During commercial photography at 200ft near downtown buildings, Yuneec H520 
experienced complete GPS signal loss. Drone drifted 150 meters from intended 
position before pilot manually landed. Temperature 45°F, overcast. No damage.""",
            "faa_report_id": "FAA_SEATTLE_WA_005",
        },
        "output": {
            "start_lat": 39.7392,
            "start_lon": -104.9903,
            "takeoff_altitude_m": 20.0,
            "max_altitude_m": 60.96,
            "flight_mode": "HOVER",
            "mission_duration_sec": 300,
            "cruise_speed_ms": 5.0,
            "waypoints_csv": "39.7392,-104.9903,10,takeoff;39.7402,-104.9903,61,hover;39.7392,-104.9903,10,land",
            "fault_type": "GPS_DROPOUT",
            "fault_onset_sec": 180,
            "fault_duration_sec": 60,
            "affected_motor": "none",
            "affected_components": "gps,navigation",
            "wind_speed_ms": 2.0,
            "wind_direction_deg": 0.0,
            "wind_gust_ms": 3.0,
            "turbulence_intensity": 0.3,
            "temperature_c": 7.2,
            "visibility_m": 5000.0,
            "gps_satellite_count": 4,
            "gps_hdop": 8.0,
            "gps_noise_m": 5.0,
            "battery_cells": 4,
            "battery_capacity_mah": 5000,
            "battery_start_pct": 95.0,
            "battery_sag_rate": 0.08,
            "failsafe_action": "RTL",
            "rtl_altitude_m": 40.0,
            "geofence_radius_m": 300.0,
            "incident_phase": "hover",
            "estimated_flight_time_sec": 180,
            "parameter_sources": "altitude=200ft <- 'at 200ft'; drift=150m <- 'drifted 150 meters'; temp=45F <- 'Temperature 45°F'",
            "px4_fault_cmd": "failure gps off",
            "reasoning": "Altitude 200ft = 60.96m. Downtown buildings caused GPS multipath. 150m drift indicates complete GPS loss. Urban canyon environment with reduced satellite visibility."
        }
    },
    {
        # Example 3: Control Loss - RC link failure
        "input": {
            "faa_report_text": """PRELIM INFO FROM FAA OPS: MIAMI, FL/UAS INCIDENT/1130E
Operator reported losing RC signal during inspection flight at 300ft over construction site.
Drone entered RTL mode but descended erratically. Emergency landing on rooftop.
Winds gusting to 25 mph. Minor damage to landing gear.""",
            "faa_report_id": "FAA_MIAMI_FL_003",
        },
        "output": {
            "start_lat": 25.7617,
            "start_lon": -80.1918,
            "takeoff_altitude_m": 15.0,
            "max_altitude_m": 91.44,
            "flight_mode": "MISSION",
            "mission_duration_sec": 240,
            "cruise_speed_ms": 8.0,
            "waypoints_csv": "25.7617,-80.1918,10,takeoff;25.7627,-80.1928,91,waypoint;25.7637,-80.1918,91,waypoint;25.7617,-80.1918,10,land",
            "fault_type": "RC_LOSS",
            "fault_onset_sec": 150,
            "fault_duration_sec": -1,
            "affected_motor": "none",
            "affected_components": "rc_receiver,control_link",
            "wind_speed_ms": 8.9,
            "wind_direction_deg": 180.0,
            "wind_gust_ms": 11.2,
            "turbulence_intensity": 0.5,
            "temperature_c": 28.0,
            "visibility_m": 8000.0,
            "gps_satellite_count": 12,
            "gps_hdop": 1.0,
            "gps_noise_m": 0.8,
            "battery_cells": 6,
            "battery_capacity_mah": 8000,
            "battery_start_pct": 85.0,
            "battery_sag_rate": 0.12,
            "failsafe_action": "RTL",
            "rtl_altitude_m": 60.0,
            "geofence_radius_m": 400.0,
            "incident_phase": "cruise",
            "estimated_flight_time_sec": 150,
            "parameter_sources": "altitude=300ft <- 'at 300ft'; wind=25mph <- 'gusting to 25 mph'; outcome=RTL <- 'entered RTL mode'",
            "px4_fault_cmd": "none",
            "reasoning": "Altitude 300ft = 91.44m. RC signal lost during construction inspection. High winds (11.2 m/s gusts) caused erratic descent during RTL. Urban environment with potential RF interference."
        }
    },
    {
        # Example 4: Battery Failure - Critical low voltage
        "input": {
            "faa_report_text": """PRELIM INFO FROM FAA OPS: LAS VEGAS, NV/UAS INCIDENT/1545P
Matrice 300 RTK performing powerline inspection experienced rapid battery voltage drop
at 500ft. Operator observed voltage drop from 46V to 38V in 30 seconds. 
Emergency descent initiated. Landed with 8% battery. No damage."""  ,
            "faa_report_id": "FAA_LAS_VEGAS_NV_004",
        },
        "output": {
            "start_lat": 36.1699,
            "start_lon": -115.1398,
            "takeoff_altitude_m": 25.0,
            "max_altitude_m": 152.4,
            "flight_mode": "MISSION",
            "mission_duration_sec": 600,
            "cruise_speed_ms": 12.0,
            "waypoints_csv": "36.1699,-115.1398,10,takeoff;36.1719,-115.1418,152,waypoint;36.1739,-115.1398,152,waypoint;36.1699,-115.1398,10,land",
            "fault_type": "BATTERY_FAILURE",
            "fault_onset_sec": 400,
            "fault_duration_sec": -1,
            "affected_motor": "none",
            "affected_components": "battery,power_system",
            "wind_speed_ms": 4.5,
            "wind_direction_deg": 270.0,
            "wind_gust_ms": 6.0,
            "turbulence_intensity": 0.3,
            "temperature_c": 38.0,
            "visibility_m": 15000.0,
            "gps_satellite_count": 14,
            "gps_hdop": 0.9,
            "gps_noise_m": 0.5,
            "battery_cells": 12,
            "battery_capacity_mah": 12000,
            "battery_start_pct": 100.0,
            "battery_sag_rate": 0.25,
            "failsafe_action": "LAND",
            "rtl_altitude_m": 80.0,
            "geofence_radius_m": 800.0,
            "incident_phase": "cruise",
            "estimated_flight_time_sec": 400,
            "parameter_sources": "altitude=500ft <- 'at 500ft'; voltage=46V->38V <- 'voltage drop from 46V to 38V'; time=30s <- 'in 30 seconds'",
            "px4_fault_cmd": "none",
            "reasoning": "Altitude 500ft = 152.4m. Rapid voltage drop (8V in 30s) indicates battery cell failure or extreme temperature stress. High ambient temp 38°C may have contributed. Emergency descent saved the aircraft."
        }
    },
    {
        # Example 5: Sensor Fault - IMU/Compass failure
        "input": {
            "faa_report_text": """PRELIM INFO FROM FAA OPS: SEATTLE, WA/UAS INCIDENT/0930P
DJI Phantom 4 Pro experienced compass error warning during mapping flight at 250ft.
Aircraft began flying in circles and drifted toward highway. Operator manually overrode
and landed in parking lot. Near magnetic interference from power substation.""",
            "faa_report_id": "FAA_SEATTLE_WA_005"
        },
        "output": {
            "start_lat": 47.6062,
            "start_lon": -122.3321,
            "takeoff_altitude_m": 10.0,
            "max_altitude_m": 76.2,
            "flight_mode": "MISSION",
            "mission_duration_sec": 360,
            "cruise_speed_ms": 6.0,
            "waypoints_csv": "47.6062,-122.3321,10,takeoff;47.6072,-122.3331,76,waypoint;47.6082,-122.3321,76,waypoint;47.6072,-122.3311,76,waypoint;47.6062,-122.3321,10,land",
            "fault_type": "SENSOR_FAILURE",
            "fault_onset_sec": 200,
            "fault_duration_sec": -1,
            "affected_motor": "none",
            "affected_components": "compass,magnetometer,imu",
            "wind_speed_ms": 3.0,
            "wind_direction_deg": 315.0,
            "wind_gust_ms": 4.5,
            "turbulence_intensity": 0.2,
            "temperature_c": 12.0,
            "visibility_m": 6000.0,
            "gps_satellite_count": 11,
            "gps_hdop": 1.3,
            "gps_noise_m": 1.2,
            "battery_cells": 4,
            "battery_capacity_mah": 5870,
            "battery_start_pct": 92.0,
            "battery_sag_rate": 0.09,
            "failsafe_action": "LAND",
            "rtl_altitude_m": 40.0,
            "geofence_radius_m": 350.0,
            "incident_phase": "cruise",
            "estimated_flight_time_sec": 200,
            "parameter_sources": "altitude=250ft <- 'at 250ft'; sensor=compass <- 'compass error warning'; cause=magnetic <- 'near magnetic interference from power substation'",
            "px4_fault_cmd": "failure mag stuck",
            "reasoning": "Altitude 250ft = 76.2m. Compass interference from power substation caused erratic circular flight pattern. Manual override prevented potential highway incident. Suburban environment with electromagnetic interference sources."
        }
    }
]


# =============================================================================
# Few-Shot Examples for Pre-Flight Safety Report (NEW 3-SECTION STRUCTURE)
# COVERAGE: 5 fault types, diverse severity levels
# =============================================================================

PREFLIGHT_REPORT_EXAMPLES = [
    {
        # Example 1: Motor Failure - CRITICAL (Crash)
        "input": {
            "incident_description": "DJI Inspire 2 lost motor 3 power at 400ft during aerial survey. Aircraft spun and crashed in open field. No injuries.",
            "incident_location": "Phoenix, Arizona",
            "fault_type": "MOTOR_FAILURE",
            "telemetry_summary": """=== FLIGHT TELEMETRY ANALYSIS ===
FLIGHT DURATION: 180.0 seconds (1800 data points)

--- ALTITUDE ---
  Max: 122.0m (400ft)
  Avg: 98.5m
  Deviation (max-min): 120.0m
  Stability (StdDev): 35.2m

--- ATTITUDE STABILITY ---
  Max Roll: 45.0deg (StdDev: 18.5deg)
  Max Pitch: 32.0deg (StdDev: 12.3deg)

--- POSITION & GPS ---
  Position Drift: 85.0m from start
  GPS Variance: 2.1m

--- ANOMALY DETECTION ---
  Severity: CRITICAL
  • high_vibration (motor imbalance detected)
  • control_saturation (actuator limits exceeded)
  • rapid_descent (>8 m/s vertical speed)"""
        },
        "output": {
            # SECTION 1: Safety Level & Cause
            "safety_level": "CRITICAL",
            "primary_hazard": "Single motor failure during cruise flight causing complete loss of control",
            "observed_effect": "Uncontrolled spin with roll deviation of 45deg and rapid descent rate exceeding 8 m/s, resulting in crash 85m from intended position",
            
            # SECTION 2: Design Constraints & Recommendations
            "design_constraints": "Do not operate single-motor-critical aircraft above 50m AGL | Avoid extended cruise missions without motor redundancy | Limit operations over hard surfaces without parachute system",
            "recommendations": "Enforce pre-flight motor spin-up test for each motor | Install motor temperature monitoring | Implement automatic failsafe landing on motor anomaly | Require octocopter configuration for survey missions above 100m",
            
            # SECTION 3: Explanation (Why) - Connects telemetry → constraints → recommendations
            "explanation": "Telemetry shows roll deviation exceeded 45deg with altitude loss of 120m during uncontrolled descent, triggering control saturation. The high_vibration anomaly indicates motor imbalance preceded the failure. This directly justifies the constraint 'Do not operate single-motor-critical aircraft above 50m AGL.' Implementing the recommended motor temperature monitoring would provide early warning, while requiring octocopter configuration would provide redundancy—reducing risk from CRITICAL to MEDIUM.",
            
            # Final Verdict
            "verdict": "NO-GO"
        }
    },
    {
        # Example 2: GPS Loss - HIGH (Controlled airspace incursion)
        "input": {
            "incident_description": "PRELIM INFO FROM FAA: DALLAS, TX/UAS at 2,200 feet observed by manned aircraft. No evasive action taken.",
            "incident_location": "Dallas, Texas",
            "fault_type": "GPS_DROPOUT",
            "telemetry_summary": """=== FLIGHT TELEMETRY ANALYSIS ===
FLIGHT DURATION: 300.0 seconds (3000 data points)

--- ALTITUDE ---
  Max: 671.0m (2200ft)
  Avg: 650.2m
  Deviation (max-min): 45.0m
  Stability (StdDev): 8.5m

--- ATTITUDE STABILITY ---
  Max Roll: 5.0deg (StdDev: 1.2deg)
  Max Pitch: 4.5deg (StdDev: 1.0deg)

--- POSITION & GPS ---
  Position Drift: 12.0m from start
  GPS Variance: 8.5m
  Satellites: min=4, avg=6.2
  HDOP Max: 7.8

--- ANOMALY DETECTION ---
  Severity: HIGH
  • gps_satellite_low (minimum satellites below 6)
  • hdop_exceeded (HDOP > 5.0)"""
        },
        "output": {
            # SECTION 1: Safety Level & Cause
            "safety_level": "HIGH",
            "primary_hazard": "Unauthorized UAS operation in controlled airspace with GPS degradation near manned aircraft",
            "observed_effect": "Stable flight attitude but GPS degradation detected with HDOP of 7.8 and satellite count dropping to 4, indicating unreliable positioning at 671m altitude in controlled airspace",
            
            # SECTION 2: Design Constraints & Recommendations
            "design_constraints": "Do not operate above 120m AGL without ATC authorization | Mandatory ADS-B receiver for operations near airports | GPS redundancy required for controlled airspace operations",
            "recommendations": "Install ADS-B In receiver for traffic awareness | Implement hard altitude ceiling at 120m AGL | Enable automatic RTL on GPS degradation (HDOP > 5.0) | Require LAANC authorization verification before flight",
            
            # SECTION 3: Explanation (Why) - Connects telemetry → constraints → recommendations
            "explanation": "Telemetry shows HDOP exceeded 7.8 with satellite count dropping to 4 during the flight at 671m altitude. This GPS unreliability in controlled airspace where manned aircraft operate justifies the constraint requiring GPS redundancy. The recommended ADS-B receiver would provide traffic awareness, while the hard altitude ceiling at 120m would prevent future incursions—reducing risk from HIGH to LOW.",
            
            # Final Verdict
            "verdict": "NO-GO"
        }
    },
    {
        # Example 3: Control Loss - HIGH (RC link failure with RTL)
        "input": {
            "incident_description": "Operator lost RC signal during bridge inspection at 300ft. Drone entered RTL mode but descended erratically in high winds. Emergency landing on rooftop with minor damage.",
            "incident_location": "Miami, Florida",
            "fault_type": "CONTROL_LOSS",
            "telemetry_summary": """=== FLIGHT TELEMETRY ANALYSIS ===
FLIGHT DURATION: 240.0 seconds (2400 data points)

--- ALTITUDE ---
  Max: 91.4m (300ft)
  Avg: 72.5m
  Deviation (max-min): 88.0m
  Stability (StdDev): 22.4m

--- ATTITUDE STABILITY ---
  Max Roll: 28.0deg (StdDev: 9.5deg)
  Max Pitch: 22.0deg (StdDev: 7.8deg)

--- SPEED ---
  Max: 12.5 m/s
  Avg: 6.8 m/s

--- POSITION & GPS ---
  Position Drift: 45.0m from start
  GPS Variance: 1.8m

--- ANOMALY DETECTION ---
  Severity: HIGH
  • rc_signal_lost (control link interrupted)
  • wind_compensation_saturated (attitude limits during RTL)
  • erratic_descent (altitude stability compromised)"""
        },
        "output": {
            # SECTION 1: Safety Level & Cause
            "safety_level": "HIGH",
            "primary_hazard": "RC link failure during inspection mission with subsequent erratic descent in high wind conditions",
            "observed_effect": "RTL failsafe activated but wind compensation saturated causing 28deg roll deviation and 88m altitude instability during emergency descent",
            
            # SECTION 2: Design Constraints & Recommendations
            "design_constraints": "Do not operate above 50m AGL when winds exceed 15 mph | Ensure clear RTL path before flight | Maintain visual line of sight for manual override capability",
            "recommendations": "Implement redundant control link (dual RC receivers) | Set conservative wind speed abort threshold at 12 m/s | Enable automatic land-in-place when RTL path obstructed | Test RTL behavior in similar wind conditions before mission",
            
            # SECTION 3: Explanation (Why) - Connects telemetry → constraints → recommendations
            "explanation": "Telemetry shows RC signal loss triggered RTL failsafe, but wind compensation saturated with roll exceeding 28deg during descent. The altitude deviation of 88m indicates the aircraft struggled to maintain stable descent. This justifies the constraint limiting operations above 50m AGL in high winds. The recommended redundant control link would prevent single-point failures, while testing RTL in similar conditions would validate safe recovery—reducing risk from HIGH to MEDIUM.",
            
            # Final Verdict
            "verdict": "NO-GO"
        }
    },
    {
        # Example 4: Battery Failure - MEDIUM (Precautionary landing)
        "input": {
            "incident_description": "Pilot reported minor battery voltage drop during inspection flight at 100ft. Landed safely with 35% battery remaining.",
            "incident_location": "Austin, Texas",
            "fault_type": "BATTERY_SAG",
            "telemetry_summary": """=== FLIGHT TELEMETRY ANALYSIS ===
FLIGHT DURATION: 420.0 seconds (4200 data points)

--- ALTITUDE ---
  Max: 30.5m (100ft)
  Avg: 28.2m
  Deviation (max-min): 5.2m
  Stability (StdDev): 1.1m

--- ATTITUDE STABILITY ---
  Max Roll: 3.0deg (StdDev: 0.8deg)
  Max Pitch: 2.5deg (StdDev: 0.6deg)

--- POSITION & GPS ---
  Position Drift: 2.0m from start
  GPS Variance: 0.5m

--- BATTERY ---
  Start: 16.8V
  End: 16.2V
  Sag Rate: 0.0014 V/s

--- ANOMALY DETECTION ---
  Severity: MEDIUM
  • battery_voltage_drop_0.6V (moderate sag during climb)"""
        },
        "output": {
            # SECTION 1: Safety Level & Cause
            "safety_level": "MEDIUM",
            "primary_hazard": "Battery voltage sag under load indicating aging battery or marginal cell health",
            "observed_effect": "0.6V voltage drop from 16.8V to 16.2V during flight with stable attitude, completed with 35% reserve indicating adequate margins",
            
            # SECTION 2: Design Constraints & Recommendations
            "design_constraints": "Maintain minimum 30% battery reserve for RTL | Limit flight duration to 70% of rated capacity for batteries over 100 cycles | Reduce mission scope if voltage sag exceeds 0.5V during climb",
            "recommendations": "Check battery internal resistance before flight | Replace batteries exceeding 150 charge cycles | Set conservative low-battery failsafe at 40% | Log battery health metrics after each flight",
            
            # SECTION 3: Explanation (Why) - Connects telemetry → constraints → recommendations
            "explanation": "Telemetry shows 0.6V voltage sag (16.8V to 16.2V) during flight with 35% landing reserve, which remains above the 30% minimum. The stable attitude (max roll 3deg) and minimal position drift (2m) indicate the flight was not compromised. This validates the constraint requiring 30% minimum battery reserve. The recommended internal resistance check would identify aging batteries before flight, while the 40% failsafe setting provides additional safety margin—risk level MEDIUM with CAUTION.",
            
            # Final Verdict
            "verdict": "CAUTION"
        }
    },
    {
        # Example 5: Sensor Fault - HIGH (Compass interference)
        "input": {
            "incident_description": "DJI Phantom experienced compass error during mapping near power substation. Aircraft flew in circles and drifted toward highway. Manual override and emergency landing.",
            "incident_location": "Seattle, Washington",
            "fault_type": "SENSOR_FAILURE",
            "telemetry_summary": """=== FLIGHT TELEMETRY ANALYSIS ===
FLIGHT DURATION: 200.0 seconds (2000 data points)

--- ALTITUDE ---
  Max: 76.2m (250ft)
  Avg: 68.5m
  Deviation (max-min): 15.0m
  Stability (StdDev): 3.8m

--- ATTITUDE STABILITY ---
  Max Roll: 18.0deg (StdDev: 6.2deg)
  Max Pitch: 15.0deg (StdDev: 5.1deg)

--- POSITION & GPS ---
  Position Drift: 120.0m from start
  GPS Variance: 2.5m
  Lateral Drift: 95.0m

--- ANOMALY DETECTION ---
  Severity: HIGH
  • compass_interference (magnetometer readings erratic)
  • heading_oscillation (yaw rate exceeded 30 deg/s)
  • circular_flight_pattern (unintended orbit behavior)"""
        },
        "output": {
            # SECTION 1: Safety Level & Cause
            "safety_level": "HIGH",
            "primary_hazard": "Compass interference from electromagnetic source causing loss of heading reference and uncontrolled circular flight pattern",
            "observed_effect": "Aircraft entered circular flight pattern with 120m position drift and 95m lateral drift toward highway, requiring manual override for emergency landing",
            
            # SECTION 2: Design Constraints & Recommendations
            "design_constraints": "Do not operate within 100m of high-voltage power infrastructure | Perform compass calibration before each flight | Abort mission if heading oscillation exceeds 15 deg/s",
            "recommendations": "Install dual-compass system for redundancy | Enable automatic switch to GPS-only navigation on compass fault | Survey operating area for electromagnetic interference sources before flight | Implement geofence exclusion zones around power substations",
            
            # SECTION 3: Explanation (Why) - Connects telemetry → constraints → recommendations
            "explanation": "Telemetry shows compass_interference anomaly with heading oscillation exceeding 30 deg/s, causing circular flight pattern and 120m position drift toward a highway. This dangerous behavior near traffic justifies the constraint requiring 100m separation from power infrastructure. The recommended dual-compass system would provide sensor redundancy, while geofence exclusion zones would prevent operations in known interference areas—reducing risk from HIGH to LOW.",
            
            # Final Verdict
            "verdict": "NO-GO"
        }
    }
]


def get_faa_to_px4_examples() -> List[dspy.Example]:
    """Get few-shot examples for FAA to PX4 config generation.
    
    Returns dspy.Example objects with input/output fields matching
    the GenerateFullPX4Config signature.
    
    COVERAGE: 5 fault types
    - motor_failure
    - gps_loss  
    - control_loss
    - battery_failure
    - sensor_fault
    """
    examples = []
    for ex in FAA_TO_PX4_EXAMPLES:
        example = dspy.Example(
            faa_report_text=ex["input"]["faa_report_text"],
            faa_report_id=ex["input"]["faa_report_id"],
            **ex["output"]
        ).with_inputs("faa_report_text", "faa_report_id")
        examples.append(example)
    return examples


def get_preflight_report_examples() -> List[dspy.Example]:
    """Get few-shot examples for pre-flight report generation (3-section structure).
    
    Returns dspy.Example objects with input/output fields matching
    the NEW GeneratePreFlightReport signature with:
    - Section 1: Safety Level & Cause
    - Section 2: Design Constraints & Recommendations
    - Section 3: Explanation (Why)
    
    COVERAGE: 5 fault types, diverse severity levels
    - MOTOR_FAILURE (CRITICAL)
    - GPS_DROPOUT (HIGH)
    - CONTROL_LOSS (HIGH)
    - BATTERY_SAG (MEDIUM - CAUTION verdict)
    - SENSOR_FAILURE (HIGH)
    """
    examples = []
    for ex in PREFLIGHT_REPORT_EXAMPLES:
        example = dspy.Example(
            **ex["input"],
            **ex["output"]
        ).with_inputs("incident_description", "incident_location", "fault_type", "telemetry_summary")
        examples.append(example)
    return examples
