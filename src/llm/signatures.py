"""
DSPy Signatures for AeroGuardian
================================
Author: AeroGuardian Member
Date: 2026-01-30

This module contains ALL DSPy signatures for the 2-LLM pipeline:
  - FAA_To_PX4_Complete: LLM #1 - FAA incident → PX4 simulation config
  - GeneratePreFlightReport: LLM #2 - Telemetry → Safety report

USAGE:
    from src.llm.signatures import FAA_To_PX4_Complete, GeneratePreFlightReport
"""

import dspy


# =============================================================================
# LLM #1 - FAA Incident to PX4 Configuration
# =============================================================================

class FAA_To_PX4_Complete(dspy.Signature):
    """
    ROLE: You are an expert UAV flight dynamics engineer and FAA incident analyst.
    You have 15+ years of experience with:
    - PX4 autopilot systems and SITL simulation configuration
    - FAA UAS incident report analysis and interpretation
    - Drone flight dynamics, failure modes, and crash investigation
    - Coordinate geometry and waypoint planning for drone operations
    
    TASK: Translate an FAA UAS incident report into a COMPLETE and ACCURATE
    PX4 SITL simulation configuration that recreates the exact scenario described.
    
    CRITICAL RULES FOR HIGH-FIDELITY TRANSLATION:
    1. ALL parameters MUST be derived DIRECTLY from the FAA report text
    2. DO NOT use default or placeholder values unless data is truly missing
    3. Location coordinates MUST match the incident city/state precisely
    4. Failure mode must EXACTLY replicate what the report describes
    5. Waypoints must recreate the actual flight path described
    6. If critical information is genuinely missing, state "MISSING_FROM_REPORT"
    
    WAYPOINT GENERATION REQUIREMENTS:
    - Generate 3-7 waypoints that replicate the described flight
    - Format as JSON array: [{"lat": X, "lon": Y, "alt": Z, "action": "takeoff|waypoint|hover|land"}]
    - First waypoint MUST be takeoff, last MUST be land
    - All coordinates must be based on the extracted incident location
    - Altitude must match what the report describes
    
    PX4 FAULT INJECTION:
    - Match the fault type to the incident (GPS_DROPOUT, MOTOR_FAILURE, etc.)
    - Set onset time based on when the incident occurred during flight
    - Severity should reflect the described outcome (crash=1.0, recovery=0.5)
    
    OUTPUT QUALITY CHECK:
    - Every parameter must be physically realistic for drone operations
    - Coordinates must be valid lat/lon for the described location
    - Reasoning field must cite specific quotes from the report
    """
    
    # Input
    faa_report_text: str = dspy.InputField(
        desc="Complete FAA incident report text"
    )
    faa_incident_id: str = dspy.InputField(
        desc="FAA incident ID"
    )
    
    # Location (extracted from report)
    city: str = dspy.OutputField(desc="City from report, or 'UNKNOWN' if not found")
    state: str = dspy.OutputField(desc="State code from report (e.g., 'CA')")
    lat: float = dspy.OutputField(desc="Latitude for location (approximate)")
    lon: float = dspy.OutputField(desc="Longitude for location (approximate)")
    
    # Flight Profile (extracted from report)
    altitude_ft: float = dspy.OutputField(desc="Altitude in feet from report")
    altitude_m: float = dspy.OutputField(desc="Altitude in meters (ft * 0.3048)")
    speed_ms: float = dspy.OutputField(desc="Speed in m/s from report or inferred")
    flight_phase: str = dspy.OutputField(desc="Flight phase: takeoff|climb|cruise|hover|descent|landing")
    uas_type: str = dspy.OutputField(desc="Aircraft type from report")
    
    # Failure (extracted from report) 
    failure_mode: str = dspy.OutputField(
        desc="DETAILED failure in snake_case (e.g., 'motor_2_bearing_seizure')"
    )
    failure_category: str = dspy.OutputField(
        desc="Category: propulsion|navigation|power|control|environmental|structural"
    )
    failure_component: str = dspy.OutputField(
        desc="Component: motor_1, gps, battery, esc_2, propeller, etc."
    )
    failure_onset_sec: int = dspy.OutputField(
        desc="Seconds after takeoff when failure occurred"
    )
    
    # Observed behavior (from report)
    symptoms: str = dspy.OutputField(
        desc="Comma-separated symptoms: loss_of_control, rapid_descent, spinning, etc."
    )
    outcome: str = dspy.OutputField(
        desc="Outcome: crash, controlled_landing, flyaway, partial_recovery"
    )
    
    # Environment (from report)
    weather: str = dspy.OutputField(desc="Weather from report or 'not_mentioned'")
    wind_speed_ms: float = dspy.OutputField(desc="Wind speed in m/s")
    wind_direction_deg: float = dspy.OutputField(desc="Wind direction in degrees (0-360)")
    environment: str = dspy.OutputField(desc="Environment: urban, suburban, rural, etc.")
    
    # PX4 Commands (generated based on failure)
    px4_fault_cmd: str = dspy.OutputField(
        desc="PX4 fault command: 'failure inject <type> <component> <severity>'"
    )
    
    # Waypoints - LLM GENERATED (not hardcoded)
    waypoints_json: str = dspy.OutputField(
        desc='JSON array of waypoints replicating the flight. Format: [{"lat": X, "lon": Y, "alt": Z, "action": "takeoff|waypoint|hover|land"}]. Generate 3-7 waypoints based on the flight described.'
    )
    
    # Reasoning
    reasoning: str = dspy.OutputField(
        desc="Explain how each parameter was extracted from the FAA report"
    )


# =============================================================================
# LLM #2 - Pre-Flight Safety Report
# =============================================================================

class GeneratePreFlightReport(dspy.Signature):
    """
    ROLE: You are a senior aviation safety analyst specializing in UAS/drone 
    safety assessments. You have expertise in:
    - FAA UAS incident analysis and root cause investigation
    - Flight telemetry interpretation and anomaly detection
    - Pre-flight risk assessment and mitigation planning
    - Aviation safety regulations (14 CFR Part 107, FAA Order 8040.4B)
    
    TASK: Generate a structured 3-section pre-flight safety report by analyzing
    the FAA incident narrative, the FAULT TYPE THAT WAS SIMULATED, and the 
    simulation telemetry data.
    
    ================================================================================
    CRITICAL RULES FOR ACCURACY:
    ================================================================================
    
    RULE 1: PRIMARY HAZARD MUST MATCH FAULT_TYPE INPUT
      - If fault_type = 'motor_failure', primary_hazard MUST be motor-related
      - If fault_type = 'gps_loss', primary_hazard MUST be GPS/navigation-related
      - If fault_type = 'battery_failure', primary_hazard MUST be power-related
      - DO NOT "hallucinate" different hazards than what was actually simulated
    
    RULE 2: BE HONEST ABOUT SIMULATION RESULTS
      - If telemetry shows NORMAL flight but fault_type indicates a failure was 
        expected, state: "Simulation did not reproduce expected [fault_type] 
        behavior. Further testing required."
      - DO NOT claim anomalies that are not actually in the telemetry
    
    RULE 3: MATCH FAA INCIDENT OUTCOME
      - If FAA report says "crashed", your analysis should explain crash risk
      - If FAA report says "recovered", acknowledge recovery potential
    
    RULE 4: RECOMMENDATIONS MUST BE RELEVANT TO FAULT_TYPE
      - If fault_type = 'motor_failure', recommend motor redundancy, NOT GPS fixes
      - All constraints and recommendations must address the actual fault type
    
    ================================================================================
    
    REPORT STRUCTURE:
    
    SECTION 1 - SAFETY LEVEL & CAUSE:
    - Determine severity: CRITICAL, HIGH, MEDIUM, or LOW
    - PRIMARY HAZARD must match the fault_type input
    - Describe actual effects observed in telemetry
    
    SECTION 2 - DESIGN CONSTRAINTS & RECOMMENDATIONS:
    - 2-4 constraints relevant to the fault_type
    - 3-5 recommendations to prevent this specific failure type
    
    SECTION 3 - EXPLANATION (WHY):
    - Explain the reasoning connecting fault_type → telemetry → recommendations
    - Be honest if telemetry doesn't show expected failure behavior
    
    VERDICT:
    - GO: Safe to fly with this configuration
    - CAUTION: Proceed with additional precautions
    - NO-GO: Significant hazards, do not fly until resolved
    """
    
    # === INPUTS ===
    incident_description: str = dspy.InputField(
        desc="Original FAA incident narrative describing what happened"
    )
    incident_location: str = dspy.InputField(
        desc="Incident location (city, state)"
    )
    fault_type: str = dspy.InputField(
        desc="Fault type that was simulated: MOTOR_FAILURE, GPS_LOSS, BATTERY_FAILURE, CONTROL_LOSS, etc. Your primary_hazard MUST match this."
    )
    expected_outcome: str = dspy.InputField(
        desc="Expected outcome from FAA report: crash, controlled_landing, flyaway, recovery. Use this to inform your analysis."
    )
    telemetry_summary: str = dspy.InputField(
        desc="Telemetry analysis with metrics: duration_sec, max_altitude_m, position_drift_m, roll_max_deg, anomalies_detected"
    )
    
    # =========================================================================
    # SECTION 1: SAFETY LEVEL & CAUSE
    # =========================================================================
    safety_level: str = dspy.OutputField(
        desc="CRITICAL, HIGH, MEDIUM, or LOW. Based on severity of hazard and potential consequences."
    )
    primary_hazard: str = dspy.OutputField(
        desc="The main hazard - MUST match the fault_type input. If fault_type='motor_failure', hazard must be motor-related. If fault_type='gps_loss', hazard must be GPS-related. Example: 'Motor failure causing uncontrolled descent'"
    )
    observed_effect: str = dspy.OutputField(
        desc="Observable effect from simulation. Example: 'Loss of position stability and lateral drift exceeding 50m safety margin'"
    )
    
    # =========================================================================
    # SECTION 2: DESIGN CONSTRAINTS & RECOMMENDATIONS
    # =========================================================================
    design_constraints: str = dspy.OutputField(
        desc="2-4 SPECIFIC constraints relevant to the fault_type, separated by |. If motor_failure: motor redundancy constraints. If gps_loss: GPS constraints. Example for motor_failure: 'Do not operate with single-motor configurations | Require motor health monitoring'"
    )
    recommendations: str = dspy.OutputField(
        desc="3-5 ACTIONABLE recommendations for the specific fault_type, separated by |. If motor_failure: motor redundancy. If gps_loss: GPS backup. Example for motor_failure: 'Implement dual-motor failsafe | Pre-flight motor vibration check | Deploy parachute on motor anomaly'"
    )
    
    # =========================================================================
    # SECTION 3: EXPLANATION (WHY) - THE NOVELTY
    # =========================================================================
    explanation: str = dspy.OutputField(
        desc="3-5 sentences explaining WHY the system reached this conclusion. MUST include: (1) State the fault_type being analyzed, (2) Describe what telemetry shows (or acknowledge if it shows normal flight despite expected failure), (3) Connect to recommendations. If simulation didn't show expected failure behavior, honestly state: 'Simulation did not reproduce [fault_type] behavior. Analysis based on FAA incident description and expected [expected_outcome].' "
    )
    
    # =========================================================================
    # FINAL VERDICT
    # =========================================================================
    verdict: str = dspy.OutputField(
        desc="GO, CAUTION, or NO-GO. Final pre-flight decision."
    )


# =============================================================================
# Signature Registry
# =============================================================================

SIGNATURES = {
    "faa_to_px4": FAA_To_PX4_Complete,
    "generate_report": GeneratePreFlightReport,
}

__all__ = [
    "FAA_To_PX4_Complete",
    "GeneratePreFlightReport",
    "SIGNATURES",
]
