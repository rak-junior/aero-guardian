"""
DSPy Signatures for AeroGuardian
================================
Author: AeroGuardian Member
Date: 2026-01-30
Updated: 2026-01-31

This module contains ALL DSPy signatures for the 2-LLM pipeline:
  - FAA_To_PX4_Complete: LLM #1 - FAA UAS sighting → PX4 simulation config
  - GeneratePreFlightReport: LLM #2 - Telemetry → Safety report (JSON/PDF)

CONTEXT:
FAA UAS Sighting Reports document abnormal UAS operations and near-miss encounters
observed by pilots, air traffic controllers, and citizens. These are OBSERVATIONAL
REPORTS, not accident investigations. AeroGuardian translates these sightings into
physics-based simulations to generate pre-flight safety intelligence.

USAGE:
    from src.llm.signatures import FAA_To_PX4_Complete, GeneratePreFlightReport
"""

import dspy


# =============================================================================
# LLM #1 - FAA UAS Sighting to PX4 Simulation Configuration
# =============================================================================

class FAA_To_PX4_Complete(dspy.Signature):
    """
    ROLE: You are a senior UAV flight dynamics engineer with 15+ years of 
    experience in flight safety analysis. Your expertise includes:
    
    - PX4 autopilot systems and Software-In-The-Loop (SITL) simulation
    - FAA UAS sighting report interpretation and operational anomaly analysis
    - UAS flight dynamics, failure mode characterization, and hazard analysis
    - Geospatial coordinate systems and mission waypoint planning
    
    CONTEXT: FAA UAS Sighting Reports document OBSERVATIONAL data about abnormal 
    UAS operations and near-miss encounters in the National Airspace System. 
    These are sightings reported by pilots, controllers, and citizens - NOT 
    accident investigations. Your task is to reconstruct what COULD have been 
    happening with the observed UAS to cause the reported behavior.
    
    TASK: Translate an FAA UAS sighting report into an executable PX4 SITL 
    simulation configuration that reconstructs the operational anomaly described.
    
    ANALYSIS APPROACH:
    1. PARSE the sighting narrative for location, altitude, UAS behavior
    2. INFER the likely operational anomaly type from described behavior:
       - "erratic movement" → control_loss or navigation issue
       - "appeared to lose control" → motor_failure or control_loss
       - "hovered then descended rapidly" → battery_failure or motor issue
       - "flew away from operator" → control_loss or gps_loss
    3. GENERATE realistic simulation parameters that would produce similar behavior
    4. CREATE waypoints that replicate the described flight path
    
    PARAMETER EXTRACTION RULES:
    1. Location: Extract city/state, generate approximate lat/lon
    2. Altitude: Convert reported altitude to meters, cap at 120m for simulation
    3. Failure Mode: Infer from behavioral description (see examples below)
    4. Environment: Note weather, time of day, proximity to airports
    5. If information is missing, use physically realistic defaults
    
    FAILURE MODE INFERENCE EXAMPLES:
    - "UAS observed spinning" → motor_failure (asymmetric thrust)
    - "drone flew erratically" → gps_loss or control_loss  
    - "appeared to lose power" → battery_failure
    - "UAS hovering near runway" → geofence_violation (healthy drone, wrong location)
    - "drone seen at high altitude" → altitude_violation
    
    WAYPOINT GENERATION:
    - Generate 4-6 waypoints that replicate the described flight
    - Format: [{"lat": X, "lon": Y, "alt": Z, "action": "takeoff|waypoint|hover|land"}]
    - First waypoint = takeoff, last = land
    - Include hover point if "hovering" mentioned
    - Altitude in meters (max 120m for drone simulation realism)
    
    OUTPUT QUALITY:
    - All coordinates must be realistic for the reported location
    - Failure parameters must produce behavior matching the sighting description
    - Reasoning must explain how you interpreted the sighting report
    """
    
    # =========================================================================
    # INPUTS
    # =========================================================================
    faa_report_text: str = dspy.InputField(
        desc="Complete FAA UAS sighting report text describing the observed event"
    )
    faa_report_id: str = dspy.InputField(
        desc="FAA sighting report ID for traceability"
    )
    
    # =========================================================================
    # LOCATION (extracted from sighting)
    # =========================================================================
    city: str = dspy.OutputField(
        desc="City from sighting report, or 'UNKNOWN' if not specified"
    )
    state: str = dspy.OutputField(
        desc="Two-letter state code from report (e.g., 'CA', 'TX')"
    )
    lat: float = dspy.OutputField(
        desc="Latitude for the sighting location (approximate geocoded value)"
    )
    lon: float = dspy.OutputField(
        desc="Longitude for the sighting location (approximate geocoded value)"
    )
    
    # =========================================================================
    # FLIGHT PROFILE (extracted/inferred from sighting)
    # =========================================================================
    altitude_ft: float = dspy.OutputField(
        desc="Reported altitude in feet. Use 200-400ft if not specified."
    )
    altitude_m: float = dspy.OutputField(
        desc="Altitude converted to meters (ft × 0.3048). Max: 120m for simulation."
    )
    speed_ms: float = dspy.OutputField(
        desc="Estimated UAS speed in m/s. Typical: 5-15 m/s for consumer drones."
    )
    flight_phase: str = dspy.OutputField(
        desc="Inferred flight phase: takeoff | climb | cruise | hover | descent | landing"
    )
    uas_type: str = dspy.OutputField(
        desc="UAS type if mentioned, else 'unknown_multirotor' or 'unknown_fixed_wing'"
    )
    
    # =========================================================================
    # OPERATIONAL ANOMALY (inferred from sighting behavior)
    # =========================================================================
    failure_mode: str = dspy.OutputField(
        desc="Inferred failure mode in snake_case. MUST be one of: motor_failure, gps_loss, gps_dropout, battery_failure, battery_depletion, control_loss, control_signal_loss, sensor_failure, compass_error, geofence_violation, altitude_violation, flyaway. Base inference ONLY on described UAS behavior in the sighting report. Do NOT guess."
    )
    failure_category: str = dspy.OutputField(
        desc="Category. MUST be one of: propulsion | navigation | power | control | environmental | airspace_violation. Match to failure_mode: motor_failure→propulsion, gps_loss→navigation, battery_*→power, control_*→control, geofence/altitude→airspace_violation."
    )
    failure_component: str = dspy.OutputField(
        desc="Primary affected component. MUST be one of: motor | gps | battery | rc_link | esc | compass | baro | imu | none. If unclear from sighting, use the most likely component for the failure_mode."
    )
    failure_onset_sec: int = dspy.OutputField(
        desc="Estimated seconds after takeoff when anomaly likely occurred. If sighting mentions 'during cruise'→60-120s, 'during takeoff'→10-30s, 'during landing'→120-180s. Default: 60s."
    )
    
    # =========================================================================
    # OBSERVED BEHAVIOR (from sighting description)
    # =========================================================================
    symptoms: str = dspy.OutputField(
        desc="Comma-separated behavioral symptoms described: erratic_movement, rapid_descent, hovering, spinning, flyaway, altitude_violation, approach_proximity"
    )
    outcome: str = dspy.OutputField(
        desc="Observed/likely outcome: unknown | landed | crashed | flew_away | recovered_by_operator"
    )
    
    # =========================================================================
    # ENVIRONMENTAL CONDITIONS
    # =========================================================================
    weather: str = dspy.OutputField(
        desc="Weather conditions if mentioned, else 'not_specified'"
    )
    wind_speed_ms: float = dspy.OutputField(
        desc="Wind speed in m/s. Use 3-5 m/s if not specified (typical conditions)."
    )
    wind_direction_deg: float = dspy.OutputField(
        desc="Wind direction 0-360 degrees. Use 270 (westerly) as default."
    )
    environment: str = dspy.OutputField(
        desc="Environment type: urban | suburban | rural | airport_vicinity | industrial"
    )
    
    # =========================================================================
    # PX4 SIMULATION COMMANDS
    # =========================================================================
    px4_fault_cmd: str = dspy.OutputField(
        desc="PX4 fault injection command. Format: 'failure <component> <type> [-i <instance>]'. Valid components: motor, gps, mag, baro, gyro, accel. Valid types: off, stuck, garbage, wrong, slow, delayed. Examples: 'failure motor off -i 1' (motor 1 off), 'failure gps off' (GPS loss), 'failure mag stuck' (compass stuck). Use 'none' if failure cannot be injected via PX4 shell (e.g., RC loss, battery sag)."
    )
    
    # =========================================================================
    # MISSION WAYPOINTS (LLM-generated)
    # =========================================================================
    waypoints_json: str = dspy.OutputField(
        desc='JSON array of 4-6 waypoints replicating the flight. Format: [{"lat": X, "lon": Y, "alt": Z, "action": "takeoff|waypoint|hover|land"}]. First=takeoff, last=land. Altitudes in meters, max 120m.'
    )
    
    # =========================================================================
    # REASONING (critical for traceability)
    # =========================================================================
    reasoning: str = dspy.OutputField(
        desc="Explain your analysis: (1) What behavior in the sighting report led to your failure_mode inference? (2) How did you determine the flight profile? (3) What assumptions did you make for missing data?"
    )


# =============================================================================
# LLM #2 - Pre-Flight Safety Report Generation
# =============================================================================

class GeneratePreFlightReport(dspy.Signature):
    """
    ROLE: You are a senior UAS safety analyst with 15+ years of experience in:
    
    - FAA UAS sighting report analysis and operational anomaly investigation
    - Flight telemetry interpretation and anomaly detection
    - Pre-flight risk assessment and safety management systems (SMS)
    - Aviation regulations: 14 CFR Part 107, FAA Order 8040.4B, DO-178C
    
    CONTEXT: AeroGuardian reconstructs FAA UAS sighting reports in physics-based
    simulation to generate evidence-backed pre-flight safety intelligence. Your 
    task is to synthesize the original sighting narrative, the simulated fault 
    type, and telemetry data into an actionable safety report.
    
    TASK: Generate a structured 3-section pre-flight safety report:
    
    ============================================================================
    SECTION 1: SAFETY LEVEL & ROOT CAUSE
    ============================================================================
    Determine severity (CRITICAL/HIGH/MEDIUM/LOW) based on:
    - Potential consequences if this anomaly occurred in flight
    - Proximity to people, property, or controlled airspace
    - Recovery potential based on simulation evidence
    
    Primary hazard MUST align with the fault_type that was simulated.
    
    ============================================================================
    SECTION 2: DESIGN CONSTRAINTS & RECOMMENDATIONS  
    ============================================================================
    Provide ACTIONABLE guidance for UAS operators:
    
    Design Constraints (2-4 items):
    - Operational limitations to mitigate the identified hazard
    - Example: "Do not operate single-motor configurations in urban areas"
    
    Recommendations (3-5 items):
    - Engineering mitigations and procedural safeguards
    - MUST be relevant to the specific fault_type
    - Example for motor_failure: "Implement redundant propulsion systems"
    
    ============================================================================
    SECTION 3: EVIDENCE-BASED EXPLANATION
    ============================================================================
    Connect the dots: fault_type → telemetry observations → recommendations
    
    Be HONEST about simulation results:
    - If telemetry shows anomalies matching the fault, describe them
    - If telemetry appears normal despite expected failure, state:
      "Simulation did not reproduce expected [fault_type] behavior. Analysis
       based on FAA sighting description and aerospace engineering principles."
    
    ============================================================================
    FINAL VERDICT
    ============================================================================
    - GO: Safe to fly with current configuration
    - CAUTION: Proceed with additional monitoring/precautions  
    - NO-GO: Unacceptable risk, do not fly until hazard is mitigated
    
    ============================================================================
    CRITICAL ACCURACY RULES
    ============================================================================
    1. Primary hazard MUST match the fault_type input
    2. Recommendations MUST address the specific fault_type
    3. Do NOT claim telemetry anomalies that aren't in the data
    4. Be honest about simulation limitations
    """
    
    # =========================================================================
    # INPUTS
    # =========================================================================
    incident_description: str = dspy.InputField(
        desc="Original FAA UAS sighting report narrative describing the observed event"
    )
    incident_location: str = dspy.InputField(
        desc="Sighting location (city, state)"
    )
    fault_type: str = dspy.InputField(
        desc="Fault type simulated: MOTOR_FAILURE, GPS_LOSS, BATTERY_FAILURE, CONTROL_LOSS, SENSOR_FAULT, GEOFENCE_VIOLATION. Your primary_hazard MUST align with this."
    )
    expected_outcome: str = dspy.InputField(
        desc="Expected outcome from sighting: crash, controlled_landing, flyaway, recovery, unknown"
    )
    telemetry_summary: str = dspy.InputField(
        desc="Comprehensive telemetry analysis with sections: FLIGHT DURATION (seconds, data points), ALTITUDE (max, avg, deviation, stability), ATTITUDE STABILITY (max roll/pitch degrees, std dev), POSITION & GPS (drift meters, variance, satellites), SPEED (max/avg m/s), VIBRATION (max/avg), BATTERY (start/end voltage, sag rate), FAILSAFE EVENTS, ANOMALY DETECTION (severity: NONE/LOW/MEDIUM/HIGH/CRITICAL, specific anomalies detected). Use these metrics to support your hazard assessment."
    )
    
    # =========================================================================
    # SECTION 1: SAFETY LEVEL & ROOT CAUSE
    # =========================================================================
    safety_level: str = dspy.OutputField(
        desc="CRITICAL (life safety risk), HIGH (significant property/operational risk), MEDIUM (operational impact), or LOW (minimal impact with mitigations). Based on SIMULATED scenario, not verified incident analysis."
    )
    primary_hazard: str = dspy.OutputField(
        desc="SIMULATED hazard scenario (inferred from FAA narrative, not confirmed). MUST match fault_type. Start with 'Simulated:' prefix. Example for motor_failure: 'Simulated: Motor failure scenario producing loss of control'. Example for gps_loss: 'Simulated: Navigation degradation scenario'."
    )
    observed_effect: str = dspy.OutputField(
        desc="Effects observed IN SIMULATION (not actual incident). Prefix with 'In simulation:'. Example: 'In simulation: Motor failure produced 15-degree roll excursion and spiral descent at 3 m/s'. Do NOT claim to reproduce actual incident behavior."
    )
    
    # =========================================================================
    # CAUSAL ANALYSIS
    # =========================================================================
    primary_failure_subsystem: str = dspy.OutputField(
        desc="Primary failure subsystem based on TEMPORAL ORDERING of anomalies (which subsystem showed abnormal behavior FIRST). MUST be one of: navigation | control | propulsion | sensor | power | undetermined. Base this on which subsystem failed FIRST, not on crash severity. If insufficient evidence, use 'undetermined'."
    )
    causal_chain: str = dspy.OutputField(
        desc="Causal chain from root cause to outcome. Format: 'subsystem1 → subsystem2 → outcome'. Example: 'navigation (GPS loss @ t=32s) → control (attitude drift @ t=45s) → crash'. The FIRST item must match primary_failure_subsystem. If undetermined, use 'insufficient_evidence'."
    )
    subsystem_evidence_summary: str = dspy.OutputField(
        desc="Evidence table showing anomalies per subsystem with timestamps, ordered by first detection time. Format: 'Navigation: [anomaly1 @ t=Xs, anomaly2 @ t=Ys] | Control: [anomaly3 @ t=Zs] | Propulsion: [none] | ...'. Include only subsystems with detected anomalies."
    )
    
    # =========================================================================
    # SECTION 2: DESIGN CONSTRAINTS & RECOMMENDATIONS
    # =========================================================================
    design_constraints: str = dspy.OutputField(
        desc="2-4 SPECIFIC operational constraints relevant to fault_type, separated by |. Prefix each with 'Consider:'. Example for motor_failure: 'Consider: Maintain minimum altitude of 30m AGL for recovery time | Consider: Limit operations over populated areas | Consider: Visual line of sight operations'."
    )
    recommendations: str = dspy.OutputField(
        desc="3-5 SUGGESTED engineering/procedural recommendations for the specific fault_type, separated by |. Prefix each with 'Consider:'. These are scenario-based suggestions, not regulatory requirements. Example for motor_failure: 'Consider: Redundant motor configuration may reduce single-point failure risk | Consider: Pre-flight vibration monitoring | Consider: Parachute recovery system'."
    )
    
    # =========================================================================
    # SECTION 3: EVIDENCE-BASED EXPLANATION
    # =========================================================================
    explanation: str = dspy.OutputField(
        desc="3-5 sentences explaining the SIMULATION analysis: (1) State this is a SIMULATED fault_type scenario, (2) Describe SIMULATION telemetry evidence, (3) Acknowledge this is proxy simulation, not incident reconstruction, (4) Connect simulation results to safety_level, (5) Note recommendations are suggestions. Example: 'This analysis simulated a motor failure scenario. In simulation, the fault produced 12-degree roll deviation. NOTE: This represents quadrotor dynamics and may not match the reported aircraft. Recommendations are suggested mitigations for similar scenarios.'"
    )
    
    # =========================================================================
    # FINAL VERDICT
    # =========================================================================
    verdict: str = dspy.OutputField(
        desc="GO (no anomalies detected in simulation - standard pre-flight checks apply), CAUTION (simulation identified conditions warranting enhanced monitoring), or NO-GO (simulation indicates elevated risk - address identified hazards before flight). Include brief justification. NOTE: This is decision support only, not operational approval."
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
