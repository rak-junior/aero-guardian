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
    faa_incident_id: str = dspy.InputField(
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
        desc="Inferred failure mode in snake_case (e.g., 'motor_failure', 'gps_loss', 'battery_depletion', 'control_signal_loss'). Base this on described UAS behavior."
    )
    failure_category: str = dspy.OutputField(
        desc="Category: propulsion | navigation | power | control | environmental | airspace_violation"
    )
    failure_component: str = dspy.OutputField(
        desc="Affected component: motor | gps | battery | rc_link | esc | compass | none"
    )
    failure_onset_sec: int = dspy.OutputField(
        desc="Seconds after takeoff when anomaly likely occurred. Default: 30-60s."
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
        desc="PX4 fault injection command. Format: 'failure inject <type> <component> <severity>'. Example: 'failure inject motor motor1 100'. Use 'none' if simulating geofence/airspace violation."
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
        desc="Telemetry analysis summary: duration_sec, max_altitude_m, position_drift_m, attitude_excursions, anomalies_detected (list), mission_success (bool)"
    )
    
    # =========================================================================
    # SECTION 1: SAFETY LEVEL & ROOT CAUSE
    # =========================================================================
    safety_level: str = dspy.OutputField(
        desc="CRITICAL (life safety risk), HIGH (significant property/operational risk), MEDIUM (operational impact), or LOW (minimal impact with mitigations)"
    )
    primary_hazard: str = dspy.OutputField(
        desc="Primary hazard - MUST match fault_type. Example for motor_failure: 'Propulsion system failure causing loss of control'. Example for gps_loss: 'Navigation degradation causing position uncertainty'."
    )
    observed_effect: str = dspy.OutputField(
        desc="Observable effects from telemetry or expected effects based on aerospace engineering. Example: 'Asymmetric thrust produced 15-degree roll excursion and spiral descent at 3 m/s'."
    )
    
    # =========================================================================
    # SECTION 2: DESIGN CONSTRAINTS & RECOMMENDATIONS
    # =========================================================================
    design_constraints: str = dspy.OutputField(
        desc="2-4 SPECIFIC operational constraints relevant to fault_type, separated by |. Example for motor_failure: 'Maintain minimum altitude of 30m AGL for recovery time | Do not fly over populated areas without parachute | Limit operations to visual line of sight'."
    )
    recommendations: str = dspy.OutputField(
        desc="3-5 ACTIONABLE engineering/procedural recommendations for the specific fault_type, separated by |. Example for motor_failure: 'Implement redundant motor configuration (6+ motors) | Install pre-flight vibration monitoring | Deploy automatic parachute system | Conduct motor health checks every 10 flight hours'."
    )
    
    # =========================================================================
    # SECTION 3: EVIDENCE-BASED EXPLANATION
    # =========================================================================
    explanation: str = dspy.OutputField(
        desc="3-5 sentences explaining the analysis chain: (1) State the fault_type being analyzed, (2) Describe telemetry evidence OR honestly acknowledge if telemetry doesn't show expected failure, (3) Connect evidence to safety_level determination, (4) Justify recommendations. Example: 'Motor failure analysis revealed asymmetric thrust patterns with 12-degree roll deviation. Telemetry confirmed loss of altitude stability. CRITICAL rating reflects proximity to urban area. Redundant propulsion recommended.'"
    )
    
    # =========================================================================
    # FINAL VERDICT
    # =========================================================================
    verdict: str = dspy.OutputField(
        desc="GO (acceptable risk), CAUTION (proceed with additional precautions), or NO-GO (unacceptable risk - do not fly). Include brief justification."
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
