"""
Scenario Fidelity Score (SFS)
=============================
Author: AeroGuardian Member
Date: 2026-01-21

Evaluates how well the LLM-generated PX4 configuration matches the original FAA report.

SCIENTIFIC RATIONALE:
---------------------
If the LLM misinterprets the FAA narrative, the entire pipeline is compromised.
SFS measures 5 dimensions of fidelity:
1. Fault type match - Does the simulated fault match what happened?
2. Trigger condition match - Are trigger conditions correctly extracted?
3. Environmental match - Are weather/location conditions consistent?
4. Temporal consistency - Is the flight phase/timeline aligned?
5. Parameter completeness - Are all required fields populated?

SCORING:
--------
Each dimension: 0.0 to 1.0
Final SFS = weighted average of dimensions
"""

import logging
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("AeroGuardian.Evaluator.SFS")


# =============================================================================
# FAULT TYPE MAPPINGS (for semantic matching)
# =============================================================================
# These mappings help score how well the LLM's inferred fault type matches
# the FAA report narrative. Keywords are searched in the lowercase report text.

FAULT_TYPE_KEYWORDS = {
    # Propulsion failures
    "motor_failure": ["motor", "propeller", "engine", "thrust", "spin", "crashed", "fell", "rotor"],
    "propulsion_failure": ["motor", "propeller", "engine", "thrust", "spin", "rotor", "esc"],
    
    # Navigation failures
    "gps_loss": ["gps", "satellite", "navigation", "position", "drift", "lost link", "erratic"],
    "gps_dropout": ["gps", "satellite", "navigation", "position", "drift"],
    "flyaway": ["flyaway", "flew away", "uncontrolled", "out of control", "lost control"],
    
    # Power/Battery failures
    "battery_failure": ["battery", "voltage", "power", "charge", "low power"],
    "battery_depletion": ["battery", "voltage", "power", "charge", "depleted", "low power"],
    
    # Control failures
    "control_loss": ["control", "malfunction", "flyaway", "unresponsive", "erratic"],
    "control_signal_loss": ["control", "signal", "link", "rc loss", "lost link"],
    "rc_loss": ["rc", "signal", "link", "lost link", "radio"],
    
    # Sensor failures
    "sensor_fault": ["sensor", "barometer", "compass", "imu", "gyro", "accelerometer"],
    "sensor_failure": ["sensor", "barometer", "compass", "imu", "gyro", "accelerometer"],
    "compass_error": ["compass", "heading", "mag", "magnetic", "yaw"],
    
    # Airspace violations - These are common in FAA sighting reports
    # No specific failure keywords - these are OBSERVED positions, not failures
    "altitude_violation": ["altitude", "feet", "ft", "high", "above", "elevated"],
    "geofence_violation": ["airspace", "airport", "approach", "runway", "near miss"],
    "airspace_violation": ["airspace", "airport", "approach", "runway", "near miss", "class"],
}

ENVIRONMENT_KEYWORDS = {
    "urban": ["city", "downtown", "building", "airport", "atct", "tower", "metropolitan"],
    "rural": ["field", "farm", "rural", "open", "country"],
    "suburban": ["residential", "neighborhood", "suburb"],
    "airport_vicinity": ["airport", "runway", "approach", "atct", "tower", "flight path"],
}


@dataclass
class SFSDimensions:
    """Individual dimension scores for SFS."""
    fault_type_match: float = 0.0
    trigger_condition_match: float = 0.0
    environmental_match: float = 0.0
    temporal_consistency: float = 0.0
    parameter_completeness: float = 0.0


@dataclass
class SFSResult:
    """Complete SFS evaluation result."""
    score: float = 0.0
    dimensions: SFSDimensions = field(default_factory=SFSDimensions)
    matched_fault_type: str = ""
    extracted_keywords: List[str] = field(default_factory=list)
    missing_parameters: List[str] = field(default_factory=list)
    confidence: str = "LOW"  # LOW, MEDIUM, HIGH
    
    def to_dict(self) -> Dict:
        return {
            "SFS": round(self.score, 3),
            "dimension_scores": {
                "fault_type_match": round(self.dimensions.fault_type_match, 3),
                "trigger_condition_match": round(self.dimensions.trigger_condition_match, 3),
                "environmental_match": round(self.dimensions.environmental_match, 3),
                "temporal_consistency": round(self.dimensions.temporal_consistency, 3),
                "parameter_completeness": round(self.dimensions.parameter_completeness, 3),
            },
            "matched_fault_type": self.matched_fault_type,
            "extracted_keywords": self.extracted_keywords,
            "missing_parameters": self.missing_parameters,
            "confidence": self.confidence,
        }


class ScenarioFidelityScorer:
    """
    Computes Scenario Fidelity Score (SFS).
    
    Measures how faithfully the LLM translated the FAA narrative into
    a PX4 simulation configuration.
    """
    
    # Dimension weights (must sum to 1.0)
    WEIGHTS = {
        "fault_type_match": 0.30,
        "trigger_condition_match": 0.20,
        "environmental_match": 0.15,
        "temporal_consistency": 0.15,
        "parameter_completeness": 0.20,
    }
    
    # Required PX4 config parameters
    REQUIRED_PARAMS = [
        "waypoints",
        "fault_injection.fault_type",
        "mission.takeoff_altitude_m",
        "mission.start_lat",
        "mission.start_lon",
    ]
    
    def __init__(self):
        logger.debug("ScenarioFidelityScorer initialized")
    
    def evaluate(self, faa_report: Dict, px4_config: Dict) -> SFSResult:
        """
        Evaluate scenario fidelity.
        
        Args:
            faa_report: Original FAA incident data
            px4_config: LLM-generated PX4 configuration
            
        Returns:
            SFSResult with score and dimension breakdown
        """
        result = SFSResult()
        dims = SFSDimensions()
        
        # Extract text content
        faa_text = self._extract_faa_text(faa_report)
        faa_text_lower = faa_text.lower()
        
        # 1. FAULT TYPE MATCH (0.30 weight)
        dims.fault_type_match, result.matched_fault_type = self._score_fault_type_match(
            faa_text_lower, 
            faa_report.get("incident_type", ""),
            px4_config.get("fault_injection", {}).get("fault_type", "")
        )
        
        # 2. TRIGGER CONDITION MATCH (0.20 weight)
        dims.trigger_condition_match, keywords = self._score_trigger_conditions(
            faa_text_lower, px4_config
        )
        result.extracted_keywords = keywords
        
        # 3. ENVIRONMENTAL MATCH (0.15 weight)
        dims.environmental_match = self._score_environmental_match(
            faa_text_lower, faa_report, px4_config
        )
        
        # 4. TEMPORAL CONSISTENCY (0.15 weight)
        dims.temporal_consistency = self._score_temporal_consistency(
            faa_text_lower, px4_config
        )
        
        # 5. PARAMETER COMPLETENESS (0.20 weight)
        dims.parameter_completeness, result.missing_parameters = self._score_parameter_completeness(
            px4_config
        )
        
        # Compute weighted final score
        result.score = (
            dims.fault_type_match * self.WEIGHTS["fault_type_match"] +
            dims.trigger_condition_match * self.WEIGHTS["trigger_condition_match"] +
            dims.environmental_match * self.WEIGHTS["environmental_match"] +
            dims.temporal_consistency * self.WEIGHTS["temporal_consistency"] +
            dims.parameter_completeness * self.WEIGHTS["parameter_completeness"]
        )
        
        result.dimensions = dims
        result.confidence = self._compute_confidence(dims)
        
        logger.info(f"SFS evaluated: {result.score:.3f} ({result.confidence} confidence)")
        return result
    
    def _extract_faa_text(self, faa_report: Dict) -> str:
        """Extract all text content from FAA report."""
        parts = [
            faa_report.get("description", ""),
            faa_report.get("summary", ""),
            faa_report.get("narrative", ""),
        ]
        return " ".join(p for p in parts if p)
    
    def _score_fault_type_match(
        self, 
        faa_text: str, 
        faa_incident_type: str,
        config_fault_type: str
    ) -> Tuple[float, str]:
        """
        Score how well the fault type matches.
        
        IMPORTANT: FAA sighting reports often don't specify incident_type explicitly.
        They are observational reports of UAS sightings, not accident investigations.
        The LLM INFERS the fault type from behavioral descriptions in the narrative.
        
        Scoring logic:
        1. If incident_type exists, check for match
        2. If no incident_type, check if LLM's inference is REASONABLE based on narrative keywords
        3. Airspace violations (altitude_violation, geofence_violation) get credit for sighting reports
        """
        
        # Direct match with incident_type field (if available)
        if faa_incident_type and config_fault_type:
            if faa_incident_type.lower() in config_fault_type.lower() or \
               config_fault_type.lower() in faa_incident_type.lower():
                return 1.0, config_fault_type
        
        # No incident_type in data - check if config fault type is REASONABLE
        # FAA sighting reports describe observed behavior, LLM infers fault type
        
        # Special case: altitude_violation / airspace sighting
        # Many FAA reports are about UAS observed at high altitude near airports
        # This is a VALID inference even without explicit failure keywords
        if config_fault_type and any(x in config_fault_type.lower() for x in ["altitude_violation", "geofence_violation", "airspace"]):
            # Check if report mentions altitude or proximity to airport
            has_altitude = re.search(r'(\d{2,5})\s*(?:feet|ft|\')', faa_text)
            has_airport = any(kw in faa_text for kw in ["atct", "tower", "approach", "runway", "airport"])
            if has_altitude or has_airport:
                return 0.9, config_fault_type  # High score - reasonable inference
        
        # Keyword-based detection from narrative
        detected_types = []
        for fault_type, keywords in FAULT_TYPE_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in faa_text)
            if matches >= 2:
                detected_types.append((fault_type, matches))
        
        if not detected_types:
            # No keywords found - give moderate credit if LLM provided any fault type
            # Many sighting reports don't describe failures, just observations
            if config_fault_type:
                return 0.5, config_fault_type  # LLM made a reasonable attempt
            return 0.3, "unknown"  # Partial credit for having any config
        
        # Check if config matches any detected type
        detected_types.sort(key=lambda x: x[1], reverse=True)
        best_match = detected_types[0][0]
        
        if config_fault_type and best_match in config_fault_type.lower():
            return 1.0, best_match
        elif config_fault_type and any(t[0] in config_fault_type.lower() for t in detected_types):
            return 0.8, config_fault_type
        else:
            return 0.5, best_match  # Detected but config doesn't match
    
    def _score_trigger_conditions(
        self, faa_text: str, config: Dict
    ) -> Tuple[float, List[str]]:
        """Score trigger condition extraction."""
        
        trigger_keywords = []
        score = 0.0
        
        # Check altitude mentioned
        alt_match = re.search(r'(\d{2,5})\s*(?:feet|ft|\')', faa_text)
        if alt_match:
            trigger_keywords.append(f"altitude:{alt_match.group(1)}ft")
            config_alt = config.get("mission", {}).get("takeoff_altitude_m", 0)
            if config_alt > 0:
                score += 0.3
        
        # Check if malfunction/failure mentioned
        if any(kw in faa_text for kw in ["malfunction", "failure", "lost", "crash"]):
            trigger_keywords.append("failure_event")
            if config.get("fault_injection", {}).get("fault_type"):
                score += 0.4
        
        # Check location mentioned
        if config.get("mission", {}).get("start_lat"):
            score += 0.3
            trigger_keywords.append("location_set")
        
        return min(score, 1.0), trigger_keywords
    
    def _score_environmental_match(
        self, faa_text: str, faa_report: Dict, config: Dict
    ) -> float:
        """Score environmental condition matching."""
        
        score = 0.0
        
        # Location match
        faa_city = faa_report.get("city", "").lower()
        faa_state = faa_report.get("state", "").lower()
        config_lat = config.get("mission", {}).get("start_lat")
        
        if config_lat and (faa_city or faa_state):
            score += 0.5  # Location was geocoded
        
        # Environment type detection
        config_env = config.get("environment", {}).get("environment_type", "")
        for env_type, keywords in ENVIRONMENT_KEYWORDS.items():
            if any(kw in faa_text for kw in keywords):
                if env_type.lower() in config_env.lower():
                    score += 0.5
                else:
                    score += 0.2  # Partial credit
                break
        
        return min(score, 1.0)
    
    def _score_temporal_consistency(self, faa_text: str, config: Dict) -> float:
        """Score temporal/phase alignment."""
        
        score = 0.5  # Base score for having a config
        
        # Check if flight phase keywords match
        phase_keywords = {
            "takeoff": ["takeoff", "departing", "climbing"],
            "cruise": ["cruise", "flying", "en route", "during flight"],
            "landing": ["landing", "approach", "descent", "returned"],
        }
        
        for phase, keywords in phase_keywords.items():
            if any(kw in faa_text for kw in keywords):
                # Check if waypoints have appropriate actions
                waypoints = config.get("waypoints", [])
                if waypoints:
                    actions = [wp.get("action", "") for wp in waypoints]
                    if phase == "takeoff" and "takeoff" in actions:
                        score += 0.25
                    elif phase == "landing" and "land" in actions:
                        score += 0.25
                    elif phase == "cruise" and "waypoint" in actions:
                        score += 0.25
        
        return min(score, 1.0)
    
    def _score_parameter_completeness(self, config: Dict) -> Tuple[float, List[str]]:
        """Score config parameter completeness."""
        
        missing = []
        present = 0
        
        for param in self.REQUIRED_PARAMS:
            value = self._get_nested_value(config, param)
            if value is not None and value != "" and value != []:
                present += 1
            else:
                missing.append(param)
        
        score = present / len(self.REQUIRED_PARAMS) if self.REQUIRED_PARAMS else 0.0
        return score, missing
    
    def _get_nested_value(self, d: Dict, key: str):
        """Get nested dictionary value using dot notation."""
        keys = key.split(".")
        value = d
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return None
        return value
    
    def _compute_confidence(self, dims: SFSDimensions) -> str:
        """Compute confidence level based on dimension scores."""
        
        avg_score = (
            dims.fault_type_match +
            dims.trigger_condition_match +
            dims.environmental_match +
            dims.temporal_consistency +
            dims.parameter_completeness
        ) / 5
        
        low_dims = sum(1 for d in [
            dims.fault_type_match,
            dims.trigger_condition_match,
            dims.environmental_match,
            dims.temporal_consistency,
            dims.parameter_completeness,
        ] if d < 0.5)
        
        if avg_score >= 0.8 and low_dims == 0:
            return "HIGH"
        elif avg_score >= 0.5 and low_dims <= 1:
            return "MEDIUM"
        else:
            return "LOW"
