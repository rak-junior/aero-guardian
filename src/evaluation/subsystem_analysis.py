"""
Subsystem Causal Analysis
=========================
Author: AeroGuardian Team
Date: 2026-02-03

Determines root cause of UAS incidents using CAUSAL reasoning, not correlation.

SCIENTIFIC APPROACH:
-------------------
1. Temporal Ordering: Which subsystem showed abnormal behavior FIRST?
2. Confidence Weighting: Ignore noise spikes, prioritize high-confidence anomalies
3. Propagation Delay: Validate that timing between subsystems is physically plausible
4. Ambiguity Handling: Admit uncertainty when evidence is insufficient

This transforms diagnosis from:
    "Crash + large deviation → control_loss" (correlation)
To:
    "GPS spike @ t=32s BEFORE attitude instability @ t=45s → navigation failure" (causation)

REFERENCES:
-----------
[1] Murphy, R. (2000). Introduction to AI Robotics. MIT Press.
[2] PX4 Autopilot Failsafe Documentation
[3] FAA AC 25-7D Flight Test Guide
"""

import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("AeroGuardian.Evaluator.Causal")


# =============================================================================
# SUBSYSTEM DEFINITIONS
# =============================================================================

class Subsystem(Enum):
    """UAS subsystem categories."""
    NAVIGATION = "navigation"      # GPS, EKF, position estimation
    CONTROL = "control"            # Attitude control, rate control
    PROPULSION = "propulsion"      # Motors, ESCs, thrust
    SENSOR = "sensor"              # IMU, baro, compass
    POWER = "power"                # Battery, voltage regulation
    AMBIGUOUS = "ambiguous"        # Requires additional evidence
    UNDETERMINED = "undetermined"  # Cannot determine from evidence


# =============================================================================
# ANOMALY TO SUBSYSTEM MAPPING (WITH CONFIDENCE)
# =============================================================================

# Format: anomaly_type -> (subsystem, base_confidence)
# Confidence represents how reliably this anomaly indicates the subsystem
ANOMALY_TO_SUBSYSTEM: Dict[str, Tuple[str, float]] = {
    # Navigation subsystem (HIGH confidence mapping)
    "position_drift": ("navigation", 0.9),
    "gps_degradation": ("navigation", 0.95),
    "ekf_innovation_spike": ("navigation", 0.85),
    
    # Control subsystem (HIGH confidence mapping)
    "roll_instability": ("control", 0.85),
    "pitch_instability": ("control", 0.85),
    "control_saturation": ("control", 0.9),
    "attitude_oscillation": ("control", 0.8),
    
    # Propulsion subsystem (requires multiple signals for high confidence)
    "motor_asymmetry": ("propulsion", 0.9),
    "yaw_bias": ("propulsion", 0.8),
    "thrust_loss": ("propulsion", 0.85),
    
    # Sensor subsystem
    "attitude_noise": ("sensor", 0.7),
    "sensor_disagreement": ("sensor", 0.85),
    "compass_error": ("sensor", 0.75),
    "sensor_degradation": ("sensor", 0.85),  # Added: matches BehaviorValidator output
    
    # Power subsystem
    "battery_voltage_drop": ("power", 0.9),
    "voltage_sag": ("power", 0.85),
    
    # AMBIGUOUS - requires additional context to classify
    # These anomalies can arise from multiple subsystem failures
    "altitude_instability": ("ambiguous", 0.3),
    "velocity_variance": ("ambiguous", 0.4),
}

# Resolution rules for ambiguous anomalies
# Format: anomaly_type -> {target_subsystem: [supporting_anomalies]}
AMBIGUOUS_RESOLUTION: Dict[str, Dict[str, List[str]]] = {
    "altitude_instability": {
        "propulsion": ["motor_asymmetry", "yaw_bias", "thrust_loss"],
        "control": ["roll_instability", "pitch_instability", "control_saturation"],
        "power": ["battery_voltage_drop", "voltage_sag"],
        "navigation": ["gps_degradation", "position_drift"],
    },
    "velocity_variance": {
        "navigation": ["position_drift", "gps_degradation"],
        "control": ["roll_instability", "pitch_instability"],
        "propulsion": ["motor_asymmetry"],
    },
}


# =============================================================================
# PROPAGATION DELAY CONSTRAINTS (PHYSICS-BASED)
# =============================================================================

# Format: (from_subsystem, to_subsystem) -> (min_delay_sec, max_delay_sec)
# These represent physically plausible delays between subsystem effects
# NOTE: Min delays are set to 0.0 to accommodate SIH emulation mode where
# failures are injected synchronously. Max delays are extended for behavioral
# emulation (fallback mode) where fault progression is simulated over longer
# time periods. For real hardware failures, delays would be shorter.
PROPAGATION_DELAYS: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("navigation", "control"): (0.0, 30.0),    # EKF delay → control response (extended for emulation)
    ("control", "propulsion"): (0.0, 15.0),    # Control output → motor response
    ("propulsion", "control"): (0.0, 15.0),    # Motor failure → attitude effect (extended for emulation)
    ("sensor", "navigation"): (0.0, 20.0),     # Sensor corruption → EKF effect
    ("sensor", "control"): (0.0, 15.0),        # Sensor noise → control noise
    ("power", "propulsion"): (0.0, 10.0),      # Voltage sag → thrust loss
    ("power", "sensor"): (0.0, 10.0),          # Voltage drop → sensor brownout
    ("navigation", "propulsion"): (0.0, 30.0), # GPS loss → EKF → control → motor (multi-hop)
    ("propulsion", "navigation"): (0.0, 30.0), # Vibration → sensor noise (secondary effect)
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SubsystemEvidence:
    """
    Evidence for a specific subsystem's involvement in an incident.
    
    Attributes:
        subsystem: The subsystem identifier (e.g., "navigation", "control")
        anomalies: List of anomaly descriptions with timing
        first_anomaly_time_sec: When the first anomaly was detected
        confidence: 0.0-1.0 confidence that this subsystem is involved
        status: Evidence status (CONFIRMED, SECONDARY, SUSPECTED, NO_EVIDENCE)
    """
    subsystem: str
    anomalies: List[str] = field(default_factory=list)
    first_anomaly_time_sec: float = float('inf')
    confidence: float = 0.0
    status: str = "NO_EVIDENCE"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "subsystem": self.subsystem,
            "anomalies": self.anomalies,
            "first_anomaly_time_sec": round(self.first_anomaly_time_sec, 2) 
                if self.first_anomaly_time_sec != float('inf') else None,
            "confidence": round(self.confidence, 3),
            "status": self.status,
        }


@dataclass
class CausalAnalysisResult:
    """
    Complete result of causal subsystem analysis.
    
    Attributes:
        primary_failure_subsystem: The subsystem that failed FIRST (or "undetermined")
        confidence: Overall confidence in the diagnosis (0.0-1.0)
        causal_chain: Ordered list of subsystems affected (e.g., ["navigation", "control"])
        subsystem_evidence: Evidence for each analyzed subsystem
        diagnosis_reasoning: Human-readable explanation of the diagnosis
        is_conclusive: Whether enough evidence exists for a definitive diagnosis
        chain_plausibility: Whether the causal chain timing is physically plausible
        warnings: Any warnings about the analysis
    """
    primary_failure_subsystem: str = "undetermined"
    confidence: float = 0.0
    causal_chain: List[str] = field(default_factory=list)
    subsystem_evidence: Dict[str, SubsystemEvidence] = field(default_factory=dict)
    diagnosis_reasoning: str = ""
    is_conclusive: bool = False
    chain_plausibility: str = "unknown"
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "primary_failure_subsystem": self.primary_failure_subsystem,
            "confidence": round(self.confidence, 3),
            "causal_chain": self.causal_chain,
            "subsystem_evidence": {
                k: v.to_dict() for k, v in self.subsystem_evidence.items()
            },
            "diagnosis_reasoning": self.diagnosis_reasoning,
            "is_conclusive": self.is_conclusive,
            "chain_plausibility": self.chain_plausibility,
            "warnings": self.warnings,
        }


# =============================================================================
# MAIN ANALYZER CLASS
# =============================================================================

class SubsystemCausalAnalyzer:
    """
    Determines root cause of UAS incidents using causal reasoning.
    
    Key Features:
    1. Confidence-weighted root cause selection (not just earliest timestamp)
    2. Propagation delay plausibility validation
    3. "Undetermined" outcome when evidence is insufficient
    4. Ambiguous subsystem resolution with context
    
    Usage:
        analyzer = SubsystemCausalAnalyzer()
        result = analyzer.analyze(detected_anomalies)
        print(result.primary_failure_subsystem)
        print(result.causal_chain)
    """
    
    # Minimum confidence threshold for conclusive diagnosis
    CONFIDENCE_THRESHOLD = 0.5
    
    # Minimum anomalies needed for subsystem to be considered
    MIN_ANOMALIES_FOR_SUBSYSTEM = 1
    
    def __init__(self):
        """Initialize the causal analyzer."""
        logger.debug("SubsystemCausalAnalyzer initialized")
    
    def analyze(self, anomalies: List[Dict], fault_type: str = "") -> CausalAnalysisResult:
        """
        Analyze detected anomalies to determine root cause.
        
        Args:
            anomalies: List of detected anomalies, each with:
                - anomaly_type: str (e.g., "position_drift")
                - first_detected_sec: float (when first exceeded threshold)
                - measured_value: float
                - threshold: float
                - severity: str (CRITICAL, HIGH, MEDIUM, LOW)
            fault_type: Known fault type from simulation config (optional).
                Used to resolve ambiguous anomalies when supporting signals
                are missing due to simulation limitations.
        
        Returns:
            CausalAnalysisResult with root cause diagnosis
        """
        result = CausalAnalysisResult()
        
        # Handle empty input
        if not anomalies:
            result.diagnosis_reasoning = "No anomalies detected. Cannot perform causal analysis."
            result.warnings.append("No anomalies provided for analysis")
            logger.warning("Causal analysis called with no anomalies")
            return result
        
        try:
            # Step 1: Resolve ambiguous anomalies and group by subsystem
            subsystem_groups = self._group_by_subsystem(anomalies, fault_type)
            
            if not subsystem_groups:
                result.diagnosis_reasoning = "Could not map anomalies to subsystems."
                result.warnings.append("All anomalies failed subsystem mapping")
                return result
            
            # Step 2: Compute evidence for each subsystem
            result.subsystem_evidence = self._build_evidence_table(subsystem_groups, anomalies)
            
            # Step 3: Select root cause using confidence-weighted approach
            primary, confidence = self._select_root_cause(result.subsystem_evidence)
            result.primary_failure_subsystem = primary
            result.confidence = confidence
            
            # Step 4: Check if diagnosis is conclusive
            if confidence < self.CONFIDENCE_THRESHOLD:
                result.is_conclusive = False
                result.primary_failure_subsystem = "undetermined"
                result.diagnosis_reasoning = (
                    f"Insufficient evidence for conclusive diagnosis. "
                    f"Highest confidence was {confidence:.2f} for '{primary}', "
                    f"below threshold of {self.CONFIDENCE_THRESHOLD}."
                )
                result.warnings.append(f"Confidence {confidence:.2f} below threshold {self.CONFIDENCE_THRESHOLD}")
                logger.info(f"Causal analysis inconclusive: confidence={confidence:.2f}")
                return result
            
            result.is_conclusive = True
            
            # Step 5: Build causal chain with temporal ordering
            result.causal_chain = self._build_causal_chain(result.subsystem_evidence)
            
            # Step 6: Validate causal chain timing plausibility
            is_plausible, plausibility_msg = self._validate_causal_chain(
                result.causal_chain, 
                result.subsystem_evidence
            )
            result.chain_plausibility = "plausible" if is_plausible else "implausible"
            
            if not is_plausible:
                result.warnings.append(f"Causal chain timing: {plausibility_msg}")
                logger.warning(f"Implausible causal chain: {plausibility_msg}")
            
            # Step 7: Generate reasoning
            result.diagnosis_reasoning = self._generate_reasoning(result)
            
            logger.info(
                f"Causal analysis complete: primary={result.primary_failure_subsystem}, "
                f"confidence={result.confidence:.2f}, chain={result.causal_chain}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Causal analysis failed: {e}", exc_info=True)
            result.warnings.append(f"Analysis error: {str(e)}")
            result.diagnosis_reasoning = f"Analysis failed due to error: {str(e)}"
            return result
    
    def _group_by_subsystem(
        self, 
        anomalies: List[Dict],
        fault_type: str = ""
    ) -> Dict[str, List[Dict]]:
        """
        Group anomalies by subsystem, resolving ambiguous mappings.
        
        Args:
            anomalies: Raw anomaly list
            fault_type: Known fault type for fallback resolution
            
        Returns:
            Dict mapping subsystem name to list of anomalies
        """
        groups: Dict[str, List[Dict]] = {}
        all_anomaly_types = [a.get("type", a.get("anomaly_type", "")) for a in anomalies]
        
        for anomaly in anomalies:
            anomaly_type = anomaly.get("type", anomaly.get("anomaly_type", "unknown"))
            
            # Get mapped subsystem
            mapping = ANOMALY_TO_SUBSYSTEM.get(anomaly_type)
            if not mapping:
                logger.debug(f"Unknown anomaly type: {anomaly_type}")
                continue
            
            subsystem, base_confidence = mapping
            
            # Resolve ambiguous subsystems using context
            if subsystem == "ambiguous":
                subsystem = self._resolve_ambiguous(anomaly_type, all_anomaly_types, fault_type)
            
            # Skip if still ambiguous
            if subsystem == "ambiguous":
                logger.debug(f"Could not resolve ambiguous anomaly: {anomaly_type}")
                continue
            
            # Add to group
            if subsystem not in groups:
                groups[subsystem] = []
            groups[subsystem].append(anomaly)
        
        return groups
    
    # Fault type to subsystem mapping (for fallback resolution)
    FAULT_TYPE_TO_SUBSYSTEM = {
        "battery_failure": "power",
        "battery_depletion": "power",
        "power_failure": "power",
        "voltage_sag": "power",
        "motor_failure": "propulsion",
        "propulsion_failure": "propulsion",
        "gps_loss": "navigation",
        "gps_failure": "navigation",
        "navigation_failure": "navigation",
        "sensor_failure": "sensor",
        "control_loss": "control",
    }
    
    def _resolve_ambiguous(
        self, 
        anomaly_type: str, 
        all_anomaly_types: List[str],
        fault_type: str = ""
    ) -> str:
        """
        Resolve an ambiguous anomaly to a specific subsystem based on context.
        
        Args:
            anomaly_type: The ambiguous anomaly type
            all_anomaly_types: All anomaly types in the incident
            fault_type: Known fault type for fallback resolution
            
        Returns:
            Resolved subsystem or "ambiguous" if cannot resolve
        """
        if anomaly_type not in AMBIGUOUS_RESOLUTION:
            return "ambiguous"
        
        rules = AMBIGUOUS_RESOLUTION[anomaly_type]
        
        # Check each potential subsystem for supporting evidence
        for subsystem, supporting_signals in rules.items():
            if any(sig in all_anomaly_types for sig in supporting_signals):
                logger.debug(
                    f"Resolved '{anomaly_type}' to '{subsystem}' via supporting signal"
                )
                return subsystem
        
        # FALLBACK: If no supporting signals but fault_type is known,
        # use fault_type to infer the subsystem (handles simulation limitations)
        if fault_type:
            fault_key = fault_type.lower().replace("-", "_").replace(" ", "_")
            inferred_subsystem = self.FAULT_TYPE_TO_SUBSYSTEM.get(fault_key)
            if inferred_subsystem and inferred_subsystem in rules:
                logger.info(
                    f"Resolved '{anomaly_type}' to '{inferred_subsystem}' via fault_type '{fault_type}' (fallback)"
                )
                return inferred_subsystem
        
        return "ambiguous"
    
    def _build_evidence_table(
        self, 
        subsystem_groups: Dict[str, List[Dict]],
        all_anomalies: List[Dict]
    ) -> Dict[str, SubsystemEvidence]:
        """
        Build evidence table for each subsystem.
        
        Args:
            subsystem_groups: Anomalies grouped by subsystem
            all_anomalies: All anomalies (for context)
            
        Returns:
            Dict mapping subsystem to SubsystemEvidence
        """
        evidence_table: Dict[str, SubsystemEvidence] = {}
        
        # Define all subsystems to track
        all_subsystems = ["navigation", "control", "propulsion", "sensor", "power"]
        
        for subsystem in all_subsystems:
            anomalies = subsystem_groups.get(subsystem, [])
            
            if not anomalies:
                evidence_table[subsystem] = SubsystemEvidence(
                    subsystem=subsystem,
                    status="NO_EVIDENCE"
                )
                continue
            
            # Compute confidence using magnitude, count, and persistence
            confidence = self._compute_subsystem_confidence(anomalies)
            
            # Find first detection time
            first_time = min(
                a.get("first_detected_sec", float('inf')) 
                for a in anomalies
            )
            
            # Build anomaly descriptions
            descriptions = []
            for a in anomalies:
                atype = a.get("type", a.get("anomaly_type", "unknown"))
                time = a.get("first_detected_sec", 0)
                descriptions.append(f"{atype} @ t={time:.1f}s")
            
            # Determine status based on confidence
            if confidence >= 0.8:
                status = "CONFIRMED"
            elif confidence >= 0.5:
                status = "LIKELY"
            elif confidence >= 0.3:
                status = "SUSPECTED"
            else:
                status = "WEAK_EVIDENCE"
            
            evidence_table[subsystem] = SubsystemEvidence(
                subsystem=subsystem,
                anomalies=descriptions,
                first_anomaly_time_sec=first_time,
                confidence=confidence,
                status=status
            )
        
        return evidence_table
    
    def _compute_subsystem_confidence(self, anomalies: List[Dict]) -> float:
        """
        Compute confidence for a subsystem based on its anomalies.
        
        Factors considered:
        1. Magnitude above threshold (how far above?)
        2. Anomaly count (multiple signals increase confidence)
        3. Severity distribution (CRITICAL > HIGH > MEDIUM > LOW)
        
        Args:
            anomalies: Anomalies for this subsystem
            
        Returns:
            Confidence score 0.0-1.0
        """
        if not anomalies:
            return 0.0
        
        # Factor 1: Magnitude ratio (average of measured/threshold)
        magnitude_scores = []
        for a in anomalies:
            measured = a.get("measured", a.get("measured_value", 0))
            threshold = a.get("threshold", 1)
            if threshold > 0:
                ratio = measured / threshold
                # Cap at 5x threshold for max contribution
                magnitude_scores.append(min(ratio / 5.0, 1.0))
        
        magnitude_factor = sum(magnitude_scores) / len(magnitude_scores) if magnitude_scores else 0.5
        
        # Factor 2: Count bonus (more anomalies = higher confidence)
        # 1 anomaly = 0.3, 2 = 0.6, 3+ = 1.0
        count_factor = min(len(anomalies) / 3.0, 1.0)
        
        # Factor 3: Severity bonus
        severity_weights = {"CRITICAL": 1.0, "HIGH": 0.8, "MEDIUM": 0.5, "LOW": 0.3}
        severity_scores = [
            severity_weights.get(a.get("severity", "MEDIUM"), 0.5) 
            for a in anomalies
        ]
        severity_factor = max(severity_scores) if severity_scores else 0.5
        
        # Combine factors with weights
        # Magnitude: 40%, Count: 30%, Severity: 30%
        confidence = (
            magnitude_factor * 0.4 +
            count_factor * 0.3 +
            severity_factor * 0.3
        )
        
        return min(confidence, 1.0)
    
    def _select_root_cause(
        self, 
        evidence_table: Dict[str, SubsystemEvidence]
    ) -> Tuple[str, float]:
        """
        Select root cause using confidence-weighted temporal ordering.
        
        Algorithm:
        1. Filter to subsystems with confidence >= threshold
        2. Among candidates, prioritize by: time / confidence
           (earlier + higher confidence wins)
        
        Args:
            evidence_table: Evidence for each subsystem
            
        Returns:
            Tuple of (primary_subsystem, confidence)
        """
        candidates = []
        
        for subsystem, evidence in evidence_table.items():
            # Skip subsystems with no evidence
            if evidence.status == "NO_EVIDENCE":
                continue
            
            # Skip low-confidence subsystems (noise filtering)
            if evidence.confidence < 0.3:
                continue
            
            candidates.append((
                subsystem,
                evidence.first_anomaly_time_sec,
                evidence.confidence
            ))
        
        if not candidates:
            return ("undetermined", 0.0)
        
        # Sort by time/confidence ratio (lower is better)
        # Earlier time + higher confidence = lower ratio
        def sort_key(c):
            subsystem, time, confidence = c
            if confidence == 0:
                return float('inf')
            return time / confidence
        
        candidates.sort(key=sort_key)
        
        best = candidates[0]
        return (best[0], best[2])
    
    def _build_causal_chain(
        self, 
        evidence_table: Dict[str, SubsystemEvidence]
    ) -> List[str]:
        """
        Build ordered causal chain based on temporal ordering.
        
        Args:
            evidence_table: Evidence for each subsystem
            
        Returns:
            List of subsystems in order of failure propagation
        """
        # Filter to subsystems with evidence
        active_subsystems = [
            (sub, ev.first_anomaly_time_sec, ev.confidence)
            for sub, ev in evidence_table.items()
            if ev.status != "NO_EVIDENCE" and ev.first_anomaly_time_sec != float('inf')
        ]
        
        # Sort by time
        active_subsystems.sort(key=lambda x: x[1])
        
        # Build chain
        chain = [sub for sub, _, _ in active_subsystems]
        
        return chain if chain else ["unknown"]
    
    def _validate_causal_chain(
        self, 
        chain: List[str],
        evidence_table: Dict[str, SubsystemEvidence]
    ) -> Tuple[bool, str]:
        """
        Validate that the causal chain timing is physically plausible.
        
        Args:
            chain: Ordered list of subsystems
            evidence_table: Evidence with timing information
            
        Returns:
            Tuple of (is_plausible, explanation)
        """
        if len(chain) < 2:
            return (True, "Single subsystem, no propagation to validate")
        
        issues = []
        
        for i in range(len(chain) - 1):
            from_sub = chain[i]
            to_sub = chain[i + 1]
            
            # Get timing
            from_ev = evidence_table.get(from_sub)
            to_ev = evidence_table.get(to_sub)
            
            if not from_ev or not to_ev:
                continue
            
            from_time = from_ev.first_anomaly_time_sec
            to_time = to_ev.first_anomaly_time_sec
            
            if from_time == float('inf') or to_time == float('inf'):
                continue
            
            delay = to_time - from_time
            
            # Check against physics constraints
            constraint = PROPAGATION_DELAYS.get((from_sub, to_sub))
            if constraint:
                min_delay, max_delay = constraint
                if delay < min_delay:
                    issues.append(
                        f"{from_sub}→{to_sub}: delay {delay:.2f}s < min {min_delay}s (too fast)"
                    )
                elif delay > max_delay:
                    issues.append(
                        f"{from_sub}→{to_sub}: delay {delay:.2f}s > max {max_delay}s (too slow)"
                    )
        
        if issues:
            return (False, "; ".join(issues))
        
        return (True, "All propagation delays within physical constraints")
    
    def _generate_reasoning(self, result: CausalAnalysisResult) -> str:
        """
        Generate human-readable reasoning for the diagnosis.
        
        Args:
            result: The analysis result
            
        Returns:
            Reasoning string
        """
        primary = result.primary_failure_subsystem
        primary_ev = result.subsystem_evidence.get(primary)
        
        if not primary_ev:
            return "Unable to generate reasoning: no primary evidence found."
        
        parts = []
        
        # State the primary finding
        parts.append(
            f"Primary failure identified in '{primary}' subsystem "
            f"(confidence: {result.confidence:.0%})."
        )
        
        # Explain timing
        if primary_ev.first_anomaly_time_sec != float('inf'):
            parts.append(
                f"First anomaly detected at t={primary_ev.first_anomaly_time_sec:.1f}s."
            )
        
        # Explain evidence
        if primary_ev.anomalies:
            if len(primary_ev.anomalies) == 1:
                parts.append(f"Evidence: {primary_ev.anomalies[0]}.")
            else:
                parts.append(f"Evidence: {', '.join(primary_ev.anomalies)}.")
        
        # Explain causal chain
        if len(result.causal_chain) > 1:
            chain_str = " → ".join(result.causal_chain)
            parts.append(f"Causal propagation: {chain_str}.")
        
        # Add plausibility note if issues
        if result.chain_plausibility == "implausible":
            parts.append("⚠️ Note: Causal chain timing may not be physically plausible.")
        
        return " ".join(parts)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_anomalies(anomalies: List[Dict]) -> CausalAnalysisResult:
    """
    Convenience function to analyze anomalies without instantiating the class.
    
    Args:
        anomalies: List of detected anomalies
        
    Returns:
        CausalAnalysisResult
    """
    analyzer = SubsystemCausalAnalyzer()
    return analyzer.analyze(anomalies)
