"""
Behavior Reproduction Rate (BRR)
================================
Author: AeroGuardian Member
Date: 2026-01-21

Verifies that the PX4 simulation actually reproduces abnormal behavior.

SCIENTIFIC RATIONALE:
---------------------
A high-fidelity config is useless if the simulation doesn't manifest abnormal behavior.
BRR uses DETERMINISTIC rules (no LLM) to detect anomalies from telemetry.

ANOMALY DETECTION (Fixed Thresholds):
--------------------------------------
| Anomaly Type        | Threshold          | Telemetry Field          |
|---------------------|--------------------| --------------------------|
| position_drift      | > 10m              | position variance         |
| velocity_variance   | > 5 m/s std        | velocity std              |
| altitude_instability| > 5m deviation     | altitude std              |
| roll_instability    | > 30 deg max       | roll max                  |
| pitch_instability   | > 30 deg max       | pitch max                 |
| control_saturation  | > 80% duration     | control output range      |
| gps_degradation     | > 3.0 HDOP         | GPS quality               |

SCORING:
--------
BRR = (detected_anomalies / expected_anomalies) * severity_weight
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("AeroGuardian.Evaluator.BRR")


# =============================================================================
# ANOMALY THRESHOLDS (DETERMINISTIC - NO LLM)
# =============================================================================

class AnomalyThresholds:
    """Fixed thresholds for deterministic anomaly detection."""
    
    # Position thresholds
    POSITION_DRIFT_M = 10.0  # meters
    POSITION_VARIANCE_M = 25.0  # meters squared
    
    # Velocity thresholds
    VELOCITY_STD_MPS = 5.0  # m/s
    
    # Altitude thresholds
    ALTITUDE_DEVIATION_M = 5.0  # meters
    ALTITUDE_STD_M = 8.0  # meters
    
    # Attitude thresholds
    ROLL_MAX_DEG = 30.0  # degrees
    PITCH_MAX_DEG = 30.0  # degrees
    ROLL_STD_DEG = 15.0  # degrees
    
    # GPS thresholds
    GPS_HDOP_MAX = 3.0
    GPS_VARIANCE_M = 5.0
    
    # Control thresholds
    CONTROL_SATURATION_PERCENT = 80.0


@dataclass
class DetectedAnomaly:
    """A single detected anomaly."""
    anomaly_type: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    measured_value: float
    threshold: float
    description: str
    
    def to_dict(self) -> Dict:
        return {
            "type": self.anomaly_type,
            "severity": self.severity,
            "measured": round(self.measured_value, 3),
            "threshold": self.threshold,
            "description": self.description,
        }


@dataclass
class BRRResult:
    """Complete BRR evaluation result."""
    score: float = 0.0
    detected_anomalies: List[DetectedAnomaly] = field(default_factory=list)
    expected_anomaly_types: List[str] = field(default_factory=list)
    telemetry_quality: str = "UNKNOWN"  # GOOD, DEGRADED, POOR
    data_points_analyzed: int = 0
    confidence: str = "LOW"
    
    def to_dict(self) -> Dict:
        return {
            "BRR": round(self.score, 3),
            "detected_anomalies": [a.to_dict() for a in self.detected_anomalies],
            "anomaly_count": len(self.detected_anomalies),
            "expected_anomaly_types": self.expected_anomaly_types,
            "telemetry_quality": self.telemetry_quality,
            "data_points_analyzed": self.data_points_analyzed,
            "confidence": self.confidence,
        }


class BehaviorValidator:
    """
    Computes Behavior Reproduction Rate (BRR).
    
    Uses DETERMINISTIC rules to detect abnormal behavior from telemetry.
    No LLM is used - only fixed thresholds.
    """
    
    # Fault type to expected anomalies mapping
    FAULT_TO_ANOMALIES = {
        "motor_failure": ["roll_instability", "pitch_instability", "altitude_instability"],
        "gps_loss": ["position_drift", "gps_degradation"],
        "gps_dropout": ["position_drift", "gps_degradation"],
        "battery_failure": ["altitude_instability", "control_saturation"],
        "control_loss": ["roll_instability", "pitch_instability", "position_drift"],
        "sensor_fault": ["altitude_instability", "attitude_instability"],
    }
    
    def __init__(self):
        logger.debug("BehaviorValidator initialized")
    
    def evaluate(
        self, 
        telemetry: List[Dict], 
        fault_type: str,
        telemetry_stats: Optional[Dict] = None
    ) -> BRRResult:
        """
        Evaluate behavior reproduction.
        
        Args:
            telemetry: Raw telemetry data points
            fault_type: Expected fault type from config
            telemetry_stats: Pre-computed telemetry statistics (optional)
            
        Returns:
            BRRResult with score and detected anomalies
        """
        result = BRRResult()
        result.data_points_analyzed = len(telemetry)
        
        # Validate telemetry quality
        if not telemetry or len(telemetry) < 10:
            result.telemetry_quality = "POOR"
            result.confidence = "LOW"
            result.score = 0.0
            logger.warning("Insufficient telemetry data for BRR")
            return result
        
        # Use pre-computed stats or compute from raw data
        stats = telemetry_stats or self._compute_telemetry_stats(telemetry)
        
        # Determine expected anomalies based on fault type
        fault_key = fault_type.lower().replace("-", "_") if fault_type else ""
        result.expected_anomaly_types = self.FAULT_TO_ANOMALIES.get(
            fault_key, ["position_drift", "altitude_instability"]  # Default
        )
        
        # Detect anomalies (DETERMINISTIC)
        result.detected_anomalies = self._detect_anomalies(stats)
        
        # Compute BRR score
        result.score = self._compute_brr_score(result)
        
        # Assess telemetry quality
        result.telemetry_quality = self._assess_telemetry_quality(stats, len(telemetry))
        result.confidence = self._compute_confidence(result, stats)
        
        logger.info(
            f"BRR evaluated: {result.score:.3f} with {len(result.detected_anomalies)} anomalies"
        )
        return result
    
    def _compute_telemetry_stats(self, telemetry: List[Dict]) -> Dict:
        """Compute statistics from raw telemetry."""
        
        if not telemetry:
            return {}
        
        import math
        
        # Extract metric arrays - handle multiple field name variations
        # Altitude: try 'alt', 'altitude_m', 'relative_alt', 'altitude'
        altitudes = []
        for t in telemetry:
            alt = t.get("alt", t.get("altitude_m", t.get("relative_alt", t.get("altitude", 0))))
            if alt and alt > 0:
                altitudes.append(alt)
        
        velocities = [t.get("velocity_m_s", t.get("groundspeed_m_s", 0)) for t in telemetry]
        
        # Roll/Pitch: try degrees first, then radians and convert
        rolls_deg = []
        pitches_deg = []
        for t in telemetry:
            # Try degrees first
            roll = t.get("roll_deg", None)
            pitch = t.get("pitch_deg", None)
            
            # If not found, use radians and convert
            if roll is None:
                roll_rad = t.get("roll", 0)
                roll = math.degrees(roll_rad) if roll_rad else 0
            if pitch is None:
                pitch_rad = t.get("pitch", 0)
                pitch = math.degrees(pitch_rad) if pitch_rad else 0
            
            rolls_deg.append(roll)
            pitches_deg.append(pitch)
        
        lats = [t.get("lat", 0) for t in telemetry if t.get("lat")]
        lons = [t.get("lon", 0) for t in telemetry if t.get("lon")]
        
        def std(arr):
            if not arr or len(arr) < 2:
                return 0.0
            mean = sum(arr) / len(arr)
            variance = sum((x - mean) ** 2 for x in arr) / len(arr)
            return math.sqrt(variance)
        
        def max_abs(arr):
            return max(abs(x) for x in arr) if arr else 0.0
        
        # Compute position drift as max distance from start
        position_drift = 0.0
        if lats and lons:
            start_lat, start_lon = lats[0], lons[0]
            for lat, lon in zip(lats, lons):
                # Approximate distance in meters
                dlat = (lat - start_lat) * 111000  # degrees to meters
                dlon = (lon - start_lon) * 111000 * math.cos(math.radians(start_lat))
                drift = math.sqrt(dlat**2 + dlon**2)
                position_drift = max(position_drift, drift)
        
        stats = {
            "max_altitude_m": max(altitudes) if altitudes else 0,
            "altitude_std_m": std(altitudes),
            "altitude_deviation": max(altitudes) - min(altitudes) if altitudes else 0,
            "velocity_std_mps": std(velocities),
            "max_roll_deg": max_abs(rolls_deg),
            "roll_std_deg": std(rolls_deg),
            "max_pitch_deg": max_abs(pitches_deg),
            "pitch_std_deg": std(pitches_deg),
            "position_drift_m": position_drift,
            "gps_variance": std(lats) * 111000 if lats else 0,  # Convert to meters
            "data_points": len(telemetry),
            "flight_duration_s": len(telemetry),  # Approximate 1Hz
        }
        
        # Log for debugging
        logger.debug(f"Computed stats: max_roll={stats['max_roll_deg']:.1f}°, max_pitch={stats['max_pitch_deg']:.1f}°, alt_dev={stats['altitude_deviation']:.1f}m")
        
        return stats
    
    def _detect_anomalies(self, stats: Dict) -> List[DetectedAnomaly]:
        """Detect anomalies using fixed thresholds."""
        
        anomalies = []
        thresholds = AnomalyThresholds()
        
        # Position drift
        drift = stats.get("position_drift_m", 0)
        if drift > thresholds.POSITION_DRIFT_M:
            severity = "CRITICAL" if drift > 50 else ("HIGH" if drift > 25 else "MEDIUM")
            anomalies.append(DetectedAnomaly(
                anomaly_type="position_drift",
                severity=severity,
                measured_value=drift,
                threshold=thresholds.POSITION_DRIFT_M,
                description=f"Position drift of {drift:.1f}m exceeds {thresholds.POSITION_DRIFT_M}m threshold"
            ))
        
        # Altitude instability
        alt_dev = stats.get("altitude_deviation", 0)
        if alt_dev > thresholds.ALTITUDE_DEVIATION_M:
            severity = "HIGH" if alt_dev > 20 else "MEDIUM"
            anomalies.append(DetectedAnomaly(
                anomaly_type="altitude_instability",
                severity=severity,
                measured_value=alt_dev,
                threshold=thresholds.ALTITUDE_DEVIATION_M,
                description=f"Altitude deviation of {alt_dev:.1f}m exceeds threshold"
            ))
        
        # Roll instability
        roll_max = stats.get("max_roll_deg", 0)
        if roll_max > thresholds.ROLL_MAX_DEG:
            severity = "CRITICAL" if roll_max > 60 else ("HIGH" if roll_max > 45 else "MEDIUM")
            anomalies.append(DetectedAnomaly(
                anomaly_type="roll_instability",
                severity=severity,
                measured_value=roll_max,
                threshold=thresholds.ROLL_MAX_DEG,
                description=f"Maximum roll of {roll_max:.1f}° exceeds {thresholds.ROLL_MAX_DEG}° threshold"
            ))
        
        # Pitch instability
        pitch_max = stats.get("max_pitch_deg", 0)
        if pitch_max > thresholds.PITCH_MAX_DEG:
            severity = "HIGH" if pitch_max > 45 else "MEDIUM"
            anomalies.append(DetectedAnomaly(
                anomaly_type="pitch_instability",
                severity=severity,
                measured_value=pitch_max,
                threshold=thresholds.PITCH_MAX_DEG,
                description=f"Maximum pitch of {pitch_max:.1f}° exceeds threshold"
            ))
        
        # GPS degradation
        gps_var = stats.get("gps_variance", 0)
        if gps_var > thresholds.GPS_VARIANCE_M:
            severity = "HIGH" if gps_var > 20 else "MEDIUM"
            anomalies.append(DetectedAnomaly(
                anomaly_type="gps_degradation",
                severity=severity,
                measured_value=gps_var,
                threshold=thresholds.GPS_VARIANCE_M,
                description=f"GPS variance of {gps_var:.1f}m indicates degradation"
            ))
        
        # Velocity variance
        vel_std = stats.get("velocity_std_mps", 0)
        if vel_std > thresholds.VELOCITY_STD_MPS:
            anomalies.append(DetectedAnomaly(
                anomaly_type="velocity_variance",
                severity="MEDIUM",
                measured_value=vel_std,
                threshold=thresholds.VELOCITY_STD_MPS,
                description=f"Velocity std of {vel_std:.1f} m/s indicates instability"
            ))
        
        return anomalies
    
    def _compute_brr_score(self, result: BRRResult) -> float:
        """Compute BRR score based on detected vs expected anomalies.
        
        Scoring logic:
        1. Primary: Check if expected anomaly types are detected
        2. Secondary: Give partial credit for ANY detected anomalies (crash behavior)
        3. Bonus: Add severity bonus for matched anomalies
        """
        
        if not result.expected_anomaly_types:
            # No specific expectations - score based on whether we detected anything
            return min(len(result.detected_anomalies) * 0.3, 1.0)
        
        detected_types = {a.anomaly_type for a in result.detected_anomalies}
        expected_types = set(result.expected_anomaly_types)
        
        # Primary score: what fraction of expected anomalies were detected?
        matches = detected_types & expected_types
        primary_score = len(matches) / len(expected_types) if expected_types else 0.0
        
        # Secondary score: Give credit for ANY detected anomalies
        # This handles crash simulations where GPS loss → kill switch → pitch/roll/altitude anomalies
        # Even if not the "expected" GPS anomalies, detecting crash behavior is valuable
        if len(result.detected_anomalies) > 0 and primary_score == 0:
            # No expected matches, but we detected crash-like behavior
            # Give partial credit: more anomalies = more credit, capped at 0.8
            secondary_score = min(len(result.detected_anomalies) * 0.25, 0.8)
            logger.debug(
                f"BRR secondary scoring: {len(result.detected_anomalies)} non-matching anomalies → {secondary_score:.2f}"
            )
        else:
            secondary_score = 0.0
        
        # Use the higher of primary or secondary
        base_score = max(primary_score, secondary_score)
        
        # Bonus for severity (on matched anomalies)
        severity_bonus = 0.0
        for anomaly in result.detected_anomalies:
            if anomaly.anomaly_type in expected_types:
                if anomaly.severity == "CRITICAL":
                    severity_bonus += 0.1
                elif anomaly.severity == "HIGH":
                    severity_bonus += 0.05
            # Also give small bonus for high-severity non-matched anomalies
            elif anomaly.severity in ("CRITICAL", "HIGH"):
                severity_bonus += 0.02
        
        return min(base_score + severity_bonus, 1.0)
    
    def _assess_telemetry_quality(self, stats: Dict, data_points: int) -> str:
        """Assess overall telemetry quality."""
        
        if data_points < 100:
            return "POOR"
        elif data_points < 500:
            return "DEGRADED"
        else:
            return "GOOD"
    
    def _compute_confidence(self, result: BRRResult, stats: Dict) -> str:
        """Compute confidence level."""
        
        data_points = stats.get("data_points", 0)
        
        if result.telemetry_quality == "POOR" or data_points < 100:
            return "LOW"
        elif result.score >= 0.7 and len(result.detected_anomalies) >= 2:
            return "HIGH"
        elif result.score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
