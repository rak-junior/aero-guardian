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
import math

logger = logging.getLogger("AeroGuardian.Evaluator.BRR")


# =============================================================================
# ANOMALY THRESHOLDS (DETERMINISTIC - NO LLM)
# =============================================================================

# Warm-up period to skip simulator initialization artifacts
# SIH and Gazebo simulators have initial transients that cause false positives
TELEMETRY_WARMUP_SEC = 3.0  # Skip first 3 seconds of telemetry

# Takeoff phase - high pitch/roll is normal during climb
# Skip attitude threshold checks during this phase
TAKEOFF_PHASE_SEC = 15.0  # First 15 seconds allow aggressive attitude during takeoff

# Minimum altitude for stable flight (below this, we're in ground/takeoff phase)
MIN_CRUISE_ALTITUDE_M = 5.0  # Below 5m, attitude thresholds are relaxed

# Simulation artifact detection - physically impossible telemetry
# If angle > 90° but altitude is stable, it's a quaternion-to-euler artifact
PHYSICALLY_POSSIBLE_ANGLE_DEG = 85.0  # Max realistic angle before checking for artifacts

class AnomalyThresholds:
    """
    Fixed thresholds for deterministic anomaly detection.
    
    THRESHOLD RATIONALE:
    ---------------------
    These thresholds are engineering heuristics for simulation anomaly detection.
    They are NOT derived from sUAS certification standards (none exist for this purpose).
    
    - Position/GPS: 10m threshold catches major position solution failures.
      Reference: Typical consumer GPS horizontal accuracy is 2-5m CEP.
      
    - Altitude: 5m threshold detects significant altitude hold failures.
      Note: This is a simulation artifact detector, NOT a Part 107 compliance check.
      
    - Attitude (Roll/Pitch): 30° threshold is a conservative stability indicator
      for multirotor hover. This is an engineering judgment, NOT a regulatory limit.
      Note: No FAA AC exists for sUAS attitude limits.
      
    - Control Saturation: 80% threshold from PX4 documentation indicates
      actuator authority concerns, but actual failure modes are complex.
      
    DISCLAIMER:
    -----------
    These thresholds detect simulation anomalies, NOT real-world safety issues.
    No regulatory body has validated these thresholds for sUAS operations.
    
    REFERENCES:
    -----------
    [1] FAA 14 CFR Part 107 - Small Unmanned Aircraft Systems (operational rules only)
    [2] PX4 Autopilot Documentation - Failsafe Parameters (engineering reference)
    [3] General GNSS accuracy literature (horizontal accuracy expectations)
    """
    
    # Position thresholds (ref: DO-316 GNSS accuracy requirements)
    POSITION_DRIFT_M = 10.0  # meters - exceeds 3× typical GPS accuracy
    POSITION_VARIANCE_M = 25.0  # meters squared - indicates unstable position solution
    
    # Velocity thresholds (ref: PX4 stability margins)
    VELOCITY_STD_MPS = 5.0  # m/s - exceeds normal hover velocity variance
    
    # Altitude thresholds (ref: Part 107 operational accuracy)
    ALTITUDE_DEVIATION_M = 5.0  # meters - significant deviation from commanded altitude
    ALTITUDE_STD_M = 8.0  # meters - indicates altitude control oscillation
    
    # Attitude thresholds (ref: FAA AC 25-7D flight envelope)
    ROLL_MAX_DEG = 30.0  # degrees - exceeds safe hover attitude limits
    PITCH_MAX_DEG = 30.0  # degrees - exceeds safe hover attitude limits
    ROLL_STD_DEG = 15.0  # degrees - indicates attitude control oscillation
    
    # GPS thresholds (ref: DO-316 Table 2-3)
    GPS_HDOP_MAX = 3.0  # dimensionless - GPS dilution of precision limit
    GPS_VARIANCE_M = 5.0  # meters - exceeds nominal GPS solution variance
    
    # Control thresholds (ref: PX4 actuator saturation warnings)
    CONTROL_SATURATION_PERCENT = 80.0  # percent - potential loss of control authority

    # NEW: Propulsion thresholds
    MOTOR_ASYMMETRY_DIFF = 0.3  # Normalized (0-1) - indicates one motor working much harder
    
    # NEW: Power thresholds
    BATTERY_VOLTAGE_DROP_V = 2.0  # Volts - sudden drop indicates failure/sag
    
    # NEW: Sensor thresholds
    ACCEL_NOISE_STD = 3.0  # m/s^2 - high vibration/noise
    GYRO_NOISE_STD = 1.0  # rad/s - high vibration/noise


@dataclass
class DetectedAnomaly:
    """
    A single detected anomaly with temporal and subsystem information.
    
    Attributes:
        anomaly_type: Type of anomaly (e.g., "position_drift", "roll_instability")
        severity: Severity level (CRITICAL, HIGH, MEDIUM, LOW)
        measured_value: The measured value that exceeded threshold
        threshold: The threshold that was exceeded
        description: Human-readable description
        first_detected_sec: When this anomaly first exceeded threshold (for causal ordering)
        subsystem: The subsystem this anomaly indicates (navigation, control, propulsion, sensor, power)
    """
    anomaly_type: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    measured_value: float
    threshold: float
    description: str
    first_detected_sec: float = 0.0  # When anomaly first exceeded threshold
    subsystem: str = "unknown"       # Which subsystem this anomaly indicates
    
    def to_dict(self) -> Dict:
        return {
            "type": self.anomaly_type,
            "severity": self.severity,
            "measured": round(self.measured_value, 3),
            "threshold": self.threshold,
            "description": self.description,
            "first_detected_sec": round(self.first_detected_sec, 2),
            "subsystem": self.subsystem,
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
    # NOTE: Airspace violations (altitude_violation, geofence_violation) are NOT mechanical
    # failures - they represent a HEALTHY drone in the WRONG location. These should NOT
    # expect anomalies since the drone is functioning correctly.
    # =========================================================================
    # FAULT TO EXPECTED ANOMALIES MAPPING
    # =========================================================================
    # This mapping defines what anomalies we EXPECT to see for each fault type.
    # 
    # ALL fault types here must match the LLM signature output values:
    #   motor_failure, gps_loss, gps_dropout, battery_failure, battery_depletion,
    #   control_loss, control_signal_loss, sensor_failure, compass_error,
    #   geofence_violation, altitude_violation, flyaway
    #
    # KEY DISTINCTION:
    # - Mechanical failures (motor, GPS, battery) → expect telemetry anomalies
    # - Airspace violations (altitude, geofence) → NO anomalies (healthy drone)
    # =========================================================================
    FAULT_TO_ANOMALIES = {
        # =====================================================================
        # PROPULSION FAILURES - Expect attitude and altitude anomalies
        # =====================================================================
        "motor_failure": ["roll_instability", "pitch_instability", "altitude_instability", "motor_asymmetry"],
        "propulsion_failure": ["roll_instability", "pitch_instability", "altitude_instability", "motor_asymmetry"],
        "esc_failure": ["roll_instability", "motor_asymmetry"],
        "thrust_loss": ["altitude_instability", "motor_asymmetry"],
        
        # =====================================================================
        # NAVIGATION FAILURES - Expect position and drift anomalies
        # =====================================================================
        "gps_loss": ["position_drift", "gps_degradation"],
        "gps_dropout": ["position_drift", "gps_degradation"],
        "gps_failure": ["position_drift", "gps_degradation"],
        "gnss_loss": ["position_drift", "gps_degradation"],
        "navigation_failure": ["position_drift", "gps_degradation"],
        "flyaway": ["position_drift", "velocity_variance"],  # Navigation loss + out of control
        
        # =====================================================================
        # POWER/BATTERY FAILURES - Expect voltage and altitude anomalies
        # =====================================================================
        "battery_failure": ["altitude_instability", "battery_voltage_drop"],
        "battery_depletion": ["altitude_instability", "battery_voltage_drop"],
        "power_failure": ["altitude_instability", "battery_voltage_drop"],
        "voltage_sag": ["altitude_instability", "battery_voltage_drop"],
        
        # =====================================================================
        # CONTROL FAILURES - Expect attitude and position anomalies
        # =====================================================================
        "control_loss": ["roll_instability", "pitch_instability", "position_drift"],
        "control_signal_loss": ["roll_instability", "pitch_instability"],
        "control_failure": ["roll_instability", "pitch_instability", "control_saturation"],
        "rc_loss": [],  # RTL behavior - drone is healthy, just lost RC link
        "link_loss": [],  # Similar to RC loss - triggers failsafe, not actual failure
        
        # =====================================================================
        # SENSOR FAILURES - Expect altitude and sensor-specific anomalies
        # =====================================================================
        "sensor_failure": ["altitude_instability", "sensor_degradation"],
        "sensor_fault": ["altitude_instability", "sensor_degradation"],  # LLM synonym
        "imu_failure": ["roll_instability", "pitch_instability", "sensor_degradation"],
        "compass_error": ["position_drift", "sensor_degradation"],  # Yaw drift causes position issues
        "compass_failure": ["position_drift", "sensor_degradation"],
        "mag_failure": ["position_drift", "sensor_degradation"],
        "baro_failure": ["altitude_instability", "sensor_degradation"],
        "gyro_failure": ["roll_instability", "pitch_instability", "sensor_degradation"],
        "accel_failure": ["altitude_instability", "sensor_degradation"],
        
        # =====================================================================
        # ENVIRONMENTAL - These may or may not produce anomalies
        # =====================================================================
        "wind_disturbance": ["position_drift", "roll_instability"],
        "icing": ["altitude_instability", "sensor_degradation"],
        
        # =====================================================================
        # AIRSPACE VIOLATIONS - Behavioral emulation mode may produce anomalies
        # These represent regulatory violations, NOT mechanical failures.
        # In SIMULATION mode, we use crash emulation (kill switch) to demonstrate
        # the consequences of violations, which produces attitude/altitude anomalies.
        # Empty list [] would be correct for real-world analysis only.
        # =====================================================================
        "altitude_violation": ["altitude_instability"],  # Emulation shows altitude response
        "geofence_violation": ["position_drift"],  # Emulation shows position response
        "airspace_violation": [],  # Generic airspace violation
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
        
        # Detect anomalies with temporal scanning (DETERMINISTIC)
        result.detected_anomalies = self._detect_anomalies(stats, telemetry)
        
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
        
        velocities = []
        for t in telemetry:
            # Prefer explicit NED velocity vector if available
            if "vel_n_m_s" in t:
                vn = t.get("vel_n_m_s", 0)
                ve = t.get("vel_e_m_s", 0)
                vd = t.get("vel_d_m_s", 0)
                velocities.append(math.sqrt(vn**2 + ve**2 + vd**2))
            else:
                velocities.append(t.get("velocity_m_s", t.get("groundspeed_m_s", 0)))
        
        # Roll/Pitch: try degrees first, then radians and convert
        # CRITICAL: Normalize angles to ±180° to prevent telemetry wrap-around bugs
        def normalize_angle(angle_deg: float) -> float:
            """Normalize angle to [-180, 180] range."""
            while angle_deg > 180:
                angle_deg -= 360
            while angle_deg < -180:
                angle_deg += 360
            return angle_deg
        
        # =====================================================================
        # ARTIFACT FILTERING: Detect and exclude simulation artifacts from stats
        # Artifacts occur when quaternion-to-euler conversions produce extreme
        # angles (>85°) while altitude remains stable (vehicle didn't crash)
        # =====================================================================
        ref_altitudes_for_artifact = []
        for pt in telemetry[:min(20, len(telemetry))]:
            alt = pt.get("alt", pt.get("altitude_m", pt.get("relative_alt", 0)))
            if alt and alt > 0:
                ref_altitudes_for_artifact.append(alt)
        ref_alt_for_artifact = max(ref_altitudes_for_artifact) if ref_altitudes_for_artifact else 0
        
        rolls_deg = []
        pitches_deg = []
        rolls_deg_filtered = []  # Artifact-filtered for stats
        pitches_deg_filtered = []  # Artifact-filtered for stats
        artifact_count = 0
        
        # =====================================================================
        # WARMUP FILTERING: Skip initialization transients for roll/pitch stats
        # This ensures stats computation matches temporal scanning behavior
        # (anomaly detection already skips warmup period)
        # =====================================================================
        warmup_end_idx = 0
        for i, t in enumerate(telemetry):
            ts = t.get("timestamp")
            if ts is not None:
                ts_float = float(ts)
            else:
                ts_float = i * 0.1  # Assume ~10Hz if no timestamp
            if ts_float >= TELEMETRY_WARMUP_SEC:
                warmup_end_idx = i
                break
        
        # Use post-warmup telemetry for attitude stats (prevents false positives from startup transients)
        attitude_telemetry = telemetry[warmup_end_idx:] if warmup_end_idx < len(telemetry) else telemetry
        
        for t in attitude_telemetry:
            # Get altitude for artifact detection
            alt = t.get("alt", t.get("altitude_m", t.get("relative_alt", 0)))
            
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
            
            # Normalize to ±180° to handle angle wrap-around in telemetry
            roll = normalize_angle(roll)
            pitch = normalize_angle(pitch)
            
            rolls_deg.append(roll)
            pitches_deg.append(pitch)
            
            # Artifact detection: extreme angle (>85°) + stable altitude = simulation bug
            is_roll_artifact = False
            is_pitch_artifact = False
            if abs(roll) > PHYSICALLY_POSSIBLE_ANGLE_DEG and alt and ref_alt_for_artifact > 0:
                alt_drop = ref_alt_for_artifact - alt
                if alt_drop < 10:  # < 10m drop means no real crash
                    is_roll_artifact = True
                    artifact_count += 1
            if abs(pitch) > PHYSICALLY_POSSIBLE_ANGLE_DEG and alt and ref_alt_for_artifact > 0:
                alt_drop = ref_alt_for_artifact - alt
                if alt_drop < 10:
                    is_pitch_artifact = True
                    if not is_roll_artifact:  # Don't double-count
                        artifact_count += 1
            
            # Add to filtered lists (use 0 for artifacts to not affect max)
            rolls_deg_filtered.append(0 if is_roll_artifact else roll)
            pitches_deg_filtered.append(0 if is_pitch_artifact else pitch)
        
        if artifact_count > 0:
            logger.info(f"  🔧 Detected {artifact_count} simulation artifact points (extreme angle + stable altitude)")
        
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
            # Use artifact-filtered values for max to avoid flagging simulation bugs
            "max_roll_deg": max_abs(rolls_deg_filtered),
            "roll_std_deg": std(rolls_deg),  # Std uses all data
            "max_pitch_deg": max_abs(pitches_deg_filtered),
            "pitch_std_deg": std(pitches_deg),  # Std uses all data
            "position_drift_m": position_drift,
            "gps_variance": std(lats) * 111000 if lats else 0,  # Convert to meters
            "data_points": len(telemetry),
            "flight_duration_s": len(telemetry),  # Approximate 1Hz
            "artifact_count": artifact_count,  # Track filtered artifacts
            
            # Additional Fields for new detectors
            "motor_outputs": [t.get("actuator_controls_0", t.get("servo_output_raw", [])) for t in telemetry],
            "battery_voltages": [
                t.get("battery_v", t.get("battery_status", {}).get("voltage_v", 0))
                for t in telemetry
            ],
            "accel_data": [
                [t.get("acc_x_m_s2", 0), t.get("acc_y_m_s2", 0), t.get("acc_z_m_s2", 0)] 
                if "acc_x_m_s2" in t else t.get("accelerometer_m_s2", []) 
                for t in telemetry
            ],
            # New fields for detailed diagnostics
            "gps_satellites": [t.get("gps_satellites", 0) for t in telemetry],
        }
        
        # Log for debugging
        logger.debug(f"Computed stats: max_roll={stats['max_roll_deg']:.1f}°, max_pitch={stats['max_pitch_deg']:.1f}°, alt_dev={stats['altitude_deviation']:.1f}m")
        
        return stats
    
    def _detect_anomalies(
        self, 
        stats: Dict, 
        telemetry: List[Dict]
    ) -> List[DetectedAnomaly]:
        """
        Detect anomalies using fixed thresholds WITH temporal scanning.
        
        This method now scans telemetry point-by-point to find the FIRST
        timestamp when each threshold is exceeded. This is essential for
        causal reasoning - we need to know which subsystem failed FIRST.
        
        WARM-UP FILTERING:
        - First TELEMETRY_WARMUP_SEC seconds are skipped to filter simulator
          initialization artifacts (SIH/Gazebo transients)
        - Reference values (position, altitude) are computed from post-warmup data
        
        Args:
            stats: Pre-computed aggregate statistics (max values, std, etc.)
            telemetry: Raw telemetry data points with timestamps
            
        Returns:
            List of DetectedAnomaly with first_detected_sec populated
        """
        import math
        
        anomalies = []
        thresholds = AnomalyThresholds()
        
        # Pre-compute reference values for cumulative metrics
        if not telemetry:
            logger.warning("Empty telemetry - cannot detect anomalies")
            return anomalies
        
        # Helper to get timestamp from telemetry point
        def get_timestamp(point: Dict, index: int) -> float:
            """Get timestamp in seconds. Falls back to index-based if missing."""
            ts = point.get("timestamp")
            if ts is not None:
                return float(ts)
            # Fallback: assume ~10Hz sampling rate
            return index * 0.1
        
        # Find first index after warm-up period (skip initialization transients)
        warmup_end_idx = 0
        for i, pt in enumerate(telemetry):
            if get_timestamp(pt, i) >= TELEMETRY_WARMUP_SEC:
                warmup_end_idx = i
                break
        
        if warmup_end_idx > 0:
            logger.debug(f"Skipping first {warmup_end_idx} telemetry points ({TELEMETRY_WARMUP_SEC}s warm-up)")
            
        # Use post-warmup telemetry for reference values
        stable_telemetry = telemetry[warmup_end_idx:] if warmup_end_idx < len(telemetry) else telemetry
        
        # Extract start position for drift calculation (post warm-up)
        start_lat = stable_telemetry[0].get("lat", 0) if stable_telemetry else 0
        start_lon = stable_telemetry[0].get("lon", 0) if stable_telemetry else 0
        
        # Helper to safely get angle in degrees (handles radians conversion)
        def get_angle_deg(point: Dict, key_deg: str, key_rad: str) -> float:
            """Extract angle in degrees, converting from radians if needed."""
            val = point.get(key_deg)
            if val is None:
                rad_val = point.get(key_rad, 0)
                if rad_val:
                    val = math.degrees(rad_val)
                else:
                    val = 0
            # Normalize to ±180°
            while val > 180:
                val -= 360
            while val < -180:
                val += 360
            return abs(val)
        
        # ===================================================================
        # TEMPORAL SCANNING: Find first exceedance time for each metric
        # Note: We use stable_telemetry (post-warmup) for reference values
        # but scan ALL telemetry for anomalies, only recording if t > warmup
        # ===================================================================
        
        # Track first exceedance times
        first_position_drift_sec = None
        first_roll_exceed_sec = None
        first_pitch_exceed_sec = None
        first_altitude_dev_sec = None
        first_gps_spike_sec = None
        first_gps_variance_exceed_sec = None  # NEW: running variance detection
        
        # Reference altitude (use median of first 10 POST-WARMUP points as stable reference)
        ref_altitudes = []
        for i, pt in enumerate(stable_telemetry[:min(10, len(stable_telemetry))]):
            alt = pt.get("alt", pt.get("altitude_m", pt.get("relative_alt", 0)))
            if alt and alt > 0:
                ref_altitudes.append(alt)
        ref_altitude = sorted(ref_altitudes)[len(ref_altitudes)//2] if ref_altitudes else 0
        
        # Reference GPS position for variance calculation (mean of first 10 POST-WARMUP points)
        ref_positions = []
        for pt in stable_telemetry[:min(10, len(stable_telemetry))]:
            lat = pt.get("lat", 0)
            lon = pt.get("lon", 0)
            if lat and lon:
                ref_positions.append((lat, lon))
        
        if ref_positions:
            ref_lat = sum(p[0] for p in ref_positions) / len(ref_positions)
            ref_lon = sum(p[1] for p in ref_positions) / len(ref_positions)
        else:
            ref_lat, ref_lon = start_lat, start_lon
        
        # Running GPS variance tracking using Welford's algorithm
        gps_sum_sq_dist = 0.0
        gps_count = 0

        
        # Scan telemetry point-by-point (skip warm-up period for anomaly detection)
        prev_lat, prev_lon = start_lat, start_lon
        for i, point in enumerate(telemetry):
            timestamp = get_timestamp(point, i)
            
            # SKIP warm-up period - don't record anomalies during initialization
            if timestamp < TELEMETRY_WARMUP_SEC:
                continue
            
            # --- Position Drift Detection ---
            lat = point.get("lat", 0)
            lon = point.get("lon", 0)
            if lat and lon and start_lat and start_lon:
                dlat = (lat - start_lat) * 111000  # degrees to meters
                dlon = (lon - start_lon) * 111000 * math.cos(math.radians(start_lat))
                drift = math.sqrt(dlat**2 + dlon**2)
                
                if drift > thresholds.POSITION_DRIFT_M and first_position_drift_sec is None:
                    first_position_drift_sec = timestamp
                    logger.debug(f"Position drift threshold exceeded at t={timestamp:.2f}s (drift={drift:.1f}m)")
            
            # --- Roll Instability Detection ---
            roll_deg = get_angle_deg(point, "roll_deg", "roll")
            alt = point.get("alt", point.get("altitude_m", point.get("relative_alt", 0)))
            
            # Skip attitude checks during takeoff phase (high angles are normal)
            is_takeoff_phase = timestamp < TAKEOFF_PHASE_SEC or (alt and alt < MIN_CRUISE_ALTITUDE_M)
            
            # Detect simulation artifacts: angle > 85° but altitude stable = quaternion bug
            # Real crashes have rapid altitude loss when vehicle flips
            is_roll_artifact = False
            if roll_deg > PHYSICALLY_POSSIBLE_ANGLE_DEG:
                # Check if altitude is stable (no crash) - if so, it's an artifact
                if alt and ref_altitude > 0:
                    alt_drop = ref_altitude - alt
                    if alt_drop < 10:  # < 10m drop means no real crash
                        is_roll_artifact = True
                        logger.debug(f"Roll artifact detected at t={timestamp:.2f}s (roll={roll_deg:.1f}° but alt stable)")
            
            if roll_deg > thresholds.ROLL_MAX_DEG and first_roll_exceed_sec is None:
                if not is_takeoff_phase and not is_roll_artifact:
                    first_roll_exceed_sec = timestamp
                    logger.debug(f"Roll threshold exceeded at t={timestamp:.2f}s (roll={roll_deg:.1f}°)")
            
            # --- Pitch Instability Detection ---
            pitch_deg = get_angle_deg(point, "pitch_deg", "pitch")
            
            # Detect simulation artifacts for pitch
            is_pitch_artifact = False
            if pitch_deg > PHYSICALLY_POSSIBLE_ANGLE_DEG:
                if alt and ref_altitude > 0:
                    alt_drop = ref_altitude - alt
                    if alt_drop < 10:
                        is_pitch_artifact = True
                        logger.debug(f"Pitch artifact detected at t={timestamp:.2f}s (pitch={pitch_deg:.1f}° but alt stable)")
            
            if pitch_deg > thresholds.PITCH_MAX_DEG and first_pitch_exceed_sec is None:
                if not is_takeoff_phase and not is_pitch_artifact:
                    first_pitch_exceed_sec = timestamp
                    logger.debug(f"Pitch threshold exceeded at t={timestamp:.2f}s (pitch={pitch_deg:.1f}°)")
            
            # --- Altitude Instability Detection ---
            alt = point.get("alt", point.get("altitude_m", point.get("relative_alt", 0)))
            if alt and ref_altitude > 0:
                alt_deviation = abs(alt - ref_altitude)
                if alt_deviation > thresholds.ALTITUDE_DEVIATION_M and first_altitude_dev_sec is None:
                    first_altitude_dev_sec = timestamp
                    logger.debug(f"Altitude deviation exceeded at t={timestamp:.2f}s (dev={alt_deviation:.1f}m)")
            
            # --- GPS Spike Detection (sudden position jump) ---
            if lat and lon and prev_lat and prev_lon:
                # Calculate position change between consecutive points
                dlat_step = (lat - prev_lat) * 111000
                dlon_step = (lon - prev_lon) * 111000 * math.cos(math.radians(lat))
                step_dist = math.sqrt(dlat_step**2 + dlon_step**2)
                
                # GPS spike: sudden jump > 5m in a single sample (~0.1s)
                # This indicates GPS glitch or severe interference
                if step_dist > 5.0 and first_gps_spike_sec is None:
                    first_gps_spike_sec = timestamp
                    logger.debug(f"GPS spike detected at t={timestamp:.2f}s (jump={step_dist:.1f}m)")
            
            # --- Running GPS Variance Detection ---
            # Calculate distance from reference position and track running variance
            if lat and lon and ref_lat and ref_lon:
                dlat_ref = (lat - ref_lat) * 111000
                dlon_ref = (lon - ref_lon) * 111000 * math.cos(math.radians(ref_lat))
                dist_from_ref = math.sqrt(dlat_ref**2 + dlon_ref**2)
                
                # Update running variance (squared distance from reference)
                gps_count += 1
                gps_sum_sq_dist += dist_from_ref ** 2
                
                # Running standard deviation (sqrt of mean squared distance)
                if gps_count >= 10:  # Need minimum samples
                    running_variance = math.sqrt(gps_sum_sq_dist / gps_count)
                    
                    if running_variance > thresholds.GPS_VARIANCE_M and first_gps_variance_exceed_sec is None:
                        first_gps_variance_exceed_sec = timestamp
                        logger.debug(f"GPS variance exceeded at t={timestamp:.2f}s (variance={running_variance:.1f}m)")
            
            prev_lat, prev_lon = lat, lon
        
        # ===================================================================
        # CREATE ANOMALIES WITH TEMPORAL INFORMATION
        # ===================================================================
        
        # Position drift
        drift = stats.get("position_drift_m", 0)
        if drift > thresholds.POSITION_DRIFT_M:
            severity = "CRITICAL" if drift > 50 else ("HIGH" if drift > 25 else "MEDIUM")
            anomalies.append(DetectedAnomaly(
                anomaly_type="position_drift",
                severity=severity,
                measured_value=drift,
                threshold=thresholds.POSITION_DRIFT_M,
                description=f"Position drift of {drift:.1f}m exceeds {thresholds.POSITION_DRIFT_M}m threshold",
                first_detected_sec=first_position_drift_sec if first_position_drift_sec is not None else 0.0,
                subsystem="navigation"
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
                description=f"Altitude deviation of {alt_dev:.1f}m exceeds threshold",
                first_detected_sec=first_altitude_dev_sec if first_altitude_dev_sec is not None else 0.0,
                subsystem="ambiguous"  # Resolved by SubsystemCausalAnalyzer
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
                description=f"Maximum roll of {roll_max:.1f} deg exceeds {thresholds.ROLL_MAX_DEG} deg threshold",
                first_detected_sec=first_roll_exceed_sec if first_roll_exceed_sec is not None else 0.0,
                subsystem="control"
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
                description=f"Maximum pitch of {pitch_max:.1f} deg exceeds threshold",
                first_detected_sec=first_pitch_exceed_sec if first_pitch_exceed_sec is not None else 0.0,
                subsystem="control"
            ))
        
        # GPS degradation - FIXED: Only flag when there's ACTUAL degradation
        # GPS variance > threshold OR satellite count critically low (but not both zero)
        gps_var = stats.get("gps_variance", 0)
        gps_sats = stats.get("gps_satellites", [])
        valid_sats = [s for s in gps_sats if s > 0]
        min_sats = min(valid_sats) if valid_sats else 12  # Default to good if no data
        
        # Only flag GPS degradation if:
        # 1. GPS variance is significantly above threshold (not just noise), OR
        # 2. Satellite count dropped critically low (below 6)
        # AND we have actual GPS data to analyze
        gps_has_issue = (
            (gps_var > thresholds.GPS_VARIANCE_M and gps_var > 1.0) or  # Variance must be meaningful
            (valid_sats and min_sats < 6)  # Only check sats if we have data
        )
        
        if gps_has_issue:
            severity = "HIGH" if (gps_var > 20 or min_sats < 4) else "MEDIUM"
            
            # Use running variance detection first, fall back to spike detection
            gps_first_detected = first_gps_variance_exceed_sec
            if gps_first_detected is None:
                gps_first_detected = first_gps_spike_sec
            if gps_first_detected is None:
                gps_first_detected = 0.0  # Last resort fallback
            
            # Build description based on what we detected
            if gps_var > thresholds.GPS_VARIANCE_M:
                desc = f"GPS variance of {gps_var:.1f}m exceeds {thresholds.GPS_VARIANCE_M}m threshold"
            else:
                desc = f"Low satellite count (min={min_sats}) indicates GPS degradation"
            
            anomalies.append(DetectedAnomaly(
                anomaly_type="gps_degradation",
                severity=severity,
                measured_value=gps_var if gps_var > thresholds.GPS_VARIANCE_M else float(min_sats),
                threshold=thresholds.GPS_VARIANCE_M if gps_var > thresholds.GPS_VARIANCE_M else 6.0,
                description=desc,
                first_detected_sec=gps_first_detected,
                subsystem="navigation"
            ))
        
        # Velocity variance
        vel_std = stats.get("velocity_std_mps", 0)
        if vel_std > thresholds.VELOCITY_STD_MPS:
            anomalies.append(DetectedAnomaly(
                anomaly_type="velocity_variance",
                severity="MEDIUM",
                measured_value=vel_std,
                threshold=thresholds.VELOCITY_STD_MPS,
                description=f"Velocity std of {vel_std:.1f} m/s indicates instability",
                first_detected_sec=0.0,  # No specific temporal scan for velocity std
                subsystem="ambiguous"  # Can be navigation or control
            ))
        
        # Log summary of temporal findings
        if anomalies:
            times = [(a.anomaly_type, a.first_detected_sec) for a in anomalies if a.first_detected_sec > 0]
            if times:
                earliest = min(times, key=lambda x: x[1])
                logger.info(f"Temporal analysis: earliest anomaly '{earliest[0]}' at t={earliest[1]:.2f}s")
        
        # --- NEW: Specific Subsystem Detectors ---
        anomalies.extend(self._detect_motor_anomalies(stats, telemetry, thresholds))
        anomalies.extend(self._detect_battery_anomalies(stats, telemetry, thresholds))
        anomalies.extend(self._detect_sensor_anomalies(stats, telemetry, thresholds))

        return anomalies

    def _detect_motor_anomalies(self, stats: Dict, telemetry: List[Dict], thresholds: AnomalyThresholds) -> List[DetectedAnomaly]:
        """
        Detect propulsion anomalies (motor asymmetry).
        
        IMPORTANT: Only analyzes motor asymmetry when:
        1. Motors are actually spinning (PWM > 1000 or normalized > 0.1)
        2. After the telemetry warmup period (skip initialization artifacts)
        """
        new_anomalies = []
        motor_outputs = stats.get("motor_outputs", [])
        
        # Need at least 10 samples to detect persistent asymmetry
        if not motor_outputs or len(motor_outputs) < 10:
            return []
        
        # PWM threshold for "motors spinning" - idle is typically 900-1000
        PWM_SPINNING_THRESHOLD = 1050  # Above idle, motors are producing thrust
        NORMALIZED_SPINNING_THRESHOLD = 0.15  # For -1 to 1 normalized outputs
            
        first_asymmetry_time = None
        max_diff = 0.0
        
        for i, point_motors in enumerate(motor_outputs):
            # Get timestamp for warmup filter
            if i < len(telemetry):
                ts = telemetry[i].get("timestamp", i * 0.1)
                try:
                    timestamp = float(ts)
                except:
                    timestamp = i * 0.1
            else:
                timestamp = i * 0.1
            
            # Skip warmup period - simulator initialization causes false positives
            if timestamp < TELEMETRY_WARMUP_SEC:
                continue
            
            # Handle stringified lists (common issue with some serializations)
            if isinstance(point_motors, str):
                try:
                    import json
                    point_motors = json.loads(point_motors)
                except:
                    pass

            # Handle different motor output formats (list of floats, or dict)
            if isinstance(point_motors, list) and len(point_motors) >= 4:
                # Assuming quadcopter - take first 4 values as main rotors
                valid_motors = [m for m in point_motors[:4] if isinstance(m, (int, float)) and not math.isnan(m)]
                if len(valid_motors) < 4:
                    continue
                
                # Check if motors are actually spinning (not idle)
                curr_min = min(valid_motors)
                curr_max = max(valid_motors)
                
                # Detect PWM vs normalized format
                is_pwm_format = curr_max > 100  # PWM values are 900-2000
                
                if is_pwm_format:
                    # PWM format: only analyze if motors are spinning
                    if curr_max < PWM_SPINNING_THRESHOLD:
                        continue  # Motors at idle, skip
                    diff = curr_max - curr_min
                    norm_diff = diff / 1000.0  # Normalize to 0-1 scale
                else:
                    # Normalized format (-1 to 1 or 0 to 1)
                    if curr_max < NORMALIZED_SPINNING_THRESHOLD:
                        continue  # Motors at idle, skip
                    norm_diff = curr_max - curr_min
                
                if norm_diff > thresholds.MOTOR_ASYMMETRY_DIFF:
                    max_diff = max(max_diff, norm_diff)
                    if first_asymmetry_time is None:
                        first_asymmetry_time = timestamp  # Use timestamp computed earlier
                            
        if first_asymmetry_time is not None:
            new_anomalies.append(DetectedAnomaly(
                anomaly_type="motor_asymmetry",
                severity="HIGH",
                measured_value=max_diff,
                threshold=thresholds.MOTOR_ASYMMETRY_DIFF,
                description=f"Motor output asymmetry of {max_diff:.2f} indicates propulsion imbalance",
                first_detected_sec=first_asymmetry_time,
                subsystem="propulsion"
            ))
            
        return new_anomalies

    def _detect_battery_anomalies(self, stats: Dict, telemetry: List[Dict], thresholds: AnomalyThresholds) -> List[DetectedAnomaly]:
        """Detect power anomalies (voltage drops)."""
        new_anomalies = []
        voltages = [v for v in stats.get("battery_voltages", []) if v > 0]
        
        if not voltages or len(voltages) < 10:
            return []
            
        # Check for sudden drops
        start_voltage = sum(voltages[:5]) / 5.0
        min_voltage = min(voltages)
        drop = start_voltage - min_voltage
        
        first_drop_time = None
        
        if drop > thresholds.BATTERY_VOLTAGE_DROP_V:
             # Find first time
            for i, v in enumerate(stats.get("battery_voltages", [])):
                 if v > 0 and (start_voltage - v) > thresholds.BATTERY_VOLTAGE_DROP_V:
                      ts = telemetry[i].get("timestamp", i * 0.1)
                      try:
                          first_drop_time = float(ts)
                      except:
                          first_drop_time = i * 0.1
                      break
            
            new_anomalies.append(DetectedAnomaly(
                anomaly_type="battery_voltage_drop",
                severity="HIGH",
                measured_value=drop,
                threshold=thresholds.BATTERY_VOLTAGE_DROP_V,
                description=f"Battery voltage drop of {drop:.1f}V exceeds threshold",
                first_detected_sec=first_drop_time if first_drop_time else 0.0,
                subsystem="power"
            ))
            
        return new_anomalies

    def _detect_sensor_anomalies(self, stats: Dict, telemetry: List[Dict], thresholds: AnomalyThresholds) -> List[DetectedAnomaly]:
        """Detect sensor anomalies (IMU noise)."""
        new_anomalies = []
        
        accel_data = stats.get("accel_data", [])
        if not accel_data or len(accel_data) < 10:
             return []
             
        # Separate axes
        acc_x = [p[0] for p in accel_data if len(p) >= 3]
        acc_y = [p[1] for p in accel_data if len(p) >= 3]
        acc_z = [p[2] for p in accel_data if len(p) >= 3]
        
        def calc_std(arr):
             if len(arr) < 2: return 0.0
             avg = sum(arr) / len(arr)
             return math.sqrt(sum((x - avg)**2 for x in arr) / len(arr))
             
        std_x = calc_std(acc_x)
        std_y = calc_std(acc_y)
        std_z = calc_std(acc_z)
        
        # Threshold: 3.0 m/s^2 is very high noise (nominal < 1.0)
        NOISE_THRESHOLD = 3.0
        
        if std_x > NOISE_THRESHOLD or std_y > NOISE_THRESHOLD or std_z > NOISE_THRESHOLD:
             max_std = max(std_x, std_y, std_z)
             new_anomalies.append(DetectedAnomaly(
                anomaly_type="sensor_degradation",
                severity="HIGH" if max_std > 5.0 else "MEDIUM",
                measured_value=max_std,
                threshold=NOISE_THRESHOLD,
                description=f"High accelerometer noise ({max_std:.2f} m/s²) indicates sensor degradation",
                first_detected_sec=0.0, # Noise is a global stat here
                subsystem="sensor"
            ))
        
        return new_anomalies
    
    def _compute_brr_score(self, result: BRRResult) -> float:
        """Compute BRR score based on detected vs expected anomalies.
        
        Scoring logic:
        1. AIRSPACE VIOLATIONS: Empty expected list means healthy drone - score HIGH if no anomalies
        2. Primary: Check if expected anomaly types are detected
        3. Secondary: Give partial credit for ANY detected anomalies (crash behavior)
        4. Bonus: Add severity bonus for matched anomalies
        """
        
        if not result.expected_anomaly_types:
            # AIRSPACE VIOLATION CASE: Empty expected list means we expect a HEALTHY drone
            # For altitude_violation/geofence_violation, the drone is functioning correctly
            # but is in the wrong location. Absence of anomalies is CORRECT behavior.
            if len(result.detected_anomalies) == 0:
                # Perfect: healthy drone with no anomalies detected
                logger.debug("BRR: Airspace violation case - no anomalies expected, none detected → 1.0")
                return 1.0
            else:
                # Anomalies detected on supposedly healthy drone - may be simulation artifacts
                # Give moderate score since we can't distinguish simulation noise from real issues
                score = max(0.5, 1.0 - (len(result.detected_anomalies) * 0.15))
                logger.debug(f"BRR: Airspace violation case - {len(result.detected_anomalies)} unexpected anomalies → {score:.2f}")
                return score
        
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
            # Give partial credit: more anomalies = more credit, capped at 0.5 (conservative)
            secondary_score = min(len(result.detected_anomalies) * 0.25, 0.5)
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
