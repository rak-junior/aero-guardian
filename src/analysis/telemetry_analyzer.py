"""
Telemetry Analyzer
==================
Author: AeroGuardian Member
Date: 2026-01-18
Updated: 2026-01-23

Analyzes raw flight telemetry to produce comprehensive engineering statistics.
Detects anomalies in stability, vibration, battery, GPS, and position.

IMPROVEMENTS:
- Added position drift analysis (critical for GPS failure scenarios)
- Added altitude stability metrics
- Added speed analysis
- Added GPS quality metrics
- Added control saturation detection
- Enhanced anomaly detection with severity levels
"""

import logging
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TelemetryStats:
    """Comprehensive flight telemetry statistics."""
    
    # Basic Flight Metrics
    duration_s: float = 0.0
    data_points: int = 0
    
    # Altitude Analysis
    max_alt_m: float = 0.0
    min_alt_m: float = 0.0
    avg_alt_m: float = 0.0
    alt_std_dev_m: float = 0.0
    alt_deviation_m: float = 0.0  # max - min
    
    # Speed Analysis
    max_speed_mps: float = 0.0
    avg_speed_mps: float = 0.0
    speed_std_dev_mps: float = 0.0
    
    # Attitude Stability (Roll/Pitch)
    max_roll_deg: float = 0.0
    max_pitch_deg: float = 0.0
    roll_std_dev: float = 0.0
    pitch_std_dev: float = 0.0
    roll_oscillation_freq: float = 0.0  # Oscillations per second
    
    # Position Drift (Critical for GPS analysis)
    position_drift_m: float = 0.0  # Max distance from start
    position_variance_m: float = 0.0  # Position scatter
    lateral_drift_m: float = 0.0
    
    # GPS Quality
    gps_satellite_min: int = 0
    gps_satellite_avg: float = 0.0
    gps_hdop_max: float = 0.0
    gps_variance_m: float = 0.0
    
    # Vibration Analysis
    vibration_avg: float = 0.0
    vibration_max: float = 0.0
    
    # Battery Analysis
    battery_start_v: float = 0.0
    battery_end_v: float = 0.0
    battery_sag_rate_vps: float = 0.0
    battery_remaining_pct: float = 0.0
    
    # Control Analysis
    control_saturation_pct: float = 0.0  # % time controls saturated
    
    # Failsafe Events
    failsafe_events: List[str] = field(default_factory=list)
    
    # Detected Anomalies (with severity)
    anomalies: List[str] = field(default_factory=list)
    anomaly_severity: str = "NONE"  # NONE, LOW, MEDIUM, HIGH, CRITICAL
    
    def to_summary_text(self) -> str:
        """Generate comprehensive text summary for LLM consumption."""
        
        # Build structured summary with all relevant metrics
        lines = [
            f"=== FLIGHT TELEMETRY ANALYSIS ===",
            f"",
            f"FLIGHT DURATION: {self.duration_s:.1f} seconds ({self.data_points} data points)",
            f"",
            f"--- ALTITUDE ---",
            f"  Max: {self.max_alt_m:.1f}m ({self.max_alt_m * 3.28:.0f}ft)",
            f"  Avg: {self.avg_alt_m:.1f}m",
            f"  Deviation (max-min): {self.alt_deviation_m:.1f}m",
            f"  Stability (StdDev): {self.alt_std_dev_m:.2f}m",
            f"",
            f"--- ATTITUDE STABILITY ---",
            f"  Max Roll: {self.max_roll_deg:.1f}deg (StdDev: {self.roll_std_dev:.1f}deg)",
            f"  Max Pitch: {self.max_pitch_deg:.1f}deg (StdDev: {self.pitch_std_dev:.1f}deg)",
        ]
        
        # Position metrics (critical for GPS issues)
        if self.position_drift_m > 0 or self.gps_variance_m > 0:
            lines.extend([
                f"",
                f"--- POSITION & GPS ---",
                f"  Position Drift: {self.position_drift_m:.1f}m from start",
                f"  GPS Variance: {self.gps_variance_m:.1f}m",
            ])
            if self.gps_satellite_min > 0:
                lines.append(f"  Satellites: min={self.gps_satellite_min}, avg={self.gps_satellite_avg:.1f}")
            if self.gps_hdop_max > 0:
                lines.append(f"  HDOP Max: {self.gps_hdop_max:.1f}")
        
        # Speed
        if self.max_speed_mps > 0:
            lines.extend([
                f"",
                f"--- SPEED ---",
                f"  Max: {self.max_speed_mps:.1f} m/s",
                f"  Avg: {self.avg_speed_mps:.1f} m/s",
            ])
        
        # Vibration
        if self.vibration_max > 0:
            lines.extend([
                f"",
                f"--- VIBRATION ---",
                f"  Max: {self.vibration_max:.2f}",
                f"  Avg: {self.vibration_avg:.2f}",
            ])
        
        # Battery
        if self.battery_start_v > 0:
            lines.extend([
                f"",
                f"--- BATTERY ---",
                f"  Start: {self.battery_start_v:.2f}V",
                f"  End: {self.battery_end_v:.2f}V",
                f"  Sag Rate: {self.battery_sag_rate_vps:.4f} V/s",
            ])
        
        # Failsafe events
        if self.failsafe_events:
            lines.extend([
                f"",
                f"--- FAILSAFE EVENTS ---",
                f"  {', '.join(self.failsafe_events)}",
            ])
        
        # Anomalies
        lines.extend([
            f"",
            f"--- ANOMALY DETECTION ---",
            f"  Severity: {self.anomaly_severity}",
        ])
        if self.anomalies:
            for anomaly in self.anomalies:
                lines.append(f"  • {anomaly}")
        else:
            lines.append(f"  • No anomalies detected")
        
        return "\n".join(lines)


class TelemetryAnalyzer:
    """
    Analyzes raw flight telemetry to produce comprehensive engineering statistics.
    
    Designed to extract all metrics needed for accurate LLM safety report generation.
    """
    
    # Anomaly detection thresholds
    THRESHOLDS = {
        "roll_instability_high": 30.0,      # degrees
        "roll_instability_critical": 45.0,   # degrees
        "roll_std_warning": 10.0,            # degrees
        "roll_std_critical": 20.0,           # degrees
        "pitch_instability": 30.0,           # degrees
        "altitude_deviation_warning": 10.0,  # meters
        "altitude_deviation_critical": 20.0, # meters
        "position_drift_warning": 15.0,      # meters
        "position_drift_critical": 50.0,     # meters
        "gps_variance_warning": 5.0,         # meters
        "gps_variance_critical": 15.0,       # meters
        "vibration_warning": 3.0,
        "vibration_critical": 5.0,
        "battery_sag_warning": -0.05,        # V/s
        "battery_sag_critical": -0.1,        # V/s
        "satellite_min_warning": 6,
        "satellite_min_critical": 4,
    }
    
    def analyze(self, telemetry: List[Dict]) -> TelemetryStats:
        """Perform comprehensive telemetry analysis."""
        
        if not telemetry:
            return self._empty_stats()
        
        stats = TelemetryStats()
        stats.data_points = len(telemetry)
        
        # Extract arrays
        t = np.array([x.get('timestamp', 0) for x in telemetry])
        stats.duration_s = t[-1] - t[0] if len(t) > 1 else 0
        
        # === ALTITUDE ANALYSIS ===
        alt = np.array([x.get('alt', x.get('altitude_m', 0)) for x in telemetry])
        if len(alt) > 0:
            stats.max_alt_m = float(np.max(alt))
            stats.min_alt_m = float(np.min(alt[alt > 0])) if np.any(alt > 0) else 0
            stats.avg_alt_m = float(np.mean(alt))
            stats.alt_std_dev_m = float(np.std(alt))
            stats.alt_deviation_m = stats.max_alt_m - stats.min_alt_m
        
        # === ATTITUDE ANALYSIS ===
        roll = np.array([x.get('roll', x.get('roll_deg', 0)) for x in telemetry])
        pitch = np.array([x.get('pitch', x.get('pitch_deg', 0)) for x in telemetry])
        
        # Convert from radians if values are small (PX4 sends radians)
        if len(roll) > 0 and np.max(np.abs(roll)) < 2 * np.pi:
            roll = roll * 57.3  # rad to deg
            pitch = pitch * 57.3
        
        if len(roll) > 0:
            stats.max_roll_deg = float(np.max(np.abs(roll)))
            stats.max_pitch_deg = float(np.max(np.abs(pitch)))
            stats.roll_std_dev = float(np.std(roll))
            stats.pitch_std_dev = float(np.std(pitch))
            
            # Roll oscillation frequency estimation
            if len(roll) > 10:
                zero_crossings = np.where(np.diff(np.signbit(roll)))[0]
                if len(zero_crossings) > 1 and stats.duration_s > 0:
                    stats.roll_oscillation_freq = len(zero_crossings) / (2 * stats.duration_s)
        
        # === POSITION ANALYSIS ===
        lat = np.array([x.get('lat', x.get('latitude', 0)) for x in telemetry])
        lon = np.array([x.get('lon', x.get('longitude', 0)) for x in telemetry])
        
        if len(lat) > 0 and np.any(lat != 0):
            # Filter valid positions
            valid_mask = (lat != 0) & (lon != 0)
            lat_valid = lat[valid_mask]
            lon_valid = lon[valid_mask]
            
            if len(lat_valid) > 1:
                # Position drift from start
                start_lat, start_lon = lat_valid[0], lon_valid[0]
                
                # Approximate meters (111000m per degree lat, adjusted for lon)
                dlat_m = (lat_valid - start_lat) * 111000
                dlon_m = (lon_valid - start_lon) * 111000 * np.cos(np.radians(start_lat))
                distances = np.sqrt(dlat_m**2 + dlon_m**2)
                
                stats.position_drift_m = float(np.max(distances))
                stats.position_variance_m = float(np.std(distances))
                stats.lateral_drift_m = float(np.max(np.abs(dlon_m)))
                
                # GPS variance (scatter)
                stats.gps_variance_m = float(np.std(lat_valid) * 111000)
        
        # === SPEED ANALYSIS ===
        speed = np.array([
            x.get('groundspeed_m_s', x.get('velocity_m_s', x.get('speed', 0))) 
            for x in telemetry
        ])
        if len(speed) > 0:
            stats.max_speed_mps = float(np.max(speed))
            stats.avg_speed_mps = float(np.mean(speed))
            stats.speed_std_dev_mps = float(np.std(speed))
        
        # === GPS QUALITY ===
        satellites = np.array([x.get('satellites', x.get('num_satellites', 0)) for x in telemetry])
        hdop = np.array([x.get('hdop', x.get('eph', 0)) for x in telemetry])
        
        if len(satellites) > 0 and np.any(satellites > 0):
            valid_sats = satellites[satellites > 0]
            stats.gps_satellite_min = int(np.min(valid_sats))
            stats.gps_satellite_avg = float(np.mean(valid_sats))
        
        if len(hdop) > 0 and np.any(hdop > 0):
            valid_hdop = hdop[hdop > 0]
            stats.gps_hdop_max = float(np.max(valid_hdop))
        
        # === VIBRATION ANALYSIS ===
        vib_x = np.array([x.get('vibration_x', 0) for x in telemetry])
        vib_y = np.array([x.get('vibration_y', 0) for x in telemetry])
        vib_z = np.array([x.get('vibration_z', 0) for x in telemetry])
        vib_mag = np.sqrt(vib_x**2 + vib_y**2 + vib_z**2)
        
        if len(vib_mag) > 0:
            stats.vibration_avg = float(np.mean(vib_mag))
            stats.vibration_max = float(np.max(vib_mag))
        
        # === BATTERY ANALYSIS ===
        volts = np.array([x.get('voltage', x.get('battery_voltage', 0)) for x in telemetry])
        valid_volts = volts[volts > 0]
        
        if len(valid_volts) > 1:
            stats.battery_start_v = float(valid_volts[0])
            stats.battery_end_v = float(valid_volts[-1])
            
            # Linear regression for sag rate
            try:
                t_volts = t[:len(valid_volts)]
                z = np.polyfit(t_volts, valid_volts, 1)
                stats.battery_sag_rate_vps = float(z[0])
            except:
                pass
        
        # Battery remaining
        remaining = np.array([x.get('battery_remaining', x.get('remaining_percent', 0)) for x in telemetry])
        if len(remaining) > 0 and np.any(remaining > 0):
            stats.battery_remaining_pct = float(remaining[-1])
        
        # === FAILSAFE DETECTION ===
        for entry in telemetry:
            mode = str(entry.get('flight_mode', entry.get('mode', ''))).upper()
            if 'RTL' in mode or 'RETURN' in mode:
                if 'RTL triggered' not in stats.failsafe_events:
                    stats.failsafe_events.append('RTL triggered')
            if 'LAND' in mode and 'AUTO' not in mode:
                if 'Emergency land' not in stats.failsafe_events:
                    stats.failsafe_events.append('Emergency land')
        
        # === ANOMALY DETECTION ===
        stats.anomalies, stats.anomaly_severity = self._detect_anomalies(stats)
        
        return stats
    
    def _detect_anomalies(self, stats: TelemetryStats) -> Tuple[List[str], str]:
        """Detect anomalies and determine severity."""
        
        anomalies = []
        severity_score = 0  # Higher = more severe
        
        T = self.THRESHOLDS
        
        # Roll instability
        if stats.max_roll_deg > T["roll_instability_critical"]:
            anomalies.append(f"CRITICAL: Extreme roll {stats.max_roll_deg:.0f}° (threshold: {T['roll_instability_critical']}°)")
            severity_score += 3
        elif stats.max_roll_deg > T["roll_instability_high"]:
            anomalies.append(f"HIGH: High roll {stats.max_roll_deg:.0f}° (threshold: {T['roll_instability_high']}°)")
            severity_score += 2
        
        # Roll standard deviation (oscillation)
        if stats.roll_std_dev > T["roll_std_critical"]:
            anomalies.append(f"CRITICAL: Roll oscillation StdDev={stats.roll_std_dev:.1f}°")
            severity_score += 3
        elif stats.roll_std_dev > T["roll_std_warning"]:
            anomalies.append(f"HIGH: Roll instability StdDev={stats.roll_std_dev:.1f}°")
            severity_score += 2
        
        # Pitch instability
        if stats.max_pitch_deg > T["pitch_instability"]:
            anomalies.append(f"HIGH: High pitch {stats.max_pitch_deg:.0f}°")
            severity_score += 2
        
        # Altitude deviation
        if stats.alt_deviation_m > T["altitude_deviation_critical"]:
            anomalies.append(f"HIGH: Altitude deviation {stats.alt_deviation_m:.1f}m")
            severity_score += 2
        elif stats.alt_deviation_m > T["altitude_deviation_warning"]:
            anomalies.append(f"MEDIUM: Altitude variation {stats.alt_deviation_m:.1f}m")
            severity_score += 1
        
        # Position drift (critical for GPS failures)
        if stats.position_drift_m > T["position_drift_critical"]:
            anomalies.append(f"CRITICAL: Position drift {stats.position_drift_m:.1f}m")
            severity_score += 3
        elif stats.position_drift_m > T["position_drift_warning"]:
            anomalies.append(f"HIGH: Position drift {stats.position_drift_m:.1f}m")
            severity_score += 2
        
        # GPS variance
        if stats.gps_variance_m > T["gps_variance_critical"]:
            anomalies.append(f"CRITICAL: GPS variance {stats.gps_variance_m:.1f}m")
            severity_score += 3
        elif stats.gps_variance_m > T["gps_variance_warning"]:
            anomalies.append(f"HIGH: GPS degradation {stats.gps_variance_m:.1f}m")
            severity_score += 2
        
        # Low satellite count
        if stats.gps_satellite_min > 0:
            if stats.gps_satellite_min < T["satellite_min_critical"]:
                anomalies.append(f"CRITICAL: Low satellites (min={stats.gps_satellite_min})")
                severity_score += 3
            elif stats.gps_satellite_min < T["satellite_min_warning"]:
                anomalies.append(f"HIGH: Low satellites (min={stats.gps_satellite_min})")
                severity_score += 2
        
        # Vibration
        if stats.vibration_max > T["vibration_critical"]:
            anomalies.append(f"CRITICAL: Severe vibration {stats.vibration_max:.1f}")
            severity_score += 3
        elif stats.vibration_max > T["vibration_warning"]:
            anomalies.append(f"MEDIUM: High vibration {stats.vibration_max:.1f}")
            severity_score += 1
        
        # Battery sag
        if stats.battery_sag_rate_vps < T["battery_sag_critical"]:
            anomalies.append(f"CRITICAL: Battery sag {stats.battery_sag_rate_vps:.3f} V/s")
            severity_score += 3
        elif stats.battery_sag_rate_vps < T["battery_sag_warning"]:
            anomalies.append(f"HIGH: Battery sag {stats.battery_sag_rate_vps:.3f} V/s")
            severity_score += 2
        
        # Failsafe events
        if stats.failsafe_events:
            anomalies.append(f"HIGH: Failsafe triggered ({', '.join(stats.failsafe_events)})")
            severity_score += 2
        
        # Determine overall severity
        if severity_score >= 6:
            severity = "CRITICAL"
        elif severity_score >= 4:
            severity = "HIGH"
        elif severity_score >= 2:
            severity = "MEDIUM"
        elif severity_score >= 1:
            severity = "LOW"
        else:
            severity = "NONE"
        
        return anomalies, severity
    
    def _empty_stats(self) -> TelemetryStats:
        """Return empty stats when no telemetry available."""
        return TelemetryStats()
