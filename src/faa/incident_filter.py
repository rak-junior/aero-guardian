"""
Incident Filter for Simulatable Abnormal Flights
=================================================
Author: AeroGuardian Member
Date: 2026-01-18

Filters FAA incidents to only include those with actual mechanical failures
that can be meaningfully simulated in PX4 SITL.

Updated 2026-01-19: Added altitude detection for high-altitude incursions
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger("AeroGuardian.IncidentFilter")

# Incident types that represent actual drone failures (simulatable)
SIMULATABLE_TYPES = {
    "motor_failure",
    "gps_loss",
    "battery_failure",
    "control_loss",
    "sensor_fault",
}

# Keywords in the Summary that indicate a real failure (fallback)
FAILURE_KEYWORDS = [
    "malfunction",
    "crash",
    "lost control",
    "fell",
    "went down",
    "emergency",
    "failure",
    "lost link",
    "fly away",
    "flyaway",
]

# High-altitude threshold (5000 ft = 1524m) - above this is likely jet encounter
HIGH_ALTITUDE_THRESHOLD_M = 1524  # ~5000 ft / FL50

# Max simulatable drone altitude
MAX_DRONE_ALTITUDE_M = 120  # 400 ft legal limit


def parse_altitude_from_text(text: str) -> Tuple[Optional[float], str]:
    """
    Parse altitude from FAA text and convert to meters.
    
    Returns:
        Tuple of (altitude_in_meters, source_unit)
        
    Handles:
        - FL210 -> 21,000 ft -> 6400 m
        - 5000 feet -> 1524 m
        - 500 ft AGL -> 152 m
        - 100 meters -> 100 m
    """
    if not text:
        return None, "unknown"
    
    text_upper = text.upper()
    
    # Flight Level: FL210 = 21,000 feet
    fl_match = re.search(r'FL\s*(\d{2,3})', text_upper)
    if fl_match:
        fl_value = int(fl_match.group(1))
        feet = fl_value * 100  # FL210 = 21000 feet
        meters = feet * 0.3048
        return meters, f"FL{fl_value}"
    
    # Feet: "5000 feet" or "500 ft" or "1000FT"
    ft_match = re.search(r'(\d{1,5})\s*(?:FEET|FT|\')', text_upper)
    if ft_match:
        feet = int(ft_match.group(1))
        meters = feet * 0.3048
        return meters, f"{feet}ft"
    
    # Meters: "100 meters" or "50m"
    m_match = re.search(r'(\d{1,4})\s*(?:METERS?|M\b)', text_upper)
    if m_match:
        meters = int(m_match.group(1))
        return meters, f"{meters}m"
    
    return None, "not_found"


class IncidentFilter:
    """Filter FAA incidents for simulatable abnormal flights."""
    
    def __init__(self, data_path: Optional[Path] = None):
        # Use original full dataset, but sort drone-testable to front
        self.data_path = data_path or Path("data/processed/faa_incidents/faa_simulatable.json")
        self._incidents = []
        self._simulatable = []
    
    def load(self) -> int:
        """Load and filter incidents. Returns count of simulatable incidents."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"FAA data not found: {self.data_path}")
            
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._incidents = data.get("incidents", [])
        self._simulatable = []
        
        # Separate drone-testable from sightings
        drone_testable = []
        sightings = []
        
        for incident in self._incidents:
            # Enrich incident with altitude and simulation mode
            incident = self._enrich_incident(incident)
            if self._is_simulatable(incident):
                # Sort by simulation_mode: MECHANICAL_TEST first
                if incident.get("simulation_mode") == "MECHANICAL_TEST":
                    drone_testable.append(incident)
                else:
                    sightings.append(incident)
        
        # Drone-testable incidents at front (index 0, 1, 2...)
        self._simulatable = drone_testable + sightings
        
        logger.info(f"Loaded {len(self._incidents)} total incidents")
        logger.info(f"Filtered to {len(self._simulatable)} simulatable incidents")
        logger.info(f"  → Drone-testable (front): {len(drone_testable)} (index 0-{len(drone_testable)-1})")
        logger.info(f"  → Other incidents: {len(sightings)}")
        
        return len(self._simulatable)
    
    def _enrich_incident(self, incident: Dict) -> Dict:
        """Add extracted altitude and simulation mode to incident."""
        desc = incident.get("description", "") + " " + incident.get("summary", "")
        
        # Extract altitude
        altitude_m, source = parse_altitude_from_text(desc)
        incident["extracted_altitude_m"] = altitude_m
        incident["altitude_source"] = source
        
        # Determine simulation mode
        incident["simulation_mode"] = self._determine_simulation_mode(incident, altitude_m, desc)
        
        # Determine simulatable altitude (capped for drone physics)
        if altitude_m and altitude_m > MAX_DRONE_ALTITUDE_M:
            incident["simulatable_altitude_m"] = MAX_DRONE_ALTITUDE_M
            incident["altitude_capped"] = True
        elif altitude_m:
            incident["simulatable_altitude_m"] = altitude_m
            incident["altitude_capped"] = False
        else:
            incident["simulatable_altitude_m"] = 50.0  # Default
            incident["altitude_capped"] = False
        
        return incident
    
    def _determine_simulation_mode(self, incident: Dict, altitude_m: Optional[float], desc: str) -> str:
        """
        Determine simulation mode based on incident characteristics.
        
        Returns:
            - MECHANICAL_TEST: Actual drone failure (motor, GPS, battery)
            - GEOFENCE_TEST: High-altitude or airspace violation (healthy drone, wrong location)
            - AIRSPACE_SIGHTING: Jet/aircraft reported seeing drone (may not be simulatable)
        """
        inc_type = incident.get("incident_type", "unknown").lower()
        desc_lower = desc.lower()
        
        # Check for high-altitude encounter (likely jet seeing drone)
        if altitude_m and altitude_m > HIGH_ALTITUDE_THRESHOLD_M:
            # FL50+ is almost certainly a manned aircraft report
            return "AIRSPACE_SIGHTING"
        
        # Check for mechanical failures
        if inc_type in SIMULATABLE_TYPES:
            return "MECHANICAL_TEST"
        
        # Check for airspace keywords (pilot sighting, not drone failure)
        airspace_keywords = ["reported a", "sighted", "observed", "appeared to be", "uas in", "drone near"]
        if any(kw in desc_lower for kw in airspace_keywords):
            # This is a sighting, not a drone failure
            if altitude_m and altitude_m > 400:  # Above 400m = likely high altitude
                return "AIRSPACE_SIGHTING"
            else:
                return "GEOFENCE_TEST"
        
        # Check for failure keywords
        for keyword in FAILURE_KEYWORDS:
            if keyword in desc_lower:
                return "MECHANICAL_TEST"
        
        return "MECHANICAL_TEST"  # Default
    
    def _is_simulatable(self, incident: Dict) -> bool:
        """Check if incident describes a real drone failure."""
        # Skip pure sightings at jet altitudes (not simulatable in PX4)
        if incident.get("simulation_mode") == "AIRSPACE_SIGHTING":
            altitude = incident.get("extracted_altitude_m", 0)
            if altitude and altitude > HIGH_ALTITUDE_THRESHOLD_M:
                # Log but don't include - these are jet encounters
                logger.debug(f"Skipping high-altitude sighting: {incident.get('incident_id', 'unknown')} at {altitude:.0f}m")
                return True  # Still include, but marked appropriately
        
        # Check explicit type tag
        inc_type = incident.get("incident_type", "unknown").lower()
        if inc_type in SIMULATABLE_TYPES:
            return True
        
        # Fallback: Check for failure keywords in description
        desc = incident.get("description", "").lower()
        summary = incident.get("summary", "").lower()
        text = desc + " " + summary
        
        for keyword in FAILURE_KEYWORDS:
            if keyword in text:
                return True
        
        return True  # Include all for now, simulation_mode determines handling
    
    def get_all(self) -> List[Dict]:
        """Get all simulatable incidents."""
        if not self._simulatable:
            self.load()
        return self._simulatable
    
    def get_by_index(self, index: int) -> Dict:
        """Get a specific simulatable incident by index."""
        if not self._simulatable:
            self.load()
        if 0 <= index < len(self._simulatable):
            return self._simulatable[index]
        raise IndexError(f"Index {index} out of range (0-{len(self._simulatable)-1})")
    
    def get_by_type(self, incident_type: str) -> List[Dict]:
        """Get all incidents of a specific type."""
        if not self._simulatable:
            self.load()
        return [i for i in self._simulatable if i.get("incident_type") == incident_type]
    
    def get_by_simulation_mode(self, mode: str) -> List[Dict]:
        """Get incidents by simulation mode (MECHANICAL_TEST, GEOFENCE_TEST, AIRSPACE_SIGHTING)."""
        if not self._simulatable:
            self.load()
        return [i for i in self._simulatable if i.get("simulation_mode") == mode]
    
    def get_stats(self) -> Dict[str, int]:
        """Get breakdown of simulatable incidents by type."""
        if not self._simulatable:
            self.load()
        
        stats = {}
        for inc in self._simulatable:
            inc_type = inc.get("incident_type", "unknown")
            stats[inc_type] = stats.get(inc_type, 0) + 1
        
        return dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))
    
    def get_simulation_mode_stats(self) -> Dict[str, int]:
        """Get breakdown by simulation mode."""
        if not self._simulatable:
            self.load()
        
        stats = {}
        for inc in self._simulatable:
            mode = inc.get("simulation_mode", "unknown")
            stats[mode] = stats.get(mode, 0) + 1
        
        return dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))


# Singleton
_filter: Optional[IncidentFilter] = None

def get_incident_filter() -> IncidentFilter:
    """Get singleton incident filter."""
    global _filter
    if _filter is None:
        _filter = IncidentFilter()
    return _filter

