"""
Sighting Filter for Simulatable UAS Events
===========================================
Author: AeroGuardian Member
Date: 2026-01-18
Updated: 2026-01-31

Filters FAA UAS sighting reports to identify those with operational anomalies
that can be meaningfully simulated in PX4 SITL for pre-flight safety analysis.

Note: FAA UAS Sighting Reports document abnormal operations and near-miss encounters,
NOT accidents. The terminology "sighting" reflects the observational nature of this data.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger("AeroGuardian.SightingFilter")

# Sighting types that represent actual drone failures (simulatable)
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


class SightingFilter:
    """
    Filter FAA UAS sighting reports for simulatable operational anomalies.
    
    FAA UAS Sighting Reports contain observational data about abnormal UAS operations
    and near-miss encounters. This filter identifies reports that describe operational
    anomalies suitable for PX4 SITL simulation.
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize the sighting filter.
        
        Args:
            data_path: Path to the FAA sightings JSON file. 
                       Defaults to data/processed/faa_reports/faa_simulatable.json
        """
        # Use original full dataset, but sort drone-testable to front
        self.data_path = data_path or Path("data/processed/faa_reports/faa_simulatable.json")
        self._sightings: List[Dict] = []
        self._simulatable: List[Dict] = []
    
    def load(self) -> int:
        """
        Load and filter sighting reports.
        
        Returns:
            Count of simulatable sightings.
            
        Raises:
            FileNotFoundError: If the data file doesn't exist.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"FAA data not found: {self.data_path}")
            
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._sightings = data.get("incidents", [])  # JSON key remains for compatibility
        self._simulatable = []
        
        # Separate drone-testable from pure sightings
        drone_testable = []
        pure_sightings = []
        
        for sighting in self._sightings:
            # Enrich sighting with altitude and simulation mode
            sighting = self._enrich_sighting(sighting)
            if self._is_simulatable(sighting):
                # Sort by simulation_mode: MECHANICAL_TEST first
                if sighting.get("simulation_mode") == "MECHANICAL_TEST":
                    drone_testable.append(sighting)
                else:
                    pure_sightings.append(sighting)
        
        # Drone-testable sightings at front (index 0, 1, 2...)
        self._simulatable = drone_testable + pure_sightings
        
        logger.info(f"Loaded {len(self._sightings)} total sightings")
        logger.info(f"Filtered to {len(self._simulatable)} simulatable sightings")
        logger.info(f"  → Drone-testable (front): {len(drone_testable)} (index 0-{len(drone_testable)-1})")
        logger.info(f"  → Other sightings: {len(pure_sightings)}")
        
        return len(self._simulatable)
    
    def _enrich_sighting(self, sighting: Dict) -> Dict:
        """
        Add extracted altitude and simulation mode to sighting data.
        
        Args:
            sighting: Raw sighting data from FAA JSON.
            
        Returns:
            Enriched sighting with additional fields.
        """
        desc = sighting.get("description", "") + " " + sighting.get("summary", "")
        
        # Extract altitude
        altitude_m, source = parse_altitude_from_text(desc)
        sighting["extracted_altitude_m"] = altitude_m
        sighting["altitude_source"] = source
        
        # Determine simulation mode
        sighting["simulation_mode"] = self._determine_simulation_mode(sighting, altitude_m, desc)
        
        # Determine simulatable altitude (capped for drone physics)
        if altitude_m and altitude_m > MAX_DRONE_ALTITUDE_M:
            sighting["simulatable_altitude_m"] = MAX_DRONE_ALTITUDE_M
            sighting["altitude_capped"] = True
        elif altitude_m:
            sighting["simulatable_altitude_m"] = altitude_m
            sighting["altitude_capped"] = False
        else:
            sighting["simulatable_altitude_m"] = 50.0  # Default
            sighting["altitude_capped"] = False
        
        return sighting
    
    def _determine_simulation_mode(
        self, 
        sighting: Dict, 
        altitude_m: Optional[float], 
        desc: str
    ) -> str:
        """
        Determine simulation mode based on sighting characteristics.
        
        Args:
            sighting: Sighting data dictionary.
            altitude_m: Extracted altitude in meters (or None).
            desc: Combined description text.
        
        Returns:
            Simulation mode string:
            - MECHANICAL_TEST: Actual drone failure (motor, GPS, battery)
            - GEOFENCE_TEST: Airspace violation (healthy drone, wrong location)
            - AIRSPACE_SIGHTING: Jet/aircraft reported seeing drone
        """
        sighting_type = sighting.get("incident_type", "unknown").lower()  # JSON key
        desc_lower = desc.lower()
        
        # Check for high-altitude encounter (likely jet seeing drone)
        if altitude_m and altitude_m > HIGH_ALTITUDE_THRESHOLD_M:
            # FL50+ is almost certainly a manned aircraft report
            return "AIRSPACE_SIGHTING"
        
        # Check for mechanical failures
        if sighting_type in SIMULATABLE_TYPES:
            return "MECHANICAL_TEST"
        
        # Check for airspace keywords (pilot sighting, not drone failure)
        airspace_keywords = [
            "reported a", "sighted", "observed", 
            "appeared to be", "uas in", "drone near"
        ]
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
    
    def _is_simulatable(self, sighting: Dict) -> bool:
        """
        Check if sighting describes an event that can be simulated.
        
        Args:
            sighting: Enriched sighting data.
            
        Returns:
            True if the sighting can be simulated in PX4 SITL.
        """
        # Skip pure sightings at jet altitudes (not simulatable in PX4)
        if sighting.get("simulation_mode") == "AIRSPACE_SIGHTING":
            altitude = sighting.get("extracted_altitude_m", 0)
            if altitude and altitude > HIGH_ALTITUDE_THRESHOLD_M:
                # Log but don't include - these are jet encounters
                logger.debug(
                    f"Skipping high-altitude sighting: "
                    f"{sighting.get('incident_id', 'unknown')} at {altitude:.0f}m"
                )
                return True  # Still include, but marked appropriately
        
        # Check explicit type tag
        sighting_type = sighting.get("incident_type", "unknown").lower()
        if sighting_type in SIMULATABLE_TYPES:
            return True
        
        # Fallback: Check for failure keywords in description
        desc = sighting.get("description", "").lower()
        summary = sighting.get("summary", "").lower()
        text = desc + " " + summary
        
        for keyword in FAILURE_KEYWORDS:
            if keyword in text:
                return True
        
        return True  # Include all for now, simulation_mode determines handling
    
    def get_all(self) -> List[Dict]:
        """Get all simulatable sightings."""
        if not self._simulatable:
            self.load()
        return self._simulatable
    
    def get_by_index(self, index: int) -> Dict:
        """
        Get a specific simulatable sighting by index.
        
        Args:
            index: Zero-based index into the simulatable sightings list.
            
        Returns:
            Sighting data dictionary.
            
        Raises:
            IndexError: If index is out of range.
        """
        if not self._simulatable:
            self.load()
        if 0 <= index < len(self._simulatable):
            return self._simulatable[index]
        raise IndexError(f"Index {index} out of range (0-{len(self._simulatable)-1})")
    
    def get_by_type(self, sighting_type: str) -> List[Dict]:
        """
        Get all sightings of a specific type.
        
        Args:
            sighting_type: The type to filter by (e.g., 'motor_failure').
            
        Returns:
            List of matching sighting dictionaries.
        """
        if not self._simulatable:
            self.load()
        return [s for s in self._simulatable if s.get("incident_type") == sighting_type]
    
    def get_by_simulation_mode(self, mode: str) -> List[Dict]:
        """
        Get sightings by simulation mode.
        
        Args:
            mode: One of MECHANICAL_TEST, GEOFENCE_TEST, AIRSPACE_SIGHTING.
            
        Returns:
            List of matching sighting dictionaries.
        """
        if not self._simulatable:
            self.load()
        return [s for s in self._simulatable if s.get("simulation_mode") == mode]
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get breakdown of simulatable sightings by type.
        
        Returns:
            Dictionary mapping sighting types to counts, sorted descending.
        """
        if not self._simulatable:
            self.load()
        
        stats = {}
        for sighting in self._simulatable:
            sighting_type = sighting.get("incident_type", "unknown")
            stats[sighting_type] = stats.get(sighting_type, 0) + 1
        
        return dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))
    
    def get_simulation_mode_stats(self) -> Dict[str, int]:
        """
        Get breakdown by simulation mode.
        
        Returns:
            Dictionary mapping simulation modes to counts, sorted descending.
        """
        if not self._simulatable:
            self.load()
        
        stats = {}
        for sighting in self._simulatable:
            mode = sighting.get("simulation_mode", "unknown")
            stats[mode] = stats.get(mode, 0) + 1
        
        return dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))
    
    def count(self) -> int:
        """Get total count of simulatable sightings."""
        if not self._simulatable:
            self.load()
        return len(self._simulatable)


# Singleton instance
_filter: Optional[SightingFilter] = None


def get_sighting_filter() -> SightingFilter:
    """
    Get singleton sighting filter instance.
    
    Returns:
        The global SightingFilter instance.
    """
    global _filter
    if _filter is None:
        _filter = SightingFilter()
    return _filter


# Backward compatibility aliases (deprecated)
IncidentFilter = SightingFilter
get_incident_filter = get_sighting_filter
