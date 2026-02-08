"""
Sighting Filter (Data Loader)
=============================
Author: AeroGuardian Member
Date: 2026-02-04

Simplified data loader for FAA UAS sighting reports.
Loads raw reports without heuristic classification or filtering,
relying on downstream LLM processing for fault determination.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger("AeroGuardian.SightingFilter")


class SightingFilter:
    """
    Load FAA UAS sighting reports for processing.
    
    Acts as a pure data loader, passing raw report descriptions to the pipeline.
    Removes all pre-filtering and hazard classification logic ("Trust LLM" approach).
    """
    
    def __init__(self, data_path: Optional[Path] = None, data_source: str = "sightings"):
        """
        Initialize the loader.
        
        Args:
            data_path: Path to the FAA sightings JSON file.
            data_source: 'sightings' (8000 high-risk) or 'failures' (31 confirmed crashes)
        """
        if data_path:
            self.data_path = data_path
        elif data_source == "failures":
            self.data_path = Path("data/processed/faa_reports/faa_actual_failures.json")
        else:
            self.data_path = Path("data/processed/faa_reports/faa_high_risk_sightings.json")
        self._sightings: List[Dict] = []
    
    def load(self) -> int:
        """
        Load sighting reports.
        
        Returns:
            Count of loaded sightings.
            
        Raises:
            FileNotFoundError: If the data file doesn't exist.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"FAA data not found: {self.data_path}")
            
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        raw_incidents = data.get("incidents", [])
        self._sightings = []
        
        for inc in raw_incidents:
            # Normalize core fields
            report_id = inc.get("report_id", inc.get("incident_id", "UNKNOWN"))
            
            # Combine description and summary for full context
            desc_parts = []
            if inc.get("description"):
                desc_parts.append(inc.get("description"))
            if inc.get("summary") and inc.get("summary") != inc.get("description"):
                desc_parts.append(inc.get("summary"))
            full_description = " ".join(desc_parts)
            
            # Create simplified report object
            report = {
                "report_id": report_id,
                "description": full_description,
                "date": inc.get("date", ""),
                "city": inc.get("city", ""),
                "state": inc.get("state", ""),
                # Preserve pre-classified fault data if available
                "fault_type": inc.get("fault_type", ""),
                "hazard_category": inc.get("hazard_category", ""),
            }
            self._sightings.append(report)
            
        logger.info(f"Loaded {len(self._sightings)} sightings (Raw Loader Mode)")
        return len(self._sightings)
    
    def get_all(self) -> List[Dict]:
        """Get all sightings."""
        if not self._sightings:
            self.load()
        return self._sightings
    
    def get_by_index(self, index: int) -> Dict:
        """Get a specific sighting by index."""
        if not self._sightings:
            self.load()
        if 0 <= index < len(self._sightings):
            return self._sightings[index]
        raise IndexError(f"Index {index} out of range (0-{len(self._sightings)-1})")
    
    def count(self) -> int:
        """Get total count."""
        if not self._sightings:
            self.load()
        return len(self._sightings)


# Singleton instance
_filter: Optional[SightingFilter] = None


def get_sighting_filter(data_source: str = "sightings") -> SightingFilter:
    """Get singleton sighting filter instance."""
    global _filter
    if _filter is None:
        _filter = SightingFilter(data_source=data_source)
    return _filter


# Backward compatibility aliases
IncidentFilter = SightingFilter
get_incident_filter = get_sighting_filter
