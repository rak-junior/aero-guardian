"""
FAA Module for AeroGuardian
============================
Author: AeroGuardian Member
Date: 2026-01-30
Updated: 2026-01-31

FAA UAS Sighting Report filtering and loading for pre-flight safety analysis.

Note: FAA UAS Sighting Reports document abnormal operations and near-miss encounters
in the National Airspace System. This module provides filtering capabilities to
identify sightings suitable for PX4 SITL simulation.
"""

from pathlib import Path

MODULE_ROOT = Path(__file__).parent

# Import sighting filter (main interface)
from .sighting_filter import (
    SightingFilter,
    get_sighting_filter,

)

# Backward compatibility aliases (deprecated - will be removed in future versions)
IncidentFilter = SightingFilter
get_incident_filter = get_sighting_filter

__all__ = [
    # Primary exports
    "SightingFilter",
    "get_sighting_filter",

    # Deprecated aliases
    "IncidentFilter",
    "get_incident_filter",
]
