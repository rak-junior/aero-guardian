"""
FAA Module for AeroGuardian
============================
Author: AeroGuardian Member
Date: 2026-01-30

Incident filtering and loading for pre-flight safety analysis.
"""

from pathlib import Path

MODULE_ROOT = Path(__file__).parent

# Import incident filter (main interface)
try:
    from .incident_filter import get_incident_filter, IncidentFilter
except ImportError:
    pass

__all__ = ["get_incident_filter", "IncidentFilter"]
