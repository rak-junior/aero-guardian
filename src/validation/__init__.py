"""
Validation Module
=================
Author: AeroGuardian Member
Date: 2026-01-18

Provides validation utilities for comparing simulation outcomes with FAA incident descriptions.
"""

from .scenario_validator import (
    ScenarioValidator,
    ValidationResult,
    compute_scenario_match_score,
    get_validator,
)

__all__ = [
    "ScenarioValidator",
    "ValidationResult",
    "compute_scenario_match_score",
    "get_validator",
]
