"""
Simulation Module for AeroGuardian
===================================
Author: AeroGuardian Member
Date: 2026-01-30

This module handles PX4 SITL simulation control and failure emulation.
"""

# Failure Emulation module (no external LLM dependencies)
from src.simulation.failure_emulator import (
    FailureEmulator,
    FailureCategory,
    FailurePhase,
    EmulationResult,
    TemporalRandomizer,
)

# Lazy import for LLM-dependent modules to avoid dspy import at module load
def _get_scenario_exports():
    """Lazy loader for scenario generator exports (requires dspy)."""
    from src.llm.scenario_generator import (
        ScenarioGenerator as FAAScenarioGenerator,
        ScenarioConfig as FAAScenarioConfig,
        ScenarioGenerationError as FAAScenarioGenerationError,
        get_scenario_generator as get_faa_generator,
        generate_scenario,
        MAX_MISSION_DURATION_SEC,
    )
    return {
        "FAAScenarioGenerator": FAAScenarioGenerator,
        "FAAScenarioConfig": FAAScenarioConfig,
        "FAAScenarioGenerationError": FAAScenarioGenerationError,
        "get_faa_generator": get_faa_generator,
        "generate_scenario": generate_scenario,
        "MAX_MISSION_DURATION_SEC": MAX_MISSION_DURATION_SEC,
    }

__all__ = [
    # Failure Emulation (always available)
    "FailureEmulator",
    "FailureCategory",
    "FailurePhase",
    "EmulationResult",
    "TemporalRandomizer",
    # Scenario generation (lazy loaded)
    "FAAScenarioGenerator",
    "FAAScenarioConfig",
    "FAAScenarioGenerationError",
    "get_faa_generator",
    "generate_scenario",
    "MAX_MISSION_DURATION_SEC",
]

def __getattr__(name):
    """Lazy loading for scenario generator exports."""
    if name in ["FAAScenarioGenerator", "FAAScenarioConfig", "FAAScenarioGenerationError",
                 "get_faa_generator", "generate_scenario", "MAX_MISSION_DURATION_SEC"]:
        exports = _get_scenario_exports()
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

