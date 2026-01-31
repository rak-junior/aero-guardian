"""
LLM Module for AeroGuardian
===========================
Author: AeroGuardian Member
Date: 2026-01-30

Standard structure:
- signatures.py: All DSPy signature definitions
- scenario_generator.py: LLM #1 generator (FAA → PX4)
- report_generator.py: LLM #2 generator (Telemetry → Report)
- client.py: Main entry point

USAGE:
    from src.llm import LLMClient
    
    client = LLMClient()
    config = client.generate_scenario_config(faa_report, incident_id)
    report = client.generate_safety_report(...)
"""

# Main entry point
from .client import (
    LLMClient,
    get_llm_client,
)

# Signatures (for advanced users)
from .signatures import (
    FAA_To_PX4_Complete,
    GeneratePreFlightReport,
    SIGNATURES,
)

# Generators (for advanced users)
from .scenario_generator import (
    ScenarioGenerator,
    ScenarioConfig,
    ScenarioGenerationError,
    get_scenario_generator,
    generate_scenario,
    MAX_MISSION_DURATION_SEC,
)

from .report_generator import (
    ReportGenerator,
    SafetyReport,
    ReportGenerationError,
    get_report_generator,
)

__all__ = [
    # Main entry point (recommended)
    "LLMClient",
    "get_llm_client",
    
    # Signatures
    "FAA_To_PX4_Complete",
    "GeneratePreFlightReport",
    "SIGNATURES",
    
    # Scenario generation
    "ScenarioGenerator",
    "ScenarioConfig",
    "ScenarioGenerationError",
    "get_scenario_generator",
    "generate_scenario",
    "MAX_MISSION_DURATION_SEC",
    
    # Report generation
    "ReportGenerator",
    "SafetyReport",
    "ReportGenerationError",
    "get_report_generator",
]
