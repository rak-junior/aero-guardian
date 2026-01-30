"""
LLM Client for AeroGuardian
===========================
Author: AeroGuardian Member
Date: 2026-01-30

Main entry point for the 2-LLM pipeline:
  - LLM #1: ScenarioGenerator - FAA incident → PX4 config
  - LLM #2: ReportGenerator - Telemetry → Safety report

USAGE:
    from src.llm import LLMClient
    
    client = LLMClient()
    
    # Generate PX4 config from FAA incident
    config = client.generate_scenario_config(faa_report, incident_id)
    
    # Generate safety report from telemetry
    report = client.generate_safety_report(incident_info, telemetry)
"""

import os
import logging
from typing import Dict, Any, Optional

from .scenario_generator import (
    ScenarioGenerator,
    ScenarioGenerationError,
    get_scenario_generator,
    MAX_MISSION_DURATION_SEC,
)
from .report_generator import (
    ReportGenerator,
    ReportGenerationError,
    get_report_generator,
)

logger = logging.getLogger("AeroGuardian.LLMClient")


class LLMClient:
    """
    Main LLM client for AeroGuardian 2-LLM Pre-Flight Safety Pipeline.
    
    This is the primary entry point for all LLM operations.
    
    Pipeline:
        LLM #1: FAA incident → PX4 simulation config
        LLM #2: Telemetry → Pre-flight safety report
    
    USAGE:
        client = LLMClient()
        
        # Step 1: Generate simulation config
        config = client.generate_scenario_config(faa_report, incident_id)
        
        # Step 2: Run simulation (external)
        # ...
        
        # Step 3: Generate safety report
        report = client.generate_safety_report(
            incident_description=faa_report,
            incident_id=incident_id,
            incident_location="City, State",
            fault_type="motor_failure",
            expected_outcome="crash",
            telemetry_summary="duration: 120s, max_alt: 50m"
        )
    """
    
    def __init__(self, model: str = None):
        """
        Initialize LLM client.
        
        Args:
            model: OpenAI model name (default: from OPENAI_MODEL env var)
        """
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._scenario_generator: Optional[ScenarioGenerator] = None
        self._report_generator: Optional[ReportGenerator] = None
        self._call_count = 0
        
        logger.info(f"LLMClient initialized: {self.model}")
    
    @property
    def scenario_generator(self) -> ScenarioGenerator:
        """Get scenario generator (lazy initialization)."""
        if self._scenario_generator is None:
            self._scenario_generator = get_scenario_generator()
        return self._scenario_generator
    
    @property
    def report_generator(self) -> ReportGenerator:
        """Get report generator (lazy initialization)."""
        if self._report_generator is None:
            self._report_generator = get_report_generator()
        return self._report_generator
    
    @property
    def is_ready(self) -> bool:
        """Check if client is ready for LLM calls."""
        return bool(os.getenv("OPENAI_API_KEY"))
    
    def generate_scenario_config(
        self,
        faa_report_text: str,
        incident_id: str,
    ) -> Dict[str, Any]:
        """
        LLM #1: Generate PX4 simulation config from FAA incident.
        
        Args:
            faa_report_text: Complete FAA incident report text
            incident_id: FAA incident ID
            
        Returns:
            Dictionary with PX4 configuration parameters
            
        Raises:
            ScenarioGenerationError on failure
        """
        self._call_count += 1
        logger.info(f"🔄 LLM Call #{self._call_count}: generate_scenario_config")
        
        config = self.scenario_generator.generate(faa_report_text, incident_id)
        
        # Convert to dict format expected by pipeline
        return {
            "mission": {
                "start_lat": config.lat,
                "start_lon": config.lon,
                "takeoff_altitude_m": config.altitude_m,
                "flight_mode": "MISSION",
                "duration_sec": MAX_MISSION_DURATION_SEC,
                "cruise_speed_ms": config.speed_ms,
                "cruise_altitude_m": config.altitude_m,
            },
            "fault_injection": {
                "fault_type": config.failure_mode,
                "fault_category": config.failure_category,
                "onset_sec": config.failure_onset_sec,
                "duration_sec": -1,  # Permanent
                "affected_components": [config.failure_component],
                "symptoms": config.symptoms,
            },
            "environment": {
                "wind_speed_ms": config.wind_speed_ms,
                "wind_direction_deg": config.wind_direction_deg,
                "weather": config.weather,
                "environment_type": config.environment,
            },
            "gps": {
                "satellite_count": 8 if "gps" not in config.failure_mode.lower() else 4,
                "noise_m": 1.0 if "gps" not in config.failure_mode.lower() else 5.0,
            },
            "waypoints": config.waypoints,
            "px4_commands": {
                "fault": config.px4_fault_cmd,
            },
            "faa_source": {
                "incident_id": config.faa_incident_id,
                "city": config.city,
                "state": config.state,
                "outcome": config.outcome,
            },
            "reasoning": config.reasoning,
            
            # Metadata
            "parameter_count": 31,
            "generated_by": "FAA_To_PX4_Complete",
        }
    
    def generate_safety_report(
        self,
        incident_description: str,
        incident_id: str,
        incident_location: str,
        fault_type: str,
        expected_outcome: str,
        telemetry_summary: str,
        **kwargs,  # Accept additional parameters for compatibility
    ) -> Dict[str, Any]:
        """
        LLM #2: Generate pre-flight safety report from telemetry.
        
        Args:
            incident_description: Original FAA incident narrative
            incident_id: FAA incident ID
            incident_location: City, State
            fault_type: MOTOR_FAILURE, GPS_LOSS, etc.
            expected_outcome: crash, controlled_landing, flyaway
            telemetry_summary: Telemetry analysis metrics
            
        Returns:
            Dictionary with safety report sections
            
        Raises:
            ReportGenerationError on failure
        """
        self._call_count += 1
        logger.info(f"🔄 LLM Call #{self._call_count}: generate_safety_report")
        
        report = self.report_generator.generate(
            incident_description=incident_description,
            incident_id=incident_id,
            incident_location=incident_location,
            fault_type=fault_type,
            expected_outcome=expected_outcome,
            telemetry_summary=telemetry_summary,
        )
        
        # Convert to dict format expected by pipeline
        return {
            "incident_id": report.incident_id,
            "incident_location": report.incident_location,
            "fault_type": report.fault_type,
            "expected_outcome": report.expected_outcome,
            
            # Section 1
            "safety_level": report.safety_level,
            "primary_hazard": report.primary_hazard,
            "observed_effect": report.observed_effect,
            
            # Section 2
            "design_constraints": report.design_constraints,
            "recommendations": report.recommendations,
            
            # Section 3
            "explanation": report.explanation,
            
            # Verdict
            "verdict": report.verdict,
            
            # Metadata
            "generated_by": "GeneratePreFlightReport",
        }
    
    # Alias for backward compatibility
    def generate_preflight_report(self, *args, **kwargs) -> Dict[str, Any]:
        """Alias for generate_safety_report (backward compatibility)."""
        return self.generate_safety_report(*args, **kwargs)
    
    # Alias for backward compatibility  
    def generate_full_px4_config(
        self,
        incident_description: str,
        incident_location: str,
        incident_type: str,
    ) -> Dict[str, Any]:
        """Alias for generate_scenario_config (backward compatibility)."""
        faa_report = f"Location: {incident_location}\nType: {incident_type}\nDescription: {incident_description}"
        return self.generate_scenario_config(faa_report, f"FAA-{incident_type.upper()}")


# =============================================================================
# Singleton Access
# =============================================================================

_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get LLM client singleton."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


__all__ = [
    "LLMClient",
    "get_llm_client",
    "ScenarioGenerationError",
    "ReportGenerationError",
]
