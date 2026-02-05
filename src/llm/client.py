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
    MAX_MISSION_DURATION_SEC,
)
from .report_generator import (
    ReportGenerator,
    ReportGenerationError,
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
        config = client.generate_scenario_config(faa_report, report_id)
        
        # Or using the raw text method
        config = client.generate_full_px4_config(
            incident_description=faa_report["description"],
            incident_location="Phoenix, AZ",
            incident_type="uas_incident",
            report_id=report_id,
        )
        
        # Step 2: Run simulation (external)
        # ...
        
        # Step 3: Generate safety report
        report = client.generate_safety_report(
            incident_description=faa_report,
            incident_id=report_id, # This should be report_id, but the diff doesn't change it here.
            incident_location="City, State",
            fault_type="motor_failure",
            expected_outcome="crash",
            telemetry_summary="duration: 120s, max_alt: 50m"
        )
    """
    
    def __init__(self, model: str = None, output_dir: str = None):
        """
        Initialize LLM client.
        
        Args:
            model: OpenAI model name (default: from OPENAI_MODEL env var)
            output_dir: Output directory for LLM logging (optional)
        """
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self._output_dir = output_dir
        self._scenario_generator: Optional[ScenarioGenerator] = None
        self._report_generator: Optional[ReportGenerator] = None
        self._call_count = 0
        
        logger.info(f"LLMClient initialized: {self.model}")
    
    def set_output_dir(self, output_dir: str) -> None:
        """
        Set output directory for LLM logging.
        
        Call this before making LLM calls to enable detailed logging.
        """
        self._output_dir = output_dir
        # Reset generators to pick up new output_dir
        self._scenario_generator = None
        self._report_generator = None
        logger.info(f"LLM logging output dir set: {output_dir}")
    
    @property
    def scenario_generator(self) -> ScenarioGenerator:
        """Get scenario generator (lazy initialization)."""
        if self._scenario_generator is None:
            self._scenario_generator = ScenarioGenerator(output_dir=self._output_dir)
        return self._scenario_generator
    
    @property
    def report_generator(self) -> ReportGenerator:
        """Get report generator (lazy initialization)."""
        if self._report_generator is None:
            self._report_generator = ReportGenerator(output_dir=self._output_dir)
        return self._report_generator
    
    @property
    def is_ready(self) -> bool:
        """Check if client is ready for LLM calls."""
        return bool(os.getenv("OPENAI_API_KEY"))
    
    def generate_scenario_config(self, faa_report: Any, report_id: str = "UNKNOWN") -> dict:
        """
        Generate PX4 configuration from an FAA report object.
        
        Args:
            faa_report: Dictionary containing report data OR raw string description
            report_id: FAA report ID
            
        Returns:
            Dictionary with PX4 mission configuration
        """
        # Extract text description
        if isinstance(faa_report, dict):
            report_text = faa_report.get("description", faa_report.get("summary", ""))
            incident_type = faa_report.get("incident_type", "unknown")
            location = f"{faa_report.get('city', 'Unknown')}, {faa_report.get('state', '')}"
        else:
            report_text = str(faa_report)
            incident_type = "unknown"
            location = "Unknown Location"
        
        return self.generate_full_px4_config(
            incident_description=report_text,
            incident_location=location,
            incident_type=incident_type,
            report_id=report_id,
        )
    
    def generate_safety_report(
        self,
        incident_description: str,
        report_id: str,
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
            report_id=report_id,
            incident_location=incident_location,
            fault_type=fault_type,
            expected_outcome=expected_outcome,
            telemetry_summary=telemetry_summary,
        )
        
        # Convert to dict format expected by pipeline
        return {
            "incident_id": report.report_id,
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
    
    def generate_full_px4_config(
        self,
        incident_description: str,
        incident_location: str,
        incident_type: str,
        report_id: str = None,
    ) -> Dict[str, Any]:
        """
        Generate full PX4 configuration from raw text description.
        
        Args:
            incident_description: The raw text description of the event
            incident_location: Contextual location string (City, State)
            incident_type: Only used for logging/filtering
            report_id: FAA report ID
            
        Returns:
            Dictionary with full PX4 configuration including:
            - mission (waypoints, altitude, speed)
            - fault_injection (type, timing, severity)
            - environment (wind, weather)
            - px4_commands (shell commands)
        """
        self._call_count += 1
        logger.info(f"🔄 LLM Call #{self._call_count}: generate_full_px4_config")

        if incident_type and incident_type.lower() not in ["unknown", "other", "uas incident", "none"]:
            faa_report_text = f"Location: {incident_location}\nType: {incident_type}\nDescription: {incident_description}"
        else:
            faa_report_text = f"Location: {incident_location}\nDescription: {incident_description}"
            
        final_id = report_id or f"FAA-{str(incident_type).upper()}"
        
        # 1. Generate core scenario (Fault inference & Waypoints)
        # Call the generator DIRECTLY to avoid recursion
        config = self.scenario_generator.generate(faa_report_text, final_id)
        
        # 2. Convert to dict format expected by pipeline
        # (Restoring logic that was accidentally removed)
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
                "incident_id": final_id, # Keep strict traceability
                "city": config.city,
                "state": config.state,
                "outcome": config.outcome,
            },
            "reasoning": config.reasoning,
            
            # P0: Uncertainty & fault alignment tracking
            "uncertainty_score": config.uncertainty_score,
            "fault_injection_supported": config.fault_injection_supported,
            
            # P0: Narrative facts vs LLM inferences
            "narrative_facts": config.narrative_facts,
            "inferred_parameters": config.inferred_parameters,
            
            # P1: Proxy simulation tags
            "proxy_modeling": config.proxy_modeling,
            
            # P2: Evidence traceability
            "evidence_map": config.evidence_map,
            "reconstruction_level": config.reconstruction_level,
            
            # Metadata
            "parameter_count": 31,
            "generated_by": "FAA_To_PX4_Complete",
            "speed_m_s": config.speed_ms,
        }


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
