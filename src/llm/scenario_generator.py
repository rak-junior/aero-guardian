"""
Scenario Generator for AeroGuardian
====================================
Author: AeroGuardian Member
Date: 2026-01-30
Updated: 2026-01-31

Generates PX4 simulation configurations from FAA UAS sighting reports.
Uses FAA_To_PX4_Complete DSPy signature to translate natural language
descriptions of operational anomalies into executable simulation parameters.

USAGE:
    from src.llm import ScenarioGenerator
    
    generator = ScenarioGenerator()
    config = generator.generate(faa_report_text, sighting_id)
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

import dspy

from .signatures import FAA_To_PX4_Complete

logger = logging.getLogger("AeroGuardian.ScenarioGenerator")


# =============================================================================
# Simulation Optimization Constants
# =============================================================================

MAX_UAV_ALTITUDE_M = 120.0  # Legal UAV limit (400 ft)
DEFAULT_TEST_ALTITUDE_M = 50.0
MAX_MISSION_DURATION_SEC = 120  # 2 minutes max
MIN_MISSION_DURATION_SEC = 60
FAULT_ONSET_RATIO = 0.5


def clamp_altitude(altitude_m: float, original_altitude_ft: float = None) -> float:
    """Clamp altitude to UAV-realistic values (max 120m)."""
    if altitude_m is None or altitude_m <= 0:
        logger.info(f"⚠️ Altitude missing/invalid, using default {DEFAULT_TEST_ALTITUDE_M}m")
        return DEFAULT_TEST_ALTITUDE_M
    
    if altitude_m > MAX_UAV_ALTITUDE_M:
        original_ft = original_altitude_ft or (altitude_m * 3.28084)
        logger.info(f"⚠️ Clamping altitude: {original_ft:.0f}ft ({altitude_m:.0f}m) → {MAX_UAV_ALTITUDE_M}m")
        return MAX_UAV_ALTITUDE_M
    
    return altitude_m


def optimize_fault_timing(llm_onset_sec: int, mission_duration_sec: int = MAX_MISSION_DURATION_SEC) -> int:
    """Optimize fault onset timing for DEMO purposes (fault at T+5s)."""
    DEMO_FAULT_ONSET_SEC = 5
    
    if llm_onset_sec != DEMO_FAULT_ONSET_SEC:
        logger.info(f"⚠️ Optimizing fault onset: {llm_onset_sec}s → {DEMO_FAULT_ONSET_SEC}s")
    
    return DEMO_FAULT_ONSET_SEC


# =============================================================================
# Data Classes
# =============================================================================

class ScenarioGenerationError(Exception):
    """Raised when scenario generation fails."""


@dataclass
class ScenarioConfig:
    """Complete PX4 config - all values from LLM."""
    
    faa_incident_id: str
    faa_report_text: str
    
    city: str
    state: str
    lat: float
    lon: float
    
    altitude_m: float
    speed_ms: float
    flight_phase: str
    uas_type: str
    
    failure_mode: str
    failure_category: str
    failure_component: str
    failure_onset_sec: int
    
    symptoms: List[str]
    outcome: str
    
    weather: str
    wind_speed_ms: float
    wind_direction_deg: float
    environment: str
    
    px4_fault_cmd: str
    waypoints: List[Dict]
    
    reasoning: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# =============================================================================
# Scenario Generator
# =============================================================================

class ScenarioGenerator:
    """
    Generate PX4 simulation configurations from FAA UAS sighting reports.
    
    Uses FAA_To_PX4_Complete DSPy signature to translate natural language
    descriptions of operational anomalies into executable simulation configs.
    
    USAGE:
        generator = ScenarioGenerator()
        config = generator.generate(faa_report, sighting_id)
    """
    
    def __init__(self):
        self.is_ready = False
        self._translator = None
        self._configure()
    
    def _configure(self):
        """Configure DSPy with OpenAI."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ScenarioGenerationError("OPENAI_API_KEY not set")
        
        try:
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            
            # Handle reasoning models (gpt-5, o1, etc.)
            is_reasoning_model = any(x in model.lower() for x in ["gpt-5", "o1-", "o3-"])
            
            lm = dspy.LM(
                model=f"openai/{model}",
                api_key=api_key,
                max_tokens=16000 if is_reasoning_model else 4096,
                temperature=1.0 if is_reasoning_model else 0.1,
            )
            
            dspy.configure(lm=lm)
            self._translator = dspy.ChainOfThought(FAA_To_PX4_Complete)
            
            # Load few-shot examples
            try:
                from .dspy_fewshot import get_faa_to_px4_examples
                examples = get_faa_to_px4_examples()
                if examples:
                    self._translator.demos = examples[:3]
                    logger.info(f"Loaded {len(examples)} few-shot examples")
            except ImportError:
                logger.debug("Few-shot examples not available")
            
            self.is_ready = True
            logger.info(f"ScenarioGenerator ready: {model}")
            
        except Exception as e:
            raise ScenarioGenerationError(f"Failed to initialize LLM: {e}")
    
    def _log_dspy_prompt(self, signature_name: str, sighting_id: str):
        """Log the formatted DSPy prompt."""
        try:
            lm = dspy.settings.lm
            if not lm or not hasattr(lm, 'history') or not lm.history:
                return
            
            last_entry = lm.history[-1] if lm.history else None
            if not last_entry or not isinstance(last_entry, dict):
                return
            
            prompt_text = ""
            if 'messages' in last_entry:
                for msg in last_entry['messages']:
                    role = msg.get('role', 'unknown').upper()
                    content = msg.get('content', '')
                    prompt_text += f"[{role}]\n{content}\n\n"
            elif 'prompt' in last_entry:
                prompt_text = last_entry['prompt']
            
            logger.info(f"{'='*80}\nDSPy PROMPT - {signature_name} ({sighting_id})\n{'='*80}\n{prompt_text}\n{'='*80}")
            
        except Exception as e:
            logger.debug(f"Prompt logging failed: {e}")
    
    def generate(self, faa_report_text: str, sighting_id: str) -> ScenarioConfig:
        """
        Generate PX4 config from FAA sighting report.
        
        Args:
            faa_report_text: Complete FAA UAS sighting report text
            sighting_id: FAA sighting ID
            
        Returns:
            ScenarioConfig with all simulation parameters
            
        Raises:
            ScenarioGenerationError on failure
        """
        if not self.is_ready:
            raise ScenarioGenerationError("Generator not initialized")
        
        if not faa_report_text or len(faa_report_text.strip()) < 20:
            raise ScenarioGenerationError("FAA report text is empty or too short")
        
        logger.info(f"Generating config: {sighting_id}")
        
        try:
            result = self._translator(
                faa_report_text=faa_report_text,
                faa_incident_id=sighting_id,  # DSPy signature uses incident_id
            )
            
            self._log_dspy_prompt("FAA_To_PX4_Complete", sighting_id)
            
            # Parse waypoints JSON
            try:
                waypoints = json.loads(result.waypoints_json)
                if not isinstance(waypoints, list) or len(waypoints) == 0:
                    raise ValueError("Waypoints must be non-empty list")
            except (json.JSONDecodeError, ValueError) as e:
                raise ScenarioGenerationError(f"Invalid waypoints from LLM: {e}")
            
            # Parse symptoms
            symptoms = [s.strip() for s in str(result.symptoms).split(",") if s.strip()]
            if not symptoms:
                raise ScenarioGenerationError("No symptoms extracted from report")
            
            # Apply simulation optimizations
            raw_altitude_m = float(result.altitude_m)
            raw_altitude_ft = float(result.altitude_ft) if hasattr(result, 'altitude_ft') else raw_altitude_m * 3.28084
            clamped_altitude_m = clamp_altitude(raw_altitude_m, raw_altitude_ft)
            
            for wp in waypoints:
                if 'alt' in wp and wp['alt'] > MAX_UAV_ALTITUDE_M:
                    wp['alt'] = clamped_altitude_m
            
            raw_onset_sec = int(result.failure_onset_sec)
            optimized_onset_sec = optimize_fault_timing(raw_onset_sec, MAX_MISSION_DURATION_SEC)
            
            logger.info(f"📊 Optimizations: Alt {raw_altitude_m:.0f}m→{clamped_altitude_m:.0f}m, Fault {raw_onset_sec}s→{optimized_onset_sec}s")
            
            config = ScenarioConfig(
                faa_incident_id=sighting_id,  # Legacy field name for compatibility
                faa_report_text=faa_report_text[:500],
                
                city=str(result.city),
                state=str(result.state),
                lat=float(result.lat),
                lon=float(result.lon),
                
                altitude_m=clamped_altitude_m,
                speed_ms=float(result.speed_ms),
                flight_phase=str(result.flight_phase),
                uas_type=str(result.uas_type),
                
                failure_mode=str(result.failure_mode),
                failure_category=str(result.failure_category),
                failure_component=str(result.failure_component),
                failure_onset_sec=optimized_onset_sec,
                
                symptoms=symptoms,
                outcome=str(result.outcome),
                
                weather=str(result.weather),
                wind_speed_ms=float(result.wind_speed_ms),
                wind_direction_deg=float(result.wind_direction_deg),
                environment=str(result.environment),
                
                px4_fault_cmd=str(result.px4_fault_cmd),
                waypoints=waypoints,
                
                reasoning=str(result.reasoning),
            )
            
            logger.info(f"✅ Generated: {config.failure_mode} at {config.city}, {config.state}")
            return config
            
        except ScenarioGenerationError:
            raise
        except Exception as e:
            raise ScenarioGenerationError(f"LLM generation failed: {e}")
    
    def generate_from_dict(self, sighting: Dict) -> ScenarioConfig:
        """
        Generate from FAA sighting dictionary.
        
        Args:
            sighting: Dictionary containing FAA sighting data.
            
        Returns:
            ScenarioConfig with all simulation parameters.
        """
        parts = []
        for key in ["date", "city", "state", "summary", "description", "altitude", "uas_type", "incident_type"]:
            if sighting.get(key):
                parts.append(f"{key.title()}: {sighting[key]}")
        
        if not parts:
            raise ScenarioGenerationError("Sighting dict has no usable fields")
        
        return self.generate(
            faa_report_text="\n".join(parts),
            sighting_id=sighting.get("incident_id", "UNKNOWN"),  # Legacy key
        )


# =============================================================================
# Singleton Access
# =============================================================================

_generator: Optional[ScenarioGenerator] = None


def get_scenario_generator() -> ScenarioGenerator:
    """Get scenario generator singleton. Raises if not available."""
    global _generator
    if _generator is None:
        _generator = ScenarioGenerator()
    return _generator


def generate_scenario(faa_report: str, sighting_id: str) -> ScenarioConfig:
    """Convenience function to generate scenario config from FAA sighting report."""
    return get_scenario_generator().generate(faa_report, sighting_id)


__all__ = [
    "ScenarioGenerator",
    "ScenarioConfig",
    "ScenarioGenerationError",
    "get_scenario_generator",
    "generate_scenario",
    "MAX_MISSION_DURATION_SEC",
]
