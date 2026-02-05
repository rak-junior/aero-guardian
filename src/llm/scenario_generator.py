"""
Scenario Generator for AeroGuardian
====================================
Author: AeroGuardian Team (Tiny Coders)
Date: 2026-01-30
Updated: 2026-02-04

Generates PX4 simulation configurations from FAA UAS sighting reports.
Uses FAA_To_PX4_Complete DSPy signature to translate natural language
descriptions of operational anomalies into executable simulation parameters.

OUTPUT: 31-parameter configuration including:
- Fault type, category, and PX4 shell command
- Waypoints with GPS coordinates
- Environmental conditions (wind, weather)
- Mission parameters (altitude, speed, duration)

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
from .llm_logger import LLMInteractionLogger, get_dspy_history, clear_dspy_history

logger = logging.getLogger("AeroGuardian.ScenarioGenerator")


# =============================================================================
# Simulation Optimization Constants
# =============================================================================

MAX_UAV_ALTITUDE_M = 120.0  # Legal UAV limit (400 ft)
DEFAULT_TEST_ALTITUDE_M = 50.0
MAX_MISSION_DURATION_SEC = 120  # 2 minutes max
MIN_MISSION_DURATION_SEC = 60
FAULT_ONSET_RATIO = 0.5

# US Continental bounds for validation
US_LAT_MIN, US_LAT_MAX = 24.0, 50.0
US_LON_MIN, US_LON_MAX = -125.0, -66.0
MAX_GEOCODE_DISTANCE_KM = 100.0  # Max acceptable drift from city center


def validate_geocoding(
    llm_lat: float, 
    llm_lon: float, 
    city: str, 
    state: str
) -> tuple[float, float, bool]:
    """
    Validate LLM-generated lat/lon against expected city location.
    
    If LLM coordinates are invalid or too far from city, falls back to
    geocoder lookup.
    
    Args:
        llm_lat: LLM-generated latitude
        llm_lon: LLM-generated longitude
        city: Expected city name
        state: Expected state name
        
    Returns:
        Tuple of (validated_lat, validated_lon, was_corrected)
    """
    import math
    
    # Check basic US bounds
    if not (US_LAT_MIN <= llm_lat <= US_LAT_MAX and US_LON_MIN <= llm_lon <= US_LON_MAX):
        logger.warning(f"⚠️ LLM lat/lon ({llm_lat:.4f}, {llm_lon:.4f}) outside US bounds")
        return _fallback_geocode(city, state, "outside_bounds")
    
    # Try to verify against actual geocoded location
    try:
        from src.core.geocoder import geocode
        actual_lat, actual_lon = geocode(city, state)
        
        # Haversine distance calculation
        R = 6371  # Earth's radius in km
        lat1, lat2 = math.radians(llm_lat), math.radians(actual_lat)
        dlat = math.radians(actual_lat - llm_lat)
        dlon = math.radians(actual_lon - llm_lon)
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance_km = R * c
        
        if distance_km > MAX_GEOCODE_DISTANCE_KM:
            logger.warning(
                f"⚠️ LLM geocoding drift: {distance_km:.1f}km from {city}, {state} "
                f"(LLM: {llm_lat:.4f}, {llm_lon:.4f} vs Actual: {actual_lat:.4f}, {actual_lon:.4f})"
            )
            return actual_lat, actual_lon, True
        
        # LLM coordinates are acceptable
        logger.debug(f"✓ LLM geocoding validated: {distance_km:.1f}km from city center")
        return llm_lat, llm_lon, False
        
    except Exception as e:
        logger.debug(f"Geocoding validation skipped (no network?): {e}")
        # Can't verify - accept LLM coordinates if within US bounds
        return llm_lat, llm_lon, False


def _fallback_geocode(city: str, state: str, reason: str) -> tuple[float, float, bool]:
    """Fallback to geocoder when LLM coordinates are invalid."""
    try:
        from src.core.geocoder import geocode
        lat, lon = geocode(city, state)
        logger.info(f"🌍 Geocoding fallback ({reason}): {city}, {state} → ({lat:.4f}, {lon:.4f})")
        return lat, lon, True
    except Exception as e:
        # Ultimate fallback: PX4 default location (Minneapolis)
        logger.warning(f"⚠️ Geocoding failed, using PX4 default: {e}")
        return 44.9778, -93.2650, True


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
        logger.info(f"Optimizing fault onset: {llm_onset_sec}s → {DEMO_FAULT_ONSET_SEC}s")
    
    return DEMO_FAULT_ONSET_SEC


# =============================================================================
# Data Classes
# =============================================================================

class ScenarioGenerationError(Exception):
    """Raised when scenario generation fails."""


@dataclass
class ScenarioConfig:
    """Complete PX4 config - all values from LLM."""
    
    faa_report_id: str
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
    
    # Uncertainty & evidence tracking
    uncertainty_score: float = 0.5  # 0=certain, 1=highly uncertain
    fault_injection_supported: bool = True  # False if px4_fault_cmd="none"
    
    # Separate narrative facts vs LLM inferences
    narrative_facts: Optional[Dict[str, Any]] = None  # Facts extracted from FAA report
    inferred_parameters: Optional[Dict[str, Any]] = None  # LLM-inferred values
    
    # Proxy simulation tags
    proxy_modeling: Optional[Dict[str, Any]] = None  # Platform substitution info
    
    # Evidence traceability
    evidence_map: Optional[Dict[str, str]] = None  # Parameter → source mapping
    reconstruction_level: str = "proxy_simulation"  # proxy_simulation | partial_match | behavioral_class
    
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
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize scenario generator with optional output logging."""
        self.is_ready = False
        self._translator = None
        self._output_dir = output_dir
        self._llm_logger: Optional[LLMInteractionLogger] = None
        self._configure()
    
    def _configure(self):
        """Configure DSPy with OpenAI."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ScenarioGenerationError("OPENAI_API_KEY not set")
        
        try:
            model = os.getenv("OPENAI_MODEL", "gpt-4o")
            
            # Handle reasoning models (gpt-5, o1, etc.)
            is_reasoning_model = any(x in model.lower() for x in ["gpt-5", "o1-", "o3-"])
            
            self.lm = dspy.LM(
                model=f"openai/{model}",
                api_key=api_key,
                max_tokens=16000 if is_reasoning_model else 4096,
                temperature=1.0 if is_reasoning_model else 0.1,
            )
            
            # Use local context instead of global dspy.configure to avoid threading issues
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
    
    def generate(self, faa_report_text: str, report_id: str) -> ScenarioConfig:
        """
        Generate PX4 config from FAA sighting report.
        
        Args:
            faa_report_text: Complete FAA UAS sighting report text
            report_id: Unique identifier for the report
            
        Returns:
            ScenarioConfig with all simulation parameters
            
        Raises:
            ScenarioGenerationError on failure
        """
        if not self.is_ready:
            raise ScenarioGenerationError("Generator not initialized")
        
        if not faa_report_text or len(faa_report_text.strip()) < 20:
            raise ScenarioGenerationError("FAA report text is empty or too short")
        
        logger.info(f"Generating config: {report_id}")
        
        # Initialize LLM logger for this request
        if self._output_dir:
            from pathlib import Path
            self._llm_logger = LLMInteractionLogger(
                output_dir=Path(self._output_dir),
                phase=1,
                report_id=report_id
            )
        
        try:
            # Log request start
            input_fields = {
                "faa_report_text": faa_report_text,
                "faa_report_id": report_id,
            }
            if self._llm_logger:
                clear_dspy_history(self.lm)  # Clear previous history
                self._llm_logger.log_request_start("FAA_To_PX4_Complete", input_fields)
            
            with dspy.context(lm=self.lm):
                result = self._translator(
                    faa_report_text=faa_report_text,
                    faa_report_id=report_id,
                )
            
            # Log response with DSPy history
            if self._llm_logger:
                self._llm_logger.log_response(result, get_dspy_history(self.lm))
            
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
            
            # Validate LLM geocoding (CRITICAL: Prevents hallucinated coordinates)
            validated_lat, validated_lon, was_corrected = validate_geocoding(
                llm_lat=float(result.lat),
                llm_lon=float(result.lon),
                city=str(result.city),
                state=str(result.state),
            )
            if was_corrected:
                logger.info(f"🌍 Geocoding corrected: LLM ({result.lat:.4f}, {result.lon:.4f}) → ({validated_lat:.4f}, {validated_lon:.4f})")
                # Also update waypoints to use corrected coordinates
                for wp in waypoints:
                    if 'lat' in wp and 'lon' in wp:
                        # Shift waypoints to corrected location (preserve relative positions)
                        lat_offset = validated_lat - float(result.lat)
                        lon_offset = validated_lon - float(result.lon)
                        wp['lat'] = wp['lat'] + lat_offset
                        wp['lon'] = wp['lon'] + lon_offset
            
            # P0: Determine if fault injection is supported
            px4_cmd = str(result.px4_fault_cmd).strip().lower()
            fault_injection_supported = px4_cmd not in ["none", "", "n/a", "not_supported"]
            
            # P0: Calculate uncertainty score based on inference confidence
            uncertainty_factors = []
            if str(result.weather) == "not_specified":
                uncertainty_factors.append(0.1)  # Weather unknown
            if px4_cmd == "none":
                uncertainty_factors.append(0.2)  # No direct fault injection
            if was_corrected:
                uncertainty_factors.append(0.1)  # Geocoding corrected
            if raw_altitude_m > MAX_UAV_ALTITUDE_M:
                uncertainty_factors.append(0.1)  # Altitude clamped
            uncertainty_score = min(1.0, 0.3 + sum(uncertainty_factors))  # Base 0.3 for LLM inference
            
            # Separate narrative facts (from FAA report) vs inferred parameters
            narrative_facts = {
                "location_stated": f"{result.city}, {result.state}",
                "malfunction_described": "malfunctioned" in faa_report_text.lower() or "malfunction" in faa_report_text.lower(),
                "parachute_deployed": "chute" in faa_report_text.lower() or "parachute" in faa_report_text.lower(),
                "outcome_stated": "landed" if "landed" in faa_report_text.lower() or "went down" in faa_report_text.lower() else "unknown",
                "aircraft_type_stated": result.uas_type if result.uas_type.lower() not in ["unknown_multirotor", "unknown_fixed_wing", "unknown"] else None,
                "altitude_stated": raw_altitude_ft if "ft" in faa_report_text.lower() or "feet" in faa_report_text.lower() else None,
            }
            
            inferred_parameters = {
                "failure_mode": str(result.failure_mode),
                "failure_category": str(result.failure_category),
                "failure_component": str(result.failure_component),
                "flight_phase": str(result.flight_phase),
                "speed_ms": float(result.speed_ms),
                "wind_speed_ms": float(result.wind_speed_ms),
                "environment_type": str(result.environment),
                "inference_reasoning": str(result.reasoning)[:500],
            }
            
            # Proxy modeling tags for platform substitution
            uas_type_lower = str(result.uas_type).lower()
            is_fixed_wing = any(x in uas_type_lower for x in ["rq-7", "rq7", "fixed", "wing", "plane"])
            proxy_modeling = {
                "source_aircraft_class": "fixed_wing" if is_fixed_wing else "multirotor",
                "source_aircraft_type": str(result.uas_type),
                "simulation_platform": "x500_quadcopter",
                "platform_substitution": is_fixed_wing,
                "substitution_reason": "PX4 SITL uses multirotor model; fixed-wing requires different simulator" if is_fixed_wing else None,
                "parachute_modeled": narrative_facts["parachute_deployed"],
                "parachute_trigger": "control_loss_recovery" if narrative_facts["parachute_deployed"] else None,
            }
            
            # Evidence traceability map (parameter → source)
            evidence_map = {
                "city": "FAA_NARRATIVE",
                "state": "FAA_NARRATIVE", 
                "lat": "GEOCODER_API" if was_corrected else "LLM_INFERENCE",
                "lon": "GEOCODER_API" if was_corrected else "LLM_INFERENCE",
                "altitude_m": "FAA_NARRATIVE" if narrative_facts["altitude_stated"] else "LLM_DEFAULT",
                "uas_type": "FAA_NARRATIVE" if narrative_facts["aircraft_type_stated"] else "LLM_INFERENCE",
                "failure_mode": "LLM_INFERENCE",
                "failure_category": "LLM_INFERENCE",
                "failure_component": "LLM_INFERENCE",
                "px4_fault_cmd": "LLM_MAPPING" if fault_injection_supported else "NOT_SUPPORTED",
                "waypoints": "LLM_GENERATED",
                "weather": "FAA_NARRATIVE" if str(result.weather) != "not_specified" else "LLM_DEFAULT",
                "wind_speed_ms": "LLM_DEFAULT",
                "outcome": "FAA_NARRATIVE" if narrative_facts["outcome_stated"] != "unknown" else "LLM_INFERENCE",
            }
            
            # Determine reconstruction level
            if not fault_injection_supported:
                reconstruction_level = "behavioral_class"  # Can only simulate class of behavior
            elif proxy_modeling["platform_substitution"]:
                reconstruction_level = "proxy_simulation"  # Different platform
            else:
                reconstruction_level = "partial_match"  # Same class, some inferences
            
            config = ScenarioConfig(
                faa_report_id=report_id,
                faa_report_text=faa_report_text[:500],
                
                city=str(result.city),
                state=str(result.state),
                lat=validated_lat,
                lon=validated_lon,
                
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
                
                # Uncertainty & evidence tracking
                uncertainty_score=uncertainty_score,
                fault_injection_supported=fault_injection_supported,
                
                # Narrative vs inferred separation
                narrative_facts=narrative_facts,
                inferred_parameters=inferred_parameters,
                
                # Proxy simulation tags
                proxy_modeling=proxy_modeling,
                
                # Evidence traceability
                evidence_map=evidence_map,
                reconstruction_level=reconstruction_level,
            )
            
            logger.info(f"✅ Generated: {config.failure_mode} at {config.city}, {config.state} (uncertainty: {uncertainty_score:.2f}, level: {reconstruction_level})")
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
            report_id=sighting.get("report_id", sighting.get("incident_id", "UNKNOWN")),
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


def generate_scenario(faa_report: str, report_id: str) -> ScenarioConfig:
    """Convenience function to generate scenario config from FAA sighting report."""
    return get_scenario_generator().generate(faa_report, report_id)


__all__ = [
    "ScenarioGenerator",
    "ScenarioConfig",
    "ScenarioGenerationError",
    "get_scenario_generator",
    "generate_scenario",
    "MAX_MISSION_DURATION_SEC",
]
