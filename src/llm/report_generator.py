"""
Report Generator for AeroGuardian
==================================
Author: AeroGuardian Member
Date: 2026-01-30

Generates pre-flight safety reports from simulation telemetry.
Uses GeneratePreFlightReport DSPy signature.

USAGE:
    from src.llm import ReportGenerator
    
    generator = ReportGenerator()
    report = generator.generate(incident_info, telemetry_summary)
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

import dspy

from .signatures import GeneratePreFlightReport

logger = logging.getLogger("AeroGuardian.ReportGenerator")


# =============================================================================
# Data Classes
# =============================================================================

class ReportGenerationError(Exception):
    """Raised when report generation fails."""


@dataclass
class SafetyReport:
    """Pre-flight safety report - all values from LLM."""
    
    incident_id: str
    incident_location: str
    fault_type: str
    expected_outcome: str
    
    # Section 1: Safety Level & Cause
    safety_level: str
    primary_hazard: str
    observed_effect: str
    
    # Section 2: Design Constraints & Recommendations
    design_constraints: list
    recommendations: list
    
    # Section 3: Explanation
    explanation: str
    
    # Final Verdict
    verdict: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Report Generator
# =============================================================================

class ReportGenerator:
    """
    Generate pre-flight safety reports from simulation telemetry.
    
    Uses GeneratePreFlightReport DSPy signature.
    
    USAGE:
        generator = ReportGenerator()
        report = generator.generate(
            incident_description="...",
            incident_id="FAA_xxx",
            incident_location="City, State",
            fault_type="motor_failure",
            expected_outcome="crash",
            telemetry_summary="duration: 120s, max_alt: 50m, ..."
        )
    """
    
    def __init__(self):
        self.is_ready = False
        self._generator = None
        self._configure()
    
    def _configure(self):
        """Configure DSPy with OpenAI."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ReportGenerationError("OPENAI_API_KEY not set")
        
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
            self._generator = dspy.ChainOfThought(GeneratePreFlightReport)
            
            # Load few-shot examples
            try:
                from .dspy_fewshot import get_preflight_report_examples
                examples = get_preflight_report_examples()
                if examples:
                    self._generator.demos = examples[:2]
                    logger.info(f"Loaded {len(examples)} few-shot examples")
            except ImportError:
                logger.debug("Few-shot examples not available")
            
            self.is_ready = True
            logger.info(f"ReportGenerator ready: {model}")
            
        except Exception as e:
            raise ReportGenerationError(f"Failed to initialize LLM: {e}")
    
    def generate(
        self,
        incident_description: str,
        incident_id: str,
        incident_location: str,
        fault_type: str,
        expected_outcome: str,
        telemetry_summary: str,
    ) -> SafetyReport:
        """
        Generate pre-flight safety report.
        
        Args:
            incident_description: Original FAA incident narrative
            incident_id: FAA incident ID
            incident_location: City, State
            fault_type: MOTOR_FAILURE, GPS_LOSS, etc.
            expected_outcome: crash, controlled_landing, flyaway
            telemetry_summary: Telemetry analysis metrics
            
        Returns:
            SafetyReport with all sections
            
        Raises:
            ReportGenerationError on failure
        """
        if not self.is_ready:
            raise ReportGenerationError("Generator not initialized")
        
        if not incident_description:
            raise ReportGenerationError("incident_description required")
        
        logger.info(f"Generating safety report: {incident_id}")
        
        try:
            result = self._generator(
                incident_description=incident_description,
                incident_location=incident_location,
                fault_type=fault_type,
                expected_outcome=expected_outcome,
                telemetry_summary=telemetry_summary,
            )
            
            # Parse constraints and recommendations
            constraints = [c.strip() for c in str(result.design_constraints).split("|") if c.strip()]
            recommendations = [r.strip() for r in str(result.recommendations).split("|") if r.strip()]
            
            report = SafetyReport(
                incident_id=incident_id,
                incident_location=incident_location,
                fault_type=fault_type,
                expected_outcome=expected_outcome,
                
                safety_level=str(result.safety_level),
                primary_hazard=str(result.primary_hazard),
                observed_effect=str(result.observed_effect),
                
                design_constraints=constraints,
                recommendations=recommendations,
                
                explanation=str(result.explanation),
                verdict=str(result.verdict),
            )
            
            logger.info(f"✅ Generated report: {report.safety_level} / {report.verdict}")
            return report
            
        except ReportGenerationError:
            raise
        except Exception as e:
            raise ReportGenerationError(f"LLM generation failed: {e}")


# =============================================================================
# Singleton Access
# =============================================================================

_generator: Optional[ReportGenerator] = None


def get_report_generator() -> ReportGenerator:
    """Get report generator singleton. Raises if not available."""
    global _generator
    if _generator is None:
        _generator = ReportGenerator()
    return _generator


__all__ = [
    "ReportGenerator",
    "SafetyReport",
    "ReportGenerationError",
    "get_report_generator",
]
