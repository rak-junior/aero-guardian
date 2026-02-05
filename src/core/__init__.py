"""
Core Module for AeroGuardian
=============================
Author: AeroGuardian Member
Date: 2026-01-30

Central utilities and connectors used across the project:
- OpenAI API connector
- Logging configuration
- PDF report generation
- Geocoding utilities

USAGE:
    from src.core import get_logger, get_openai
    
    logger = get_logger("MyModule")
    openai = get_openai()
"""

from pathlib import Path

MODULE_ROOT = Path(__file__).parent

# Logging (most commonly used)
from .logging_config import (
    get_logger,
    get_llm_logger,
    get_dspy_logger,
    log_exception,
)

# OpenAI connector
from .openai_connector import (
    OpenAIConnector,
    get_openai,
    ChatResponse,
)

# Geocoding
from .geocoder import (
    geocode,
    geocode_incident,
)

# PDF generation
from .pdf_report_generator import (
    PDFGenerator,
    generate_pdf,
)

# Configuration
from .config import (
    Config,
    get_config,
)

# Safety Guardrails (Optional - may not be installed)
try:
    from .guardrails import (
        GuardrailSystem,
        GuardrailViolation,
        ProvenanceTracker,
        ConfidenceCeiling,
        AircraftClassValidator,
        AircraftClass,
        SimulationEnvelopeDisclosure,
        ProvenanceError,
        AircraftMismatchError,
    )
    HAS_GUARDRAILS = True
except ImportError:
    HAS_GUARDRAILS = False
    # Define stubs for optional guardrails
    GuardrailSystem = None
    GuardrailViolation = None
    ProvenanceTracker = None
    ConfidenceCeiling = None
    AircraftClassValidator = None
    AircraftClass = None
    SimulationEnvelopeDisclosure = None
    ProvenanceError = None
    AircraftMismatchError = None

__all__ = [
    # Logging
    "get_logger",
    "get_llm_logger",
    "get_dspy_logger",
    "log_exception",
    
    # OpenAI
    "OpenAIConnector",
    "get_openai",
    "ChatResponse",
    
    # Geocoding
    "geocode",
    "geocode_incident",
    
    # PDF
    "PDFGenerator",
    "generate_pdf",
    
    # Configuration
    "Config",
    "get_config",
    
    # Safety Guardrails
    "GuardrailSystem",
    "GuardrailViolation",
    "ProvenanceTracker",
    "ConfidenceCeiling",
    "AircraftClassValidator",
    "AircraftClass",
    "SimulationEnvelopeDisclosure",
    "ProvenanceError",
    "AircraftMismatchError",
]
