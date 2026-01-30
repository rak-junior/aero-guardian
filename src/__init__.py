"""
AeroGuardian Source Module
===========================
Author: AeroGuardian Member
Date: 2026-01-01

End-to-End Pre-Flight Safety Analysis Pipeline:
- FAA incident loading
- LLM-based scenario generation (GPT-4o)
- PX4 SITL simulation
- Safety report generation
"""

from pathlib import Path

PHASE_B_ROOT = Path(__file__).parent
PROJECT_ROOT = PHASE_B_ROOT.parent.parent
