"""
Reporting Module for AeroGuardian
==================================
Author: AeroGuardian Member
Date: 2026-01-30

Multi-format safety report generation for pre-flight analysis.

Output Formats:
- JSON: Structured data for programmatic access
- Excel: Multi-sheet workbook for analysis
- PDF: Professional report for stakeholders

USAGE:
    from src.reporting import UnifiedReporter
    
    reporter = UnifiedReporter(output_dir)
    reporter.generate(incident, config, telemetry, report)
"""

from pathlib import Path

MODULE_ROOT = Path(__file__).parent

# Main entry point
from .unified_reporter import (
    UnifiedReporter,
    generate_reports,
)

__all__ = [
    "UnifiedReporter",
    "generate_reports",
]
