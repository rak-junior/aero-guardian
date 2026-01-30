"""
Evaluation Module for AeroGuardian
===================================
Author: AeroGuardian Member
Date: 2026-01-30

Research-grade evaluation framework for quantifying LLM-based safety analysis.

Metrics (ESRI Framework):
- SFS: Scenario Fidelity Score (LLM translation accuracy)
- BRR: Behavior Reproduction Rate (Simulation validity)  
- ECC: Evidence-Conclusion Consistency (Report groundedness)
- ESRI: Executable Safety Reliability Index = SFS × BRR × ECC

USAGE:
    from src.evaluation import CaseEvaluator, get_case_evaluator
    
    evaluator = get_case_evaluator()
    result = evaluator.evaluate(incident, config, telemetry, report)
    
    print(f"ESRI Score: {result.esri.score:.2%}")
"""

# Main entry point (recommended)
from .evaluate_case import (
    CaseEvaluator,
    CaseEvaluationResult,
    EvaluationExcelExporter,
    get_case_evaluator,
)

# Individual metrics (for advanced usage)
from .scenario_fidelity import ScenarioFidelityScorer, SFSResult
from .behavior_validation import BehaviorValidator, BRRResult
from .evidence_consistency import EvidenceConsistencyChecker, ECCResult
from .esri import ESRICalculator, ESRIResult

__all__ = [
    # Main entry point (recommended)
    "CaseEvaluator",
    "CaseEvaluationResult",
    "EvaluationExcelExporter",
    "get_case_evaluator",
    
    # Scenario Fidelity Score (SFS)
    "ScenarioFidelityScorer",
    "SFSResult",
    
    # Behavior Reproduction Rate (BRR)
    "BehaviorValidator",
    "BRRResult",
    
    # Evidence-Conclusion Consistency (ECC)
    "EvidenceConsistencyChecker",
    "ECCResult",
    
    # ESRI (combined metric)
    "ESRICalculator",
    "ESRIResult",
]
