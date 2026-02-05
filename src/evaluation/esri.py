"""
Executable Safety Reliability Index (ESRI)
==========================================
Author: AeroGuardian Member
Date: 2026-01-21
Updated: 2026-02-01

Single end-to-end INTERNAL CONSISTENCY score combining all evaluation metrics.

DISCLAIMER:
-----------
ESRI measures INTERNAL CONSISTENCY, NOT external validation or safety certification.
A high ESRI means pipeline components agree with each other, not that the output is "safe".
ESRI cannot detect: LLM hallucinations, simulation physics errors, or FAA report inaccuracies.

FORMULA:
--------
ESRI = SFS × BRR × ECC

Where:
- SFS (Scenario Fidelity Score): Measures LLM translation consistency [0-1]
- BRR (Behavior Reproduction Rate): Measures simulation reproduction consistency [0-1]
- ECC (Evidence-Conclusion Consistency): Measures claim-evidence alignment [0-1]

SCIENTIFIC RATIONALE:
---------------------
Using multiplication (product) rather than averaging ensures that weakness in ANY
stage propagates to the final score. This aligns with fault propagation principles
where a single inconsistency compromises the overall pipeline coherence:

  - Defense-in-depth: Each stage must pass independently
  - Multiplicative penalty: 0.9 × 0.9 × 0.5 = 0.405 (correctly identifies weak ECC)
  - Zero propagation: Any component at 0 yields ESRI = 0

Low ESRI indicates pipeline inconsistency:
- The LLM translation may not align with FAA report (low SFS)
- The simulation may not match expected behavior (low BRR)
- The report claims may not align with telemetry (low ECC)

CONSISTENCY LEVELS:
-------------------
- ESRI >= 0.7: HIGH CONSISTENCY - Pipeline stages align well
- ESRI >= 0.4: MEDIUM CONSISTENCY - Some disagreement, review recommended
- ESRI < 0.4: LOW CONSISTENCY - Significant pipeline disagreement

NOTE ON REGULATORY REFERENCES:
------------------------------
The multiplicative structure is inspired by risk assessment frameworks (e.g., severity × probability).
However, ESRI is NOT a regulatory metric and has not been validated by any aviation authority.
Do not cite ESRI as evidence of regulatory compliance.
"""

import logging
from typing import Dict
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("AeroGuardian.Evaluator.ESRI")


@dataclass
class ESRIResult:
    """Complete ESRI evaluation result."""
    
    # Component scores
    sfs: float = 0.0
    brr: float = 0.0
    ecc: float = 0.0
    
    # Final score
    esri: float = 0.0
    
    # Consistency assessment (NOT safety validation)
    consistency_level: str = "LOW"  # HIGH, MEDIUM, LOW
    consistency_justification: str = ""
    
    # Detailed breakdowns
    sfs_details: Dict = field(default_factory=dict)
    brr_details: Dict = field(default_factory=dict)
    ecc_details: Dict = field(default_factory=dict)
    
    # Metadata
    evaluation_timestamp: str = ""
    incident_id: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "ESRI": round(self.esri, 4),
            "component_scores": {
                "SFS": round(self.sfs, 3),
                "BRR": round(self.brr, 3),
                "ECC": round(self.ecc, 3),
            },
            "consistency_level": self.consistency_level,
            "consistency_justification": self.consistency_justification,
            "details": {
                "scenario_fidelity": self.sfs_details,
                "behavior_reproduction": self.brr_details,
                "evidence_consistency": self.ecc_details,
            },
            "metadata": {
                "incident_id": self.incident_id,
                "evaluation_timestamp": self.evaluation_timestamp,
            }
        }
    
    def to_summary(self) -> str:
        """Generate human-readable summary with percentages."""
        return (
            f"ESRI: {self.esri * 100:.1f}% ({self.consistency_level} CONSISTENCY)\n"
            f"├─ SFS (Scenario Fidelity): {self.sfs * 100:.1f}%\n"
            f"├─ BRR (Behavior Reproduction): {self.brr * 100:.1f}%\n"
            f"└─ ECC (Evidence Consistency): {self.ecc * 100:.1f}%\n"
            f"\nJustification: {self.consistency_justification}"
        )


class ESRICalculator:
    """
    Computes Executable Safety Reliability Index (ESRI).
    
    Aggregates SFS, BRR, and ECC into a single CONSISTENCY score.
    NOTE: This measures internal pipeline consistency, NOT safety validation.
    """
    
    # Consistency level thresholds
    HIGH_CONSISTENCY_THRESHOLD = 0.7
    MEDIUM_CONSISTENCY_THRESHOLD = 0.4
    
    def __init__(self):
        logger.debug("ESRICalculator initialized")
    
    def calculate(
        self,
        sfs_result: Dict,
        brr_result: Dict,
        ecc_result: Dict,
        incident_id: str = ""
    ) -> ESRIResult:
        """
        Calculate ESRI from component scores.
        
        Args:
            sfs_result: Scenario Fidelity Score result dict
            brr_result: Behavior Reproduction Rate result dict
            ecc_result: Evidence-Conclusion Consistency result dict
            incident_id: Optional incident identifier
            
        Returns:
            ESRIResult with final score and consistency assessment
        """
        result = ESRIResult()
        result.incident_id = incident_id
        result.evaluation_timestamp = datetime.now().isoformat()
        
        # Extract scores
        result.sfs = sfs_result.get("SFS", 0.0)
        result.brr = brr_result.get("BRR", 0.0)
        result.ecc = ecc_result.get("ECC", 0.0)
        
        # Store details
        result.sfs_details = sfs_result
        result.brr_details = brr_result
        result.ecc_details = ecc_result
        
        # Calculate ESRI = SFS × BRR × ECC
        result.esri = result.sfs * result.brr * result.ecc
        
        # Determine consistency level
        result.consistency_level, result.consistency_justification = self._assess_consistency(
            result.sfs, result.brr, result.ecc, result.esri
        )
        
        logger.info(
            f"ESRI calculated: {result.esri * 100:.1f}% = "
            f"{result.sfs * 100:.0f}% × {result.brr * 100:.0f}% × {result.ecc * 100:.0f}% "
            f"({result.consistency_level} CONSISTENCY)"
        )
        
        return result
    
    def _assess_consistency(
        self,
        sfs: float,
        brr: float,
        ecc: float,
        esri: float
    ) -> tuple:
        """Assess consistency level and generate justification."""
        
        weakest = min(sfs, brr, ecc)
        weakest_name = "SFS" if sfs == weakest else ("BRR" if brr == weakest else "ECC")
        
        if esri >= self.HIGH_CONSISTENCY_THRESHOLD:
            return "HIGH", (
                f"All components align. ESRI {esri:.3f} exceeds threshold {self.HIGH_CONSISTENCY_THRESHOLD}. "
                f"Pipeline stages are consistent for this scenario."
            )
        elif esri >= self.MEDIUM_CONSISTENCY_THRESHOLD:
            return "MEDIUM", (
                f"ESRI {esri:.3f} is acceptable but {weakest_name} ({weakest:.3f}) shows disagreement. "
                f"Manual review recommended."
            )
        else:
            # Identify the inconsistency
            if sfs < 0.4:
                bottleneck = "LLM translation may not align with FAA report"
            elif brr < 0.4:
                bottleneck = "Simulation did not match expected scenario behavior"
            elif ecc < 0.4:
                bottleneck = "Report claims do not align with telemetry evidence"
            else:
                bottleneck = "Multiple stages show disagreement"
            
            return "LOW", (
                f"ESRI {esri:.3f} is below minimum threshold. "
                f"Issue: {bottleneck}. "
                f"Pipeline has significant inconsistencies for this scenario."
            )
    
    def calculate_aggregate(self, individual_results: list) -> Dict:
        """
        Calculate aggregate ESRI across multiple scenarios.
        
        Args:
            individual_results: List of ESRIResult dicts
            
        Returns:
            Aggregate statistics
        """
        if not individual_results:
            return {
                "aggregate_ESRI": 0.0,
                "count": 0,
                "consistency_distribution": {},
            }
        
        esri_scores = [r.get("ESRI", 0) for r in individual_results]
        sfs_scores = [r.get("component_scores", {}).get("SFS", 0) for r in individual_results]
        brr_scores = [r.get("component_scores", {}).get("BRR", 0) for r in individual_results]
        ecc_scores = [r.get("component_scores", {}).get("ECC", 0) for r in individual_results]
        
        consistency_counts = {
            "HIGH": sum(1 for r in individual_results if r.get("consistency_level") == "HIGH"),
            "MEDIUM": sum(1 for r in individual_results if r.get("consistency_level") == "MEDIUM"),
            "LOW": sum(1 for r in individual_results if r.get("consistency_level") == "LOW"),
        }
        
        def avg(arr):
            return sum(arr) / len(arr) if arr else 0.0
        
        def std(arr):
            """Compute population standard deviation."""
            if not arr or len(arr) < 2:
                return 0.0
            mean = avg(arr)
            variance = sum((x - mean) ** 2 for x in arr) / len(arr)
            return variance ** 0.5
        
        return {
            "aggregate_ESRI": {
                "mean": round(avg(esri_scores), 4),
                "std": round(std(esri_scores), 4),
                "min": round(min(esri_scores), 4),
                "max": round(max(esri_scores), 4),
            },
            "component_averages": {
                "SFS": {"mean": round(avg(sfs_scores), 3), "std": round(std(sfs_scores), 3)},
                "BRR": {"mean": round(avg(brr_scores), 3), "std": round(std(brr_scores), 3)},
                "ECC": {"mean": round(avg(ecc_scores), 3), "std": round(std(ecc_scores), 3)},
            },
            "count": len(individual_results),
            "consistency_distribution": consistency_counts,
            "high_consistency_rate": round(
                (consistency_counts["HIGH"] + consistency_counts["MEDIUM"]) / len(individual_results), 3
            ) if individual_results else 0.0,
        }

