"""
Executable Safety Reliability Index (ESRI)
==========================================
Author: AeroGuardian Member
Date: 2026-01-21

Single end-to-end trust score combining all evaluation metrics.

FORMULA:
--------
ESRI = SFS × BRR × ECC

SCIENTIFIC RATIONALE:
---------------------
Using multiplication ensures that weakness in ANY stage propagates to the final score.
A system cannot be trusted if:
- The LLM mistranslates the FAA report (low SFS)
- The simulation fails to reproduce behavior (low BRR)
- The report is not grounded in evidence (low ECC)

TRUST LEVELS:
-------------
- ESRI >= 0.7: HIGH TRUST - System is reliable
- ESRI >= 0.4: MEDIUM TRUST - Review recommended
- ESRI < 0.4: LOW TRUST - Do not rely on output
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
    
    # Trust assessment
    trust_level: str = "LOW"  # HIGH, MEDIUM, LOW
    trust_justification: str = ""
    
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
            "trust_level": self.trust_level,
            "trust_justification": self.trust_justification,
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
            f"ESRI: {self.esri * 100:.1f}% ({self.trust_level} TRUST)\n"
            f"├─ SFS (Scenario Fidelity): {self.sfs * 100:.1f}%\n"
            f"├─ BRR (Behavior Reproduction): {self.brr * 100:.1f}%\n"
            f"└─ ECC (Evidence Consistency): {self.ecc * 100:.1f}%\n"
            f"\nJustification: {self.trust_justification}"
        )


class ESRICalculator:
    """
    Computes Executable Safety Reliability Index (ESRI).
    
    Aggregates SFS, BRR, and ECC into a single trust score.
    """
    
    # Trust level thresholds
    HIGH_TRUST_THRESHOLD = 0.7
    MEDIUM_TRUST_THRESHOLD = 0.4
    
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
            ESRIResult with final score and trust assessment
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
        
        # Determine trust level
        result.trust_level, result.trust_justification = self._assess_trust(
            result.sfs, result.brr, result.ecc, result.esri
        )
        
        logger.info(
            f"ESRI calculated: {result.esri * 100:.1f}% = "
            f"{result.sfs * 100:.0f}% × {result.brr * 100:.0f}% × {result.ecc * 100:.0f}% "
            f"({result.trust_level} TRUST)"
        )
        
        return result
    
    def _assess_trust(
        self,
        sfs: float,
        brr: float,
        ecc: float,
        esri: float
    ) -> tuple:
        """Assess trust level and generate justification."""
        
        weakest = min(sfs, brr, ecc)
        weakest_name = "SFS" if sfs == weakest else ("BRR" if brr == weakest else "ECC")
        
        if esri >= self.HIGH_TRUST_THRESHOLD:
            return "HIGH", (
                f"All components strong. ESRI {esri:.3f} exceeds threshold {self.HIGH_TRUST_THRESHOLD}. "
                f"System output is reliable for this incident."
            )
        elif esri >= self.MEDIUM_TRUST_THRESHOLD:
            return "MEDIUM", (
                f"ESRI {esri:.3f} is acceptable but {weakest_name} ({weakest:.3f}) is weak. "
                f"Manual review recommended before acting on output."
            )
        else:
            # Identify the bottleneck
            if sfs < 0.4:
                bottleneck = "LLM translation of FAA report may be inaccurate"
            elif brr < 0.4:
                bottleneck = "Simulation did not reproduce expected abnormal behavior"
            elif ecc < 0.4:
                bottleneck = "Safety report claims are not well-supported by telemetry"
            else:
                bottleneck = "Multiple weak components"
            
            return "LOW", (
                f"ESRI {esri:.3f} is below minimum threshold. "
                f"Bottleneck: {bottleneck}. "
                f"System output should NOT be trusted for this incident."
            )
    
    def calculate_aggregate(self, individual_results: list) -> Dict:
        """
        Calculate aggregate ESRI across multiple incidents.
        
        Args:
            individual_results: List of ESRIResult dicts
            
        Returns:
            Aggregate statistics
        """
        if not individual_results:
            return {
                "aggregate_ESRI": 0.0,
                "count": 0,
                "trust_distribution": {},
            }
        
        esri_scores = [r.get("ESRI", 0) for r in individual_results]
        sfs_scores = [r.get("component_scores", {}).get("SFS", 0) for r in individual_results]
        brr_scores = [r.get("component_scores", {}).get("BRR", 0) for r in individual_results]
        ecc_scores = [r.get("component_scores", {}).get("ECC", 0) for r in individual_results]
        
        trust_counts = {
            "HIGH": sum(1 for r in individual_results if r.get("trust_level") == "HIGH"),
            "MEDIUM": sum(1 for r in individual_results if r.get("trust_level") == "MEDIUM"),
            "LOW": sum(1 for r in individual_results if r.get("trust_level") == "LOW"),
        }
        
        def avg(arr):
            return sum(arr) / len(arr) if arr else 0.0
        
        return {
            "aggregate_ESRI": {
                "mean": round(avg(esri_scores), 4),
                "min": round(min(esri_scores), 4),
                "max": round(max(esri_scores), 4),
            },
            "component_averages": {
                "SFS": round(avg(sfs_scores), 3),
                "BRR": round(avg(brr_scores), 3),
                "ECC": round(avg(ecc_scores), 3),
            },
            "count": len(individual_results),
            "trust_distribution": trust_counts,
            "reliability_rate": round(
                (trust_counts["HIGH"] + trust_counts["MEDIUM"]) / len(individual_results), 3
            ) if individual_results else 0.0,
        }
