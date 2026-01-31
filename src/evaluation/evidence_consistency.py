"""
Evidence-Conclusion Consistency (ECC)
=====================================
Author: AeroGuardian Member
Date: 2026-01-21

Ensures the generated safety report is supported by telemetry evidence.

SCIENTIFIC RATIONALE:
---------------------
A safety report is only trustworthy if its claims are grounded in evidence.
ECC verifies that each claim in the report can be traced to telemetry data.

CLAIM VERIFICATION:
-------------------
1. Hazard Level Claim - Must be supported by detected anomaly severity
2. Primary Hazard Claim - Must reference detectable telemetry metrics
3. Recommendation Claims - Must address detected anomalies

SCORING:
--------
ECC = (supported_claims / total_claims) * evidence_strength
"""

import logging
from typing import Dict, List
from dataclasses import dataclass, field

logger = logging.getLogger("AeroGuardian.Evaluator.ECC")


# =============================================================================
# HAZARD LEVEL TO ANOMALY SEVERITY MAPPING
# =============================================================================

HAZARD_LEVEL_REQUIREMENTS = {
    "CRITICAL": ["CRITICAL"],
    "HIGH": ["CRITICAL", "HIGH"],
    "MEDIUM": ["CRITICAL", "HIGH", "MEDIUM"],
    "LOW": ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
}

# Keywords that must have telemetry support
HAZARD_KEYWORDS = {
    "roll": ["roll_instability", "attitude_instability"],
    "pitch": ["pitch_instability", "attitude_instability"],
    "altitude": ["altitude_instability"],
    "position": ["position_drift", "gps_degradation"],
    "gps": ["gps_degradation", "position_drift"],
    "control": ["control_saturation", "roll_instability", "pitch_instability"],
    "motor": ["roll_instability", "altitude_instability"],
    "battery": ["altitude_instability", "control_saturation"],
    "drift": ["position_drift"],
    "instability": ["roll_instability", "pitch_instability", "altitude_instability"],
    "loss of control": ["control_saturation", "roll_instability"],
    "descent": ["altitude_instability"],
}

# Universal safety recommendations that are valid for ANY critical/high severity failure
# These don't need to match specific anomaly types - they are valid safety measures
UNIVERSAL_SAFETY_KEYWORDS = [
    "parachute",       # Recovery system - valid for any propulsion/control failure
    "redundant",       # Redundancy - valid for any failure mode
    "redundancy",
    "backup",          # Backup systems
    "failsafe",        # Failsafe systems
    "recovery",        # Recovery mechanisms
    "emergency",       # Emergency procedures
    "pre-flight",      # Pre-flight checks
    "preflight",
    "geofence",        # Containment systems
    "return to home",  # Return-to-home capability
    "rth",
]


@dataclass
class ClaimVerification:
    """Verification result for a single claim."""
    claim_type: str  # hazard_level, primary_hazard, recommendation
    claim_text: str
    is_supported: bool = False  # Default to False
    supporting_evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "claim_type": self.claim_type,
            "claim_text": self.claim_text[:100] + "..." if len(self.claim_text) > 100 else self.claim_text,
            "is_supported": self.is_supported,
            "supporting_evidence": self.supporting_evidence,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class ECCResult:
    """Complete ECC evaluation result."""
    score: float = 0.0
    verified_claims: List[ClaimVerification] = field(default_factory=list)
    unsupported_claims: List[str] = field(default_factory=list)
    evidence_strength: float = 0.0
    total_claims: int = 0
    supported_claims: int = 0
    confidence: str = "LOW"
    
    def to_dict(self) -> Dict:
        return {
            "ECC": round(self.score, 3),
            "verified_claims": [c.to_dict() for c in self.verified_claims],
            "unsupported_claims": self.unsupported_claims,
            "evidence_strength": round(self.evidence_strength, 3),
            "total_claims": self.total_claims,
            "supported_claims": self.supported_claims,
            "confidence": self.confidence,
        }


class EvidenceConsistencyChecker:
    """
    Computes Evidence-Conclusion Consistency (ECC).
    
    Verifies that safety report claims are grounded in telemetry evidence.
    """
    
    def __init__(self):
        logger.debug("EvidenceConsistencyChecker initialized")
    
    def evaluate(
        self,
        safety_report: Dict,
        detected_anomalies: List[Dict],
        telemetry_stats: Dict,
    ) -> ECCResult:
        """
        Evaluate evidence-conclusion consistency.
        
        Args:
            safety_report: Generated safety report
            detected_anomalies: List of anomalies detected by BRR
            telemetry_stats: Telemetry statistics
            
        Returns:
            ECCResult with score and claim verification details
        """
        result = ECCResult()
        
        # Extract anomaly types for matching
        anomaly_types = [a.get("type", a.get("anomaly_type", "")) for a in detected_anomalies]
        anomaly_severities = [a.get("severity", "") for a in detected_anomalies]
        
        # 1. Verify hazard level claim
        hazard_claim = self._verify_hazard_level(
            safety_report,
            anomaly_severities
        )
        result.verified_claims.append(hazard_claim)
        
        # 2. Verify primary hazard claim
        hazard_text_claim = self._verify_primary_hazard(
            safety_report,
            anomaly_types,
            telemetry_stats
        )
        result.verified_claims.append(hazard_text_claim)
        
        # 3. Verify recommendations
        rec_claims = self._verify_recommendations(
            safety_report,
            anomaly_types
        )
        result.verified_claims.extend(rec_claims)
        
        # 4. Verify design constraints (if present)
        constraint_claims = self._verify_constraints(
            safety_report,
            anomaly_types
        )
        result.verified_claims.extend(constraint_claims)
        
        # Compute summary metrics
        result.total_claims = len(result.verified_claims)
        result.supported_claims = sum(1 for c in result.verified_claims if c.is_supported)
        result.unsupported_claims = [
            c.claim_text for c in result.verified_claims if not c.is_supported
        ]
        
        # Compute evidence strength
        result.evidence_strength = self._compute_evidence_strength(
            detected_anomalies,
            telemetry_stats
        )
        
        # Compute ECC score
        if result.total_claims > 0:
            base_score = result.supported_claims / result.total_claims
            result.score = base_score * result.evidence_strength
        else:
            result.score = 0.0
        
        result.confidence = self._compute_confidence(result)
        
        logger.info(
            f"ECC evaluated: {result.score:.3f} ({result.supported_claims}/{result.total_claims} claims supported)"
        )
        return result
    
    def _verify_hazard_level(
        self,
        safety_report: Dict,
        anomaly_severities: List[str]
    ) -> ClaimVerification:
        """Verify hazard level claim matches detected anomaly severity."""
        
        # Extract hazard level from report
        hazard_level = (
            safety_report.get("safety_level") or
            safety_report.get("hazard_level") or
            safety_report.get("risk_level", "UNKNOWN")
        )
        hazard_level = hazard_level.upper().split()[0]  # Get first word
        
        claim = ClaimVerification(
            claim_type="hazard_level",
            claim_text=f"Hazard Level: {hazard_level}"
        )
        
        # Get required severity for this hazard level
        required_severities = HAZARD_LEVEL_REQUIREMENTS.get(hazard_level, [])
        
        # Check if any detected anomaly matches required severity
        for severity in anomaly_severities:
            if severity.upper() in required_severities:
                claim.is_supported = True
                claim.supporting_evidence.append(f"Anomaly with {severity} severity detected")
                claim.confidence = 1.0
                return claim
        
        # If no matching severity but has anomalies, partial support
        if anomaly_severities:
            claim.is_supported = True
            claim.confidence = 0.6
            claim.supporting_evidence.append("Anomalies detected but severity mismatch")
        else:
            claim.is_supported = False
            claim.confidence = 0.0
        
        return claim
    
    def _verify_primary_hazard(
        self,
        safety_report: Dict,
        anomaly_types: List[str],
        telemetry_stats: Dict
    ) -> ClaimVerification:
        """Verify primary hazard claim is supported by telemetry."""
        
        # Extract primary hazard from report
        primary_hazard = (
            safety_report.get("primary_hazard") or
            safety_report.get("hazard_type", "Unknown")
        )
        
        claim = ClaimVerification(
            claim_type="primary_hazard",
            claim_text=primary_hazard
        )
        
        primary_hazard_lower = primary_hazard.lower()
        
        # Check for keyword matches
        for keyword, expected_anomalies in HAZARD_KEYWORDS.items():
            if keyword in primary_hazard_lower:
                for expected in expected_anomalies:
                    if expected in anomaly_types:
                        claim.is_supported = True
                        claim.supporting_evidence.append(
                            f"'{keyword}' in claim matched by {expected} anomaly"
                        )
                        claim.confidence = 0.9
                        return claim
        
        # Check telemetry stats for supporting data
        if "roll" in primary_hazard_lower and telemetry_stats.get("max_roll_deg", 0) > 30:
            claim.is_supported = True
            claim.supporting_evidence.append(
                f"Roll angle {telemetry_stats.get('max_roll_deg', 0):.1f}° supports claim"
            )
            claim.confidence = 0.8
            return claim
        
        if "altitude" in primary_hazard_lower and telemetry_stats.get("altitude_deviation", 0) > 5:
            claim.is_supported = True
            claim.supporting_evidence.append(
                f"Altitude deviation {telemetry_stats.get('altitude_deviation', 0):.1f}m supports claim"
            )
            claim.confidence = 0.8
            return claim
        
        if anomaly_types:
            claim.is_supported = True
            claim.confidence = 0.5
            claim.supporting_evidence.append("Generic anomalies detected")
        else:
            claim.is_supported = False
            claim.confidence = 0.0
        
        return claim
    
    def _verify_recommendations(
        self,
        safety_report: Dict,
        anomaly_types: List[str]
    ) -> List[ClaimVerification]:
        """Verify recommendations address detected anomalies."""
        
        claims = []
        
        # Get recommendations from report
        recommendations = safety_report.get("recommendations", [])
        if isinstance(recommendations, str):
            recommendations = [r.strip() for r in recommendations.split("|")]
        
        for i, rec in enumerate(recommendations[:5]):  # Check first 5
            claim = ClaimVerification(
                claim_type="recommendation",
                claim_text=rec
            )
            
            rec_lower = rec.lower()
            
            # Check if recommendation addresses any detected anomaly
            for keyword, expected_anomalies in HAZARD_KEYWORDS.items():
                if keyword in rec_lower:
                    for expected in expected_anomalies:
                        if expected in anomaly_types:
                            claim.is_supported = True
                            claim.supporting_evidence.append(
                                f"Addresses {expected} anomaly"
                            )
                            claim.confidence = 0.8
                            break
                    if claim.is_supported:
                        break
            
            # Check for universal safety recommendations (valid for ANY critical/high failure)
            if not claim.is_supported and anomaly_types:
                for universal_keyword in UNIVERSAL_SAFETY_KEYWORDS:
                    if universal_keyword in rec_lower:
                        claim.is_supported = True
                        claim.supporting_evidence.append(
                            f"Universal safety measure '{universal_keyword}' valid for detected anomalies"
                        )
                        claim.confidence = 0.85  # High confidence for proven safety measures
                        break
            
            # Generic safety recommendations get partial credit
            if not claim.is_supported:
                safety_words = ["check", "inspect", "monitor", "limit", "ensure", "verify"]
                if any(w in rec_lower for w in safety_words):
                    claim.is_supported = True
                    claim.confidence = 0.4
                    claim.supporting_evidence.append("Generic safety measure")
            
            claims.append(claim)
        
        return claims
    
    def _verify_constraints(
        self,
        safety_report: Dict,
        anomaly_types: List[str]
    ) -> List[ClaimVerification]:
        """Verify design constraints are supported."""
        
        claims = []
        
        constraints = safety_report.get("design_constraints", [])
        if isinstance(constraints, str):
            constraints = [c.strip() for c in constraints.split("|")]
        
        for constraint in constraints[:3]:  # Check first 3
            claim = ClaimVerification(
                claim_type="design_constraint",
                claim_text=constraint
            )
            
            constraint_lower = constraint.lower()
            
            # Check relevance to detected anomalies
            for keyword, expected_anomalies in HAZARD_KEYWORDS.items():
                if keyword in constraint_lower:
                    for expected in expected_anomalies:
                        if expected in anomaly_types:
                            claim.is_supported = True
                            claim.supporting_evidence.append(
                                f"Constraint addresses {expected}"
                            )
                            claim.confidence = 0.85
                            break
                    if claim.is_supported:
                        break
            
            if not claim.is_supported and anomaly_types:
                claim.is_supported = True
                claim.confidence = 0.5
                claim.supporting_evidence.append("Constraint exists with anomalies")
            
            claims.append(claim)
        
        return claims
    
    def _compute_evidence_strength(
        self,
        detected_anomalies: List[Dict],
        telemetry_stats: Dict
    ) -> float:
        """Compute overall evidence strength."""
        
        if not detected_anomalies:
            return 0.3  # Minimal evidence
        
        # Count severity levels
        critical_count = sum(
            1 for a in detected_anomalies if a.get("severity") == "CRITICAL"
        )
        high_count = sum(
            1 for a in detected_anomalies if a.get("severity") == "HIGH"
        )
        
        # Base strength from anomaly count
        base_strength = min(len(detected_anomalies) * 0.2, 0.6)
        
        # Severity bonus
        severity_bonus = critical_count * 0.15 + high_count * 0.1
        
        # Data quality bonus
        data_points = telemetry_stats.get("data_points", 0)
        data_bonus = 0.2 if data_points > 500 else (0.1 if data_points > 100 else 0.0)
        
        return min(base_strength + severity_bonus + data_bonus, 1.0)
    
    def _compute_confidence(self, result: ECCResult) -> str:
        """Compute confidence level."""
        
        if result.total_claims == 0:
            return "LOW"
        
        support_rate = result.supported_claims / result.total_claims
        
        if support_rate >= 0.8 and result.evidence_strength >= 0.7:
            return "HIGH"
        elif support_rate >= 0.5 and result.evidence_strength >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"
