"""
Scenario Validator - Semantic Similarity Validation
====================================================
Author: AeroGuardian Member
Date: 2026-01-18

Validates simulation fidelity by computing semantic similarity between:
- Original FAA incident description
- Simulation outcome summary

Uses sentence-transformers for embedding-based comparison.
This addresses the expert feedback regarding validation metrics.

Usage:
    from src.validation.scenario_validator import ScenarioValidator
    
    validator = ScenarioValidator()
    score = validator.compute_match_score(
        faa_description="UAS lost GPS signal and crashed...",
        simulation_summary="Simulated GPS dropout caused loss of control..."
    )
    print(f"Match score: {score:.2f}")  # 0.0 to 1.0
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger("AeroGuardian.Validator")

# Try importing sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")


@dataclass
class ValidationResult:
    """Result of scenario validation."""
    match_score: float  # 0.0 to 1.0
    faa_keywords_found: List[str]
    simulation_keywords_found: List[str]
    is_valid: bool  # True if score >= threshold
    reasoning: str


class ScenarioValidator:
    """
    Validates simulation fidelity against FAA incident descriptions.
    
    Uses semantic similarity (sentence embeddings) to compare:
    - What the FAA report described
    - What the simulation produced
    
    This provides a quantitative metric for "behavior reproduction fidelity".
    """
    
    # Key failure-related terms to look for
    FAILURE_KEYWORDS = {
        "gps": ["gps", "satellite", "navigation", "position", "lost signal"],
        "motor": ["motor", "propeller", "prop", "engine", "thrust", "spinning"],
        "battery": ["battery", "power", "voltage", "charge", "low battery"],
        "control": ["control", "lost control", "erratic", "unstable", "uncontrollable"],
        "crash": ["crash", "fell", "impacted", "ground", "descended rapidly"],
        "flyaway": ["flyaway", "fly away", "lost link", "disconnected", "out of range"],
    }
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.5):
        """
        Initialize validator.
        
        Args:
            model_name: Sentence transformer model to use
            threshold: Minimum similarity score for validation pass
        """
        self.threshold = threshold
        self.model = None
        self.model_name = model_name
        
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                logger.info(f"Loading sentence transformer: {model_name}")
                self.model = SentenceTransformer(model_name)
                logger.info("✓ Sentence transformer loaded")
            except Exception as e:
                logger.error(f"Failed to load sentence transformer: {e}")
    
    def is_ready(self) -> bool:
        """Check if validator is ready."""
        return self.model is not None
    
    def compute_match_score(
        self, 
        faa_description: str, 
        simulation_summary: str
    ) -> float:
        """
        Compute semantic similarity between FAA description and simulation summary.
        
        Args:
            faa_description: Original FAA incident report text
            simulation_summary: Summary of simulation outcome
            
        Returns:
            Similarity score from 0.0 (no match) to 1.0 (perfect match)
        """
        if not self.is_ready():
            logger.warning("Validator not ready, using keyword fallback")
            return self._keyword_similarity(faa_description, simulation_summary)
        
        try:
            # Encode both texts
            emb_faa = self.model.encode(faa_description, convert_to_numpy=True)
            emb_sim = self.model.encode(simulation_summary, convert_to_numpy=True)
            
            # Cosine similarity
            similarity = np.dot(emb_faa, emb_sim) / (
                np.linalg.norm(emb_faa) * np.linalg.norm(emb_sim)
            )
            
            # Clamp to [0, 1]
            return float(max(0.0, min(1.0, similarity)))
            
        except Exception as e:
            logger.error(f"Embedding computation failed: {e}")
            return self._keyword_similarity(faa_description, simulation_summary)
    
    def _keyword_similarity(self, text1: str, text2: str) -> float:
        """Fallback: Compute keyword-based similarity."""
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        # Find matching keyword categories
        matches = 0
        total = 0
        
        for category, keywords in self.FAILURE_KEYWORDS.items():
            text1_has = any(kw in text1_lower for kw in keywords)
            text2_has = any(kw in text2_lower for kw in keywords)
            
            if text1_has or text2_has:
                total += 1
                if text1_has and text2_has:
                    matches += 1
        
        if total == 0:
            return 0.5  # Neutral if no keywords found
        
        return matches / total
    
    def validate(
        self, 
        faa_description: str, 
        simulation_summary: str,
        simulation_telemetry: Optional[Dict] = None
    ) -> ValidationResult:
        """
        Full validation with detailed results.
        
        Args:
            faa_description: Original FAA incident text
            simulation_summary: Summary of simulation
            simulation_telemetry: Optional telemetry stats for additional checks
            
        Returns:
            ValidationResult with score, keywords, and reasoning
        """
        # Compute similarity score
        score = self.compute_match_score(faa_description, simulation_summary)
        
        # Extract keywords found
        faa_lower = faa_description.lower()
        sim_lower = simulation_summary.lower()
        
        faa_keywords = []
        sim_keywords = []
        
        for category, keywords in self.FAILURE_KEYWORDS.items():
            for kw in keywords:
                if kw in faa_lower:
                    faa_keywords.append(kw)
                if kw in sim_lower:
                    sim_keywords.append(kw)
        
        # Determine if valid
        is_valid = score >= self.threshold
        
        # Generate reasoning
        if is_valid:
            reasoning = f"Simulation matches FAA description with {score*100:.0f}% similarity (threshold: {self.threshold*100:.0f}%)"
        else:
            reasoning = f"Simulation similarity ({score*100:.0f}%) below threshold ({self.threshold*100:.0f}%). Keywords found in FAA: {faa_keywords[:3]}, in simulation: {sim_keywords[:3]}"
        
        return ValidationResult(
            match_score=score,
            faa_keywords_found=list(set(faa_keywords))[:10],
            simulation_keywords_found=list(set(sim_keywords))[:10],
            is_valid=is_valid,
            reasoning=reasoning
        )
    
    def batch_validate(
        self, 
        pairs: List[Tuple[str, str]]
    ) -> Dict[str, float]:
        """
        Validate multiple FAA/simulation pairs.
        
        Args:
            pairs: List of (faa_description, simulation_summary) tuples
            
        Returns:
            Dict with avg_score, min_score, max_score, pass_rate
        """
        if not pairs:
            return {"error": "No pairs provided"}
        
        scores = [self.compute_match_score(faa, sim) for faa, sim in pairs]
        
        return {
            "count": len(scores),
            "avg_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "pass_rate": sum(1 for s in scores if s >= self.threshold) / len(scores),
        }


# Convenience function
def compute_scenario_match_score(faa_description: str, simulation_summary: str) -> float:
    """
    Compute semantic similarity between FAA description and simulation summary.
    
    This is the recommended function for quick validation checks.
    
    Args:
        faa_description: Original FAA incident report text
        simulation_summary: Summary of simulation outcome
        
    Returns:
        Similarity score from 0.0 (no match) to 1.0 (perfect match)
    """
    validator = ScenarioValidator()
    return validator.compute_match_score(faa_description, simulation_summary)


# Singleton
_validator: Optional[ScenarioValidator] = None

def get_validator() -> ScenarioValidator:
    """Get singleton validator instance."""
    global _validator
    if _validator is None:
        _validator = ScenarioValidator()
    return _validator
