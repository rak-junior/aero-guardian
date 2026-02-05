"""
FAA Data Processing Pipeline
============================
Author: AeroGuardian
Date: 2026-01-28
Updated: 2026-01-28

Reads raw FAA Excel files and converts them to simulatable JSON format.

Classification:
- ACTUAL_FAILURE: Real drone malfunctions (crash, lost control, flyaway, malfunction)
- HIGH_RISK_SIGHTING: Sightings in dangerous situations (simulated as potential failures)
- NORMAL_SIGHTING: Standard sightings (excluded from simulatable data)

Output: Only ACTUAL_FAILURE + HIGH_RISK_SIGHTING are included in simulatable.json

Usage:
    python process_faa_data.py
    python process_faa_data.py --input data/raw/faa --output data/processed/faa_reports
    python process_faa_data.py --exclude-high-risk
"""

import argparse
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# =============================================================================
# Logging Configuration
# =============================================================================

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the module."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class FAAConfig:
    """Configuration for FAA data processing pipeline.
    
    Attributes:
        raw_data_dir: Directory containing raw FAA Excel files.
        output_dir: Directory for processed JSON output.
        include_high_risk: Whether to include HIGH_RISK_SIGHTING in output.
        verbose: Enable debug logging.
    """
    raw_data_dir: Path = field(default_factory=lambda: Path("data/raw/faa"))
    output_dir: Path = field(default_factory=lambda: Path("data/processed/faa_reports"))
    include_high_risk: bool = True
    verbose: bool = False


@dataclass(frozen=True)
class FaultTypeInfo:
    """Information about a PX4-simulatable fault type."""
    hazard_category: str
    px4_fault_type: str
    hazard_description: str
    default_severity: float


# =============================================================================
# Constants
# =============================================================================

FAULT_TYPES: Dict[str, FaultTypeInfo] = {
    "gps_loss": FaultTypeInfo(
        hazard_category="NAVIGATION",
        px4_fault_type="gps_dropout",
        hazard_description="GPS signal loss or position hold failure",
        default_severity=0.6,
    ),
    "motor_failure": FaultTypeInfo(
        hazard_category="PROPULSION",
        px4_fault_type="motor_degradation",
        hazard_description="Motor/propeller/ESC failure causing thrust loss",
        default_severity=0.7,
    ),
    "battery_failure": FaultTypeInfo(
        hazard_category="POWER",
        px4_fault_type="battery_low",
        hazard_description="Battery failure or low voltage causing power loss",
        default_severity=0.8,
    ),
    "control_loss": FaultTypeInfo(
        hazard_category="CONTROL",
        px4_fault_type="rc_loss",
        hazard_description="Control link loss or RC failsafe activation",
        default_severity=0.5,
    ),
    "sensor_fault": FaultTypeInfo(
        hazard_category="SENSOR",
        px4_fault_type="sensor_failure",
        hazard_description="IMU/compass/barometer sensor failure",
        default_severity=0.6,
    ),
}


# =============================================================================
# Incident Classifier
# =============================================================================

class FAAClassifier:
    """Classifies FAA incidents into safety categories.
    
    This classifier analyzes incident descriptions to determine:
    - Whether it's an actual drone failure or just a sighting
    - The type of fault for PX4 simulation
    - Confidence level of classification
    
    Example:
        >>> classifier = FAAClassifier()
        >>> classification, fault_type, confidence = classifier.classify(summary)
    """
    
    # Keywords indicating real drone malfunction
    ACTUAL_FAILURE_KEYWORDS: List[str] = [
        "malfunction", "malfunctioned",
        "crash", "crashed", "crashing",
        "fell", "fall", "fallen", "falling",
        "went down", "going down",
        "lost control", "out of control",
        "fly away", "flyaway", "flew away", "flown away",
        "runaway", "run away",
        "emergency landing", "forced landing",
        "chute deployed", "parachute deployed",
        "struck", "impacted", "hit the ground",
        "operator stated", "operator reported",
        "lost power", "power loss",
        "low battery", "dead battery",
        "motor stopped", "motor failed",
        "propeller broke", "prop failure",
        "gps malfunction", "gps failure",
        "lost signal", "signal lost", "lost link",
        "unresponsive", "uncommanded",
        "spun out", "spinning", "tumbling",
        "descended rapidly", "rapid descent",
    ]
    
    # Keywords indicating dangerous situation
    HIGH_RISK_KEYWORDS: List[str] = [
        "evasive action taken",
        "near miss", "close call",
        "within 50 feet", "within 100 feet", "within 200 feet",
        "passed directly", "directly overhead", "directly below",
        "same altitude", "at same alt",
        "runway", "final approach", "departure end",
        "ascending", "descending through",
        "fl", "flight level",
        "emergency", "declared emergency",
        "suspended operations", "held departures",
        "traffic pattern", "on approach", "on final",
        "laser", "interference",
    ]
    
    # Keywords indicating normal sighting
    SIGHTING_ONLY_KEYWORDS: List[str] = [
        "no evasive action taken",
        "no evasive action reported",
        "evasive action not reported",
        "no impact to operations",
        "no further information",
    ]
    
    def classify(self, summary: str) -> Tuple[str, str, float]:
        """Classify an incident into categories.
        
        Args:
            summary: Incident description text.
            
        Returns:
            Tuple of (classification, fault_type, confidence):
            - classification: ACTUAL_FAILURE, HIGH_RISK_SIGHTING, or NORMAL_SIGHTING
            - fault_type: gps_loss, motor_failure, etc.
            - confidence: 0.0-1.0
        """
        text_lower = summary.lower()
        
        # Count matching keywords
        actual_failure_count = sum(
            1 for kw in self.ACTUAL_FAILURE_KEYWORDS if kw in text_lower
        )
        high_risk_count = sum(
            1 for kw in self.HIGH_RISK_KEYWORDS if kw in text_lower
        )
        
        # Check for operator-reported issues (strongest signal)
        if "operator stated" in text_lower or "operator reported" in text_lower:
            return "ACTUAL_FAILURE", self.detect_fault_type(summary), 0.95
        
        # Check for clear malfunction keywords
        if any(kw in text_lower for kw in ["malfunctioned", "malfunction"]):
            return "ACTUAL_FAILURE", self.detect_fault_type(summary), 0.9
        
        # Check for crash/impact
        if any(kw in text_lower for kw in ["crashed", "struck", "impacted", "hit the ground"]):
            return "ACTUAL_FAILURE", self.detect_fault_type(summary), 0.85
        
        # Check for loss of control
        if "lost control" in text_lower or "out of control" in text_lower:
            return "ACTUAL_FAILURE", self.detect_fault_type(summary), 0.85
        
        # Check for flyaway
        if any(kw in text_lower for kw in ["fly away", "flyaway", "flew away", "runaway"]):
            return "ACTUAL_FAILURE", self.detect_fault_type(summary), 0.85
        
        # Check for chute/parachute deployment (emergency)
        if "chute deployed" in text_lower or "parachute" in text_lower:
            return "ACTUAL_FAILURE", self.detect_fault_type(summary), 0.9
        
        # Check for actual failures with lower confidence
        if actual_failure_count >= 2:
            return "ACTUAL_FAILURE", self.detect_fault_type(summary), 0.7
        
        # HIGH_RISK: Flight level or very high altitude
        altitude = self.parse_altitude(summary)
        if altitude and altitude > 1500:  # Above 5000 feet (FL50)
            return "HIGH_RISK_SIGHTING", self.detect_fault_type(summary), 0.8
        
        # HIGH_RISK: Evasive action taken
        if "evasive action taken" in text_lower:
            return "HIGH_RISK_SIGHTING", self.detect_fault_type(summary), 0.75
        
        # HIGH_RISK: Near runway or on approach
        if any(kw in text_lower for kw in ["runway", "final approach", "on final", "departure end"]):
            return "HIGH_RISK_SIGHTING", self.detect_fault_type(summary), 0.7
        
        # HIGH_RISK: Very close encounter
        if any(kw in text_lower for kw in ["within 50 feet", "within 100 feet", "passed directly", "same altitude"]):
            return "HIGH_RISK_SIGHTING", self.detect_fault_type(summary), 0.7
        
        # HIGH_RISK: Moderate altitude near aircraft
        if altitude and altitude > 600:  # Above 2000 feet
            return "HIGH_RISK_SIGHTING", self.detect_fault_type(summary), 0.65
        
        # Check for high risk keywords
        if high_risk_count >= 2:
            return "HIGH_RISK_SIGHTING", self.detect_fault_type(summary), 0.6
        
        # Normal sighting
        return "NORMAL_SIGHTING", "none", 0.5
    
    def detect_fault_type(self, text: str) -> str:
        """Detect the fault type from incident description.
        
        Args:
            text: Incident description text.
            
        Returns:
            Fault type string for PX4 simulation.
        """
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in ["gps", "navigation", "position", "drift"]):
            return "gps_loss"
        
        if any(kw in text_lower for kw in ["motor", "propeller", "prop", "esc", "thrust"]):
            return "motor_failure"
        
        if any(kw in text_lower for kw in ["battery", "power", "voltage", "charge"]):
            return "battery_failure"
        
        if any(kw in text_lower for kw in ["control", "link", "signal", "rc", "command"]):
            return "control_loss"
        
        if any(kw in text_lower for kw in ["sensor", "compass", "imu", "gyro", "barometer"]):
            return "sensor_fault"
        
        # High altitude sightings -> GPS loss (simulate navigation issue)
        altitude = self.parse_altitude(text)
        if altitude and altitude > 300:  # Above 300m (1000ft)
            return "gps_loss"
        
        # Default: GPS loss is the most common and safest to simulate
        return "gps_loss"
    
    def parse_altitude(self, text: str) -> Optional[float]:
        """Parse altitude from FAA text and convert to meters.
        
        Args:
            text: Text containing altitude information.
            
        Returns:
            Altitude in meters, or None if not found.
        """
        if not text:
            return None
        
        text_upper = text.upper()
        
        # Flight Level: FL210 = 21,000 feet
        fl_match = re.search(r'FL\s*(\d{2,3})', text_upper)
        if fl_match:
            fl_value = int(fl_match.group(1))
            feet = fl_value * 100
            return feet * 0.3048
        
        # Feet patterns: "5000 feet", "500 FT", "AT 4,300 FEET"
        ft_patterns = [
            r'AT\s+([\d,]+)\s*(?:FEET|FT)',
            r'([\d,]+)\s*(?:FEET|FT)',
            r'ALTITUDE\s+(?:OF\s+)?([\d,]+)',
        ]
        for pattern in ft_patterns:
            match = re.search(pattern, text_upper)
            if match:
                feet_str = match.group(1).replace(',', '')
                try:
                    feet = int(feet_str)
                    return feet * 0.3048
                except ValueError:
                    continue
        
        return None


# =============================================================================
# Data Processor
# =============================================================================

class FAAProcessor:
    """Processes FAA Excel files into simulatable JSON format.
    
    This processor reads raw FAA incident data, classifies each incident,
    and outputs filtered JSON files suitable for PX4 simulation.
    
    Example:
        >>> config = FAAConfig()
        >>> processor = FAAProcessor(config)
        >>> processor.run()
    """
    
    def __init__(self, config: FAAConfig) -> None:
        """Initialize the processor.
        
        Args:
            config: Processing configuration.
        """
        self.config = config
        self.classifier = FAAClassifier()
        self.stats: Dict[str, Any] = {
            "total_raw": 0,
            "ACTUAL_FAILURE": 0,
            "HIGH_RISK_SIGHTING": 0,
            "NORMAL_SIGHTING": 0,
            "by_fault_type": Counter(),
            "by_file": Counter(),
        }
    
    def run(self) -> List[Dict[str, Any]]:
        """Run the full processing pipeline.
        
        Returns:
            List of processed incident records.
        """
        logger.info("=" * 60)
        logger.info("FAA DATA PROCESSING PIPELINE")
        logger.info("=" * 60)
        
        # Find all Excel files
        excel_files = self._find_excel_files()
        logger.info(f"Found {len(excel_files)} Excel files in {self.config.raw_data_dir}")
        
        # Process each file
        all_incidents: List[Dict[str, Any]] = []
        for excel_file in sorted(excel_files):
            incidents = self._process_file(excel_file)
            all_incidents.extend(incidents)
        
        # Sort: ACTUAL_FAILURE first, then HIGH_RISK_SIGHTING
        # Sort: ACTUAL_FAILURE first, then HIGH_RISK_SIGHTING
        all_incidents.sort(
            key=lambda x: (0 if x.get("classification", "ACTUAL_FAILURE") == "ACTUAL_FAILURE" else 1, x["report_id"])
        )
        
        # Save output
        self._save_output(all_incidents)
        self._print_summary(all_incidents)
        
        return all_incidents
    
    def _find_excel_files(self) -> List[Path]:
        """Find all Excel files in the raw data directory.
        
        Returns:
            List of Excel file paths.
        """
        files = list(self.config.raw_data_dir.glob("*.xlsx"))
        # Skip temp files
        return [f for f in files if not f.name.startswith("~$")]
    
    def _process_file(self, filepath: Path) -> List[Dict[str, Any]]:
        """Process a single Excel file.
        
        Args:
            filepath: Path to the Excel file.
            
        Returns:
            List of incident records from this file.
        """
        logger.info(f"Processing: {filepath.name}")
        
        try:
            df = pd.read_excel(filepath)
        except Exception as e:
            logger.error(f"  ERROR reading file: {e}")
            return []
        
        # Normalize column names
        df.columns = [col.strip() for col in df.columns]
        
        # Find the summary column
        summary_col = self._find_summary_column(df)
        if not summary_col:
            logger.warning(f"  No summary column found. Columns: {list(df.columns)}")
            return []
        
        # Find date column
        date_col = self._find_date_column(df)
        
        # Process each row
        incidents: List[Dict[str, Any]] = []
        for idx, row in df.iterrows():
            self.stats["total_raw"] += 1
            
            summary = str(row[summary_col]) if pd.notna(row[summary_col]) else ""
            if not summary or len(summary) < 10:
                continue
            
            # Classify the incident
            classification, fault_type, confidence = self.classifier.classify(summary)
            self.stats[classification] += 1
            
            # Skip NORMAL_SIGHTING
            if classification == "NORMAL_SIGHTING":
                continue
            
            # Skip HIGH_RISK if not included
            if classification == "HIGH_RISK_SIGHTING" and not self.config.include_high_risk:
                continue
            
            # Create incident record
            incident = self._create_incident_record(
                row=row,
                idx=idx,
                filepath=filepath,
                summary=summary,
                summary_col=summary_col,
                date_col=date_col,
                classification=classification,
                fault_type=fault_type,
                confidence=confidence,
            )
            
            incidents.append(incident)
            self.stats["by_fault_type"][fault_type] += 1
            self.stats["by_file"][filepath.name] += 1
        
        logger.info(f"  → Extracted {len(incidents)} simulatable incidents")
        return incidents
    
    def _find_summary_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the summary/description column in the dataframe."""
        for col in df.columns:
            if col.lower() in ['summary', 'description', 'narrative']:
                return col
        return None
    
    def _find_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the date column in the dataframe."""
        for col in df.columns:
            if 'date' in col.lower() or 'sighting' in col.lower():
                return col
        return None
    
    def _create_incident_record(
        self,
        row: pd.Series,
        idx: int,
        filepath: Path,
        summary: str,
        summary_col: str,
        date_col: Optional[str],
        classification: str,
        fault_type: str,
        confidence: float,
    ) -> Dict[str, Any]:
        """Create an incident record dictionary.
        
        Args:
            row: DataFrame row.
            idx: Row index.
            filepath: Source file path.
            summary: Incident summary text.
            summary_col: Name of summary column.
            date_col: Name of date column (if found).
            classification: Incident classification.
            fault_type: Detected fault type.
            confidence: Classification confidence.
            
        Returns:
            Incident record dictionary.
        """
        # Extract location
        city = str(row.get('City', '')).strip() if pd.notna(row.get('City')) else ''
        state = str(row.get('State', '')).strip() if pd.notna(row.get('State')) else ''
        
        # Extract date
        date_str = ""
        if date_col:
            date_val = row.get(date_col, '')
            date_str = self._format_date(date_val)
        
        # Get fault type metadata
        fault_info = FAULT_TYPES.get(fault_type, FAULT_TYPES["gps_loss"])
        
        # Extract altitude
        altitude = self.classifier.parse_altitude(summary)
        
        # Create record - IMPORTANT: Preserve ALL classification metadata
        report_id = f"{filepath.stem}_{idx + 1}"
        return {
            "report_id": report_id,
            "date": date_str,
            "city": city.upper(),
            "state": state.upper(),
            "description": summary,
            # Classification metadata (CRITICAL: Do not discard these!)
            "classification": classification,  # ACTUAL_FAILURE or HIGH_RISK_SIGHTING
            "fault_type": fault_type,  # LLM can override, but provides initial hint
            "classification_confidence": confidence,
            "altitude_m": altitude if altitude else None,
            # Fault type metadata from FAULT_TYPES dict
            "hazard_category": fault_info.hazard_category,
            "hazard_description": fault_info.hazard_description,
        }
    
    def _format_date(self, date_val: Any) -> str:
        """Format date value to ISO string."""
        if pd.isna(date_val):
            return ""
        try:
            if isinstance(date_val, datetime):
                return date_val.isoformat()
            return str(date_val)
        except Exception:
            return ""
    
    def _save_output(self, incidents: List[Dict[str, Any]]) -> None:
        """Save processed incidents to JSON files.
        
        Args:
            incidents: List of incident records.
        """
        logger.info("")
        logger.info("=" * 60)
        logger.info("GENERATING OUTPUT FILES")
        logger.info("=" * 60)
        
        # 1. Full incidents file
        full_output = {
            "version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "description": "FAA incidents processed for AeroGuardian pre-flight safety analysis",
            "statistics": {
                "total_raw_records": self.stats["total_raw"],
                "simulatable_total": len(incidents),
            },
            "incidents": incidents,
        }
        
        full_path = self.config.output_dir / "faa_reports.json"
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(full_output, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved: {full_path} ({len(incidents)} incidents)")
        
        # 2. Simulatable incidents file
        simulatable_output = {
            "version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "description": "FAA incidents filtered for PX4 SITL simulation",
            "original_total": self.stats["total_raw"],
            "simulatable_total": len(incidents),
            "filter_rate": f"{len(incidents) / max(self.stats['total_raw'], 1) * 100:.1f}%",
            "incidents": incidents,
        }
        
        sim_path = self.config.output_dir / "faa_simulatable.json"
        with open(sim_path, 'w', encoding='utf-8') as f:
            json.dump(simulatable_output, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved: {sim_path} ({len(incidents)} incidents)")
        
        # 3. CRITICAL: Actual failures ONLY file (for competition demos)
        # These are the 31 cases with REAL drone malfunctions - highest credibility
        actual_failures = [i for i in incidents if i.get("classification") == "ACTUAL_FAILURE"]
        actual_output = {
            "version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "description": "FAA incidents with CONFIRMED drone malfunctions (crashes, flyaways, malfunctions) - highest credibility subset",
            "data_quality_note": "These cases contain explicit evidence of drone failure (operator stated, crashed, malfunctioned, etc.) unlike HIGH_RISK_SIGHTING which are just altitude/proximity sightings.",
            "competition_note": "Use this dataset for demos - these are real failures, not hypothetical scenarios.",
            "total_failures": len(actual_failures),
            "incidents": actual_failures,
        }
        
        actual_path = self.config.output_dir / "faa_actual_failures.json"
        with open(actual_path, 'w', encoding='utf-8') as f:
            json.dump(actual_output, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved: {actual_path} ({len(actual_failures)} ACTUAL_FAILURE incidents)")
        
        # 4. High-risk sightings file (hypothetical scenarios)
        high_risk = [i for i in incidents if i.get("classification") == "HIGH_RISK_SIGHTING"]
        high_risk_output = {
            "version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "description": "FAA sightings in high-risk situations (altitude/proximity) - hypothetical failure scenarios",
            "data_quality_note": "These are NOT confirmed failures. They are sightings where a drone was observed in a situation that COULD indicate a problem (high altitude, near runway, etc.) but likely was just normal operation.",
            "total_sightings": len(high_risk),
            "incidents": high_risk,
        }
        
        high_risk_path = self.config.output_dir / "faa_high_risk_sightings.json"
        with open(high_risk_path, 'w', encoding='utf-8') as f:
            json.dump(high_risk_output, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved: {high_risk_path} ({len(high_risk)} HIGH_RISK_SIGHTING incidents)")

    def _print_summary(self, incidents: List[Dict[str, Any]]) -> None:
        """Print summary statistics."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total raw records: {self.stats['total_raw']}")
        logger.info(f"ACTUAL_FAILURE: {self.stats['ACTUAL_FAILURE']}")
        logger.info(f"HIGH_RISK_SIGHTING: {self.stats['HIGH_RISK_SIGHTING']}")
        logger.info(f"NORMAL_SIGHTING: {self.stats['NORMAL_SIGHTING']}")
        logger.info(f"Simulatable total: {len(incidents)}")
        
        logger.info("By fault type (Internal Classification):")
        for ftype, count in self.stats["by_fault_type"].most_common():
            logger.info(f"  {ftype}: {count}")
        
        logger.info("✓ Data ready for AeroGuardian pipeline!")
    
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total raw records: {self.stats['total_raw']}")
        logger.info(f"ACTUAL_FAILURE: {self.stats['ACTUAL_FAILURE']}")
        logger.info(f"HIGH_RISK_SIGHTING: {self.stats['HIGH_RISK_SIGHTING']}")
        logger.info(f"NORMAL_SIGHTING: {self.stats['NORMAL_SIGHTING']}")
        logger.info(f"Simulatable total: {len(incidents)}")
        
        logger.info("By fault type:")
        for ftype, count in self.stats["by_fault_type"].most_common():
            logger.info(f"  {ftype}: {count}")
        
        logger.info("✓ Data ready for AeroGuardian pipeline!")


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="FAA Data Processing Pipeline for AeroGuardian",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python process_faa_data.py
    python process_faa_data.py --input data/raw/faa --output data/processed
    python process_faa_data.py --exclude-high-risk
    python process_faa_data.py --verbose
        """,
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("data/raw/faa"),
        help="Input directory containing FAA Excel files (default: data/raw/faa)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/processed/faa_reports"),
        help="Output directory for JSON files (default: data/processed/faa_reports)",
    )
    parser.add_argument(
        "--exclude-high-risk",
        action="store_true",
        help="Exclude HIGH_RISK_SIGHTING incidents from output",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose debug logging",
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point.
    
    Returns:
        Exit code (0 for success, 1 for error).
    """
    args = parse_args()
    setup_logging(args.verbose)
    
    config = FAAConfig(
        raw_data_dir=args.input,
        output_dir=args.output,
        include_high_risk=not args.exclude_high_risk,
        verbose=args.verbose,
    )
    
    try:
        processor = FAAProcessor(config)
        processor.run()
        return 0
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
