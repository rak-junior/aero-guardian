"""
AeroGuardian Batch Pipeline Runner
===================================
Author: AeroGuardian Member
Date: 2026-01-20

Run the automated pipeline for multiple FAA incidents in batch mode.
Generates safety reports for each incident and aggregates evaluation results.

Usage:
    python scripts/run_batch_pipeline.py                     # Run first 10 incidents
    python scripts/run_batch_pipeline.py --count 50          # Run 50 incidents
    python scripts/run_batch_pipeline.py --start 0 --count 20  # Run incidents 0-19
    python scripts/run_batch_pipeline.py --headless          # No Gazebo GUI
"""

import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.core.logging_config import get_logger
logger = get_logger("AeroGuardian.BatchPipeline")


def run_batch_pipeline(
    start_index: int = 0,
    count: int = 10,
    headless: bool = True,
    skip_px4: bool = False,
) -> Dict[str, Any]:
    """
    Run the automated pipeline for multiple incidents.
    
    Args:
        start_index: Starting incident index
        count: Number of incidents to process
        headless: Run without Gazebo GUI
        skip_px4: Skip PX4 startup (use existing instance)
        
    Returns:
        Dictionary with batch results summary
    """
    # Lazy import to avoid loading everything at startup
    from scripts.run_automated_pipeline import AutomatedPipeline, PipelineConfig
    
    logger.info("=" * 70)
    logger.info("  AEROGUARDIAN BATCH PIPELINE")
    logger.info("=" * 70)
    logger.info(f"  Start Index: {start_index}")
    logger.info(f"  Count: {count}")
    logger.info(f"  Headless Mode: {headless}")
    logger.info("=" * 70)
    
    # Initialize config
    config = PipelineConfig(headless=headless)
    
    # Track results
    results = {
        "start_time": datetime.now().isoformat(),
        "start_index": start_index,
        "count": count,
        "headless": headless,
        "incidents": [],
        "summary": {
            "total": 0,
            "success": 0,
            "failed": 0,
            "go": 0,
            "caution": 0,
            "nogo": 0,
        }
    }
    
    # Process each incident
    for i in range(start_index, start_index + count):
        incident_result = {
            "index": i,
            "incident_id": None,
            "success": False,
            "verdict": None,
            "hazard_level": None,
            "telemetry_points": 0,
            "error": None,
            "output_dir": None,
        }
        
        try:
            logger.info("")
            logger.info("=" * 70)
            logger.info(f"  PROCESSING INCIDENT {i + 1}/{start_index + count} (Index: {i})")
            logger.info("=" * 70)
            
            # Create fresh pipeline for each incident
            pipeline = AutomatedPipeline(config)
            paths = pipeline.run(incident_index=i, skip_px4=skip_px4)
            
            # Load result from generated report
            report_json = paths.get("json")
            if report_json and report_json.exists():
                with open(report_json) as f:
                    report = json.load(f)
                    incident_result["incident_id"] = report.get("incident_id")
                    incident_result["verdict"] = report.get("go_nogo_recommendation", "UNKNOWN")
                    incident_result["hazard_level"] = report.get("hazard_level", "UNKNOWN")
                    
                    # Count telemetry
                    telemetry_file = paths.get("full_telemetry")
                    if telemetry_file and telemetry_file.exists():
                        with open(telemetry_file) as tf:
                            telemetry = json.load(tf)
                            incident_result["telemetry_points"] = len(telemetry)
            
            incident_result["success"] = True
            incident_result["output_dir"] = str(paths.get("report_dir", ""))
            
            results["summary"]["success"] += 1
            
            # Count verdicts
            verdict = incident_result["verdict"].upper() if incident_result["verdict"] else "UNKNOWN"
            if verdict == "GO":
                results["summary"]["go"] += 1
            elif verdict == "CAUTION":
                results["summary"]["caution"] += 1
            elif verdict in ["NO-GO", "NOGO"]:
                results["summary"]["nogo"] += 1
                
        except Exception as e:
            logger.error(f"Failed to process incident {i}: {e}")
            incident_result["error"] = str(e)
            results["summary"]["failed"] += 1
        
        results["incidents"].append(incident_result)
        results["summary"]["total"] += 1
        
        # Brief pause between incidents
        time.sleep(2)
    
    # Finalize
    results["end_time"] = datetime.now().isoformat()
    results["duration_sec"] = (
        datetime.fromisoformat(results["end_time"]) - 
        datetime.fromisoformat(results["start_time"])
    ).total_seconds()
    
    # Calculate rates
    total = results["summary"]["total"]
    if total > 0:
        results["summary"]["success_rate"] = results["summary"]["success"] / total
        results["summary"]["go_rate"] = results["summary"]["go"] / total
        results["summary"]["caution_rate"] = results["summary"]["caution"] / total
        results["summary"]["nogo_rate"] = results["summary"]["nogo"] / total
    
    # Print summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("  BATCH PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Total Processed: {results['summary']['total']}")
    logger.info(f"  Success: {results['summary']['success']}")
    logger.info(f"  Failed: {results['summary']['failed']}")
    logger.info(f"  GO: {results['summary']['go']}")
    logger.info(f"  CAUTION: {results['summary']['caution']}")
    logger.info(f"  NO-GO: {results['summary']['nogo']}")
    logger.info(f"  Duration: {results['duration_sec']:.1f} seconds")
    logger.info("=" * 70)
    
    # Save batch results
    output_file = PROJECT_ROOT / "outputs" / f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"  Results saved to: {output_file}")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AeroGuardian Batch Pipeline Runner")
    parser.add_argument("--start", type=int, default=0, help="Starting incident index")
    parser.add_argument("--count", type=int, default=10, help="Number of incidents to process")
    parser.add_argument("--headless", action="store_true", help="Run without Gazebo GUI")
    parser.add_argument("--skip-px4", action="store_true", help="Skip PX4 startup")
    
    args = parser.parse_args()
    
    results = run_batch_pipeline(
        start_index=args.start,
        count=args.count,
        headless=args.headless,
        skip_px4=args.skip_px4,
    )
    
    return 0 if results["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    exit(main())
