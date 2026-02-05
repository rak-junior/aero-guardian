"""
Benchmark Runner Script

Run ground truth validation against ALFA and RflyMAD datasets.
Produces competition-ready quantitative metrics.

Usage:
    python scripts/run_benchmark.py                    # Full benchmark
    python scripts/run_benchmark.py --quick            # Quick test (100 flights)
    python scripts/run_benchmark.py --dataset alfa     # ALFA only
    python scripts/run_benchmark.py --dataset rflymad  # RflyMAD only
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.ground_truth_benchmark import GroundTruthBenchmark, BenchmarkConfig
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description="Run ground truth benchmark validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_benchmark.py                     # Full benchmark (all flights)
  python scripts/run_benchmark.py --quick             # Quick test (100 flights)
  python scripts/run_benchmark.py --dataset alfa      # ALFA dataset only
  python scripts/run_benchmark.py --dataset rflymad   # RflyMAD dataset only
  python scripts/run_benchmark.py --max-flights 500   # Custom flight limit
        """
    )
    
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick test mode (100 flights only)"
    )
    
    parser.add_argument(
        "--dataset", "-d",
        choices=["alfa", "rflymad", "both"],
        default="both",
        help="Dataset to benchmark (default: both)"
    )
    
    parser.add_argument(
        "--max-flights", "-m",
        type=int,
        default=None,
        help="Maximum number of flights to evaluate"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for JSON report"
    )
    
    parser.add_argument(
        "--position-threshold",
        type=float,
        default=10.0,
        help="Position drift threshold in meters (default: 10.0)"
    )
    
    parser.add_argument(
        "--roll-threshold",
        type=float,
        default=30.0,
        help="Roll angle threshold in degrees (default: 30.0)"
    )
    
    args = parser.parse_args()
    
    # Determine max flights
    max_flights = args.max_flights
    if args.quick and max_flights is None:
        max_flights = 100
    
    # Configure
    config = BenchmarkConfig(
        position_drift_threshold=args.position_threshold,
        roll_threshold=args.roll_threshold,
        pitch_threshold=args.roll_threshold  # Use same for pitch
    )
    
    print("="*60)
    print("  AEROGUARDIAN BENCHMARK RUNNER")
    print("="*60)
    print(f"\nDataset: {args.dataset.upper()}")
    print(f"Max flights: {max_flights or 'ALL'}")
    print(f"Position threshold: {config.position_drift_threshold}m")
    print(f"Attitude threshold: {config.roll_threshold}°")
    print()
    
    # Run benchmark
    benchmark = GroundTruthBenchmark(config=config)
    
    try:
        report = benchmark.run_benchmark(
            dataset=args.dataset,
            max_flights=max_flights
        )
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nMake sure ALFA and RflyMAD datasets are in:")
        print("  - data/raw/alfa/alfa_cleaned.csv")
        print("  - data/raw/rflymad/rflymad_cleaned.csv")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Benchmark failed: {e}")
        sys.exit(1)
    
    # Print summary
    benchmark.print_summary(report)
    
    # Save JSON report
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path(__file__).parent.parent / "outputs" / "benchmark"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = output_dir / f"benchmark_{args.dataset}_{timestamp}.json"
    
    report_dict = benchmark.generate_report_json(report, output_path)
    
    # Print key metrics for copy-paste
    print("\n" + "="*60)
    print("  COPY-PASTE METRICS (for demo/presentation)")
    print("="*60)
    print(f"""
┌────────────────────────────────────────────────┐
│ Ground Truth Benchmark Results                 │
│ ─────────────────────────────                  │
│ Dataset: {args.dataset.upper():40s}│
│ Flights: {report.total_flights:40d}│
│                                                │
│ Detection Rate:       {report.detection_rate*100:5.1f}%                   │
│ False Positive Rate:  {report.false_positive_rate*100:5.1f}%                   │
│ Precision:            {report.precision*100:5.1f}%                   │
│ F1 Score:             {report.f1_score*100:5.1f}%                   │
│ Attribution Accuracy: {report.attribution_accuracy*100:5.1f}%                   │
│                                                │
│ Mean Onset Delay:     {report.mean_onset_delay:.3f}s                   │
│ Processing Speed:     {report.total_flights/report.processing_time_sec:.1f} flights/sec          │
└────────────────────────────────────────────────┘
""")
    
    # Return exit code based on detection rate
    if report.detection_rate < 0.5:
        print("\n[WARNING] Detection rate below 50% - review thresholds")
        sys.exit(2)
    elif report.detection_rate < 0.7:
        print("\n[INFO] Detection rate moderate - consider threshold tuning")
    else:
        print("\n[SUCCESS] Detection rate good!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
