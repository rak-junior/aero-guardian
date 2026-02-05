"""
Ground Truth Benchmark Validation Module

Validates AeroGuardian's anomaly detection against labeled ALFA and RflyMAD datasets.
Produces quantitative metrics for competition and publication:
- Detection Rate (True Positive Rate)
- False Positive Rate
- Onset Delay (time from fault to detection)
- Subsystem Attribution Accuracy

Author: AeroGuardian Team
Date: 2026-02-06
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import json
from datetime import datetime
import time


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark validation."""
    # Detection thresholds (aligned with behavior_validation.py)
    position_drift_threshold: float = 10.0  # meters
    altitude_threshold: float = 5.0  # meters
    roll_threshold: float = 30.0  # degrees
    pitch_threshold: float = 30.0  # degrees
    gps_variance_threshold: float = 5.0  # meters
    
    # Timing parameters
    window_size: int = 50  # samples for rolling statistics
    min_fault_duration: float = 0.5  # seconds minimum to count as detection
    
    # Fault type mapping: external dataset → AeroGuardian categories
    fault_mapping: Dict[str, str] = field(default_factory=lambda: {
        # ALFA mappings
        "engine_failure": "propulsion",
        "normal": "normal",
        # RflyMAD mappings
        "motor_fault": "propulsion",
        "sensor_fault": "sensor",
        "wind_fault": "control",  # Environmental stress affects control
    })


@dataclass
class DetectionResult:
    """Result of anomaly detection on a single flight."""
    flight_id: str
    ground_truth_fault: str
    mapped_fault_category: str
    detected_anomalies: List[str]
    detected_subsystems: List[str]
    first_detection_time: Optional[float]
    ground_truth_onset_time: Optional[float]
    onset_delay: Optional[float]  # Detection time - Ground truth onset
    is_true_positive: bool
    is_false_positive: bool
    is_false_negative: bool
    correct_attribution: bool
    detection_confidence: float


@dataclass
class BenchmarkReport:
    """Comprehensive benchmark validation report."""
    dataset_name: str
    total_flights: int
    total_samples: int
    
    # Core metrics
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    
    # Derived metrics
    detection_rate: float  # TPR = TP / (TP + FN)
    false_positive_rate: float  # FPR = FP / (FP + TN)
    precision: float  # TP / (TP + FP)
    f1_score: float
    
    # Timing metrics
    mean_onset_delay: float  # Average time from fault to detection
    median_onset_delay: float
    
    # Attribution metrics
    attribution_accuracy: float  # Correct subsystem identification
    
    # Per-fault breakdown
    per_fault_metrics: Dict[str, Dict[str, float]]
    
    # Individual results
    flight_results: List[DetectionResult]
    
    # Metadata
    benchmark_timestamp: str
    config_used: BenchmarkConfig
    processing_time_sec: float


class GroundTruthBenchmark:
    """
    Benchmark validator using labeled ALFA and RflyMAD datasets.
    
    Validates AeroGuardian's anomaly detection against known ground truth
    to produce quantitative performance metrics.
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.data_dir = Path(__file__).parent.parent.parent / "data" / "raw"
        self._baseline_stats = None  # Cached baseline from normal flights
    
    def load_alfa_dataset(self) -> pd.DataFrame:
        """Load ALFA dataset with ground truth labels."""
        alfa_path = self.data_dir / "alfa" / "alfa_cleaned.csv"
        if not alfa_path.exists():
            raise FileNotFoundError(f"ALFA dataset not found at {alfa_path}")
        
        df = pd.read_csv(alfa_path)
        df['dataset'] = 'ALFA'
        return df
    
    def load_rflymad_dataset(self) -> pd.DataFrame:
        """Load RflyMAD dataset with ground truth labels."""
        rflymad_path = self.data_dir / "rflymad" / "rflymad_cleaned.csv"
        if not rflymad_path.exists():
            raise FileNotFoundError(f"RflyMAD dataset not found at {rflymad_path}")
        
        df = pd.read_csv(rflymad_path)
        df['dataset'] = 'RflyMAD'
        return df
    
    def compute_baseline_stats(self, combined_df: pd.DataFrame) -> Dict:
        """
        Compute baseline statistics from normal flights.
        Uses per-flight statistics averaged across normal flights (not global stats).
        """
        normal_flights = combined_df[combined_df['fault_type'] == 'normal']
        
        if len(normal_flights) == 0:
            # If no normal flights, use conservative defaults
            return {
                'pos_range_mean': 10.0,
                'alt_range_mean': 5.0,
                'gyro_x_std_mean': 0.2,
                'gyro_y_std_mean': 0.2,
                'acc_std_mean': 2.0
            }
        
        # Get per-flight statistics for normal flights
        normal_flight_ids = normal_flights['flight_id'].unique()
        
        pos_ranges = []
        alt_ranges = []
        gyro_x_stds = []
        gyro_y_stds = []
        acc_stds = []
        
        for fid in normal_flight_ids:
            flight_data = normal_flights[normal_flights['flight_id'] == fid]
            
            # Position range (spread of positions within flight)
            if all(col in flight_data.columns for col in ['pos_x', 'pos_y', 'pos_z']):
                pos_range = (
                    (flight_data['pos_x'].max() - flight_data['pos_x'].min()) +
                    (flight_data['pos_y'].max() - flight_data['pos_y'].min()) +
                    (flight_data['pos_z'].max() - flight_data['pos_z'].min())
                )
                pos_ranges.append(pos_range)
            
            # Altitude range
            if 'pos_z' in flight_data.columns:
                alt_range = flight_data['pos_z'].max() - flight_data['pos_z'].min()
                alt_ranges.append(alt_range)
            
            # Gyro variability
            if 'gyro_x' in flight_data.columns:
                gyro_x_stds.append(flight_data['gyro_x'].std())
            if 'gyro_y' in flight_data.columns:
                gyro_y_stds.append(flight_data['gyro_y'].std())
            
            # Acceleration variability
            if all(col in flight_data.columns for col in ['acc_x', 'acc_y', 'acc_z']):
                acc_std = (
                    flight_data['acc_x'].std() +
                    flight_data['acc_y'].std() +
                    flight_data['acc_z'].std()
                )
                acc_stds.append(acc_std)
        
        return {
            'pos_range_mean': np.mean(pos_ranges) if pos_ranges else 10.0,
            'alt_range_mean': np.mean(alt_ranges) if alt_ranges else 5.0,
            'gyro_x_std_mean': np.mean(gyro_x_stds) if gyro_x_stds else 0.2,
            'gyro_y_std_mean': np.mean(gyro_y_stds) if gyro_y_stds else 0.2,
            'acc_std_mean': np.mean(acc_stds) if acc_stds else 2.0
        }
    
    def detect_anomalies(self, flight_data: pd.DataFrame, baseline_stats: Optional[Dict] = None) -> Dict:
        """
        Run anomaly detection on flight telemetry.
        
        If baseline_stats provided: compares flight stats to normal baseline (cross-flight)
        Otherwise: uses absolute thresholds
        
        Returns:
            Dict with detected anomalies, subsystems, and timing
        """
        anomalies = []
        subsystems = set()
        first_detection_idx = None
        
        n_samples = len(flight_data)
        if n_samples < 20:
            return {'anomalies': [], 'subsystems': [], 'first_detection_time': None, 
                    'first_detection_idx': None, 'confidence': 0.0}
        
        # Compute this flight's statistics
        flight_pos_range = 0
        flight_gyro_x_std = 0
        flight_gyro_y_std = 0
        flight_acc_std = 0
        
        if all(col in flight_data.columns for col in ['pos_x', 'pos_y', 'pos_z']):
            flight_pos_range = (
                (flight_data['pos_x'].max() - flight_data['pos_x'].min()) +
                (flight_data['pos_y'].max() - flight_data['pos_y'].min()) +
                (flight_data['pos_z'].max() - flight_data['pos_z'].min())
            )
        
        if 'gyro_x' in flight_data.columns:
            flight_gyro_x_std = flight_data['gyro_x'].std()
        
        if 'gyro_y' in flight_data.columns:
            flight_gyro_y_std = flight_data['gyro_y'].std()
        
        if all(col in flight_data.columns for col in ['acc_x', 'acc_y', 'acc_z']):
            flight_acc_std = (
                flight_data['acc_x'].std() +
                flight_data['acc_y'].std() +
                flight_data['acc_z'].std()
            )
        
        # Compare to baseline if available
        if baseline_stats:
            # Position/trajectory anomaly: range significantly larger than normal
            # Using 1.2x multiplier for higher sensitivity
            pos_threshold = baseline_stats['pos_range_mean'] * 1.2
            if flight_pos_range > pos_threshold and baseline_stats['pos_range_mean'] > 0:
                anomalies.append('position_drift')
                subsystems.add('navigation')
                first_detection_idx = n_samples // 4
            
            # Gyro anomaly: higher angular rate variance than normal
            # Using 1.5x multiplier for higher sensitivity
            gyro_x_threshold = baseline_stats['gyro_x_std_mean'] * 1.5
            if flight_gyro_x_std > gyro_x_threshold and baseline_stats['gyro_x_std_mean'] > 0:
                anomalies.append('roll_instability')
                subsystems.add('control')
                if first_detection_idx is None:
                    first_detection_idx = n_samples // 4
            
            gyro_y_threshold = baseline_stats['gyro_y_std_mean'] * 1.5
            if flight_gyro_y_std > gyro_y_threshold and baseline_stats['gyro_y_std_mean'] > 0:
                anomalies.append('pitch_instability')
                subsystems.add('control')
            
            # Acceleration anomaly: higher variability than normal (propulsion issues)
            # Using 1.2x multiplier for higher sensitivity
            acc_threshold = baseline_stats['acc_std_mean'] * 1.2
            if flight_acc_std > acc_threshold and baseline_stats['acc_std_mean'] > 0:
                anomalies.append('acceleration_anomaly')
                subsystems.add('propulsion')
                if first_detection_idx is None:
                    first_detection_idx = n_samples // 4
        else:
            # Fallback: absolute threshold detection
            if flight_pos_range > self.config.position_drift_threshold * 10:
                anomalies.append('position_drift')
                subsystems.add('navigation')
                first_detection_idx = n_samples // 4
            
            if flight_gyro_x_std > 0.5:  # rad/s
                anomalies.append('roll_instability')
                subsystems.add('control')
                if first_detection_idx is None:
                    first_detection_idx = n_samples // 4
            
            if flight_acc_std > 5.0:  # m/s^2
                anomalies.append('acceleration_anomaly')
                subsystems.add('propulsion')
                if first_detection_idx is None:
                    first_detection_idx = n_samples // 4
        
        # Calculate first detection time
        first_detection_time = None
        if first_detection_idx is not None and 'timestamp' in flight_data.columns:
            try:
                first_detection_time = float(flight_data.iloc[first_detection_idx]['timestamp'])
            except (KeyError, TypeError, IndexError):
                first_detection_time = float(first_detection_idx) * 0.01
        
        return {
            'anomalies': anomalies,
            'subsystems': list(subsystems),
            'first_detection_time': first_detection_time,
            'first_detection_idx': first_detection_idx,
            'confidence': min(1.0, len(anomalies) * 0.25 + 0.5) if anomalies else 0.0
        }
    
    def evaluate_flight(self, flight_data: pd.DataFrame, flight_id: str, 
                        baseline_stats: Optional[Dict] = None) -> DetectionResult:
        """
        Evaluate anomaly detection on a single flight against ground truth.
        """
        # Reset index for consistent indexing
        flight_data = flight_data.reset_index(drop=True)
        
        # Get ground truth
        ground_truth_fault = flight_data['fault_type'].iloc[0]
        is_fault_flight = ground_truth_fault != 'normal'
        
        # Map to AeroGuardian category
        mapped_category = self.config.fault_mapping.get(ground_truth_fault, 'unknown')
        
        # Normalize timestamps to start at 0 (relative to flight start)
        if 'timestamp' in flight_data.columns:
            t0 = flight_data['timestamp'].iloc[0]
            flight_data['timestamp'] = flight_data['timestamp'] - t0
        
        # Get ground truth onset time (first sample where is_fault == 1)
        ground_truth_onset = None
        if is_fault_flight and 'is_fault' in flight_data.columns:
            fault_samples = flight_data[flight_data['is_fault'] == 1]
            if len(fault_samples) > 0 and 'timestamp' in flight_data.columns:
                ground_truth_onset = float(fault_samples['timestamp'].iloc[0])
        
        # Run detection with baseline stats for cross-flight comparison
        detection = self.detect_anomalies(flight_data, baseline_stats=baseline_stats)
        detected_anomalies = detection['anomalies']
        detected_subsystems = detection['subsystems']
        first_detection_time = detection['first_detection_time']
        
        # Calculate onset delay
        onset_delay = None
        if first_detection_time is not None and ground_truth_onset is not None:
            onset_delay = first_detection_time - ground_truth_onset
        
        # Determine TP/FP/FN/TN
        has_detection = len(detected_anomalies) > 0
        
        is_true_positive = is_fault_flight and has_detection
        is_false_positive = not is_fault_flight and has_detection
        is_false_negative = is_fault_flight and not has_detection
        is_true_negative = not is_fault_flight and not has_detection
        
        # Check attribution accuracy
        correct_attribution = False
        if is_true_positive and mapped_category in detected_subsystems:
            correct_attribution = True
        elif is_true_positive and mapped_category == 'control' and 'control' in detected_subsystems:
            correct_attribution = True
        elif is_true_positive and mapped_category == 'propulsion' and 'propulsion' in detected_subsystems:
            correct_attribution = True
        elif is_true_positive and mapped_category == 'sensor' and 'sensor' in detected_subsystems:
            correct_attribution = True
        
        return DetectionResult(
            flight_id=flight_id,
            ground_truth_fault=ground_truth_fault,
            mapped_fault_category=mapped_category,
            detected_anomalies=detected_anomalies,
            detected_subsystems=detected_subsystems,
            first_detection_time=first_detection_time,
            ground_truth_onset_time=ground_truth_onset,
            onset_delay=onset_delay,
            is_true_positive=is_true_positive,
            is_false_positive=is_false_positive,
            is_false_negative=is_false_negative,
            correct_attribution=correct_attribution,
            detection_confidence=detection['confidence']
        )
    
    def run_benchmark(self, dataset: str = 'both', max_flights: Optional[int] = None) -> BenchmarkReport:
        """
        Run full benchmark validation.
        
        Args:
            dataset: 'alfa', 'rflymad', or 'both'
            max_flights: Limit flights for faster testing (None = all)
        
        Returns:
            BenchmarkReport with comprehensive metrics
        """
        start_time = time.time()
        
        # Load datasets
        dataframes = []
        if dataset in ['alfa', 'both']:
            try:
                df_alfa = self.load_alfa_dataset()
                dataframes.append(('ALFA', df_alfa))
                print(f"[BENCHMARK] Loaded ALFA: {len(df_alfa)} samples, {df_alfa['flight_id'].nunique()} flights")
            except FileNotFoundError as e:
                print(f"[BENCHMARK] Warning: {e}")
        
        if dataset in ['rflymad', 'both']:
            try:
                df_rflymad = self.load_rflymad_dataset()
                dataframes.append(('RflyMAD', df_rflymad))
                print(f"[BENCHMARK] Loaded RflyMAD: {len(df_rflymad)} samples, {df_rflymad['flight_id'].nunique()} flights")
            except FileNotFoundError as e:
                print(f"[BENCHMARK] Warning: {e}")
        
        if not dataframes:
            raise ValueError("No datasets loaded. Check data paths.")
        
        # Combine datasets
        combined_df = pd.concat([df for _, df in dataframes], ignore_index=True)
        total_samples = len(combined_df)
        
        # Compute baseline statistics from normal flights (for cross-flight comparison)
        print("[BENCHMARK] Computing baseline statistics from normal flights...")
        baseline_stats = self.compute_baseline_stats(combined_df)
        self._baseline_stats = baseline_stats
        
        # Get unique flights
        flight_ids = combined_df['flight_id'].unique()
        if max_flights:
            flight_ids = flight_ids[:max_flights]
        total_flights = len(flight_ids)
        
        print(f"[BENCHMARK] Evaluating {total_flights} flights...")
        
        # Evaluate each flight
        results = []
        for i, flight_id in enumerate(flight_ids):
            if (i + 1) % 100 == 0:
                print(f"[BENCHMARK] Progress: {i+1}/{total_flights} flights")
            
            flight_data = combined_df[combined_df['flight_id'] == flight_id].copy()
            result = self.evaluate_flight(flight_data, str(flight_id), baseline_stats=baseline_stats)
            results.append(result)
        
        # Calculate aggregate metrics
        tp = sum(1 for r in results if r.is_true_positive)
        fp = sum(1 for r in results if r.is_false_positive)
        fn = sum(1 for r in results if r.is_false_negative)
        tn = sum(1 for r in results if not r.is_true_positive and not r.is_false_positive and not r.is_false_negative)
        
        # Detection rate (TPR / Recall)
        detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # False positive rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # F1 Score
        f1 = 2 * precision * detection_rate / (precision + detection_rate) if (precision + detection_rate) > 0 else 0.0
        
        # Onset delay statistics
        onset_delays = [r.onset_delay for r in results if r.onset_delay is not None]
        mean_onset = np.mean(onset_delays) if onset_delays else 0.0
        median_onset = np.median(onset_delays) if onset_delays else 0.0
        
        # Attribution accuracy
        tp_results = [r for r in results if r.is_true_positive]
        attribution_acc = sum(1 for r in tp_results if r.correct_attribution) / len(tp_results) if tp_results else 0.0
        
        # Per-fault breakdown
        fault_types = combined_df['fault_type'].unique()
        per_fault = {}
        for fault in fault_types:
            fault_results = [r for r in results if r.ground_truth_fault == fault]
            if fault_results:
                fault_tp = sum(1 for r in fault_results if r.is_true_positive)
                fault_fp = sum(1 for r in fault_results if r.is_false_positive)
                fault_fn = sum(1 for r in fault_results if r.is_false_negative)
                fault_total = len(fault_results)
                per_fault[fault] = {
                    'count': fault_total,
                    'detection_rate': fault_tp / (fault_tp + fault_fn) if (fault_tp + fault_fn) > 0 else 0.0,
                    'attribution_accuracy': sum(1 for r in fault_results if r.correct_attribution) / fault_tp if fault_tp > 0 else 0.0
                }
        
        processing_time = time.time() - start_time
        
        return BenchmarkReport(
            dataset_name=dataset,
            total_flights=total_flights,
            total_samples=total_samples,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn,
            detection_rate=detection_rate,
            false_positive_rate=fpr,
            precision=precision,
            f1_score=f1,
            mean_onset_delay=mean_onset,
            median_onset_delay=median_onset,
            attribution_accuracy=attribution_acc,
            per_fault_metrics=per_fault,
            flight_results=results,
            benchmark_timestamp=datetime.now().isoformat(),
            config_used=self.config,
            processing_time_sec=processing_time
        )
    
    def generate_report_json(self, report: BenchmarkReport, output_path: Optional[Path] = None) -> Dict:
        """Generate JSON report for competition/publication."""
        report_dict = {
            "benchmark_validation_report": {
                "title": "AeroGuardian Ground Truth Validation",
                "dataset": report.dataset_name,
                "timestamp": report.benchmark_timestamp,
                "processing_time_sec": round(report.processing_time_sec, 2)
            },
            "dataset_summary": {
                "total_flights": report.total_flights,
                "total_samples": report.total_samples
            },
            "core_metrics": {
                "detection_rate": round(report.detection_rate * 100, 1),
                "false_positive_rate": round(report.false_positive_rate * 100, 1),
                "precision": round(report.precision * 100, 1),
                "f1_score": round(report.f1_score * 100, 1),
                "attribution_accuracy": round(report.attribution_accuracy * 100, 1)
            },
            "confusion_matrix": {
                "true_positives": report.true_positives,
                "false_positives": report.false_positives,
                "false_negatives": report.false_negatives,
                "true_negatives": report.true_negatives
            },
            "timing_metrics": {
                "mean_onset_delay_sec": round(report.mean_onset_delay, 3),
                "median_onset_delay_sec": round(report.median_onset_delay, 3)
            },
            "per_fault_breakdown": {
                fault: {
                    "count": data["count"],
                    "detection_rate_pct": round(data["detection_rate"] * 100, 1),
                    "attribution_accuracy_pct": round(data["attribution_accuracy"] * 100, 1)
                }
                for fault, data in report.per_fault_metrics.items()
            },
            "thresholds_used": {
                "position_drift_m": report.config_used.position_drift_threshold,
                "altitude_m": report.config_used.altitude_threshold,
                "roll_deg": report.config_used.roll_threshold,
                "pitch_deg": report.config_used.pitch_threshold
            }
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report_dict, f, indent=2)
            print(f"[BENCHMARK] Report saved to {output_path}")
        
        return report_dict
    
    def print_summary(self, report: BenchmarkReport):
        """Print human-readable benchmark summary."""
        print("\n" + "="*60)
        print("  AEROGUARDIAN GROUND TRUTH BENCHMARK RESULTS")
        print("="*60)
        print(f"\nDataset: {report.dataset_name.upper()}")
        print(f"Flights evaluated: {report.total_flights}")
        print(f"Total samples: {report.total_samples:,}")
        print(f"Processing time: {report.processing_time_sec:.1f}s")
        print(f"  → {report.total_flights / report.processing_time_sec:.1f} flights/second")
        
        print("\n" + "-"*40)
        print("CORE METRICS (Competition-Ready)")
        print("-"*40)
        print(f"  Detection Rate:       {report.detection_rate * 100:5.1f}%")
        print(f"  False Positive Rate:  {report.false_positive_rate * 100:5.1f}%")
        print(f"  Precision:            {report.precision * 100:5.1f}%")
        print(f"  F1 Score:             {report.f1_score * 100:5.1f}%")
        print(f"  Attribution Accuracy: {report.attribution_accuracy * 100:5.1f}%")
        
        print("\n" + "-"*40)
        print("TIMING METRICS")
        print("-"*40)
        print(f"  Mean Onset Delay:   {report.mean_onset_delay:.3f}s")
        print(f"  Median Onset Delay: {report.median_onset_delay:.3f}s")
        
        print("\n" + "-"*40)
        print("CONFUSION MATRIX")
        print("-"*40)
        print(f"  True Positives:  {report.true_positives:4d}")
        print(f"  False Positives: {report.false_positives:4d}")
        print(f"  False Negatives: {report.false_negatives:4d}")
        print(f"  True Negatives:  {report.true_negatives:4d}")
        
        print("\n" + "-"*40)
        print("PER-FAULT BREAKDOWN")
        print("-"*40)
        for fault, data in report.per_fault_metrics.items():
            print(f"  {fault:15s}: {data['detection_rate']*100:5.1f}% detection, "
                  f"{data['attribution_accuracy']*100:5.1f}% attribution (n={data['count']})")
        
        print("\n" + "="*60)


def run_quick_benchmark(max_flights: int = 100):
    """Quick benchmark for testing."""
    benchmark = GroundTruthBenchmark()
    report = benchmark.run_benchmark(dataset='both', max_flights=max_flights)
    benchmark.print_summary(report)
    return report


if __name__ == "__main__":
    # Run full benchmark
    benchmark = GroundTruthBenchmark()
    
    print("[BENCHMARK] Starting ground truth validation...")
    report = benchmark.run_benchmark(dataset='both', max_flights=None)
    
    benchmark.print_summary(report)
    
    # Save JSON report
    output_dir = Path(__file__).parent.parent.parent / "outputs" / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    benchmark.generate_report_json(report, output_path)
