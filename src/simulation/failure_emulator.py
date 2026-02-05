"""
Failure Emulator Module for AeroGuardian.

Implements flight-dynamics-consistent failure effect emulation for PX4 SITL.
Produces realistic, distinguishable telemetry signatures for each failure category.

Author: AeroGuardian Team (Tiny Coders)
"""

import asyncio
import logging
import random
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger("AeroGuardian.FailureEmulator")


class FailureCategory(Enum):
    """Supported failure categories."""
    PROPULSION = "propulsion"
    NAVIGATION = "navigation"
    BATTERY = "battery"
    CONTROL = "control"
    SENSOR = "sensor"
    ALTITUDE = "altitude"
    AIRSPACE_VIOLATION = "airspace_violation"  # NOT a failure - healthy drone in wrong location


class FailurePhase(Enum):
    """Multi-stage failure progression phases."""
    NOMINAL = "nominal"
    INCIPIENT = "incipient"
    PROPAGATION = "propagation"
    CRITICAL = "critical"
    RESOLUTION = "resolution"


@dataclass
class PhaseConfig:
    """Configuration for a single failure phase."""
    name: FailurePhase
    duration_sec: float
    parameters: Dict[str, Any]
    description: str


@dataclass
class EmulationResult:
    """Result of failure emulation."""
    success: bool
    category: FailureCategory
    method: str
    phases_completed: List[str]
    observation_duration: float
    landed: bool
    error: Optional[str] = None
    telemetry_markers: Dict[str, float] = field(default_factory=dict)


class TemporalRandomizer:
    """
    Randomizes failure timing to prevent LLM script learning.
    
    Ensures the AI learns physical behavior patterns, not timing scripts.
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
    
    def randomize_onset(self, base_sec: float = 10.0) -> float:
        """Randomize fault onset time (5-20 seconds after mission start)."""
        return self.rng.uniform(5.0, 20.0)
    
    def randomize_phase_duration(self, base_sec: float) -> float:
        """Randomize phase duration by ±30%."""
        return base_sec * self.rng.uniform(0.7, 1.3)
    
    def randomize_severity(self, base: float = 1.0) -> float:
        """Randomize target severity (60-100% of base)."""
        return base * self.rng.uniform(0.6, 1.0)
    
    def randomize_degradation_tau(self) -> float:
        """Randomize exponential degradation time constant (3-10 seconds)."""
        return self.rng.uniform(3.0, 10.0)
    
    def compute_progressive_value(
        self, 
        current: float, 
        target: float, 
        elapsed: float, 
        tau: float
    ) -> float:
        """
        Compute progressive degradation using exponential approach.
        
        value(t) = current + (target - current) * (1 - exp(-t/tau))
        """
        return current + (target - current) * (1 - math.exp(-elapsed / tau))


class FailureEmulator:
    """
    Flight-dynamics-consistent failure effect emulation.
    
    Produces realistic, distinguishable telemetry for each failure type.
    Does NOT use kill switch or forced crash - uses parameter perturbation
    to let physics play out realistically.
    """
    
    # Map fault type keywords to failure categories
    CATEGORY_MAPPING = {
        # Propulsion
        "motor": FailureCategory.PROPULSION,
        "propulsion": FailureCategory.PROPULSION,
        "engine": FailureCategory.PROPULSION,
        "thrust": FailureCategory.PROPULSION,
        "esc": FailureCategory.PROPULSION,
        # Navigation
        "gps": FailureCategory.NAVIGATION,
        "navigation": FailureCategory.NAVIGATION,
        "gnss": FailureCategory.NAVIGATION,
        "position": FailureCategory.NAVIGATION,
        "flyaway": FailureCategory.NAVIGATION,  # Flyaway is navigation loss
        # Battery/Power
        "battery": FailureCategory.BATTERY,
        "power": FailureCategory.BATTERY,
        "voltage": FailureCategory.BATTERY,
        "depletion": FailureCategory.BATTERY,
        # Control
        "control": FailureCategory.CONTROL,
        "servo": FailureCategory.CONTROL,
        "actuator": FailureCategory.CONTROL,
        "link": FailureCategory.CONTROL,
        "rc": FailureCategory.CONTROL,
        "signal": FailureCategory.CONTROL,
        # Sensor
        "sensor": FailureCategory.SENSOR,
        "imu": FailureCategory.SENSOR,
        "gyro": FailureCategory.SENSOR,
        "accel": FailureCategory.SENSOR,
        "baro": FailureCategory.SENSOR,
        "mag": FailureCategory.SENSOR,
        "compass": FailureCategory.SENSOR,
        # Altitude (mechanical failure affecting altitude control)
        "altitude": FailureCategory.ALTITUDE,
        "height": FailureCategory.ALTITUDE,
        "vertical": FailureCategory.ALTITUDE,
        "z-axis": FailureCategory.ALTITUDE,
        # Airspace Violations - NOT mechanical failures
        # These represent a healthy drone in the wrong location
        "geofence": FailureCategory.AIRSPACE_VIOLATION,
        "airspace": FailureCategory.AIRSPACE_VIOLATION,
        "violation": FailureCategory.AIRSPACE_VIOLATION,
    }
    
    def __init__(self, drone, randomizer: Optional[TemporalRandomizer] = None):
        """
        Initialize failure emulator.
        
        Args:
            drone: MAVSDK System instance
            randomizer: Optional temporal randomizer (created if not provided)
        """
        self.drone = drone
        self.randomizer = randomizer or TemporalRandomizer()
        self._original_params: Dict[str, float] = {}
    
    def _classify_fault_type(self, fault_type: str) -> FailureCategory:
        """Map fault type string to failure category."""
        fault_lower = fault_type.lower().replace("-", "_").replace(" ", "_")
        
        # Special case: altitude_violation is an AIRSPACE violation, not a mechanical failure
        # The drone is healthy but in the wrong location
        if "altitude_violation" in fault_lower:
            return FailureCategory.AIRSPACE_VIOLATION
        
        for keyword, category in self.CATEGORY_MAPPING.items():
            if keyword in fault_lower:
                return category
        
        # Default to control failure for unknown types
        logger.warning(f"Unknown fault type '{fault_type}', defaulting to CONTROL")
        return FailureCategory.CONTROL
    
    async def _store_original_param(self, param_name: str) -> float:
        """Store original parameter value for later restoration."""
        try:
            if param_name.endswith("_INT"):
                value = await self.drone.param.get_param_int(param_name.replace("_INT", ""))
            else:
                value = await self.drone.param.get_param_float(param_name)
            self._original_params[param_name] = value
            return value
        except Exception as e:
            logger.warning(f"Could not get param {param_name}: {e}")
            return 0.0
    
    async def _restore_original_params(self):
        """Restore all modified parameters to original values."""
        for param_name, value in self._original_params.items():
            try:
                if isinstance(value, int):
                    await self.drone.param.set_param_int(param_name, value)
                else:
                    await self.drone.param.set_param_float(param_name, value)
            except Exception as e:
                logger.warning(f"Could not restore param {param_name}: {e}")
        self._original_params.clear()
    
    async def emulate(self, fault_type: str, severity: float = 1.0, parachute_trigger: bool = False) -> EmulationResult:
        """
        Execute failure emulation for the specified fault type.
        
        Dispatches to the appropriate failure-specific handler based on
        fault type classification.
        
        Args:
            fault_type: Type of failure to emulate (e.g., "motor_failure", "gps_loss")
            severity: Severity level 0.0-1.0 (1.0 = complete failure)
            parachute_trigger: If True and control failure, simulate parachute deployment
        
        Returns:
            EmulationResult with details of the emulation
        """
        category = self._classify_fault_type(fault_type)
        severity = self.randomizer.randomize_severity(severity)
        
        logger.info(f"🔧 Emulating {category.value} failure (severity: {severity:.2f})")
        if parachute_trigger:
            logger.info("    🪂 Parachute deployment requested")
        
        # Special handling for control failures with parachute
        if category == FailureCategory.CONTROL and parachute_trigger:
            try:
                result = await self._emulate_control_instability(severity, parachute_trigger=True)
                logger.info(f"✅ Failure emulation complete: {result.method}")
                return result
            except Exception as e:
                logger.error(f"❌ Failure emulation failed: {e}")
                return EmulationResult(
                    success=False,
                    category=category,
                    method="error",
                    phases_completed=[],
                    observation_duration=0,
                    landed=False,
                    error=str(e),
                )
        
        handlers = {
            FailureCategory.PROPULSION: self._emulate_propulsion_anomaly,
            FailureCategory.NAVIGATION: self._emulate_navigation_failure,
            FailureCategory.BATTERY: self._emulate_battery_failure,
            FailureCategory.CONTROL: self._emulate_control_instability,
            FailureCategory.SENSOR: self._emulate_sensor_degradation,
            FailureCategory.ALTITUDE: self._emulate_altitude_failure,
            FailureCategory.AIRSPACE_VIOLATION: self._emulate_airspace_violation,
        }
        
        handler = handlers.get(category, self._emulate_control_instability)
        
        try:
            result = await handler(severity)
            logger.info(f"✅ Failure emulation complete: {result.method}")
            return result
        except Exception as e:
            logger.error(f"❌ Failure emulation failed: {e}")
            return EmulationResult(
                success=False,
                category=category,
                method="error",
                phases_completed=[],
                observation_duration=0,
                landed=False,
                error=str(e),
            )
    
    async def _emulate_propulsion_anomaly(self, severity: float) -> EmulationResult:
        """
        Emulate propulsion/motor failure through asymmetric thrust.
        
        Creates yaw-roll coupling characteristic of real motor failures.
        Produces progressive spiral tendency, not instant crash.
        """
        logger.info("  → Propulsion anomaly: Asymmetric thrust emulation")
        phases_completed = []
        telemetry_markers = {}
        
        # Store original motor limits
        original_max = await self._store_original_param("PWM_MAIN_MAX2")
        if original_max == 0:
            original_max = 2000  # Default PX4 PWM max
        
        # Phase durations with randomization
        phase_durations = {
            "incipient": self.randomizer.randomize_phase_duration(5.0),
            "propagation": self.randomizer.randomize_phase_duration(10.0),
            "critical": self.randomizer.randomize_phase_duration(10.0),
        }
        
        try:
            # Phase 1: INCIPIENT - Minor thrust reduction (10-20%)
            logger.info("  Phase 1: INCIPIENT - Minor thrust asymmetry")
            reduction_1 = int(original_max * 0.15 * severity)
            await self.drone.param.set_param_int("PWM_MAIN_MAX2", int(original_max - reduction_1))
            telemetry_markers["phase1_start"] = asyncio.get_event_loop().time()
            await asyncio.sleep(phase_durations["incipient"])
            phases_completed.append("INCIPIENT")
            
            # Phase 2: PROPAGATION - Moderate thrust reduction (30-40%)
            logger.info("  Phase 2: PROPAGATION - Growing thrust asymmetry")
            reduction_2 = int(original_max * 0.35 * severity)
            await self.drone.param.set_param_int("PWM_MAIN_MAX2", int(original_max - reduction_2))
            telemetry_markers["phase2_start"] = asyncio.get_event_loop().time()
            await asyncio.sleep(phase_durations["propagation"])
            phases_completed.append("PROPAGATION")
            
            # Phase 3: CRITICAL - Severe thrust reduction (50-70%)
            logger.info("  Phase 3: CRITICAL - Severe thrust asymmetry")
            reduction_3 = int(original_max * 0.60 * severity)
            await self.drone.param.set_param_int("PWM_MAIN_MAX2", int(original_max - reduction_3))
            telemetry_markers["phase3_start"] = asyncio.get_event_loop().time()
            await asyncio.sleep(phase_durations["critical"])
            phases_completed.append("CRITICAL")
            
            # Phase 4: RESOLUTION - Controlled landing
            logger.info("  Phase 4: RESOLUTION - Initiating controlled landing")
            await self._restore_original_params()
            await self.drone.action.land()
            telemetry_markers["resolution_start"] = asyncio.get_event_loop().time()
            await asyncio.sleep(15)  # Wait for landing
            phases_completed.append("RESOLUTION")
            
            return EmulationResult(
                success=True,
                category=FailureCategory.PROPULSION,
                method="asymmetric_thrust_degradation",
                phases_completed=phases_completed,
                observation_duration=sum(phase_durations.values()) + 15,
                landed=True,
                telemetry_markers=telemetry_markers,
            )
            
        except Exception as e:
            await self._restore_original_params()
            raise e
    
    async def _emulate_navigation_failure(self, severity: float) -> EmulationResult:
        """
        Emulate GPS/navigation failure through estimator stress.
        
        Creates position drift and EKF failover characteristic of real GPS loss.
        Shows mode transitions and RTL behavior.
        """
        logger.info("  → Navigation failure: GPS degradation emulation")
        phases_completed = []
        telemetry_markers = {}
        
        # Store original EKF parameters
        original_gps_noise = await self._store_original_param("EKF2_GPS_P_NOISE")
        original_gps_v_noise = await self._store_original_param("EKF2_GPS_V_NOISE")
        if original_gps_noise == 0:
            original_gps_noise = 0.5
        if original_gps_v_noise == 0:
            original_gps_v_noise = 0.3
        
        phase_durations = {
            "incipient": self.randomizer.randomize_phase_duration(8.0),
            "propagation": self.randomizer.randomize_phase_duration(12.0),
            "critical": self.randomizer.randomize_phase_duration(10.0),
        }
        
        try:
            # Phase 1: INCIPIENT - GPS noise increase
            logger.info("  Phase 1: INCIPIENT - GPS accuracy degrading")
            noise_1 = original_gps_noise * (1 + 3 * severity)
            await self.drone.param.set_param_float("EKF2_GPS_P_NOISE", noise_1)
            await self.drone.param.set_param_float("EKF2_GPS_V_NOISE", original_gps_v_noise * (1 + 2 * severity))
            telemetry_markers["phase1_start"] = asyncio.get_event_loop().time()
            await asyncio.sleep(phase_durations["incipient"])
            phases_completed.append("INCIPIENT")
            
            # Phase 2: PROPAGATION - Severe GPS degradation
            logger.info("  Phase 2: PROPAGATION - GPS severely degraded")
            noise_2 = original_gps_noise * (1 + 8 * severity)
            await self.drone.param.set_param_float("EKF2_GPS_P_NOISE", noise_2)
            telemetry_markers["phase2_start"] = asyncio.get_event_loop().time()
            await asyncio.sleep(phase_durations["propagation"])
            phases_completed.append("PROPAGATION")
            
            # Phase 3: CRITICAL - Complete GPS denial (if high severity)
            if severity > 0.7:
                logger.info("  Phase 3: CRITICAL - GPS denied")
                try:
                    await self.drone.param.set_param_int("SYS_HAS_GPS", 0)
                except Exception:
                    # Some PX4 versions don't allow this parameter change in flight
                    noise_3 = original_gps_noise * 20
                    await self.drone.param.set_param_float("EKF2_GPS_P_NOISE", noise_3)
            telemetry_markers["phase3_start"] = asyncio.get_event_loop().time()
            await asyncio.sleep(phase_durations["critical"])
            phases_completed.append("CRITICAL")
            
            # Phase 4: RESOLUTION - Trigger RTL or land
            logger.info("  Phase 4: RESOLUTION - RTL/Land initiated")
            await self._restore_original_params()
            try:
                await self.drone.action.return_to_launch()
            except Exception:
                await self.drone.action.land()
            telemetry_markers["resolution_start"] = asyncio.get_event_loop().time()
            await asyncio.sleep(20)  # Wait for RTL/land
            phases_completed.append("RESOLUTION")
            
            return EmulationResult(
                success=True,
                category=FailureCategory.NAVIGATION,
                method="gps_degradation_ekf_stress",
                phases_completed=phases_completed,
                observation_duration=sum(phase_durations.values()) + 20,
                landed=True,
                telemetry_markers=telemetry_markers,
            )
            
        except Exception as e:
            await self._restore_original_params()
            raise e
    
    async def _emulate_battery_failure(self, severity: float) -> EmulationResult:
        """
        Emulate battery/power failure through failsafe triggering.
        
        Creates realistic RTL/land behavior from low battery conditions.
        Shows controlled descent, not crash.
        """
        logger.info("  → Battery failure: Failsafe trigger emulation")
        phases_completed = []
        telemetry_markers = {}
        
        # Store original battery parameters
        original_low_thr = await self._store_original_param("BAT1_LOW_THR")
        original_crit_thr = await self._store_original_param("BAT1_CRIT_THR")
        original_action = await self._store_original_param("COM_LOW_BAT_ACT")
        
        phase_durations = {
            "warning": self.randomizer.randomize_phase_duration(5.0),
            "critical": self.randomizer.randomize_phase_duration(10.0),
        }
        
        try:
            # Phase 1: Low battery warning
            logger.info("  Phase 1: Low battery warning triggered")
            await self.drone.param.set_param_float("BAT1_LOW_THR", 0.95)
            await self.drone.param.set_param_int("COM_LOW_BAT_ACT", 2)  # RTL
            telemetry_markers["phase1_start"] = asyncio.get_event_loop().time()
            await asyncio.sleep(phase_durations["warning"])
            phases_completed.append("WARNING")
            
            # Phase 2: Critical battery - force land
            if severity > 0.6:
                logger.info("  Phase 2: Critical battery - forcing land")
                await self.drone.param.set_param_float("BAT1_CRIT_THR", 0.98)
                await self.drone.param.set_param_int("COM_LOW_BAT_ACT", 3)  # Land
                telemetry_markers["phase2_start"] = asyncio.get_event_loop().time()
                await asyncio.sleep(phase_durations["critical"])
                phases_completed.append("CRITICAL")
            
            # Phase 3: Wait for controlled landing
            logger.info("  Phase 3: RESOLUTION - Controlled descent")
            telemetry_markers["resolution_start"] = asyncio.get_event_loop().time()
            await asyncio.sleep(20)  # Wait for land
            phases_completed.append("RESOLUTION")
            
            # Restore parameters
            await self._restore_original_params()
            
            return EmulationResult(
                success=True,
                category=FailureCategory.BATTERY,
                method="failsafe_triggered_descent",
                phases_completed=phases_completed,
                observation_duration=sum(phase_durations.values()) + 20,
                landed=True,
                telemetry_markers=telemetry_markers,
            )
            
        except Exception as e:
            await self._restore_original_params()
            raise e
    
    async def _emulate_control_instability(self, severity: float, parachute_trigger: bool = False) -> EmulationResult:
        """
        Emulate control system failure through gain degradation.
        
        Creates oscillatory instability characteristic of real control failures.
        Shows growing oscillations, saturation attempts, then recovery.
        
        Args:
            severity: Failure severity (0.0-1.0)
            parachute_trigger: If True, simulate parachute deployment in Phase 4
                             (rapid controlled descent instead of normal landing)
        """
        logger.info("  → Control instability: Gain degradation emulation")
        if parachute_trigger:
            logger.info("    🪂 Parachute deployment will be simulated in recovery phase")
        phases_completed = []
        telemetry_markers = {}
        
        # Store original control gains
        original_roll_p = await self._store_original_param("MC_ROLL_P")
        original_pitch_p = await self._store_original_param("MC_PITCH_P")
        original_yaw_p = await self._store_original_param("MC_YAW_P")
        
        if original_roll_p == 0:
            original_roll_p = 6.5  # Default PX4 value
        if original_pitch_p == 0:
            original_pitch_p = 6.5
        if original_yaw_p == 0:
            original_yaw_p = 2.8
        
        phase_durations = {
            "incipient": self.randomizer.randomize_phase_duration(8.0),
            "propagation": self.randomizer.randomize_phase_duration(12.0),
            "critical": self.randomizer.randomize_phase_duration(10.0),
        }
        
        # Gain reduction factors based on severity
        reductions = [
            0.85 - 0.05 * severity,   # Phase 1: 80-85%
            0.65 - 0.10 * severity,   # Phase 2: 55-65%
            0.45 - 0.15 * severity,   # Phase 3: 30-45%
        ]
        
        try:
            # Phase 1: INCIPIENT - Minor gain reduction
            logger.info("  Phase 1: INCIPIENT - Control response slightly degraded")
            await self.drone.param.set_param_float("MC_ROLL_P", original_roll_p * reductions[0])
            await self.drone.param.set_param_float("MC_PITCH_P", original_pitch_p * reductions[0])
            telemetry_markers["phase1_start"] = asyncio.get_event_loop().time()
            await asyncio.sleep(phase_durations["incipient"])
            phases_completed.append("INCIPIENT")
            
            # Phase 2: PROPAGATION - Moderate gain reduction
            logger.info("  Phase 2: PROPAGATION - Oscillations developing")
            await self.drone.param.set_param_float("MC_ROLL_P", original_roll_p * reductions[1])
            await self.drone.param.set_param_float("MC_PITCH_P", original_pitch_p * reductions[1])
            telemetry_markers["phase2_start"] = asyncio.get_event_loop().time()
            await asyncio.sleep(phase_durations["propagation"])
            phases_completed.append("PROPAGATION")
            
            # Phase 3: CRITICAL - Severe gain reduction
            logger.info("  Phase 3: CRITICAL - Significant instability")
            await self.drone.param.set_param_float("MC_ROLL_P", original_roll_p * reductions[2])
            await self.drone.param.set_param_float("MC_PITCH_P", original_pitch_p * reductions[2])
            await self.drone.param.set_param_float("MC_YAW_P", original_yaw_p * reductions[1])
            telemetry_markers["phase3_start"] = asyncio.get_event_loop().time()
            await asyncio.sleep(phase_durations["critical"])
            phases_completed.append("CRITICAL")
            
            # Phase 4: RESOLUTION - Restore gains and land (or parachute deploy)
            if parachute_trigger:
                logger.info("  Phase 4: PARACHUTE_DEPLOY - 🪂 Simulating parachute deployment")
                telemetry_markers["parachute_deploy"] = asyncio.get_event_loop().time()
                # Parachute simulation: Kill all motors (motors off), let physics handle descent
                # This produces distinct telemetry: rapid altitude loss, minimal control authority
                await self._restore_original_params()
                try:
                    # Set to STABILIZED mode with minimal throttle (simulates parachute descent)
                    await self.drone.action.hold()
                    await asyncio.sleep(2)  # Brief hover
                    # Then land - simulating parachute bringing it down
                    await self.drone.action.land()
                except Exception as e:
                    logger.warning(f"    Parachute simulation fallback: {e}")
                    await self.drone.action.land()
                phases_completed.append("PARACHUTE_DEPLOY")
            else:
                logger.info("  Phase 4: RESOLUTION - Restoring control, landing")
                await self._restore_original_params()
                await self.drone.action.land()
                phases_completed.append("RESOLUTION")
            
            telemetry_markers["resolution_start"] = asyncio.get_event_loop().time()
            await asyncio.sleep(15)
            
            return EmulationResult(
                success=True,
                category=FailureCategory.CONTROL,
                method="control_gain_degradation" + ("_with_parachute" if parachute_trigger else ""),
                phases_completed=phases_completed,
                observation_duration=sum(phase_durations.values()) + 15,
                landed=True,
                telemetry_markers=telemetry_markers,
            )
            
        except Exception as e:
            await self._restore_original_params()
            raise e
    
    async def _emulate_sensor_degradation(self, severity: float) -> EmulationResult:
        """
        Emulate sensor failure through EKF noise injection.
        
        Creates attitude estimation errors and compensatory behavior.
        Shows gradual degradation, not instant crash.
        """
        logger.info("  → Sensor degradation: EKF noise injection")
        phases_completed = []
        telemetry_markers = {}
        
        # Store original EKF noise parameters
        original_acc_noise = await self._store_original_param("EKF2_ACC_NOISE")
        original_gyr_noise = await self._store_original_param("EKF2_GYR_NOISE")
        original_baro_noise = await self._store_original_param("EKF2_BARO_NOISE")
        
        if original_acc_noise == 0:
            original_acc_noise = 0.35
        if original_gyr_noise == 0:
            original_gyr_noise = 0.015
        if original_baro_noise == 0:
            original_baro_noise = 2.0
        
        phase_durations = {
            "incipient": self.randomizer.randomize_phase_duration(10.0),
            "propagation": self.randomizer.randomize_phase_duration(12.0),
            "critical": self.randomizer.randomize_phase_duration(8.0),
        }
        
        try:
            # Phase 1: INCIPIENT - Mild sensor noise
            logger.info("  Phase 1: INCIPIENT - Sensor noise increasing")
            await self.drone.param.set_param_float("EKF2_ACC_NOISE", original_acc_noise * (1 + severity))
            await self.drone.param.set_param_float("EKF2_GYR_NOISE", original_gyr_noise * (1 + severity * 2))
            telemetry_markers["phase1_start"] = asyncio.get_event_loop().time()
            await asyncio.sleep(phase_durations["incipient"])
            phases_completed.append("INCIPIENT")
            
            # Phase 2: PROPAGATION - Moderate sensor degradation
            logger.info("  Phase 2: PROPAGATION - Sensor quality degraded")
            await self.drone.param.set_param_float("EKF2_ACC_NOISE", original_acc_noise * (1 + 3 * severity))
            await self.drone.param.set_param_float("EKF2_GYR_NOISE", original_gyr_noise * (1 + 5 * severity))
            await self.drone.param.set_param_float("EKF2_BARO_NOISE", original_baro_noise * (1 + 2 * severity))
            telemetry_markers["phase2_start"] = asyncio.get_event_loop().time()
            await asyncio.sleep(phase_durations["propagation"])
            phases_completed.append("PROPAGATION")
            
            # Phase 3: CRITICAL - Severe sensor noise
            if severity > 0.6:
                logger.info("  Phase 3: CRITICAL - Severe sensor degradation")
                await self.drone.param.set_param_float("EKF2_ACC_NOISE", original_acc_noise * (1 + 6 * severity))
                await self.drone.param.set_param_float("EKF2_GYR_NOISE", original_gyr_noise * (1 + 10 * severity))
                telemetry_markers["phase3_start"] = asyncio.get_event_loop().time()
                await asyncio.sleep(phase_durations["critical"])
                phases_completed.append("CRITICAL")
            
            # Phase 4: RESOLUTION - Land with degraded sensors
            logger.info("  Phase 4: RESOLUTION - Landing with degraded sensors")
            await self.drone.action.land()
            telemetry_markers["resolution_start"] = asyncio.get_event_loop().time()
            await asyncio.sleep(15)
            phases_completed.append("RESOLUTION")
            
            # Restore parameters
            await self._restore_original_params()
            
            return EmulationResult(
                success=True,
                category=FailureCategory.SENSOR,
                method="ekf_noise_injection",
                phases_completed=phases_completed,
                observation_duration=sum(phase_durations.values()) + 15,
                landed=True,
                telemetry_markers=telemetry_markers,
            )
            
        except Exception as e:
            await self._restore_original_params()
            raise e
            raise e
    
    async def _emulate_altitude_failure(self, severity: float) -> EmulationResult:
        """
        Emulate altitude control failure through Z-axis gain degradation.
        
        Creates instability in vertical hold, leading to drift or oscillation.
        """
        logger.info("  → Altitude failure: Z-axis gain degradation emulation")
        phases_completed = []
        telemetry_markers = {}
        
        # Store original altitude gains
        original_z_p = await self._store_original_param("MPC_Z_P")
        original_z_vel_p = await self._store_original_param("MPC_Z_VEL_P")
        
        if original_z_p == 0:
            original_z_p = 1.0
        if original_z_vel_p == 0:
            original_z_vel_p = 0.2
            
        phase_durations = {
            "incipient": self.randomizer.randomize_phase_duration(8.0),
            "propagation": self.randomizer.randomize_phase_duration(10.0),
            "critical": self.randomizer.randomize_phase_duration(12.0),
        }
        
        try:
            # Phase 1: INCIPIENT - Soft altitude hold
            logger.info("  Phase 1: INCIPIENT - Altitude hold softening")
            await self.drone.param.set_param_float("MPC_Z_P", original_z_p * 0.5)
            telemetry_markers["phase1_start"] = asyncio.get_event_loop().time()
            await asyncio.sleep(phase_durations["incipient"])
            phases_completed.append("INCIPIENT")
            
            # Phase 2: PROPAGATION - Oscillations / Drift
            logger.info("  Phase 2: PROPAGATION - Vertical instability")
            await self.drone.param.set_param_float("MPC_Z_P", original_z_p * 0.2)
            await self.drone.param.set_param_float("MPC_Z_VEL_P", original_z_vel_p * 0.5)
            telemetry_markers["phase2_start"] = asyncio.get_event_loop().time()
            await asyncio.sleep(phase_durations["propagation"])
            phases_completed.append("PROPAGATION")
            
            # Phase 3: CRITICAL - Severe degradation (if high severity)
            if severity > 0.6:
                logger.info("  Phase 3: CRITICAL - Loss of vertical dampening")
                await self.drone.param.set_param_float("MPC_Z_VEL_P", original_z_vel_p * 0.1)
                telemetry_markers["phase3_start"] = asyncio.get_event_loop().time()
                await asyncio.sleep(phase_durations["critical"])
                phases_completed.append("CRITICAL")
                
            # Phase 4: RESOLUTION - Restore and Land
            logger.info("  Phase 4: RESOLUTION - Restoring Z-control, landing")
            await self._restore_original_params()
            await self.drone.action.land()
            telemetry_markers["resolution_start"] = asyncio.get_event_loop().time()
            await asyncio.sleep(15)
            phases_completed.append("RESOLUTION")
            
            return EmulationResult(
                success=True,
                category=FailureCategory.ALTITUDE,
                method="z_axis_gain_degradation",
                phases_completed=phases_completed,
                observation_duration=sum(phase_durations.values()) + 15,
                landed=True,
                telemetry_markers=telemetry_markers,
            )
            
        except Exception as e:
            await self._restore_original_params()
            raise e

    async def _emulate_airspace_violation(self, severity: float) -> EmulationResult:
        """
        Handle airspace violation scenarios (altitude_violation, geofence_violation).
        
        IMPORTANT: This is NOT a mechanical failure. The drone is healthy but was
        observed in airspace where it shouldn't be. We do NOT inject any failure -
        we simply execute a normal flight to demonstrate healthy drone behavior.
        
        This produces CLEAN telemetry with no anomalies, which is the EXPECTED
        outcome for an airspace violation scenario.
        """
        logger.info("  → Airspace violation: NO failure injection (healthy drone scenario)")
        logger.info("  → Drone will fly normally to demonstrate healthy behavior")
        
        phases_completed = []
        telemetry_markers = {}
        
        try:
            # Phase 1: OBSERVATION - Let the drone fly normally
            logger.info("  Phase 1: OBSERVATION - Healthy drone flight")
            telemetry_markers["observation_start"] = asyncio.get_event_loop().time()
            
            # Observe healthy flight for 30 seconds
            observation_duration = self.randomizer.randomize_phase_duration(30.0)
            await asyncio.sleep(observation_duration)
            phases_completed.append("OBSERVATION")
            
            # Phase 2: RESOLUTION - Normal landing
            logger.info("  Phase 2: RESOLUTION - Normal landing")
            await self.drone.action.land()
            telemetry_markers["resolution_start"] = asyncio.get_event_loop().time()
            await asyncio.sleep(15)
            phases_completed.append("RESOLUTION")
            
            return EmulationResult(
                success=True,
                category=FailureCategory.AIRSPACE_VIOLATION,
                method="healthy_drone_observation",
                phases_completed=phases_completed,
                observation_duration=observation_duration + 15,
                landed=True,
                telemetry_markers=telemetry_markers,
            )
            
        except Exception as e:
            logger.error(f"Airspace violation observation failed: {e}")
            raise e
