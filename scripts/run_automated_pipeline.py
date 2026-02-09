"""
AeroGuardian Automated Pipeline Runner
======================================
Author: AeroGuardian Member
Date: 2026-01-19
Updated: 2026-01-31
Version: 1.0

Fully automated pipeline that runs everything without user input:
1. Start PX4 SITL with Gazebo GUI in WSL
2. Load FAA UAS sighting report and generate configuration via LLM
3. Execute flight mission and capture telemetry
4. Generate safety reports (PDF/JSON) & Evaluation report (Excel/JSON)

Usage:
    python scripts/run_automated_pipeline.py --wsl-ip [IP_ADDRESS]                    # Default: sighting 0
    python scripts/run_automated_pipeline.py -r 5 --wsl-ip [IP_ADDRESS]               # Specific sighting
    python scripts/run_automated_pipeline.py --headless --wsl-ip [IP_ADDRESS]         # No Gazebo GUI
"""

import os
import sys
import time
import asyncio
import subprocess
import argparse

# =============================================================================
# CRITICAL: Fix gRPC asyncio race condition on Windows
# =============================================================================
# This prevents "RuntimeError: Event loop is closed" when gRPC poller thread
# tries to send events after the event loop has been closed during cleanup.
# Reference: https://github.com/grpc/grpc/issues/25364

# Set Windows-compatible event loop policy
if sys.platform == 'win32':
    # Use WindowsSelectorEventLoopPolicy to avoid ProactorEventLoop issues
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Suppress gRPC internal thread exceptions during shutdown
import threading
_original_excepthook = threading.excepthook

def _grpc_excepthook(args):
    """Suppress gRPC shutdown exceptions."""
    if args.exc_type == RuntimeError and "Event loop is closed" in str(args.exc_value):
        # Silently ignore this known gRPC cleanup issue
        return
    _original_excepthook(args)

threading.excepthook = _grpc_excepthook
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# Suppress pydantic serialization warnings from dspy/litellm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Setup project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

# Use centralized logging (single daily log file)
from src.core.logging_config import get_logger
logger = get_logger("AeroGuardian.Pipeline")

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    # WSL settings
    wsl_distro: str = "Ubuntu"
    px4_dir: str = "~/PX4-Autopilot"
    # WSL IP (obtained from WSL via: ip addr show eth0)
    # QGroundControl on Windows connects TO this WSL IP
    wsl_ip: str = "[WSL_IP]"
    qgc_port: int = 18570
    mavsdk_port: int = 14580 #14540
    
    # Timeouts
    px4_startup_timeout: int = 120
    mission_timeout: int = 300
    telemetry_rate_hz: int = 10
    
    # Simulation settings
    vehicle: str = "iris"
    world: str = "empty"
    headless: bool = False
    data_source: str = "sightings"  # 'sightings' (8000) or 'failures' (31)
    
    # Output
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")


# =============================================================================
# WSL Controller
# =============================================================================

class WSLController:
    """Control PX4 SITL in WSL2."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._px4_process = None
        self._wsl_ip = None
        self._windows_ip = None
        
    def run_wsl(self, cmd: str, timeout: int = 60) -> Tuple[int, str]:
        """Execute command in WSL."""
        try:
            result = subprocess.run(
                ["wsl", "-d", self.config.wsl_distro, "bash", "-c", cmd],
                capture_output=True, text=True, timeout=timeout
            )
            return result.returncode, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return -1, "Command timed out"
        except Exception as e:
            return -1, str(e)
    
    def get_wsl_ip(self) -> str:
        """Get WSL2 IP address."""
        if self._wsl_ip:
            return self._wsl_ip
        code, output = self.run_wsl("hostname -I | awk '{print $1}'", timeout=10)
        if code == 0:
            self._wsl_ip = output.strip()
        return self._wsl_ip or "127.0.0.1"
    
    def get_windows_ip(self) -> str:
        """Get Windows host IP from WSL."""
        if self._windows_ip:
            return self._windows_ip
        code, output = self.run_wsl("grep nameserver /etc/resolv.conf | awk '{print $2}' | head -1", timeout=10)
        if code == 0 and output.strip():
            # Extract just the IP address
            ip = output.strip().split()[-1]  # Get the last word which should be the IP
            if ip and '.' in ip:  # Basic IP validation
                self._windows_ip = ip
        return self._windows_ip or "172.27.160.1"
    
    def is_px4_running(self) -> bool:
        """Check if PX4 SITL is running."""
        code, _ = self.run_wsl("pgrep -f 'px4.*sitl'", timeout=5)
        return code == 0
    
    def start_px4_gazebo(self) -> bool:
        """Start PX4 SITL with Gazebo."""
        if self.is_px4_running():
            logger.info("PX4 SITL is already running")
            return True
        
        logger.info(f"Starting PX4 SITL...")
        
        # Get Windows IP for DISPLAY (in case Gazebo is needed)
        windows_ip = self.get_windows_ip()
        
        # Choose simulator target:
        # - sihsim_quadx: Software-in-the-loop, no graphics needed (headless friendly)
        # - gazebo-classic_iris: Gazebo Classic with iris drone - SUPPORTS FAILURE INJECTION
        #                        (requires: git submodule update --init --recursive Tools/simulation/gazebo-classic)
        # - gz_x500: Gazebo Harmonic with X500 drone (limited failure injection support)
        # NOTE: Using gz_x500 because Gazebo Classic submodule is not initialized
        if self.config.headless:
            sim_target = "sihsim_quadx"  # SIH simulator - no graphics
        else:
            sim_target = "gz_x500"  # Gazebo Harmonic - fallback to crash simulation for failures
        
        # Get home location if set (from incident geocoding)
        home_lat = getattr(self, '_home_lat', 47.397742)  # Default: Switzerland test location
        home_lon = getattr(self, '_home_lon', 8.545594)
        home_alt = getattr(self, '_home_alt', 488.0)
        
        # For WSLg, use :0 directly; for VcXsrv use windows_ip:0
        # WSLg is default on Windows 11
        display_env = ":0"  # WSLg default
        
        launch_cmd = f"""
            pkill -9 px4 2>/dev/null || true
            pkill -9 gz 2>/dev/null || true
            pkill -9 ruby 2>/dev/null || true
            sleep 2
            cd {self.config.px4_dir}
            
            # Source environment
            source ~/.bashrc 2>/dev/null || true
            source /opt/ros/*/setup.bash 2>/dev/null || true
            
            # Display for Gazebo GUI
            export DISPLAY={display_env}
            export LIBGL_ALWAYS_SOFTWARE=1
            
            # PX4 configuration
            export PX4_SIM_HOST_ADDR={self.config.wsl_ip}
            export PX4_HOME_LAT={home_lat}
            export PX4_HOME_LON={home_lon}
            export PX4_HOME_ALT={home_alt}
            
            # ENABLE FAILURE INJECTION for demo
            # This allows MAVSDK failure.inject() to work
            export PX4_SYS_FAILURE_EN=1
            
            # Set Gazebo world (default, baylands, forest, lawn, windy, etc.)
            export PX4_GZ_WORLD=default
            
            # Gazebo model paths (for both Classic and Harmonic)
            export GAZEBO_MODEL_PATH=$HOME/PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models:$GAZEBO_MODEL_PATH
            export GZ_SIM_RESOURCE_PATH=$HOME/PX4-Autopilot/Tools/simulation/gz/models:$GZ_SIM_RESOURCE_PATH
            
            # Launch PX4 with Gazebo
            # Add param override for failure injection
            cd $HOME/PX4-Autopilot && make px4_sitl {sim_target} 2>&1
        """
        
        logger.info(f"  Simulator: {sim_target}")
        logger.info(f"  PX4_SIM_HOST_ADDR={self.config.wsl_ip}")
        
        # Start PX4 in background
        self._px4_process = subprocess.Popen(
            ["wsl", "-d", self.config.wsl_distro, "bash", "-c", launch_cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Wait for PX4 to be ready, capturing output
        logger.info("Waiting for PX4 SITL to initialize...")
        start_time = time.time()
        output_lines = []
        
        while time.time() - start_time < self.config.px4_startup_timeout:
            # Check if process is still running
            if self._px4_process.poll() is not None:
                # Process ended, capture remaining output
                remaining = self._px4_process.stdout.read()
                if remaining:
                    output_lines.append(remaining)
                logger.error(f"PX4 process exited with code: {self._px4_process.returncode}")
                logger.error(f"Last output:\n{''.join(output_lines[-10:])}")
                return False
            
            # Read available output without blocking
            try:
                line = self._px4_process.stdout.readline()
                if line:
                    output_lines.append(line)
                    # Log PX4 startup progress
                    if any(x in line.lower() for x in ['error', 'fail', 'ready', 'armed', 'connected']):
                        logger.info(f"  PX4: {line.strip()[:80]}")
            except:
                pass
            
            # Check if ready
            if self._check_px4_ready():
                logger.info("✓ PX4 SITL is ready and accepting connections")
                return True
            
            time.sleep(2)
        
        logger.error("PX4 SITL failed to start within timeout")
        logger.error(f"Last 20 lines of output:")
        for line in output_lines[-20:]:
            logger.error(f"  {line.strip()}")
        return False
    
    def _check_px4_ready(self) -> bool:
        """Check if PX4 is ready to accept connections."""
        # Check if PX4 process is running and has initialized
        # Look for the px4 process and check if it has started the mavlink interface
        code, output = self.run_wsl("pgrep -a px4 | grep -v grep", timeout=5)
        if code != 0:
            return False
        
        # Also check for the MAVLink output in the process
        code2, _ = self.run_wsl("ss -tuln | grep -E '(14540|14580|18570)'", timeout=5)
        return code2 == 0
    
    def start_mavproxy_bridge(self) -> bool:
        """Start MAVProxy to bridge MAVLink from WSL to Windows."""
        windows_ip = self.get_windows_ip()
        
        logger.info(f"Starting MAVProxy bridge (WSL -> Windows:{self.config.mavsdk_port})...")
        
        # MAVProxy command: listen on UDP 14580 (PX4 onboard), forward to Windows
        mavproxy_cmd = f"""
            python3 -m MAVProxy.mavproxy \
                --master=udpin:127.0.0.1:14580 \
                --out=udpout:{windows_ip}:{self.config.mavsdk_port} \
                --daemon &
            sleep 2
            echo "MAVProxy started"
        """
        
        code, output = self.run_wsl(mavproxy_cmd, timeout=30)
        if code == 0:
            logger.info("✓ MAVProxy bridge started")
            return True
        else:
            logger.warning(f"MAVProxy failed to start: {output}")
            return False
    
    def set_home_location(self, lat: float, lon: float, alt: float = 0) -> bool:
        """Set PX4 home location via environment variables before startup."""
        logger.info(f"Setting home location: ({lat:.4f}, {lon:.4f})")
        
        # PX4 uses PX4_HOME_LAT, PX4_HOME_LON, PX4_HOME_ALT environment variables
        # These are set in the launch command
        self._home_lat = lat
        self._home_lon = lon
        self._home_alt = alt
        return True
    
    def launch_qgroundcontrol(self) -> bool:
        """Launch QGroundControl on Windows."""
        logger.info("Launching QGroundControl...")
        
        # Common QGC installation paths
        qgc_paths = [
            r"C:\Program Files\QGroundControl\bin\QGroundControl.exe",
        ]
        
        # Check if QGC is already running
        try:
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq QGroundControl.exe"],
                capture_output=True,
                text=True
            )
            if "QGroundControl.exe" in result.stdout:
                logger.info("✓ QGroundControl is already running")
                return True
        except:
            pass
        
        # Find and launch QGC
        for path in qgc_paths:
            if os.path.exists(path):
                try:
                    subprocess.Popen([path], start_new_session=True)
                    logger.info(f"✓ Launched QGroundControl from: {path}")
                    time.sleep(3)  # Give QGC time to start
                    return True
                except Exception as e:
                    logger.warning(f"Failed to launch QGC: {e}")
        
        logger.warning("QGroundControl not found. Please start it manually.")
        return False
    
    def stop_px4(self):
        """Stop PX4 SITL."""
        logger.info("Stopping PX4 SITL...")
        self.run_wsl("pkill -9 px4 2>/dev/null || true", timeout=10)
        self.run_wsl("pkill -9 ruby 2>/dev/null || true", timeout=10)
        self.run_wsl("pkill -9 gz 2>/dev/null || true", timeout=10)
        self.run_wsl("pkill -9 mavproxy 2>/dev/null || true", timeout=10)
        if self._px4_process:
            self._px4_process.terminate()
            self._px4_process = None




# =============================================================================
# MAVSDK Mission Executor
# =============================================================================

class MissionExecutor:
    """Execute missions using MAVSDK."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.drone = None
        self.telemetry_data: List[Dict] = []
        self._capturing = False
    
    async def connect(self, max_retries: int = 3, retry_delay: float = 5.0) -> bool:
        """
        Connect to PX4 via MAVSDK with retry logic.
        
        Args:
            max_retries: Maximum connection attempts before giving up
            retry_delay: Seconds to wait between retry attempts
            
        Returns:
            True if connected, False if all retries failed
        """
        from mavsdk import System
        
        connection_url = f"udp://{self.config.wsl_ip}:{self.config.mavsdk_port}"
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Connection attempt {attempt}/{max_retries} to {connection_url}...")
                
                # Create fresh System instance for each retry
                self.drone = System()
                await self.drone.connect(system_address=connection_url)
                
                # Wait for connection with timeout
                connected = False
                timeout_counter = 0
                async for state in self.drone.core.connection_state():
                    if state.is_connected:
                        logger.info("✓ Connected to PX4")
                        connected = True
                        break
                    timeout_counter += 1
                    if timeout_counter > 30:  # ~30 second timeout
                        raise TimeoutError("Connection state timeout")
                    await asyncio.sleep(1)
                
                if not connected:
                    raise ConnectionError("Failed to establish connection")
                
                # Wait for GPS fix with timeout
                logger.info("Waiting for GPS fix...")
                gps_timeout = 0
                async for health in self.drone.telemetry.health():
                    if health.is_global_position_ok:
                        logger.info("✓ GPS fix acquired")
                        return True
                    gps_timeout += 1
                    if gps_timeout > 60:  # ~60 second GPS timeout
                        raise TimeoutError("GPS fix timeout")
                    await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                logger.warning(f"Connection attempt {attempt} cancelled")
                await self._cleanup_connection()
                raise  # Re-raise cancellation
                
            except Exception as e:
                logger.warning(f"Connection attempt {attempt} failed: {e}")
                await self._cleanup_connection()
                
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {max_retries} connection attempts failed")
                    return False
        
        return False
    
    async def _cleanup_connection(self):
        """Cleanup connection state after failure."""
        try:
            if self.drone:
                # Attempt graceful disconnect if possible
                try:
                    # Give gRPC time to cleanup
                    await asyncio.sleep(0.5)
                except Exception:
                    pass
                self.drone = None
        except Exception as e:
            logger.debug(f"Cleanup exception (safe to ignore): {e}")
    
    async def start_telemetry_capture(self):
        """Start capturing telemetry in background."""
        self._capturing = True
        self.telemetry_data = []
        self._current_position = None
        self._current_attitude = None
        self._current_battery = None
        self._current_actuator = None
        self._capture_start_time = time.time()
        
        # Create separate tasks for each telemetry stream
        async def position_capture():
            async for pos in self.drone.telemetry.position():
                if not self._capturing:
                    break
                self._current_position = pos
                
        async def attitude_capture():
            async for att in self.drone.telemetry.attitude_euler():
                if not self._capturing:
                    break
                self._current_attitude = att
                
        async def battery_capture():
            async for bat in self.drone.telemetry.battery():
                if not self._capturing:
                    break
                self._current_battery = bat

        async def actuator_capture():
            async for act in self.drone.telemetry.actuator_output_status():
                if not self._capturing:
                    break
                self._current_actuator = act
        
        async def data_recorder():
            """Record combined telemetry at fixed rate."""
            while self._capturing:
                if self._current_position:
                    record = {
                        "timestamp": time.time() - self._capture_start_time,
                        "lat": self._current_position.latitude_deg,
                        "lon": self._current_position.longitude_deg,
                        "alt": self._current_position.absolute_altitude_m,
                        "relative_alt": self._current_position.relative_altitude_m,
                        "roll": self._current_attitude.roll_deg if self._current_attitude else 0,
                        "pitch": self._current_attitude.pitch_deg if self._current_attitude else 0,
                        "yaw": self._current_attitude.yaw_deg if self._current_attitude else 0,
                        "battery_v": self._current_battery.voltage_v if self._current_battery else 0,
                        "battery_pct": self._current_battery.remaining_percent * 100 if self._current_battery else 0,
                        # Add actuator outputs (motor values)
                        # actuator_output_status yields .actuator which is a list of floats
                        "actuator_controls_0": self._current_actuator.actuator if self._current_actuator else [],
                    }
                    self.telemetry_data.append(record)
                await asyncio.sleep(1.0 / self.config.telemetry_rate_hz)
        
        # Start all capture tasks concurrently
        asyncio.create_task(position_capture())
        asyncio.create_task(attitude_capture())
        asyncio.create_task(battery_capture())
        asyncio.create_task(actuator_capture())
        asyncio.create_task(data_recorder())
    
    def stop_telemetry_capture(self):
        """Stop telemetry capture."""
        self._capturing = False
    
    async def execute_mission(
        self,
        waypoints: List[Dict],
        takeoff_alt: float,
        speed_m_s: float,
        fault_type: str = None,
        fault_onset_sec: int = 60,
        fault_severity: float = 1.0
    ) -> Tuple[bool, List[Dict]]:
        """
        Execute a complete mission with LLM-specified parameters and fault injection.
        
        Args:
            waypoints: List of waypoint dicts with lat, lon, alt, action
            takeoff_alt: Takeoff altitude in meters
            speed_m_s: Cruise speed in m/s
            fault_type: Type of fault to inject (motor_failure, gps_loss, etc.)
            fault_onset_sec: Seconds after takeoff to inject fault
            fault_severity: Fault severity 0.0-1.0 (1.0 = complete failure)
        
        Returns:
            Tuple of (success, telemetry_data)
        """
        from mavsdk.mission import MissionItem, MissionPlan
        
        try:
            # =========================================================
            # ENABLE FAILURE INJECTION (required for demo)
            # =========================================================
            if fault_type:
                try:
                    logger.info("   → Enabling failure injection (SYS_FAILURE_EN=1)...")
                    await self.drone.param.set_param_int("SYS_FAILURE_EN", 1)
                    logger.info("   ✓ SYS_FAILURE_EN enabled")
                except Exception as e:
                    logger.warning(f"   ⚠ Could not set SYS_FAILURE_EN: {e}")
            
            # Build mission items
            # IMPORTANT: PX4 missions should NOT have TAKEOFF/LAND vehicle actions
            # in middle items - only use NONE for intermediate waypoints
            mission_items = []
            for i, wp in enumerate(waypoints):
                # All mission items should have NONE action - the mission
                # executor handles takeoff/land separately
                vehicle_action = MissionItem.VehicleAction.NONE
                
                item = MissionItem(
                    latitude_deg=wp.get("lat", 0),
                    longitude_deg=wp.get("lon", 0),
                    relative_altitude_m=wp.get("alt", takeoff_alt),
                    speed_m_s=speed_m_s,
                    is_fly_through=True,
                    gimbal_pitch_deg=0,
                    gimbal_yaw_deg=0,
                    camera_action=MissionItem.CameraAction.NONE,
                    loiter_time_s=0,
                    camera_photo_interval_s=0,
                    acceptance_radius_m=3.0,  # Reduced from 5.0 for more accurate waypoint hitting
                    yaw_deg=0,
                    camera_photo_distance_m=0,
                    vehicle_action=vehicle_action
                )
                mission_items.append(item)
            
            mission_plan = MissionPlan(mission_items)
            
            # Upload mission
            logger.info(f"Uploading mission with {len(waypoints)} waypoints...")
            await self.drone.mission.upload_mission(mission_plan)
            
            # Start telemetry capture
            await self.start_telemetry_capture()
            
            # Wait for vehicle to be ready (health checks)
            logger.info("Waiting for vehicle health checks...")
            arm_ready = False
            for attempt in range(30):  # 30 second timeout
                async for health in self.drone.telemetry.health():
                    if health.is_armable:
                        arm_ready = True
                        logger.info("✓ Vehicle is armable")
                    break
                if arm_ready:
                    break
                await asyncio.sleep(1)
            
            if not arm_ready:
                logger.warning("Vehicle health checks not passed, attempting arm anyway...")
            
            # Arm with retry
            logger.info("Arming vehicle...")
            for attempt in range(3):  # 3 retries
                try:
                    await self.drone.action.arm()
                    logger.info("✓ Vehicle armed")
                    break
                except Exception as arm_error:
                    if attempt < 2:
                        logger.warning(f"Arm attempt {attempt+1} failed: {arm_error}, retrying...")
                        await asyncio.sleep(2)
                    else:
                        raise arm_error
            
            # Set takeoff altitude before taking off
            logger.info(f"Taking off to {takeoff_alt}m...")
            await self.drone.action.set_takeoff_altitude(takeoff_alt)
            await self.drone.action.takeoff()
            
            # Wait for drone to reach target altitude (with timeout)
            target_reached = False
            current_alt = 0.0 # Initialize current_alt for logging outside the loop
            for wait_cycle in range(60):  # 60 second timeout
                await asyncio.sleep(1)
                async for position in self.drone.telemetry.position():
                    current_alt = position.relative_altitude_m
                    if current_alt >= takeoff_alt * 0.9:  # 90% of target
                        target_reached = True
                        logger.info(f"✓ Reached altitude: {current_alt:.1f}m")
                    break
                if target_reached:
                    break
                if wait_cycle > 0 and wait_cycle % 10 == 0:
                    logger.info(f"   Climbing... (current: {current_alt:.1f}m)")
            
            if not target_reached:
                logger.warning(f"Takeoff did not reach target altitude, continuing anyway...")
            
            # Start mission
            logger.info("Starting mission...")
            await self.drone.mission.start_mission()
            
            # =========================================================
            # INJECT FAULT IMMEDIATELY AFTER MISSION START
            # =========================================================
            # Since missions complete in ~10 seconds, we inject immediately
            # instead of waiting for a specific time.
            fault_injected = False
            crash_simulated = False
            
            if fault_type:
                logger.info(f"🔧 EMULATING FAILURE: {fault_type} (severity: {fault_severity})")
                try:
                    # First try native PX4 fault injection
                    injection_success = await self._trigger_px4_fault(fault_type, fault_severity)
                    if injection_success:
                        logger.info(f"✓ Fault {fault_type} injected successfully via PX4 native")
                        fault_injected = True
                        # Wait for native fault to take effect and complete
                        await asyncio.sleep(30)
                    else:
                        # Use FailureEmulator for realistic behavioral emulation
                        logger.info("⚠ Native injection unavailable - using behavioral emulation")
                        from src.simulation.failure_emulator import FailureEmulator
                        
                        emulator = FailureEmulator(self.drone)
                        emulation_result = await emulator.emulate(fault_type, fault_severity)
                        
                        if emulation_result.success:
                            logger.info(f"✅ Failure emulated: {emulation_result.method}")
                            logger.info(f"   Phases: {' → '.join(emulation_result.phases_completed)}")
                            logger.info(f"   Observation duration: {emulation_result.observation_duration:.1f}s")
                            fault_injected = True
                            # Emulator handles its own timing and landing
                        else:
                            logger.warning(f"⚠ Emulation failed: {emulation_result.error}")
                            logger.info("   Falling back to controlled landing")
                            await self.drone.action.land()
                            await asyncio.sleep(15)
                            fault_injected = True
                    
                except Exception as fault_err:
                    logger.error(f"Fault emulation failed: {fault_err}")
                    # Graceful fallback - land instead of crash
                    try:
                        await self.drone.action.land()
                        await asyncio.sleep(15)
                    except Exception:
                        pass
                    fault_injected = True  # Mark as done to avoid retry
            else:
                # No fault - wait for mission to complete
                logger.info("🎯 No fault injection - normal mission execution")
            
            # =========================================================
            # MONITOR MISSION WITH TIMEOUT
            # =========================================================
            # NOTE: If fault was injected, the emulator already handled landing
            # Skip mission monitoring to avoid blocking on mission_progress()
            if fault_injected:
                logger.info("🔧 Fault injection complete - skipping mission monitor")
            else:
                mission_start_time = time.time()
                mission_timeout = 120  # Max 120 seconds for mission
                
                while True:
                    elapsed = time.time() - mission_start_time
                    
                    # Check mission timeout
                    if elapsed > mission_timeout:
                        logger.warning(f"⏰ Mission timeout ({mission_timeout}s reached)")
                        break
                    
                    # Check mission progress (if fault wasn't already injected and we're crashing)
                    if crash_simulated:
                        # Fault was injected and crash initiated - exit loop
                        break
                    
                    # Check mission progress
                    mission_complete = False
                    try:
                        async for progress in self.drone.mission.mission_progress():
                            if progress.current > 0:
                                logger.info(f"  Waypoint {progress.current}/{progress.total}")
                            if progress.current >= progress.total:
                                logger.info("✓ All waypoints reached")
                                mission_complete = True
                                break
                            break  # Only check once per loop iteration
                    except Exception:
                        pass  # Mission progress may fail during fault
                    
                    # Exit loop if mission is complete
                    if mission_complete:
                        break
                    
                    await asyncio.sleep(1)  # Check every second
            
            # Capture final telemetry
            
            # Land (if not already crashed)
            try:
                logger.info("Landing...")
                await self.drone.action.land()
                await asyncio.sleep(10)
            except Exception as land_error:
                logger.warning(f"Landing failed (vehicle may have crashed): {land_error}")
            
            # Disarm
            try:
                logger.info("Disarming...")
                await self.drone.action.disarm()
            except Exception as disarm_error:
                logger.warning(f"Disarm failed: {disarm_error}")
            
            self.stop_telemetry_capture()
            
            logger.info(f"✓ Mission complete. Captured {len(self.telemetry_data)} telemetry points")
            return True, self.telemetry_data
            
        except Exception as e:
            logger.error(f"Mission execution failed: {e}")
            self.stop_telemetry_capture()
            
            # Emergency land
            try:
                await self.drone.action.return_to_launch()
            except:
                pass
            
            return False, self.telemetry_data
    
    async def _inject_fault_delayed(self, fault_type: str, delay_sec: int, severity: float = 1.0):
        """
        Inject fault after specified delay.
        
        This runs as a background task during mission execution.
        
        Args:
            fault_type: Type of fault (motor_failure, gps_loss, battery_failure, etc.)
            delay_sec: Seconds to wait before injecting fault
            severity: Fault severity 0.0-1.0
        """
        try:
            logger.info(f"⏱️ Fault injection countdown: waiting {delay_sec}s...")
            await asyncio.sleep(delay_sec)
            
            logger.info(f"💥 INJECTING FAULT: {fault_type} (severity: {severity})")
            injection_result = await self._trigger_px4_fault(fault_type, severity)
            
            if injection_result is True:
                logger.info(f"✓ Fault {fault_type} injected successfully via PX4 native")
            elif injection_result is None:
                # Behavioral violation - no hardware fault to inject
                logger.info(f"✓ Fault {fault_type} is behavioral - normal flight (no hardware injection)")
            else:
                # injection_result is False - actual injection failed
                logger.warning(f"⚠ Actual {fault_type} injection failed - fallback to crash simulation")
            
        except asyncio.CancelledError:
            logger.info("Fault injection cancelled (mission ended)")
        except Exception as e:
            logger.error(f"Fault injection failed: {e}")
    
    async def _trigger_px4_fault(self, fault_type: str, severity: float = 1.0):
        """
        Trigger actual PX4 fault using MAVSDK failure module.
        
        PX4 requires:
        1. SYS_FAILURE_EN=1 to enable failure injection
        2. Use drone.failure.inject() with FailureUnit and FailureType
        
        FailureUnit options: SENSOR_GYRO, SENSOR_ACCEL, SENSOR_MAG, SENSOR_BARO, 
                             SENSOR_GPS, SENSOR_OPTICAL_FLOW, SENSOR_VIO, 
                             SENSOR_DISTANCE_SENSOR, SENSOR_AIRSPEED, 
                             SYSTEM_BATTERY, SYSTEM_MOTOR, SYSTEM_SERVO,
                             SYSTEM_AVOIDANCE, SYSTEM_RC_SIGNAL, SYSTEM_MAVLINK_SIGNAL
        
        FailureType options: OK, OFF, STUCK, GARBAGE, WRONG, SLOW, DELAYED, INTERMITTENT
        
        Args:
            fault_type: Normalized fault type from LLM
            severity: 0.0-1.0 severity (1.0 = complete failure = OFF)
        """
        from mavsdk.failure import FailureUnit, FailureType
        
        fault_type_lower = fault_type.lower().replace("-", "_").replace(" ", "_")
        
        try:
            # Step 1: Enable failure injection in PX4
            # Note: This must be done BEFORE attempting to inject failures
            logger.info("   → Enabling PX4 failure injection (SYS_FAILURE_EN=1)...")
            try:
                await self.drone.param.set_param_int("SYS_FAILURE_EN", 1)
                logger.info("   ✓ SYS_FAILURE_EN enabled")
                # Wait for parameter to propagate (important for MAVSDK timing)
                await asyncio.sleep(1.0)
            except Exception as e:
                logger.warning(f"   ⚠ Could not set SYS_FAILURE_EN (may already be enabled): {e}")
            
            # Step 2: Determine failure type based on severity
            # severity 1.0 = OFF (complete failure)
            # severity 0.5 = INTERMITTENT (partial failure)
            # severity < 0.5 = STUCK (sensor gives fixed value)
            if severity >= 0.8:
                failure_type = FailureType.OFF
                failure_type_name = "OFF"
            elif severity >= 0.5:
                failure_type = FailureType.INTERMITTENT
                failure_type_name = "INTERMITTENT"
            else:
                failure_type = FailureType.STUCK
                failure_type_name = "STUCK"
            
            # Step 3: Map fault types to MAVSDK FailureUnit
            if "motor" in fault_type_lower or "propulsion" in fault_type_lower or "engine" in fault_type_lower:
                failure_unit = FailureUnit.SYSTEM_MOTOR
                unit_name = "MOTOR"
                
            elif "gps" in fault_type_lower or "navigation" in fault_type_lower:
                failure_unit = FailureUnit.SENSOR_GPS
                unit_name = "GPS"
                
            elif "battery" in fault_type_lower or "power" in fault_type_lower:
                failure_unit = FailureUnit.SYSTEM_BATTERY
                unit_name = "BATTERY"
                
            elif "control" in fault_type_lower or "servo" in fault_type_lower:
                # X500 multicopter has NO servos - only motors
                # SYSTEM_SERVO will timeout because no servo actuator exists
                # Map to SYSTEM_MOTOR which represents the actual actuators
                failure_unit = FailureUnit.SYSTEM_MOTOR
                unit_name = "MOTOR"
                
            elif "gyro" in fault_type_lower:
                failure_unit = FailureUnit.SENSOR_GYRO
                unit_name = "GYRO"
                
            elif "accel" in fault_type_lower:
                failure_unit = FailureUnit.SENSOR_ACCEL
                unit_name = "ACCEL"
                
            elif "mag" in fault_type_lower or "compass" in fault_type_lower:
                failure_unit = FailureUnit.SENSOR_MAG
                unit_name = "MAG"
                
            # IMPORTANT: Check for behavioral violations BEFORE hardware mappings
            # This must come before "altitude" check since "altitude" is substring of "altitude_violation"
            elif any(kw in fault_type_lower for kw in ["geofence", "airspace", "altitude_violation", "none", "unknown"]):
                # Behavioral violations - NO hardware fault injection
                # These are operational/procedural issues, not hardware failures
                logger.info(f"   ⚠ Fault type '{fault_type}' is a behavioral violation (not hardware)")
                logger.info(f"   → Skipping hardware fault injection - running normal flight")
                return None  # Signal: no fault injected (not a failure, just not applicable)
                
            elif ("sensor" in fault_type_lower or "baro" in fault_type_lower) or ("altitude" in fault_type_lower and "violation" not in fault_type_lower):
                # Only map to BARO if it's an altitude SENSOR issue, not a violation
                failure_unit = FailureUnit.SENSOR_BARO
                unit_name = "BARO"
                
            elif "rc" in fault_type_lower or "remote" in fault_type_lower:
                failure_unit = FailureUnit.SYSTEM_RC_SIGNAL
                unit_name = "RC_SIGNAL"
                
            else:
                # Default to motor failure for unknown hardware-related types
                logger.warning(f"   Unknown fault type '{fault_type}', defaulting to MOTOR failure")
                failure_unit = FailureUnit.SYSTEM_MOTOR
                unit_name = "MOTOR"
            
            # Step 4: Inject the failure with retry
            # instance=0 means first instance of this unit type
            logger.info(f"   → Injecting {unit_name} failure ({failure_type_name})...")
            
            # Retry logic - some Gazebo setups need multiple attempts
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    await self.drone.failure.inject(failure_unit, failure_type, 0)
                    logger.info(f"   ✓ Failure injected: {unit_name} = {failure_type_name}")
                    return True  # Success!
                except Exception as inject_error:
                    if attempt < max_retries - 1:
                        logger.info(f"   ⏳ Retry {attempt + 2}/{max_retries}...")
                        await asyncio.sleep(1.0)  # Wait before retry
                    else:
                        # All retries failed
                        logger.warning(f"   ⚠ MAVSDK failure.inject() failed after {max_retries} attempts: {inject_error}")
                        logger.warning(f"   ⚠ Gazebo Harmonic (gz_x500) may not fully support failure injection")
                        logger.warning(f"   ⚠ Will fallback to crash simulation instead")
                        return False  # Signal that actual fault injection failed
                
        except Exception as e:
            logger.error(f"Failed to inject PX4 fault: {e}")
            return False  # Signal failure
        
        return True  # Signal success


# =============================================================================
# Automated Pipeline
# =============================================================================

class AutomatedPipeline:
    """Fully automated AeroGuardian pipeline."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.wsl = WSLController(config)
        self.executor = None
        
    def run(self, incident_index: int = 0, skip_px4: bool = False) -> Dict[str, Path]:
        """Run the full automated pipeline."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("  AEROGUARDIAN AUTOMATED PIPELINE")
        logger.info("=" * 70)
        logger.info(f"  Incident Index: {incident_index}")
        logger.info(f"  QGC Connection: {self.config.wsl_ip}:{self.config.qgc_port}")
        logger.info(f"  Headless Mode: {self.config.headless}")
        logger.info("=" * 70)
        logger.info("")
        
        try:
            # Step 0: Launch QGroundControl (if not already running)
            logger.info("")
            logger.info("Launching QGroundControl...")
            self.wsl.launch_qgroundcontrol()
            
            # Step 1: Load FAA incident (do first to get location for PX4 home)
            self._step_header(1, "Load FAA Incident")
            incident = self._load_incident(incident_index)
            logger.info(f"  ID: {incident.get('report_id', incident.get('incident_id', 'Unknown'))}")
            logger.info(f"  Location: {incident.get('city', 'Unknown')}, {incident.get('state', 'Unknown')}")
            logger.info(f"  Type: {incident.get('fault_type', incident.get('hazard_category', 'Unknown'))}")
            
            # Step 2: Generate LLM configuration (before PX4 to get home location)
            self._step_header(2, "Generate LLM Configuration")
            flight_config = self._generate_config(incident)
            fault_type = flight_config.get('fault_injection', 'none').get('fault_type', 'none')
            logger.info(f"  Fault Type: {fault_type}")
            logger.info(f"  Waypoints: {len(flight_config.get('waypoints', []))}")
            
            # Copy home location to WSL controller for PX4 startup
            if hasattr(self, '_home_lat'):
                self.wsl._home_lat = self._home_lat
                self.wsl._home_lon = self._home_lon
                self.wsl._home_alt = self._home_alt
            
            # Step 3: Start PX4 (with home location from incident)
            self._step_header(3, "PX4 SITL Initialization")
            if skip_px4:
                logger.info("Skipping PX4 startup (--skip-px4 flag)")
            else:
                if not self.wsl.start_px4_gazebo():
                    raise RuntimeError("Failed to start PX4 SITL")
            
            # Step 4: Execute mission
            self._step_header(4, "Execute Flight Mission")
            success, telemetry = asyncio.run(self._execute_mission(flight_config))
            logger.info(f"  Mission Success: {success}")
            logger.info(f"  Telemetry Points: {len(telemetry)}")
            
            self._step_header(5, "Generate Safety Report")
            safety_report = self._generate_safety_report(incident, flight_config, telemetry)
            logger.info(f"  Hazard Level: {safety_report.get('safety_level', 'UNKNOWN')}")
            logger.info(f"  Recommendation: {safety_report.get('verdict', 'REVIEW')}")
            
            # Step 6: Save reports
            self._step_header(6, "Save Reports")
            paths = self._save_reports(incident, flight_config, telemetry, safety_report)
            
            # Summary
            self._print_summary(incident, safety_report, len(telemetry), paths)
            
            return paths
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            if not skip_px4:
                self.wsl.stop_px4()
    
    def run_from_incident(self, incident: Dict, skip_px4: bool = False) -> Dict[str, Path]:
        """Run the pipeline for a specific incident dict (for batch processing)."""
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  PROCESSING: {incident.get('report_id', incident.get('incident_id', 'Unknown'))}")
        logger.info("=" * 60)
        logger.info(f"  Location: {incident.get('city', 'Unknown')}, {incident.get('state', '')}")
        logger.info(f"  Type: {incident.get('fault_type', incident.get('incident_type', 'other'))}")
        logger.info("")
        
        try:
            # Step 1: Generate LLM configuration
            self._step_header(1, "Generate LLM Configuration")
            flight_config = self._generate_config(incident)
            logger.info(f"  Fault Type: {flight_config.get('fault_type', 'none')}")
            
            # Copy home location to WSL controller
            if hasattr(self, '_home_lat'):
                self.wsl._home_lat = self._home_lat
                self.wsl._home_lon = self._home_lon
                self.wsl._home_alt = self._home_alt
            
            # Step 2: Start PX4 (if not skipping)
            self._step_header(2, "PX4 SITL Initialization")
            if skip_px4:
                logger.info("Skipping PX4 startup")
            else:
                if not self.wsl.start_px4_gazebo():
                    raise RuntimeError("Failed to start PX4 SITL")
            
            # Step 3: Execute mission
            self._step_header(3, "Execute Flight Mission")
            success, telemetry = asyncio.run(self._execute_mission(flight_config))
            logger.info(f"  Mission Success: {success}")
            logger.info(f"  Telemetry Points: {len(telemetry)}")
            
            # Step 4: Generate safety report
            self._step_header(4, "Generate Safety Report")
            safety_report = self._generate_safety_report(incident, flight_config, telemetry)
            logger.info(f"  Verdict: {safety_report.get('verdict', 'REVIEW')}")
            
            # Step 5: Save reports
            self._step_header(5, "Save Reports")
            paths = self._save_reports(incident, flight_config, telemetry, safety_report)
            
            # Summary
            logger.info("")
            logger.info("=" * 40)
            logger.info(f"✓ COMPLETE: {incident.get('report_id', incident.get('incident_id', 'Unknown'))}")
            logger.info(f"  Output: {paths.get('report_dir', 'N/A')}")
            logger.info("=" * 40)
            
            return paths
            
        except Exception as e:
            logger.error(f"Failed: {e}")
            raise
        finally:
            if not skip_px4:
                self.wsl.stop_px4()
    
    def _step_header(self, num: int, name: str):
        """Print step header."""
        logger.info("")
        logger.info("-" * 60)
        logger.info(f"  STEP {num}: {name}")
        logger.info("-" * 60)
    
    def _load_incident(self, index: int) -> Dict:
        """Load FAA sighting by index."""
        from src.faa.sighting_filter import get_sighting_filter
        
        sighting_filter = get_sighting_filter(data_source=self.config.data_source)
        count = sighting_filter.load()
        logger.info(f"  Available sightings: {count}")
        
        sighting = sighting_filter.get_by_index(index)
        return sighting
    
    def _generate_config(self, incident: Dict) -> Dict:
        """Generate flight configuration using LLM."""
        from src.core.geocoder import geocode_incident
        from src.llm.client import get_llm_client
        
        # Geocode location
        lat, lon = geocode_incident(incident)
        logger.info(f"  Geocoded: ({lat:.4f}, {lon:.4f})")
        
        # Store home location for PX4 (use incident location)
        self._home_lat = lat
        self._home_lon = lon
        self._home_alt = 0
        
        # Generate config using LLM
        client = get_llm_client()
        # Enable LLM logging to output directory
        client.set_output_dir(str(self.config.output_dir))
        config = client.generate_full_px4_config(
            incident_description=incident.get("description", incident.get("summary", "")),
            incident_location=f"{incident.get('city', '')}, {incident.get('state', '')}",
            incident_type=incident.get("incident_type", "unknown"),
            report_id=incident.get("report_id", incident.get("incident_id", "Unknown")),
        )
        
        # Log simulation mode from incident filter (MECHANICAL_TEST / AIRSPACE_SIGHTING)
        sim_mode = incident.get("simulation_mode", "MECHANICAL_TEST")
        if sim_mode == "AIRSPACE_SIGHTING":
            logger.warning(f"  ⚠ High-altitude incident detected: simulation capped to drone-realistic altitude")
            logger.info(f"  Extracted altitude: {incident.get('extracted_altitude_m', 0):.0f}m ({incident.get('altitude_source', 'unknown')})")
        
        # Extract altitude from LLM config
        mission_config = config.get("mission", {})
        llm_takeoff_alt = mission_config.get("takeoff_altitude_m")
        llm_cruise_alt = mission_config.get("cruise_altitude_m")
        
        # Use simulatable_altitude_m from incident filter (capped to 120m for high-altitude incidents)
        incident_alt = incident.get("simulatable_altitude_m", 50.0)
        
        # Prefer LLM altitude if valid (>0 and <200m), otherwise use incident filter's capped value
        if llm_takeoff_alt is not None and llm_takeoff_alt > 0 and llm_takeoff_alt < 200:
            takeoff_alt = llm_takeoff_alt
        else:
            takeoff_alt = min(incident_alt, 120.0) if incident_alt > 0 else 50.0  # Default to 50m if invalid
        
        if llm_cruise_alt is not None and llm_cruise_alt > 0 and llm_cruise_alt < 200:
            cruise_alt = llm_cruise_alt
        else:
            cruise_alt = min(incident_alt, 120.0) if incident_alt > 0 else 50.0
        
        # Speed: use LLM config or sensible default
        speed = mission_config.get("speed_m_s")
        if speed is None:
            speed = 5.0  # Default drone cruise speed
            logger.info(f"  Using default speed: {speed} m/s")
        
        # Store speed in config for mission execution
        config["speed_m_s"] = speed
        
        # =================================================================
        # COMPACT WAYPOINT PATTERN for 120s mission duration
        # =================================================================
        # At 8 m/s:
        #   - 100m offset = ~12.5s per segment
        #   - 4 segments = ~50s total flight
        #   - Leaves ~70s for fault effects and landing
        #
        # Distance conversion: ~111m per 0.001 degree lat (at equator)
        # 100m = 0.0009 degrees approximately
        offset_deg = 0.0009  # ~100m offset (was 0.0036 = 400m)
        
        # Create compact waypoint pattern around incident location
        config["waypoints"] = [
            # Home/Launch point (at incident location)
            {"lat": lat, "lon": lon, "alt": takeoff_alt, "action": "takeoff"},
            # Waypoint 1: 100m North
            {"lat": lat + offset_deg, "lon": lon, "alt": cruise_alt, "action": "waypoint"},
            # Waypoint 2: 100m NE (fault typically triggers here at T+60s)
            {"lat": lat + offset_deg, "lon": lon + offset_deg, "alt": cruise_alt, "action": "waypoint"},
            # Waypoint 3: 100m East
            {"lat": lat, "lon": lon + offset_deg, "alt": cruise_alt, "action": "waypoint"},
            # Return and land at home
            {"lat": lat, "lon": lon, "alt": takeoff_alt, "action": "land"},
        ]
        
        # CRITICAL: Store capped altitudes back to config so mission executor uses them
        if "mission" not in config:
            config["mission"] = {}
        config["mission"]["takeoff_altitude_m"] = takeoff_alt
        config["mission"]["cruise_altitude_m"] = cruise_alt
        
        logger.info(f"  Home location: ({lat:.4f}, {lon:.4f})")
        logger.info(f"  Altitude: takeoff={takeoff_alt}m, cruise={cruise_alt}m")
        logger.info(f"  Speed: {speed} m/s")
        
        return config
    
    async def _execute_mission(self, config: Dict) -> Tuple[bool, List[Dict]]:
        """Execute flight mission with fault injection support."""
        self.executor = MissionExecutor(self.config)
        
        if not await self.executor.connect():
            return False, []
        
        waypoints = config.get("waypoints", [])
        
        # Extract mission parameters from config - NO DEFAULTS
        mission = config.get("mission", {})
        takeoff_alt = mission.get("takeoff_altitude_m")
        speed_m_s = config.get("speed_m_s") or mission.get("speed_m_s")
        
        if takeoff_alt is None:
            raise ValueError("Config missing mission.takeoff_altitude_m")
        if speed_m_s is None:
            raise ValueError("Config missing speed_m_s")
        
        # Extract fault injection parameters
        fault_config = config.get("fault_injection", {})
        fault_type = fault_config.get("fault_type", None)
        fault_onset_sec = fault_config.get("onset_sec", 60)  # Optimized by faa_scenario_generator
        fault_severity = fault_config.get("severity", 1.0)
        
        logger.info(f"  Executing mission: alt={takeoff_alt}m, speed={speed_m_s}m/s, waypoints={len(waypoints)}")
        if fault_type:
            logger.info(f"  🎯 Fault injection: {fault_type} at T+{fault_onset_sec}s (severity: {fault_severity})")
        
        return await self.executor.execute_mission(
            waypoints=waypoints,
            takeoff_alt=takeoff_alt,
            speed_m_s=speed_m_s,
            fault_type=fault_type,
            fault_onset_sec=fault_onset_sec,
            fault_severity=fault_severity
        )
    
    def _generate_safety_report(self, incident: Dict, config: Dict, telemetry: List[Dict]) -> Dict:
        """Generate safety report using LLM."""
        from src.llm.client import get_llm_client
        from src.analysis.telemetry_analyzer import TelemetryAnalyzer
        
        client = get_llm_client()
        analyzer = TelemetryAnalyzer()
        
        # Analyze telemetry - comprehensive analysis
        stats = analyzer.analyze(telemetry)
        
        # Use the new comprehensive structured summary
        telemetry_text = stats.to_summary_text()
        
        # Build simulation params from config - use actual values, no defaults
        waypoints = config.get("waypoints", [])
        mission = config.get("mission", {})
        altitude = waypoints[0].get('alt') if waypoints else mission.get("cruise_altitude_m", "N/A")
        speed = config.get("speed_m_s") or mission.get("speed_m_s", "N/A")
        
        # Extract expected outcome from FAA source in config
        faa_source = config.get("faa_source", {})
        expected_outcome = faa_source.get("outcome", "unknown")
        
        # Extract fault_type from correct nested path
        fault_injection = config.get("fault_injection", {})
        fault_type = fault_injection.get("fault_type", "unknown")
        
        sim_params = f"""Waypoints: {len(waypoints)}
Altitude: {altitude}m
Speed: {speed} m/s
Behavior: {fault_type} simulation"""
        
        # Generate report with new 6-section format
        return client.generate_preflight_report(
            incident_description=incident.get("description", incident.get("summary", "")),
            report_id=incident.get("report_id", incident.get("incident_id", "Unknown")),
            incident_location=f"{incident.get('city', '')}, {incident.get('state', '')}",
            incident_date=incident.get("date", "Unknown"),
            fault_type=fault_type,  # From config["fault_injection"]["fault_type"]
            expected_outcome=expected_outcome,  # From config["faa_source"]["outcome"]
            simulation_params=sim_params,
            telemetry_summary=telemetry_text,
        )
    
    def _save_reports(self, incident: Dict, config: Dict, telemetry: List[Dict], safety: Dict) -> Dict[str, Path]:
        """Save all reports."""
        from src.reporting.unified_reporter import UnifiedReporter
        
        reporter = UnifiedReporter(self.config.output_dir)
        return reporter.generate(
            incident=incident,
            flight_config=config,
            telemetry=telemetry,
            safety_analysis=safety,
        )
    
    def _print_summary(self, incident: Dict, safety: Dict, telemetry_count: int, paths: Dict):
        """Print final summary."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("  PIPELINE COMPLETE")
        logger.info("=" * 70)
        logger.info(f"  Incident: {incident.get('report_id', incident.get('incident_id', 'Unknown'))}")
        logger.info(f"  Telemetry Points: {telemetry_count}")
        logger.info(f"  Hazard Level: {safety.get('safety_level', 'UNKNOWN')}")
        logger.info(f"  Recommendation: {safety.get('verdict', 'REVIEW')}")
        logger.info("")
        logger.info("  Output Files:")
        for key, path in paths.items():
            if path:
                logger.info(f"    - {key}: {path}")
        logger.info("")
        logger.info("=" * 70)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AeroGuardian Automated Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_automated_pipeline.py                   # Run with default settings
    python run_automated_pipeline.py --incident 5      # Process incident #5
    python run_automated_pipeline.py --headless        # No Gazebo GUI
    python run_automated_pipeline.py --skip-px4        # PX4 already running
    python run_automated_pipeline.py --batch data.json # Batch processing from JSON

QGroundControl Connection:
    IP:   {WSL_IP}
    Port: 18570
        """
    )
    
    parser.add_argument("--report", "-r", type=int, default=0,
                        help="FAA report index to process (default: 0)")
    parser.add_argument("--batch", "-b", type=str, default=None,
                        help="JSON file for batch processing (single object or array)")
    parser.add_argument("--headless", action="store_true",
                        help="Run without Gazebo GUI")
    parser.add_argument("--skip-px4", action="store_true",
                        help="Skip PX4 startup (assume already running)")
    parser.add_argument("--wsl-ip", type=str, default=None,
                        help="WSL2 IP address (get via: ip addr show eth0 in WSL). Required for QGC connection.")
    parser.add_argument("--qgc-port", type=int, default=18570,
                        help="QGroundControl UDP port (default: 18570)")
    parser.add_argument("--vehicle", type=str, default="iris",
                        choices=["iris", "typhoon_h480", "plane", "rover"],
                        help="PX4 vehicle type")
    parser.add_argument("--data-source", type=str, default="sightings",
                        choices=["sightings", "failures"],
                        help="Data source: 'sightings' (8000 high-risk) or 'failures' (31 confirmed crashes)")
    
    args = parser.parse_args()
    
    # Build config
    config = PipelineConfig(
        wsl_ip=args.wsl_ip if args.wsl_ip else "127.0.0.1",
        qgc_port=args.qgc_port,
        headless=args.headless,
        vehicle=args.vehicle,
        data_source=args.data_source,
    )
    
    # Run pipeline
    pipeline = AutomatedPipeline(config)
    
    try:
        # Batch mode - process from JSON file
        if args.batch:
            import json as json_module
            batch_file = Path(args.batch)
            if not batch_file.exists():
                logger.error(f"Batch file not found: {batch_file}")
                return 1
            
            with open(batch_file, "r", encoding="utf-8") as f:
                data = json_module.load(f)
            
            # Handle single object or array
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                # Check for nested array keys
                for key in ["incidents", "data", "records", "items", "results"]:
                    if key in data and isinstance(data[key], list):
                        records = data[key]
                        break
                else:
                    records = [data]
            else:
                logger.error("Invalid JSON format")
                return 1
            
            logger.info(f"Batch processing {len(records)} record(s) from {batch_file}")
            
            all_reports = []
            for idx, record in enumerate(records):
                logger.info("")
                logger.info(f"{'='*60}")
                logger.info(f"  PROCESSING RECORD {idx+1} of {len(records)}")
                logger.info(f"{'='*60}")
                
                # Create temp incident in expected format
                incident = {
                    "incident_id": record.get("incident_id", f"Batch_{idx+1}"),
                    "date": record.get("date", ""),
                    "city": record.get("city", "Unknown"),
                    "state": record.get("state", ""),
                    "description": record.get("description", record.get("summary", "")),
                    "summary": record.get("summary", record.get("description", "")),
                    "incident_type": record.get("incident_type", "other"),
                }
                
                try:
                    # Store incident temporarily for pipeline
                    pipeline._batch_incident = incident
                    paths = pipeline.run_from_incident(
                        incident=incident,
                        skip_px4=args.skip_px4 or (idx > 0)  # Skip PX4 after first
                    )
                    all_reports.append({
                        "incident_id": incident["incident_id"],
                        "output_dir": str(paths.get("report_dir", "")),
                        "status": "success",
                    })
                except Exception as e:
                    logger.error(f"Record {idx+1} failed: {e}")
                    all_reports.append({
                        "incident_id": incident["incident_id"],
                        "status": "failed",
                        "error": str(e),
                    })
            
            # Summary
            logger.info("")
            logger.info(f"{'='*60}")
            logger.info(f"  BATCH COMPLETE: {len([r for r in all_reports if r['status']=='success'])}/{len(records)} successful")
            logger.info(f"{'='*60}")
            
            # Save combined report
            batch_output = config.output_dir / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(batch_output, "w", encoding="utf-8") as f:
                json_module.dump({"reports": all_reports, "total": len(records)}, f, indent=2)
            logger.info(f"Batch summary saved to: {batch_output}")
            
            return 0
        
        # Single incident mode
        paths = pipeline.run(
            incident_index=args.report,
            skip_px4=args.skip_px4
        )
        logger.info("Pipeline completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    from datetime import datetime
    sys.exit(main())