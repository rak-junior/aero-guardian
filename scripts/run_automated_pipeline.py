"""
AeroGuardian Automated Pipeline Runner
======================================
Author: AeroGuardian Team (Tiny Coders)
Date: 2026-01-19
Updated: 2026-02-04
Version: 2.0

Fully automated pipeline that runs the complete AeroGuardian workflow:

PIPELINE STEPS:
1. Load FAA UAS sighting report from processed dataset
2. Generate PX4 simulation config via LLM #1 (GPT-4o + DSPy)
3. Start PX4 SITL with Gazebo (Harmonic or Classic) in WSL2
4. Execute flight mission with fault injection
5. Capture telemetry at 10-50Hz (IMU, GPS, motors)
6. Analyze telemetry with physics-based anomaly detection
7. Generate safety report via LLM #2 (GPT-4o + DSPy)
8. Evaluate output with ESRI framework (SFS × BRR × ECC)
9. Save reports (PDF, JSON, Excel)

SUPPORTED SIMULATORS:
- gz_x500: Gazebo Harmonic X500 quadcopter (recommended)
- gazebo-classic_iris: Gazebo Classic Iris quadcopter
- sihsim_quadx: Software-In-Hardware (no physics, fastest)

Usage:
    python scripts/run_automated_pipeline.py -r 0 --wsl-ip 172.x.x.x --headless -s gz_x500
    python scripts/run_automated_pipeline.py -r 5 --wsl-ip 172.x.x.x -s gazebo-classic_iris
    python scripts/run_automated_pipeline.py --skip-px4  # If PX4 already running
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
    
    # Simulator selection
    # Options: "auto", "sihsim_quadx", "gz_x500", "gazebo-classic_iris"
    # auto: Uses sihsim_quadx for headless, gz_x500 for GUI
    # sihsim_quadx: SIH simulator - fast, headless only, no failure injection
    # gz_x500: Gazebo Harmonic - GUI, limited failure injection on WSL2
    # gazebo-classic_iris: Gazebo Classic - GUI, FULL failure injection (requires submodule init)
    simulator: str = "auto"
    
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
        """
        Check if PX4 SITL is actually running (not zombie/suspended).
        
        Uses ps to check process state - only 'R' (running) or 'S' (sleeping) are valid.
        Processes in 'T' (stopped) or 'Z' (zombie) state are not functional.
        """
        # Check for running px4 process using awk to filter by STAT column (column 8)
        # STAT column shows: R=running, S=sleeping, T=stopped, Z=zombie
        # We want only R or S states (start with R or S, may have modifiers like Ss, S+)
        code, output = self.run_wsl(
            "ps aux | grep -E 'px4.*sitl|bin/px4' | grep -v grep | awk '$8 ~ /^[RS]/ {print; exit}'",
            timeout=5
        )
        if code == 0 and output.strip():
            logger.info(f"Found running PX4 process: {output.strip()[:100]}")
            return True
        
        # Also check if there are zombie/stopped processes (for logging)
        code2, output2 = self.run_wsl(
            "ps aux | grep -E 'px4.*sitl|bin/px4' | grep -v grep | awk '$8 ~ /^[TZ]/ {print}'",
            timeout=5
        )
        if code2 == 0 and output2.strip():
            logger.warning(f"Found zombie/stopped PX4 (will clean up): {output2.strip()[:80]}")
        
        return False
    
    def cleanup_stale_processes(self):
        """Kill any zombie/suspended PX4/Gazebo processes."""
        logger.debug("Cleaning up stale simulation processes...")
        self.run_wsl("pkill -9 px4 2>/dev/null; pkill -9 gz 2>/dev/null; pkill -9 ruby 2>/dev/null", timeout=5)
        time.sleep(1)
    
    def _validate_simulator_target(self, requested: str) -> str:
        """
        Validate simulator target and return a valid alternative if needed.
        
        PX4 simulator targets:
        - Modern PX4 (v1.14+): gz_x500, sihsim_quadx
        - Legacy: gazebo-classic_iris (requires gazebo-classic submodule)
        
        Returns the best available simulator target.
        """
        # Known valid targets in order of preference for fault injection
        VALID_TARGETS = [
            "gz_x500",           # Gazebo Harmonic - best for modern setups
            "sihsim_quadx",      # SIH - always works, limited fault injection
        ]
        
        # Mapping from legacy names to valid targets
        LEGACY_MAPPING = {
            "gazebo-classic_iris": "gz_x500",
            "gazebo_iris": "gz_x500",
            "gazebo-classic": "gz_x500",
            "jmavsim": "sihsim_quadx",
        }
        
        # If requesting a known legacy target, map it
        if requested in LEGACY_MAPPING:
            mapped = LEGACY_MAPPING[requested]
            logger.warning(f"  Simulator '{requested}' not available in your PX4 build")
            logger.warning(f"  → Using '{mapped}' instead (Gazebo Harmonic)")
            return mapped
        
        # Check if requested target is in the valid list
        if requested in VALID_TARGETS:
            return requested
        
        # For unknown targets, try to validate with a quick build test
        logger.info(f"  Validating simulator target: {requested}")
        code, output = self.run_wsl(
            f"cd ~/PX4-Autopilot && ninja -t targets 2>/dev/null | grep -q '{requested}' && echo 'valid' || echo 'invalid'",
            timeout=15
        )
        
        if "valid" in output:
            return requested
        
        # Unknown and invalid - fall back to gz_x500
        logger.warning(f"  Simulator '{requested}' not found in PX4 build")
        logger.warning(f"  → Falling back to 'gz_x500'")
        return "gz_x500"
    
    def start_px4_gazebo(self) -> bool:
        """Start PX4 SITL with Gazebo."""
        # First cleanup any stale processes
        self.cleanup_stale_processes()
        
        if self.is_px4_running():
            logger.info("PX4 SITL is already running")
            return True
        
        logger.info(f"Starting PX4 SITL...")
        
        # Get Windows IP for DISPLAY (in case Gazebo is needed)
        windows_ip = self.get_windows_ip()
        
        # Choose simulator target based on config or auto-select
        # Simulator options:
        # - sihsim_quadx: Software-in-the-loop, no graphics (headless), no failure injection
        # - gazebo-classic_iris: Gazebo Classic, FULL failure injection (requires submodule init)
        # - gz_x500: Gazebo Harmonic, limited failure injection on WSL2
        if self.config.simulator == "auto":
            if self.config.headless:
                sim_target = "sihsim_quadx"  # SIH simulator - no graphics, fast
            else:
                sim_target = "gz_x500"  # Gazebo Harmonic - GUI with X500 drone
        else:
            # Validate user-specified simulator and provide fallback
            sim_target = self._validate_simulator_target(self.config.simulator)
        
        logger.info(f"  Simulator: {sim_target}")
        
        # Get home location if set (from incident geocoding)
        home_lat = getattr(self, '_home_lat', 47.397742)  # Default: Switzerland test location
        home_lon = getattr(self, '_home_lon', 8.545594)
        home_alt = getattr(self, '_home_alt', 488.0)
        
        # For WSLg, use :0 directly; for VcXsrv use windows_ip:0
        # WSLg is default on Windows 11
        display_env = ":0"  # WSLg default
        
        # Gazebo Harmonic (gz_x500) requires additional setup
        is_gazebo_harmonic = "gz_x500" in sim_target or "gz_" in sim_target
        headless_arg = "" if not self.config.headless else "HEADLESS=1"
        
        if is_gazebo_harmonic:
            logger.info("  Mode: Gazebo Harmonic (gz sim)")
            if self.config.headless:
                logger.info("  GUI: Disabled (headless mode)")
        
        launch_cmd = f"""
            pkill -9 px4 2>/dev/null || true
            pkill -9 gz 2>/dev/null || true
            pkill -9 ruby 2>/dev/null || true
            sleep 2
            cd {self.config.px4_dir}
            
            # Source environment
            source ~/.bashrc 2>/dev/null || true
            source /opt/ros/*/setup.bash 2>/dev/null || true
            
            # Display for Gazebo GUI (WSLg on Windows 11)
            export DISPLAY={display_env}
            export LIBGL_ALWAYS_SOFTWARE=1
            
            # Gazebo Harmonic specific environment (for gz_x500)
            export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/PX4-Autopilot/build/px4_sitl_default/build_gz_plugins
            export GZ_SIM_RESOURCE_PATH=$HOME/PX4-Autopilot/Tools/simulation/gz/models:$GZ_SIM_RESOURCE_PATH
            export GZ_VERSION=harmonic
            
            # PX4 configuration
            export PX4_SIM_HOST_ADDR={self.config.wsl_ip}
            export PX4_HOME_LAT={home_lat}
            export PX4_HOME_LON={home_lon}
            export PX4_HOME_ALT={home_alt}
            
            # ENABLE FAILURE INJECTION for PX4
            # This allows MAVSDK failure.inject() and shell 'failure' commands to work
            export PX4_SYS_FAILURE_EN=1
            
            # Set Gazebo world (default, baylands, etc.)
            export PX4_GZ_WORLD=default
            
            # Gazebo model paths (for both Classic and Harmonic)
            export GAZEBO_MODEL_PATH=$HOME/PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models:$GAZEBO_MODEL_PATH
            
            # Headless mode for Gazebo (no GUI)
            {'export HEADLESS=1' if self.config.headless else ''}
            
            # Launch PX4 with simulator
            cd $HOME/PX4-Autopilot && {headless_arg} make px4_sitl {sim_target} 2>&1
        """
        
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
        # Gazebo Harmonic takes longer to start than SIH simulator
        effective_timeout = self.config.px4_startup_timeout
        if is_gazebo_harmonic:
            effective_timeout = max(180, self.config.px4_startup_timeout)  # 180s for Gazebo
            logger.info(f"Waiting for PX4 SITL to initialize (Gazebo timeout: {effective_timeout}s)...")
        else:
            logger.info("Waiting for PX4 SITL to initialize...")
        start_time = time.time()
        output_lines = []
        
        while time.time() - start_time < effective_timeout:
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
        # Track fault injection status for reporting
        self.fault_injection_success = False  # True if native PX4 fault injection worked
        self.fault_injection_mode = "none"    # "native", "emulated", "fallback", or "none"
    
    async def connect(self, max_retries: int = 3, retry_delay: float = 5.0) -> bool:
        """
        Connect to PX4 via MAVSDK with retry logic.
        
        Args:
            max_retries: Maximum connection attempts before giving up
            retry_delay: Seconds to wait between retry attempts
            
        Returns:
            True if connected, False if all retries failed
        """
        # Validate IP first
        if "x.x" in self.config.wsl_ip or "[" in self.config.wsl_ip:
            logger.error(f"Invalid WSL IP detected: {self.config.wsl_ip}. Please configure correct IP.")
            return False

        # Pre-check connectivity with ping (Windows)
        try:
            # -n 1 = 1 count, -w 1000 = 1000ms timeout
            ping_cmd = ["ping", "-n", "1", "-w", "1000", self.config.wsl_ip]
            if subprocess.run(ping_cmd, capture_output=True).returncode != 0:
                logger.error(f"Host {self.config.wsl_ip} is unreachable via ping. Aborting connection.")
                return False
        except Exception:
            pass # Ignore ping errors (e.g. permission), let MAVSDK try

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
                    if timeout_counter > 10:  # REDUCED: 10 second timeout for faster feedback
                        raise TimeoutError("Connection state timeout")
                    await asyncio.sleep(1)
                
                if not connected:
                    raise ConnectionError("Failed to establish connection")
                
                # Wait for GPS fix with timeout
                # NOTE: Gazebo Harmonic on WSL2 may need longer GPS initialization
                gps_max_timeout = 60 if not self.config.headless else 30  # 60s for Gazebo, 30s for SIH
                logger.info(f"Waiting for GPS fix (timeout: {gps_max_timeout}s)...")
                gps_timeout = 0
                async for health in self.drone.telemetry.health():
                    if health.is_global_position_ok:
                        logger.info("✓ GPS fix acquired")
                        return True
                    gps_timeout += 1
                    if gps_timeout % 10 == 0:
                        logger.info(f"  Still waiting for GPS... ({gps_timeout}s)")
                    if gps_timeout > gps_max_timeout:
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
                    # If it was a generic "Connection failed" (likely IP issue), 
                    # fail fast instead of retrying 3 times if we are sure it's dead
                    if "Connection failed" in str(e) or "Destination IP unknown" in str(e):
                        logger.error("Fatal connection error (Invalid IP?). Stopping retries.")
                        return False

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
        
        # State variables for all streams
        self._current_position = None
        self._current_attitude = None
        self._current_battery = None
        self._current_actuator = None
        self._current_gps_info = None
        self._current_velocity = None
        self._current_imu = None
        
        self._capture_start_time = time.time()
        
        # --- Telemetry Stream Coroutines ---
        
        async def position_capture():
            async for pos in self.drone.telemetry.position():
                if not self._capturing: break
                self._current_position = pos
                
        async def attitude_capture():
            async for att in self.drone.telemetry.attitude_euler():
                if not self._capturing: break
                self._current_attitude = att
                
        async def battery_capture():
            async for bat in self.drone.telemetry.battery():
                if not self._capturing: break
                self._current_battery = bat

        async def actuator_capture():
            try:
                # Explicitly request the stream at 20Hz
                await self.drone.telemetry.set_rate_actuator_output_status(20.0)
                logger.info("Subscribed to actuator_output_status at 20Hz")
            except Exception as e:
                logger.warning(f"Failed to set actuator rate: {e}")

            async for act in self.drone.telemetry.actuator_output_status():
                if not self._capturing: break
                self._current_actuator = act
                
        async def gps_info_capture():
            try:
                await self.drone.telemetry.set_rate_gps_info(5.0) # 5Hz enough for status
            except Exception as e:
                logger.warning(f"Failed to set gps_info rate: {e}")
                
            async for gps in self.drone.telemetry.gps_info():
                if not self._capturing: break
                self._current_gps_info = gps
                
        async def velocity_capture():
            try:
                await self.drone.telemetry.set_rate_velocity_ned(20.0)
            except Exception as e:
                logger.warning(f"Failed to set velocity_ned rate: {e}")
                
            async for vel in self.drone.telemetry.velocity_ned():
                if not self._capturing: break
                self._current_velocity = vel
                
        async def imu_capture():
            try:
                # IMU needs high rate for vibration/noise analysis
                await self.drone.telemetry.set_rate_imu(50.0)
                logger.info("Subscribed to imu at 50Hz")
            except Exception as e:
                # Fallback if 50Hz fails
                try: 
                    await self.drone.telemetry.set_rate_imu(20.0)
                except: 
                    pass
                logger.warning(f"Failed to set imu rate (attempted 50Hz): {e}")

            async for imu in self.drone.telemetry.imu():
                if not self._capturing: break
                self._current_imu = imu
        
        async def data_recorder():
            """Record combined telemetry at fixed rate."""
            while self._capturing:
                # Create base record with timestamp
                record = {
                    "timestamp": time.time() - self._capture_start_time,
                }
                
                # --- Position ---
                if self._current_position:
                    record.update({
                        "lat": self._current_position.latitude_deg,
                        "lon": self._current_position.longitude_deg,
                        "alt": self._current_position.absolute_altitude_m,
                        "relative_alt": self._current_position.relative_altitude_m,
                    })
                    
                # --- Attitude ---
                if self._current_attitude:
                    record.update({
                        "roll": self._current_attitude.roll_deg,
                        "pitch": self._current_attitude.pitch_deg,
                        "yaw": self._current_attitude.yaw_deg,
                    })
                
                # --- Battery ---
                if self._current_battery:
                    record.update({
                        "battery_v": self._current_battery.voltage_v,
                        "battery_pct": self._current_battery.remaining_percent * 100,
                    })
                
                # --- Actuators (Propulsion) ---
                if self._current_actuator:
                    # Explicit list cast for serialization safety
                    record["actuator_controls_0"] = list(self._current_actuator.actuator)
                else:
                    record["actuator_controls_0"] = []
                    
                # --- GPS Info (Navigation) ---
                if self._current_gps_info:
                    record["gps_satellites"] = self._current_gps_info.num_satellites
                    record["gps_fix_type"] = str(self._current_gps_info.fix_type) # Convert enum to string
                
                # --- Velocity (Control/Nav) ---
                if self._current_velocity:
                    record.update({
                        "vel_n_m_s": self._current_velocity.north_m_s,
                        "vel_e_m_s": self._current_velocity.east_m_s,
                        "vel_d_m_s": self._current_velocity.down_m_s,
                    })
                    
                # --- IMU (Sensor/Control) ---
                if self._current_imu:
                    # Acceleration
                    acc = self._current_imu.acceleration_frd
                    record.update({
                        "acc_x_m_s2": acc.forward_m_s2,
                        "acc_y_m_s2": acc.right_m_s2,
                        "acc_z_m_s2": acc.down_m_s2,
                    })
                    # Angular Velocity (Body Rates)
                    gyro = self._current_imu.angular_velocity_frd
                    record.update({
                        "gyro_x_rad_s": gyro.forward_rad_s,
                        "gyro_y_rad_s": gyro.right_rad_s,
                        "gyro_z_rad_s": gyro.down_rad_s,
                    })

                self.telemetry_data.append(record)
                await asyncio.sleep(1.0 / self.config.telemetry_rate_hz)
        
        # Start all capture tasks concurrently
        asyncio.create_task(position_capture())
        asyncio.create_task(attitude_capture())
        asyncio.create_task(battery_capture())
        asyncio.create_task(actuator_capture())
        asyncio.create_task(gps_info_capture())
        asyncio.create_task(velocity_capture())
        asyncio.create_task(imu_capture())
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
        fault_severity: float = 1.0,
        px4_fault_cmd: str = None,
        parachute_trigger: bool = False
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
            parachute_trigger: If True, simulate parachute deployment (P1 enhancement)
        
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
                    # First try native PX4 fault injection (includes param fallback)
                    injection_success = await self._trigger_px4_fault(fault_type, fault_severity)
                    if injection_success:
                        logger.info(f"✓ Fault {fault_type} injected successfully via PX4")
                        fault_injected = True
                        self.fault_injection_success = True
                        # Mode is set by _trigger_px4_fault: "native" or "param"
                        if self.fault_injection_mode == "none":
                            self.fault_injection_mode = "native"  # Default
                        # Wait for fault to take effect and complete
                        await asyncio.sleep(30)
                    else:
                        # Use FailureEmulator for realistic behavioral emulation
                        logger.info("⚠ Native + param injection unavailable - using behavioral emulation")
                        self.fault_injection_mode = "emulated"
                        from src.simulation.failure_emulator import FailureEmulator
                        
                        emulator = FailureEmulator(self.drone)
                        emulation_result = await emulator.emulate(
                            fault_type, 
                            fault_severity,
                            parachute_trigger=parachute_trigger
                        )
                        
                        if emulation_result.success:
                            logger.info(f"✅ Failure emulated: {emulation_result.method}")
                            logger.info(f"   Phases: {' → '.join(emulation_result.phases_completed)}")
                            logger.info(f"   Observation duration: {emulation_result.observation_duration:.1f}s")
                            fault_injected = True
                            self.fault_injection_success = True
                            # Emulator handles its own timing and landing
                        else:
                            logger.warning(f"⚠ Emulation failed: {emulation_result.error}")
                            logger.info("   Falling back to controlled landing")
                            self.fault_injection_mode = "fallback"
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
    
    async def _inject_fault_delayed(self, fault_type: str, delay_sec: int, severity: float = 1.0, px4_fault_cmd: str = None):
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
            injection_success = await self._trigger_px4_fault(fault_type, severity, px4_fault_cmd)
            if injection_success:
                logger.info(f"✓ Fault {fault_type} injected successfully via PX4")
            else:
                logger.warning(f"⚠ Actual {fault_type} injection failed - fallback to crash simulation")
            
        except asyncio.CancelledError:
            logger.info("Fault injection cancelled (mission ended)")
        except Exception as e:
            logger.error(f"Fault injection failed: {e}")
    
    def _generate_px4_failure_cmd(self, fault_type_lower: str, severity: float) -> str:
        """
        Generate PX4 shell failure command based on fault type.
        
        PX4 failure command format: failure <unit> <type>
        Units: gyro, accel, mag, baro, gps, airspeed, motor, servo, avoidance, rc, mavlink
        Types: ok, off, stuck, garbage, wrong, slow, delayed, intermittent
        
        IMPORTANT: Airspace violations (altitude_violation, geofence_violation) do NOT
        inject any fault - the drone was healthy but in wrong location.
        
        Returns:
            PX4 shell command string or empty string if unknown/no fault to inject
        """
        # SKIP fault injection for airspace violations
        # These are NOT mechanical failures - drone is healthy but in wrong location
        if any(x in fault_type_lower for x in ['altitude_violation', 'geofence_violation', 'airspace']):
            logger.info("  → Airspace violation: NO fault injection (healthy drone demonstration)")
            return ""  # No fault command - healthy flight
        
        # Map fault types to PX4 failure units
        fault_mappings = {
            'motor': 'motor',
            'propulsion': 'motor',
            'engine': 'motor',
            'gps': 'gps',
            'navigation': 'gps',
            'flyaway': 'gps',
            'battery': 'battery',  # Note: battery failure may not be supported
            'power': 'battery',
            'gyro': 'gyro',
            'accelerometer': 'accel',
            'magnetometer': 'mag',
            'compass': 'mag',
            'baro': 'baro',
            'barometer': 'baro',
            'rc': 'rc',
            'remote': 'rc',
        }
        
        # Find matching unit
        unit = None
        for key, value in fault_mappings.items():
            if key in fault_type_lower:
                unit = value
                break
        
        if not unit:
            return ""
        
        # Determine failure type based on severity
        if severity >= 0.8:
            fail_type = "off"
        elif severity >= 0.5:
            fail_type = "intermittent"
        else:
            fail_type = "stuck"
        
        return f"failure {unit} {fail_type}"
    
    async def _trigger_px4_fault(self, fault_type: str, severity: float = 1.0, px4_fault_cmd: str = None):
        """
        Trigger actual PX4 fault using MAVSDK failure module.
        
        PX4 requires:
        1. SYS_FAILURE_EN=1 to enable failure injection
        2. Use drone.failure.inject(FailureUnit, FailureType, instance)
        
        Reference: https://docs.px4.io/main/en/debug/failure_injection.html
        
        FailureUnit options (MAVSDK Python enums):
          Sensors: SENSOR_GYRO, SENSOR_ACCEL, SENSOR_MAG, SENSOR_BARO, SENSOR_GPS,
                   SENSOR_OPTICAL_FLOW, SENSOR_VIO, SENSOR_DISTANCE_SENSOR, SENSOR_AIRSPEED
          Systems: SYSTEM_BATTERY, SYSTEM_MOTOR, SYSTEM_SERVO, SYSTEM_AVOIDANCE,
                   SYSTEM_RC_SIGNAL, SYSTEM_MAVLINK_SIGNAL
        
        FailureType options: OK, OFF, STUCK, GARBAGE, WRONG, SLOW, DELAYED, INTERMITTENT
        
        Correct API: await drone.failure.inject(FailureUnit.SENSOR_GPS, FailureType.OFF, 0)
        
        Args:
            fault_type: Normalized fault type from LLM
            severity: 0.0-1.0 severity (1.0 = complete failure = OFF)
            px4_fault_cmd: Optional raw PX4 shell command from LLM
        """
        from mavsdk.failure import FailureUnit, FailureType
        from datetime import datetime
        
        fault_type_lower = fault_type.lower().replace("-", "_").replace(" ", "_")
        injection_timestamp = datetime.now().isoformat()
        
        # =====================================================================
        # DETAILED FAULT INJECTION LOG - For Developer Alerting
        # =====================================================================
        logger.info("=" * 70)
        logger.info("  🚨 FAULT INJECTION INITIATED")
        logger.info("=" * 70)
        logger.info(f"  Timestamp:      {injection_timestamp}")
        logger.info(f"  Fault Type:     {fault_type}")
        logger.info(f"  Severity:       {severity:.2f} (1.0=complete failure)")
        logger.info(f"  LLM Command:    {px4_fault_cmd if px4_fault_cmd else 'None (auto-generate)'}")
        logger.info(f"  Simulator:      {self.config.simulator}")
        logger.info("-" * 70)
        
        try:
            # Step 1: Enable failure injection in PX4
            logger.info("  [STEP 1] Enabling PX4 failure injection parameter...")
            logger.info(f"           Parameter: SYS_FAILURE_EN = 1")
            try:
                await self.drone.param.set_param_int("SYS_FAILURE_EN", 1)
                logger.info("           ✓ SUCCESS: SYS_FAILURE_EN enabled")
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.warning(f"           ⚠ WARNING: Could not set SYS_FAILURE_EN: {e}")
                logger.info("           (Parameter may already be enabled via environment)")

            # =====================================================================
            # NEW: Try parameter-based injection FIRST (most reliable for all simulators)
            # =====================================================================
            # Parameter injection works with SIH, Gazebo Harmonic, and Gazebo Classic
            # Unlike MAVSDK failure.inject() which requires simulator-specific plugins
            logger.info("  [STEP 2] Trying parameter-based fault injection (most reliable)...")
            param_success = await self._inject_fault_via_params(fault_type_lower, severity)
            if param_success:
                logger.info("-" * 70)
                logger.info(f"  ✅ FAULT INJECTION COMPLETE via Parameters")
                logger.info(f"     Method: PX4 parameter manipulation")
                logger.info(f"     Expected Effect: {self._describe_fault_effect(fault_type_lower)}")
                logger.info("=" * 70)
                return True
            
            logger.info("           → Parameter injection not applicable for this fault type")
            logger.info("             Falling back to shell/MAVSDK methods...")
            
            # Step 3: Try raw PX4 shell command if available (Trust the LLM)
            # The PX4 shell `failure` command works with SIH simulator when SYS_FAILURE_EN=1
            shell_cmd = px4_fault_cmd if (px4_fault_cmd and len(px4_fault_cmd) > 5) else None
            
            # If no LLM command, generate a standard shell command based on fault type
            if not shell_cmd:
                shell_cmd = self._generate_px4_failure_cmd(fault_type_lower, severity)
                logger.info(f"  [STEP 3] Auto-generated PX4 shell command:")
            else:
                logger.info(f"  [STEP 3] Using LLM-provided PX4 shell command:")
            
            if shell_cmd:
                logger.info(f"           Command: '{shell_cmd}'")
                logger.info(f"           Method:  PX4 Shell (MAVSDK shell.send)")
                shell_start = datetime.now()
                try:
                    feedback = await asyncio.wait_for(
                        self.drone.shell.send(shell_cmd),
                        timeout=5.0
                    )
                    shell_elapsed = (datetime.now() - shell_start).total_seconds() * 1000
                    logger.info(f"           ✓ SUCCESS: Command executed in {shell_elapsed:.0f}ms")
                    logger.info(f"           Response:  {feedback if feedback else '(no feedback)'}")
                    logger.info("-" * 70)
                    logger.info(f"  ✅ FAULT INJECTION COMPLETE via PX4 Shell")
                    logger.info(f"     Expected Effect: {self._describe_fault_effect(fault_type_lower)}")
                    logger.info("=" * 70)
                    self.fault_injection_mode = "native"
                    return True
                except asyncio.TimeoutError:
                    shell_elapsed = (datetime.now() - shell_start).total_seconds() * 1000
                    logger.warning(f"           ⚠ TIMEOUT: Shell command took >{shell_elapsed:.0f}ms")
                    logger.info("           Fallback: Trying MAVSDK failure.inject()...")
                except Exception as shell_error:
                    logger.warning(f"           ⚠ FAILED: {shell_error}")
                    logger.info("           Fallback: Trying MAVSDK failure.inject()...")
                # Fallthrough to MAVSDK logic
            # Step 4: Determine failure type based on severity
            logger.info(f"  [STEP 4] Mapping fault to MAVSDK FailureUnit...")
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
            
            # Map fault types to MAVSDK FailureUnit (correct enum names)
            # Reference: https://docs.px4.io/main/en/debug/failure_injection.html
            # MAVSDK Python uses: SENSOR_* for sensors, SYSTEM_* for systems
            if "motor" in fault_type_lower or "propulsion" in fault_type_lower or "engine" in fault_type_lower:
                failure_unit = FailureUnit.SYSTEM_MOTOR
                unit_name = "SYSTEM_MOTOR"
                
            elif "gps" in fault_type_lower or "navigation" in fault_type_lower or "flyaway" in fault_type_lower:
                failure_unit = FailureUnit.SENSOR_GPS
                unit_name = "SENSOR_GPS"
                
            elif "battery" in fault_type_lower or "power" in fault_type_lower:
                failure_unit = FailureUnit.SYSTEM_BATTERY
                unit_name = "SYSTEM_BATTERY"
                
            elif "control" in fault_type_lower or "servo" in fault_type_lower:
                failure_unit = FailureUnit.SYSTEM_SERVO
                unit_name = "SYSTEM_SERVO"
                
            elif "gyro" in fault_type_lower:
                failure_unit = FailureUnit.SENSOR_GYRO
                unit_name = "SENSOR_GYRO"
                
            elif "accel" in fault_type_lower:
                failure_unit = FailureUnit.SENSOR_ACCEL
                unit_name = "SENSOR_ACCEL"
                
            elif "mag" in fault_type_lower or "compass" in fault_type_lower:
                failure_unit = FailureUnit.SENSOR_MAG
                unit_name = "SENSOR_MAG"
                
            elif "baro" in fault_type_lower or "altitude" in fault_type_lower:
                failure_unit = FailureUnit.SENSOR_BARO
                unit_name = "SENSOR_BARO"
                
            elif "rc" in fault_type_lower or "remote" in fault_type_lower:
                failure_unit = FailureUnit.SYSTEM_RC_SIGNAL
                unit_name = "SYSTEM_RC_SIGNAL"
                
            else:
                # Default to motor failure for unknown types
                logger.warning(f"           Unknown fault '{fault_type}', defaulting to SYSTEM_MOTOR")
                failure_unit = FailureUnit.SYSTEM_MOTOR
                unit_name = "SYSTEM_MOTOR"
            
            logger.info(f"           FailureUnit:  {unit_name}")
            logger.info(f"           FailureType:  {failure_type_name}")
            logger.info(f"           Instance:     0 (primary)")
            
            # Step 5: Inject the failure with retry (MAVSDK method)
            # Correct API: inject(failure_unit, failure_type, instance)
            # Reference: MAVSDK Python - Failure.inject(failure_unit, failure_type, instance)
            # instance=0 means first instance of this unit type
            logger.info(f"  [STEP 5] Injecting via MAVSDK failure.inject()...")
            logger.info(f"           API: inject({unit_name}, {failure_type_name}, 0)")
            
            # Retry logic with timeout - some Gazebo setups need multiple attempts
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    mavsdk_start = datetime.now()
                    # Correct MAVSDK API: inject(failure_unit, failure_type, instance)
                    await asyncio.wait_for(
                        self.drone.failure.inject(
                            failure_unit,
                            failure_type, 
                            0  # instance 0 = primary/first
                        ),
                        timeout=3.0  # 3 second timeout per attempt
                    )
                    mavsdk_elapsed = (datetime.now() - mavsdk_start).total_seconds() * 1000
                    logger.info(f"           ✓ SUCCESS: Injected in {mavsdk_elapsed:.0f}ms")
                    logger.info("-" * 70)
                    logger.info(f"  ✅ FAULT INJECTION COMPLETE via MAVSDK")
                    logger.info(f"     Unit: {unit_name}, Type: {failure_type_name}, Instance: 0")
                    logger.info(f"     Expected Effect: {self._describe_fault_effect(fault_type_lower)}")
                    logger.info("=" * 70)
                    self.fault_injection_mode = "native"
                    return True  # Success!
                except asyncio.TimeoutError:
                    logger.warning(f"           ⚠ TIMEOUT (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.5)  # Brief wait before retry
                except Exception as inject_error:
                    if attempt < max_retries - 1:
                        logger.info(f"           ⏳ Retry {attempt + 2}/{max_retries}...")
                        await asyncio.sleep(0.5)  # Wait before retry
                    else:
                        # All retries failed
                        logger.warning(f"           ⚠ FAILED after {max_retries} attempts: {inject_error}")
            
            # All injection methods failed
            # This is reached when:
            # - Parameter injection not applicable (fault type not mappable)
            # - Shell command failed/timed out
            # - MAVSDK injection failed/timed out
            logger.info("-" * 70)
            logger.warning(f"  ⚠ FAULT INJECTION FALLBACK MODE")
            logger.info(f"     All injection methods failed for fault type: {fault_type}")
            logger.info(f"     Impact: Telemetry may show normal flight patterns")
            logger.info(f"             Behavioral analysis will still work based on telemetry")
            logger.info("=" * 70)
            self.fault_injection_mode = "fallback"
            return False  # Signal that actual fault injection failed
                
        except Exception as e:
            logger.error(f"  ❌ FAULT INJECTION ERROR: {e}")
            logger.info("=" * 70)
            return False  # Signal failure
    
    async def _inject_fault_via_params(self, fault_type_lower: str, severity: float) -> bool:
        """
        Inject fault using PX4 parameter manipulation.
        
        This is the MOST RELIABLE injection method - works with all simulators:
        - SIH (sihsim_quadx)
        - Gazebo Harmonic (gz_x500)
        - Gazebo Classic
        
        Supported injections:
        - Baro/Altitude: SIM_BARO_OFF_P (pressure offset in Pa) → altitude error
        - GPS/Navigation/Flyaway: EKF2_GPS_P_NOISE (position noise in m) → position uncertainty
        - Mag/Compass: SIM_MAG_OFFSET_X (offset in Gauss) → heading error
        - Accel: EKF2_ACC_NOISE (m/s²) → velocity estimation error
        - Gyro: EKF2_GYR_NOISE (rad/s) → attitude estimation error
        - Motor/Propulsion: CA_FAILURE_MODE (control allocation failure mode)
        """
        try:
            if "baro" in fault_type_lower or "altitude" in fault_type_lower:
                # Inject 500Pa offset (~40m altitude error)
                # Even airspace violation should use GPS drift to simulate position error
                offset = 500 * severity  # 0-500 Pa based on severity
                await self.drone.param.set_param_float("SIM_BARO_OFF_P", float(offset))
                logger.info(f"           ✓ Set SIM_BARO_OFF_P = {offset:.0f} Pa (altitude offset)")
                self.fault_injection_mode = "param"
                return True
                
            elif "gps" in fault_type_lower or "navigation" in fault_type_lower or "flyaway" in fault_type_lower:
                # Increase GPS noise to simulate degradation
                noise = 5.0 + (45.0 * severity)  # 5-50m noise
                await self.drone.param.set_param_float("EKF2_GPS_P_NOISE", float(noise))
                logger.info(f"           ✓ Set EKF2_GPS_P_NOISE = {noise:.1f} m (GPS degradation)")
                self.fault_injection_mode = "param"
                return True
                
            elif "mag" in fault_type_lower or "compass" in fault_type_lower:
                # Inject magnetometer offset (0.5 Gauss = ~45° heading error)
                offset = 0.5 * severity
                await self.drone.param.set_param_float("SIM_MAG_OFFSET_X", float(offset))
                logger.info(f"           ✓ Set SIM_MAG_OFFSET_X = {offset:.2f} Ga (compass error)")
                self.fault_injection_mode = "param"
                return True
                
            elif "gyro" in fault_type_lower:
                # Increase gyro noise
                noise = 0.01 + (0.1 * severity)  # 0.01-0.11 rad/s noise
                await self.drone.param.set_param_float("EKF2_GYR_NOISE", float(noise))
                logger.info(f"           ✓ Set EKF2_GYR_NOISE = {noise:.3f} rad/s")
                self.fault_injection_mode = "param"
                return True
                
            elif "accel" in fault_type_lower:
                # Increase accelerometer noise
                noise = 0.35 + (2.0 * severity)  # 0.35-2.35 m/s² noise
                await self.drone.param.set_param_float("EKF2_ACC_NOISE", float(noise))
                logger.info(f"           ✓ Set EKF2_ACC_NOISE = {noise:.2f} m/s²")
                self.fault_injection_mode = "param"
                return True
            
            elif "motor" in fault_type_lower or "propulsion" in fault_type_lower:
                # Motor failure via control allocation
                # CA_FAILURE_MODE: 0=disabled, 1=remove motors, 2=reduce motors
                failure_mode = 1 if severity >= 0.8 else 2
                await self.drone.param.set_param_int("CA_FAILURE_MODE", failure_mode)
                logger.info(f"           ✓ Set CA_FAILURE_MODE = {failure_mode} (motor failure simulation)")
                self.fault_injection_mode = "param"
                return True
            
            elif "none" in fault_type_lower or "violation" in fault_type_lower:
                # No fault injection for airspace violations - normal drone behavior
                logger.info(f"           ✓ No fault injection needed (airspace violation / healthy drone)")
                self.fault_injection_mode = "emulated"
                return True
            
            else:
                logger.info(f"           No parameter mapping for '{fault_type_lower}'")
                return False
                
        except Exception as e:
            logger.warning(f"           Parameter injection failed: {e}")
            return False
    
    def _describe_fault_effect(self, fault_type_lower: str) -> str:
        """Return human-readable description of expected fault effects."""
        # Airspace violations are NOT mechanical failures
        if any(x in fault_type_lower for x in ['altitude_violation', 'geofence_violation', 'airspace']):
            return "No fault injected (healthy drone in restricted airspace) → Normal flight telemetry"
        
        effects = {
            "motor": "Motor power reduction → Yaw/roll instability, possible controlled descent",
            "propulsion": "Thrust asymmetry → Attitude oscillation, altitude loss",
            "gps": "Position estimate degradation → EKF mode switch, possible RTL",
            "navigation": "Navigation uncertainty → Drift, failsafe activation",
            "flyaway": "Loss of position control → Uncontrolled horizontal movement",
            "battery": "Low voltage simulation → RTL or forced landing sequence",
            "power": "Power system fault → Failsafe landing behavior",
            "gyro": "Angular rate sensor noise → Attitude estimation errors",
            "accel": "Accelerometer fault → Velocity estimation drift",
            "mag": "Compass interference → Heading errors, yaw drift",
            "compass": "Magnetometer stuck → EKF fallback to GPS heading",
            "baro": "Altitude sensor off → Altitude hold degradation",
            "rc": "RC signal loss → Failsafe behavior (RTL/Land)",
            "control": "Servo malfunction → Reduced control authority",
        }
        for key, desc in effects.items():
            if key in fault_type_lower:
                return desc
        return "Unknown fault type → Monitor telemetry for anomalies"


# =============================================================================
# Automated Pipeline
# =============================================================================

class AutomatedPipeline:
    """Fully automated AeroGuardian pipeline."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.wsl = WSLController(config)
        self.executor = None
        self.timing_metrics = {}  # Track step durations
        
    def _track_time(self, step_name: str, start_time: float) -> float:
        """Record step duration and return current time."""
        duration = time.time() - start_time
        self.timing_metrics[step_name] = round(duration, 2)
        logger.info(f"  ⏱ {step_name}: {duration:.2f}s")
        return time.time()
        
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
        
        # Initialize timing
        pipeline_start = time.time()
        self.timing_metrics = {}
        step_start = time.time()
        
        try:
            # Step 0: Launch QGroundControl (if not already running)
            logger.info("")
            logger.info("Launching QGroundControl...")
            self.wsl.launch_qgroundcontrol()
            
            # Step 1: Load FAA incident (do first to get location for PX4 home)
            self._step_header(1, "Load FAA Report")
            step_start = time.time()
            incident = self._load_report(incident_index)
            logger.info(f"  ID: {incident.get('report_id', 'Unknown')}")
            logger.info(f"  Location: {incident.get('city', 'Unknown')}, {incident.get('state', 'Unknown')}")
            step_start = self._track_time("T_load", step_start)
            
            # Step 2: Generate LLM configuration (before PX4 to get home location)
            self._step_header(2, "Generate LLM Configuration")
            flight_config = self._generate_config(incident)
            fault_type = flight_config.get('fault_injection', {}).get('fault_type', 'none')
            logger.info(f"  Fault Type: {fault_type}")
            logger.info(f"  Waypoints: {len(flight_config.get('waypoints', []))}")
            step_start = self._track_time("T_translate", step_start)
            
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
            step_start = self._track_time("T_px4_init", step_start)
            
            # Step 4: Execute mission
            self._step_header(4, "Execute Flight Mission")
            success, telemetry = asyncio.run(self._execute_mission(flight_config))
            logger.info(f"  Mission Success: {success}")
            logger.info(f"  Telemetry Points: {len(telemetry)}")
            
            # Capture fault injection status for reporting
            if self.executor:
                flight_config["fault_injection_status"] = {
                    "success": self.executor.fault_injection_success,
                    "mode": self.executor.fault_injection_mode,
                }
                if not self.executor.fault_injection_success and flight_config.get("fault_injection", {}).get("fault_type"):
                    logger.warning(f"  ⚠ Fault injection fallback: {self.executor.fault_injection_mode}")
            step_start = self._track_time("T_simulate", step_start)
            
            self._step_header(5, "Generate Safety Report")
            safety_report = self._generate_safety_report(incident, flight_config, telemetry)
            logger.info(f"  Hazard Level: {safety_report.get('safety_level', 'UNKNOWN')}")
            logger.info(f"  Recommendation: {safety_report.get('verdict', 'REVIEW')}")
            step_start = self._track_time("T_analyze", step_start)
            
            # Step 6: Save reports
            self._step_header(6, "Save Reports")
            paths = self._save_reports(incident, flight_config, telemetry, safety_report)
            step_start = self._track_time("T_save", step_start)
            
            # Total pipeline time
            self.timing_metrics["T_total"] = round(time.time() - pipeline_start, 2)
            
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
        logger.info(f"  PROCESSING: {incident.get('report_id', 'Unknown')}")
        logger.info("=" * 60)
        logger.info(f"  Location: {incident.get('city', 'Unknown')}, {incident.get('state', '')}")
        logger.info(f"  Type: {incident.get('incident_type', 'other')}")
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
            
            # Capture fault injection status for reporting
            if self.executor:
                flight_config["fault_injection_status"] = {
                    "success": self.executor.fault_injection_success,
                    "mode": self.executor.fault_injection_mode,
                }
                if not self.executor.fault_injection_success and flight_config.get("fault_injection", {}).get("fault_type"):
                    logger.warning(f"  ⚠ Fault injection fallback: {self.executor.fault_injection_mode}")
            
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
            logger.info(f"✓ COMPLETE: {incident.get('report_id', 'Unknown')}")
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
    
    def _load_report(self, index: int) -> Dict:
        """Load FAA report by index."""
        from src.faa.sighting_filter import get_sighting_filter
        
        sighting_filter = get_sighting_filter()
        count = sighting_filter.load()
        logger.info(f"  Available reports: {count}")
        
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
            report_id=incident.get("report_id", "Unknown"),
        )
        
        # Log simulation mode (if available, mostly legacy)
        sim_mode = incident.get("simulation_mode", "MECHANICAL_TEST")
        if sim_mode == "AIRSPACE_SIGHTING":
             logger.info(f"  Note: Legacy high-altitude tag detected (AIRSPACE_SIGHTING)")
        
        # Extract altitude from LLM config
        mission_config = config.get("mission", {})
        llm_takeoff_alt = mission_config.get("takeoff_altitude_m")
        llm_cruise_alt = mission_config.get("cruise_altitude_m")
        
        # Prefer LLM altitude if valid (>0 and <200m), otherwise use default
        if llm_takeoff_alt is not None and llm_takeoff_alt > 0 and llm_takeoff_alt < 200:
            takeoff_alt = llm_takeoff_alt
        else:
            takeoff_alt = 50.0  # Default safe altitude
            
        if llm_cruise_alt is not None and llm_cruise_alt > 0 and llm_cruise_alt < 200:
            cruise_alt = llm_cruise_alt
        else:
            cruise_alt = 50.0
        

        
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
        
        # P1: Extract parachute trigger from proxy_modeling
        proxy_modeling = config.get("proxy_modeling", {})
        parachute_trigger = proxy_modeling.get("parachute_modeled", False)
        
        logger.info(f"  Executing mission: alt={takeoff_alt}m, speed={speed_m_s}m/s, waypoints={len(waypoints)}")
        if fault_type:
            logger.info(f"  🎯 Fault injection: {fault_type} at T+{fault_onset_sec}s (severity: {fault_severity})")
        if parachute_trigger:
            logger.info(f"  🪂 Parachute deployment will be simulated")
        
        return await self.executor.execute_mission(
            waypoints=waypoints,
            takeoff_alt=takeoff_alt,
            speed_m_s=speed_m_s,
            fault_type=fault_type,
            fault_onset_sec=fault_onset_sec,
            fault_severity=fault_severity,
            px4_fault_cmd=config.get("px4_commands", {}).get("fault"),
            parachute_trigger=parachute_trigger,
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
            report_id=incident.get("report_id", "Unknown"),
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
        
        # Add timing metrics to config for inclusion in output
        config['timing_metrics'] = self.timing_metrics
        
        reporter = UnifiedReporter(self.config.output_dir)
        return reporter.generate(
            incident=incident,
            flight_config=config,
            telemetry=telemetry,
            safety_analysis=safety,
        )
    
    def _print_summary(self, incident: Dict, safety: Dict, telemetry_count: int, paths: Dict):
        """Print final summary with timing metrics."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("  PIPELINE COMPLETE")
        logger.info("=" * 70)
        logger.info(f"  Report: {incident.get('report_id', 'Unknown')}")
        logger.info(f"  Telemetry Points: {telemetry_count}")
        logger.info(f"  Hazard Level: {safety.get('safety_level', 'UNKNOWN')}")
        logger.info(f"  Recommendation: {safety.get('verdict', 'REVIEW')}")
        logger.info("")
        
        # Print timing metrics
        if self.timing_metrics:
            logger.info("  Timing Metrics:")
            for step, duration in self.timing_metrics.items():
                logger.info(f"    {step}: {duration}s")
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
                        help="Run without Gazebo GUI (uses SIH simulator)")
    parser.add_argument("--skip-px4", action="store_true",
                        help="Skip PX4 startup (assume already running)")
    parser.add_argument("--wsl-ip", type=str, default=None,
                        help="WSL2 IP address (get via: ip addr show eth0 in WSL). Required for QGC connection.")
    parser.add_argument("--qgc-port", type=int, default=18570,
                        help="QGroundControl UDP port (default: 18570)")
    parser.add_argument("--vehicle", type=str, default="iris",
                        choices=["iris", "typhoon_h480", "plane", "rover"],
                        help="PX4 vehicle type")
    parser.add_argument("--simulator", "-s", type=str, default="auto",
                        choices=["auto", "sihsim_quadx", "gz_x500", "gazebo-classic_iris"],
                        help="Simulator target (auto: sihsim_quadx for headless, gz_x500 for GUI)")
    
    args = parser.parse_args()
    
    # Build config
    config = PipelineConfig(
        wsl_ip=args.wsl_ip if args.wsl_ip else "127.0.0.1",
        qgc_port=args.qgc_port,
        headless=args.headless,
        vehicle=args.vehicle,
        simulator=args.simulator,
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
                report_id = record.get("report_id", f"Batch_{idx+1}")
                incident = {
                    "report_id": report_id,
                    "incident_id": report_id,  # Alias for compatibility
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
                        "incident_id": report_id,
                        "output_dir": str(paths.get("report_dir", "")),
                        "status": "success",
                    })
                except Exception as e:
                    logger.error(f"Record {idx+1} failed: {e}")
                    all_reports.append({
                        "incident_id": report_id,
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