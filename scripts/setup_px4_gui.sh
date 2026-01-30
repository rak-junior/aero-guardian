# =============================================================================
# AeroGuardian PX4 + Gazebo GUI Setup Script (WSL2)
# =============================================================================
# Author: AeroGuardian Member
# Date: 2026-01-19
# Version: 1.0
#
# Professional setup script for PX4 SITL with Gazebo GUI in WSL2.
# Configures networking for QGroundControl connection.
#
# Usage:
#     chmod +x setup_px4_gui.sh
#     ./setup_px4_gui.sh [--install-px4] [--install-deps] [--configure-only]
#
# QGroundControl Connection:
#     IP: 172.27.166.100
#     Port: 18570
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================
PX4_VERSION="v1.14.3"
PX4_DIR="$HOME/PX4-Autopilot"
LOG_FILE="$HOME/px4_setup_$(date +%Y%m%d_%H%M%S).log"

# QGroundControl configuration
QGC_HOST_IP="${QGC_HOST_IP:-172.27.166.100}"
QGC_PORT="${QGC_PORT:-18570}"
MAVSDK_PORT="${MAVSDK_PORT:-14540}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# =============================================================================
# Utility Functions
# =============================================================================

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo -e "$msg" | tee -a "$LOG_FILE"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

log_step() {
    echo "" | tee -a "$LOG_FILE"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" | tee -a "$LOG_FILE"
    echo -e "${BOLD}  $1${NC}" | tee -a "$LOG_FILE"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" | tee -a "$LOG_FILE"
}

check_command() {
    if command -v "$1" &>/dev/null; then
        log_success "$1 is installed"
        return 0
    else
        log_warn "$1 is not installed"
        return 1
    fi
}

# =============================================================================
# Environment Detection
# =============================================================================

detect_environment() {
    log_step "Phase 1: Environment Detection"
    
    # Detect WSL
    if grep -qi microsoft /proc/version 2>/dev/null; then
        log_success "Running in WSL2"
        IS_WSL=true
    else
        log_warn "Not running in WSL - some features may not work"
        IS_WSL=false
    fi
    
    # Get WSL IP
    WSL_IP=$(hostname -I | awk '{print $1}')
    log_info "WSL IP: $WSL_IP"
    
    # Get Windows Host IP (usually the gateway in WSL)
    WINDOWS_IP=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
    log_info "Windows Host IP: $WINDOWS_IP"
    
    # Detect display
    if [ -n "$DISPLAY" ]; then
        log_success "DISPLAY is set: $DISPLAY"
    else
        DISPLAY="${WINDOWS_IP}:0"
        export DISPLAY
        log_info "Set DISPLAY to: $DISPLAY"
    fi
    
    # Check X server
    if command -v xset &>/dev/null; then
        if xset q &>/dev/null 2>&1; then
            log_success "X11 connection working"
        else
            log_warn "X11 connection failed - GUI may not work"
            log_info "Ensure VcXsrv or WSLg is running on Windows"
        fi
    fi
    
    log_info "QGroundControl target: $QGC_HOST_IP:$QGC_PORT"
}

# =============================================================================
# Dependencies Installation  
# =============================================================================

install_dependencies() {
    log_step "Phase 2: Installing System Dependencies"
    
    sudo apt-get update
    
    # Essential build tools
    sudo apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        curl \
        python3 \
        python3-pip \
        python3-venv \
        ninja-build \
        ccache \
        gdb
    
    # PX4 dependencies
    sudo apt-get install -y \
        libxml2-dev \
        libxslt1-dev \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        libeigen3-dev \
        libprotobuf-dev \
        protobuf-compiler \
        libcurl4-openssl-dev \
        libyaml-cpp-dev
    
    # Networking tools
    sudo apt-get install -y \
        net-tools \
        netcat \
        socat
    
    # X11 for GUI (required for Gazebo display)
    sudo apt-get install -y \
        x11-apps \
        x11-xserver-utils \
        dbus-x11
    
    log_success "System dependencies installed"
}

# =============================================================================
# Gazebo Installation
# =============================================================================

install_gazebo() {
    log_step "Phase 3: Installing Gazebo Classic"
    
    if check_command gazebo; then
        log_info "Gazebo is already installed"
        gazebo --version | head -1
        return 0
    fi
    
    # Add Gazebo repository
    sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" > /etc/apt/sources.list.d/gazebo-stable.list'
    wget https://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
    
    sudo apt-get update
    sudo apt-get install -y gazebo libgazebo-dev
    
    log_success "Gazebo installed"
}

# =============================================================================
# PX4-Autopilot Installation
# =============================================================================

install_px4() {
    log_step "Phase 4: Installing PX4-Autopilot"
    
    if [ -d "$PX4_DIR" ] && [ -f "$PX4_DIR/Tools/simulation/gazebo-classic/sitl_gazebo-classic/CMakeLists.txt" ]; then
        log_info "PX4-Autopilot already installed at $PX4_DIR"
        cd "$PX4_DIR"
        git describe --tags 2>/dev/null || echo "Unknown version"
        return 0
    fi
    
    # Clone PX4
    log_info "Cloning PX4-Autopilot $PX4_VERSION..."
    cd "$HOME"
    git clone --recursive https://github.com/PX4/PX4-Autopilot.git --branch "$PX4_VERSION" --depth 1
    
    cd "$PX4_DIR"
    
    # Run PX4 setup script
    log_info "Running PX4 setup script..."
    bash ./Tools/setup/ubuntu.sh --no-nuttx
    
    # First build to ensure everything works
    log_info "Building PX4 SITL (this may take 10-15 minutes)..."
    make px4_sitl_default
    
    log_success "PX4-Autopilot installed and built"
}

# =============================================================================
# MAVSDK Installation
# =============================================================================

install_mavsdk() {
    log_step "Phase 5: Installing MAVSDK"
    
    # Python MAVSDK
    pip3 install --upgrade mavsdk pymavlink
    
    log_success "MAVSDK installed"
}

# =============================================================================
# Configure QGroundControl Connection
# =============================================================================

configure_qgc_connection() {
    log_step "Phase 6: Configuring QGroundControl Connection"
    
    log_info "Target QGC: $QGC_HOST_IP:$QGC_PORT"
    
    # Create environment file for PX4
    cat > "$HOME/.px4_env" << EOF
# PX4 Environment Configuration
# Generated by AeroGuardian setup script
# Date: $(date)

# Display for GUI
export DISPLAY=${WINDOWS_IP}:0

# QGroundControl connection
export PX4_GCS_URL="udp://@${QGC_HOST_IP}:${QGC_PORT}"
export PX4_SIM_HOST_ADDR="${QGC_HOST_IP}"

# Gazebo settings
export GAZEBO_IP=$(hostname -I | awk '{print $1}')
export GZ_SIM_SYSTEM_PLUGIN_PATH="\$HOME/PX4-Autopilot/build/px4_sitl_default/build_gazebo-classic"

# MAVLink ports
export PX4_MAVSDK_PORT=$MAVSDK_PORT
export PX4_QGC_PORT=$QGC_PORT

# Silence Gazebo deprecation warnings
export IGNITION_SILENT=1
export GAZEBO_MASTER_URI=http://localhost:11345
EOF

    # Add to bashrc if not already there
    if ! grep -q "source.*px4_env" "$HOME/.bashrc"; then
        echo "" >> "$HOME/.bashrc"
        echo "# PX4 Environment" >> "$HOME/.bashrc"
        echo "[ -f ~/.px4_env ] && source ~/.px4_env" >> "$HOME/.bashrc"
    fi
    
    source "$HOME/.px4_env"
    
    log_success "QGC connection configured"
    log_info "  Host IP: $QGC_HOST_IP"
    log_info "  Port: $QGC_PORT"
}

# =============================================================================
# Create Launcher Script
# =============================================================================

create_launcher() {
    log_step "Phase 7: Creating Launcher Script"
    
    # Create the main launcher script
    cat > "$HOME/launch_px4_gazebo.sh" << 'LAUNCHER_SCRIPT'
#!/bin/bash
# =============================================================================
# PX4 + Gazebo Launcher with QGroundControl Connection
# =============================================================================
# Usage: ./launch_px4_gazebo.sh [vehicle] [world]
#   vehicle: iris (default), typhoon_h480, plane, rover
#   world: empty (default), warehouse, baylands
# =============================================================================

set -e

# Load environment
source "$HOME/.px4_env" 2>/dev/null || true

# Configuration
VEHICLE="${1:-iris}"
WORLD="${2:-empty}"
PX4_DIR="$HOME/PX4-Autopilot"

# QGroundControl target (from environment or default)
QGC_HOST="${PX4_SIM_HOST_ADDR:-172.27.166.100}"
QGC_PORT="${PX4_QGC_PORT:-18570}"

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  AeroGuardian PX4 + Gazebo Launcher${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  Vehicle: ${YELLOW}$VEHICLE${NC}"
echo -e "  World:   ${YELLOW}$WORLD${NC}"
echo -e "  QGC:     ${YELLOW}$QGC_HOST:$QGC_PORT${NC}"
echo -e "  Display: ${YELLOW}$DISPLAY${NC}"
echo ""

cd "$PX4_DIR"

# Build command with QGC forwarding
export PX4_SIM_HOST_ADDR="$QGC_HOST"

echo -e "${GREEN}[+]${NC} Starting PX4 SITL + Gazebo..."
echo -e "${GREEN}[+]${NC} MAVLink will connect to QGroundControl at $QGC_HOST:$QGC_PORT"
echo ""

# Launch with proper MAVLink forwarding
# The -o flag sets startup options for MAVLink output
HEADLESS=0 make px4_sitl gazebo-classic_${VEHICLE}__${WORLD}

LAUNCHER_SCRIPT

    chmod +x "$HOME/launch_px4_gazebo.sh"
    
    # Create headless launcher
    cat > "$HOME/launch_px4_headless.sh" << 'HEADLESS_SCRIPT'
#!/bin/bash
# Headless PX4 SITL Launcher (no GUI)
source "$HOME/.px4_env" 2>/dev/null || true

VEHICLE="${1:-iris}"
PX4_DIR="$HOME/PX4-Autopilot"
QGC_HOST="${PX4_SIM_HOST_ADDR:-172.27.166.100}"

echo "Starting PX4 SITL (headless) - Vehicle: $VEHICLE"
echo "QGroundControl target: $QGC_HOST:${PX4_QGC_PORT:-18570}"

cd "$PX4_DIR"
export PX4_SIM_HOST_ADDR="$QGC_HOST"
HEADLESS=1 make px4_sitl gazebo-classic_${VEHICLE}

HEADLESS_SCRIPT

    chmod +x "$HOME/launch_px4_headless.sh"
    
    log_success "Launcher scripts created:"
    log_info "  GUI:      ~/launch_px4_gazebo.sh [vehicle] [world]"
    log_info "  Headless: ~/launch_px4_headless.sh [vehicle]"
}

# =============================================================================
# Validation
# =============================================================================

validate_installation() {
    log_step "Phase 8: Validating Installation"
    
    local errors=0
    
    # Check PX4
    if [ -d "$PX4_DIR" ] && [ -f "$PX4_DIR/build/px4_sitl_default/bin/px4" ]; then
        log_success "PX4 SITL binary found"
    else
        log_error "PX4 SITL binary not found"
        ((errors++))
    fi
    
    # Check Gazebo
    if check_command gazebo; then
        log_success "Gazebo available"
    else
        log_error "Gazebo not found"
        ((errors++))
    fi
    
    # Check MAVSDK
    if python3 -c "import mavsdk" 2>/dev/null; then
        log_success "MAVSDK Python package available"
    else
        log_warn "MAVSDK Python not installed"
    fi
    
    # Check launcher
    if [ -x "$HOME/launch_px4_gazebo.sh" ]; then
        log_success "Launcher script ready"
    else
        log_warn "Launcher script not found"
    fi
    
    echo ""
    if [ $errors -eq 0 ]; then
        log_success "All validations passed!"
    else
        log_error "$errors validation(s) failed"
    fi
    
    return $errors
}

# =============================================================================
# Print Summary
# =============================================================================

print_summary() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  SETUP COMPLETE${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "  ${BOLD}QGroundControl Connection:${NC}"
    echo -e "    Host: ${YELLOW}$QGC_HOST_IP${NC}"
    echo -e "    Port: ${YELLOW}$QGC_PORT${NC}"
    echo ""
    echo -e "  ${BOLD}To launch PX4 with Gazebo GUI:${NC}"
    echo -e "    ${GREEN}~/launch_px4_gazebo.sh${NC}"
    echo ""
    echo -e "  ${BOLD}To launch PX4 headless:${NC}"
    echo -e "    ${GREEN}~/launch_px4_headless.sh${NC}"
    echo ""
    echo -e "  ${BOLD}Log file:${NC} $LOG_FILE"
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# =============================================================================
# Main
# =============================================================================

main() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  AeroGuardian PX4 + Gazebo Setup${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    # Parse arguments
    INSTALL_DEPS=false
    INSTALL_PX4=false
    CONFIGURE_ONLY=false
    
    for arg in "$@"; do
        case $arg in
            --install-deps)
                INSTALL_DEPS=true
                ;;
            --install-px4)
                INSTALL_PX4=true
                ;;
            --configure-only)
                CONFIGURE_ONLY=true
                ;;
            --help|-h)
                echo "Usage: $0 [options]"
                echo ""
                echo "Options:"
                echo "  --install-deps    Install system dependencies"
                echo "  --install-px4     Install PX4-Autopilot"
                echo "  --configure-only  Only configure, don't install"
                echo ""
                echo "Environment variables:"
                echo "  QGC_HOST_IP       QGroundControl host IP (default: 172.27.166.100)"
                echo "  QGC_PORT          QGroundControl port (default: 18570)"
                exit 0
                ;;
        esac
    done
    
    # Always detect environment
    detect_environment
    
    if [ "$CONFIGURE_ONLY" = true ]; then
        configure_qgc_connection
        create_launcher
        validate_installation
        print_summary
        exit 0
    fi
    
    if [ "$INSTALL_DEPS" = true ]; then
        install_dependencies
        install_gazebo
        install_mavsdk
    fi
    
    if [ "$INSTALL_PX4" = true ]; then
        install_px4
    fi
    
    # Always configure and create launcher
    configure_qgc_connection
    create_launcher
    validate_installation
    print_summary
}

# Run
main "$@"
