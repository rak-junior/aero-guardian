#!/bin/bash
# =============================================================================
# AeroGuardian PX4 + Gazebo Setup Script (WSL2)
# =============================================================================
# Author: AeroGuardian Team (Tiny Coders)
# Date: 2026-02-04
# Version: 1.0
#
# Professional setup script for PX4 SITL with Gazebo in WSL2.
# Supports both Gazebo Harmonic (gz_x500) and Gazebo Classic (iris).
#
# Usage:
#     chmod +x setup_px4_gui.sh
#     ./setup_px4_gui.sh [OPTIONS]
#
# Options:
#     --install-deps      Install system dependencies
#     --install-px4       Install PX4-Autopilot
#     --install-gazebo    Install Gazebo Harmonic (recommended)
#     --configure-only    Only configure, don't install
#     --help              Show this help message
#
# Recommended first-time setup:
#     ./setup_px4_gui.sh --install-deps --install-px4 --install-gazebo
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================
PX4_VERSION="v1.15.2"
PX4_DIR="$HOME/PX4-Autopilot"
LOG_FILE="$HOME/px4_setup_$(date +%Y%m%d_%H%M%S).log"

# QGroundControl configuration
QGC_HOST_IP="$(hostname -I | awk '{print $1}')"  # Automatically set to WSL network IP
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
# Enhanced Error Handling
# =============================================================================
check_dependency() {
    if ! command -v "$1" &>/dev/null; then
        log_error "$1 is required but not installed. Please install it and re-run the script."
        exit 1
    fi
}

# Check critical dependencies
log_step "Checking Critical Dependencies"
check_dependency "awk"
check_dependency "hostname"
check_dependency "cat"
check_dependency "grep"
check_dependency "curl"
check_dependency "git"
check_dependency "python3"

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
    
    # Get Ubuntu version
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        log_info "OS: $NAME $VERSION_ID"
        UBUNTU_VERSION="$VERSION_ID"
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
    
    log_info "QGroundControl target: $QGC_HOST_IP:$QGC_PORT"
}

# =============================================================================
# Dependencies Installation  
# =============================================================================

install_dependencies() {
    log_step "Phase 2: Installing System Dependencies"
    
    sudo apt-get update
    
    # Essential build tools
    log_info "Installing build tools..."
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
        gdb \
        lsb-release \
        gnupg
    
    # PX4 dependencies
    log_info "Installing PX4 dependencies..."
    sudo apt-get install -y \
        libxml2-dev \
        libxslt1-dev \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        libeigen3-dev \
        libprotobuf-dev \
        protobuf-compiler \
        libcurl4-openssl-dev \
        libyaml-cpp-dev \
        libopencv-dev
    
    # Networking tools
    log_info "Installing networking tools..."
    sudo apt-get install -y \
        net-tools \
        netcat-openbsd \
        socat
    
    # X11 for GUI (optional, for Gazebo display)
    log_info "Installing X11 tools..."
    sudo apt-get install -y \
        x11-apps \
        x11-xserver-utils \
        dbus-x11 \
        mesa-utils
    
    # Python packages for MAVSDK
    log_info "Installing Python packages..."
    pip3 install --upgrade pip
    pip3 install mavsdk pymavlink
    
    log_success "System dependencies installed"
}

# =============================================================================
# Gazebo Harmonic Installation (Recommended)
# =============================================================================

install_gazebo_harmonic() {
    log_step "Phase 3: Installing Gazebo Harmonic"
    
    # Check if already installed
    if command -v gz &>/dev/null; then
        local gz_version=$(gz --version 2>/dev/null | head -1)
        log_info "Gazebo already installed: $gz_version"
        return 0
    fi
    
    log_info "Adding Gazebo repository..."
    
    # Add OSRF repository for Gazebo
    sudo curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
    
    sudo apt-get update
    
    # Install Gazebo Harmonic
    log_info "Installing Gazebo Harmonic (this may take 5-10 minutes)..."
    sudo apt-get install -y gz-harmonic
    
    log_success "Gazebo Harmonic installed"
    gz --version | head -1
}

# =============================================================================
# Gazebo Classic Installation (Alternative)
# =============================================================================

install_gazebo_classic() {
    log_step "Phase 3: Installing Gazebo Classic"
    
    if check_command gazebo; then
        log_info "Gazebo Classic is already installed"
        gazebo --version | head -1
        return 0
    fi
    
    # Add Gazebo repository
    sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" > /etc/apt/sources.list.d/gazebo-stable.list'
    wget https://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
    
    sudo apt-get update
    sudo apt-get install -y gazebo libgazebo-dev
    
    log_success "Gazebo Classic installed"
}

# =============================================================================
# PX4-Autopilot Installation
# =============================================================================

install_px4() {
    log_step "Phase 4: Installing PX4-Autopilot"
    
    if [ -d "$PX4_DIR" ] && [ -f "$PX4_DIR/build/px4_sitl_default/bin/px4" ]; then
        log_info "PX4-Autopilot already installed at $PX4_DIR"
        cd "$PX4_DIR"
        git describe --tags 2>/dev/null || echo "Unknown version"
        return 0
    fi
    
    # Clone PX4
    log_info "Cloning PX4-Autopilot $PX4_VERSION..."
    cd "$HOME"
    
    if [ -d "$PX4_DIR" ]; then
        log_info "Removing incomplete PX4 installation..."
        rm -rf "$PX4_DIR"
    fi
    
    git clone --recursive https://github.com/PX4/PX4-Autopilot.git --branch "$PX4_VERSION" --depth 1
    
    cd "$PX4_DIR"
    
    # Run PX4 setup script
    log_info "Running PX4 setup script..."
    bash ./Tools/setup/ubuntu.sh --no-nuttx
    
    # Build for Gazebo Harmonic (gz_x500)
    log_info "Building PX4 SITL for Gazebo Harmonic (this may take 10-15 minutes)..."
    make px4_sitl gz_x500
    
    log_success "PX4-Autopilot installed and built for Gazebo Harmonic"
}

# =============================================================================
# Configure Environment
# =============================================================================

configure_environment() {
    log_step "Phase 5: Configuring Environment"
    
    log_info "Target QGC: $QGC_HOST_IP:$QGC_PORT"
    
    # Create environment file for PX4
    cat > "$HOME/.px4_env" << EOF
# PX4 Environment Configuration
# Generated by AeroGuardian setup script
# Date: $(date)

# Display for GUI (WSLg or VcXsrv)
export DISPLAY=${WINDOWS_IP}:0

# QGroundControl connection
export PX4_GCS_URL="udp://@${QGC_HOST_IP}:${QGC_PORT}"
export PX4_SIM_HOST_ADDR="${QGC_HOST_IP}"

# Gazebo settings
export GZ_SIM_RESOURCE_PATH="\$HOME/PX4-Autopilot/Tools/simulation/gz/models:\$HOME/PX4-Autopilot/Tools/simulation/gz/worlds"

# MAVLink ports
export PX4_MAVSDK_PORT=$MAVSDK_PORT
export PX4_QGC_PORT=$QGC_PORT

# Silence deprecation warnings
export IGN_GAZEBO_SYSTEM_PLUGIN_PATH=""
EOF

    # Add to bashrc if not already there
    if ! grep -q "source.*px4_env" "$HOME/.bashrc"; then
        echo "" >> "$HOME/.bashrc"
        echo "# PX4 Environment" >> "$HOME/.bashrc"
        echo "[ -f ~/.px4_env ] && source ~/.px4_env" >> "$HOME/.bashrc"
    fi
    
    source "$HOME/.px4_env"
    
    log_success "Environment configured"
    log_info "  Host IP: $QGC_HOST_IP"
    log_info "  Port: $QGC_PORT"
}

# =============================================================================
# Create Launcher Scripts
# =============================================================================

create_launchers() {
    log_step "Phase 6: Creating Launcher Scripts"
    
    # Gazebo Harmonic launcher (GUI)
    cat > "$HOME/launch_px4_gz.sh" << 'LAUNCHER_SCRIPT'
#!/bin/bash
# =============================================================================
# PX4 + Gazebo Harmonic Launcher
# Usage: ./launch_px4_gz.sh [--headless]
# =============================================================================

source "$HOME/.px4_env" 2>/dev/null || true

PX4_DIR="$HOME/PX4-Autopilot"
QGC_HOST="${PX4_SIM_HOST_ADDR:-APT::Update::Post-Invoke-Success {"if /usr/bin/test -w /var/lib/command-not-found/ -a -e /usr/lib/cnf-update-db; then /usr/lib/cnf-update-db > /dev/null; fi";};}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  AeroGuardian PX4 + Gazebo Harmonic Launcher"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Vehicle: X500 Quadcopter"
echo "  Simulator: Gazebo Harmonic (gz sim)"
echo "  QGC Target: $QGC_HOST:${PX4_QGC_PORT:-18570}"
echo ""

cd "$PX4_DIR"
export PX4_SIM_HOST_ADDR="$QGC_HOST"

if [[ "$1" == "--headless" ]]; then
    echo "[+] Starting PX4 SITL + Gazebo (headless)..."
    HEADLESS=1 make px4_sitl gz_x500
else
    echo "[+] Starting PX4 SITL + Gazebo (GUI)..."
    make px4_sitl gz_x500
fi
LAUNCHER_SCRIPT

    chmod +x "$HOME/launch_px4_gz.sh"
    
    # Headless launcher
    cat > "$HOME/launch_px4_headless.sh" << 'HEADLESS_SCRIPT'
#!/bin/bash
# Headless PX4 SITL Launcher (no GUI, for automation)
source "$HOME/.px4_env" 2>/dev/null || true

PX4_DIR="$HOME/PX4-Autopilot"
QGC_HOST="${PX4_SIM_HOST_ADDR:-172.27.166.100}"

echo "Starting PX4 SITL with Gazebo Harmonic (headless)"
echo "QGroundControl target: $QGC_HOST:${PX4_QGC_PORT:-18570}"
echo ""

cd "$PX4_DIR"
export PX4_SIM_HOST_ADDR="$QGC_HOST"
HEADLESS=1 make px4_sitl gz_x500
HEADLESS_SCRIPT

    chmod +x "$HOME/launch_px4_headless.sh"
    
    # Gazebo Classic launcher (alternative)
    cat > "$HOME/launch_px4_classic.sh" << 'CLASSIC_SCRIPT'
#!/bin/bash
# PX4 + Gazebo Classic Launcher (alternative)
source "$HOME/.px4_env" 2>/dev/null || true

PX4_DIR="$HOME/PX4-Autopilot"
QGC_HOST="${PX4_SIM_HOST_ADDR:-172.27.166.100}"

echo "Starting PX4 SITL with Gazebo Classic (iris)"
cd "$PX4_DIR"
export PX4_SIM_HOST_ADDR="$QGC_HOST"
make px4_sitl gazebo-classic_iris
CLASSIC_SCRIPT

    chmod +x "$HOME/launch_px4_classic.sh"
    
    log_success "Launcher scripts created:"
    log_info "  Gazebo Harmonic:    ~/launch_px4_gz.sh [--headless]"
    log_info "  Headless mode:      ~/launch_px4_headless.sh"
    log_info "  Gazebo Classic:     ~/launch_px4_classic.sh"
}

# =============================================================================
# Validation
# =============================================================================

validate_installation() {
    log_step "Phase 7: Validating Installation"
    
    local errors=0
    
    # Check PX4
    if [ -f "$PX4_DIR/build/px4_sitl_default/bin/px4" ]; then
        log_success "PX4 SITL binary found"
    else
        log_error "PX4 SITL binary not found"
        ((errors++))
    fi
    
    # Check Gazebo Harmonic
    if command -v gz &>/dev/null; then
        log_success "Gazebo Harmonic available"
        gz --version | head -1
    else
        log_warn "Gazebo Harmonic not found (optional)"
    fi
    
    # Check Gazebo Classic
    if command -v gazebo &>/dev/null; then
        log_success "Gazebo Classic available"
    else
        log_warn "Gazebo Classic not found (optional)"
    fi
    
    # Check MAVSDK
    if python3 -c "import mavsdk" 2>/dev/null; then
        log_success "MAVSDK Python package available"
    else
        log_warn "MAVSDK Python not installed"
        ((errors++))
    fi
    
    # Check launcher scripts
    if [ -x "$HOME/launch_px4_gz.sh" ]; then
        log_success "Launcher scripts ready"
    else
        log_warn "Launcher scripts not found"
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
    WSL_IP=$(hostname -I | awk '{print $1}')
    
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  SETUP COMPLETE${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "  ${BOLD}WSL IP Address:${NC} ${YELLOW}$WSL_IP${NC}"
    echo ""
    echo -e "  ${BOLD}To launch PX4 with Gazebo Harmonic:${NC}"
    echo -e "    ${GREEN}~/launch_px4_gz.sh${NC}           # With GUI"
    echo -e "    ${GREEN}~/launch_px4_gz.sh --headless${NC} # Headless"
    echo ""
    echo -e "  ${BOLD}From Windows PowerShell:${NC}"
    echo -e "    ${GREEN}\$wsl_ip = (wsl -- hostname -I).Trim().Split()[0]${NC}"
    echo -e "    ${GREEN}python scripts/run_automated_pipeline.py -r 0 --wsl-ip \$wsl_ip --headless -s gz_x500${NC}"
    echo ""
    echo -e "  ${BOLD}Log file:${NC} $LOG_FILE"
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# =============================================================================
# Help
# =============================================================================

show_help() {
    echo "AeroGuardian PX4 + Gazebo Setup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --install-deps      Install system dependencies"
    echo "  --install-px4       Install PX4-Autopilot (v$PX4_VERSION)"
    echo "  --install-gazebo    Install Gazebo Harmonic (recommended)"
    echo "  --configure-only    Only configure environment, don't install"
    echo "  --help, -h          Show this help message"
    echo ""
    echo "Recommended first-time setup:"
    echo "  $0 --install-deps --install-px4 --install-gazebo"
    echo ""
    echo "Environment variables:"
    echo "  QGC_HOST_IP         QGroundControl host IP (default: 172.27.166.100)"
    echo "  QGC_PORT            QGroundControl port (default: 18570)"
    echo ""
}

# =============================================================================
# Main
# =============================================================================

main() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  AeroGuardian PX4 + Gazebo Setup (v2.0)${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    # Parse arguments
    INSTALL_DEPS=false
    INSTALL_PX4=false
    INSTALL_GAZEBO=false
    CONFIGURE_ONLY=false
    
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    for arg in "$@"; do
        case $arg in
            --install-deps)
                INSTALL_DEPS=true
                ;;
            --install-px4)
                INSTALL_PX4=true
                ;;
            --install-gazebo)
                INSTALL_GAZEBO=true
                ;;
            --configure-only)
                CONFIGURE_ONLY=true
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                echo "Unknown option: $arg"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Always detect environment
    detect_environment
    
    if [ "$CONFIGURE_ONLY" = true ]; then
        configure_environment
        create_launchers
        validate_installation
        print_summary
        exit 0
    fi
    
    if [ "$INSTALL_DEPS" = true ]; then
        install_dependencies
    fi
    
    if [ "$INSTALL_GAZEBO" = true ]; then
        install_gazebo_harmonic
    fi
    
    if [ "$INSTALL_PX4" = true ]; then
        install_px4
    fi
    
    # Always configure and create launchers
    configure_environment
    create_launchers
    validate_installation
    print_summary
}

# Run
main "$@"
