@echo off
REM =============================================================================
REM AeroGuardian Setup Script for Windows
REM =============================================================================
REM Author: AeroGuardian Team (Tiny Coders)
REM Date: 2026-02-04
REM Version: 1.0
REM
REM This script sets up the AeroGuardian Python environment on Windows.
REM For PX4/Gazebo setup in WSL2, see scripts/setup_px4_gui.sh
REM =============================================================================

setlocal enabledelayedexpansion

echo.
echo ======================================================================
echo   AeroGuardian Setup for Windows
echo   Version 1.0
echo ======================================================================
echo.

REM Check Python version
echo [1/5] Checking Python installation...
python --version 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+
    echo Download from: https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation!
    pause
    exit /b 1
)

REM Verify Python version is 3.10+
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    if %%a LSS 3 (
        echo [ERROR] Python 3.10+ required, found %PYVER%
        pause
        exit /b 1
    )
    if %%a EQU 3 if %%b LSS 10 (
        echo [ERROR] Python 3.10+ required, found %PYVER%
        pause
        exit /b 1
    )
)
echo [OK] Python %PYVER% detected

REM Create virtual environment if not exists
echo.
echo [2/5] Setting up virtual environment...
if not exist "venv" (
    echo       Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)

REM Activate virtual environment
echo.
echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated

REM Install/upgrade pip and dependencies
echo.
echo [4/5] Installing dependencies...
echo       This may take 2-5 minutes on first run...
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [WARNING] Some dependencies may have failed to install
    echo          Try running: pip install -r requirements.txt
) else (
    echo [OK] All dependencies installed
)

REM Verify critical packages
echo.
echo       Verifying critical packages...
python -c "import mavsdk" 2>nul
if errorlevel 1 (
    echo [WARNING] MAVSDK not installed correctly
    echo          Run: pip install mavsdk
) else (
    echo       - MAVSDK: OK
)

python -c "import dspy" 2>nul
if errorlevel 1 (
    echo [WARNING] DSPy not installed correctly
    echo          Run: pip install dspy-ai
) else (
    echo       - DSPy: OK
)

python -c "import openai" 2>nul
if errorlevel 1 (
    echo [WARNING] OpenAI not installed correctly
    echo          Run: pip install openai
) else (
    echo       - OpenAI: OK
)

REM Check for .env file
echo.
echo [5/5] Checking configuration...
if not exist ".env" (
    echo.
    echo [WARNING] .env file not found!
    echo          Creating template .env file...
    (
        echo # AeroGuardian Configuration
        echo # Replace with your actual OpenAI API key
        echo OPENAI_API_KEY=sk-your-api-key-here
        echo OPENAI_MODEL=gpt-4o
    ) > .env
    echo.
    echo [IMPORTANT] Please edit .env and add your OPENAI_API_KEY
    echo             The file has been created with a template.
    echo.
) else (
    REM Check if API key is set (not the placeholder)
    findstr /C:"sk-your-api-key-here" .env >nul 2>&1
    if not errorlevel 1 (
        echo [WARNING] .env contains placeholder API key
        echo          Please edit .env and add your real OPENAI_API_KEY
    ) else (
        echo [OK] .env file configured
    )
)

REM Check WSL2 availability
echo.
echo Checking WSL2 availability...
wsl --status >nul 2>&1
if errorlevel 1 (
    echo [WARNING] WSL2 not detected or not running
    echo          WSL2 is required for PX4 SITL simulation
    echo          Install from: https://docs.microsoft.com/en-us/windows/wsl/install
) else (
    echo [OK] WSL2 is available
    echo.
    echo Getting WSL IP address...
    for /f "tokens=*" %%i in ('wsl -- hostname -I 2^>nul') do (
        for /f "tokens=1" %%j in ("%%i") do set WSL_IP=%%j
    )
    if defined WSL_IP (
        echo [OK] WSL IP: !WSL_IP!
    ) else (
        echo [WARNING] Could not determine WSL IP
        echo          Run in WSL: ip addr show eth0
    )
)

echo.
echo ======================================================================
echo   Setup Complete!
echo ======================================================================
echo.
echo NEXT STEPS:
echo.
echo 1. Edit .env and add your OPENAI_API_KEY (if not already done)
echo.
echo 2. Setup PX4 in WSL2 (if not already done):
echo    wsl
echo    cd /mnt/c/path/to/aero-guardian/scripts
echo    chmod +x setup_px4_gui.sh
echo    ./setup_px4_gui.sh --install-deps --install-px4
echo.
echo 3. Run the pipeline:
echo    .\venv\Scripts\activate
echo    $wsl_ip = (wsl -- hostname -I).Trim().Split()[0]
echo    python scripts/run_automated_pipeline.py -r 0 --wsl-ip $wsl_ip --headless -s gz_x500
echo.
echo ======================================================================
echo.
pause
