@echo off
REM =============================================================================
REM AeroGuardian Setup Script for Windows
REM =============================================================================
REM Author: AeroGuardian Member
REM Date: 2026-01-19
REM Version: 1.0
REM
REM This script sets up the AeroGuardian environment on Windows.
REM =============================================================================

echo.
echo =====================================================
echo   AeroGuardian Setup for Windows
echo =====================================================
echo.

REM Check Python version
python --version 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Create virtual environment if not exists
if not exist "venv" (
    echo [1/4] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
) else (
    echo [1/4] Virtual environment already exists
)

REM Activate virtual environment
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo [3/4] Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo [WARNING] Some dependencies may have failed to install
)

REM Check for .env file
echo [4/4] Checking configuration...
if not exist ".env" (
    echo.
    echo [WARNING] .env file not found!
    echo Creating template .env file...
    (
        echo # AeroGuardian Configuration
        echo # Add your OpenAI API key below
        echo OPENAI_API_KEY=sk-your-api-key-here
        echo OPENAI_MODEL=gpt-4o-mini
    ) > .env
    echo.
    echo [IMPORTANT] Please edit .env and add your OPENAI_API_KEY
    echo.
)

echo.
echo =====================================================
echo   Setup Complete!
echo =====================================================
echo.
echo To run the automated pipeline:
echo   1. Edit .env with your OPENAI_API_KEY
echo   2. Start PX4 in WSL (optional - scripts/setup_px4_gui.sh)
echo   3. Run: python scripts/run_automated_pipeline.py --incident 0
echo.
echo Options:
echo   --incident N    Process incident #N
echo   --headless      No Gazebo GUI
echo   --skip-px4      Skip PX4 startup (if already running)
echo.
pause
