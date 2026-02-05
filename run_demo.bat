@echo off
REM ============================================================
REM AeroGuardian Demo Launcher
REM Quick demo script to run the automated pipeline
REM ============================================================
REM Author: AeroGuardian Team (Tiny Coders)
REM Date: 2026-02-04
REM Version: 2.0
REM
REM Usage: run_demo.bat [OPTIONS]
REM   Options are passed directly to run_automated_pipeline.py
REM
REM Examples:
REM   run_demo.bat                    # Process report #0
REM   run_demo.bat -r 5               # Process report #5
REM   run_demo.bat -r 10 --headless   # Process report #10 headless
REM ============================================================

setlocal enabledelayedexpansion

echo.
echo ======================================================================
echo   AEROGUARDIAN DEMO LAUNCHER v2.0
echo ======================================================================
echo.

REM Get script directory
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

REM Check if virtual environment exists
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found!
    echo         Please run setup.bat first.
    pause
    exit /b 1
)

REM Get WSL IP address
echo [1/3] Getting WSL IP address...
for /f "tokens=*" %%i in ('wsl -- hostname -I 2^>nul') do (
    for /f "tokens=1" %%j in ("%%i") do set WSL_IP=%%j
)

if not defined WSL_IP (
    echo [ERROR] Could not get WSL IP address!
    echo         Make sure WSL2 is running.
    pause
    exit /b 1
)
echo       WSL IP: %WSL_IP%

REM Check for .env file
echo.
echo [2/3] Checking configuration...
if not exist ".env" (
    echo [ERROR] .env file not found!
    echo         Please create .env with your OPENAI_API_KEY.
    pause
    exit /b 1
)
echo       .env file found

REM Run the automated pipeline
echo.
echo [3/3] Running AeroGuardian Pipeline...
echo.
echo ======================================================================
echo   Command: python scripts/run_automated_pipeline.py --wsl-ip %WSL_IP% --headless -s gz_x500 %*
echo ======================================================================
echo.

call .\venv\Scripts\python.exe scripts\run_automated_pipeline.py --wsl-ip %WSL_IP% --headless -s gz_x500 %*

echo.
echo ======================================================================
echo   Demo Complete!
echo ======================================================================
echo.
echo Output files saved to: outputs\
echo.
echo To view results:
echo   - Open the PDF report in outputs\{latest}\report\report.pdf
echo   - Check evaluation scores in outputs\{latest}\evaluation\evaluation.json
echo.
pause
