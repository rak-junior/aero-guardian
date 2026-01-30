@echo off
REM ============================================================
REM AeroGuardian Demo Launcher
REM Starts X Server, QGC, and runs the automated pipeline
REM ============================================================
REM Author: AeroGuardian Member
REM Date: 2026-01-19
REM Version: 1.0
REM ============================================================

echo ============================================================
echo   AEROGUARDIAN DEMO LAUNCHER
echo ============================================================
echo.

REM Get script directory
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

REM Check if VcXsrv is installed
set VCXSRV_PATH="C:\Program Files\VcXsrv\vcxsrv.exe"
if not exist %VCXSRV_PATH% (
    set VCXSRV_PATH="C:\Program Files (x86)\VcXsrv\vcxsrv.exe"
)

REM Start VcXsrv if not running
tasklist /FI "IMAGENAME eq vcxsrv.exe" 2>NUL | find /I /N "vcxsrv.exe">NUL
if "%ERRORLEVEL%"=="1" (
    echo [1/4] Starting VcXsrv X Server...
    if exist %VCXSRV_PATH% (
        start "" %VCXSRV_PATH% :0 -multiwindow -clipboard -wgl -ac
        timeout /t 3 /nobreak > NUL
        echo       VcXsrv started successfully
    ) else (
        echo       WARNING: VcXsrv not found. Please install from https://sourceforge.net/projects/vcxsrv/
        echo       Continuing without X server...
    )
) else (
    echo [1/4] VcXsrv already running
)

REM Get Windows IP for WSL
echo [2/4] Getting Windows IP for WSL...
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do (
    for /f "tokens=1" %%b in ("%%a") do set WINDOWS_IP=%%b
)
echo       Windows IP: %WINDOWS_IP%

REM Start QGC in WSL (background)
echo [3/4] Starting QGroundControl in WSL...
start "" wsl -d Ubuntu -e bash -c "export DISPLAY=$(grep nameserver /etc/resolv.conf | awk '{print $2}'):0 && cd ~ && ./QGroundControl.AppImage --appimage-extract-and-run 2>/dev/null"
timeout /t 5 /nobreak > NUL
echo       QGC launched in WSL

REM Run the automated pipeline
echo [4/4] Running AeroGuardian Automated Pipeline...
echo.
echo ============================================================
call .\venv\Scripts\python.exe scripts\run_automated_pipeline.py %*

echo.
echo ============================================================
echo   Demo Complete!
echo ============================================================
echo.
echo Output files saved to: outputs\incidents\
echo Log file: logs\%date:~-4%-%date:~3,2%-%date:~0,2%.log
echo.
pause
