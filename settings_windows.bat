@echo off
REM ============================================================
REM  ALMA - open the settings menu on Windows
REM  Double-click this after setup_windows.bat has finished.
REM ============================================================
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] No virtual environment found.
    echo   Run setup_windows.bat first.
    echo(
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m core.config_menu

echo(
pause
endlocal
