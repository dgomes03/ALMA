@echo off
REM ============================================================
REM  ALMA - start the app on Windows
REM  Double-click this after setup_windows.bat has finished.
REM  Settings menu first:  run_windows.bat --config
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

echo Starting ALMA ...
echo   ^(first run downloads the embedding + reranker models from
echo    Hugging Face - this needs internet and takes a few minutes^)
echo(

".venv\Scripts\python.exe" main.py %*

echo(
echo The app has stopped.
pause
endlocal
