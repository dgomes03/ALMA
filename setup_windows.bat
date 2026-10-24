@echo off
REM ============================================================
REM  ALMA - one-time Windows setup
REM  Double-click this file once. When it finishes, use
REM  run_windows.bat to start the app.
REM  For the benchmark/evaluation extras, run from a terminal:
REM      setup_windows.bat --benchmark
REM ============================================================
setlocal
cd /d "%~dp0"

set "REQS=requirements.txt"
if /i "%~1"=="--benchmark" set "REQS=requirements-benchmark.txt"

echo(
echo ============================================================
echo   ALMA - Windows setup
echo ============================================================
echo(

REM --- 1. Find a supported Python (3.10 to 3.13) ----------------
REM  The "py" launcher finds every installed version, so a newer
REM  Python (or the Microsoft Store placeholder) earlier on PATH
REM  does not get in the way.
set "PY="
for %%V in (3.13 3.12 3.11 3.10) do (
    if not defined PY (
        py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
    )
)
if not defined PY (
    python -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo [ERROR] No Python between 3.10 and 3.13 was found.
    echo(
    echo   Install Python 3.13 from:
    echo       https://www.python.org/downloads/windows/
    echo   Other versions can stay installed. Then run this setup again.
    echo(
    pause
    exit /b 1
)

echo Using this Python:
%PY% --version
echo(

REM --- 2. Create a virtual environment (.venv) ------------------
REM  A .venv made earlier with an unsupported Python is replaced.
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)" >nul 2>&1
    if errorlevel 1 (
        echo The existing .venv uses an unsupported Python - recreating it.
        rmdir /s /q ".venv"
    )
)
if exist ".venv\Scripts\python.exe" (
    echo Virtual environment already exists ^(.venv^) - reusing it.
) else (
    echo Creating virtual environment in .venv ...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Could not create the virtual environment.
        pause
        exit /b 1
    )
)
echo(

REM --- 3. Install the Python packages ---------------------------
echo Upgrading pip ...
".venv\Scripts\python.exe" -m pip install --upgrade pip

echo(
echo Installing packages from %REQS% ...
echo ^(this can take several minutes and downloads ~2-3 GB^)
".venv\Scripts\python.exe" -m pip install -r %REQS%
if errorlevel 1 (
    echo(
    echo [ERROR] Package installation failed. Scroll up to see which one.
    echo   Most common fix: install the "Microsoft C++ Build Tools" from
    echo   https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo   then run this setup again.
    pause
    exit /b 1
)
echo(

REM --- 4. Pre-download the small NLTK data it needs -------------
echo Downloading NLTK data ...
".venv\Scripts\python.exe" -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('punkt', quiet=True)"
echo(

if not exist "documents" mkdir "documents"
if not exist "model" mkdir "model"

echo ============================================================
echo   Setup finished successfully.
echo ============================================================
echo(
echo   STILL TO DO before the first run:
echo(
echo   1) Install Ollama for Windows and keep it running:
echo          https://ollama.com/download
echo      On the first start ALMA asks which Ministral 3 model to pull.
echo(
echo   2) Put the files you want to search in the documents folder.
echo(
echo   3) (Optional) For OCR of scanned PDFs/images, install
echo      Tesseract:  https://github.com/UB-Mannheim/tesseract/wiki
echo      In the installer, under "Additional language data",
echo      tick Portuguese. Without Tesseract, text documents
echo      still work fine.
echo(
echo   When ready, double-click:  run_windows.bat
echo   To change settings later, double-click:  settings_windows.bat
echo(
set /p "OPEN_MENU=Open the settings menu now? [y/N] "
if /i "%OPEN_MENU%"=="y" ".venv\Scripts\python.exe" -m core.config_menu
pause
endlocal
