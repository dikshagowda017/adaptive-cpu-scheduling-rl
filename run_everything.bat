@echo off
REM ======================================================================
REM  Double-click this file to set up AND verify the whole project.
REM  First run: creates the "venv" folder and installs the libraries
REM  (needs internet, a few minutes). Later runs: about 3-4 minutes.
REM  Add --fast to skip mutation testing:  run_everything.bat --fast
REM ======================================================================
setlocal
cd /d "%~dp0"
title Adaptive CPU Scheduling - full verification

REM ---- 1. Find Python 3.10 or newer --------------------------------
set "PY="
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if not errorlevel 1 set "PY=python"
if defined PY goto :check_venv
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if not errorlevel 1 set "PY=py -3"
if not defined PY goto :nopython

REM ---- 2. Make sure the venv works ON THIS COMPUTER ------------------
REM A venv copied from another PC points to that PC's Python and is broken.
:check_venv
if not exist "venv\Scripts\python.exe" goto :create
"venv\Scripts\python.exe" -c "import sys" >nul 2>nul
if not errorlevel 1 goto :deps
echo The venv folder was made on another computer and does not work here.
echo Rebuilding it...
rmdir /s /q venv

:create
echo [setup] Creating the virtual environment - first run only...
%PY% -m venv venv
if errorlevel 1 goto :fail

:deps
"venv\Scripts\python.exe" -c "import gymnasium, numpy, pandas, matplotlib, pytest, pytest_cov, hypothesis" >nul 2>nul
if not errorlevel 1 goto :run
echo [setup] Installing libraries from requirements.txt - needs internet, a few minutes...
"venv\Scripts\python.exe" -m pip install --upgrade pip
"venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :fail

REM ---- 3. Run everything --------------------------------------------
:run
"venv\Scripts\python.exe" verify_all.py %*
if errorlevel 1 echo. & echo Some checks failed - scroll up to see which step.
goto :end

:nopython
echo.
echo Python 3.10 or newer was not found on this computer.
echo Install it from https://www.python.org/downloads/ and TICK "Add python.exe to PATH",
echo then double-click this file again.
goto :end

:fail
echo.
echo Setup failed. Check your internet connection and try again.
echo If it keeps failing, delete the "venv" folder and run this file again.

:end
echo.
pause
