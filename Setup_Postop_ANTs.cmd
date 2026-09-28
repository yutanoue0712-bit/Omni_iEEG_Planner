@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Set up the main application environment first. See README.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m venv ".venv-ants"
if errorlevel 1 goto failed
set "ANTS_REQUIREMENTS=requirements-postop-ants.txt"
if exist "requirements-postop-ants-lock.txt" set "ANTS_REQUIREMENTS=requirements-postop-ants-lock.txt"
".venv-ants\Scripts\python.exe" -m pip install --only-binary=:all: -r "%ANTS_REQUIREMENTS%"
if errorlevel 1 goto failed
".venv-ants\Scripts\python.exe" -c "import ants; print('ANTsPy ready:', ants.__version__)"
if errorlevel 1 goto failed
echo Ready. Restart Omni-iEEG planner.
pause
exit /b 0
:failed
echo Setup did not complete. The main application environment was not changed.
pause
exit /b 1
