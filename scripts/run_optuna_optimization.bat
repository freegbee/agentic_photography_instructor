@echo off
REM Change working directory to project root (one level up from scripts folder)
cd /d "%~dp0.."

REM Load environment variables from .env file if it exists
if exist ".env" (
    echo Loading environment variables from .env...
    for /f "eol=# tokens=*" %%i in (.env) do set "%%i"
)

call .venv\Scripts\activate
set PYTHONPATH=src

echo Start Optuna Optimization...
echo PYTHONPATH set to: %PYTHONPATH%

python src/training/stable_baselines/training/optimize_ppo.py

pause