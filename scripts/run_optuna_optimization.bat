@echo off
REM Skript zum Starten der Optuna-Optimierung ausserhalb der IDE
REM Annahme: Dieses Skript liegt im Ordner "scripts" im Repository-Root

REM Wechselt in das Verzeichnis über dem Skript-Verzeichnis (Repository Root)
cd /d "%~dp0.."

REM Setzt den PYTHONPATH auf den src Ordner, damit Python die Module findet
set PYTHONPATH=%CD%\src

echo Start Optuna Optimization...
echo PYTHONPATH set to: %PYTHONPATH%

python src/training/stable_baselines/training/optimize_ppo.py

pause