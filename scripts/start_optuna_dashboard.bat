@echo off
REM Skript zum Starten des Optuna Dashboards

REM Wechselt in das Verzeichnis über dem Skript-Verzeichnis (Repository Root)
cd /d "%~dp0.."

echo Starting Optuna Dashboard...
echo Database: src/training/stable_baselines/training/optuna_studies/ppo_optimization.db
echo Open http://127.0.0.1:8080 in your browser.

optuna-dashboard sqlite:///src/training/stable_baselines/training/optuna_studies/ppo_optimization.db

pause