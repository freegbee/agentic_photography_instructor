import os
import sys
import webbrowser
from threading import Timer

try:
    from optuna_dashboard import run_server
except ImportError:
    print("Fehler: 'optuna-dashboard' ist nicht installiert.")
    print("Bitte installiere es mit: pip install optuna-dashboard")
    sys.exit(1)


def main():
    # Pfad zur Datenbank relativ zu diesem Skript (gleiche Logik wie in optimize_ppo.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    storage_dir = os.path.join(script_dir, "optuna_studies")
    db_path = os.path.join(storage_dir, "ppo_optimization.db")
    
    # SQLite URL für Optuna
    storage_url = f"sqlite:///{db_path}"
    
    print("==================================================")
    print("           Start Optuna Dashboard                 ")
    print("==================================================")
    print(f"Datenbank: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"WARNUNG: Die Datenbankdatei existiert nicht!")
        print(f"Pfad: {db_path}")
        print("Bitte führe zuerst 'optimize_ppo.py' aus, um Daten zu generieren.")
    
    host = "127.0.0.1"
    port = 8080
    dashboard_url = f"http://{host}:{port}"
    
    print(f"Dashboard URL: {dashboard_url}")
    print("Drücke STRG+C zum Beenden.")

    # Browser automatisch öffnen nach 1.5 Sekunden
    def open_browser():
        print(f"Öffne Browser: {dashboard_url}")
        webbrowser.open(dashboard_url)
        
    Timer(1.5, open_browser).start()

    # Server starten (blockiert, bis User abbricht)
    run_server(storage_url, host=host, port=port)


if __name__ == "__main__":
    main()