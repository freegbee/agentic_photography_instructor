import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sqlalchemy import create_engine, text

# Konfiguration
# HIER BITTE DEINEN CONNECTION STRING EINFÜGEN
DB_CONNECTION_STR = os.getenv('MLFLOW_BACKEND_STORE_URI')

# Definition der Runs und ihrer Labels
# Korrektur basierend auf deinem SQL Snippet:
# eefeb... -> Haupttraining 1 (im SQL stand es als erstes Filter Argument, aber das Label war "Haupttraining 1")
# Warte, im SQL stand:
# run_uuid = 'eefeb...' AS "Haupttraining 1"
# run_uuid = '105c4...' AS "Haupttraining 2"
# run_uuid = '0a0de...' AS "Haupttraining 3"
# Ich korrigiere das Dictionary:

RUNS = {
    'eefeb7e047784d5e9584276505d2f12c': 'Haupttraining 1',
    '105c494bca5943cdbb78d7365291d505': 'Haupttraining 2',
    '0a0deea790034af3bb521bbc61dbbcbe': 'Haupttraining 3'
}

def get_db_engine():
    return create_engine(DB_CONNECTION_STR)

def load_data(metric_base):
    print(f"Lade Daten für {metric_base}...")
    run_uuids = list(RUNS.keys())
    run_uuids_str = "', '".join(run_uuids)
    
    # Wir laden Train UND Eval Daten in einer Abfrage
    query = f"""
    SELECT 
        m.step,
        m.value,
        m.run_uuid,
        CASE 
            WHEN m.key LIKE 'train/%' THEN 'Train' 
            ELSE 'Eval' 
        END as metric_type
    FROM metrics m
    WHERE 
        m.run_uuid IN ('{run_uuids_str}')
        AND (m.key = 'train/{metric_base}' OR m.key = 'eval/{metric_base}')
    ORDER BY m.step ASC
    """

    engine = get_db_engine()
    with engine.connect() as connection:
        df = pd.read_sql(text(query), connection)

    if df.empty:
        print(f"Keine Daten für {metric_base} gefunden.")
        return None

    # Datentypen anpassen
    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    # X-Achse Skalierung:
    # Falls die Steps groß sind (z.B. 4000, 8000...), skalieren wir sie runter.
    # Falls sie klein sind (0..74), nutzen wir sie direkt.
    if df['step'].max() > 1000:
        df['step'] = df['step'] / 4000

    # Sicherstellen, dass wir bei 1 anfangen (Rollout Batch 1-75)
    if df['step'].min() == 0:
        df['step'] = df['step'] + 1
    
    # Run UUIDs durch lesbare Namen ersetzen
    df['Run'] = df['run_uuid'].map(RUNS)

    return df

def create_plot(df, title, ylabel, filename):
    if df is None or df.empty:
        return

    # Sortierung für die Legende (Alphabetisch)
    sorted_runs = sorted(df['Run'].unique())

    # Schriftgröße verdoppeln (font_scale=2.0) für bessere Lesbarkeit
    sns.set_context("notebook", font_scale=2.0)

    # Wir plotten zweimal, um volle Kontrolle über Style (Alpha/Dicke) zu haben
    plt.figure(figsize=(16,9))

    # 1. Training: Dünner und transparenter (alpha)
    sns.lineplot(
        data=df[df['metric_type'] == 'Train'],
        x='step',
        y='value',
        hue='Run',
        hue_order=sorted_runs,
        linewidth=1.5,  # Etwas dünner
        alpha=0.5,  # Heller/Transparenter
        legend=False  # Keine Legende hier, sonst haben wir doppelte Einträge
    )

    # 2. Eval: Dick und voll deckend
    sns.lineplot(
        data=df[df['metric_type'] == 'Eval'],
        x='step',
        y='value',
        hue='Run',
        hue_order=sorted_runs,
        linewidth=2.5,  # Dick
        alpha=1.0  # Volle Deckkraft
    )

    # Legende anpassen:
    # 1. Bestehende Handles (Farben für die Runs) holen
    handles, labels = plt.gca().get_legend_handles_labels()

    # 2. Manuelle Einträge für den Linienstil (Train vs Eval) erstellen
    # Wir nutzen neutrale graue Linien, um nur den Stil (Dicke/Alpha) zu zeigen
    line_train = Line2D([0], [0], color='gray', linewidth=1.5, alpha=0.5, label='Training')
    line_eval = Line2D([0], [0], color='gray', linewidth=2.5, alpha=1.0, label='Validierung')

    # 3. Alles zusammenfügen
    # Wir fügen die Stil-Erklärung am Ende der Legende an
    plt.legend(handles=handles + [line_train, line_eval], loc='best')

    plt.title(title)
    plt.xlabel('Rollout Batch')
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(filename, dpi=300)
    print(f"Grafik gespeichert unter: {filename}")
    plt.close()

def plot_mean_reward():
    df = load_data('mean_reward')
    create_plot(df, 'Mean Reward Verlauf (Train vs Eval)', 'Mean Reward', 'comparison_mean_reward.png')

def plot_mean_episodes_len():
    df = load_data('mean_episodes_len')
    create_plot(df, 'Mean Episode Length Verlauf (Train vs Eval)', 'Mean Episode Length', 'comparison_mean_episodes_len.png')

if __name__ == "__main__":
    plot_mean_reward()
    plot_mean_episodes_len()