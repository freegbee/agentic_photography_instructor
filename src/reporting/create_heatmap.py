import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text

# Konfiguration
# HIER BITTE DEINEN CONNECTION STRING EINFÜGEN
DB_CONNECTION_STR = os.getenv('MLFLOW_BACKEND_STORE_URI')
RUN_UUID = 'eefeb7e047784d5e9584276505d2f12c'
OUTPUT_FILENAME = 'transformer_usage_heatmap.png'
TRANSFORMER_PREFIX = 'train_transformer_usage/'
TITLE="Haupttraining 1 - Training"
ROLLOUT_BATCH_STEPS=4000

def create_heatmap():
    # 1. Datenbankverbindung herstellen
    engine = create_engine(DB_CONNECTION_STR)

    # 2. SQL Query vorbereiten (mit f-string für die UUID, um es flexibel zu halten)
    query = f"""
    SELECT
        REPLACE(k.key, '{TRANSFORMER_PREFIX}', '') AS layer_metric,
        (s.step / {ROLLOUT_BATCH_STEPS}) as step,
        m.value
    FROM
        -- 1. Alle Steps generieren (X-Achse Gerüst)
        generate_series({ROLLOUT_BATCH_STEPS}, 300000, {ROLLOUT_BATCH_STEPS}) AS s(step)
    CROSS JOIN
        -- 2. Alle vorkommenden Keys ermitteln (Y-Achse Gerüst)
        (SELECT DISTINCT key FROM metrics
         WHERE run_uuid = '{RUN_UUID}' AND key LIKE '{TRANSFORMER_PREFIX}%') k
    LEFT JOIN
        metrics m
        ON m.step = s.step
        AND m.key = k.key
        AND m.run_uuid = '{RUN_UUID}'
    ORDER BY
        layer_metric, s.step;
    """

    print("Lade Daten aus der Datenbank...")
    with engine.connect() as connection:
        df = pd.read_sql(text(query), connection)

    if df.empty:
        print("Keine Daten gefunden. Bitte überprüfe die RUN_UUID.")
        return

    # Sicherstellen, dass die Werte numerisch sind (verhindert 'dtype object' Fehler)
    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    # 3. Daten pivotisieren (Long-Format zu Wide-Format für die Heatmap)
    # Index = Zeilen (Layer), Columns = Spalten (Steps), Values = Farbe (Value)
    heatmap_data = df.pivot(index='layer_metric', columns='step', values='value')

    # 4. Plot erstellen
    plt.figure(figsize=(16, 5))  # Höhe verringert (z.B. 5), damit die Zeilen flacher sind
    
    # Erstellen der Heatmap
    # cmap="Blues" sorgt für den Verlauf von blassblau zu kräftigem Blau
    # cbar_kws labelt die Farbskala
    ax = sns.heatmap(heatmap_data, cmap="Blues", linewidths=.5, cbar_kws={'label': 'Usage Value'})

    plt.title(f'{TITLE} - Transformer Usage Heatmap')
    plt.xlabel(f'Step (scaled / {ROLLOUT_BATCH_STEPS})')
    plt.ylabel('Transformer')
    
    # Layout straffen, damit nichts abgeschnitten wird
    plt.tight_layout()

    # Speichern und Anzeigen
    plt.savefig(OUTPUT_FILENAME, dpi=300)
    print(f"Grafik gespeichert unter: {OUTPUT_FILENAME}")
    # plt.show() # Optional: Fenster öffnen, falls gewünscht

def create_combined_heatmap():
    # 1. Datenbankverbindung herstellen
    engine = create_engine(DB_CONNECTION_STR)

    # 2. SQL Query für Train und Eval
    query = f"""
    SELECT
--         k.key AS layer_metric,
        CASE 
            WHEN k.key LIKE 'train_transformer_usage/%' THEN REPLACE(k.key, 'train_transformer_usage/', '')
            WHEN k.key LIKE 'eval_transformer_usage/%' THEN REPLACE(k.key, 'eval_transformer_usage/', '')
        END AS layer_metric,
        CASE 
            WHEN k.key LIKE 'train_transformer_usage/%' THEN 'train'
            ELSE 'eval'
        END AS metric_type,
        (s.step / {ROLLOUT_BATCH_STEPS}) as step,
        m.value
    FROM
        generate_series({ROLLOUT_BATCH_STEPS}, 300000, {ROLLOUT_BATCH_STEPS}) AS s(step)
    CROSS JOIN
        (SELECT DISTINCT key FROM metrics
         WHERE run_uuid = '{RUN_UUID}' 
         AND (key LIKE 'train_transformer_usage/%' OR key LIKE 'eval_transformer_usage/%')) k
    LEFT JOIN
        metrics m
        ON m.step = s.step
        AND m.key = k.key
        AND m.run_uuid = '{RUN_UUID}'
    ORDER BY
        layer_metric, metric_type, s.step;
    """

    print("Lade kombinierte Daten (Train & Eval)...")
    with engine.connect() as connection:
        df = pd.read_sql(text(query), connection)

    if df.empty:
        print("Keine Daten gefunden.")
        return

    # Sicherstellen, dass die Werte numerisch sind (verhindert 'dtype object' Fehler)
    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    # 3. Pivotisieren
    heatmap_data = df.pivot_table(index=['layer_metric', 'metric_type'], columns='step', values='value', fill_value=0)

    # Sortieren und Reindexieren: Pro Transformer erst Train (oben), dann Eval (unten)
    transformers = sorted(df['layer_metric'].unique())
    new_index = []
    for t in transformers:
        new_index.append((t, 'train'))
        new_index.append((t, 'eval'))
    
    heatmap_data = heatmap_data.reindex(new_index, fill_value=0)

    # 4. Daten für Plot vorbereiten: Eval-Werte negativ machen für die Farbcodierung
    plot_data = heatmap_data.copy()
    # Jede zweite Zeile (Eval) mit -1 multiplizieren
    plot_data.iloc[1::2] *= -1

    # 5. Custom Colormap: Orange (negativ) bis Blau (positiv)
    colors_neg = plt.cm.Oranges(np.linspace(1, 0, 128))
    colors_pos = plt.cm.Blues(np.linspace(0, 1, 128))
    combined_colors = np.vstack((colors_neg, colors_pos))
    cmap = mcolors.LinearSegmentedColormap.from_list('OrangeBlue', combined_colors)

    # Skalierung symmetrisch machen
    max_val = plot_data.abs().max().max()

    # Plot erstellen
    plt.figure(figsize=(16, len(transformers) * 0.4 + 2))
    
    ax = sns.heatmap(plot_data, cmap=cmap, vmin=-max_val, vmax=max_val, linewidths=0, cbar=False)

    # Y-Achsen Beschriftung anpassen (Zentriert zwischen Train und Eval Zeile)
    ax.set_yticks([i * 2 + 1 for i in range(len(transformers))])
    ax.set_yticklabels(transformers, rotation=0)
    ax.set_ylabel('Transformer')
    ax.set_xlabel(f'Rollout Batch (Steps / {ROLLOUT_BATCH_STEPS})')
    plt.title(f'{TITLE} - Kombinierter Transformernutzung (blau: Training, orange: Validierung)')

    # Trennlinien zwischen den Transformern (alle 2 Zeilen)
    for i in range(2, len(transformers) * 2, 2):
        ax.hlines(i, *ax.get_xlim(), colors='white', linewidths=2)

    # Eigene Colorbar hinzufügen
    cbar = plt.colorbar(ax.collections[0], ax=ax, pad=0.02, aspect=30)
    cbar.set_label('Usage Value')
    ticks = cbar.get_ticks()
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{int(abs(t))}" for t in ticks])

    plt.tight_layout()
    output_filename = 'combined_transformer_usage_heatmap.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Kombinierte Grafik gespeichert unter: {output_filename}")

if __name__ == "__main__":
    create_heatmap()
    create_combined_heatmap()