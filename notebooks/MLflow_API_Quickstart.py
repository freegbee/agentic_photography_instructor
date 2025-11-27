#%% md
# MLflow API Quickstart (im Notebook-Container)

# Dieses Notebook (Python-Script mit Notebook-Zellen) zeigt, wie Sie per MLflow-API
# Experimente, Runs und Artefakte abfragen. Es liegt lokal unter
# `notebooks/MLflow_API_Quickstart.py` und ist im Container unter
# `/workspace/notebooks/MLflow_API_Quickstart.py` sichtbar.

# Voraussetzungen:
# - Notebook-Container läuft via Docker Compose.
# - `mlruns`-Volume ist gemountet (bereits in docker-compose eingerichtet).
# - `MLFLOW_TRACKING_URI` ist gesetzt (z. B. `http://mlflow:${MLFLOW_PORT}`).

# Hinweis: Diese Datei nutzt Jupyter-kompatible Zellmarker (`#%%`). In JupyterLab können Sie
# sie wie ein Notebook ausführen (falls Jupytext installiert ist), sonst einfach als
# Script in Zellen ausführen. Alternativ können Sie den Inhalt in ein klassisches
# `.ipynb`-Notebook kopieren.

#%%
# Optional: Fehlende Pakete nachinstallieren (nur wenn nötig).
# In der Regel sind 'mlflow' und 'pandas' bereits über requirements installiert.
try:
    import mlflow  # noqa: F401
    import pandas as pd  # noqa: F401
except Exception:
    # Die Magie %pip funktioniert in Jupyter/IPython
    try:
        get_ipython().run_line_magic('pip', 'install -q mlflow pandas')  # type: ignore[name-defined]
    except Exception:
        pass

try:
    import matplotlib.pyplot as plt  # noqa: F401
    from PIL import Image  # noqa: F401
except Exception:
    try:
        get_ipython().run_line_magic('pip', 'install -q matplotlib pillow')  # type: ignore[name-defined]
    except Exception:
        pass

print('Pakete bereit.')

#%%
import os
import mlflow
from mlflow.tracking import MlflowClient

print('Tracking URI:', mlflow.get_tracking_uri())
print('MLFLOW_TRACKING_URI (env):', os.getenv('MLFLOW_TRACKING_URI'))

client = MlflowClient()  # nutzt automatisch MLFLOW_TRACKING_URI aus der Umgebung
experiments = client.search_experiments()
print('Experimente gefunden:', len(experiments))
for exp in experiments:
    print(f"- {exp.experiment_id}: {exp.name} (lifecycle_stage={exp.lifecycle_stage})")

#%%
# Läufe (Runs) eines Experiments suchen
from mlflow.entities import ViewType

exp_name = 'Default'  # ggf. anpassen (z. B. 'demo' oder Ihr Experimentname)
exp = client.get_experiment_by_name(exp_name)
if exp is None:
    raise ValueError(f"Experiment '{exp_name}' nicht gefunden – bitte Namen prüfen.")

runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string='',
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=20,
    order_by=['attributes.start_time DESC'],
)

print(f"Gefundene Runs in '{exp_name}': {len(runs)}")
for r in runs:
    print(r.info.run_id, r.data.params, r.data.metrics)

#%%
# Neuesten Run auswählen und Artefakte auflisten
if not runs:
    raise ValueError('Keine Runs gefunden. Bitte Training ausführen oder anderes Experiment wählen.')

run = runs[0]
run_id = run.info.run_id
print('Gewählter Run:', run_id)

def list_artifacts_recursive(client: MlflowClient, run_id: str, path: str = '') -> None:
    for item in client.list_artifacts(run_id, path):
        print(('DIR ' if item.is_dir else 'FILE'), item.path)
        if item.is_dir:
            list_artifacts_recursive(client, run_id, item.path)

list_artifacts_recursive(client, run_id)

#%%
# Beispiel: CSV-Artefakt laden und anzeigen
import tempfile
import pandas as pd

artifact_rel_path = 'metrics.csv'  # ggf. anpassen
with tempfile.TemporaryDirectory() as tmpdir:
    local_path = client.download_artifacts(run_id, artifact_rel_path, dst_path=tmpdir)
    print('Lokal gespeichert:', local_path)
    df = pd.read_csv(local_path)

print(df.head())

#%%
# Beispiel: Bild-Artefakt anzeigen
import tempfile
from PIL import Image
import matplotlib.pyplot as plt

image_rel_path = 'images/example.png'  # ggf. anpassen
with tempfile.TemporaryDirectory() as tmpdir:
    img_path = client.download_artifacts(run_id, image_rel_path, dst_path=tmpdir)

img = Image.open(img_path)
plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.axis('off')
plt.title(f'Run {run_id}: {image_rel_path}')
plt.show()

#%%
# Beispiel: Zeitreihen-Metrik (History) visualisieren
from pandas import DataFrame
import matplotlib.pyplot as plt

metric_key = 'loss'  # Beispiel
hist = client.get_metric_history(run_id, metric_key)
loss_df = DataFrame([{ 'step': m.step, 'value': m.value, 'timestamp': m.timestamp } for m in hist])
if not loss_df.empty:
    loss_df = loss_df.sort_values('step')
    ax = loss_df.plot(x='step', y='value', title=f'{metric_key} over steps')
    ax.set_xlabel('step')
    ax.set_ylabel(metric_key)
else:
    print(f"Keine History-Einträge für Metrik '{metric_key}' gefunden.")
