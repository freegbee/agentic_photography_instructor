DQN Agent (subset_training)
===========================

Dieser Ordner enthält eine kleine DQN-Agent-Implementierung mit einer alternativen Q-Netz-Klasse, die
ResNet18/ResNet50 als Vision-Backbone verwendet und danach eigene lineare Layer (Kopf) trainiert.

Zweck
-----
Die Implementierung erlaubt es, statt des einfachen Conv-Netzes (`SimpleQNetwork`) ein ResNet-Backbone
für die Feature-Extraktion zu nutzen und nur den Kopf (oder Kopf + Feintuning des Backbones)
für die Q-Werte zu trainieren.

Wichtige Dateien
----------------
- `DQNAgent.py` - Implementierung des `DQNAgent`, `SimpleQNetwork` und `ResNetFeatureQNetwork`.
- `tests/test_dqn_agent_resnet.py` - Unit-Tests / Smoke-Tests für das neue Netzwerk und die Utilities.

Design-Highlights
-----------------
- `ResNetFeatureQNetwork(input_shape, n_actions, backbone='resnet18'|'resnet50', pretrained=False, freeze_backbone=True)`
  - Nutzt das ResNet-Backbone als Feature-Extractor. `resnet.fc` wird durch `Identity` ersetzt und ein eigener MLP-Kopf
    (Linear -> ReLU -> Linear) erzeugt die `n_actions` Q-Werte.
  - Wenn `input_shape[0] != 3` wird automatisch ein 1x1 Conv verwendet, um auf 3 Kanäle zu mappen.
  - Unterstützt die moderne `weights=` API von `torchvision` mit Fallback auf `pretrained=` für ältere Versionen.

- `DQNAgent` Erweiterungen:
  - `network_constructor` und `network_kwargs` erlauben den Austausch des verwendeten Netzes
    (z. B. `ResNetFeatureQNetwork`).
  - Input-Preprocessing Utility: `_states_to_tensor(...)`
    - Skaliert automatisch 0-255 -> 0-1 (Heuristik) und wendet optional ImageNet-Normalisierung an.
    - Normalisierung gesteuert über `network_kwargs`: `pretrained`, `use_imagenet_norm`, `input_mean`, `input_std`.
  - Per-Parameter-Optimizer-Gruppen: falls das Netz `backbone` und `head` Module exportiert, erstellt der Agent
    separate Parameterguppen mit eigener Lernrate (Argumente: `lr_backbone`, `lr_head`) und `weight_decay`.

Konfigurations-/Übergabebeispiele
---------------------------------
- Einfacher Agent mit Standard-SimpleQNetwork:

```python
from experiments.subset_training.DQNAgent import DQNAgent

agent = DQNAgent(action_space=['a','b'], state_shape=(3,64,64), lr=1e-3)
```

- Agent mit ResNet18-Backbone, Kopf trainieren (Backbone gefroren):

```python
from experiments.subset_training.DQNAgent import DQNAgent, ResNetFeatureQNetwork

network_kwargs = dict(backbone='resnet18', pretrained=False, freeze_backbone=True, use_imagenet_norm=False)
agent = DQNAgent(action_space, (3,64,64), lr=1e-3, network_constructor=ResNetFeatureQNetwork, network_kwargs=network_kwargs)
```

- Agent mit ResNet18, unterschiedliche Lernraten für Backbone und Kopf (Feintuning-Szenario):

```python
network_kwargs = dict(backbone='resnet18', pretrained=True, freeze_backbone=False, use_imagenet_norm=True)
agent = DQNAgent(action_space, (3,224,224), lr=1e-4, lr_backbone=1e-5, lr_head=1e-4,
                 network_constructor=ResNetFeatureQNetwork, network_kwargs=network_kwargs)
```

Hinweise zur Eingabe-Normalisierung
-----------------------------------
- Wenn `pretrained=True` oder `use_imagenet_norm=True` wird die ImageNet-Standard-Normalisierung
  (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) automatisch angewendet.
- Alternativ kannst du eigene `input_mean` und `input_std` (listen) via `network_kwargs` übergeben.
- `_states_to_tensor()` skaliert Werte von 0-255 zu 0-1, falls der maximale Wert im Tensor > 2.0 ist.

Optimierung & Fine-tuning
-------------------------
- Übliche Vorgehensweise:
  1. Backbone einfrieren (`freeze_backbone=True`) und nur Kopf trainieren (schnelles Training, vermeidet Überanpassung).
  2. Falls nötig, Backbone teilweise oder komplett freigeben und mit niedrigerer Lernrate feintunen.
- Per-Parameter-Gruppen: `lr_backbone` und `lr_head` steuern getrennte Lernraten für Backbone und Kopf.
- Weight decay wird an den Optimizer weitergereicht (Argument `weight_decay`).

Optimierung / Optimierungsinformationen
---------------------------------------

Dieser Abschnitt fasst praktische Empfehlungen und häufig verwendete Hyperparameter für das Training eines DQN-Agenten mit ResNet-Vision-Backbone zusammen. Die Empfehlungen sind bewusst pragmatisch und sollen als Ausgangspunkt dienen; Grid-/Bayes-Search auf wichtigen Parametern (LR, weight_decay, batch_size) ist empfohlen.

1) Optimizer
   - Empfohlen: AdamW (oder Adam) für den Kopf; für Backbone ist AdamW oder SGD möglich, je nach Feintuning-Szenario.
   - Verwende per-Parameter-Gruppen:
     - Kopf (`head`): höhere Lernrate (z. B. 1e-4 .. 1e-3)
     - Backbone (`backbone`): niedrigere Lernrate (z. B. 1e-6 .. 1e-4) beim Feintuning
   - weight_decay: typischer Startwert 1e-4 bis 1e-2 (bei AdamW), 0.0 für Adam wenn nötig.

2) Lernraten und Schedules
   - Warmup: kurzes lineares Warmup (z. B. 500–2000 Steps) kann sehr hilfreich sein.
   - Scheduler: CosineAnnealingLR, StepLR oder ReduceLROnPlateau sind passende Optionen.
   - Empfehlung für Feintuning: starte mit lr_head z.B. 1e-4 und lr_backbone 1e-5, dann ggf. lr_backbone erhöhen.

3) Batch-Größe und BatchNorm
   - ResNet-Modelle enthalten BatchNorm — für stabile BN-Statistiken sind größere Batches besser.
   - Falls GPU-Memory limitiert: Gradient Accumulation (mehrere Micro-Batches) verwenden, oder Backbone einfrieren.
   - Alternativen: SyncBatchNorm (bei Multi-GPU) oder GroupNorm/LayerNorm als Architekturänderung.

4) Mixed Precision (AMP)
   - Empfohlen, um Trainingsgeschwindigkeit und GPU-Memory-Auslastung zu verbessern.
   - Verwende PyTorch native AMP (autocast + GradScaler).

5) Gradient-Clipping und Regularisierung
   - Gradient clipping (z. B. Norm clipping mit max_norm=1.0..5.0) kann Instabilitäten reduzieren.
   - Dropout im Kopf kann helfen, falls Overfitting auftritt.

6) Fine-tuning-Strategie / Zeitplan
   - Phase A: Backbone einfrieren, nur Kopf trainieren (schnell, stabil).
   - Phase B: Backbone teilweise/komplett freigeben und mit kleinerer Lernrate feintunen.
   - Überwache Validierungs-/Evaluations-Metriken und wähle das beste Checkpoint nach Evaluation.

7) Checkpointing, Logging und Evaluation
   - Häufige Checkpoints (z. B. alle N Trainings-Episoden oder Steps) und separates Speichern des besten Modells.
   - Logge mindestens: training-loss, eval-reward, lr, gradient norms, memory usage.
   - Nutze MLflow/TensorBoard/Weights&Biases für verlässliches Experiment-Tracking.

8) Weitere Praxis-Tipps
   - Pretrained-Backbone: Wenn `pretrained=True`, achte auf ImageNet-Normalisierung (mean/std). Oft besser bei kleinen Datensätzen.
   - Data augmentation: random crop/resize, color jitter, random horizontal flip (nur wenn sinnvoll für die Aufgabe).
   - Reproduzierbarkeit: Setze Seeds (PyTorch/NumPy/Python), deaktiviere ggf. non-deterministische CUDA-Algorithmen, aber beachte Performance-Tradeoffs.

9) Beispiel: Optimizer- und Scheduler-Setup (PyTorch)

```python
# Beispiel: Per-Parameter Optimizer mit AdamW und CosineLR + AMP
backbone_lr = 1e-5
head_lr = 1e-4
weight_decay = 1e-4

param_groups = [
    {"params": backbone_params, "lr": backbone_lr},
    {"params": head_params, "lr": head_lr},
]
optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

# optional: scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

# AMP setup
scaler = torch.cuda.amp.GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        loss = model(batch)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
```

10) Empfohlene Startwerte (Hinweise)
    - lr_head: 1e-4
    - lr_backbone: 1e-5 (beim Feintuning)
    - weight_decay: 1e-4
    - batch_size: so groß wie möglich (z. B. 16/32/64) abhängig von GPU
    - gradient_clip_norm: 1.0..5.0
    - warmup_steps: 500..2000

11) Monitoring spezifischer Probleme
    - Starke Schwankungen des Q-Loss: zu große LR, instability — verkleinere LR, aktiviere grad clipping.
    - Sehr langsame Lernkurven: evtl. Backbone zu stark gefroren oder zu kleine LR für Kopf.
    - Überanpassung: mehr Regularisierung, Data-Augmentation, oder Backbone einfrieren.

12) Tests und Validierung
    - Teste optimizer-Parametergruppen in Unit-Tests (z. B. prüfe, dass `optimizer.param_groups` erwartete LRs enthält).
    - Smoke-Tests (vor jedem Experiment) mit kleinen zufälligen Batches, um CUDA-/Shape-Probleme früh zu entdecken.


Weitere Schritte
---------------
- Wenn du möchtest, ergänze ich gerne einen Unit-Test, der explizit prüft, ob `DQNAgent` die erwarteten Optimizer-Parameterguppen
  (z. B. korrekte LRs für Backbone vs Head) erstellt. Das ist nützlich, um Regressionsfehler beim Param-Gruppen-Aufbau zu erkennen.

Tests / Developer-Hinweise
--------------------------

Kurze Anleitung, wie du die Unit-Tests für diesen Teil des Projekts lokal ausführst und wie du die Optimizer-Parameterguppen (Backbone vs Head) schnell inspizierst.

1) Tests ausführen
   - Einzelne Testdatei (z.B. ResNet-bezogene Tests):

```bash
pytest -q tests/test_dqn_agent_resnet.py
```

   - Optimizer-Param-Group Tests:

```bash
pytest -q tests/test_dqn_agent_optimizer_groups.py
```

   - Beide Tests zusammen (kompakt):

```bash
pytest -q tests/test_dqn_agent_resnet.py tests/test_dqn_agent_optimizer_groups.py
```

2) Inspektion der Optimizer-Param-Gruppen (Runtime)
   - Falls du während eines Debug-Laufs prüfen möchtest, welche Parameterguppen und Lernraten aktuell verwendet werden, kannst du folgendes Snippet nutzen:

```python
# agent ist eine Instanz von DQNAgent
for i, g in enumerate(agent.optimizer.param_groups):
    lr = g.get('lr', agent.optimizer.defaults.get('lr'))
    wd = g.get('weight_decay', agent.optimizer.defaults.get('weight_decay'))
    n_params = sum(p.numel() for p in g['params'])
    print(f"group {i}: lr={lr}, weight_decay={wd}, params={n_params}")
```

   - Das hilft zu verifizieren, dass `lr_backbone` / `lr_head` korrekt auf die jeweiligen Gruppen angewendet wurden.

3) CI / Automation
   - Fügen Sie diese Testbefehle (oder ein pytest-call) in Ihren CI-Workflow ein (z.B. GitHub Actions) — die Tests sind schnell Smoke-Tests und sollten als Gate fungieren.

4) Debugging-Tipps
   - Wenn ein Test fehlschlägt, führe ihn einzeln mit `-q -k <testname>` aus, um die Ausgabe zu fokussieren.
   - Nutze `print()` oder `pdb` in Tests kurzfristig, um Inhalte (Shapes, LRs) zu inspizieren.
