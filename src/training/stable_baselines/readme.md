# Stable Baselines und gymnasium

## Allgemeines
- Stable Baselines ist eine Sammlung von bewährten Implementierungen von Reinforcement-Learning-Algorithmen in Python. Es baut auf der OpenAI Gym-Bibliothek auf, die eine Vielzahl von Umgebungen für das Training und die Evaluierung von RL-Agenten bereitstellt.
- Gymnasium (früher bekannt als OpenAI Gym) ist eine Bibliothek, die standardisierte Umgebungen für die Entwicklung und das Testen von Reinforcement-Learning-Algorithmen bereitstellt.

## Begrifflichkeiten
- **Umgebung (Environment)**: Die Umgebung ist die Welt, in der der RL-Agent agiert. Sie stellt Zustände bereit, empfängt Aktionen und gibt Belohnungen zurück.
- **Agent**: Der Agent ist das Modell, das lernt, wie es in der Umgebung agieren soll, um Belohnungen zu maximieren.
- **Episode**: Eine Episode ist eine Sequenz von Zuständen, Aktionen und Belohnungen, die mit einem Anfangszustand beginnt und mit einem Endzustand (`done` oder `truncated`) endet.
- **Step**: Ein Schritt ist eine einzelne Interaktion des Agenten mit der Umgebung, bei der der Agent eine Aktion auswählt, die Umgebung daraufhin den nächsten Zustand und die Belohnung zurückgibt.
  - In Gymnasium wird ein Schritt typischerweise durch den Aufruf der Methode `env.step(action)` durchgeführt, die ein Tupel `(next_state, reward, done, truncated, info)` zurückgibt.
  - `next_state`: Der Zustand der Umgebung nach der Aktion, in unserem Falle das angepasste Bild
  - `reward`: Die Belohnung, die der Agent für die Aktion erhält, also die verbesserung oder verschlechterung des Scores
  - `done`: Ein boolescher Wert, der angibt, ob die Episode beendet ist (z.B. wenn der Score hoch genug ist.
  - `truncated`: Ein boolescher Wert, der angibt, ob die Episode vorzeitig abgebrochen wurde (z.B. durch eine Zeitbegrenzung, oder die Anzahl der verfügbaren Aktionen abgelaufen ist).
  - `info`: Ein optionales Dictionary, das zusätzliche Informationen über den Schritt enthalten kann.
- **`reset()`**: Die `reset`-Methode wird verwendet, um die Umgebung auf ihren Anfangszustand zurückzusetzen und eine neue Episode zu starten. Sie gibt den initialen Zustand der Umgebung zurück. Z.b. wird ein zufälliges Bild geladen
- **Rollout Batch**: Ein Rollout-Batch ist eine Sammlung von Übergängen (Zustand → Aktion → Belohnung → nächster Zustand), die vor einem einzigen Policy-Update gesammelt werden.
- **Epoche**: Eine Epoche ist ein vollständiger Durchlauf durch den gesamten Trainingsdatensatz (also alle Bilder). In RL kann dies bedeuten, dass der Agent eine bestimmte Anzahl von Episoden durchläuft.
- **Mini-Batch**: Ein Mini-Batch (`minibatch_size`; Anzahl Mini-Batches = `rollout_batch_size / minibatch_size`) ist eine kleinere Untermenge des Trainingsdatensatzes, die verwendet wird, um die Modellparameter während des Trainings zu aktualisieren.
- **Policy**: Die Policy ist die Strategie, die der Agent verwendet, um Aktionen basierend auf dem aktuellen Zustand auszuwählen.
- **Vectorized Environments** ermöglichen paralleles Training über mehrere Instanzen der Umgebung hinweg, was die Effizienz des Trainingsprozesses erheblich steigert. Also ähnlich einer Batch-Verarbeitung.
- `n_steps`: Je nach Algorithmus Anzahl der steps, die durchgeführt werden (pro Umgebung), bis die Policy updated wird. Kommt z.B. bei PPO vor. 
  - Bei z.B. `n_steps=128` und `n_envs=8` werden insgesamt 1024 Schritte (128 Schritte pro Umgebung x 8 Umgebungen) durchgeführt, bevor die Policy aktualisiert wird.
- `time_steps`: Gesamtanzahl der Schritte, die im Training durchgeführt werden sollen. Dies bestimmt die Gesamtdauer des Trainingsprozesses.
  - D.h. bei `time_steps=1_000_000` wird das Training fortgesetzt, bis insgesamt 1 Million Schritte in allen Umgebungen durchgeführt wurden.
  - Beispiel: Bei `n_envs=8` und `n_steps=128` werden pro Update 1024 Schritte durchgeführt. Um 1 Million Schritte zu erreichen, sind etwa 977 Updates erforderlich (1_000_000 / 1024 ≈ 977).

## Was wie oft?
- Rollout-Batch Grösse pro Policy Update `= n_steps * num_envs`
- Anzahl Updates ≈ `total_timesteps / (n_steps * num_envs)`
- Pro Update werden die gesammelten `n_steps * num_envs` Übergänge für n_epochs Optimierungsdurchläufe verwendet.
- Episodes können innerhalb eines Rollouts enden; wenn eine einzelne Environment-Episode fertig ist, wird diese Environment sofort neu gestartet und zählt weiter zu den n_steps.
- Bei `num_envs=8, n_steps=128 → rollout_batch_size=1024. Mit n_epochs=4 und minibatch_size=256 → 4 Updates pro Rollout-Batch`

## Training-Loop
- Step ist eine einzelne Transformation, also `action -> next_state -> reward`
- Episode ist 1...n Steps (vom initialen `reset` bis `done` oder `truncated`)
- 