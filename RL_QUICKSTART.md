# RL Training Quickstart

Fast track guide to get started with RL training in 5 minutes.

## Prerequisites

1. Docker and Docker Compose installed
2. Services running: `cd docker && docker-compose up -d`
3. Source dataset available (e.g., flickr8k)

## Quick Start (3 Steps)

### Step 1: Generate Training Dataset (~10-30 min)

```bash
# Start experiment service
cd docker && docker-compose up -d experiment-service

# Attach and generate dataset
docker exec -it experiment-service bash
cd /app
python -m experiments.rl_training.generate_dataset
```

**Accept defaults or customize:**
- Source dataset: `flickr8k` (8k images)
- Output directory: `rl_training_data`
- Split ratios: `0.7/0.15/0.15` (train/val/test)
- Use random transformers: `Y`

This creates `rl_training_data/` with train/val/test splits.

### Step 2: Train RL Agent (~1-6 hours depending on dataset size)

```bash
# Start train container
cd docker && docker-compose up -d train

# Attach and train
docker exec -it train bash
cd /app/src
python -m experiments.rl_training.entrypoint
```

**Accept defaults for first run:**
- Dataset root: `rl_training_data`
- Max steps per episode: `5`
- Number of epochs: `10`
- Batch size: `32`

Monitor training at: `http://localhost:5000` (MLflow)

### Step 3: Evaluate Results

```bash
# Inside train container
python -m experiments.rl_training.evaluate \
    --checkpoint /mlruns/<exp_id>/<run_id>/artifacts/checkpoint_epoch_9.pth \
    --dataset rl_training_data \
    --output-dir /app/volumes/resources/rl_evaluation
```

Find checkpoint path in MLflow UI under Artifacts.

Results saved to `/app/volumes/resources/rl_evaluation/` including:
- `results.csv` - detailed metrics
- `example_*_initial.jpg` - before images
- `example_*_final.jpg` - after images
- `example_*_metadata.txt` - action sequences

## Expected Results

After 10 epochs with default settings:

- **Success Rate**: 60-80% (images improved)
- **Avg Improvement**: +0.05 to +0.15 score
- **Avg Steps**: 2-4 transformations per image

## Quick Tips

### Faster Training (Small Dataset)
Use fewer images for testing:
- Source: `twenty_images` (20 images)
- Epochs: `3`
- Training time: ~5-10 minutes

### Better Results (More Training)
- Increase epochs: `20` or `50`
- Larger dataset: `flickr8k` (8000 images)
- More steps: `max_steps=7`

### Debug Issues
Check logs:
```bash
docker logs train
docker logs juror-service
```

Check MLflow: `http://localhost:5000`

## Common Commands

**List available datasets:**
```bash
ls /app/volumes/resources/
```

**Check dataset structure:**
```bash
ls -R /app/volumes/resources/rl_training_data/
```

**View training progress:**
```bash
# MLflow UI
open http://localhost:5000

# Grafana
open http://localhost:5030
```

**Copy results to host:**
```bash
# From host machine
docker cp train:/app/volumes/resources/rl_evaluation ./rl_evaluation_results
```

## Next Steps

See `RL_TRAINING_GUIDE.md` for:
- Hyperparameter tuning
- Custom action spaces
- Advanced training techniques
- Troubleshooting

## Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│                 RL Training Pipeline                 │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  Step 1: Generate Dataset (experiment-service)      │
│  ─────────────────────────────────────────────────  │
│  ImageDegradationExperiment                         │
│    ├─ Source: flickr8k (original images)            │
│    ├─ Apply: Random transformers (degrades images)  │
│    ├─ Score: Juror scores both versions             │
│    └─ Split: train (70%) / val (15%) / test (15%)   │
│                                                      │
│  Output: rl_training_data/{train,val,test}/         │
│    ├─ images/ (degraded images)                     │
│    └─ annotations.json (scores + metadata)          │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  Step 2: Train Agent (train container)              │
│  ─────────────────────────────────────────────────  │
│  RLTrainingExperiment                               │
│    ├─ Agent: DQN (policy + target networks)         │
│    ├─ Environment: Multi-step episodes              │
│    │   • State: Preprocessed image (3,64,64)        │
│    │   • Action: Transform (14 options + STOP)      │
│    │   • Reward: score_delta - step_penalty         │
│    ├─ Training: Experience replay + ε-greedy        │
│    └─ Validation: Greedy evaluation every epoch     │
│                                                      │
│  Outputs: Checkpoints + Metrics (MLflow)            │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  Step 3: Evaluate (train container)                 │
│  ─────────────────────────────────────────────────  │
│  RLEvaluator                                        │
│    ├─ Load: Best checkpoint from training           │
│    ├─ Test: Run greedy episodes on test set         │
│    └─ Save: Results CSV + example images            │
│                                                      │
│  Outputs: Evaluation metrics + visualizations       │
└─────────────────────────────────────────────────────┘
```

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/experiments/rl_training/generate_dataset.py` | Generate train/val/test splits |
| `src/experiments/rl_training/entrypoint.py` | Train RL agent |
| `src/experiments/rl_training/evaluate.py` | Evaluate trained agent |
| `src/experiments/rl_training/RLTrainingExperiment.py` | Main training logic |
| `src/experiments/rl_training/RLDataset.py` | Dataset loader |
| `src/experiments/subset_training/DQNAgent.py` | DQN implementation |
| `configs/default.yaml` | Configuration (under `rl_training`) |
| `RL_TRAINING_GUIDE.md` | Full documentation |

## Support

For issues or questions:
1. Check `RL_TRAINING_GUIDE.md` for detailed docs
2. Review `CLAUDE.md` for project architecture
3. Check MLflow logs at `http://localhost:5000`
4. Open an issue on GitHub
