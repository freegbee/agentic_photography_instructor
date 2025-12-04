# Reinforcement Learning Training Guide

This guide explains how to train an RL agent to optimize image aesthetics using transformations.

## Overview

The RL training system uses **Deep Q-Networks (DQN)** to learn which sequences of transformations improve image aesthetic scores. The agent:
- **Observes**: Degraded images as states
- **Acts**: Applies transformations from the action space
- **Rewards**: Score improvements (with shaped rewards)
- **Goal**: Learn to maximize aesthetic score through multi-step episodes

## Architecture

### Components

1. **RLDataset** (`src/experiments/rl_training/RLDataset.py`)
   - Loads degraded images with original scores from COCO annotations
   - Returns tuples: (degraded_image_data, original_score, transformation)

2. **RLTrainingExperiment** (`src/experiments/rl_training/RLTrainingExperiment.py`)
   - Main training loop with multi-step episodes
   - Handles experience replay, target network updates, validation
   - Logs metrics to MLflow

3. **DQNAgent** (`src/experiments/subset_training/DQNAgent.py`)
   - Simple CNN-based Q-network
   - Epsilon-greedy exploration
   - Model persistence (save/load checkpoints)

4. **TransformationActor** (`src/experiments/subset_training/TransformationActor.py`)
   - Applies transformations and scores results via Juror service

## Workflow

### Step 1: Generate Training Dataset

First, generate degraded images with train/val/test splits:

```bash
# Start experiment service
cd docker && docker-compose up -d experiment-service

# Attach to container
docker exec -it experiment-service bash

# Run image degradation experiment with splits
cd /app
python -m experiments.image_degrade.entrypoint
```

When prompted:
- **Experiment Name**: "RL Dataset Generation"
- **Target directory**: `rl_training_data`
- **Target dataset ID**: `rl_flickr8k`
- **Transformer name**: `RANDOM` (uses all reversible transformers)
- **Source dataset**: `flickr8k`
- **Batch size**: `16`

To enable splits, you need to modify the entrypoint or create a new script. Here's a quick example:

```python
# In Python shell or script
from experiments.image_degrade.ImageDegradationExperiment import ImageDegradationExperiment

experiment = ImageDegradationExperiment(
    experiment_name="RL Dataset Generation",
    target_directory_root="rl_training_data",
    target_dataset_id="rl_flickr8k",
    transformer_name="RANDOM",
    source_dataset_id="flickr8k",
    batch_size=16,
    split_ratios=(0.7, 0.15, 0.15),  # train, val, test
    seed=42
)

experiment.run()
```

This will create:
```
volumes/resources/rl_training_data/
├── train/
│   ├── images/
│   └── annotations.json
├── val/
│   ├── images/
│   └── annotations.json
└── test/
    ├── images/
    └── annotations.json
```

### Step 2: Start Training Container

```bash
cd docker
docker-compose up -d train

# Attach to container
docker exec -it train bash
```

### Step 3: Run Training

Inside the train container:

```bash
# Interactive training
python -m experiments.rl_training.entrypoint
```

Follow the prompts or accept defaults:
- **Experiment Name**: `RL Image Optimization`
- **Dataset root**: `rl_training_data`
- **Max steps per episode**: `5`
- **Number of epochs**: `10`
- **Batch size**: `32`
- **Learning rate**: `0.001`
- **Add STOP action**: `y` or `n`

The training will:
1. Load train/val datasets
2. Train for specified epochs
3. Validate every N epochs
4. Save checkpoints to `/tmp/rl_checkpoints/`
5. Log all metrics to MLflow

### Step 4: Monitor Training

Access MLflow UI at `http://localhost:5000` to monitor:
- Training metrics: `train_avg_reward`, `train_success_rate`, `train_avg_loss`
- Validation metrics: `val_avg_improvement`, `val_success_rate`
- Hyperparameters
- Checkpoints (artifacts)

You can also view metrics in Grafana at `http://localhost:5030`.

### Step 5: Evaluate Trained Agent

After training completes, evaluate on test set:

```bash
# Inside train container
python -m experiments.rl_training.evaluate \
    --checkpoint /mlruns/<experiment_id>/<run_id>/artifacts/checkpoint_epoch_9.pth \
    --dataset rl_training_data \
    --max-steps 5 \
    --output-dir /app/volumes/resources/rl_evaluation
```

This will:
- Run greedy evaluation (epsilon=0)
- Save before/after images for first 20 samples
- Generate `results.csv` with detailed metrics
- Log summary statistics

## Configuration

Configuration is in `configs/default.yaml` under `rl_training`:

```yaml
rl_training:
  dataset_generation:
    source_dataset: "flickr8k"
    output_dir: "rl_training_data"
    split_ratios: [0.7, 0.15, 0.15]
    use_random_transformer: true
    seed: 42

  training:
    max_steps_per_episode: 5
    batch_size: 32
    replay_capacity: 10000
    learning_rate: 0.001
    gamma: 0.99
    epsilon_start: 1.0
    epsilon_end: 0.05
    epsilon_decay: 0.995
    target_update_frequency: 100
    num_epochs: 10
    dataloader_batch_size: 8
    validation_frequency: 1

  reward:
    step_penalty: 0.01
    terminal_bonus: 0.1

  action_space:
    add_stop_action: true

  state:
    shape: [3, 64, 64]
```

## Hyperparameter Tuning

### Key Hyperparameters

1. **max_steps_per_episode** (5)
   - Number of transformations the agent can apply
   - Higher = more complex sequences but longer episodes
   - Recommended: 3-7

2. **learning_rate** (0.001)
   - DQN optimizer learning rate
   - Lower = more stable but slower learning
   - Recommended: 0.0001 - 0.001

3. **gamma** (0.99)
   - Discount factor for future rewards
   - Higher = values long-term improvements more
   - Recommended: 0.95 - 0.99

4. **epsilon_decay** (0.995)
   - Rate of exploration decay
   - Lower = explores longer
   - Recommended: 0.99 - 0.999

5. **step_penalty** (0.01)
   - Penalty per transformation step
   - Encourages efficiency
   - Recommended: 0.01 - 0.05

6. **terminal_bonus** (0.1)
   - Bonus for reaching/exceeding target score
   - Encourages goal achievement
   - Recommended: 0.05 - 0.2

### Reward Shaping

The reward function is:
```python
reward = (new_score - prev_score) - step_penalty

# At terminal step, if new_score >= target_score:
reward += terminal_bonus
```

Adjust `step_penalty` and `terminal_bonus` to shape behavior:
- Higher step_penalty → fewer steps, more conservative
- Higher terminal_bonus → more aggressive pursuit of target

## Action Space

The action space consists of:
- **REVERSIBLE_TRANSFORMERS** (14 transformers):
  - Color channel swaps (GB, RB, RG)
  - Color channel inversions (B, G, R)
  - Color plane shifts (Left/Down for B, G, R, All)
- **STOP action** (optional):
  - Allows early termination
  - Agent learns when to stop transforming

## State Representation

Images are preprocessed to state shape `(3, 64, 64)`:
1. Convert BGR → RGB
2. Resize to 64×64
3. Transpose to (C, H, W)
4. Normalize to [0, 255] uint8

Larger state sizes (e.g., 128×128) capture more detail but:
- Require more memory
- Slower training
- May need deeper networks

## Tips for Better Training

### 1. Dataset Quality
- Use high-quality source images (Places365, DIV2K)
- Ensure good diversity in transformations
- Balance easy/hard samples

### 2. Training Stability
- Start with small datasets (1000-5000 images)
- Monitor validation metrics for overfitting
- Use target network updates (every 100 steps)

### 3. Exploration
- Start with epsilon=1.0 for full exploration
- Decay slowly (0.995-0.999)
- Consider epsilon-greedy with minimum 0.05

### 4. Episode Length
- Start with max_steps=3-5
- Increase gradually as agent learns
- Too long → sparse rewards, slow learning

### 5. Debugging
- Check replay buffer size (should fill up quickly)
- Monitor Q-values (shouldn't explode/vanish)
- Validate action distribution (not all same action)

## Common Issues

### Issue: Agent always selects same action
**Solution**:
- Increase epsilon or decay rate
- Check action space diversity
- Verify reward signal is meaningful

### Issue: Training loss not decreasing
**Solution**:
- Reduce learning rate
- Increase batch size
- Check for gradient clipping issues

### Issue: Validation worse than training
**Solution**:
- Add dropout/regularization
- Increase dataset size
- Reduce model capacity

### Issue: Agent stops immediately (STOP action)
**Solution**:
- Increase step_penalty (make stopping costly)
- Remove/reduce terminal_bonus
- Check if degraded images already have high scores

## Advanced Usage

### Custom Action Space

```python
from experiments.rl_training.RLTrainingExperiment import RLTrainingExperiment

# Use only specific transformers
custom_actions = [
    "SwapColorChannelTransformerGB",
    "InvertColorChannelTransformerR",
    "GaussianBlur3",
]

experiment = RLTrainingExperiment(
    experiment_name="Custom Actions RL",
    run_name="gaussian_swap_invert",
    dataset_root=dataset_root,
    action_space=custom_actions,
    add_stop_action=False,
    # ... other params
)
```

### Resume Training

```python
from experiments.subset_training.DQNAgent import DQNAgent

agent = DQNAgent(action_space, state_shape)
agent.load_checkpoint("/path/to/checkpoint.pth")

# Continue training with this agent
experiment.agent = agent
experiment.run()
```

### Multi-Step Transformations

To train on multi-step degraded images (not just single transformations):

1. Generate dataset with double transformations:
   ```python
   # Use double_transformation experiment
   from experiments.double_transformation import ...
   ```

2. Adjust max_steps_per_episode accordingly

## Performance Expectations

### Training Time
- **Small dataset** (1k images): ~1-2 hours on CPU, ~15-30 min on GPU
- **Medium dataset** (5k images): ~4-6 hours on CPU, ~1-2 hours on GPU
- **Large dataset** (10k+ images): ~8+ hours on CPU, ~3+ hours on GPU

### Success Rates
- **Baseline** (random): ~14% (1/14 actions improves)
- **Early training** (epoch 1-3): 20-30%
- **Mid training** (epoch 4-7): 40-60%
- **Late training** (epoch 8+): 60-80%

Target success rate depends on:
- Dataset difficulty
- Action space size
- Max steps allowed

## Next Steps

1. **Experiment with different architectures**: Replace SimpleQNetwork with ResNet/EfficientNet
2. **Try different RL algorithms**: DQN → Double DQN, Dueling DQN, Rainbow
3. **Add more transformations**: Include cropping, brightness, contrast adjustments
4. **Multi-objective rewards**: Balance score improvement with diversity
5. **Curriculum learning**: Start with easy samples, increase difficulty

## References

- DQN Paper: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- Aesthetic Predictor: [aesthetic-predictor-v2-5](https://github.com/discus0434/aesthetic-predictor-v2-5)
- Project Documentation: `CLAUDE.md`
