# Multi-Step Transformation Training

## Overview

The `MultiStepTransformWrapper` allows training RL agents to apply **multiple transformations** before receiving a reward. This is useful when you want the agent to learn complex multi-step transformation sequences rather than single transformations.

## Problem Statement

In single-step training:
- Agent applies 1 transformation → receives immediate reward
- Agent learns individual transformation effects
- No planning of multi-step sequences

In multi-step training:
- Agent applies N transformations → receives reward only after all steps
- Agent must learn to plan sequences
- Reward based on cumulative effect of all transformations

## Architecture

### Wrapper Structure

```
SuccessCountingWrapper
  └── ImageRenderWrapper
        └── ImageObservationWrapper
              └── MultiStepTransformWrapper  ← NEW!
                    └── ImageTransformEnv
```

### How It Works

**Without wrapper** (single-step):
```
reset() → obs₁
step(action₁) → obs₂, reward₁, done=True
reset() → obs₁
step(action₁) → obs₂, reward₁, done=True
...
```

**With wrapper** (multi-step, N=2):
```
reset() → obs₁
step(action₁) → obs₂, reward=0, done=False  ← No reward yet
step(action₂) → obs₃, reward=final, done=True  ← Reward after 2 steps
reset() → obs₁
...
```

## Configuration

### Basic Configuration

```python
training_params.set({
    # ... other params ...
    "use_multi_step_wrapper": True,  # Enable wrapper
    "steps_per_episode": 2,          # Number of transformations per episode
    "multi_step_intermediate_reward": False,  # No intermediate rewards
    "multi_step_reward_shaping": False,       # No reward shaping
})
```

### Configuration Options

#### `use_multi_step_wrapper` (bool)
- **Default:** `False`
- **Description:** Whether to enable the multi-step wrapper
- **When to use:** Enable when training with datasets that have multiple degradations

#### `steps_per_episode` (int)
- **Default:** `2`
- **Description:** Number of transformations the agent must apply per episode
- **Important:** Should match the number of degradations in your dataset
- **Example:** If your dataset has 2 degradations per image, set this to 2

#### `multi_step_intermediate_reward` (bool)
- **Default:** `False`
- **Description:** Give small rewards (0.1x) at intermediate steps
- **When to use:** If you want some feedback during the episode
- **Trade-off:** May bias agent toward greedy short-term improvements

#### `multi_step_reward_shaping` (bool)
- **Default:** `False`
- **Description:** Provide shaped rewards based on progress toward the goal
- **When to use:** If agent struggles to learn with purely delayed rewards
- **Mechanism:** Small rewards (±0.05-0.1) based on score improvement

## Usage Examples

### Example 1: Pure Delayed Reward (Recommended)

```python
training_params.set({
    "use_multi_step_wrapper": True,
    "steps_per_episode": 2,
    "multi_step_intermediate_reward": False,  # Zero reward until end
    "multi_step_reward_shaping": False,        # No shaping
})
```

**Pros:**
- Agent learns true multi-step planning
- No bias toward intermediate goals
- Cleaner learning signal

**Cons:**
- May be harder to learn initially
- Requires more exploration

**Best for:** 2-3 step sequences, sufficient training time

### Example 2: With Reward Shaping

```python
training_params.set({
    "use_multi_step_wrapper": True,
    "steps_per_episode": 2,
    "multi_step_intermediate_reward": False,
    "multi_step_reward_shaping": True,  # Small shaped rewards
})
```

**Pros:**
- Easier to learn
- Faster initial progress
- Guides agent toward goal

**Cons:**
- May not learn optimal sequences
- Shaped reward might be misleading

**Best for:** >3 step sequences, limited training time

### Example 3: With Intermediate Rewards

```python
training_params.set({
    "use_multi_step_wrapper": True,
    "steps_per_episode": 2,
    "multi_step_intermediate_reward": True,  # 0.1x rewards at each step
    "multi_step_reward_shaping": False,
})
```

**Pros:**
- Some feedback at each step
- Less variance in learning

**Cons:**
- May bias toward greedy choices
- Less true multi-step planning

**Best for:** Debugging, initial experiments

## Reward Structure

### Pure Delayed Reward

```
Step 1: action₁ applied → reward = 0.0
Step 2: action₂ applied → reward = (final_score - initial_score) + bonus
```

### With Intermediate Rewards

```
Step 1: action₁ applied → reward = (score₁ - initial_score) * 0.1
Step 2: action₂ applied → reward = (final_score - initial_score) + bonus
```

### With Reward Shaping

```
Step 1: action₁ applied → reward = clip((score₁ - initial_score) * 0.05, -0.1, 0.1)
Step 2: action₂ applied → reward = (final_score - initial_score) + bonus
```

## Dataset Requirements

The multi-step wrapper works best with datasets where images have multiple degradations applied:

```python
# Create dataset with 2 degradations per image
transform_preprocessing_params.set({
    "num_transformations": 2,  # Match steps_per_episode
    "transformer_names": POC_MULTI_ONE_STEP_TRANSFORMERS,
    "use_random_transformer": True,
})
```

**Important:** `steps_per_episode` should typically match the number of degradations in your dataset.

## Training Considerations

### Episode Length

With N-step episodes:
- Each episode requires N actions
- Rollout buffer fills N times slower
- Adjust `n_steps` accordingly:

```python
# Single-step training
training_params.set({"n_steps": 200})  # 200 single-step episodes

# 2-step training
training_params.set({
    "n_steps": 200,  # Still 200, but fewer complete episodes
    "steps_per_episode": 2
})
```

### Learning Rate

Multi-step learning may benefit from adjusted learning rates:

```python
# Consider slightly lower learning rate for multi-step
general_params.set({"learning_rate": 1e-4})  # vs 3e-4 for single-step
```

### Exploration

Multi-step planning requires more exploration:

```python
# May need more training steps
training_params.set({"total_training_steps": 600_000})  # vs 400_000 for single-step
```

## Logging and Monitoring

The wrapper logs detailed information about each episode:

```python
logger.info(
    f"Multi-step: Episode complete after {steps} steps. "
    f"Initial: {initial_score:.4f}, Final: {final_score:.4f}, "
    f"Improvement: {improvement:.4f}, Success: {success}"
)
logger.info(f"Transformation sequence: {' -> '.join(transformation_sequence)}")
```

### MLflow Metrics

The wrapper provides additional info in the `step()` return:

```python
info["multi_step"] = {
    "steps_taken": 2,
    "transformation_sequence": ["CA_INV_B", "CA_SWAP_RG"],
    "intermediate_scores": [4.52, 5.31],
    "initial_score": 4.21,
    "final_score": 5.31,
    "total_improvement": 1.10,
    "success": True
}
```

## Running Multi-Step Training

### Quick Start

```bash
# Use the provided example entrypoint
python -m training.stable_baselines.training.entrypoint_multi_step
```

### Customization

1. **Copy the example entrypoint:**
   ```bash
   cp src/training/stable_baselines/training/entrypoint_multi_step.py \
      src/training/stable_baselines/training/my_multi_step_training.py
   ```

2. **Adjust parameters:**
   - Change `steps_per_episode` to match your dataset
   - Enable/disable reward shaping based on learning progress
   - Adjust learning rate and training steps

3. **Run:**
   ```bash
   python -m training.stable_baselines.training.my_multi_step_training
   ```

## Troubleshooting

### Agent Not Learning

**Symptoms:** Reward not improving, random behavior

**Solutions:**
1. Enable reward shaping: `multi_step_reward_shaping=True`
2. Start with fewer steps: `steps_per_episode=2` before increasing
3. Increase exploration: higher learning rate, more training steps
4. Check dataset: ensure degradations are reversible

### Agent Learning Greedy Solutions

**Symptoms:** Good first step, poor second step

**Solutions:**
1. Disable intermediate rewards: `multi_step_intermediate_reward=False`
2. Disable reward shaping: `multi_step_reward_shaping=False`
3. Increase planning horizon: higher `n_steps`

### Training Too Slow

**Symptoms:** Takes very long to train

**Solutions:**
1. Reduce `steps_per_episode` (start with 2)
2. Use more parallel environments: higher `num_vector_envs`
3. Optimize `n_steps` and `mini_batch_size`

## Implementation Details

### Files

- **Wrapper:** `src/training/stable_baselines/environment/multi_step_wrapper.py`
- **Integration:** `src/training/stable_baselines/training/trainer.py`
- **Parameters:** `src/training/stable_baselines/training/hyper_params.py`
- **Example:** `src/training/stable_baselines/training/entrypoint_multi_step.py`

### Key Methods

**`MultiStepTransformWrapper.reset()`**
- Resets tracking variables
- Calls base environment reset
- Returns initial observation

**`MultiStepTransformWrapper.step(action)`**
- Applies transformation
- Tracks intermediate scores
- Returns zero/shaped reward for steps 1 to N-1
- Returns final reward at step N

## Comparison: Single-Step vs Multi-Step

| Aspect | Single-Step | Multi-Step (N=2) |
|--------|-------------|------------------|
| Actions per episode | 1 | 2 |
| Reward frequency | Every step | After N steps |
| Learning difficulty | Easier | Harder |
| Planning horizon | None | N steps |
| Best for | Simple tasks | Complex sequences |
| Training time | Shorter | Longer |
| Quality of solutions | Local optima | Global optima |

## Future Extensions

### Variable-Length Episodes

Allow agent to decide when to stop (up to max_steps):

```python
# Not yet implemented
action = {"transformer_index": ..., "stop": False}
```

### Hierarchical Actions

Learn high-level policies that invoke low-level transformations:

```python
# Future work
high_level_action = "improve_brightness"  # → sequence of low-level transformations
```

### Multi-Image Episodes

Apply sequences to multiple images in one episode:

```python
# Future work
MultiImageMultiStepWrapper(env, images_per_episode=5, steps_per_image=2)
```
