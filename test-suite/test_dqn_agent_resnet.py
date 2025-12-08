import importlib.util
from pathlib import Path
import numpy as np
import torch
import math

# load DQNAgent module from path
p = Path(r"/experiments/subset_training/DQNAgent.py")
spec = importlib.util.spec_from_file_location("dqn", str(p))
dqn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dqn)


def test_resnet_feature_network_forward_shape():
    """ResNetFeatureQNetwork returns correct shape for a batch."""
    model = dqn.ResNetFeatureQNetwork((3, 64, 64), n_actions=5, backbone='resnet18', pretrained=False, freeze_backbone=True)
    model.eval()
    x = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.float32)
    out = model(x)
    assert out.shape == (2, 5)


def test_states_to_tensor_and_select_action():
    """DQNAgent._states_to_tensor converts numpy input and select_action returns a valid index."""
    action_space = ['a', 'b', 'c']
    agent = dqn.DQNAgent(action_space, (3, 64, 64), lr=1e-3,
                         network_constructor=dqn.ResNetFeatureQNetwork,
                         network_kwargs={'backbone': 'resnet18', 'pretrained': False, 'freeze_backbone': True, 'use_imagenet_norm': False})

    s = np.random.randint(0, 256, size=(3, 64, 64), dtype=np.uint8)
    t = agent._states_to_tensor(s, add_batch_dim=True)
    assert isinstance(t, torch.Tensor)
    assert t.shape == (1, 3, 64, 64)

    a = agent.select_action(s)
    assert isinstance(a, int)
    assert 0 <= a < len(action_space)


def test_optimize_step_smoke():
    """Run a single optimize_step with a tiny random batch to ensure no runtime errors and finite loss."""
    action_space = ['a', 'b', 'c']
    agent = dqn.DQNAgent(action_space, (3, 64, 64), lr=1e-3,
                         network_constructor=dqn.ResNetFeatureQNetwork,
                         network_kwargs={'backbone': 'resnet18', 'pretrained': False, 'freeze_backbone': True, 'use_imagenet_norm': False})

    N = 2
    states = np.random.randint(0, 256, size=(N, 3, 64, 64), dtype=np.uint8)
    next_states = np.random.randint(0, 256, size=(N, 3, 64, 64), dtype=np.uint8)
    actions = np.random.randint(0, 3, size=(N,))
    rewards = np.random.randn(N).astype(np.float32)
    dones = np.zeros(N, dtype=np.float32)

    loss = agent.optimize_step((states, actions, rewards, next_states, dones), target_update=True)
    assert isinstance(loss, float)
    assert not math.isnan(loss) and math.isfinite(loss)

