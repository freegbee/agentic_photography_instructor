import importlib.util
from pathlib import Path
import numpy as np
import torch

# load DQNAgent module from path
p = Path(r"/experiments/subset_training/DQNAgent.py")
spec = importlib.util.spec_from_file_location("dqn", str(p))
dqn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dqn)


def _get_group_lrs(optimizer):
    return sorted([float(g.get('lr', optimizer.defaults.get('lr', 0.0))) for g in optimizer.param_groups])


def test_optimizer_param_groups_resnet():
    """Check that optimizer has separate groups for backbone and head with requested LRs and weight_decay."""
    action_space = ['a', 'b', 'c']
    backbone_lr = 1e-5
    head_lr = 1e-4
    weight_decay = 1e-3

    agent = dqn.DQNAgent(action_space, (3, 64, 64), lr=1e-4, lr_backbone=backbone_lr, lr_head=head_lr, weight_decay=weight_decay,
                         network_constructor=dqn.ResNetFeatureQNetwork,
                         network_kwargs={'backbone': 'resnet18', 'pretrained': False, 'freeze_backbone': False, 'use_imagenet_norm': False})

    opt = agent.optimizer
    # ensure we have at least two groups (backbone + head)
    assert len(opt.param_groups) >= 2

    lrs = _get_group_lrs(opt)
    # check that both requested LRs are in groups
    assert any(abs(l - backbone_lr) < 1e-12 for l in lrs)
    assert any(abs(l - head_lr) < 1e-12 for l in lrs)

    # check weight_decay is set on optimizer defaults (applies across groups)
    assert abs(opt.defaults.get('weight_decay', 0.0) - weight_decay) < 1e-12


def test_optimizer_no_backbone_when_frozen():
    """If backbone is frozen, backbone param group should not be created."""
    action_space = ['a', 'b', 'c']
    backbone_lr = 5e-6
    head_lr = 5e-5

    agent = dqn.DQNAgent(action_space, (3, 64, 64), lr=1e-4, lr_backbone=backbone_lr, lr_head=head_lr,
                         network_constructor=dqn.ResNetFeatureQNetwork,
                         network_kwargs={'backbone': 'resnet18', 'pretrained': False, 'freeze_backbone': True, 'use_imagenet_norm': False})

    opt = agent.optimizer
    lrs = _get_group_lrs(opt)
    # backbone_lr should NOT be present because backbone was frozen
    assert not any(abs(l - backbone_lr) < 1e-12 for l in lrs)
    # head_lr should be present
    assert any(abs(l - head_lr) < 1e-12 for l in lrs)


def test_optimizer_fallback_single_group_when_no_special_modules():
    """If network doesn't expose head/backbone, optimizer should still be created (single group)."""
    action_space = ['a', 'b']
    # Use SimpleQNetwork (default) which doesn't expose `head`/`backbone`
    agent = dqn.DQNAgent(action_space, (3, 64, 64), lr=2e-4, network_constructor=dqn.SimpleQNetwork)
    opt = agent.optimizer
    # should have at least one param group; since SimpleQNetwork has trainable params, single/few groups ok
    assert len(opt.param_groups) >= 1
    # ensure lr for at least one group equals the base lr
    lrs = _get_group_lrs(opt)
    assert any(abs(l - 2e-4) < 1e-12 for l in lrs)

