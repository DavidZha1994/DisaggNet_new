import torch
from omegaconf import OmegaConf

from src.losses.losses import create_loss_function


def test_regression_seq_loss_basic():
    cfg = OmegaConf.create({
        "regression_weight": 1.0,
        "classification_weight": 0.0,
        "normalize_per_device": True,
        "off_penalty_weight": 0.2,
        "active_threshold_rel": 0.02,
    })
    loss_fn = create_loss_function(cfg)
    B, L, K = 2, 16, 3
    target = torch.zeros(B, L, K)
    status = torch.zeros(B, L, K)
    target[:, 4:8, 0] = 100.0
    status[:, 4:8, 0] = 1.0
    pred = target.clone() * 0.8
    vm = torch.ones(B, L, K, dtype=torch.bool)
    scale = torch.tensor([100.0, 100.0, 100.0])
    v = loss_fn.regression_seq_loss(pred, target, status, vm, scale)
    assert torch.isfinite(v)
    assert v.item() > 0.0


def test_off_penalty_punishes_negative():
    cfg = OmegaConf.create({
        "normalize_per_device": True,
        "off_penalty_weight": 0.5,
        "active_threshold_rel": 0.02,
    })
    loss_fn = create_loss_function(cfg)
    B, L, K = 1, 8, 2
    target = torch.zeros(B, L, K)
    status = torch.zeros(B, L, K)
    pred_pos = torch.zeros(B, L, K)
    pred_neg = torch.zeros(B, L, K)
    pred_pos[:, :, 0] = 10.0
    pred_neg[:, :, 0] = -10.0
    vm = torch.ones(B, L, K, dtype=torch.bool)
    scale = torch.tensor([100.0, 100.0])
    v_pos = loss_fn.regression_seq_loss(pred_pos, target, status, vm, scale)
    v_neg = loss_fn.regression_seq_loss(pred_neg, target, status, vm, scale)
    assert v_pos.item() > 0.0
    assert v_neg.item() > 0.0
    assert v_neg.item() >= v_pos.item()