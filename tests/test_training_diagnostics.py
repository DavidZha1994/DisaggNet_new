import os
import math
import torch
from omegaconf import OmegaConf

from src.train import NILMLightningModule


def make_config(reg_w: float = 1.0, cls_w: float = 0.0):
    cfg = OmegaConf.create({
        "data": {
            "device_names": [
                "washing_machine",
                "dishwasher",
                "kettle",
                "microwave",
                "fridge",
            ]
        },
        "model": {
            "d_model": 256,
            "n_heads": 8,
            "num_layers": 6,
            "dropout": 0.0,
            "time_encoder": {
                "d_model": 256,
                "n_heads": 8,
                "num_layers": 6,
                "dropout": 0.0,
                "input_conv_embed": True,
                "causal_mask": True,
            },
            "freq_encoder": {"enable": False, "proj_dim": 96, "dropout": 0.0},
            "fusion": {"type": "none", "gated": False},
            "aux_encoder": {"enable": False},
            "heads": {
                "regression": {"hidden": 128, "init_bias": -4.0, "seq_emb_scale": 0.05},
                "conditioning": {"enable_film": False, "dropout": 0.0},
                "classification": {"enable": False, "init_p": [0.1]},
                "unknown": {"enable": False},
                "routing": {"enable": False},
            },
        },
        "training": {"visualization": {"enable": False}},
        "loss": {
            "regression_weight": reg_w,
            "classification_weight": cls_w,
            "conservation_weight": 1.0,
            "consistency_weight": 0.0,
            "nonneg_penalty_weight": 0.3,
            "normalize_per_device": True,
            "active_threshold_rel": 0.02,
            "active_boost_weight": 2.0,
            "off_penalty_weight": 0.15,
            "rel_loss_weight": 1.5,
            "peak_focus_top_p": 0.1,
            "peak_focus_weight": 0.8,
            "shape_loss_weight": 0.2,
            "derivative_loss_weight": 0.3,
            "edge_focus_weight": 0.2,
            "edge_focus_thr_rel": 0.03,
            "multiscale_shapes": [2, 4, 8],
        },
        "evaluation": {"threshold_method": "optimal"},
        "aux_training": {"metric_learning": {"enable": False}},
    })
    return cfg


def make_batch(B: int = 2, L: int = 64, K: int = 5, device: str = "cpu"):
    torch.manual_seed(0)
    tf = torch.randn(B, L, 4, device=device).float()
    mask_t = torch.ones(B, L, device=device).bool()
    targ = torch.zeros(B, L, K, device=device).float()
    for b in range(B):
        for k in range(K):
            s = 4 + 7 * k
            e = min(L, s + 8)
            targ[b, s:e, k] = torch.rand(1).item() * (100.0 + 50.0 * k)
    mains = targ.sum(dim=2) + 0.01 * torch.randn(B, L, device=device)
    return {
        "time_features": tf,
        "time_valid_mask": mask_t,
        "target_seq": targ,
        "target_seq_valid_mask": torch.ones(B, L, device=device).bool(),
        "mains_seq": mains,
    }


def grad_norm_on_reg_head(module: NILMLightningModule):
    head = module.model.prediction_head.regression_heads[0]
    return float(head[-2].weight.grad.norm().item() if head[-2].weight.grad is not None else 0.0)


def test_regression_weight_scales_gradients():
    torch.manual_seed(0)
    cfg1 = make_config(reg_w=1.0, cls_w=0.0)
    cfg1.loss.conservation_weight = 0.0
    cfg1.loss.consistency_weight = 0.0
    cfg1.loss.nonneg_penalty_weight = 0.0
    cfg1.loss.shape_loss_weight = 0.0
    cfg1.loss.derivative_loss_weight = 0.0
    cfg1.loss.edge_focus_weight = 0.0
    m1 = NILMLightningModule(cfg1, {"accelerator": "cpu"}, cfg1.data.device_names)
    m1.power_scale = torch.tensor([100.0, 150.0, 200.0, 250.0, 50.0]).float()
    batch = make_batch(device="cpu")
    m1.zero_grad()
    pred, loss = m1._forward_and_compute_loss(batch, stage="train")
    loss.backward()
    g1 = grad_norm_on_reg_head(m1)

    torch.manual_seed(0)
    cfg10 = make_config(reg_w=10.0, cls_w=0.0)
    cfg10.loss.conservation_weight = 0.0
    cfg10.loss.consistency_weight = 0.0
    cfg10.loss.nonneg_penalty_weight = 0.0
    cfg10.loss.shape_loss_weight = 0.0
    cfg10.loss.derivative_loss_weight = 0.0
    cfg10.loss.edge_focus_weight = 0.0
    m10 = NILMLightningModule(cfg10, {"accelerator": "cpu"}, cfg10.data.device_names)
    m10.power_scale = m1.power_scale.clone()
    batch2 = make_batch(device="cpu")
    m10.zero_grad()
    pred2, loss2 = m10._forward_and_compute_loss(batch2, stage="train")
    loss2.backward()
    g10 = grad_norm_on_reg_head(m10)

    ratio = g10 / max(g1, 1e-8)
    assert ratio > 6.0 and ratio < 14.0


def test_external_scale_applied_in_forward_seq():
    torch.manual_seed(1)
    cfg = make_config(reg_w=1.0, cls_w=0.0)
    module = NILMLightningModule(cfg, {"accelerator": "cpu"}, cfg.data.device_names)
    tf = torch.randn(2, 32, 4).float()
    ones = torch.ones(1, 1, len(cfg.data.device_names)).float()
    big = 1000.0 * ones
    out1 = module.model.forward_seq(tf, external_scale=ones)[0]
    out2 = module.model.forward_seq(tf, external_scale=big)[0]
    r = float(out2.abs().mean() / out1.abs().mean())
    assert r > 900.0 and r < 1100.0


def test_gradients_flow_to_regression_heads():
    torch.manual_seed(2)
    cfg = make_config(reg_w=5.0, cls_w=0.0)
    m = NILMLightningModule(cfg, {"accelerator": "cpu"}, cfg.data.device_names)
    m.power_scale = torch.tensor([120.0, 140.0, 160.0, 180.0, 60.0]).float()
    b = make_batch(device="cpu")
    m.zero_grad()
    _, loss = m._forward_and_compute_loss(b, stage="train")
    loss.backward()
    g_head = grad_norm_on_reg_head(m)
    g_seq = float(m.model.prediction_head.seq_conv.weight.grad.norm().item())
    assert g_head > 0.0
    assert g_seq > 0.0


def test_classification_disabled_no_drag():
    torch.manual_seed(3)
    cfg = make_config(reg_w=1.0, cls_w=0.0)
    m = NILMLightningModule(cfg, {"accelerator": "cpu"}, cfg.data.device_names)
    assert m.classification_enabled is False
    assert m.model.prediction_head.classification_heads is None
    b = make_batch(device="cpu")
    _, loss = m._forward_and_compute_loss(b, stage="train")
    assert math.isfinite(float(loss.item()))