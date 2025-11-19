import os
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.data.datamodule import NILMDataModule
from src.models.fusion_transformer import FusionTransformer
from src.train import NILMLightningModule, load_device_info


def _load_cfg(cfg_path: str, prepared_dir: str):
    cfg = OmegaConf.load(cfg_path)
    if not hasattr(cfg, "paths"):
        cfg.paths = OmegaConf.create({})
    cfg.paths.prepared_dir = prepared_dir
    if not hasattr(cfg, "data"):
        cfg.data = OmegaConf.create({})
    cfg.data.batch_size = 4
    cfg.data.num_workers = 0
    if not hasattr(cfg, "training"):
        cfg.training = OmegaConf.create({})
    if not hasattr(cfg.training, "visualization"):
        cfg.training.visualization = OmegaConf.create({})
    cfg.training.visualization.enable = False
    return cfg


@pytest.mark.usefixtures("prepared_dir")
def test_datamodule_batch_step_by_step(prepared_dir):
    cfg = _load_cfg("configs/training/optimized_stable.yaml", prepared_dir)
    dm = NILMDataModule(cfg, data_root=prepared_dir)
    dm.prepare_data()
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    if isinstance(dm.power_scale_vec, torch.Tensor):
        print("power_scale_vec:", dm.power_scale_vec.tolist())
    tseq = batch.get("target_seq")
    if isinstance(tseq, torch.Tensor):
        B,L,K = tseq.shape
        flat = tseq.reshape(-1, K)
        p95 = torch.quantile(torch.clamp(flat, min=0.0), 0.95, dim=0)
        print("target_seq_p95:", p95.tolist())
    assert isinstance(batch["time_features"], torch.Tensor)
    assert batch["time_features"].dim() == 3
    if "freq_features" in batch:
        assert batch["freq_features"].dim() == 3
    if "target_seq" in batch:
        assert batch["target_seq"].dim() == 3
    B, L, C = batch["time_features"].shape
    tseq = batch.get("target_seq")
    sseq = batch.get("status_seq")
    tp = batch.get("target_power")
    tvm = batch.get("target_seq_valid_mask")
    m = None
    if isinstance(tvm, torch.Tensor):
        m = tvm.to(torch.float32)
    if isinstance(tseq, torch.Tensor) and isinstance(tp, torch.Tensor) and tseq.size(0) == tp.size(0):
        if m is None:
            mean_seq = tseq.mean(dim=1)
        else:
            num = (tseq * m.unsqueeze(-1)).sum(dim=1)
            den = m.unsqueeze(-1).sum(dim=1).clamp_min(1.0)
            mean_seq = num / den
        if tp.size(1) == mean_seq.size(1):
            diff = torch.abs(mean_seq - tp)
            assert torch.isfinite(diff).all()
    print("time_features:", batch["time_features"].shape)
    if "freq_features" in batch:
        print("freq_features:", batch["freq_features"].shape)
    if "target_seq" in batch:
        print("target_seq:", batch["target_seq"].shape)
    if "status_seq" in batch:
        print("status_seq:", batch["status_seq"].shape)
    print("target_power:", batch["target_power"].shape)
    if "total_power" in batch:
        print("total_power:", batch["total_power"].shape)


@pytest.mark.usefixtures("prepared_dir")
def test_model_forward_seq_step_by_step(prepared_dir):
    cfg = _load_cfg("configs/training/optimized_stable.yaml", prepared_dir)
    dm = NILMDataModule(cfg, data_root=prepared_dir)
    dm.prepare_data()
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    device_info, device_names = load_device_info(cfg)
    model = FusionTransformer(cfg.model, len(device_names))
    tf = batch["time_features"]
    tvm = batch.get("time_valid_mask")
    fs = batch.get("freq_features")
    fvm = batch.get("freq_valid_mask")
    ext_scale = dm.power_scale_vec if isinstance(dm.power_scale_vec, torch.Tensor) else None
    time_repr = model.time_encoder(tf, None, mask=tvm)
    freq_repr = None
    if fs is not None:
        freq_repr = model.freq_encoder(fs, fvm) if model.freq_encoder is not None else None
    fused_repr = model.fusion(time_repr, freq_repr) if model.fusion is not None else time_repr
    seq_pred_norm = model.prediction_head.forward_seq(fused_repr)
    print("time_repr mean/std:", float(time_repr.mean()), float(time_repr.std()))
    if isinstance(freq_repr, torch.Tensor):
        print("freq_repr mean/std:", float(freq_repr.mean()), float(freq_repr.std()))
    print("seq_pred_norm mean/std:", float(seq_pred_norm.mean()), float(seq_pred_norm.std()))
    out = model.forward_seq(tf, fs, None, batch.get("aux_features"), tvm, fvm, batch.get("aux_valid_mask"), ext_scale)
    pred_seq, reg_win, cls_win, unk_win, cls_seq = out
    assert pred_seq.shape[:2] == tf.shape[:2]
    assert pred_seq.shape[2] == len(device_names)
    nz_ratio = (pred_seq.abs() > 1e-12).float().mean().item()
    print("pred_seq nz_ratio:", nz_ratio)
    assert nz_ratio > 0.01
    m = tvm
    if isinstance(m, torch.Tensor):
        m = m.to(torch.float32).unsqueeze(-1)
        num = (pred_seq * m).sum(dim=1)
        den = m.sum(dim=1).clamp_min(1.0)
        mean_seq = num / den
    else:
        mean_seq = pred_seq.mean(dim=1)
    mean_diff = torch.abs(mean_seq - reg_win)
    print("mean_diff mean:", mean_diff.mean().item())
    assert torch.isfinite(mean_diff).all()


@pytest.mark.usefixtures("prepared_dir")
def test_train_losses_and_gradients_step_by_step(prepared_dir):
    cfg = _load_cfg("configs/training/optimized_stable.yaml", prepared_dir)
    device_info, device_names = load_device_info(cfg)
    dm = NILMDataModule(cfg, data_root=prepared_dir)
    dm.prepare_data()
    dm.setup()
    module = NILMLightningModule(cfg, device_info, device_names)
    if isinstance(dm.power_scale_vec, torch.Tensor):
        module.power_scale = dm.power_scale_vec
    batch = next(iter(dm.train_dataloader()))
    if 'target_seq' not in batch:
        import pytest as _pytest
        _pytest.skip("缺少 target_seq 标签，跳过梯度检查（合成数据仅用于管道形状验证）")
    pred_seq, total_loss = module._forward_and_compute_loss(batch, stage="train")
    assert torch.isfinite(total_loss)
    assert total_loss.item() >= 0.0
    total_loss.backward()
    nonzero = 0
    for p in module.model.parameters():
        if p.grad is not None:
            g = p.grad.data
            if torch.isfinite(g).all() and g.abs().sum().item() > 0:
                nonzero += 1
    print("grad_nonzero_count:", nonzero)
    assert nonzero > 10