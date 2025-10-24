import os
import sys
from pathlib import Path

import torch
import numpy as np
from omegaconf import OmegaConf

# 允许脚本直接导入项目模块
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from src.data.datamodule import NILMDataModule
from src.train import NILMLightningModule, load_device_info


def build_config(device_names):
    return OmegaConf.create({
        'project_name': 'DisaggNet',
        'paths': {
            'output_dir': 'outputs',
            'data_dir': 'Data',
            'prepared_dir': 'Data/prepared'
        },
        'data': {
            'batch_size': 2,
            'num_workers': 0,
            'pin_memory': False,
            'device_names': device_names,
            'window_size': 64
        },
        'model': {
            'd_model': 64,
            'n_heads': 4,
            'num_layers': 2,
            'dropout': 0.1,
            'time_encoder': {
                'd_model': 64, 'n_heads': 4, 'num_layers': 2,
                'dropout': 0.1, 'input_conv_embed': False, 'causal_mask': True
            },
            'freq_encoder': {'enable': True, 'proj_dim': 32, 'conv1d_kernel': 3, 'small_transformer_layers': 0, 'dropout': 0.1},
            'fusion': {'type': 'cross_attention', 'gated': True},
            'aux_encoder': {'enable': True, 'hidden': 32, 'dropout': 0.1},
            'heads': {
                'regression': {'hidden': 32},
                'classification': {'init_p': 0.1}
            },
            'calibration': {'enable': False}
        },
        'training': {
            'max_epochs': 1,
            'min_epochs': 1,
            'log_every_n_steps': 1,
            'optimizer': {'name': 'adamw', 'lr': 1e-3},
            'monitor_device_stats': False,
            'num_sanity_val_steps': 0,
        },
        'loss': {
            'classification_weight': 0.0,
            'regression_weight': 1.0,
            'conservation_weight': 0.5,
            'consistency_weight': 1.0,
            'focal_alpha': 0.25,
            'focal_gamma': 2.0,
            'huber_delta': 1.0,
            'normalize_per_device': True,
            'huber_beta_rel': 0.05,
            'rel_loss_weight': 0.5,
            'rel_eps': 0.05,
            'active_threshold_rel': 0.05,
        },
        'evaluation': {'threshold_method': 'optimal'},
        'debug': {'track_grad_norm': 0, 'strict_validation': False},
        'reproducibility': {'deterministic': False, 'benchmark': False},
    })


def pct_near_zero(x, tol=1e-8):
    if x.numel() == 0:
        return 0.0
    return float((x.abs() < tol).float().mean().item())


def stats(name, x):
    x_np = x.detach().cpu()
    finite = torch.isfinite(x_np)
    safe = torch.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0)
    mean = float(safe.mean().item())
    std = float(safe.std().item())
    minv = float(safe.min().item())
    maxv = float(safe.max().item())
    zeros = pct_near_zero(safe)
    print(f"[STAT] {name}: shape={tuple(x_np.shape)}, finite={float(finite.float().mean().item()):.4f}, mean={mean:.6f}, std={std:.6f}, min={minv:.6f}, max={maxv:.6f}, pct(|x|<1e-8)={zeros:.4f}")


def main():
    os.environ.setdefault('DISAGGNET_FORCE_CPU', '1')

    prepared_dir = repo_root / 'Data' / 'prepared'
    if not prepared_dir.exists():
        print(f"[ERROR] prepared_dir not found: {prepared_dir}")
        return

    # 设备信息
    # 优先使用映射文件；若缺失则走内部回退
    device_info, device_names = load_device_info(build_config([]))
    print(f"[INFO] devices: {device_names}")

    # 构建配置、模块与数据
    config = build_config(device_names)
    dm = NILMDataModule(config, data_root=str(prepared_dir), fold_id=0)
    dm.prepare_data()
    dm.setup()

    device_info, device_names2 = load_device_info(config)
    lm = NILMLightningModule(config, device_info, device_names2)
    lm.eval()

    # 读取 power_scale_vec
    power_scale_vec = getattr(dm, 'power_scale_vec', None)
    if isinstance(power_scale_vec, torch.Tensor):
        print(f"[INFO] power_scale_vec: shape={tuple(power_scale_vec.shape)}, min={float(power_scale_vec.min().item()):.6f}, max={float(power_scale_vec.max().item()):.6f}")
    else:
        print("[WARN] dm.power_scale_vec is None")

    print(f"[INFO] normalize_per_device={lm.normalize_per_device}, huber_beta_rel={lm.huber_beta_rel}, rel_loss_weight={lm.rel_loss_weight}, rel_eps={lm.rel_eps}")

    # 抓取一个验证批次
    val_batch = next(iter(dm.val_dataloader()))
    # 批次字段统计
    for key in ['time_features', 'time_valid_mask', 'freq_features', 'freq_valid_mask', 'aux_features', 'aux_valid_mask', 'target_seq', 'target_seq_valid_mask', 'target_power', 'target_states', 'mains_seq', 'total_power']:
        if key in val_batch:
            x = val_batch[key]
            print(f"[BATCH] {key}: shape={tuple(x.shape)} dtype={x.dtype}")
            if x.dtype.is_floating_point:
                stats(key, x)

    # 前向路径：时域编码 -> 频域编码 -> 融合 -> 序列预测
    time_features = val_batch['time_features']
    freq_features = val_batch.get('freq_features', None)
    aux_features = val_batch['aux_features']
    time_valid_mask = val_batch.get('time_valid_mask', None)
    freq_valid_mask = val_batch.get('freq_valid_mask', None)
    aux_valid_mask = val_batch.get('aux_valid_mask', None)

    # 时域编码
    time_repr = lm.model.time_encoder(time_features, None, mask=time_valid_mask)
    stats('time_repr', time_repr)

    # 频域编码
    freq_repr = None
    if freq_features is not None:
        freq_repr = lm.model.freq_encoder(freq_features, freq_valid_mask)
        if freq_repr is not None:
            stats('freq_repr', freq_repr if freq_repr.dim() == 2 else freq_repr.mean(dim=1))

    # 辅助特征融合
    if lm.model.aux_encoder is not None and aux_features is not None:
        aux_repr = lm.model.aux_encoder(aux_features, aux_valid_mask)
        stats('aux_repr', aux_repr)
        seq_len = time_repr.size(1)
        aux_seq = aux_repr.unsqueeze(1).expand(-1, seq_len, -1)
        gate = lm.model.aux_gate(aux_seq)
        time_repr = time_repr + lm.model.aux_weight * gate * aux_seq
        stats('time_repr+aux', time_repr)

    # 融合
    fused_repr = lm.model.fusion(time_repr, freq_repr) if lm.model.fusion is not None else time_repr
    stats('fused_repr', fused_repr)

    # 序列预测
    seq_pred = lm.model.prediction_head.forward_seq(fused_repr)
    stats('seq_pred', seq_pred)

    # 窗口级预测（回归/分类）
    reg_pred, cls_pred, unknown_pred = lm.model.prediction_head(fused_repr)
    stats('reg_pred', reg_pred)
    stats('cls_pred', cls_pred)
    if unknown_pred is not None:
        stats('unknown_pred', unknown_pred)

    # 近似零比例报告
    nz = pct_near_zero(seq_pred)
    print(f"[RESULT] seq_pred near-zero fraction: {nz:.4f}")


if __name__ == '__main__':
    main()