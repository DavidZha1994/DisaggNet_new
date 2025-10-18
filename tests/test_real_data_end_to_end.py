import os
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from src.data.datamodule import NILMDataModule
from src.train import NILMLightningModule, load_device_info


PREPARED_DIR = "/Users/yu/Workspace/DisaggNet_new/Data/prepared"
FOLD_ID = 0


def _get_device_names_from_prepared(prepared_dir: str):
    mapping_path = Path(prepared_dir) / "device_name_to_id.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"missing device_name_to_id.json at {mapping_path}")
    with open(mapping_path, "r") as f:
        mapping = json.load(f)

    # 支持 name->id 或 id->name 两种格式
    def _to_int(x):
        try:
            return int(x)
        except Exception:
            return None

    # 判定方向
    sample_key = next(iter(mapping.keys()))
    sample_val = mapping[sample_key]
    key_is_int_like = _to_int(sample_key) is not None
    val_is_int_like = _to_int(sample_val) is not None

    device_names = []
    if key_is_int_like and not val_is_int_like:
        # id(str)->name(str)
        pairs = sorted(((int(k), v) for k, v in mapping.items()), key=lambda kv: kv[0])
        device_names = [name for _, name in pairs]
    elif not key_is_int_like and val_is_int_like:
        # name->id
        pairs = sorted(((v, k) for k, v in mapping.items()), key=lambda kv: kv[0])
        device_names = [name for _, name in pairs]
    else:
        # 回退：按键排序
        device_names = sorted(list(mapping.keys()))

    # 与targets_seq维度一致性快速检查（若存在）
    seq_path = Path(prepared_dir) / f"fold_{FOLD_ID}" / "train_targets_seq.npy"
    if seq_path.exists():
        arr = np.load(seq_path)
        n_devices_from_seq = arr.shape[-1]
        assert n_devices_from_seq == len(device_names), (
            f"targets_seq设备维度({n_devices_from_seq})与映射设备数({len(device_names)})不一致")
    return device_names


def _build_real_config(device_names):
    return OmegaConf.create({
        'project_name': 'DisaggNet',
        'paths': {
            'output_dir': 'outputs',
            'data_dir': 'Data',
            'prepared_dir': 'Data/prepared'
        },
        'data': {
            'batch_size': 4,
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
            'log_every_n_steps': 1,
            'optimizer': {'name': 'adamw', 'lr': 1e-3},
        },
        'loss': {
            # 显式关闭分类以匹配当前回归-only训练
            'classification_weight': 0.0,
            'regression_weight': 1.0,
            'conservation_weight': 0.5,
            'consistency_weight': 1.0,
            'focal_alpha': 0.25,
            'focal_gamma': 2.0,
            'huber_delta': 1.0,
        },
        'evaluation': {'threshold_method': 'optimal'},
        'debug': {'track_grad_norm': 0, 'strict_validation': False},
    })


def test_datamodule_real_batch_shapes():
    device_names = _get_device_names_from_prepared(PREPARED_DIR)
    config = _build_real_config(device_names)

    dm = NILMDataModule(config, data_root=PREPARED_DIR, fold_id=FOLD_ID)
    dm.prepare_data()
    dm.setup()

    # 取一个训练批次
    train_batch = next(iter(dm.train_dataloader()))

    assert 'time_features' in train_batch
    assert 'aux_features' in train_batch
    assert 'target_states' in train_batch

    # 可选项存在则检查形状
    if 'freq_features' in train_batch:
        assert train_batch['freq_features'].dim() == 3
    if 'target_seq' in train_batch:
        assert train_batch['target_seq'].dim() == 3

    # 基本形状匹配
    B, T, F_time = train_batch['time_features'].shape
    n_devices = len(device_names)
    assert train_batch['aux_features'].shape[0] == B
    assert train_batch['target_states'].shape[0] == B

    # target_power 维度与设备数一致（若为占位，则与states一致）
    assert train_batch['target_power'].shape[0] == B
    assert train_batch['target_power'].shape[1] in (n_devices, train_batch['target_states'].shape[1])


def test_lightningmodule_forward_and_training_step_real():
    device_names = _get_device_names_from_prepared(PREPARED_DIR)
    config = _build_real_config(device_names)
    device_info, device_names2 = load_device_info(config)

    lm = NILMLightningModule(config, device_info, device_names2)
    assert lm.n_devices == len(device_names)
    assert lm.classification_enabled is False

    dm = NILMDataModule(config, data_root=PREPARED_DIR, fold_id=FOLD_ID)
    dm.prepare_data()
    dm.setup()
    batch = next(iter(dm.train_dataloader()))

    preds = lm(batch)
    if isinstance(preds, tuple) and len(preds) == 3:
        pred_power, pred_states, _ = preds
    else:
        pred_power, pred_states = preds

    B = batch['time_features'].shape[0]
    n_devices = len(device_names)
    assert pred_power.shape == (B, n_devices)
    assert pred_states.shape == (B, n_devices)
    assert torch.isfinite(pred_power).all()
    assert torch.isfinite(pred_states).all()

    loss = lm.training_step(batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_val_metrics_regression_only():
    device_names = _get_device_names_from_prepared(PREPARED_DIR)
    config = _build_real_config(device_names)
    device_info, device_names2 = load_device_info(config)
    lm = NILMLightningModule(config, device_info, device_names2)

    dm = NILMDataModule(config, data_root=PREPARED_DIR, fold_id=FOLD_ID)
    dm.prepare_data()
    dm.setup()
    batch = next(iter(dm.val_dataloader()))

    # 产生窗口级预测并计算指标
    preds = lm(batch)
    if isinstance(preds, tuple) and len(preds) == 3:
        pred_power, pred_states, _ = preds
    else:
        pred_power, pred_states = preds

    metrics = lm._compute_metrics(batch, (pred_power, pred_states), stage='val')

    # 回归指标应存在
    for k in ['mae', 'nde', 'sae', 'teca', 'score']:
        assert k in metrics

    # 分类相关指标在分类关闭时不应出现
    for k in ['f1', 'mcc', 'roc_auc', 'pr_auc']:
        assert k not in metrics


def test_forward_seq_shapes_real():
    device_names = _get_device_names_from_prepared(PREPARED_DIR)
    config = _build_real_config(device_names)
    device_info, device_names2 = load_device_info(config)
    lm = NILMLightningModule(config, device_info, device_names2)

    dm = NILMDataModule(config, data_root=PREPARED_DIR, fold_id=FOLD_ID)
    dm.prepare_data()
    dm.setup()
    batch = next(iter(dm.train_dataloader()))

    seq_pred, reg_pred, cls_pred, unk_pred = lm.model.forward_seq(
        time_features=batch['time_features'],
        freq_features=batch.get('freq_features'),
        time_positional=None,
        aux_features=batch.get('aux_features'),
    )

    B, T, _ = batch['time_features'].shape
    n_devices = len(device_names)
    assert seq_pred.shape == (B, T, n_devices)
    assert reg_pred.shape == (B, n_devices)
    assert cls_pred.shape == (B, n_devices)
    if unk_pred is not None:
        assert unk_pred.shape == (B, 1)
    assert torch.isfinite(seq_pred).all()