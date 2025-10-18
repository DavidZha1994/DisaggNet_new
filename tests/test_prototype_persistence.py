import torch
from omegaconf import OmegaConf

from src.train import NILMLightningModule, load_device_info


def build_min_config_for_metric(n_devices=3, embed_dim=32):
    # 最小可运行配置，开启度量学习
    return OmegaConf.create({
        'project_name': 'DisaggNet',
        'paths': {'output_dir': 'outputs', 'data_dir': 'Data', 'prepared_dir': 'Data/prepared'},
        'data': {'batch_size': 2, 'num_workers': 0, 'pin_memory': False, 'device_names': [f'device_{i+1}' for i in range(n_devices)], 'window_size': 8},
        'model': {
            'd_model': embed_dim,
            'n_heads': 2,
            'num_layers': 1,
            'dropout': 0.1,
            'time_encoder': {
                'd_model': embed_dim, 'n_heads': 2, 'num_layers': 1,
                'dropout': 0.1, 'input_conv_embed': False, 'causal_mask': True
            },
            'freq_encoder': {'enable': False, 'proj_dim': 16, 'conv1d_kernel': 3, 'small_transformer_layers': 0, 'dropout': 0.1},
            'fusion': {'type': 'cross_attention', 'gated': True},
            'aux_encoder': {'enable': False, 'hidden': 16, 'dropout': 0.1},
            'heads': {
                'regression': {'hidden': 16},
                'classification': {'init_p': 0.1},
                'unknown': {'enable': True}
            },
            'calibration': {'enable': False}
        },
        'aux_training': {
            'metric_learning': {
                'enable': True,
                'margin': 0.2,
                'weight': 0.2,
                'use_power': True
            }
        },
        'training': {
            'max_epochs': 1,
            'log_every_n_steps': 1,
            'optimizer': {'name': 'adamw', 'lr': 1e-3},
        },
        'loss': {
            'classification_weight': 2.0,
            'regression_weight': 1.0,
            'conservation_weight': 0.5,
            'consistency_weight': 1.0,
            'focal_alpha': 0.25,
            'focal_gamma': 2.0,
            'huber_delta': 1.0,
        },
        'evaluation': {'threshold_method': 'optimal'},
        'debug': {'track_grad_norm': 0, 'strict_validation': True},
    })


def build_fake_batch(B=2, W=8, C=3, n_devices=3):
    time_features = torch.randn(B, W, C, dtype=torch.float32)
    # 仅使用时域以简化
    target_power = torch.rand(B, n_devices, dtype=torch.float32)
    target_states = torch.randint(0, 2, (B, n_devices), dtype=torch.float32)

    return {
        'time_features': time_features,
        'freq_features': None,
        'time_positional': None,
        'aux_features': None,
        'target_power': target_power,
        'target_states': target_states,
    }


def test_prototype_library_persistence_roundtrip(tmp_path):
    n_devices = 3
    embed_dim = 32
    config = build_min_config_for_metric(n_devices=n_devices, embed_dim=embed_dim)
    device_info, device_names = load_device_info(config)

    lm = NILMLightningModule(config, device_info, device_names)
    assert lm.metric_learning_enable
    assert lm.prototype_library is not None

    # 运行一次 forward_with_embeddings 并更新原型库
    batch = build_fake_batch(B=4, W=8, C=3, n_devices=n_devices)
    reg, cls, unk, emb = lm.forward_with_embeddings(batch)
    # 使用 states 更新统计；这里用分类 logits 的 sigmoid 作为近似激活
    states = torch.sigmoid(cls.detach())
    lm.prototype_library.update(emb.detach(), states.detach())

    # 保存检查点（调用模块钩子以持久化原型库）
    ckpt = {}
    lm.on_save_checkpoint(ckpt)
    assert 'prototype_library_state' in ckpt

    # 创建新模块并加载检查点
    lm2 = NILMLightningModule(config, device_info, device_names)
    lm2.on_load_checkpoint(ckpt)

    # 比较状态字典是否一致
    s1 = lm.prototype_library.state_dict()
    s2 = lm2.prototype_library.state_dict()
    # 逐键比较张量近似相等
    for k in s1.keys():
        v1 = s1[k]
        v2 = s2[k]
        assert torch.allclose(v1, v2, atol=1e-6), f"state mismatch on {k}"

    # 进一步验证：用相同 embeddings 计算 Mahalanobis 距离应一致
    d1 = lm.prototype_library.mahalanobis(emb.detach())
    d2 = lm2.prototype_library.mahalanobis(emb.detach())
    assert torch.allclose(d1, d2, atol=1e-6)