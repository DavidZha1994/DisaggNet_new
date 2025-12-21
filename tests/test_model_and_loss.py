import torch
from omegaconf import OmegaConf

from src.models.fusion_transformer import FusionTransformer
from src.losses.losses import create_loss_function
from src.train import NILMLightningModule, load_device_info


def build_min_config(n_devices=2):
    # 以 base.yaml 为参考的最小配置
    return OmegaConf.create({
        'project_name': 'DisaggNet',
        'paths': {'output_dir': 'outputs', 'data_dir': 'Data', 'prepared_dir': 'Data/prepared'},
        'data': {'batch_size': 4, 'num_workers': 0, 'pin_memory': False, 'device_names': [f'device_{i+1}' for i in range(n_devices)], 'window_size': 8},
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


def build_fake_batch(B=4, W=8, C=3, Ff=12, Fa=5, n_devices=2):
    # time_features: (B, W, C)
    time_features = torch.randn(B, W, C, dtype=torch.float32)
    time_features[:, :, 0] = torch.linspace(0.5, 1.5, W).unsqueeze(0).repeat(B, 1)

    # freq_features: (B, 1, Ff)
    freq_features = torch.randn(B, 1, Ff, dtype=torch.float32)

    # aux_features: (B, Fa)
    aux_features = torch.randn(B, Fa, dtype=torch.float32)
    # total_power from P_kW_mean (index 1)
    aux_features[:, 1] = 1.23

    # targets split: power (B, n_devices), states (B, n_devices)
    target_power = torch.rand(B, n_devices, dtype=torch.float32)
    target_states = torch.randint(0, 2, (B, n_devices), dtype=torch.float32)

    # total_power: (B, 1)
    total_power = aux_features[:, 1].unsqueeze(1)

    return {
        'time_features': time_features,
        'freq_features': freq_features,
        'aux_features': aux_features,
        'target_power': target_power,
        'target_states': target_states,
        'total_power': total_power,
    }


def test_fusion_transformer_forward_shapes():
    n_devices = 3
    config = build_min_config(n_devices=n_devices)
    model = FusionTransformer(config.model, n_devices)

    batch = build_fake_batch(B=5, W=8, C=3, Ff=16, Fa=6, n_devices=n_devices)

    out = model(
        time_features=batch['time_features'],
        freq_features=batch['freq_features'],
        time_positional=None,
        aux_features=batch['aux_features']
    )

    if isinstance(out, tuple) and len(out) == 3:
        reg, cls, unk = out
    else:
        reg, cls = out

    assert reg.shape == (5, n_devices)
    assert (cls is None) or (cls.shape == (5, n_devices))
    # Unknown 默认关闭时为 None；如开启则形状为 (B, 1)
    if 'unk' in locals():
        assert (unk is None) or (unk.shape == (5, 1))
    assert not torch.isnan(reg).any()
    if cls is not None:
        assert not torch.isnan(cls).any()


def test_fusion_transformer_forward_with_embeddings_shapes():
    n_devices = 3
    config = build_min_config(n_devices=n_devices)
    # 启用 unknown 头以覆盖两种返回路径
    config.model.heads['unknown'] = {'enable': True}
    model = FusionTransformer(config.model, n_devices)

    batch = build_fake_batch(B=5, W=8, C=3, Ff=16, Fa=6, n_devices=n_devices)

    reg, cls, unk, emb = model.forward_with_embeddings(
        time_features=batch['time_features'],
        freq_features=batch['freq_features'],
        time_positional=None,
        aux_features=batch['aux_features']
    )

    assert reg.shape == (5, n_devices)
    assert (cls is None) or (cls.shape == (5, n_devices))
    assert unk is None or unk.shape == (5, 1)
    assert emb.shape == (5, n_devices, config.model.time_encoder.d_model)
    assert not torch.isnan(reg).any()
    if cls is not None:
        assert not torch.isnan(cls).any()
    assert not torch.isnan(emb).any()


def test_multitask_losses_exist_and_scalar():
    cfg = {
        'regression_weight': 1.0,
        'classification_weight': 2.0,
        'conservation_weight': 0.5,
        'consistency_weight': 1.0,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'huber_delta': 1.0,
    }
    loss_fn = create_loss_function(cfg)

    B, L, K = 4, 16, 2
    pred_seq = torch.rand(B, L, K)
    target_seq = torch.rand(B, L, K)
    status_seq = torch.randint(0, 2, (B, L, K)).float()
    valid_mask = torch.ones(B, L, K, dtype=torch.bool)
    mains_seq = torch.rand(B, L)
    scale = torch.tensor([100., 100.])

    reg = loss_fn.regression_seq_loss(pred_seq, target_seq, status_seq, valid_mask, scale)
    cls = loss_fn.classification_seq_loss(pred_seq.clamp(0,1), status_seq, valid_mask)
    cons = loss_fn.conservation_loss(mains_seq, pred_seq)
    consw = loss_fn.consistency_window_loss(pred_seq, pred_seq.mean(dim=1), valid_mask, scale)
    total = reg + cls + cons + consw

    for v in [reg, cls, cons, consw, total]:
        assert isinstance(v, torch.Tensor)
        assert v.dim() == 0


def test_lightningmodule_training_step_end_to_end():
    n_devices = 2
    config = build_min_config(n_devices=n_devices)
    device_info, device_names = load_device_info(config)

    lm = NILMLightningModule(config, device_info, device_names)

    batch = build_fake_batch(B=3, W=8, C=3, Ff=12, Fa=6, n_devices=n_devices)

    # forward via module call
    preds = lm(batch)
    if isinstance(preds, tuple) and len(preds) == 3:
        pred_power, pred_states, _ = preds
    else:
        pred_power, pred_states = preds
    assert pred_power.shape == (3, n_devices)
    assert (pred_states is None) or (pred_states.shape == (3, n_devices))

    # compute loss via training_step path
    loss = lm.training_step({
        **batch,
    }, batch_idx=0)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
