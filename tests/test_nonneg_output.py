import torch
from omegaconf import OmegaConf
from src.train import NILMLightningModule, load_device_info
from src.models.fusion_transformer import FusionTransformer


def build_config():
    return OmegaConf.create({
        'paths': {'prepared_dir': 'Data/prepared'},
        'data': {'batch_size': 2, 'num_workers': 0, 'pin_memory': False, 'device_names': ['dev1', 'dev2'], 'window_size': 16},
        'model': {
            'time_encoder': {'d_model': 96, 'n_heads': 8, 'num_layers': 3, 'dropout': 0.2, 'input_conv_embed': False, 'causal_mask': True},
            'freq_encoder': {'enable': False, 'proj_dim': 96},
            'fusion': {'type': 'none'},
            'aux_encoder': {'enable': False, 'hidden': 64, 'dropout': 0.1},
            'heads': {'regression': {'hidden': 64, 'seq_use_softplus': True}}
        },
        'training': {'max_epochs': 1, 'optimizer': {'name': 'adamw', 'lr': 1e-4}},
        'loss': {},
        'evaluation': {'test_after_training': False}
    })


def build_batch(B=2, T=16, F=3, K=2):
    time_features = torch.rand(B, T, F)
    target_seq = torch.rand(B, T, K)
    status_seq = torch.randint(0, 2, (B, T, K)).float()
    target_seq_valid_mask = torch.ones(B, T, dtype=torch.bool)
    mains_seq = torch.rand(B, T)
    return {
        'time_features': time_features,
        'target_seq': target_seq,
        'status_seq': status_seq,
        'target_seq_valid_mask': target_seq_valid_mask,
        'mains_seq': mains_seq
    }


def test_nonnegative_seq_outputs():
    cfg = build_config()
    info, names = load_device_info(cfg)
    lm = NILMLightningModule(cfg, info, names)
    batch = build_batch()
    seq_pred, loss = lm._forward_and_compute_loss(batch, 'test')
    assert seq_pred.min().item() >= 0.0
