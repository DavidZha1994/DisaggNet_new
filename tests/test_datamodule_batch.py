import torch
import numpy as np
from omegaconf import OmegaConf

from src.data.datamodule import NILMDataModule


def build_fake_item(W=8, C=3, Fm=5, has_pkw_mean=True):
    # 构造工程/统计特征（mains）向量
    mains = torch.randn(Fm, dtype=torch.float32)
    if has_pkw_mean:
        # 将第1个位置作为 P_kW_mean，确保正值以便比较
        mains[1] = torch.tensor(1.23, dtype=torch.float32)

    # 原始窗口作为 aux_features，形状 (W, C)
    raw = torch.randn(W, C, dtype=torch.float32)
    # 将第0通道视为 P_kW，确保均值可计算
    raw[:, 0] = torch.linspace(0.5, 1.5, W)

    # 频域帧（可选），使用二维以测试 unsqueeze 行为
    freq = torch.randn(10, 6, dtype=torch.float32)

    # targets：第一列为回归功率，第二列为二值状态
    target_power = torch.rand(1, dtype=torch.float32)
    target_state = torch.randint(0, 2, (1,), dtype=torch.float32)
    targets = torch.cat([target_power, target_state], dim=0)

    item = {
        "mains": mains,
        "aux_features": raw,
        "freq_features": freq,
        "targets": targets,
        "timestamps": 1690000000,
    }
    return item


def test_collate_and_map_with_pkw_mean_extraction():
    # 构造最小配置
    config = OmegaConf.create({
        'data': {'batch_size': 2, 'num_workers': 0, 'pin_memory': False},
        'imbalance_handling': {'sampling_strategy': 'mixed'}
    })
    dm = NILMDataModule(config)

    # 定义特征名称，包含 P_kW_mean
    dm.feature_names = ["f0", "P_kW_mean", "f2", "f3", "f4"]
    # 定义原始通道名称，包含 P_kW
    dm.raw_channel_names = ["P_kW", "Q_kVAR", "S_kVA"]

    # 构造批次
    batch = [build_fake_item(has_pkw_mean=True) for _ in range(3)]

    out = dm._collate_and_map(batch)

    # 断言必要字段存在
    assert 'time_features' in out
    assert 'aux_features' in out
    assert 'freq_features' in out
    assert 'target_power' in out
    assert 'target_states' in out
    assert 'timestamps' in out
    assert 'total_power' in out

    # 形状检查
    B = 3
    assert out['time_features'].shape == (B, 8, 3)
    assert out['aux_features'].shape == (B, 5)
    # freq_features 为二维输入，期望被扩展为 (B, 1, Ff)
    assert out['freq_features'].dim() == 3
    assert out['target_power'].shape[0] == B
    assert out['target_states'].shape[0] == B
    assert out['timestamps'].dim() in (1, 2)
    assert out['timestamps'].shape[0] == B
    assert out['total_power'].shape == (B, 1)

    # total_power 应来自 P_kW_mean 的列
    expected = torch.stack([b['mains'][1] for b in batch], dim=0).unsqueeze(1)
    assert torch.allclose(out['total_power'], expected, atol=1e-6)


def test_collate_and_map_fallback_total_power_from_raw_channel():
    # 构造最小配置
    config = OmegaConf.create({
        'data': {'batch_size': 2, 'num_workers': 0, 'pin_memory': False},
        'imbalance_handling': {'sampling_strategy': 'mixed'}
    })
    dm = NILMDataModule(config)

    # 不包含 P_kW_mean，使其回退到原始窗口的 P_kW 通道
    dm.feature_names = ["f0", "f1", "f2", "f3", "f4"]
    dm.raw_channel_names = ["P_kW", "Q_kVAR", "S_kVA"]

    W = 8
    C = 3
    batch = [build_fake_item(W=W, C=C, Fm=5, has_pkw_mean=False) for _ in range(4)]

    out = dm._collate_and_map(batch)

    # 形状检查
    B = 4
    assert out['time_features'].shape == (B, W, C)
    assert out['total_power'].shape == (B, 1)

    # 验证 total_power 等于 P_kW 通道的时间均值
    pkw_means = []
    for b in batch:
        tf = b['aux_features']  # (W, C)
        pkw_means.append(tf[:, 0].mean())
    expected = torch.stack(pkw_means, dim=0).unsqueeze(1)
    assert torch.allclose(out['total_power'], expected, atol=1e-6)