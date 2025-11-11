import os
import json
import pickle
from pathlib import Path

import torch


def test_pipeline_outputs_structure(prepared_dir):
    root = Path(prepared_dir)
    assert root.exists(), f"prepared_dir 不存在: {root}"

    # 顶层文件
    top_files = [
        "cv_splits.pkl",
        "labels.pkl",
        "device_name_to_id.json",
        "label_map.json",
        "segments_meta.csv",
    ]
    for name in top_files:
        fp = root / name
        assert fp.exists(), f"缺少顶层文件: {name}"

    # 至少存在 fold_0
    fold0 = root / "fold_0"
    assert fold0.exists(), "缺少 fold_0 目录"

    # 折内文件
    for name in [
        "train_raw.pt", "val_raw.pt",
        "train_freq.pt", "val_freq.pt",
        "train_features.pt", "val_features.pt",
        "train_targets_seq.pt", "val_targets_seq.pt",
        "train_mask.pt", "val_mask.pt",
        "train_indices.pt", "val_indices.pt",
        "train_metadata.csv", "val_metadata.csv",
        "feature_names.json", "raw_channel_names.json",
    ]:
        assert (fold0 / name).exists(), f"fold_0 缺少文件: {name}"


def test_loaded_shapes_match_indices(prepared_dir):
    root = Path(prepared_dir)
    fold0 = root / "fold_0"

    # 加载张量
    train_feats = torch.load(fold0 / "train_features.pt")
    val_feats = torch.load(fold0 / "val_features.pt")
    train_raw = torch.load(fold0 / "train_raw.pt")
    val_raw = torch.load(fold0 / "val_raw.pt")
    train_freq = torch.load(fold0 / "train_freq.pt")
    val_freq = torch.load(fold0 / "val_freq.pt")
    train_idx = torch.load(fold0 / "train_indices.pt")
    val_idx = torch.load(fold0 / "val_indices.pt")
    train_mask = torch.load(fold0 / "train_mask.pt")
    val_mask = torch.load(fold0 / "val_mask.pt")

    # 维度一致性：第一维与索引长度一致
    assert train_feats.shape[0] == train_idx.numel()
    assert val_feats.shape[0] == val_idx.numel()
    assert train_raw.shape[0] == train_idx.numel()
    assert val_raw.shape[0] == val_idx.numel()
    assert train_freq.shape[0] == train_idx.numel()
    assert val_freq.shape[0] == val_idx.numel()
    assert train_mask.shape[:2] == train_raw.shape[:2]
    assert val_mask.shape[:2] == val_raw.shape[:2]

    # 名称与最后一维校验
    feat_names = json.loads((fold0 / "feature_names.json").read_text())
    raw_names = json.loads((fold0 / "raw_channel_names.json").read_text())
    assert train_feats.shape[1] == len(feat_names)
    assert train_raw.shape[2] == len(raw_names)

    # 值有效性
    assert torch.isfinite(train_feats).all()
    assert torch.isfinite(val_feats).all()
    assert torch.isfinite(train_raw).all()
    assert torch.isfinite(val_raw).all()
    assert torch.isfinite(train_freq).all()
    assert torch.isfinite(val_freq).all()

    # labels.pkl 内容校验
    with open(root / "labels.pkl", "rb") as f:
        labels_data = pickle.load(f)
    assert "labels" in labels_data and "label_type" in labels_data
    assert labels_data["label_type"] in ("regression", "classification")
    # labels 行数应不少于任一折的总样本数（因折分按窗口索引构建）
    assert labels_data["labels"].shape[0] >= max(train_idx.numel(), val_idx.numel())