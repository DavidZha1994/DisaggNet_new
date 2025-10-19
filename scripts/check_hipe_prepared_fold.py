#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 Data/prepared 下由 HIPEDataPreparationPipeline 生成的数据折结构与形状一致性。
- 检查顶层文件：cv_splits.pkl、labels.pkl、device_name_to_id.json
- 对每个 fold_*：检查 train/val 的 raw/freq/features/indices 是否存在（仅 .pt 文件）
- 校验形状一致：features/raw/freq 的第一维应与对应 indices 长度一致
- 校验 metadata CSV 行数与 indices 一致
"""
import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd

TOP_FILES = ["cv_splits.pkl", "labels.pkl", "device_name_to_id.json"]
FOLD_FILES = [
    "train_raw.pt", "val_raw.pt",
    "train_freq.pt", "val_freq.pt",
    "train_features.pt", "val_features.pt",
    "train_indices.pt", "val_indices.pt",
    "train_metadata.csv", "val_metadata.csv",
    "feature_names.json", "raw_channel_names.json",
]


def load_optional_pt(fp: str):
    import torch
    return torch.load(fp) if os.path.exists(fp) else None


def check_fold(fold_dir: str) -> dict:
    out = {"fold_dir": os.path.basename(fold_dir)}
    # 存在性
    present = {f: os.path.exists(os.path.join(fold_dir, f)) for f in FOLD_FILES}
    out["present"] = present
    # 读取
    train_features = load_optional_pt(os.path.join(fold_dir, "train_features.pt"))
    val_features = load_optional_pt(os.path.join(fold_dir, "val_features.pt"))
    train_raw = load_optional_pt(os.path.join(fold_dir, "train_raw.pt"))
    val_raw = load_optional_pt(os.path.join(fold_dir, "val_raw.pt"))
    train_freq = load_optional_pt(os.path.join(fold_dir, "train_freq.pt"))
    val_freq = load_optional_pt(os.path.join(fold_dir, "val_freq.pt"))
    train_idx = load_optional_pt(os.path.join(fold_dir, "train_indices.pt"))
    val_idx = load_optional_pt(os.path.join(fold_dir, "val_indices.pt"))
    # 统一转 numpy 便于下游检查
    def to_np(x):
        try:
            import torch
            return x.detach().cpu().numpy() if hasattr(x, 'detach') else (x.cpu().numpy() if isinstance(x, torch.Tensor) else x)
        except Exception:
            return x
    train_features_np = None if train_features is None else to_np(train_features)
    val_features_np = None if val_features is None else to_np(val_features)
    train_raw_np = None if train_raw is None else to_np(train_raw)
    val_raw_np = None if val_raw is None else to_np(val_raw)
    train_freq_np = None if train_freq is None else to_np(train_freq)
    val_freq_np = None if val_freq is None else to_np(val_freq)
    train_idx_np = None if train_idx is None else to_np(train_idx)
    val_idx_np = None if val_idx is None else to_np(val_idx)

    out.update({
        "train_features_shape": None if train_features_np is None else tuple(train_features_np.shape),
        "val_features_shape": None if val_features_np is None else tuple(val_features_np.shape),
        "train_raw_shape": None if train_raw_np is None else tuple(train_raw_np.shape),
        "val_raw_shape": None if val_raw_np is None else tuple(val_raw_np.shape),
        "train_freq_shape": None if train_freq_np is None else tuple(train_freq_np.shape),
        "val_freq_shape": None if val_freq_np is None else tuple(val_freq_np.shape),
        "train_indices_len": None if train_idx_np is None else int(len(train_idx_np)),
        "val_indices_len": None if val_idx_np is None else int(len(val_idx_np)),
    })
    # 元数据CSV
    tm_path = os.path.join(fold_dir, "train_metadata.csv")
    vm_path = os.path.join(fold_dir, "val_metadata.csv")
    tm_len = None
    vm_len = None
    if os.path.exists(tm_path):
        try:
            tm_len = len(pd.read_csv(tm_path))
        except Exception:
            tm_len = -1
    if os.path.exists(vm_path):
        try:
            vm_len = len(pd.read_csv(vm_path))
        except Exception:
            vm_len = -1
    out.update({"train_metadata_len": tm_len, "val_metadata_len": vm_len})
    # 一致性检查
    checks = []
    def add_check(name: str, cond: bool):
        checks.append((name, bool(cond)))
    if train_features_np is not None and train_idx_np is not None:
        add_check("train_features_vs_indices", train_features_np.shape[0] == len(train_idx_np))
    if val_features_np is not None and val_idx_np is not None:
        add_check("val_features_vs_indices", val_features_np.shape[0] == len(val_idx_np))
    if train_raw_np is not None and train_idx_np is not None:
        add_check("train_raw_vs_indices", train_raw_np.shape[0] == len(train_idx_np))
    if val_raw_np is not None and val_idx_np is not None:
        add_check("val_raw_vs_indices", val_raw_np.shape[0] == len(val_idx_np))
    if train_freq_np is not None and train_idx_np is not None:
        add_check("train_freq_vs_indices", train_freq_np.shape[0] == len(train_idx_np))
    if val_freq_np is not None and val_idx_np is not None:
        add_check("val_freq_vs_indices", val_freq_np.shape[0] == len(val_idx_np))
    if tm_len is not None and train_idx_np is not None:
        add_check("train_metadata_vs_indices", tm_len == len(train_idx_np))
    if vm_len is not None and val_idx_np is not None:
        add_check("val_metadata_vs_indices", vm_len == len(val_idx_np))
    out["checks"] = checks
    out["ok"] = all(c for _, c in checks) if checks else False
    return out


def main():
    parser = argparse.ArgumentParser(description="验证HIPE管线生成的prepared数据折结构与形状一致性（.pt 文件）")
    parser.add_argument("--prepared", default=os.path.join("Data", "prepared"), help="prepared数据目录")
    args = parser.parse_args()

    root = args.prepared
    if not os.path.isdir(root):
        print(f"✗ prepared目录不存在: {root}")
        sys.exit(1)
    # 顶层文件检查
    top_presence = {f: os.path.exists(os.path.join(root, f)) for f in TOP_FILES}
    for f, ok in top_presence.items():
        print(f"{'✓' if ok else '✗'} {f}")
    # fold 目录
    folds = [d for d in os.listdir(root) if d.startswith("fold_") and os.path.isdir(os.path.join(root, d))]
    if not folds:
        print("✗ 未找到 fold_* 目录")
        sys.exit(2)
    print(f"找到 {len(folds)} 个折：{', '.join(sorted(folds))}")
    fails = 0
    for fold in sorted(folds):
        res = check_fold(os.path.join(root, fold))
        print(f"\n[{res['fold_dir']}]")
        print(f"  train_features: {res['train_features_shape']} | val_features: {res['val_features_shape']}")
        print(f"  train_raw:      {res['train_raw_shape']} | val_raw:      {res['val_raw_shape']}")
        print(f"  train_freq:     {res['train_freq_shape']} | val_freq:     {res['val_freq_shape']}")
        print(f"  indices: train={res['train_indices_len']} val={res['val_indices_len']} | metadata: train={res['train_metadata_len']} val={res['val_metadata_len']}")
        for name, ok in res["checks"]:
            print(f"    {'✓' if ok else '✗'} {name}")
        if not res.get("ok"):
            fails += 1
    print(f"\n完成：通过 {len(folds)-fails}，失败 {fails}")
    sys.exit(0 if fails == 0 else 3)


if __name__ == "__main__":
    main()