import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_preparation.hipe_pipeline import HIPEDataPreparationPipeline


def test_resample_align_merge(hipe_raw_setup):
    raw_dir, prepared_dir, cfg_path = hipe_raw_setup
    pipe = HIPEDataPreparationPipeline(config_path=cfg_path)

    mains_fp = pipe._find_mains_file(raw_dir)
    dev_fps = pipe._find_device_files(raw_dir)
    assert mains_fp and dev_fps, "应能找到主端与设备CSV"

    df_main = pipe._read_mains(mains_fp)
    dev_dfs, dev_names = pipe._read_devices(dev_fps)
    df_merged, label_map = pipe._align_and_merge(df_main, dev_dfs, dev_names)

    ts_col = pipe.hipe.timestamp_col
    # 时间戳单调递增
    assert pd.Series(df_merged[ts_col]).is_monotonic_increasing
    # 重采样步长应以 resample_seconds 为主（允许少量异常间隔）
    diffs = pd.Series(df_merged[ts_col]).diff().dropna().dt.total_seconds().to_numpy()
    assert diffs.size > 0
    # 最常见的步长应等于配置的重采样秒数
    step = int(pipe.hipe.resample_seconds)
    counts = pd.Series(diffs).round().value_counts()
    assert int(counts.index[0]) == step

    # 至少一个设备成功合并，且每个设备的功率列存在且有有效值
    assert len(label_map) > 0
    for i in sorted(label_map.keys()):
        name = label_map[i]
        col = f"{name}_P_kW"
        assert col in df_merged.columns
        assert df_merged[col].notna().sum() > 0


def test_repair_small_gaps_unit():
    # 构造含短缺口与长缺口的序列
    ts = pd.date_range("2020-01-01 00:00:00", periods=12, freq="5s")
    P = np.array([1., 2., np.nan, 4., 5., np.nan, np.nan, 8., np.nan, np.nan, np.nan, 12.], dtype=np.float32)
    df = pd.DataFrame({"timestamp": ts, "P_kW": P})
    pipe = HIPEDataPreparationPipeline(config_path=None)
    # 使用默认配置中的时间列名
    df = df.rename(columns={"timestamp": pipe.hipe.timestamp_col})
    fixed, _ = pipe._repair_small_gaps(df)
    # 单个缺口应被插值
    assert np.isfinite(fixed["P_kW"].iloc[2])
    # 两个连续缺口应被插值
    assert np.isfinite(fixed["P_kW"].iloc[5]) and np.isfinite(fixed["P_kW"].iloc[6])
    # 总体缺口数量应减少（修复短缺口），长缺口行为保持鲁棒（可能部分填充）
    before_nans = np.isnan(P).sum()
    after_nans = np.isnan(fixed["P_kW"].to_numpy()).sum()
    assert after_nans < before_nans


def test_features_targets_and_windowing(hipe_raw_setup):
    raw_dir, prepared_dir, cfg_path = hipe_raw_setup
    pipe = HIPEDataPreparationPipeline(config_path=cfg_path)
    mains_fp = pipe._find_mains_file(raw_dir)
    dev_fps = pipe._find_device_files(raw_dir)
    df_main = pipe._read_mains(mains_fp)
    dev_dfs, dev_names = pipe._read_devices(dev_fps)
    df_merged, label_map = pipe._align_and_merge(df_main, dev_dfs, dev_names)
    df_merged, _ = pipe._repair_small_gaps(df_merged)

    # 特征与目标形状正确
    X_full = pipe._build_mains_features(df_merged)
    assert X_full.ndim == 2 and X_full.shape[1] == 7
    P = df_merged.get("P_kW").to_numpy(dtype=np.float32) if "P_kW" in df_merged.columns else np.full(len(df_merged), np.nan, dtype=np.float32)
    dP = X_full[:, 4]
    if len(P) > 1:
        assert np.isclose(dP[0], 0.0)
        # 允许 NaN 比较：当 P 含 NaN 时跳过对应位置
        mask = np.isfinite(P[1:]) & np.isfinite(P[:-1]) & np.isfinite(dP[1:])
        assert np.allclose(dP[1:][mask], (P[1:] - P[:-1])[mask])

    eff_dev_names = [label_map[i] for i in sorted(label_map.keys())]
    Y_full = pipe._build_targets(df_merged, eff_dev_names, kind="P")
    assert Y_full.shape[0] == X_full.shape[0]
    assert Y_full.shape[1] == len(eff_dev_names)
    if len(eff_dev_names) > 0:
        assert np.isfinite(Y_full).sum() > 0

    # 切窗应符合长度与步长规则（保留所有窗口）
    L = int(pipe.hipe.window_length)
    H = int(pipe.hipe.step_size)
    starts_all = np.arange(0, max(0, X_full.shape[0] - L + 1), H, dtype=np.int64)
    X_seq, Y_seq, starts = pipe._slide_window(X_full, Y_full, L=L, H=H, starts_override=starts_all)
    assert X_seq.shape[0] == len(starts_all)
    assert X_seq.shape[1] == L and X_seq.shape[2] == 7
    assert Y_seq.shape[0] == len(starts_all) and Y_seq.shape[1] == L and Y_seq.shape[2] == len(eff_dev_names)


def test_full_pipeline_outputs_and_final_trainset(hipe_raw_setup):
    raw_dir, prepared_dir, cfg_path = hipe_raw_setup
    pipe = HIPEDataPreparationPipeline(config_path=cfg_path)
    _ = pipe.run_full_pipeline(data_path=raw_dir)

    # 顶层文件存在（精简后不再要求 cv_splits.pkl 与 labels.pkl）
    top_files = ["device_name_to_id.json", "pipeline_results.json"]
    for name in top_files:
        assert os.path.exists(os.path.join(prepared_dir, name))

    # 至少一个折目录存在
    fold0 = os.path.join(prepared_dir, "fold_0")
    assert os.path.isdir(fold0)

    # 加载折数据并校验形状一致性
    data = pipe.load_processed_data(fold_id=0)
    for key in ["train_raw", "val_raw", "train_freq", "val_freq", "train_features", "val_features", "train_indices", "val_indices", "train_mask", "val_mask"]:
        assert data[key] is not None, f"缺少输出: {key}"

    train_raw = data["train_raw"].numpy()
    val_raw = data["val_raw"].numpy()
    train_idx = data["train_indices"].numpy()
    val_idx = data["val_indices"].numpy()
    train_mask = data["train_mask"].numpy()
    # 原始输入应无 NaN（已按窗均值安全填充）
    assert np.isfinite(train_raw).all() and np.isfinite(val_raw).all()
    # 掩码应仅包含 0/1
    assert set(np.unique(train_mask)).issubset({0, 1})
    # 第一维与 indices 一致
    assert train_raw.shape[0] == len(train_idx)
    assert val_raw.shape[0] == len(val_idx)

    # 元数据CSV行数与indices一致
    md_train = pd.read_csv(os.path.join(fold0, "train_metadata.csv"))
    md_val = pd.read_csv(os.path.join(fold0, "val_metadata.csv"))
    assert len(md_train) == len(train_idx)
    assert len(md_val) == len(val_idx)

    # 原始通道名称与工程特征名称存在
    with open(os.path.join(fold0, "raw_channel_names.json"), "r") as f:
        raw_names = json.load(f)
    with open(os.path.join(fold0, "feature_names.json"), "r") as f:
        feat_names = json.load(f)
    assert isinstance(raw_names, list) and len(raw_names) == 7
    assert isinstance(feat_names, list) and len(feat_names) > 0

    # 训练集即最终输出：确认路径与数量
    assert pipe.get_pipeline_summary().get("output_dir") == prepared_dir
    assert pipe.get_pipeline_summary().get("n_windows", 0) >= (len(train_idx) + len(val_idx))
