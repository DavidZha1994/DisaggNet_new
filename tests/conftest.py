import os
import sys
from pathlib import Path

import pytest
import pandas as pd

# Avoid OpenMP runtime duplication aborts on macOS with mixed Python wheels
# (e.g., PyTorch, NumPy, and other libraries linking different libomp).
# This setting is an unsafe workaround but acceptable for unit tests.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Keep thread count small to reduce contention during tests.
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Ensure repository root is in sys.path for absolute imports like `import src.*`
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# 项目内导入（用于运行数据准备流水线）
from src.data_preparation.hipe_pipeline import HIPEDataPreparationPipeline


@pytest.fixture(scope="session")
def prepared_dir(tmp_path_factory):
    """
    会话级准备数据夹：
    - 从仓库 Data/raw 复制并截断一部分CSV到临时 raw 目录（保留主端与少量设备）
    - 生成最小化测试配置（关闭可视化、缩短窗口长度、简化CV）
    - 运行 HIPEDataPreparationPipeline，输出到临时 prepared 目录
    - 返回 prepared 目录路径字符串
    """
    repo_root = Path(REPO_ROOT)
    raw_src = repo_root / "Data" / "raw"
    assert raw_src.exists(), f"原始数据目录不存在: {raw_src}"

    work_root = tmp_path_factory.mktemp("hipe_work")
    raw_dst = work_root / "raw"
    prepared_dst = work_root / "prepared"
    raw_dst.mkdir(parents=True, exist_ok=True)
    prepared_dst.mkdir(parents=True, exist_ok=True)

    # 选择主端与前3个设备CSV，截断前 N 行以加速测试
    mains_csv = None
    device_csvs = []
    for fp in sorted(raw_src.glob("*.csv")):
        name = fp.name
        if mains_csv is None and ("MainTerminal" in name or "main" in name.lower() or "mains" in name.lower()):
            mains_csv = fp
            continue
        device_csvs.append(fp)
    device_csvs = device_csvs[:3]
    assert mains_csv is not None, "未找到主端CSV（包含MainTerminal/main/mains）"
    assert device_csvs, "未找到设备CSV"

    # 截断函数：读取前 N 行并写入目标目录，保留原列
    def _truncate_copy(src_fp: Path, dst_fp: Path, n_rows: int = 5000):
        df = pd.read_csv(src_fp, nrows=n_rows)
        df.to_csv(dst_fp, index=False)

    # 拷贝主端与设备到临时 raw 目录
    _truncate_copy(mains_csv, raw_dst / mains_csv.name)
    for dev in device_csvs:
        _truncate_copy(dev, raw_dst / dev.name)

    # 构造测试配置：关闭可视化、缩短窗口长度、简化CV
    cfg = {
        "data_storage": {"output_directory": str(prepared_dst)},
        "hipe": {
            # 明确指定主端文件名与设备匹配模式，兼容仓库示例数据
            "mains_file": mains_csv.name,
            "device_pattern": "*.csv",
            # 使用默认时间列自动推断；保持命名模式以匹配复制的文件
            "resample_seconds": 5,
            "window_length": 256,
            "step_size": 128,
            "label_mode": "regression",
            "stft": {"n_fft": 128, "hop_length": 64, "win_length": 128, "window": "hann"},
        },
        # 降低窗口有效比例阈值，避免小样本数据导致无窗口
        "masking": {"min_valid_ratio": 0.0},
        "compute": {"n_jobs": 1, "use_polars": False},
        "visualization": {"enable": False},
        "cross_validation": {
            "n_folds": 2,
            "purge_gap_minutes": 1,
            "val_span_days": 0,
            "test_span_days": 0,
            "min_train_days": 0,
            # 关闭段隔离，按时间分割，增加训练集非空概率
            "segment_isolation": False,
            "holdout_test": False,
        },
        "imbalance_handling": {"neg_to_pos_ratio": 2.0},
    }

    # 写入临时配置文件
    cfg_fp = work_root / "prep_config_test.yaml"
    import yaml
    with open(cfg_fp, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    # 运行流水线
    pipeline = HIPEDataPreparationPipeline(config_path=str(cfg_fp))
    _ = pipeline.run_full_pipeline(data_path=str(raw_dst))

    return str(prepared_dst)


@pytest.fixture(scope="session")
def hipe_raw_setup(tmp_path_factory):
    """
    会话级原始数据与配置构建：
    - 从 Data/raw 复制主端与前3个设备，截断以加速
    - 生成管线配置（启用最近邻对齐、保留全部窗口、关闭可视化、简化CV）
    - 返回 (raw_dir, prepared_dir, config_path) 供分步测试使用
    """
    repo_root = Path(REPO_ROOT)
    raw_src = repo_root / "Data" / "raw"
    assert raw_src.exists(), f"原始数据目录不存在: {raw_src}"

    work_root = tmp_path_factory.mktemp("hipe_steps")
    raw_dst = work_root / "raw"
    prepared_dst = work_root / "prepared"
    raw_dst.mkdir(parents=True, exist_ok=True)
    prepared_dst.mkdir(parents=True, exist_ok=True)

    mains_csv = None
    device_csvs = []
    for fp in sorted(raw_src.glob("*.csv")):
        name = fp.name
        if mains_csv is None and ("MainTerminal" in name or "main" in name.lower() or "mains" in name.lower()):
            mains_csv = fp
            continue
        device_csvs.append(fp)
    device_csvs = device_csvs[:3]
    assert mains_csv is not None, "未找到主端CSV（包含MainTerminal/main/mains）"
    assert device_csvs, "未找到设备CSV"

    def _truncate_copy(src_fp: Path, dst_fp: Path, n_rows: int = 5000):
        df = pd.read_csv(src_fp, nrows=n_rows)
        df.to_csv(dst_fp, index=False)

    _truncate_copy(mains_csv, raw_dst / mains_csv.name)
    for dev in device_csvs:
        _truncate_copy(dev, raw_dst / dev.name)

    cfg = {
        "data_storage": {"output_directory": str(prepared_dst)},
        "hipe": {
            "mains_file": mains_csv.name,
            "device_pattern": "*.csv",
            "resample_seconds": 5,
            "window_length": 256,
            "step_size": 128,
            "label_mode": "regression",
            "stft": {"n_fft": 128, "hop_length": 64, "win_length": 128, "window": "hann"},
        },
        "alignment": {"direction": "nearest", "tolerance_seconds": 5},
        "masking": {"keep_all_windows": True},
        "compute": {"n_jobs": 1, "use_polars": False},
        "visualization": {"enable": False},
        "cross_validation": {
            "n_folds": 2,
            "purge_gap_minutes": 1,
            "val_span_days": 0,
            "test_span_days": 0,
            "min_train_days": 0,
            "segment_isolation": False,
            "holdout_test": False,
        },
        "imbalance_handling": {"neg_to_pos_ratio": 2.0},
    }

    cfg_fp = work_root / "prep_config_steps.yaml"
    import yaml
    with open(cfg_fp, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    return str(raw_dst), str(prepared_dst), str(cfg_fp)