import os
import glob
from typing import List, Optional, Any, Dict
import pandas as pd


def safe_to_csv(df: pd.DataFrame, fp: str) -> None:
    """轻量安全写CSV，失败不抛出以保证流水线继续。"""
    try:
        df.to_csv(fp, index=False)
    except Exception:
        pass


def extract_device_name(fp: str) -> str:
    base = os.path.basename(fp)
    stem = os.path.splitext(base)[0]
    if "_PhaseCount_" in stem:
        return stem.split("_PhaseCount_")[0]
    if stem.startswith("device_"):
        return stem[len("device_"):]
    return stem


def find_mains_file(data_path: str, hipe: Any) -> Optional[str]:
    """根据配置与常见命名模式寻找主端CSV文件。"""
    cand = os.path.join(data_path, getattr(hipe, "mains_file", "main.csv"))
    if os.path.exists(cand):
        return cand
    patterns = [
        "*main*.csv",
        "*mains*.csv",
        "*Main*.csv",
        "*MainTerminal*.csv",
        "*Main*PhaseCount*.csv",
    ]
    for p in patterns:
        hits = glob.glob(os.path.join(data_path, p))
        if hits:
            hits = sorted(hits, key=lambda x: (len(os.path.basename(x)), x))
            return hits[0]
    return None


def find_device_files(data_path: str, hipe: Any) -> List[str]:
    """根据模式收集设备CSV文件，并排除主端文件。"""
    pattern = os.path.join(data_path, getattr(hipe, "device_pattern", "device_*.csv"))
    files = sorted(glob.glob(pattern))
    if not files:
        fallback_patterns = [
            "*_PhaseCount_*.csv",
            "*PhaseCount*.csv",
            "*.csv",
        ]
        hits: List[str] = []
        for p in fallback_patterns:
            hits.extend(glob.glob(os.path.join(data_path, p)))
        files = sorted(set(hits))
    mains_fp = find_mains_file(data_path, hipe)
    if mains_fp:
        files = [fp for fp in files if os.path.abspath(fp) != os.path.abspath(mains_fp)]
        files = [fp for fp in files if ("Main" not in os.path.basename(fp))]
    return files


def read_table(fp: str, hipe: Any, rename_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """统一读取主表/设备表：标准化时间戳列并按映射重命名，保留所有列。

    - 自动检测时间戳列（默认 `hipe.timestamp_col`，回退到包含 "time" 的列名）
    - 将时间戳解析为 UTC 再移除时区，删除无效行并按时间排序
    - 若提供 `rename_map`（或列存在），执行 {规范名: 原始名} 的重命名
    - 统一时间戳列名为 `hipe.timestamp_col`
    """
    df = pd.read_csv(fp)
    ts_col_conf = getattr(hipe, "timestamp_col", "timestamp")
    ts_col = ts_col_conf
    if ts_col not in df.columns:
        time_cols = [c for c in df.columns if "time" in c.lower()]
        if time_cols:
            ts_col = time_cols[0]
        else:
            raise ValueError(f"CSV缺少时间戳列: {ts_col_conf}")
    dt = pd.to_datetime(df[ts_col], errors='coerce', utc=True)
    df[ts_col] = dt.dt.tz_localize(None)
    df = df.dropna(subset=[ts_col]).copy()
    df = df.sort_values(ts_col)
    if rename_map:
        ren = {src: canon for canon, src in rename_map.items() if isinstance(src, str) and src in df.columns}
        if ren:
            df = df.rename(columns=ren)
    df = df.rename(columns={ts_col: ts_col_conf})
    return df


# 统一读取逻辑已由 read_table 提供；不再区分 mains/devices 包装函数。
