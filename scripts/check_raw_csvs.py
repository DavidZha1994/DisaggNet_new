#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 Data/raw 下原始 CSV 的时间戳与数值列是否可正确解析与重采样。
- 识别时间列：['timestamp','ts_utc','SensorDateTime']（大小写兼容）
- 时间解析策略：优先按秒单位；失败则使用通用解析 utc=True，并移除时区
- 重采样：按 --resample-seconds 聚合 mean(numeric_only=True)
- 输出：每文件的行数、时间范围、NaT计数、数值列数量与重采样长度
"""
import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np

TS_CANDIDATES = ["SensorDateTime"]


def find_ts_column(df: pd.DataFrame) -> str | None:
    # 直接匹配候选
    for cand in TS_CANDIDATES:
        if cand in df.columns:
            return cand
    # 大小写兼容匹配
    lowmap = {str(c).lower(): c for c in df.columns}
    for cand in TS_CANDIDATES:
        k = cand.lower()
        if k in lowmap:
            return lowmap[k]
    return None


def parse_ts_to_datetime(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors='coerce', utc=True)
    try:
        s = s.dt.tz_convert(None)
    except Exception:
        pass
    return s


def check_csv(fp: str, resample_seconds: int) -> dict:
    df = pd.read_csv(fp)
    ts_col = find_ts_column(df)
    info = {"file": os.path.basename(fp)}
    if ts_col is None:
        info.update({"ok": False, "reason": "no_timestamp_column"})
        return info
    df['datetime'] = parse_ts_to_datetime(df[ts_col])
    nat_count = int(df['datetime'].isna().sum())
    df = df[df['datetime'].notna()].copy()
    info.update({
        "rows": int(len(df)),
        "nat": nat_count,
        "first_ts": str(df['datetime'].iloc[0]) if len(df) else None,
        "last_ts": str(df['datetime'].iloc[-1]) if len(df) else None,
    })
    # 数值列识别（排除统一的 datetime 列）
    num_cols = [c for c in df.columns if c != 'datetime' and np.issubdtype(df[c].dtype, np.number)]
    info["num_cols_count"] = len(num_cols)
    # 重采样（仅数值列参与 mean）
    try:
        df_res = df.set_index('datetime').resample(f"{resample_seconds}s").mean(numeric_only=True)
        info["resampled_len"] = int(len(df_res))
        info["ok"] = True
    except Exception as e:
        info["ok"] = False
        info["reason"] = f"resample_error: {e}"
    return info


def main():
    parser = argparse.ArgumentParser(description="检查原始CSV的时间戳与重采样可用性")
    parser.add_argument("--data", default=os.path.join("Data", "raw"), help="原始数据目录")
    parser.add_argument("--resample-seconds", type=int, default=5, help="重采样秒数")
    args = parser.parse_args()

    root = args.data
    if not os.path.isdir(root):
        print(f"✗ 原始数据目录不存在: {root}")
        sys.exit(1)
    csvs = sorted(glob.glob(os.path.join(root, "*.csv")))
    if not csvs:
        print(f"✗ 未找到CSV文件于: {root}")
        sys.exit(1)

    print(f"在 {root} 发现 {len(csvs)} 个CSV，开始检查（重采样={args.resample_seconds}s）...")
    oks, fails = 0, 0
    for fp in csvs:
        info = check_csv(fp, args.resample_seconds)
        status = "✓" if info.get("ok") else "✗"
        print(f"{status} {info['file']}: rows={info.get('rows')}, nat={info.get('nat')}, num_cols={info.get('num_cols_count')}, resampled_len={info.get('resampled_len')}, first={info.get('first_ts')}, last={info.get('last_ts')}{' reason='+info.get('reason','') if not info.get('ok') else ''}")
        if info.get("ok"):
            oks += 1
        else:
            fails += 1
    print(f"完成：成功 {oks}，失败 {fails}")
    sys.exit(0 if fails == 0 else 2)


if __name__ == "__main__":
    main()