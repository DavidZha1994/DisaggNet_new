#!/usr/bin/env python3
"""
检查 raw 目录下每一个时刻，总表与各分设备在 P/Q/S 三个通道上的关系：
- 分别输出 P、Q、S 三个 CSV 文件；
- 每个文件的列为：timestamp、mains（总表）、每个设备列（设备名）、sum_devices（分表和）、diff（mains - sum_devices）；
- 默认时间列为 SensorDateTime，可通过 --ts-col 覆盖。

运行示例：
python scripts/check_mains_vs_devices.py --raw-dir data/raw --ts-col SensorDateTime --out-dir reports/debug_manual
"""
import argparse
import os
import glob
import pandas as pd
import numpy as np


def extract_device_name(path: str) -> str:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    if '_PhaseCount_' in stem:
        return stem.split('_PhaseCount_')[0]
    if stem.startswith('device_'):
        return stem[len('device_'):]
    if 'MainTerminal' in stem:
        return 'MainTerminal'
    return stem


def is_mains(path: str) -> bool:
    low = os.path.basename(path).lower()
    return ('mainterminal' in low) or ('main' in low) or ('mains' in low)


def detect_ts_col(cols, preferred: str = 'SensorDateTime') -> str:
    if preferred in cols:
        return preferred
    candidates = [c for c in cols if 'time' in c.lower() or 'date' in c.lower() or 'timestamp' in c.lower()]
    if candidates:
        return candidates[0]
    raise ValueError(f"无法识别时间列，期望 {preferred} 或包含 time/date/timestamp 的列")


def _derive_from_energy(df: pd.DataFrame, energy_col: str) -> pd.Series:
    # 按时间排序以确保差分正确
    df = df.sort_values('timestamp')
    if energy_col not in df.columns:
        return pd.Series(index=df.index, dtype='float64')
    # 计算小时级差分功率：Δ能量(kWh) / Δ时间(小时)
    dt_hours = df['timestamp'].diff().dt.total_seconds().div(3600.0)
    dE = df[energy_col].diff()
    with np.errstate(divide='ignore', invalid='ignore'):
        derived = dE / dt_hours
    # 去除无效/负值（能量不应负增长；重置导致负差分，记为0）
    derived = derived.where(derived >= 0, 0.0)
    return derived


def _ensure_channels(df: pd.DataFrame) -> pd.DataFrame:
    # 为 P/Q/S 提供回退：若列缺失或全为零，尝试用对应能量列差分推断
    # P: E_PP_kWh；Q: E_QP_kvarh；S: E_SP_kVAh
    # 统一 P_kW 为正向消耗（若设备采集为负值则取绝对值）
    # 注意：Q_kvar 可为正/负，保持原符号；S_kVA 应为非负
    #
    # 构建副本避免修改原 df
    df = df.copy()

    # P_kW
    if 'P_kW' not in df.columns or (np.nan_to_num(df['P_kW'].astype(float)).sum() == 0):
        if 'E_PP_kWh' in df.columns:
            derived_p = _derive_from_energy(df, 'E_PP_kWh')
            df['P_kW'] = derived_p
    # 统一为正向消耗
    if 'P_kW' in df.columns:
        df['P_kW'] = df['P_kW'].astype(float).abs()

    # Q_kvar
    if 'Q_kvar' not in df.columns or (np.nan_to_num(df['Q_kvar'].astype(float)).sum() == 0):
        if 'E_QP_kvarh' in df.columns:
            derived_q = _derive_from_energy(df, 'E_QP_kvarh')
            df['Q_kvar'] = derived_q
    if 'Q_kvar' in df.columns:
        df['Q_kvar'] = df['Q_kvar'].astype(float)

    # S_kVA
    if 'S_kVA' not in df.columns or (np.nan_to_num(df['S_kVA'].astype(float)).sum() == 0):
        if 'E_SP_kVAh' in df.columns:
            derived_s = _derive_from_energy(df, 'E_SP_kVAh')
            df['S_kVA'] = derived_s
    if 'S_kVA' in df.columns:
        df['S_kVA'] = df['S_kVA'].astype(float).clip(lower=0)

    cols = ['timestamp'] + [c for c in ['P_kW', 'Q_kvar', 'S_kVA'] if c in df.columns]
    return df[cols]


def read_csv_basic(fp: str, ts_col: str):
    df = pd.read_csv(fp)
    ts_col = detect_ts_col(df.columns, preferred=ts_col)
    # 统一按 UTC 解析并去除时区标记，避免混合时区告警
    dt = pd.to_datetime(df[ts_col], errors='coerce', utc=True)
    df['timestamp'] = dt.dt.tz_localize(None)
    df = df.dropna(subset=['timestamp'])
    df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last')
    # 保留原始数值列用于派生
    if 'E_PP_kWh' in df.columns:
        df['E_PP_kWh'] = pd.to_numeric(df['E_PP_kWh'], errors='coerce')
    if 'E_QP_kvarh' in df.columns:
        df['E_QP_kvarh'] = pd.to_numeric(df['E_QP_kvarh'], errors='coerce')
    if 'E_SP_kVAh' in df.columns:
        df['E_SP_kVAh'] = pd.to_numeric(df['E_SP_kVAh'], errors='coerce')
    for c in ['P_kW', 'Q_kvar', 'S_kVA']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return _ensure_channels(df)


def collect_files(raw_dir: str):
    files = glob.glob(os.path.join(raw_dir, '*.csv'))
    if not files:
        raise FileNotFoundError(f"目录为空：{raw_dir}")
    mains = [f for f in files if is_mains(f)]
    devices = [f for f in files if f not in mains]
    if not mains:
        raise FileNotFoundError('未找到主端CSV（文件名包含 MainTerminal/main/mains）')
    return mains, devices


def build_mains_df(mains_fps, ts_col):
    dfs = [read_csv_basic(fp, ts_col) for fp in mains_fps]
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last')
    return df.set_index('timestamp')


def build_device_series(device_fps, ts_col, channel):
    series = {}
    for fp in device_fps:
        df = read_csv_basic(fp, ts_col)
        if channel not in df.columns:
            continue
        name = extract_device_name(fp)
        s = df.set_index('timestamp')[channel].astype(float)
        # 如果同一设备跨多个文件，拼接后保留最后一个重复时间戳
        if name in series:
            s = pd.concat([series[name], s]).sort_index()
            s = s[~s.index.duplicated(keep='last')]
        series[name] = s
    return series

# 新增：按近邻时间对齐设备数据到主端索引

def align_series_to_index(base_index: pd.DatetimeIndex, s: pd.Series, tolerance_seconds: int = 3) -> pd.Series:
    s = s.sort_index()
    s.name = s.name or 'value'
    left = pd.DataFrame({'timestamp': base_index}).sort_values('timestamp')
    right = s.reset_index().sort_values('timestamp')
    # 强制确保时间戳为 datetime64[ns]，避免对象类型导致合并错误
    left['timestamp'] = pd.to_datetime(left['timestamp'], errors='coerce')
    right['timestamp'] = pd.to_datetime(right['timestamp'], errors='coerce')
    merged = pd.merge_asof(
        left,
        right,
        on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta(seconds=tolerance_seconds),
    )
    return merged[s.name]


def make_check_csv(raw_dir: str, out_dir: str, ts_col: str = 'SensorDateTime'):
    os.makedirs(out_dir, exist_ok=True)
    mains_fps, device_fps = collect_files(raw_dir)
    mains_df = build_mains_df(mains_fps, ts_col)  # index: timestamp
    channels = [
        ('P_kW', 'P_mains_vs_devices.csv'),
        ('Q_kvar', 'Q_mains_vs_devices.csv'),
        ('S_kVA', 'S_mains_vs_devices.csv'),
    ]

    for ch, out_name in channels:
        if ch not in mains_df.columns:
            print(f"警告: 主端缺少 {ch} 列，跳过该通道")
            continue
        dev_series = build_device_series(device_fps, ts_col, ch)
        # 构建组合表
        frame = pd.DataFrame(index=mains_df.index)
        frame['timestamp'] = frame.index
        frame['mains'] = mains_df[ch]
        for name, s in sorted(dev_series.items()):
            s.name = name
            aligned = align_series_to_index(frame.index, s, tolerance_seconds=3)
            frame[name] = aligned.values
        device_cols = [c for c in frame.columns if c not in ['timestamp', 'mains']]
        frame['sum_devices'] = frame[device_cols].sum(axis=1, skipna=True)
        frame['diff'] = frame['mains'] - frame['sum_devices']
        # 保存CSV
        out_fp = os.path.join(out_dir, out_name)
        frame.to_csv(out_fp, index=False)
        # 简要统计打印（避免除以0导致 inf）
        valid = frame[['mains', 'sum_devices', 'diff']].dropna()
        mae = float(np.abs(valid['diff']).mean()) if len(valid) else np.nan
        ratio_series = (valid['sum_devices'] / valid['mains']).where(valid['mains'] != 0)
        ratio = float(ratio_series.mean(skipna=True)) if len(valid) else np.nan
        print(
            f"{ch}: 保存 {out_fp}；设备列数={len(device_cols)}，有效行={len(valid)}，"
            f"MAE差={mae:.6f}，设备/总表平均比={ratio:.6f}"
        )
    print('完成。')


def main():
    ap = argparse.ArgumentParser(description='检查每时刻总表与分设备功率之和的差异，分别输出P/Q/S CSV。')
    ap.add_argument('--raw-dir', type=str, default=os.path.join(os.getcwd(), 'data', 'raw'), help='原始CSV目录')
    ap.add_argument('--ts-col', type=str, default='SensorDateTime', help='时间列名，默认 SensorDateTime')
    ap.add_argument('--out-dir', type=str, default=os.path.join(os.getcwd(), 'reports', 'debug_manual'), help='输出目录')
    args = ap.parse_args()
    make_check_csv(args.raw_dir, args.out_dir, ts_col=args.ts_col)


if __name__ == '__main__':
    main()