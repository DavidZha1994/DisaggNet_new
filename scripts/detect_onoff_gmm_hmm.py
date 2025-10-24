#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 Data/raw 读取分设备功率 P，使用 GMM/HMM 的多状态→二值化方法判断开关状态并绘图。
输出每设备的 PNG 图，包含：原始功率曲线、GMM二值化状态、HMM样式二值化状态。

用法示例：
python scripts/detect_onoff_gmm_hmm.py --raw-dir Data/raw --ts-col SensorDateTime --out-dir reports/onoff_gmm_hmm 

说明：
- GMM：对功率进行多高斯拟合（自动BIC选择2-5个成分），通过最大均值间隙划分OFF与ON簇，映射为二值状态，并做短促状态去抖。
- HMM-like：复用 src/tools/advanced_onoff_methods.py 中的 hmm_like 方法，作为简化的HMM近似。
- 时间统一解析为UTC并去除时区。
"""
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# 为了复用项目内的高级方法
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
from src.tools.advanced_onoff_methods import AdvancedOnOffDetector


def is_mains(path: str) -> bool:
    low = os.path.basename(path).lower()
    return ('mainterminal' in low) or ('main' in low) or ('mains' in low) or ('mainmeter' in low)


def detect_ts_col(cols, preferred: str = 'SensorDateTime') -> str:
    if preferred in cols:
        return preferred
    candidates = [c for c in cols if ('time' in str(c).lower()) or ('date' in str(c).lower()) or ('timestamp' in str(c).lower())]
    if candidates:
        return candidates[0]
    raise ValueError(f"无法识别时间列，期望 {preferred} 或包含 time/date/timestamp 的列")


def read_power_series(fp: str, ts_col: str) -> pd.DataFrame:
    df = pd.read_csv(fp)
    ts_col = detect_ts_col(df.columns, preferred=ts_col)
    dt = pd.to_datetime(df[ts_col], errors='coerce', utc=True)
    df['timestamp'] = dt.dt.tz_localize(None)
    df = df.dropna(subset=['timestamp']).sort_values('timestamp')
    # 选择功率列
    power_col = None
    for cand in ['P_kW', 'power', 'P']:
        if cand in df.columns:
            power_col = cand
            break
    if power_col is None:
        # 若不存在，尝试能量差分（不稳定，仅回退）
        if 'E_PP_kWh' in df.columns:
            s = pd.to_numeric(df['E_PP_kWh'], errors='coerce')
            # 差分并按小时→秒缩放（假设间隔近似均匀，作为回退估计）
            dt_sec = df['timestamp'].diff().dt.total_seconds().fillna(method='bfill')
            p = s.diff() * 3600.0 / dt_sec
            df['P_kW'] = p
            power_col = 'P_kW'
        else:
            raise ValueError(f"{os.path.basename(fp)} 未找到功率列 P_kW/power/P")
    df[power_col] = pd.to_numeric(df[power_col], errors='coerce')
    df = df[['timestamp', power_col]].dropna().copy()
    df = df.drop_duplicates(subset=['timestamp'], keep='last')
    df = df.rename(columns={power_col: 'P_kW'})
    return df


def extract_device_name(fp: str) -> str:
    name = os.path.basename(fp)
    stem = os.path.splitext(name)[0]
    # 尝试去掉常见前缀
    for pref in ['cleaned_', 'device_', 'dev_']:
        if stem.startswith(pref):
            stem = stem[len(pref):]
    # 尝试 MainTerminal 模式之前的设备名
    parts = stem.split('_')
    if len(parts) > 0:
        return parts[0]
    return stem


def choose_best_gmm(x: np.ndarray, min_k: int = 2, max_k: int = 5) -> GaussianMixture:
    best_bic = np.inf
    best_model = None
    x_ = x.reshape(-1, 1)
    for k in range(min_k, max_k + 1):
        try:
            m = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
            m.fit(x_)
            bic = m.bic(x_)
            if bic < best_bic:
                best_bic = bic
                best_model = m
        except Exception:
            continue
    if best_model is None:
        # 回退到2成分
        best_model = GaussianMixture(n_components=2, covariance_type='full', random_state=42).fit(x_)
    return best_model


def gmm_multi_to_binary(x: np.ndarray, min_run: int = 5) -> tuple[np.ndarray, dict]:
    """拟合GMM并将多状态映射为二值开关状态。返回 state(0/1) 与信息。"""
    x = np.asarray(x)
    model = choose_best_gmm(x, 2, 5)
    x_ = x.reshape(-1, 1)
    resp = model.predict(x_)
    means = model.means_.flatten()
    # 依据最大相邻均值间隙分割为OFF/ON组
    order = np.argsort(means)
    means_sorted = means[order]
    # 计算相邻间隙
    gaps = np.diff(means_sorted)
    if len(gaps) == 0:
        split_idx = 0
    else:
        split_idx = int(np.argmax(gaps))
    # OFF组为低均值侧，ON组为高均值侧
    off_components = set(order[:split_idx + 1])
    on_components = set(order[split_idx + 1:])
    state = np.array([1 if resp[i] in on_components else 0 for i in range(len(resp))], dtype=int)
    # 去抖：移除短促片段
    state = remove_short_runs(state, min_run=min_run)
    info = {
        'components': int(model.n_components),
        'means': means.tolist(),
        'split_idx': split_idx,
        'on_components': sorted(list(on_components)),
        'off_components': sorted(list(off_components)),
    }
    return state, info


def remove_short_runs(state: np.ndarray, min_run: int = 5) -> np.ndarray:
    s = state.copy()
    if len(s) == 0:
        return s
    # 统计连续段
    start = 0
    while start < len(s):
        end = start
        val = s[start]
        while end + 1 < len(s) and s[end + 1] == val:
            end += 1
        run_len = end - start + 1
        if run_len < min_run:
            # 将短促段合并到前或后，选相邻更长的方向
            left_len = start if start > 0 else 0
            right_len = len(s) - end - 1
            if right_len >= left_len:
                s[start:end + 1] = s[end + 1] if end + 1 < len(s) else val
            else:
                s[start:end + 1] = s[start - 1] if start > 0 else val
        start = end + 1
    return s


def hmm_like_binary(x: np.ndarray, n_states: int = 2, min_run: int = 5) -> tuple[np.ndarray, dict]:
    detector = AdvancedOnOffDetector()
    # 使用项目内的简化HMM方法
    state, info = detector.hmm_like_method(x, n_states=n_states)
    # 去抖
    state = remove_short_runs(state.astype(int), min_run=min_run)
    info['n_states'] = n_states
    return state, info


def plot_device(df: pd.DataFrame, gmm_state: np.ndarray, hmm_state: np.ndarray, device_name: str, out_path: str):
    time = df['timestamp'].values
    x = df['P_kW'].values
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(18, 10), sharex=True)
    # 原始功率
    axes[0].plot(time, x, color='tab:blue', lw=1.0, alpha=0.85)
    axes[0].set_ylabel('P_kW')
    axes[0].set_title(f'{device_name} - Power')
    axes[0].grid(True, alpha=0.3)
    # GMM状态
    ymin, ymax = np.nanmin(x), np.nanmax(x)
    if np.isclose(ymin, ymax):
        ymax = ymin + 1.0
    axes[1].plot(time, x, color='tab:gray', lw=0.8, alpha=0.5, label='P_kW')
    axes[1].step(time, ymin + (ymax - ymin) * gmm_state.astype(float), where='post', color='tab:red', lw=1.5, label='GMM state')
    axes[1].set_ylabel('state(GMM scaled)')
    axes[1].set_title('GMM multi→binary')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper left')
    # HMM状态
    axes[2].plot(time, x, color='tab:gray', lw=0.8, alpha=0.5, label='P_kW')
    axes[2].step(time, ymin + (ymax - ymin) * hmm_state.astype(float), where='post', color='tab:orange', lw=1.5, label='HMM-like state')
    axes[2].set_ylabel('state(HMM scaled)')
    axes[2].set_title('HMM-like multi→binary')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper left')
    axes[-1].set_xlabel('Time')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='GMM/HMM 多状态→二值化开关状态并绘图')
    ap.add_argument('--raw-dir', type=str, default=os.path.join(os.getcwd(), 'Data', 'raw'), help='原始CSV目录')
    ap.add_argument('--ts-col', type=str, default='SensorDateTime', help='时间列名')
    ap.add_argument('--out-dir', type=str, default=os.path.join(os.getcwd(), 'reports', 'onoff_gmm_hmm'), help='输出图目录')
    ap.add_argument('--min-run', type=int, default=5, help='最短连续状态长度用于去抖')
    ap.add_argument('--limit', type=int, default=0, help='最多处理的设备文件数，0表示全部')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.raw_dir, '*.csv')))
    if not files:
        print(f"✗ 未找到CSV文件于: {args.raw_dir}")
        sys.exit(1)

    processed = 0
    for fp in files:
        if is_mains(fp):
            continue
        try:
            df = read_power_series(fp, args.ts_col)
            if len(df) < 10:
                print(f"跳过 {os.path.basename(fp)}：数据太短")
                continue
            x = df['P_kW'].to_numpy()
            # GMM多状态→二值化
            gmm_state, gmm_info = gmm_multi_to_binary(x, min_run=args.min_run)
            # HMM-like二值化
            hmm_state, hmm_info = hmm_like_binary(x, n_states=2, min_run=args.min_run)
            # 绘图
            dev_name = extract_device_name(fp)
            out_fp = os.path.join(args.out_dir, f"{dev_name}_onoff_gmm_hmm.png")
            plot_device(df, gmm_state, hmm_state, dev_name, out_fp)
            print(f"✓ {os.path.basename(fp)} → {out_fp} | GMM: k={gmm_info['components']}, means={np.round(gmm_info['means'], 3)}")
            processed += 1
            if args.limit and processed >= args.limit:
                break
        except Exception as e:
            print(f"✗ 处理 {os.path.basename(fp)} 失败: {e}")
            continue

    print(f"完成：处理设备文件 {processed} 个，输出目录 {args.out_dir}")


if __name__ == '__main__':
    main()