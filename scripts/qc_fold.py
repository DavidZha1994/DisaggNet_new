import os
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


def _load_pt_array(fp: str) -> Optional[np.ndarray]:
    """Load a .pt file as numpy array if it's a Tensor; otherwise return None."""
    try:
        import torch
        if os.path.exists(fp):
            t = torch.load(fp)
            if hasattr(t, 'detach'):
                return t.detach().cpu().numpy()
            if hasattr(t, 'numpy'):
                return t.numpy()
    except Exception:
        pass
    return None


def _load_pt_obj(fp: str):
    """Load a .pt file and return the raw Python object (dict/Tensor/etc)."""
    try:
        import torch
        if os.path.exists(fp):
            return torch.load(fp)
    except Exception:
        return None
    return None


def _load_json(fp: str) -> Optional[Dict]:
    try:
        if os.path.exists(fp):
            with open(fp, 'r') as f:
                return json.load(f)
    except Exception:
        return None
    return None


def _load_cfg_resample_seconds(repo_root: str) -> int:
    # Default fallback
    default_seconds = 5
    cfg_fp = os.path.join(repo_root, 'configs', 'pipeline', 'prep_config.yaml')
    try:
        import yaml
        if os.path.exists(cfg_fp):
            with open(cfg_fp, 'r') as f:
                cfg = yaml.safe_load(f) or {}
            return int(((cfg.get('hipe') or {}).get('resample_seconds', default_seconds)))
    except Exception:
        pass
    return default_seconds


def _contiguous_segments(mask: np.ndarray) -> List[Tuple[int, int, int]]:
    """Return list of (start_idx, end_idx, length) for True runs in boolean mask."""
    segs: List[Tuple[int, int, int]] = []
    if mask.size == 0:
        return segs
    # Ensure boolean
    m = mask.astype(bool)
    n = m.size
    i = 0
    while i < n:
        if m[i]:
            s = i
            while i + 1 < n and m[i + 1]:
                i += 1
            e = i
            segs.append((s, e, e - s + 1))
        i += 1
    return segs


def _aggregate_timeline(start_ts: np.ndarray, L: int, step_seconds: int) -> Tuple[np.ndarray, Dict[int, int]]:
    """Build a global timeline from window starts and return mapping from absolute ts -> global index.
    Returns (timeline_ts, ts_to_idx).
    """
    if start_ts.size == 0:
        return np.array([], dtype=np.int64), {}
    # Inclusive last point across all windows
    first = int(np.min(start_ts))
    last_start = int(np.max(start_ts))
    last_point = last_start + (L - 1) * step_seconds
    total_points = int((last_point - first) // step_seconds) + 1
    timeline_ts = np.arange(first, first + total_points * step_seconds, step_seconds, dtype=np.int64)
    ts_to_idx = {int(t): int(i) for i, t in enumerate(timeline_ts)}
    return timeline_ts, ts_to_idx


def _reconstruct_mask_over_timeline(
    start_ts: np.ndarray,
    win_mask: np.ndarray,
    channel_idx: int,
    L: int,
    step_seconds: int,
    timeline_ts: np.ndarray,
    ts_to_idx: Dict[int, int],
) -> np.ndarray:
    """Aggregate a window-level boolean mask for one channel over the full timeline.
    Uses OR to mark a point as valid if any overlapping window has it valid.
    """
    # win_mask shape: [N, L, C]
    N = win_mask.shape[0]
    global_mask = np.zeros_like(timeline_ts, dtype=np.uint8)
    for i in range(N):
        s_ts = int(start_ts[i])
        base_idx = ts_to_idx.get(s_ts)
        if base_idx is None:
            continue
        # window mask for selected channel
        m = win_mask[i, :, channel_idx].astype(np.uint8)
        # Place into global timeline with OR aggregation
        e_idx = base_idx + L
        if e_idx > global_mask.size:
            e_idx = global_mask.size
            m = m[: max(0, e_idx - base_idx)]
        global_mask[base_idx:e_idx] = np.maximum(global_mask[base_idx:e_idx], m[: e_idx - base_idx])
    return global_mask


def _reconstruct_gapfill_mask_over_timeline(
    start_ts: np.ndarray,
    gap_mask_seq: Optional[np.ndarray],
    gap_channel_idx: int,
    L: int,
    step_seconds: int,
    timeline_ts: np.ndarray,
    ts_to_idx: Dict[int, int],
) -> np.ndarray:
    """Aggregate gap-fill mask (1=filled) for mains across timeline using OR."""
    if gap_mask_seq is None or gap_mask_seq.size == 0:
        return np.zeros_like(timeline_ts, dtype=np.uint8)
    N = gap_mask_seq.shape[0]
    global_filled = np.zeros_like(timeline_ts, dtype=np.uint8)
    for i in range(N):
        s_ts = int(start_ts[i])
        base_idx = ts_to_idx.get(s_ts)
        if base_idx is None:
            continue
        m = gap_mask_seq[i, :, gap_channel_idx].astype(np.uint8)
        e_idx = base_idx + L
        if e_idx > global_filled.size:
            e_idx = global_filled.size
            m = m[: max(0, e_idx - base_idx)]
        global_filled[base_idx:e_idx] = np.maximum(global_filled[base_idx:e_idx], m[: e_idx - base_idx])
    return global_filled

def _reconstruct_coverage_over_timeline(
    start_ts: np.ndarray,
    L: int,
    timeline_ts: np.ndarray,
    ts_to_idx: Dict[int, int],
) -> np.ndarray:
    """将窗口覆盖范围映射到全局时间线，覆盖处记为1。"""
    if start_ts is None or start_ts.size == 0:
        return np.zeros_like(timeline_ts, dtype=np.uint8)
    cov = np.zeros_like(timeline_ts, dtype=np.uint8)
    for s_ts in start_ts:
        base_idx = ts_to_idx.get(int(s_ts))
        if base_idx is None:
            continue
        end_idx = min(base_idx + L, timeline_ts.size)
        cov[base_idx:end_idx] = 1
    return cov

def _reconstruct_device_missing_over_timeline(
    start_ts: np.ndarray,
    Y_seq: np.ndarray,
    device_idx: int,
    L: int,
    step_seconds: int,
    timeline_ts: np.ndarray,
    ts_to_idx: Dict[int, int],
) -> np.ndarray:
    """Build device missing mask (True if NaN) over the timeline using AND across overlaps to detect persistent gaps.
    We mark a timepoint as missing if ALL overlapping windows have NaN at that point (strict gap persistence).
    """
    N = Y_seq.shape[0]
    global_missing = np.zeros_like(timeline_ts, dtype=np.uint8)
    # Track counts of overlaps and counts of NaN occurrences
    overlap_cnt = np.zeros_like(timeline_ts, dtype=np.int32)
    missing_cnt = np.zeros_like(timeline_ts, dtype=np.int32)
    for i in range(N):
        s_ts = int(start_ts[i])
        base_idx = ts_to_idx.get(s_ts)
        if base_idx is None:
            continue
        vec = Y_seq[i, :, device_idx]
        miss = np.isnan(vec).astype(np.uint8)
        e_idx = base_idx + L
        limit = min(e_idx, timeline_ts.size)
        wlen = max(0, limit - base_idx)
        if wlen <= 0:
            continue
        overlap_cnt[base_idx:limit] += 1
        missing_cnt[base_idx:limit] += miss[:wlen]
    # Persistent missing where all overlapping windows are missing
    with np.errstate(divide='ignore', invalid='ignore'):
        persistent_missing = (missing_cnt > 0) & (missing_cnt == overlap_cnt)
    global_missing = persistent_missing.astype(np.uint8)
    return global_missing


def _parse_datetime_series_to_sec(dt_series: pd.Series) -> np.ndarray:
    """Convert a pandas datetime series to seconds since epoch (int64)."""
    try:
        return (pd.to_datetime(dt_series, errors='coerce').astype('int64') // 1_000_000_000).to_numpy(dtype=np.int64)
    except Exception:
        return np.array([], dtype=np.int64)


def _downsample(ts: np.ndarray, y: np.ndarray, factor: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample by taking every k-th sample to reduce plotting cost (default: 12 -> ~1分钟)."""
    if ts.size == 0 or y.size == 0:
        return ts, y
    idx = np.arange(0, ts.size, max(1, int(factor)), dtype=np.int64)
    return ts[idx], y[idx]


def _hmm_onoff(series: np.ndarray) -> np.ndarray:
    """Run HMM-like on/off detector on a 1D series, return binary on/off mask (uint8)."""
    try:
        from src.tools.advanced_onoff_methods import AdvancedOnOffDetector
        det = AdvancedOnOffDetector()
        x = np.nan_to_num(np.asarray(series, dtype=np.float32), nan=0.0)
        st, _info = det.hmm_like_method(x)
        st = np.asarray(st, dtype=np.uint8)
        return st
    except Exception:
        # Fallback: median threshold
        m = float(np.nanmedian(series)) if np.isfinite(np.nanmedian(series)) else 0.0
        return (np.nan_to_num(series, nan=0.0) >= m).astype(np.uint8)


def _plot_channel(ts: np.ndarray, y: np.ndarray, on_mask: np.ndarray, valid_mask: np.ndarray, title: str, save_fp: str) -> None:
    """Plot power series, on/off mask, and missing(valid) mask.
    - Do NOT fill areas.
    - Highlight segments where missing==0 (i.e., valid==0) with thicker line on the power plot.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
    ax_p, ax_on, ax_m = axes
    # Downsample for plotting
    ts_d, y_d = _downsample(ts, y, factor=12)
    _, on_d = _downsample(ts, on_mask, factor=12)
    _, val_d = _downsample(ts, valid_mask, factor=12)
    # Use datetime x-axis
    dt_x = pd.to_datetime(ts_d, unit='s')
    # Power line
    ax_p.plot(dt_x, y_d, color='tab:blue', linewidth=0.8, label='power')
    # Overlay thicker segments where valid==0
    try:
        miss_d = (val_d == 0)
        # Extract contiguous segments where missing
        segs = _contiguous_segments(miss_d.astype(bool))
        for s, e, _l in segs:
            # plot the same y over the segment with thicker line
            sl_ts = dt_x[s:e+1]
            sl_y = y_d[s:e+1]
            ax_p.plot(sl_ts, sl_y, color='black', linewidth=2.0)
    except Exception:
        pass
    ax_p.set_title(title)
    ax_p.set_ylabel('Power')
    ax_p.grid(True, alpha=0.3)
    # On/off mask (step line)
    # Step line for on/off; no fill, crisp edges
    try:
        ax_on.step(dt_x, on_d, where='post', color='tab:green', linewidth=1.0)
    except Exception:
        ax_on.plot(dt_x, on_d, color='tab:green', linewidth=1.0, drawstyle='steps-post')
    ax_on.set_ylabel('ON=1/OFF=0')
    ax_on.grid(True, alpha=0.3)
    # Valid mask (1=valid, 0=missing)
    # Step line for valid/missing; no fill
    try:
        ax_m.step(dt_x, val_d, where='post', color='tab:orange', linewidth=1.0)
    except Exception:
        ax_m.plot(dt_x, val_d, color='tab:orange', linewidth=1.0, drawstyle='steps-post')
    ax_m.set_ylabel('Valid=1/Missing=0')
    ax_m.grid(True, alpha=0.3)
    # X-axis as datetime if seconds
    ax_m.set_xlabel('Time')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_fp), exist_ok=True)
    plt.savefig(save_fp, dpi=160)
    plt.close(fig)


def _cross_correlation_lag(x: np.ndarray, y: np.ndarray, max_lag: int = 50) -> Tuple[int, float]:
    """Estimate alignment lag by cross-correlation around zero lag. Returns (best_lag, corr_value)."""
    # normalize to zero mean unit variance to avoid scale effects
    def _norm(v: np.ndarray) -> np.ndarray:
        v = v.astype(np.float64)
        m = np.nanmean(v)
        s = np.nanstd(v)
        if s <= 1e-12:
            s = 1.0
        return (v - m) / s
    xn = _norm(x)
    yn = _norm(y)
    best_lag = 0
    best_corr = -1.0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a = xn[-lag:]
            b = yn[: xn.size + lag]
        elif lag > 0:
            a = xn[: xn.size - lag]
            b = yn[lag:]
        else:
            a = xn
            b = yn
        if a.size == 0 or b.size == 0:
            continue
        # Handle NaNs by masking
        m = (~np.isnan(a)) & (~np.isnan(b))
        if not np.any(m):
            continue
        corr = float(np.dot(a[m], b[m]) / max(1, m.sum()))
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    return best_lag, best_corr


def _compute_raw_mains_gap_segments(raw_fp: str, ts_col: str, step_seconds: int, overlap_range: Optional[Tuple[int, int]] = None) -> List[Dict]:
    """从原始主端CSV中计算采样缺口段（基于相邻时间戳差）。
    缺口段定义：相邻时间戳差 diff > step_seconds，则缺口区间为 [prev+step, next-step]。
    overlap_range 限定 [min_ts, max_ts] 范围，仅保留与其有交叠的缺口段。
    返回：包含 start_ts, end_ts, points, duration_seconds 的列表。
    """
    if not os.path.exists(raw_fp):
        return []
    try:
        df = pd.read_csv(raw_fp)
        if ts_col not in df.columns:
            # 尝试常见备选（含大小写与模糊匹配）
            cols = list(df.columns)
            lower_map = {c.lower(): c for c in cols}
            cand = None
            for key in ['sensordatetime', 'datetime', 'timestamp', 'ts', 'time', 'date', 'date_time', 'sensor_datetime']:
                if key in lower_map:
                    cand = lower_map[key]
                    break
            if cand is None:
                # 模糊：同时包含 date 和 time 的列优先
                for c in cols:
                    cl = c.lower()
                    if ('date' in cl) and ('time' in cl):
                        cand = c
                        break
            if cand is None:
                # 次选：包含 time 或 date
                for c in cols:
                    cl = c.lower()
                    if ('time' in cl) or ('date' in cl):
                        cand = c
                        break
            ts_col = cand or ts_col or (cols[0] if cols else None)
        # 时间戳与值
        ts_ser = pd.to_datetime(df[ts_col], errors='coerce')
        ts = (ts_ser.astype('int64') // 1_000_000_000).to_numpy()
        vals = df['P_kW'].to_numpy() if 'P_kW' in df.columns else None
        # 过滤无效时间戳
        m_valid_ts = np.isfinite(ts)
        ts = ts[m_valid_ts]
        vals = vals[m_valid_ts] if vals is not None else None
        if ts.size == 0:
            return []
        # 排序（保持对应值同步）
        order = np.argsort(ts)
        ts = ts[order]
        vals = vals[order] if vals is not None else None
        rows: List[Dict] = []
        # 1) 基于采样间隔的缺口
        for i in range(ts.size - 1):
            prev = int(ts[i])
            nxt = int(ts[i + 1])
            diff = int(nxt - prev)
            if diff > step_seconds:
                start_missing = prev + step_seconds
                end_missing = nxt - step_seconds
                points = max(0, int((diff - step_seconds) // step_seconds))
                if overlap_range is not None:
                    lo, hi = int(overlap_range[0]), int(overlap_range[1])
                    if end_missing < lo or start_missing > hi:
                        continue
                rows.append({
                    'start_ts': int(start_missing),
                    'end_ts': int(end_missing),
                    'points': int(points),
                    'duration_seconds': int(points * step_seconds),
                })
        # 2) 基于值NaN的缺口（P_kW列）
        if vals is not None:
            if overlap_range is not None:
                lo, hi = int(overlap_range[0]), int(overlap_range[1])
                m_range = (ts >= lo) & (ts <= hi)
                ts_in = ts[m_range]
                vals_in = vals[m_range]
            else:
                ts_in = ts
                vals_in = vals
            if ts_in.size > 0:
                nan_mask = np.isnan(vals_in)
                segs = _contiguous_segments(nan_mask)
                for s, e, l in segs:
                    start_ts = int(ts_in[s])
                    end_ts = int(ts_in[e])
                    # 合并去重：如果与采样缺口行区间重叠，后续可由上层合并或直接追加
                    rows.append({
                        'start_ts': start_ts,
                        'end_ts': end_ts,
                        'points': int(l),
                        'duration_seconds': int(l * step_seconds),
                    })
        return rows
    except Exception:
        return []


def _match_segments(
    mains_segs: List[Tuple[int, int, int]],
    raw_rows: List[Dict],
    timeline_ts: np.ndarray,
    step_seconds: int,
) -> Tuple[List[Dict], Dict]:
    """将拼接后的主端残余缺口段与原始主端缺口段做对齐匹配。
    匹配准则：区间交叠 >= step_seconds，选最大交叠作为匹配。
    返回：comparison_rows, metrics_summary
    """
    # 转原始段为 (s,e,dur) 列表
    raw_segs = [(int(r['start_ts']), int(r['end_ts']), int(r['duration_seconds'])) for r in raw_rows]
    comp_rows: List[Dict] = []
    matched_raw_idx = set()
    matched_mains_idx = set()
    for i, (s_idx, e_idx, points) in enumerate(mains_segs):
        s_ts = int(timeline_ts[s_idx])
        e_ts = int(timeline_ts[e_idx])
        dur_m = int(points * step_seconds)
        best_j = -1
        best_overlap = 0
        for j, (rs, re, rdur) in enumerate(raw_segs):
            # 交叠秒数
            overlap = min(e_ts, re) - max(s_ts, rs)
            if overlap >= step_seconds and overlap > best_overlap:
                best_overlap = overlap
                best_j = j
        if best_j >= 0:
            rs, re, rdur = raw_segs[best_j]
            matched_raw_idx.add(best_j)
            matched_mains_idx.add(i)
            comp_rows.append({
                'mains_start_ts': s_ts,
                'mains_end_ts': e_ts,
                'mains_duration_seconds': dur_m,
                'raw_start_ts_match': rs,
                'raw_end_ts_match': re,
                'raw_duration_seconds_match': rdur,
                'start_delta_seconds': int(abs(s_ts - rs)),
                'duration_delta_seconds': int(abs(dur_m - rdur)),
                'overlap_seconds': int(best_overlap),
            })
        else:
            comp_rows.append({
                'mains_start_ts': s_ts,
                'mains_end_ts': e_ts,
                'mains_duration_seconds': dur_m,
                'raw_start_ts_match': np.nan,
                'raw_end_ts_match': np.nan,
                'raw_duration_seconds_match': np.nan,
                'start_delta_seconds': np.nan,
                'duration_delta_seconds': np.nan,
                'overlap_seconds': 0,
            })
    # 统计指标
    matched_cnt = len(matched_mains_idx)
    unmatched_mains = len(mains_segs) - matched_cnt
    unmatched_raw = len(raw_segs) - len(matched_raw_idx)
    start_deltas = [r['start_delta_seconds'] for r in comp_rows if not np.isnan(r['start_delta_seconds'])]
    dur_deltas = [r['duration_delta_seconds'] for r in comp_rows if not np.isnan(r['duration_delta_seconds'])]
    summary = {
        'matched_segments_count': int(matched_cnt),
        'unmatched_mains_segments_count': int(unmatched_mains),
        'unmatched_raw_segments_count': int(unmatched_raw),
        'mean_abs_start_delta_seconds': float(np.mean(start_deltas)) if start_deltas else np.nan,
        'mean_abs_duration_delta_seconds': float(np.mean(dur_deltas)) if dur_deltas else np.nan,
    }
    return comp_rows, summary


def qc_fold(fold_dir: str, output_dir: Optional[str] = None, plots: bool = False, raw_mains_fp: Optional[str] = None, ts_col_hint: Optional[str] = None) -> Dict:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    out_dir = output_dir or os.path.join(fold_dir, 'quality', 'qc')
    os.makedirs(out_dir, exist_ok=True)

    # Load per-fold labels metadata (datetime_iso) for window starts; fallback to empty if missing
    train_labels = _load_pt_obj(os.path.join(fold_dir, 'train_labels.pt'))
    val_labels = _load_pt_obj(os.path.join(fold_dir, 'val_labels.pt'))
    def _labels_to_ts(ld):
        if not ld or not isinstance(ld, dict):
            return np.array([], dtype=np.int64)
        md = ld.get('label_metadata') or []
        if not md:
            return np.array([], dtype=np.int64)
        iso_list = [m.get('datetime_iso', '') for m in md if isinstance(m, dict)]
        return _parse_datetime_series_to_sec(pd.Series(iso_list))
    train_ts = _labels_to_ts(train_labels)
    val_ts = _labels_to_ts(val_labels)
    starts_all = np.concatenate([train_ts, val_ts]) if train_ts.size or val_ts.size else np.array([], dtype=np.int64)
    # Load raw channel names
    raw_names = _load_json(os.path.join(fold_dir, 'raw_channel_names.json')) or {}
    raw_cols: List[str] = list(raw_names) if isinstance(raw_names, list) else (
        raw_names.get('raw_channel_names') if isinstance(raw_names, dict) else []
    )
    # Find mains channel index (P_kW preferred)
    mains_idx = 0
    if raw_cols:
        try:
            mains_idx = raw_cols.index('P_kW')
        except ValueError:
            mains_idx = 0

    # Load tensors
    train_raw = _load_pt_array(os.path.join(fold_dir, 'train_raw.pt'))
    val_raw = _load_pt_array(os.path.join(fold_dir, 'val_raw.pt'))
    # 新版不再提供时域掩码与gap掩码；目标序列仍存在
    train_Y = _load_pt_array(os.path.join(fold_dir, 'train_targets_seq.pt'))
    val_Y = _load_pt_array(os.path.join(fold_dir, 'val_targets_seq.pt'))

    # Basic shapes
    L = int(train_raw.shape[1]) if train_raw is not None else int(val_raw.shape[1]) if val_raw is not None else 0
    step_seconds = _load_cfg_resample_seconds(repo_root)

    # Timeline
    timeline_ts, ts_to_idx = _aggregate_timeline(starts_all, L, step_seconds)

    # Aggregate mains validity and filled-gaps over timeline (train+val)
    # 主端有效掩码：从折内原始窗计算（按时间点、按通道），聚合为P_kW通道的 OR 覆盖
    if train_raw is not None or val_raw is not None:
        # channel idx for P_kW
        p_idx = mains_idx
        def _win_valid_mask(raw_seq: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
            if raw_seq is None or raw_seq.size == 0:
                return np.array([], dtype=np.int64), np.zeros_like(timeline_ts, dtype=np.uint8)
            N = raw_seq.shape[0]
            starts_arr = train_ts if raw_seq is train_raw else val_ts
            global_valid = np.zeros_like(timeline_ts, dtype=np.uint8)
            for i in range(N):
                s_ts = int(starts_arr[i]) if i < starts_arr.size else None
                base_idx = ts_to_idx.get(s_ts)
                if base_idx is None:
                    continue
                vec = raw_seq[i, :, p_idx]
                vm = np.isfinite(vec).astype(np.uint8)
                end = min(base_idx + L, timeline_ts.size)
                wlen = max(0, end - base_idx)
                if wlen <= 0:
                    continue
                global_valid[base_idx:end] = np.maximum(global_valid[base_idx:end], vm[:wlen])
            return starts_arr, global_valid
        _, mains_valid_train = _win_valid_mask(train_raw)
        _, mains_valid_val = _win_valid_mask(val_raw)
        mains_valid = np.maximum(mains_valid_train, mains_valid_val)
    else:
        mains_valid = np.ones_like(timeline_ts, dtype=np.uint8)
    # Coverage mask: union of train/val window coverage
    coverage_train = _reconstruct_coverage_over_timeline(train_ts, L, timeline_ts, ts_to_idx)
    coverage_val = _reconstruct_coverage_over_timeline(val_ts, L, timeline_ts, ts_to_idx)
    coverage = np.maximum(coverage_train, coverage_val)
    # Residual gaps on mains: only within coverage, originally missing and not gap-filled
    mains_missing_persistent = (coverage == 1) & (mains_valid == 0)

    # Device residual gaps over timeline
    device_gaps_summary: List[Dict] = []
    device_gaps_segments: List[Dict] = []
    if train_Y is not None or val_Y is not None:
        # Combine Y
        Y_all = train_Y if train_Y is not None else np.empty((0, L, 0), dtype=np.float32)
        Y_all = np.concatenate([Y_all, val_Y], axis=0) if val_Y is not None else Y_all
        dev_cnt = int(Y_all.shape[2]) if Y_all.ndim == 3 else 0
        starts_for_Y = np.concatenate([train_ts[: (train_Y.shape[0] if train_Y is not None else 0)],
                                       val_ts[: (val_Y.shape[0] if val_Y is not None else 0)]])
        for d in range(dev_cnt):
            miss_mask = _reconstruct_device_missing_over_timeline(starts_for_Y, Y_all, d, L, step_seconds, timeline_ts, ts_to_idx)
            segs = _contiguous_segments(miss_mask.astype(bool))
            durations = [int(l * step_seconds) for (_, _, l) in segs]
            device_gaps_segments.extend([
                {
                    'device_idx': int(d),
                    'start_ts': int(timeline_ts[s]),
                    'end_ts': int(timeline_ts[e]),
                    'points': int(l),
                    'duration_seconds': int(l * step_seconds),
                }
                for (s, e, l) in segs
            ])
            device_gaps_summary.append({
                'device_idx': int(d),
                'num_gaps': int(len(segs)),
                'total_gap_seconds': int(np.sum(durations)) if durations else 0,
                'max_gap_seconds': int(np.max(durations)) if durations else 0,
                'median_gap_seconds': int(np.median(durations)) if durations else 0,
            })

    # Mains residual gap segments
    mains_segs = _contiguous_segments(mains_missing_persistent.astype(bool))
    mains_durations = [int(l * step_seconds) for (_, _, l) in mains_segs]

    # Alignment check via cross-correlation: use aggregated mains (P_kW proxy) and sum of devices power if available
    # Build coarse series by averaging first occurrences across overlapping windows for train set
    def _reconstruct_series(starts: np.ndarray, win: np.ndarray, ch_idx: int) -> np.ndarray:
        if starts.size == 0 or win is None:
            return np.full_like(timeline_ts, np.nan, dtype=np.float64)
        series = np.full_like(timeline_ts, np.nan, dtype=np.float64)
        counts = np.zeros_like(timeline_ts, dtype=np.int32)
        for i in range(win.shape[0]):
            s_ts = int(starts[i])
            base_idx = ts_to_idx.get(s_ts)
            if base_idx is None:
                continue
            vec = win[i, :, ch_idx].astype(np.float64)
            end = min(base_idx + L, timeline_ts.size)
            wlen = max(0, end - base_idx)
            if wlen <= 0:
                continue
            # Mean aggregation over overlaps
            sl = slice(base_idx, end)
            # replace NaNs with 0 in aggregation and track counts
            valid = ~np.isnan(vec[:wlen])
            add = np.where(valid, vec[:wlen], 0.0)
            series[sl] = np.where(np.isnan(series[sl]), add, series[sl] + add)
            counts[sl] += valid.astype(np.int32)
        with np.errstate(invalid='ignore'):
            series = np.divide(series, counts, out=np.full_like(series, np.nan, dtype=np.float64), where=counts > 0)
        return series

    mains_series = _reconstruct_series(train_ts, train_raw, mains_idx)
    if val_raw is not None:
        mains_series_val = _reconstruct_series(val_ts, val_raw, mains_idx)
        # merge train/val averages
        with np.errstate(invalid='ignore'):
            merged = np.nanmean(np.vstack([mains_series, mains_series_val]), axis=0)
            mains_series = merged
    # Sum of devices power across device dimension (targets_seq has shape [N, L, K])
    dev_sum_series = None
    if train_Y is not None or val_Y is not None:
        Y_base = train_Y if train_Y is not None else np.empty((0, L, 0), dtype=np.float32)
        Y_base = np.concatenate([Y_base, val_Y], axis=0) if val_Y is not None else Y_base
        dev_sum_series = np.full_like(timeline_ts, np.nan, dtype=np.float64)
        counts = np.zeros_like(timeline_ts, dtype=np.int32)
        # 窗口起始时间与 Y 对应（train 后接 val）
        starts_Y = np.concatenate([
            train_ts[: (train_Y.shape[0] if train_Y is not None else 0)],
            val_ts[: (val_Y.shape[0] if val_Y is not None else 0)]
        ])
        for i in range(Y_base.shape[0]):
            s_ts = int(starts_Y[i]) if i < starts_Y.size else None
            if s_ts is None:
                continue
            base_idx = ts_to_idx.get(s_ts)
            if base_idx is None:
                continue
            vec = Y_base[i, :, :]
            # sum across devices (K) to obtain total target power per timepoint
            dev_p = np.nansum(vec.astype(np.float64), axis=1) if vec.shape[1] > 0 else np.full((L,), np.nan, dtype=np.float64)
            end = min(base_idx + L, timeline_ts.size)
            wlen = max(0, end - base_idx)
            if wlen <= 0:
                continue
            sl = slice(base_idx, end)
            valid = ~np.isnan(dev_p[:wlen])
            add = np.where(valid, dev_p[:wlen], 0.0)
            dev_sum_series[sl] = np.where(np.isnan(dev_sum_series[sl]), add, dev_sum_series[sl] + add)
            counts[sl] += valid.astype(np.int32)
        with np.errstate(invalid='ignore'):
            dev_sum_series = np.divide(dev_sum_series, counts, out=np.full_like(dev_sum_series, np.nan, dtype=np.float64), where=counts > 0)

    best_lag, corr_val = (0, np.nan)
    if dev_sum_series is not None and mains_series is not None:
        # Use indices where both are finite
        idx = (~np.isnan(mains_series)) & (~np.isnan(dev_sum_series))
        if np.any(idx):
            best_lag, corr_val = _cross_correlation_lag(mains_series[idx], dev_sum_series[idx], max_lag=50)

    # Save reports
    # Use safe_to_csv from our IO module if available
    try:
        from src.data_preparation.io import safe_to_csv as _safe_csv
    except Exception:
        def _safe_csv(df: pd.DataFrame, fp: str):
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            df.to_csv(fp, index=False)

    # Summary（基础）
    summary_rows = [
        {'metric': 'timeline_points', 'value': int(timeline_ts.size)},
        {'metric': 'window_length_L', 'value': int(L)},
        {'metric': 'step_seconds', 'value': int(step_seconds)},
        {'metric': 'mains_residual_gaps_count', 'value': int(len(mains_segs))},
        {'metric': 'mains_residual_total_seconds', 'value': int(np.sum(mains_durations)) if mains_durations else 0},
        {'metric': 'mains_residual_max_seconds', 'value': int(np.max(mains_durations)) if mains_durations else 0},
        {'metric': 'alignment_best_lag_points', 'value': int(best_lag)},
        {'metric': 'alignment_best_lag_seconds', 'value': int(best_lag) * int(step_seconds)},
        {'metric': 'alignment_corr_at_best_lag', 'value': float(corr_val) if not np.isnan(corr_val) else np.nan},
        {'metric': 'alignment_verified', 'value': bool((abs(int(best_lag)) <= 2) and (not np.isnan(corr_val)) and (float(corr_val) >= 0.25))},
    ]
    _safe_csv(pd.DataFrame(summary_rows), os.path.join(out_dir, 'fold_qc_summary.csv'))

    # Detailed segments
    mains_seg_rows = [
        {
            'start_ts': int(timeline_ts[s]),
            'end_ts': int(timeline_ts[e]),
            'points': int(l),
            'duration_seconds': int(l * step_seconds),
        }
        for (s, e, l) in mains_segs
    ]
    mains_seg_df = pd.DataFrame(mains_seg_rows)
    _safe_csv(mains_seg_df, os.path.join(out_dir, 'mains_residual_gap_segments.csv'))

    # 原始主端缺口计算与对比
    if raw_mains_fp:
        # 读取配置中的 timestamp_col
        ts_col_cfg = ts_col_hint
        if not ts_col_cfg:
            try:
                import yaml
                cfg_fp = os.path.join(repo_root, 'configs', 'pipeline', 'prep_config.yaml')
                if os.path.exists(cfg_fp):
                    with open(cfg_fp, 'r') as f:
                        cfg = yaml.safe_load(f) or {}
                    ts_col_cfg = (cfg.get('hipe') or {}).get('timestamp_col', 'datetime')
                else:
                    ts_col_cfg = 'datetime'
            except Exception:
                ts_col_cfg = 'datetime'

        timeline_range = (int(timeline_ts[0]) if timeline_ts.size else None, int(timeline_ts[-1]) if timeline_ts.size else None)
        raw_rows = _compute_raw_mains_gap_segments(raw_mains_fp, ts_col_cfg, int(step_seconds), overlap_range=timeline_range if timeline_ts.size else None)
        raw_df = pd.DataFrame(raw_rows)
        _safe_csv(raw_df, os.path.join(out_dir, 'raw_mains_gap_segments_overlapping.csv'))

        comp_rows, comp_summary = _match_segments(mains_segs, raw_rows, timeline_ts, int(step_seconds))
        comp_df = pd.DataFrame(comp_rows)
        _safe_csv(comp_df, os.path.join(out_dir, 'mains_vs_raw_gap_comparison.csv'))
        # 扩展汇总
        for k, v in comp_summary.items():
            summary_rows.append({'metric': k, 'value': v})

    if device_gaps_summary:
        _safe_csv(pd.DataFrame(device_gaps_summary), os.path.join(out_dir, 'device_gap_summary.csv'))
    if device_gaps_segments:
        _safe_csv(pd.DataFrame(device_gaps_segments), os.path.join(out_dir, 'device_gap_segments.csv'))

    # =========================
    # 绘图（基于折内 .pt 文件重建时间序列）：
    # - 主端：P/Q/S/PF 按通道绘制功率 + 开关 + 有效掩码
    # - 设备：每设备 P_kW 绘制功率 + 开关 + 有效掩码
    # =========================
    try:
        plots_dir = os.path.join(out_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        # 主端通道索引映射（仅绘制前四个基础通道）
        main_channel_names = []
        if raw_cols:
            main_channel_names = [c for c in ['P_kW', 'Q_kvar', 'S_kVA', 'PF'] if c in raw_cols]
        # 构造每个主端通道的时间序列、on/off 与 valid 掩码
        for ch in main_channel_names:
            try:
                ch_idx = raw_cols.index(ch)
            except ValueError:
                continue
            # 重建主端通道时间序列（train+val平均）
            y_train = _reconstruct_series(train_ts, train_raw, ch_idx) if train_raw is not None else np.full_like(timeline_ts, np.nan, dtype=np.float64)
            y_val = _reconstruct_series(val_ts, val_raw, ch_idx) if val_raw is not None else np.full_like(timeline_ts, np.nan, dtype=np.float64)
            with np.errstate(invalid='ignore'):
                y_main = np.nanmean(np.vstack([y_train, y_val]), axis=0) if (y_train.size and y_val.size) else (y_train if y_train.size else y_val)
            # 有效掩码（任意窗口在该点有效即记为1）
            def _aggregate_valid(starts_arr, raw_seq):
                if starts_arr.size == 0 or raw_seq is None or raw_seq.size == 0:
                    return np.zeros_like(timeline_ts, dtype=np.uint8)
                N = raw_seq.shape[0]
                vm = np.zeros_like(timeline_ts, dtype=np.uint8)
                for i in range(N):
                    s_ts = int(starts_arr[i]) if i < starts_arr.size else None
                    base_idx = ts_to_idx.get(s_ts)
                    if base_idx is None:
                        continue
                    vec = raw_seq[i, :, ch_idx]
                    m = np.isfinite(vec).astype(np.uint8)
                    end = min(base_idx + L, timeline_ts.size)
                    wlen = max(0, end - base_idx)
                    if wlen <= 0:
                        continue
                    vm[base_idx:end] = np.maximum(vm[base_idx:end], m[:wlen])
                return vm
            v_train = _aggregate_valid(train_ts, train_raw)
            v_val = _aggregate_valid(val_ts, val_raw)
            v_main = np.maximum(v_train, v_val)
            # 开关掩码
            on_main = _hmm_onoff(y_main)
            _plot_channel(timeline_ts, y_main.astype(np.float32), on_main, v_main, title=f"Mains {ch}", save_fp=os.path.join(plots_dir, f"mains_{ch}.png"))

        # 设备：使用处理后保存的掩码与序列（train/val_device_masks.pt / targets_seq.pt）
        dm_train = _load_pt_obj(os.path.join(fold_dir, 'train_device_masks.pt'))
        dm_val = _load_pt_obj(os.path.join(fold_dir, 'val_device_masks.pt'))
        det_obj = _load_pt_obj(os.path.join(fold_dir, 'train_onoff_detected.pt'))
        on_train = dm_train.get('onoff').detach().cpu().numpy() if (isinstance(dm_train, dict) and hasattr(dm_train.get('onoff'), 'detach')) else (dm_train.get('onoff').numpy() if isinstance(dm_train, dict) and hasattr(dm_train.get('onoff'), 'numpy') else None)
        on_val = dm_val.get('onoff').detach().cpu().numpy() if (isinstance(dm_val, dict) and hasattr(dm_val.get('onoff'), 'detach')) else (dm_val.get('onoff').numpy() if isinstance(dm_val, dict) and hasattr(dm_val.get('onoff'), 'numpy') else None)
        v_train = dm_train.get('valid').detach().cpu().numpy() if (isinstance(dm_train, dict) and hasattr(dm_train.get('valid'), 'detach')) else (dm_train.get('valid').numpy() if isinstance(dm_train, dict) and hasattr(dm_train.get('valid'), 'numpy') else None)
        v_val = dm_val.get('valid').detach().cpu().numpy() if (isinstance(dm_val, dict) and hasattr(dm_val.get('valid'), 'detach')) else (dm_val.get('valid').numpy() if isinstance(dm_val, dict) and hasattr(dm_val.get('valid'), 'numpy') else None)
        Y_base = train_Y if train_Y is not None else np.empty((0, L, 0), dtype=np.float32)
        Y_base = np.concatenate([Y_base, val_Y], axis=0) if val_Y is not None else Y_base
        starts_Y = np.concatenate([
            train_ts[: (train_Y.shape[0] if train_Y is not None else 0)],
            val_ts[: (val_Y.shape[0] if val_Y is not None else 0)]
        ])
        dev_cnt = int(Y_base.shape[2]) if Y_base.ndim == 3 else 0

        def _aggregate_over_timeline(starts_arr: np.ndarray, win_mask: np.ndarray, dev_idx: int, agg: str = 'mean') -> np.ndarray:
            """将窗口级掩码聚合到时间线：agg='mean' 取重叠窗口平均，'or' 取并集。"""
            if win_mask is None or win_mask.size == 0:
                return np.zeros_like(timeline_ts, dtype=np.float32)
            N = win_mask.shape[0]
            acc = np.zeros_like(timeline_ts, dtype=np.float32)
            cnt = np.zeros_like(timeline_ts, dtype=np.float32)
            for i in range(N):
                s_ts = int(starts_arr[i]) if i < starts_arr.size else None
                base_idx = ts_to_idx.get(s_ts)
                if base_idx is None:
                    continue
                vec = win_mask[i, :, dev_idx].astype(np.float32)
                end = min(base_idx + L, timeline_ts.size)
                wlen = max(0, end - base_idx)
                if wlen <= 0:
                    continue
                sl = slice(base_idx, end)
                if agg == 'or':
                    acc[sl] = np.maximum(acc[sl], vec[:wlen])
                else:
                    acc[sl] = acc[sl] + vec[:wlen]
                    cnt[sl] = cnt[sl] + 1.0
            if agg == 'or':
                return acc
            with np.errstate(invalid='ignore'):
                return np.divide(acc, cnt, out=np.zeros_like(acc), where=cnt > 0)

        def _reconstruct_series_device(dev_idx: int) -> np.ndarray:
            """从 targets_seq 重建设备功率时间线（重叠平均）。"""
            if Y_base.size == 0 or dev_idx >= dev_cnt:
                return np.full_like(timeline_ts, np.nan, dtype=np.float64)
            series = np.full_like(timeline_ts, np.nan, dtype=np.float64)
            counts = np.zeros_like(timeline_ts, dtype=np.int32)
            for i in range(Y_base.shape[0]):
                s_ts = int(starts_Y[i]) if i < starts_Y.size else None
                base_idx = ts_to_idx.get(s_ts)
                if base_idx is None:
                    continue
                vec = Y_base[i, :, dev_idx].astype(np.float64)
                end = min(base_idx + L, timeline_ts.size)
                wlen = max(0, end - base_idx)
                if wlen <= 0:
                    continue
                sl = slice(base_idx, end)
                val_vec = ~np.isnan(vec[:wlen])
                add = np.where(val_vec, vec[:wlen], 0.0)
                series[sl] = np.where(np.isnan(series[sl]), add, series[sl] + add)
                counts[sl] += val_vec.astype(np.int32)
            with np.errstate(invalid='ignore'):
                series = np.divide(series, counts, out=np.full_like(series, np.nan, dtype=np.float64), where=counts > 0)
            return series

        # 从 unified/merged.csv 提取设备名称映射（列名以 *_P_kW 结尾且不等于主端 P_kW）
        dev_names: List[str] = []
        try:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            merged_csv = os.path.join(repo_root, 'Data', 'prepared', 'unified', 'merged.csv')
            if os.path.exists(merged_csv):
                dfm = pd.read_csv(merged_csv, nrows=1)
                for c in dfm.columns:
                    if c.endswith('_P_kW') and c != 'P_kW':
                        # 设备名在前缀
                        dev_name = c[:-len('_P_kW')]
                        dev_names.append(dev_name)
        except Exception:
            dev_names = []

        # 如果存在独立检测结果，则使用其 onoff/valid 时间线，并同步替换 timeline_ts
        det_on = None
        det_valid = None
        if isinstance(det_obj, dict) and det_obj.get('onoff') is not None:
            try:
                import torch
                raw_on = det_obj.get('onoff')
                raw_val = det_obj.get('valid')
                det_on = raw_on.detach().cpu().numpy() if hasattr(raw_on, 'detach') else (raw_on.numpy() if hasattr(raw_on, 'numpy') else np.asarray(raw_on))
                det_valid = raw_val.detach().cpu().numpy() if hasattr(raw_val, 'detach') else (raw_val.numpy() if hasattr(raw_val, 'numpy') else np.asarray(raw_val))
                timeline_ts = det_obj.get('timeline_sec', timeline_ts)
            except Exception:
                det_on = None
                det_valid = None

        for d in range(dev_cnt):
            # 处理后 on/off 掩码（窗口平均阈值0.5）与 valid（并集）
            if det_on is not None and det_valid is not None and det_on.shape[0] > d:
                on_bin = det_on[d].astype(np.uint8)
                val_bin = det_valid[d].astype(np.uint8)
            else:
                on_timeline_train = _aggregate_over_timeline(train_ts, on_train, d, agg='mean') if on_train is not None else np.zeros_like(timeline_ts, dtype=np.float32)
                on_timeline_val = _aggregate_over_timeline(val_ts, on_val, d, agg='mean') if on_val is not None else np.zeros_like(timeline_ts, dtype=np.float32)
                on_timeline = np.maximum(on_timeline_train, on_timeline_val)
                on_bin = (on_timeline >= 0.5).astype(np.uint8)
                val_timeline_train = _aggregate_over_timeline(train_ts, v_train, d, agg='or') if v_train is not None else np.zeros_like(timeline_ts, dtype=np.float32)
                val_timeline_val = _aggregate_over_timeline(val_ts, v_val, d, agg='or') if v_val is not None else np.zeros_like(timeline_ts, dtype=np.float32)
                val_bin = np.maximum(val_timeline_train, val_timeline_val).astype(np.uint8)
            # 功率序列（处理后窗口重叠平均）
            y_dev = _reconstruct_series_device(d)
            name = dev_names[d] if (dev_names and d < len(dev_names)) else f"Device_{d}"
            safe_name = name.replace(' ', '_')
            _plot_channel(timeline_ts, y_dev.astype(np.float32), on_bin, val_bin, title=f"{name} P_kW", save_fp=os.path.join(plots_dir, f"{safe_name}_P_kW.png"))
    except Exception:
        # 绘图失败不影响 QC 汇总
        pass

    # Optional plots
    if plots:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 4))
            plt.plot(timeline_ts, mains_series, label='mains_P_kW', linewidth=0.8)
            if dev_sum_series is not None:
                plt.plot(timeline_ts, dev_sum_series, label='sum_devices_P_kW', linewidth=0.6)
            # Overlay residual mains gaps as vertical spans
            for (s, e, _) in mains_segs:
                plt.axvspan(timeline_ts[s], timeline_ts[e], color='red', alpha=0.15)
            plt.legend()
            plt.xlabel('timestamp (s)')
            plt.ylabel('power')
            plt.tight_layout()
            fig_fp = os.path.join(out_dir, 'alignment_and_gaps.png')
            plt.savefig(fig_fp, dpi=160)
            plt.close()
        except Exception:
            pass

    result = {r['metric']: r['value'] for r in summary_rows}
    return result


def main():
    parser = argparse.ArgumentParser(description='Quality check for prepared fold .pt outputs')
    parser.add_argument('--fold-dir', type=str, required=True, help='Path to prepared fold directory (e.g., Data/prepared/fold_0)')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to write QC reports (default: fold_dir/quality/qc)')
    parser.add_argument('--plots', action='store_true', help='Whether to generate plots')
    parser.add_argument('--raw-mains', type=str, default=None, help='Path to raw mains CSV for gap comparison')
    parser.add_argument('--ts-col', type=str, default=None, help='Timestamp column name in raw mains CSV (default: from config or datetime)')
    args = parser.parse_args()

    res = qc_fold(args.fold_dir, output_dir=args.output_dir, plots=args.plots, raw_mains_fp=args.raw_mains, ts_col_hint=args.ts_col)
    # Print concise summary to stdout
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()