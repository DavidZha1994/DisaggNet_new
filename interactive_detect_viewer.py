import os
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.set_page_config(page_title="检测结果交互对照(HMM微调预览)", layout="wide")
PLOT_TEMPLATE = "plotly_white"
COLORWAY = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]


def _load_pt(fp: str):
    import torch
    try:
        return torch.load(fp)
    except Exception:
        return torch.load(fp, weights_only=False)


def _to_numpy(x):
    try:
        import torch
        if hasattr(x, 'detach'):
            return x.detach().cpu().numpy()
        if hasattr(x, 'numpy'):
            return x.numpy()
    except Exception:
        pass
    return np.asarray(x)


# 已移除从 CSV 合并文件加载设备名的逻辑；统一从检测 PT 读取


def _reconstruct_series_from_windows(targets: np.ndarray, starts_iso: List[str], timeline_sec: np.ndarray, L: int) -> np.ndarray:
    ser = pd.Series(list(starts_iso))
    starts_sec = (pd.to_datetime(ser, errors='coerce').astype('int64') // 1_000_000_000).to_numpy(dtype=np.int64)
    N, Lw, K = targets.shape
    assert Lw == L
    start0 = int(timeline_sec[0]) if timeline_sec.size > 0 else 0
    step_seconds = int(np.median(np.diff(timeline_sec))) if timeline_sec.size > 1 else 5
    T = timeline_sec.size
    ts_to_idx = {int(start0 + i * step_seconds): int(i) for i in range(T)}
    series = np.full((K, T), np.nan, dtype=np.float64)
    counts = np.zeros((K, T), dtype=np.int32)
    for i in range(N):
        base = ts_to_idx.get(int(starts_sec[i]))
        if base is None:
            continue
        end = min(base + L, T)
        wlen = end - base
        if wlen <= 0:
            continue
        win = targets[i, :wlen, :]  # [wlen,K]
        val = ~np.isnan(win)
        add = np.where(val, win, 0.0)
        sl = slice(base, end)
        series[:, sl] = np.where(np.isnan(series[:, sl]), add.T, series[:, sl] + add.T)
        counts[:, sl] += val.T.astype(np.int32)
    with np.errstate(invalid='ignore'):
        series = np.divide(series, counts, out=np.full_like(series, np.nan, dtype=np.float64), where=counts > 0)
    return series


def _reconstruct_state_from_windows(onoff_win: np.ndarray, starts_iso: List[str], timeline_sec: np.ndarray, L: int) -> np.ndarray:
    # onoff_win: [N,L,K] uint8/bool; reconstruct to [K,T] via majority over overlaps
    ser = pd.Series(list(starts_iso))
    starts_sec = (pd.to_datetime(ser, errors='coerce').astype('int64') // 1_000_000_000).to_numpy(dtype=np.int64)
    N, Lw, K = onoff_win.shape
    assert Lw == L
    start0 = int(timeline_sec[0]) if timeline_sec.size > 0 else 0
    step_seconds = int(np.median(np.diff(timeline_sec))) if timeline_sec.size > 1 else 5
    T = timeline_sec.size
    ts_to_idx = {int(start0 + i * step_seconds): int(i) for i in range(T)}
    agg = np.zeros((K, T), dtype=np.float64)
    counts = np.zeros((K, T), dtype=np.int32)
    for i in range(N):
        base = ts_to_idx.get(int(starts_sec[i]))
        if base is None:
            continue
        end = min(base + L, T)
        wlen = end - base
        if wlen <= 0:
            continue
        win = onoff_win[i, :wlen, :].astype(np.float64)  # [wlen,K]
        sl = slice(base, end)
        agg[:, sl] += win.T
        counts[:, sl] += 1
    with np.errstate(invalid='ignore'):
        mean = np.divide(agg, counts, out=np.zeros_like(agg), where=counts > 0)
    return (mean >= 0.5).astype(np.uint8)


def _enforce_min_runs(state: np.ndarray, min_on_pts: int, min_off_pts: int) -> np.ndarray:
    if state.size == 0:
        return state
    runs = []
    cur = int(state[0])
    start = 0
    for i in range(1, state.size):
        if int(state[i]) != cur:
            runs.append((start, i - 1, cur))
            start = i
            cur = int(state[i])
    runs.append((start, state.size - 1, cur))
    out = state.copy()
    for a, b, v in runs:
        length = b - a + 1
        need = min_on_pts if v == 1 else min_off_pts
        if length < need:
            if a > 0:
                out[a:b+1] = out[a - 1]
            elif b < (out.size - 1):
                out[a:b+1] = out[b + 1]
    return out

# 已移除所有 HMM 识别和预览逻辑，查看器只读取并展示检测脚本生成的掩码


@st.cache_data(show_spinner=True)
def load_detection_fold(fold_dir: str):
    # 统一读取管线生成的窗口级目标与掩码（train_targets_seq.pt）
    targ_fp = os.path.join(fold_dir, 'train_targets_seq.pt')
    if not os.path.exists(targ_fp):
        raise FileNotFoundError('未找到训练窗口文件，请先运行数据准备管线。')
    obj = _load_pt(targ_fp)
    # 兼容：旧版为张量 (N,L,K)，新版为字典
    if isinstance(obj, dict):
        # 设备名与时间线
        dev_names = obj.get('device_names') or []
        step = int(obj.get('step_seconds', 5))
        timeline = obj.get('timeline_sec')
        timeline = _to_numpy(timeline) if timeline is not None else None
        meta = obj.get('label_metadata') or []
        iso = [m.get('datetime_iso', '') for m in meta]

        # 窗口维度
        tP = obj.get('targets_P'); tQ = obj.get('targets_Q'); tS = obj.get('targets_S')
        onP = obj.get('onoff_P'); vP = obj.get('valid_P')
        onQ = obj.get('onoff_Q'); vQ = obj.get('valid_Q')
        onS = obj.get('onoff_S'); vS = obj.get('valid_S')
        import torch
        def to_np(x):
            if x is None:
                return None
            try:
                return (x.detach().cpu().numpy() if hasattr(x, 'detach') else x)
            except Exception:
                return x
        tP = to_np(tP); tQ = to_np(tQ); tS = to_np(tS)
        onP = to_np(onP); vP = to_np(vP)
        onQ = to_np(onQ); vQ = to_np(vQ)
        onS = to_np(onS); vS = to_np(vS)
        if tP is None:
            raise ValueError('train_targets_seq 缺少 targets_P。')
        N, L, K = tP.shape
        # 重建连续时间序列与掩码
        if timeline is None:
            T = int(N * L)
            timeline = np.arange(0, T * step, step, dtype=np.int64)
        series_P = _reconstruct_series_from_windows(tP, iso, timeline, L)
        series_Q = _reconstruct_series_from_windows(tQ, iso, timeline, L) if isinstance(tQ, np.ndarray) else np.zeros_like(series_P)
        series_S = _reconstruct_series_from_windows(tS, iso, timeline, L) if isinstance(tS, np.ndarray) else np.zeros_like(series_P)
        on_P = _reconstruct_state_from_windows(onP.astype(np.uint8), iso, timeline, L) if isinstance(onP, np.ndarray) else np.zeros_like(series_P, dtype=np.uint8)
        val_P = _reconstruct_state_from_windows(vP.astype(np.uint8), iso, timeline, L) if isinstance(vP, np.ndarray) else np.ones_like(series_P, dtype=np.uint8)
        on_Q = _reconstruct_state_from_windows(onQ.astype(np.uint8), iso, timeline, L) if isinstance(onQ, np.ndarray) else np.zeros_like(series_P, dtype=np.uint8)
        val_Q = _reconstruct_state_from_windows(vQ.astype(np.uint8), iso, timeline, L) if isinstance(vQ, np.ndarray) else np.ones_like(series_P, dtype=np.uint8)
        on_S = _reconstruct_state_from_windows(onS.astype(np.uint8), iso, timeline, L) if isinstance(onS, np.ndarray) else np.zeros_like(series_P, dtype=np.uint8)
        val_S = _reconstruct_state_from_windows(vS.astype(np.uint8), iso, timeline, L) if isinstance(vS, np.ndarray) else np.ones_like(series_P, dtype=np.uint8)

        return {
            'timeline': timeline.astype(np.int64),
            'step_seconds': step,
            'dev_names': dev_names if len(dev_names) == K else [f'Device_{i}' for i in range(K)],
            'series_P': series_P.astype(np.float32),
            'series_Q': series_Q.astype(np.float32),
            'series_S': series_S.astype(np.float32),
            'onoff_P': on_P.astype(np.uint8),
            'valid_P': val_P.astype(np.uint8),
            'onoff_Q': on_Q.astype(np.uint8),
            'valid_Q': val_Q.astype(np.uint8),
            'onoff_S': on_S.astype(np.uint8),
            'valid_S': val_S.astype(np.uint8),
        }
    else:
        # 旧版回退：obj 为张量 (N,L,K)，仅包含 P 通道窗口功率
        arr = _to_numpy(obj)
        if not isinstance(arr, np.ndarray) or arr.ndim != 3:
            raise ValueError("不支持的 train_targets_seq 格式：期望 dict 或 3D 张量")
        N, L, K = arr.shape
        step = 5
        # 设备名尝试从顶层映射加载
        root_dir = os.path.dirname(os.path.dirname(fold_dir))
        dev_map_fp = os.path.join(root_dir, 'device_name_to_id.json')
        dev_names: List[str] = []
        if os.path.exists(dev_map_fp):
            try:
                import json
                with open(dev_map_fp, 'r') as f:
                    mp = json.load(f)
                # 支持 name->id 或 id->name
                if isinstance(mp, dict):
                    if all(isinstance(v, int) for v in mp.values()):
                        # name->id
                        dev_names = [x for x, _ in sorted(mp.items(), key=lambda kv: kv[1])]
                    elif all(isinstance(k, int) for k in mp.keys()):
                        # id->name
                        dev_names = [v for _, v in sorted(mp.items(), key=lambda kv: kv[0])]
            except Exception:
                dev_names = [f'Device_{i}' for i in range(K)]
        if not dev_names:
            dev_names = [f'Device_{i}' for i in range(K)]
        # 时间线与ISO尝试从 train_labels.pt 获取
        labels_fp = os.path.join(fold_dir, 'train_labels.pt')
        iso: List[str] = []
        if os.path.exists(labels_fp):
            try:
                ld = _load_pt(labels_fp)
                meta = ld.get('label_metadata') if isinstance(ld, dict) else []
                iso = [m.get('datetime_iso', '') for m in meta] if meta else []
            except Exception:
                iso = []
        T = int(N * L)
        timeline = np.arange(0, T * step, step, dtype=np.int64)
        series_P = _reconstruct_series_from_windows(arr, iso, timeline, L)
        series_Q = np.zeros_like(series_P)
        series_S = np.zeros_like(series_P)
        on_P = np.zeros_like(series_P, dtype=np.uint8)
        val_P = np.ones_like(series_P, dtype=np.uint8)
        on_Q = np.zeros_like(series_P, dtype=np.uint8)
        val_Q = np.ones_like(series_P, dtype=np.uint8)
        on_S = np.zeros_like(series_P, dtype=np.uint8)
        val_S = np.ones_like(series_P, dtype=np.uint8)
        return {
            'timeline': timeline.astype(np.int64),
            'step_seconds': step,
            'dev_names': dev_names,
            'series_P': series_P.astype(np.float32),
            'series_Q': series_Q.astype(np.float32),
            'series_S': series_S.astype(np.float32),
            'onoff_P': on_P.astype(np.uint8),
            'valid_P': val_P.astype(np.uint8),
            'onoff_Q': on_Q.astype(np.uint8),
            'valid_Q': val_Q.astype(np.uint8),
            'onoff_S': on_S.astype(np.uint8),
            'valid_S': val_S.astype(np.uint8),
        }


st.sidebar.title("交互配置")
default_fold = os.path.join(os.getcwd(), 'Data', 'prepared', 'fold_2')
fold_dir = st.sidebar.text_input("Fold 目录", value=default_fold)

try:
    det_obj = load_detection_fold(fold_dir)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

timeline = det_obj['timeline']
dev_names = det_obj['dev_names']
step_seconds = int(det_obj['step_seconds'])

# 选择通道
channel = st.sidebar.selectbox("选择通道", options=["P_kW", "Q_kvar", "S_kVA"], index=0)
if channel == "P_kW":
    series = det_obj['series_P']
    onoff = det_obj['onoff_P']
    valid = det_obj['valid_P']
elif channel == "Q_kvar":
    series = det_obj['series_Q']
    onoff = det_obj['onoff_Q']
    valid = det_obj['valid_Q']
else:
    series = det_obj['series_S']
    onoff = det_obj['onoff_S']
    valid = det_obj['valid_S']

# 时间选择
dt_all = pd.to_datetime(timeline, unit='s')
min_date = dt_all.min().date()
max_date = dt_all.max().date()
start_date, end_date = st.sidebar.date_input("展示日期范围", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(start_date, tuple):
    start_date, end_date = start_date

# DatetimeIndex 没有 .dt，直接使用 .date 属性
dates = dt_all.date
mask = (dates >= start_date) & (dates <= end_date)
dt = dt_all[mask]

# 设备选择（仅展示已生成的检测掩码）
d_idx = st.sidebar.selectbox("选择设备", options=list(range(len(dev_names))), format_func=lambda i: dev_names[i])

y = series[d_idx][mask]
s_on = onoff[d_idx][mask]
s_val = valid[d_idx][mask]

# 限制点数避免超大页面
MAX_POINTS = 250000
if len(dt) > MAX_POINTS:
    step = int(np.ceil(len(dt) / MAX_POINTS))
    dt = dt[::step]
    y = y[::step]
    s_on = s_on[::step]
    s_val = s_val[::step]

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)
fig.add_trace(go.Scatter(x=dt, y=y, name=f"{dev_names[d_idx]} {channel}", mode="lines"), row=1, col=1)
fig.add_trace(go.Scatter(x=dt, y=s_on.astype(np.float32), name="Detected ON/OFF", mode="lines", line=dict(color="#2ca02c"), line_shape="hv"), row=2, col=1)
fig.add_trace(go.Scatter(x=dt, y=s_val.astype(np.float32), name="Valid", mode="lines", line=dict(color="#ff7f0e"), line_shape="hv"), row=3, col=1)

# 已移除 HMM 预览曲线，统一展示脚本生成的检测掩码

fig.update_layout(template=PLOT_TEMPLATE, colorway=COLORWAY, legend=dict(orientation="h"))
fig.update_xaxes(title_text="timestamp", rangeslider=dict(visible=True), row=3, col=1)
fig.update_yaxes(title_text="Power", row=1, col=1)
fig.update_yaxes(title_text="ON/OFF", row=2, col=1, range=[-0.1, 1.1])
fig.update_yaxes(title_text="Valid", row=3, col=1, range=[-0.1, 1.1])
st.plotly_chart(fig, use_container_width=True)

toggles = int(np.sum(np.diff(s_on) != 0))
tph = float(toggles) / max(1.0, (len(s_on) * step_seconds) / 3600.0)
st.info(f"检测摘要：切换 {toggles} 次，≈ {tph:.3f} 次/小时；步长 {step_seconds}s")