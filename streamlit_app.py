import os
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# 项目内导入
from src.data_preparation.hipe_pipeline import HIPEDataPreparationPipeline
from src.data_preparation.cross_validation import WalkForwardCV

# ------------------------------
# 页面与样式
# ------------------------------
st.set_page_config(page_title="HIPE 可视化", layout="wide")
PLOT_TEMPLATE = "plotly_white"
COLORWAY = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

# ------------------------------
# 工具函数
# ------------------------------
@st.cache_data(show_spinner=False)
def load_config(config_path: str) -> Dict:
    import yaml
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

@st.cache_data(show_spinner=True)
def load_merged_df(config_path: str, data_dir: str) -> Tuple[pd.DataFrame, Dict[int, str], HIPEDataPreparationPipeline]:
    """使用管线的内部方法读取主端与设备，重采样并合并。返回合并后的DataFrame与设备label_map。"""
    pipe = HIPEDataPreparationPipeline(config_path)
    mains_fp = pipe._find_mains_file(data_dir)
    dev_fps = pipe._find_device_files(data_dir)
    if mains_fp is None:
        raise FileNotFoundError("未找到主端CSV。请检查配置 hipe.mains_file 或数据目录。")
    if not dev_fps:
        raise FileNotFoundError("未找到设备CSV。请检查配置 hipe.device_pattern 或数据目录。")
    df_main = pipe._read_mains(mains_fp)
    dev_dfs, dev_names = pipe._read_devices(dev_fps)
    df_merged, label_map = pipe._align_and_merge(df_main, dev_dfs, dev_names)
    # 修复微小缺口
    df_merged = pipe._repair_small_gaps(df_merged)
    return df_merged, label_map, pipe

@st.cache_data(show_spinner=False)
def compute_windows_and_features(df_merged: pd.DataFrame, _pipe: HIPEDataPreparationPipeline, min_valid_ratio: float = 0.8):
    """计算窗口起点、有效窗口掩码、辅助特征。返回 starts, L, aux_feats, aux_names, X_seq。"""
    X_full = _pipe._build_mains_features(df_merged)
    # 基于列后缀直接解析设备名（不依赖文件名）
    eff_dev_names = []
    for c in df_merged.columns:
        if c.endswith("_P_kW"):
            eff_dev_names.append(c.rsplit("_", 1)[0])
        elif c.endswith("_P_W"):
            eff_dev_names.append(c.rsplit("_", 1)[0])
    Yp_full = _pipe._build_targets(df_merged, eff_dev_names, kind="P")
    L = int(_pipe.hipe.window_length)
    H = int(_pipe.hipe.step_size)
    starts_all = np.arange(0, max(0, X_full.shape[0] - L + 1), H, dtype=np.int64)
    valid_mask = _pipe._valid_window_mask_by_ratio(Yp_full, starts_all, L, min_ratio=float(min_valid_ratio))
    starts = starts_all[valid_mask]
    X_seq, Yp_seq, _ = _pipe._slide_window(X_full, Yp_full, L=L, H=H, starts_override=starts)
    aux_feats, aux_names = _pipe._aggregate_aux_features(X_seq, df_merged, starts)
    return starts, L, aux_feats, aux_names, X_seq


def compute_missing_segments(df: pd.DataFrame, ts_col: str, resample_seconds: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """按采样周期只标记连续超过一个采样间隔的缺口。返回 [(start_ts, end_ts), ...]。"""
    s = pd.to_datetime(df[ts_col])
    ts = (s.astype("int64") // 1_000_000_000).to_numpy()
    # 主端有效性：只看 P/Q/S/PF 是否存在 NaN
    mains_cols = [c for c in ["P_kW", "Q_kvar", "S_kVA", "PF"] if c in df.columns]
    arr = df[mains_cols].to_numpy(dtype=np.float32) if mains_cols else np.empty((len(df), 0), dtype=np.float32)
    gap_thr = max(1, int(resample_seconds))  # 一个采样周期
    segs = []
    n = len(df)
    cur_missing_start = None
    for i in range(1, n):
        gap = ts[i] - ts[i-1]
        row_nan = np.isnan(arr[i]).any() if arr.size else False
        if gap > gap_thr or row_nan:
            if cur_missing_start is None:
                cur_missing_start = s.iloc[i-1]
        else:
            if cur_missing_start is not None:
                segs.append((cur_missing_start, s.iloc[i]))
                cur_missing_start = None
    if cur_missing_start is not None:
        segs.append((cur_missing_start, s.iloc[-1]))
    return segs


def stft_window(sig: np.ndarray, win_len: int, hop: int, n_fft: int) -> np.ndarray:
    """简单STFT，返回 [frames, F] 幅度谱。NaN以均值填充。"""
    m = np.nanmean(sig)
    if np.isnan(m):
        m = 0.0
    sig = np.where(np.isnan(sig), m, sig).astype(np.float32)
    frames = 1 if len(sig) < win_len else (1 + (len(sig) - win_len) // hop)
    F = (n_fft // 2) + 1
    out = np.empty((frames, F), dtype=np.float32)
    window = np.hanning(win_len).astype(np.float32)
    for t in range(frames):
        start = t * hop
        end = start + win_len
        if end <= len(sig):
            frame = sig[start:end]
        else:
            pad = np.zeros(win_len, dtype=np.float32)
            take = max(0, len(sig) - start)
            if take > 0:
                pad[:take] = sig[start:start+take]
            frame = pad
        spec = np.fft.rfft(frame * window, n=n_fft)
        out[t] = np.abs(spec).astype(np.float32)
    return out

# ------------------------------
# 侧边栏与数据加载
# ------------------------------
st.sidebar.title("可视化配置")
default_config = os.path.join(os.getcwd(), "config/prep_config.yaml")
default_data = os.path.join(os.getcwd(), "data/raw")
config_path = st.sidebar.text_input("配置文件路径", value=default_config)
data_dir = st.sidebar.text_input("原始数据目录", value=default_data)

if not os.path.exists(config_path):
    st.sidebar.error("配置文件不存在")
if not os.path.exists(data_dir):
    st.sidebar.error("原始数据目录不存在")

if st.sidebar.button("加载数据"):
    st.experimental_rerun()

# 强制清理缓存并重新加载原始数据
if st.sidebar.button("强制重新加载原始数据"):
    st.cache_data.clear()
    st.experimental_rerun()

cfg = load_config(config_path)
df_merged, label_map, pipe = load_merged_df(config_path, data_dir)
res_s = int(pipe.hipe.resample_seconds)

# 限制展示时间范围，避免一次性传输超大图像数据
all_ts = pd.to_datetime(df_merged[pipe.hipe.timestamp_col])
min_date = all_ts.min().date()
max_date = all_ts.max().date()
start_date, end_date = st.sidebar.date_input(
    "选择展示日期范围",
    value=(min_date, min(max_date, min_date)),
    min_value=min_date,
    max_value=max_date
)
if isinstance(start_date, tuple):
    # 兼容旧版本返回
    start_date, end_date = start_date
mask_range = (all_ts.dt.date >= start_date) & (all_ts.dt.date <= end_date)
df_view = df_merged.loc[mask_range].reset_index(drop=True)

# 根据选择的日期范围自动选择下采样
span_days = max(1, (end_date - start_date).days + 1)
base_resample_s = int(pipe.hipe.resample_seconds)
# 一天或一周以内：用 5S（若基础采样更粗，则取基础采样秒数）；更长：按天
fine_str = f"{max(5, base_resample_s)}S"
downsample_auto = fine_str if span_days <= 7 else "1D"
# ------------------------------
# 页面布局
# ------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "总表与设备缺失", "窗口划分", "频域样例", "手工特征", "Walk-Forward"
])

# ------------------------------
# Tab1: 总表与设备缺失
# ------------------------------
with tab1:
    st.subheader("总表与所有设备的数据与缺失段")
    dev_cols = [c for c in df_view.columns if c.endswith("_P_W") or c.endswith("_P_kW")]
    show_devices = st.multiselect("选择设备通道", options=dev_cols, default=dev_cols[:min(4, len(dev_cols))])
    # 原始数据模式（不下采样）
    raw_mode = st.checkbox("显示原始数据（不下采样）", value=True)
    ts_col = pipe.hipe.timestamp_col
    mains_candidates = ["P_kW", "P_total_W", "P", "power"]
    mains_col = next((c for c in mains_candidates if c in df_view.columns), None)
    base_cols = [ts_col] + ([mains_col] if mains_col else [])
    df_plot = df_view[base_cols + show_devices].copy()
    df_plot[ts_col] = pd.to_datetime(df_plot[ts_col])
    if not raw_mode:
        downsample = downsample_auto
        st.caption(f"自动下采样: {downsample}（跨度 {span_days} 天，基础采样 {base_resample_s}s）")
        df_plot = df_plot.set_index(ts_col).resample(downsample).mean().reset_index()

    # 进一步限制最大点数，避免消息过大（允许一周5S完整显示）
    MAX_POINTS = 200000
    if len(df_plot) > MAX_POINTS:
        step = int(np.ceil(len(df_plot) / MAX_POINTS))
        df_plot = df_plot.iloc[::step]

    fig = go.Figure()
    fig.update_layout(template=PLOT_TEMPLATE, colorway=COLORWAY, legend=dict(orientation="h"))
    if mains_col:
        fig.add_trace(go.Scatter(x=df_plot[ts_col], y=df_plot.get(mains_col), name=mains_col, mode="lines"))
    for c in show_devices:
        fig.add_trace(go.Scatter(x=df_plot[ts_col], y=df_plot.get(c), name=c, mode="lines", opacity=0.7))

    # 缺失段叠加（主端，按原始数据计算，但只在当前展示范围内绘制）
    miss_all = compute_missing_segments(df_merged, ts_col=ts_col, resample_seconds=res_s)
    for a, b in miss_all:
        if (a.date() > end_date) or (b.date() < start_date):
            continue
        fig.add_vrect(x0=a, x1=b, fillcolor="rgba(255,0,0,0.08)", line_width=0, layer="below")
    fig.update_xaxes(title_text="timestamp")
    fig.update_yaxes(title_text="功率")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Tab2: 窗口划分
# ------------------------------
with tab2:
    st.subheader("完整数据的窗口划分与详情")
    min_ratio = st.slider("最小有效比例(过滤缺失窗口)", 0.0, 1.0, float(cfg.get('masking', {}).get('min_valid_ratio', 0.8)), 0.05)
    starts, L, aux_feats, aux_names, X_seq = compute_windows_and_features(df_merged, pipe, min_valid_ratio=min_ratio)

    # 仅在当前展示范围内绘制主端曲线与窗口起点
    ts_all = pd.to_datetime(df_merged[pipe.hipe.timestamp_col])
    mask_tab2 = (ts_all.dt.date >= start_date) & (ts_all.dt.date <= end_date)
    ts = ts_all[mask_tab2]
    mains_candidates = ["P_kW", "P_total_W", "P", "power"]
    mains_col = next((c for c in mains_candidates if c in df_merged.columns), None)
    pk = df_merged.get(mains_col)[mask_tab2] if mains_col else pd.Series([], dtype=float)

    # 进一步限制最大点数，避免消息过大
    MAX_POINTS2 = 40000
    if len(ts) > MAX_POINTS2:
        step2 = int(np.ceil(len(ts) / MAX_POINTS2))
        ts = ts.iloc[::step2]
        pk = pk.iloc[::step2]

    fig2 = go.Figure()
    fig2.update_layout(template=PLOT_TEMPLATE, colorway=COLORWAY)
    if mains_col:
        fig2.add_trace(go.Scatter(x=ts, y=pk, name=mains_col, mode="lines"))
    # 边界线（每个窗口的起点，中点落在当前范围内才绘制）
    ts_secs = (ts_all.astype("int64") // 1_000_000_000).to_numpy()
    starts_mid = ts_secs[starts] + (res_s * L) // 2
    ts_mid = pd.to_datetime(starts_mid, unit="s")
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    mask_mid = (ts_mid >= start_ts) & (ts_mid <= end_ts)
    fig2.add_trace(go.Scatter(x=ts_mid[mask_mid], y=np.ones(mask_mid.sum()),
                              name="window_mid", mode="markers", marker=dict(size=3, color="#ff7f0e")))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("选中窗口详情")
    if len(starts) > 0:
        win_idx = st.slider("窗口索引", 0, len(starts)-1, 0)
        s = int(starts[win_idx])
        e = s + L
        df_win = df_merged.iloc[s:e]
        figw = go.Figure()
        figw.update_layout(template=PLOT_TEMPLATE, colorway=COLORWAY)
        mains_candidates = ["P_kW", "P_total_W", "P", "power"]
        mains_col = next((c for c in mains_candidates if c in df_win.columns), None)
        if mains_col:
            figw.add_trace(go.Scatter(x=pd.to_datetime(df_win[pipe.hipe.timestamp_col]), y=df_win.get(mains_col), name=mains_col, mode="lines"))
        # 显示所有设备通道（兼容两种后缀）
        show_cols = [c for c in df_win.columns if c.endswith("_P_W") or c.endswith("_P_kW")]
        for c in show_cols:
            figw.add_trace(go.Scatter(x=pd.to_datetime(df_win[pipe.hipe.timestamp_col]), y=df_win.get(c), name=c, mode="lines", opacity=0.7))
        st.plotly_chart(figw, use_container_width=True)
    else:
        st.info("暂无有效窗口。请调整最小有效比例或日期范围。")

# ------------------------------
# Tab3: 频域样例（STFT）
# ------------------------------
with tab3:
    st.subheader("设备开启窗口的频域数据（交互样例）")
    if len(starts) > 0:
        win_idx3 = st.slider("选择窗口索引", 0, len(starts)-1, 0)
        s3 = int(starts[win_idx3])
        e3 = s3 + L
        channel = st.selectbox("选择通道", ["P_kW", "Q_kvar", "S_kVA", "PF"])
        sig = df_merged[channel].iloc[s3:e3].to_numpy(dtype=np.float32)
        n_fft = int(pipe.hipe.stft_n_fft)
        hop = int(pipe.hipe.stft_hop)
        win_len = int(pipe.hipe.stft_win_length)
        spec = stft_window(sig, win_len=win_len, hop=hop, n_fft=n_fft)
        fig3 = px.imshow(spec.T, origin="lower", aspect="auto", color_continuous_scale="Turbo")
        fig3.update_layout(template=PLOT_TEMPLATE)
        fig3.update_xaxes(title_text="frame")
        fig3.update_yaxes(title_text="freq bin")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("暂无窗口用于频域样例。")

# ------------------------------
# Tab4: 手工特征
# ------------------------------
with tab4:
    st.subheader("窗口级手工特征分布")
    feat_name = st.selectbox("选择特征", aux_names if len(aux_names) else ["无特征"])
    if feat_name in aux_names:
        idx = aux_names.index(feat_name)
        vals = aux_feats[:, idx]
        vals = vals[np.isfinite(vals)]
        fig4 = px.histogram(vals, nbins=50)
        fig4.update_layout(template=PLOT_TEMPLATE, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("暂无特征数据")

# ------------------------------
# Tab5: Walk-Forward 折叠
# ------------------------------
with tab5:
    st.subheader("Walk-Forward 数据折叠划分")
    seg_meta = pipe._create_segments_meta(df_merged)
    cv_cfg = pipe._ensure_cv_config()
    cv = WalkForwardCV({"cross_validation": cv_cfg})
    folds = cv.create_folds(seg_meta)
    # 构造窗口元数据用于散点绘制
    ts_secs_full = (pd.to_datetime(df_merged[pipe.hipe.timestamp_col]).astype("int64") // 1_000_000_000).to_numpy()
    # 使用窗口中点作为散点横轴
    starts_mid_full = ts_secs_full[starts] + (res_s * L) // 2
    ts_mid_full = pd.to_datetime(starts_mid_full, unit="s")

    # 仅在当前展示范围内绘制散点
    start_ts2 = pd.Timestamp(start_date)
    end_ts2 = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    mask_mid2 = (ts_mid_full >= start_ts2) & (ts_mid_full <= end_ts2)
    mask_mid2 = np.asarray(mask_mid2, dtype=bool)
    ts_mid = np.asarray(ts_mid_full)[mask_mid2]

    seg_ids = np.zeros(len(starts), dtype=int)
    # 将窗口映射到段
    seg_start = seg_meta["start_ts"].to_numpy()
    seg_end = seg_meta["end_ts"].to_numpy()
    seg_id_vals = seg_meta["segment_id"].to_numpy()
    for i in range(len(starts)):
        m = (starts_mid_full[i] >= seg_start) & (starts_mid_full[i] <= seg_end)
        if np.any(m):
            seg_ids[i] = int(seg_id_vals[np.argmax(m)])

    y = np.zeros(len(starts), dtype=int)
    colors = {0: "lightgray", 1: "steelblue", 2: "darkorange", 3: "darkgreen"}
    for fi, f in enumerate(folds):
        train_seg = set(getattr(f, "train_segments", []) or [])
        val_seg = set(getattr(f, "val_segments", []) or [])
        test_seg = set(getattr(f, "test_segments", []) or [])
        mask_train = np.asarray([sid in train_seg for sid in seg_ids], dtype=bool)[mask_mid2]
        mask_val = np.asarray([sid in val_seg for sid in seg_ids], dtype=bool)[mask_mid2]
        mask_test = np.asarray([sid in test_seg for sid in seg_ids], dtype=bool)[mask_mid2]

        fig5 = go.Figure()
        fig5.update_layout(template=PLOT_TEMPLATE)
        fig5.add_trace(go.Scatter(x=ts_mid[mask_train], y=np.ones(mask_train.sum())*1, mode="markers", name=f"Fold {fi} train", marker=dict(color=colors[1], size=5)))
        fig5.add_trace(go.Scatter(x=ts_mid[mask_val], y=np.ones(mask_val.sum())*2, mode="markers", name=f"Fold {fi} val", marker=dict(color=colors[2], size=5)))
        if len(test_seg) > 0:
            fig5.add_trace(go.Scatter(x=ts_mid[mask_test], y=np.ones(mask_test.sum())*3, mode="markers", name=f"Fold {fi} test", marker=dict(color=colors[3], size=5)))
        fig5.update_yaxes(range=[0,4], tickvals=[1,2,3], ticktext=["train","val","test"])
        fig5.update_xaxes(title_text="timestamp")
        st.plotly_chart(fig5, use_container_width=True)

st.success("页面加载完成。可在侧边栏调整展示范围与下采样等级。")