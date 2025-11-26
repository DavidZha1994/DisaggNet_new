#!/usr/bin/env python3
import os
import json
import numpy as np
import torch
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)

st.set_page_config(page_title="REFIT/UKDALE Pipeline Viewer", layout="wide")


# 读取训练配置中的可视化保存目录，避免与训练输出路径不一致
def _get_viz_save_dir(project_root: str, default: str = "outputs/viz") -> str:
    try:
        import yaml
        cfg_path = os.path.join(project_root, "configs", "training", "optimized_stable.yaml")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        viz = ((cfg.get("training") or {}).get("visualization") or {})
        sd = viz.get("save_dir")
        if isinstance(sd, str) and sd.strip():
            return sd.strip()
    except Exception:
        pass
    return default


def load_fold(dir_path: str):
    fold_dir = dir_path
    data = {}
    # 必要文件
    data["train_raw"] = torch.load(os.path.join(fold_dir, "train_raw.pt"))
    data["val_raw"] = torch.load(os.path.join(fold_dir, "val_raw.pt"))
    data["train_freq"] = torch.load(os.path.join(fold_dir, "train_freq.pt"))
    data["val_freq"] = torch.load(os.path.join(fold_dir, "val_freq.pt"))
    data["train_features"] = torch.load(os.path.join(fold_dir, "train_features.pt"))
    data["val_features"] = torch.load(os.path.join(fold_dir, "val_features.pt"))
    data["train_targets_seq"] = torch.load(os.path.join(fold_dir, "train_targets_seq.pt"))
    data["val_targets_seq"] = torch.load(os.path.join(fold_dir, "val_targets_seq.pt"))
    # 可选文件
    feat_names_fp = os.path.join(fold_dir, "feature_names.json")
    raw_names_fp = os.path.join(fold_dir, "raw_channel_names.json")
    data["feature_names"] = json.load(open(feat_names_fp)) if os.path.exists(feat_names_fp) else []
    data["raw_channel_names"] = json.load(open(raw_names_fp)) if os.path.exists(raw_names_fp) else []
    return data


def normalize_image(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if not np.isfinite(x).any():
        return np.zeros_like(x, dtype=np.float32)
    m = np.nanmin(x)
    M = np.nanmax(x)
    if M <= m:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - m) / (M - m)
    y = np.clip(y, 0.0, 1.0)
    return y


st.title("REFIT/UKDALE 数据准备结果查看")

# 选择数据集子目录
base_prepared = os.path.join(PROJECT_ROOT, "Data", "prepared")
subdirs = [d for d in os.listdir(base_prepared) if os.path.isdir(os.path.join(base_prepared, d))]
dataset_choice = st.selectbox("选择数据集", options=subdirs, index=(subdirs.index("ukdale") if "ukdale" in subdirs else (subdirs.index("refit") if "refit" in subdirs else 0)))

# 选择折目录（自动列出该数据集下的 fold_*）
dataset_dir = os.path.join(base_prepared, dataset_choice)
fold_options = sorted([d for d in os.listdir(dataset_dir) if d.startswith("fold_") and os.path.isdir(os.path.join(dataset_dir, d))])
fold_choice = st.selectbox("选择折", options=(fold_options if fold_options else ["fold_0"]))
fold_dir = os.path.join(dataset_dir, fold_choice)

use_val = st.checkbox("查看验证集", value=False)
if os.path.isdir(fold_dir):
    data = load_fold(fold_dir)
    # 选择样本
    n_samples = data["val_raw"].size(0) if use_val else data["train_raw"].size(0)
    idx = st.slider("样本索引", min_value=0, max_value=max(0, n_samples - 1), value=0)

    # 切换 split
    raw_tensor = data["val_raw"] if use_val else data["train_raw"]
    raw = raw_tensor[idx].detach().cpu().numpy()
    freq_obj = data["val_freq"] if use_val else data["train_freq"]
    if isinstance(freq_obj, dict):
        frames = freq_obj.get("frames")
        conf = freq_obj.get("confidence")
        freq = frames[idx].detach().cpu().numpy()
        confidence = float(conf[idx].detach().cpu().numpy()) if conf is not None else None
    else:
        freq = freq_obj[idx].detach().cpu().numpy()
        confidence = None
    aux = data["val_features"][idx].detach().cpu().numpy() if use_val else data["train_features"][idx].detach().cpu().numpy()
    ts_obj = data["val_targets_seq"] if use_val else data["train_targets_seq"]
    if isinstance(ts_obj, dict):
        target_seq = ts_obj.get("seq")[idx].detach().cpu().numpy()
        status_seq = ts_obj.get("status")[idx].detach().cpu().numpy() if ts_obj.get("status") is not None else None
    else:
        target_seq = ts_obj[idx].detach().cpu().numpy()
        status_seq = None

    # 概览
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("窗口长度", raw.shape[0])
    with c2:
        st.metric("时域通道数", raw.shape[1])
    with c3:
        st.metric("设备数(K)", target_seq.shape[-1])

    # 读取采样率与时间轴
    try:
        import yaml
        with open(os.path.join(PROJECT_ROOT, "configs", "expes.yaml"), "r", encoding="utf-8") as f:
            expes_cfg = yaml.safe_load(f) or {}
        sampling_rate = str(expes_cfg.get("sampling_rate", "1min"))
    except Exception:
        sampling_rate = "1min"
    # 读取起始时间（ISO）
    labels_obj = torch.load(os.path.join(fold_dir, "val_labels.pt" if use_val else "train_labels.pt"))
    meta_list = labels_obj.get("label_metadata", [])
    start_iso = meta_list[idx].get("datetime_iso", None) if isinstance(meta_list, list) and idx < len(meta_list) else None
    if start_iso is None:
        start_iso = meta_list[idx].get("timestamp", None) if isinstance(meta_list, list) and idx < len(meta_list) else None
    # 构建时间轴
    try:
        import pandas as pd
        t_index = pd.date_range(start=pd.to_datetime(start_iso), periods=raw.shape[0], freq=sampling_rate)
    except Exception:
        t_index = None

    st.subheader("时域窗口（P_W）")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    if t_index is not None:
        ax.plot(t_index, raw[:, 0], label="P_W")
        ax.set_xlabel("时间")
    else:
        ax.plot(raw[:, 0], label="P_W")
        ax.set_xlabel("样本")
    ax.set_ylabel("功率 (W)")
    ax.legend()
    # 缺失掩码叠加显示（来自 raw 的最后一通道）
    try:
        if raw.shape[1] >= 2:
            miss = raw[:, -1].astype(np.uint8)
            if t_index is not None:
                for i, m in enumerate(miss):
                    if m:
                        ax.axvspan(t_index[i], t_index[i], color='red', alpha=0.15)
            else:
                for i, m in enumerate(miss):
                    if m:
                        ax.axvspan(i, i, color='red', alpha=0.15)
    except Exception:
        pass
    st.pyplot(fig)

    st.subheader("频域帧（归一化显示）")
    # 频域帧为 [T_frames, F_bins*C_eff]，重塑为 [T_frames, F_bins]
    spec = freq
    try:
        c_eff = len(data.get("raw_channel_names", [])) or 1
        f_bins = spec.shape[-1] // c_eff
        spec_2d = spec[:, :f_bins] if c_eff == 1 else spec[:, :f_bins]  # 默认展示第0通道
    except Exception:
        spec_2d = spec
    img = normalize_image(spec_2d)
    import matplotlib.pyplot as plt
    fig_s, ax_s = plt.subplots(1, 1, figsize=(10, 3))
    ax_s.imshow(img.T, aspect='auto', origin='lower')
    ax_s.set_xlabel("帧")
    ax_s.set_ylabel("频率 Bin")
    st.pyplot(fig_s)

    st.subheader("手工特征（时间外生变量）")
    fn = data.get("feature_names", []) or []
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 3))
    ax2.bar(np.arange(len(aux)), aux)
    ax2.set_xticks(np.arange(len(aux)))
    if fn and len(fn) == len(aux):
        ax2.set_xticklabels(fn, rotation=45, ha='right')
    ax2.set_ylabel("特征值")
    ax2.set_xlabel("特征名称")
    st.pyplot(fig2)

    st.subheader("目标序列（每设备功率，单位 W）")
    # 加载设备名称映射
    dev_map_fp = os.path.join(base_prepared, dataset_choice, "device_name_to_id.json")
    dev_names = []
    if os.path.exists(dev_map_fp):
        try:
            dev_map = json.load(open(dev_map_fp))
            inv = {v: k for k, v in dev_map.items()}
            dev_names = [inv.get(i, f"dev_{i}") for i in range(target_seq.shape[-1])]
        except Exception:
            dev_names = [f"dev_{i}" for i in range(target_seq.shape[-1])]
    else:
        dev_names = [f"dev_{i}" for i in range(target_seq.shape[-1])]

    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 3))
    if t_index is not None:
        for k_i in range(target_seq.shape[-1]):
            ax3.plot(t_index, target_seq[:, k_i], label=(dev_names[k_i] if dev_names else f"dev_{k_i}"))
        ax3.set_xlabel("时间")
    else:
        for k_i in range(target_seq.shape[-1]):
            ax3.plot(target_seq[:, k_i], label=(dev_names[k_i] if dev_names else f"dev_{k_i}"))
        ax3.set_xlabel("样本")
    ax3.set_ylabel("功率 (W)")
    ax3.legend(ncol=2)
    st.pyplot(fig3)

    # 开关掩码显示（来自 targets_seq 的字典 'status'）
    st.subheader("开关掩码（选择设备）")
    if status_seq is None:
        st.info("未在 targets_seq 中找到 'status' 掩码（字典形式）")
    else:
        dev_choice = st.selectbox("设备选择（显示其开关掩码）", options=list(range(status_seq.shape[-1])), format_func=(lambda i: dev_names[i] if i < len(dev_names) else f"dev_{i}"))
        fig4, ax4 = plt.subplots(1, 1, figsize=(10, 2.5))
        if t_index is not None:
            ax4.step(t_index, status_seq[:, dev_choice], where='post', label=f"{dev_names[dev_choice]} 开关")
            ax4.set_xlabel("时间")
        else:
            ax4.step(np.arange(status_seq.shape[0]), status_seq[:, dev_choice], where='post', label=f"{dev_names[dev_choice]} 开关")
            ax4.set_xlabel("样本")
        ax4.set_yticks([0, 1])
        ax4.set_yticklabels(["OFF", "ON"])
        ax4.set_ylabel("状态")
        ax4.legend()
        st.pyplot(fig4)
else:
    st.warning(f"当前数据集未找到有效折目录，例如 {os.path.join('Data','prepared',dataset_choice,'fold_0')}")

# ===== 验证交互可视化（HTML嵌入） =====
st.header("验证交互可视化（拼接结果，Plotly HTML）")

# 优先使用训练配置中设定的保存目录（默认 outputs/viz），并兼容旧版 outputs/
viz_save_dir = _get_viz_save_dir(PROJECT_ROOT, default="outputs/viz")
candidates = [
    os.path.join(PROJECT_ROOT, viz_save_dir, "val_interactive", dataset_choice),
    os.path.join(PROJECT_ROOT, viz_save_dir, "val_interactive", dataset_choice.upper()),
    os.path.join(PROJECT_ROOT, "outputs", "val_interactive", dataset_choice),
    os.path.join(PROJECT_ROOT, "outputs", "val_interactive", dataset_choice.upper()),
]
base_html_root = next((p for p in candidates if os.path.isdir(p)), candidates[0])
try:
    st.caption(f"HTML根目录: {os.path.relpath(base_html_root, PROJECT_ROOT)}")
except Exception:
    st.caption(f"HTML根目录: {base_html_root}")
fold_html_dir = os.path.join(base_html_root, fold_choice)
if os.path.isdir(fold_html_dir):
    # rank 选择
    rank_dirs = [d for d in os.listdir(fold_html_dir) if d.startswith("rank_") and os.path.isdir(os.path.join(fold_html_dir, d))]
    rank_choice = st.selectbox("选择进程/Rank", options=(rank_dirs if rank_dirs else ["rank_0"]))
    rank_dir = os.path.join(fold_html_dir, rank_choice)
    html_files = []
    if os.path.isdir(rank_dir):
        html_files = sorted([f for f in os.listdir(rank_dir) if f.endswith(".html")])
    else:
        html_files = sorted([f for f in os.listdir(fold_html_dir) if f.endswith(".html")])
    if html_files:
        html_choice = st.selectbox("选择Epoch图", options=html_files, index=len(html_files) - 1)
        html_path = os.path.join(rank_dir if os.path.isdir(rank_dir) else fold_html_dir, html_choice)
        try:
            import streamlit.components.v1 as components
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            components.html(html_content, height=900, scrolling=True)
        except Exception as e:
            st.error(f"嵌入HTML失败：{e}")
    else:
        st.info("该fold目录下暂未生成验证交互HTML。完成一次验证epoch后将自动生成。")
else:
    st.info("尚未找到验证交互HTML根目录。请运行训练以生成对应文件。")
