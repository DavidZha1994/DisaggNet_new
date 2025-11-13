import os
import json
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import streamlit as st


def _load_pt(fp: str):
    # 适配 PyTorch 2.6 的安全反序列化默认行为
    # 显式设置 weights_only=False 以加载包含 numpy/字典的 .pt 文件
    obj = torch.load(fp, map_location='cpu', weights_only=False)
    return obj


def _get_config_defaults() -> Dict:
    # 兼容：若无法读取配置文件，使用合理默认值
    defaults = {
        'resample_seconds': 5,
        'window_length': 256,
        'step_size': 128,
    }
    cfg_path = os.path.join('configs', 'pipeline', 'prep_config.yaml')
    try:
        import yaml
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        hipe = (cfg or {}).get('hipe', {})
        defaults['resample_seconds'] = int(hipe.get('resample_seconds', defaults['resample_seconds']))
        defaults['window_length'] = int(hipe.get('window_length', defaults['window_length']))
        defaults['step_size'] = int(hipe.get('step_size', defaults['step_size']))
    except Exception:
        pass
    return defaults


def overlap_add(frames: np.ndarray,
                 starts: Optional[np.ndarray],
                 step_size: int,
                 mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    将窗口级帧重建为完整时间序列（重叠平均）。
    frames: [N, L] 或 [N, L, C]（需已选定通道后为 [N,L]）
    starts: [N] 开始索引；若为 None，则使用 i*step_size
    mask: [N, L] 有效掩码（True/1 表示有效）
    """
    if frames.ndim == 3:
        # 若仍含通道维，合并为 [N, L]（调用方应预先选通道，但这里兜底取第0通道）
        frames = frames[..., 0]
    N, L = frames.shape
    if starts is None:
        starts = np.arange(N, dtype=np.int64) * int(step_size)
    total_len = int(starts.max()) + int(L)
    acc = np.zeros(total_len, dtype=np.float64)
    cnt = np.zeros(total_len, dtype=np.int64)
    if mask is None:
        mask = np.ones_like(frames, dtype=np.bool_)
    for i in range(N):
        s = int(starts[i])
        e = s + L
        valid = mask[i]
        f = frames[i]
        acc[s:e][valid] += f[valid]
        cnt[s:e][valid] += 1
    series = np.full(total_len, np.nan, dtype=np.float64)
    nz = cnt > 0
    series[nz] = acc[nz] / cnt[nz]
    return series


def rolling_low_quantile(x: np.ndarray, window_points: int, q: float) -> np.ndarray:
    ser = pd.Series(x)
    # 使用 min_periods=1 保证前期也有值；对 NaN 自动跳过
    base = ser.rolling(window_points, min_periods=1).quantile(q)
    return base.to_numpy()


def load_fold_paths(prepared_dir: str, fold_id: int) -> Dict[str, str]:
    fold_dir = os.path.join(prepared_dir, f"fold_{fold_id}")
    return {
        'fold_dir': fold_dir,
        'raw_pt': os.path.join(fold_dir, 'train_raw.pt'),
        'targets_pt': os.path.join(fold_dir, 'train_targets_seq.pt'),
        'raw_names_json': os.path.join(fold_dir, 'raw_channel_names.json'),
        'feature_names_json': os.path.join(fold_dir, 'feature_names.json'),
        'indices_pt': os.path.join(fold_dir, 'train_indices.pt'),
        'mask_pt': os.path.join(fold_dir, 'train_mask.pt'),
        'labels_pt': os.path.join(fold_dir, 'train_labels.pt'),
    }


def resolve_mains_channel(frames_obj, raw_names_path: str) -> int:
    # 默认取 P_kW 为主表功率通道
    try:
        with open(raw_names_path, 'r', encoding='utf-8') as f:
            names = json.load(f) or []
        if isinstance(names, list) and names:
            for idx, nm in enumerate(names):
                if str(nm).lower().startswith('p'):
                    return idx
    except Exception:
        pass
    # 兜底：第0通道
    return 0


def main():
    st.set_page_config(page_title="24h低分位基线对比", layout="wide")
    st.title("24小时滚动低分位基线对比（mains 与设备总功率）")

    # 基本输入
    prepared_dir = st.text_input("prepared目录", value="Data/prepared/hipe")
    fold_id = st.number_input("fold_id", min_value=0, value=2, step=1)
    q = st.slider("分位数 q", 0.01, 0.50, 0.05, 0.01)
    show_debug = st.checkbox("显示调试信息", value=False)
    clip_neg = st.checkbox("主表去基线负值截断为0", value=False, help="优先通过调整 q 避免负值；若仍出现负值，可勾选此项将负值截断为0。")
    cap_baseline = st.checkbox("限制基线不超过主表", value=True, help="将滚动低分位基线与主表逐点取最小值，避免基线超过瞬时主表导致负值。")

    cfg = _get_config_defaults()
    resample_seconds = int(cfg['resample_seconds'])
    window_length = int(cfg['window_length'])
    step_size = int(cfg['step_size'])
    day_points = int(24 * 3600 // max(resample_seconds, 1))

    st.caption(f"采样间隔: {resample_seconds}s, 窗长: {window_length}, 步长: {step_size}, 24h点数: {day_points}")

    paths = load_fold_paths(prepared_dir, int(fold_id))
    # 加载原始窗口
    if not os.path.exists(paths['raw_pt']):
        st.error(f"未找到 {paths['raw_pt']}。请确认已生成 .pt 文件或选择正确折。")
        return
    raw_obj = _load_pt(paths['raw_pt'])

    # 兼容对象格式（避免对张量使用布尔短路导致报错）
    if isinstance(raw_obj, dict):
        if 'frames' in raw_obj:
            frames = raw_obj['frames']
        elif 'X' in raw_obj:
            frames = raw_obj['X']
        elif 'X_seq' in raw_obj:
            frames = raw_obj['X_seq']
        else:
            frames = None
    else:
        frames = raw_obj
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().numpy()
    if frames is None:
        st.error("train_raw.pt 不包含可识别的窗口帧。")
        return
    # 调试：显示原始窗口维度
    try:
        N, L = frames.shape[:2]
        C = frames.shape[2] if frames.ndim == 3 else 1
        st.caption(f"raw frames 维度: N={N}, L={L}, C={C}")
    except Exception:
        pass
    # 加载 indices / mask（可选）
    starts = None
    if os.path.exists(paths['indices_pt']):
        idx_obj = _load_pt(paths['indices_pt'])
        if isinstance(idx_obj, torch.Tensor):
            starts = idx_obj.detach().cpu().numpy().astype(np.int64)
        elif isinstance(idx_obj, dict):
            starts = idx_obj.get('indices')
            if isinstance(starts, torch.Tensor):
                starts = starts.detach().cpu().numpy().astype(np.int64)
    # 若 indices 缺失，尝试从 labels 的时间戳构造绝对时间索引用于展示
    labels_meta = None
    abs_start_sec = None
    if os.path.exists(paths['labels_pt']):
        lbl_obj = _load_pt(paths['labels_pt'])
        if isinstance(lbl_obj, dict):
            labels_meta = lbl_obj.get('label_metadata')
            if isinstance(labels_meta, list) and labels_meta:
                iso0 = labels_meta[0].get('datetime_iso')
                try:
                    abs_start_sec = int(pd.to_datetime(iso0, errors='coerce').value // 1_000_000_000)
                except Exception:
                    abs_start_sec = None
    mask = None
    if os.path.exists(paths['mask_pt']):
        m_obj = _load_pt(paths['mask_pt'])
        m = m_obj.get('mask') if isinstance(m_obj, dict) else m_obj
        if isinstance(m, torch.Tensor):
            mask = m.detach().cpu().numpy().astype(bool)

    # 选择主表 P 通道
    mains_ch = resolve_mains_channel(frames, paths['raw_names_json'])
    mains_frames = frames[..., mains_ch]
    mains_series = overlap_add(mains_frames, starts, step_size, mask)

    # 若存在绝对起点，构造时间索引用于更直观展示
    dt_index = None
    if abs_start_sec is not None and isinstance(mains_series, np.ndarray) and mains_series.size > 0:
        try:
            dt_index = pd.to_datetime(np.arange(abs_start_sec, abs_start_sec + len(mains_series) * resample_seconds, resample_seconds), unit='s')
        except Exception:
            dt_index = None

    # 24h滚动低分位基线与去基线
    baseline_raw = rolling_low_quantile(mains_series, day_points, q)
    baseline = np.minimum(baseline_raw, mains_series) if cap_baseline else baseline_raw
    mains_detrended = mains_series - baseline
    mains_detrended_display = np.maximum(mains_detrended, 0) if clip_neg else mains_detrended

    # 加载设备窗口并求和
    if not os.path.exists(paths['targets_pt']):
        st.warning(f"未找到 {paths['targets_pt']}，仅展示主表曲线。")
        targets_P = None
    else:
        t_obj = _load_pt(paths['targets_pt'])
        targets_P = None
        picked_key = None
        if isinstance(t_obj, dict):
            # 优先适配新版结构
            if 'targets_P' in t_obj:
                targets_P = t_obj['targets_P']; picked_key = 'targets_P'
            elif 'targets' in t_obj:
                targets_P = t_obj['targets']; picked_key = 'targets'
            else:
                # 兼容：自动发现形状为 [N,L,D] 的张量键
                for k, v in t_obj.items():
                    if isinstance(v, torch.Tensor) and v.ndim == 3:
                        targets_P = v; picked_key = k; break
        elif isinstance(t_obj, torch.Tensor):
            # 兼容：直接保存为张量 [N,L,D]
            targets_P = t_obj; picked_key = '<tensor>'
        if isinstance(targets_P, torch.Tensor):
            targets_P = targets_P.detach().cpu().numpy()
        if show_debug:
            st.caption("targets文件结构调试信息：")
            if isinstance(t_obj, dict):
                keys = list(t_obj.keys())
                st.text(f"keys: {keys}")
                if picked_key:
                    st.text(f"picked key: {picked_key}")
            else:
                st.text(f"type: {type(t_obj)}")

    device_sum_series = None
    device_sum_detrended = None
    if targets_P is not None:
        # targets_P 假设为 [N, L, D]
        dev_sum_frames = np.nansum(targets_P, axis=-1)
        device_sum_series = overlap_add(dev_sum_frames, starts, step_size, mask)
        dev_baseline = rolling_low_quantile(device_sum_series, day_points, q)
        device_sum_detrended = device_sum_series - dev_baseline

    # 画图
    st.subheader("主表功率：原始 vs 去基线" + ("（负值截断为0）" if clip_neg else "") + ("；基线不超过主表" if cap_baseline else ""))
    df_main = pd.DataFrame({
        'mains_raw': mains_series,
        'baseline': baseline,
        ('mains_detrended' if not clip_neg else 'mains_detrended_clipped'): mains_detrended_display,
    })
    if dt_index is not None:
        df_main.index = dt_index
    st.line_chart(df_main)
    # 负值统计提示
    try:
        neg_ratio = float(np.mean(np.nan_to_num(mains_detrended) < 0))
        min_val = float(np.nanmin(mains_detrended))
        st.caption(f"主表去基线负值比例：{neg_ratio:.2%}；最小值：{min_val:.3f}")
    except Exception:
        pass

    if device_sum_series is not None:
        st.subheader("设备总功率：原始 vs 24h低分位去基线")
        df_dev = pd.DataFrame({
            'devices_sum_raw': device_sum_series,
            'devices_sum_detrended': device_sum_detrended,
        })
        if dt_index is not None:
            df_dev.index = dt_index
        st.line_chart(df_dev)

        st.subheader("对比：主表原始 vs 设备总原始")
        df_cmp = pd.DataFrame({
            'mains_raw': mains_series,
            'devices_sum_raw': device_sum_series,
        })
        if dt_index is not None:
            df_cmp.index = dt_index
        st.line_chart(df_cmp)

        # 新增：主表去基线 vs 设备总原始
        st.subheader("对比：主表去基线 vs 设备总原始" + ("（主表负值已截断）" if clip_neg else ""))
        df_cmp2 = pd.DataFrame({
            ('mains_detrended' if not clip_neg else 'mains_detrended_clipped'): mains_detrended_display,
            'devices_sum_raw': device_sum_series,
        })
        if dt_index is not None:
            df_cmp2.index = dt_index
        st.line_chart(df_cmp2)

    if device_sum_series is None and os.path.exists(paths['targets_pt']):
        st.info("已找到 targets 文件，但未识别到设备P目标键。可打开“显示调试信息”查看键列表。")
    # 显式提示数据是否为空
    if isinstance(mains_series, np.ndarray) and mains_series.size == 0:
        st.error("重建后的主表序列为空，请检查 starts/step_size 是否正确或数据是否存在。")
    else:
        st.success("完成：可用滑块调整分位数 q（Q10–Q20），观察对比效果。")


if __name__ == '__main__':
    main()