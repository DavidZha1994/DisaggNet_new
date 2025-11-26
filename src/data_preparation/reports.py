import os
from typing import List, Optional

import numpy as np
import pandas as pd

from .io import safe_to_csv
# 可选依赖：自适应迟滞开关检测器
try:
    from src.tools.advanced_onoff_methods import AdaptiveHysteresisDetector  # type: ignore
except Exception:
    AdaptiveHysteresisDetector = None  # 依赖缺失时回退
try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None


def _get_main_cols(df: pd.DataFrame, ts_col: str) -> List[str]:
    base = [
        "P_kW", "Q_kvar", "S_kVA", "PF",
        "F_Hz",
        "U12_V", "U23_V", "U31_V",
        "V1_V", "V2_V", "V3_V",
        "I1_A", "I2_A", "I3_A", "IN_A",
        "IAVR_A", "UAVR_V", "VAVR_V",
        "E_PP_kWh", "E_QP_kvarh", "E_SP_kVAh",
        "THD_U12_F", "THD_U23_F", "THD_U31_F",
        "THD_V1_F", "THD_V2_F", "THD_V3_F",
        "THD_I1_F", "THD_I2_F", "THD_I3_F",
    ]
    cols = [ts_col] + [c for c in base if c in df.columns]
    return cols


def _get_device_cols(df: pd.DataFrame, name: str, ts_col: str) -> List[str]:
    dev_cols = [c for c in [f"{name}_P_kW", f"{name}_Q_kvar", f"{name}_S_kVA"] if c in df.columns]
    return [ts_col] + dev_cols if dev_cols else []


def export_unified_tables_core(
    df: pd.DataFrame,
    eff_dev_names: List[str],
    filled: bool,
    ts_col: str,
    output_dir: str,
) -> None:
    """合并统一导出逻辑：merged/main/devices，支持填充与未填充。"""
    try:
        base_dir = os.path.join(output_dir, "unified_filled" if filled else "unified")
        dev_dir = os.path.join(base_dir, "devices")
        os.makedirs(dev_dir, exist_ok=True)

        merged_name = "merged_filled.csv" if filled else "merged.csv"
        main_name = "main_filled.csv" if filled else "main.csv"

        # merged
        safe_to_csv(df, os.path.join(base_dir, merged_name))
        # main
        main_cols = _get_main_cols(df, ts_col)
        if main_cols:
            safe_to_csv(df[main_cols], os.path.join(base_dir, main_name))
        # devices
        for name in eff_dev_names:
            cols = _get_device_cols(df, name, ts_col)
            if not cols:
                continue
            dev_name = f"{name}_filled.csv" if filled else f"{name}.csv"
            safe_to_csv(df[cols], os.path.join(dev_dir, dev_name))
    except Exception:
        # 导出失败不阻塞主流程
        pass


def export_alignment_coverage_core(
    df: pd.DataFrame,
    eff_dev_names: List[str],
    filled: bool,
    output_dir: str,
) -> None:
    """统一覆盖率导出，支持填充与未填充。"""
    try:
        total = len(df)
        rows = []
        for name in eff_dev_names:
            col = f"{name}_P_kW"
            if col not in df.columns:
                continue
            aligned = int(np.isfinite(df[col]).sum())
            coverage = (float(aligned) / float(max(1, total))) if total > 0 else 0.0
            rows.append({
                "device": name,
                "aligned_points": aligned,
                "total_grid_points": int(total),
                "coverage_ratio": round(coverage, 6),
            })
        base_dir = os.path.join(output_dir, "unified_filled" if filled else "unified")
        os.makedirs(base_dir, exist_ok=True)
        out_name = "alignment_coverage_filled.csv" if filled else "alignment_coverage.csv"
        safe_to_csv(pd.DataFrame(rows), os.path.join(base_dir, out_name))
    except Exception:
        pass


def export_device_interruptions(
    df_filled: pd.DataFrame,
    eff_dev_names: List[str],
    ts_col: str,
    step_seconds: int,
    output_dir: str,
) -> None:
    """导出每设备在修复后的统一网格上的残余中断段（连续NaN）。"""
    try:
        seg_rows = []
        sum_rows = []
        for name in eff_dev_names:
            col = f"{name}_P_kW"
            if col not in df_filled.columns:
                continue
            mask = df_filled[col].isna().to_numpy()
            if mask.size == 0:
                sum_rows.append({
                    "device": name,
                    "num_interruptions": 0,
                    "total_seconds": 0,
                    "max_seconds": 0,
                    "median_seconds": 0,
                })
                continue
            starts = []
            ends = []
            for i in range(mask.size):
                if mask[i] and (i == 0 or not mask[i - 1]):
                    starts.append(i)
                if mask[i] and (i == mask.size - 1 or not mask[i + 1]):
                    ends.append(i)
            durations = []
            for s, e in zip(starts, ends):
                points = int(e - s + 1)
                seconds = int(points * step_seconds)
                try:
                    st = pd.to_datetime(df_filled[ts_col].iloc[s])
                    et = pd.to_datetime(df_filled[ts_col].iloc[e])
                except Exception:
                    st = df_filled[ts_col].iloc[s]
                    et = df_filled[ts_col].iloc[e]
                seg_rows.append({
                    "device": name,
                    "start_time": st,
                    "end_time": et,
                    "duration_seconds": seconds,
                    "points": points,
                })
                durations.append(seconds)
            if durations:
                total_sec = int(np.sum(durations))
                max_sec = int(np.max(durations))
                median_sec = int(np.median(durations))
                num = len(durations)
            else:
                total_sec = 0
                max_sec = 0
                median_sec = 0
                num = 0
            sum_rows.append({
                "device": name,
                "num_interruptions": int(num),
                "total_seconds": int(total_sec),
                "max_seconds": int(max_sec),
                "median_seconds": int(median_sec),
            })
        qdir = os.path.join(output_dir, "quality")
        os.makedirs(qdir, exist_ok=True)
        safe_to_csv(pd.DataFrame(seg_rows), os.path.join(qdir, "device_interruptions_segments.csv"))
        safe_to_csv(pd.DataFrame(sum_rows), os.path.join(qdir, "device_interruptions_summary.csv"))
    except Exception:
        pass


def export_device_interruptions_all_channels(
    df_filled: pd.DataFrame,
    eff_dev_names: List[str],
    ts_col: str,
    step_seconds: int,
    output_dir: str,
) -> None:
    """按设备导出各通道（P/Q/S）在修复后的统一网格上的残余中断段（连续NaN）。
    生成汇总与明细 CSV，包含 channel 字段。
    """
    try:
        qdir = os.path.join(output_dir, "quality")
        os.makedirs(qdir, exist_ok=True)

        seg_rows = []
        sum_rows = []
        ts_series = df_filled[ts_col] if ts_col in df_filled.columns else None
        is_num_ts = (ts_series is not None) and np.issubdtype(ts_series.dtype, np.number)

        channels = ["P_kW", "Q_kvar", "S_kVA"]
        for name in eff_dev_names:
            for ch in channels:
                col = f"{name}_{ch}"
                if col not in df_filled.columns:
                    continue
                mask = df_filled[col].isna().to_numpy()
                if mask.size == 0:
                    sum_rows.append({
                        "device": name,
                        "channel": ch,
                        "num_interruptions": 0,
                        "total_seconds": 0,
                        "max_seconds": 0,
                        "median_seconds": 0,
                    })
                    continue
                starts = []
                ends = []
                for i in range(mask.size):
                    if mask[i] and (i == 0 or not mask[i - 1]):
                        starts.append(i)
                    if mask[i] and (i == mask.size - 1 or not mask[i + 1]):
                        ends.append(i)
                durations = []
                for s, e in zip(starts, ends):
                    points = int(e - s + 1)
                    seconds = int(points * step_seconds)
                    try:
                        if ts_series is not None:
                            if is_num_ts:
                                st = pd.to_datetime(ts_series.iloc[s], unit="s")
                                et = pd.to_datetime(ts_series.iloc[e], unit="s")
                            else:
                                st = pd.to_datetime(ts_series.iloc[s])
                                et = pd.to_datetime(ts_series.iloc[e])
                        else:
                            st = s
                            et = e
                    except Exception:
                        st = ts_series.iloc[s] if ts_series is not None else s
                        et = ts_series.iloc[e] if ts_series is not None else e
                    seg_rows.append({
                        "device": name,
                        "channel": ch,
                        "start_time": st,
                        "end_time": et,
                        "duration_seconds": seconds,
                        "points": points,
                    })
                    durations.append(seconds)
                if durations:
                    total_sec = int(np.sum(durations))
                    max_sec = int(np.max(durations))
                    median_sec = int(np.median(durations))
                    num = len(durations)
                else:
                    total_sec = 0
                    max_sec = 0
                    median_sec = 0
                    num = 0
                sum_rows.append({
                    "device": name,
                    "channel": ch,
                    "num_interruptions": int(num),
                    "total_seconds": int(total_sec),
                    "max_seconds": int(max_sec),
                    "median_seconds": int(median_sec),
                })

        safe_to_csv(pd.DataFrame(seg_rows), os.path.join(qdir, "device_interruptions_segments_all_channels.csv"))
        safe_to_csv(pd.DataFrame(sum_rows), os.path.join(qdir, "device_interruptions_summary_all_channels.csv"))
    except Exception:
        pass


def export_mains_interruptions_all_channels(
    df_filled: pd.DataFrame,
    ts_col: str,
    step_seconds: int,
    output_dir: str,
) -> None:
    """按总表导出各通道（P/Q/S/PF）在修复后的统一网格上的残余中断段（连续NaN）。
    生成汇总与明细 CSV，包含 channel 字段。
    """
    try:
        qdir = os.path.join(output_dir, "quality")
        os.makedirs(qdir, exist_ok=True)

        seg_rows = []
        sum_rows = []
        ts_series = df_filled[ts_col] if ts_col in df_filled.columns else None
        is_num_ts = (ts_series is not None) and np.issubdtype(ts_series.dtype, np.number)

        channels = ["P_kW", "Q_kvar", "S_kVA", "PF"]
        for ch in channels:
            if ch not in df_filled.columns:
                continue
            mask = df_filled[ch].isna().to_numpy()
            if mask.size == 0:
                sum_rows.append({
                    "channel": ch,
                    "num_interruptions": 0,
                    "total_seconds": 0,
                    "max_seconds": 0,
                    "median_seconds": 0,
                })
                continue
            starts = []
            ends = []
            for i in range(mask.size):
                if mask[i] and (i == 0 or not mask[i - 1]):
                    starts.append(i)
                if mask[i] and (i == mask.size - 1 or not mask[i + 1]):
                    ends.append(i)
            durations = []
            for s, e in zip(starts, ends):
                points = int(e - s + 1)
                seconds = int(points * step_seconds)
                try:
                    if ts_series is not None:
                        if is_num_ts:
                            st = pd.to_datetime(ts_series.iloc[s], unit="s")
                            et = pd.to_datetime(ts_series.iloc[e], unit="s")
                        else:
                            st = pd.to_datetime(ts_series.iloc[s])
                            et = pd.to_datetime(ts_series.iloc[e])
                    else:
                        st = s
                        et = e
                except Exception:
                    st = ts_series.iloc[s] if ts_series is not None else s
                    et = ts_series.iloc[e] if ts_series is not None else e
                seg_rows.append({
                    "channel": ch,
                    "start_time": st,
                    "end_time": et,
                    "duration_seconds": seconds,
                    "points": points,
                })
                durations.append(seconds)
            if durations:
                total_sec = int(np.sum(durations))
                max_sec = int(np.max(durations))
                median_sec = int(np.median(durations))
                num = len(durations)
            else:
                total_sec = 0
                max_sec = 0
                median_sec = 0
                num = 0
            sum_rows.append({
                "channel": ch,
                "num_interruptions": int(num),
                "total_seconds": int(total_sec),
                "max_seconds": int(max_sec),
                "median_seconds": int(median_sec),
            })

        safe_to_csv(pd.DataFrame(seg_rows), os.path.join(qdir, "mains_interruptions_segments.csv"))
        safe_to_csv(pd.DataFrame(sum_rows), os.path.join(qdir, "mains_interruptions_summary.csv"))
    except Exception:
        pass


def export_onoff_masks(
    df_filled: pd.DataFrame,
    eff_dev_names: List[str],
    ts_col: str,
    output_dir: str,
    method: str = "hmm_like",
) -> None:
    """导出设备在统一网格上的开关(on/off)掩码（不导出主端）。
    - 设备：基于每设备 `*_P_kW` 检测（若存在）。
    输出到 `quality/` 目录：
    - `device_onoff_masks.csv`，列：timestamp, device, on, method
    改进：
    - 使用向量化的 HMM-like 检测（见 AdvancedOnOffDetector 加速版）。
    - 采用按设备分块流式写入，避免一次性构建千万级行的列表导致内存膨胀。
    - 在设备级进度上显示处理进度。
    """
    try:
        qdir = os.path.join(output_dir, "quality")
        os.makedirs(qdir, exist_ok=True)
        ts = pd.to_datetime(df_filled[ts_col], errors="coerce") if ts_col in df_filled.columns else None
        if ts is None:
            return
        # 优先使用自适应迟滞检测器；若不存在则回退为中位数阈值法
        det = None
        try:
            if AdaptiveHysteresisDetector is not None:
                det = AdaptiveHysteresisDetector()
        except Exception:
            det = None

        out_fp = os.path.join(qdir, "device_onoff_masks.csv")
        header_written = False
        chunk_size = 200_000  # 分块写入，降低峰值内存

        iterator = eff_dev_names
        if _tqdm is not None:
            iterator = _tqdm(eff_dev_names, desc="生成设备开关掩码", unit="device")

        for name in iterator:
            col = f"{name}_P_kW"
            if col not in df_filled.columns:
                continue
            x = df_filled[col].to_numpy(dtype=np.float32)
            try:
                if det is not None:
                    # 统一使用自适应迟滞法（与管线一致）
                    st, info = det.detect(x)
                else:
                    raise RuntimeError("Adaptive detector unavailable")
            except Exception:
                st = (x > np.nanmedian(x)).astype(int)
                info = {"method": "median_threshold"}
            method_tag = info.get("method", method)

            # 分块流式写入 CSV，避免一次性构建巨型 DataFrame
            T = len(ts)
            for s in range(0, T, chunk_size):
                e = min(s + chunk_size, T)
                df_chunk = pd.DataFrame({
                    "timestamp": ts.iloc[s:e].to_numpy(),
                    "device": name,
                    "on": np.asarray(st[s:e], dtype=int),
                    "method": method_tag,
                })
                df_chunk.to_csv(out_fp, mode=("a" if header_written else "w"), header=(not header_written), index=False)
                header_written = True
    except Exception:
        pass


def export_quality_report(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    X_seq: np.ndarray,
    Yp_seq: np.ndarray,
    freq_mask: Optional[np.ndarray],
    gap_mask_seq: Optional[np.ndarray],
    windows_meta: pd.DataFrame,
    segments_meta: pd.DataFrame,
    starts: np.ndarray,
    L: int,
    H: int,
    frame_valid_ratio_thr: float,
    device_intersection_thr: float,
    window_valid_ratio_thr: float,
    ts_col: str,
    output_dir: str,
) -> None:
    """导出数据质量报告（CSV）。"""
    out_dir = os.path.join(output_dir, "quality")
    os.makedirs(out_dir, exist_ok=True)
    T = int(df_after.shape[0])

    # 1) 基线 naive 窗口（不考虑分段）
    if T >= L:
        naive_starts = np.arange(0, T - L + 1, H, dtype=np.int64)
    else:
        naive_starts = np.array([], dtype=np.int64)

    seg_bounds = segments_meta[["start_idx", "end_idx"]].to_numpy(dtype=int) if not segments_meta.empty else np.zeros((0, 2), dtype=int)
    naive_in_segment = np.zeros_like(naive_starts, dtype=np.uint8)
    naive_seg_id = np.full_like(naive_starts, fill_value=-1, dtype=np.int64)
    for i, s in enumerate(naive_starts):
        e = s + L - 1
        ok = False
        sid = -1
        for j, (a, b) in enumerate(seg_bounds):
            if s >= a and e <= b:
                ok = True
                sid = j
                break
        naive_in_segment[i] = 1 if ok else 0
        naive_seg_id[i] = sid
    dropped_by_segment = int(naive_starts.shape[0] - starts.shape[0])

    # 保存 naive 窗口明细
    try:
        df_naive = pd.DataFrame({
            "start_idx": naive_starts,
            "end_idx": (naive_starts + L - 1),
            "in_segment": naive_in_segment.astype(np.uint8),
            "segment_id": naive_seg_id,
        })
        safe_to_csv(df_naive, os.path.join(out_dir, "naive_windows.csv"))
    except Exception:
        pass

    # 2) 实际窗口质量明细
    N = int(starts.shape[0])
    frames = int(freq_mask.shape[1]) if (freq_mask is not None and getattr(freq_mask, "size", 0) > 0) else 0
    if frames > 0 and freq_mask is not None and getattr(freq_mask, "size", 0) > 0:
        freq_valid_frames = freq_mask.sum(axis=1)
        freq_valid_ratio = (freq_valid_frames / float(frames)).astype(np.float32)
    else:
        freq_valid_frames = np.zeros(N, dtype=np.int64)
        freq_valid_ratio = np.zeros(N, dtype=np.float32)

    if getattr(X_seq, "size", 0) > 0:
        Lx = X_seq.shape[1]
        Cx = X_seq.shape[2]
        x_valid_ratio = (np.isfinite(X_seq).sum(axis=(1, 2)) / float(max(1, Lx * Cx))).astype(np.float32)
    else:
        x_valid_ratio = np.zeros(N, dtype=np.float32)

    if getattr(Yp_seq, "size", 0) > 0:
        Ly = Yp_seq.shape[1]
        Ky = Yp_seq.shape[2]
        y_valid_ratio_overall = (np.isfinite(Yp_seq).sum(axis=(1, 2)) / float(max(1, Ly * Ky))).astype(np.float32)
        y_all_valid_ratio = (np.all(np.isfinite(Yp_seq), axis=2).sum(axis=1) / float(max(1, Ly))).astype(np.float32)
        y_all_nan = np.all(np.isnan(Yp_seq), axis=(1, 2)).astype(np.uint8)
    else:
        y_valid_ratio_overall = np.zeros(N, dtype=np.float32)
        y_all_valid_ratio = np.zeros(N, dtype=np.float32)
        y_all_nan = np.zeros(N, dtype=np.uint8)

    if gap_mask_seq is not None and getattr(gap_mask_seq, "size", 0) > 0:
        Lg = gap_mask_seq.shape[1]
        Cg = gap_mask_seq.shape[2]
        gap_ratio = (gap_mask_seq.sum(axis=(1, 2)) / float(max(1, Lg * Cg))).astype(np.float32)
    else:
        gap_ratio = np.zeros(N, dtype=np.float32)

    would_drop_low_freq = (freq_valid_ratio < float(frame_valid_ratio_thr)).astype(np.uint8)
    would_drop_low_x = (x_valid_ratio < float(window_valid_ratio_thr)).astype(np.uint8)
    would_drop_low_device_intersection = (y_all_valid_ratio < float(device_intersection_thr)).astype(np.uint8)

    try:
        detail = windows_meta.copy()
        detail["start_idx"] = starts.astype(np.int64)
        detail["end_idx"] = (starts + L - 1).astype(np.int64)
        detail["window_length"] = int(L)
        detail["step"] = int(H)
        detail["x_valid_ratio"] = x_valid_ratio
        detail["freq_valid_frames"] = freq_valid_frames.astype(np.int64)
        detail["freq_total_frames"] = int(frames)
        detail["freq_valid_ratio"] = freq_valid_ratio
        detail["gap_ratio_mains"] = gap_ratio
        detail["y_valid_ratio_overall"] = y_valid_ratio_overall
        detail["y_all_devices_intersection_ratio"] = y_all_valid_ratio
        detail["y_all_nan_label"] = y_all_nan.astype(np.uint8)
        detail["would_drop_low_freq"] = would_drop_low_freq
        detail["would_drop_low_x"] = would_drop_low_x
        detail["would_drop_low_device_intersection"] = would_drop_low_device_intersection
        if "start_ts" in detail.columns:
            try:
                detail["start_time"] = pd.to_datetime(detail["start_ts"], unit="s")
                detail["end_time"] = pd.to_datetime(detail["end_ts"], unit="s")
            except Exception:
                pass
        safe_to_csv(detail, os.path.join(out_dir, "windows_detail.csv"))
    except Exception:
        pass

    try:
        summary_rows = []
        summary_rows.append(("naive_possible_windows", int(naive_starts.shape[0])))
        summary_rows.append(("kept_windows_segment_aware", int(N)))
        summary_rows.append(("dropped_by_segment_boundary", int(dropped_by_segment)))
        summary_rows.append(("windows_would_drop_low_freq", int(would_drop_low_freq.sum())))
        summary_rows.append(("windows_would_drop_low_x", int(would_drop_low_x.sum())))
        summary_rows.append(("windows_would_drop_low_device_intersection", int(would_drop_low_device_intersection.sum())))
        summary_rows.append(("windows_label_all_nan", int(y_all_nan.sum())))
        if not segments_meta.empty:
            summary_rows.append(("segments_count", int(segments_meta.shape[0])))
            summary_rows.append(("segment_len_mean", float(segments_meta["n_rows"].mean())))
            summary_rows.append(("segment_len_median", float(segments_meta["n_rows"].median())))
        df_sum = pd.DataFrame(summary_rows, columns=["metric", "value"])
        safe_to_csv(df_sum, os.path.join(out_dir, "windows_summary.csv"))
    except Exception:
        pass

    try:
        ex_dir = os.path.join(out_dir, "examples")
        os.makedirs(ex_dir, exist_ok=True)
        mains_cols = [c for c in ["P_kW", "Q_kvar", "S_kVA", "PF"] if c in df_after.columns]
        good = np.where((freq_valid_ratio >= 0.95) & (x_valid_ratio >= 0.95) & (gap_ratio <= 1e-6))[0]
        lowf = np.where(would_drop_low_freq == 1)[0]
        lowi = np.where(would_drop_low_device_intersection == 1)[0]
        heavy_gap = np.where(gap_ratio > 0.1)[0]

        def _emit_examples(idxs: np.ndarray, tag: str, max_n: int = 3):
            for idx in idxs[:max_n]:
                s = int(starts[idx])
                e = int(s + L)
                bef = df_before.iloc[s:e].copy()
                aft = df_after.iloc[s:e].copy()
                out = pd.DataFrame({
                    ts_col: pd.to_datetime(aft[ts_col]).reset_index(drop=True)
                }) if ts_col in aft.columns else pd.DataFrame({"idx": np.arange(s, e)})
                for c in mains_cols:
                    if c in bef.columns:
                        out[f"{c}_before"] = bef[c].reset_index(drop=True)
                    else:
                        out[f"{c}_before"] = np.nan
                    out[f"{c}_after"] = aft[c].reset_index(drop=True) if c in aft.columns else np.nan
                if gap_mask_seq is not None and getattr(gap_mask_seq, "size", 0) > 0 and len(mains_cols) > 0:
                    for j, c in enumerate(mains_cols[:gap_mask_seq.shape[2]]):
                        try:
                            out[f"gap_mask_{c}"] = gap_mask_seq[idx, :, j].astype(np.uint8)
                        except Exception:
                            pass
                fp = os.path.join(ex_dir, f"example_{tag}_win{int(idx)}.csv")
                safe_to_csv(out, fp)

        _emit_examples(good, "clean")
        _emit_examples(lowf, "low_freq")
        _emit_examples(lowi, "low_device_intersection")
        _emit_examples(heavy_gap, "heavy_gap")

        cross_idxs = np.where(naive_in_segment == 0)[0]
        for i in cross_idxs[:3]:
            s = int(naive_starts[i])
            e = int(s + L)
            bef = df_before.iloc[s:e].copy()
            aft = df_after.iloc[s:e].copy()
            out = pd.DataFrame({
                ts_col: pd.to_datetime(aft[ts_col]).reset_index(drop=True)
            }) if ts_col in aft.columns else pd.DataFrame({"idx": np.arange(s, e)})
            for c in mains_cols:
                out[f"{c}_before"] = bef[c].reset_index(drop=True) if c in bef.columns else np.nan
                out[f"{c}_after"] = aft[c].reset_index(drop=True) if c in aft.columns else np.nan
            safe_to_csv(out, os.path.join(ex_dir, f"example_cross_segment_naive_{i}.csv"))
    except Exception:
        pass
