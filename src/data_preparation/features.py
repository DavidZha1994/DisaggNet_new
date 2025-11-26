import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


def build_mains_features(df: pd.DataFrame) -> np.ndarray:
    """构建主端原始+导数特征矩阵 [T, 7]：P/Q/S/PF 及其导数 dP/dQ/dS。
    期望列名为对齐后的规范名：`P_kW`, `Q_kvar`, `S_kVA`, `PF`。
    """
    T = len(df)
    P = df["P_kW"].to_numpy(dtype=np.float32) if "P_kW" in df.columns else np.full(T, np.nan, dtype=np.float32)
    Q = df["Q_kvar"].to_numpy(dtype=np.float32) if "Q_kvar" in df.columns else np.full(T, np.nan, dtype=np.float32)
    S = df["S_kVA"].to_numpy(dtype=np.float32) if "S_kVA" in df.columns else np.full(T, np.nan, dtype=np.float32)
    PF = df["PF"].to_numpy(dtype=np.float32) if "PF" in df.columns else np.full(T, np.nan, dtype=np.float32)
    dP = np.concatenate([np.array([0.0], dtype=np.float32), np.diff(P).astype(np.float32)]) if T > 0 else np.empty(0, dtype=np.float32)
    dQ = np.concatenate([np.array([0.0], dtype=np.float32), np.diff(Q).astype(np.float32)]) if T > 0 else np.empty(0, dtype=np.float32)
    dS = np.concatenate([np.array([0.0], dtype=np.float32), np.diff(S).astype(np.float32)]) if T > 0 else np.empty(0, dtype=np.float32)
    X = np.stack([P, Q, S, PF, dP, dQ, dS], axis=1)
    return X.astype(np.float32)


def build_targets(df: pd.DataFrame, dev_names: List[str], kind: str = "P") -> np.ndarray:
    """构建目标矩阵 [T, K]，每列为设备的功率序列（P/Q/S）。"""
    T = len(df)
    mats = []
    for name in dev_names:
        col = f"{name}_P_kW" if kind == "P" else f"{name}_Q_kvar" if kind == "Q" else f"{name}_S_kVA"
        if col in df.columns:
            mats.append(df[col].to_numpy(dtype=np.float32))
        else:
            mats.append(np.full(T, np.nan, dtype=np.float32))
    if mats:
        Y = np.stack(mats, axis=1)
    else:
        Y = np.empty((T, 0), dtype=np.float32)
    return Y


def slide_window(
    X_full: np.ndarray,
    Y_full: np.ndarray,
    L: int,
    H: int,
    starts_override: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按长度 L、步长 H 在全序列上滑窗，返回窗口序列及 starts。"""
    T = X_full.shape[0]
    if T == 0 or L <= 0:
        return (
            np.empty((0, 0, 0), dtype=np.float32),
            np.empty((0, 0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    starts = starts_override if starts_override is not None else np.arange(0, max(0, T - L + 1), H, dtype=np.int64)
    X_list = []
    Y_list = []
    for s in starts:
        e = s + L
        X_list.append(X_full[s:e, :])
        Y_list.append(Y_full[s:e, :])
    X_seq = np.stack(X_list, axis=0).astype(np.float32) if X_list else np.empty((0, L, X_full.shape[1]), dtype=np.float32)
    Y_seq = np.stack(Y_list, axis=0).astype(np.float32) if Y_list else np.empty((0, L, Y_full.shape[1]), dtype=np.float32)
    return X_seq, Y_seq, starts


def valid_window_mask_by_ratio(
    Y_full: np.ndarray,
    starts: np.ndarray,
    L: int,
    min_ratio: float = 0.8,
) -> np.ndarray:
    """按窗口内有效标注比例筛选窗口。返回布尔掩码。"""
    if Y_full.size == 0:
        return np.ones_like(starts, dtype=bool)
    K = Y_full.shape[1]
    mask = np.zeros(len(starts), dtype=bool)
    total = L * max(1, K)
    for i, s in enumerate(starts):
        e = s + L
        win = Y_full[s:e, :]
        valid = np.isfinite(win).sum()
        ratio = float(valid) / float(total)
        mask[i] = (ratio >= float(min_ratio))
    return mask


def window_stft_frames(
    X_seq: np.ndarray,
    n_fft: int,
    hop: int,
    win_length: int,
    window_type: str = "hann",
    valid_ratio_threshold: float = 0.85,
) -> Tuple[np.ndarray, np.ndarray]:
    """按窗口对 P/Q/S/PF 通道计算 STFT 幅度谱，并生成帧掩码。
    - 返回: (freq_feats, freq_mask)
      freq_feats: [N, T_frames, F_bins*C]
      freq_mask:  [N, T_frames]  (uint8; 1=有效, 0=无效)
    - 帧有效性判断：帧内非 NaN 比例（跨所选通道聚合）≥ 阈值。
    """
    if X_seq.size == 0:
        return np.empty((0, 0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.uint8)
    N, L, C = X_seq.shape
    candidate = [0, 1, 2, 3]
    ch_valid_counts = [int(np.isfinite(X_seq[:, :, ch]).sum()) for ch in candidate if ch < C]
    stft_channels = [ch for ch, cnt in zip(candidate, ch_valid_counts) if cnt > 0 and ch < C]
    if len(stft_channels) == 0:
        stft_channels = [0] if C > 0 else []
    if n_fft < win_length:
        n_fft = win_length
    wtype = (window_type or "hann").lower()
    if wtype == "hann":
        window = np.hanning(win_length).astype(np.float32)
    elif wtype == "hamming":
        window = np.hamming(win_length).astype(np.float32)
    else:
        window = np.ones(win_length, dtype=np.float32)
    frames = 1 if L < win_length else (1 + (L - win_length) // hop)
    F = (n_fft // 2) + 1
    C_eff = len(stft_channels)
    out = np.empty((N, frames, F * C_eff), dtype=np.float32)
    mask = np.zeros((N, frames), dtype=np.uint8)
    for i in range(N):
        sigs = []
        for ci, ch in enumerate(stft_channels):
            sig = X_seq[i, :, ch].astype(np.float32)
            cnt = np.isfinite(sig).sum()
            if cnt > 0:
                m = float(np.nansum(sig) / cnt)
            else:
                m = 0.0
            sigs.append(np.where(np.isnan(sig), m, sig))
        sigs = np.stack(sigs, axis=0)
        for t in range(frames):
            start = t * hop
            end = start + win_length
            if end <= L:
                frame_block = X_seq[i, start:end, stft_channels]
            else:
                take = max(0, L - start)
                blk = np.full((win_length, C_eff), np.nan, dtype=np.float32)
                if take > 0:
                    blk[:take, :] = X_seq[i, start:start + take, stft_channels]
                frame_block = blk
            valid = np.isfinite(frame_block).sum()
            ratio = float(valid) / float(win_length * C_eff) if (win_length * C_eff) > 0 else 0.0
            frame_valid = ratio >= float(valid_ratio_threshold)
            mask[i, t] = 1 if frame_valid else 0
            for ci in range(C_eff):
                sig = sigs[ci]
                if end <= L:
                    frame = sig[start:end]
                else:
                    pad = np.zeros(win_length, dtype=np.float32)
                    take = max(0, L - start)
                    if take > 0:
                        pad[:take] = sig[start:start + take]
                    frame = pad
                frame_win = frame * window
                spec = np.fft.rfft(frame_win, n=n_fft)
                mag = np.abs(spec).astype(np.float32)
                if not frame_valid:
                    mag[:] = 0.0
                out[i, t, ci * F:(ci + 1) * F] = mag
    return out, mask


def create_segments_meta(
    df_merged: pd.DataFrame,
    ts_col: str,
    resample_seconds: int,
    k_median: float = 4.0,
    gap_seconds_min: int = 30,
) -> pd.DataFrame:
    """按 Δt > max(30s, k×medianΔt) 与主端缺失切段，输出连续段元数据。"""
    s = pd.to_datetime(df_merged[ts_col])
    ts_secs = (s.astype("int64") // 1_000_000_000).to_numpy()
    mains_cols = [c for c in ["P_kW", "Q_kvar", "S_kVA", "PF"] if c in df_merged.columns]
    arr = df_merged[mains_cols].to_numpy(dtype=np.float32) if mains_cols else np.empty((len(df_merged), 0), dtype=np.float32)
    if len(ts_secs) >= 2:
        dts = np.diff(ts_secs)
        dts = dts[(dts > 0) & np.isfinite(dts)]
        if dts.size == 0:
            baseline = max(1, int(resample_seconds))
        else:
            baseline = float(np.median(dts))
    else:
        baseline = max(1, int(resample_seconds))
    gap_thr = max(int(gap_seconds_min), float(k_median) * baseline)
    segments = []
    n = len(df_merged)
    seg_start = None
    for i in range(n):
        row_nan = np.isnan(arr[i]).any() if arr.size else False
        if seg_start is None:
            if not row_nan:
                seg_start = i
            continue
        gap = ts_secs[i] - ts_secs[i - 1]
        if row_nan or (gap > gap_thr):
            end = i - 1
            if end >= seg_start:
                segments.append((seg_start, end))
            seg_start = None
            if not row_nan:
                seg_start = i
    if seg_start is not None and (n - 1) >= seg_start:
        segments.append((seg_start, n - 1))
    seg_rows = []
    for sid, (a, b) in enumerate(segments):
        start_ts = int(ts_secs[a])
        end_ts = int(ts_secs[b])
        n_rows = int(b - a + 1)
        duration_hours = float((end_ts - start_ts) / 3600.0)
        seg_rows.append({
            "segment_id": sid,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "start_idx": int(a),
            "end_idx": int(b),
            "n_rows": n_rows,
            "duration_hours": duration_hours,
        })
    return pd.DataFrame(seg_rows)


def build_windows_metadata(
    df_merged: pd.DataFrame,
    starts: np.ndarray,
    L: int,
    ts_col: str,
    resample_seconds: int,
    k_median: float = 4.0,
    gap_seconds_min: int = 30,
) -> pd.DataFrame:
    """为每个窗口生成元数据并映射所属段。"""
    s = pd.to_datetime(df_merged[ts_col])
    ts_secs = (s.astype("int64") // 1_000_000_000).to_numpy()
    n = len(starts)
    start_ts = ts_secs[starts]
    end_ts = ts_secs[starts + L - 1]
    seg_df = create_segments_meta(df_merged, ts_col=ts_col, resample_seconds=resample_seconds, k_median=k_median, gap_seconds_min=gap_seconds_min)
    seg_ids = np.zeros(n, dtype=int)
    if not seg_df.empty:
        seg_start = seg_df["start_ts"].to_numpy()
        seg_end = seg_df["end_ts"].to_numpy()
        seg_id_vals = seg_df["segment_id"].to_numpy()
        for i in range(n):
            m = (start_ts[i] >= seg_start) & (start_ts[i] <= seg_end)
            if np.any(m):
                seg_ids[i] = int(seg_id_vals[np.argmax(m)])
    meta = pd.DataFrame({
        "start_ts": start_ts.astype(np.int64),
        "end_ts": end_ts.astype(np.int64),
        "window_idx": np.arange(n, dtype=np.int64),
        "segment_id": seg_ids.astype(np.int64),
    })
    return meta


def aggregate_aux_features(
    X_seq: np.ndarray,
    df_merged: pd.DataFrame,
    starts: np.ndarray,
    ts_col: str,
    resample_seconds: int,
    k_median: float = 4.0,
    gap_seconds_min: int = 30,
) -> Tuple[np.ndarray, List[str]]:
    """计算每窗口的鲁棒统计特征（NaN感知）：mean/std/median 等辅助特征。"""
    if X_seq.size == 0:
        return np.empty((0, 0), dtype=np.float32), []
    N, L, C = X_seq.shape
    channel_names = ["P_kW", "Q_kvar", "S_kVA", "PF", "dP", "dQ", "dS"]
    if len(channel_names) != C:
        channel_names = [f"ch_{i}" for i in range(C)]
    feats: List[np.ndarray] = []
    names: List[str] = []

    # 基础统计特征
    for ci, cname in enumerate(channel_names):
        x = X_seq[:, :, ci]
        has_valid = np.any(~np.isnan(x), axis=1)
        mu = np.zeros(N, dtype=np.float32)
        sd = np.zeros(N, dtype=np.float32)
        median = np.zeros(N, dtype=np.float32)
        if np.any(has_valid):
            xx = x[has_valid]
            mu[has_valid] = np.nanmean(xx, axis=1).astype(np.float32)
            sd[has_valid] = np.nanstd(xx, axis=1).astype(np.float32)
            median[has_valid] = np.nanmedian(xx, axis=1).astype(np.float32)
        feats.extend([mu, sd, median])
        names.extend([f"mean_{cname}", f"std_{cname}", f"median_{cname}"])
        valid_ratio = (~np.isnan(x)).sum(axis=1) / L
        feats.append(valid_ratio.astype(np.float32))
        names.append(f"valid_ratio_{cname}")

    # 工业电气派生特征
    try:
        P_seq = X_seq[:, :, 0]
        Q_seq = X_seq[:, :, 1]
        PF_seq = X_seq[:, :, 3] if C >= 4 else np.full((N, L), np.nan, dtype=np.float32)
        phi_seq = np.full((N, L), np.nan, dtype=np.float32)
        valid_pq = (~np.isnan(P_seq)) & (~np.isnan(Q_seq)) & (np.abs(P_seq) > 1e-6)
        phi_seq[valid_pq] = np.arctan2(Q_seq[valid_pq], P_seq[valid_pq])
        has_phi = np.any(~np.isnan(phi_seq), axis=1)
        phi_mean = np.zeros(N, dtype=np.float32)
        phi_std = np.zeros(N, dtype=np.float32)
        if np.any(has_phi):
            phi_mean[has_phi] = np.nanmean(phi_seq[has_phi], axis=1).astype(np.float32)
            phi_std[has_phi] = np.nanstd(phi_seq[has_phi], axis=1).astype(np.float32)
        feats.extend([phi_mean, phi_std])
        names.extend(["mean_phi", "std_phi"])
        if L > 1:
            dPF_seq = np.diff(PF_seq, axis=1)
            has_dpf = np.any(~np.isnan(dPF_seq), axis=1)
            dPF_mean = np.zeros(N, dtype=np.float32)
            dPF_std = np.zeros(N, dtype=np.float32)
            if np.any(has_dpf):
                dPF_mean[has_dpf] = np.nanmean(dPF_seq[has_dpf], axis=1).astype(np.float32)
                dPF_std[has_dpf] = np.nanstd(dPF_seq[has_dpf], axis=1).astype(np.float32)
        else:
            dPF_mean = np.zeros(N, dtype=np.float32)
            dPF_std = np.zeros(N, dtype=np.float32)
        feats.extend([dPF_mean, dPF_std])
        names.extend(["mean_dPF", "std_dPF"])
    except Exception:
        pass

    # 能量增量（kWh）
    energy_delta = np.zeros(N, dtype=np.float32)
    if "E_PP_kWh" in df_merged.columns:
        e = df_merged["E_PP_kWh"].to_numpy(dtype=np.float64)
        for i, s in enumerate(starts):
            w = e[s:s + L]
            if w.size > 1:
                a, b = w[0], w[-1]
                if np.isfinite(a) and np.isfinite(b):
                    energy_delta[i] = float(b - a)
    feats.append(energy_delta)
    names.append("energy_delta_kWh")

    # 聚合函数
    def _window_mean_std(columns: list) -> tuple:
        m = np.zeros(N, dtype=np.float32)
        s = np.zeros(N, dtype=np.float32)
        arrs = [df_merged[c].to_numpy(dtype=np.float64) for c in columns if c in df_merged.columns]
        if arrs:
            mat = np.stack(arrs, axis=1)
            avg_ts = np.nanmean(mat, axis=1)
            for i, st in enumerate(starts):
                w = avg_ts[st:st + L]
                if w.size:
                    m[i] = np.nanmean(w)
                    s[i] = np.nanstd(w)
        return m, s

    # Vrms 与 Irms
    vr_m, vr_s = _window_mean_std(["V1_V", "V2_V", "V3_V"])  # 相电压
    ir_m, ir_s = _window_mean_std(["I1_A", "I2_A", "I3_A"])  # 相电流
    feats.extend([vr_m, vr_s, ir_m, ir_s])
    names.extend(["Vrms_mean", "Vrms_std", "Irms_mean", "Irms_std"])

    # 频率
    f_m, f_s = _window_mean_std(["F_Hz"])
    feats.extend([f_m, f_s])
    names.extend(["F_Hz_mean", "F_Hz_std"])

    # THD
    thdv_m, thdv_s = _window_mean_std(["THD_V1_F", "THD_V2_F", "THD_V3_F"])  # 电压 THD
    thdi_m, thdi_s = _window_mean_std(["THD_I1_F", "THD_I2_F", "THD_I3_F"])  # 电流 THD
    feats.extend([thdv_m, thdv_s, thdi_m, thdi_s])
    names.extend(["THD_V_mean", "THD_V_std", "THD_I_mean", "THD_I_std"])

    # 电压不平衡度（std/mean 按相*时间平均）
    v_unbalance = np.zeros(N, dtype=np.float32)
    if all(c in df_merged.columns for c in ["V1_V", "V2_V", "V3_V"]):
        v1 = df_merged["V1_V"].to_numpy(dtype=np.float64)
        v2 = df_merged["V2_V"].to_numpy(dtype=np.float64)
        v3 = df_merged["V3_V"].to_numpy(dtype=np.float64)
        mat = np.stack([v1, v2, v3], axis=1)
        for i, st in enumerate(starts):
            w = mat[st:st + L, :]
            mu_ts = np.nanmean(w, axis=1)
            sd_ts = np.nanstd(w, axis=1)
            with np.errstate(invalid='ignore', divide='ignore'):
                ratio = np.divide(sd_ts, mu_ts, out=np.zeros_like(sd_ts), where=mu_ts != 0)
            v_unbalance[i] = float(np.nanmean(ratio))
    feats.append(v_unbalance)
    names.append("voltage_unbalance_mean")

    # 时间编码与连续段相对位置
    try:
        ts_secs = (pd.to_datetime(df_merged[ts_col]).astype("int64") // 1_000_000_000).to_numpy()
        seg_df = create_segments_meta(
            df_merged,
            ts_col=ts_col,
            resample_seconds=int(resample_seconds),
            k_median=float(k_median),
            gap_seconds_min=int(gap_seconds_min),
        )
        seg_starts = seg_df[["segment_id", "start_idx"]].set_index("segment_id")["start_idx"].to_dict() if not seg_df.empty else {}
        seg_ends = seg_df[["segment_id", "end_idx"]].set_index("segment_id")["end_idx"].to_dict() if not seg_df.empty else {}
        win_meta = build_windows_metadata(
            df_merged,
            starts,
            L,
            ts_col,
            resample_seconds=int(resample_seconds),
            k_median=float(k_median),
            gap_seconds_min=int(gap_seconds_min),
        )
        seg_ids = win_meta["segment_id"].to_numpy(dtype=int)
        dt_med = np.zeros(N, dtype=np.float32)
        rel_pos = np.zeros(N, dtype=np.float32)
        seg_len = np.zeros(N, dtype=np.float32)
        for i, st in enumerate(starts):
            ed = min(len(ts_secs), st + L)
            d = np.diff(ts_secs[st:ed])
            dt_med[i] = float(np.median(d[d > 0])) if d.size and np.any(d > 0) else float(max(1, int(resample_seconds)))
            sid = int(seg_ids[i]) if i < len(seg_ids) else -1
            a = seg_starts.get(sid, st)
            b = seg_ends.get(sid, ed - 1)
            length = max(1, int(b - a + 1))
            seg_len[i] = float(length)
            rel_pos[i] = float(st - a) / float(max(1, length - 1))
        feats.extend([dt_med, rel_pos, seg_len])
        names.extend(["dt_median_s", "segment_rel_pos", "segment_len_samples"])
    except Exception:
        pass

    # 总体有效比例（跨全部通道）
    valid_overall = np.isfinite(X_seq).sum(axis=(1, 2)) / float(max(1, L * C))
    feats.append(valid_overall.astype(np.float32))
    names.append("valid_ratio_overall")

    Fd = np.stack(feats, axis=1).astype(np.float32)
    Fd = np.nan_to_num(Fd, nan=0.0, posinf=1e6, neginf=-1e6)
    invalid_count = np.sum(~np.isfinite(Fd))
    if invalid_count > 0:
        print(f"警告: 辅助特征中发现 {invalid_count} 个无效值，已自动清理")
    return Fd, names
