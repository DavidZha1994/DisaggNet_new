import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


def _to_numpy(x):
    try:
        if hasattr(x, 'detach'):
            x = x.detach().cpu()
        if hasattr(x, 'numpy'):
            return x.numpy()
        return np.asarray(x)
    except Exception:
        return np.asarray(x)


def reconstruct_dense_curve(
    predictions_per_offset: Dict[int, np.ndarray],
    stride: int,
    total_length: int,
) -> np.ndarray:
    if total_length <= 0 or stride <= 0 or not predictions_per_offset:
        return np.zeros((0,), dtype=np.float32)
    ks = []
    for v in predictions_per_offset.values():
        arr = _to_numpy(v)
        if arr.ndim == 1:
            ks.append(1)
        elif arr.ndim >= 2:
            ks.append(int(arr.shape[-1]))
    k_out = max(ks) if ks else 1
    dense = np.full((total_length, k_out), np.nan, dtype=np.float32)
    count = np.zeros((total_length, 1), dtype=np.float32)
    for offset, vals in predictions_per_offset.items():
        arr = _to_numpy(vals)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        n, k = arr.shape
        if n == 0:
            continue
        if k < k_out:
            pad = np.zeros((n, k_out - k), dtype=arr.dtype)
            arr = np.concatenate([arr, pad], axis=1)
        idx = offset + np.arange(n, dtype=np.int64) * int(stride)
        mask = (idx >= 0) & (idx < total_length)
        if not np.any(mask):
            continue
        idx = idx[mask]
        arr_valid = arr[mask]
        cur = dense[idx]
        cur = np.nan_to_num(cur, nan=0.0)
        dense[idx] = cur + arr_valid
        count[idx, 0] += 1.0
    valid = count[:, 0] > 0
    if not np.any(valid):
        if k_out == 1:
            return np.full((total_length,), np.nan, dtype=np.float32)
        return np.full((total_length, k_out), np.nan, dtype=np.float32)
    dense_valid = dense[valid] / count[valid]
    dense[valid] = dense_valid.astype(np.float32)
    if k_out == 1:
        return dense[:, 0]
    return dense


def save_validation_interactive_plot(
    buffers: List[Dict[str, Any]],
    device_names: List[str],
    vis_output_dir: str,
    dataset_name: str,
    fold_id: int,
    epoch: int,
    max_power: Optional[np.ndarray] = None,
) -> Path:
    if len(buffers) == 0:
        return Path()
    bufs = sorted(buffers, key=lambda d: float(d.get('start', 0.0)))

    def _get_ln(d):
        x = d.get('pred')
        y = d.get('true')
        xn = _to_numpy(x)
        yn = _to_numpy(y)
        if xn.ndim >= 2:
            return int(xn.shape[0])
        if yn.ndim >= 2:
            return int(yn.shape[0])
        return int(xn.shape[0] if xn.ndim == 1 else yn.shape[0])

    def _get_k(d):
        x = _to_numpy(d.get('pred'))
        y = _to_numpy(d.get('true'))
        xk = int(x.shape[1]) if x.ndim >= 2 else 1
        yk = int(y.shape[1]) if y.ndim >= 2 else 1
        return max(xk, yk)

    K = 1
    for d in bufs:
        K = max(K, _get_k(d))

    total_points = int(sum(max(0, _get_ln(d)) for d in bufs))
    if total_points <= 0:
        return Path()
    mains_concat = np.full((total_points,), np.nan, dtype=np.float32)
    pred_concat = np.full((total_points, K), np.nan, dtype=np.float32)
    true_concat = np.full((total_points, K), np.nan, dtype=np.float32)
    unknown_concat = np.full((total_points,), np.nan, dtype=np.float32)
    idx_cur = 0
    for d in bufs:
        ln = _get_ln(d)
        if ln <= 0:
            continue
        p = _to_numpy(d.get('pred'))
        t = _to_numpy(d.get('true'))
        vmask = d.get('valid')
        vmask_np = None
        try:
            if vmask is not None:
                vmask_np = _to_numpy(vmask).astype(np.float32)
                if vmask_np.ndim > 1:
                    vmask_np = np.squeeze(vmask_np)
        except Exception:
            vmask_np = None
        if p.ndim == 1:
            p = p.reshape(-1, 1)
        if t.ndim == 1:
            t = t.reshape(-1, 1)
        pk = int(p.shape[1])
        tk = int(t.shape[1])
        m = d.get('mains')
        mm = None
        if m is not None:
            mm = _to_numpy(m)
        if mm is None:
            mm = np.full((ln,), np.nan, dtype=np.float32)
        else:
            if mm.ndim > 1:
                mm = mm.astype(np.float32)
                try:
                    mm = mm.sum(axis=-1)
                except Exception:
                    mm = np.squeeze(mm)
            else:
                mm = mm.astype(np.float32)
            mm = mm[:ln]
            if mm.shape[0] < ln:
                pad_len = ln - mm.shape[0]
                pad = np.full((pad_len,), np.nan, dtype=np.float32)
                mm = np.concatenate([mm, pad], axis=0)
            try:
                if vmask_np is not None and vmask_np.shape[0] >= ln:
                    vv = vmask_np[:ln]
                    mm = np.where(vv > 0.5, mm, np.nan)
            except Exception:
                pass
        sl = slice(idx_cur, idx_cur + ln)
        if pk > 0:
            new_pred = p[:ln, :pk].astype(np.float32)
            if vmask_np is not None and vmask_np.shape[0] >= ln:
                vv = vmask_np[:ln]
                new_pred = np.where(vv[:, None] > 0.5, new_pred, np.nan)
            pred_concat[sl, :pk] = new_pred
        if tk > 0:
            new_true = t[:ln, :tk].astype(np.float32)
            if vmask_np is not None and vmask_np.shape[0] >= ln:
                vv = vmask_np[:ln]
                new_true = np.where(vv[:, None] > 0.5, new_true, np.nan)
            true_concat[sl, :tk] = new_true
        mains_concat[sl] = mm[:ln].astype(np.float32)

        u = d.get('unknown')
        if u is not None:
            try:
                uu = _to_numpy(u).astype(np.float32)
                if uu.ndim > 1:
                    uu = uu.reshape(-1)
                uu = uu[:ln]
                if uu.shape[0] < ln:
                    pad_len = ln - uu.shape[0]
                    pad = np.full((pad_len,), np.nan, dtype=np.float32)
                    uu = np.concatenate([uu, pad], axis=0)
                if vmask_np is not None and vmask_np.shape[0] >= ln:
                    vv = vmask_np[:ln]
                    uu = np.where(vv > 0.5, uu, np.nan)
                unknown_concat[sl] = uu
            except Exception:
                pass
        idx_cur += ln
    try:
        valid_m = np.isfinite(mains_concat)
        valid_p = np.isfinite(pred_concat).any(axis=1)
        valid_t = np.isfinite(true_concat).any(axis=1)
        valid_u = np.isfinite(unknown_concat)
        global_valid = valid_m | valid_p | valid_t | valid_u
    except Exception:
        global_valid = np.ones((total_points,), dtype=bool)
    if not np.any(global_valid):
        global_valid = np.ones((total_points,), dtype=bool)
    idx_all = np.where(global_valid)[0]
    mains_concat = mains_concat[idx_all]
    pred_concat = pred_concat[idx_all, :]
    true_concat = true_concat[idx_all, :]
    unknown_concat = unknown_concat[idx_all]
    total_points = int(mains_concat.shape[0])
    max_points = 200000
    if total_points > max_points:
        stride = int(np.ceil(total_points / float(max_points)))
        idx = np.arange(0, total_points, stride, dtype=np.int64)
        mains_concat = mains_concat[idx]
        pred_concat = pred_concat[idx, :]
        true_concat = true_concat[idx, :]
        unknown_concat = unknown_concat[idx]
    else:
        idx = np.arange(total_points, dtype=np.int64)
    x_index = np.arange(int(mains_concat.shape[0]), dtype=np.int64)
    hover_text = [str(int(i)) for i in x_index]

    try:
        titles = [str(device_names[i]) for i in range(K)]
    except Exception:
        titles = [f'Device_{i}' for i in range(K)]

    has_unknown = np.isfinite(unknown_concat).any()
    rows_total = int(K + 1 + (1 if has_unknown else 0))
    max_vs = (1.0 / max(rows_total - 1, 1)) - 1e-6
    vs = min(0.02, max(0.001, max_vs * 0.9))
    subplot_titles = ["总功率"] + titles
    if has_unknown:
        subplot_titles = subplot_titles + ["未知功率"]

    fig = make_subplots(
        rows=rows_total,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=vs,
        subplot_titles=subplot_titles,
    )
    fig.add_trace(
        go.Scatter(
            x=x_index,
            y=mains_concat,
            name="总功率",
            mode="lines",
            line=dict(color="#7f7f7f"),
            legendgroup="mains",
            showlegend=True,
            text=hover_text,
            hovertemplate="时间: %{text}<br>功率=%{y:.2f}W<extra></extra>",
        ),
        row=1,
        col=1,
    )

    for i in range(K):
        fig.add_trace(
            go.Scatter(
                x=x_index,
                y=true_concat[:, i],
                name="目标功率",
                mode="lines",
                line=dict(color="#2ca02c"),
                legendgroup="target",
                showlegend=True if i == 0 else False,
                text=hover_text,
                hovertemplate="时间: %{text}<br>功率=%{y:.2f}W<extra></extra>",
            ),
            row=i + 2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_index,
                y=pred_concat[:, i],
                name="预测功率",
                mode="lines",
                line=dict(color="#ff7f0e"),
                legendgroup="prediction",
                showlegend=True if i == 0 else False,
                text=hover_text,
                hovertemplate="时间: %{text}<br>功率=%{y:.2f}W<extra></extra>",
            ),
            row=i + 2,
            col=1,
        )
        try:
            vals = np.stack([true_concat[:, i], pred_concat[:, i]], axis=0).reshape(-1)
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                q = float(np.nanpercentile(vals, 99.5))
                upper = float(max(np.nanmax(vals), q))
                if np.isfinite(upper) and upper > 0:
                    fig.update_yaxes(range=[0.0, upper * 1.15], row=i + 2, col=1)
        except Exception:
            pass

    if has_unknown:
        fig.add_trace(
            go.Scatter(
                x=x_index,
                y=unknown_concat,
                name="未知功率",
                mode="lines",
                line=dict(color="#1f77b4"),
                legendgroup="unknown",
                showlegend=True,
                text=hover_text,
                hovertemplate="时间: %{text}<br>功率=%{y:.2f}W<extra></extra>",
            ),
            row=rows_total,
            col=1,
        )

    try:
        mvals = mains_concat[np.isfinite(mains_concat)]
        if mvals.size > 0:
            q_m = float(np.nanpercentile(mvals, 99.5))
            u_m = float(max(np.nanmax(mvals), q_m))
            if np.isfinite(u_m) and u_m > 0:
                fig.update_yaxes(range=[0.0, u_m * 1.15], row=1, col=1)
    except Exception:
        pass

    if has_unknown:
        try:
            uvals = unknown_concat[np.isfinite(unknown_concat)]
            if uvals.size > 0:
                q_u = float(np.nanpercentile(uvals, 99.5))
                u_u = float(max(np.nanmax(uvals), q_u))
                if np.isfinite(u_u) and u_u > 0:
                    fig.update_yaxes(range=[0.0, u_u * 1.15], row=rows_total, col=1)
        except Exception:
            pass

    fig.update_layout(
        template="plotly_white",
        legend=dict(orientation="h"),
        height=int(220 * rows_total),
    )
    try:
        for r in range(1, rows_total + 1):
            fig.update_xaxes(type='linear', row=r, col=1)
    except Exception:
        pass
    out_dir = (
        Path(vis_output_dir)
        / "val_interactive"
        / str(dataset_name)
        / f"fold_{int(fold_id)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"epoch_{int(epoch):04d}.html"
    pio.write_html(fig, file=str(fp), include_plotlyjs='cdn', auto_open=False)
    return fp
