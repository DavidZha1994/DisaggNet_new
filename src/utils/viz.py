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

    times = []
    mains_list = []
    pred_series = [[] for _ in range(K)]
    true_series = [[] for _ in range(K)]
    last_end_ts = None

    for d in bufs:
        ln = _get_ln(d)
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
        for i in range(K):
            if i < pk:
                arr = p[:ln, i].astype(np.float32)
                if vmask_np is not None and vmask_np.shape[0] >= ln:
                    vv = vmask_np[:ln]
                    arr = np.where(vv > 0.5, arr, np.nan)
                pred_series[i].append(arr)
            else:
                pred_series[i].append(np.full((ln,), np.nan, dtype=np.float32))
            if i < tk:
                arrt = t[:ln, i].astype(np.float32)
                if vmask_np is not None and vmask_np.shape[0] >= ln:
                    vv = vmask_np[:ln]
                    arrt = np.where(vv > 0.5, arrt, np.nan)
                true_series[i].append(arrt)
            else:
                true_series[i].append(np.full((ln,), np.nan, dtype=np.float32))

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
            # 若提供有效掩码，则屏蔽无效位置，避免主表为 0 与分设备不一致
            try:
                if vmask_np is not None and vmask_np.shape[0] >= ln:
                    vv = vmask_np[:ln]
                    mm = np.where(vv > 0.5, mm, np.nan)
            except Exception:
                pass
        start = float(d.get('start', 0.0))
        step = float(d.get('step', 1.0))
        ts = start + np.arange(ln, dtype=np.float64) * step
        if last_end_ts is not None:
            cont = np.isclose(start, last_end_ts + step, atol=max(step * 0.01, 1e-6))
            if not cont:
                times.append(np.array([np.nan], dtype=np.float64))
                mains_list.append(np.array([np.nan], dtype=np.float32))
                for i in range(K):
                    pred_series[i].append(np.full((1,), np.nan, dtype=np.float32))
                    true_series[i].append(np.full((1,), np.nan, dtype=np.float32))
        times.append(ts)
        mains_list.append(mm)
        last_end_ts = float(ts[-1]) if ts.size > 0 else start

    timeline = np.concatenate(times, axis=0)
    try:
        timeline_dt = [datetime.utcfromtimestamp(float(v)) if np.isfinite(v) else None for v in timeline]
    except Exception:
        try:
            timeline_dt = [None if not np.isfinite(v) else np.datetime64(int(v), 's') for v in timeline]
        except Exception:
            timeline_dt = [None if not np.isfinite(v) else float(v) for v in timeline]

    mains_concat = np.concatenate(mains_list, axis=0)
    pred_concat = np.stack(
        [np.concatenate(pred_series[i], axis=0) for i in range(K)],
        axis=1,
    )
    true_concat = np.stack(
        [np.concatenate(true_series[i], axis=0) for i in range(K)],
        axis=1,
    )

    x_index = np.arange(int(mains_concat.shape[0]), dtype=np.int64)
    def _fmt_time(v):
        try:
            if v is None:
                return ""
            if isinstance(v, datetime):
                return v.strftime("%Y-%m-%d %H:%M:%S")
            if isinstance(v, np.datetime64):
                try:
                    s = np.datetime_as_string(v, unit='s')
                    return s.replace("T", " ")
                except Exception:
                    return str(v)
            if isinstance(v, (int, float)):
                if np.isfinite(v):
                    dt = datetime.utcfromtimestamp(float(v))
                    return dt.strftime("%Y-%m-%d %H:%M:%S")
                return ""
            return str(v)
        except Exception:
            try:
                return str(v) if v is not None else ""
            except Exception:
                return ""
    hover_text = [ _fmt_time(v) for v in timeline_dt ]

    try:
        titles = [str(device_names[i]) for i in range(K)]
    except Exception:
        titles = [f'Device_{i}' for i in range(K)]

    rows_total = int(K + 1)
    max_vs = (1.0 / max(rows_total - 1, 1)) - 1e-6
    vs = min(0.02, max(0.001, max_vs * 0.9))
    fig = make_subplots(
        rows=rows_total,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=vs,
        subplot_titles=["总功率"] + titles,
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

    try:
        mvals = mains_concat[np.isfinite(mains_concat)]
        if mvals.size > 0:
            q_m = float(np.nanpercentile(mvals, 99.5))
            u_m = float(max(np.nanmax(mvals), q_m))
            if np.isfinite(u_m) and u_m > 0:
                fig.update_yaxes(range=[0.0, u_m * 1.15], row=1, col=1)
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
