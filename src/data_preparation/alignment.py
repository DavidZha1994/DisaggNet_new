import os
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from .io import safe_to_csv


def resample_df(
    df: pd.DataFrame,
    cols: List[str],
    ts_col: str,
    rule_seconds: int,
    use_polars: bool = False,
) -> pd.DataFrame:
    """按统一秒级网格重采样并补全时间网格（保留NaN显式缺口）。"""
    if use_polars:
        try:
            import polars as pl
            keep_cols = [ts_col] + [c for c in cols if c in df.columns]
            dfp = pl.from_pandas(df[keep_cols])
            if dfp.schema.get(ts_col) != pl.Datetime:
                dfp = dfp.with_columns(pl.col(ts_col).str.strptime(pl.Datetime, strict=False))
            dfp = dfp.with_columns(pl.col(ts_col).dt.replace_time_zone(None))
            step_ms = int(rule_seconds * 1000)
            half_ms = int(step_ms // 2)
            dfp = dfp.with_columns([
                pl.from_epoch(
                    ((pl.col(ts_col).dt.epoch(time_unit='ms') + half_ms) // step_ms) * step_ms,
                    time_unit='ms'
                ).alias("_snap_ts")
            ])
            out = (
                dfp.group_by("_snap_ts")
                .agg([pl.col(c).mean().alias(c) for c in cols if c in dfp.columns])
                .sort("_snap_ts")
            )
            x = out.to_pandas()
            if len(x) > 0:
                start = pd.to_datetime(x["_snap_ts"].min())
                end = pd.to_datetime(x["_snap_ts"].max())
                grid = pd.date_range(start=start, end=end, freq=f"{rule_seconds}s")
                xr = x.set_index("_snap_ts").reindex(grid)
                xr[ts_col] = xr.index
                xr = xr.reset_index(drop=True)
                return xr
            else:
                return pd.DataFrame({ts_col: pd.to_datetime([])})
        except Exception:
            pass

    x = df.copy()
    x[ts_col] = pd.to_datetime(x[ts_col], errors="coerce", utc=True).dt.tz_localize(None)
    x = x.dropna(subset=[ts_col])
    present_cols = [c for c in cols if c in x.columns]
    x["_snap_ts"] = x[ts_col].dt.round(f"{rule_seconds}s")
    agg = x.groupby("_snap_ts", as_index=False)[present_cols].mean()
    if len(agg) > 0:
        start = pd.to_datetime(agg["_snap_ts"].min())
        end = pd.to_datetime(agg["_snap_ts"].max())
        grid = pd.date_range(start=start, end=end, freq=f"{rule_seconds}s")
        out = agg.set_index("_snap_ts").reindex(grid)
        out[ts_col] = out.index
        out = out.reset_index(drop=True)
    else:
        out = pd.DataFrame({ts_col: pd.to_datetime([])})
    return out.sort_values(ts_col).reset_index(drop=True)


def resample_rename_device(
    df: pd.DataFrame,
    name: str,
    ts_col: str,
    rule_seconds: int,
    use_polars: bool = False,
) -> Optional[pd.DataFrame]:
    cols = [c for c in ["P_kW", "Q_kvar", "S_kVA"] if c in df.columns]
    if "P_kW" not in cols:
        return None
    if not cols:
        return None
    dfr = resample_df(df, cols, ts_col=ts_col, rule_seconds=rule_seconds, use_polars=use_polars)
    ren = {}
    if "P_kW" in dfr.columns:
        ren["P_kW"] = f"{name}_P_kW"
    if "Q_kvar" in dfr.columns:
        ren["Q_kvar"] = f"{name}_Q_kvar"
    if "S_kVA" in dfr.columns:
        ren["S_kVA"] = f"{name}_S_kVA"
    dfr = dfr.rename(columns=ren)
    return dfr


def align_and_merge(
    df_main: pd.DataFrame,
    dev_dfs: List[pd.DataFrame],
    dev_names: List[str],
    ts_col: str,
    rule_seconds: int,
    n_jobs: int = 1,
    use_polars: bool = False,
) -> Tuple[pd.DataFrame, Dict[int, str]]:
    main_cols = [c for c in [
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
    ] if c in df_main.columns]
    dfm = resample_df(df_main, main_cols, ts_col=ts_col, rule_seconds=rule_seconds, use_polars=use_polars)
    base = dfm.copy()
    merged_names: List[str] = []
    futures = []
    with ThreadPoolExecutor(max_workers=max(1, int(n_jobs))) as ex:
        for df, name in zip(dev_dfs, dev_names):
            futures.append((name, ex.submit(resample_rename_device, df, name, ts_col, rule_seconds, use_polars)))
        for name, fut in futures:
            dfr = fut.result()
            if dfr is None:
                continue
            base = base.merge(dfr, on=ts_col, how="left", sort=False)
            merged_names.append(name)
    base = base.sort_values(ts_col).reset_index(drop=True)
    label_map = {i: name for i, name in enumerate(merged_names)}
    return base, label_map


def export_alignment_drift(
    df_main_raw: pd.DataFrame,
    dev_raw_list: List[pd.DataFrame],
    dev_names: List[str],
    ts_col: str,
    resample_seconds: int,
    output_dir: str,
) -> None:
    """导出毫秒级对齐漂移诊断到 quality/alignment_drift.csv。"""
    try:
        step_ms = int(resample_seconds) * 1000
        half_ms = step_ms // 2

        def _drift_stats(df: pd.DataFrame) -> Tuple[float, float, float]:
            dt = pd.to_datetime(df[ts_col], errors='coerce', utc=True).dt.tz_localize(None)
            ms = (dt.astype('int64') // 1_000_000).to_numpy()
            snap = ((ms + half_ms) // step_ms) * step_ms
            diff = ms - snap
            if diff.size == 0:
                return 0.0, 0.0, 0.0
            return float(np.mean(np.abs(diff))), float(np.std(diff)), float(np.max(np.abs(diff)))

        rows = []
        m_mean, m_std, m_max = _drift_stats(df_main_raw)
        rows.append({"name": "__MAIN__", "mean_abs_ms": m_mean, "std_ms": m_std, "max_abs_ms": m_max})
        for df, name in zip(dev_raw_list, dev_names):
            d_mean, d_std, d_max = _drift_stats(df)
            rows.append({"name": name, "mean_abs_ms": d_mean, "std_ms": d_std, "max_abs_ms": d_max})

        qdir = os.path.join(output_dir, "quality")
        os.makedirs(qdir, exist_ok=True)
        safe_to_csv(pd.DataFrame(rows), os.path.join(qdir, "alignment_drift.csv"))
    except Exception:
        pass