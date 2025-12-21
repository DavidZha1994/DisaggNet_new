import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from .io import safe_to_csv


def repair_small_gaps(
    df: pd.DataFrame,
    ts_col: str,
    resample_seconds: int,
    max_fill_seconds: int = 60,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    填补短缺口并返回缺口掩码（仅主端列记录插值位置）。
    - 最大缺口长度由 `max_fill_seconds` 控制（按步长换算为点数限制）。
    - 返回: (df_filled, gap_mask_df)，gap_mask_df 仅含主端列（True=由插值填补）。
    """
    x = df.copy()
    step_s = max(1, int(resample_seconds))
    limit = max(0, int(np.floor(int(max_fill_seconds) / step_s)))
    # 主端列集合
    # 固定主端列顺序，缺失列以零掩码占位，保证后续窗口级掩码通道一致
    all_mains = ["P_kW", "Q_kvar", "S_kVA", "PF"]
    mains_cols_present = [c for c in all_mains if c in x.columns]
    mask_dict = {c: np.zeros(len(x), dtype=bool) for c in mains_cols_present}

    for c in x.columns:
        if c == ts_col:
            continue
        ser = x[c]
        if ser.dtype.kind in 'bif':
            before_nan = ser.isna().to_numpy()
            try:
                x[c] = ser.ffill(limit=limit)
            except Exception:
                # 插值失败不影响其他列
                pass
            after_nan = x[c].isna().to_numpy()
            if c in mask_dict:
                mask_dict[c] = (~after_nan) & before_nan

    # 完整主端掩码：为未出现的主端列补零，并固定列顺序为 P/Q/S/PF
    if len(all_mains) > 0:
        for c in all_mains:
            if c not in mask_dict:
                mask_dict[c] = np.zeros(len(x), dtype=bool)
        gap_mask_df = pd.DataFrame({c: mask_dict[c] for c in all_mains}, index=x.index)
    else:
        gap_mask_df = pd.DataFrame(index=x.index)
    return x, gap_mask_df


def export_gap_repair_report(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    gap_mask_df: pd.DataFrame,
    eff_dev_names: List[str],
    output_dir: str,
    ts_col: str,
) -> None:
    """生成修复短缺口报告到 prepared/quality/gap_repair_report.csv。
    - 主端列：使用 gap_mask_df 统计填补点数与残余缺口率。
    - 设备列：用非NaN计数变化估计填补量，同时记录覆盖率变化。
    """
    try:
        out_dir = os.path.join(output_dir, "quality")
        os.makedirs(out_dir, exist_ok=True)
        total = len(df_after)
        rows = []
        # 主端列
        mains_cols = [c for c in ["P_kW", "Q_kvar", "S_kVA", "PF"] if c in df_after.columns]
        for c in mains_cols:
            filled = int(gap_mask_df[c].sum()) if c in gap_mask_df.columns else 0
            before_nan = int(df_before[c].isna().sum()) if c in df_before.columns else 0
            after_nan = int(df_after[c].isna().sum()) if c in df_after.columns else 0
            rows.append({
                "type": "mains",
                "column": c,
                "filled_points": filled,
                "before_nan": before_nan,
                "after_nan": after_nan,
                "total_points": int(total),
                "filled_ratio": round(filled / float(max(1, total)), 8),
                "residual_gap_ratio": round(after_nan / float(max(1, total)), 8)
            })
        # 设备列
        for name in eff_dev_names:
            for c in [f"{name}_P_kW", f"{name}_Q_kvar", f"{name}_S_kVA"]:
                if c in df_after.columns and c in df_before.columns:
                    before_non_nan = int(np.isfinite(df_before[c]).sum())
                    after_non_nan = int(np.isfinite(df_after[c]).sum())
                    filled = max(0, after_non_nan - before_non_nan)
                    before_nan = int(df_before[c].isna().sum())
                    after_nan = int(df_after[c].isna().sum())
                    rows.append({
                        "type": "device",
                        "column": c,
                        "filled_points": int(filled),
                        "before_nan": int(before_nan),
                        "after_nan": int(after_nan),
                        "total_points": int(total),
                        "filled_ratio": round(filled / float(max(1, total)), 8),
                        "coverage_ratio_before": round(before_non_nan / float(max(1, total)), 8),
                        "coverage_ratio_after": round(after_non_nan / float(max(1, total)), 8)
                    })
        safe_to_csv(pd.DataFrame(rows), os.path.join(out_dir, "gap_repair_report.csv"))
    except Exception:
        # 报告生成失败不影响主流程
        pass
