#!/usr/bin/env python3
import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def downsample_ts(ts: pd.Series, y: pd.Series, rule: str = '5min') -> Tuple[pd.Series, pd.Series]:
    try:
        df = pd.DataFrame({'ts': pd.to_datetime(ts, errors='coerce'), 'y': y})
        df = df.dropna(subset=['ts'])
        df = df.set_index('ts')
        agg = df.resample(rule).mean(numeric_only=True)
        return agg.index, agg['y']
    except Exception:
        n = len(ts)
        step = max(1, n // 5000)
        idx = np.arange(0, n, step)
        return ts.iloc[idx], y.iloc[idx]


@dataclass
class VizConfig:
    max_devices: int = 10
    resample_rule: str = '5min'
    sample_windows: int = 3


class PipelineVisualizer:
    def __init__(self, out_dir: str, cfg: Optional[VizConfig] = None):
        self.out_dir = out_dir
        self.cfg = cfg or VizConfig()
        ensure_dir(self.out_dir)

    def plot_mains_pqs(self, df_main: pd.DataFrame, ts_col: str = 'timestamp'):
        out = os.path.join(self.out_dir, '01_mains_pqs_overview.png')
        if ts_col not in df_main.columns:
            return
        ts = pd.to_datetime(df_main[ts_col], errors='coerce')
        ts = ts.dt.tz_localize(None)
        channels = [c for c in ['P_kW', 'Q_kvar', 'S_kVA', 'PF'] if c in df_main.columns]
        if not channels:
            return
        fig, axes = plt.subplots(len(channels), 1, figsize=(12, 3 * len(channels)), sharex=True)
        axes = np.atleast_1d(axes)
        for i, c in enumerate(channels):
            x, y = downsample_ts(ts, df_main[c], rule=self.cfg.resample_rule)
            axes[i].plot(x, y, lw=0.8)
            axes[i].set_ylabel(c)
        axes[-1].set_xlabel('timestamp')
        fig.suptitle('Mains P/Q/S/PF overview')
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)

    def plot_devices_p(self, df_merged: pd.DataFrame, label_map: Dict[int, str], ts_col: str = 'timestamp'):
        out = os.path.join(self.out_dir, '02_devices_P_overview.png')
        if ts_col not in df_merged.columns:
            return
        ts = pd.to_datetime(df_merged[ts_col], errors='coerce')
        ts = ts.dt.tz_localize(None)
        names = [label_map[i] for i in sorted(label_map.keys())][: self.cfg.max_devices]
        cols = [f'{nm}_P_kW' for nm in names if f'{nm}_P_kW' in df_merged.columns]
        if not cols:
            return
        n = len(cols)
        fig, axes = plt.subplots(n, 1, figsize=(12, 2.2 * n), sharex=True)
        axes = np.atleast_1d(axes)
        for i, c in enumerate(cols):
            x, y = downsample_ts(ts, df_merged[c], rule=self.cfg.resample_rule)
            axes[i].plot(x, y, lw=0.7)
            axes[i].set_ylabel(c)
        axes[-1].set_xlabel('timestamp')
        fig.suptitle('Top devices P overview')
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)

    def plot_missing_heatmap(self, df_before: pd.DataFrame, df_after: pd.DataFrame, ts_col: str = 'timestamp'):
        out = os.path.join(self.out_dir, '03_missing_gaps_timeline.png')
        if ts_col not in df_before.columns:
            return
        # 时间轴（无时区）
        ts_before = pd.to_datetime(df_before[ts_col], errors='coerce').dt.tz_localize(None)
        # 估计采样周期（秒），用于缺口阈值
        diffs = ts_before.diff().dt.total_seconds()
        step_seconds = float(np.nanmedian(diffs[diffs > 0])) if np.isfinite(diffs).any() else 5.0
        gap_threshold = step_seconds  # 仅标记连续缺口时间 > 一个采样周期

        # 选择主要数值列
        numeric_cols = [c for c in df_before.columns if c != ts_col and df_before[c].dtype.kind in 'bif']
        if not numeric_cols:
            return
        key_cols = numeric_cols[:min(4, len(numeric_cols))]

        fig, axes = plt.subplots(len(key_cols), 1, figsize=(14, 3 * len(key_cols)), sharex=True)
        axes = np.atleast_1d(axes)

        for i, col in enumerate(key_cols):
            valid_mask = ~df_before[col].isna()
            # 下采样后仅展示有效数据曲线
            x_before, y_before = downsample_ts(ts_before[valid_mask], df_before[col][valid_mask])
            axes[i].plot(x_before, y_before, 'b-', alpha=0.8, linewidth=0.8, label='Valid data')

            # 连续缺口分段（只标记长度>阈值的缺口）
            missing = ~valid_mask
            in_gap = False
            gap_start_idx = None

            for j in range(len(missing)):
                if missing.iloc[j] and not in_gap:
                    in_gap = True
                    gap_start_idx = j
                elif not missing.iloc[j] and in_gap:
                    gap_end_idx = j - 1
                    # 缺口持续时间
                    gap_duration = (ts_before.iloc[gap_end_idx] - ts_before.iloc[gap_start_idx]).total_seconds()
                    if gap_duration > gap_threshold + 1e-6:  # 超过一个采样周期
                        axes[i].axvspan(ts_before.iloc[gap_start_idx], ts_before.iloc[gap_end_idx],
                                        alpha=0.25, color='red', label='Missing data' if gap_start_idx == 0 else "")
                    in_gap = False
                    gap_start_idx = None
            # 尾部缺口
            if in_gap and gap_start_idx is not None:
                gap_end_idx = len(missing) - 1
                gap_duration = (ts_before.iloc[gap_end_idx] - ts_before.iloc[gap_start_idx]).total_seconds()
                if gap_duration > gap_threshold + 1e-6:
                    axes[i].axvspan(ts_before.iloc[gap_start_idx], ts_before.iloc[gap_end_idx],
                                    alpha=0.25, color='red', label='Missing data' if gap_start_idx == 0 else "")

            axes[i].set_title(f'{col} - Missing data gaps in timeline (> {int(round(gap_threshold))}s)')
            axes[i].set_ylabel(col)
            if i == 0:
                axes[i].legend()

        axes[-1].set_xlabel('Time')
        fig.suptitle('Missing Data Gaps Timeline View', fontsize=14)
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def plot_window_boundaries(self, df_merged: pd.DataFrame, starts: np.ndarray, L: int, ts_col: str = 'timestamp'):
        out = os.path.join(self.out_dir, '04_window_boundaries.png')
        if ts_col not in df_merged.columns or len(starts) == 0:
            return
        ts = pd.to_datetime(df_merged[ts_col], errors='coerce')
        ts = ts.dt.tz_localize(None)
        if 'P_kW' not in df_merged.columns:
            return
        x, y = downsample_ts(ts, df_merged['P_kW'], rule=self.cfg.resample_rule)
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(x, y, lw=0.8, color='steelblue')
        stride = max(1, len(starts) // 200)
        for s in starts[::stride]:
            s_idx = int(s)
            if s_idx < len(ts):
                s_ts = ts.iloc[s_idx]
                ax.axvline(pd.to_datetime(s_ts), color='red', alpha=0.2, lw=0.5)
        ax.set_title('Window boundaries overlay on P_kW')
        ax.set_xlabel('timestamp')
        ax.set_ylabel('P_kW')
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)

    def plot_stft_samples(self, freq_feats: np.ndarray, n_fft: int, hop: int):
        out = os.path.join(self.out_dir, '05_stft_spectrograms.png')
        if freq_feats.size == 0:
            return
        N = freq_feats.shape[0]
        K = min(self.cfg.sample_windows, N)
        fig, axes = plt.subplots(K, 1, figsize=(10, 3*K))
        axes = np.atleast_1d(axes)
        for i in range(K):
            im = axes[i].imshow(freq_feats[i].T, aspect='auto', origin='lower', cmap='magma')
            axes[i].set_title(f'Window {i} STFT magnitude (n_fft={n_fft}, hop={hop})')
            axes[i].set_xlabel('frame')
            axes[i].set_ylabel('bin x channel')
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        try:
            frames = freq_feats.shape[1]
            with open(os.path.join(self.out_dir, '05_stft_mapping.json'), 'w') as f:
                json.dump({'frames_per_window': int(frames), 'n_fft': int(n_fft), 'hop': int(hop)}, f, indent=2)
        except Exception:
            pass

    def plot_aux_feature_hist(self, aux_feats: np.ndarray, aux_names: List[str]):
        out = os.path.join(self.out_dir, '06_aux_features_hist.png')
        if aux_feats.size == 0:
            return
        m = aux_feats.shape[1]
        cols = min(4, m)
        rows = int(np.ceil(m / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(12, 2.5*rows))
        axes = np.atleast_2d(axes)
        for idx in range(m):
            r = idx // cols
            c = idx % cols
            ax = axes[r, c]
            vals = aux_feats[:, idx]
            vals = vals[np.isfinite(vals)]
            ax.hist(vals, bins=50, color='gray')
            name = aux_names[idx] if idx < len(aux_names) else f'feat_{idx}'
            ax.set_title(name)
        for idx in range(m, rows*cols):
            r = idx // cols
            c = idx % cols
            axes[r, c].axis('off')
        fig.suptitle('Aux feature distributions')
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)

    def plot_walk_forward(self, windows_meta: pd.DataFrame, folds: List[Dict]):
        out = os.path.join(self.out_dir, '07_walk_forward_splits.png')
        if windows_meta is None or len(windows_meta) == 0:
            return
        
        # 创建子图，每个fold一行
        n_folds = len(folds)
        fig, axes = plt.subplots(n_folds, 1, figsize=(12, 2 * n_folds), sharex=True)
        if n_folds == 1:
            axes = [axes]
        
        ts_mid = (windows_meta['start_ts'].to_numpy() + windows_meta['end_ts'].to_numpy()) // 2
        ts_datetime = pd.to_datetime(ts_mid, unit='s')
        colors = np.array(['lightgray', 'steelblue', 'darkorange', 'darkgreen'])
        
        for fold_idx, f in enumerate(folds):
            ax = axes[fold_idx]
            y = np.zeros(len(ts_mid), dtype=np.int32)
            
            train_seg = set(getattr(f, 'train_segments', []) or [])
            val_seg = set(getattr(f, 'val_segments', []) or [])
            test_seg = set(getattr(f, 'test_segments', []) or [])
            seg_ids = windows_meta['segment_id'].to_numpy()
            
            train_mask = np.array([sid in train_seg for sid in seg_ids], dtype=bool)
            val_mask = np.array([sid in val_seg for sid in seg_ids], dtype=bool)
            test_mask = np.array([sid in test_seg for sid in seg_ids], dtype=bool)
            
            y[train_mask] = 1
            y[val_mask] = 2
            y[test_mask] = 3
            
            ax.scatter(ts_datetime, y, c=colors[y], s=2)
            ax.set_yticks([0,1,2,3])
            ax.set_yticklabels(['none','train','val','test'])
            ax.set_title(f'Fold {fold_idx}: Walk-forward window assignments')
            ax.set_ylabel('set')
            
            # 添加fold时间范围信息
            if hasattr(f, 'train_start_ts') and hasattr(f, 'val_end_ts'):
                train_start = pd.to_datetime(f.train_start_ts, unit='s')
                val_end = pd.to_datetime(f.val_end_ts, unit='s')
                ax.axvline(train_start, color='blue', linestyle='--', alpha=0.5, label='Train start')
                ax.axvline(val_end, color='orange', linestyle='--', alpha=0.5, label='Val end')
                if fold_idx == 0:  # 只在第一个子图显示图例
                    ax.legend(loc='upper right', fontsize=8)
        
        axes[-1].set_xlabel('timestamp')
        fig.suptitle('Walk-forward Cross-Validation Splits', fontsize=14)
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)