#!/usr/bin/env python3
import os
import glob
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TS_CANDIDATES = {"timestamp", "ts_utc", "sensordatetime", "SensorDateTime"}

def parse_ts(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, unit="s", errors="coerce")
    if s.isna().any():
        s = pd.to_datetime(series, errors="coerce", utc=True)
        try:
            s = s.dt.tz_convert(None)
        except Exception:
            pass
    return s

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def visualize_raw(raw_dir, out_dir, resample_seconds=60, file_limit=0, row_limit=0):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted([fp for fp in glob.glob(os.path.join(raw_dir, '*.csv'))])
    if file_limit and file_limit > 0:
        files = files[:file_limit]
    nat_counts = []
    resampled_lengths = []
    numeric_counts = []
    file_names = []
    for fp in files:
        try:
            df = pd.read_csv(fp, low_memory=False, nrows=(row_limit if row_limit and row_limit > 0 else None))
            cols_lower = {c.lower(): c for c in df.columns}
            ts_col = None
            for candidate in ['timestamp', 'ts_utc', 'sensordatetime']:
                if candidate in cols_lower:
                    ts_col = cols_lower[candidate]
                    break
            if ts_col is None:
                continue
            ts = pd.to_datetime(df[ts_col], errors='coerce', utc=True)
            ts = ts.tz_localize(None)
            nat_counts.append(ts.isna().sum())
            # numeric
            num_df = df.select_dtypes(include=[np.number])
            numeric_counts.append(num_df.shape[1])
            if ts.isna().all():
                resampled_lengths.append(0)
            else:
                tmp = pd.DataFrame({'ts': ts})
                tmp['val'] = 1.0
                tmp = tmp.dropna(subset=['ts']).set_index('ts')
                res = tmp.resample(f'{resample_seconds}s').mean(numeric_only=True)
                resampled_lengths.append(len(res))
            file_names.append(os.path.basename(fp))
        except Exception:
            continue
    if len(file_names) == 0:
        return
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(file_names)), nat_counts)
    plt.xticks(range(len(file_names)), [fn[:20] for fn in file_names], rotation=45, ha='right')
    plt.title('NaT counts per CSV (sample)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'raw_nat_counts.png'))
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(file_names)), resampled_lengths)
    plt.xticks(range(len(file_names)), [fn[:20] for fn in file_names], rotation=45, ha='right')
    plt.title('Resampled lengths per CSV (sample)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'raw_resampled_lengths.png'))
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(file_names)), numeric_counts)
    plt.xticks(range(len(file_names)), [fn[:20] for fn in file_names], rotation=45, ha='right')
    plt.title('Numeric column counts per CSV (sample)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'raw_numeric_cols.png'))
    plt.close()

def visualize_prepared(prep_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.isdir(prep_dir):
        return
    labels_fp = os.path.join(prep_dir, 'labels.pkl')
    if os.path.exists(labels_fp):
        labels = pd.read_pickle(labels_fp)
        # Handle both DataFrame and dict
        if isinstance(labels, pd.DataFrame):
            counts = labels.sum(axis=0)
        elif isinstance(labels, dict):
            keys = list(labels.keys())
            vals = []
            for k in keys:
                v = labels[k]
                try:
                    vals.append(int(np.sum(v)))
                except Exception:
                    vals.append(len(v) if hasattr(v, '__len__') else 0)
            counts = pd.Series(vals, index=keys)
        else:
            counts = None
        if counts is not None and len(counts) > 0:
            plt.figure(figsize=(8, 4))
            counts.sort_values(ascending=False).plot(kind='bar')
            plt.title('Label totals')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'labels_total_hist.png'))
            plt.close()
    folds = sorted([d for d in os.listdir(prep_dir) if d.startswith('fold_')])
    if len(folds) == 0:
        return
    # labels
    lbl_fp = os.path.join(prep_dir, 'labels.pkl')
    if os.path.exists(lbl_fp):
        with open(lbl_fp, 'rb') as f:
            data = pickle.load(f)
        labels = np.asarray(data.get('labels'))
        if labels.ndim == 2:
            total = labels.sum(axis=1)
        else:
            total = labels
        total = np.asarray(total)
        # 过滤掉 NaN/Inf，避免绘图报错
        finite_mask = np.isfinite(total)
        if finite_mask.any():
            total_clean = total[finite_mask]
            plt.figure(figsize=(8, 4))
            plt.hist(total_clean, bins=50)
            plt.xlabel('labels_total')
            plt.ylabel('count')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'labels_total_hist.png'))
            plt.close()
        if labels.ndim == 2 and labels.shape[1] > 0:
            means = labels.mean(axis=0)
            plt.figure(figsize=(8, 4))
            plt.plot(means, marker='o')
            plt.xlabel('device id')
            plt.ylabel('mean label')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'labels_device_means.png'))
            plt.close()
    # folds train/val counts
    folds = sorted([d for d in os.listdir(prep_dir) if d.startswith('fold_')])
    train_counts, val_counts, fids = [], [], []
    for fd in folds:
        fdir = os.path.join(prep_dir, fd)
        tri = os.path.join(fdir, 'train_indices.pt')
        vai = os.path.join(fdir, 'val_indices.pt')
        if os.path.exists(tri) and os.path.exists(vai):
            import torch
            train_counts.append(len(torch.load(tri)))
            val_counts.append(len(torch.load(vai)))
            fids.append(fd)
    if fids:
        x = np.arange(len(fids))
        width = 0.35
        plt.figure(figsize=(8, 4))
        plt.bar(x - width/2, train_counts, width, label='train')
        plt.bar(x + width/2, val_counts, width, label='val')
        plt.xticks(x, fids)
        plt.ylabel('windows')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'fold_train_val_counts.png'))
        plt.close()
    # samples from fold_0
    f0 = os.path.join(prep_dir, 'fold_0')
    tr_raw = os.path.join(f0, 'train_raw.pt')
    tr_freq = os.path.join(f0, 'train_freq.pt')
    if os.path.exists(tr_raw):
        import torch
        raw_t = torch.load(tr_raw)
        raw = raw_t.detach().cpu().numpy() if hasattr(raw_t, 'detach') else raw_t
        n = min(3, raw.shape[0])
        fig, axes = plt.subplots(n, 1, figsize=(10, 3*n), sharex=True)
        ch = 0
        for i in range(n):
            axes[i].plot(raw[i, :, ch], lw=1)
            axes[i].set_ylabel(f'sample {i} P_kW')
        axes[-1].set_xlabel('time step')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'fold0_train_raw_PkW_samples.png'))
        plt.close(fig)
    if os.path.exists(tr_freq):
        import torch
        freq_t = torch.load(tr_freq)
        freq = freq_t.detach().cpu().numpy() if hasattr(freq_t, 'detach') else freq_t
        if freq.ndim == 3 and freq.shape[0] > 0:
            plt.figure(figsize=(8, 4))
            plt.imshow(freq[0].T, aspect='auto', origin='lower', cmap='magma')
            plt.colorbar(label='magnitude')
            plt.xlabel('frame')
            plt.ylabel('bin')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'fold0_train_freq_sample0.png'))
            plt.close()

def main():
    ap = argparse.ArgumentParser(description='Visualize raw CSV checks and prepared folds')
    ap.add_argument('--raw', default='Data/raw')
    ap.add_argument('--prepared', default='Data/prepared')
    ap.add_argument('--out', default='reports/prep_visualization')
    ap.add_argument('--resample-seconds', type=int, default=60)
    ap.add_argument('--file-limit', type=int, default=3)
    ap.add_argument('--row-limit', type=int, default=500000)
    args = ap.parse_args()
    ensure_dir(args.out)
    visualize_raw(args.raw, args.out, resample_seconds=args.resample_seconds, file_limit=args.file_limit, row_limit=args.row_limit)
    visualize_prepared(args.prepared, args.out)
    print(f'Saved figures to {args.out}')

if __name__ == '__main__':
    main()