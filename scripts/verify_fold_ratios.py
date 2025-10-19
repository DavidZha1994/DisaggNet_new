import os
import csv
import json
import yaml
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE_DIR)
PREPARED_DIR = os.path.join(ROOT, 'Data', 'prepared')
CONFIG_PATH = os.path.join(ROOT, 'config', 'prep_config.yaml')
SEGMENTS_META = os.path.join(PREPARED_DIR, 'segments_meta.csv')


def read_yaml(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def read_csv_timestamps(fp):
    ts = []
    with open(fp, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # metadata uses epoch seconds in 'timestamp'
            t = int(row['timestamp'])
            ts.append(t)
    return min(ts), max(ts)


def read_segments_meta(fp):
    # CSV with columns: segment_id,start_ts,end_ts
    starts = []
    ends = []
    with open(fp, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            starts.append(int(row['start_ts']))
            ends.append(int(row['end_ts']))
    return min(starts), max(ends)


def fmt_ts(ts):
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')


def main():
    cfg = read_yaml(CONFIG_PATH)
    purge_gap_minutes = int(cfg.get('cross_validation', {}).get('purge_gap_minutes', 0))
    purge_gap_seconds = purge_gap_minutes * 60
    n_folds = int(cfg.get('cross_validation', {}).get('n_folds', 3))

    global_start, global_end = read_segments_meta(SEGMENTS_META)
    print('[Global]')
    print(f'  start={fmt_ts(global_start)}')
    print(f'  end  ={fmt_ts(global_end)}')

    # Verify each fold
    for i in range(n_folds):
        fold_dir = os.path.join(PREPARED_DIR, f'fold_{i}')
        train_meta = os.path.join(fold_dir, 'train_metadata.csv')
        val_meta = os.path.join(fold_dir, 'val_metadata.csv')
        if not (os.path.exists(train_meta) and os.path.exists(val_meta)):
            print(f'[Fold {i}] missing metadata files')
            continue
        train_start, train_end = read_csv_timestamps(train_meta)
        val_start, val_end = read_csv_timestamps(val_meta)

        train_dur = max(0, train_end - train_start)
        val_dur = max(0, val_end - val_start)
        used_total_span = max(1, val_end - global_start)
        
        # Ratios
        ratio_val_used = val_dur / used_total_span
        denom = train_dur + purge_gap_seconds + val_dur
        ratio_val_nominal = val_dur / denom if denom > 0 else 0.0
        
        print(f'\n[Fold {i}]')
        print(f'  train: start={fmt_ts(train_start)}, end={fmt_ts(train_end)}, dur_s={train_dur}')
        print(f'  val:   start={fmt_ts(val_start)}, end={fmt_ts(val_end)}, dur_s={val_dur}')
        print(f'  used_total_span_s={used_total_span}')
        print(f'  ratio_val_over_used={ratio_val_used:.4f} (target≈0.20)')
        print(f'  ratio_val_over_train+gap+val={ratio_val_nominal:.4f} (target≈0.20, gap={purge_gap_seconds}s)')
        
        # Consistency checks
        if i > 0:
            prev_val_end_fp = os.path.join(PREPARED_DIR, f'fold_{i-1}', 'val_metadata.csv')
            _, prev_val_end = read_csv_timestamps(prev_val_end_fp)
            if train_end <= prev_val_end:
                print(f'  WARN: train_end not expanding vs previous val_end ({fmt_ts(prev_val_end)})')
        
        # Coverage check on final fold
        if i == n_folds - 1:
            extra = global_end - val_end
            print(f'  coverage_gap_seconds_to_global_end={extra}')
            print(f'  coverage_gap_hours={extra/3600:.3f}')

    print('\nDone.')


if __name__ == '__main__':
    main()