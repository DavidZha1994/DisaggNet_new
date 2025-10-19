import os
import pandas as pd

base = os.path.join(os.path.dirname(__file__), "..", "Data", "prepared")
base = os.path.abspath(base)
print(f"Prepared base: {base}")

seg_fp = os.path.join(base, "segments_meta.csv")
print("[Global segments]")
if os.path.exists(seg_fp):
    seg = pd.read_csv(seg_fp)
    smin = pd.to_datetime(seg['start_ts'], unit='s').min()
    emax = pd.to_datetime(seg['end_ts'], unit='s').max()
    dur = emax - smin
    print(f"  start={smin}")
    print(f"  end  ={emax}")
    print(f"  duration_days={dur.days}, hours={dur.seconds//3600}")
else:
    print("  segments_meta.csv not found")

for f in range(3):
    fdir = os.path.join(base, f"fold_{f}")
    tr = os.path.join(fdir, "train_metadata.csv")
    va = os.path.join(fdir, "val_metadata.csv")
    print(f"[Fold {f}]")
    if os.path.exists(tr):
        dtr = pd.read_csv(tr)
        if len(dtr):
            tmin = pd.to_datetime(dtr['timestamp'], unit='s').min()
            tmax = pd.to_datetime(dtr['timestamp'], unit='s').max()
            print(f"  train: start={tmin}, end={tmax}")
        else:
            print("  train: empty")
    else:
        print("  train_metadata.csv not found")
    if os.path.exists(va):
        dva = pd.read_csv(va)
        if len(dva):
            vmin = pd.to_datetime(dva['timestamp'], unit='s').min()
            vmax = pd.to_datetime(dva['timestamp'], unit='s').max()
            print(f"  val:   start={vmin}, end={vmax}")
        else:
            print("  val: empty")
    else:
        print("  val_metadata.csv not found")
    print()