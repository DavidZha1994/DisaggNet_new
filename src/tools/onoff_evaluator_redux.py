import os
import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Smoothing & Robust Stats
# -------------------------

def hampel_filter(x: np.ndarray, window: int = 7, n_sigma: float = 3.0) -> np.ndarray:
    s = pd.Series(x)
    rolling_med = s.rolling(window, center=True, min_periods=1).median()
    diff = (s - rolling_med).abs()
    mad = diff.rolling(window, center=True, min_periods=1).median()
    threshold = n_sigma * 1.4826 * mad
    cleaned = s.copy()
    mask = diff > threshold
    cleaned[mask] = rolling_med[mask]
    return cleaned.to_numpy()


def smooth_series(x: np.ndarray, median_window: int = 5, mean_window: int = 9, use_hampel: bool = True) -> np.ndarray:
    y = x
    if use_hampel:
        y = hampel_filter(y, window=max(5, median_window), n_sigma=3.0)
    if median_window > 1:
        y = pd.Series(y).rolling(median_window, center=True, min_periods=1).median().to_numpy()
    if mean_window > 1:
        y = pd.Series(y).rolling(mean_window, center=True, min_periods=1).mean().to_numpy()
    return y


def otsu_threshold(x: np.ndarray, bins: int = 256) -> float:
    data = x[~np.isnan(x)]
    hist, bin_edges = np.histogram(data, bins=bins)
    hist = hist.astype(float)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mean1 = np.cumsum(hist * centers)
    mean2 = (np.cumsum((hist * centers)[::-1]))[::-1]
    inter_class = (mean1 * weight2 - mean2 * weight1) ** 2 / (weight1 * weight2 + 1e-9)
    idx = np.nanargmax(inter_class)
    return centers[idx]


def distribution_valley_threshold(x: np.ndarray, bins: int = 128, min_sep: float = 0.05) -> float:
    data = x[~np.isnan(x)]
    if data.size < 10:
        return float(np.nanmedian(data))
    hist, edges = np.histogram(data, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    peak_idxs = np.argsort(hist)[-2:]
    peak_vals = centers[peak_idxs]
    p1, p2 = np.sort(peak_vals)
    if abs(p2 - p1) < min_sep * (centers.max() - centers.min()):
        return otsu_threshold(data, bins=bins)
    left = min(peak_idxs)
    right = max(peak_idxs)
    if right - left <= 1:
        return otsu_threshold(data, bins=bins)
    valley_idx_rel = int(np.argmin(hist[left+1:right]))
    valley_idx = left + 1 + valley_idx_rel
    return centers[valley_idx]


def _runs(state: np.ndarray) -> List[int]:
    if len(state) == 0:
        return []
    runs = []
    cur = state[0]
    length = 1
    for v in state[1:]:
        if v == cur:
            length += 1
        else:
            runs.append(length)
            cur = v
            length = 1
    runs.append(length)
    return runs


def enforce_min_durations(state: np.ndarray, min_on: int = 32, min_off: int = 32) -> np.ndarray:
    s = state.astype(int)
    if len(s) == 0:
        return s
    runs = _runs(s)
    if not runs:
        return s
    out = s.copy()
    i = 0
    cur = s[0]
    for r in runs:
        if cur == 1 and r < min_on:
            out[i:i+r] = 0
        elif cur == 0 and r < min_off:
            out[i:i+r] = 1
        i += r
        cur = 1 - cur
    return out


# -------------------------
# State Computation
# -------------------------

def compute_absolute_state(x: np.ndarray) -> Tuple[np.ndarray, Dict]:
    xs = smooth_series(x, median_window=5, mean_window=9, use_hampel=True)
    thr = distribution_valley_threshold(xs)
    mad = np.median(np.abs(xs - np.median(xs))) + 1e-6
    hys = 0.75 * mad
    upper = thr + hys
    lower = thr - hys
    on_count = off_count = 0
    min_on = min_off = 32
    state = np.zeros_like(xs, dtype=int)
    cur = 0
    for i, v in enumerate(xs):
        if cur == 0:
            if v > upper:
                on_count += 1
                if on_count >= min_on:
                    cur = 1
                    on_count = 0
            else:
                on_count = 0
        else:
            if v < lower:
                off_count += 1
                if off_count >= min_off:
                    cur = 0
                    off_count = 0
            else:
                off_count = 0
        state[i] = cur
    state = enforce_min_durations(state, min_on=min_on, min_off=min_off)
    info = {"thr": float(thr), "upper": float(upper), "lower": float(lower), "mad": float(mad)}
    return state, info


def compute_delta_state(x: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    基于归一化数据的改进delta检测方法
    
    核心思想：
    1. 数据归一化：使用robust scaling (基于中位数和MAD)
    2. 相对变化率：检测归一化后的相对变化
    3. 自适应阈值：基于数据分布自动调整阈值
    """
    xs = smooth_series(x, median_window=3, mean_window=5, use_hampel=True)
    if len(xs) < 10:
        return np.zeros_like(xs, dtype=int), {}
    
    # Step 1: Robust normalization
    # 使用中位数和MAD进行robust scaling，避免异常值影响
    median_x = np.median(xs)
    mad_x = np.median(np.abs(xs - median_x)) + 1e-6
    
    # 归一化到[-1, 1]范围，但保持原始分布特征
    xs_norm = (xs - median_x) / (3 * mad_x)  # 3*MAD约等于标准差
    xs_norm = np.clip(xs_norm, -3, 3)  # 限制极值
    
    # Step 2: 计算归一化后的变化率
    dx_norm = np.diff(xs_norm, prepend=xs_norm[0])
    
    # Step 3: 自适应阈值计算
    # 基于归一化变化率的分布特征
    mad_dx = np.median(np.abs(dx_norm - np.median(dx_norm))) + 1e-6
    
    # 使用更小的滚动窗口提高敏感度
    w_candidates = [6, 8, 12, 16]
    # 使用相对阈值系数
    k_candidates = [1.5, 1.2, 1.0, 0.8]
    
    best_state = None
    best_info = None
    best_score = -np.inf
    
    for w in w_candidates:
        # 计算滚动变化率
        roll_pos = pd.Series(np.clip(dx_norm, 0, None)).rolling(w, min_periods=1).sum().to_numpy()
        roll_neg = pd.Series(np.clip(-dx_norm, 0, None)).rolling(w, min_periods=1).sum().to_numpy()
        
        # 基于分布的自适应基线
        base_pos = np.percentile(roll_pos, 85)  # 降低基线百分位
        base_neg = np.percentile(roll_neg, 85)
        
        for k in k_candidates:
            # 相对阈值，基于MAD
            thr_pos = base_pos + k * mad_dx
            thr_neg = base_neg + k * mad_dx
            
            # 状态检测逻辑
            state = np.zeros_like(xs, dtype=int)
            cur = 0
            
            # 使用固定的最小持续时间，但相对较短
            min_on = 6
            min_off = 6
            min_toggle_gap = 8
            
            on_count = off_count = 0
            last_toggle_idx = -min_toggle_gap
            
            for i in range(len(xs)):
                if cur == 0:  # 当前为OFF状态
                    if roll_pos[i] > thr_pos:
                        on_count += 1
                        if on_count >= min_on and (i - last_toggle_idx) >= min_toggle_gap:
                            # 额外检查：确保有足够的功率变化
                            if xs_norm[i] > -0.5:  # 归一化后的功率不能太低
                                cur = 1
                                on_count = 0
                                last_toggle_idx = i
                    else:
                        on_count = 0
                        
                else:  # 当前为ON状态
                    if roll_neg[i] > thr_neg:
                        off_count += 1
                        if off_count >= min_off and (i - last_toggle_idx) >= min_toggle_gap:
                            # 额外检查：确保功率确实下降
                            if xs_norm[i] < 0.5:  # 归一化后的功率不能太高
                                cur = 0
                                off_count = 0
                                last_toggle_idx = i
                    else:
                        off_count = 0
                
                state[i] = cur
            
            state = enforce_min_durations(state, min_on=min_on, min_off=min_off)
            
            # 计算评分
            toggles = int(np.sum(np.diff(state) != 0))
            runs = _runs(state)
            avg_run = float(np.mean(runs)) if runs else 1.0
            short_runs = int(sum(1 for r in runs if r < 12))  # 短运行定义为<12个时间点
            
            # 改进的评分函数：平衡切换数量和运行长度
            if toggles == 0:
                score = -1000
            elif toggles <= 10:  # 期望的切换数量范围
                toggle_penalty = 0
            elif toggles <= 50:
                toggle_penalty = (toggles - 10) * 10
            else:
                toggle_penalty = 400 + (toggles - 50) * 50
            
            short_run_penalty = short_runs * 20
            score = avg_run - toggle_penalty - short_run_penalty
            
            if score > best_score:
                best_score = score
                best_state = state
                best_info = {
                    "mad": float(mad_dx),
                    "window": int(w),
                    "k": float(k),
                    "thr_pos": float(thr_pos),
                    "thr_neg": float(thr_neg),
                    "toggles": toggles,
                    "avg_run": avg_run,
                    "median_x": float(median_x),
                    "mad_x": float(mad_x),
                    "mad_dx": float(mad_dx),
                    "min_toggle_gap": int(min_toggle_gap),
                }
    
    if best_state is None:
        # fallback conservative zero state
        min_on = min_off = 12
        return enforce_min_durations(np.zeros_like(xs, dtype=int), min_on=min_on, min_off=min_off), {"mad": 0.0, "window": 12, "k": 1.0}
    return best_state, best_info


def compute_hybrid_state(x: np.ndarray) -> Tuple[np.ndarray, Dict]:
    sa, ia = compute_absolute_state(x)
    sd, id_ = compute_delta_state(x)
    xs = smooth_series(x, median_window=5, mean_window=9, use_hampel=True)
    thr = ia.get("thr", np.nanmedian(xs))
    margin = 0.5 * ia.get("mad", np.median(np.abs(xs - np.median(xs))))
    near_thr = (xs > thr - margin) & (xs < thr + 3 * margin)
    combined = np.where((sd == 1) & near_thr, 1, sa)
    combined = enforce_min_durations(combined, min_on=32, min_off=32)
    
    # 计算混合方法的评分
    toggles = int(np.sum(np.diff(combined) != 0))
    runs = _runs(combined)
    avg_run = float(np.mean(runs)) if runs else 0.0
    short_runs = sum(1 for r in runs if r < 12)
    
    # 评分函数
    if toggles == 0:
        score = -1000.0
    elif toggles <= 10:
        score = avg_run - short_runs * 10
    else:
        score = avg_run - (toggles - 10) * 20 - short_runs * 10
    
    info = {
        "absolute": ia, 
        "delta": id_,
        "score": float(score),
        "toggles": toggles,
        "avg_run": avg_run,
        "short_runs": short_runs
    }
    return combined, info


# -------------------------
# Plotting
# -------------------------

def plot_methods(time: np.ndarray, x: np.ndarray, states: Dict[str, np.ndarray], device_name: str, out_path: Path):
    # Widen horizontal axis for clearer switch visualization across all devices
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(28, 10), dpi=300, sharex=True, sharey=True)
    methods = ["absolute", "delta", "hybrid"]
    xs_smooth = smooth_series(x, median_window=5, mean_window=9, use_hampel=True)
    y_min = min(np.nanmin(x), np.nanmin(xs_smooth))
    y_max = max(np.nanmax(x), np.nanmax(xs_smooth))
    if np.isclose(y_min, y_max):
        y_max = y_min + 1.0
    for i, m in enumerate(methods):
        ax = axes[i]
        ax.plot(time, x, color='tab:blue', alpha=0.6, label='power(raw)')
        ax.plot(time, xs_smooth, color='tab:orange', alpha=0.85, label='power(smooth)')
        ax.set_ylabel('Power')
        ax.set_ylim(y_min, y_max)
        s = states[m]
        scaled = y_min + (y_max - y_min) * (s.astype(float))
        ax.step(time, scaled, where='post', color='tab:red', alpha=0.9, label='state(0/1 scaled)')
        ax.set_title(f"{device_name} - {m}")
        ax.grid(True, alpha=0.3)
        # Reduce default padding on x to utilize widened horizontal space
        ax.margins(x=0.005)
        ax.legend(loc='upper left')
    axes[-1].set_xlabel('Time')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# -------------------------
# Evaluation & IO
# -------------------------

def evaluate_series(time: np.ndarray, x: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
    sa, ia = compute_absolute_state(x)
    sd, id_ = compute_delta_state(x)
    sh, ih = compute_hybrid_state(x)
    states = {"absolute": sa, "delta": sd, "hybrid": sh}
    infos = {"absolute": ia, "delta": id_, "hybrid": ih}
    return states, infos


def score_states(x: np.ndarray, state: np.ndarray) -> Dict[str, float]:
    toggles = int(np.sum(np.diff(state) != 0))
    runs = _runs(state)
    avg_run = float(np.mean(runs)) if runs else 0.0
    short_runs = int(np.sum(np.array(runs) < 16))
    on_std = float(np.std(x[state == 1])) if np.any(state == 1) else 0.0
    off_std = float(np.std(x[state == 0])) if np.any(state == 0) else 0.0
    # Strongly penalize zero-toggle (indicates no switching captured)
    zero_penalty = len(x) if toggles == 0 else 0
    score = avg_run - 25 * short_runs - 10 * toggles - 5 * on_std - 2 * off_std - zero_penalty
    return {
        "toggles": toggles,
        "avg_run": avg_run,
        "short_runs": short_runs,
        "on_std": on_std,
        "off_std": off_std,
        "score": score,
    }


def find_device_name(filename: str) -> str:
    # Heuristic: between 'cleaned_' and next '_' is device name; else stem
    stem = Path(filename).stem
    if stem.startswith("cleaned_"):
        parts = stem.split("_")
        if len(parts) > 1:
            return parts[1]
    return stem


def main():
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "Data"
    out_dir = data_dir / "analysis_onoff"
    out_dir.mkdir(parents=True, exist_ok=True)

    # collect cleaned csv files
    csv_files = []
    for p in data_dir.rglob("*.csv"):
        if p.name.startswith("cleaned_"):
            csv_files.append(p)
    csv_files = sorted(csv_files)

    report = []
    best_methods = {}
    per_device_params: Dict[str, Dict] = {}

    for csv in csv_files:
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        # pick source column
        source_col = None
        for cand in ["P_kW", "power", "P"]:
            if cand in df.columns:
                source_col = cand
                break
        if source_col is None:
            continue
        x = df[source_col].to_numpy()
        time = np.arange(len(x))

        states, infos = evaluate_series(time, x)
        metrics = {m: score_states(x, s) for m, s in states.items()}
        # choose best by score
        best = max(metrics.items(), key=lambda kv: kv[1]["score"])[0]
        device_name = find_device_name(csv.name)

        # plot
        plot_path = out_dir / f"{csv.name}_onoff_detection.png"
        plot_methods(time, x, states, device_name, plot_path)

        item = {
            "file": csv.name,
            "source": source_col,
            "absolute": metrics["absolute"],
            "delta": metrics["delta"],
            "hybrid": metrics["hybrid"],
            "best": best,
            "plot_path": str(plot_path),
        }
        report.append(item)
        best_methods[device_name] = best

        # Suggest per-device parameters based on robust stats
        xs_s = smooth_series(x, median_window=5, mean_window=9, use_hampel=True)
        mad_x = float(np.median(np.abs(xs_s - np.median(xs_s))) + 1e-6)
        noise_span = float(np.percentile(xs_s, 95) - np.percentile(xs_s, 5))
        # adapt smoothing: small noise -> small window; large noise -> larger window
        if noise_span < 0.2:
            median_w, mean_w = 3, 5
        elif noise_span < 1.0:
            median_w, mean_w = 5, 9
        else:
            median_w, mean_w = 7, 13
        # durations: prefer the median of observed runs clipped
        runs = _runs(states[best])
        med_run = int(np.median(runs)) if runs else 32
        min_on = int(max(16, min(96, med_run // 2)))
        min_off = int(max(16, min(96, med_run // 2)))
        params = {
            "method": best,
            "smooth_window": mean_w,
            "median_window": median_w,
            "min_on_duration": min_on,
            "min_off_duration": min_off,
        }
        if best == "absolute":
            ia = infos["absolute"]
            params.update({
                "on_threshold": float(ia.get("upper", np.nan)),
                "off_threshold": float(ia.get("lower", np.nan)),
                "hysteresis_margin": float(ia.get("upper", 0) - ia.get("thr", 0)),
                "mad": float(ia.get("mad", mad_x)),
            })
        elif best == "delta":
            id_ = infos["delta"]
            params.update({
                "delta_k": float(id_.get("k", 2.0)),
                "delta_threshold": float(id_.get("thr", 0.0)),
                "mad_diff": float(id_.get("mad", mad_x)),
            })
        else:  # hybrid
            ia = infos["absolute"]
            id_ = infos["delta"]
            params.update({
                "on_threshold": float(ia.get("upper", np.nan)),
                "off_threshold": float(ia.get("lower", np.nan)),
                "hysteresis_margin": float(ia.get("upper", 0) - ia.get("thr", 0)),
                "delta_k": float(id_.get("k", 2.0)),
            })
        # Skip writing per-device parameters for main meter-like devices
        if device_name.lower() not in {"mainterminal", "mainmeter", "main"}:
            per_device_params[device_name] = params

    # save
    report_path = out_dir / "onoff_method_report.json"
    best_path = out_dir / "onoff_best_methods.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_methods, f, ensure_ascii=False, indent=2)
    print(f"Saved report to {report_path}")
    print(f"Saved best methods summary to {best_path}")

    # Render per-device YAML config alongside for review
    yaml_lines = ["labels:", "  onoff:", "    per_device:"]
    for dev, p in per_device_params.items():
        yaml_lines.append(f"      {dev}:")
        for k, v in p.items():
            if isinstance(v, float):
                yaml_lines.append(f"        {k}: {v:.6f}")
            else:
                yaml_lines.append(f"        {k}: {v}")
    per_dev_yaml_path = out_dir / "onoff_per_device_config.yaml"
    with open(per_dev_yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines) + "\n")
    print(f"Saved per-device config suggestion to {per_dev_yaml_path}")


if __name__ == "__main__":
    main()