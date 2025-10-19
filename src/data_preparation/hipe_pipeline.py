import os
import json
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml


@dataclass
class HIPEConfig:
    enable: bool = True
    timestamp_col: str = "timestamp"
    mains_file: str = "main.csv"
    device_pattern: str = "device_*.csv"
    resample_seconds: int = 5
    window_length: int = 1024
    step_size: int = 256
    # Columns mapping to internal canonical names
    mains_cols: Dict[str, str] = None  # {"P_kW": "P_total_W", "Q_kvar": "Q_total_var", "S_kVA": "S_total_VA", "PF": "PF_total"}
    device_cols: Dict[str, str] = None  # {"P": "P_W", "Q": "Q_var", "S": "S_VA"}
    stft_n_fft: int = 256
    stft_hop: int = 64
    stft_win_length: int = 256
    stft_window: str = "hann"  # hann|hamming|boxcar
    # Label generation options
    label_mode: str = "regression"  # "regression" | "classification"
    on_power_threshold_w: float = 30.0  # device on threshold in watts
    on_ratio_threshold: float = 0.5  # window on ratio threshold


class HIPEDataPreparationPipeline:
    """
    HIPE 风格的数据准备管线：
    - 仅用主端时序作为输入 X（可叠加导数与频域能量带）。
    - 用分设备功率序列作为监督 Y（多输出序列回归）。
    - 不输入设备名，设备索引通过 label_map 固定。
    输出文件与现有 datamodule 兼容：train_raw.npy、train_freq.npy、train_features.npy、indices/metadata、feature_names.json、raw_channel_names.json、device_name_to_id.json、cv_splits.pkl、labels.pkl。
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f) or {}
        # 解析 HIPE 配置块
        hcfg = (self.config.get("hipe") or {})
        self.hipe = HIPEConfig(
            enable=hcfg.get("enable", True),
            timestamp_col=hcfg.get("timestamp_col", "timestamp"),
            mains_file=hcfg.get("mains_file", "main.csv"),
            device_pattern=hcfg.get("device_pattern", "device_*.csv"),
            resample_seconds=int(hcfg.get("resample_seconds", 5)),
            window_length=int(hcfg.get("window_length", 1024)),
            step_size=int(hcfg.get("step_size", 256)),
            mains_cols=hcfg.get("mains_cols") or {
                "P_kW": "P_total_W",
                "Q_kvar": "Q_total_var",
                "S_kVA": "S_total_VA",
                "PF": "PF_total",
            },
            device_cols=hcfg.get("device_cols") or {
                "P": "P_W",
                "Q": "Q_var",
                "S": "S_VA",
            },
            stft_n_fft=int(hcfg.get("stft", {}).get("n_fft", 256)),
            stft_hop=int(hcfg.get("stft", {}).get("hop_length", 64)),
            stft_win_length=int(hcfg.get("stft", {}).get("win_length", hcfg.get("stft", {}).get("n_fft", 256))),
            stft_window=str(hcfg.get("stft", {}).get("window", "hann")),
            label_mode=str(hcfg.get("label_mode", "regression")),
            on_power_threshold_w=float(hcfg.get("on_power_threshold_w", 30.0)),
            on_ratio_threshold=float(hcfg.get("on_ratio_threshold", 0.5)),
        )

        # 输出目录（兼容原配置）
        self.output_dir = (
            self.config.get("data_storage", {}).get("output_directory")
            or self.config.get("output_dir")
            or os.path.join("Data", "prepared")
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self.summary_: Dict = {}

    def get_pipeline_summary(self) -> Dict:
        return self.summary_.copy()

    # [已移除重复的旧版 run_full_pipeline，以下为最新实现]

        return self.summary_.copy()

    # ------------------------
    # 读取与对齐
    # ------------------------
    def _find_mains_file(self, root: str) -> Optional[str]:
        fp = os.path.join(root, self.hipe.mains_file)
        if os.path.exists(fp):
            return fp
        # fallback: 任何含 main 的 csv
        for cand in glob.glob(os.path.join(root, "*.csv")):
            if "main" in os.path.basename(cand).lower():
                return cand
        return None

    def _find_device_files(self, root: str) -> List[str]:
        fps = glob.glob(os.path.join(root, self.hipe.device_pattern))
        mains = self._find_mains_file(root)
        mains_abs = os.path.abspath(mains) if mains else None
        # 从匹配结果中排除主表文件（避免作为设备并入）
        if mains_abs is not None:
            fps = [f for f in fps if os.path.abspath(f) != mains_abs]
        # fallback: 除 mains 外的其余 csv
        if not fps:
            candidates = glob.glob(os.path.join(root, "*.csv"))
            if mains_abs is not None:
                candidates = [f for f in candidates if os.path.abspath(f) != mains_abs]
            fps = candidates
        # 去重并排序
        fps = sorted(set(fps))
        return fps

    def _read_mains(self, fp: str) -> pd.DataFrame:
        # 仅读取必要列以降低内存占用
        try:
            header = pd.read_csv(fp, nrows=0).columns.tolist()
        except Exception:
            header = None
        ts_col = self.hipe.timestamp_col
        rename_map = self.hipe.mains_cols or {}
        candidate = set([ts_col, "ts_utc", "SensorDateTime", "P_kW", "Q_kvar", "S_kVA", "PF"]) | set(rename_map.values() or [])
        usecols = None
        if header is not None:
            usecols = [c for c in header if c in candidate]
            # 若时间列缺失，则允许读全部以便回退重命名
            if ts_col not in usecols and ("ts_utc" not in usecols and "SensorDateTime" not in usecols):
                usecols = None
        # 优先尝试使用 polars.scan_csv 流式读取，失败则回退 pandas
        try:
            import polars as pl
            lf = pl.scan_csv(fp)
            if usecols is not None:
                lf = lf.select(usecols)
            # 在流式阶段对常用数值列降精度为 Float32
            num_cols = [c for c in ["P_kW", "Q_kvar", "S_kVA", "PF"] if c in lf.columns]
            if num_cols:
                lf = lf.with_columns([pl.col(c).cast(pl.Float32) for c in num_cols])
            df = lf.collect(streaming=True).to_pandas()
        except Exception:
            df = pd.read_csv(fp, usecols=usecols)
        # 解析时间列（支持回退）
        if ts_col not in df.columns:
            for fallback in ["ts_utc", "SensorDateTime"]:
                if fallback in df.columns:
                    df = df.rename(columns={fallback: ts_col})
                    break
        if ts_col not in df.columns:
            raise KeyError(f"主端 CSV 缺少时间列: {ts_col}")
        s = pd.to_datetime(df[ts_col], unit="s", errors="coerce")
        if s.isna().any():
            s = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
            try:
                s = s.dt.tz_convert(None)
            except Exception:
                pass
        df[ts_col] = s
        df = df[df[ts_col].notna()].copy()
        # 数值列降为 float32
        for c in list(df.columns):
            if c != ts_col and np.issubdtype(df[c].dtype, np.number):
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
        return df

    def _read_devices(self, fps: List[str]) -> Tuple[List[pd.DataFrame], List[str]]:
        dev_dfs = []
        names = []
        for fp in fps:
            name = os.path.splitext(os.path.basename(fp))[0]
            # 仅读取必要列以降低内存占用
            try:
                header = pd.read_csv(fp, nrows=0).columns.tolist()
            except Exception:
                header = None
            ts_col = self.hipe.timestamp_col
            # 设备列既考虑原始键（如 P/Q/S），也考虑规范名（P_W/Q_var/S_VA）
            dev_map = self.hipe.device_cols or {"P": "P_W", "Q": "Q_var", "S": "S_VA"}
            wanted = set([ts_col, "ts_utc", "SensorDateTime", "P", "Q", "S", "P_W", "Q_var", "S_VA"]) | set(dev_map.keys()) | set(dev_map.values())
            usecols = None
            if header is not None:
                usecols = [c for c in header if c in wanted]
                if ts_col not in usecols and ("ts_utc" not in usecols and "SensorDateTime" not in usecols):
                    usecols = None
            # 优先尝试使用 polars.scan_csv 流式读取，失败则回退 pandas
            try:
                import polars as pl
                lf = pl.scan_csv(fp)
                if usecols is not None:
                    lf = lf.select(usecols)
                num_cols = [c for c in ["P", "Q", "S", "P_W", "Q_var", "S_VA"] if c in lf.columns]
                if num_cols:
                    lf = lf.with_columns([pl.col(c).cast(pl.Float32) for c in num_cols])
                df = lf.collect(streaming=True).to_pandas()
            except Exception:
                df = pd.read_csv(fp, usecols=usecols)
            if ts_col not in df.columns:
                for fallback in ["ts_utc", "SensorDateTime"]:
                    if fallback in df.columns:
                        df = df.rename(columns={fallback: ts_col})
                        break
            if ts_col not in df.columns:
                raise KeyError(f"设备 CSV 缺少时间列: {ts_col}")
            s = pd.to_datetime(df[ts_col], unit="s", errors="coerce")
            if s.isna().any():
                s = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
                try:
                    s = s.dt.tz_convert(None)
                except Exception:
                    pass
            df[ts_col] = s
            df = df[df[ts_col].notna()].copy()
            # 数值列降为 float32
            for c in list(df.columns):
                if c != ts_col and np.issubdtype(df[c].dtype, np.number):
                    df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
            dev_dfs.append(df)
            names.append(name)
        return dev_dfs, names

    def _align_and_merge(self, df_main: pd.DataFrame, dev_dfs: List[pd.DataFrame], names: List[str]) -> Tuple[pd.DataFrame, Dict[int, str]]:
        # 以主端时间为基准，按秒级重采样并保持索引为统一时间轴
        ts_col = self.hipe.timestamp_col
        df_main = df_main.sort_values(ts_col).set_index(ts_col).resample(f"{self.hipe.resample_seconds}s").mean(numeric_only=True)
        # 主端列重命名/选择
        rename_map = self.hipe.mains_cols or {}
        for k, v in rename_map.items():
            if v in df_main.columns:
                df_main = df_main.rename(columns={v: k})
        for col in ["P_kW", "Q_kvar", "S_kVA", "PF"]:
            if col not in df_main.columns:
                df_main[col] = 0.0
        # 仅保留需要的主端列以降低内存占用
        df_main = df_main[["P_kW", "Q_kvar", "S_kVA", "PF"]]

        # 合并设备功率到主端时间轴
        merged = df_main.copy()
        label_map: Dict[int, str] = {}
        for i, (df_dev, name) in enumerate(zip(dev_dfs, names)):
            dev_res = df_dev.sort_values(ts_col).set_index(ts_col).resample(f"{self.hipe.resample_seconds}s").mean(numeric_only=True)
            # 对齐到主端索引（统一长度，避免赋值长度不匹配）
            dev_res = dev_res.reindex(merged.index)
            # 设备列重命名
            for dev_col, canon in (self.hipe.device_cols or {}).items():
                if dev_col in dev_res.columns:
                    dev_res = dev_res.rename(columns={dev_col: f"{name}_{canon}"})
            # 仅保留需要的设备列以降低内存占用
            needed = [f"{name}_{c}" for c in ["P_W", "Q_var", "S_VA"] if f"{name}_{c}" in dev_res.columns]
            if needed:
                dev_res = dev_res[needed]
            # 仅并入 P/Q/S（保留缺失为 NaN 以支持掩码）
            for canon in ["P_W", "Q_var", "S_VA"]:
                colname = f"{name}_{canon}"
                if colname not in dev_res.columns:
                    # 保留为 NaN，避免将缺失误当作 0
                    dev_res[colname] = np.nan
                # 不填充 0，直接转为 float32，NaN 原样保留
                merged[colname] = pd.to_numeric(dev_res[colname], errors="coerce").to_numpy(dtype=np.float32)
            label_map[i] = name
        merged = merged.reset_index()
        return merged, label_map

    # ------------------------
    # 特征与目标
    # ------------------------
    def _build_mains_features(self, df: pd.DataFrame) -> np.ndarray:
        # 仅保留总表的 P/Q/S 通道，确保与目标数据类型一致
        arrs = []
        for col in ["P_kW", "Q_kvar", "S_kVA"]:
            if col in df.columns:
                arrs.append(df[col].to_numpy(dtype=np.float32))
            else:
                arrs.append(np.zeros(len(df), dtype=np.float32))
        X = np.stack(arrs, axis=1)  # [T,3]
        return X

    def _build_targets(self, df: pd.DataFrame, names: List[str], kind: str = "P") -> np.ndarray:
        cols = [f"{name}_{'P_W' if kind=='P' else ('Q_var' if kind=='Q' else 'S_VA')}" for name in names]
        arrs = []
        for c in cols:
            arrs.append(df[c].to_numpy(dtype=np.float32) if c in df.columns else np.zeros(len(df), dtype=np.float32))
        return np.stack(arrs, axis=1)  # [T,K]

    def _slide_window(self, X: np.ndarray, Yp: np.ndarray, L: int, H: int, starts_override: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''滑窗切片，返回 X/Y 序列与起始索引。兼容空窗口情况。支持在内存压力大时使用 np.memmap 预分配。'''
        n = X.shape[0]
        starts = starts_override if starts_override is not None else np.arange(0, max(0, n - L + 1), H, dtype=np.int64)
        N = starts.size
        if N == 0:
            X_seq = np.empty((0, L, X.shape[1]), dtype=np.float32)
            Yp_seq = np.empty((0, L, Yp.shape[1]), dtype=np.float32)
            return X_seq, Yp_seq, starts
        C = int(X.shape[1])
        K = int(Yp.shape[1])
        bytes_needed = int(N) * int(L) * (int(C) + int(K)) * 4
        # 默认阈值约 6GB，可通过环境变量覆盖
        mem_limit = int(os.environ.get("HIPE_MEMMAP_LIMIT_BYTES", str(6_000_000_000)))
        use_memmap = bytes_needed > mem_limit
        if use_memmap:
            tmp_dir = os.path.join(self.output_dir, "_tmp_windows")
            os.makedirs(tmp_dir, exist_ok=True)
            x_path = os.path.join(tmp_dir, "X_seq.dat")
            y_path = os.path.join(tmp_dir, "Yp_seq.dat")
            X_seq = np.memmap(x_path, dtype=np.float32, mode="w+", shape=(N, L, C))
            Yp_seq = np.memmap(y_path, dtype=np.float32, mode="w+", shape=(N, L, K))
            for i, s in enumerate(starts):
                X_seq[i, :, :] = X[s:s+L].astype(np.float32)
                Yp_seq[i, :, :] = Yp[s:s+L].astype(np.float32)
            X_seq.flush()
            Yp_seq.flush()
        else:
            # 直接预分配目标数组，避免列表累积后 stack 的峰值内存
            X_seq = np.empty((N, L, C), dtype=np.float32)
            Yp_seq = np.empty((N, L, K), dtype=np.float32)
            for i, s in enumerate(starts):
                X_seq[i, :, :] = X[s:s+L].astype(np.float32)
                Yp_seq[i, :, :] = Yp[s:s+L].astype(np.float32)
        return X_seq, Yp_seq, starts

    def _repair_small_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        '''填补较小缺口（秒级到分钟级），保留较大缺口为 NaN 以便后续窗口丢弃。'''
        cfg = self.config.get('segmentation', {})
        ts_col = self.hipe.timestamp_col
        small_gap_seconds = int(cfg.get('small_gap_threshold_seconds', 60))
        resample_s = max(1, int(self.hipe.resample_seconds))
        limit = max(1, small_gap_seconds // resample_s)

        df2 = df.copy()
        if ts_col in df2.columns:
            df2[ts_col] = pd.to_datetime(df2[ts_col])
            df2 = df2.set_index(ts_col)
        # 仅对数值列进行插值与受限填充
        num_cols = [c for c in df2.columns if np.issubdtype(df2[c].dtype, np.number)]
        if num_cols:
            # 时间插值（基于 DatetimeIndex）
            df2[num_cols] = df2[num_cols].interpolate(method='time', limit=limit, limit_direction='both')
            # 受限前填与后填
            df2[num_cols] = df2[num_cols].fillna(method='ffill', limit=limit)
            df2[num_cols] = df2[num_cols].fillna(method='bfill', limit=limit)
        # 恢复时间列
        if ts_col not in df2.columns:
            df2 = df2.reset_index()
        return df2

    def _valid_window_mask(self, df: pd.DataFrame, starts: np.ndarray, L: int) -> np.ndarray:
        '''窗口有效性判断：若窗内存在未修复的 NaN（表示跨越较大中断），则标记为无效。'''
        ts_col = self.hipe.timestamp_col
        num_cols = [c for c in df.columns if c != ts_col and np.issubdtype(df[c].dtype, np.number)]
        n = len(starts)
        valid = np.ones(n, dtype=bool)
        for i, s in enumerate(starts):
            win = df.iloc[s:s+L]
            if num_cols and win[num_cols].isna().any().any():
                valid[i] = False
        return valid

    def _valid_window_mask_by_ratio(self, Yp_full: np.ndarray, starts: np.ndarray, L: int, min_ratio: float = 0.8) -> np.ndarray:
        '''基于设备有效点比例的窗口过滤：至少有一个设备在该窗口的有效比例 >= min_ratio 才保留。全局无设备数据则全部保留。'''
        n = len(starts)
        valid = np.ones(n, dtype=bool)
        if Yp_full.size == 0:
            return valid
        # 全局设备有效性：有无任何非 NaN 样本
        global_has_data = np.sum(~np.isnan(Yp_full), axis=0) > 0  # [K]
        for i, s in enumerate(starts):
            win = Yp_full[s:s+L, :]  # [L,K]
            count_valid = np.sum(~np.isnan(win), axis=0)  # 每设备有效点数
            ratio = count_valid.astype(np.float32) / float(L)
            if np.any(global_has_data):
                # 至少一个有数据的设备达到阈值则保留
                keep = np.any(ratio[global_has_data] >= max(0.0, float(min_ratio)))
                valid[i] = bool(keep)
            else:
                # 若全局没有任何设备数据，保留所有窗口（用于仅主端的下游特征）
                valid[i] = True
        return valid

    def run_full_pipeline(self, data_path: str) -> Dict:
        """运行完整 HIPE 数据准备（含 Walk-Forward、连续性修复、防泄漏与不平衡控制）。"""
        import shutil
        # 1) 清理旧折数据
        for name in os.listdir(self.output_dir):
            if name.startswith("fold_"):
                shutil.rmtree(os.path.join(self.output_dir, name), ignore_errors=True)
        cv_plan_path = os.path.join(self.output_dir, "cv_splits.pkl")
        if os.path.exists(cv_plan_path):
            os.remove(cv_plan_path)

        # 2) 读取并对齐
        mains_fp = self._find_mains_file(data_path)
        device_fps = self._find_device_files(data_path)
        if mains_fp is None:
            raise FileNotFoundError("未找到主端 CSV（默认 main.csv），请在 hipe.mains_file 配置或命名文件为 main.csv")
        if not device_fps:
            raise FileNotFoundError("未找到设备 CSV（默认 device_*.csv），请在 hipe.device_pattern 配置或命名符合模式")
        df_main = self._read_mains(mains_fp)
        dev_dfs, dev_names = self._read_devices(device_fps)
        df_merged, label_map = self._align_and_merge(df_main, dev_dfs, dev_names)

        # 3) 连续性修复：填短缺口、保留长缺口为 NaN
        df_merged = self._repair_small_gaps(df_merged)

        # 4) 特征与目标
        X_full = self._build_mains_features(df_merged)
        Yp_full = self._build_targets(df_merged, dev_names, kind="P")
        # 切窗（保留窗口，按设备有效比例过滤）
        L = self.hipe.window_length
        H = self.hipe.step_size
        starts_all = np.arange(0, max(0, X_full.shape[0] - L + 1), H, dtype=np.int64)
        min_ratio = float(self.config.get('masking', {}).get('min_valid_ratio', 0.8))
        valid_mask = self._valid_window_mask_by_ratio(Yp_full, starts_all, L, min_ratio=min_ratio)
        starts = starts_all[valid_mask]
        X_seq, Yp_seq, _ = self._slide_window(X_full, Yp_full, L=L, H=H, starts_override=starts)
        # 频域与辅助特征
        freq_feats = self._window_stft_frames(X_seq)
        aux_feats, aux_names = self._aggregate_aux_features(X_seq, df_merged, starts)

        # 5) 生成窗口元数据与段元数据
        windows_meta = self._build_windows_metadata(df_merged, starts, L)
        segments_meta = self._create_segments_meta(df_merged)

        # 6) Walk-Forward CV 计划
        cv_cfg = self._ensure_cv_config()
        from .cross_validation import WalkForwardCV
        cv = WalkForwardCV({"cross_validation": cv_cfg})
        folds = cv.create_folds(segments_meta)
        # 以工程/统计特征作为 X，标签为窗内设备功率均值（忽略 NaN）
        labels_mat = np.nanmean(Yp_seq, axis=1).astype(np.float32)  # [n_windows, K]
        labels_total = labels_mat.sum(axis=1).astype(np.float32)  # [n_windows]
        windows_dataset = {"X": aux_feats, "metadata": windows_meta, "y": labels_total}

        # 7) 按折分割并保存
        splits = {}
        for fold in folds:
            train_ds, val_ds, test_ds = cv.split_windows(windows_dataset, fold)
            # 泄漏自检（时间间隔与段隔离）
            try:
                cv.validate_no_leakage(fold, train_ds, val_ds)
            except Exception:
                pass
            # 训练集不平衡控制：保留所有正样本，负样本上限 2x（可从配置覆盖）；若无正样本则保留全部负样本
            neg_to_pos = float(self.config.get("imbalance_handling", {}).get("neg_to_pos_ratio", 2.0))
            train_idx = train_ds["metadata"].get("window_idx").to_numpy(dtype=np.int64)
            val_idx = val_ds["metadata"].get("window_idx").to_numpy(dtype=np.int64)
            y_train = labels_total[train_idx]
            pos_mask = y_train > 0.0
            neg_mask = ~pos_mask
            pos_idx = train_idx[pos_mask]
            neg_idx = train_idx[neg_mask]
            if len(pos_idx) == 0:
                # 无正样本：不要进一步下采样负样本，保留所有训练窗口
                train_idx_bal = train_idx.copy()
            else:
                max_neg = int(np.ceil(neg_to_pos * len(pos_idx)))
                if len(neg_idx) > max_neg:
                    rng = np.random.default_rng(42)
                    neg_idx = rng.choice(neg_idx, size=max_neg, replace=False)
                train_idx_bal = np.concatenate([pos_idx, neg_idx])
                train_idx_bal.sort()
            fold_key = f"fold_{fold.fold_id}"
            splits[fold_key] = {"train_indices": train_idx_bal, "val_indices": val_idx}
        device_name_to_id = {name: i for i, name in enumerate(dev_names)}
        self._save_outputs(
            df_merged=df_merged,
            X_seq=X_seq,
            Yp_seq=Yp_seq,
            aux_feats=aux_feats,
            aux_names=aux_names,
            freq_feats=freq_feats,
            starts=starts,
            splits=splits,
            label_map=label_map,
            device_name_to_id=device_name_to_id,
            windows_meta=windows_meta,
        )

        # 汇总
        n = int(X_seq.shape[0])
        Tf = int(freq_feats.shape[1]) if freq_feats.ndim == 3 else 0
        Ff = int(freq_feats.shape[2]) if freq_feats.ndim == 3 else 0
        self.summary_ = {
            "n_windows": n,
            "window_length": int(L),
            "step_size": int(H),
            "n_devices": len(dev_names),
            "channels": ["P_kW", "Q_kvar", "S_kVA", "PF", "dP", "dQ", "dS"],
            "freq_repr": {"type": "stft", "frames": Tf, "bins": Ff, "n_fft": self.hipe.stft_n_fft, "hop": self.hipe.stft_hop},
            "aux_features_count": int(aux_feats.shape[1]),
            "output_dir": self.output_dir,
            "folds": list(splits.keys()),
        }
        return self.summary_.copy()

    def _save_outputs(
        self,
        df_merged: pd.DataFrame,
        X_seq: np.ndarray,
        Yp_seq: np.ndarray,
        aux_feats: np.ndarray,
        aux_names: List[str],
        freq_feats: np.ndarray,
        starts: np.ndarray,
        splits: Dict,
        label_map: Dict[int, str],
        device_name_to_id: Dict[str, int],
        windows_meta: pd.DataFrame,
    ) -> None:
        # 顶层元数据与 CV 计划
        with open(os.path.join(self.output_dir, "cv_splits.pkl"), "wb") as f:
            import pickle
            pickle.dump(splits, f)
        with open(os.path.join(self.output_dir, "device_name_to_id.json"), "w") as f:
            json.dump(device_name_to_id, f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.output_dir, "label_map.json"), "w") as f:
            json.dump({str(k): v for k, v in label_map.items()}, f, ensure_ascii=False, indent=2)

        # labels.pkl：按照配置生成分类或回归标签，并附带窗口元数据
        # 构建标签元数据（按样本列表组织，包含 timestamp 与 segment_id）
        if "start_ts" in windows_meta.columns:
            start_ts_all = windows_meta["start_ts"].astype('int64').to_numpy()
        else:
            ts_col = self.hipe.timestamp_col
            # 回退为 df_merged 中的时间列（转换为整数秒）
            start_ts_all = (pd.to_datetime(df_merged[ts_col].iloc[starts]).astype('int64') // 1_000_000_000).to_numpy()
        seg_ids_all = (
            windows_meta["segment_id"].astype(int).to_numpy()
            if "segment_id" in windows_meta.columns
            else np.zeros_like(start_ts_all, dtype=int)
        )
        window_idx_all = (
            windows_meta["window_idx"].astype(int).to_numpy()
            if "window_idx" in windows_meta.columns
            else np.arange(len(start_ts_all), dtype=int)
        )
        dt_strs = pd.to_datetime(start_ts_all, unit="s").strftime("%Y-%m-%d %H:%M:%S").to_numpy()
        label_meta_list = [
            {"timestamp": int(ts), "datetime": dt, "segment_id": int(seg_id), "window_idx": int(wi)}
            for ts, dt, seg_id, wi in zip(start_ts_all, dt_strs, seg_ids_all, window_idx_all)
        ]

        # 根据配置生成标签矩阵
        if getattr(self.hipe, "label_mode", "regression") == "classification":
            on_thr = float(getattr(self.hipe, "on_power_threshold_w", 30.0))
            ratio_thr = float(getattr(self.hipe, "on_ratio_threshold", 0.5))
            # 每窗每设备 on/off 比例
            on_states = (Yp_seq >= on_thr).astype(np.float32)  # [N,L,K]
            on_ratio = np.where(np.isfinite(on_states), on_states, 0.0).mean(axis=1)  # [N,K]
            labels_mat = (on_ratio >= ratio_thr).astype(np.float32)
            label_type = "classification"
        else:
            labels_mat = np.nanmean(Yp_seq, axis=1).astype(np.float32)
            label_type = "regression"

        labels_data = {
            "labels": labels_mat,
            "label_type": label_type,
            "label_metadata": label_meta_list,
        }
        # 可选：附带 HIPE 配置
        try:
            from dataclasses import asdict
            labels_data["meta"] = {"hipe_config": asdict(self.hipe)}
        except Exception:
            labels_data["meta"] = {"hipe_config": {}}
        with open(os.path.join(self.output_dir, "labels.pkl"), "wb") as f:
            import pickle
            pickle.dump(labels_data, f)

        # 每折保存
        for fold_key, part in splits.items():
            fold_dir = os.path.join(self.output_dir, fold_key)
            os.makedirs(fold_dir, exist_ok=True)
            train_idx = part["train_indices"]
            val_idx = part["val_indices"]

            # 原始序列供 TimeEncoder：对 NaN 做安全填充（窗均值），遇到全 NaN 则回退为 0
            X_train = X_seq[train_idx].astype(np.float32)
            X_val = X_seq[val_idx].astype(np.float32)
            # 原始掩码：True=有效，False=缺失；保存为 uint8 以节省空间
            train_mask = (~np.isnan(X_train)).astype(np.uint8)
            val_mask = (~np.isnan(X_val)).astype(np.uint8)
            # 安全填充：窗均值；遇到全 NaN 则回退为 0.0
            m_train = np.nanmean(X_train, axis=1, keepdims=True)
            m_val = np.nanmean(X_val, axis=1, keepdims=True)
            m_train = np.where(np.isnan(m_train), 0.0, m_train)
            m_val = np.where(np.isnan(m_val), 0.0, m_val)
            X_train_filled = np.where(np.isnan(X_train), m_train, X_train).astype(np.float32)
            X_val_filled = np.where(np.isnan(X_val), m_val, X_val).astype(np.float32)
            # 保存掩码与原始窗（仅 .pt）
            import torch
            torch.save(torch.from_numpy(train_mask.astype(np.uint8)), os.path.join(fold_dir, "train_mask.pt"))
            torch.save(torch.from_numpy(val_mask.astype(np.uint8)), os.path.join(fold_dir, "val_mask.pt"))
            # 保存张量版本，减少后续加载转换
            torch.save(torch.from_numpy(X_train_filled).float(), os.path.join(fold_dir, "train_raw.pt"))
            torch.save(torch.from_numpy(X_val_filled).float(), os.path.join(fold_dir, "val_raw.pt"))
            torch.save(torch.from_numpy(freq_feats[train_idx].astype(np.float32)).float(), os.path.join(fold_dir, "train_freq.pt"))
            torch.save(torch.from_numpy(freq_feats[val_idx].astype(np.float32)).float(), os.path.join(fold_dir, "val_freq.pt"))
            torch.save(torch.from_numpy(aux_feats[train_idx].astype(np.float32)).float(), os.path.join(fold_dir, "train_features.pt"))
            torch.save(torch.from_numpy(aux_feats[val_idx].astype(np.float32)).float(), os.path.join(fold_dir, "val_features.pt"))
            torch.save(torch.from_numpy(Yp_seq[train_idx].astype(np.float32)).float(), os.path.join(fold_dir, "train_targets_seq.pt"))
            torch.save(torch.from_numpy(Yp_seq[val_idx].astype(np.float32)).float(), os.path.join(fold_dir, "val_targets_seq.pt"))
            torch.save(torch.from_numpy(train_idx.astype(np.int64)), os.path.join(fold_dir, "train_indices.pt"))
            torch.save(torch.from_numpy(val_idx.astype(np.int64)), os.path.join(fold_dir, "val_indices.pt"))
            # 已移除 NPY 兼容保存（统一生成 .pt 文件）
            # metadata（时间戳，直接使用秒整数）
            if "start_ts" in windows_meta.columns:
                train_ts = windows_meta["start_ts"].iloc[train_idx].astype('int64').to_numpy()
                val_ts = windows_meta["start_ts"].iloc[val_idx].astype('int64').to_numpy()
            else:
                ts_col = self.hipe.timestamp_col
                starts_ts = (pd.to_datetime(df_merged[ts_col].iloc[starts]).astype('int64') // 1_000_000_000).to_numpy()
                train_ts = starts_ts[train_idx]
                val_ts = starts_ts[val_idx]
            train_dt = pd.to_datetime(train_ts, unit="s").strftime("%Y-%m-%d %H:%M:%S").to_numpy()
            val_dt = pd.to_datetime(val_ts, unit="s").strftime("%Y-%m-%d %H:%M:%S").to_numpy()
            md_train = pd.DataFrame({"timestamp": train_ts, "datetime": train_dt})
            md_val = pd.DataFrame({"timestamp": val_ts, "datetime": val_dt})
            md_train.to_csv(os.path.join(fold_dir, "train_metadata.csv"), index=False)
            md_val.to_csv(os.path.join(fold_dir, "val_metadata.csv"), index=False)

            # 名称文件
            with open(os.path.join(fold_dir, "feature_names.json"), "w") as f:
                json.dump(aux_names, f, ensure_ascii=False, indent=2)
            with open(os.path.join(fold_dir, "raw_channel_names.json"), "w") as f:
                json.dump(["P_kW", "Q_kvar", "S_kVA"], f, ensure_ascii=False, indent=2)

        # 清理临时窗口 memmap 目录，避免磁盘膨胀
        try:
            import shutil
            tmp_dir = os.path.join(self.output_dir, "_tmp_windows")
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    def load_processed_data(self, fold_id: int = 0) -> Dict:
        fold_dir = os.path.join(self.output_dir, f"fold_{fold_id}")
        import torch
        data = {
            "train_raw": torch.load(os.path.join(fold_dir, "train_raw.pt")) if os.path.exists(os.path.join(fold_dir, "train_raw.pt")) else None,
            "val_raw": torch.load(os.path.join(fold_dir, "val_raw.pt")) if os.path.exists(os.path.join(fold_dir, "val_raw.pt")) else None,
            "train_freq": torch.load(os.path.join(fold_dir, "train_freq.pt")) if os.path.exists(os.path.join(fold_dir, "train_freq.pt")) else None,
            "val_freq": torch.load(os.path.join(fold_dir, "val_freq.pt")) if os.path.exists(os.path.join(fold_dir, "val_freq.pt")) else None,
            "train_features": torch.load(os.path.join(fold_dir, "train_features.pt")) if os.path.exists(os.path.join(fold_dir, "train_features.pt")) else None,
            "val_features": torch.load(os.path.join(fold_dir, "val_features.pt")) if os.path.exists(os.path.join(fold_dir, "val_features.pt")) else None,
            "train_indices": torch.load(os.path.join(fold_dir, "train_indices.pt")) if os.path.exists(os.path.join(fold_dir, "train_indices.pt")) else None,
            "val_indices": torch.load(os.path.join(fold_dir, "val_indices.pt")) if os.path.exists(os.path.join(fold_dir, "val_indices.pt")) else None,
            "train_mask": torch.load(os.path.join(fold_dir, "train_mask.pt")) if os.path.exists(os.path.join(fold_dir, "train_mask.pt")) else None,
            "val_mask": torch.load(os.path.join(fold_dir, "val_mask.pt")) if os.path.exists(os.path.join(fold_dir, "val_mask.pt")) else None,
        }
        return data

    def _ensure_cv_config(self) -> Dict:
        cv = self.config.get("cross_validation") or {}
        return {
            "n_folds": int(cv.get("n_folds", 5)),
            "purge_gap_minutes": int(cv.get("purge_gap_minutes", 10)),
            "val_span_days": int(cv.get("val_span_days", 7)),
            "test_span_days": int(cv.get("test_span_days", 0)),
            "min_train_days": int(cv.get("min_train_days", 7)),
            "segment_isolation": bool(cv.get("segment_isolation", True)),
            "holdout_test": bool(cv.get("holdout_test", False)),
        }

    def _window_stft_frames(self, X_seq: np.ndarray) -> np.ndarray:
        """按窗口对 P_kW 通道计算STFT幅度谱，NaN安全填充。返回形状 [N, T_frames, F_bins]."""
        if X_seq.size == 0:
            return np.empty((0, 0, 0), dtype=np.float32)
        N, L, C = X_seq.shape
        # 选择 P_kW 通道（默认第0通道）
        p_channel = 0
        win_len = int(self.hipe.stft_win_length)
        hop = int(self.hipe.stft_hop)
        n_fft = int(self.hipe.stft_n_fft)
        if n_fft < win_len:
            n_fft = win_len
        # 窗函数
        wtype = (self.hipe.stft_window or "hann").lower()
        if wtype == "hann":
            window = np.hanning(win_len).astype(np.float32)
        elif wtype == "hamming":
            window = np.hamming(win_len).astype(np.float32)
        else:
            window = np.ones(win_len, dtype=np.float32)
        # 计算每窗帧数
        frames = 1 if L < win_len else (1 + (L - win_len) // hop)
        F = (n_fft // 2) + 1
        out = np.empty((N, frames, F), dtype=np.float32)
        for i in range(N):
            sig = X_seq[i, :, p_channel].astype(np.float32)
            m = np.nanmean(sig)
            if np.isnan(m):
                m = 0.0
            sig = np.where(np.isnan(sig), m, sig)
            for t in range(frames):
                start = t * hop
                end = start + win_len
                if end <= L:
                    frame = sig[start:end]
                else:
                    # 零填充最后一帧
                    pad = np.zeros(win_len, dtype=np.float32)
                    take = max(0, L - start)
                    if take > 0:
                        pad[:take] = sig[start:start+take]
                    frame = pad
                frame_win = frame * window
                spec = np.fft.rfft(frame_win, n=n_fft)
                mag = np.abs(spec).astype(np.float32)
                out[i, t, :] = mag
        return out

    def _aggregate_aux_features(self, X_seq: np.ndarray, df_merged: pd.DataFrame, starts: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """计算每窗口的统计特征（NaN感知）：mean/std，覆盖所有输入通道。"""
        if X_seq.size == 0:
            return np.empty((0, 0), dtype=np.float32), []
        N, L, C = X_seq.shape
        channel_names = ["P_kW", "Q_kvar", "S_kVA", "PF", "dP", "dQ", "dS"]
        # 对齐通道数量
        if len(channel_names) != C:
            channel_names = [f"ch_{i}" for i in range(C)]
        feats = []
        names = []
        for ci, cname in enumerate(channel_names):
            x = X_seq[:, :, ci]
            mu = np.nanmean(x, axis=1)
            sd = np.nanstd(x, axis=1)
            feats.append(mu)
            feats.append(sd)
            names.append(f"mean_{cname}")
            names.append(f"std_{cname}")
        Fd = np.stack(feats, axis=1).astype(np.float32)
        return Fd, names

    def _create_segments_meta(self, df_merged: pd.DataFrame) -> pd.DataFrame:
        """基于时间间隔和缺失值生成连续段元数据。"""
        ts_col = self.hipe.timestamp_col
        s = pd.to_datetime(df_merged[ts_col])
        ts_secs = (s.astype("int64") // 1_000_000_000).to_numpy()
        # 仅用主端列判断行有效性，忽略设备列的 NaN
        mains_cols = [c for c in ["P_kW", "Q_kvar", "S_kVA", "PF"] if c in df_merged.columns]
        arr = df_merged[mains_cols].to_numpy(dtype=np.float32) if mains_cols else np.empty((len(df_merged), 0), dtype=np.float32)
        res_s = max(1, int(self.hipe.resample_seconds))
        max_gap = res_s * 3
        segments = []
        n = len(df_merged)
        seg_start = None
        for i in range(n):
            row_nan = np.isnan(arr[i]).any() if arr.size else False
            if seg_start is None:
                if not row_nan:
                    seg_start = i
                continue
            # 正常推进
            gap = ts_secs[i] - ts_secs[i-1]
            if row_nan or gap > max_gap:
                # 结束当前段
                end = i - 1
                if end >= seg_start:
                    segments.append((seg_start, end))
                seg_start = None
                if not row_nan:
                    seg_start = i
        # 收尾
        if seg_start is not None and (n - 1) >= seg_start:
            segments.append((seg_start, n - 1))
        # 构造 DataFrame
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
                "n_rows": n_rows,
                "duration_hours": duration_hours,
            })
        return pd.DataFrame(seg_rows)

    def _build_windows_metadata(self, df_merged: pd.DataFrame, starts: np.ndarray, L: int) -> pd.DataFrame:
        """为每个窗口生成元数据并映射所属段。"""
        ts_col = self.hipe.timestamp_col
        s = pd.to_datetime(df_merged[ts_col])
        ts_secs = (s.astype("int64") // 1_000_000_000).to_numpy()
        n = len(starts)
        start_ts = ts_secs[starts]
        end_ts = ts_secs[starts + L - 1]
        # 段映射
        seg_df = self._create_segments_meta(df_merged)
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

    @staticmethod
    def _safe_load(fp: str) -> Optional[np.ndarray]:
        return np.load(fp) if os.path.exists(fp) else None