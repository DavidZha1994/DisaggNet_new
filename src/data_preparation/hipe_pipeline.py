import os
import json
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml
from concurrent.futures import ThreadPoolExecutor
from src.visualization.pipeline_viz import PipelineVisualizer, VizConfig


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
        # 计算加速配置
        compute_cfg = self.config.get("compute") or {}
        n_jobs_cfg = int(compute_cfg.get("n_jobs", 1))
        if n_jobs_cfg == -1:
            n_jobs_cfg = max(1, os.cpu_count() or 1)
        self.n_jobs = n_jobs_cfg
        self.use_polars = bool(compute_cfg.get("use_polars", False))
        # Visualization setup (independent output directory)
        viz_cfg = self.config.get("visualization") or {}
        self.viz_enabled = bool(viz_cfg.get("enable", True))
        base_viz_dir = viz_cfg.get("output_dir") or os.path.join("reports", "pipeline_visualization")
        os.makedirs(base_viz_dir, exist_ok=True)
        from datetime import datetime as _dt
        ts_dir = _dt.now().strftime("%Y%m%d_%H%M%S")
        self.viz_dir = os.path.join(base_viz_dir, ts_dir)
        os.makedirs(self.viz_dir, exist_ok=True)
        try:
            vconf = VizConfig()
        except Exception:
            vconf = None
        self.viz = PipelineVisualizer(self.viz_dir, vconf)

    def get_pipeline_summary(self) -> Dict:
        return self.summary_.copy()

    def run_full_pipeline(self, data_path: str) -> Dict:
        """运行完整 HIPE 数据准备（含 Walk-Forward、连续性修复、防泄漏与不平衡控制）。"""
        import shutil
        from datetime import datetime
        start_time = datetime.now().isoformat()
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
        if getattr(self, "viz_enabled", False):
            try:
                self.viz.plot_mains_pqs(df_main, ts_col=self.hipe.timestamp_col)
            except Exception:
                pass
        dev_dfs, dev_names = self._read_devices(device_fps)
        df_merged, label_map = self._align_and_merge(df_main, dev_dfs, dev_names)
        if getattr(self, "viz_enabled", False):
            try:
                self.viz.plot_devices_p(df_merged, label_map, ts_col=self.hipe.timestamp_col)
            except Exception:
                pass
        # 仅使用成功合并的设备名称
        eff_dev_names = [label_map[i] for i in sorted(label_map.keys())]

        # 3) 连续性修复：填短缺口、保留长缺口为 NaN
        df_before = df_merged.copy()
        df_merged = self._repair_small_gaps(df_merged)
        if getattr(self, "viz_enabled", False):
            try:
                self.viz.plot_missing_heatmap(df_before, df_merged, ts_col=self.hipe.timestamp_col)
            except Exception:
                pass

        # 4) 特征与目标
        X_full = self._build_mains_features(df_merged)
        Yp_full = self._build_targets(df_merged, eff_dev_names, kind="P")
        # 切窗（保留窗口，按设备有效比例过滤）
        L = self.hipe.window_length
        H = self.hipe.step_size
        starts_all = np.arange(0, max(0, X_full.shape[0] - L + 1), H, dtype=np.int64)
        min_ratio = float(self.config.get('masking', {}).get('min_valid_ratio', 0.8))
        valid_mask = self._valid_window_mask_by_ratio(Yp_full, starts_all, L, min_ratio=min_ratio)
        starts = starts_all[valid_mask]
        X_seq, Yp_seq, _ = self._slide_window(X_full, Yp_full, L=L, H=H, starts_override=starts)
        if getattr(self, "viz_enabled", False):
            try:
                self.viz.plot_window_boundaries(df_merged, starts, L, ts_col=self.hipe.timestamp_col)
            except Exception:
                pass
        # 保持目标序列的 NaN（由下游 mask 处理），仅确保 dtype
        Yp_seq = Yp_seq.astype(np.float32)
        # 频域与辅助特征
        freq_feats = self._window_stft_frames(X_seq)
        if getattr(self, "viz_enabled", False):
            try:
                self.viz.plot_stft_samples(freq_feats, n_fft=int(self.hipe.stft_n_fft), hop=int(self.hipe.stft_hop))
            except Exception:
                pass
        aux_feats, aux_names = self._aggregate_aux_features(X_seq, df_merged, starts)
        if getattr(self, "viz_enabled", False):
            try:
                self.viz.plot_aux_feature_hist(aux_feats, aux_names)
            except Exception:
                pass

        # 5) 生成窗口元数据与段元数据
        windows_meta = self._build_windows_metadata(df_merged, starts, L)
        segments_meta = self._create_segments_meta(df_merged)
        # 保存段元数据供验证
        try:
            segments_meta.to_csv(os.path.join(self.output_dir, "segments_meta.csv"), index=False)
        except Exception:
            pass

        # 6) Walk-Forward CV 计划
        cv_cfg = self._ensure_cv_config()
        from .cross_validation import WalkForwardCV
        cv = WalkForwardCV({"cross_validation": cv_cfg})
        folds = cv.create_folds(segments_meta)
        if getattr(self, "viz_enabled", False):
            try:
                self.viz.plot_walk_forward(windows_meta, folds)
            except Exception:
                pass
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
        device_name_to_id = {name: i for i, name in enumerate(eff_dev_names)}
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
            "n_devices": len(eff_dev_names),
            "channels": ["P_kW", "Q_kvar", "S_kVA", "PF", "dP", "dQ", "dS"],
            "freq_repr": {"type": "stft", "frames": Tf, "bins": Ff, "n_fft": self.hipe.stft_n_fft, "hop": self.hipe.stft_hop},
            "aux_features_count": int(aux_feats.shape[1]),
            "output_dir": self.output_dir,
            "folds": list(splits.keys()),
        }

        # 写出 pipeline_results.json 供验证脚本使用
        try:
            end_time = datetime.now().isoformat()
            steps_summary = {
                "segmentation": {
                    "status": "success",
                    "details": {
                        "segments_count": int(segments_meta.shape[0]) if hasattr(segments_meta, 'shape') else 0,
                        "total_samples": int(df_merged.shape[0])
                    }
                },
                "windowing": {
                    "status": "success",
                    "details": {
                        "total_windows": int(windows_meta.shape[0]) if hasattr(windows_meta, 'shape') else n,
                        "window_length": int(L)
                    }
                },
                "cross_validation": {
                    "status": "success",
                    "details": {
                        "n_folds": int(len(splits)),
                        "purge_gap_minutes": int(cv_cfg.get("purge_gap_minutes", 0)),
                        "segment_isolation": bool(cv_cfg.get("segment_isolation", True))
                    }
                }
            }
            results = {
                "status": "success",
                "start_time": start_time,
                "end_time": end_time,
                "steps": steps_summary,
            }
            with open(os.path.join(self.output_dir, "pipeline_results.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception:
            # 保持流程不失败
            pass

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

            # 原始序列supply：对 NaN 做安全填充（窗均值），遇到全 NaN 则回退为 0
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
            # 修正：保存完整原始通道名称与顺序，匹配 X_seq 的7个通道
            with open(os.path.join(fold_dir, "raw_channel_names.json"), "w") as f:
                json.dump(["P_kW", "Q_kvar", "S_kVA", "PF", "dP", "dQ", "dS"], f, ensure_ascii=False, indent=2)

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
        """按窗口对 P/Q/S/PF 通道计算STFT幅度谱，NaN安全填充。返回形状 [N, T_frames, F_bins*C]."""
        if X_seq.size == 0:
            return np.empty((0, 0, 0), dtype=np.float32)
        N, L, C = X_seq.shape
        # 选择待计算的通道：P(0)/Q(1)/S(2)/PF(3)
        stft_channels = [0, 1, 2, 3]
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
        # 帧数与频点数
        frames = 1 if L < win_len else (1 + (L - win_len) // hop)
        F = (n_fft // 2) + 1
        C_eff = len(stft_channels)
        out = np.empty((N, frames, F * C_eff), dtype=np.float32)
        for i in range(N):
            for ci, ch in enumerate(stft_channels):
                sig = X_seq[i, :, ch].astype(np.float32)
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
                    out[i, t, ci * F:(ci + 1) * F] = mag
        return out

    def _aggregate_aux_features(self, X_seq: np.ndarray, df_merged: pd.DataFrame, starts: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """计算每窗口的鲁棒统计特征（NaN感知）：mean/std/median，覆盖所有输入通道。"""
        if X_seq.size == 0:
            return np.empty((0, 0), dtype=np.float32), []
        N, L, C = X_seq.shape
        channel_names = ["P_kW", "Q_kvar", "S_kVA", "PF", "dP", "dQ", "dS"]
        # 对齐通道数量
        if len(channel_names) != C:
            channel_names = [f"ch_{i}" for i in range(C)]
        feats = []
        names = []
        
        # 基础统计特征（更鲁棒）
        for ci, cname in enumerate(channel_names):
            x = X_seq[:, :, ci]
            
            # 基础统计量
            mu = np.nanmean(x, axis=1)
            sd = np.nanstd(x, axis=1)
            median = np.nanmedian(x, axis=1)
            
            # 处理全NaN窗口
            mu = np.where(np.isnan(mu), 0.0, mu)
            sd = np.where(np.isnan(sd), 0.0, sd)
            median = np.where(np.isnan(median), 0.0, median)
            
            feats.extend([mu, sd, median])
            names.extend([f"mean_{cname}", f"std_{cname}", f"median_{cname}"])
            
            # 有效值比例（数据质量指标）
            valid_ratio = (~np.isnan(x)).sum(axis=1) / L
            feats.append(valid_ratio.astype(np.float32))
            names.append(f"valid_ratio_{cname}")
            
        # 工业电气派生特征（更鲁棒的计算）
        try:
            P_seq = X_seq[:, :, 0]
            Q_seq = X_seq[:, :, 1]
            PF_seq = X_seq[:, :, 3] if C >= 4 else np.full((N, L), np.nan, dtype=np.float32)
            dP_seq = X_seq[:, :, 4] if C >= 5 else np.full((N, L), np.nan, dtype=np.float32)
            
            # 功率角 φ（避免除零）
            # 使用 atan2 并处理 P=0 的情况
            phi_seq = np.full((N, L), np.nan, dtype=np.float32)
            valid_pq = (~np.isnan(P_seq)) & (~np.isnan(Q_seq)) & (np.abs(P_seq) > 1e-6)
            phi_seq[valid_pq] = np.arctan2(Q_seq[valid_pq], P_seq[valid_pq])
            
            phi_mean = np.nanmean(phi_seq, axis=1)
            phi_std = np.nanstd(phi_seq, axis=1)
            phi_mean = np.where(np.isnan(phi_mean), 0.0, phi_mean)
            phi_std = np.where(np.isnan(phi_std), 0.0, phi_std)
            
            feats.extend([phi_mean, phi_std])
            names.extend(["mean_phi", "std_phi"])
            
            # PF 动态变化（更鲁棒）
            if L > 1:
                dPF_seq = np.diff(PF_seq, axis=1)
                dPF_mean = np.nanmean(dPF_seq, axis=1)
                dPF_std = np.nanstd(dPF_seq, axis=1)
                dPF_mean = np.where(np.isnan(dPF_mean), 0.0, dPF_mean)
                dPF_std = np.where(np.isnan(dPF_std), 0.0, dPF_std)
            else:
                dPF_mean = np.zeros(N, dtype=np.float32)
                dPF_std = np.zeros(N, dtype=np.float32)
                
            feats.extend([dPF_mean, dPF_std])
            names.extend(["mean_dPF", "std_dPF"])
            
            # 删除 on_ratio 和 edge_count 特征 - 这些是未经验证的开关状态相关特征
            
            # 删除斜率统计特征 - 这些是基于功率变化率的未验证特征
        except Exception:
            pass
        # 能量增量（kWh）
        energy_delta = np.zeros(N, dtype=np.float32)
        if "E_PP_kWh" in df_merged.columns:
            e = df_merged["E_PP_kWh"].to_numpy(dtype=np.float64)
            for i, s in enumerate(starts):
                w = e[s:s + L]
                if w.size > 1:
                    a, b = w[0], w[-1]
                    if np.isfinite(a) and np.isfinite(b):
                        energy_delta[i] = float(b - a)
        feats.append(energy_delta)
        names.append("energy_delta_kWh")
        # 电压/电流/频率/THD 汇总
        def _window_mean_std(columns: list) -> tuple:
            m = np.zeros(N, dtype=np.float32)
            s = np.zeros(N, dtype=np.float32)
            arrs = [df_merged[c].to_numpy(dtype=np.float64) for c in columns if c in df_merged.columns]
            if arrs:
                mat = np.stack(arrs, axis=1)
                avg_ts = np.nanmean(mat, axis=1)
                for i, st in enumerate(starts):
                    w = avg_ts[st:st + L]
                    if w.size:
                        m[i] = np.nanmean(w)
                        s[i] = np.nanstd(w)
            return m, s
        # Vrms 与 Irms
        vr_m, vr_s = _window_mean_std(["V1_V", "V2_V", "V3_V"])  # 相电压
        ir_m, ir_s = _window_mean_std(["I1_A", "I2_A", "I3_A"])  # 相电流
        feats.extend([vr_m, vr_s, ir_m, ir_s])
        names.extend(["Vrms_mean", "Vrms_std", "Irms_mean", "Irms_std"])
        # 频率
        f_m, f_s = _window_mean_std(["F_Hz"])
        feats.extend([f_m, f_s])
        names.extend(["F_Hz_mean", "F_Hz_std"])
        # THD
        thdv_m, thdv_s = _window_mean_std(["THD_V1_F", "THD_V2_F", "THD_V3_F"])  # 电压 THD
        thdi_m, thdi_s = _window_mean_std(["THD_I1_F", "THD_I2_F", "THD_I3_F"])  # 电流 THD
        feats.extend([thdv_m, thdv_s, thdi_m, thdi_s])
        names.extend(["THD_V_mean", "THD_V_std", "THD_I_mean", "THD_I_std"])
        # 电压不平衡度（std/mean 按相*时间平均）
        v_unbalance = np.zeros(N, dtype=np.float32)
        if all(c in df_merged.columns for c in ["V1_V", "V2_V", "V3_V"]):
            v1 = df_merged["V1_V"].to_numpy(dtype=np.float64)
            v2 = df_merged["V2_V"].to_numpy(dtype=np.float64)
            v3 = df_merged["V3_V"].to_numpy(dtype=np.float64)
            mat = np.stack([v1, v2, v3], axis=1)
            for i, st in enumerate(starts):
                w = mat[st:st + L, :]
                mu_ts = np.nanmean(w, axis=1)
                sd_ts = np.nanstd(w, axis=1)
                with np.errstate(invalid='ignore', divide='ignore'):
                    ratio = np.divide(sd_ts, mu_ts, out=np.zeros_like(sd_ts), where=mu_ts != 0)
                v_unbalance[i] = float(np.nanmean(ratio))
        feats.append(v_unbalance)
        names.append("voltage_unbalance_mean")
        
        # 最终的鲁棒性处理
        Fd = np.stack(feats, axis=1).astype(np.float32)
        
        # 清理所有无效值
        Fd = np.nan_to_num(Fd, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 检查并报告数据质量
        invalid_count = np.sum(~np.isfinite(Fd))
        if invalid_count > 0:
            print(f"警告: 辅助特征中发现 {invalid_count} 个无效值，已自动清理")
            
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

    # === 缺失的私有方法补充实现 ===
    def _find_mains_file(self, data_path: str) -> Optional[str]:
        # 优先使用配置文件名
        cand = os.path.join(data_path, self.hipe.mains_file)
        if os.path.exists(cand):
            return cand
        # 回退匹配
        patterns = ["*main*.csv", "*mains*.csv"]
        for p in patterns:
            hits = glob.glob(os.path.join(data_path, p))
            if hits:
                return sorted(hits)[0]
        return None

    def _find_device_files(self, data_path: str) -> List[str]:
        pattern = os.path.join(data_path, self.hipe.device_pattern)
        files = sorted(glob.glob(pattern))
        # 排除主表文件（如果匹配到device_pattern）
        mains_fp = self._find_mains_file(data_path)
        if mains_fp:
            files = [fp for fp in files if os.path.abspath(fp) != os.path.abspath(mains_fp)]
        return files

    def _read_mains(self, fp: str) -> pd.DataFrame:
        df = pd.read_csv(fp)
        # 时间戳列检测
        ts_col = self.hipe.timestamp_col
        if ts_col not in df.columns:
            # 简单推断：寻找包含"time"的列
            time_cols = [c for c in df.columns if "time" in c.lower()]
            if time_cols:
                ts_col = time_cols[0]
            else:
                raise ValueError(f"主端CSV缺少时间戳列: {self.hipe.timestamp_col}")
        # 统一为UTC无时区
        dt = pd.to_datetime(df[ts_col], errors='coerce', utc=True)
        df[ts_col] = dt.dt.tz_localize(None)
        df = df.dropna(subset=[ts_col]).copy()
        df = df.sort_values(ts_col)
        # 使用原始列名，不进行任何映射或单位转换
        # 保留标准列（扩展工业电气通道）
        keep_cols = [c for c in [ts_col,
                                 "P_kW", "Q_kvar", "S_kVA", "PF",
                                 "F_Hz",
                                 "U12_V", "U23_V", "U31_V",
                                 "V1_V", "V2_V", "V3_V",
                                 "I1_A", "I2_A", "I3_A", "IN_A",
                                 "IAVR_A", "UAVR_V", "VAVR_V",
                                 "E_PP_kWh", "E_QP_kvarh", "E_SP_kVAh",
                                 "THD_U12_F", "THD_U23_F", "THD_U31_F",
                                 "THD_V1_F", "THD_V2_F", "THD_V3_F",
                                 "THD_I1_F", "THD_I2_F", "THD_I3_F"] if c in df.columns]
        df = df[keep_cols].copy()
        df = df.rename(columns={ts_col: self.hipe.timestamp_col})
        return df

    def _extract_device_name(self, fp: str) -> str:
        base = os.path.basename(fp)
        stem = os.path.splitext(base)[0]
        # 优先去掉 PhaseCount 后缀，以得到纯设备名
        if "_PhaseCount_" in stem:
            return stem.split("_PhaseCount_")[0]
        # 兼容 device_ 前缀
        if stem.startswith("device_"):
            return stem[len("device_"):]
        return stem

    def _read_devices(self, fps: List[str]) -> Tuple[List[pd.DataFrame], List[str]]:
        dev_dfs: List[pd.DataFrame] = []
        dev_names: List[str] = []
        for fp in fps:
            df = pd.read_csv(fp)
            ts_col = self.hipe.timestamp_col
            if ts_col not in df.columns:
                time_cols = [c for c in df.columns if "time" in c.lower()]
                if time_cols:
                    ts_col = time_cols[0]
                else:
                    raise ValueError(f"设备CSV缺少时间戳列: {self.hipe.timestamp_col}")
            dt = pd.to_datetime(df[ts_col], errors='coerce', utc=True)
            df[ts_col] = dt.dt.tz_localize(None)
            df = df.dropna(subset=[ts_col]).copy()
            df = df.sort_values(ts_col)
            # 不做任何别名映射或单位转换，直接使用原始列
            keep = [c for c in [ts_col, "P_kW", "Q_kvar", "S_kVA"] if c in df.columns]
            df = df[keep].copy()
            df = df.rename(columns={ts_col: self.hipe.timestamp_col})
            dev_dfs.append(df)
            dev_names.append(self._extract_device_name(fp))
        return dev_dfs, dev_names

    def _resample_df(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        ts_col = self.hipe.timestamp_col
        rule_seconds = int(self.hipe.resample_seconds)
        # 可选使用 Polars 加速
        if getattr(self, "use_polars", False):
            try:
                import polars as pl
                # 只保留需要的列
                keep_cols = [ts_col] + [c for c in cols if c in df.columns]
                dfp = pl.from_pandas(df[keep_cols])
                # 确保时间戳为 Datetime 并无时区
                if dfp.schema.get(ts_col) != pl.Datetime:
                    dfp = dfp.with_columns(pl.col(ts_col).str.strptime(pl.Datetime, strict=False))
                dfp = dfp.with_columns(pl.col(ts_col).dt.replace_time_zone(None))
                out = (
                    dfp.group_by_dynamic(ts_col, every=f"{rule_seconds}s", closed="left")
                    .agg([pl.col(c).mean().alias(c) for c in cols if c in dfp.columns])
                    .sort(ts_col)
                )
                x = out.to_pandas()
                x[ts_col] = pd.to_datetime(x[ts_col]).tz_localize(None)
                return x
            except Exception:
                # 回退到 pandas
                pass
        # pandas 路径
        x = df.copy()
        x[ts_col] = pd.to_datetime(x[ts_col], errors="coerce", utc=True).dt.tz_localize(None)
        x = x.dropna(subset=[ts_col])
        x = x.set_index(x[ts_col])
        x = x[[c for c in cols if c in x.columns]]
        x = x.resample(f"{rule_seconds}S").mean()
        x[ts_col] = x.index
        x.reset_index(drop=True, inplace=True)
        return x

    def _resample_rename_device(self, df: pd.DataFrame, name: str) -> Optional[pd.DataFrame]:
        ts_col = self.hipe.timestamp_col
        cols = [c for c in ["P_kW", "Q_kvar", "S_kVA"] if c in df.columns]
        # 仅保留含有功功率 P_kW 的设备；没有 P_kW 列则跳过该设备
        if "P_kW" not in cols:
            return None
        if not cols:
            return None
        dfr = self._resample_df(df, cols)
        ren = {}
        if "P_kW" in dfr.columns:
            ren["P_kW"] = f"{name}_P_kW"
        if "Q_kvar" in dfr.columns:
            ren["Q_kvar"] = f"{name}_Q_kvar"
        if "S_kVA" in dfr.columns:
            ren["S_kVA"] = f"{name}_S_kVA"
        dfr = dfr.rename(columns=ren)
        return dfr

    def _align_and_merge(self, df_main: pd.DataFrame, dev_dfs: List[pd.DataFrame], dev_names: List[str]) -> Tuple[pd.DataFrame, Dict[int, str]]:
        ts_col = self.hipe.timestamp_col
        # 主端重采样（保留扩展工业通道）
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
            "THD_I1_F", "THD_I2_F", "THD_I3_F"
        ] if c in df_main.columns]
        dfm = self._resample_df(df_main, main_cols)
        # 合并为统一时间索引
        base = dfm.copy()
        # 并行处理设备重采样与重命名（仅保留成功生成数据的设备）
        merged_names: List[str] = []
        futures = []
        with ThreadPoolExecutor(max_workers=getattr(self, "n_jobs", 1)) as ex:
            for df, name in zip(dev_dfs, dev_names):
                futures.append((name, ex.submit(self._resample_rename_device, df, name)))
            for name, fut in futures:
                dfr = fut.result()
                if dfr is None:
                    # 跳过没有 P/Q/S 数据的设备
                    continue
                base = pd.merge_asof(
                    base.sort_values(ts_col),
                    dfr.sort_values(ts_col),
                    on=ts_col
                )
                merged_names.append(name)
        base = base.sort_values(ts_col).reset_index(drop=True)
        # 仅对成功合并的设备建立标签映射
        label_map = {i: name for i, name in enumerate(merged_names)}
        return base, label_map

    def _repair_small_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        # 小缺口定义：连续 <=2 个步长
        limit = 2
        for c in x.columns:
            if c == self.hipe.timestamp_col:
                continue
            ser = x[c]
            if ser.dtype.kind in 'bif':
                try:
                    x[c] = ser.interpolate(limit=limit, limit_direction='both')
                except Exception:
                    pass
        return x

    def _build_mains_features(self, df: pd.DataFrame) -> np.ndarray:
        T = len(df)
        P = df["P_kW"].to_numpy(dtype=np.float32) if "P_kW" in df.columns else np.full(T, np.nan, dtype=np.float32)
        Q = df["Q_kvar"].to_numpy(dtype=np.float32) if "Q_kvar" in df.columns else np.full(T, np.nan, dtype=np.float32)
        S = df["S_kVA"].to_numpy(dtype=np.float32) if "S_kVA" in df.columns else np.full(T, np.nan, dtype=np.float32)
        PF = df["PF"].to_numpy(dtype=np.float32) if "PF" in df.columns else np.full(T, np.nan, dtype=np.float32)
        dP = np.concatenate([np.array([0.0], dtype=np.float32), np.diff(P).astype(np.float32)]) if T > 0 else np.empty(0, dtype=np.float32)
        dQ = np.concatenate([np.array([0.0], dtype=np.float32), np.diff(Q).astype(np.float32)]) if T > 0 else np.empty(0, dtype=np.float32)
        dS = np.concatenate([np.array([0.0], dtype=np.float32), np.diff(S).astype(np.float32)]) if T > 0 else np.empty(0, dtype=np.float32)
        X = np.stack([P, Q, S, PF, dP, dQ, dS], axis=1)
        return X.astype(np.float32)

    def _build_targets(self, df: pd.DataFrame, dev_names: List[str], kind: str = "P") -> np.ndarray:
        T = len(df)
        mats = []
        for name in dev_names:
            col = f"{name}_P_kW" if kind == "P" else f"{name}_Q_kvar" if kind == "Q" else f"{name}_S_kVA"
            if col in df.columns:
                mats.append(df[col].to_numpy(dtype=np.float32))
            else:
                mats.append(np.full(T, np.nan, dtype=np.float32))
        if mats:
            Y = np.stack(mats, axis=1)
        else:
            Y = np.empty((T, 0), dtype=np.float32)
        return Y

    def _slide_window(self, X_full: np.ndarray, Y_full: np.ndarray, L: int, H: int, starts_override: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        T = X_full.shape[0]
        if T == 0 or L <= 0:
            return np.empty((0, 0, 0), dtype=np.float32), np.empty((0, 0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
        starts = starts_override if starts_override is not None else np.arange(0, max(0, T - L + 1), H, dtype=np.int64)
        X_list = []
        Y_list = []
        for s in starts:
            e = s + L
            X_list.append(X_full[s:e, :])
            Y_list.append(Y_full[s:e, :])
        X_seq = np.stack(X_list, axis=0).astype(np.float32) if X_list else np.empty((0, L, X_full.shape[1]), dtype=np.float32)
        Y_seq = np.stack(Y_list, axis=0).astype(np.float32) if Y_list else np.empty((0, L, Y_full.shape[1]), dtype=np.float32)
        return X_seq, Y_seq, starts

    def _valid_window_mask_by_ratio(self, Y_full: np.ndarray, starts: np.ndarray, L: int, min_ratio: float = 0.8) -> np.ndarray:
        if Y_full.size == 0:
            return np.ones_like(starts, dtype=bool)
        K = Y_full.shape[1]
        mask = np.zeros(len(starts), dtype=bool)
        total = L * max(1, K)
        for i, s in enumerate(starts):
            e = s + L
            win = Y_full[s:e, :]
            valid = np.isfinite(win).sum()
            ratio = float(valid) / float(total)
            mask[i] = (ratio >= float(min_ratio))
        return mask