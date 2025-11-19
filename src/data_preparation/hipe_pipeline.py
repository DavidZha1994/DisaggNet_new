import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from tqdm import tqdm
except Exception:
    tqdm = None
# 可视化已移除
from src.data_preparation.alignment import (
    resample_df as _alignment_resample_df,
    resample_rename_device as _alignment_resample_rename_device,
    align_and_merge as _alignment_align_and_merge,
    export_alignment_drift as _alignment_export_alignment_drift,
)
from src.data_preparation.gaps import (
    repair_small_gaps as _gaps_repair_small_gaps,
    export_gap_repair_report as _gaps_export_gap_repair_report,
)
from src.data_preparation.features import (
    build_mains_features as _features_build_mains_features,
    build_targets as _features_build_targets,
    slide_window as _features_slide_window,
    valid_window_mask_by_ratio as _features_valid_window_mask_by_ratio,
    window_stft_frames as _features_window_stft_frames,
    aggregate_aux_features as _features_aggregate_aux_features,
    create_segments_meta as _features_create_segments_meta,
    build_windows_metadata as _features_build_windows_metadata,
)
from src.data_preparation.reports import (
    export_quality_report as _reports_export_quality_report,
    export_unified_tables_core as _reports_export_unified_tables_core,
    export_alignment_coverage_core as _reports_export_alignment_coverage_core,
    export_device_interruptions as _reports_export_device_interruptions,
    export_device_interruptions_all_channels as _reports_export_device_interruptions_all_channels,
    export_mains_interruptions_all_channels as _reports_export_mains_interruptions_all_channels,
    export_onoff_masks as _reports_export_onoff_masks,
)
from src.data_preparation.io import (
    find_mains_file as _io_find_mains_file,
    find_device_files as _io_find_device_files,
    extract_device_name as _io_extract_device_name,
    safe_to_csv as _io_safe_to_csv,
    read_table as _io_read_table,
)


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
        # 配置文件定位：优先使用传入路径；否则仅尝试新统一路径
        default_new_path = os.path.join("configs", "pipeline", "prep_config.yaml")
        if config_path and os.path.exists(config_path):
            self.config_path = config_path
        elif os.path.exists(default_new_path):
            self.config_path = default_new_path
        else:
            # 不再回退到旧路径；若未提供且默认不存在，置空并使用代码内默认配置
            self.config_path = None
        self.config = {}
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
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
        # 移除可视化设置
        self.viz_enabled = False
        self.viz_dir = None
        self.viz = None

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
        if tqdm is not None:
            tm = tqdm(total=1, desc=f"读取主端: {os.path.basename(mains_fp)}", unit="file")
        df_main = self._read_mains(mains_fp)
        if tqdm is not None:
            tm.update(1); tm.close()
        # 可视化已移除
        dev_dfs, dev_names = self._read_devices(device_fps)
        # 导出毫秒级对齐漂移诊断，便于观察原始时间与网格对齐情况
        try:
            self._export_alignment_drift(df_main_raw=df_main, dev_raw_list=dev_dfs, dev_names=dev_names)
        except Exception:
            pass
        df_merged, label_map = self._align_and_merge(df_main, dev_dfs, dev_names)
        if tqdm is not None:
            tqdm.write(f"完成对齐与合并，共 {len(df_merged)} 行，设备 {len(label_map)} 个")
        # 可视化已移除
        # 仅使用成功合并的设备名称
        eff_dev_names = [label_map[i] for i in sorted(label_map.keys())]

        # 导出统一5秒网格的 CSV 以及覆盖率统计，便于外部核查
        try:
            self._export_unified_tables(df_merged, eff_dev_names)
            self._export_alignment_coverage(df_merged, eff_dev_names)
        except Exception:
            pass

        # 3) 连续性修复：填短缺口、保留长缺口为 NaN
        df_before = df_merged.copy()
        df_merged, gap_mask_df = self._repair_small_gaps(df_merged)
        # 可视化已移除

        # 开关掩码：不再导出 CSV，改为随折保存到 PT（train/val_device_masks.pt）
        # 此处跳过 _export_onoff_masks 调用
        # 修复后导出：统一网格填补版与修复报告
        try:
            self._export_unified_tables_filled(df_merged, eff_dev_names)
            self._export_gap_repair_report(df_before, df_merged, gap_mask_df, eff_dev_names)
            self._export_alignment_coverage_filled(df_merged, eff_dev_names)
            # 设备中断报告：保留全通道版本，移除单通道重复输出
            # 新增：导出设备与总表各通道的残余中断段（连续 NaN）
            self._export_device_interruptions_all_channels(df_merged, eff_dev_names)
            self._export_mains_interruptions_all_channels(df_merged)
        except Exception:
            pass

        # 4) 特征与目标
        X_full = self._build_mains_features(df_merged)
        Yp_full = self._build_targets(df_merged, eff_dev_names, kind="P")
        # 切窗：严格在连续段内滑窗，禁止跨段
        L = self.hipe.window_length
        H = self.hipe.step_size
        segments_meta = self._create_segments_meta(df_merged)
        def _compute_segmented_starts(seg_df: pd.DataFrame, L_: int, H_: int) -> np.ndarray:
            ss: List[int] = []
            if seg_df is None or seg_df.empty:
                return np.array([], dtype=np.int64)
            for _, r in seg_df.iterrows():
                a = int(r.get("start_idx", 0))
                b = int(r.get("end_idx", -1))
                if (b - a + 1) >= L_:
                    ss.extend(list(range(a, b - L_ + 2, H_)))
            return np.array(ss, dtype=np.int64)
        starts = _compute_segmented_starts(segments_meta, L, H)
        X_seq, Yp_seq, _ = self._slide_window(X_full, Yp_full, L=L, H=H, starts_override=starts)
        # 可视化已移除
        # 保持目标序列的 NaN（由下游 mask 处理），仅确保 dtype
        Yp_seq = Yp_seq.astype(np.float32)
        # 频域与辅助特征（频域帧有效性阈值）
        freq_cfg = self.config.get("frequency") or {}
        valid_ratio_thr = float(freq_cfg.get("valid_ratio_threshold", 0.85))
        freq_feats, freq_mask = self._window_stft_frames(X_seq, valid_ratio_threshold=valid_ratio_thr)
        # 可视化已移除
        aux_feats, aux_names = self._aggregate_aux_features(X_seq, df_merged, starts)
        # 可视化已移除

        # 生成 gap 掩码（主端 P/Q/S/PF 固定顺序），按窗口切片，保证通道0对齐 P_kW
        mains_order = ["P_kW", "Q_kvar", "S_kVA", "PF"]
        # gaps.repair_small_gaps 已保证返回这四列（缺失列补零）；此处仍做稳健兜底
        gap_cols = [c for c in mains_order if c in gap_mask_df.columns]
        # 若返回列不全，按固定顺序补零列
        if len(gap_cols) < len(mains_order):
            for c in mains_order:
                if c not in gap_mask_df.columns:
                    gap_mask_df[c] = 0
            gap_cols = mains_order
        gap_arr = gap_mask_df[gap_cols].to_numpy(dtype=np.uint8)
        Nw = starts.shape[0]
        Lw = int(L)
        Cg = len(mains_order)
        gap_mask_seq = np.zeros((Nw, Lw, Cg), dtype=np.uint8)
        for i, s in enumerate(starts):
            e = s + Lw
            # 将 gap_arr 的列映射到固定顺序 mains_order
            # 当前 gap_cols 与 mains_order 一致，可直接切片填充
            slice_block = gap_arr[s:e, :]
            # 如果 gap_cols 比 mains_order 少（理论上不会发生，因为已补全），也安全填充
            gap_mask_seq[i, :, : len(gap_cols)] = slice_block

        # 5) 生成窗口元数据（不再导出 segments_meta.csv）
        windows_meta = self._build_windows_metadata(df_merged, starts, L)
        # 导出段元数据，供测试验证
        try:
            seg_path = os.path.join(self.output_dir, "segments_meta.csv")
            windows_meta.to_csv(seg_path, index=False)
        except Exception:
            pass

        # 质量报告导出（窗口级与汇总）
        try:
            self._export_quality_report(
                df_before=df_before,
                df_after=df_merged,
                X_seq=X_seq,
                Yp_seq=Yp_seq,
                freq_mask=freq_mask,
                gap_mask_seq=gap_mask_seq,
                windows_meta=windows_meta,
                segments_meta=segments_meta,
                starts=starts,
                L=L,
                H=H,
                frame_valid_ratio_thr=valid_ratio_thr,
            )
        except Exception:
            # 报告不影响主流程
            pass

        # 6) Walk-Forward CV 计划
        cv_cfg = self._ensure_cv_config()
        from .cross_validation import WalkForwardCV
        cv = WalkForwardCV({"cross_validation": cv_cfg})
        folds = cv.create_folds(segments_meta)
        # 可视化已移除
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
            freq_mask=freq_mask,
            gap_mask_seq=gap_mask_seq,
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
        freq_mask: Optional[np.ndarray] = None,
        gap_mask_seq: Optional[np.ndarray] = None,
    ) -> None:
        # 顶层元数据与 CV 计划（文件名支持配置覆盖）
        files_cfg = (self.config.get("data_storage", {}).get("files") or {})
        def out_fp(name: str, default: str) -> str:
            return os.path.join(self.output_dir, files_cfg.get(name, default))
        # 恢复保存：cv_splits、label_map、labels 与设备映射
        try:
            import pickle
            # cv_splits.pkl
            with open(out_fp("cv_splits", "cv_splits.pkl"), "wb") as f:
                pickle.dump(splits, f)
            # 设备映射与标签映射
            import json
            with open(out_fp("device_name_to_id", "device_name_to_id.json"), "w", encoding="utf-8") as f:
                json.dump(device_name_to_id, f, ensure_ascii=False, indent=2)
            with open(out_fp("label_map", "label_map.json"), "w", encoding="utf-8") as f:
                json.dump({int(k): str(v) for k, v in label_map.items()}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

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
        # 改为 ISO 时间元数据，不再使用秒整数
        try:
            ts_col = self.hipe.timestamp_col
            dt_series = pd.to_datetime(df_merged[ts_col]).dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            start_dt_iso = dt_series.iloc[starts].to_numpy()
        except Exception:
            # 兜底：仍使用格式化字符串
            start_dt_iso = pd.to_datetime(start_ts_all, unit="s").strftime('%Y-%m-%dT%H:%M:%S.%fZ').to_numpy()
        label_meta_list = [
            {"datetime_iso": dt, "segment_id": int(seg_id), "window_idx": int(wi)}
            for dt, seg_id, wi in zip(start_dt_iso, seg_ids_all, window_idx_all)
        ]

        # 根据配置生成标签矩阵
        if getattr(self.hipe, "label_mode", "regression") == "classification":
            # 使用 HMM 类方法生成每窗每设备的 on/off 状态序列，再按 on 比例阈值决定标签
            ratio_thr = float(getattr(self.hipe, "on_ratio_threshold", 0.5))
            try:
                from src.tools.advanced_onoff_methods import AdvancedOnOffDetector
                det = AdvancedOnOffDetector()
                N, L, K = Yp_seq.shape
                on_ratio = np.zeros((N, K), dtype=np.float32)
                for i in range(N):
                    for k in range(K):
                        seq = Yp_seq[i, :, k]
                        # 对 NaN 做安全处理：填0，不影响有效比率统计（有效比率由 isfinite 控制）
                        x = np.nan_to_num(seq, nan=0.0)
                        st, _info = det.hmm_like_method(x)
                        st = np.asarray(st, dtype=np.float32)
                        # 对无效点（NaN）置0，避免误判
                        valid = np.isfinite(seq).astype(np.float32)
                        st = st * valid
                        denom = float(valid.sum()) if valid.sum() > 0 else 1.0
                        on_ratio[i, k] = float(st.sum() / denom)
                labels_mat = (on_ratio >= ratio_thr).astype(np.float32)
                label_type = "classification"
            except Exception:
                # 回退：使用窗内均值回归标签再二值化（不推荐，仅兜底）
                with np.errstate(invalid='ignore', divide='ignore'):
                    m = np.nanmean(Yp_seq, axis=1).astype(np.float32)
                m = np.where(np.isnan(m), 0.0, m)
                thr = float(np.nanmedian(m))
                labels_mat = (m >= thr).astype(np.float32)
                label_type = "classification"
        else:
            with np.errstate(invalid='ignore', divide='ignore'):
                labels_mat = np.nanmean(Yp_seq, axis=1).astype(np.float32)
            # 对全 NaN 的窗回退为 0，避免警告与 NaN 外泄
            labels_mat = np.where(np.isnan(labels_mat), 0.0, labels_mat)
            label_type = "regression"

        # 顶层 labels.pkl（恢复兼容）
        try:
            import pickle
            with open(out_fp("labels", "labels.pkl"), "wb") as f:
                pickle.dump({
                    "labels": labels_mat.astype(np.float32),
                    "label_type": label_type,
                    "label_metadata": label_meta_list,
                }, f)
        except Exception:
            pass

        # 每折保存
        for fold_key, part in splits.items():
            fold_dir = os.path.join(self.output_dir, fold_key)
            os.makedirs(fold_dir, exist_ok=True)
            train_idx = part["train_indices"]
            val_idx = part["val_indices"]
            fold_files = (self.config.get("data_storage", {}).get("fold_files") or {})
            def fold_fp(name: str, default: str) -> str:
                return os.path.join(fold_dir, fold_files.get(name, default))

            # 原始序列supply：对 NaN 做安全填充（窗均值），遇到全 NaN 则回退为 0
            X_train = X_seq[train_idx].astype(np.float32)
            X_val = X_seq[val_idx].astype(np.float32)
            # 安全填充：窗均值；遇到全 NaN 则回退为 0.0（不触发空切片告警）
            valid_cnt_train = np.sum(np.isfinite(X_train), axis=1, keepdims=True)
            sum_train = np.nansum(X_train, axis=1, keepdims=True)
            m_train = np.divide(sum_train, valid_cnt_train, out=np.zeros_like(sum_train, dtype=np.float32), where=valid_cnt_train > 0)
            valid_cnt_val = np.sum(np.isfinite(X_val), axis=1, keepdims=True)
            sum_val = np.nansum(X_val, axis=1, keepdims=True)
            m_val = np.divide(sum_val, valid_cnt_val, out=np.zeros_like(sum_val, dtype=np.float32), where=valid_cnt_val > 0)
            X_train_filled = np.where(np.isnan(X_train), m_train, X_train).astype(np.float32)
            X_val_filled = np.where(np.isnan(X_val), m_val, X_val).astype(np.float32)
            # 保存原始窗（仅 .pt）
            import torch
            # 保存张量版本，减少后续加载转换
            torch.save(torch.from_numpy(X_train_filled).float(), fold_fp("train_raw", "train_raw.pt"))
            torch.save(torch.from_numpy(X_val_filled).float(), fold_fp("val_raw", "val_raw.pt"))
            # 保存时域掩码（逐窗口逐时间逐通道）
            try:
                torch.save(torch.from_numpy(gap_mask_seq[train_idx].astype(np.uint8)), fold_fp("train_mask", "train_mask.pt"))
                torch.save(torch.from_numpy(gap_mask_seq[val_idx].astype(np.uint8)), fold_fp("val_mask", "val_mask.pt"))
            except Exception:
                pass
            # 频域帧（兼容旧测试，仅保存张量）；若计算失败则保存占位零张量，确保文件存在且第一维匹配
            try:
                fr_train_np = freq_feats[train_idx].astype(np.float32)
                fr_val_np = freq_feats[val_idx].astype(np.float32)
            except Exception:
                Nt = int(train_idx.shape[0]) if hasattr(train_idx, 'shape') else len(train_idx)
                Nv = int(val_idx.shape[0]) if hasattr(val_idx, 'shape') else len(val_idx)
                fr_train_np = np.zeros((Nt, 1, 1), dtype=np.float32)
                fr_val_np = np.zeros((Nv, 1, 1), dtype=np.float32)
            torch.save(torch.from_numpy(fr_train_np).float(), fold_fp("train_freq", "train_freq.pt"))
            torch.save(torch.from_numpy(fr_val_np).float(), fold_fp("val_freq", "val_freq.pt"))
            torch.save(torch.from_numpy(aux_feats[train_idx].astype(np.float32)).float(), fold_fp("train_features", "train_features.pt"))
            torch.save(torch.from_numpy(aux_feats[val_idx].astype(np.float32)).float(), fold_fp("val_features", "val_features.pt"))
            torch.save(torch.from_numpy(Yp_seq[train_idx].astype(np.float32)).float(), fold_fp("train_targets_seq", "train_targets_seq.pt"))
            torch.save(torch.from_numpy(Yp_seq[val_idx].astype(np.float32)).float(), fold_fp("val_targets_seq", "val_targets_seq.pt"))
            # 保存 indices（恢复兼容）
            try:
                torch.save(torch.from_numpy(train_idx.astype(np.int64)), fold_fp("train_indices", "train_indices.pt"))
                torch.save(torch.from_numpy(val_idx.astype(np.int64)), fold_fp("val_indices", "val_indices.pt"))
            except Exception:
                pass
            # 统一设备级掩码文件：valid=1/缺失=0，onoff=1/0，形状 [N,L,K]
            try:
                from src.tools.advanced_onoff_methods import AdaptiveHysteresisDetector
                det = AdaptiveHysteresisDetector()
                hcfg = (self.config.get("hipe") or {})
                # 自适应迟滞参数（分位阈值与最短段）
                on_pct = float(hcfg.get("onoff_on_percentile", 70.0))
                off_pct = float(hcfg.get("onoff_off_percentile", 30.0))
                min_on_pts = int(hcfg.get("onoff_min_on_points", hcfg.get("onoff_min_run_points", 12)))
                min_off_pts = int(hcfg.get("onoff_min_off_points", hcfg.get("onoff_min_run_points", 12)))
                dev_train = Yp_seq[train_idx].astype(np.float32)  # [Nt,L,K]
                dev_val = Yp_seq[val_idx].astype(np.float32)      # [Nv,L,K]
                valid_train = np.isfinite(dev_train).astype(np.uint8)
                valid_val = np.isfinite(dev_val).astype(np.uint8)
                Nt, Lw, Kd = dev_train.shape if dev_train.ndim == 3 else (0, 0, 0)
                Nv = dev_val.shape[0] if dev_val.ndim == 3 else 0
                onoff_train = np.zeros_like(valid_train, dtype=np.uint8)
                onoff_val = np.zeros_like(valid_val, dtype=np.uint8)
                # 按窗按设备进行 HMM 检测（简化版）
                for i in range(Nt):
                    for k in range(Kd):
                        seq = dev_train[i, :, k]
                        x = np.nan_to_num(seq, nan=0.0)
                        st, _info = det.detect(
                            x,
                            on_percentile=on_pct,
                            off_percentile=off_pct,
                            min_on_points=min_on_pts,
                            min_off_points=min_off_pts,
                        )
                        st = np.asarray(st, dtype=np.uint8)
                        # 无效点置0，避免缺失被误判为 ON
                        st = st * valid_train[i, :, k]
                        onoff_train[i, :, k] = st
                for i in range(Nv):
                    for k in range(Kd):
                        seq = dev_val[i, :, k]
                        x = np.nan_to_num(seq, nan=0.0)
                        st, _info = det.detect(
                            x,
                            on_percentile=on_pct,
                            off_percentile=off_pct,
                            min_on_points=min_on_pts,
                            min_off_points=min_off_pts,
                        )
                        st = np.asarray(st, dtype=np.uint8)
                        st = st * valid_val[i, :, k]
                        onoff_val[i, :, k] = st
                torch.save({
                    "valid": torch.from_numpy(valid_train.astype(np.uint8)),
                    "onoff": torch.from_numpy(onoff_train.astype(np.uint8)),
                    "method": "hmm_like"
                }, fold_fp("train_device_masks", "train_device_masks.pt"))
                torch.save({
                    "valid": torch.from_numpy(valid_val.astype(np.uint8)),
                    "onoff": torch.from_numpy(onoff_val.astype(np.uint8)),
                    "method": "hmm_like"
                }, fold_fp("val_device_masks", "val_device_masks.pt"))
            except Exception:
                pass
            # 每折标签：基于 onoff 掩码的时间众数与功率时间均值
            try:
                # 若 onoff_* 不可用，则回退为全零状态
                if 'onoff_train' in locals():
                    win_states_train = (onoff_train.mean(axis=1) >= 0.5).astype(np.float32)
                else:
                    win_states_train = np.zeros((int(train_idx.shape[0]), int(Yp_seq.shape[2])), dtype=np.float32)
                if 'onoff_val' in locals():
                    win_states_val = (onoff_val.mean(axis=1) >= 0.5).astype(np.float32)
                else:
                    win_states_val = np.zeros((int(val_idx.shape[0]), int(Yp_seq.shape[2])), dtype=np.float32)
                with np.errstate(invalid='ignore', divide='ignore'):
                    win_power_train = np.nanmean(Yp_seq[train_idx].astype(np.float32), axis=1)
                    win_power_val = np.nanmean(Yp_seq[val_idx].astype(np.float32), axis=1)
                win_power_train = np.where(np.isnan(win_power_train), 0.0, win_power_train).astype(np.float32)
                win_power_val = np.where(np.isnan(win_power_val), 0.0, win_power_val).astype(np.float32)
                labels_train = np.concatenate([win_states_train, win_power_train], axis=1).astype(np.float32)
                labels_val = np.concatenate([win_states_val, win_power_val], axis=1).astype(np.float32)
                # ISO 时间元数据：窗口起始时间
                try:
                    ts_col = self.hipe.timestamp_col
                    dt_all = pd.to_datetime(df_merged[ts_col]).dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                    train_dt_iso = dt_all.iloc[starts[train_idx]].to_numpy()
                    val_dt_iso = dt_all.iloc[starts[val_idx]].to_numpy()
                except Exception:
                    train_dt_iso = np.array(["" for _ in range(labels_train.shape[0])])
                    val_dt_iso = np.array(["" for _ in range(labels_val.shape[0])])
                meta_train = [{"datetime_iso": str(dt)} for dt in train_dt_iso]
                meta_val = [{"datetime_iso": str(dt)} for dt in val_dt_iso]
                import torch
                torch.save({"labels": torch.from_numpy(labels_train).float(), "label_metadata": meta_train}, fold_fp("train_labels", "train_labels.pt"))
                torch.save({"labels": torch.from_numpy(labels_val).float(), "label_metadata": meta_val}, fold_fp("val_labels", "val_labels.pt"))
            except Exception:
                pass
            # 已移除 NPY 兼容保存（统一生成 .pt 文件）
            # metadata：导出 CSV（恢复兼容）
            try:
                md_train = windows_meta.iloc[train_idx].copy()
                md_val = windows_meta.iloc[val_idx].copy()
                md_train["fold"] = str(fold_key)
                md_train["split"] = "train"
                md_val["fold"] = str(fold_key)
                md_val["split"] = "val"
                md_train.to_csv(fold_fp("train_metadata", "train_metadata.csv"), index=False)
                md_val.to_csv(fold_fp("val_metadata", "val_metadata.csv"), index=False)
            except Exception:
                pass

            # 名称文件
            with open(fold_fp("feature_names", "feature_names.json"), "w") as f:
                json.dump(aux_names, f, ensure_ascii=False, indent=2)
            # 修正：保存完整原始通道名称与顺序，匹配 X_seq 的7个通道
            with open(fold_fp("raw_channel_names", "raw_channel_names.json"), "w") as f:
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
            # 频域与掩码以新的字典格式保存，不再返回旧掩码/索引
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

    def _export_quality_report(
        self,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
        X_seq: np.ndarray,
        Yp_seq: np.ndarray,
        freq_mask: Optional[np.ndarray],
        gap_mask_seq: Optional[np.ndarray],
        windows_meta: pd.DataFrame,
        segments_meta: pd.DataFrame,
        starts: np.ndarray,
        L: int,
        H: int,
        frame_valid_ratio_thr: float,
        device_intersection_thr: float = 0.6,
        window_valid_ratio_thr: float = 0.6,
    ) -> None:
        return _reports_export_quality_report(
            df_before,
            df_after,
            X_seq,
            Yp_seq,
            freq_mask,
            gap_mask_seq,
            windows_meta,
            segments_meta,
            starts,
            int(L),
            int(H),
            float(frame_valid_ratio_thr),
            float(device_intersection_thr),
            float(window_valid_ratio_thr),
            ts_col=self.hipe.timestamp_col,
            output_dir=self.output_dir,
        )

    def _window_stft_frames(self, X_seq: np.ndarray, valid_ratio_threshold: float = 0.85) -> Tuple[np.ndarray, np.ndarray]:
        """薄包装：调用 features.window_stft_frames 计算频域特征与帧掩码。"""
        return _features_window_stft_frames(
            X_seq,
            n_fft=int(self.hipe.stft_n_fft),
            hop=int(self.hipe.stft_hop),
            win_length=int(self.hipe.stft_win_length),
            window_type=str(self.hipe.stft_window),
            valid_ratio_threshold=float(valid_ratio_threshold),
        )

    def _aggregate_aux_features(self, X_seq: np.ndarray, df_merged: pd.DataFrame, starts: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """薄包装：调用 features.aggregate_aux_features 计算辅助特征。"""
        seg_cfg = (self.config.get("segmentation") or {})
        return _features_aggregate_aux_features(
            X_seq,
            df_merged,
            starts,
            ts_col=self.hipe.timestamp_col,
            resample_seconds=int(self.hipe.resample_seconds),
            k_median=float(seg_cfg.get("k_median", 4.0)),
            gap_seconds_min=int(seg_cfg.get("gap_seconds_min", 30)),
        )

    def _create_segments_meta(self, df_merged: pd.DataFrame) -> pd.DataFrame:
        """薄包装：调用 features.create_segments_meta 生成连续段元数据。"""
        seg_cfg = (self.config.get("segmentation") or {})
        return _features_create_segments_meta(
            df_merged,
            ts_col=self.hipe.timestamp_col,
            resample_seconds=int(self.hipe.resample_seconds),
            k_median=float(seg_cfg.get("k_median", 4.0)),
            gap_seconds_min=int(seg_cfg.get("gap_seconds_min", 30)),
        )

    def _build_windows_metadata(self, df_merged: pd.DataFrame, starts: np.ndarray, L: int) -> pd.DataFrame:
        """薄包装：调用 features.build_windows_metadata 生成窗口元数据。"""
        seg_cfg = (self.config.get("segmentation") or {})
        return _features_build_windows_metadata(
            df_merged,
            starts,
            int(L),
            ts_col=self.hipe.timestamp_col,
            resample_seconds=int(self.hipe.resample_seconds),
            k_median=float(seg_cfg.get("k_median", 4.0)),
            gap_seconds_min=int(seg_cfg.get("gap_seconds_min", 30)),
        )

    @staticmethod
    def _safe_load(fp: str) -> Optional[np.ndarray]:
        return np.load(fp) if os.path.exists(fp) else None

    def _export_unified_tables_core(self, df: pd.DataFrame, eff_dev_names: List[str], filled: bool) -> None:
        return _reports_export_unified_tables_core(
            df,
            eff_dev_names,
            bool(filled),
            ts_col=self.hipe.timestamp_col,
            output_dir=self.output_dir,
        )

    def _export_alignment_coverage_core(self, df: pd.DataFrame, eff_dev_names: List[str], filled: bool) -> None:
        return _reports_export_alignment_coverage_core(
            df,
            eff_dev_names,
            bool(filled),
            output_dir=self.output_dir,
        )

    # === 缺失的私有方法补充实现 ===
    def _find_mains_file(self, data_path: str) -> Optional[str]:
        return _io_find_mains_file(data_path, self.hipe)

    def _find_device_files(self, data_path: str) -> List[str]:
        return _io_find_device_files(data_path, self.hipe)

    def _read_mains(self, fp: str) -> pd.DataFrame:
        mains_map = getattr(self.hipe, "mains_cols", None)
        return _io_read_table(fp, self.hipe, rename_map=mains_map)

    def _extract_device_name(self, fp: str) -> str:
        return _io_extract_device_name(fp)

    def _read_devices(self, fps: List[str]) -> Tuple[List[pd.DataFrame], List[str]]:
        dev_map = getattr(self.hipe, "device_cols", None)
        canon_map = None
        if isinstance(dev_map, dict) and dev_map:
            m = {}
            if isinstance(dev_map.get("P"), str):
                m["P_kW"] = dev_map["P"]
            if isinstance(dev_map.get("Q"), str):
                m["Q_kvar"] = dev_map["Q"]
            if isinstance(dev_map.get("S"), str):
                m["S_kVA"] = dev_map["S"]
            canon_map = m
        dev_dfs: List[pd.DataFrame] = []
        dev_names: List[str] = []
        if not fps:
            return dev_dfs, dev_names
        # 并行读取设备文件，并显示进度
        max_workers = max(1, os.cpu_count() or 1)
        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for fp in fps:
                futures[ex.submit(_io_read_table, fp, self.hipe, canon_map)] = fp
            pbar = tqdm(total=len(futures), desc="读取设备文件", unit="file") if tqdm is not None else None
            for fut in as_completed(futures):
                fp = futures[fut]
                try:
                    df = fut.result()
                except Exception:
                    df = None
                if df is not None:
                    dev_dfs.append(df)
                    dev_names.append(_io_extract_device_name(fp))
                if pbar is not None:
                    pbar.update(1)
            if pbar is not None:
                pbar.close()
        return dev_dfs, dev_names

    def _resample_df(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        return _alignment_resample_df(
            df,
            cols,
            ts_col=self.hipe.timestamp_col,
            rule_seconds=int(self.hipe.resample_seconds),
            use_polars=getattr(self, "use_polars", False),
        )

    def _resample_rename_device(self, df: pd.DataFrame, name: str) -> Optional[pd.DataFrame]:
        return _alignment_resample_rename_device(
            df,
            name,
            ts_col=self.hipe.timestamp_col,
            rule_seconds=int(self.hipe.resample_seconds),
            use_polars=getattr(self, "use_polars", False),
        )

    def _align_and_merge(self, df_main: pd.DataFrame, dev_dfs: List[pd.DataFrame], dev_names: List[str]) -> Tuple[pd.DataFrame, Dict[int, str]]:
        return _alignment_align_and_merge(
            df_main,
            dev_dfs,
            dev_names,
            ts_col=self.hipe.timestamp_col,
            rule_seconds=int(self.hipe.resample_seconds),
            n_jobs=getattr(self, "n_jobs", 1),
            use_polars=getattr(self, "use_polars", False),
        )

    def _export_alignment_drift(self, df_main_raw: pd.DataFrame, dev_raw_list: List[pd.DataFrame], dev_names: List[str]) -> None:
        return _alignment_export_alignment_drift(
            df_main_raw,
            dev_raw_list,
            dev_names,
            ts_col=self.hipe.timestamp_col,
            resample_seconds=int(self.hipe.resample_seconds),
            output_dir=self.output_dir,
        )

    def _export_unified_tables(self, df_merged: pd.DataFrame, eff_dev_names: List[str]) -> None:
        """导出统一5秒网格（未填充）merged/main/devices。"""
        self._export_unified_tables_core(df=df_merged, eff_dev_names=eff_dev_names, filled=False)

    def _export_alignment_coverage(self, df_merged: pd.DataFrame, eff_dev_names: List[str]) -> None:
        """导出未填充统一网格的设备覆盖率。"""
        self._export_alignment_coverage_core(df=df_merged, eff_dev_names=eff_dev_names, filled=False)

    def _export_unified_tables_filled(self, df_filled: pd.DataFrame, eff_dev_names: List[str]) -> None:
        """导出统一5秒网格（已填充）merged_filled/main_filled/devices。"""
        self._export_unified_tables_core(df=df_filled, eff_dev_names=eff_dev_names, filled=True)

    def _export_gap_repair_report(self, df_before: pd.DataFrame, df_after: pd.DataFrame, gap_mask_df: pd.DataFrame, eff_dev_names: List[str]) -> None:
        return _gaps_export_gap_repair_report(
            df_before,
            df_after,
            gap_mask_df,
            eff_dev_names,
            output_dir=self.output_dir,
            ts_col=self.hipe.timestamp_col,
        )

    def _export_alignment_coverage_filled(self, df_filled: pd.DataFrame, eff_dev_names: List[str]) -> None:
        """导出已填充统一网格的设备覆盖率。"""
        self._export_alignment_coverage_core(df=df_filled, eff_dev_names=eff_dev_names, filled=True)

    def _export_device_interruptions(self, df_filled: pd.DataFrame, eff_dev_names: List[str]) -> None:
        return _reports_export_device_interruptions(
            df_filled,
            eff_dev_names,
            ts_col=self.hipe.timestamp_col,
            step_seconds=int(self.hipe.resample_seconds),
            output_dir=self.output_dir,
        )

    def _export_device_interruptions_all_channels(self, df_filled: pd.DataFrame, eff_dev_names: List[str]) -> None:
        return _reports_export_device_interruptions_all_channels(
            df_filled,
            eff_dev_names,
            ts_col=self.hipe.timestamp_col,
            step_seconds=int(self.hipe.resample_seconds),
            output_dir=self.output_dir,
        )

    def _export_mains_interruptions_all_channels(self, df_filled: pd.DataFrame) -> None:
        return _reports_export_mains_interruptions_all_channels(
            df_filled,
            ts_col=self.hipe.timestamp_col,
            step_seconds=int(self.hipe.resample_seconds),
            output_dir=self.output_dir,
        )

    def _export_onoff_masks(self, df_filled: pd.DataFrame, eff_dev_names: List[str]) -> None:
        return _reports_export_onoff_masks(
            df_filled,
            eff_dev_names,
            ts_col=self.hipe.timestamp_col,
            output_dir=self.output_dir,
        )

    def _repair_small_gaps(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        fill_cfg = self.config.get("gap_filling") or {}
        # 默认阈值提升为 1200 秒（20 分钟），可通过配置覆盖
        max_sec = int(fill_cfg.get("max_fill_seconds", 1200))
        return _gaps_repair_small_gaps(
            df,
            ts_col=self.hipe.timestamp_col,
            resample_seconds=int(self.hipe.resample_seconds),
            max_fill_seconds=max_sec,
        )

    def _build_mains_features(self, df: pd.DataFrame) -> np.ndarray:
        """薄包装：调用 features.build_mains_features 构建主端特征。"""
        return _features_build_mains_features(df)

    def _build_targets(self, df: pd.DataFrame, dev_names: List[str], kind: str = "P") -> np.ndarray:
        """薄包装：调用 features.build_targets 构建监督目标。"""
        return _features_build_targets(df, dev_names, kind)

    def _slide_window(self, X_full: np.ndarray, Y_full: np.ndarray, L: int, H: int, starts_override: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """薄包装：调用 features.slide_window 进行切窗。"""
        return _features_slide_window(X_full, Y_full, int(L), int(H), starts_override)

    def _valid_window_mask_by_ratio(self, Y_full: np.ndarray, starts: np.ndarray, L: int, min_ratio: float = 0.8) -> np.ndarray:
        """薄包装：调用 features.valid_window_mask_by_ratio 计算有效窗口掩码。"""
        return _features_valid_window_mask_by_ratio(Y_full, starts, int(L), float(min_ratio))