import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml

# helpers for REFIT / UKDALE
from src.helpers.preprocessing import (
    UKDALE_DataBuilder,
    REFIT_DataBuilder,
    create_exogene as _helpers_create_exogene,
)

# reuse HIPE features and CV for frequency and splitting
from src.data_preparation.features import (
    window_stft_frames as _features_window_stft_frames,
)
from src.data_preparation.cross_validation import WalkForwardCV


@dataclass
class RUConfig:
    dataset: str = "UKDALE"
    data_path: str = "Data/"
    result_path: str = "Data/prepared"
    sampling_rate: str = "1min"
    window_size: int = 256
    window_stride: Optional[int] = None
    list_exo_variables: Optional[List[str]] = None
    # STFT defaults (can be overridden by configs/pipeline/prep_config.yaml)
    stft_n_fft: int = 256
    stft_hop: int = 64
    stft_win_length: int = 256
    stft_window: str = "hann"


class REFITUKDALEPipeline:
    """
    REFIT/UKDALE 数据准备管线：
    - 读取 helpers 中的 DataBuilder（单位均为 W）
    - 生成时域原始窗口（聚合功率与其导数通道）
    - 生成频域 STFT 帧（来自 HIPE 的 features.window_stft_frames）
    - 生成手工特征（时间外生变量：hour/dow/month/minute）
    - 按 HIPE 风格进行 Walk-Forward 划分并保存 .pt 文件
    - 使用 configs/expes.yaml 与 configs/datasets.yaml 的配置
    """

    def __init__(self, expes_cfg_path: Optional[str] = None, datasets_cfg_path: Optional[str] = None, prep_cfg_path: Optional[str] = None, dataset_override: Optional[str] = None):
        self.expes_cfg_path = expes_cfg_path or os.path.join("configs", "expes.yaml")
        self.datasets_cfg_path = datasets_cfg_path or os.path.join("configs", "datasets.yaml")
        self.prep_cfg_path = prep_cfg_path or os.path.join("configs", "pipeline", "prep_config.yaml")
        self.dataset_override = dataset_override
        self.config: Dict = {}
        self.datasets_cfg: Dict = {}
        self.hipe_cfg: Dict = {}
        self.prep_cfg: Dict = {}
        self.ru = RUConfig()
        self.output_dir = "Data/prepared"
        self.summary_: Dict = {}
        self._load_configs()

    def _load_configs(self) -> None:
        """仅读取 prep_config.yaml 并初始化管线配置。
        - STFT 参数取自 frequency 节点（若缺失则回退到 hipe.stft）
        - REFIT/UKDALE 基本参数取自 refit_ukdale 节点（窗口、采样、外生变量、数据根目录）
        - 数据集的 mask 与划分取自 datasets 节点（统一合并自原 datasets.yaml）
        """
        # 载入 prep_config.yaml
        if os.path.exists(self.prep_cfg_path):
            with open(self.prep_cfg_path, "r", encoding="utf-8") as f:
                self.prep_cfg = yaml.safe_load(f) or {}
        else:
            self.prep_cfg = {}

        # 选择数据集名称
        ds_name = str(self.dataset_override or self.prep_cfg.get("dataset", self.ru.dataset))

        # STFT/frequency 参数（优先 frequency，再回退 hipe.stft）
        freq = (self.prep_cfg.get("frequency", {}) or {})
        if isinstance(freq, dict) and freq:
            self.ru.stft_n_fft = int(freq.get("n_fft", self.ru.stft_n_fft))
            self.ru.stft_hop = int(freq.get("hop_length", self.ru.stft_hop))
            self.ru.stft_win_length = int(freq.get("win_length", self.ru.stft_win_length))
            self.ru.stft_window = str(freq.get("window", self.ru.stft_window))
        else:
            stft = (self.prep_cfg.get("hipe", {}) or {}).get("stft", {})
            if isinstance(stft, dict) and stft:
                self.ru.stft_n_fft = int(stft.get("n_fft", self.ru.stft_n_fft))
                self.ru.stft_hop = int(stft.get("hop_length", self.ru.stft_hop))
                self.ru.stft_win_length = int(stft.get("win_length", self.ru.stft_win_length))
                self.ru.stft_window = str(stft.get("window", self.ru.stft_window))

        # REFIT/UKDALE 参数
        ru_cfg = (self.prep_cfg.get("refit_ukdale", {}) or {})
        self.ru.sampling_rate = str(ru_cfg.get("sampling_rate", self.ru.sampling_rate))
        self.ru.window_size = int(ru_cfg.get("window_size", self.ru.window_size))
        self.ru.window_stride = int(ru_cfg.get("window_stride", self.ru.window_size))
        self.ru.list_exo_variables = ru_cfg.get("list_exo_variables", ["minute", "hour", "dow", "month"]) or ["minute", "hour", "dow", "month"]
        data_root = str(ru_cfg.get("data_root", "Data"))

        # 数据目录拼接
        base = data_root.rstrip(os.sep)
        if ds_name.upper() == "UKDALE":
            data_path = os.path.join(base, "UKDALE") + os.sep
            if not os.path.exists(data_path):
                alt_base = "data"
                data_path_alt = os.path.join(alt_base, "UKDALE") + os.sep
                if os.path.exists(data_path_alt):
                    data_path = data_path_alt
        elif ds_name.upper() == "REFIT":
            data_path = os.path.join(base, "REFIT", "RAW_DATA_CLEAN") + os.sep
            if not os.path.exists(data_path):
                alt_base = "data"
                data_path_alt = os.path.join(alt_base, "REFIT", "RAW_DATA_CLEAN") + os.sep
                if os.path.exists(data_path_alt):
                    data_path = data_path_alt
        else:
            raise ValueError(f"未知数据集 {ds_name}，仅支持 REFIT 或 UKDALE")

        self.ru.dataset = ds_name
        self.ru.data_path = data_path
        # 输出目录：优先 prep_config.data_storage.output_directory，其次 RUConfig.result_path，最后默认
        try:
            ds_cfg = (self.prep_cfg.get("data_storage", {}) or {})
            out_dir_cfg = str(ds_cfg.get("output_directory", "") or "").strip()
        except Exception:
            out_dir_cfg = ""
        if out_dir_cfg:
            self.output_dir = out_dir_cfg
        else:
            self.output_dir = self.ru.result_path or self.output_dir or "Data/prepared"
        os.makedirs(self.output_dir, exist_ok=True)

        # 合并数据集定义（来自 prep_config.yaml 的 datasets 节点）
        self.datasets_cfg = (self.prep_cfg.get("datasets", {}) or {})

    def get_pipeline_summary(self) -> Dict:
        return self.summary_.copy()

    def _select_mask_and_splits(self) -> Tuple[List[str], Dict[str, List[int]]]:
        """从 datasets.yaml 中读取 mask_app 和 house splits"""
        ds_block = self.datasets_cfg.get(self.ru.dataset, {}) or {}
        mask_app: List[str] = []
        splits: Dict[str, List[int]] = {"train": [], "valid": [], "test": []}

        # 优先使用统一结构：mask_app + splits
        mask_app = list(ds_block.get("mask_app", []) or [])
        _splits = ds_block.get("splits", {}) or {}
        if _splits:
            splits["train"] = list(_splits.get("train", []) or [])
            splits["valid"] = list(_splits.get("valid", []) or [])
            splits["test"] = list(_splits.get("test", []) or [])

        # 若未提供统一结构，则兼容原 datasets.yaml 的按设备定义
        if not mask_app:
            if self.ru.dataset.upper() == "UKDALE":
                for _, app_cfg in ds_block.items():
                    app_name = str(app_cfg.get("app", "")).strip()
                    if app_name:
                        mask_app.append(app_name)
                        if not splits["train"]:
                            splits["train"] = list(app_cfg.get("ind_house_train", []) or [])
                            splits["valid"] = list(app_cfg.get("ind_house_valid", []) or [])
                            splits["test"] = list(app_cfg.get("ind_house_test", []) or [])
            elif self.ru.dataset.upper() == "REFIT":
                for _, app_cfg in ds_block.items():
                    app_name = str(app_cfg.get("app", "")).strip()
                    if app_name:
                        mask_app.append(app_name)
                        if not splits["train"] and not splits["test"]:
                            houses = list(app_cfg.get("house_with_app_i", []) or [])
                            if houses:
                                n = len(houses)
                                n_test = max(1, int(round(0.2 * n)))
                                splits["test"] = houses[:n_test]
                                splits["train"] = houses[n_test:]
                                splits["valid"] = splits["train"][-1:] if len(splits["train"]) > 1 else splits["train"][:1]
            else:
                raise ValueError(f"未知数据集 {self.ru.dataset}，仅支持 REFIT 或 UKDALE")

        if not mask_app:
            raise ValueError("未能从 prep_config.yaml 的 datasets 节点解析到有效的 mask_app 列表")
        return mask_app, splits

    def _build_data(self, house_indices: List[int]) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
        """调用 helpers DataBuilder 生成 NILM 数据集；返回 (X, st_date, device_names)
        X: [N, M, 2, L]: M 包含 aggregate 及每个设备；2: [power, status]
        st_date: DataFrame，index 为 house id，列 'start_date'
        device_names: mask_app（不含 aggregate）
        """
        mask_app, _ = self._select_mask_and_splits()
        if self.ru.dataset.upper() == "UKDALE":
            db = UKDALE_DataBuilder(
                data_path=self.ru.data_path,
                mask_app=mask_app,
                sampling_rate=self.ru.sampling_rate,
                window_size=self.ru.window_size,
                window_stride=self.ru.window_stride,
                soft_label=False,
                use_status_from_kelly_paper=True,
            )
        else:
            db = REFIT_DataBuilder(
                data_path=self.ru.data_path,
                mask_app=mask_app,
                sampling_rate=self.ru.sampling_rate,
                window_size=self.ru.window_size,
                window_stride=self.ru.window_stride,
                use_status_from_kelly_paper=False,
                soft_label=False,
            )
        X, st_date = db.get_nilm_dataset(house_indices)
        return X, st_date, mask_app

    def _build_time_domain(self, X: np.ndarray) -> np.ndarray:
        """构建时域窗口：聚合功率与其导数；输出 [N, L, C]，单位 W。
        C=2: ['P_W','dP_W']
        """
        agg = X[:, 0, 0, :].astype(np.float32)  # [N,L]
        # derivative: safe gradient over time
        dP = np.gradient(agg, axis=1).astype(np.float32)
        raw = np.stack([agg, dP], axis=-1).astype(np.float32)
        return raw

    def _build_targets_seq(self, X: np.ndarray, device_names: List[str]) -> np.ndarray:
        """构建每窗目标序列 [N, L, K]（设备功率，单位 W）。按 device_names 顺序。"""
        N = X.shape[0]
        L = X.shape[-1]
        K = len(device_names)
        Y = np.zeros((N, L, K), dtype=np.float32)
        # helpers DataBuilder 组织：mask_app = ['aggregate'] + devices；功率在 [:, j, 0, :]
        for k, _ in enumerate(device_names):
            # device index in X = 1 + k
            Y[:, :, k] = X[:, 1 + k, 0, :].astype(np.float32)
        return Y

    def _build_status_seq(self, X: np.ndarray, device_names: List[str]) -> np.ndarray:
        """构建每窗设备开关掩码 [N, L, K]，按 device_names 顺序。"""
        N = X.shape[0]
        L = X.shape[-1]
        K = len(device_names)
        S = np.zeros((N, L, K), dtype=np.float32)
        for k, _ in enumerate(device_names):
            S[:, :, k] = X[:, 1 + k, 1, :].astype(np.float32)
        # 规范化为 {0,1}
        S = np.where(S > 0.5, 1.0, 0.0).astype(np.float32)
        return S

    def _build_labels(self, Y_seq: np.ndarray) -> np.ndarray:
        """窗口级标签：每设备窗均值功率；与 HIPE 一致，保持 shape [N,K]。"""
        with np.errstate(invalid='ignore', divide='ignore'):
            labels = np.nanmean(Y_seq, axis=1).astype(np.float32)
        labels = np.where(np.isnan(labels), 0.0, labels)
        return labels

    def _build_exogene_features(self, raw_seq: np.ndarray, st_date: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """生成窗口级手工特征：时间外生变量 embedding（对每窗按时间序列生成并在窗内均值池化）。
        返回 (aux_feats[N,F], feature_names)
        """
        N, L, C = raw_seq.shape
        feats: List[np.ndarray] = []
        # 构建每窗 exogene 时序，再做均值池化 -> 向量
        for i in range(N):
            try:
                start_dt = st_date.iloc[i]["start_date"]
            except Exception:
                # 兜底：若 st_date 匹配不上，回退为第一行
                start_dt = st_date.iloc[0]["start_date"] if "start_date" in st_date.columns else None
            # 以聚合通道的长度作为时间序列长度
            x_win = raw_seq[i, :, 0]
            exo = _helpers_create_exogene(x_win, start_dt, self.ru.list_exo_variables, self.ru.sampling_rate, cosinbase=True, new_range=(-1, 1))
            # exo shape: (1, n_var, L)
            exo_mean = np.nanmean(exo[0], axis=1).astype(np.float32)  # [n_var]
            feats.append(exo_mean)
        aux = np.stack(feats, axis=0).astype(np.float32)
        # feature names（按 list_exo_variables 的正弦/余弦展开）
        fn: List[str] = []
        for v in (self.ru.list_exo_variables or []):
            fn.append(f"{v}_sin")
            fn.append(f"{v}_cos")
        return aux, fn

    def _build_freq_frames(self, raw_seq: np.ndarray, valid_ratio_threshold: float = 0.85) -> Tuple[np.ndarray, np.ndarray]:
        return _features_window_stft_frames(
            raw_seq,  # expects shape [N,L,C]
            n_fft=int(self.ru.stft_n_fft),
            hop=int(self.ru.stft_hop),
            win_length=int(self.ru.stft_win_length),
            window_type=str(self.ru.stft_window),
            valid_ratio_threshold=float(valid_ratio_threshold),
        )

    def run_full_pipeline(self) -> Dict:
        """运行完整管线并保存到 output_dir/fold_0/"""
        # 选择 mask 与划分
        mask_app, splits = self._select_mask_and_splits()
        # 构建数据（训练+验证合并用于生成总体窗口集，后续由 Walk-Forward 划分）
        house_all = sorted(set((splits.get("train", []) or []) + (splits.get("valid", []) or [])))
        X_all, st_date_all, device_names = self._build_data(house_all)

        def _step_seconds(sr: str) -> int:
            try:
                s = str(sr).lower()
                if "min" in s:
                    return int(s.replace("min", "")) * 60
                if s.endswith("s"):
                    return int(s[:-1])
                if s.endswith("t"):
                    v = int(s[:-1])
                    return 60 * v
                return 60
            except Exception:
                return 60

        def _sequence_non_overlapping(X: np.ndarray, st_df: pd.DataFrame, win_size: int, step_s: int) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
            df = st_df.copy().reset_index()
            df = df.rename(columns={df.columns[0]: "house"}) if df.columns[0] != "house" else df
            df["start_ts"] = pd.to_datetime(df["start_date"]).astype("int64") // 1_000_000_000
            df["end_ts"] = df["start_ts"] + int(win_size) * int(step_s)
            g = df.groupby("house").agg(earliest=("start_ts", "min"), latest=("end_ts", "max")).reset_index()
            order = g.sort_values("earliest")["house"].tolist()
            selected: List[int] = []
            prev_end: Optional[int] = None
            for h in order:
                rows = df[df["house"] == h].sort_values("start_ts")
                if prev_end is None:
                    keep_idx = rows.index.tolist()
                    if len(rows) > 0:
                        prev_end = int(rows["end_ts"].max())
                else:
                    rows2 = rows[rows["start_ts"] >= int(prev_end)]
                    keep_idx = rows2.index.tolist()
                    if len(rows2) > 0:
                        prev_end = int(rows2["end_ts"].max())
                selected.extend(keep_idx)
            if len(selected) == 0:
                return X, st_df, np.arange(X.shape[0], dtype=np.int64)
            sel = np.array(selected, dtype=np.int64)
            return X[sel], st_df.iloc[sel], sel

        step_s = _step_seconds(self.ru.sampling_rate)
        X_all, st_date_all, selected_idx = _sequence_non_overlapping(X_all, st_date_all, int(self.ru.window_size), int(step_s))

        raw_all = self._build_time_domain(X_all)
        Y_seq_all = self._build_targets_seq(X_all, device_names)
        S_seq_all = self._build_status_seq(X_all, device_names)
        labels_all = self._build_labels(Y_seq_all)

        # 频域与外生特征
        freq_frames_all, freq_mask_all = self._build_freq_frames(raw_all)
        aux_all, aux_names = self._build_exogene_features(raw_all, st_date_all)

        def _to_epoch_seconds(x) -> int:
            try:
                return int(pd.to_datetime(x).timestamp())
            except Exception:
                try:
                    dt64 = np.datetime64(str(x))
                    return int(pd.Timestamp(dt64).timestamp())
                except Exception:
                    return 0
        start_ts = np.array([_to_epoch_seconds(sd) for sd in st_date_all["start_date"].tolist()], dtype=np.int64)
        end_ts = start_ts + int(self.ru.window_size) * int(step_s)
        try:
            house_ids = np.array(st_date_all.index.tolist(), dtype=np.int64)
        except Exception:
            try:
                house_ids = np.array(st_date_all.index.tolist())
            except Exception:
                house_ids = np.zeros(len(start_ts), dtype=np.int64)
        windows_meta = pd.DataFrame({
            "window_idx": np.arange(len(start_ts), dtype=np.int64),
            "start_ts": start_ts,
            "end_ts": end_ts,
            "house": house_ids,
            # 段ID：按连续时间分组（相邻窗起始差==stride*step_s 且 house 不变 视为同段）
            "segment_id": self._assign_segment_ids(start_ts, stride_seconds=int(self.ru.window_stride) * int(step_s), houses=house_ids),
        })

        # Walk-Forward CV（与 HIPE 相同接口）
        # 从 prep_config.yaml 读取交叉验证配置（带默认回退）
        _cv = (self.prep_cfg.get("cross_validation") or {}) if isinstance(self.prep_cfg, dict) else {}
        cv_cfg = {
            "cross_validation": {
                "n_folds": int(_cv.get("n_folds", 5)),
                "purge_gap_minutes": int(_cv.get("purge_gap_minutes", 10)),
                "val_span_days": float(_cv.get("val_span_days", 7.0)),
                "test_span_days": float(_cv.get("test_span_days", 0.0)),
                "min_train_days": float(_cv.get("min_train_days", 7.0)),
                "segment_isolation": bool(_cv.get("segment_isolation", True)),
                "holdout_test": bool(_cv.get("holdout_test", False)),
            }
        }
        cv = WalkForwardCV(cv_cfg)
        # 以段为单位创建折
        segments_meta = self._build_segments_meta_from_windows(windows_meta)
        folds = cv.create_folds(segments_meta)

        # 构建窗口数据集（X=aux_all，y=labels总和）
        labels_total = labels_all.sum(axis=1).astype(np.float32)
        windows_dataset = {"X": aux_all, "metadata": windows_meta, "y": labels_total}

        # 多折输出（保存到数据集子目录）
        if not folds:
            raise RuntimeError("无法创建 CV 折，请检查时间范围或段元数据")

        dataset_dir = os.path.join(self.output_dir, str(self.ru.dataset).lower())
        os.makedirs(dataset_dir, exist_ok=True)

        import torch

        # 原始时域窗口（NaN 用窗均值填充）
        def _fill_na(win: np.ndarray) -> np.ndarray:
            m = np.nanmean(win, axis=1, keepdims=True)
            m = np.where(np.isfinite(m), m, 0.0)
            return np.where(np.isnan(win), m, win).astype(np.float32)

        # 置信度：按原始时域有效比例
        def _valid_ratio(arr: np.ndarray) -> np.ndarray:
            B, L, C = arr.shape
            total = float(L * C) if L * C > 0 else 1.0
            return (np.sum(np.isfinite(arr), axis=(1, 2)) / total).astype(np.float32)

        # 缺失掩码：P_W 通道非有限值位置为 1
        def _missing_mask(arr: np.ndarray) -> np.ndarray:
            # arr: [N,L,2]，取通道 0（P_W）
            pw = arr[:, :, 0]
            mask = ~np.isfinite(pw)
            return mask.astype(np.uint8)

        # ISO 起始时间全集（按窗口索引）
        start_iso_all = pd.to_datetime(windows_meta["start_ts"], unit="s").dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ').to_numpy()

        # 顶层保存设备映射与窗口元数据
        try:
            dev_map = {name: i for i, name in enumerate(device_names)}
            with open(os.path.join(dataset_dir, "device_name_to_id.json"), "w", encoding="utf-8") as f:
                json.dump(dev_map, f, ensure_ascii=False, indent=2)
            with open(os.path.join(dataset_dir, "device_names.json"), "w", encoding="utf-8") as f:
                json.dump(device_names, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        # 窗口元数据（ISO 时间）
        try:
            df_meta = pd.DataFrame({
                "window_idx": windows_meta["window_idx"].astype(int),
                "start_iso": pd.to_datetime(windows_meta["start_ts"], unit="s").astype(str),
                "end_iso": pd.to_datetime(windows_meta["end_ts"], unit="s").astype(str),
                "segment_id": windows_meta["segment_id"].astype(int),
                "house": windows_meta["house"].astype(int) if "house" in windows_meta.columns else 0,
            })
            df_meta.to_csv(os.path.join(dataset_dir, "windows_meta.csv"), index=False)
        except Exception:
            pass

        # 保存交叉验证计划（所有折的窗口划分）
        try:
            rows = []
            for fold in folds:
                tr_ds, va_ds, _ = cv.split_windows(windows_dataset, fold)
                tr_idx = tr_ds["metadata"]["window_idx"].to_numpy(dtype=np.int64)
                va_idx = va_ds["metadata"]["window_idx"].to_numpy(dtype=np.int64)
                for w in tr_idx.tolist():
                    rows.append({"fold_id": int(fold.fold_id), "split": "train", "window_idx": int(w)})
                for w in va_idx.tolist():
                    rows.append({"fold_id": int(fold.fold_id), "split": "val", "window_idx": int(w)})
            cv_plan_df = pd.DataFrame(rows)
            cv_plan_df.to_csv(os.path.join(dataset_dir, "cv_plan.csv"), index=False)
        except Exception:
            pass

        # 遍历每个折并保存
        for fold in folds:
            fold_dir = os.path.join(dataset_dir, f"fold_{fold.fold_id}")
            os.makedirs(fold_dir, exist_ok=True)

            train_ds, val_ds, _ = cv.split_windows(windows_dataset, fold)
            train_idx = train_ds["metadata"]["window_idx"].to_numpy(dtype=np.int64)
            val_idx = val_ds["metadata"]["window_idx"].to_numpy(dtype=np.int64)
            # 保存绝对窗口索引（供 DataModule 使用泄露检查与对齐）
            torch.save(torch.from_numpy(train_idx), os.path.join(fold_dir, "train_indices.pt"))
            torch.save(torch.from_numpy(val_idx), os.path.join(fold_dir, "val_indices.pt"))

            # 原始时域窗口 + 缺失掩码（作为额外通道）
            miss_all = _missing_mask(raw_all)  # [N,L]
            X_train_raw = _fill_na(raw_all[train_idx])  # [N,L,C]
            X_val_raw = _fill_na(raw_all[val_idx])
            # 叠加缺失掩码为最后一通道（float32）
            miss_train = miss_all[train_idx].astype(np.float32)[..., np.newaxis]
            miss_val = miss_all[val_idx].astype(np.float32)[..., np.newaxis]
            X_train_raw_aug = np.concatenate([X_train_raw, miss_train], axis=2).astype(np.float32)
            X_val_raw_aug = np.concatenate([X_val_raw, miss_val], axis=2).astype(np.float32)
            torch.save(torch.from_numpy(X_train_raw_aug).float(), os.path.join(fold_dir, "train_raw.pt"))
            torch.save(torch.from_numpy(X_val_raw_aug).float(), os.path.join(fold_dir, "val_raw.pt"))

            # 频域帧（含置信度）
            fr_train = freq_frames_all[train_idx].astype(np.float32)
            fr_val = freq_frames_all[val_idx].astype(np.float32)
            conf_train = _valid_ratio(raw_all[train_idx])
            conf_val = _valid_ratio(raw_all[val_idx])
            torch.save({
                "frames": torch.from_numpy(fr_train).float(),
                "confidence": torch.from_numpy(conf_train).float(),
            }, os.path.join(fold_dir, "train_freq.pt"))
            torch.save({
                "frames": torch.from_numpy(fr_val).float(),
                "confidence": torch.from_numpy(conf_val).float(),
            }, os.path.join(fold_dir, "val_freq.pt"))

            # 手工特征
            torch.save(torch.from_numpy(aux_all[train_idx]).float(), os.path.join(fold_dir, "train_features.pt"))
            torch.save(torch.from_numpy(aux_all[val_idx]).float(), os.path.join(fold_dir, "val_features.pt"))

            # 目标序列 + 开关掩码（合并到同一文件，字典形式）
            torch.save({
                "seq": torch.from_numpy(Y_seq_all[train_idx]).float(),
                "status": torch.from_numpy(S_seq_all[train_idx]).float(),
            }, os.path.join(fold_dir, "train_targets_seq.pt"))
            torch.save({
                "seq": torch.from_numpy(Y_seq_all[val_idx]).float(),
                "status": torch.from_numpy(S_seq_all[val_idx]).float(),
            }, os.path.join(fold_dir, "val_targets_seq.pt"))

            # 不再另存开关掩码与缺失掩码；已合并至 targets_seq 与 raw

            # 每折标签与元数据
            labels_train = labels_all[train_idx].astype(np.float32)
            labels_val = labels_all[val_idx].astype(np.float32)
            meta_train = [{
                "datetime_iso": str(start_iso_all[i]),
                "timestamp": str(start_iso_all[i]),
                "segment_id": int(windows_meta.iloc[i]["segment_id"]),
                "house": int(windows_meta.iloc[i]["house"]) if "house" in windows_meta.columns else 0,
            } for i in train_idx]
            meta_val = [{
                "datetime_iso": str(start_iso_all[i]),
                "timestamp": str(start_iso_all[i]),
                "segment_id": int(windows_meta.iloc[i]["segment_id"]),
                "house": int(windows_meta.iloc[i]["house"]) if "house" in windows_meta.columns else 0,
            } for i in val_idx]
            torch.save({"labels": torch.from_numpy(labels_train).float(), "label_metadata": meta_train}, os.path.join(fold_dir, "train_labels.pt"))
            torch.save({"labels": torch.from_numpy(labels_val).float(), "label_metadata": meta_val}, os.path.join(fold_dir, "val_labels.pt"))

            # 名称文件（每折自足）
            with open(os.path.join(fold_dir, "feature_names.json"), "w", encoding="utf-8") as f:
                json.dump(aux_names, f, ensure_ascii=False, indent=2)
            with open(os.path.join(fold_dir, "raw_channel_names.json"), "w", encoding="utf-8") as f:
                json.dump(["P_W", "dP_W", "missing_mask"], f, ensure_ascii=False, indent=2)
            # 设备顺序文件（列表形式，与 device_name_to_id.json 相互补充）
            try:
                with open(os.path.join(fold_dir, "device_names.json"), "w", encoding="utf-8") as f:
                    json.dump(device_names, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        # 总结
        self.summary_ = {
            "dataset": self.ru.dataset,
            "n_windows": int(raw_all.shape[0]),
            "window_length": int(raw_all.shape[1]),
            "channels": ["P_W"],
            "n_devices": len(mask_app),
            "freq_repr": {"type": "stft", "n_fft": self.ru.stft_n_fft, "hop": self.ru.stft_hop},
            "aux_features_count": int(aux_all.shape[1]),
            "output_dir": os.path.join(self.output_dir, str(self.ru.dataset).lower()),
            "folds": [f"fold_{f.fold_id}" for f in folds],
        }

        # 结果文件
        try:
            with open(os.path.join(self.output_dir, "pipeline_results.json"), "w", encoding="utf-8") as f:
                json.dump({"status": "success", "summary": self.summary_}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        return self.summary_.copy()

    @staticmethod
    def _assign_segment_ids(start_ts: np.ndarray, stride_seconds: int, houses: Optional[np.ndarray] = None) -> np.ndarray:
        seg = np.zeros_like(start_ts, dtype=np.int64)
        if start_ts.size == 0:
            return seg
        sid = 0
        seg[0] = sid
        for i in range(1, len(start_ts)):
            same_house = True
            if houses is not None and len(houses) == len(start_ts):
                try:
                    same_house = bool(houses[i] == houses[i - 1])
                except Exception:
                    same_house = True
            if same_house and int(start_ts[i] - start_ts[i - 1]) == int(stride_seconds):
                seg[i] = sid
            else:
                sid += 1
                seg[i] = sid
        return seg

    @staticmethod
    def _build_segments_meta_from_windows(windows_meta: pd.DataFrame) -> pd.DataFrame:
        """按 segment_id 聚合为段元数据，包含 start_ts/end_ts/segment_id"""
        df = windows_meta.copy()
        g = df.groupby("segment_id")
        out = pd.DataFrame({
            "segment_id": g["segment_id"].first(),
            "start_ts": g["start_ts"].min(),
            "end_ts": g["end_ts"].max(),
        }).reset_index(drop=True)
        return out
