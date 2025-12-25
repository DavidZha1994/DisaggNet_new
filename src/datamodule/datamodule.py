"""
工业级数据模块

提供 NILMDataModule 与 NILMDataset，用于从 Data/prepared 加载折数据，
支持：
- 训练/验证/测试 DataLoader
- 类权重计算与 WeightedRandomSampler（仅训练集）
- 可选的时间过滤（与 Walk-Forward 输出一致）
- 防泄露检查（时间间隔 purge gap 与 segment 隔离的提示）
- 批次包含 mains、targets、timestamps、aux_features（若可用）
"""

import os
import sys
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
import platform


class NILMDataset(Dataset):
    """简单的数据集封装，返回标准化特征与标签。

    期望 Data/prepared/fold_{k}/ 下存在：
    - train_features.pt / val_features.pt
    - train_indices.pt / val_indices.pt
    - train_raw.pt / val_raw.pt（原始窗口，供时域Transformer）
    """

    def __init__(
        self,
        features: Any,
        indices: np.ndarray,
        labels_data: Dict[str, Any],
        raw_windows: Optional[Any] = None,
        freq_windows: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
        target_seq: Optional[Any] = None,
        status_seq: Optional[Any] = None,
    ):
        # 统一为 torch.Tensor，避免每次 __getitem__ 转换
        if isinstance(features, torch.Tensor):
            self.features = features.float()
        else:
            self.features = torch.from_numpy(np.asarray(features).astype(np.float32))
        self.indices = indices.astype(np.int64)
        self.labels_data = labels_data
        self.raw_windows = None if raw_windows is None else (
            raw_windows.float() if isinstance(raw_windows, torch.Tensor) else torch.from_numpy(np.asarray(raw_windows).astype(np.float32))
        )
        self.freq_windows = None if freq_windows is None else (
            freq_windows.float() if isinstance(freq_windows, torch.Tensor) else torch.from_numpy(np.asarray(freq_windows).astype(np.float32))
        )
        self.feature_names = feature_names or []
        self.target_seq = None if target_seq is None else (
            target_seq.float() if isinstance(target_seq, torch.Tensor) else torch.from_numpy(np.asarray(target_seq).astype(np.float32))
        )
        self.status_seq = None if status_seq is None else (
            status_seq.float() if isinstance(status_seq, torch.Tensor) else torch.from_numpy(np.asarray(status_seq).astype(np.float32))
        )

        labels = labels_data.get("labels")
        if labels is None:
            raise ValueError("labels_data 缺少 'labels' 字段")
        labels_arr = np.asarray(labels)
        if labels_arr.ndim == 1:
            labels_arr = labels_arr.reshape(-1, 1)
        else:
            labels_arr = labels_arr.astype(np.float32)
        n_feat = int(self.features.size(0))
        if labels_arr.shape[0] != n_feat:
            if labels_arr.shape[0] >= int(self.indices.max()) + 1 and len(self.indices) == n_feat:
                targets_np = labels_arr[self.indices]
            else:
                raise ValueError(f"标签数量与特征数量不一致: labels={labels_arr.shape[0]}, features={n_feat}")
        else:
            targets_np = labels_arr
        self.targets = torch.from_numpy(targets_np.astype(np.float32))

        meta_list = labels_data.get("label_metadata", [])
        if meta_list:
            if len(meta_list) == n_feat:
                indices_for_meta = range(n_feat)
            else:
                indices_for_meta = self.indices
            self.timestamps = np.array([
                (meta_list[i].get("timestamp") if isinstance(meta_list[i], dict) else None)
                for i in indices_for_meta
            ])
            self.segment_ids = np.array([
                (meta_list[i].get("segment_id") if isinstance(meta_list[i], dict) else None)
                for i in indices_for_meta
            ])
        else:
            self.timestamps = None
            self.segment_ids = None

    def __len__(self) -> int:
        return int(self.features.size(0))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        mains = self.features[idx]
        target = self.targets[idx]
        item: Dict[str, Any] = {
            "mains": mains,
            "targets": target,
        }
        if self.timestamps is not None and self.timestamps[idx] is not None:
            item["timestamps"] = self.timestamps[idx]
        if self.raw_windows is not None and self.raw_windows.size()[0] > 0:
            item["aux_features"] = self.raw_windows[idx]
        if self.freq_windows is not None and self.freq_windows.size()[0] > 0:
            item["freq_features"] = self.freq_windows[idx]
        if self.target_seq is not None and self.target_seq.numel() > 0:
            item["target_seq"] = self.target_seq[idx]
        if self.status_seq is not None and self.status_seq.numel() > 0:
            item["status_seq"] = self.status_seq[idx]
        # 新增：标记事件窗口（若 label_metadata 提供）
        try:
            abs_idx = int(self.indices[idx])
            meta_list = self.labels_data.get("label_metadata", []) if isinstance(self.labels_data, dict) else []
            if meta_list and abs_idx < len(meta_list):
                m = meta_list[abs_idx]
                flag = bool(m.get("has_events", False)) if isinstance(m, dict) else bool(getattr(m, "has_events", False))
                item["has_events"] = flag
        except Exception:
            pass
        return item


class NILMDataModule(pl.LightningDataModule):
    """数据模块，负责从 Data/prepared 构建 DataLoader。

    参数：
    - config: OmegaConf/DictConfig 或 dict，需包含 data.batch_size 等。
    - data_root: 默认 'Data/prepared'
    - fold_id: 可选，指定使用的折。若未指定，默认使用 0。
    - use_weighted_sampler: 是否启用训练集的加权采样（默认根据配置自动）。
    """

    def __init__(
        self,
        config: Any,
        data_root: Optional[str] = None,
        fold_id: Optional[int] = None,
        use_weighted_sampler: Optional[bool] = None,
    ):
        super().__init__()
        self.config = config
        # 允许在仅调用 _collate_and_map 的单元测试场景下缺失 prepared_dir
        if data_root is not None:
            self.data_root = Path(data_root)
        else:
            cfg_paths = getattr(config, 'paths', None)
            cfg_prepared = getattr(cfg_paths, 'prepared_dir', None) if cfg_paths is not None else None
            dataset_name = str(getattr(config, 'dataset', '') or '').strip().lower()
            if not cfg_prepared:
                # 回退占位目录，延迟检查到 setup 阶段
                base_prepared = Path('Data/prepared')
                # 动态附加 dataset 子目录（小写）
                if dataset_name:
                    self.data_root = base_prepared / dataset_name
                else:
                    self.data_root = base_prepared
                self._allow_missing_root = True
            else:
                base_prepared = Path(cfg_prepared)
                # 若 prepared_dir 未包含数据集子目录，则附加小写数据集名称
                if dataset_name:
                    parts_lower = [p.lower() for p in base_prepared.parts]
                    if dataset_name not in parts_lower:
                        self.data_root = base_prepared / dataset_name
                    else:
                        self.data_root = base_prepared
                else:
                    self.data_root = base_prepared
                self._allow_missing_root = False
        self.fold_id = fold_id if fold_id is not None else 0
        # 缓存特征与原始通道名称，供批处理阶段使用
        self.feature_names: List[str] = []
        self.raw_channel_names: List[str] = []
        # 新增：设备数量（用于占位标签维度）
        self.n_devices: int = 0
        # 新增：采样间隔（秒），用于展开时间戳序列；固定为 5.0s
        self.resample_seconds: float = 5.0

        data_cfg = getattr(config, "data", None) or {}
        self.batch_size = int(getattr(data_cfg, "batch_size", 256))
        raw_num_workers = getattr(data_cfg, "num_workers", None)
        try:
            raw_num_workers = int(raw_num_workers) if raw_num_workers is not None else -1
        except Exception:
            raw_num_workers = -1
        try:
            sys_name = platform.system()
        except Exception:
            sys_name = ""
        if raw_num_workers is not None and raw_num_workers > 0:
            self.num_workers = raw_num_workers
        else:
            try:
                cpu_count = os.cpu_count() or 4
            except Exception:
                cpu_count = 4
            if sys_name in ("Windows", "Linux"):
                base = max(1, cpu_count // 2)
                self.num_workers = min(8, base)
            else:
                self.num_workers = 0
        self.pin_memory = bool(getattr(data_cfg, "pin_memory", True))
        self.normalize_total_power_to_watts = bool(getattr(data_cfg, "normalize_total_power_to_watts", False))
        try:
            is_mac = (sys.platform == 'darwin')
        except Exception:
            is_mac = False
        mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        if is_mac and mps_available:
            self.num_workers = 0
            self.pin_memory = False
            print("[信息] 检测到 macOS + MPS，已将 num_workers=0 且禁用 pin_memory 以确保稳定运行。")
        if os.environ.get("PYTEST_CURRENT_TEST"):
            self.num_workers = 0

        # 是否使用加权采样
        if use_weighted_sampler is None:
            imb_cfg = getattr(config, "imbalance_handling", None) or {}
            strat = getattr(imb_cfg, "sampling_strategy", "none")
            self.sampling_strategy = strat
            self.use_weighted_sampler = strat in ("mixed", "oversample", "hybrid", "weighted")
            self.event_boost = float(getattr(imb_cfg, "event_boost", 1.5))
            self.pos_count_inverse_enable = bool(getattr(imb_cfg, "pos_count_inverse_enable", True))
            self.early_oversample_epochs = int(getattr(imb_cfg, "early_oversample_epochs", 0))
        else:
            self.use_weighted_sampler = bool(use_weighted_sampler)
            self.event_boost = 1.5
            self.pos_count_inverse_enable = True
            self.sampling_strategy = "mixed"
            self.early_oversample_epochs = 0

        # 占位符
        self.train_ds: Optional[NILMDataset] = None
        self.val_ds: Optional[NILMDataset] = None
        self.test_ds: Optional[NILMDataset] = None
        self.class_weights: Optional[torch.Tensor] = None
        # 统计信息（用于自动配置）
        self.pos_weight_vec: Optional[torch.Tensor] = None
        self.prior_p_vec: Optional[torch.Tensor] = None
        # 新增：每设备功率尺度（例如P95），用于相对刻度训练
        self.power_scale_vec: Optional[torch.Tensor] = None
        # 新增：有效性与事件率缓存（用于采样加权）
        self.train_valid_ratio: Optional[np.ndarray] = None
        self.val_valid_ratio: Optional[np.ndarray] = None
        self.train_event_rate: Optional[np.ndarray] = None
        self.val_event_rate: Optional[np.ndarray] = None
        # 新增：基于状态序列的每设备激活先验与稀有增强
        self.active_prior_vec: Optional[torch.Tensor] = None
        self.rarity_boost_vec: Optional[torch.Tensor] = None
        # 采样加权参数（可从配置覆盖）
        imb_cfg = getattr(config, "imbalance_handling", None) or {}
        self.validity_weight_enable: bool = bool(getattr(imb_cfg, "validity_weight_enable", True))
        self.validity_exponent: float = float(getattr(imb_cfg, "validity_exponent", 2.0))
        self.validity_floor: float = float(getattr(imb_cfg, "validity_floor", 0.05))
        self.invalid_boost: float = float(getattr(imb_cfg, "invalid_boost", 0.15))

    def _load_device_names_from_mapping(self) -> Optional[List[str]]:
        try:
            base = getattr(self, "data_root", None)
            if base is None:
                return None
            base = Path(base)
            candidates = [base, base.parent, base.parent.parent]
            mapping = None
            for d in candidates:
                if d is None:
                    continue
                fp = Path(d) / "device_name_to_id.json"
                if fp.exists():
                    with open(fp, "r", encoding="utf-8") as f:
                        mapping = json.load(f)
                    break
            if not mapping:
                return None
            def _to_int(x):
                try:
                    return int(x)
                except Exception:
                    return None
            sample_key = next(iter(mapping.keys()))
            sample_val = mapping[sample_key]
            key_is_int_like = _to_int(sample_key) is not None
            val_is_int_like = _to_int(sample_val) is not None
            if key_is_int_like and not val_is_int_like:
                pairs = sorted(((int(k), v) for k, v in mapping.items()), key=lambda kv: kv[0])
                names = [name for _, name in pairs]
            elif not key_is_int_like and val_is_int_like:
                pairs = sorted(((v, k) for k, v in mapping.items()), key=lambda kv: kv[0])
                names = [name for _, name in pairs]
            else:
                names = sorted(list(mapping.keys()))
            return [str(n) for n in names]
        except Exception as e:
            try:
                print(f"[警告] 从 device_name_to_id.json 读取设备名称失败：{e}")
            except Exception:
                pass
            return None

    def setup(self, stage: Optional[str] = None):
        fold_dir = self.data_root / f"fold_{self.fold_id}"
        if not fold_dir.exists():
            # 在仅用于批映射的测试场景下跳过严格检查
            if getattr(self, '_allow_missing_root', False):
                return
            raise FileNotFoundError(f"未找到折目录: {fold_dir}")
        # 新增：从 Data/prepared 识别设备数量
        # 简化：不再依赖设备映射文件，后续从目标序列维度自动推断

        # 加载特征与索引
        # 加载特征与索引（仅 .pt 张量文件）
        tf_pt = fold_dir / "train_features.pt"
        vf_pt = fold_dir / "val_features.pt"
        if not (tf_pt.exists() and vf_pt.exists()):
            raise FileNotFoundError(f"缺少 .pt 特征文件: {tf_pt} 或 {vf_pt}")
        train_features = torch.load(tf_pt)
        val_features = torch.load(vf_pt)
        # 索引：优先读取折内提供的索引；否则使用本地顺序索引
        tr_idx_pt = fold_dir / "train_indices.pt"
        va_idx_pt = fold_dir / "val_indices.pt"
        if tr_idx_pt.exists() and va_idx_pt.exists():
            train_indices = torch.load(tr_idx_pt)
            val_indices = torch.load(va_idx_pt)
            train_indices = np.asarray(train_indices).astype(np.int64)
            val_indices = np.asarray(val_indices).astype(np.int64)
        else:
            # 回退：使用本地顺序索引（非绝对索引）
            train_indices = np.arange(train_features.size(0), dtype=np.int64)
            val_indices = np.arange(val_features.size(0), dtype=np.int64)

        # 原始窗口（仅 .pt）
        train_raw = None
        val_raw = None
        tr_pt = fold_dir / "train_raw.pt"
        vr_pt = fold_dir / "val_raw.pt"
        if not (tr_pt.exists() and vr_pt.exists()):
            raise FileNotFoundError(f"缺少 .pt 原始窗口: {tr_pt} 或 {vr_pt}")
        train_raw = torch.load(tr_pt)
        val_raw = torch.load(vr_pt)

        # 设备功率尺度估计（max 与 P95）
        try:
            ts_train_path = fold_dir / "train_targets_seq.pt"
            ts_train_obj = torch.load(ts_train_path) if ts_train_path.exists() else None
            if ts_train_obj is not None:
                ts_train = ts_train_obj.get("seq") if isinstance(ts_train_obj, dict) else ts_train_obj
                vals = torch.nan_to_num(ts_train.float(), nan=0.0).contiguous()
                K = int(vals.size(-1))
                max_per_dev = torch.amax(vals, dim=(0, 1))
                max_per_dev = torch.clamp(max_per_dev, min=1.0).float()
                flat = vals.view(-1, K)
                p95_list = []
                for k in range(K):
                    vk = flat[:, k]
                    nz = vk[vk > 1.0]
                    base = nz if nz.numel() > 0 else vk
                    q = torch.quantile(base, 0.95)
                    p95_list.append(q)
                p95_vec = torch.stack(p95_list).clamp_min(1.0).float()
                self.max_power_vec = max_per_dev.detach()
                self.power_scale_vec = p95_vec.detach()
                self.n_devices = K
        except Exception:
            self.max_power_vec = None
            self.power_scale_vec = None

        # 频域特征（可选，仅 .pt；缺失则置为 None）
        train_freq = None
        val_freq = None
        tfreq_pt = fold_dir / "train_freq.pt"
        vfreq_pt = fold_dir / "val_freq.pt"
        if tfreq_pt.exists() and vfreq_pt.exists():
            tf_obj = torch.load(tfreq_pt)
            vf_obj = torch.load(vfreq_pt)
            # 兼容：频域文件可为张量或字典 {frames, confidence}
            try:
                if isinstance(tf_obj, dict):
                    self.train_freq_confidence = tf_obj.get("confidence", None)
                    train_freq = tf_obj.get("frames", None)
                else:
                    train_freq = tf_obj
                if isinstance(vf_obj, dict):
                    self.val_freq_confidence = vf_obj.get("confidence", None)
                    val_freq = vf_obj.get("frames", None)
                else:
                    val_freq = vf_obj
            except Exception:
                train_freq = tf_obj
                val_freq = vf_obj

        # 特征名称（可选）
        feature_names = []
        feat_name_path = fold_dir / "feature_names.json"
        if feat_name_path.exists():
            with open(feat_name_path, "r") as f:
                feature_names = json.load(f)
        # 缓存至实例
        self.feature_names = feature_names or []

        # 原始通道名称（可选，用于从原始窗口中提取总有功功率）
        raw_channel_path = fold_dir / "raw_channel_names.json"
        if raw_channel_path.exists():
            try:
                with open(raw_channel_path, "r") as f:
                    self.raw_channel_names = json.load(f) or []
            except Exception:
                self.raw_channel_names = []

        # 加载顶层 labels.pkl（优先，包含绝对索引下的标签与元数据）
        # 每折标签：优先加载 train_labels.pt / val_labels.pt，否则基于 device_masks 与 target_seq 构造
        train_labels_path = fold_dir / "train_labels.pt"
        val_labels_path = fold_dir / "val_labels.pt"
        if train_labels_path.exists() and val_labels_path.exists():
            train_labels_data = torch.load(train_labels_path)
            val_labels_data = torch.load(val_labels_path)
        else:
            # 构造标签：
            # - window_states 来自 device_masks.onoff 的时间众数（>=0.5）
            # - window_power 来自 target_seq 的时间均值
            # - 元数据 datetime_iso 从原始时间列回推困难，此处留空
            train_labels_data = {"labels": np.zeros((train_features.size(0), 1), dtype=np.float32), "label_metadata": []}
            val_labels_data = {"labels": np.zeros((val_features.size(0), 1), dtype=np.float32), "label_metadata": []}
            try:
                dm_train = torch.load(fold_dir / "train_device_masks.pt")
                dm_val = torch.load(fold_dir / "val_device_masks.pt")
                ts_train = torch.load(fold_dir / "train_targets_seq.pt")
                ts_val = torch.load(fold_dir / "val_targets_seq.pt")
                on_train = dm_train.get("onoff") if isinstance(dm_train, dict) else None
                on_val = dm_val.get("onoff") if isinstance(dm_val, dict) else None
                if isinstance(on_train, torch.Tensor):
                    ws_train = (on_train.float().mean(dim=1) >= 0.5).float()  # (N,K)
                else:
                    ws_train = torch.zeros(ts_train.size(0), ts_train.size(-1))
                if isinstance(on_val, torch.Tensor):
                    ws_val = (on_val.float().mean(dim=1) >= 0.5).float()
                else:
                    ws_val = torch.zeros(ts_val.size(0), ts_val.size(-1))
                wp_train = torch.nanmean(ts_train.float(), dim=1)
                wp_val = torch.nanmean(ts_val.float(), dim=1)
                labels_train = torch.cat([ws_train, torch.nan_to_num(wp_train, nan=0.0)], dim=1)
                labels_val = torch.cat([ws_val, torch.nan_to_num(wp_val, nan=0.0)], dim=1)
                train_labels_data = {"labels": labels_train.detach().cpu().numpy(), "label_metadata": []}
                val_labels_data = {"labels": labels_val.detach().cpu().numpy(), "label_metadata": []}
            except Exception as e:
                print(f"[警告] 构造每折标签失败，使用占位：{e}")

        # 采样间隔固定为 5 秒（HIPE 数据集标准）
        self.resample_seconds = 5.0

        # 每折独立标签数据
        def _ensure_labels(ld):
            arr = ld.get("labels")
            arr = np.asarray(arr)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1).astype(np.float32)
            else:
                arr = arr.astype(np.float32)
            md = ld.get("label_metadata", [])
            return {"labels": arr, "label_metadata": md}
        train_labels_obj = _ensure_labels(train_labels_data)
        val_labels_obj = _ensure_labels(val_labels_data)

        # 构建数据集（移除 .npy 读取逻辑，见下方 .pt 版本）
        # 读取并传递序列级回归目标（若存在，优先 .pt）
        train_seq_arr = None
        val_seq_arr = None
        try:
            seq_train_pt = fold_dir / "train_targets_seq.pt"
            seq_val_pt = fold_dir / "val_targets_seq.pt"
            if seq_train_pt.exists():
                ts_tr_obj = torch.load(seq_train_pt)
                train_seq_arr = ts_tr_obj.get("seq") if isinstance(ts_tr_obj, dict) else ts_tr_obj
                # 设备开关掩码（逐时间步）
                self.train_status_seq = ts_tr_obj.get("status") if isinstance(ts_tr_obj, dict) else None
            if seq_val_pt.exists():
                ts_va_obj = torch.load(seq_val_pt)
                val_seq_arr = ts_va_obj.get("seq") if isinstance(ts_va_obj, dict) else ts_va_obj
                self.val_status_seq = ts_va_obj.get("status") if isinstance(ts_va_obj, dict) else None
            # 推断设备数
            try:
                if isinstance(train_seq_arr, torch.Tensor) and train_seq_arr.numel() > 0:
                    self.n_devices = int(train_seq_arr.size(-1))
                elif isinstance(val_seq_arr, torch.Tensor) and val_seq_arr.numel() > 0:
                    self.n_devices = int(val_seq_arr.size(-1))
                else:
                    self.n_devices = 0
            except Exception:
                self.n_devices = 0
        except Exception as e:
            print(f"[警告] 加载 targets_seq 失败：{e}")

        self.train_ds = NILMDataset(
            train_features, train_indices, train_labels_obj,
            raw_windows=train_raw, freq_windows=train_freq,
            feature_names=feature_names,
            target_seq=train_seq_arr,
            status_seq=getattr(self, 'train_status_seq', None),
        )
        self.val_ds = NILMDataset(
            val_features, val_indices, val_labels_obj,
            raw_windows=val_raw, freq_windows=val_freq,
            feature_names=feature_names,
            target_seq=val_seq_arr,
            status_seq=getattr(self, 'val_status_seq', None),
        )
        mapping_names = self._load_device_names_from_mapping()
        cfg_names = getattr(getattr(self.config, "data", None), "device_names", None)
        source = "default"
        if isinstance(mapping_names, list) and len(mapping_names) > 0:
            names = [str(x) for x in mapping_names]
            source = "mapping"
        elif isinstance(cfg_names, list) and len(cfg_names) > 0:
            names = [str(x) for x in cfg_names]
            source = "config"
        else:
            k = int(self.n_devices) if self.n_devices > 0 else 1
            names = [f"device_{i+1}" for i in range(k)]
            source = "fallback"
        k = int(self.n_devices) if self.n_devices > 0 else len(names)
        self.device_names = names[:k]
        self.device_name_to_id = {str(n): i for i, n in enumerate(self.device_names)}
        try:
            print(f"[Info] Device names from {source}: {self.device_names}")
        except Exception:
            pass

        # 预计算采样加权所需的有效比例与事件率
        try:
            def _compute_valid_ratio(raw_tensor: Optional[torch.Tensor], channel_names: List[str]) -> Optional[np.ndarray]:
                if not isinstance(raw_tensor, torch.Tensor) or raw_tensor.numel() == 0:
                    return None
                try:
                    miss_idx = channel_names.index("missing_mask") if channel_names else (raw_tensor.size(-1) - 1)
                except Exception:
                    miss_idx = raw_tensor.size(-1) - 1
                miss = raw_tensor.detach().cpu()[:, :, miss_idx].numpy().astype(np.float32)
                L = miss.shape[1] if miss.ndim == 2 else 1
                if L <= 0:
                    return None
                missing_ratio = np.clip(np.nanmean(miss, axis=1), 0.0, 1.0)
                valid_ratio = 1.0 - missing_ratio
                return valid_ratio.astype(np.float32)

            def _compute_event_rate(status_seq: Optional[torch.Tensor]) -> Optional[np.ndarray]:
                if not isinstance(status_seq, torch.Tensor) or status_seq.numel() == 0:
                    return None
                try:
                    s = status_seq.detach().cpu().float()  # (N,L,K)
                    if s.dim() != 3 or s.size(1) < 2:
                        return None
                    diff = torch.abs(s[:, 1:, :] - s[:, :-1, :])  # transitions
                    trans = diff.sum(dim=(1, 2)).numpy().astype(np.float32)  # (N,)
                    denom = float((s.size(1) - 1) * max(s.size(2), 1))
                    rate = (trans / denom) if denom > 0 else trans
                    rate = np.clip(rate, 0.0, 1.0)
                    return rate.astype(np.float32)
                except Exception:
                    return None

            self.train_valid_ratio = _compute_valid_ratio(train_raw, self.raw_channel_names)
            self.val_valid_ratio = _compute_valid_ratio(val_raw, self.raw_channel_names)
            self.train_event_rate = _compute_event_rate(getattr(self, 'train_status_seq', None))
            self.val_event_rate = _compute_event_rate(getattr(self, 'val_status_seq', None))
        except Exception as e:
            print(f"[警告] 预计算有效比例/事件率失败：{e}")

        # 预计算每设备功率尺度（P95），用于相对刻度训练（在设备激活时序上统计，避免零占比压低P95）
        try:
            if isinstance(train_seq_arr, torch.Tensor) and train_seq_arr.numel() > 0:
                # train_seq_arr: (N, L, K)
                N, L, K = int(train_seq_arr.size(0)), int(train_seq_arr.size(1)), int(train_seq_arr.size(2))
                ts = train_seq_arr.detach().cpu().reshape(-1, K).numpy()  # [N*L, K]
                ts = np.where(np.isfinite(ts), ts, np.nan)
                ts = np.clip(ts, a_min=0.0, a_max=None)
                # 若存在逐时间步开关掩码，则在激活时统计P95
                st = getattr(self, 'train_status_seq', None)
                scale = np.zeros((K,), dtype=np.float32)
                with np.errstate(all='ignore'):
                    if isinstance(st, torch.Tensor) and st.numel() == (N * L * K):
                        st_flat = (st.detach().cpu().reshape(-1, K).numpy() > 0.5).astype(np.uint8)
                        for k_ in range(K):
                            sel = ts[:, k_][st_flat[:, k_] == 1]
                            if sel.size >= 50:
                                scale[k_] = float(np.nanpercentile(sel, 95))
                            else:
                                # 有效点过少，回退为无条件P95
                                scale[k_] = float(np.nanpercentile(ts[:, k_], 95))
                    else:
                        scale = np.nanpercentile(ts, 95, axis=0)
                # 处理NaN或非正数：回退为1.0
                if not np.all(np.isfinite(scale)):
                    scale = np.where(np.isfinite(scale), scale, 1.0)
                # 防止极小值导致归一化爆炸；若P95过小（<1.0W），提升到1.0回退值
                scale = np.maximum(scale, 1.0)
                self.power_scale_vec = torch.tensor(scale, dtype=torch.float32)
            else:
                # 回退：使用单位尺度（1.0）
                k = int(self.n_devices) if self.n_devices > 0 else 1
                self.power_scale_vec = torch.ones(k, dtype=torch.float32)
        except Exception as e:
            print(f"[警告] 计算功率尺度失败，将使用单位尺度：{e}")
            k = int(self.n_devices) if self.n_devices > 0 else 1
            self.power_scale_vec = torch.ones(k, dtype=torch.float32)
        # 基于状态序列的每设备激活先验与自适应稀疏增强（自动化，无需配置）
        try:
            st = getattr(self, 'train_status_seq', None)
            if isinstance(st, torch.Tensor) and st.dim() == 3 and st.size(-1) > 0:
                p = (st.float() > 0.5).float().mean(dim=(0, 1))  # (K,)
                self.active_prior_vec = p.detach().clone().float()
                # 自适应：按设备激活先验的逆频率加权，并做归一化与限幅
                v = torch.clamp(self.active_prior_vec, min=1e-4).float()
                inv = 1.0 / v
                inv_norm = inv / torch.mean(inv).clamp(min=1e-8)
                rb = torch.clamp(inv_norm, min=1.0, max=5.0)
                self.rarity_boost_vec = rb.detach().clone().float()
            else:
                self.active_prior_vec = None
                self.rarity_boost_vec = None
        except Exception:
            self.active_prior_vec = None
            self.rarity_boost_vec = None

        # 可选：测试集从验证集复制（若无单独测试集，保持 None）
        # 不再依赖 cv_splits.pkl 构造 test 数据集
        self.test_ds = None

        # 计算类权重（仅用于二分类或多标签的每列二分类）
        self.class_weights = self._compute_class_weights(self.train_ds.targets)

        # 统计每设备阳性比例与 pos_weight（仅对二值列）
        self._compute_label_stats()

        # 简单的泄露检查：时间戳间隔与索引交集
        self._validate_no_leakage()

    def _build_sampler(self, dataset: NILMDataset) -> Optional[WeightedRandomSampler]:
        if not self.use_weighted_sampler:
            return None
        y = np.asarray(dataset.targets)
        if y.ndim == 2:
            # 仅基于二值列计算阳性数量，避免回归列影响采样权重
            binary_mask = []
            for j in range(y.shape[1]):
                col = y[:, j]
                vals = np.unique(col[~np.isnan(col)])
                is_binary = len(vals) > 0 and np.all(np.isin(vals, [0.0, 1.0]))
                binary_mask.append(is_binary)
            binary_mask = np.array(binary_mask, dtype=bool)
            if binary_mask.any():
                y_bin = y[:, binary_mask]
                pos_counts = (y_bin == 1).sum(axis=1)
            else:
                pos_counts = np.zeros(y.shape[0], dtype=np.float32)
            # 可选：禁用按阳性数量的逆权重，避免多事件窗口被过度降权
            if getattr(self, 'pos_count_inverse_enable', True):
                weights = 1.0 / (pos_counts + 1.0)
            else:
                weights = np.ones(y.shape[0], dtype=np.float32)
        else:
            # 二分类：正类权重大，负类权重小
            weights = np.where(y == 1, 1.0, 0.25).astype(np.float32)
        # 新增：按有效比例与事件率加权
        try:
            N = len(dataset)
            # 有效比例（来自 raw 的 missing_mask 通道）
            if self.validity_weight_enable and isinstance(self.train_valid_ratio, np.ndarray) and self.train_valid_ratio.size == N:
                vr = self.train_valid_ratio.astype(np.float32)
                vr = np.clip(vr, 0.0, 1.0)
                # floor + (1-floor) * vr^exp
                w_valid = self.validity_floor + (1.0 - self.validity_floor) * np.power(vr, self.validity_exponent)
                weights = weights * w_valid.astype(np.float32)
                # 低有效比例直接丢弃（权重置零）
                validity_min_ratio = float(getattr(getattr(self.config, "imbalance_handling", {}), "validity_min_ratio", 0.85))
                mask_keep = (vr >= validity_min_ratio).astype(np.float32)
                weights = weights * mask_keep
            # 事件率（来自 status 序列）
            if isinstance(self.train_event_rate, np.ndarray) and self.train_event_rate.size == N:
                er = self.train_event_rate.astype(np.float32)
                er = np.clip(er, 0.0, 1.0)
                boost = float(self.event_boost)
                if getattr(self, 'sampling_strategy', None) == 'oversample':
                    boost = max(boost, 3.0)
                    self.pos_count_inverse_enable = False
                w_event = 1.0 + boost * er
                weights = weights * w_event.astype(np.float32)
        except Exception:
            pass
        weights = torch.tensor(weights.astype(np.float32))
        # 稀有设备窗口增强（自适应比例控制）
        try:
            N = len(dataset)
            st = getattr(self, 'train_status_seq', None)
            rb = getattr(self, 'rarity_boost_vec', None)
            if isinstance(st, torch.Tensor) and isinstance(rb, torch.Tensor):
                if int(st.size(0)) == N and st.dim() == 3 and int(st.size(2)) == int(rb.numel()):
                    a = (st.detach().cpu().float() > 0.5).numpy().astype(np.uint8)  # (N,L,K)
                    any_active = a.max(axis=1)  # (N,K)
                    rbn = rb.detach().cpu().numpy().astype(np.float32).reshape(1, -1)  # (1,K)
                    # 使用加权求和以增强含多个稀疏设备的窗口
                    sb_sum = (any_active.astype(np.float32) * rbn).sum(axis=1)  # (N,)
                    sb_sum = np.clip(sb_sum, 0.0, None).astype(np.float32)
                    # 基础增强：1 + 加权稀疏分数
                    base_boost = 1.0 + sb_sum
                    # 目标比例控制：确保含稀疏设备的窗口在采样中占有至少 target_ratio 的份额
                    rare_mask = (sb_sum > 0.0)
                    cur_ratio = float(np.mean(rare_mask)) if rare_mask.size > 0 else 0.0
                    target_ratio = 0.30
                    ratio_boost = 1.0
                    if cur_ratio > 0.0 and cur_ratio < target_ratio:
                        ratio_boost = min(12.0, max(2.0, target_ratio / cur_ratio))
                    scaled_boost = base_boost * (ratio_boost if cur_ratio < target_ratio else 1.0)
                    weights = weights * torch.tensor(scaled_boost, dtype=weights.dtype)
        except Exception:
            pass
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    def set_sampling_strategy(self, strategy: str, event_boost: Optional[float] = None, pos_count_inverse_enable: Optional[bool] = None) -> None:
        self.sampling_strategy = str(strategy)
        self.use_weighted_sampler = self.sampling_strategy in ("mixed", "oversample", "hybrid", "weighted")
        if event_boost is not None:
            self.event_boost = float(event_boost)
        if pos_count_inverse_enable is not None:
            self.pos_count_inverse_enable = bool(pos_count_inverse_enable)

    def _compute_label_stats(self) -> None:
        """统计二值状态列的阳性比例与 pos_weight，供训练前自动写入配置。"""
        y = np.asarray(self.train_ds.targets)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        binary_cols = []
        pos_list = []
        neg_list = []
        for j in range(y.shape[1]):
            col = y[:, j]
            if self._is_binary_column(col):
                binary_cols.append(j)
                pos = int((col == 1).sum())
                neg = int((col == 0).sum())
                pos_list.append(pos)
                neg_list.append(neg)
        if not binary_cols:
            self.pos_weight_vec = None
            self.prior_p_vec = None
            return
        pos_arr = np.array(pos_list, dtype=np.float32)
        neg_arr = np.array(neg_list, dtype=np.float32)
        total = pos_arr + neg_arr
        p = np.where(total > 0, pos_arr / total, 0.0)
        # 使用安全除法避免除零产生 RuntimeWarning
        pos_weight = np.divide(neg_arr, pos_arr, out=np.ones_like(pos_arr), where=pos_arr > 0)
        self.pos_weight_vec = torch.tensor(pos_weight, dtype=torch.float32)
        self.prior_p_vec = torch.tensor(p, dtype=torch.float32)

    def train_dataloader(self) -> DataLoader:
        sampler = self._build_sampler(self.train_ds)
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_and_map,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_and_map,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_ds is None:
            return None
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_and_map,
        )

    def predict_dataloader(self) -> Optional[DataLoader]:
        """可选：供 Trainer.predict 使用。默认复用验证集。"""
        if self.val_ds is None:
            return None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_and_map,
        )

    def get_class_weights(self) -> Optional[torch.Tensor]:
        """供外部（如损失函数）使用的类权重。"""
        return self.class_weights

    def _compute_class_weights(self, targets: Any) -> Optional[torch.Tensor]:
        """基于训练集标签计算每列二分类的权重向量。
        返回按二值列顺序排列的 pos_weight 向量（neg/pos），若不存在二值列返回 None。
        """
        try:
            if isinstance(targets, torch.Tensor):
                y = targets.detach().cpu().numpy()
            else:
                y = np.asarray(targets)
        except Exception:
            try:
                y = np.asarray(targets)
            except Exception:
                return None
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        binary_cols = []
        pos_list = []
        neg_list = []
        for j in range(y.shape[1]):
            col = y[:, j]
            vals = np.unique(col[~np.isnan(col)])
            is_binary = len(vals) > 0 and np.all(np.isin(vals, [0.0, 1.0]))
            if is_binary:
                binary_cols.append(j)
                pos = int((col == 1).sum())
                neg = int((col == 0).sum())
                pos_list.append(pos)
                neg_list.append(neg)
        if not binary_cols:
            return None
        pos_arr = np.array(pos_list, dtype=np.float32)
        neg_arr = np.array(neg_list, dtype=np.float32)
        with np.errstate(divide='ignore', invalid='ignore'):
            pos_weight = np.where(pos_arr > 0, neg_arr / pos_arr, 1.0)
        cw = torch.tensor(pos_weight, dtype=torch.float32)
        return cw

    # -----------------------
    # 适配 & 统计
    # -----------------------
    def _is_binary_column(self, col: np.ndarray) -> bool:
        """判断列是否为二值标签（仅包含0/1）"""
        vals = np.unique(col[~np.isnan(col)])
        if len(vals) == 0:
            return False
        return np.all(np.isin(vals, [0, 1]))

    def _compute_label_stats(self) -> None:
        """统计二值状态列的阳性比例与 pos_weight，供训练前自动写入配置。"""
        y = np.asarray(self.train_ds.targets)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        binary_cols = []
        pos_list = []
        neg_list = []
        for j in range(y.shape[1]):
            col = y[:, j]
            if self._is_binary_column(col):
                binary_cols.append(j)
                pos = int((col == 1).sum())
                neg = int((col == 0).sum())
                pos_list.append(pos)
                neg_list.append(neg)
        if not binary_cols:
            self.pos_weight_vec = None
            self.prior_p_vec = None
            return
        pos_arr = np.array(pos_list, dtype=np.float32)
        neg_arr = np.array(neg_list, dtype=np.float32)
        total = pos_arr + neg_arr
        p = np.where(total > 0, pos_arr / total, 0.0)
        # 使用安全除法避免除零产生 RuntimeWarning
        pos_weight = np.divide(neg_arr, pos_arr, out=np.ones_like(pos_arr), where=pos_arr > 0)
        self.pos_weight_vec = torch.tensor(pos_weight, dtype=torch.float32)
        self.prior_p_vec = torch.tensor(p, dtype=torch.float32)

    def get_pos_weight(self) -> Optional[torch.Tensor]:
        return self.pos_weight_vec

    def get_prior_p(self) -> Optional[torch.Tensor]:
        return self.prior_p_vec
    def get_active_prior_from_status(self) -> Optional[torch.Tensor]:
        return self.active_prior_vec

    def _validate_no_leakage(self) -> None:
        """轻量防泄露检查：
        - 索引交集检查（train/val 不应复用同一绝对索引）
        - 时间间隔检查（val 起始时间需晚于 train 结束时间，且满足 purge_gap）
        - segment 隔离（如开启 segment_isolation，则 train/val 不能出现相同 segment_id）
        仅打印告警，不抛异常，避免影响训练流程。
        """
        try:
            if self.train_ds is None or self.val_ds is None:
                return

            # 读取 walk-forward 配置
            wf_cfg = None
            try:
                wf_cfg = getattr(self.config, 'walk_forward', None) or (self.config.get('walk_forward') if hasattr(self.config, 'get') else None)
            except Exception:
                wf_cfg = None
            # purge gap（秒）
            purge_gap_s = 0.0
            try:
                if wf_cfg is not None:
                    if hasattr(wf_cfg, 'purge_gap_seconds'):
                        purge_gap_s = float(getattr(wf_cfg, 'purge_gap_seconds'))
                    elif hasattr(wf_cfg, 'purge_gap_minutes'):
                        purge_gap_s = float(getattr(wf_cfg, 'purge_gap_minutes')) * 60.0
                    elif isinstance(wf_cfg, dict):
                        if 'purge_gap_seconds' in wf_cfg:
                            purge_gap_s = float(wf_cfg.get('purge_gap_seconds', 0.0))
                        elif 'purge_gap_minutes' in wf_cfg:
                            purge_gap_s = float(wf_cfg.get('purge_gap_minutes', 0.0)) * 60.0
            except Exception:
                purge_gap_s = 0.0

            # segment 隔离开关
            segment_isolation = False
            try:
                if wf_cfg is not None:
                    if hasattr(wf_cfg, 'segment_isolation'):
                        segment_isolation = bool(getattr(wf_cfg, 'segment_isolation'))
                    elif isinstance(wf_cfg, dict):
                        segment_isolation = bool(wf_cfg.get('segment_isolation', False))
            except Exception:
                segment_isolation = False

            # 1) segment 隔离检查（可选）
            try:
                tr_segs = getattr(self.train_ds, 'segment_ids', None)
                va_segs = getattr(self.val_ds, 'segment_ids', None)
                if tr_segs is not None and va_segs is not None:
                    tr_set = {s for s in tr_segs.tolist() if s is not None}
                    va_set = {s for s in va_segs.tolist() if s is not None}
                    overlap = tr_set.intersection(va_set)
                    if overlap:
                        if segment_isolation:
                            print(f"[警告] 泄露检查：已开启 segment_isolation，但 train/val 存在相同 segment_id：{list(overlap)[:5]} ...")
                        else:
                            print(f"[信息] 泄露检查：train/val 存在共享 segment_id（未开启隔离）：{list(overlap)[:5]} ...")
            except Exception:
                pass

            # 2) 时间间隔与 purge gap 检查
            try:
                tr_ts = getattr(self.train_ds, 'timestamps', None)
                va_ts = getattr(self.val_ds, 'timestamps', None)
                if tr_ts is not None and va_ts is not None and len(tr_ts) > 0 and len(va_ts) > 0:
                    def _to_sec(arr) -> np.ndarray:
                        secs: List[float] = []
                        for x in arr:
                            try:
                                if x is None:
                                    continue
                                # 数值型（假定为秒）
                                if isinstance(x, (int, float, np.integer, np.floating)):
                                    fx = float(x)
                                    if np.isfinite(fx):
                                        secs.append(fx)
                                    continue
                                # numpy datetime64
                                if isinstance(x, np.datetime64):
                                    try:
                                        v = x.astype('datetime64[s]').astype('int64')
                                        secs.append(float(v))
                                        continue
                                    except Exception:
                                        pass
                                # pandas/py datetime 带 timestamp()
                                if hasattr(x, 'timestamp'):
                                    try:
                                        secs.append(float(x.timestamp()))
                                        continue
                                    except Exception:
                                        pass
                                # 字符串可解析为 datetime64
                                try:
                                    dt64 = np.datetime64(str(x))
                                    v = dt64.astype('datetime64[s]').astype('int64')
                                    secs.append(float(v))
                                    continue
                                except Exception:
                                    pass
                            except Exception:
                                continue
                        return np.array(secs, dtype=np.float64) if len(secs) > 0 else np.array([], dtype=np.float64)

                    tr_sec = _to_sec(tr_ts)
                    va_sec = _to_sec(va_ts)
                    if tr_sec.size > 0 and va_sec.size > 0:
                        train_end = float(np.nanmax(tr_sec))
                        val_start = float(np.nanmin(va_sec))
                        if np.isfinite(train_end) and np.isfinite(val_start):
                            gap = val_start - train_end
                            if gap < float(purge_gap_s) - 1e-6:
                                print(f"[警告] 泄露检查：val 起点与 train 终点间隔 {gap:.1f}s，小于 purge_gap {purge_gap_s:.1f}s。")
                    else:
                        print("[信息] 泄露检查：无法可靠解析时间戳，已跳过时间间隔校验。")
            except Exception:
                pass
        except Exception as e:
            print(f"[信息] 泄露检查流程异常：{e}")

    def _collate_and_map(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """将 Dataset 的样本映射为模型期望的 batch 字段。
        规范化后的行为：
        - 原始时域窗口(raw_windows) -> time_features（供 Transformer 时域分支）
        - 频域帧(freq_windows) -> freq_features（供 Transformer 频域分支）
        - 工程/统计特征(features) -> aux_features（供 MLP 辅助分支）
        - targets -> 拆分为 target_power 与 target_states（按列类型自动判定）
        """
        # 工程/统计特征（向量），仅用于 MLP 辅助分支
        mains_list = [item["mains"].float() for item in batch]

        # 堆叠targets
        targets_list = [item["targets"].float() for item in batch]
        targets = torch.stack(targets_list, dim=0).contiguous()
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)

        # 自动拆分二值状态列与回归功率列
        # 规则：仅含0/1的列作为状态；其余作为功率
        y_np = targets.detach().cpu().numpy()
        binary_mask = []
        for j in range(y_np.shape[1]):
            col = y_np[:, j]
            vals = np.unique(col[~np.isnan(col)])
            is_binary = len(vals) > 0 and np.all(np.isin(vals, [0.0, 1.0]))
            binary_mask.append(is_binary)
        binary_mask = np.array(binary_mask, dtype=bool)

        if binary_mask.any():
            target_states = targets[:, binary_mask].contiguous()
        else:
            # 若没有二值列，使用全零占位，维度按设备数推断（若未知则用1）
            n_dev = getattr(self, 'n_devices', 1) or 1
            target_states = torch.zeros(targets.size(0), n_dev, dtype=targets.dtype, device=targets.device).contiguous()

        if (~binary_mask).any():
            target_power = targets[:, ~binary_mask].contiguous()
        else:
            target_power = torch.zeros_like(target_states).contiguous()

        # 先构建基本输出字典，后续再进行列数对齐

        out: Dict[str, Any] = {
            "target_power": target_power,
            "target_states": target_states,
        }
        # 新增：目标有效掩码（用于训练阶段跳过无效标签日志）
        out["target_power_valid_mask"] = torch.isfinite(target_power)
        # 原始时域窗口 -> time_features（必须存在）
        if all("aux_features" in item for item in batch):
            raw_list = [item["aux_features"].float() for item in batch]
            time_features = torch.stack(raw_list, dim=0)
            if time_features.dim() == 2:
                time_features = time_features.unsqueeze(1)
            time_features = time_features.contiguous()
            time_features_clean = torch.nan_to_num(time_features, nan=0.0, posinf=0.0, neginf=0.0)
            out["time_features"] = time_features_clean
            try:
                mains_seq = None
                if hasattr(self, "raw_channel_names") and self.raw_channel_names:
                    # 优先使用 P_W，如果提供 P_kW 则统一转换到 W
                    if "P_W" in self.raw_channel_names:
                        pw_idx = self.raw_channel_names.index("P_W")
                        mains_seq = time_features_clean[:, :, pw_idx]
                    elif "P_kW" in self.raw_channel_names:
                        pkw_idx = self.raw_channel_names.index("P_kW")
                        mains_seq = time_features_clean[:, :, pkw_idx] * 1000.0
                # 回退：各通道的均值（假设为功率近似，单位随原始数据）
                if mains_seq is None:
                    mains_seq = time_features_clean.mean(dim=-1)
                out["mains_seq"] = mains_seq.contiguous()
            except Exception:
                out["mains_seq"] = time_features_clean.mean(dim=-1).contiguous()
        else:
            raise RuntimeError("数据集中缺少原始时域窗口(aux_features)，无法构建 time_features。请在数据准备管线中生成 train_raw/val_raw。")

        # 工程/统计特征 -> aux_features（供 MLP）
        aux_features_batch = torch.stack(mains_list, dim=0).contiguous()
        
        # 生成辅助特征有效掩码（保持原始数据不变）
        aux_valid_mask = torch.isfinite(aux_features_batch)  # (B, n_features)
        
        # 只做最小的安全处理：极值裁剪，但保持NaN标记
        aux_features_safe = aux_features_batch.clone()
        finite_mask = torch.isfinite(aux_features_safe)
        if finite_mask.any():
            # 对有限值进行极值裁剪
            finite_values = aux_features_safe[finite_mask]
            if finite_values.numel() > 0:
                q01 = torch.quantile(finite_values, 0.01)
                q99 = torch.quantile(finite_values, 0.99)
                aux_features_safe = torch.where(
                    finite_mask,
                    torch.clamp(aux_features_safe, q01, q99),
                    aux_features_safe  # 保持NaN/Inf不变
                )
        
        out["aux_features"] = aux_features_safe
        out["aux_valid_mask"] = aux_valid_mask.contiguous()
        
        # 频域特征（若样本中包含，则拼接）
        if all("freq_features" in item for item in batch):
            freq_list = [item["freq_features"].float() for item in batch]
            freq_features = torch.stack(freq_list, dim=0)
            if freq_features.dim() == 2:
                freq_features = freq_features.unsqueeze(1)
            freq_features = freq_features.contiguous()
            # 先计算掩码，再清理
            freq_valid_mask = torch.isfinite(freq_features).all(dim=-1)
            freq_features_clean = torch.nan_to_num(freq_features, nan=0.0, posinf=0.0, neginf=0.0)
            out["freq_features"] = freq_features_clean
            out["freq_valid_mask"] = freq_valid_mask.contiguous()
            try:
                freq_valid_ratio = freq_valid_mask.float().mean(dim=1)
            except Exception:
                freq_valid_ratio = torch.ones(freq_valid_mask.size(0), device=freq_valid_mask.device, dtype=torch.float32)
            out["freq_valid_ratio"] = freq_valid_ratio.contiguous()
        # 时间戳：将起始秒展开为长度为 L 的序列（间隔为 resample_seconds）
        if all("timestamps" in item for item in batch):
            try:
                # 兼容 ISO 字符串：优先尝试解析为秒
                def _parse_to_sec(x):
                    try:
                        if isinstance(x, (int, float, np.integer, np.floating)):
                            fx = float(x)
                            return fx if np.isfinite(fx) else np.nan
                        # 字符串 -> datetime64[s] -> int64
                        dt64 = np.datetime64(str(x))
                        v = dt64.astype('datetime64[s]').astype('int64')
                        return float(v)
                    except Exception:
                        return np.nan
                start_vals = [ _parse_to_sec(item["timestamps"]) for item in batch ]
                start_ts = torch.tensor([0.0 if (np.isnan(v)) else v for v in start_vals], dtype=torch.float32)
                L = int(out["time_features"].size(1))
                delta = float(getattr(self, "resample_seconds", 5.0))
                steps = torch.arange(L, dtype=torch.float32)
                ts_mat = start_ts.view(-1, 1) + steps.view(1, -1) * delta
                out["timestamps"] = ts_mat.contiguous()
            except Exception:
                # 回退为批次级起始时间戳（与原实现一致）
                try:
                    out["timestamps"] = torch.tensor([float(item["timestamps"]) for item in batch], dtype=torch.float32)
                except Exception:
                    pass

        # 基于时间戳构造 TimeRPE 所需的周期特征（minute/hour/dow/month 的 sin/cos）
        # 输出：time_positional -> (B, T, 8)
        try:
            ts_mat = out.get("timestamps", None)
            if ts_mat is not None:
                import math
                from datetime import datetime
                bsz, seq_len = int(ts_mat.size(0)), int(ts_mat.size(1))
                tp = torch.empty(bsz, seq_len, 8, dtype=torch.float32)
                ts_cpu = ts_mat.detach().cpu().numpy()
                for b in range(bsz):
                    for t in range(seq_len):
                        v = float(ts_cpu[b, t])
                        try:
                            dt = datetime.utcfromtimestamp(v)
                        except Exception:
                            # 若解析失败，以 1970-01-01 起点相对秒粗略近似
                            base = datetime(1970, 1, 1)
                            dt = base
                        minute = dt.minute
                        hour = dt.hour
                        dow = dt.weekday()  # 0-6
                        month = dt.month - 1  # 0-11
                        # 周期角度
                        ang_min = 2.0 * math.pi * (minute / 60.0)
                        ang_hour = 2.0 * math.pi * (hour / 24.0)
                        ang_dow = 2.0 * math.pi * (dow / 7.0)
                        ang_month = 2.0 * math.pi * (month / 12.0)
                        tp[b, t, 0] = math.sin(ang_min)
                        tp[b, t, 1] = math.cos(ang_min)
                        tp[b, t, 2] = math.sin(ang_hour)
                        tp[b, t, 3] = math.cos(ang_hour)
                        tp[b, t, 4] = math.sin(ang_dow)
                        tp[b, t, 5] = math.cos(ang_dow)
                        tp[b, t, 6] = math.sin(ang_month)
                        tp[b, t, 7] = math.cos(ang_month)
                out["time_positional"] = tp.to(out["time_features"].device).contiguous()
        except Exception:
            # 若时间戳缺失或解析失败，则不输出 time_positional，由模型内部回退
            pass

        # 可选：事件标记（若样本提供），聚合为批次级布尔张量
        if all("has_events" in item for item in batch):
            try:
                flags = torch.tensor([bool(item["has_events"]) for item in batch], dtype=torch.bool)
            except Exception:
                flags = torch.zeros(len(batch), dtype=torch.bool)
            out["has_events"] = flags.contiguous()

        # 可选：时序回归目标（每窗口每设备序列）
        if all("target_seq" in item for item in batch):
            seq_list = [item["target_seq"].float() for item in batch]
            target_seq = torch.stack(seq_list, dim=0)  # (B, L, K)
            target_seq = target_seq.contiguous()
            out["target_seq"] = target_seq
            
            # Seq2Point Support: Extract center point
            # 严格对齐：center = L // 2
            L_seq = int(target_seq.size(1))
            center_idx = L_seq // 2
            out["target_point"] = target_seq[:, center_idx, :].contiguous()
            
            # 新增：序列目标有效掩码（逐设备与逐时间步）
            per_dev_valid = torch.isfinite(target_seq).contiguous()  # (B, L, K)
            out["target_seq_per_device_valid_mask"] = per_dev_valid
            # 汇总掩码（逐时间步）：过去使用 any(dim=-1)，现保留做兜底
            out["target_seq_valid_mask"] = per_dev_valid.any(dim=-1).contiguous()

        # 可选：时序分类目标（逐时间步开关状态）
        if all("status_seq" in item for item in batch):
            s_list = [item["status_seq"].float() for item in batch]
            status_seq = torch.stack(s_list, dim=0)  # (B, L, K)
            status_seq = status_seq.contiguous()
            out["status_seq"] = status_seq
            out["status_seq_valid_mask"] = torch.isfinite(status_seq).contiguous()

        # 对齐 target_power 的列数与设备数（若不一致）
        try:
            n_dev = int(getattr(self, 'n_devices', 0))
            if n_dev <= 0:
                n_dev = int(out["target_states"].size(1)) if out["target_states"].dim() == 2 else int(out["target_power"].size(1))
        except Exception:
            n_dev = int(out["target_states"].size(1)) if out["target_states"].dim() == 2 else int(out["target_power"].size(1))

        tp = out["target_power"]
        if isinstance(tp, torch.Tensor) and tp.dim() == 2 and n_dev > 0 and tp.size(1) != n_dev:
            # 若存在 target_seq，则用时间均值生成每设备窗口级功率
            if "target_seq" in out and isinstance(out["target_seq"], torch.Tensor) and out["target_seq"].dim() == 3:
                out["target_power"] = torch.nanmean(out["target_seq"], dim=1).contiguous()
            else:
                B, K_old = tp.size(0), tp.size(1)
                if K_old < n_dev:
                    pad = torch.zeros(B, n_dev - K_old, dtype=tp.dtype, device=tp.device)
                    out["target_power"] = torch.cat([tp, pad], dim=1).contiguous()
                else:
                    out["target_power"] = tp[:, :n_dev].contiguous()
            out["target_power_valid_mask"] = torch.isfinite(out["target_power"]) 

        # 提取总功率 total_power (B, 1)，优先从工程特征中的 P_kW_mean，其次从原始窗口中的 P_kW 通道均值
        total_power: Optional[torch.Tensor] = None
        # 1) 从工程特征中提取 P_kW_mean
        if self.feature_names:
            try:
                pkw_mean_idx = self.feature_names.index("P_kW_mean")
                total_power = aux_features_batch[:, pkw_mean_idx].unsqueeze(1)
                if self.normalize_total_power_to_watts:
                    total_power = total_power * 1000.0
            except ValueError:
                total_power = None
        # 2) 回退：从原始窗口的 P_kW 通道按时间平均得到窗口总功率代表值
        if total_power is None and hasattr(self, "raw_channel_names") and self.raw_channel_names and out.get("time_features") is not None:
            try:
                pkw_idx = self.raw_channel_names.index("P_kW")
                total_power = out["time_features"][:, :, pkw_idx].mean(dim=1, keepdim=True)
                if self.normalize_total_power_to_watts:
                    total_power = total_power * 1000.0
            except ValueError:
                total_power = None
        if total_power is None:
            # 若仍不可用，构造占位列
            total_power = torch.zeros(out["aux_features"].size(0), 1, dtype=out["aux_features"].dtype)
        out["total_power"] = total_power.contiguous()

        return out

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning 标准钩子：资源释放占位。当前无需特殊处理。"""
        return

# 兼容性别名：历史脚本可能使用 PreparedDataset 名称
PreparedDataset = NILMDataset
