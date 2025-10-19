"""
数据准备主流程
Data Preparation Pipeline

整合所有数据准备功能的主流程脚本，包括：
- 数据分段
- 窗口化
- 交叉验证
- 标签处理
- 特征工程
- 评估
"""

import os
import yaml
import logging
import pandas as pd
import polars as pl
import numpy as np
from datetime import datetime
from scipy import signal
from typing import Dict, List, Tuple, Optional, Any
import pickle
import json
from dataclasses import asdict

from .segmentation import DataSegmenter
from .windowing import WindowGenerator
from .cross_validation import WalkForwardCV, CVFold
from .label_handling import LabelHandler
from .feature_engineering import FeatureEngineer
from .evaluation import EventEvaluator

logger = logging.getLogger(__name__)

class DataPreparationPipeline:
    """数据准备主流程"""
    
    def __init__(self, config_path: str):
        """
        初始化数据准备流程
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # 先设置输出目录
        self.output_dir = self.config['data_storage']['output_directory']
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 然后设置日志和随机种子
        self._setup_logging()
        self._setup_random_seed()
        
        # 初始化各个组件
        self.segmenter = DataSegmenter(self.config)
        self.window_generator = WindowGenerator(self.config)
        self.cv_generator = WalkForwardCV(self.config)
        self.label_handler = LabelHandler(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.evaluator = EventEvaluator(self.config)
        
        logger.info("数据准备流程初始化完成")
    
    def _load_config(self) -> Dict:
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_logging(self):
        """设置日志"""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.output_dir, 'pipeline.log')),
                logging.StreamHandler()
            ]
        )
    
    def _setup_random_seed(self):
        """设置随机种子"""
        seed = self.config.get('randomness_control', {}).get('random_seed', 42)
        np.random.seed(seed)
        logger.info(f"设置随机种子: {seed}")
    
    def run_full_pipeline(self, data_path: str) -> Dict[str, Any]:
        """
        运行完整的数据准备流程
        
        Args:
            data_path: 输入数据路径
            
        Returns:
            pipeline_results: 流程结果
        """
        logger.info("开始运行完整数据准备流程")
        
        pipeline_results = {
            'start_time': datetime.now().isoformat(),
            'config': self.config,
            'steps': {}
        }
        
        try:
            # 步骤1: 加载数据
            logger.info("步骤1: 加载数据")
            data = self._load_data(data_path)
            pipeline_results['steps']['data_loading'] = {
                'status': 'completed',
                'data_shape': data.shape,
                'columns': list(data.columns)
            }
            
            # 步骤2: 数据分段
            logger.info("步骤2: 数据分段")
            # 转换为polars DataFrame
            data_pl = pl.from_pandas(data)
            segments, segments_meta = self.segmenter.segment_device_data(data_pl, "device_1")
            pipeline_results['steps']['segmentation'] = {
                'status': 'completed',
                'num_segments': len(segments),
                'total_samples': sum(len(seg) for seg in segments)
            }
            
            # 步骤3: 窗口化
            logger.info("步骤3: 窗口化")
            windows_data = self._generate_windows(segments, segments_meta)
            pipeline_results['steps']['windowing'] = {
                'status': 'completed',
                'num_windows': len(windows_data['windows']),
                'window_length': self.config['windowing']['default_window_length']
            }
            
            # 步骤4: 标签处理
            logger.info("步骤4: 标签处理")
            labels_data = self._process_labels(windows_data, segments_meta)
            pipeline_results['steps']['label_processing'] = {
                'status': 'completed',
                'label_distribution': self._get_label_distribution(labels_data['labels'])
            }
            
            # 步骤5: 交叉验证分割
            logger.info("步骤5: 交叉验证分割")
            cv_splits = self._create_cv_splits(windows_data, labels_data, segments_meta)
            pipeline_results['steps']['cross_validation'] = {
                'status': 'completed',
                'num_folds': len(cv_splits),
                'fold_info': {f"fold_{fold.fold_id}": f"train_{len(fold.train_segments or [])} val_{len(fold.val_segments or [])}" for fold in cv_splits}
            }
            
            # 步骤6: 特征工程
            logger.info("步骤6: 特征工程")
            processed_data = self._process_features(windows_data, cv_splits)
            pipeline_results['steps']['feature_engineering'] = {
                'status': 'completed',
                'num_features': processed_data['feature_names'] if 'feature_names' in processed_data else 'unknown'
            }
            
            # 步骤7: 保存处理后的数据
            logger.info("步骤7: 保存数据")
            self._save_processed_data(processed_data, labels_data, cv_splits, segments_meta)
            pipeline_results['steps']['data_saving'] = {
                'status': 'completed',
                'output_directory': self.output_dir
            }
            
            pipeline_results['status'] = 'completed'
            pipeline_results['end_time'] = datetime.now().isoformat()
            
            logger.info("完整数据准备流程运行完成")
            
        except Exception as e:
            logger.error(f"流程运行失败: {e}")
            pipeline_results['status'] = 'failed'
            pipeline_results['error'] = str(e)
            pipeline_results['end_time'] = datetime.now().isoformat()
            raise
        
        # 保存流程结果
        self._save_pipeline_results(pipeline_results)
        
        return pipeline_results
    
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """加载数据"""
        if os.path.isfile(data_path):
            # 单个文件
            if data_path.endswith('.csv'):
                data = pd.read_csv(data_path)
            elif data_path.endswith('.parquet'):
                data = pd.read_parquet(data_path)
            else:
                raise ValueError(f"不支持的文件格式: {data_path}")
        elif os.path.isdir(data_path):
            # 目录，合并所有CSV文件
            csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
            if not csv_files:
                raise ValueError(f"目录 {data_path} 中没有找到CSV文件")
            
            logger.info(f"找到 {len(csv_files)} 个CSV文件，开始合并")
            data_frames = []
            
            for csv_file in sorted(csv_files):
                file_path = os.path.join(data_path, csv_file)
                logger.info(f"加载文件: {csv_file}")
                df = pd.read_csv(file_path)
                
                # 添加设备名称列（从文件名提取）
                device_name = csv_file.replace('cleaned_', '').split('_')[0]
                df['device_name'] = device_name
                
                data_frames.append(df)
            
            # 合并所有数据
            data = pd.concat(data_frames, ignore_index=True)
            logger.info(f"数据合并完成，总共 {len(data_frames)} 个文件")
        else:
            raise ValueError(f"路径不存在: {data_path}")
        
        logger.info(f"加载数据完成: {data.shape}")
        return data
    
    def _generate_windows(self, segments: List[pl.DataFrame], 
                         segments_meta: List) -> Dict[str, Any]:
        """生成窗口"""
        all_windows = []
        all_metadata = []
        
        feature_columns_global = None
        for i, segment_data in enumerate(segments):
            segment_meta = segments_meta[i]
            
            windows, metadata, feature_columns = self.window_generator.generate_windows(
                segment_data, segment_meta.device_name, segment_meta.segment_id
            )
            
            all_windows.extend(windows)
            all_metadata.extend(metadata)
            # 记录首个段的列名，若后续不一致则告警
            if feature_columns_global is None:
                feature_columns_global = feature_columns
            elif feature_columns_global != feature_columns:
                logger.warning("不同段的特征列不一致，后续提取按首段列名进行")
        
        return {
            'windows': all_windows,
            'metadata': all_metadata,
            'feature_columns': feature_columns_global or []
        }
    
    def _process_labels(self, windows_data: Dict[str, Any], 
                       segments_meta: List) -> Dict[str, Any]:
        """处理标签：优先使用开关(on/off)配置生成；否则回退事件检测。"""
        windows = windows_data['windows']
        metadata = windows_data['metadata']
        feature_columns = windows_data.get('feature_columns', [])

        # 优先尝试基于on/off阈值的标签生成
        labels_onoff = None
        try:
            labels_onoff = self.label_handler.generate_onoff_labels_from_windows(
                windows, feature_columns, metadata
            )
        except Exception as e:
            logger.warning(f"on/off 标签生成异常，回退事件检测: {e}")

        if labels_onoff is not None:
            return {
                'labels': labels_onoff,
                'label_metadata': metadata
            }

        # 回退：基于事件定义的窗口检测
        labels = []
        for window, meta in zip(windows, metadata):
            has_event = self._window_has_event(window, feature_columns)
            labels.append(1 if has_event else 0)

        return {
            'labels': np.array(labels),
            'label_metadata': metadata
        }

    def _window_has_event(self, window: np.ndarray, feature_columns: List[str]) -> bool:
        """基于配置的事件定义，在单个窗口内检测是否存在事件。"""
        if not hasattr(self, 'label_handler') or self.label_handler is None:
            # 若未初始化，创建一个以当前配置为准的处理器
            self.label_handler = LabelHandler(self.config)

        # 遍历事件定义，任一命中即认为窗口有事件
        for ed in self.label_handler.event_definitions:
            if ed.feature_column not in feature_columns:
                continue
            idx = feature_columns.index(ed.feature_column)
            series = window[:, idx]
            if len(series) < 2:
                continue
            diff = np.diff(series, prepend=series[0])
            if ed.direction in ('up', 'above'):
                raw = diff > ed.threshold
            elif ed.direction in ('down', 'below'):
                raw = diff < -ed.threshold
            else:  # both
                raw = np.abs(diff) > ed.threshold

            if not np.any(raw):
                continue

            # 最小持续时间过滤（简单连续True判定）
            if ed.min_duration and ed.min_duration > 1:
                # 计算连续True的最长长度
                max_run = 0
                current = 0
                for flag in raw:
                    if flag:
                        current += 1
                        if current > max_run:
                            max_run = current
                    else:
                        current = 0
                if max_run < ed.min_duration:
                    continue

            # 最大间隔（简单策略：不在窗口内展开桥接，保持保守）
            # 如需更严格，可以在后续版本加入对 max_gap 的桥接逻辑

            return True

        return False
    
    def _create_cv_splits(self, windows_data: Dict[str, Any], 
                         labels_data: Dict[str, Any],
                         segments_meta: List) -> List[CVFold]:
        """创建交叉验证分割"""
        metadata = pd.DataFrame(windows_data['metadata'])
        
        # 将segments_meta转换为DataFrame
        segments_df = pd.DataFrame([
            {
                'device_name': seg.device_name,
                'segment_id': seg.segment_id,
                'start_ts': seg.start_ts,
                'end_ts': seg.end_ts,
                'n_rows': seg.n_rows,
                'duration_hours': seg.duration_hours,
                'has_events': seg.has_events,
                'event_density': seg.event_density,
                'max_gap_seconds': seg.max_gap_seconds,
                'filled_ratio': seg.filled_ratio
            }
            for seg in segments_meta
        ])
        
        cv_folds = self.cv_generator.create_folds(segments_df)
        return cv_folds
    
    def _process_features(self, windows_data: Dict[str, Any], 
                         cv_splits: List[CVFold]) -> Dict[str, Any]:
        """处理特征"""
        windows = np.array(windows_data['windows'])
        metadata = windows_data['metadata']
        feature_columns = windows_data.get('feature_columns', [])
        
        processed_folds = []
        # 选择要导出的原始通道（按列名动态匹配）
        raw_channel_candidates = ['P_kW', 'Q_kvar', 'S_kVA', 'V2_V', 'V3_V', 'I2_A', 'I3_A']
        # 基于首段列名，过滤出实际存在的原始通道
        raw_channel_indices: List[int] = []
        raw_channel_names: List[str] = []
        if feature_columns:
            for cname in raw_channel_candidates:
                if cname in feature_columns:
                    idx = feature_columns.index(cname)
                    raw_channel_indices.append(idx)
                    raw_channel_names.append(cname)
        # 兜底：若候选列均不存在但存在至少一列，则取前1列作为原始功率信号
        if not raw_channel_indices and feature_columns:
            raw_channel_indices = [0]
            raw_channel_names = [feature_columns[0]]

        # 预先为每个折收集索引（不做列筛选），以便随后计算全局掩码
        indices_per_fold: List[Tuple[np.ndarray, np.ndarray]] = []
        for fold_idx, fold in enumerate(cv_splits):
            train_indices = []
            val_indices = []
            for i, meta in enumerate(metadata):
                window_start_ts = meta.start_ts
                window_end_ts = meta.end_ts
                if (window_start_ts >= fold.train_start_ts and 
                    window_end_ts <= fold.train_end_ts):
                    train_indices.append(i)
                elif (window_start_ts >= fold.val_start_ts and 
                      window_end_ts <= fold.val_end_ts):
                    val_indices.append(i)
            train_indices = np.array(train_indices)
            val_indices = np.array(val_indices)
            if len(train_indices) == 0 or len(val_indices) == 0:
                logger.warning(f"折 {fold_idx} 训练或验证集为空，跳过")
                indices_per_fold.append((np.array([]), np.array([])))
                continue
            indices_per_fold.append((train_indices, val_indices))

        # 计算全局列掩码：基于所有折的训练集特征合并后进行有效性与方差筛选
        global_train_features_list = []
        for fold_idx, (train_indices, val_indices) in enumerate(indices_per_fold):
            if train_indices.size == 0:
                continue
            train_windows = windows[train_indices]
            train_meta = [metadata[i] for i in train_indices]
            for window, meta in zip(train_windows, train_meta):
                wf = self._extract_window_features(window, feature_columns)
                tf = self._extract_time_and_energy_features(window, meta, feature_columns)
                uf = self._extract_unbalance_features(window, feature_columns)
                global_train_features_list.append(np.concatenate([wf, tf, uf]))

        if not global_train_features_list:
            logger.warning("所有折均为空，无法处理特征")
            return {'folds': [], 'feature_names': []}

        global_train_features = np.array(global_train_features_list)
        # 全局中位数填充（全NaN列回退0）
        global_col_median = np.nanmedian(global_train_features, axis=0)
        global_col_median = np.where(np.isfinite(global_col_median), global_col_median, 0.0)
        global_train_features = np.where(np.isfinite(global_train_features), global_train_features, global_col_median)

        # 全局列有效性筛选（统一掩码）
        global_var = np.nanvar(global_train_features, axis=0)
        global_valid_cols_mask = np.isfinite(global_var) & (global_var > 1e-12)
        if not np.all(global_valid_cols_mask):
            removed_count = int(np.sum(~global_valid_cols_mask))
            logger.warning(f"全局列筛选：移除{removed_count}个低方差/无效特征列（统一应用于所有折）")

        feature_names_all = self._get_feature_names()
        unified_feature_names = [name for name, keep in zip(feature_names_all, global_valid_cols_mask) if keep]

        # 逐折处理并应用统一掩码
        for fold_idx, (train_indices, val_indices) in enumerate(indices_per_fold):
            if train_indices.size == 0 or val_indices.size == 0:
                continue
            logger.info(f"处理第 {fold_idx + 1} 折特征（统一掩码）")

            # 训练集
            train_windows = windows[train_indices]
            train_meta = [metadata[i] for i in train_indices]
            train_features_list = []
            train_raw_list = []
            train_freq_list = []
            for window, meta in zip(train_windows, train_meta):
                wf = self._extract_window_features(window, feature_columns)
                tf = self._extract_time_and_energy_features(window, meta, feature_columns)
                uf = self._extract_unbalance_features(window, feature_columns)
                train_features_list.append(np.concatenate([wf, tf, uf]))
                # 提取原始通道序列（形状: [T, C_raw]）
                if raw_channel_indices:
                    try:
                        train_raw_list.append(np.stack([window[:, idx] for idx in raw_channel_indices], axis=-1))
                    except Exception:
                        # 兜底：仅保留功率列
                        train_raw_list.append(window[:, :1])
                # 频域表示（支持摘要/FFT/STFT）
                freq_repr = self._extract_frequency_representation(window, feature_columns)
                train_freq_list.append(freq_repr)
            train_features = np.array(train_features_list)
            train_features = np.where(np.isfinite(train_features), train_features, global_col_median)
            train_features = train_features[:, global_valid_cols_mask]

            # 拟合缩放器（按折）
            self.feature_engineer.fit_normalization(train_features)
            train_features_scaled = self.feature_engineer.apply_normalization(train_features)
            # 组装频域数组（summary/fft 为 (N, F)，stft 为 (N, T_f, F)）
            train_freq = np.stack(train_freq_list, axis=0) if train_freq_list else np.empty((0,))

            # 验证集
            val_windows = windows[val_indices]
            val_meta = [metadata[i] for i in val_indices]
            val_features_list = []
            val_raw_list = []
            val_freq_list = []
            for window, meta in zip(val_windows, val_meta):
                wf = self._extract_window_features(window, feature_columns)
                tf = self._extract_time_and_energy_features(window, meta, feature_columns)
                uf = self._extract_unbalance_features(window, feature_columns)
                val_features_list.append(np.concatenate([wf, tf, uf]))
                # 提取原始通道序列（形状: [T, C_raw]）
                if raw_channel_indices:
                    try:
                        val_raw_list.append(np.stack([window[:, idx] for idx in raw_channel_indices], axis=-1))
                    except Exception:
                        val_raw_list.append(window[:, :1])
                # 频域表示（支持摘要/FFT/STFT）
                freq_repr = self._extract_frequency_representation(window, feature_columns)
                val_freq_list.append(freq_repr)
            val_features = np.array(val_features_list)
            val_features = np.where(np.isfinite(val_features), val_features, global_col_median)
            val_features = val_features[:, global_valid_cols_mask]
            val_features_scaled = self.feature_engineer.apply_normalization(val_features)
            # 组装频域数组
            val_freq = np.stack(val_freq_list, axis=0) if val_freq_list else np.empty((0,))

            # 组装原始序列数组
            train_raw = np.array(train_raw_list) if train_raw_list else np.empty((0, 0, 0))
            val_raw = np.array(val_raw_list) if val_raw_list else np.empty((0, 0, 0))

            processed_fold = {
                'fold_idx': fold_idx,
                'train_features': train_features_scaled,
                'val_features': val_features_scaled,
                'train_raw': train_raw,
                'val_raw': val_raw,
                'train_freq': train_freq,
                'val_freq': val_freq,
                'train_indices': train_indices,
                'val_indices': val_indices,
                'scaler': self.feature_engineer.scaler,
                'feature_names': unified_feature_names,
                'raw_channel_names': raw_channel_names
            }
            processed_folds.append(processed_fold)

        return {
            'folds': processed_folds,
            'feature_names': unified_feature_names if processed_folds else []
        }

    def _extract_frequency_summary(self, window: np.ndarray, feature_columns: List[str]) -> np.ndarray:
        """
        从窗口优选信号提取频域摘要特征。

        返回: [total_energy, low_freq_energy, high_freq_energy, spectral_centroid, bandwidth]
        """
        # 优先选择功率或电压/电流通道
        candidates = ['P_kW', 'V2_V', 'I2_A', 'Q_kvar']
        sig_idx = None
        for cname in candidates:
            if cname in feature_columns:
                idx = feature_columns.index(cname)
                if idx < window.shape[1]:
                    sig_idx = idx
                    break
        if sig_idx is None:
            # 兜底：使用第0列
            sig_idx = 0 if window.shape[1] > 0 else None
        if sig_idx is None:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        x = window[:, sig_idx].astype(np.float32)
        if x.size < 8:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        spec = np.fft.rfft(x)
        mag = np.abs(spec).astype(np.float32)
        total = float(np.sum(mag))
        if total <= 1e-8:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        n = mag.shape[0]
        low_band = max(1, n // 4)
        high_band_start = n // 2
        low_energy = float(np.sum(mag[:low_band]))
        high_energy = float(np.sum(mag[high_band_start:]))

        freqs = np.linspace(0.0, 1.0, n, dtype=np.float32)
        centroid = float(np.sum(freqs * mag) / total)
        var = float(np.sum(((freqs - centroid) ** 2) * mag) / total)
        bandwidth = float(np.sqrt(max(var, 0.0)))

        return np.array([total, low_energy, high_energy, centroid, bandwidth], dtype=np.float32)

    def _extract_frequency_representation(self, window: np.ndarray, feature_columns: List[str]) -> np.ndarray:
        """
        最佳实践：统一使用 STFT 生成频域时序波形。
        返回：(T_frames, F_bins)，默认返回幅度谱（magnitude=True）。

        配置示例：
        frequency:
          signal: P_kW
          n_fft: 256
          win_length: 256
          hop_length: 64
          magnitude: true
        """
        freq_cfg = (self.config or {}).get('frequency', {})
        signal_pref = freq_cfg.get('signal', 'P_kW')
        n_fft = int(freq_cfg.get('n_fft', max(8, window.shape[0])))
        win_length = int(freq_cfg.get('win_length', n_fft))
        hop_length = int(freq_cfg.get('hop_length', max(1, win_length // 4)))
        use_magnitude = bool(freq_cfg.get('magnitude', True))

        # 选择信号列（优先功率，退化到电压/电流，再兜底第0列）
        sig_idx: Optional[int] = None
        if signal_pref in feature_columns:
            idx = feature_columns.index(signal_pref)
            if idx < window.shape[1]:
                sig_idx = idx
        if sig_idx is None:
            for cname in ['P_kW', 'V2_V', 'I2_A', 'Q_kvar']:
                if cname in feature_columns:
                    idx = feature_columns.index(cname)
                    if idx < window.shape[1]:
                        sig_idx = idx
                        break
        if sig_idx is None:
            sig_idx = 0 if window.shape[1] > 0 else None
        if sig_idx is None:
            # 空窗口兜底
            return np.zeros((1,), dtype=np.float32)

        x = window[:, sig_idx].astype(np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # 统一 STFT：按帧提取谱
        frames: List[np.ndarray] = []
        if win_length <= 0:
            win_length = min(x.size, n_fft)
        if hop_length <= 0:
            hop_length = max(1, win_length // 4)
        start = 0
        while start + win_length <= x.size:
            seg = x[start:start + win_length]
            if seg.size < win_length:
                seg = np.pad(seg, (0, win_length - seg.size))
            spec = np.fft.rfft(seg, n=n_fft)
            mag = np.abs(spec).astype(np.float32) if use_magnitude else spec.astype(np.complex64)
            frames.append(mag)
            start += hop_length
        if not frames:
            # 不足一帧时，用整体零填充生成一帧
            seg = x
            if seg.size < win_length:
                seg = np.pad(seg, (0, win_length - seg.size))
            spec = np.fft.rfft(seg, n=n_fft)
            mag = np.abs(spec).astype(np.float32) if use_magnitude else spec.astype(np.complex64)
            frames.append(mag)
        return np.stack(frames, axis=0)
    
    def _extract_window_features(self, window: np.ndarray, feature_columns: List[str]) -> np.ndarray:
        """
        从单个窗口中提取特征
        
        Args:
            window: 形状为 (window_length, n_features) 的窗口数据
            
        Returns:
            window_features: 提取的特征向量
        """
        features = []
        
        # 仅使用指定通道（排除相1），按列名动态索引
        desired_cols = ['P_kW', 'Q_kvar', 'S_kVA', 'V2_V', 'V3_V', 'I2_A', 'I3_A', 'F_Hz', 'THD_IN_F']
        for col_name in desired_cols:
            if col_name not in feature_columns:
                continue
            col_idx = feature_columns.index(col_name)
            if col_idx >= window.shape[1]:
                continue
            col_data = window[:, col_idx]
            
            # 1. 原始统计特征
            features.extend([
                float(np.nanmean(col_data)),           # 均值
                float(np.nanstd(col_data)),            # 标准差
                float(np.nanmin(col_data)),            # 最小值
                float(np.nanmax(col_data)),            # 最大值
                float(np.nanmedian(col_data)),         # 中位数
            ])
            
            # 2. 变化率特征
            if len(col_data) > 1:
                diff = np.diff(col_data)
                features.extend([
                    float(np.nanmean(diff)),           # 平均变化率
                    float(np.nanstd(diff)),            # 变化率标准差
                    float(np.nansum(np.abs(diff))),    # 总变化量
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
            
            # 3. 趋势特征（线性斜率）
            if len(col_data) > 2:
                x = np.arange(len(col_data))
                slope = np.polyfit(x, col_data, 1)[0]
                features.append(float(slope))
            else:
                features.append(0.0)
        
        # === 进阶特征 ===
        
        # 4. 功率相关特征（如果有P和Q）
        if window.shape[1] >= 2:
            P = window[:, 0]  # P_kW
            Q = window[:, 1]  # Q_kvar
            
            # 功率因数 (Power Factor)
            S = np.sqrt(P**2 + Q**2)
            power_factor = np.mean(P / (S + 1e-8))
            features.append(power_factor)
            
            # 稳态功率指纹 - 开关后功率平均值
            # 检测功率变化点（简化版）
            P_diff = np.abs(np.diff(P))
            change_threshold = np.std(P_diff) * 2
            change_points = np.where(P_diff > change_threshold)[0]
            
            if len(change_points) > 0:
                # 取最后一个变化点后的稳态值
                last_change = change_points[-1] + 1
                steady_state_P = np.mean(P[last_change:])
                steady_state_Q = np.mean(Q[last_change:])
                steady_state_S = np.mean(S[last_change:])
            else:
                # 如果没有明显变化，取整个窗口的平均值
                steady_state_P = np.mean(P)
                steady_state_Q = np.mean(Q)
                steady_state_S = np.mean(S)
            
            features.extend([steady_state_P, steady_state_Q, steady_state_S])
            
            # 无功有功比
            reactive_active_ratio = np.mean(Q / (P + 1e-8))
            features.append(reactive_active_ratio)
        
        # 5. 瞬态波形特征
        if window.shape[1] >= 1:
            P = window[:, 0]  # 使用功率信号
            
            # 瞬态斜率特征
            if len(P) > 1:
                diff = np.diff(P)
                # 最大上升斜率
                max_rise_slope = np.max(diff) if len(diff) > 0 else 0
                # 最大下降斜率
                max_fall_slope = np.min(diff) if len(diff) > 0 else 0
                # 平均绝对斜率
                avg_abs_slope = np.mean(np.abs(diff)) if len(diff) > 0 else 0
                
                features.extend([max_rise_slope, max_fall_slope, avg_abs_slope])
                
                # 瞬态持续时间特征
                # 检测瞬态事件（变化超过阈值的连续区间）
                threshold = np.std(diff) * 1.5
                transient_mask = np.abs(diff) > threshold
                
                if np.any(transient_mask):
                    # 计算瞬态事件的平均持续时间
                    transient_durations = []
                    in_transient = False
                    current_duration = 0
                    
                    for is_transient in transient_mask:
                        if is_transient:
                            if not in_transient:
                                in_transient = True
                                current_duration = 1
                            else:
                                current_duration += 1
                        else:
                            if in_transient:
                                transient_durations.append(current_duration)
                                in_transient = False
                                current_duration = 0
                    
                    if in_transient:  # 如果最后还在瞬态中
                        transient_durations.append(current_duration)
                    
                    avg_transient_duration = np.mean(transient_durations) if transient_durations else 0
                    max_transient_duration = np.max(transient_durations) if transient_durations else 0
                else:
                    avg_transient_duration = 0
                    max_transient_duration = 0
                
                features.extend([avg_transient_duration, max_transient_duration])
            else:
                features.extend([0, 0, 0, 0, 0])
        
        # 6. 扩展谐波特征（H3-H15）与能量比例（仅使用2/3相）
        signal_candidates = ['I2_A', 'I3_A', 'V2_V', 'V3_V']
        sig_idx = None
        for cname in signal_candidates:
            if cname in feature_columns:
                idx = feature_columns.index(cname)
                if idx < window.shape[1]:
                    sig_idx = idx
                    break
        if sig_idx is not None:
            sig = window[:, sig_idx]
            if len(sig) >= 8:
                fft_result = np.fft.fft(sig)
                n_samples = len(sig)
                half_len = len(fft_result) // 2

                harmonic_amps = []
                for h in [3, 5, 7, 9, 11, 13, 15]:
                    idx = int(h * n_samples / 50) if n_samples > (h - 1) else h
                    idx = min(idx, half_len)
                    amp = float(np.abs(fft_result[idx])) if idx < len(fft_result) else 0.0
                    harmonic_amps.append(amp)

                features.extend(harmonic_amps)

                fft_mag = np.abs(fft_result[:half_len])
                total_energy_fft = float(np.sum(fft_mag ** 2)) if half_len > 0 else 0.0
                harmonic_energy = float(np.sum(np.array(harmonic_amps) ** 2))
                harmonic_energy_ratio = harmonic_energy / (total_energy_fft + 1e-8)
                features.append(harmonic_energy_ratio)
            else:
                features.extend([0.0] * 8)
        else:
            features.extend([0.0] * 8)
        
        # 7. 频域能量特征与瞬态增强（FFT/STFT/CWT）
        if window.shape[1] >= 1:
            p_signal = window[:, 0]  # 使用功率信号

            if len(p_signal) >= 4:
                # FFT能量特征
                fft_result = np.fft.fft(p_signal)
                fft_magnitude = np.abs(fft_result[:len(fft_result)//2])

                total_energy = float(np.sum(fft_magnitude ** 2))
                low_freq_energy = float(np.sum(fft_magnitude[:len(fft_magnitude)//4] ** 2))
                high_freq_energy = float(np.sum(fft_magnitude[3 * len(fft_magnitude)//4:] ** 2))
                freqs = np.arange(len(fft_magnitude))
                spectral_centroid = float(np.sum(freqs * fft_magnitude) / (np.sum(fft_magnitude) + 1e-8))

                # STFT瞬态能量：时间轴上高频能量的方差衡量瞬态
                try:
                    f, t, Zxx = signal.stft(p_signal, nperseg=min(64, len(p_signal)))
                    Z_mag = np.abs(Zxx)
                    # 取上四分之一频段的能量随时间的方差
                    hf_start = int(3 * len(f) / 4)
                    hf_energy_t = np.sum(Z_mag[hf_start:, :] ** 2, axis=0)
                    stft_transient_energy = float(np.var(hf_energy_t))
                except Exception:
                    stft_transient_energy = 0.0

                # CWT小波能量（Ricker/Morlet简单能量聚合）
                try:
                    widths = np.arange(1, max(2, min(32, len(p_signal) // 2)))
                    cwt_coeffs = signal.cwt(p_signal - np.mean(p_signal), signal.ricker, widths)
                    cwt_energy = float(np.sum(cwt_coeffs ** 2))
                except Exception:
                    cwt_energy = 0.0

                features.extend([
                    total_energy, low_freq_energy, high_freq_energy, spectral_centroid,
                    stft_transient_energy, cwt_energy
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # 8. 占空比和周期性检测特征
        if window.shape[1] >= 1:
            signal = window[:, 0]  # 使用功率信号
            
            # 占空比计算（信号高于平均值的比例）
            signal_mean = np.mean(signal)
            duty_cycle = np.mean(signal > signal_mean)
            features.append(duty_cycle)
            
            # 周期性检测（自相关）
            if len(signal) > 1:
                # 计算自相关
                signal_centered = signal - np.mean(signal)
                autocorr = np.correlate(signal_centered, signal_centered, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                
                # 寻找第一个局部最大值（除了lag=0）
                if len(autocorr) > 2:
                    # 归一化自相关
                    autocorr_norm = autocorr / (autocorr[0] + 1e-8)
                    
                    # 寻找周期性指标
                    max_autocorr = np.max(autocorr_norm[1:]) if len(autocorr_norm) > 1 else 0
                    
                    # 周期性强度
                    periodicity_strength = max_autocorr
                else:
                    periodicity_strength = 0
            else:
                periodicity_strength = 0
            
            features.append(periodicity_strength)
        
        # 9. 频率特征（如果有频率列）与过零率/相位漂移代理
        if window.shape[1] >= 6:  # F_Hz是第6列
            freq = window[:, 5]
            freq_deviation = float(np.mean(freq - 50.0))  # 频率偏差
            freq_stability = float(np.std(freq))  # 频率稳定性

            # 过零率（电压/电流信号）与相位漂移代理
            zcr_v = 0.0
            zcr_i = 0.0
            phase_shift_proxy = 0.0
            try:
                v_signal = window[:, 3] if window.shape[1] >= 4 else None
                i_signal = window[:, 4] if window.shape[1] >= 5 else None
                if v_signal is not None:
                    zcr_v = float(np.mean((v_signal[:-1] <= 0) & (v_signal[1:] > 0)))
                if i_signal is not None:
                    zcr_i = float(np.mean((i_signal[:-1] <= 0) & (i_signal[1:] > 0)))
                if v_signal is not None and i_signal is not None:
                    # 简单的相位漂移代理：首次上升过零位置差/窗口长度
                    vz_idx = np.argmax((v_signal[:-1] <= 0) & (v_signal[1:] > 0)) if len(v_signal) > 1 else 0
                    iz_idx = np.argmax((i_signal[:-1] <= 0) & (i_signal[1:] > 0)) if len(i_signal) > 1 else 0
                    phase_shift_proxy = float((iz_idx - vz_idx) / max(1, len(v_signal)))
            except Exception:
                pass

            features.extend([freq_deviation, freq_stability, zcr_v, zcr_i, phase_shift_proxy])
        
        # 10. THD特征（优先使用中性或2/3相，避免1相）
        thd_v_cols = [c for c in feature_columns if c.startswith('THD_V') and c.endswith('_F') and not c.startswith('THD_V1')]
        thd_i_cols = [c for c in feature_columns if c.startswith('THD_I') and c.endswith('_F') and not c.startswith('THD_I1')]
        if 'THD_IN_F' in feature_columns:
            thd_i_cols = ['THD_IN_F'] + thd_i_cols
        thd_v = np.zeros(window.shape[0])
        thd_i = np.zeros(window.shape[0])
        if thd_v_cols:
            v_idx = feature_columns.index(thd_v_cols[0])
            if v_idx < window.shape[1]:
                thd_v = window[:, v_idx]
        if thd_i_cols:
            i_idx = feature_columns.index(thd_i_cols[0])
            if i_idx < window.shape[1]:
                thd_i = window[:, i_idx]
        features.extend([
            float(np.nanmean(thd_v)),
            float(np.nanmean(thd_i)),
            float(np.nanmean(thd_v + thd_i)),  # THD总和
            float(np.nanstd(thd_v)),           # THD电压变化
            float(np.nanstd(thd_i)),           # THD电流变化
        ])
        
        return np.array(features)

    def _extract_time_and_energy_features(self, window: np.ndarray, meta, feature_columns: List[str]) -> np.ndarray:
        """基于窗口元数据的时间上下文与能耗积分特征"""
        feats: List[float] = []
        try:
            # 时间上下文
            start_dt = datetime.utcfromtimestamp(int(meta.start_ts))
            hour = start_dt.hour
            dow = start_dt.weekday()  # 0=周一
            is_weekend = 1.0 if dow >= 5 else 0.0
            # 班次（默认三班制：夜22-6，早6-14，晚14-22）
            if 22 <= hour or hour < 6:
                shift_id = 0
            elif 6 <= hour < 14:
                shift_id = 1
            else:
                shift_id = 2
            is_business_hours = 1.0 if 9 <= hour < 18 and is_weekend == 0.0 else 0.0
            # 节假日（从配置）
            holidays = set(self.config.get('calendar', {}).get('holidays', []))
            date_str = start_dt.strftime('%Y-%m-%d')
            is_holiday = 1.0 if date_str in holidays else 0.0
            # 周期编码
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            dow_sin = np.sin(2 * np.pi * dow / 7)
            dow_cos = np.cos(2 * np.pi * dow / 7)

            feats.extend([
                float(hour), float(dow), is_weekend, float(shift_id), is_holiday,
                float(hour_sin), float(hour_cos), float(dow_sin), float(dow_cos), is_business_hours
            ])

            # 能耗积分
            idx_P = feature_columns.index('P_kW') if 'P_kW' in feature_columns else None
            idx_Q = feature_columns.index('Q_kvar') if 'Q_kvar' in feature_columns else None
            idx_S = feature_columns.index('S_kVA') if 'S_kVA' in feature_columns else None

            duration_hours = max(1e-8, (int(meta.end_ts) - int(meta.start_ts)) / 3600.0)
            p_mean = float(np.mean(window[:, idx_P])) if idx_P is not None else 0.0
            q_mean = float(np.mean(window[:, idx_Q])) if idx_Q is not None else 0.0
            s_mean = float(np.mean(window[:, idx_S])) if idx_S is not None else 0.0

            e_active_kwh = p_mean * duration_hours
            e_reactive_kvarh = q_mean * duration_hours
            e_apparent_kvah = s_mean * duration_hours

            # 能效KPI：功率因数代理与能耗效率比
            pf_proxy = p_mean / max(1e-8, np.sqrt(p_mean ** 2 + q_mean ** 2)) if (p_mean != 0.0 or q_mean != 0.0) else 0.0
            energy_eff_ratio = e_active_kwh / max(1e-8, e_apparent_kvah) if e_apparent_kvah > 0 else 0.0

            feats.extend([e_active_kwh, e_reactive_kvarh, pf_proxy, energy_eff_ratio])
        except Exception:
            # 与特征名对齐的占位（时间10项+能耗4项）
            feats.extend([0.0] * 14)
        return np.array(feats)

    def _extract_unbalance_features(self, window: np.ndarray, feature_columns: List[str]) -> np.ndarray:
        """三相不平衡度（若存在V2_V/V3_V或I2_A/I3_A）"""
        vu = 0.0
        iu = 0.0
        try:
            v_cols = [c for c in feature_columns if c.startswith('V') and c.endswith('_V')]
            i_cols = [c for c in feature_columns if c.startswith('I') and c.endswith('_A')]
            # 排除相1，仅保留2/3相
            v_cols = [c for c in v_cols if not c.startswith('V1')]
            i_cols = [c for c in i_cols if not c.startswith('I1')]
            # 仅在存在至少两相时计算
            if len(v_cols) >= 2:
                v_vals = []
                for c in v_cols[:3]:
                    idx = feature_columns.index(c)
                    v_vals.append(float(np.mean(window[:, idx])))
                v_vals = np.array(v_vals)
                if np.mean(v_vals) != 0:
                    vu = float((np.max(v_vals) - np.min(v_vals)) / np.mean(v_vals))
            if len(i_cols) >= 2:
                i_vals = []
                for c in i_cols[:3]:
                    idx = feature_columns.index(c)
                    i_vals.append(float(np.mean(window[:, idx])))
                i_vals = np.array(i_vals)
                if np.mean(i_vals) != 0:
                    iu = float((np.max(i_vals) - np.min(i_vals)) / np.mean(i_vals))
        except Exception:
            pass
        return np.array([vu, iu])
    
    def _get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        feature_names = []
        
        # 基础特征名称（排除相1，仅使用2/3相与中性THD）
        base_features = ['P_kW', 'Q_kvar', 'S_kVA', 'V2_V', 'V3_V', 'I2_A', 'I3_A', 'F_Hz', 'THD_IN_F']
        
        for col_name in base_features:
            # 统计特征
            feature_names.extend([
                f'{col_name}_mean',
                f'{col_name}_std', 
                f'{col_name}_min',
                f'{col_name}_max',
                f'{col_name}_median',
            ])
            
            # 变化率特征
            feature_names.extend([
                f'{col_name}_diff_mean',
                f'{col_name}_diff_std',
                f'{col_name}_total_change',
            ])
            
            # 趋势特征
            feature_names.append(f'{col_name}_slope')
        
        # 进阶派生特征
        feature_names.extend([
            # 功率相关特征
            'power_factor',
            'steady_state_P',
            'steady_state_Q', 
            'steady_state_S',
            'reactive_active_ratio',
            
            # 瞬态波形特征
            'max_rise_slope',
            'max_fall_slope',
            'avg_abs_slope',
            'avg_transient_duration',
            'max_transient_duration',
            
            # 谐波特征（H3-H15）与比例
            'harmonic_3_amp',
            'harmonic_5_amp',
            'harmonic_7_amp',
            'harmonic_9_amp',
            'harmonic_11_amp',
            'harmonic_13_amp',
            'harmonic_15_amp',
            'harmonic_energy_ratio',
            
            # 频域能量与瞬态特征
            'total_energy',
            'low_freq_energy',
            'high_freq_energy',
            'spectral_centroid',
            'stft_transient_energy',
            'cwt_energy',
            
            # 占空比和周期性特征
            'duty_cycle',
            'periodicity_strength',
            
            # 频率与过零/相位特征
            'freq_deviation',
            'freq_stability',
            'zcr_v',
            'zcr_i',
            'phase_shift_proxy',

            # 时间上下文特征
            'hour_of_day', 'day_of_week', 'is_weekend', 'shift_id', 'is_holiday',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_business_hours',

            # 能耗积分与能效KPI
            'E_active_kWh', 'E_reactive_kvarh', 'pf_proxy', 'energy_efficiency_ratio',

            # 三相不平衡指标
            'voltage_unbalance', 'current_unbalance',
            
            # THD特征
            'THD_V_mean',
            'THD_I_mean', 
            'THD_total_mean',
            'THD_V_std',
            'THD_I_std'
        ])
        
        return feature_names
    
    def _get_label_distribution(self, labels: np.ndarray) -> Dict[str, int]:
        unique, counts = np.unique(labels, return_counts=True)
        return {str(k): int(v) for k, v in zip(unique, counts)}
    
    def _save_processed_data(self, processed_data: Dict[str, Any], 
                            labels_data: Dict[str, Any],
                            cv_splits: List[CVFold],
                            segments_meta: List):
        """保存处理后的数据"""
        
        # 保存交叉验证分割
        cv_path = os.path.join(self.output_dir, 'cv_splits.pkl')
        with open(cv_path, 'wb') as f:
            pickle.dump(cv_splits, f)
        
        # 保存标签数据
        labels_path = os.path.join(self.output_dir, 'labels.pkl')
        with open(labels_path, 'wb') as f:
            pickle.dump(labels_data, f)
        
        # 保存段元数据
        segments_df = pd.DataFrame([
            {
                'device_name': meta.device_name,
                'segment_id': meta.segment_id,
                'start_ts': meta.start_ts,
                'end_ts': meta.end_ts,
                'n_rows': meta.n_rows,
                'duration_hours': meta.duration_hours,
                'has_events': meta.has_events,
                'event_density': meta.event_density,
                'max_gap_seconds': meta.max_gap_seconds,
                'filled_ratio': meta.filled_ratio
            }
            for meta in segments_meta
        ])
        segments_df.to_csv(os.path.join(self.output_dir, 'segments_meta.csv'), index=False)

        # 构建并保存设备名称到ID的映射（基于所有窗口元数据）
        all_label_meta = labels_data.get('label_metadata', [])
        try:
            all_device_names = [(
                m['device_name'] if isinstance(m, dict) else getattr(m, 'device_name', None)
            ) for m in all_label_meta]
        except Exception:
            all_device_names = []
        unique_devices = sorted({name for name in all_device_names if name is not None})
        device_name_to_id = {name: idx for idx, name in enumerate(unique_devices)}
        with open(os.path.join(self.output_dir, 'device_name_to_id.json'), 'w') as f:
            json.dump(device_name_to_id, f, ensure_ascii=False, indent=2)

        # 保存每个折的处理数据
        for fold_data in processed_data['folds']:
            fold_idx = fold_data['fold_idx']
            fold_dir = os.path.join(self.output_dir, f'fold_{fold_idx}')
            os.makedirs(fold_dir, exist_ok=True)
            
            # 保存特征数据（仅 .pt）
            import torch
            torch.save(torch.from_numpy(fold_data['train_features']).float(), os.path.join(fold_dir, 'train_features.pt'))
            torch.save(torch.from_numpy(fold_data['val_features']).float(), os.path.join(fold_dir, 'val_features.pt'))
            
            # 保存原始窗口序列（与特征对齐，仅 .pt）
            if 'train_raw' in fold_data and fold_data['train_raw'] is not None and fold_data['train_raw'].size > 0:
                torch.save(torch.from_numpy(fold_data['train_raw']).float(), os.path.join(fold_dir, 'train_raw.pt'))
            if 'val_raw' in fold_data and fold_data['val_raw'] is not None and fold_data['val_raw'].size > 0:
                torch.save(torch.from_numpy(fold_data['val_raw']).float(), os.path.join(fold_dir, 'val_raw.pt'))

            # 保存频域摘要帧（可选，仅 .pt）
            if 'train_freq' in fold_data and fold_data['train_freq'] is not None and np.size(fold_data['train_freq']) > 0:
                torch.save(torch.from_numpy(fold_data['train_freq']).float(), os.path.join(fold_dir, 'train_freq.pt'))
            if 'val_freq' in fold_data and fold_data['val_freq'] is not None and np.size(fold_data['val_freq']) > 0:
                torch.save(torch.from_numpy(fold_data['val_freq']).float(), os.path.join(fold_dir, 'val_freq.pt'))
            
            # 保存索引（仅 .pt）
            torch.save(torch.from_numpy(np.asarray(fold_data['train_indices']).astype(np.int64)), os.path.join(fold_dir, 'train_indices.pt'))
            torch.save(torch.from_numpy(np.asarray(fold_data['val_indices']).astype(np.int64)), os.path.join(fold_dir, 'val_indices.pt'))

            # 保存该折的元数据快照（包含设备名等，便于下游按设备分析）
            train_meta_rows = []
            val_meta_rows = []
            for i in fold_data['train_indices']:
                meta_obj = labels_data['label_metadata'][int(i)]
                # dataclass 或字典均统一转为字典行
                if hasattr(meta_obj, '__dict__'):
                    train_meta_rows.append(asdict(meta_obj))
                elif isinstance(meta_obj, dict):
                    train_meta_rows.append(meta_obj)
                else:
                    train_meta_rows.append({'index': int(i)})
            for i in fold_data['val_indices']:
                meta_obj = labels_data['label_metadata'][int(i)]
                if hasattr(meta_obj, '__dict__'):
                    val_meta_rows.append(asdict(meta_obj))
                elif isinstance(meta_obj, dict):
                    val_meta_rows.append(meta_obj)
                else:
                    val_meta_rows.append({'index': int(i)})
            if train_meta_rows:
                pd.DataFrame(train_meta_rows).to_csv(os.path.join(fold_dir, 'train_metadata.csv'), index=False)
            if val_meta_rows:
                pd.DataFrame(val_meta_rows).to_csv(os.path.join(fold_dir, 'val_metadata.csv'), index=False)

            # 设备ID序列保存已移除，统一 .pt 文件策略，仅保留设备名称映射
            if device_name_to_id:
                pass

            # 保存缩放器
            with open(os.path.join(fold_dir, 'scaler.pkl'), 'wb') as f:
                pickle.dump(fold_data['scaler'], f)

            # 保存特征名称
            with open(os.path.join(fold_dir, 'feature_names.json'), 'w') as f:
                json.dump(fold_data['feature_names'], f)
            
            # 保存原始通道名称
            if 'raw_channel_names' in fold_data:
                with open(os.path.join(fold_dir, 'raw_channel_names.json'), 'w') as f:
                    json.dump(fold_data['raw_channel_names'], f)
        
        logger.info(f"处理后的数据已保存到: {self.output_dir}")
    
    def _save_pipeline_results(self, results: Dict[str, Any]):
        """保存流程结果"""
        results_path = os.path.join(self.output_dir, 'pipeline_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"流程结果已保存到: {results_path}")
    
    def load_processed_data(self, fold_idx: int) -> Dict[str, Any]:
        """
        加载处理后的数据
        
        Args:
            fold_idx: 折索引
            
        Returns:
            loaded_data: 加载的数据
        """
        fold_dir = os.path.join(self.output_dir, f'fold_{fold_idx}')
        
        if not os.path.exists(fold_dir):
            raise FileNotFoundError(f"找不到折 {fold_idx} 的数据: {fold_dir}")
        
        # 加载特征数据（.pt -> numpy）
        import torch
        train_features_t = torch.load(os.path.join(fold_dir, 'train_features.pt'))
        val_features_t = torch.load(os.path.join(fold_dir, 'val_features.pt'))
        train_features = train_features_t.detach().cpu().numpy() if hasattr(train_features_t, 'detach') else train_features_t
        val_features = val_features_t.detach().cpu().numpy() if hasattr(val_features_t, 'detach') else val_features_t
        
        # 加载原始窗口序列（可选，.pt -> numpy）
        train_raw = None
        val_raw = None
        raw_channel_names = []
        train_raw_path_pt = os.path.join(fold_dir, 'train_raw.pt')
        val_raw_path_pt = os.path.join(fold_dir, 'val_raw.pt')
        raw_names_path = os.path.join(fold_dir, 'raw_channel_names.json')
        if os.path.exists(train_raw_path_pt):
            tr = torch.load(train_raw_path_pt)
            train_raw = tr.detach().cpu().numpy() if hasattr(tr, 'detach') else tr
        if os.path.exists(val_raw_path_pt):
            vr = torch.load(val_raw_path_pt)
            val_raw = vr.detach().cpu().numpy() if hasattr(vr, 'detach') else vr
        if os.path.exists(raw_names_path):
            with open(raw_names_path, 'r') as f:
                raw_channel_names = json.load(f)
        
        # 加载频域摘要（可选，.pt -> numpy）
        train_freq = None
        val_freq = None
        train_freq_path = os.path.join(fold_dir, 'train_freq.pt')
        val_freq_path = os.path.join(fold_dir, 'val_freq.pt')
        if os.path.exists(train_freq_path):
            tf = torch.load(train_freq_path)
            train_freq = tf.detach().cpu().numpy() if hasattr(tf, 'detach') else tf
        if os.path.exists(val_freq_path):
            vf = torch.load(val_freq_path)
            val_freq = vf.detach().cpu().numpy() if hasattr(vf, 'detach') else vf
        
        # 加载索引（.pt -> numpy）
        train_indices_t = torch.load(os.path.join(fold_dir, 'train_indices.pt'))
        val_indices_t = torch.load(os.path.join(fold_dir, 'val_indices.pt'))
        train_indices = train_indices_t.detach().cpu().numpy() if hasattr(train_indices_t, 'detach') else train_indices_t
        val_indices = val_indices_t.detach().cpu().numpy() if hasattr(val_indices_t, 'detach') else val_indices_t

        # 加载设备名称映射（不再加载设备ID数组）
        device_map_path = os.path.join(self.output_dir, 'device_name_to_id.json')
        device_name_to_id = {}
        if os.path.exists(device_map_path):
            with open(device_map_path, 'r') as f:
                device_name_to_id = json.load(f)
        train_device_ids = None
        val_device_ids = None
        
        # 加载缩放器
        with open(os.path.join(fold_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        
        # 加载特征名称
        with open(os.path.join(fold_dir, 'feature_names.json'), 'r') as f:
            feature_names = json.load(f)
        
        # 加载标签数据
        with open(os.path.join(self.output_dir, 'labels.pkl'), 'rb') as f:
            labels_data = pickle.load(f)
        
        train_labels = labels_data['labels'][train_indices]
        val_labels = labels_data['labels'][val_indices]
        
        return {
            'train_features': train_features,
            'val_features': val_features,
            'train_raw': train_raw,
            'val_raw': val_raw,
            'train_freq': train_freq,
            'val_freq': val_freq,
            'train_labels': train_labels,
            'val_labels': val_labels,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'train_device_ids': train_device_ids,
            'val_device_ids': val_device_ids,
            'device_name_to_id': device_name_to_id,
            'scaler': scaler,
            'feature_names': feature_names,
            'raw_channel_names': raw_channel_names
        }
    
    def evaluate_fold(self, fold_idx: int, y_pred: np.ndarray, 
                     y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        评估单个折的结果
        
        Args:
            fold_idx: 折索引
            y_pred: 预测标签
            y_prob: 预测概率
            
        Returns:
            evaluation_results: 评估结果
        """
        # 加载验证集真实标签
        fold_data = self.load_processed_data(fold_idx)
        y_true = fold_data['val_labels']
        
        # 加载元数据
        with open(os.path.join(self.output_dir, 'labels.pkl'), 'rb') as f:
            labels_data = pickle.load(f)
        
        val_metadata = pd.DataFrame([labels_data['label_metadata'][i] for i in fold_data['val_indices']])
        
        # 评估
        evaluation_results = self.evaluator.evaluate_predictions(
            y_true, y_pred, y_prob, val_metadata
        )
        
        # 生成报告
        report_path = os.path.join(self.output_dir, f'fold_{fold_idx}', 'evaluation_report.html')
        self.evaluator.generate_evaluation_report(evaluation_results, report_path, fold_idx)
        
        # 生成可视化
        plots_dir = os.path.join(self.output_dir, f'fold_{fold_idx}', 'plots')
        self.evaluator.create_visualization_plots(y_true, y_pred, y_prob, val_metadata, plots_dir)
        
        return evaluation_results
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """获取流程摘要"""
        results_path = os.path.join(self.output_dir, 'pipeline_results.json')
        
        if not os.path.exists(results_path):
            return {"error": "流程结果文件不存在"}
        
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # 加载段元数据
        segments_meta_path = os.path.join(self.output_dir, 'segments_meta.csv')
        if os.path.exists(segments_meta_path):
            segments_meta = pd.read_csv(segments_meta_path)
            results['segments_summary'] = {
                'total_segments': len(segments_meta),
                'total_samples': segments_meta['n_rows'].sum(),
                'devices': segments_meta['device_name'].unique().tolist(),
                'time_range': {
                    'start': segments_meta['start_ts'].min(),
                    'end': segments_meta['end_ts'].max()
                }
            }
        
        return results


def main():
    """主函数示例"""
    import argparse
    
    parser = argparse.ArgumentParser(description='运行数据准备流程')
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument('--data', required=True, help='输入数据路径')
    parser.add_argument('--output', help='输出目录（可选，会覆盖配置文件中的设置）')
    
    args = parser.parse_args()
    
    # 创建流程实例
    pipeline = DataPreparationPipeline(args.config)
    
    # 如果指定了输出目录，更新配置
    if args.output:
        pipeline.output_dir = args.output
        pipeline.config['data_storage']['output_directory'] = args.output
        os.makedirs(args.output, exist_ok=True)
    
    # 运行流程
    try:
        results = pipeline.run_full_pipeline(args.data)
        print("数据准备流程运行成功！")
        print(f"结果保存在: {pipeline.output_dir}")
        
        # 打印摘要
        summary = pipeline.get_pipeline_summary()
        print("\n流程摘要:")
        print(f"- 状态: {summary.get('status', 'unknown')}")
        if 'segments_summary' in summary:
            seg_summary = summary['segments_summary']
            print(f"- 总段数: {seg_summary['total_segments']}")
            print(f"- 总样本数: {seg_summary['total_samples']}")
            print(f"- 设备数: {len(seg_summary['devices'])}")
        
    except Exception as e:
        print(f"数据准备流程运行失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())