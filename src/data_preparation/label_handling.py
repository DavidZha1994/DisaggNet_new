"""
标签处理模块
Label Handling Module

实现时序数据的标签处理，包括：
- 事件定义和检测
- 标签生成策略
- 不平衡数据处理
- 采样策略
"""

import numpy as np
import polars as pl
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from sklearn.utils import resample
from collections import Counter
import logging
from .evaluation import EventEvaluator

logger = logging.getLogger(__name__)

@dataclass
class EventDefinition:
    """事件定义"""
    event_type: str
    feature_column: str
    threshold: float
    direction: str  # 'up', 'down', 'both'
    min_duration: int  # 最小持续时间（样本数）
    max_gap: int  # 最大间隔（样本数）

@dataclass
class LabelConfig:
    """标签配置"""
    horizon: int  # 预测视野（步数）
    label_type: str  # 'binary', 'multiclass', 'regression'
    aggregation: str  # 'any', 'count', 'max', 'mean'
    causality_check: bool  # 是否检查因果性

@dataclass
class OnOffConfig:
    """开关状态标签配置"""
    enabled: bool = False
    feature_column: str = "P_kW"  # 观测功率/电压/电流列名
    method: str = "absolute"      # 'absolute'|'delta'|'hybrid'
    on_threshold: float = 0.5      # 上电阈值（绝对法）
    off_threshold: float = 0.3     # 断电阈值（绝对法）
    hysteresis_margin: float = 0.0 # 迟滞边界增强（可选）
    min_on_duration: int = 5       # 连续样本达到上电判定的最小长度
    min_off_duration: int = 5      # 连续样本达到断电判定的最小长度
    on_ratio_threshold: float = 0.5 # 窗口内判为“开启”的比例阈值
    smooth_window: int = 1         # 平滑窗口长度（>1时进行简单移动平均）
    per_device: Optional[Dict[str, Dict[str, float]]] = None  # 每设备阈值

class LabelHandler:
    """标签处理器"""
    
    def __init__(self, config: Dict):
        self.event_definitions = self._parse_event_definitions(config['labels']['event_definitions'])
        self.label_config = LabelConfig(**config['labels']['label_config'])
        self.imbalance_handling = config['labels'].get('imbalance_handling', {'enabled': False, 'strategy': 'mixed'})
        self.onoff_config = self._parse_onoff_config(config['labels'].get('onoff', {}))
        self.sampling_strategy = config['labels']['sampling_strategy']
        self.config = config
        
    def _parse_event_definitions(self, event_defs: List[Dict]) -> List[EventDefinition]:
        """解析事件定义"""
        definitions = []
        for event_def in event_defs:
            definitions.append(EventDefinition(**event_def))
        return definitions

    def _parse_onoff_config(self, onoff_cfg: Dict[str, Any]) -> OnOffConfig:
        """解析开关状态配置"""
        if not onoff_cfg:
            return OnOffConfig()  # 默认禁用
        # 兼容配置中缺失字段
        defaults = OnOffConfig().__dict__.copy()
        merged = {**defaults, **onoff_cfg}
        return OnOffConfig(**merged)
    
    def detect_events(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        检测数据中的事件
        
        Args:
            df: 输入数据
            
        Returns:
            df_with_events: 带事件标记的数据
        """
        logger.info("开始事件检测")
        
        # 复制数据框
        df_events = df.clone()
        
        # 为每种事件类型添加检测列
        for event_def in self.event_definitions:
            event_column = f"event_{event_def.event_type}"
            
            # 检测事件
            events = self._detect_single_event_type(df, event_def)
            df_events = df_events.with_columns(pl.Series(event_column, events))
            
            event_count = np.sum(events)
            logger.info(f"检测到 {event_def.event_type} 事件: {event_count} 个")
        
        return df_events

    # === 开关状态生成（绝对阈值/跃变/混合） ===
    def _smooth_series(self, x: np.ndarray, k: int) -> np.ndarray:
        if k is None or k <= 1:
            return x
        k = int(k)
        if k > len(x):
            k = len(x)
        kernel = np.ones(k, dtype=float) / float(k)
        return np.convolve(x, kernel, mode='same')

    def _compute_onoff_state_absolute(self, x: np.ndarray, on_thr: float, off_thr: float,
                                      min_on: int, min_off: int) -> np.ndarray:
        """基于绝对阈值并带滞回与最小持续的开关状态计算。"""
        state = np.zeros_like(x, dtype=bool)
        cur = False
        on_run = 0
        off_run = 0
        for i, v in enumerate(x):
            if not cur:
                if v >= on_thr:
                    on_run += 1
                    if on_run >= min_on:
                        cur = True
                        on_run = 0
                        off_run = 0
                else:
                    on_run = 0
            else:
                if v <= off_thr:
                    off_run += 1
                    if off_run >= min_off:
                        cur = False
                        off_run = 0
                        on_run = 0
                else:
                    off_run = 0
            state[i] = cur
        return state

    def _compute_onoff_state_delta(self, x: np.ndarray, up_delta: float, down_delta: float,
                                   min_on: int, min_off: int) -> np.ndarray:
        """基于跃变检测的开关状态（上升/下降突变）。"""
        state = np.zeros_like(x, dtype=bool)
        cur = False
        diff = np.diff(x, prepend=x[0])
        on_run = 0
        off_run = 0
        for i, d in enumerate(diff):
            if not cur:
                if d >= up_delta:
                    on_run += 1
                    if on_run >= min_on:
                        cur = True
                        on_run = 0
                        off_run = 0
                else:
                    on_run = 0
            else:
                if d <= -down_delta:
                    off_run += 1
                    if off_run >= min_off:
                        cur = False
                        off_run = 0
                        on_run = 0
                else:
                    off_run = 0
            state[i] = cur
        return state

    def generate_onoff_labels_from_windows(self,
                                           windows: List[np.ndarray],
                                           feature_columns: List[str],
                                           window_metadata: List) -> np.ndarray:
        """基于窗口内的功率/电压/电流生成开关状态标签。
        - 支持绝对阈值、跃变与混合策略；
        - 支持每设备阈值；
        - 标签为窗口内“开启”占比超过 on_ratio_threshold 的二值。
        """
        cfg = self.onoff_config
        if not cfg.enabled:
            logger.info("on/off 标签未启用，回退事件标签")
            return None

        if cfg.feature_column not in feature_columns:
            logger.warning(f"开关观测列 {cfg.feature_column} 不存在，回退事件标签")
            return None

        col_idx = feature_columns.index(cfg.feature_column)

        labels: List[int] = []
        for i, window in enumerate(windows):
            x = window[:, col_idx].astype(float)
            # 平滑
            x_proc = self._smooth_series(np.nan_to_num(x, nan=0.0), cfg.smooth_window)

            # 每设备阈值覆盖
            on_thr = cfg.on_threshold
            off_thr = cfg.off_threshold
            meta = window_metadata[i]
            dev_name = getattr(meta, 'device_name', None) if hasattr(meta, 'device_name') else (
                meta.get('device_name') if isinstance(meta, dict) else None
            )
            if cfg.per_device and dev_name and dev_name in cfg.per_device:
                pd_cfg = cfg.per_device[dev_name]
                on_thr = float(pd_cfg.get('on_threshold', on_thr))
                off_thr = float(pd_cfg.get('off_threshold', off_thr))

            # 迟滞边界增强
            on_thr_eff = on_thr + max(0.0, cfg.hysteresis_margin)
            off_thr_eff = off_thr - max(0.0, cfg.hysteresis_margin)

            # 状态序列
            if cfg.method == 'absolute':
                state = self._compute_onoff_state_absolute(x_proc, on_thr_eff, off_thr_eff,
                                                           cfg.min_on_duration, cfg.min_off_duration)
            elif cfg.method == 'delta':
                state = self._compute_onoff_state_delta(x_proc, on_thr_eff, off_thr_eff,
                                                        cfg.min_on_duration, cfg.min_off_duration)
            else:  # hybrid：先绝对，后用跃变修正边缘
                state_abs = self._compute_onoff_state_absolute(x_proc, on_thr_eff, off_thr_eff,
                                                               cfg.min_on_duration, cfg.min_off_duration)
                state_delta = self._compute_onoff_state_delta(x_proc, on_thr_eff, off_thr_eff,
                                                              cfg.min_on_duration, cfg.min_off_duration)
                state = np.where(state_delta, True, state_abs)

            # 窗口标签：开启占比是否超过阈值
            on_ratio = float(np.mean(state)) if state.size > 0 else 0.0
            labels.append(1 if on_ratio >= cfg.on_ratio_threshold else 0)

        y = np.asarray(labels, dtype=np.int64)
        logger.info(f"on/off 标签分布: {dict(Counter(y))}")
        return y
    
    def _detect_single_event_type(self, df: pl.DataFrame, event_def: EventDefinition) -> np.ndarray:
        """
        检测单一类型事件
        
        Args:
            df: 输入数据
            event_def: 事件定义
            
        Returns:
            events: 事件标记数组
        """
        if event_def.feature_column not in df.columns:
            logger.warning(f"特征列 {event_def.feature_column} 不存在")
            return np.zeros(len(df), dtype=bool)
        
        # 获取特征数据
        feature_data = df.select(event_def.feature_column).to_numpy().flatten()
        
        # 计算变化
        diff_data = np.diff(feature_data, prepend=feature_data[0])
        
        # 根据方向检测事件
        if event_def.direction == 'up':
            raw_events = diff_data > event_def.threshold
        elif event_def.direction == 'down':
            raw_events = diff_data < -event_def.threshold
        elif event_def.direction == 'both':
            raw_events = np.abs(diff_data) > event_def.threshold
        else:
            raise ValueError(f"未知的事件方向: {event_def.direction}")
        
        # 应用持续时间和间隔过滤
        filtered_events = self._filter_events_by_duration(
            raw_events, event_def.min_duration, event_def.max_gap
        )
        
        return filtered_events
    
    def _filter_events_by_duration(self, events: np.ndarray, min_duration: int, max_gap: int) -> np.ndarray:
        """
        根据持续时间和间隔过滤事件
        
        Args:
            events: 原始事件标记
            min_duration: 最小持续时间
            max_gap: 最大间隔
            
        Returns:
            filtered_events: 过滤后的事件标记
        """
        if min_duration <= 1 and max_gap <= 0:
            return events
        
        filtered_events = np.zeros_like(events, dtype=bool)
        
        # 找到事件区间
        event_starts = np.where(np.diff(events.astype(int), prepend=0) == 1)[0]
        event_ends = np.where(np.diff(events.astype(int), append=0) == -1)[0]
        
        for start, end in zip(event_starts, event_ends):
            duration = end - start + 1
            
            # 检查持续时间
            if duration >= min_duration:
                filtered_events[start:end+1] = True
        
        # 合并间隔较小的事件
        if max_gap > 0:
            filtered_events = self._merge_close_events(filtered_events, max_gap)
        
        return filtered_events
    
    def _merge_close_events(self, events: np.ndarray, max_gap: int) -> np.ndarray:
        """
        合并间隔较小的事件
        
        Args:
            events: 事件标记
            max_gap: 最大间隔
            
        Returns:
            merged_events: 合并后的事件标记
        """
        merged_events = events.copy()
        
        # 找到事件区间
        event_starts = np.where(np.diff(events.astype(int), prepend=0) == 1)[0]
        event_ends = np.where(np.diff(events.astype(int), append=0) == -1)[0]
        
        if len(event_starts) <= 1:
            return merged_events
        
        # 检查相邻事件间隔
        for i in range(len(event_starts) - 1):
            gap = event_starts[i + 1] - event_ends[i] - 1
            
            if gap <= max_gap:
                # 填充间隔
                merged_events[event_ends[i]+1:event_starts[i+1]] = True
        
        return merged_events
    
    def generate_labels(self, df_with_events: pl.DataFrame, window_metadata: List) -> np.ndarray:
        """
        为窗口生成标签
        
        Args:
            df_with_events: 带事件标记的数据
            window_metadata: 窗口元数据列表
            
        Returns:
            labels: 窗口标签数组
        """
        logger.info("开始生成窗口标签")
        
        labels = []
        
        # 获取事件列
        event_columns = [col for col in df_with_events.columns if col.startswith('event_')]
        
        if not event_columns:
            logger.warning("未找到事件列，返回零标签")
            return np.zeros(len(window_metadata))
        
        # 转换为pandas以便索引
        df_pd = df_with_events.to_pandas()
        
        for window_meta in window_metadata:
            # 获取窗口对应的数据索引范围
            window_start_idx = window_meta.start_idx
            window_end_idx = window_meta.end_idx
            
            # 根据标签配置生成标签
            if self.label_config.label_type == 'binary':
                label = self._generate_binary_label(
                    df_pd, window_start_idx, window_end_idx, event_columns
                )
            elif self.label_config.label_type == 'multiclass':
                label = self._generate_multiclass_label(
                    df_pd, window_start_idx, window_end_idx, event_columns
                )
            elif self.label_config.label_type == 'regression':
                label = self._generate_regression_label(
                    df_pd, window_start_idx, window_end_idx, event_columns
                )
            else:
                raise ValueError(f"未知的标签类型: {self.label_config.label_type}")
            
            labels.append(label)
        
        labels = np.array(labels)
        
        # 统计标签分布
        if self.label_config.label_type in ['binary', 'multiclass']:
            label_counts = Counter(labels)
            logger.info(f"标签分布: {dict(label_counts)}")
        
        return labels
    
    def _generate_binary_label(self, df: pd.DataFrame, start_idx: int, end_idx: int, event_columns: List[str]) -> int:
        """
        生成二分类标签
        
        Args:
            df: 数据框
            start_idx: 窗口开始索引
            end_idx: 窗口结束索引
            event_columns: 事件列名
            
        Returns:
            label: 二分类标签 (0 或 1)
        """
        # 考虑预测视野
        label_start_idx = end_idx
        label_end_idx = min(end_idx + self.label_config.horizon, len(df))
        
        # 检查因果性
        if self.label_config.causality_check and label_end_idx > len(df):
            return 0  # 无法获取未来标签
        
        # 获取标签窗口内的事件
        label_window = df.iloc[label_start_idx:label_end_idx]
        
        # 聚合事件
        if self.label_config.aggregation == 'any':
            has_event = label_window[event_columns].any().any()
            return int(has_event)
        elif self.label_config.aggregation == 'count':
            event_count = label_window[event_columns].sum().sum()
            return int(event_count > 0)
        else:
            # 默认使用any
            has_event = label_window[event_columns].any().any()
            return int(has_event)
    
    def _generate_multiclass_label(self, df: pd.DataFrame, start_idx: int, end_idx: int, event_columns: List[str]) -> int:
        """
        生成多分类标签
        
        Args:
            df: 数据框
            start_idx: 窗口开始索引
            end_idx: 窗口结束索引
            event_columns: 事件列名
            
        Returns:
            label: 多分类标签
        """
        # 考虑预测视野
        label_start_idx = end_idx
        label_end_idx = min(end_idx + self.label_config.horizon, len(df))
        
        # 获取标签窗口内的事件
        label_window = df.iloc[label_start_idx:label_end_idx]
        
        # 找到最主要的事件类型
        event_counts = label_window[event_columns].sum()
        
        if event_counts.sum() == 0:
            return 0  # 无事件
        
        # 返回事件数量最多的类型（+1因为0表示无事件）
        return event_counts.idxmax().split('_')[-1] if isinstance(event_counts.idxmax(), str) else 1
    
    def _generate_regression_label(self, df: pd.DataFrame, start_idx: int, end_idx: int, event_columns: List[str]) -> float:
        """
        生成回归标签
        
        Args:
            df: 数据框
            start_idx: 窗口开始索引
            end_idx: 窗口结束索引
            event_columns: 事件列名
            
        Returns:
            label: 回归标签
        """
        # 考虑预测视野
        label_start_idx = end_idx
        label_end_idx = min(end_idx + self.label_config.horizon, len(df))
        
        # 获取标签窗口内的事件
        label_window = df.iloc[label_start_idx:label_end_idx]
        
        # 计算事件强度
        if self.label_config.aggregation == 'count':
            return float(label_window[event_columns].sum().sum())
        elif self.label_config.aggregation == 'mean':
            return float(label_window[event_columns].mean().mean())
        elif self.label_config.aggregation == 'max':
            return float(label_window[event_columns].max().max())
        else:
            return float(label_window[event_columns].sum().sum())
    
    def handle_imbalanced_labels(self, X: np.ndarray, y: np.ndarray, metadata: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        处理不平衡标签
        
        Args:
            X: 特征数据
            y: 标签数据
            metadata: 元数据
            
        Returns:
            X_balanced: 平衡后的特征数据
            y_balanced: 平衡后的标签数据
            metadata_balanced: 平衡后的元数据
        """
        if not self.imbalance_handling.get('enabled', False):
            return X, y, metadata
        
        logger.info("开始处理不平衡标签")
        
        strategy = self.imbalance_handling['strategy']
        
        if strategy == 'oversample':
            return self._oversample(X, y, metadata)
        elif strategy == 'undersample':
            return self._undersample(X, y, metadata)
        elif strategy == 'mixed':
            return self._mixed_sampling(X, y, metadata)
        elif strategy == 'segment_weighted':
            return self._segment_weighted_sampling(X, y, metadata)
        else:
            logger.warning(f"未知的采样策略: {strategy}")
            return X, y, metadata
    
    def _oversample(self, X: np.ndarray, y: np.ndarray, metadata: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """过采样少数类"""
        target_ratio = self.imbalance_handling.get('target_ratio', 0.3)
        
        # 统计类别分布
        unique_labels, counts = np.unique(y, return_counts=True)
        majority_count = counts.max()
        
        X_resampled = []
        y_resampled = []
        metadata_resampled = []
        
        for label in unique_labels:
            label_mask = y == label
            label_count = np.sum(label_mask)
            
            # 计算目标数量
            if label_count == majority_count:
                target_count = label_count  # 多数类保持不变
            else:
                target_count = int(majority_count * target_ratio)
            
            # 重采样
            if target_count > label_count:
                # 过采样
                indices = np.where(label_mask)[0]
                resampled_indices = resample(indices, n_samples=target_count, random_state=42)
            else:
                # 保持原样
                resampled_indices = np.where(label_mask)[0]
            
            X_resampled.append(X[resampled_indices])
            y_resampled.append(y[resampled_indices])
            metadata_resampled.append(metadata.iloc[resampled_indices])
        
        # 合并结果
        X_balanced = np.vstack(X_resampled)
        y_balanced = np.hstack(y_resampled)
        metadata_balanced = pd.concat(metadata_resampled, ignore_index=True)
        
        logger.info(f"过采样完成: {len(y)} -> {len(y_balanced)} 样本")
        return X_balanced, y_balanced, metadata_balanced
    
    def _undersample(self, X: np.ndarray, y: np.ndarray, metadata: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """欠采样多数类"""
        target_ratio = self.imbalance_handling.get('target_ratio', 0.3)
        
        # 统计类别分布
        unique_labels, counts = np.unique(y, return_counts=True)
        minority_count = counts.min()
        
        X_resampled = []
        y_resampled = []
        metadata_resampled = []
        
        for label in unique_labels:
            label_mask = y == label
            label_count = np.sum(label_mask)
            
            # 计算目标数量
            if label_count == minority_count:
                target_count = label_count  # 少数类保持不变
            else:
                target_count = int(minority_count / target_ratio)
            
            # 重采样
            indices = np.where(label_mask)[0]
            if target_count < label_count:
                # 欠采样
                resampled_indices = resample(indices, n_samples=target_count, random_state=42, replace=False)
            else:
                # 保持原样
                resampled_indices = indices
            
            X_resampled.append(X[resampled_indices])
            y_resampled.append(y[resampled_indices])
            metadata_resampled.append(metadata.iloc[resampled_indices])
        
        # 合并结果
        X_balanced = np.vstack(X_resampled)
        y_balanced = np.hstack(y_resampled)
        metadata_balanced = pd.concat(metadata_resampled, ignore_index=True)
        
        logger.info(f"欠采样完成: {len(y)} -> {len(y_balanced)} 样本")
        return X_balanced, y_balanced, metadata_balanced
    
    def _mixed_sampling(self, X: np.ndarray, y: np.ndarray, metadata: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """混合采样策略"""
        # 先进行适度的过采样，再进行适度的欠采样
        X_over, y_over, meta_over = self._oversample(X, y, metadata)
        X_balanced, y_balanced, metadata_balanced = self._undersample(X_over, y_over, meta_over)
        
        logger.info(f"混合采样完成: {len(y)} -> {len(y_balanced)} 样本")
        return X_balanced, y_balanced, metadata_balanced
    
    def _segment_weighted_sampling(self, X: np.ndarray, y: np.ndarray, metadata: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """基于段的加权采样"""
        # 计算每个段的事件密度
        segment_event_density = metadata.groupby('segment_id')['has_events'].mean()
        
        # 根据事件密度调整采样权重
        sampling_weights = []
        for _, row in metadata.iterrows():
            segment_id = row['segment_id']
            density = segment_event_density[segment_id]
            
            # 事件密度高的段获得更高权重
            weight = 1.0 + density * 2.0
            sampling_weights.append(weight)
        
        sampling_weights = np.array(sampling_weights)
        sampling_weights = sampling_weights / sampling_weights.sum()
        
        # 根据权重重采样
        n_samples = len(X)
        resampled_indices = np.random.choice(
            n_samples, size=n_samples, replace=True, p=sampling_weights
        )
        
        X_balanced = X[resampled_indices]
        y_balanced = y[resampled_indices]
        metadata_balanced = metadata.iloc[resampled_indices].reset_index(drop=True)
        
        logger.info(f"段加权采样完成: 保持 {len(y_balanced)} 样本")
        return X_balanced, y_balanced, metadata_balanced

    def tune_thresholds(self, df: Union[pl.DataFrame, pd.DataFrame], strategy: str = 'percentile', p: float = 0.95, per_device: bool = False) -> None:
        """根据数据自适应调整事件阈值。
        - strategy: 'percentile' 或 'std'（以均值±kσ）
        - p: 分位点或标准差系数
        - per_device: 是否按设备名称分组自适应
        """
        if df is None:
            return
        df_pd = df.to_pandas() if isinstance(df, pl.DataFrame) else df
        groups = [df_pd]
        device_col = 'device_name' if 'device_name' in df_pd.columns else None
        if per_device and device_col:
            groups = [g for _, g in df_pd.groupby(device_col)]
        for group in groups:
            for ed in self.event_definitions:
                col = ed.feature_column
                if col not in group.columns:
                    continue
                series = pd.to_numeric(group[col], errors='coerce')
                if strategy == 'percentile':
                    if ed.direction in ('up', 'above'):
                        thr = float(np.nanpercentile(series.values, p * 100.0))
                    elif ed.direction in ('down', 'below'):
                        thr = float(np.nanpercentile(series.values, (1.0 - p) * 100.0))
                    else:  # both
                        upper = float(np.nanpercentile(series.values, p * 100.0))
                        lower = float(np.nanpercentile(series.values, (1.0 - p) * 100.0))
                        thr = max(abs(upper), abs(lower))
                else:  # 'std'
                    mu = float(np.nanmean(series.values))
                    sigma = float(np.nanstd(series.values))
                    k = p
                    if ed.direction in ('up', 'above'):
                        thr = mu + k * sigma
                    elif ed.direction in ('down', 'below'):
                        thr = mu - k * sigma
                    else:
                        thr = k * sigma
                ed.threshold = thr

    def evaluate_events(self, y_true: np.ndarray, y_pred: np.ndarray, metadata: pd.DataFrame) -> Dict[str, Any]:
        """事件级评估：返回事件级精确率/召回率/F1及错误分析。"""
        evaluator = EventEvaluator(self.config)
        results = evaluator.evaluate_predictions(y_true, y_pred, y_prob=None, metadata=metadata)
        return results.get('event_metrics', {})
    
    def get_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        计算类别权重
        
        Args:
            y: 标签数组
            
        Returns:
            class_weights: 类别权重字典
        """
        unique_labels, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        n_classes = len(unique_labels)
        
        class_weights = {}
        for label, count in zip(unique_labels, counts):
            weight = total_samples / (n_classes * count)
            class_weights[int(label)] = weight
        
        logger.info(f"类别权重: {class_weights}")
        return class_weights