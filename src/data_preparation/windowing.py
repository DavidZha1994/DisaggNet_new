"""
窗口化模块
Windowing Module

实现时序数据的窗口化策略，包括：
- 滑动窗口生成
- 重叠控制
- 窗口质量过滤
- 多尺度窗口支持
"""

import numpy as np
import polars as pl
import pandas as pd
from typing import List, Dict, Tuple, Iterator, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class WindowMetadata:
    """窗口元数据"""
    device_name: str
    segment_id: int
    window_id: int
    start_idx: int
    end_idx: int
    start_ts: int
    end_ts: int
    window_length: int
    filled_ratio: float
    has_events: bool
    event_count: int

class WindowGenerator:
    """窗口生成器"""
    
    def __init__(self, config: Dict):
        self.window_lengths = config['windowing']['window_lengths']
        self.step_sizes = config['windowing']['step_sizes']
        self.default_window_length = config['windowing']['default_window_length']
        self.default_step_size = config['windowing']['default_step_size']
        self.max_filled_ratio = config['windowing']['max_filled_ratio']
        self.overlap_within_set = config['windowing']['overlap_within_set']
        self.overlap_across_sets = config['windowing']['overlap_across_sets']
        
    def generate_windows(self, df: pl.DataFrame, device_name: str, segment_id: int,
                        window_length: Optional[int] = None, 
                        step_size: Optional[int] = None) -> Tuple[List[np.ndarray], List[WindowMetadata], List[str]]:
        """
        为单个段生成窗口
        
        Args:
            df: 段数据
            device_name: 设备名称
            segment_id: 段ID
            window_length: 窗口长度
            step_size: 步长
            
        Returns:
            windows: 窗口数据列表
            metadata: 窗口元数据列表
        """
        if window_length is None:
            window_length = self.default_window_length
        if step_size is None:
            step_size = self.default_step_size
            
        logger.debug(f"生成窗口: 设备={device_name}, 段={segment_id}, 长度={window_length}, 步长={step_size}")
        
        # 获取数值特征列
        feature_columns = self._get_feature_columns(df)
        
        # 转换为numpy数组以提高性能
        data_array = df.select(feature_columns).to_numpy()
        ts_array = df.select("ts_utc").to_numpy().flatten()
        
        # 确保mask_filled不包含None值
        if "mask_filled" in df.columns:
            filled_mask = df.select("mask_filled").to_numpy().flatten()
            # 将None值转换为False
            filled_mask = np.where(filled_mask == None, False, filled_mask).astype(bool)
        else:
            filled_mask = np.zeros(len(df), dtype=bool)
        
        windows = []
        metadata = []
        window_id = 0
        
        # 滑动窗口生成
        for start_idx in range(0, len(data_array) - window_length + 1, step_size):
            end_idx = start_idx + window_length
            
            # 提取窗口数据
            window_data = data_array[start_idx:end_idx]
            window_filled = filled_mask[start_idx:end_idx]
            
            # 计算窗口质量指标
            filled_ratio = np.mean(window_filled)
            
            # 过滤低质量窗口
            if filled_ratio > self.max_filled_ratio:
                logger.debug(f"跳过低质量窗口: filled_ratio={filled_ratio:.3f}")
                continue
            
            # 检测窗口内事件（简化处理）
            has_events, event_count = self._detect_window_events(window_data, feature_columns)
            
            # 创建窗口元数据
            window_meta = WindowMetadata(
                device_name=device_name,
                segment_id=segment_id,
                window_id=window_id,
                start_idx=start_idx,
                end_idx=end_idx,
                start_ts=int(ts_array[start_idx]),
                end_ts=int(ts_array[end_idx-1]),
                window_length=window_length,
                filled_ratio=filled_ratio,
                has_events=has_events,
                event_count=event_count
            )
            
            windows.append(window_data)
            metadata.append(window_meta)
            window_id += 1
        
        logger.info(f"设备 {device_name} 段 {segment_id} 生成 {len(windows)} 个窗口")
        return windows, metadata, feature_columns
    
    def generate_multiscale_windows(self, df: pl.DataFrame, device_name: str, segment_id: int) -> Dict[str, Tuple[List[np.ndarray], List[WindowMetadata]]]:
        """
        生成多尺度窗口
        
        Args:
            df: 段数据
            device_name: 设备名称
            segment_id: 段ID
            
        Returns:
            multiscale_windows: 多尺度窗口字典
        """
        multiscale_windows = {}
        
        for window_length in self.window_lengths:
            for step_size in self.step_sizes:
                scale_key = f"L{window_length}_S{step_size}"
                
                windows, metadata = self.generate_windows(
                    df, device_name, segment_id, window_length, step_size
                )
                
                if windows:  # 只保存非空结果
                    multiscale_windows[scale_key] = (windows, metadata)
        
        return multiscale_windows
    
    def _get_feature_columns(self, df: pl.DataFrame) -> List[str]:
        """
        获取特征列名
        
        Args:
            df: 数据框
            
        Returns:
            feature_columns: 特征列名列表
        """
        # 排除非特征列
        exclude_columns = {'ts_utc', 't_rel', 'mask_filled', 'device_id'}
        all_columns = set(df.columns)
        feature_columns = list(all_columns - exclude_columns)
        
        # 确保数值列
        numeric_columns = []
        for col in feature_columns:
            col_dtype = df.select(pl.col(col)).dtypes[0]
            if col_dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                numeric_columns.append(col)
        
        return numeric_columns
    
    def _detect_window_events(self, window_data: np.ndarray, feature_columns: List[str]) -> Tuple[bool, int]:
        """
        检测窗口内的事件
        
        Args:
            window_data: 窗口数据
            feature_columns: 特征列名
            
        Returns:
            has_events: 是否有事件
            event_count: 事件数量
        """
        # 简化的事件检测逻辑
        # 实际应用中应该根据具体业务逻辑实现
        
        has_events = False
        event_count = 0
        
        # 检测功率变化事件
        if 'P_kW' in feature_columns:
            try:
                power_idx = feature_columns.index('P_kW')
                power_data = window_data[:, power_idx]
                
                # 计算功率变化
                power_diff = np.diff(power_data)
                large_changes = np.abs(power_diff) > 1.0  # 1kW阈值
                
                event_count = int(np.sum(large_changes))
                has_events = bool(event_count > 0)
            except (IndexError, ValueError) as e:
                logger.warning(f"功率事件检测失败: {e}")
                has_events = False
                event_count = 0
        
        # 确保返回值类型正确
        return bool(has_events), int(event_count)
    
    def filter_windows_by_quality(self, windows: List[np.ndarray], 
                                 metadata: List[WindowMetadata],
                                 min_event_ratio: float = 0.0,
                                 max_filled_ratio: Optional[float] = None) -> Tuple[List[np.ndarray], List[WindowMetadata]]:
        """
        根据质量指标过滤窗口
        
        Args:
            windows: 窗口数据列表
            metadata: 窗口元数据列表
            min_event_ratio: 最小事件比例
            max_filled_ratio: 最大填充比例
            
        Returns:
            filtered_windows: 过滤后的窗口
            filtered_metadata: 过滤后的元数据
        """
        if max_filled_ratio is None:
            max_filled_ratio = self.max_filled_ratio
        
        filtered_windows = []
        filtered_metadata = []
        
        for window, meta in zip(windows, metadata):
            # 质量检查
            if meta.filled_ratio > max_filled_ratio:
                continue
            
            # 事件密度检查
            event_ratio = meta.event_count / meta.window_length
            if event_ratio < min_event_ratio:
                continue
            
            filtered_windows.append(window)
            filtered_metadata.append(meta)
        
        logger.info(f"质量过滤: {len(windows)} -> {len(filtered_windows)} 个窗口")
        return filtered_windows, filtered_metadata
    
    def create_window_dataset(self, windows: List[np.ndarray], 
                             metadata: List[WindowMetadata],
                             labels: Optional[List[int]] = None) -> Dict:
        """
        创建窗口数据集
        
        Args:
            windows: 窗口数据列表
            metadata: 窗口元数据列表
            labels: 标签列表
            
        Returns:
            dataset: 窗口数据集字典
        """
        if not windows:
            return {}
        
        # 堆叠窗口数据
        X = np.stack(windows)  # Shape: (n_windows, window_length, n_features)
        
        # 创建元数据DataFrame
        meta_data = []
        for meta in metadata:
            meta_data.append({
                'device_name': meta.device_name,
                'segment_id': meta.segment_id,
                'window_id': meta.window_id,
                'start_ts': meta.start_ts,
                'end_ts': meta.end_ts,
                'window_length': meta.window_length,
                'filled_ratio': meta.filled_ratio,
                'has_events': meta.has_events,
                'event_count': meta.event_count
            })
        
        meta_df = pd.DataFrame(meta_data)
        
        dataset = {
            'X': X,
            'metadata': meta_df
        }
        
        if labels is not None:
            dataset['y'] = np.array(labels)
        
        logger.info(f"创建窗口数据集: {X.shape[0]} 个窗口, 形状 {X.shape}")
        return dataset
    
    def save_windows(self, dataset: Dict, output_path: str):
        """
        保存窗口数据集
        
        Args:
            dataset: 窗口数据集
            output_path: 输出路径
        """
        np.savez_compressed(
            output_path,
            X=dataset['X'],
            y=dataset.get('y'),
            metadata=dataset['metadata'].to_dict('records')
        )
        
        logger.info(f"窗口数据集已保存到: {output_path}")
    
    def load_windows(self, input_path: str) -> Dict:
        """
        加载窗口数据集
        
        Args:
            input_path: 输入路径
            
        Returns:
            dataset: 窗口数据集
        """
        data = np.load(input_path, allow_pickle=True)
        
        dataset = {
            'X': data['X'],
            'metadata': pd.DataFrame(data['metadata'].item())
        }
        
        if 'y' in data and data['y'] is not None:
            dataset['y'] = data['y']
        
        logger.info(f"从 {input_path} 加载窗口数据集")
        return dataset