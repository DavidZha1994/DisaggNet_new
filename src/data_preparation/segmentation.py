"""
数据分段模块
Data Segmentation Module

处理时序数据的分段，包括：
- 基于时间间隔的数据切分
- 段过滤和元数据生成
- 相对时间生成和缺失标记
"""

import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SegmentMetadata:
    """段元数据"""
    device_name: str
    segment_id: int
    start_ts: int
    end_ts: int
    n_rows: int
    duration_hours: float
    has_events: bool
    event_density: float
    max_gap_seconds: int
    filled_ratio: float

class DataSegmenter:
    """数据分段器"""
    
    def __init__(self, config: Dict):
        self.gap_threshold = config['segmentation']['gap_threshold_seconds']
        self.min_segment_length = config['segmentation']['min_segment_length']
        self.fill_small_gaps = config['segmentation']['fill_small_gaps']
        self.small_gap_threshold = config['segmentation']['small_gap_threshold']
        self.fill_method = config['segmentation']['fill_method']
        
    def segment_device_data(self, df: pl.DataFrame, device_name: str) -> Tuple[List[pl.DataFrame], List[SegmentMetadata]]:
        """
        对单个设备的数据进行分段
        
        Args:
            df: 设备数据
            device_name: 设备名称
            
        Returns:
            segments: 分段后的数据列表
            metadata: 段元数据列表
        """
        logger.info(f"开始分段设备数据: {device_name}")
        
        # 确保数据按时间排序
        df = df.sort("ts_utc")
        
        # 计算时间间隔
        df = df.with_columns([
            pl.col("ts_utc").diff().alias("time_diff")
        ])
        
        # 标识大间隔位置
        large_gaps = df.filter(pl.col("time_diff") > self.gap_threshold)
        gap_indices = large_gaps.select(pl.col("ts_utc")).to_numpy().flatten()
        
        logger.info(f"发现 {len(gap_indices)} 个大间隔 (>{self.gap_threshold}s)")
        
        # 分段
        segments = []
        metadata = []
        
        start_idx = 0
        segment_id = 0
        
        for gap_ts in gap_indices:
            # 找到间隔前的结束位置
            end_idx = df.filter(pl.col("ts_utc") < gap_ts).height
            
            if end_idx > start_idx:
                segment_df = df.slice(start_idx, end_idx - start_idx)
                
                # 检查段长度
                if segment_df.height >= self.min_segment_length:
                    # 处理段数据
                    processed_segment, seg_metadata = self._process_segment(
                        segment_df, device_name, segment_id
                    )
                    segments.append(processed_segment)
                    metadata.append(seg_metadata)
                    segment_id += 1
                else:
                    logger.debug(f"丢弃短段: {segment_df.height} < {self.min_segment_length}")
            
            # 更新起始位置
            start_idx = df.filter(pl.col("ts_utc") >= gap_ts).select(pl.int_range(pl.len())).item(0, 0)
        
        # 处理最后一段
        if start_idx < df.height:
            segment_df = df.slice(start_idx, df.height - start_idx)
            if segment_df.height >= self.min_segment_length:
                processed_segment, seg_metadata = self._process_segment(
                    segment_df, device_name, segment_id
                )
                segments.append(processed_segment)
                metadata.append(seg_metadata)
        
        logger.info(f"设备 {device_name} 分段完成: {len(segments)} 个有效段")
        return segments, metadata
    
    def _process_segment(self, df: pl.DataFrame, device_name: str, segment_id: int) -> Tuple[pl.DataFrame, SegmentMetadata]:
        """
        处理单个段的数据
        
        Args:
            df: 段数据
            device_name: 设备名称
            segment_id: 段ID
            
        Returns:
            processed_df: 处理后的段数据
            metadata: 段元数据
        """
        # 计算时间间隔
        df = df.with_columns([
            pl.col("ts_utc").diff().alias("time_diff")
        ])
        
        # 填充小间隔
        mask_filled = pl.Series("mask_filled", [False] * df.height, dtype=pl.Boolean)
        
        if self.fill_small_gaps:
            small_gaps = df.filter(
                (pl.col("time_diff") > 5) & 
                (pl.col("time_diff") <= self.small_gap_threshold)
            )
            
            if small_gaps.height > 0:
                logger.debug(f"填充 {small_gaps.height} 个小间隔")
                # 这里简化处理，实际应该根据fill_method进行不同的填充
                # 标记填充的位置
                mask_filled = df.with_columns([
                    ((pl.col("time_diff") > 5) & 
                     (pl.col("time_diff") <= self.small_gap_threshold)).fill_null(False).alias("is_filled")
                ]).select("is_filled").to_series()
        
        # 生成相对时间
        start_ts = df.select("ts_utc").item(0, 0)
        df = df.with_columns([
            (pl.col("ts_utc") - start_ts).alias("t_rel"),
            mask_filled.alias("mask_filled")
        ])
        
        # 计算段统计信息
        end_ts = df.select("ts_utc").item(-1, 0)
        duration_hours = (end_ts - start_ts) / 3600.0
        filled_ratio = mask_filled.sum() / len(mask_filled)
        max_gap = df.select("time_diff").max().item(0, 0) if df.height > 1 else 0
        
        # 检测事件（这里简化处理，实际应该根据具体业务逻辑）
        # 假设功率变化超过阈值为事件
        has_events = False
        event_density = 0.0
        
        if "P_kW" in df.columns:
            power_changes = df.select(pl.col("P_kW").diff().abs())
            large_changes = power_changes.filter(pl.col("P_kW") > 1.0)  # 1kW变化阈值
            has_events = large_changes.height > 0
            event_density = large_changes.height / df.height
        
        # 创建元数据
        metadata = SegmentMetadata(
            device_name=device_name,
            segment_id=segment_id,
            start_ts=start_ts,
            end_ts=end_ts,
            n_rows=df.height,
            duration_hours=duration_hours,
            has_events=has_events,
            event_density=event_density,
            max_gap_seconds=max_gap,
            filled_ratio=filled_ratio
        )
        
        return df, metadata
    
    def save_segments(self, segments: List[pl.DataFrame], metadata: List[SegmentMetadata], 
                     output_dir: Path, device_name: str, format: str = "parquet"):
        """
        保存分段数据和元数据
        
        Args:
            segments: 分段数据列表
            metadata: 元数据列表
            output_dir: 输出目录
            device_name: 设备名称
            format: 保存格式
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存各段数据
        for i, segment in enumerate(segments):
            if format == "parquet":
                filename = f"{device_name}_segment_{i:03d}.parquet"
                segment.write_parquet(output_dir / filename)
            else:
                filename = f"{device_name}_segment_{i:03d}.csv"
                segment.write_csv(output_dir / filename)
        
        logger.info(f"保存 {len(segments)} 个段到 {output_dir}")
    
    def create_metadata_summary(self, all_metadata: List[SegmentMetadata]) -> pl.DataFrame:
        """
        创建所有段的元数据汇总
        
        Args:
            all_metadata: 所有段的元数据
            
        Returns:
            metadata_df: 元数据DataFrame
        """
        data = []
        for meta in all_metadata:
            data.append({
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
            })
        
        return pl.DataFrame(data)
    
    def segment_all_devices(self, data_dir: Path, output_dir: Path, 
                           device_pattern: str = "*_PhaseCount_*.csv") -> pl.DataFrame:
        """
        分段所有设备的数据
        
        Args:
            data_dir: 数据目录
            output_dir: 输出目录
            device_pattern: 设备文件模式
            
        Returns:
            metadata_summary: 元数据汇总
        """
        data_files = list(Path(data_dir).glob(device_pattern))
        all_metadata = []
        
        for file_path in data_files:
            # 提取设备名称
            device_name = file_path.stem.split('_PhaseCount_')[0]
            
            try:
                # 读取数据
                df = pl.read_csv(file_path)
                
                # 分段处理
                segments, metadata = self.segment_device_data(df, device_name)
                
                # 保存段数据
                self.save_segments(segments, metadata, output_dir, device_name)
                
                # 收集元数据
                all_metadata.extend(metadata)
                
            except Exception as e:
                logger.error(f"处理设备 {device_name} 时出错: {e}")
                continue
        
        # 创建并保存元数据汇总
        metadata_summary = self.create_metadata_summary(all_metadata)
        metadata_summary.write_csv(output_dir / "segments_meta.csv")
        
        logger.info(f"所有设备分段完成，共 {len(all_metadata)} 个段")
        return metadata_summary