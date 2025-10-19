"""
交叉验证模块
Cross Validation Module

实现时序数据的walk-forward交叉验证，包括：
- Rolling Origin划分
- Purge Gap防泄漏
- 段级隔离
- 时间感知的数据分割
"""

import numpy as np
import polars as pl
import pandas as pd
from typing import List, Dict, Tuple, Iterator, Optional
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class CVFold:
    """交叉验证折"""
    fold_id: int
    train_start_ts: int
    train_end_ts: int
    val_start_ts: int
    val_end_ts: int
    test_start_ts: Optional[int] = None
    test_end_ts: Optional[int] = None
    purge_gap_seconds: int = 0
    train_segments: List[int] = None
    val_segments: List[int] = None
    test_segments: List[int] = None

class WalkForwardCV:
    """Walk-Forward交叉验证器"""
    
    def __init__(self, config: Dict):
        self.n_folds = config['cross_validation']['n_folds']
        self.purge_gap_minutes = config['cross_validation']['purge_gap_minutes']
        self.val_span_days = config['cross_validation']['val_span_days']
        self.test_span_days = config['cross_validation']['test_span_days']
        self.min_train_days = config['cross_validation']['min_train_days']
        self.segment_isolation = config['cross_validation']['segment_isolation']
        self.holdout_test = config['cross_validation']['holdout_test']
        
        self.purge_gap_seconds = self.purge_gap_minutes * 60
        
    def create_folds(self, segments_meta: pd.DataFrame) -> List[CVFold]:
        """
        创建walk-forward交叉验证折
        
        Args:
            segments_meta: 段元数据
            
        Returns:
            folds: CV折列表
        """
        logger.info(f"创建 {self.n_folds} 折 walk-forward CV")
        
        # 获取全局时间范围
        global_start = segments_meta['start_ts'].min()
        global_end = segments_meta['end_ts'].max()
        
        logger.info(f"数据时间范围: {datetime.fromtimestamp(global_start)} - {datetime.fromtimestamp(global_end)}")
        
        # 预留测试集
        if self.holdout_test:
            test_start = global_end - (self.test_span_days * 24 * 3600)
            cv_end = test_start - self.purge_gap_seconds
        else:
            cv_end = global_end
            test_start = None
        
        # 计算每折的时间跨度
        cv_span = cv_end - global_start
        fold_span = cv_span // self.n_folds
        val_span_seconds = self.val_span_days * 24 * 3600
        min_train_seconds = self.min_train_days * 24 * 3600
        
        folds = []
        
        for fold_id in range(self.n_folds):
            # 扩展式时间窗口：每次使用从 global_start 到当前折终点的全部数据
            used_span = cv_span * (fold_id + 1) // self.n_folds
            fold_end = global_start + used_span
            if fold_id == self.n_folds - 1:
                fold_end = cv_end

            # 80/20 时间比例，验证集为末尾20%（相对于当前使用的总跨度）
            total_used_span = fold_end - global_start
            val_span_seconds_ratio = max(int(total_used_span * 0.2), 1)
            val_end = fold_end
            val_start = val_end - val_span_seconds_ratio

            # 训练集为前80%，并与验证集之间留 purge gap；训练起点固定为 global_start，实现滚动扩展
            train_start = global_start
            train_end = val_start - self.purge_gap_seconds

            # 确保训练集足够长
            if train_end - train_start < min_train_seconds:
                logger.warning(f"折 {fold_id} 训练集时间不足，跳过")
                continue

            # 创建CV折
            fold = CVFold(
                fold_id=fold_id,
                train_start_ts=int(train_start),
                train_end_ts=int(train_end),
                val_start_ts=int(val_start),
                val_end_ts=int(val_end),
                purge_gap_seconds=self.purge_gap_seconds
            )
            
            # 添加测试集信息
            if self.holdout_test and fold_id == self.n_folds - 1:
                fold.test_start_ts = int(test_start)
                fold.test_end_ts = int(global_end)
            
            # 分配段到各集合
            fold = self._assign_segments_to_fold(fold, segments_meta)
            
            folds.append(fold)
            
            logger.info(f"折 {fold_id}: 训练 {datetime.fromtimestamp(train_start)} - {datetime.fromtimestamp(train_end)}")
            logger.info(f"折 {fold_id}: 验证 {datetime.fromtimestamp(val_start)} - {datetime.fromtimestamp(val_end)}")
        
        return folds
    
    def _assign_segments_to_fold(self, fold: CVFold, segments_meta: pd.DataFrame) -> CVFold:
        """
        为CV折分配段
        
        Args:
            fold: CV折
            segments_meta: 段元数据
            
        Returns:
            fold: 更新后的CV折
        """
        train_segments = []
        val_segments = []
        test_segments = []
        
        for _, segment in segments_meta.iterrows():
            segment_start = segment['start_ts']
            segment_end = segment['end_ts']
            
            # 段级隔离：段不能跨越集合边界
            if self.segment_isolation:
                # 训练集：段完全在训练时间窗口内
                if segment_end <= fold.train_end_ts:
                    train_segments.append(segment['segment_id'])
                
                # 验证集：段完全在验证时间窗口内
                elif (segment_start >= fold.val_start_ts and 
                      segment_end <= fold.val_end_ts):
                    val_segments.append(segment['segment_id'])
                
                # 测试集：段完全在测试时间窗口内
                elif (fold.test_start_ts is not None and
                      segment_start >= fold.test_start_ts and
                      segment_end <= fold.test_end_ts):
                    test_segments.append(segment['segment_id'])
            
            else:
                # 允许段分割：按时间点分配
                if segment_start < fold.train_end_ts:
                    train_segments.append(segment['segment_id'])
                elif segment_start >= fold.val_start_ts and segment_start < fold.val_end_ts:
                    val_segments.append(segment['segment_id'])
                elif (fold.test_start_ts is not None and 
                      segment_start >= fold.test_start_ts):
                    test_segments.append(segment['segment_id'])
        
        fold.train_segments = train_segments
        fold.val_segments = val_segments
        fold.test_segments = test_segments
        
        logger.debug(f"折 {fold.fold_id}: 训练段 {len(train_segments)}, 验证段 {len(val_segments)}, 测试段 {len(test_segments)}")
        
        return fold
    
    def split_windows(self, windows_dataset: Dict, fold: CVFold) -> Tuple[Dict, Dict, Optional[Dict]]:
        """
        根据CV折分割窗口数据集
        
        Args:
            windows_dataset: 窗口数据集
            fold: CV折
            
        Returns:
            train_dataset: 训练集
            val_dataset: 验证集
            test_dataset: 测试集（可选）
        """
        metadata = windows_dataset['metadata']
        X = windows_dataset['X']
        y = windows_dataset.get('y')
        
        # 根据时间和段ID分割
        if self.segment_isolation:
            # 段级分割
            train_mask = metadata['segment_id'].isin(fold.train_segments)
            val_mask = metadata['segment_id'].isin(fold.val_segments)
            test_mask = metadata['segment_id'].isin(fold.test_segments) if fold.test_segments else np.zeros(len(metadata), dtype=bool)
        else:
            # 时间点分割
            train_mask = metadata['end_ts'] <= fold.train_end_ts
            val_mask = ((metadata['start_ts'] >= fold.val_start_ts) & 
                       (metadata['end_ts'] <= fold.val_end_ts))
            test_mask = ((metadata['start_ts'] >= fold.test_start_ts) & 
                        (metadata['end_ts'] <= fold.test_end_ts)) if fold.test_start_ts else np.zeros(len(metadata), dtype=bool)
        
        # 创建数据集
        train_dataset = {
            'X': X[train_mask],
            'metadata': metadata[train_mask].reset_index(drop=True)
        }
        
        val_dataset = {
            'X': X[val_mask],
            'metadata': metadata[val_mask].reset_index(drop=True)
        }
        
        test_dataset = None
        if np.any(test_mask):
            test_dataset = {
                'X': X[test_mask],
                'metadata': metadata[test_mask].reset_index(drop=True)
            }
        
        # 添加标签
        if y is not None:
            train_dataset['y'] = y[train_mask]
            val_dataset['y'] = y[val_mask]
            if test_dataset is not None:
                test_dataset['y'] = y[test_mask]
        
        logger.info(f"折 {fold.fold_id} 数据分割: 训练 {len(train_dataset['X'])}, 验证 {len(val_dataset['X'])}, 测试 {len(test_dataset['X']) if test_dataset else 0}")
        
        return train_dataset, val_dataset, test_dataset
    
    def validate_no_leakage(self, fold: CVFold, train_dataset: Dict, val_dataset: Dict) -> bool:
        """
        验证数据泄漏
        
        Args:
            fold: CV折
            train_dataset: 训练集
            val_dataset: 验证集
            
        Returns:
            is_valid: 是否无泄漏
        """
        train_meta = train_dataset['metadata']
        val_meta = val_dataset['metadata']
        
        # 检查时间重叠
        train_max_ts = train_meta['end_ts'].max()
        val_min_ts = val_meta['start_ts'].min()
        
        time_gap = val_min_ts - train_max_ts
        
        if time_gap < self.purge_gap_seconds:
            logger.error(f"时间泄漏检测: 间隔 {time_gap}s < 要求 {self.purge_gap_seconds}s")
            return False
        
        # 检查段重叠
        if self.segment_isolation:
            train_segments = set(train_meta['segment_id'].unique())
            val_segments = set(val_meta['segment_id'].unique())
            
            overlap_segments = train_segments & val_segments
            if overlap_segments:
                logger.error(f"段重叠检测: 重叠段 {overlap_segments}")
                return False
        
        logger.info(f"折 {fold.fold_id} 泄漏检查通过: 时间间隔 {time_gap}s")
        return True
    
    def get_fold_summary(self, fold: CVFold) -> Dict:
        """
        获取CV折摘要
        
        Args:
            fold: CV折
            
        Returns:
            summary: 折摘要
        """
        summary = {
            'fold_id': fold.fold_id,
            'train_period': f"{datetime.fromtimestamp(fold.train_start_ts)} - {datetime.fromtimestamp(fold.train_end_ts)}",
            'val_period': f"{datetime.fromtimestamp(fold.val_start_ts)} - {datetime.fromtimestamp(fold.val_end_ts)}",
            'purge_gap_hours': fold.purge_gap_seconds / 3600,
            'train_segments': len(fold.train_segments) if fold.train_segments else 0,
            'val_segments': len(fold.val_segments) if fold.val_segments else 0,
            'test_segments': len(fold.test_segments) if fold.test_segments else 0
        }
        
        if fold.test_start_ts:
            summary['test_period'] = f"{datetime.fromtimestamp(fold.test_start_ts)} - {datetime.fromtimestamp(fold.test_end_ts)}"
        
        return summary
    
    def save_cv_plan(self, folds: List[CVFold], output_path: str):
        """
        保存CV计划
        
        Args:
            folds: CV折列表
            output_path: 输出路径
        """
        cv_plan = []
        for fold in folds:
            fold_data = {
                'fold_id': fold.fold_id,
                'train_start_ts': fold.train_start_ts,
                'train_end_ts': fold.train_end_ts,
                'val_start_ts': fold.val_start_ts,
                'val_end_ts': fold.val_end_ts,
                'test_start_ts': fold.test_start_ts,
                'test_end_ts': fold.test_end_ts,
                'purge_gap_seconds': fold.purge_gap_seconds,
                'train_segments': fold.train_segments,
                'val_segments': fold.val_segments,
                'test_segments': fold.test_segments
            }
            cv_plan.append(fold_data)
        
        cv_df = pd.DataFrame(cv_plan)
        cv_df.to_csv(output_path, index=False)
        
        logger.info(f"CV计划已保存到: {output_path}")
    
    def load_cv_plan(self, input_path: str) -> List[CVFold]:
        """
        加载CV计划
        
        Args:
            input_path: 输入路径
            
        Returns:
            folds: CV折列表
        """
        cv_df = pd.read_csv(input_path)
        
        folds = []
        for _, row in cv_df.iterrows():
            fold = CVFold(
                fold_id=row['fold_id'],
                train_start_ts=row['train_start_ts'],
                train_end_ts=row['train_end_ts'],
                val_start_ts=row['val_start_ts'],
                val_end_ts=row['val_end_ts'],
                test_start_ts=row['test_start_ts'] if pd.notna(row['test_start_ts']) else None,
                test_end_ts=row['test_end_ts'] if pd.notna(row['test_end_ts']) else None,
                purge_gap_seconds=row['purge_gap_seconds'],
                train_segments=eval(row['train_segments']) if pd.notna(row['train_segments']) else [],
                val_segments=eval(row['val_segments']) if pd.notna(row['val_segments']) else [],
                test_segments=eval(row['test_segments']) if pd.notna(row['test_segments']) else []
            )
            folds.append(fold)
        
        logger.info(f"从 {input_path} 加载CV计划")
        return folds