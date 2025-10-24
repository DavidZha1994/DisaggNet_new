"""
鲁棒特征处理器
Robust Feature Handler

专门处理辅助特征中的 NaN/Inf 值，提供多种策略：
1. 数据生成阶段的预防性处理
2. 模型训练阶段的实时清理
3. 特征质量监控和报告
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)


class RobustFeatureHandler:
    """鲁棒特征处理器"""
    
    def __init__(self, 
                 strategy: str = "adaptive",
                 nan_fill_method: str = "median",
                 inf_fill_method: str = "clip",
                 enable_monitoring: bool = True):
        """
        Args:
            strategy: 处理策略 ['conservative', 'adaptive', 'aggressive']
            nan_fill_method: NaN填充方法 ['zero', 'mean', 'median', 'forward', 'interpolate']
            inf_fill_method: Inf处理方法 ['zero', 'clip', 'percentile']
            enable_monitoring: 是否启用质量监控
        """
        self.strategy = strategy
        self.nan_fill_method = nan_fill_method
        self.inf_fill_method = inf_fill_method
        self.enable_monitoring = enable_monitoring
        
        # 统计信息
        self.stats = {
            'total_processed': 0,
            'nan_counts': {},
            'inf_counts': {},
            'feature_quality_scores': {}
        }
        
        # 特征统计缓存（用于填充）
        self.feature_stats = {}
        
    def process_aux_features(self, 
                           aux_features: Union[torch.Tensor, np.ndarray],
                           feature_names: Optional[List[str]] = None,
                           update_stats: bool = True) -> Union[torch.Tensor, np.ndarray]:
        """
        处理辅助特征中的无效值
        
        Args:
            aux_features: 辅助特征 [batch_size, n_features] 或 [n_features]
            feature_names: 特征名称列表
            update_stats: 是否更新统计信息
            
        Returns:
            cleaned_features: 清理后的特征
        """
        is_torch = isinstance(aux_features, torch.Tensor)
        device = aux_features.device if is_torch else None
        
        # 转换为numpy进行处理
        if is_torch:
            features = aux_features.detach().cpu().numpy()
        else:
            features = aux_features.copy()
            
        original_shape = features.shape
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        batch_size, n_features = features.shape
        
        if feature_names is None:
            feature_names = [f"aux_feat_{i}" for i in range(n_features)]
            
        # 检测无效值
        nan_mask = np.isnan(features)
        inf_mask = np.isinf(features)
        invalid_mask = nan_mask | inf_mask
        
        if update_stats:
            self._update_statistics(features, feature_names, nan_mask, inf_mask)
            
        # 如果没有无效值，直接返回
        if not invalid_mask.any():
            if is_torch:
                return aux_features
            else:
                return features.reshape(original_shape)
                
        # 处理无效值
        cleaned_features = self._clean_features(features, nan_mask, inf_mask, feature_names)
        
        # 转换回原始格式
        cleaned_features = cleaned_features.reshape(original_shape)
        
        if is_torch:
            return torch.from_numpy(cleaned_features).to(device).to(aux_features.dtype)
        else:
            return cleaned_features.astype(aux_features.dtype)
            
    def _clean_features(self, 
                       features: np.ndarray,
                       nan_mask: np.ndarray,
                       inf_mask: np.ndarray,
                       feature_names: List[str]) -> np.ndarray:
        """清理特征中的无效值"""
        cleaned = features.copy()
        batch_size, n_features = features.shape
        
        for i, feat_name in enumerate(feature_names):
            feat_col = cleaned[:, i]
            feat_nan_mask = nan_mask[:, i]
            feat_inf_mask = inf_mask[:, i]
            
            # 处理 NaN 值
            if feat_nan_mask.any():
                fill_value = self._get_nan_fill_value(feat_col, feat_name)
                feat_col[feat_nan_mask] = fill_value
                
            # 处理 Inf 值
            if feat_inf_mask.any():
                feat_col = self._handle_inf_values(feat_col, feat_inf_mask, feat_name)
                cleaned[:, i] = feat_col
                
        return cleaned
        
    def _get_nan_fill_value(self, feature_col: np.ndarray, feature_name: str) -> float:
        """获取NaN填充值"""
        valid_values = feature_col[~np.isnan(feature_col)]
        
        if len(valid_values) == 0:
            return 0.0
            
        if self.nan_fill_method == "zero":
            return 0.0
        elif self.nan_fill_method == "mean":
            return np.mean(valid_values)
        elif self.nan_fill_method == "median":
            return np.median(valid_values)
        elif self.nan_fill_method == "forward":
            # 简单的前向填充
            last_valid = None
            for val in feature_col:
                if not np.isnan(val):
                    last_valid = val
            return last_valid if last_valid is not None else 0.0
        else:
            return 0.0
            
    def _handle_inf_values(self, feature_col: np.ndarray, inf_mask: np.ndarray, feature_name: str) -> np.ndarray:
        """处理Inf值"""
        valid_values = feature_col[~np.isinf(feature_col)]
        
        if len(valid_values) == 0:
            feature_col[inf_mask] = 0.0
            return feature_col
            
        if self.inf_fill_method == "zero":
            feature_col[inf_mask] = 0.0
        elif self.inf_fill_method == "clip":
            # 裁剪到有效值的范围
            min_val, max_val = np.min(valid_values), np.max(valid_values)
            feature_col[inf_mask & (feature_col > 0)] = max_val
            feature_col[inf_mask & (feature_col < 0)] = min_val
        elif self.inf_fill_method == "percentile":
            # 使用95%分位数
            p95 = np.percentile(valid_values, 95)
            p5 = np.percentile(valid_values, 5)
            feature_col[inf_mask & (feature_col > 0)] = p95
            feature_col[inf_mask & (feature_col < 0)] = p5
            
        return feature_col
        
    def _update_statistics(self, 
                          features: np.ndarray,
                          feature_names: List[str],
                          nan_mask: np.ndarray,
                          inf_mask: np.ndarray):
        """更新统计信息"""
        self.stats['total_processed'] += features.shape[0]
        
        for i, feat_name in enumerate(feature_names):
            if feat_name not in self.stats['nan_counts']:
                self.stats['nan_counts'][feat_name] = 0
                self.stats['inf_counts'][feat_name] = 0
                
            self.stats['nan_counts'][feat_name] += int(nan_mask[:, i].sum())
            self.stats['inf_counts'][feat_name] += int(inf_mask[:, i].sum())
            
            # 计算特征质量分数
            total_values = features.shape[0]
            invalid_count = nan_mask[:, i].sum() + inf_mask[:, i].sum()
            quality_score = 1.0 - (invalid_count / total_values)
            self.stats['feature_quality_scores'][feat_name] = quality_score
            
    def get_quality_report(self) -> Dict[str, Any]:
        """获取特征质量报告"""
        if self.stats['total_processed'] == 0:
            return {"message": "No data processed yet"}
            
        report = {
            "total_samples_processed": self.stats['total_processed'],
            "feature_quality_summary": {},
            "problematic_features": [],
            "recommendations": []
        }
        
        # 特征质量汇总
        for feat_name, quality_score in self.stats['feature_quality_scores'].items():
            nan_count = self.stats['nan_counts'].get(feat_name, 0)
            inf_count = self.stats['inf_counts'].get(feat_name, 0)
            
            report["feature_quality_summary"][feat_name] = {
                "quality_score": quality_score,
                "nan_count": nan_count,
                "inf_count": inf_count,
                "nan_rate": nan_count / self.stats['total_processed'],
                "inf_rate": inf_count / self.stats['total_processed']
            }
            
            # 识别问题特征
            if quality_score < 0.9:  # 质量分数低于90%
                report["problematic_features"].append({
                    "feature": feat_name,
                    "quality_score": quality_score,
                    "main_issue": "high_nan_rate" if nan_count > inf_count else "high_inf_rate"
                })
                
        # 生成建议
        if len(report["problematic_features"]) > 0:
            report["recommendations"].extend([
                "检查数据生成管道中的特征计算逻辑",
                "考虑在特征工程阶段添加更鲁棒的计算方法",
                "对于频繁出现NaN的特征，考虑使用更稳定的统计量"
            ])
            
        return report
        
    def save_report(self, filepath: str):
        """保存质量报告"""
        report = self.get_quality_report()
        
        if filepath.endswith('.json'):
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        else:
            # 保存为文本格式
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("=== 辅助特征质量报告 ===\n\n")
                f.write(f"总处理样本数: {report['total_samples_processed']}\n\n")
                
                f.write("特征质量汇总:\n")
                for feat_name, stats in report['feature_quality_summary'].items():
                    f.write(f"  {feat_name}:\n")
                    f.write(f"    质量分数: {stats['quality_score']:.3f}\n")
                    f.write(f"    NaN率: {stats['nan_rate']:.3f}\n")
                    f.write(f"    Inf率: {stats['inf_rate']:.3f}\n\n")
                    
                if report['problematic_features']:
                    f.write("问题特征:\n")
                    for prob_feat in report['problematic_features']:
                        f.write(f"  - {prob_feat['feature']}: {prob_feat['main_issue']}\n")
                    f.write("\n")
                    
                if report['recommendations']:
                    f.write("建议:\n")
                    for rec in report['recommendations']:
                        f.write(f"  - {rec}\n")


def create_robust_aux_feature_processor(window_data: np.ndarray, 
                                       feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """
    创建鲁棒的辅助特征处理器（用于数据生成阶段）
    
    Args:
        window_data: 窗口数据 [n_windows, window_length, n_channels]
        feature_names: 通道名称
        
    Returns:
        aux_features: 处理后的辅助特征 [n_windows, n_aux_features]
        aux_names: 特征名称列表
    """
    if window_data.size == 0:
        return np.empty((0, 0), dtype=np.float32), []
        
    N, L, C = window_data.shape
    
    if feature_names is None:
        feature_names = [f"ch_{i}" for i in range(C)]
        
    feats = []
    names = []
    
    # 基础统计特征（NaN感知）
    for ci, cname in enumerate(feature_names):
        x = window_data[:, :, ci]
        
        # 均值和标准差（使用nanmean/nanstd）
        mu = np.nanmean(x, axis=1)
        sd = np.nanstd(x, axis=1)
        
        # 处理全NaN窗口
        mu = np.where(np.isnan(mu), 0.0, mu)
        sd = np.where(np.isnan(sd), 0.0, sd)
        
        feats.extend([mu, sd])
        names.extend([f"mean_{cname}", f"std_{cname}"])
        
        # 添加更鲁棒的统计量
        # 中位数（对异常值更鲁棒）
        median_vals = np.nanmedian(x, axis=1)
        median_vals = np.where(np.isnan(median_vals), 0.0, median_vals)
        feats.append(median_vals)
        names.append(f"median_{cname}")
        
        # 有效值比例
        valid_ratio = (~np.isnan(x)).sum(axis=1) / L
        feats.append(valid_ratio)
        names.append(f"valid_ratio_{cname}")
        
    # 组合特征（如果有多个通道）
    if C >= 2:
        # 通道间相关性（使用有效值）
        for i in range(C):
            for j in range(i+1, C):
                corr_vals = []
                for n in range(N):
                    x_i = window_data[n, :, i]
                    x_j = window_data[n, :, j]
                    valid_mask = ~(np.isnan(x_i) | np.isnan(x_j))
                    if valid_mask.sum() > 1:
                        corr = np.corrcoef(x_i[valid_mask], x_j[valid_mask])[0, 1]
                        corr_vals.append(corr if np.isfinite(corr) else 0.0)
                    else:
                        corr_vals.append(0.0)
                        
                feats.append(np.array(corr_vals, dtype=np.float32))
                names.append(f"corr_{feature_names[i]}_{feature_names[j]}")
    
    # 转换为数组并进行最终清理
    aux_features = np.column_stack(feats).astype(np.float32)
    
    # 最终的NaN/Inf清理
    aux_features = np.nan_to_num(aux_features, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return aux_features, names