"""
Conformal Prediction 不确定性量化模块
提供回归和分类任务的置信区间预测，支持可控的覆盖率
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class BaseConformalPredictor(ABC):
    """Conformal Prediction基础类"""
    
    def __init__(self, coverage: float = 0.9):
        """
        Args:
            coverage: 目标覆盖率 (0, 1)
        """
        assert 0 < coverage < 1, "Coverage must be between 0 and 1"
        self.coverage = coverage
        self.alpha = 1 - coverage
        self.is_calibrated = False
        self.calibration_scores = None
        
    @abstractmethod
    def compute_nonconformity_score(self, predictions: torch.Tensor, 
                                  targets: torch.Tensor) -> torch.Tensor:
        """计算非一致性分数"""
        pass
    
    @abstractmethod
    def predict_with_interval(self, predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """预测带置信区间的结果"""
        pass
    
    def calibrate(self, predictions: torch.Tensor, targets: torch.Tensor):
        """在验证集上进行标定"""
        scores = self.compute_nonconformity_score(predictions, targets)
        self.calibration_scores = scores
        self.is_calibrated = True
        
        # 计算分位数
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.quantile = torch.quantile(scores, q_level)
        
        logger.info(f"Conformal calibration completed. Quantile: {self.quantile:.4f}")
    
    def save_calibration(self, path: str):
        """保存标定结果"""
        if not self.is_calibrated:
            raise ValueError("Must calibrate before saving")
        
        calibration_data = {
            'coverage': self.coverage,
            'alpha': self.alpha,
            'quantile': self.quantile.item() if isinstance(self.quantile, torch.Tensor) else self.quantile,
            'calibration_scores': self.calibration_scores.cpu().numpy() if isinstance(self.calibration_scores, torch.Tensor) else self.calibration_scores
        }
        
        with open(path, 'wb') as f:
            pickle.dump(calibration_data, f)
        
        logger.info(f"Calibration data saved to {path}")
    
    def load_calibration(self, path: str):
        """加载标定结果"""
        with open(path, 'rb') as f:
            calibration_data = pickle.load(f)
        
        self.coverage = calibration_data['coverage']
        self.alpha = calibration_data['alpha']
        self.quantile = calibration_data['quantile']
        self.calibration_scores = calibration_data['calibration_scores']
        self.is_calibrated = True
        
        logger.info(f"Calibration data loaded from {path}")


class RegressionConformalPredictor(BaseConformalPredictor):
    """回归任务的Conformal Prediction"""
    
    def __init__(self, coverage: float = 0.9, score_type: str = 'absolute'):
        """
        Args:
            coverage: 目标覆盖率
            score_type: 非一致性分数类型 ('absolute', 'squared', 'normalized')
        """
        super().__init__(coverage)
        self.score_type = score_type
    
    def compute_nonconformity_score(self, predictions: torch.Tensor, 
                                  targets: torch.Tensor) -> torch.Tensor:
        """计算回归的非一致性分数"""
        if self.score_type == 'absolute':
            scores = torch.abs(predictions - targets)
        elif self.score_type == 'squared':
            scores = (predictions - targets) ** 2
        elif self.score_type == 'normalized':
            # 使用预测值的标准差进行归一化（需要模型输出不确定性）
            if predictions.dim() > targets.dim():
                # 假设预测包含均值和方差
                pred_mean = predictions[..., 0]
                pred_std = torch.clamp(predictions[..., 1], min=1e-6)
                scores = torch.abs(pred_mean - targets) / pred_std
            else:
                scores = torch.abs(predictions - targets)
        else:
            raise ValueError(f"Unknown score type: {self.score_type}")
        
        return scores
    
    def predict_with_interval(self, predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """预测功率区间"""
        if not self.is_calibrated:
            raise ValueError("Must calibrate before prediction")
        
        # 对于回归，预测区间为 [pred - quantile, pred + quantile]
        lower_bound = predictions - self.quantile
        upper_bound = predictions + self.quantile
        
        # 确保功率非负
        lower_bound = torch.clamp(lower_bound, min=0.0)
        
        return {
            'predictions': predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'interval_width': upper_bound - lower_bound,
            'coverage': self.coverage
        }


class ClassificationConformalPredictor(BaseConformalPredictor):
    """分类任务的Conformal Prediction"""
    
    def __init__(self, coverage: float = 0.9, score_type: str = 'aps'):
        """
        Args:
            coverage: 目标覆盖率
            score_type: 非一致性分数类型 ('aps', 'raps', 'lac')
        """
        super().__init__(coverage)
        self.score_type = score_type
    
    def compute_nonconformity_score(self, predictions: torch.Tensor, 
                                  targets: torch.Tensor) -> torch.Tensor:
        """计算分类的非一致性分数"""
        if self.score_type == 'aps':  # Adaptive Prediction Sets
            # 使用1 - 真实类别的概率作为分数
            probs = torch.softmax(predictions, dim=-1)
            # 对于二分类，targets是0/1，对于多分类需要转换为one-hot
            if targets.dim() == 1:
                scores = 1 - probs.gather(1, targets.long().unsqueeze(1)).squeeze(1)
            else:
                scores = 1 - (probs * targets).sum(dim=-1)
        elif self.score_type == 'lac':  # Least Ambiguous set-valued Classifier
            # 使用负对数似然作为分数
            log_probs = torch.log_softmax(predictions, dim=-1)
            if targets.dim() == 1:
                scores = -log_probs.gather(1, targets.long().unsqueeze(1)).squeeze(1)
            else:
                scores = -(log_probs * targets).sum(dim=-1)
        else:
            raise ValueError(f"Unknown score type: {self.score_type}")
        
        return scores
    
    def predict_with_interval(self, predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """预测校准后的置信区间"""
        if not self.is_calibrated:
            raise ValueError("Must calibrate before prediction")
        
        probs = torch.softmax(predictions, dim=-1)
        
        # 计算预测集合
        if self.score_type == 'aps':
            # 按概率降序排列，累积概率超过(1-quantile)的类别包含在预测集中
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # 找到需要包含的类别数量
            prediction_sets = cumsum_probs <= (1 - self.quantile)
            
            # 至少包含一个类别
            prediction_sets[:, 0] = True
            
            # 计算校准后的置信度
            calibrated_confidence = 1 - self.quantile
            
        else:  # lac
            # 对于LAC，使用阈值方法
            calibrated_confidence = 1 - self.quantile
            prediction_sets = probs >= self.quantile
        
        return {
            'predictions': predictions,
            'probabilities': probs,
            'prediction_sets': prediction_sets,
            'calibrated_confidence': calibrated_confidence,
            'coverage': self.coverage
        }


class MultiTaskConformalPredictor:
    """多任务Conformal Prediction"""
    
    def __init__(self, regression_coverage: float = 0.9, 
                 classification_coverage: float = 0.9,
                 device_names: List[str] = None):
        """
        Args:
            regression_coverage: 回归任务覆盖率
            classification_coverage: 分类任务覆盖率
            device_names: 设备名称列表
        """
        self.device_names = device_names or []
        self.n_devices = len(self.device_names)
        self.is_calibrated = False
        
        # 为每个设备创建回归和分类预测器
        self.regression_predictors = {}
        self.classification_predictors = {}
        
        for device_name in self.device_names:
            self.regression_predictors[device_name] = RegressionConformalPredictor(
                coverage=regression_coverage, score_type='absolute'
            )
            self.classification_predictors[device_name] = ClassificationConformalPredictor(
                coverage=classification_coverage, score_type='aps'
            )
    
    def calibrate(self, predictions: Tuple[torch.Tensor, torch.Tensor], 
                  targets: Tuple[torch.Tensor, torch.Tensor]):
        """
        在验证集上标定
        Args:
            predictions: (power_pred, state_pred)
            targets: (power_target, state_target)
        """
        power_pred, state_pred = predictions
        power_target, state_target = targets
        
        # 为每个设备分别标定
        for i, device_name in enumerate(self.device_names):
            # 回归标定
            self.regression_predictors[device_name].calibrate(
                power_pred[:, i], power_target[:, i]
            )
            
            # 分类标定
            self.classification_predictors[device_name].calibrate(
                state_pred[:, i], state_target[:, i]
            )
        
        self.is_calibrated = True
        logger.info("Multi-task conformal calibration completed")
    
    def predict_with_intervals(self, predictions: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, Dict]:
        """
        预测带置信区间的结果
        Returns:
            Dict with device-wise predictions including intervals
        """
        power_pred, state_pred = predictions
        results = {}
        
        for i, device_name in enumerate(self.device_names):
            # 回归区间预测
            power_intervals = self.regression_predictors[device_name].predict_with_interval(
                power_pred[:, i]
            )
            
            # 分类置信区间预测
            state_intervals = self.classification_predictors[device_name].predict_with_interval(
                state_pred[:, i]
            )
            
            results[device_name] = {
                'power': power_intervals,
                'state': state_intervals
            }
        
        return results
    
    def save_calibration(self, save_dir: str):
        """保存所有标定结果"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for device_name in self.device_names:
            # 保存回归标定
            reg_path = save_path / f"{device_name}_regression_calibration.pkl"
            self.regression_predictors[device_name].save_calibration(str(reg_path))
            
            # 保存分类标定
            cls_path = save_path / f"{device_name}_classification_calibration.pkl"
            self.classification_predictors[device_name].save_calibration(str(cls_path))
    
    def load_calibration(self, save_dir: str):
        """加载所有标定结果"""
        save_path = Path(save_dir)
        
        for device_name in self.device_names:
            # 加载回归标定
            reg_path = save_path / f"{device_name}_regression_calibration.pkl"
            if reg_path.exists():
                self.regression_predictors[device_name].load_calibration(str(reg_path))
            
            # 加载分类标定
            cls_path = save_path / f"{device_name}_classification_calibration.pkl"
            if cls_path.exists():
                self.classification_predictors[device_name].load_calibration(str(cls_path))
    
    def get_alert_thresholds(self, confidence_level: float = 0.95) -> Dict[str, Dict]:
        """
        获取告警阈值设计
        Args:
            confidence_level: 置信水平
        Returns:
            设备级别的告警阈值
        """
        thresholds = {}
        
        for device_name in self.device_names:
            reg_predictor = self.regression_predictors[device_name]
            cls_predictor = self.classification_predictors[device_name]
            
            if reg_predictor.is_calibrated and cls_predictor.is_calibrated:
                thresholds[device_name] = {
                    'power_uncertainty_threshold': reg_predictor.quantile,
                    'state_confidence_threshold': cls_predictor.quantile,
                    'coverage': confidence_level
                }
        
        return thresholds


def evaluate_coverage(intervals: Dict[str, torch.Tensor], 
                     targets: torch.Tensor) -> Dict[str, float]:
    """
    评估实际覆盖率
    Args:
        intervals: 预测区间
        targets: 真实值
    Returns:
        覆盖率统计
    """
    results = {}
    
    if 'lower_bound' in intervals and 'upper_bound' in intervals:
        # 回归覆盖率
        lower = intervals['lower_bound']
        upper = intervals['upper_bound']
        
        coverage = ((targets >= lower) & (targets <= upper)).float().mean()
        avg_width = (upper - lower).mean()
        
        results.update({
            'empirical_coverage': coverage.item(),
            'average_interval_width': avg_width.item(),
            'target_coverage': intervals.get('coverage', 0.9)
        })
    
    return results
