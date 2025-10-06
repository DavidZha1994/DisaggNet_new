"""
训练稳定性优化工具
提供数据不平衡处理和训练稳定性优化功能
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import logging
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
import warnings

logger = logging.getLogger(__name__)


class DataImbalanceHandler:
    """数据不平衡处理器"""
    
    def __init__(self, strategy: str = "hybrid", config: Optional[Dict] = None):
        self.strategy = strategy
        self.config = config or {}
        self.samplers = {}
        
    def compute_class_weights(self, labels: np.ndarray) -> Dict[int, float]:
        """计算类别权重"""
        unique_labels = np.unique(labels)
        class_weights = compute_class_weight(
            'balanced', 
            classes=unique_labels, 
            y=labels
        )
        return dict(zip(unique_labels, class_weights))
    
    def setup_samplers(self, X: np.ndarray, y: np.ndarray) -> None:
        """设置采样器"""
        # 过采样器
        if self.strategy in ["oversample", "hybrid"]:
            try:
                self.samplers['oversample'] = SMOTE(
                    sampling_strategy='minority',
                    random_state=42,
                    k_neighbors=min(5, len(np.unique(y)) - 1)
                )
            except Exception:
                # 如果SMOTE失败，使用ADASYN
                self.samplers['oversample'] = ADASYN(
                    sampling_strategy='minority',
                    random_state=42,
                    n_neighbors=min(5, len(np.unique(y)) - 1)
                )
        
        # 欠采样器
        if self.strategy in ["undersample", "hybrid"]:
            self.samplers['undersample'] = RandomUnderSampler(
                sampling_strategy='majority',
                random_state=42
            )
        
        # 混合采样器
        if self.strategy == "hybrid_advanced":
            self.samplers['hybrid'] = SMOTEENN(
                smote=SMOTE(random_state=42),
                enn=EditedNearestNeighbours(),
                random_state=42
            )
    
    def balance_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """平衡数据"""
        self.setup_samplers(X, y)
        
        original_shape = X.shape
        logger.info(f"原始数据形状: {original_shape}, 类别分布: {np.bincount(y)}")
        
        try:
            if self.strategy == "oversample":
                X_resampled, y_resampled = self.samplers['oversample'].fit_resample(X, y)
            elif self.strategy == "undersample":
                X_resampled, y_resampled = self.samplers['undersample'].fit_resample(X, y)
            elif self.strategy == "hybrid":
                # 先过采样，再欠采样
                X_over, y_over = self.samplers['oversample'].fit_resample(X, y)
                X_resampled, y_resampled = self.samplers['undersample'].fit_resample(X_over, y_over)
            elif self.strategy == "hybrid_advanced":
                X_resampled, y_resampled = self.samplers['hybrid'].fit_resample(X, y)
            else:
                X_resampled, y_resampled = X, y
            
            logger.info(f"重采样后数据形状: {X_resampled.shape}, 类别分布: {np.bincount(y_resampled)}")
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.warning(f"数据重采样失败: {e}, 使用原始数据")
            return X, y


class NumericalStabilityOptimizer:
    """数值稳定性优化器"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.eps = self.config.get('eps', 1e-8)
        self.clip_value = self.config.get('clip_value', 100.0)
        
    def stabilize_tensor(self, tensor: torch.Tensor, 
                        clip_range: Optional[Tuple[float, float]] = None) -> torch.Tensor:
        """稳定化张量"""
        # 检查NaN和Inf
        if not torch.isfinite(tensor).all():
            logger.warning("检测到非有限值，进行修复")
            tensor = torch.where(torch.isfinite(tensor), tensor, torch.zeros_like(tensor))
        
        # 裁剪值
        if clip_range:
            tensor = torch.clamp(tensor, clip_range[0], clip_range[1])
        else:
            tensor = torch.clamp(tensor, -self.clip_value, self.clip_value)
        
        return tensor
    
    def safe_log(self, tensor: torch.Tensor, eps: Optional[float] = None) -> torch.Tensor:
        """安全对数变换"""
        if eps is None:
            eps = self.eps
        return torch.log(torch.clamp(tensor, min=eps))
    
    def safe_sqrt(self, tensor: torch.Tensor, eps: Optional[float] = None) -> torch.Tensor:
        """安全平方根"""
        if eps is None:
            eps = self.eps
        return torch.sqrt(torch.clamp(tensor, min=eps))
    
    def safe_division(self, numerator: torch.Tensor, 
                     denominator: torch.Tensor, 
                     eps: Optional[float] = None) -> torch.Tensor:
        """安全除法"""
        if eps is None:
            eps = self.eps
        return numerator / torch.clamp(denominator, min=eps)
    
    def gradient_clipping(self, model: nn.Module, 
                         max_norm: float = 1.0, 
                         norm_type: float = 2.0) -> float:
        """梯度裁剪"""
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type)


class LossStabilizer:
    """损失函数稳定化器"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.eps = self.config.get('eps', 1e-8)
        self.loss_clip = self.config.get('loss_clip', 100.0)
        
    def stabilize_focal_loss(self, pred: torch.Tensor, target: torch.Tensor,
                           alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
        """稳定化Focal Loss"""
        # 数值稳定性保护
        pred = torch.clamp(pred, self.eps, 1 - self.eps)
        
        # 计算交叉熵
        ce_loss = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        
        # 计算权重
        p_t = target * pred + (1 - target) * (1 - pred)
        alpha_t = target * alpha + (1 - target) * (1 - alpha)
        
        # Focal权重
        focal_weight = alpha_t * torch.pow(1 - p_t, gamma)
        
        # 最终损失
        focal_loss = focal_weight * ce_loss
        
        # 裁剪损失
        focal_loss = torch.clamp(focal_loss, 0, self.loss_clip)
        
        return focal_loss.mean()
    
    def stabilize_huber_loss(self, pred: torch.Tensor, target: torch.Tensor,
                           delta: float = 1.0) -> torch.Tensor:
        """稳定化Huber Loss"""
        residual = torch.abs(pred - target)
        
        # 使用平滑的Huber损失
        loss = torch.where(
            residual < delta,
            0.5 * residual ** 2,
            delta * (residual - 0.5 * delta)
        )
        
        # 裁剪损失
        loss = torch.clamp(loss, 0, self.loss_clip)
        
        return loss.mean()


class TrainingStabilityMonitor:
    """训练稳定性监控器"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.history = {
            'losses': [],
            'gradients': [],
            'weights': [],
            'lr': []
        }
        self.anomaly_threshold = self.config.get('anomaly_threshold', 3.0)
        
    def log_training_step(self, loss: float, grad_norm: float, 
                         lr: float, step: int) -> Dict[str, Any]:
        """记录训练步骤"""
        self.history['losses'].append(loss)
        self.history['gradients'].append(grad_norm)
        self.history['lr'].append(lr)
        
        # 检测异常
        anomalies = self.detect_anomalies()
        
        return {
            'step': step,
            'loss': loss,
            'grad_norm': grad_norm,
            'lr': lr,
            'anomalies': anomalies
        }
    
    def detect_anomalies(self) -> List[str]:
        """检测训练异常"""
        anomalies = []
        
        if len(self.history['losses']) > 10:
            recent_losses = self.history['losses'][-10:]
            loss_std = np.std(recent_losses)
            loss_mean = np.mean(recent_losses)
            
            # 检测损失爆炸
            if recent_losses[-1] > loss_mean + self.anomaly_threshold * loss_std:
                anomalies.append("loss_explosion")
            
            # 检测损失震荡
            if loss_std > loss_mean * 0.5:
                anomalies.append("loss_oscillation")
        
        if len(self.history['gradients']) > 5:
            recent_grads = self.history['gradients'][-5:]
            
            # 检测梯度爆炸
            if any(g > 10.0 for g in recent_grads):
                anomalies.append("gradient_explosion")
            
            # 检测梯度消失
            if all(g < 1e-6 for g in recent_grads):
                anomalies.append("gradient_vanishing")
        
        return anomalies
    
    def get_stability_metrics(self) -> Dict[str, float]:
        """获取稳定性指标"""
        if not self.history['losses']:
            return {}
        
        losses = np.array(self.history['losses'])
        gradients = np.array(self.history['gradients'])
        
        return {
            'loss_stability': 1.0 / (1.0 + np.std(losses[-20:]) / (np.mean(losses[-20:]) + 1e-8)),
            'gradient_stability': 1.0 / (1.0 + np.std(gradients[-20:]) / (np.mean(gradients[-20:]) + 1e-8)),
            'training_progress': max(0, (losses[0] - losses[-1]) / (losses[0] + 1e-8)) if len(losses) > 1 else 0
        }


class StabilityOptimizer:
    """综合稳定性优化器"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.imbalance_handler = DataImbalanceHandler(
            strategy=self.config.get('imbalance_strategy', 'hybrid'),
            config=self.config.get('imbalance_config', {})
        )
        self.numerical_optimizer = NumericalStabilityOptimizer(
            config=self.config.get('numerical_config', {})
        )
        self.loss_stabilizer = LossStabilizer(
            config=self.config.get('loss_config', {})
        )
        self.monitor = TrainingStabilityMonitor(
            config=self.config.get('monitor_config', {})
        )
    
    def optimize_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """优化数据"""
        # 处理数据不平衡
        X_balanced, y_balanced = self.imbalance_handler.balance_data(X, y)
        
        # 计算类别权重
        class_weights = self.imbalance_handler.compute_class_weights(y_balanced)
        
        return X_balanced, y_balanced, {'class_weights': class_weights}
    
    def optimize_model(self, model: nn.Module) -> None:
        """优化模型"""
        # 初始化权重
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def optimize_training_step(self, model: nn.Module, loss: torch.Tensor, 
                             optimizer: torch.optim.Optimizer, step: int) -> Dict[str, Any]:
        """优化训练步骤"""
        # 梯度裁剪
        grad_norm = self.numerical_optimizer.gradient_clipping(model)
        
        # 稳定化损失
        loss = self.numerical_optimizer.stabilize_tensor(loss)
        
        # 记录监控信息
        lr = optimizer.param_groups[0]['lr']
        monitor_info = self.monitor.log_training_step(
            loss.item(), grad_norm, lr, step
        )
        
        return monitor_info
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化总结"""
        stability_metrics = self.monitor.get_stability_metrics()
        
        return {
            'stability_metrics': stability_metrics,
            'optimization_config': self.config,
            'recommendations': self._generate_recommendations(stability_metrics)
        }
    
    def _generate_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        if metrics.get('loss_stability', 0) < 0.8:
            recommendations.append("考虑降低学习率或增加梯度裁剪")
        
        if metrics.get('gradient_stability', 0) < 0.8:
            recommendations.append("检查模型架构，可能存在梯度流问题")
        
        if metrics.get('training_progress', 0) < 0.1:
            recommendations.append("训练进展缓慢，考虑调整优化器或学习率调度")
        
        return recommendations