"""多任务损失函数"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from omegaconf import DictConfig
import numpy as np


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: (batch_size, n_devices) - logits
        # targets: (batch_size, n_devices) - binary targets
        
        # 数值稳定性保护 - 限制输入范围
        inputs = torch.clamp(inputs, min=-10.0, max=10.0)
        
        # 使用binary_cross_entropy_with_logits替代binary_cross_entropy以支持autocast
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 防止BCE损失数值不稳定
        bce_loss = torch.clamp(bce_loss, min=1e-8, max=100.0)
        
        # 计算概率用于focal weight计算
        probs = torch.sigmoid(inputs)
        
        # 计算pt，添加数值保护
        pt = torch.where(targets == 1, probs, 1 - probs)
        pt = pt.clamp(min=1e-8, max=1.0 - 1e-8)  # 数值保护
        
        # 计算focal weight
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # 应用focal weight
        focal_loss = focal_weight * bce_loss
        
        # 检查NaN和Inf
        if torch.isnan(focal_loss).any() or torch.isinf(focal_loss).any():
            print(f"Warning: NaN or Inf detected in FocalLoss")
            focal_loss = torch.where(torch.isnan(focal_loss) | torch.isinf(focal_loss), 
                                   torch.zeros_like(focal_loss), focal_loss)
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedMSELoss(nn.Module):
    """加权MSE损失（用于处理不平衡的回归问题）"""
    
    def __init__(self, device_weights: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super().__init__()
        self.device_weights = device_weights
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: (batch_size, n_devices)
        # targets: (batch_size, n_devices)
        
        # 数值稳定性保护 - 限制输入范围
        inputs = torch.clamp(inputs, min=0.0, max=1e6)
        targets = torch.clamp(targets, min=0.0, max=1e6)
        
        mse_loss = (inputs - targets) ** 2
        
        # 防止梯度爆炸
        mse_loss = torch.clamp(mse_loss, max=1e6)
        
        if self.device_weights is not None:
            # 应用设备权重
            weights = self.device_weights.to(inputs.device).unsqueeze(0)
            mse_loss = mse_loss * weights
        
        # 检查NaN和Inf
        if torch.isnan(mse_loss).any() or torch.isinf(mse_loss).any():
            print(f"Warning: NaN or Inf detected in WeightedMSELoss")
            mse_loss = torch.where(torch.isnan(mse_loss) | torch.isinf(mse_loss), 
                                 torch.zeros_like(mse_loss), mse_loss)
        
        if self.reduction == 'mean':
            return mse_loss.mean()
        elif self.reduction == 'sum':
            return mse_loss.sum()
        else:
            return mse_loss


class MAELoss(nn.Module):
    """平均绝对误差损失"""
    
    def __init__(self, device_weights: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super().__init__()
        self.device_weights = device_weights
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: (batch_size, n_devices)
        # targets: (batch_size, n_devices)
        
        mae_loss = torch.abs(inputs - targets)
        
        if self.device_weights is not None:
            # 应用设备权重
            weights = self.device_weights.to(inputs.device).unsqueeze(0)
            mae_loss = mae_loss * weights
        
        if self.reduction == 'mean':
            return mae_loss.mean()
        elif self.reduction == 'sum':
            return mae_loss.sum()
        else:
            return mae_loss


class HuberLoss(nn.Module):
    """Huber损失（对异常值鲁棒）"""
    
    def __init__(self, delta: float = 1.0, device_weights: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super().__init__()
        self.delta = delta
        self.device_weights = device_weights
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: (batch_size, n_devices)
        # targets: (batch_size, n_devices)
        # 忽略无效标签：仅对有限值位置计算损失
        valid = torch.isfinite(inputs) & torch.isfinite(targets)

        # 计算残差
        residual = torch.abs(inputs - targets)
        
        # Huber损失：当残差小于delta时使用平方损失，否则使用线性损失
        huber_loss = torch.where(
            residual < self.delta,
            0.5 * residual ** 2,
            self.delta * (residual - 0.5 * self.delta)
        )
        
        # 将无效位置的损失置零
        huber_loss = torch.where(valid, huber_loss, torch.zeros_like(huber_loss))
        
        if self.device_weights is not None:
            # 应用设备权重
            weights = self.device_weights.to(inputs.device).unsqueeze(0)
            huber_loss = huber_loss * weights
            # 有效位置的权重和作为归一化分母
            denom = torch.where(valid, weights, torch.zeros_like(weights)).sum().clamp_min(1.0)
        else:
            denom = valid.float().sum().clamp_min(1.0)
        
        if self.reduction == 'mean':
            return huber_loss.sum() / denom
        elif self.reduction == 'sum':
            return huber_loss.sum()
        else:
            # 返回逐元素损失（已屏蔽无效位置）
            return huber_loss


class QuantileLoss(nn.Module):
    """分位数损失（用于不确定性估计）"""
    
    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9], reduction: str = 'mean'):
        super().__init__()
        self.quantiles = quantiles
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: (batch_size, n_devices * n_quantiles)
        # targets: (batch_size, n_devices)
        
        batch_size, n_devices = targets.size()
        n_quantiles = len(self.quantiles)
        
        # 重塑inputs
        inputs = inputs.view(batch_size, n_devices, n_quantiles)
        targets = targets.unsqueeze(-1).expand(-1, -1, n_quantiles)
        
        # 计算分位数损失
        errors = targets - inputs
        quantile_tensor = torch.tensor(self.quantiles, device=inputs.device).view(1, 1, -1)
        
        quantile_loss = torch.where(
            errors >= 0,
            quantile_tensor * errors,
            (quantile_tensor - 1) * errors
        )
        
        if self.reduction == 'mean':
            return quantile_loss.mean()
        elif self.reduction == 'sum':
            return quantile_loss.sum()
        else:
            return quantile_loss





# 统一损失函数创建接口
def create_loss_function(config: dict) -> 'UnifiedMultiTaskLoss':
    """
    创建统一的多任务损失函数
    
    Args:
        config: 损失函数配置字典，包含各种权重和参数
    
    Returns:
        UnifiedMultiTaskLoss: 配置好的多任务损失函数实例
    """
    # 默认参数
    default_config = {
        'classification_weight': 2.0,
        'regression_weight': 1.0,
        'conservation_weight': 0.5,
        'consistency_weight': 1.0,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'huber_delta': 1.0
    }
    
    # 合并配置
    final_config = {**default_config, **config}
    
    # 兼容旧版/配置文件中的聚合权重字段
    weights_cfg = final_config.pop('weights', None)
    if weights_cfg:
        # 分类/回归权重映射
        if 'classification' in weights_cfg:
            final_config['classification_weight'] = weights_cfg['classification']
        if 'regression' in weights_cfg:
            final_config['regression_weight'] = weights_cfg['regression']
        # 一致性权重（若有）
        if 'consistency' in weights_cfg:
            final_config['consistency_weight'] = weights_cfg['consistency']
        # 如果提供了调度计划，则开启调度开关
        if 'consistency_schedule' in weights_cfg and weights_cfg['consistency_schedule']:
            final_config['enable_scheduling'] = True
    
    # 仅保留 UnifiedMultiTaskLoss 支持的键，避免传入未知参数
    allowed_keys = {
        'classification_weight', 'regression_weight', 'conservation_weight', 'consistency_weight',
        'focal_alpha', 'focal_gamma', 'huber_delta',
        'class_weights', 'enable_adaptive_weighting', 'enable_scheduling'
    }
    filtered_config = {k: final_config[k] for k in allowed_keys if k in final_config}
    
    # 创建损失函数实例
    return UnifiedMultiTaskLoss(**filtered_config)


class UnifiedMultiTaskLoss(nn.Module):
    """统一的多任务损失函数，整合分类、回归、守恒和一致性约束"""
    
    def __init__(self, 
                 classification_weight: float = 2.0,
                 regression_weight: float = 1.0,
                 conservation_weight: float = 0.5,
                 consistency_weight: float = 1.0,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 huber_delta: float = 1.0,
                 class_weights: Optional[torch.Tensor] = None,
                 enable_adaptive_weighting: bool = False,
                 enable_scheduling: bool = False):
        super().__init__()
        
        # 初始损失权重
        self.initial_classification_weight = classification_weight
        self.initial_regression_weight = regression_weight
        self.initial_conservation_weight = conservation_weight
        self.initial_consistency_weight = consistency_weight
        
        # 当前损失权重（可动态调整）
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        self.conservation_weight = conservation_weight
        self.consistency_weight = consistency_weight
        
        # 动态权重调整配置
        self.enable_adaptive_weighting = enable_adaptive_weighting
        self.enable_scheduling = enable_scheduling
        self.current_epoch = 0
        
        # 损失历史记录（用于自适应权重）
        self.loss_history = {
            'classification': [],
            'regression': [],
            'conservation': [],
            'consistency': []
        }
        
        # 损失函数组件
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.huber_loss = HuberLoss(delta=huber_delta)
        
        # 类别权重（如果提供）
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
    
    def conservation_constraint(self, pred_power: torch.Tensor, 
                              total_power: torch.Tensor,
                              pred_switch: torch.Tensor,
                              unknown_pred: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        能量守恒约束：预测的设备功率总和应接近总功率
        
        Args:
            pred_power: 预测的设备功率 (batch_size, n_devices)
            total_power: 总功率 (batch_size, 1)
            pred_switch: 预测的开关状态 (batch_size, n_devices)
            unknown_pred: 预测的未知残差功率 (batch_size, 1)
        
        Returns:
            torch.Tensor: 守恒约束损失
        """
        # 只考虑预测为开启状态的设备
        active_power = pred_power * torch.sigmoid(pred_switch)
        predicted_total = active_power.sum(dim=1, keepdim=True)
        if unknown_pred is not None:
            unknown = F.softplus(unknown_pred)
            predicted_total = predicted_total + unknown
        
        # 使用相对误差，避免总功率为0的情况
        relative_error = torch.abs(predicted_total - total_power) / (total_power + 1e-6)
        return relative_error.mean()
    
    def consistency_constraint(self, pred_power: torch.Tensor, 
                             pred_switch: torch.Tensor,
                             threshold: float = 0.1) -> torch.Tensor:
        """
        一致性约束：功率和开关状态应该一致
        
        Args:
            pred_power: 预测的功率 (batch_size, n_devices)
            pred_switch: 预测的开关状态logits (batch_size, n_devices)
            threshold: 功率阈值，低于此值应该关闭
        
        Returns:
            torch.Tensor: 一致性约束损失
        """
        # 将开关状态logits转换为概率
        switch_prob = torch.sigmoid(pred_switch)
        
        # 低功率时应该关闭（开关概率应该低）
        low_power_mask = pred_power < threshold
        low_power_penalty = switch_prob[low_power_mask].mean() if low_power_mask.any() else torch.tensor(0.0, device=pred_power.device)
        
        # 高功率时应该开启（开关概率应该高）
        high_power_mask = pred_power >= threshold
        high_power_penalty = (1 - switch_prob[high_power_mask]).mean() if high_power_mask.any() else torch.tensor(0.0, device=pred_power.device)
        
        return low_power_penalty + high_power_penalty
    
    def forward(self, 
                pred_power: torch.Tensor,
                pred_switch: torch.Tensor,
                target_power: torch.Tensor,
                target_switch: torch.Tensor,
                total_power: Optional[torch.Tensor] = None,
                unknown_pred: Optional[torch.Tensor] = None,
                sample_weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播计算损失
        
        Args:
            pred_power: 预测功率 (batch_size, n_devices)
            pred_switch: 预测开关状态logits (batch_size, n_devices)
            target_power: 目标功率 (batch_size, n_devices)
            target_switch: 目标开关状态 (batch_size, n_devices)
            total_power: 总功率，用于守恒约束 (batch_size, 1)
            unknown_pred: 预测的未知残差功率 (batch_size, 1)
            sample_weights: 样本权重 (batch_size,)；None 时回退为均值聚合
        
        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: (总损失, 各项损失详情)
        """
        losses = {}
        
        # 回归损失（功率预测）— 支持按样本加权
        # 逐元素Huber损失，随后按样本聚合
        valid = torch.isfinite(pred_power) & torch.isfinite(target_power)
        residual = torch.abs(pred_power - target_power)
        delta = getattr(self.huber_loss, 'delta', 1.0)
        huber_element = torch.where(
            residual < delta,
            0.5 * residual ** 2,
            delta * (residual - 0.5 * delta)
        )
        huber_element = torch.where(valid, huber_element, torch.zeros_like(huber_element))
        if getattr(self.huber_loss, 'device_weights', None) is not None:
            dev_w = self.huber_loss.device_weights.to(pred_power.device).unsqueeze(0).expand_as(huber_element)
            huber_element = huber_element * dev_w
            denom_per_sample = torch.where(valid, dev_w, torch.zeros_like(dev_w)).sum(dim=1).clamp_min(1.0)
        else:
            denom_per_sample = valid.float().sum(dim=1).clamp_min(1.0)
        per_sample_reg = huber_element.sum(dim=1) / denom_per_sample
        if sample_weights is not None:
            w = sample_weights.to(pred_power.device).float().clamp_min(0.0)
            denom_w = w.sum().clamp_min(1e-6)
            regression_loss = (per_sample_reg * w).sum() / denom_w
        else:
            regression_loss = per_sample_reg.mean()
        losses['regression'] = self.regression_weight * regression_loss

        # 分类损失（开关状态）— 支持按样本加权
        if self.classification_weight > 0.0:
            # 数值稳定性与无效标签掩蔽
            inputs = torch.clamp(pred_switch, min=-10.0, max=10.0)
            targets = target_switch
            valid = torch.isfinite(inputs) & torch.isfinite(targets)
            
            # 逐元素 BCE 并掩蔽无效项
            bce_element = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            bce_element = torch.where(valid, torch.clamp(bce_element, min=1e-8, max=100.0), torch.zeros_like(bce_element))
            
            # 计算 focal 权重（按有效项）
            probs = torch.sigmoid(inputs)
            pt = torch.where(targets >= 0.5, probs, 1 - probs).clamp(min=1e-8, max=1.0 - 1e-8)
            focal_weight = getattr(self.focal_loss, 'alpha', 1.0) * (1 - pt) ** getattr(self.focal_loss, 'gamma', 2.0)
            focal_element = focal_weight * bce_element
            
            # 按样本聚合（仅统计有效元素）
            denom_per_sample = valid.float().sum(dim=1).clamp_min(1.0)
            per_sample_cls = focal_element.sum(dim=1) / denom_per_sample
            
            if sample_weights is not None:
                w = sample_weights.to(pred_power.device).float().clamp_min(0.0)
                denom_w = w.sum().clamp_min(1e-6)
                classification_loss = (per_sample_cls * w).sum() / denom_w
            else:
                classification_loss = per_sample_cls.mean()
            losses['classification'] = self.classification_weight * classification_loss
        else:
            losses['classification'] = torch.tensor(0.0, device=pred_power.device)

        # 守恒约束（如果提供总功率）— 支持按样本加权
        if total_power is not None and self.conservation_weight > 0.0:
            if self.classification_weight > 0.0:
                conservation_loss = self.conservation_constraint(pred_power, total_power, pred_switch, unknown_pred=unknown_pred)
            else:
                predicted_total = pred_power.sum(dim=1, keepdim=True)
                if unknown_pred is not None:
                    predicted_total = predicted_total + F.softplus(unknown_pred)
                # 掩蔽无效项，避免 NaN/Inf
                valid_total = torch.isfinite(predicted_total) & torch.isfinite(total_power)
                rel_err = torch.where(
                    valid_total,
                    torch.abs(predicted_total - total_power) / (total_power.abs() + 1e-6),
                    torch.zeros_like(total_power)
                )
                per_sample_cons = rel_err.squeeze(1)
                if sample_weights is not None:
                    w = sample_weights.to(pred_power.device).float().clamp_min(0.0)
                    denom_w = w.sum().clamp_min(1e-6)
                    conservation_loss = (per_sample_cons * w).sum() / denom_w
                else:
                    conservation_loss = per_sample_cons.mean()
            losses['conservation'] = self.conservation_weight * conservation_loss

        # 一致性约束 — 仅在分类启用时计算（支持按样本加权）
        if self.classification_weight > 0.0 and self.consistency_weight > 0.0:
            switch_prob = torch.sigmoid(pred_switch)
            low_mask = (pred_power < 0.1).float()
            high_mask = (pred_power >= 0.1).float()
            # 低功率应关闭：惩罚为开概率
            low_sum = (switch_prob * low_mask).sum(dim=1)
            low_count = low_mask.sum(dim=1)
            low_penalty = torch.where(low_count > 0, low_sum / low_count.clamp_min(1.0), torch.zeros_like(low_count))
            # 高功率应开启：惩罚为(1-开概率)
            high_sum = ((1 - switch_prob) * high_mask).sum(dim=1)
            high_count = high_mask.sum(dim=1)
            high_penalty = torch.where(high_count > 0, high_sum / high_count.clamp_min(1.0), torch.zeros_like(high_count))
            per_sample_consistency = low_penalty + high_penalty
            if sample_weights is not None:
                w = sample_weights.to(pred_power.device).float().clamp_min(0.0)
                denom_w = w.sum().clamp_min(1e-6)
                consistency_loss = (per_sample_consistency * w).sum() / denom_w
            else:
                consistency_loss = per_sample_consistency.mean()
            losses['consistency'] = self.consistency_weight * consistency_loss
        else:
            losses['consistency'] = torch.tensor(0.0, device=pred_power.device)
        
        # 总损失
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        # 记录损失历史（用于自适应权重调整）
        if self.enable_adaptive_weighting:
            self._update_loss_history(losses)
        
        return total_loss, losses
    
    def update_epoch(self, epoch: int):
        """更新训练轮次，用于损失权重调度"""
        self.current_epoch = epoch
        
        if self.enable_scheduling:
            self._update_scheduled_weights(epoch)
        
        if self.enable_adaptive_weighting and len(self.loss_history['classification']) > 10:
            self._update_adaptive_weights()
    
    def _update_loss_history(self, losses: Dict[str, torch.Tensor]):
        """更新损失历史记录"""
        for key in ['classification', 'regression', 'conservation', 'consistency']:
            if key in losses:
                self.loss_history[key].append(losses[key].item())
                # 保持历史记录长度不超过100
                if len(self.loss_history[key]) > 100:
                    self.loss_history[key] = self.loss_history[key][-100:]
    
    def _update_scheduled_weights(self, epoch: int):
        """基于训练轮次调度损失权重"""
        # 示例调度策略：前期注重分类，后期平衡各任务
        if epoch < 10:
            # 前10个epoch，强调分类任务
            self.classification_weight = self.initial_classification_weight * 1.5
            self.regression_weight = self.initial_regression_weight * 0.8
        elif epoch < 30:
            # 中期，逐渐平衡
            progress = (epoch - 10) / 20.0
            self.classification_weight = self.initial_classification_weight * (1.5 - 0.5 * progress)
            self.regression_weight = self.initial_regression_weight * (0.8 + 0.2 * progress)
        else:
            # 后期，使用初始权重
            self.classification_weight = self.initial_classification_weight
            self.regression_weight = self.initial_regression_weight
    
    def _update_adaptive_weights(self):
        """基于损失历史自适应调整权重"""
        # 计算最近10个epoch的损失变化趋势
        recent_window = 10
        
        for loss_type in ['classification', 'regression', 'conservation', 'consistency']:
            if len(self.loss_history[loss_type]) >= recent_window:
                recent_losses = self.loss_history[loss_type][-recent_window:]
                
                # 计算损失下降趋势
                if len(recent_losses) >= 2:
                    trend = (recent_losses[-1] - recent_losses[0]) / len(recent_losses)
                    
                    # 如果某个损失下降缓慢或上升，增加其权重
                    if loss_type == 'classification':
                        if trend > -0.001:  # 下降缓慢
                            self.classification_weight = min(
                                self.classification_weight * 1.1, 
                                self.initial_classification_weight * 2.0
                            )
                        elif trend < -0.01:  # 下降很快
                            self.classification_weight = max(
                                self.classification_weight * 0.95,
                                self.initial_classification_weight * 0.5
                            )
                    
                    elif loss_type == 'regression':
                        if trend > -0.001:
                            self.regression_weight = min(
                                self.regression_weight * 1.1,
                                self.initial_regression_weight * 2.0
                            )
                        elif trend < -0.01:
                            self.regression_weight = max(
                                self.regression_weight * 0.95,
                                self.initial_regression_weight * 0.5
                            )


# 推荐的损失函数配置
RECOMMENDED_LOSS_CONFIGS = {
    'balanced': {
        'classification_weight': 2.0,
        'regression_weight': 1.0,
        'conservation_weight': 0.5,
        'consistency_weight': 1.0,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'huber_delta': 1.0,
        'enable_adaptive_weighting': False,
        'enable_scheduling': False
    },
    'classification_focused': {
        'classification_weight': 3.0,
        'regression_weight': 1.0,
        'conservation_weight': 0.3,
        'consistency_weight': 1.5,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'huber_delta': 1.0,
        'enable_adaptive_weighting': False,
        'enable_scheduling': False
    },
    'regression_focused': {
        'classification_weight': 1.5,
        'regression_weight': 2.0,
        'conservation_weight': 0.8,
        'consistency_weight': 0.8,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'huber_delta': 1.0,
        'enable_adaptive_weighting': False,
        'enable_scheduling': False
    },
    'adaptive': {
        'classification_weight': 2.0,
        'regression_weight': 1.0,
        'conservation_weight': 0.5,
        'consistency_weight': 1.0,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'huber_delta': 1.0,
        'enable_adaptive_weighting': True,
        'enable_scheduling': False
    },
    'scheduled': {
        'classification_weight': 2.0,
        'regression_weight': 1.0,
        'conservation_weight': 0.5,
        'consistency_weight': 1.0,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'huber_delta': 1.0,
        'enable_adaptive_weighting': False,
        'enable_scheduling': True
    },
    'full_dynamic': {
        'classification_weight': 2.0,
        'regression_weight': 1.0,
        'conservation_weight': 0.5,
        'consistency_weight': 1.0,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'huber_delta': 1.0,
        'enable_adaptive_weighting': True,
        'enable_scheduling': True
    }
}