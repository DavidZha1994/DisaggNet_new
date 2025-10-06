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
        
        residual = torch.abs(inputs - targets)
        
        # Huber损失：当残差小于delta时使用平方损失，否则使用线性损失
        huber_loss = torch.where(
            residual < self.delta,
            0.5 * residual ** 2,
            self.delta * (residual - 0.5 * self.delta)
        )
        
        if self.device_weights is not None:
            # 应用设备权重
            weights = self.device_weights.to(inputs.device).unsqueeze(0)
            huber_loss = huber_loss * weights
        
        if self.reduction == 'mean':
            return huber_loss.mean()
        elif self.reduction == 'sum':
            return huber_loss.sum()
        else:
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


class MultiTaskLoss(nn.Module):
    """多任务损失函数集成"""
    
    def __init__(self, config: DictConfig, device_info: Dict, n_devices: int):
        super().__init__()
        
        self.config = config
        self.n_devices = n_devices
        
        # 损失权重
        self.regression_weight = config.weights.regression
        self.classification_weight = config.weights.classification
        self.consistency_weight_schedule = self._create_weight_schedule(
            config.weights.consistency_schedule
        )
        
        # 设备权重（处理不平衡）
        device_weights = None
        if config.device_weights.enable:
            weights = [device_info.get(i, {}).get('weight', 1.0) for i in range(n_devices)]
            device_weights = torch.tensor(weights, dtype=torch.float32)
        
        # 回归损失
        if config.regression.type == 'mae':
            self.regression_loss = MAELoss(device_weights, reduction='mean')
        elif config.regression.type == 'mse':
            self.regression_loss = WeightedMSELoss(device_weights, reduction='mean')
        elif config.regression.type == 'huber':
            self.regression_loss = HuberLoss(
                delta=config.regression.huber_delta, 
                device_weights=device_weights, 
                reduction='mean'
            )
        elif config.regression.type == 'quantile':
            self.regression_loss = QuantileLoss(
                quantiles=config.regression.quantiles,
                reduction='mean'
            )
        else:
            raise ValueError(f"Unknown regression loss type: {config.regression.type}")
        
        # 分类损失
        if config.classification.type == 'bce':
            # 设备无关的pos_weight处理：注册为buffer以随模块迁移设备
            pos_weights = None
            if config.classification.pos_weight.enable:
                pos_weights = torch.tensor([
                    device_info.get(i, {}).get('pos_weight', 1.0) for i in range(n_devices)
                ], dtype=torch.float32)
            if pos_weights is not None:
                self.register_buffer('pos_weight', pos_weights)
            else:
                self.pos_weight = None
            # 使用函数式binary_cross_entropy_with_logits在forward中计算，避免设备不匹配
            self.classification_loss = None
        elif config.classification.type == 'focal':
            self.classification_loss = FocalLoss(
                alpha=config.classification.focal_alpha,
                gamma=config.classification.focal_gamma,
                reduction='mean'
            )
        else:
            raise ValueError(f"Unknown classification loss type: {config.classification.type}")
        
        # 先验损失（如果启用）
        self.use_priors = config.priors.enable
        if self.use_priors:
            from ..models.priors import PriorKnowledgeIntegrator
            self.prior_integrator = PriorKnowledgeIntegrator(
                config.priors, device_info, n_devices
            )
        
        # 当前epoch（用于动态权重调整）
        self.register_buffer('current_epoch', torch.tensor(0))
    
    def _create_weight_schedule(self, schedule_config: DictConfig) -> Dict:
        """创建权重调度"""
        return {
            'type': schedule_config.type,
            'initial': schedule_config.initial,
            'final': schedule_config.final,
            'warmup_epochs': schedule_config.warmup_epochs
        }
    
    def _get_consistency_weight(self) -> float:
        """获取当前epoch的一致性权重"""
        epoch = self.current_epoch.item()
        schedule = self.consistency_weight_schedule
        
        if schedule['type'] == 'linear':
            if epoch < schedule['warmup_epochs']:
                # 线性增长
                weight = schedule['initial'] + (schedule['final'] - schedule['initial']) * \
                        (epoch / schedule['warmup_epochs'])
            else:
                weight = schedule['final']
        elif schedule['type'] == 'exponential':
            if epoch < schedule['warmup_epochs']:
                # 指数增长
                progress = epoch / schedule['warmup_epochs']
                weight = schedule['initial'] * (schedule['final'] / schedule['initial']) ** progress
            else:
                weight = schedule['final']
        else:
            weight = schedule['final']
        
        return weight
    
    def forward(self, predictions: Tuple[torch.Tensor, torch.Tensor], 
                targets: Tuple[torch.Tensor, torch.Tensor],
                historical_power: Optional[torch.Tensor] = None,
                window_length: int = 256) -> Dict[str, torch.Tensor]:
        """计算多任务损失"""
        
        pred_power, pred_states = predictions
        target_power, target_states = targets
        
        # 数值稳定性检查
        if torch.isnan(pred_power).any() or torch.isnan(pred_states).any():
            print("Warning: NaN detected in model predictions")
            return {
                'total': torch.tensor(0.0, device=pred_power.device, requires_grad=True),
                'regression': torch.tensor(0.0, device=pred_power.device),
                'classification': torch.tensor(0.0, device=pred_power.device)
            }
        
        losses = {}
        
        # 回归损失（功率预测）
        regression_loss = self.regression_loss(pred_power, target_power)
        # 损失值范围保护
        regression_loss = torch.clamp(regression_loss, max=100.0)
        losses['regression'] = self.regression_weight * regression_loss
        
        # 分类损失（开关状态）
        if self.config.classification.type == 'bce':
            # 直接在此处计算BCE with logits，确保pos_weight与设备一致
            pos_weight = getattr(self, 'pos_weight', None)
            classification_loss = F.binary_cross_entropy_with_logits(
                pred_states, target_states,
                pos_weight=pos_weight if pos_weight is not None else None,
                reduction='mean'
            )
        else:
            # 其他损失（如FocalLoss）同样接收logits
            classification_loss = self.classification_loss(pred_states, target_states)
        # 损失值范围保护
        classification_loss = torch.clamp(classification_loss, max=100.0)
        losses['classification'] = self.classification_weight * classification_loss
        
        # 先验损失
        if self.use_priors:
            consistency_weight = self._get_consistency_weight()
            prior_losses = self.prior_integrator(
                pred_power, pred_states, target_power, target_states,
                historical_power, window_length
            )
            
            for loss_name, loss_value in prior_losses.items():
                # 先验损失也需要数值保护
                loss_value = torch.clamp(loss_value, max=100.0)
                if loss_name == 'consistency':
                    losses[f'prior_{loss_name}'] = consistency_weight * loss_value
                else:
                    # 其他先验损失使用配置中的权重
                    weight = getattr(self.config.priors.weights, loss_name, 1.0)
                    losses[f'prior_{loss_name}'] = weight * loss_value
        
        # 总损失
        total_loss = sum(losses.values())
        
        # 最终检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Warning: Invalid total loss detected: {total_loss}")
            total_loss = torch.tensor(0.0, device=total_loss.device, requires_grad=True)
        
        losses['total'] = total_loss
        
        return losses
    
    def update_epoch(self, epoch: int):
        """更新当前epoch"""
        self.current_epoch.fill_(epoch)
        if self.use_priors:
            self.prior_integrator.update_epoch(epoch)


class LossScheduler:
    """损失权重调度器"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.schedules = {}
        
        # 解析各种调度配置
        for loss_name, schedule_config in config.items():
            if hasattr(schedule_config, 'schedule') and schedule_config.schedule.enable:
                self.schedules[loss_name] = {
                    'type': schedule_config.schedule.type,
                    'initial': schedule_config.schedule.initial,
                    'final': schedule_config.schedule.final,
                    'warmup_epochs': schedule_config.schedule.warmup_epochs,
                    'decay_epochs': getattr(schedule_config.schedule, 'decay_epochs', 100)
                }
    
    def get_weight(self, loss_name: str, epoch: int) -> float:
        """获取指定损失在当前epoch的权重"""
        if loss_name not in self.schedules:
            return 1.0
        
        schedule = self.schedules[loss_name]
        
        if schedule['type'] == 'linear':
            if epoch < schedule['warmup_epochs']:
                progress = epoch / schedule['warmup_epochs']
                weight = schedule['initial'] + (schedule['final'] - schedule['initial']) * progress
            else:
                weight = schedule['final']
        elif schedule['type'] == 'cosine':
            if epoch < schedule['warmup_epochs']:
                progress = epoch / schedule['warmup_epochs']
                weight = schedule['initial'] + (schedule['final'] - schedule['initial']) * \
                        (1 - np.cos(progress * np.pi)) / 2
            else:
                weight = schedule['final']
        elif schedule['type'] == 'exponential':
            if epoch < schedule['warmup_epochs']:
                progress = epoch / schedule['warmup_epochs']
                weight = schedule['initial'] * (schedule['final'] / schedule['initial']) ** progress
            else:
                # 指数衰减
                decay_progress = (epoch - schedule['warmup_epochs']) / schedule['decay_epochs']
                weight = schedule['final'] * (0.1 ** decay_progress)
        else:
            weight = schedule['final']
        
        return max(weight, 0.0)  # 确保权重非负
    
    def get_all_weights(self, epoch: int) -> Dict[str, float]:
        """获取所有损失在当前epoch的权重"""
        return {loss_name: self.get_weight(loss_name, epoch) for loss_name in self.schedules.keys()}


class AdaptiveLossWeighting(nn.Module):
    """自适应损失权重（基于梯度范数平衡）"""
    
    def __init__(self, n_tasks: int, alpha: float = 0.16):
        super().__init__()
        self.n_tasks = n_tasks
        self.alpha = alpha
        
        # 可学习的权重参数
        self.log_weights = nn.Parameter(torch.zeros(n_tasks))
        
        # 梯度范数的移动平均
        self.register_buffer('grad_norm_avg', torch.ones(n_tasks))
        self.register_buffer('update_count', torch.tensor(0))
    
    def forward(self, losses: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算自适应加权的总损失"""
        # 计算权重
        weights = F.softmax(self.log_weights, dim=0) * self.n_tasks
        
        # 加权损失
        weighted_losses = [w * loss for w, loss in zip(weights, losses)]
        total_loss = sum(weighted_losses)
        
        return total_loss, weights
    
    def update_weights(self, losses: List[torch.Tensor], shared_parameters):
        """基于梯度范数更新权重"""
        # 计算每个任务的梯度范数
        grad_norms = []
        for loss in losses:
            # 计算相对于共享参数的梯度
            grads = torch.autograd.grad(loss, shared_parameters, retain_graph=True, allow_unused=True)
            grad_norm = torch.norm(torch.cat([g.flatten() for g in grads if g is not None]))
            grad_norms.append(grad_norm)
        
        grad_norms = torch.stack(grad_norms)
        
        # 更新移动平均
        self.update_count += 1
        momentum = 1.0 / self.update_count if self.update_count < 100 else 0.01
        self.grad_norm_avg = (1 - momentum) * self.grad_norm_avg + momentum * grad_norms
        
        # 计算相对梯度范数
        relative_grad_norms = grad_norms / (self.grad_norm_avg + 1e-8)
        
        # 更新权重（梯度范数大的任务权重应该小一些）
        target_weights = 1.0 / (relative_grad_norms + 1e-8)
        target_weights = target_weights / target_weights.sum() * self.n_tasks
        
        # 软更新
        current_weights = F.softmax(self.log_weights, dim=0) * self.n_tasks
        new_weights = (1 - self.alpha) * current_weights + self.alpha * target_weights
        
        # 更新log_weights
        self.log_weights.data = torch.log(new_weights / self.n_tasks + 1e-8)