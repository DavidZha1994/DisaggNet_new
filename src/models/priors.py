"""工业先验和自适应一致性模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from omegaconf import DictConfig


class AdaptiveConsistencyLoss(nn.Module):
    """自适应一致性损失"""
    
    def __init__(self, config: DictConfig):
        super().__init__()
        
        self.config = config
        self.tolerance_init = config.tolerance_init  # 初始容忍度
        self.tolerance_decay = config.tolerance_decay  # 容忍度衰减率
        self.min_tolerance = config.min_tolerance  # 最小容忍度
        self.warmup_epochs = config.warmup_epochs  # 预热轮数
        
        # 可学习的容忍度参数（每个设备独立）
        self.register_buffer('current_epoch', torch.tensor(0))
        
    def get_adaptive_tolerance(self, n_devices: int, device: torch.device) -> torch.Tensor:
        """计算自适应容忍度"""
        epoch = self.current_epoch.item()
        
        if epoch < self.warmup_epochs:
            # 预热期间使用较大的容忍度
            tolerance = self.tolerance_init
        else:
            # 指数衰减
            decay_epochs = epoch - self.warmup_epochs
            tolerance = self.tolerance_init * (self.tolerance_decay ** decay_epochs)
            tolerance = max(tolerance, self.min_tolerance)
        
        return torch.full((n_devices,), tolerance, device=device)
    
    def forward(self, pred_power: torch.Tensor, pred_states: torch.Tensor, 
                target_power: torch.Tensor, target_states: torch.Tensor) -> torch.Tensor:
        """计算自适应一致性损失"""
        # pred_power: (batch_size, n_devices)
        # pred_states: (batch_size, n_devices)
        # target_power: (batch_size, n_devices)
        # target_states: (batch_size, n_devices)
        
        batch_size, n_devices = pred_power.size()
        device = pred_power.device
        
        # 获取自适应容忍度
        tolerance = self.get_adaptive_tolerance(n_devices, device)
        
        # 计算功率预测的总和
        pred_total = torch.sum(pred_power, dim=1)  # (batch_size,)
        target_total = torch.sum(target_power, dim=1)  # (batch_size,)
        
        # 计算一致性误差
        consistency_error = torch.abs(pred_total - target_total)  # (batch_size,)
        
        # 自适应惩罚：超出容忍度的部分进行惩罚
        avg_tolerance = torch.mean(tolerance)
        penalty = F.relu(consistency_error - avg_tolerance)
        
        # 返回平均一致性损失
        return torch.mean(penalty)
    
    def update_epoch(self, epoch: int):
        """更新当前轮数"""
        self.current_epoch.fill_(epoch)


class RampRegularizer(nn.Module):
    """爬坡正则化（针对加热类设备）"""
    
    def __init__(self, config: DictConfig, heating_device_indices: List[int]):
        super().__init__()
        
        self.config = config
        self.heating_device_indices = heating_device_indices
        self.ramp_weight = config.ramp_weight
        self.max_ramp_rate = config.max_ramp_rate  # 最大允许爬坡率
        
    def forward(self, pred_power: torch.Tensor, window_length: int) -> torch.Tensor:
        """计算爬坡正则化损失"""
        if not self.heating_device_indices or self.ramp_weight == 0:
            return torch.tensor(0.0, device=pred_power.device)
        
        # pred_power: (batch_size, n_devices)
        # 这里假设我们有时间序列的功率预测，需要在时间维度上计算梯度
        # 由于当前输出是单点预测，我们使用batch内的变化作为代理
        
        total_ramp_loss = 0.0
        
        for device_idx in self.heating_device_indices:
            if device_idx < pred_power.size(1):
                device_power = pred_power[:, device_idx]  # (batch_size,)
                
                # 计算相邻样本间的功率变化率（作为爬坡率的代理）
                if device_power.size(0) > 1:
                    power_diff = torch.diff(device_power)  # (batch_size-1,)
                    ramp_rate = torch.abs(power_diff)
                    
                    # 超出最大爬坡率的部分进行惩罚
                    ramp_penalty = F.relu(ramp_rate - self.max_ramp_rate)
                    total_ramp_loss += torch.mean(ramp_penalty)
        
        return self.ramp_weight * total_ramp_loss


class RangeRegularizer(nn.Module):
    """范围正则化（确保功率在合理范围内）"""
    
    def __init__(self, config: DictConfig, device_power_ranges: Dict[int, Tuple[float, float]]):
        super().__init__()
        
        self.config = config
        self.device_power_ranges = device_power_ranges  # {device_idx: (min_power, max_power)}
        self.range_weight = config.range_weight
        
    def forward(self, pred_power: torch.Tensor) -> torch.Tensor:
        """计算范围正则化损失"""
        if not self.device_power_ranges or self.range_weight == 0:
            return torch.tensor(0.0, device=pred_power.device)
        
        total_range_loss = 0.0
        
        for device_idx, (min_power, max_power) in self.device_power_ranges.items():
            if device_idx < pred_power.size(1):
                device_power = pred_power[:, device_idx]  # (batch_size,)
                
                # 低于最小值的惩罚
                under_penalty = F.relu(min_power - device_power)
                
                # 高于最大值的惩罚
                over_penalty = F.relu(device_power - max_power)
                
                total_range_loss += torch.mean(under_penalty + over_penalty)
        
        return self.range_weight * total_range_loss


class IndustrialPriors(nn.Module):
    """工业先验知识集成模块"""
    
    def __init__(self, config: DictConfig, device_info: Dict):
        super().__init__()
        
        self.config = config
        self.device_info = device_info
        
        # 提取设备类型信息
        self.heating_devices = [i for i, info in device_info.items() 
                               if info.get('type') == 'heating']
        self.motor_devices = [i for i, info in device_info.items() 
                             if info.get('type') == 'motor']
        
        # 提取功率范围信息
        self.power_ranges = {i: (info.get('min_power', 0), info.get('max_power', 1000)) 
                            for i, info in device_info.items()}
        
        # 初始化各种正则化器
        if config.adaptive_consistency.enable:
            self.consistency_loss = AdaptiveConsistencyLoss(config.adaptive_consistency)
        else:
            self.consistency_loss = None
            
        if config.ramp_regularizer.enable:
            self.ramp_regularizer = RampRegularizer(config.ramp_regularizer, self.heating_devices)
        else:
            self.ramp_regularizer = None
            
        if config.range_regularizer.enable:
            self.range_regularizer = RangeRegularizer(config.range_regularizer, self.power_ranges)
        else:
            self.range_regularizer = None
    
    def forward(self, pred_power: torch.Tensor, pred_states: torch.Tensor,
                target_power: torch.Tensor, target_states: torch.Tensor,
                window_length: int = 256) -> Dict[str, torch.Tensor]:
        """计算所有先验损失"""
        
        losses = {}
        
        # 自适应一致性损失
        if self.consistency_loss is not None:
            losses['consistency'] = self.consistency_loss(
                pred_power, pred_states, target_power, target_states
            )
        
        # 爬坡正则化
        if self.ramp_regularizer is not None:
            losses['ramp'] = self.ramp_regularizer(pred_power, window_length)
        
        # 范围正则化
        if self.range_regularizer is not None:
            losses['range'] = self.range_regularizer(pred_power)
        
        return losses
    
    def update_epoch(self, epoch: int):
        """更新轮数相关的参数"""
        if self.consistency_loss is not None:
            self.consistency_loss.update_epoch(epoch)


class SoftPriorConstraints(nn.Module):
    """软先验约束（可学习的约束权重）"""
    
    def __init__(self, config: DictConfig, n_devices: int):
        super().__init__()
        
        self.config = config
        self.n_devices = n_devices
        
        # 可学习的设备间相关性权重
        if config.device_correlation.enable:
            self.correlation_weights = nn.Parameter(
                torch.eye(n_devices) * config.device_correlation.init_weight
            )
        else:
            self.correlation_weights = None
        
        # 可学习的时间依赖权重
        if config.temporal_dependency.enable:
            self.temporal_weights = nn.Parameter(
                torch.ones(config.temporal_dependency.max_lag) * config.temporal_dependency.init_weight
            )
        else:
            self.temporal_weights = None
    
    def device_correlation_loss(self, pred_power: torch.Tensor) -> torch.Tensor:
        """设备间相关性损失"""
        if self.correlation_weights is None:
            return torch.tensor(0.0, device=pred_power.device)
        
        # 计算预测功率的相关性
        batch_size = pred_power.size(0)
        
        # 标准化功率预测
        pred_normalized = F.normalize(pred_power, p=2, dim=0)
        
        # 计算相关性矩阵
        correlation_matrix = torch.mm(pred_normalized.t(), pred_normalized) / batch_size
        
        # 与先验相关性的差异
        correlation_loss = F.mse_loss(correlation_matrix, self.correlation_weights)
        
        return correlation_loss
    
    def forward(self, pred_power: torch.Tensor, 
                historical_power: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """计算软先验约束损失"""
        
        losses = {}
        
        # 设备间相关性损失
        if self.correlation_weights is not None:
            losses['device_correlation'] = self.device_correlation_loss(pred_power)
        
        # 时间依赖性损失（如果有历史数据）
        if self.temporal_weights is not None and historical_power is not None:
            # 这里可以实现时间依赖性的损失计算
            # 暂时返回零损失
            losses['temporal_dependency'] = torch.tensor(0.0, device=pred_power.device)
        
        return losses


class PriorKnowledgeIntegrator(nn.Module):
    """先验知识集成器（主接口）"""
    
    def __init__(self, config: DictConfig, device_info: Dict, n_devices: int):
        super().__init__()
        
        self.config = config
        
        # 工业先验
        if config.industrial_priors.enable:
            self.industrial_priors = IndustrialPriors(config.industrial_priors, device_info)
        else:
            self.industrial_priors = None
        
        # 软约束
        if config.soft_constraints.enable:
            self.soft_constraints = SoftPriorConstraints(config.soft_constraints, n_devices)
        else:
            self.soft_constraints = None
    
    def forward(self, pred_power: torch.Tensor, pred_states: torch.Tensor,
                target_power: torch.Tensor, target_states: torch.Tensor,
                historical_power: Optional[torch.Tensor] = None,
                window_length: int = 256) -> Dict[str, torch.Tensor]:
        """计算所有先验知识损失"""
        
        all_losses = {}
        
        # 工业先验损失
        if self.industrial_priors is not None:
            industrial_losses = self.industrial_priors(
                pred_power, pred_states, target_power, target_states, window_length
            )
            all_losses.update(industrial_losses)
        
        # 软约束损失
        if self.soft_constraints is not None:
            soft_losses = self.soft_constraints(pred_power, historical_power)
            all_losses.update(soft_losses)
        
        return all_losses
    
    def update_epoch(self, epoch: int):
        """更新轮数相关的参数"""
        if self.industrial_priors is not None:
            self.industrial_priors.update_epoch(epoch)