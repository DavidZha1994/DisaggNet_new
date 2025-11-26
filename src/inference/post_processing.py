"""
推理后处理模块
实现工业级稳定预测的后处理策略，包括滞回阈值、最短持续时间过滤和重叠窗口融合
"""

import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging


@dataclass
class PostProcessingConfig:
    """后处理配置参数"""
    
    # 滞回阈值参数
    switch_on_threshold: float = 0.6    # 开启阈值
    switch_off_threshold: float = 0.4   # 关闭阈值
    power_on_threshold: float = 0.1     # 功率开启阈值
    power_off_threshold: float = 0.05   # 功率关闭阈值
    
    # 最短持续时间过滤（以时间步为单位）
    min_on_duration: int = 3            # 最短开机时长
    min_off_duration: int = 2           # 最短关机时长
    
    # 重叠窗口融合参数
    overlap_ratio: float = 0.5          # 重叠比例
    fusion_method: str = 'weighted'     # 融合方法: 'weighted', 'average', 'max'
    
    # 平滑参数
    temporal_smoothing: bool = True     # 是否启用时间平滑
    smoothing_window: int = 3           # 平滑窗口大小
    
    # 能量守恒后处理
    enforce_conservation: bool = True   # 是否强制能量守恒
    conservation_tolerance: float = 0.1  # 能量守恒容忍度


class HysteresisFilter:
    """滞回阈值过滤器，避免开关状态抖动"""
    
    def __init__(self, config: PostProcessingConfig):
        self.config = config
        self.previous_states = {}  # 存储每个设备的前一状态
        
    def apply(self, switch_probs: torch.Tensor, 
              power_preds: torch.Tensor,
              device_ids: Optional[List[str]] = None) -> torch.Tensor:
        """
        应用滞回阈值过滤
        
        Args:
            switch_probs: 开关概率预测 (batch_size, n_devices) 或 (n_devices,)
            power_preds: 功率预测 (batch_size, n_devices) 或 (n_devices,)
            device_ids: 设备ID列表，用于状态跟踪
            
        Returns:
            filtered_states: 过滤后的开关状态 (batch_size, n_devices) 或 (n_devices,)
        """
        if switch_probs.dim() == 1:
            switch_probs = switch_probs.unsqueeze(0)
            power_preds = power_preds.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        batch_size, n_devices = switch_probs.shape
        filtered_states = torch.zeros_like(switch_probs)
        
        for b in range(batch_size):
            for d in range(n_devices):
                device_key = f"{device_ids[d] if device_ids else d}_{b}"
                
                current_prob = switch_probs[b, d].item()
                current_power = power_preds[b, d].item()
                
                # 获取前一状态
                prev_state = self.previous_states.get(device_key, 0.0)
                
                # 应用滞回逻辑
                if prev_state < 0.5:  # 前一状态为关闭
                    # 需要同时满足概率和功率的开启条件
                    if (
                        current_prob > self.config.switch_on_threshold
                        and current_power > self.config.power_on_threshold
                    ):
                        new_state = 1.0
                    else:
                        new_state = 0.0
                else:  # 前一状态为开启
                    # 需要同时满足概率和功率的关闭条件
                    if (
                        current_prob < self.config.switch_off_threshold
                        and current_power < self.config.power_off_threshold
                    ):
                        new_state = 0.0
                    else:
                        new_state = 1.0
                
                filtered_states[b, d] = new_state
                self.previous_states[device_key] = new_state
        
        return filtered_states.squeeze(0) if squeeze_output else filtered_states


class DurationFilter:
    """最短持续时间过滤器"""
    
    def __init__(self, config: PostProcessingConfig):
        self.config = config
        self.state_history = {}  # 存储每个设备的状态历史
        
    def apply(self, switch_states: torch.Tensor,
              device_ids: Optional[List[str]] = None) -> torch.Tensor:
        """
        应用最短持续时间过滤
        
        Args:
            switch_states: 开关状态 (batch_size, n_devices) 或 (n_devices,)
            device_ids: 设备ID列表
            
        Returns:
            filtered_states: 过滤后的开关状态
        """
        if switch_states.dim() == 1:
            switch_states = switch_states.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        batch_size, n_devices = switch_states.shape
        filtered_states = switch_states.clone()
        
        for b in range(batch_size):
            for d in range(n_devices):
                device_key = f"{device_ids[d] if device_ids else d}_{b}"
                
                current_state = switch_states[b, d].item()
                
                # 获取状态历史
                if device_key not in self.state_history:
                    self.state_history[device_key] = []
                
                history = self.state_history[device_key]
                history.append(current_state)
                
                # 保持历史长度在合理范围内
                max_history = max(self.config.min_on_duration, self.config.min_off_duration) + 5
                if len(history) > max_history:
                    history.pop(0)
                
                # 应用最短持续时间过滤
                if len(history) >= 2:
                    filtered_state = self._filter_duration(history, current_state)
                    filtered_states[b, d] = filtered_state
        
        return filtered_states.squeeze(0) if squeeze_output else filtered_states
    
    def _filter_duration(self, history: List[float], current_state: float) -> float:
        """应用持续时间过滤逻辑"""
        if len(history) < 2:
            return current_state
            
        # 检查状态变化
        prev_state = history[-2]
        
        if prev_state != current_state:  # 状态发生变化
            if current_state > 0.5:  # 从关闭到开启
                # 检查前面的关闭持续时间
                off_duration = self._count_consecutive_states(history[:-1], 0.0, reverse=True)
                if off_duration < self.config.min_off_duration:
                    return prev_state  # 保持前一状态
            else:  # 从开启到关闭
                # 检查前面的开启持续时间
                on_duration = self._count_consecutive_states(history[:-1], 1.0, reverse=True)
                if on_duration < self.config.min_on_duration:
                    return prev_state  # 保持前一状态
        
        return current_state
    
    def _count_consecutive_states(self, history: List[float], target_state: float, 
                                reverse: bool = False) -> int:
        """计算连续状态的持续时间"""
        if not history:
            return 0
            
        sequence = reversed(history) if reverse else history
        count = 0
        
        for state in sequence:
            if abs(state - target_state) < 0.5:
                count += 1
            else:
                break
                
        return count


class OverlapWindowFusion:
    """重叠窗口融合器"""
    
    def __init__(self, config: PostProcessingConfig):
        self.config = config
        self.window_buffer = {}  # 存储重叠窗口的预测结果
        
    def add_prediction(self, window_id: str, 
                      switch_preds: torch.Tensor,
                      power_preds: torch.Tensor,
                      window_start: int,
                      window_end: int):
        """
        添加窗口预测结果
        
        Args:
            window_id: 窗口标识
            switch_preds: 开关预测 (seq_len, n_devices)
            power_preds: 功率预测 (seq_len, n_devices)
            window_start: 窗口开始位置
            window_end: 窗口结束位置
        """
        self.window_buffer[window_id] = {
            'switch_preds': switch_preds,
            'power_preds': power_preds,
            'start': window_start,
            'end': window_end
        }
    
    def fuse_overlapping_windows(self, target_start: int, target_end: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        融合重叠窗口的预测结果
        
        Args:
            target_start: 目标时间段开始
            target_end: 目标时间段结束
            
        Returns:
            fused_switch_preds: 融合后的开关预测
            fused_power_preds: 融合后的功率预测
        """
        overlapping_windows = []
        
        # 找到所有重叠的窗口
        for window_id, window_data in self.window_buffer.items():
            if (window_data['start'] < target_end and window_data['end'] > target_start):
                overlapping_windows.append(window_data)
        
        if not overlapping_windows:
            raise ValueError("No overlapping windows found for the target range")
        
        # 计算融合结果
        target_length = target_end - target_start
        n_devices = overlapping_windows[0]['switch_preds'].shape[1]
        
        fused_switch = torch.zeros(target_length, n_devices)
        fused_power = torch.zeros(target_length, n_devices)
        weight_sum = torch.zeros(target_length, n_devices)
        
        for window_data in overlapping_windows:
            # 计算重叠区域
            overlap_start = max(target_start, window_data['start'])
            overlap_end = min(target_end, window_data['end'])
            
            if overlap_start >= overlap_end:
                continue
                
            # 计算在目标序列和窗口序列中的索引
            target_idx_start = overlap_start - target_start
            target_idx_end = overlap_end - target_start
            window_idx_start = overlap_start - window_data['start']
            window_idx_end = overlap_end - window_data['start']
            
            # 计算权重（距离窗口中心越近权重越大）
            window_center = (window_data['start'] + window_data['end']) / 2
            distances = torch.abs(torch.arange(overlap_start, overlap_end, dtype=torch.float) - window_center)
            weights = torch.exp(-distances / (window_data['end'] - window_data['start']) * 4)  # 高斯权重
            weights = weights.unsqueeze(1).expand(-1, n_devices)
            
            # 累加加权预测
            switch_segment = window_data['switch_preds'][window_idx_start:window_idx_end]
            power_segment = window_data['power_preds'][window_idx_start:window_idx_end]
            
            fused_switch[target_idx_start:target_idx_end] += weights * switch_segment
            fused_power[target_idx_start:target_idx_end] += weights * power_segment
            weight_sum[target_idx_start:target_idx_end] += weights
        
        # 归一化
        fused_switch = fused_switch / (weight_sum + 1e-8)
        fused_power = fused_power / (weight_sum + 1e-8)
        
        return fused_switch, fused_power


class TemporalSmoother:
    """时间平滑器"""
    
    def __init__(self, config: PostProcessingConfig):
        self.config = config
        
    def apply(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        应用时间平滑
        
        Args:
            predictions: 预测结果 (seq_len, n_devices)
            
        Returns:
            smoothed_predictions: 平滑后的预测结果
        """
        if not self.config.temporal_smoothing:
            return predictions
            
        window_size = self.config.smoothing_window
        if window_size <= 1:
            return predictions
            
        # 使用移动平均进行平滑
        kernel = torch.ones(window_size) / window_size
        
        smoothed = torch.zeros_like(predictions)
        for d in range(predictions.shape[1]):
            smoothed[:, d] = torch.conv1d(
                predictions[:, d].unsqueeze(0).unsqueeze(0),
                kernel.unsqueeze(0).unsqueeze(0),
                padding=window_size // 2
            ).squeeze()
            
        return smoothed


class ConservationEnforcer:
    """能量守恒强制器"""
    
    def __init__(self, config: PostProcessingConfig):
        self.config = config
        
    def apply(self, power_preds: torch.Tensor, 
              switch_states: torch.Tensor,
              total_power: torch.Tensor) -> torch.Tensor:
        """
        强制能量守恒约束
        
        Args:
            power_preds: 功率预测 (seq_len, n_devices)
            switch_states: 开关状态 (seq_len, n_devices)
            total_power: 总功率 (seq_len,)
            
        Returns:
            adjusted_power: 调整后的功率预测
        """
        if not self.config.enforce_conservation:
            return power_preds
            
        adjusted_power = power_preds.clone()
        
        for t in range(power_preds.shape[0]):
            active_devices = switch_states[t] > 0.5
            if active_devices.sum() == 0:
                continue
                
            current_total = (power_preds[t] * switch_states[t]).sum()
            target_total = total_power[t]
            
            # 如果偏差超过容忍度，进行调整
            if abs(current_total - target_total) > self.config.conservation_tolerance * target_total:
                # 按比例调整活跃设备的功率
                scale_factor = target_total / (current_total + 1e-8)
                adjusted_power[t, active_devices] *= scale_factor
                
        return adjusted_power


class InferencePostProcessor:
    """推理后处理主类"""
    
    def __init__(self, config: PostProcessingConfig):
        self.config = config
        self.hysteresis_filter = HysteresisFilter(config)
        self.duration_filter = DurationFilter(config)
        self.overlap_fusion = OverlapWindowFusion(config)
        self.temporal_smoother = TemporalSmoother(config)
        self.conservation_enforcer = ConservationEnforcer(config)
        
        self.logger = logging.getLogger(__name__)
        
    def process_single_window(self, 
                            switch_probs: torch.Tensor,
                            power_preds: torch.Tensor,
                            total_power: Optional[torch.Tensor] = None,
                            device_ids: Optional[List[str]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        处理单个窗口的预测结果
        
        Args:
            switch_probs: 开关概率预测 (seq_len, n_devices)
            power_preds: 功率预测 (seq_len, n_devices)
            total_power: 总功率 (seq_len,)
            device_ids: 设备ID列表
            
        Returns:
            final_switch_states: 最终开关状态
            final_power_preds: 最终功率预测
        """
        seq_len, n_devices = switch_probs.shape
        
        # 1. 时间平滑
        smoothed_switch_probs = self.temporal_smoother.apply(switch_probs)
        smoothed_power_preds = self.temporal_smoother.apply(power_preds)
        
        # 2. 逐时间步应用滞回阈值过滤
        switch_states = torch.zeros_like(smoothed_switch_probs)
        for t in range(seq_len):
            switch_states[t] = self.hysteresis_filter.apply(
                smoothed_switch_probs[t], 
                smoothed_power_preds[t],
                device_ids
            )
        
        # 3. 应用最短持续时间过滤
        for t in range(seq_len):
            switch_states[t] = self.duration_filter.apply(
                switch_states[t],
                device_ids
            )
        
        # 4. 强制能量守恒（如果提供了总功率）
        final_power_preds = smoothed_power_preds
        if total_power is not None:
            final_power_preds = self.conservation_enforcer.apply(
                smoothed_power_preds,
                switch_states,
                total_power
            )
        
        return switch_states, final_power_preds
    
    def process_overlapping_windows(self,
                                  window_predictions: List[Dict],
                                  target_start: int,
                                  target_end: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        处理重叠窗口的预测结果
        
        Args:
            window_predictions: 窗口预测结果列表
            target_start: 目标时间段开始
            target_end: 目标时间段结束
            
        Returns:
            final_switch_states: 最终开关状态
            final_power_preds: 最终功率预测
        """
        # 添加所有窗口预测到缓冲区
        for i, pred in enumerate(window_predictions):
            self.overlap_fusion.add_prediction(
                window_id=f"window_{i}",
                switch_preds=pred['switch_probs'],
                power_preds=pred['power_preds'],
                window_start=pred['start'],
                window_end=pred['end']
            )
        
        # 融合重叠窗口
        fused_switch, fused_power = self.overlap_fusion.fuse_overlapping_windows(
            target_start, target_end
        )
        
        # 应用后处理
        final_switch, final_power = self.process_single_window(
            fused_switch,
            fused_power,
            total_power=window_predictions[0].get('total_power'),
            device_ids=window_predictions[0].get('device_ids')
        )
        
        return final_switch, final_power
    
    def reset_state(self):
        """重置所有状态缓存"""
        self.hysteresis_filter.previous_states.clear()
        self.duration_filter.state_history.clear()
        self.overlap_fusion.window_buffer.clear()
        
        self.logger.info("Post-processor state reset")


def create_post_processor(config_dict: Dict) -> InferencePostProcessor:
    """
    根据配置字典创建后处理器
    
    Args:
        config_dict: 配置字典
        
    Returns:
        post_processor: 配置好的后处理器
    """
    config = PostProcessingConfig(**config_dict)
    return InferencePostProcessor(config)


# 推荐的后处理配置
RECOMMENDED_POST_PROCESSING_CONFIGS = {
    'industrial_stable': {
        'switch_on_threshold': 0.65,
        'switch_off_threshold': 0.35,
        'power_on_threshold': 0.1,
        'power_off_threshold': 0.05,
        'min_on_duration': 5,
        'min_off_duration': 3,
        'overlap_ratio': 0.5,
        'fusion_method': 'weighted',
        'temporal_smoothing': True,
        'smoothing_window': 3,
        'enforce_conservation': True,
        'conservation_tolerance': 0.1
    },
    'responsive': {
        'switch_on_threshold': 0.55,
        'switch_off_threshold': 0.45,
        'power_on_threshold': 0.05,
        'power_off_threshold': 0.02,
        'min_on_duration': 2,
        'min_off_duration': 1,
        'overlap_ratio': 0.3,
        'fusion_method': 'average',
        'temporal_smoothing': False,
        'smoothing_window': 1,
        'enforce_conservation': False,
        'conservation_tolerance': 0.15
    },
    'conservative': {
        'switch_on_threshold': 0.75,
        'switch_off_threshold': 0.25,
        'power_on_threshold': 0.15,
        'power_off_threshold': 0.08,
        'min_on_duration': 8,
        'min_off_duration': 5,
        'overlap_ratio': 0.7,
        'fusion_method': 'weighted',
        'temporal_smoothing': True,
        'smoothing_window': 5,
        'enforce_conservation': True,
        'conservation_tolerance': 0.05
    }
}
