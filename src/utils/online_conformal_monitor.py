"""
在线Conformal Prediction监控器
用于实时监控预测区间的覆盖率和校准质量
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque, defaultdict
import time
import logging
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class OnlineMetrics:
    """在线监控指标"""
    coverage_rate: float
    interval_width: float
    calibration_error: float
    sample_count: int
    timestamp: float

class OnlineConformalMonitor:
    """在线Conformal Prediction监控器"""
    
    def __init__(
        self,
        device_names: List[str],
        alpha: float = 0.1,
        window_size: int = 1000,
        alert_threshold: float = 0.05,
        log_interval: int = 100
    ):
        """
        初始化在线监控器
        
        Args:
            device_names: 设备名称列表
            alpha: 置信水平
            window_size: 滑动窗口大小
            alert_threshold: 告警阈值
            log_interval: 日志记录间隔
        """
        self.device_names = device_names
        self.alpha = alpha
        self.target_coverage = 1 - alpha
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.log_interval = log_interval
        
        # 为每个设备维护滑动窗口
        self.regression_windows = {
            device: {
                'predictions': deque(maxlen=window_size),
                'targets': deque(maxlen=window_size),
                'intervals': deque(maxlen=window_size),
                'coverage': deque(maxlen=window_size),
                'widths': deque(maxlen=window_size)
            }
            for device in device_names
        }
        
        self.classification_windows = {
            device: {
                'predictions': deque(maxlen=window_size),
                'targets': deque(maxlen=window_size),
                'calibrated_probs': deque(maxlen=window_size),
                'confidence': deque(maxlen=window_size)
            }
            for device in device_names
        }
        
        # 统计计数器
        self.sample_counts = defaultdict(int)
        self.alert_counts = defaultdict(int)
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
    def update_regression(
        self,
        device_name: str,
        prediction: float,
        target: float,
        interval: Tuple[float, float]
    ) -> Optional[Dict[str, Any]]:
        """
        更新回归任务监控数据
        
        Args:
            device_name: 设备名称
            prediction: 预测值
            target: 真实值
            interval: 预测区间 (lower, upper)
            
        Returns:
            监控指标字典（如果触发更新）
        """
        if device_name not in self.regression_windows:
            return None
            
        window = self.regression_windows[device_name]
        
        # 添加新数据
        window['predictions'].append(prediction)
        window['targets'].append(target)
        window['intervals'].append(interval)
        
        # 计算覆盖率
        lower, upper = interval
        is_covered = lower <= target <= upper
        window['coverage'].append(is_covered)
        
        # 计算区间宽度
        width = upper - lower
        window['widths'].append(width)
        
        self.sample_counts[device_name] += 1
        
        # 每隔一定样本数计算指标
        if self.sample_counts[device_name] % self.log_interval == 0:
            return self._compute_regression_metrics(device_name)
            
        return None
    
    def update_classification(
        self,
        device_name: str,
        prediction: torch.Tensor,
        target: int,
        calibrated_prob: float
    ) -> Optional[Dict[str, Any]]:
        """
        更新分类任务监控数据
        
        Args:
            device_name: 设备名称
            prediction: 预测概率分布
            target: 真实标签
            calibrated_prob: 校准后的置信度
            
        Returns:
            监控指标字典（如果触发更新）
        """
        if device_name not in self.classification_windows:
            return None
            
        window = self.classification_windows[device_name]
        
        # 添加新数据
        window['predictions'].append(prediction.cpu().numpy())
        window['targets'].append(target)
        window['calibrated_probs'].append(calibrated_prob)
        
        # 计算置信度
        confidence = torch.max(prediction).item()
        window['confidence'].append(confidence)
        
        self.sample_counts[f"{device_name}_cls"] += 1
        
        # 每隔一定样本数计算指标
        if self.sample_counts[f"{device_name}_cls"] % self.log_interval == 0:
            return self._compute_classification_metrics(device_name)
            
        return None
    
    def _compute_regression_metrics(self, device_name: str) -> Dict[str, Any]:
        """计算回归任务监控指标"""
        window = self.regression_windows[device_name]
        
        if not window['coverage']:
            return {}
            
        # 计算覆盖率
        coverage_rate = np.mean(window['coverage'])
        
        # 计算平均区间宽度
        avg_width = np.mean(window['widths'])
        
        # 计算校准误差
        calibration_error = abs(coverage_rate - self.target_coverage)
        
        metrics = {
            'coverage_rate': coverage_rate,
            'interval_width': avg_width,
            'calibration_error': calibration_error,
            'sample_count': len(window['coverage']),
            'timestamp': time.time()
        }
        
        # 检查是否需要告警
        if calibration_error > self.alert_threshold:
            self.alert_counts[device_name] += 1
            self.logger.warning(
                f"Coverage alert for {device_name}: "
                f"coverage={coverage_rate:.3f}, target={self.target_coverage:.3f}, "
                f"error={calibration_error:.3f}"
            )
            metrics['alert'] = True
        else:
            metrics['alert'] = False
            
        return metrics
    
    def _compute_classification_metrics(self, device_name: str) -> Dict[str, Any]:
        """计算分类任务监控指标"""
        window = self.classification_windows[device_name]
        
        if not window['calibrated_probs']:
            return {}
            
        # 计算平均校准置信度
        avg_calibrated_prob = np.mean(window['calibrated_probs'])
        
        # 计算平均原始置信度
        avg_confidence = np.mean(window['confidence'])
        
        # 计算校准改进
        calibration_improvement = avg_calibrated_prob - avg_confidence
        
        metrics = {
            'avg_calibrated_prob': avg_calibrated_prob,
            'avg_confidence': avg_confidence,
            'calibration_improvement': calibration_improvement,
            'sample_count': len(window['calibrated_probs']),
            'timestamp': time.time()
        }
        
        return metrics
    
    def get_current_status(self) -> Dict[str, Dict[str, Any]]:
        """获取当前监控状态"""
        status = {}
        
        # 回归任务状态
        for device_name in self.device_names:
            if device_name in self.regression_windows:
                metrics = self._compute_regression_metrics(device_name)
                if metrics:
                    status[f"{device_name}_regression"] = metrics
                    
            if device_name in self.classification_windows:
                metrics = self._compute_classification_metrics(device_name)
                if metrics:
                    status[f"{device_name}_classification"] = metrics
                    
        return status
    
    def save_monitoring_report(self, output_path: str) -> None:
        """保存监控报告"""
        report = {
            'config': {
                'device_names': self.device_names,
                'alpha': self.alpha,
                'target_coverage': self.target_coverage,
                'window_size': self.window_size,
                'alert_threshold': self.alert_threshold
            },
            'current_status': self.get_current_status(),
            'sample_counts': dict(self.sample_counts),
            'alert_counts': dict(self.alert_counts),
            'timestamp': time.time()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        self.logger.info(f"Monitoring report saved to {output_path}")
    
    def reset_windows(self) -> None:
        """重置所有监控窗口"""
        for device_name in self.device_names:
            if device_name in self.regression_windows:
                for key in self.regression_windows[device_name]:
                    self.regression_windows[device_name][key].clear()
                    
            if device_name in self.classification_windows:
                for key in self.classification_windows[device_name]:
                    self.classification_windows[device_name][key].clear()
                    
        self.sample_counts.clear()
        self.alert_counts.clear()
        
        self.logger.info("Monitoring windows reset")

class ConformalAlertSystem:
    """Conformal Prediction告警系统"""
    
    def __init__(
        self,
        alert_config: Dict[str, Any],
        notification_channels: Optional[List[str]] = None
    ):
        """
        初始化告警系统
        
        Args:
            alert_config: 告警配置
            notification_channels: 通知渠道列表
        """
        self.alert_config = alert_config
        self.notification_channels = notification_channels or []
        
        # 告警阈值
        self.coverage_threshold = alert_config.get('coverage_threshold', 0.05)
        self.width_threshold = alert_config.get('width_threshold', 2.0)
        self.consecutive_alerts = alert_config.get('consecutive_alerts', 3)
        
        # 告警状态跟踪
        self.alert_history = defaultdict(list)
        self.consecutive_counts = defaultdict(int)
        
        self.logger = logging.getLogger(__name__)
    
    def check_alerts(
        self,
        device_name: str,
        metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        检查告警条件
        
        Args:
            device_name: 设备名称
            metrics: 监控指标
            
        Returns:
            告警列表
        """
        alerts = []
        current_time = time.time()
        
        # 检查覆盖率告警
        if 'coverage_rate' in metrics:
            coverage_error = abs(metrics['coverage_rate'] - (1 - metrics.get('alpha', 0.1)))
            if coverage_error > self.coverage_threshold:
                alert = {
                    'type': 'coverage_deviation',
                    'device': device_name,
                    'severity': 'high' if coverage_error > 2 * self.coverage_threshold else 'medium',
                    'message': f"Coverage rate deviation: {coverage_error:.3f}",
                    'timestamp': current_time,
                    'metrics': metrics
                }
                alerts.append(alert)
                self.consecutive_counts[f"{device_name}_coverage"] += 1
            else:
                self.consecutive_counts[f"{device_name}_coverage"] = 0
        
        # 检查区间宽度告警
        if 'interval_width' in metrics:
            if metrics['interval_width'] > self.width_threshold:
                alert = {
                    'type': 'wide_intervals',
                    'device': device_name,
                    'severity': 'medium',
                    'message': f"Interval width too large: {metrics['interval_width']:.3f}",
                    'timestamp': current_time,
                    'metrics': metrics
                }
                alerts.append(alert)
        
        # 检查连续告警
        for alert_type in ['coverage']:
            key = f"{device_name}_{alert_type}"
            if self.consecutive_counts[key] >= self.consecutive_alerts:
                alert = {
                    'type': 'consecutive_alerts',
                    'device': device_name,
                    'severity': 'critical',
                    'message': f"Consecutive {alert_type} alerts: {self.consecutive_counts[key]}",
                    'timestamp': current_time,
                    'metrics': metrics
                }
                alerts.append(alert)
                # 重置计数器
                self.consecutive_counts[key] = 0
        
        # 记录告警历史
        for alert in alerts:
            self.alert_history[device_name].append(alert)
            self._send_notification(alert)
        
        return alerts
    
    def _send_notification(self, alert: Dict[str, Any]) -> None:
        """发送告警通知"""
        message = f"[{alert['severity'].upper()}] {alert['type']}: {alert['message']}"
        
        # 记录到日志
        if alert['severity'] == 'critical':
            self.logger.critical(message)
        elif alert['severity'] == 'high':
            self.logger.error(message)
        else:
            self.logger.warning(message)
        
        # 发送到其他通知渠道
        for channel in self.notification_channels:
            try:
                self._send_to_channel(channel, alert)
            except Exception as e:
                self.logger.error(f"Failed to send alert to {channel}: {e}")
    
    def _send_to_channel(self, channel: str, alert: Dict[str, Any]) -> None:
        """发送告警到指定渠道"""
        # 这里可以实现具体的通知渠道逻辑
        # 例如：邮件、短信、Slack、钉钉等
        pass
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """获取告警摘要"""
        summary = {
            'total_alerts': sum(len(alerts) for alerts in self.alert_history.values()),
            'devices_with_alerts': len(self.alert_history),
            'consecutive_counts': dict(self.consecutive_counts),
            'recent_alerts': []
        }
        
        # 获取最近的告警
        all_alerts = []
        for device_alerts in self.alert_history.values():
            all_alerts.extend(device_alerts)
        
        # 按时间排序，取最近10个
        all_alerts.sort(key=lambda x: x['timestamp'], reverse=True)
        summary['recent_alerts'] = all_alerts[:10]
        
        return summary