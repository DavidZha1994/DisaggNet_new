"""评估指标模块"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import (
    f1_score, matthews_corrcoef, precision_recall_curve, roc_auc_score,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')


class NILMMetrics:
    """NILM专用评估指标"""
    
    def __init__(self, device_names: List[str], threshold_method: str = 'optimal'):
        self.device_names = device_names
        self.n_devices = len(device_names)
        self.threshold_method = threshold_method
        self.thresholds = {}

    def save_thresholds(self, filepath: str) -> None:
        """保存当前阈值到文件（JSON）"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(self.thresholds, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存阈值失败: {e}")

    def load_thresholds(self, filepath: str) -> Optional[Dict[str, float]]:
        """从文件（JSON）加载阈值并设置为当前阈值"""
        try:
            if not os.path.exists(filepath):
                return None
            with open(filepath, 'r') as f:
                thresholds = json.load(f)
            # 基于设备名过滤/校正
            valid = {name: float(thresholds.get(name, 0.5)) for name in self.device_names}
            self.thresholds = valid
            return valid
        except Exception as e:
            print(f"加载阈值失败: {e}")
            return None
        
    def _to_numpy(self, tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """转换为numpy数组"""
        if isinstance(tensor, torch.Tensor):
            # 如果是BFloat16，先转换为Float32
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            return tensor.detach().cpu().numpy()
        return tensor
    
    def _safe_divide(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """安全除法，避免除零"""
        return numerator / denominator if denominator != 0 else default
    
    def mean_absolute_error(self, y_pred: torch.Tensor, y_true: torch.Tensor, 
                           per_device: bool = False) -> Union[float, Dict[str, float]]:
        """平均绝对误差 (MAE)"""
        y_pred = self._to_numpy(y_pred)
        y_true = self._to_numpy(y_true)
        
        mae_values = np.mean(np.abs(y_pred - y_true), axis=0)
        
        if per_device:
            return {self.device_names[i]: mae_values[i] for i in range(self.n_devices)}
        else:
            return np.mean(mae_values)
    
    def normalized_disaggregation_error(self, y_pred: torch.Tensor, y_true: torch.Tensor,
                                      per_device: bool = False) -> Union[float, Dict[str, float]]:
        """归一化分解误差 (NDE)"""
        y_pred = self._to_numpy(y_pred)
        y_true = self._to_numpy(y_true)
        
        # 计算每个设备的NDE
        nde_values = []
        for i in range(self.n_devices):
            numerator = np.sum(np.abs(y_pred[:, i] - y_true[:, i]))
            denominator = np.sum(y_true[:, i])
            nde = self._safe_divide(numerator, denominator, 1.0)
            nde_values.append(nde)
        
        nde_values = np.array(nde_values)
        
        if per_device:
            return {self.device_names[i]: nde_values[i] for i in range(self.n_devices)}
        else:
            return np.mean(nde_values)
    
    def signal_aggregate_error(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        """信号聚合误差 (SAE)"""
        y_pred = self._to_numpy(y_pred)
        y_true = self._to_numpy(y_true)
        
        # 计算总功率
        pred_total = np.sum(y_pred, axis=1)
        true_total = np.sum(y_true, axis=1)
        
        # SAE
        numerator = np.sum(np.abs(pred_total - true_total))
        denominator = np.sum(true_total)
        
        return self._safe_divide(numerator, denominator, 1.0)
    
    def total_energy_correctly_assigned(self, y_pred: torch.Tensor, y_true: torch.Tensor,
                                      per_device: bool = False) -> Union[float, Dict[str, float]]:
        """总能量正确分配比例 (TECA)"""
        y_pred = self._to_numpy(y_pred)
        y_true = self._to_numpy(y_true)
        
        teca_values = []
        for i in range(self.n_devices):
            pred_energy = np.sum(y_pred[:, i])
            true_energy = np.sum(y_true[:, i])
            
            if true_energy == 0:
                teca = 1.0 if pred_energy == 0 else 0.0
            else:
                teca = 1.0 - abs(pred_energy - true_energy) / true_energy
                teca = max(0.0, teca)  # 确保非负
            
            teca_values.append(teca)
        
        teca_values = np.array(teca_values)
        
        if per_device:
            return {self.device_names[i]: teca_values[i] for i in range(self.n_devices)}
        else:
            return np.mean(teca_values)
    
    def optimize_thresholds(self, y_pred_proba: torch.Tensor, y_true: torch.Tensor,
                          method: str = 'f1') -> Dict[str, float]:
        """优化分类阈值"""
        y_pred_proba = self._to_numpy(y_pred_proba)
        y_true = self._to_numpy(y_true)
        
        optimal_thresholds = {}
        
        for i in range(self.n_devices):
            device_name = self.device_names[i]
            
            if len(np.unique(y_true[:, i])) < 2:
                # 如果只有一个类别，使用默认阈值
                optimal_thresholds[device_name] = 0.5
                continue
            
            if method == 'f1':
                # 基于F1分数优化
                precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_pred_proba[:, i])
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                best_idx = np.argmax(f1_scores)
                optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            
            elif method == 'youden':
                # 基于Youden指数优化
                from sklearn.metrics import roc_curve
                fpr, tpr, thresholds = roc_curve(y_true[:, i], y_pred_proba[:, i])
                youden_index = tpr - fpr
                best_idx = np.argmax(youden_index)
                optimal_threshold = thresholds[best_idx]
            
            else:
                optimal_threshold = 0.5
            
            optimal_thresholds[device_name] = optimal_threshold
        
        self.thresholds = optimal_thresholds
        return optimal_thresholds
    
    def apply_thresholds(self, y_pred_proba: torch.Tensor, 
                        thresholds: Optional[Dict[str, float]] = None) -> np.ndarray:
        """应用阈值进行二值化"""
        y_pred_proba = self._to_numpy(y_pred_proba)
        
        if thresholds is None:
            thresholds = self.thresholds
        
        y_pred_binary = np.zeros_like(y_pred_proba)
        
        for i, device_name in enumerate(self.device_names):
            threshold = thresholds.get(device_name, 0.5)
            y_pred_binary[:, i] = (y_pred_proba[:, i] >= threshold).astype(int)
        
        return y_pred_binary
    
    def f1_score(self, y_pred_proba: torch.Tensor, y_true: torch.Tensor,
                thresholds: Optional[Dict[str, float]] = None,
                per_device: bool = False, average: str = 'macro') -> Union[float, Dict[str, float]]:
        """F1分数"""
        y_true = self._to_numpy(y_true)
        y_pred_binary = self.apply_thresholds(y_pred_proba, thresholds)
        
        if per_device:
            f1_scores = {}
            for i, device_name in enumerate(self.device_names):
                if len(np.unique(y_true[:, i])) < 2:
                    f1_scores[device_name] = 0.0
                else:
                    f1_scores[device_name] = f1_score(y_true[:, i], y_pred_binary[:, i], average='binary')
            return f1_scores
        else:
            # 计算macro或micro平均
            if average == 'macro':
                f1_scores = []
                for i in range(self.n_devices):
                    if len(np.unique(y_true[:, i])) >= 2:
                        f1 = f1_score(y_true[:, i], y_pred_binary[:, i], average='binary')
                        f1_scores.append(f1)
                return np.mean(f1_scores) if f1_scores else 0.0
            else:
                return f1_score(y_true.flatten(), y_pred_binary.flatten(), average=average)
    
    def matthews_correlation_coefficient(self, y_pred_proba: torch.Tensor, y_true: torch.Tensor,
                                       thresholds: Optional[Dict[str, float]] = None,
                                       per_device: bool = False) -> Union[float, Dict[str, float]]:
        """马修斯相关系数 (MCC)"""
        y_true = self._to_numpy(y_true)
        y_pred_binary = self.apply_thresholds(y_pred_proba, thresholds)
        
        if per_device:
            mcc_scores = {}
            for i, device_name in enumerate(self.device_names):
                if len(np.unique(y_true[:, i])) < 2:
                    mcc_scores[device_name] = 0.0
                else:
                    try:
                        mcc = matthews_corrcoef(y_true[:, i], y_pred_binary[:, i])
                        mcc_scores[device_name] = mcc if not np.isnan(mcc) else 0.0
                    except:
                        mcc_scores[device_name] = 0.0
            return mcc_scores
        else:
            try:
                mcc = matthews_corrcoef(y_true.flatten(), y_pred_binary.flatten())
                return mcc if not np.isnan(mcc) else 0.0
            except:
                return 0.0
    
    def precision_recall_auc(self, y_pred_proba: torch.Tensor, y_true: torch.Tensor,
                           per_device: bool = False) -> Union[float, Dict[str, float]]:
        """Precision-Recall AUC"""
        y_pred_proba = self._to_numpy(y_pred_proba)
        y_true = self._to_numpy(y_true)
        
        if per_device:
            pr_auc_scores = {}
            for i, device_name in enumerate(self.device_names):
                if len(np.unique(y_true[:, i])) < 2:
                    pr_auc_scores[device_name] = 0.0
                else:
                    try:
                        from sklearn.metrics import average_precision_score
                        pr_auc = average_precision_score(y_true[:, i], y_pred_proba[:, i])
                        pr_auc_scores[device_name] = pr_auc
                    except:
                        pr_auc_scores[device_name] = 0.0
            return pr_auc_scores
        else:
            try:
                from sklearn.metrics import average_precision_score
                pr_auc_scores = []
                for i in range(self.n_devices):
                    if len(np.unique(y_true[:, i])) >= 2:
                        pr_auc = average_precision_score(y_true[:, i], y_pred_proba[:, i])
                        pr_auc_scores.append(pr_auc)
                return np.mean(pr_auc_scores) if pr_auc_scores else 0.0
            except:
                return 0.0
    
    def roc_auc(self, y_pred_proba: torch.Tensor, y_true: torch.Tensor,
               per_device: bool = False) -> Union[float, Dict[str, float]]:
        """ROC AUC"""
        y_pred_proba = self._to_numpy(y_pred_proba)
        y_true = self._to_numpy(y_true)
        
        if per_device:
            roc_auc_scores = {}
            for i, device_name in enumerate(self.device_names):
                if len(np.unique(y_true[:, i])) < 2:
                    roc_auc_scores[device_name] = 0.5
                else:
                    try:
                        roc_auc = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
                        roc_auc_scores[device_name] = roc_auc
                    except:
                        roc_auc_scores[device_name] = 0.5
            return roc_auc_scores
        else:
            try:
                roc_auc_scores = []
                for i in range(self.n_devices):
                    if len(np.unique(y_true[:, i])) >= 2:
                        roc_auc = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
                        roc_auc_scores.append(roc_auc)
                return np.mean(roc_auc_scores) if roc_auc_scores else 0.5
            except:
                return 0.5
    
    def event_detection_metrics(self, y_pred_proba: torch.Tensor, y_true: torch.Tensor,
                              thresholds: Optional[Dict[str, float]] = None,
                              tolerance_seconds: int = 30) -> Dict[str, Dict[str, float]]:
        """事件检测指标（考虑时间容忍度）"""
        y_true = self._to_numpy(y_true)
        y_pred_binary = self.apply_thresholds(y_pred_proba, thresholds)
        
        event_metrics = {}
        
        for i, device_name in enumerate(self.device_names):
            # 检测事件（状态变化）
            true_events = self._detect_events(y_true[:, i])
            pred_events = self._detect_events(y_pred_binary[:, i])
            
            # 计算事件级别的精确率、召回率
            tp, fp, fn = self._match_events(true_events, pred_events, tolerance_seconds)
            
            precision = self._safe_divide(tp, tp + fp, 0.0)
            recall = self._safe_divide(tp, tp + fn, 0.0)
            f1 = self._safe_divide(2 * precision * recall, precision + recall, 0.0)
            
            event_metrics[device_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'true_events': len(true_events),
                'pred_events': len(pred_events)
            }
        
        return event_metrics
    
    def _detect_events(self, signal: np.ndarray) -> List[Tuple[int, int]]:
        """检测信号中的事件（开启和关闭）"""
        events = []
        state = signal[0]
        start_idx = 0
        
        for i in range(1, len(signal)):
            if signal[i] != state:
                if state == 1:  # 结束一个开启事件
                    events.append((start_idx, i - 1))
                state = signal[i]
                start_idx = i
        
        # 处理最后一个事件
        if state == 1:
            events.append((start_idx, len(signal) - 1))
        
        return events
    
    def _match_events(self, true_events: List[Tuple[int, int]], 
                     pred_events: List[Tuple[int, int]], 
                     tolerance: int) -> Tuple[int, int, int]:
        """匹配真实事件和预测事件"""
        tp = 0
        matched_pred = set()
        
        for true_start, true_end in true_events:
            for j, (pred_start, pred_end) in enumerate(pred_events):
                if j in matched_pred:
                    continue
                
                # 检查是否在容忍范围内
                if (abs(true_start - pred_start) <= tolerance or 
                    abs(true_end - pred_end) <= tolerance or
                    (pred_start <= true_start <= pred_end) or
                    (true_start <= pred_start <= true_end)):
                    tp += 1
                    matched_pred.add(j)
                    break
        
        fp = len(pred_events) - len(matched_pred)
        fn = len(true_events) - tp
        
        return tp, fp, fn
    
    def compute_all_metrics(self, y_pred_power: torch.Tensor, y_pred_proba: torch.Tensor,
                          y_true_power: torch.Tensor, y_true_states: torch.Tensor,
                          optimize_thresholds: bool = True,
                          classification_enabled: bool = True) -> Dict[str, Union[float, Dict[str, float]]]:
        """计算所有指标"""
        metrics = {}
        
        # 优化阈值（仅分类启用时）
        if classification_enabled and optimize_thresholds:
            try:
                self.optimize_thresholds(y_pred_proba, y_true_states)
            except Exception:
                pass
        
        # 回归指标
        metrics['mae'] = self.mean_absolute_error(y_pred_power, y_true_power)
        metrics['mae_per_device'] = self.mean_absolute_error(y_pred_power, y_true_power, per_device=True)
        metrics['nde'] = self.normalized_disaggregation_error(y_pred_power, y_true_power)
        metrics['nde_per_device'] = self.normalized_disaggregation_error(y_pred_power, y_true_power, per_device=True)
        metrics['sae'] = self.signal_aggregate_error(y_pred_power, y_true_power)
        metrics['teca'] = self.total_energy_correctly_assigned(y_pred_power, y_true_power)
        metrics['teca_per_device'] = self.total_energy_correctly_assigned(y_pred_power, y_true_power, per_device=True)
        
        # 分类指标（可选）
        if classification_enabled:
            try:
                metrics['f1'] = self.f1_score(y_pred_proba, y_true_states)
                metrics['f1_per_device'] = self.f1_score(y_pred_proba, y_true_states, per_device=True)
                metrics['mcc'] = self.matthews_correlation_coefficient(y_pred_proba, y_true_states)
                metrics['mcc_per_device'] = self.matthews_correlation_coefficient(y_pred_proba, y_true_states, per_device=True)
                metrics['pr_auc'] = self.precision_recall_auc(y_pred_proba, y_true_states)
                metrics['pr_auc_per_device'] = self.precision_recall_auc(y_pred_proba, y_true_states, per_device=True)
                metrics['roc_auc'] = self.roc_auc(y_pred_proba, y_true_states)
                metrics['roc_auc_per_device'] = self.roc_auc(y_pred_proba, y_true_states, per_device=True)
                # 事件检测指标
                metrics['event_metrics'] = self.event_detection_metrics(y_pred_proba, y_true_states)
            except Exception:
                pass
        
        # 综合评分（会根据已有指标自动加权）
        metrics['score'] = self._compute_composite_score(metrics)
        
        return metrics
    
    def _compute_composite_score(self, metrics: Dict) -> float:
        """计算综合评分"""
        # 权重配置
        weights = {
            'mae': -0.2,  # 负权重，越小越好
            'nde': -0.2,
            'sae': -0.1,
            'teca': 0.2,  # 正权重，越大越好
            'f1': 0.3,
            'mcc': 0.2,
            'pr_auc': 0.2
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric_name, weight in weights.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                # 统一将 numpy 标量/torch 标量 转换为 Python float
                try:
                    if isinstance(value, (int, float)):
                        val = float(value)
                    elif isinstance(value, np.ndarray) and value.shape == ():
                        val = float(value.item())
                    elif isinstance(value, np.floating):
                        val = float(value)
                    elif torch.is_tensor(value) and value.dim() == 0:
                        val = float(value.item())
                    else:
                        # 非标量（如字典/数组）不计入综合分
                        continue
                except Exception:
                    continue
                
                if not np.isnan(val):
                    if weight < 0:
                        # 对于越小越好的指标，使用负指数变换
                        normalized_value = np.exp(-val)
                    else:
                        # 对于越大越好的指标，直接使用
                        normalized_value = val
                    
                    score += abs(weight) * normalized_value
                    total_weight += abs(weight)
        
        return score / total_weight if total_weight > 0 else 0.0


class ConsistencyMetrics:
    """一致性相关指标"""
    
    @staticmethod
    def power_balance_error(y_pred_power: torch.Tensor, y_true_power: torch.Tensor) -> float:
        """功率平衡误差"""
        pred_total = torch.sum(y_pred_power, dim=1)
        true_total = torch.sum(y_true_power, dim=1)
        
        balance_error = torch.mean(torch.abs(pred_total - true_total))
        return balance_error.item()
    
    @staticmethod
    def state_power_consistency(y_pred_power: torch.Tensor, y_pred_states: torch.Tensor,
                              threshold: float = 0.1) -> float:
        """状态-功率一致性"""
        # 当状态为0时，功率应该接近0
        off_state_mask = y_pred_states < 0.5
        off_state_power = y_pred_power[off_state_mask]
        
        if len(off_state_power) > 0:
            inconsistency = torch.mean((off_state_power > threshold).float())
            return 1.0 - inconsistency.item()
        else:
            return 1.0
    
    @staticmethod
    def temporal_consistency(y_pred_power: torch.Tensor, max_change_rate: float = 0.5) -> float:
        """时间一致性（相邻时间步的变化不应过大）"""
        if y_pred_power.size(0) <= 1:
            return 1.0
        
        # 计算相邻时间步的变化率
        power_diff = torch.diff(y_pred_power, dim=0)
        change_rate = torch.abs(power_diff) / (y_pred_power[:-1] + 1e-8)
        
        # 计算超出最大变化率的比例
        violations = (change_rate > max_change_rate).float()
        consistency = 1.0 - torch.mean(violations)
        
        return consistency.item()


class DelayMetrics:
    """延迟相关指标"""
    
    @staticmethod
    def detection_delay(y_pred_proba: torch.Tensor, y_true: torch.Tensor,
                       threshold: float = 0.5) -> Dict[str, float]:
        """检测延迟"""
        y_pred_proba = y_pred_proba.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        
        delays = {}
        
        for device_idx in range(y_pred_proba.shape[1]):
            true_signal = y_true[:, device_idx]
            pred_signal = (y_pred_proba[:, device_idx] >= threshold).astype(int)
            
            # 找到真实信号的上升沿
            true_edges = np.where(np.diff(true_signal) == 1)[0] + 1
            
            if len(true_edges) == 0:
                delays[f'device_{device_idx}'] = 0.0
                continue
            
            total_delay = 0
            valid_detections = 0
            
            for edge in true_edges:
                # 在上升沿后寻找预测信号的响应
                search_window = min(edge + 50, len(pred_signal))  # 搜索窗口
                
                for i in range(edge, search_window):
                    if pred_signal[i] == 1:
                        delay = i - edge
                        total_delay += delay
                        valid_detections += 1
                        break
            
            if valid_detections > 0:
                avg_delay = total_delay / valid_detections
                delays[f'device_{device_idx}'] = avg_delay
            else:
                delays[f'device_{device_idx}'] = float('inf')  # 未检测到
        
        return delays
    
    @staticmethod
    def response_time_distribution(y_pred_proba: torch.Tensor, y_true: torch.Tensor,
                                 threshold: float = 0.5) -> Dict[str, List[float]]:
        """响应时间分布"""
        y_pred_proba = y_pred_proba.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        
        response_times = {}
        
        for device_idx in range(y_pred_proba.shape[1]):
            true_signal = y_true[:, device_idx]
            pred_signal = (y_pred_proba[:, device_idx] >= threshold).astype(int)
            
            # 找到所有状态变化
            true_changes = np.where(np.diff(true_signal) != 0)[0] + 1
            device_response_times = []
            
            for change_point in true_changes:
                # 寻找预测信号的对应变化
                search_start = max(0, change_point - 10)
                search_end = min(len(pred_signal), change_point + 20)
                
                for i in range(search_start, search_end):
                    if i > 0 and pred_signal[i] != pred_signal[i-1]:
                        response_time = abs(i - change_point)
                        device_response_times.append(response_time)
                        break
            
            response_times[f'device_{device_idx}'] = device_response_times
        
        return response_times