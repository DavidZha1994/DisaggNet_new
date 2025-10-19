"""准实时推理模块"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import deque
import time
from datetime import datetime, timedelta
import json
from omegaconf import DictConfig, OmegaConf
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from src.train import NILMLightningModule
from src.utils.metrics import ConsistencyMetrics
from src.utils.conformal_prediction import MultiTaskConformalPredictor
from src.utils.conformal_evaluation import ConformalEvaluator
from src.utils.online_conformal_monitor import OnlineConformalMonitor, ConformalAlertSystem
from src.inference.post_processing import InferencePostProcessor, create_post_processor, RECOMMENDED_POST_PROCESSING_CONFIGS
from src.utils.prototypes import PrototypeLibrary


class CircularBuffer:
    """环形缓冲区"""
    
    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self.buffer = deque(maxlen=maxsize)
        self.timestamps = deque(maxlen=maxsize)
    
    def append(self, data: np.ndarray, timestamp: datetime):
        """添加新数据"""
        self.buffer.append(data)
        self.timestamps.append(timestamp)
    
    def get_window(self, window_size: int) -> Tuple[np.ndarray, List[datetime]]:
        """获取指定大小的窗口数据"""
        if len(self.buffer) < window_size:
            raise ValueError(f"Buffer size {len(self.buffer)} < window_size {window_size}")
        
        # 获取最后window_size个数据点
        data = np.array(list(self.buffer)[-window_size:])
        timestamps = list(self.timestamps)[-window_size:]
        
        return data, timestamps
    
    def is_full(self, min_size: int) -> bool:
        """检查缓冲区是否有足够的数据"""
        return len(self.buffer) >= min_size
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
        self.timestamps.clear()


class IncrementalSTFT:
    """增量STFT计算器"""
    
    def __init__(self, n_fft: int, hop_length: int, window: str = 'hann'):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window
        
        # 创建窗口函数
        if window == 'hann':
            self.window_func = torch.hann_window(n_fft)
        elif window == 'hamming':
            self.window_func = torch.hamming_window(n_fft)
        else:
            self.window_func = torch.ones(n_fft)
        
        # 历史数据缓存
        self.history_buffer = deque(maxlen=n_fft)
        self.last_stft = None
    
    def compute_incremental(self, new_data: np.ndarray) -> torch.Tensor:
        """增量计算STFT"""
        # 将新数据添加到历史缓冲区
        for sample in new_data:
            self.history_buffer.append(sample)
        
        # 如果缓冲区不够大，返回空
        if len(self.history_buffer) < self.n_fft:
            return None
        
        # 获取当前窗口数据
        current_window = np.array(list(self.history_buffer)[-self.n_fft:])
        current_window = torch.from_numpy(current_window).float()
        
        # 应用窗口函数
        windowed_data = current_window * self.window_func
        
        # 计算FFT
        fft_result = torch.fft.fft(windowed_data)
        
        # 只保留正频率部分
        stft_result = fft_result[:self.n_fft // 2 + 1]
        
        # 计算幅度谱
        magnitude = torch.abs(stft_result)
        
        return magnitude.unsqueeze(0)  # 添加批次维度
    
    def reset(self):
        """重置STFT计算器"""
        self.history_buffer.clear()
        self.last_stft = None


class HysteresisFilter:
    """滞回滤波器"""
    
    def __init__(self, thresholds: Dict[str, float], hysteresis_ratio: float = 0.1):
        self.thresholds = thresholds
        self.hysteresis_ratio = hysteresis_ratio
        
        # 计算滞回阈值
        self.on_thresholds = {}
        self.off_thresholds = {}
        
        for device, threshold in thresholds.items():
            self.on_thresholds[device] = threshold
            self.off_thresholds[device] = threshold * (1 - hysteresis_ratio)
        
        # 当前状态
        self.current_states = {device: False for device in thresholds.keys()}
    
    def filter(self, predictions: Dict[str, float]) -> Dict[str, bool]:
        """应用滞回滤波"""
        filtered_states = {}
        
        for device, power in predictions.items():
            if device not in self.thresholds:
                filtered_states[device] = power > 0
                continue
            
            current_state = self.current_states[device]
            
            if current_state:
                # 当前为开启状态，检查是否应该关闭
                if power < self.off_thresholds[device]:
                    filtered_states[device] = False
                else:
                    filtered_states[device] = True
            else:
                # 当前为关闭状态，检查是否应该开启
                if power > self.on_thresholds[device]:
                    filtered_states[device] = True
                else:
                    filtered_states[device] = False
            
            # 更新当前状态
            self.current_states[device] = filtered_states[device]
        
        return filtered_states
    
    def reset(self):
        """重置滤波器状态"""
        self.current_states = {device: False for device in self.thresholds.keys()}


class VotingFilter:
    """短期投票滤波器"""
    
    def __init__(self, window_size: int = 3, threshold: float = 0.5):
        self.window_size = window_size
        self.threshold = threshold
        self.history = {}
    
    def filter(self, states: Dict[str, bool]) -> Dict[str, bool]:
        """应用投票滤波"""
        filtered_states = {}
        
        for device, state in states.items():
            if device not in self.history:
                self.history[device] = deque(maxlen=self.window_size)
            
            # 添加当前状态
            self.history[device].append(state)
            
            # 计算投票结果
            if len(self.history[device]) >= self.window_size:
                vote_ratio = sum(self.history[device]) / len(self.history[device])
                filtered_states[device] = vote_ratio >= self.threshold
            else:
                # 历史不够，直接使用当前状态
                filtered_states[device] = state
        
        return filtered_states
    
    def reset(self):
        """重置投票历史"""
        self.history.clear()


class OnlineConsistencyMonitor:
    """在线一致性监控器"""
    
    def __init__(self, error_threshold: float = 0.2, alert_duration: int = 3):
        self.error_threshold = error_threshold
        self.alert_duration = alert_duration
        
        self.consistency_metrics = ConsistencyMetrics()
        self.error_history = deque(maxlen=alert_duration)
        self.audit_windows = []
    
    def check_consistency(self, mains_power: float, device_powers: Dict[str, float], 
                         timestamp: datetime) -> Dict[str, Any]:
        """检查功率一致性"""
        
        # 计算总设备功率
        total_device_power = sum(device_powers.values())
        
        # 计算一致性误差
        if mains_power > 0:
            consistency_error = abs(mains_power - total_device_power) / mains_power
        else:
            consistency_error = abs(total_device_power)
        
        # 添加到历史
        self.error_history.append(consistency_error)
        
        # 检查是否需要警报
        alert = False
        if len(self.error_history) >= self.alert_duration:
            recent_errors = list(self.error_history)
            if all(error > self.error_threshold for error in recent_errors):
                alert = True
                
                # 记录审计窗口
                audit_window = {
                    'timestamp': timestamp,
                    'mains_power': mains_power,
                    'device_powers': device_powers.copy(),
                    'consistency_error': consistency_error,
                    'recent_errors': recent_errors.copy()
                }
                self.audit_windows.append(audit_window)
        
        return {
            'consistency_error': consistency_error,
            'alert': alert,
            'recent_avg_error': np.mean(list(self.error_history)) if self.error_history else 0.0
        }
    
    def get_audit_windows(self) -> List[Dict[str, Any]]:
        """获取审计窗口"""
        return self.audit_windows.copy()
    
    def reset(self):
        """重置监控器"""
        self.error_history.clear()
        self.audit_windows.clear()


class RealTimeInferenceEngine:
    """准实时推理引擎"""
    
    def __init__(self, config: DictConfig, model_path: str, device_info: Dict[str, Any]):
        self.config = config
        self.device_info = device_info
        self.device_names = list(device_info.keys())
        
        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # 初始化组件
        self._initialize_components()
        
        # 性能统计
        self.inference_times = deque(maxlen=100)
        self.total_inferences = 0
    
    def _load_model(self, model_path: str) -> NILMLightningModule:
        """加载训练好的模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # 加载检查点
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 创建模型实例
        model = NILMLightningModule.load_from_checkpoint(
            model_path,
            config=self.config,
            device_info=self.device_info,
            device_names=self.device_names,
            map_location='cpu'
        )

        # 若检查点包含原型库状态且模型启用了度量学习，则加载原型库
        try:
            proto_state = checkpoint.get('prototype_library_state', None)
            if proto_state is not None and getattr(model, 'prototype_library', None) is not None:
                model.prototype_library.load_state_dict(proto_state)
                print("原型库状态已从检查点加载，推理时启用异常距离监控。")
            else:
                print("检查点未包含原型库状态或模型未启用度量学习。")
        except Exception as e:
            print(f"加载原型库状态失败（推理继续）: {e}")
        
        return model
    
    def _initialize_components(self):
        """初始化各个组件"""
        
        # 环形缓冲区
        buffer_size = self.config.inference.buffer_size
        self.buffer = CircularBuffer(buffer_size)
        
        # 特征提取配置
        self.feature_config = self.config.data.features if hasattr(self.config.data, 'features') else None
        
        # 增量STFT
        if self.config.data.features.freq_domain.enable:
            self.stft_computer = IncrementalSTFT(
                n_fft=self.config.data.features.freq_domain.stft.n_fft,
                hop_length=self.config.data.features.freq_domain.stft.hop_length
            )
        else:
            self.stft_computer = None
        
        # 阈值加载与区分：分类阈值（0~1）与功率滞回阈值（功率域）
        self.class_thresholds: Optional[Dict[str, float]] = None
        power_thresholds: Optional[Dict[str, float]] = None

        try:
            # 1) 可能存在模型训练得到的“最佳分类阈值”
            model_thresholds = getattr(self.model, 'best_thresholds', None)
            if isinstance(model_thresholds, dict) and len(model_thresholds) > 0:
                # 判断数值范围：若均 <= 1.0，则视为分类阈值
                if max(map(float, model_thresholds.values())) <= 1.0 + 1e-8:
                    self.class_thresholds = {name: float(model_thresholds.get(name, 0.5)) for name in self.device_names}
                else:
                    power_thresholds = {name: float(model_thresholds.get(name, 10.0)) for name in self.device_names}

            # 2) 若未从模型拿到，尝试从 JSON 文件加载（train.py 落盘的通常为分类阈值）
            if self.class_thresholds is None and power_thresholds is None:
                eval_cfg = getattr(self.config, 'evaluation', None)
                default_path = Path(self.config.paths.output_dir) / 'best_thresholds.json'
                thresholds_path = (eval_cfg.thresholds_path if (eval_cfg and hasattr(eval_cfg, 'thresholds_path')) else str(default_path))
                if os.path.exists(thresholds_path):
                    with open(thresholds_path, 'r') as f:
                        data = json.load(f)
                    loaded = {name: float(data.get(name, 0.5)) for name in self.device_names}
                    if len(loaded) > 0:
                        if max(loaded.values()) <= 1.0 + 1e-8:
                            self.class_thresholds = loaded
                        else:
                            power_thresholds = loaded
        except Exception as e:
            print(f"阈值加载失败，将使用默认：{e}")

        # 3) 构建滞回功率阈值（当未可用时使用默认功率域阈值）
        if power_thresholds is None:
            default_power_thr = float(self.config.inference.get('default_power_threshold', 10.0))
            power_thresholds = {device: default_power_thr for device in self.device_names}

        # 保留原有的滤波器用于兼容性（但主要使用统一后处理器）
        self.hysteresis_filter = HysteresisFilter(
            power_thresholds,
            hysteresis_ratio=self.config.inference.hysteresis_ratio
        )
        
        # 投票滤波器
        self.voting_filter = VotingFilter(
            window_size=self.config.inference.voting_window,
            threshold=self.config.inference.voting_threshold
        )
        
        # 一致性监控器
        self.consistency_monitor = OnlineConsistencyMonitor(
            error_threshold=self.config.inference.consistency_threshold,
            alert_duration=self.config.inference.alert_duration
        )
        
        # 统一后处理器 - 使用推荐配置
        post_processing_config = RECOMMENDED_POST_PROCESSING_CONFIGS.get('balanced', {})
        # 如果config中有自定义后处理配置，则覆盖默认值
        if hasattr(self.config, 'post_processing') and self.config.post_processing:
            post_processing_config.update(OmegaConf.to_container(self.config.post_processing, resolve=True))
        
        # 设置设备名称用于后处理器
        post_processing_config['device_names'] = self.device_names
        
        self.post_processor = create_post_processor(post_processing_config)
        
        # 打印后处理配置用于验证
        print(f"后处理器配置: 滞回比例={self.post_processor.config.hysteresis_ratio}, "
              f"最短持续时间={self.post_processor.config.min_on_duration}, "
              f"重叠窗口融合={self.post_processor.config.overlap_fusion_method}")
        
        # Conformal Prediction预测器
        if hasattr(self.model, 'conformal_predictor') and self.model.conformal_predictor is not None:
            self.conformal_predictor = self.model.conformal_predictor
            # 创建评估器用于在线监控
            self.conformal_evaluator = ConformalEvaluator(
                device_names=list(self.device_info.keys()),
                alpha=self.config.get('conformal_prediction', {}).get('alpha', 0.1)
            )
            # 创建在线监控器
            self.conformal_monitor = OnlineConformalMonitor(
                device_names=list(self.device_info.keys()),
                alpha=self.config.get('conformal_prediction', {}).get('alpha', 0.1),
                window_size=self.config.get('conformal_prediction', {}).get('monitoring', {}).get('window_size', 1000),
                alert_threshold=self.config.get('conformal_prediction', {}).get('monitoring', {}).get('alert_threshold', 0.05)
            )
            # 创建告警系统
            alert_config = self.config.get('conformal_prediction', {}).get('alerts', {})
            if alert_config.get('enable', False):
                self.alert_system = ConformalAlertSystem(alert_config)
            else:
                self.alert_system = None
        else:
            self.conformal_predictor = None
            self.conformal_evaluator = None
            self.conformal_monitor = None
            self.alert_system = None
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.config.inference.use_gpu else 'cpu')
        self.model.to(self.device)
    
    def process_new_data(self, mains_data: Dict[str, float], timestamp: datetime) -> Dict[str, Any]:
        """处理新的主线数据"""
        
        start_time = time.time()
        
        try:
            # 提取功率数据
            mains_power = np.array([mains_data.get('P', 0.0), mains_data.get('Q', 0.0), mains_data.get('S', 0.0)])
            
            # 添加到缓冲区
            self.buffer.append(mains_power, timestamp)
            
            # 检查是否有足够的数据进行推理
            window_size = self.config.data.window_size
            if not self.buffer.is_full(window_size):
                return {
                    'status': 'insufficient_data',
                    'buffer_size': len(self.buffer.buffer),
                    'required_size': window_size
                }
            
            # 获取窗口数据
            window_data, window_timestamps = self.buffer.get_window(window_size)
            
            # 特征工程
            features = self._extract_features(window_data, window_timestamps)
            
            # 模型推理
            predictions = self._run_inference(features)
            
            # 后处理
            processed_results = self._post_process(predictions, mains_data['P'], timestamp)
            
            # 记录推理时间
            inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
            self.inference_times.append(inference_time)
            self.total_inferences += 1
            
            # 构建结果
            result = {
                'status': 'success',
                'timestamp': timestamp,
                'mains_power': mains_data['P'],
                'device_powers': processed_results['device_powers'],
                'device_states': processed_results['device_states'],
                'consistency_check': processed_results['consistency_check'],
                'inference_time_ms': inference_time,
                'avg_inference_time_ms': np.mean(list(self.inference_times)),
                'total_inferences': self.total_inferences
            }
            
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': timestamp
            }
    
    def _extract_features(self, window_data: np.ndarray, timestamps: List[datetime]) -> Dict[str, torch.Tensor]:
        """提取特征：时域、频域与时间位置编码"""
        out: Dict[str, torch.Tensor] = {}
        
        # 时域特征 - 直接使用窗口数据
        out['time_features'] = torch.from_numpy(window_data).float().unsqueeze(0).to(self.device)
        
        # 频域特征 - 如果启用STFT
        if self.stft_computer is not None:
            try:
                # 使用增量STFT计算频域特征
                freq_data = self.stft_computer.compute_stft(window_data)
                if freq_data is not None:
                    out['freq_features'] = torch.from_numpy(freq_data).float().unsqueeze(0).to(self.device)
                else:
                    out['freq_features'] = None
            except Exception as e:
                print(f"频域特征计算失败: {e}")
                out['freq_features'] = None
        else:
            out['freq_features'] = None
        
        # 时间位置编码 - 简单的序列位置
        seq_len = window_data.shape[0]
        time_pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).to(self.device)
        out['time_positional'] = time_pos
        
        # 辅助特征占位
        out['aux_features'] = None
        return out
    
    def _run_inference(self, features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """运行模型推理"""
        with torch.no_grad():
            batch = {
                'time_features': features.get('time_features'),
                'freq_features': features.get('freq_features'),
                'time_positional': features.get('time_positional'),
                'aux_features': features.get('aux_features')
            }
            # 优先尝试获取嵌入以进行异常距离计算
            outputs = None
            pred_embeddings = None
            try:
                if hasattr(self.model, 'forward_with_embeddings') and self.model.metric_learning_enable:
                    out = self.model.forward_with_embeddings(batch)
                    if isinstance(out, tuple) and len(out) == 4:
                        pred_power, pred_states, unknown_pred, pred_embeddings = out
                        outputs = (pred_power, pred_states, unknown_pred)
                    else:
                        pred_power, pred_states, pred_embeddings = out
                        outputs = (pred_power, pred_states)
                else:
                    outputs = self.model(batch)
            except Exception:
                outputs = self.model(batch)

            # 推理期计算 Mahalanobis 距离并进行门控（可选）
            try:
                if pred_embeddings is not None and getattr(self.model, 'prototype_library', None) is not None:
                    distances = self.model.prototype_library.mahalanobis(pred_embeddings)  # (B, N)
                    # 将距离作为 anomaly_score 融入输出，供后续 _post_process 使用
                    self._last_anomaly_distances = distances.squeeze(0).detach().cpu().numpy().tolist()
                else:
                    self._last_anomaly_distances = None
            except Exception as e:
                print(f"推理期距离计算失败（忽略）: {e}")
                self._last_anomaly_distances = None

            return outputs

    def _post_process(self, predictions: Tuple[torch.Tensor, torch.Tensor], mains_power: float, 
                     timestamp: datetime) -> Dict[str, Any]:
        """后处理预测结果 - 使用统一后处理器"""
        
        # 提取功率预测和状态预测（logits），兼容unknown
        if isinstance(predictions, tuple) and len(predictions) == 3:
            pred_power, pred_states, unknown_pred = predictions
        else:
            pred_power, pred_states = predictions
            unknown_pred = None
        power_preds = pred_power.cpu().numpy().flatten()
        # 温度缩放 + Sigmoid 生成状态概率
        logits = pred_states
        if hasattr(self.model, 'temperature_scaling') and self.model.temperature_scaling is not None:
            logits = self.model.temperature_scaling(logits)
        state_probs_np = torch.sigmoid(logits).cpu().numpy().flatten()

        # 异常距离门控：根据训练期原型的 Mahalanobis 距离与 Unknown 能量协同调整
        anomaly_scores = self._last_anomaly_distances if hasattr(self, '_last_anomaly_distances') else None
        if anomaly_scores is not None:
            # 根据配置设置门槛与权重
            gating_conf = self.config.get('inference', {}).get('prototype_gating', {})
            threshold = float(gating_conf.get('distance_threshold', 2.5))
            unknown_boost = float(gating_conf.get('unknown_boost', 0.3))
            suppress_power_ratio = float(gating_conf.get('suppress_power_ratio', 0.5))
            # 若模型输出 Unknown，则提升其权重；同时抑制高异常设备的激活
            high_anomaly = [i for i, d in enumerate(anomaly_scores) if d >= threshold]
            if len(high_anomaly) > 0:
                # 抑制对应设备的状态概率与功率
                for i in high_anomaly:
                    d = float(anomaly_scores[i])
                    # 异常严重度：距离超过阈值的相对超额，截断到 [0, 1]
                    severity = min(max(d / threshold - 1.0, 0.0), 1.0)
                    # 抑制状态概率
                    state_probs_np[i] = state_probs_np[i] * (1.0 - 0.5 * severity)
                    # 抑制功率预测
                    power_preds[i] = power_preds[i] * (1.0 - suppress_power_ratio * severity)
                # Unknown 加权提升（若可用）
                if unknown_pred is not None:
                    unknown_pred = unknown_pred + unknown_boost * torch.tensor([1.0], device=unknown_pred.device)
        
        # 构建设备功率字典
        device_powers = {}
        for i, device in enumerate(self.device_names):
            device_powers[device] = float(power_preds[i])
        
        # 构建设备状态概率字典
        state_probs = {}
        for i, device in enumerate(self.device_names):
            state_probs[device] = float(state_probs_np[i])
        
        # 使用统一后处理器进行后处理
        try:
            # 准备输入数据
            inference_data = {
                'device_powers': device_powers,
                'state_probs': state_probs,
                'mains_power': mains_power,
                'timestamp': timestamp,
                'anomaly_scores': {self.device_names[i]: float(anomaly_scores[i]) for i in range(len(self.device_names))} if anomaly_scores is not None else None
            }
            
            # 调用统一后处理器
            processed_results = self.post_processor.process_single_window(inference_data)
            
            # 提取处理后的状态
            final_states = processed_results.get('final_states', {})
            
            print(f"统一后处理器处理结果: {processed_results}")
            
        except Exception as e:
            print(f"统一后处理器处理失败，回退到原有逻辑: {e}")
            # 回退到原有的后处理逻辑
            hysteresis_states = self.hysteresis_filter.filter(device_powers)
            final_states = self.voting_filter.filter(hysteresis_states)
            processed_results = {}
        
        # Conformal Prediction区间预测
        conformal_results = {}
        if self.conformal_predictor is not None:
            # 准备预测数据
            pred_power_tensor = pred_power.unsqueeze(0)
            pred_state_tensor = pred_states.unsqueeze(0)
            
            # 获取区间预测
            intervals = self.conformal_predictor.predict_intervals(
                predictions=(pred_power_tensor, pred_state_tensor)
            )
            
            # 解析区间结果
            power_intervals = intervals['regression']
            state_intervals = intervals['classification']
            
            for i, device in enumerate(self.device_names):
                power_interval = {
                    'lower': float(power_intervals['lower'][0, i]),
                    'upper': float(power_intervals['upper'][0, i]),
                    'coverage': float(power_intervals['coverage'])
                }
                state_confidence = {
                    'threshold': float(state_intervals['threshold'][0, i]),
                    'calibrated_prob': float(state_intervals['calibrated_prob'][0, i]),
                    'coverage': float(state_intervals['coverage'])
                }
                
                conformal_results[device] = {
                    'power_interval': power_interval,
                    'state_confidence': state_confidence
                }
                
                # 在线监控更新
                if self.conformal_monitor is not None:
                    # 更新回归监控（需要真实值时才能计算覆盖率，这里先记录预测）
                    # 在实际应用中，可以通过延迟获取真实值来更新监控
                    pass
                    
                # 检查告警
                if self.alert_system is not None:
                    # 区间宽度基础告警
                    interval_width = power_interval['upper'] - power_interval['lower']
                    metrics = {
                        'interval_width': interval_width,
                        'coverage_rate': power_interval['coverage'],
                        'timestamp': timestamp.timestamp()
                    }
                    alerts = self.alert_system.check_alerts(device, metrics) or []

                    # 结合功率滞回阈值与区间宽度的不确定性提示
                    try:
                        thr = self.hysteresis_filter.thresholds.get(device, 0.0)
                        margin = abs(thr) * self.hysteresis_filter.hysteresis_ratio
                        near_threshold = abs(device_powers[device] - thr) <= max(margin, 1e-6)
                        width_ratio = interval_width / max(abs(device_powers[device]) + 1e-6, 1.0)
                        if near_threshold and width_ratio > (self.config.inference.get('uncertainty_width_ratio', 0.5)):
                            alerts.append({'type': 'uncertainty_near_threshold', 'width_ratio': float(width_ratio)})
                    except Exception:
                        pass

                    # 结合分类阈值与状态概率的“临界”提醒
                    try:
                        cls_thr = None
                        if self.class_thresholds is not None:
                            cls_thr = float(self.class_thresholds.get(device, 0.5))
                        else:
                            # 若分类阈值未加载，使用共形输出的阈值
                            state_conf = conformal_results[device].get('state_confidence', {})
                            cls_thr = float(state_conf.get('threshold', 0.5))
                        prob = float(state_probs.get(device, 0.0))
                        cls_margin = float(self.config.inference.get('cls_threshold_margin', 0.05))
                        if abs(prob - cls_thr) <= cls_margin:
                            alerts.append({'type': 'state_prob_near_threshold', 'margin': float(abs(prob - cls_thr))})
                    except Exception:
                        pass

                    if alerts:
                        conformal_results[device]['alerts'] = alerts
        
        # 一致性检查（保持原有逻辑）
        consistency_check = self.consistency_monitor.check_consistency(
            mains_power, device_powers, timestamp
        )
        
        # 计算unknown与残差
        unknown_power = None
        residual_power = None
        try:
            if unknown_pred is not None:
                unknown_power = float(F.softplus(unknown_pred).squeeze().cpu().numpy())
            # 残差 = 主表功率 - 已预测设备激活功率和 - unknown（若存在）
            active_power = pred_power * torch.sigmoid(pred_states)
            predicted_total = float(active_power.sum().cpu().item())
            if unknown_power is not None:
                predicted_total += unknown_power
            residual_power = float(mains_power - predicted_total)
        except Exception:
            pass

        result = {
            'device_powers': device_powers,
            'device_states': final_states,
            'state_probs': state_probs,
            'consistency_check': consistency_check,
            'post_processing_details': processed_results  # 添加后处理详情
        }

        # 若存在分类阈值，补充基于概率的二值化状态（与功率滞回并行提供）
        try:
            if self.class_thresholds is not None:
                prob_based_states = {}
                for device in self.device_names:
                    threshold = self.class_thresholds.get(device, 0.5)
                    prob_based_states[device] = state_probs[device] > threshold
                result['prob_based_states'] = prob_based_states
        except Exception:
            pass

        # 添加Unknown与残差监控输出
        if unknown_power is not None:
            result['unknown_power'] = unknown_power
        if residual_power is not None:
            result['residual_power'] = residual_power

        # 添加Conformal结果
        if conformal_results:
            result['conformal_prediction'] = conformal_results

        return result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        if not self.inference_times:
            return {'status': 'no_data'}
        
        times = list(self.inference_times)
        return {
            'total_inferences': self.total_inferences,
            'avg_inference_time_ms': np.mean(times),
            'min_inference_time_ms': np.min(times),
            'max_inference_time_ms': np.max(times),
            'std_inference_time_ms': np.std(times),
            'p95_inference_time_ms': np.percentile(times, 95),
            'p99_inference_time_ms': np.percentile(times, 99)
        }
    
    def get_audit_windows(self) -> List[Dict[str, Any]]:
        """获取一致性审计窗口"""
        return self.consistency_monitor.get_audit_windows()
    
    def reset(self):
        """重置推理引擎"""
        self.buffer.clear()
        if self.stft_computer:
            self.stft_computer.reset()
        self.hysteresis_filter.reset()
        self.voting_filter.reset()
        self.consistency_monitor.reset()
        self.inference_times.clear()
        self.total_inferences = 0
    
    def export_to_onnx(self, output_path: str, input_shape: Tuple[int, ...]):
        """导出模型为ONNX格式"""
        
        # 创建示例输入
        dummy_input = torch.randn(1, *input_shape).to(self.device)
        dummy_batch = {
            'mains': dummy_input,
            'targets': torch.zeros(1, len(self.device_names)).to(self.device)
        }
        
        # 导出ONNX
        torch.onnx.export(
            self.model,
            dummy_batch,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['mains'],
            output_names=['predictions'],
            dynamic_axes={
                'mains': {0: 'batch_size'},
                'predictions': {0: 'batch_size'}
            }
        )
        
        print(f"模型已导出为ONNX格式: {output_path}")


def create_inference_engine(config: DictConfig, model_path: str, device_info: Dict[str, Any]) -> RealTimeInferenceEngine:
    """创建推理引擎"""
    return RealTimeInferenceEngine(config, model_path, device_info)


def simulate_real_time_inference(engine: RealTimeInferenceEngine, data_file: str, 
                                interval_seconds: float = 5.0) -> List[Dict[str, Any]]:
    """模拟实时推理"""
    
    # 加载测试数据
    df = pd.read_csv(data_file)
    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='5S')
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    results = []
    
    print(f"开始模拟实时推理，数据点数: {len(df)}")
    
    for i, row in df.iterrows():
        # 构建主线数据
        mains_data = {
            'P': row.get('mains_P', row.get('P', 0.0)),
            'Q': row.get('mains_Q', row.get('Q', 0.0)),
            'S': row.get('mains_S', row.get('S', 0.0))
        }
        
        timestamp = row['timestamp']
        
        # 处理数据
        result = engine.process_new_data(mains_data, timestamp)
        results.append(result)
        
        # 打印进度
        if i % 100 == 0:
            print(f"处理进度: {i}/{len(df)} ({i/len(df)*100:.1f}%)")
        
        # 模拟实时间隔
        if interval_seconds > 0:
            time.sleep(interval_seconds / 1000)  # 转换为毫秒并加速
    
    print("模拟推理完成")
    
    # 打印性能统计
    stats = engine.get_performance_stats()
    print("\n=== 性能统计 ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 打印审计窗口
    audit_windows = engine.get_audit_windows()
    if audit_windows:
        print(f"\n检测到 {len(audit_windows)} 个一致性异常窗口")
    
    return results


def main(config: DictConfig, model_path: str = None, data_path: str = None, output_dir: Path = None) -> Dict[str, Any]:
    """主推理函数"""
    # 模型路径
    if model_path is None:
        model_path = config.get('infer_model_path', 'outputs/best_model.ckpt')
    
    # 设备信息
    device_info = {
        'device1': {'max_power': 100, 'type': 'heating'},
        'device2': {'max_power': 50, 'type': 'lighting'}
    }
    
    # 创建推理引擎
    engine = create_inference_engine(config, model_path, device_info)
    
    # 模拟实时推理
    if data_path is None:
        data_path = "path/to/test_data.csv"
    results = simulate_real_time_inference(engine, data_path)
    
    # 保存结果
    if output_dir is None:
        output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "inference_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"推理结果保存至: {output_file}")
    return results


# 注意：此脚本不应直接执行，请使用统一入口 main.py
# 示例：python main.py infer --checkpoint-path=path/to/model.ckpt --data-path=path/to/data.csv