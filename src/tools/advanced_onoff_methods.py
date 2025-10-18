#!/usr/bin/env python3
"""
先进的开关检测方法
包括机器学习、信号处理、统计学习等方法
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class AdvancedOnOffDetector:
    """先进的开关检测器集合"""
    
    def __init__(self):
        self.methods = {
            'spectral': self.spectral_method,
            'clustering': self.clustering_method,
            'change_point': self.change_point_method,
            'wavelet': self.wavelet_method,
            'hmm_like': self.hmm_like_method,
            'adaptive_threshold': self.adaptive_threshold_method,
            'energy_based': self.energy_based_method
        }
    
    def spectral_method(self, power_data, **kwargs):
        """
        基于频谱分析的方法
        利用功率信号的频域特征检测开关状态
        """
        try:
            # 参数设置
            window_size = kwargs.get('window_size', 100)
            overlap = kwargs.get('overlap', 0.5)
            freq_threshold = kwargs.get('freq_threshold', 0.1)
            
            # 计算短时傅里叶变换
            f, t, Zxx = signal.stft(
                power_data, 
                nperseg=window_size, 
                noverlap=int(window_size * overlap)
            )
            
            # 计算功率谱密度
            psd = np.abs(Zxx) ** 2
            
            # 低频能量占比（表示稳定状态）
            low_freq_idx = f < freq_threshold
            low_freq_energy = np.sum(psd[low_freq_idx, :], axis=0)
            total_energy = np.sum(psd, axis=0)
            stability_ratio = low_freq_energy / (total_energy + 1e-8)
            
            # 插值到原始长度
            stability_interp = np.interp(
                np.arange(len(power_data)),
                np.linspace(0, len(power_data)-1, len(stability_ratio)),
                stability_ratio
            )
            
            # 基于稳定性比例确定状态
            threshold = np.percentile(stability_interp, 50)
            state = (stability_interp > threshold).astype(int)
            
            # 后处理：去除短暂状态
            state = self._post_process_state(state, min_duration=20)
            
            info = {
                'method': 'spectral',
                'window_size': window_size,
                'freq_threshold': freq_threshold,
                'stability_threshold': threshold,
                'toggles': int(np.sum(np.diff(state) != 0)),
                'avg_run': float(np.mean(self._get_run_lengths(state)))
            }
            
            return state, info
            
        except Exception as e:
            return np.zeros_like(power_data, dtype=int), {'error': str(e)}
    
    def clustering_method(self, power_data, **kwargs):
        """
        基于聚类的方法
        使用功率及其衍生特征进行聚类分析
        """
        try:
            # 参数设置
            n_clusters = kwargs.get('n_clusters', 2)
            window_size = kwargs.get('window_size', 50)
            
            # 特征工程
            features = self._extract_clustering_features(power_data, window_size)
            
            # 标准化特征
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # 使用高斯混合模型进行聚类
            gmm = GaussianMixture(n_components=n_clusters, random_state=42)
            labels = gmm.fit_predict(features_scaled)
            
            # 确定哪个簇代表ON状态（通常是功率更高的簇）
            cluster_means = []
            for i in range(n_clusters):
                cluster_power = power_data[labels == i]
                cluster_means.append(np.mean(cluster_power) if len(cluster_power) > 0 else 0)
            
            on_cluster = np.argmax(cluster_means)
            state = (labels == on_cluster).astype(int)
            
            # 后处理
            state = self._post_process_state(state, min_duration=15)
            
            info = {
                'method': 'clustering',
                'n_clusters': n_clusters,
                'cluster_means': cluster_means,
                'on_cluster': int(on_cluster),
                'toggles': int(np.sum(np.diff(state) != 0)),
                'avg_run': float(np.mean(self._get_run_lengths(state)))
            }
            
            return state, info
            
        except Exception as e:
            return np.zeros_like(power_data, dtype=int), {'error': str(e)}
    
    def change_point_method(self, power_data, **kwargs):
        """
        基于变点检测的方法
        检测功率序列中的结构性变化点
        """
        try:
            # 参数设置
            penalty = kwargs.get('penalty', 'l2')
            min_size = kwargs.get('min_size', 20)
            jump_threshold = kwargs.get('jump_threshold', None)
            
            # 计算累积和
            cumsum = np.cumsum(power_data - np.mean(power_data))
            
            # 检测变点（简化版本）
            change_points = []
            window = min_size
            
            for i in range(window, len(cumsum) - window):
                # 计算前后窗口的统计差异
                before = cumsum[i-window:i]
                after = cumsum[i:i+window]
                
                # 使用t检验统计量
                mean_diff = np.abs(np.mean(after) - np.mean(before))
                std_pooled = np.sqrt((np.var(before) + np.var(after)) / 2)
                
                if std_pooled > 0:
                    t_stat = mean_diff / std_pooled
                    if jump_threshold is None:
                        jump_threshold = np.percentile(power_data, 75) * 0.1
                    
                    if t_stat > jump_threshold:
                        change_points.append(i)
            
            # 基于变点构建状态序列
            state = np.zeros_like(power_data, dtype=int)
            
            if change_points:
                # 简单策略：交替分配状态
                current_state = 0
                last_point = 0
                
                for cp in change_points:
                    # 根据功率变化方向决定状态
                    power_before = np.mean(power_data[max(0, cp-10):cp])
                    power_after = np.mean(power_data[cp:min(len(power_data), cp+10)])
                    
                    if power_after > power_before:
                        current_state = 1
                    else:
                        current_state = 0
                    
                    state[last_point:cp] = 1 - current_state
                    last_point = cp
                
                state[last_point:] = current_state
            
            # 后处理
            state = self._post_process_state(state, min_duration=min_size)
            
            info = {
                'method': 'change_point',
                'change_points': len(change_points),
                'jump_threshold': jump_threshold,
                'toggles': int(np.sum(np.diff(state) != 0)),
                'avg_run': float(np.mean(self._get_run_lengths(state)))
            }
            
            return state, info
            
        except Exception as e:
            return np.zeros_like(power_data, dtype=int), {'error': str(e)}
    
    def wavelet_method(self, power_data, **kwargs):
        """
        基于小波变换的方法
        利用小波分解检测不同尺度的变化
        """
        try:
            from scipy import signal as scipy_signal
            import pywt
            
            # 参数设置
            wavelet = kwargs.get('wavelet', 'db4')
            scales = kwargs.get('scales', np.arange(1, 31))
            
            # 使用PyWavelets进行连续小波变换
            try:
                # 尝试使用pywt.cwt
                coeffs, freqs = pywt.cwt(power_data, scales, wavelet)
                cwt_matrix = coeffs
            except:
                # 如果pywt不可用，使用简化的小波分析
                # 基于滑动窗口的能量分析作为替代
                window_sizes = scales
                cwt_matrix = np.zeros((len(window_sizes), len(power_data)))
                
                for i, window_size in enumerate(window_sizes):
                    if window_size >= len(power_data):
                        continue
                    # 计算滑动窗口的方差作为"小波系数"
                    for j in range(len(power_data) - window_size + 1):
                        window_data = power_data[j:j+window_size]
                        cwt_matrix[i, j + window_size//2] = np.var(window_data)
            
            # 计算小波能量
            wavelet_energy = np.sum(np.abs(cwt_matrix) ** 2, axis=0)
            
            # 基于能量分布检测状态
            if len(wavelet_energy) > 21:
                energy_smooth = scipy_signal.savgol_filter(wavelet_energy, 21, 3)
            else:
                energy_smooth = wavelet_energy
            
            threshold = np.percentile(energy_smooth, 60)
            
            state = (energy_smooth > threshold).astype(int)
            
            # 后处理
            state = self._post_process_state(state, min_duration=15)
            
            info = {
                'method': 'wavelet',
                'wavelet': wavelet,
                'scales': len(scales),
                'energy_threshold': threshold,
                'toggles': int(np.sum(np.diff(state) != 0)),
                'avg_run': float(np.mean(self._get_run_lengths(state)))
            }
            
            return state, info
            
        except Exception as e:
            return np.zeros_like(power_data, dtype=int), {'error': str(e)}
    
    def hmm_like_method(self, power_data, **kwargs):
        """
        类HMM方法
        使用状态转移概率和观测概率的简化版本
        """
        try:
            # 参数设置
            n_states = kwargs.get('n_states', 2)
            transition_penalty = kwargs.get('transition_penalty', 0.1)
            
            # 计算观测概率（基于功率分布）
            power_bins = np.linspace(np.min(power_data), np.max(power_data), 50)
            power_digitized = np.digitize(power_data, power_bins)
            
            # 初始化状态概率
            state_probs = np.zeros((len(power_data), n_states))
            
            # 简化的前向算法
            for t in range(len(power_data)):
                power_val = power_data[t]
                
                # 观测概率（OFF状态偏向低功率，ON状态偏向高功率）
                if n_states == 2:
                    # OFF状态概率
                    state_probs[t, 0] = np.exp(-power_val / (np.mean(power_data) + 1e-6))
                    # ON状态概率
                    state_probs[t, 1] = 1 - state_probs[t, 0]
                
                # 考虑转移概率（惩罚频繁切换）
                if t > 0:
                    for s in range(n_states):
                        # 保持当前状态的奖励
                        same_state_bonus = 1.0
                        # 切换状态的惩罚
                        switch_penalty = transition_penalty
                        
                        # 调整概率
                        state_probs[t, s] *= same_state_bonus
            
            # Viterbi解码（简化版）
            state = np.argmax(state_probs, axis=1)
            
            # 后处理：平滑状态序列
            state = self._smooth_state_sequence(state, window=10)
            state = self._post_process_state(state, min_duration=20)
            
            info = {
                'method': 'hmm_like',
                'n_states': n_states,
                'transition_penalty': transition_penalty,
                'toggles': int(np.sum(np.diff(state) != 0)),
                'avg_run': float(np.mean(self._get_run_lengths(state)))
            }
            
            return state, info
            
        except Exception as e:
            return np.zeros_like(power_data, dtype=int), {'error': str(e)}
    
    def adaptive_threshold_method(self, power_data, **kwargs):
        """
        自适应阈值方法
        根据局部统计特性动态调整阈值
        """
        try:
            # 参数设置
            window_size = kwargs.get('window_size', 100)
            percentile = kwargs.get('percentile', 75)
            adaptation_rate = kwargs.get('adaptation_rate', 0.1)
            
            state = np.zeros_like(power_data, dtype=int)
            threshold_history = []
            
            # 初始阈值
            initial_threshold = np.percentile(power_data[:window_size], percentile)
            current_threshold = initial_threshold
            
            for i in range(len(power_data)):
                # 当前功率值
                current_power = power_data[i]
                
                # 更新状态
                state[i] = 1 if current_power > current_threshold else 0
                
                # 自适应更新阈值
                if i >= window_size:
                    # 计算局部统计
                    local_data = power_data[i-window_size:i]
                    local_threshold = np.percentile(local_data, percentile)
                    
                    # 指数移动平均更新
                    current_threshold = (1 - adaptation_rate) * current_threshold + \
                                     adaptation_rate * local_threshold
                
                threshold_history.append(current_threshold)
            
            # 后处理
            state = self._post_process_state(state, min_duration=15)
            
            info = {
                'method': 'adaptive_threshold',
                'window_size': window_size,
                'percentile': percentile,
                'adaptation_rate': adaptation_rate,
                'final_threshold': current_threshold,
                'threshold_std': np.std(threshold_history),
                'toggles': int(np.sum(np.diff(state) != 0)),
                'avg_run': float(np.mean(self._get_run_lengths(state)))
            }
            
            return state, info
            
        except Exception as e:
            return np.zeros_like(power_data, dtype=int), {'error': str(e)}
    
    def energy_based_method(self, power_data, **kwargs):
        """
        基于能量的方法
        分析功率信号的能量分布和变化模式
        """
        try:
            # 参数设置
            window_size = kwargs.get('window_size', 50)
            energy_threshold = kwargs.get('energy_threshold', None)
            
            # 计算滑动窗口能量
            energy_series = []
            for i in range(len(power_data)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(power_data), i + window_size // 2)
                window_data = power_data[start_idx:end_idx]
                
                # 计算多种能量指标
                rms_energy = np.sqrt(np.mean(window_data ** 2))
                variance_energy = np.var(window_data)
                peak_energy = np.max(window_data)
                
                # 综合能量指标
                combined_energy = 0.5 * rms_energy + 0.3 * variance_energy + 0.2 * peak_energy
                energy_series.append(combined_energy)
            
            energy_series = np.array(energy_series)
            
            # 自动确定阈值
            if energy_threshold is None:
                # 使用双峰分布的谷值作为阈值
                hist, bins = np.histogram(energy_series, bins=50)
                # 寻找直方图的局部最小值
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(-hist)  # 寻找谷值
                
                if len(peaks) > 0:
                    valley_idx = peaks[np.argmax(hist[peaks])]  # 选择最深的谷
                    energy_threshold = bins[valley_idx]
                else:
                    energy_threshold = np.percentile(energy_series, 50)
            
            # 基于能量阈值确定状态
            state = (energy_series > energy_threshold).astype(int)
            
            # 后处理
            state = self._post_process_state(state, min_duration=20)
            
            info = {
                'method': 'energy_based',
                'window_size': window_size,
                'energy_threshold': energy_threshold,
                'energy_mean': float(np.mean(energy_series)),
                'energy_std': float(np.std(energy_series)),
                'toggles': int(np.sum(np.diff(state) != 0)),
                'avg_run': float(np.mean(self._get_run_lengths(state)))
            }
            
            return state, info
            
        except Exception as e:
            return np.zeros_like(power_data, dtype=int), {'error': str(e)}
    
    def _extract_clustering_features(self, power_data, window_size):
        """提取用于聚类的特征"""
        features = []
        
        for i in range(len(power_data)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(power_data), i + window_size // 2)
            window_data = power_data[start_idx:end_idx]
            
            # 基本统计特征
            mean_power = np.mean(window_data)
            std_power = np.std(window_data)
            max_power = np.max(window_data)
            min_power = np.min(window_data)
            
            # 变化特征
            if len(window_data) > 1:
                diff_mean = np.mean(np.abs(np.diff(window_data)))
                diff_std = np.std(np.diff(window_data))
            else:
                diff_mean = 0
                diff_std = 0
            
            # 分布特征
            skewness = self._calculate_skewness(window_data)
            kurtosis = self._calculate_kurtosis(window_data)
            
            features.append([
                mean_power, std_power, max_power, min_power,
                diff_mean, diff_std, skewness, kurtosis
            ])
        
        return np.array(features)
    
    def _calculate_skewness(self, data):
        """计算偏度"""
        if len(data) < 2:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data):
        """计算峰度"""
        if len(data) < 2:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _smooth_state_sequence(self, state, window=5):
        """平滑状态序列"""
        if window <= 1:
            return state
        
        smoothed = np.copy(state)
        for i in range(window, len(state) - window):
            window_states = state[i-window:i+window+1]
            # 使用众数
            unique, counts = np.unique(window_states, return_counts=True)
            smoothed[i] = unique[np.argmax(counts)]
        
        return smoothed
    
    def _post_process_state(self, state, min_duration=10):
        """后处理：去除过短的状态段"""
        if len(state) == 0:
            return state
        
        processed = np.copy(state)
        
        # 找到所有运行段
        runs = []
        current_state = state[0]
        start_idx = 0
        
        for i in range(1, len(state)):
            if state[i] != current_state:
                runs.append((start_idx, i-1, current_state))
                start_idx = i
                current_state = state[i]
        runs.append((start_idx, len(state)-1, current_state))
        
        # 去除过短的运行段
        for start, end, run_state in runs:
            if end - start + 1 < min_duration:
                # 将短段设置为相邻段的状态
                if start > 0:
                    processed[start:end+1] = processed[start-1]
                elif end < len(state) - 1:
                    processed[start:end+1] = processed[end+1]
        
        return processed
    
    def _get_run_lengths(self, state):
        """获取运行长度"""
        if len(state) == 0:
            return [0]
        
        runs = []
        current_run = 1
        
        for i in range(1, len(state)):
            if state[i] == state[i-1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        
        return runs if runs else [1]
    
    def evaluate_all_methods(self, power_data, **kwargs):
        """评估所有方法"""
        results = {}
        
        for method_name, method_func in self.methods.items():
            try:
                state, info = method_func(power_data, **kwargs)
                
                # 计算评分
                toggles = info.get('toggles', 0)
                avg_run = info.get('avg_run', 0)
                runs = self._get_run_lengths(state)
                short_runs = sum(1 for r in runs if r < 12)
                
                # 评分函数
                if toggles == 0:
                    score = -1000
                elif toggles <= 10:
                    score = avg_run - short_runs * 10
                else:
                    score = avg_run - (toggles - 10) * 20 - short_runs * 10
                
                info['score'] = score
                info['short_runs'] = short_runs
                
                results[method_name] = {
                    'state': state,
                    'info': info
                }
                
            except Exception as e:
                results[method_name] = {
                    'state': np.zeros_like(power_data, dtype=int),
                    'info': {'error': str(e), 'score': -1000}
                }
        
        return results