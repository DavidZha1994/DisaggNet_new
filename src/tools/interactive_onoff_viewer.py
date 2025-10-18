#!/usr/bin/env python3
"""
交互式开关检测结果查看器
支持动态缩放、时间范围选择、多方法对比等功能
"""

import hashlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import streamlit as st
import json
import os
import pickle
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.tools.onoff_evaluator_redux import (
    compute_absolute_state, 
    compute_delta_state, 
    compute_hybrid_state
)
from src.tools.advanced_onoff_methods import AdvancedOnOffDetector

class InteractiveOnOffViewer:
    def __init__(self, data_dir="/Users/yu/Workspace/DisaggNet_new/Data"):
        """初始化交互式查看器"""
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"  # 修改为processed目录
        self.analysis_dir = self.data_dir / "analysis_onoff"
        self.analysis_results = []
        self.best_methods = {}
        self.advanced_detector = AdvancedOnOffDetector()
        
        # 创建持久化缓存目录
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # 内存缓存
        self.device_data_cache = {}  # 缓存原始数据
        self.method_results_cache = {}  # 缓存计算结果
        
        # 加载持久化缓存
        self._load_persistent_cache()
        
        # 加载已有的分析结果
        self.load_analysis_results()
        
    def _load_persistent_cache(self):
        """加载持久化缓存"""
        try:
            # 加载设备数据缓存
            device_cache_file = self.cache_dir / "device_data_cache.pkl"
            if device_cache_file.exists():
                with open(device_cache_file, 'rb') as f:
                    self.device_data_cache = pickle.load(f)
            
            # 加载方法结果缓存
            method_cache_file = self.cache_dir / "method_results_cache.pkl"
            if method_cache_file.exists():
                with open(method_cache_file, 'rb') as f:
                    self.method_results_cache = pickle.load(f)
                    
        except Exception as e:
            print(f"加载持久化缓存时出错: {e}")
            self.device_data_cache = {}
            self.method_results_cache = {}
    
    def _save_persistent_cache(self):
        """保存持久化缓存"""
        try:
            # 保存设备数据缓存
            device_cache_file = self.cache_dir / "device_data_cache.pkl"
            with open(device_cache_file, 'wb') as f:
                pickle.dump(self.device_data_cache, f)
            
            # 保存方法结果缓存
            method_cache_file = self.cache_dir / "method_results_cache.pkl"
            with open(method_cache_file, 'wb') as f:
                pickle.dump(self.method_results_cache, f)
                
        except Exception as e:
            print(f"保存持久化缓存时出错: {e}")
    
    def _get_device_cache_key(self, device_name):
        """生成设备数据缓存键"""
        return f"device_{device_name}"
    
    def _get_method_cache_key(self, device_name, power_data):
        """生成方法结果缓存键，基于设备名称和数据特征"""
        # 使用设备名称、数据长度和数据统计特征生成稳定的缓存键
        data_stats = {
            'length': len(power_data),
            'mean': float(np.mean(power_data)),
            'std': float(np.std(power_data)),
            'max': float(np.max(power_data)),
            'min': float(np.min(power_data))
        }
        
        # 创建基于统计特征的哈希
        stats_str = json.dumps(data_stats, sort_keys=True)
        stats_hash = hashlib.md5(stats_str.encode()).hexdigest()[:16]
        
        return f"method_{device_name}_{stats_hash}"
        
    def _optimize_parameters_for_device(self, power_data, device_name):
        """为特定设备优化算法参数"""
        # 分析功率数据特征
        power_stats = {
            'mean': np.mean(power_data),
            'std': np.std(power_data),
            'max': np.max(power_data),
            'min': np.min(power_data),
            'range': np.max(power_data) - np.min(power_data),
            'cv': np.std(power_data) / np.mean(power_data) if np.mean(power_data) > 0 else 0,
            'non_zero_ratio': np.sum(power_data > 0.01) / len(power_data),
            'length': len(power_data)
        }
        
        # 基于数据特征的自适应参数
        optimized_params = {}
        
        # Absolute方法参数优化
        if power_stats['cv'] < 0.5:  # 低变异性
            optimized_params['absolute'] = {'threshold': power_stats['mean'] * 0.1}
        elif power_stats['cv'] < 1.0:  # 中等变异性
            optimized_params['absolute'] = {'threshold': power_stats['mean'] * 0.15}
        else:  # 高变异性
            optimized_params['absolute'] = {'threshold': power_stats['mean'] * 0.2}
        
        # Delta方法参数优化
        delta_threshold = min(power_stats['std'] * 2, power_stats['range'] * 0.1)
        optimized_params['delta'] = {'threshold': delta_threshold}
        
        # 高级方法参数优化
        # Spectral方法
        if power_stats['length'] > 10000:
            nperseg = 1024
        elif power_stats['length'] > 1000:
            nperseg = 256
        else:
            nperseg = min(64, power_stats['length'] // 4)
        
        optimized_params['spectral'] = {
            'nperseg': nperseg,
            'freq_threshold': 0.1 if power_stats['cv'] > 1.0 else 0.05
        }
        
        # Clustering方法
        optimized_params['clustering'] = {
            'n_clusters': 3 if power_stats['non_zero_ratio'] > 0.5 else 2,
            'window_size': max(10, min(50, power_stats['length'] // 100))
        }
        
        # Change Point方法
        optimized_params['change_point'] = {
            'penalty': power_stats['std'] * 10,
            'min_size': max(5, power_stats['length'] // 200)
        }
        
        # Wavelet方法
        max_scale = min(30, power_stats['length'] // 10)
        optimized_params['wavelet'] = {
            'scales': np.arange(1, max_scale + 1),
            'wavelet': 'db4' if power_stats['cv'] < 1.0 else 'haar'
        }
        
        # HMM-like方法
        optimized_params['hmm_like'] = {
            'n_states': 3 if power_stats['non_zero_ratio'] > 0.7 else 2,
            'window_size': max(20, min(100, power_stats['length'] // 50))
        }
        
        # Adaptive Threshold方法
        optimized_params['adaptive_threshold'] = {
            'window_size': max(50, min(200, power_stats['length'] // 20)),
            'alpha': 0.1 if power_stats['cv'] < 0.5 else 0.2
        }
        
        # Energy方法
        optimized_params['energy_based'] = {
            'window_size': max(20, min(100, power_stats['length'] // 30)),
            'overlap': 0.5 if power_stats['cv'] > 1.0 else 0.3
        }
        
        return optimized_params
        
    def load_analysis_results(self):
        """加载已有的分析结果"""
        try:
            report_path = self.analysis_dir / "onoff_method_report.json"
            if report_path.exists():
                with open(report_path, 'r') as f:
                    self.analysis_results = json.load(f)
            else:
                self.analysis_results = []
                
            best_methods_path = self.analysis_dir / "onoff_best_methods.json"
            if best_methods_path.exists():
                with open(best_methods_path, 'r') as f:
                    self.best_methods = json.load(f)
            else:
                self.best_methods = {}
        except Exception as e:
            st.error(f"加载分析结果失败: {e}")
            self.analysis_results = []
            self.best_methods = {}
    
    def get_available_devices(self):
        """获取可用的设备列表"""
        devices = []
        for result in self.analysis_results:
            filename = result['file']
            # 从文件名提取设备名
            if 'cleaned_' in filename:
                device_name = filename.split('cleaned_')[1].split('_PhaseCount')[0]
                devices.append(device_name)
        return sorted(list(set(devices)))
    
    def load_device_data(self, device_name):
        """加载指定设备的原始数据（带缓存）"""
        # 生成缓存键
        cache_key = self._get_device_cache_key(device_name)
        
        # 检查内存缓存
        if cache_key in self.device_data_cache:
            return self.device_data_cache[cache_key]
        
        # 查找对应的文件
        target_file = None
        for result in self.analysis_results:
            if device_name in result['file']:
                target_file = result['file']
                break
        
        if not target_file:
            return None, None
            
        file_path = self.processed_dir / target_file
        if not file_path.exists():
            return None, None
            
        try:
            df = pd.read_csv(file_path)
            # 确保时间列存在并转换
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['time'])
            else:
                # 创建时间索引
                df['timestamp'] = pd.date_range(start='2017-10-01', periods=len(df), freq='1min')
            
            power_data = df['P_kW'].values
            
            # 缓存结果到内存
            self.device_data_cache[cache_key] = (df, power_data)
            
            # 保存到持久化缓存
            self._save_persistent_cache()
            
            return df, power_data
        except Exception as e:
            st.error(f"加载设备数据失败: {e}")
            return None, None
    
    def compute_all_methods(self, power_data, device_name):
        """计算所有检测方法的结果（带缓存和自适应参数）"""
        # 生成稳定的缓存键
        cache_key = self._get_method_cache_key(device_name, power_data)
        
        if cache_key in self.method_results_cache:
            st.info("✅ 从持久化缓存加载检测结果，无需重新计算")
            return self.method_results_cache[cache_key]
        
        # 获取设备特定的优化参数
        optimized_params = self._optimize_parameters_for_device(power_data, device_name)
        
        st.info("🔧 使用设备特定的优化参数进行计算...")
        
        # 显示参数优化信息
        with st.expander("📊 查看优化后的参数设置"):
            for method, params in optimized_params.items():
                st.write(f"**{method.title()}方法:**")
                for param, value in params.items():
                    if isinstance(value, float):
                        st.write(f"  - {param}: {value:.4f}")
                    elif isinstance(value, np.ndarray):
                        st.write(f"  - {param}: 数组长度 {len(value)}")
                    else:
                        st.write(f"  - {param}: {value}")
        
        methods = {}
        
        # 显示计算进度
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_methods = 10  # 3个传统方法 + 7个先进方法
        completed = 0
        
        # 传统方法 - 这些方法不支持参数优化，使用固定实现
        try:
            status_text.text("正在计算 Absolute Threshold 方法...")
            progress_bar.progress(completed / total_methods)
            
            # Absolute方法使用固定实现（不支持参数传递）
            abs_state, abs_info = compute_absolute_state(power_data)
            
            # 为传统方法添加统一评分
            abs_score = self._calculate_unified_score(abs_state, abs_info)
            abs_info['score'] = abs_score
            
            methods['absolute'] = {
                'state': abs_state,
                'info': abs_info,
                'name': 'Absolute Threshold',
                'category': 'traditional'
            }
            completed += 1
            progress_bar.progress(completed / total_methods)
            st.write(f"✅ Absolute方法完成 - 阈值: {abs_info.get('thr', 'N/A'):.3f} - 评分: {abs_score:.2f}")
            
        except Exception as e:
            st.warning(f"Absolute方法计算失败: {e}")
            completed += 1
            
        try:
            status_text.text("正在计算 Delta (Normalized) 方法...")
            progress_bar.progress(completed / total_methods)
            
            # Delta方法使用固定实现（不支持参数传递）
            delta_state, delta_info = compute_delta_state(power_data)
            
            # 为传统方法添加统一评分
            delta_score = self._calculate_unified_score(delta_state, delta_info)
            delta_info['score'] = delta_score
            
            methods['delta'] = {
                'state': delta_state,
                'info': delta_info,
                'name': 'Delta (Normalized)',
                'category': 'traditional'
            }
            completed += 1
            progress_bar.progress(completed / total_methods)
            st.write(f"✅ Delta方法完成 - 切换次数: {delta_info.get('toggles', 'N/A')} - 评分: {delta_score:.2f}")
            
        except Exception as e:
            st.warning(f"Delta方法计算失败: {e}")
            completed += 1
            
        try:
            status_text.text("正在计算 Hybrid 方法...")
            progress_bar.progress(completed / total_methods)
            
            # Hybrid方法不需要特殊参数
            hybrid_state, hybrid_info = compute_hybrid_state(power_data)
            
            # 为传统方法添加统一评分（Hybrid方法已经有score，但我们使用统一的计算方式）
            hybrid_score = self._calculate_unified_score(hybrid_state, hybrid_info)
            hybrid_info['score'] = hybrid_score
            
            methods['hybrid'] = {
                'state': hybrid_state,
                'info': hybrid_info,
                'name': 'Hybrid Method',
                'category': 'traditional'
            }
            completed += 1
            progress_bar.progress(completed / total_methods)
            st.write(f"✅ Hybrid方法完成 - 评分: {hybrid_score:.2f}")
            
        except Exception as e:
            st.warning(f"Hybrid方法计算失败: {e}")
            completed += 1
        
        # 先进方法 - 使用优化参数
        advanced_method_names = ['spectral', 'clustering', 'change_point', 'wavelet', 'hmm_like', 'adaptive_threshold', 'energy_based']
        
        for method_name in advanced_method_names:
            try:
                status_text.text(f"正在计算 {method_name.title()} 方法...")
                progress_bar.progress(completed / total_methods)
                
                # 获取该方法的优化参数
                method_params = optimized_params.get(method_name, {})
                
                # 调用对应的方法
                if hasattr(self.advanced_detector, f'{method_name}_method'):
                    method_func = getattr(self.advanced_detector, f'{method_name}_method')
                    # 高级方法返回 (state, info) 元组
                    state, info = method_func(power_data, **method_params)
                    
                    if 'error' not in info:
                        # 为高级方法添加统一的评分计算
                        score = self._calculate_unified_score(state, info)
                        info['score'] = score
                        
                        methods[f'advanced_{method_name}'] = {
                            'state': state,
                            'info': info,
                            'name': f'Advanced: {method_name.title()}',
                            'category': 'advanced'
                        }
                        st.write(f"✅ {method_name.title()}方法完成 - 评分: {score:.2f}")
                    else:
                        st.write(f"❌ {method_name.title()}方法失败: {info['error']}")
                else:
                    st.write(f"❌ {method_name.title()}方法不存在")
                
                completed += 1
                progress_bar.progress(completed / total_methods)
                
            except Exception as e:
                st.warning(f"{method_name.title()}方法计算失败: {e}")
                completed += 1
                progress_bar.progress(completed / total_methods)
        
        # 完成
        progress_bar.progress(1.0)
        status_text.text("所有方法计算完成！")
        
        # 缓存结果到内存
        self.method_results_cache[cache_key] = methods
        
        # 保存到持久化缓存
        self._save_persistent_cache()
        
        st.success(f"✅ 计算完成！共计算了 {len(methods)} 种检测方法，使用了设备优化参数，结果已缓存")
        
        # 清理进度显示
        progress_bar.empty()
        status_text.empty()
        
        return methods
    
    def _calculate_unified_score(self, state, info):
        """
        为所有检测方法计算统一的评分
        评分标准：
        - 基础分：平均运行长度 (avg_run)
        - 惩罚项：短运行段数量、切换次数过多
        - 奖励项：稳定的开关模式
        """
        try:
            # 计算基本统计信息
            toggles = info.get('toggles', int(np.sum(np.diff(state) != 0)))
            
            # 计算运行长度
            runs = self._get_run_lengths(state)
            if len(runs) == 0:
                return -1000.0
                
            avg_run = float(np.mean(runs))
            short_runs = sum(1 for r in runs if r < 12)  # 短于12个时间点的运行段
            
            # 基础评分：平均运行长度
            base_score = avg_run
            
            # 惩罚项
            toggle_penalty = 0
            if toggles > 10:
                toggle_penalty = (toggles - 10) * 20  # 过多切换的惩罚
            
            short_run_penalty = short_runs * 10  # 短运行段惩罚
            
            # 计算最终评分
            final_score = base_score - toggle_penalty - short_run_penalty
            
            # 特殊情况处理
            if toggles == 0:  # 没有切换（全0或全1状态）
                final_score = -1000.0
            
            return float(final_score)
            
        except Exception as e:
            return -1000.0
    
    def _get_run_lengths(self, state):
        """计算状态序列中每个连续段的长度"""
        if len(state) == 0:
            return []
        
        runs = []
        current_run = 1
        
        for i in range(1, len(state)):
            if state[i] == state[i-1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        
        runs.append(current_run)  # 添加最后一个运行段
        return runs
    
    def create_power_plot(self, df, power_data, device_name, sync_key="power_plot"):
        """创建独立的可缩放功率时间序列图"""
        # 数据降采样以避免消息大小限制
        max_points = 10000  # 限制最大数据点数
        if len(df) > max_points:
            step = len(df) // max_points
            df_sampled = df.iloc[::step].copy()
            power_sampled = power_data[::step]
        else:
            df_sampled = df.copy()
            power_sampled = power_data
        
        fig = go.Figure()
        
        # 添加功率数据
        fig.add_trace(
            go.Scatter(
                x=df_sampled['timestamp'],
                y=power_sampled,
                mode='lines',
                name='功率 (kW)',
                line=dict(color='#1f77b4', width=1.5),
                hovertemplate='<b>时间</b>: %{x}<br><b>功率</b>: %{y:.3f} kW<extra></extra>'
            )
        )
        
        # 检查是否有存储的时间范围
        time_range_key = f"{sync_key}_time_range"
        if time_range_key in st.session_state:
            time_range = st.session_state[time_range_key]
            xaxis_range = [time_range[0], time_range[1]]
        else:
            xaxis_range = None
        
        # 更新布局，启用缩放功能
        fig.update_layout(
            title=f'{device_name} - 功率时间序列 (可缩放)',
            xaxis_title="时间",
            yaxis_title="功率 (kW)",
            height=500,  # 增加高度
            showlegend=True,
            hovermode='x unified',
            # 启用缩放和平移
            xaxis=dict(
                rangeslider=dict(visible=True),  # 添加范围滑块
                type='date',
                range=xaxis_range  # 设置时间范围
            ),
            # 添加工具栏按钮
            modebar=dict(
                add=['pan2d', 'select2d', 'lasso2d', 'resetScale2d', 'autoScale2d']
            ),
            # 添加唯一标识符用于时间范围同步
            uirevision=sync_key
        )
        
        # 如果进行了降采样，添加提示
        if len(df) > max_points:
            fig.add_annotation(
                text=f"数据已降采样 (显示 {len(df_sampled)}/{len(df)} 点)",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=10, color="red"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="red",
                borderwidth=1
            )
        
        return fig
    
    def create_state_plot(self, df, methods, device_name, sync_key="state_plot"):
        """创建独立的可缩放开关状态图"""
        if not methods:
            return None
        
        # 数据降采样以避免消息大小限制
        max_points = 10000  # 限制最大数据点数
        if len(df) > max_points:
            step = len(df) // max_points
            df_sampled = df.iloc[::step].copy()
            # 对状态数据也进行降采样
            methods_sampled = {}
            for key, method_data in methods.items():
                methods_sampled[key] = {
                    'state': method_data['state'][::step],
                    'info': method_data['info'],
                    'name': method_data['name'],
                    'category': method_data['category']
                }
        else:
            df_sampled = df.copy()
            methods_sampled = methods
            
        fig = go.Figure()
        
        # 为每种方法添加状态线
        colors = ['#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i, (method_key, method_data) in enumerate(methods_sampled.items()):
            state = method_data['state']
            color = colors[i % len(colors)]
            
            # 添加状态线，使用偏移避免重叠
            y_offset = i * 1.2  # 增加偏移量以更好地分离不同方法
            fig.add_trace(
                go.Scatter(
                    x=df_sampled['timestamp'],
                    y=state + y_offset,
                    mode='lines',
                    name=method_data['name'],
                    line=dict(color=color, width=2),
                    hovertemplate=f'<b>{method_data["name"]}</b><br>时间: %{{x}}<br>状态: %{{customdata}}<extra></extra>',
                    customdata=['ON' if s == 1 else 'OFF' for s in state]
                )
            )
            
            # 添加填充区域显示ON状态（简化以减少数据量）
            on_periods = []
            current_start = None
            
            for j, s in enumerate(state):
                if s == 1 and current_start is None:
                    current_start = j
                elif s == 0 and current_start is not None:
                    on_periods.append((current_start, j-1))
                    current_start = None
            
            # 处理最后一个ON期间
            if current_start is not None:
                on_periods.append((current_start, len(state)-1))
            
            # 限制填充区域数量以减少数据量
            max_shapes = 100
            if len(on_periods) > max_shapes:
                # 只显示较长的ON期间
                on_periods = sorted(on_periods, key=lambda x: x[1] - x[0], reverse=True)[:max_shapes]
            
            # 添加ON状态的填充区域
            for start, end in on_periods:
                if start < len(df_sampled) and end < len(df_sampled):
                    fig.add_shape(
                        type="rect",
                        x0=df_sampled.iloc[start]['timestamp'],
                        x1=df_sampled.iloc[end]['timestamp'],
                        y0=y_offset - 0.05,
                        y1=y_offset + 1.05,
                        fillcolor=color,
                        opacity=0.2,
                        layer="below",
                        line_width=0
                    )
        
        # 检查是否有存储的时间范围
        time_range_key = f"{sync_key}_time_range"
        if time_range_key in st.session_state:
            time_range = st.session_state[time_range_key]
            xaxis_range = [time_range[0], time_range[1]]
        else:
            xaxis_range = None
        
        # 更新布局
        fig.update_layout(
            title=f'{device_name} - 开关状态对比 (可缩放)',
            xaxis_title="时间",
            yaxis_title="开关状态",
            height=max(500, len(methods_sampled) * 120),  # 根据方法数量动态调整高度
            showlegend=True,
            hovermode='x unified',
            # 启用缩放和平移
            xaxis=dict(
                rangeslider=dict(visible=True),  # 添加范围滑块
                type='date',
                range=xaxis_range  # 设置时间范围
            ),
            yaxis=dict(
                range=[-0.2, len(methods_sampled) * 1.2 + 0.2],  # 调整y轴范围以匹配新的偏移量
                tickvals=[i * 1.2 + 0.5 for i in range(len(methods_sampled))],
                ticktext=[method_data['name'] for method_data in methods_sampled.values()]
            ),
            # 添加工具栏按钮
            modebar=dict(
                add=['pan2d', 'select2d', 'lasso2d', 'resetScale2d', 'autoScale2d']
            ),
            # 添加唯一标识符用于时间范围同步
            uirevision=sync_key
        )
        
        # 如果进行了降采样，添加提示
        if len(df) > max_points:
            fig.add_annotation(
                text=f"数据已降采样 (显示 {len(df_sampled)}/{len(df)} 点)",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=10, color="red"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="red",
                borderwidth=1
            )
        
        return fig

    def create_interactive_plot(self, df, power_data, methods, device_name, time_range=None):
        """创建交互式图表"""
        # 时间范围选择
        if time_range:
            start_idx = max(0, int(time_range[0] * len(df)))
            end_idx = min(len(df), int(time_range[1] * len(df)))
            df_plot = df.iloc[start_idx:end_idx].copy()
            power_plot = power_data[start_idx:end_idx]
        else:
            df_plot = df.copy()
            power_plot = power_data
            start_idx = 0
        
        # 创建子图
        n_methods = len(methods)
        fig = make_subplots(
            rows=n_methods + 1, cols=1,
            shared_xaxes=True,
            subplot_titles=['功率数据'] + [methods[key]['name'] for key in methods.keys()],
            vertical_spacing=0.02,
            row_heights=[0.4] + [0.6/n_methods] * n_methods
        )
        
        # 添加功率数据
        fig.add_trace(
            go.Scatter(
                x=df_plot['timestamp'],
                y=power_plot,
                mode='lines',
                name='功率 (kW)',
                line=dict(color='blue', width=1),
                hovertemplate='时间: %{x}<br>功率: %{y:.3f} kW<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 为每种方法添加状态图
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for i, (method_key, method_data) in enumerate(methods.items()):
            state = method_data['state'][start_idx:start_idx+len(df_plot)]
            info = method_data['info']
            
            # 创建状态填充区域
            on_periods = []
            current_start = None
            
            for j, s in enumerate(state):
                if s == 1 and current_start is None:
                    current_start = j
                elif s == 0 and current_start is not None:
                    on_periods.append((current_start, j-1))
                    current_start = None
            
            # 处理最后一个ON期间
            if current_start is not None:
                on_periods.append((current_start, len(state)-1))
            
            # 添加ON状态的填充区域
            for start, end in on_periods:
                fig.add_shape(
                    type="rect",
                    x0=df_plot.iloc[start]['timestamp'],
                    x1=df_plot.iloc[end]['timestamp'],
                    y0=0,
                    y1=1,
                    fillcolor=colors[i % len(colors)],
                    opacity=0.3,
                    layer="below",
                    line_width=0,
                    row=i+2, col=1
                )
            
            # 添加状态线
            fig.add_trace(
                go.Scatter(
                    x=df_plot['timestamp'],
                    y=state,
                    mode='lines',
                    name=f'{method_data["name"]}',
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=f'{method_data["name"]}<br>时间: %{{x}}<br>状态: %{{y}}<extra></extra>',
                    yaxis=f'y{i+2}' if i > 0 else 'y2'
                ),
                row=i+2, col=1
            )
            
            # 添加方法信息作为注释
            toggles = info.get('toggles', 0)
            avg_run = info.get('avg_run', 0)
            fig.add_annotation(
                text=f"切换: {toggles}, 平均运行: {avg_run:.1f}",
                xref="paper", yref="paper",
                x=0.02, y=1 - (i+1.5)/(n_methods+1),
                showarrow=False,
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1
            )
        
        # 更新布局
        fig.update_layout(
            title=f'{device_name} 开关检测结果对比',
            height=200 + n_methods * 150,
            showlegend=True,
            hovermode='x unified',
            xaxis_title="时间",
            font=dict(size=12)
        )
        
        # 设置y轴
        fig.update_yaxes(title_text="功率 (kW)", row=1, col=1)
        for i in range(n_methods):
            fig.update_yaxes(
                title_text="状态", 
                row=i+2, col=1,
                range=[-0.1, 1.1],
                tickvals=[0, 1],
                ticktext=['OFF', 'ON']
            )
        
        return fig
    
    def run_streamlit_app(self):
        """运行Streamlit应用"""
        st.set_page_config(
            page_title="交互式开关检测查看器",
            page_icon="🔌",
            layout="wide"
        )
        
        st.title("🔌 交互式开关检测结果查看器")
        st.markdown("---")
        
        # 侧边栏控制
        st.sidebar.header("控制面板")
        
        # 在侧边栏添加缓存状态信息
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 缓存状态")
        st.sidebar.info(f"已缓存设备数据: {len(self.device_data_cache)}")
        st.sidebar.info(f"已缓存计算结果: {len(self.method_results_cache)}")
        
        if st.sidebar.button("🗑️ 清空缓存"):
            # 清空内存缓存
            self.device_data_cache.clear()
            self.method_results_cache.clear()
            
            # 清空持久化缓存文件
            try:
                device_cache_file = self.cache_dir / "device_data_cache.pkl"
                method_cache_file = self.cache_dir / "method_results_cache.pkl"
                
                if device_cache_file.exists():
                    device_cache_file.unlink()
                if method_cache_file.exists():
                    method_cache_file.unlink()
                    
                st.sidebar.success("内存和持久化缓存已全部清空")
            except Exception as e:
                st.sidebar.error(f"清空持久化缓存时出错: {e}")
                st.sidebar.success("内存缓存已清空")
            
            st.rerun()
        
        st.sidebar.markdown("---")
        
        # 设备选择
        devices = self.get_available_devices()
        if not devices:
            st.error("未找到可用的设备数据")
            return
            
        selected_device = st.sidebar.selectbox("选择设备", devices)
        
        # 加载设备数据
        df, power_data = self.load_device_data(selected_device)
        if df is None or power_data is None:
            st.error(f"无法加载设备 {selected_device} 的数据")
            return
        
        # 时间范围选择
        st.sidebar.subheader("时间范围选择")
        time_range = st.sidebar.slider(
            "选择查看的时间范围 (%)",
            min_value=0.0,
            max_value=100.0,
            value=(0.0, 100.0),
            step=0.1,
            format="%.1f%%"
        )
        time_range = (time_range[0]/100, time_range[1]/100)
        
        # 方法选择
        st.sidebar.subheader("检测方法选择")
        
        # 按类别组织方法选择
        st.sidebar.write("**传统方法:**")
        traditional_methods = ['absolute', 'delta', 'hybrid']
        selected_traditional = []
        for method in traditional_methods:
            if st.sidebar.checkbox(method.title(), value=True, key=f"trad_{method}"):
                selected_traditional.append(method)
        
        st.sidebar.write("**先进方法:**")
        advanced_methods = ['spectral', 'clustering', 'change_point', 'wavelet', 'hmm_like', 'adaptive_threshold', 'energy_based']
        selected_advanced = []
        for method in advanced_methods:
            if st.sidebar.checkbox(method.replace('_', ' ').title(), value=False, key=f"adv_{method}"):
                selected_advanced.append(f'advanced_{method}')
        
        selected_methods = selected_traditional + selected_advanced
        
        # 计算所有方法的结果
        with st.spinner("计算检测结果..."):
            all_methods = self.compute_all_methods(power_data, selected_device)
            
        # 过滤选中的方法
        filtered_methods = {k: v for k, v in all_methods.items() if k in selected_methods}
        
        if not filtered_methods:
            st.warning("请至少选择一种检测方法")
            return
        
        # 显示设备信息
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("设备名称", selected_device)
        with col2:
            st.metric("数据点数", len(power_data))
        with col3:
            best_method = self.best_methods.get(selected_device, "未知")
            st.metric("推荐方法", best_method)
        
        # 显示功率统计信息
        st.subheader("功率数据统计")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("最大功率", f"{np.max(power_data):.3f} kW")
        with col2:
            st.metric("平均功率", f"{np.mean(power_data):.3f} kW")
        with col3:
            st.metric("功率标准差", f"{np.std(power_data):.3f} kW")
        with col4:
            non_zero_ratio = np.sum(power_data > 0.01) / len(power_data) * 100
            st.metric("非零功率比例", f"{non_zero_ratio:.1f}%")
        
        # 显示图表选择选项
        st.subheader("📊 可视化选项")
        
        # 图表类型选择
        chart_type = st.radio(
            "选择图表类型:",
            ["分离式可缩放图表", "组合式图表"],
            index=0,
            help="分离式图表提供更好的缩放体验，组合式图表便于对比"
        )
        
        if chart_type == "分离式可缩放图表":
            # 使用相同的同步键实现时间范围联动
            sync_key = f"{selected_device}_sync"
            
            # 添加时间范围选择器
            st.subheader("⏰ 时间范围选择")
            col1, col2 = st.columns(2)
            
            with col1:
                start_time = st.date_input(
                    "开始时间",
                    value=df['timestamp'].min().date(),
                    min_value=df['timestamp'].min().date(),
                    max_value=df['timestamp'].max().date(),
                    key=f"{sync_key}_start"
                )
            
            with col2:
                end_time = st.date_input(
                    "结束时间", 
                    value=df['timestamp'].max().date(),
                    min_value=df['timestamp'].min().date(),
                    max_value=df['timestamp'].max().date(),
                    key=f"{sync_key}_end"
                )
            
            # 转换为datetime对象
            import datetime
            start_datetime = datetime.datetime.combine(start_time, datetime.time.min)
            end_datetime = datetime.datetime.combine(end_time, datetime.time.max)
            
            # 存储时间范围到session_state
            time_range_key = f"{sync_key}_time_range"
            st.session_state[time_range_key] = [start_datetime, end_datetime]
            
            # 显示独立的可缩放功率图
            st.subheader("🔌 功率时间序列")
            power_fig = self.create_power_plot(df, power_data, selected_device, sync_key)
            st.plotly_chart(power_fig, use_container_width=True)
            
            # 显示独立的可缩放状态图
            if filtered_methods:
                st.subheader("🔄 开关状态对比")
                state_fig = self.create_state_plot(df, filtered_methods, selected_device, sync_key)
                if state_fig:
                    st.plotly_chart(state_fig, use_container_width=True)
                    
            # 添加联动说明
            st.info("💡 提示：使用上方的时间范围选择器来同步两个图表的显示范围。图表会自动更新以显示选定的时间段。")
        else:
            # 显示原有的组合图表
            st.subheader("📈 组合图表")
            fig = self.create_interactive_plot(df, power_data, filtered_methods, selected_device, time_range)
            st.plotly_chart(fig, use_container_width=True)
        
        # 显示方法详细信息
        st.subheader("检测方法详细信息")
        for method_key, method_data in filtered_methods.items():
            with st.expander(f"{method_data['name']} 详细信息"):
                info = method_data['info']
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**基本统计:**")
                    st.write(f"- 切换次数: {info.get('toggles', 0)}")
                    st.write(f"- 平均运行长度: {info.get('avg_run', 0):.2f}")
                    st.write(f"- 短运行数: {info.get('short_runs', 0)}")
                    
                with col2:
                    st.write("**参数设置:**")
                    for key, value in info.items():
                        if key not in ['toggles', 'avg_run', 'short_runs']:
                            if isinstance(value, float):
                                st.write(f"- {key}: {value:.4f}")
                            else:
                                st.write(f"- {key}: {value}")
        
        # 导出功能
        st.sidebar.subheader("导出选项")
        if st.sidebar.button("导出当前结果"):
            # 准备导出数据
            export_data = {
                'device': selected_device,
                'time_range': time_range,
                'methods': {k: v['info'] for k, v in filtered_methods.items()},
                'power_stats': {
                    'max': float(np.max(power_data)),
                    'mean': float(np.mean(power_data)),
                    'std': float(np.std(power_data)),
                    'non_zero_ratio': float(non_zero_ratio)
                }
            }
            
            # 保存到文件
            export_path = self.analysis_dir / f"{selected_device}_interactive_results.json"
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            st.sidebar.success(f"结果已导出到: {export_path}")

def main():
    """主函数"""
    viewer = InteractiveOnOffViewer()
    viewer.run_streamlit_app()

if __name__ == "__main__":
    main()