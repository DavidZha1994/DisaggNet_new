"""
特征工程模块
Feature Engineering Module

实现时序数据的特征工程，包括：
- 防泄漏的归一化
- 派生特征生成
- 时间特征编码
- 统计特征计算
"""

import numpy as np
import polars as pl
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
import joblib
import logging
from datetime import datetime
import holidays

logger = logging.getLogger(__name__)

class LeakagePreventionScaler:
    """防泄漏的缩放器"""
    
    def __init__(self, scaler_type: str = 'standard', **kwargs):
        self.scaler_type = scaler_type
        self.kwargs = kwargs
        self.scalers = {}  # 每个设备/段的独立scaler
        self.feature_names = None
        
    def fit(self, X: np.ndarray, device_ids: Optional[np.ndarray] = None, 
            segment_ids: Optional[np.ndarray] = None) -> 'LeakagePreventionScaler':
        """
        拟合缩放器（仅在训练集上）
        
        Args:
            X: 特征数据
            device_ids: 设备ID数组
            segment_ids: 段ID数组
            
        Returns:
            self: 拟合后的缩放器
        """
        if device_ids is not None:
            # 按设备分别拟合
            unique_devices = np.unique(device_ids)
            for device_id in unique_devices:
                device_mask = device_ids == device_id
                device_X = X[device_mask]
                
                if len(device_X) > 0:
                    scaler = self._create_scaler()
                    scaler.fit(device_X)
                    self.scalers[f'device_{device_id}'] = scaler
        else:
            # 全局拟合
            scaler = self._create_scaler()
            scaler.fit(X)
            self.scalers['global'] = scaler
        
        logger.info(f"缩放器拟合完成: {len(self.scalers)} 个scaler")
        return self
    
    def transform(self, X: np.ndarray, device_ids: Optional[np.ndarray] = None,
                  segment_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """
        应用缩放变换
        
        Args:
            X: 特征数据
            device_ids: 设备ID数组
            segment_ids: 段ID数组
            
        Returns:
            X_scaled: 缩放后的特征数据
        """
        if not self.scalers:
            raise ValueError("缩放器未拟合")
        
        X_scaled = np.zeros_like(X)
        
        if device_ids is not None:
            # 按设备分别变换
            unique_devices = np.unique(device_ids)
            for device_id in unique_devices:
                device_mask = device_ids == device_id
                scaler_key = f'device_{device_id}'
                
                if scaler_key in self.scalers:
                    X_scaled[device_mask] = self.scalers[scaler_key].transform(X[device_mask])
                elif 'global' in self.scalers:
                    # 回退到全局scaler
                    X_scaled[device_mask] = self.scalers['global'].transform(X[device_mask])
                else:
                    logger.warning(f"设备 {device_id} 无对应scaler，保持原值")
                    X_scaled[device_mask] = X[device_mask]
        else:
            # 全局变换
            if 'global' in self.scalers:
                X_scaled = self.scalers['global'].transform(X)
            else:
                logger.warning("无全局scaler，保持原值")
                X_scaled = X
        
        return X_scaled
    
    def fit_transform(self, X: np.ndarray, device_ids: Optional[np.ndarray] = None,
                      segment_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """拟合并变换"""
        return self.fit(X, device_ids, segment_ids).transform(X, device_ids, segment_ids)
    
    def _create_scaler(self):
        """创建缩放器实例"""
        if self.scaler_type == 'standard':
            return StandardScaler(**self.kwargs)
        elif self.scaler_type == 'robust':
            return RobustScaler(**self.kwargs)
        elif self.scaler_type == 'minmax':
            return MinMaxScaler(**self.kwargs)
        else:
            raise ValueError(f"未知的缩放器类型: {self.scaler_type}")
    
    def save(self, filepath: str):
        """保存缩放器"""
        joblib.dump({
            'scaler_type': self.scaler_type,
            'kwargs': self.kwargs,
            'scalers': self.scalers,
            'feature_names': self.feature_names
        }, filepath)
        logger.info(f"缩放器已保存到: {filepath}")
    
    def load(self, filepath: str):
        """加载缩放器"""
        data = joblib.load(filepath)
        self.scaler_type = data['scaler_type']
        self.kwargs = data['kwargs']
        self.scalers = data['scalers']
        self.feature_names = data.get('feature_names')
        logger.info(f"从 {filepath} 加载缩放器")

class FeatureEngineer:
    """特征工程器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.time_features = config['feature_engineering']['time_features']
        self.statistical_features = config['feature_engineering']['statistical_features']
        self.derived_features = config['feature_engineering']['derived_features']
        self.normalization = config['feature_engineering']['normalization']
        self.feature_selection = config['feature_engineering']['feature_selection']
        
        # 初始化组件
        self.scaler = None
        self.pca = None
        self.feature_names = None
        self.selected_features = None
        
    def engineer_features(self, df: pl.DataFrame, is_training: bool = True) -> pl.DataFrame:
        """
        执行特征工程
        
        Args:
            df: 输入数据
            is_training: 是否为训练阶段
            
        Returns:
            df_engineered: 特征工程后的数据
        """
        logger.info(f"开始特征工程 (训练模式: {is_training})")
        
        df_engineered = df.clone()
        
        # 1. 时间特征
        if self.time_features['enabled']:
            df_engineered = self._add_time_features(df_engineered)
        
        # 2. 统计特征
        if self.statistical_features['enabled']:
            df_engineered = self._add_statistical_features(df_engineered)
        
        # 3. 派生特征
        if self.derived_features['enabled']:
            df_engineered = self._add_derived_features(df_engineered)
        
        # 4. 缺失值处理
        df_engineered = self._handle_missing_values(df_engineered)
        
        logger.info(f"特征工程完成: {len(df.columns)} -> {len(df_engineered.columns)} 列")
        return df_engineered
    
    def _add_time_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """添加时间特征"""
        logger.debug("添加时间特征")
        
        df_time = df.clone()
        
        # 从时间戳提取时间特征
        if 'ts_utc' in df.columns:
            df_time = df_time.with_columns([
                # 小时
                pl.col('ts_utc').map_elements(
                    lambda x: datetime.fromtimestamp(x).hour, 
                    return_dtype=pl.Int32
                ).alias('hour'),
                
                # 星期几
                pl.col('ts_utc').map_elements(
                    lambda x: datetime.fromtimestamp(x).weekday(), 
                    return_dtype=pl.Int32
                ).alias('day_of_week'),
                
                # 月份
                pl.col('ts_utc').map_elements(
                    lambda x: datetime.fromtimestamp(x).month, 
                    return_dtype=pl.Int32
                ).alias('month'),
                
                # 季度
                pl.col('ts_utc').map_elements(
                    lambda x: (datetime.fromtimestamp(x).month - 1) // 3 + 1, 
                    return_dtype=pl.Int32
                ).alias('quarter')
            ])
            
            # 周末标记
            if self.time_features.get('weekend', True):
                df_time = df_time.with_columns([
                    (pl.col('day_of_week') >= 5).alias('is_weekend')
                ])
            
            # 工作时间标记
            if self.time_features.get('business_hours', True):
                df_time = df_time.with_columns([
                    ((pl.col('hour') >= 9) & (pl.col('hour') <= 17) & 
                     (pl.col('day_of_week') < 5)).alias('is_business_hours')
                ])
            
            # 节假日标记（简化处理）
            if self.time_features.get('holidays', False):
                # 这里可以集成holidays库
                df_time = df_time.with_columns([
                    pl.lit(False).alias('is_holiday')  # 简化处理
                ])
            
            # 周期性编码
            if self.time_features.get('cyclical_encoding', True):
                df_time = df_time.with_columns([
                    # 小时的周期性编码
                    (2 * np.pi * pl.col('hour') / 24).sin().alias('hour_sin'),
                    (2 * np.pi * pl.col('hour') / 24).cos().alias('hour_cos'),
                    
                    # 星期的周期性编码
                    (2 * np.pi * pl.col('day_of_week') / 7).sin().alias('dow_sin'),
                    (2 * np.pi * pl.col('day_of_week') / 7).cos().alias('dow_cos'),
                    
                    # 月份的周期性编码
                    (2 * np.pi * pl.col('month') / 12).sin().alias('month_sin'),
                    (2 * np.pi * pl.col('month') / 12).cos().alias('month_cos')
                ])
        
        return df_time
    
    def _add_statistical_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """添加统计特征"""
        logger.debug("添加统计特征")
        
        df_stats = df.clone()
        
        # 获取数值特征列
        numeric_columns = self._get_numeric_columns(df)
        
        # 滚动窗口统计
        window_sizes = self.statistical_features.get('window_sizes', [12, 60, 300])  # 1分钟、5分钟、25分钟
        
        for window_size in window_sizes:
            for col in numeric_columns:
                if col in ['ts_utc', 't_rel']:
                    continue
                
                # 滚动均值
                if self.statistical_features.get('rolling_mean', True):
                    df_stats = df_stats.with_columns([
                        pl.col(col).rolling_mean(window_size).alias(f'{col}_rmean_{window_size}')
                    ])
                
                # 滚动标准差
                if self.statistical_features.get('rolling_std', True):
                    df_stats = df_stats.with_columns([
                        pl.col(col).rolling_std(window_size).alias(f'{col}_rstd_{window_size}')
                    ])
                
                # 滚动最值
                if self.statistical_features.get('rolling_minmax', True):
                    df_stats = df_stats.with_columns([
                        pl.col(col).rolling_min(window_size).alias(f'{col}_rmin_{window_size}'),
                        pl.col(col).rolling_max(window_size).alias(f'{col}_rmax_{window_size}')
                    ])
                
                # 滚动变化率
                if self.statistical_features.get('rolling_change', True):
                    df_stats = df_stats.with_columns([
                        (pl.col(col) - pl.col(col).shift(window_size)).alias(f'{col}_change_{window_size}'),
                        ((pl.col(col) - pl.col(col).shift(window_size)) / 
                         pl.col(col).shift(window_size)).alias(f'{col}_pct_change_{window_size}')
                    ])
        
        return df_stats
    
    def _add_derived_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """添加派生特征"""
        logger.debug("添加派生特征")
        
        df_derived = df.clone()
        
        # 功率相关派生特征
        if all(col in df.columns for col in ['P_kW', 'Q_kvar']):
            df_derived = df_derived.with_columns([
                # 视在功率
                (pl.col('P_kW').pow(2) + pl.col('Q_kvar').pow(2)).sqrt().alias('S_kVA_derived'),
                
                # 功率因数
                (pl.col('P_kW') / (pl.col('P_kW').pow(2) + pl.col('Q_kvar').pow(2)).sqrt()).alias('power_factor'),
                
                # 功率比值
                (pl.col('Q_kvar') / pl.col('P_kW')).alias('reactive_active_ratio')
            ])
        
        # 电压电流比值（阻抗）
        voltage_cols = [col for col in df.columns if col.startswith('V') and col.endswith('_V')]
        current_cols = [col for col in df.columns if col.startswith('I') and col.endswith('_A')]
        
        for v_col, i_col in zip(voltage_cols, current_cols):
            if v_col in df.columns and i_col in df.columns:
                impedance_col = f'Z_{v_col[0]}'
                df_derived = df_derived.with_columns([
                    (pl.col(v_col) / pl.col(i_col)).alias(impedance_col)
                ])
        
        # THD相关特征
        thd_cols = [col for col in df.columns if 'THD' in col]
        if len(thd_cols) > 1:
            # THD总和
            df_derived = df_derived.with_columns([
                pl.sum_horizontal(thd_cols).alias('THD_total')
            ])
            
            # THD平均
            df_derived = df_derived.with_columns([
                pl.mean_horizontal(thd_cols).alias('THD_mean')
            ])
        
        # 频率偏差
        if 'F_Hz' in df.columns:
            nominal_freq = 50.0  # 标称频率
            df_derived = df_derived.with_columns([
                (pl.col('F_Hz') - nominal_freq).alias('freq_deviation'),
                ((pl.col('F_Hz') - nominal_freq) / nominal_freq).alias('freq_deviation_pct')
            ])
        
        return df_derived
    
    def _handle_missing_values(self, df: pl.DataFrame) -> pl.DataFrame:
        """处理缺失值"""
        logger.debug("处理缺失值")
        
        # 获取数值列
        numeric_columns = self._get_numeric_columns(df)
        
        # 前向填充
        df_filled = df.with_columns([
            pl.col(col).forward_fill() for col in numeric_columns
        ])
        
        # 后向填充剩余的缺失值
        df_filled = df_filled.with_columns([
            pl.col(col).backward_fill() for col in numeric_columns
        ])
        
        # 用0填充仍然缺失的值
        df_filled = df_filled.fill_null(0)
        
        return df_filled
    
    def _get_numeric_columns(self, df: pl.DataFrame) -> List[str]:
        """获取数值列"""
        numeric_columns = []
        for col in df.columns:
            dtype = df.select(pl.col(col).dtype).item(0, 0)
            if dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                numeric_columns.append(col)
        return numeric_columns
    
    def fit_normalization(self, X: np.ndarray, device_ids: Optional[np.ndarray] = None) -> 'FeatureEngineer':
        """
        拟合归一化器（仅在训练集上）
        
        Args:
            X: 特征数据
            device_ids: 设备ID数组
            
        Returns:
            self: 拟合后的特征工程器
        """
        if not self.normalization['enabled']:
            return self
        
        logger.info("拟合归一化器")
        
        scaler_type = self.normalization.get('method', 'standard')
        scaler_kwargs = self.normalization.get('params', {})
        
        self.scaler = LeakagePreventionScaler(scaler_type, **scaler_kwargs)
        self.scaler.fit(X, device_ids)
        
        return self
    
    def apply_normalization(self, X: np.ndarray, device_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """
        应用归一化
        
        Args:
            X: 特征数据
            device_ids: 设备ID数组
            
        Returns:
            X_normalized: 归一化后的特征数据
        """
        if not self.normalization['enabled'] or self.scaler is None:
            return X
        
        return self.scaler.transform(X, device_ids)
    
    def fit_feature_selection(self, X: np.ndarray, y: np.ndarray) -> 'FeatureEngineer':
        """
        拟合特征选择（仅在训练集上）
        
        Args:
            X: 特征数据
            y: 标签数据
            
        Returns:
            self: 拟合后的特征工程器
        """
        if not self.feature_selection['enabled']:
            return self
        
        logger.info("拟合特征选择")
        
        method = self.feature_selection.get('method', 'variance')
        
        if method == 'variance':
            # 方差阈值
            threshold = self.feature_selection.get('variance_threshold', 0.01)
            variances = np.var(X, axis=0)
            self.selected_features = np.where(variances > threshold)[0]
            
        elif method == 'correlation':
            # 相关性阈值
            threshold = self.feature_selection.get('correlation_threshold', 0.95)
            corr_matrix = np.corrcoef(X.T)
            
            # 找到高相关的特征对
            high_corr_pairs = np.where(np.abs(corr_matrix) > threshold)
            features_to_remove = set()
            
            for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
                if i != j and i not in features_to_remove:
                    features_to_remove.add(j)
            
            all_features = set(range(X.shape[1]))
            self.selected_features = np.array(list(all_features - features_to_remove))
            
        elif method == 'pca':
            # PCA降维
            n_components = self.feature_selection.get('n_components', 0.95)
            self.pca = PCA(n_components=n_components)
            self.pca.fit(X)
            
        logger.info(f"特征选择完成: {X.shape[1]} -> {len(self.selected_features) if self.selected_features is not None else self.pca.n_components_} 特征")
        
        return self
    
    def apply_feature_selection(self, X: np.ndarray) -> np.ndarray:
        """
        应用特征选择
        
        Args:
            X: 特征数据
            
        Returns:
            X_selected: 选择后的特征数据
        """
        if not self.feature_selection['enabled']:
            return X
        
        if self.selected_features is not None:
            return X[:, self.selected_features]
        elif self.pca is not None:
            return self.pca.transform(X)
        else:
            return X
    
    def save_feature_engineer(self, filepath: str):
        """保存特征工程器"""
        data = {
            'config': self.config,
            'feature_names': self.feature_names,
            'selected_features': self.selected_features
        }
        
        # 保存scaler
        if self.scaler is not None:
            scaler_path = filepath.replace('.pkl', '_scaler.pkl')
            self.scaler.save(scaler_path)
            data['scaler_path'] = scaler_path
        
        # 保存PCA
        if self.pca is not None:
            pca_path = filepath.replace('.pkl', '_pca.pkl')
            joblib.dump(self.pca, pca_path)
            data['pca_path'] = pca_path
        
        joblib.dump(data, filepath)
        logger.info(f"特征工程器已保存到: {filepath}")
    
    def load_feature_engineer(self, filepath: str):
        """加载特征工程器"""
        data = joblib.load(filepath)
        
        self.config = data['config']
        self.feature_names = data.get('feature_names')
        self.selected_features = data.get('selected_features')
        
        # 加载scaler
        if 'scaler_path' in data:
            self.scaler = LeakagePreventionScaler()
            self.scaler.load(data['scaler_path'])
        
        # 加载PCA
        if 'pca_path' in data:
            self.pca = joblib.load(data['pca_path'])
        
        logger.info(f"从 {filepath} 加载特征工程器")