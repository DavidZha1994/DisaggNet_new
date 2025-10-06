"""Walk-Forward交叉验证实现"""

import logging
import pickle
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .data.datamodule import NILMDataModule  # 使用新的工业级数据模块
from .train import NILMLightningModule, create_trainer, setup_logging
from .models.fusion_transformer import FusionTransformer
from .utils.metrics import NILMMetrics

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardFold:
    """Walk-Forward折叠数据结构"""
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    
    def to_dict(self) -> Dict:
        return {
            'fold_id': self.fold_id,
            'train_start': self.train_start.isoformat(),
            'train_end': self.train_end.isoformat(),
            'val_start': self.val_start.isoformat(),
            'val_end': self.val_end.isoformat(),
            'test_start': self.test_start.isoformat(),
            'test_end': self.test_end.isoformat()
        }


class WalkForwardValidator:
    """Walk-Forward交叉验证器"""
    
    def __init__(self, config: DictConfig, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.wf_config = config.walk_forward
        
        # 加载设备信息
        from .train import load_device_info
        self.device_info, self.device_names = load_device_info(config)
        
        # 创建输出目录
        self.wf_output_dir = output_dir / "walk_forward"
        self.wf_output_dir.mkdir(exist_ok=True)
        
        # 缓存目录
        self.cache_dir = self.wf_output_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # 结果存储
        self.fold_results: List[Dict] = []
        self.folds: List[WalkForwardFold] = []
        
    def _get_cache_path(self, cache_type: str) -> Path:
        """获取缓存文件路径"""
        config_hash = self._get_config_hash()
        return self.cache_dir / f"{cache_type}_{config_hash}.pkl"
    
    def _get_config_hash(self) -> str:
        """生成配置哈希值"""
        import hashlib
        
        # 只包含影响数据分割的配置
        relevant_config = {
            'walk_forward': OmegaConf.to_container(self.wf_config),
            'data_paths': {
                'data_dir': str(self.config.paths.data_dir),
                # 使用 prepared 目录而非 H5 文件，避免对 H5 的依赖
                'prepared_dir': str(Path(self.config.paths.data_dir) / 'prepared')
            },
            'windowing': OmegaConf.to_container(self.config.windowing)
        }
        
        config_str = str(sorted(relevant_config.items()))
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _load_cached_folds(self) -> Optional[List[WalkForwardFold]]:
        """加载缓存的折叠信息"""
        cache_path = self._get_cache_path("folds")
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # 验证缓存完整性
                if 'folds' in cached_data and 'config_hash' in cached_data:
                    if cached_data['config_hash'] == self._get_config_hash():
                        logger.info(f"Loading cached folds from {cache_path}")
                        
                        # 重建WalkForwardFold对象
                        folds = []
                        for fold_data in cached_data['folds']:
                            fold = WalkForwardFold(
                                fold_id=fold_data['fold_id'],
                                train_start=pd.Timestamp(fold_data['train_start']),
                                train_end=pd.Timestamp(fold_data['train_end']),
                                val_start=pd.Timestamp(fold_data['val_start']),
                                val_end=pd.Timestamp(fold_data['val_end']),
                                test_start=pd.Timestamp(fold_data['test_start']),
                                test_end=pd.Timestamp(fold_data['test_end'])
                            )
                            folds.append(fold)
                        
                        logger.info(f"Loaded {len(folds)} cached folds")
                        return folds
                    else:
                        logger.info("Cached folds are outdated, will regenerate")
                        
            except Exception as e:
                logger.warning(f"Failed to load cached folds: {e}")
        
        return None
    
    def _save_cached_folds(self, folds: List[WalkForwardFold]) -> None:
        """保存折叠信息到缓存"""
        cache_path = self._get_cache_path("folds")
        
        try:
            cached_data = {
                'folds': [fold.to_dict() for fold in folds],
                'config_hash': self._get_config_hash(),
                'timestamp': datetime.now().isoformat(),
                'num_folds': len(folds)
            }
            
            # 原子性写入
            temp_path = cache_path.with_suffix('.tmp')
            with open(temp_path, 'wb') as f:
                pickle.dump(cached_data, f)
            temp_path.replace(cache_path)
            
            logger.info(f"Cached {len(folds)} folds to {cache_path}")
            
        except Exception as e:
            logger.warning(f"Failed to cache folds: {e}")
    
    def generate_folds(self, data_df: pd.DataFrame) -> List[WalkForwardFold]:
        """生成Walk-Forward折叠"""
        # 尝试加载缓存的折叠
        cached_folds = self._load_cached_folds()
        if cached_folds is not None:
            self.folds = cached_folds
            return self.folds
        
        logger.info("Generating Walk-Forward folds...")
        
        # 确保数据按时间排序
        timestamp_col = self.config.data.timestamp_col
        data_df = data_df.sort_values(timestamp_col)
        
        start_time = data_df[timestamp_col].min()
        end_time = data_df[timestamp_col].max()
        total_duration = end_time - start_time
        
        # 计算折叠参数
        num_folds = self.wf_config.num_folds
        train_size = pd.Timedelta(days=self.wf_config.train_days)
        val_size = pd.Timedelta(days=self.wf_config.val_days)
        test_size = pd.Timedelta(days=self.wf_config.test_days)
        step_size = pd.Timedelta(days=self.wf_config.step_days)
        
        folds = []
        current_start = start_time
        
        for fold_id in range(num_folds):
            # 训练集时间范围
            train_start = current_start
            train_end = train_start + train_size
            
            # 验证集时间范围
            val_start = train_end
            val_end = val_start + val_size
            
            # 测试集时间范围
            test_start = val_end
            test_end = test_start + test_size
            
            # 检查是否超出数据范围
            if test_end > end_time:
                logger.info(f"Stopping at fold {fold_id}: insufficient data")
                break
            
            fold = WalkForwardFold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end
            )
            
            folds.append(fold)
            current_start += step_size
        
        logger.info(f"Generated {len(folds)} Walk-Forward folds")
        
        # 保存到缓存
        self._save_cached_folds(folds)
        
        self.folds = folds
        return folds
    
    def run_fold(self, fold: WalkForwardFold, data_df: pd.DataFrame) -> Dict:
        """运行单个折叠的训练和评估"""
        logger.info(f"Running fold {fold.fold_id}")
        
        # 检查是否已有该折叠的结果
        fold_result_path = self.wf_output_dir / f"fold_{fold.fold_id}_results.yaml"
        if fold_result_path.exists() and not self.wf_config.get('force_retrain', False):
            logger.info(f"Loading existing results for fold {fold.fold_id}")
            with open(fold_result_path, 'r') as f:
                return yaml.safe_load(f)
        
        # 创建折叠特定的配置
        fold_config = OmegaConf.create(OmegaConf.to_container(self.config, resolve=True))
        
        # 设置时间范围过滤
        fold_config.data.time_filter = {
            'train_start': fold.train_start.isoformat(),
            'train_end': fold.train_end.isoformat(),
            'val_start': fold.val_start.isoformat(),
            'val_end': fold.val_end.isoformat(),
            'test_start': fold.test_start.isoformat(),
            'test_end': fold.test_end.isoformat()
        }
        
        # 创建折叠输出目录
        fold_output_dir = self.wf_output_dir / f"fold_{fold.fold_id}"
        fold_output_dir.mkdir(exist_ok=True)
        
        try:
            # 创建数据模块
            data_module = NILMDataModule(fold_config)
            data_module.setup()
            
            # 创建模型
            model = NILMLightningModule(fold_config, self.device_info, self.device_names)
            
            # 设置日志记录
            logger_tb = setup_logging(fold_config, fold_output_dir)
            
            # 创建训练器
            trainer = create_trainer(fold_config, logger_tb)
            
            # 训练模型
            trainer.fit(model, datamodule=data_module)
            
            # 测试模型
            test_results = trainer.test(model, datamodule=data_module)
            
            # 收集结果
            fold_result = {
                'fold_id': fold.fold_id,
                'fold_info': fold.to_dict(),
                'test_results': test_results[0] if test_results else {},
                'best_val_score': float(model.best_val_score),
                'best_thresholds': {k: float(v) for k, v in model.best_thresholds.items()},
                'best_model_path': str(trainer.checkpoint_callback.best_model_path) if trainer.checkpoint_callback else None,
                'training_time': datetime.now().isoformat()
            }
            
            # 保存折叠结果
            with open(fold_result_path, 'w') as f:
                yaml.dump(fold_result, f, default_flow_style=False)
            
            logger.info(f"Fold {fold.fold_id} completed. Val score: {model.best_val_score:.4f}")
            
            return fold_result
            
        except Exception as e:
            logger.error(f"Error in fold {fold.fold_id}: {e}")
            error_result = {
                'fold_id': fold.fold_id,
                'fold_info': fold.to_dict(),
                'error': str(e),
                'training_time': datetime.now().isoformat()
            }
            
            # 保存错误结果
            with open(fold_result_path, 'w') as f:
                yaml.dump(error_result, f, default_flow_style=False)
            
            return error_result
    
    def run_validation(self) -> Dict:
        """运行完整的Walk-Forward交叉验证"""
        logger.info("Starting Walk-Forward Cross-Validation")
        
        # 加载数据
        data_module = NILMDataModule(self.config)
        data_module.setup()
        
        # 获取原始数据用于时间分割
        raw_data = data_module._load_raw_data()
        
        # 生成折叠
        folds = self.generate_folds(raw_data)
        
        if not folds:
            raise ValueError("No valid folds generated")
        
        # 运行每个折叠
        fold_results = []
        for fold in folds:
            result = self.run_fold(fold, raw_data)
            fold_results.append(result)
            self.fold_results.append(result)
        
        # 计算总体统计
        summary = self._compute_summary_statistics(fold_results)
        
        # 保存总体结果
        final_results = {
            'summary': summary,
            'fold_results': fold_results,
            'config': OmegaConf.to_container(self.config, resolve=True),
            'validation_time': datetime.now().isoformat()
        }
        
        results_path = self.wf_output_dir / "walk_forward_results.yaml"
        with open(results_path, 'w') as f:
            yaml.dump(final_results, f, default_flow_style=False)
        
        logger.info(f"Walk-Forward validation completed. Results saved to {results_path}")
        if 'avg_val_score' in summary:
            logger.info(f"Average validation score: {summary['avg_val_score']:.4f} ± {summary['std_val_score']:.4f}")
        else:
            logger.warning("No valid fold results available for summary statistics")
        
        return final_results
    
    def _compute_summary_statistics(self, fold_results: List[Dict]) -> Dict:
        """计算汇总统计信息"""
        valid_results = [r for r in fold_results if 'error' not in r]
        
        if not valid_results:
            return {'error': 'No valid fold results'}
        
        # 提取验证分数
        val_scores = [r['best_val_score'] for r in valid_results]
        
        # 提取测试指标
        test_metrics = {}
        if valid_results[0].get('test_results'):
            for metric_name in valid_results[0]['test_results'].keys():
                metric_values = [r['test_results'][metric_name] for r in valid_results if metric_name in r['test_results']]
                if metric_values:
                    test_metrics[f'avg_{metric_name}'] = np.mean(metric_values)
                    test_metrics[f'std_{metric_name}'] = np.std(metric_values)
                    test_metrics[f'min_{metric_name}'] = np.min(metric_values)
                    test_metrics[f'max_{metric_name}'] = np.max(metric_values)
        
        summary = {
            'num_folds': len(fold_results),
            'num_valid_folds': len(valid_results),
            'num_failed_folds': len(fold_results) - len(valid_results),
            'avg_val_score': np.mean(val_scores),
            'std_val_score': np.std(val_scores),
            'min_val_score': np.min(val_scores),
            'max_val_score': np.max(val_scores),
            **test_metrics
        }
        
        return summary


def run_walk_forward_validation(config: DictConfig, output_dir: Path) -> Dict:
    """运行Walk-Forward交叉验证的主函数"""
    validator = WalkForwardValidator(config, output_dir)
    return validator.run_validation()