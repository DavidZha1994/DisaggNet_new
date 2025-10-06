"""Optuna超参数优化"""

import os
import sys
import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, DeviceStatsMonitor
from optuna.integration import PyTorchLightningPruningCallback
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.train import NILMLightningModule, create_trainer, setup_logging, load_device_info
from src.data.datamodule import NILMDataModule


class OptunaPruningCallback(PyTorchLightningPruningCallback):
    """自定义Optuna剪枝回调"""
    
    def __init__(self, trial: optuna.trial.Trial, monitor: str = 'val_score'):
        super().__init__(trial, monitor)
        self.trial = trial
        self.monitor = monitor
    
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """验证结束时检查是否需要剪枝"""
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is not None:
            self.trial.report(current_score, step=trainer.current_epoch)
            if self.trial.should_prune():
                message = f"Trial was pruned at epoch {trainer.current_epoch}."
                raise optuna.TrialPruned(message)


class OptunaObjective:
    """Optuna目标函数"""
    
    def __init__(self, config: DictConfig, base_output_dir: Path):
        self.config = config
        self.base_output_dir = base_output_dir
        self.device_info, self.device_names = load_device_info(config)
        
        # 注意：数据模块将在每个试验中重新创建以使用不同的批次大小
        self.base_datamodule_config = config
        
        # 试验计数器
        self.trial_count = 0
    
    def suggest_hyperparameters(self, trial: optuna.trial.Trial) -> DictConfig:
        """建议超参数"""
        # 复制基础配置
        trial_config = OmegaConf.create(OmegaConf.to_container(self.config, resolve=True))
        
        # 模型结构参数
        trial_config.data.window_size = trial.suggest_categorical(
            'window_size', self.config.hpo.search_space.structure.window_sizes
        )
        
        trial_config.model.time_encoder.d_model = trial.suggest_categorical(
            'd_model', self.config.hpo.search_space.structure.d_models
        )
        
        trial_config.model.time_encoder.num_layers = trial.suggest_categorical(
            'time_layers', self.config.hpo.search_space.structure.time_layers
        )
        
        trial_config.model.time_encoder.n_heads = trial.suggest_categorical(
            'n_heads', self.config.hpo.search_space.structure.n_heads
        )
        
        if trial_config.model.freq_encoder.enable:
            trial_config.model.freq_encoder.small_transformer_layers = trial.suggest_categorical(
                'freq_layers', self.config.hpo.search_space.structure.freq_layers
            )
            
            trial_config.model.freq_encoder.proj_dim = trial.suggest_categorical(
                'freq_proj_dim', self.config.hpo.search_space.structure.freq_proj_dims
            )
        
        # STFT参数
        if trial_config.data.features.freq_domain.enable:
            trial_config.data.features.freq_domain.stft.n_fft = trial.suggest_categorical(
                'n_fft', self.config.hpo.search_space.stft.n_ffts
            )
            
            trial_config.data.features.freq_domain.stft.hop_length = trial.suggest_categorical(
                'hop_length', self.config.hpo.search_space.stft.hop_lengths
            )
        
        # 损失权重
        trial_config.loss.weights.classification = trial.suggest_float(
            'alpha', 
            self.config.hpo.search_space.loss_weights.alpha_range[0],
            self.config.hpo.search_space.loss_weights.alpha_range[1]
        )
        
        trial_config.loss.weights.consistency_schedule.final = trial.suggest_float(
            'beta_final',
            self.config.hpo.search_space.loss_weights.beta_final_range[0],
            self.config.hpo.search_space.loss_weights.beta_final_range[1]
        )
        
        trial_config.loss.weights.consistency_schedule.warmup_epochs = trial.suggest_categorical(
            'warmup_epochs', self.config.hpo.search_space.loss_weights.warmup_epochs
        )
        
        # 正则化参数
        trial_config.model.time_encoder.dropout = trial.suggest_float(
            'dropout',
            self.config.hpo.search_space.regularization.dropout_range[0],
            self.config.hpo.search_space.regularization.dropout_range[1]
        )
        
        if trial_config.loss.priors.enable:
            trial_config.loss.priors.weights.ramp = trial.suggest_float(
                'ramp_weight',
                self.config.hpo.search_space.regularization.ramp_range[0],
                self.config.hpo.search_space.regularization.ramp_range[1]
            )
            
            trial_config.loss.priors.weights.range = trial.suggest_float(
                'range_weight',
                self.config.hpo.search_space.regularization.range_range[0],
                self.config.hpo.search_space.regularization.range_range[1]
            )
        
        # 学习率（对数均匀分布）
        trial_config.training.optimizer.lr = trial.suggest_float(
            'learning_rate',
            self.config.hpo.search_space.learning_rate.min_lr,
            self.config.hpo.search_space.learning_rate.max_lr,
            log=True
        )
        
        # 训练精度和相关参数（新增）
        if hasattr(self.config.hpo.search_space, 'training'):
            trial_config.training.precision = trial.suggest_categorical(
                'precision',
                self.config.hpo.search_space.training.precision_options
            )
            
            trial_config.training.batch_size = trial.suggest_categorical(
                'batch_size',
                self.config.hpo.search_space.training.batch_sizes
            )
            
            trial_config.training.gradient_clip_val = trial.suggest_categorical(
                'gradient_clip_val',
                self.config.hpo.search_space.training.gradient_clip_vals
            )

        # 推理/后处理相关参数（新增，用于阈值与策略优化）
        # 若配置中未提供搜索空间，使用合理的默认范围
        trial_config.inference = trial_config.inference if 'inference' in trial_config else {}
        hys_min = getattr(self.config.hpo.search_space, 'inference', {}).get('hysteresis_ratio_min', 0.05)
        hys_max = getattr(self.config.hpo.search_space, 'inference', {}).get('hysteresis_ratio_max', 0.30)
        trial_config.inference.hysteresis_ratio = trial.suggest_float('hysteresis_ratio', hys_min, hys_max)

        thr_min = getattr(self.config.hpo.search_space, 'inference', {}).get('default_power_threshold_min', 5.0)
        thr_max = getattr(self.config.hpo.search_space, 'inference', {}).get('default_power_threshold_max', 100.0)
        trial_config.inference.default_power_threshold = trial.suggest_float('default_power_threshold', thr_min, thr_max)

        method_options = getattr(self.config.hpo.search_space, 'inference', {}).get('power_threshold_methods', ['fixed', 'otsu', 'percentile'])
        trial_config.inference.power_threshold_method = trial.suggest_categorical('power_threshold_method', method_options)

        pctl_options = getattr(self.config.hpo.search_space, 'inference', {}).get('power_threshold_percentiles', [80, 85, 90, 95])
        trial_config.inference.power_threshold_percentile = trial.suggest_categorical('power_threshold_percentile', pctl_options)

        vote_window_options = getattr(self.config.hpo.search_space, 'inference', {}).get('voting_windows', [1, 3, 5, 7])
        trial_config.inference.voting = trial_config.inference.voting if 'voting' in trial_config.inference else {}
        trial_config.inference.voting.window = trial.suggest_categorical('voting_window', vote_window_options)

        vote_thr_min = getattr(self.config.hpo.search_space, 'inference', {}).get('voting_threshold_min', 0.5)
        vote_thr_max = getattr(self.config.hpo.search_space, 'inference', {}).get('voting_threshold_max', 0.9)
        trial_config.inference.voting.threshold = trial.suggest_float('voting_threshold', vote_thr_min, vote_thr_max)

        cls_margin_min = getattr(self.config.hpo.search_space, 'inference', {}).get('cls_threshold_margin_min', 0.01)
        cls_margin_max = getattr(self.config.hpo.search_space, 'inference', {}).get('cls_threshold_margin_max', 0.20)
        trial_config.inference.cls_threshold_margin = trial.suggest_float('cls_threshold_margin', cls_margin_min, cls_margin_max)
        
        return trial_config
    
    def __call__(self, trial: optuna.trial.Trial) -> float:
        """目标函数"""
        self.trial_count += 1
        
        try:
            # 建议超参数
            trial_config = self.suggest_hyperparameters(trial)
            
            # 创建试验特定的输出目录
            trial_output_dir = self.base_output_dir / f"trial_{trial.number}"
            trial_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 更新配置中的路径
            trial_config.paths.output_dir = str(trial_output_dir)
            trial_config.logging.save_dir = str(trial_output_dir)
            
            # 设置随机种子
            pl.seed_everything(trial_config.experiment.seed + trial.number, workers=True)
            
            # 创建数据模块（使用建议的批次大小）
            datamodule = NILMDataModule(trial_config)
            datamodule.setup()
            
            # 创建模型
            model = NILMLightningModule(trial_config, self.device_info, self.device_names)
            
            # 创建日志记录器
            experiment_name = f"trial_{trial.number}"
            logger = setup_logging(trial_config, experiment_name)
            
            # 创建回调函数（使用与正常训练相同的配置）
            callbacks = []
            
            # Optuna剪枝回调
            pruning_callback = OptunaPruningCallback(trial, monitor='val_score')
            callbacks.append(pruning_callback)
            
            # 早停回调（使用配置文件中的设置）
            if trial_config.training.early_stopping.enable:
                early_stopping = EarlyStopping(
                    monitor=trial_config.training.early_stopping.monitor,
                    patience=trial_config.training.early_stopping.patience,
                    mode=trial_config.training.early_stopping.mode,
                    verbose=False  # HPO时保持安静
                )
                callbacks.append(early_stopping)
            
            # 学习率监控
            lr_monitor = LearningRateMonitor(logging_interval='epoch')
            callbacks.append(lr_monitor)
            
            # 设备状态监控
            if trial_config.training.monitor_device_stats:
                device_stats = DeviceStatsMonitor()
                callbacks.append(device_stats)
            
            # 临时覆盖训练配置以适应HPO
            original_max_epochs = trial_config.training.max_epochs
            original_min_epochs = trial_config.training.min_epochs
            trial_config.training.max_epochs = self.config.hpo.max_epochs
            trial_config.training.min_epochs = self.config.hpo.min_epochs
            
            # 创建训练器（使用完整的训练配置）
            trainer = create_trainer(trial_config, logger=logger)
            
            # 恢复原始配置
            trial_config.training.max_epochs = original_max_epochs
            trial_config.training.min_epochs = original_min_epochs
            
            # 手动添加HPO特定的回调
            trainer.callbacks.extend(callbacks)
            
            # 禁用HPO不需要的功能
            trainer.enable_checkpointing = False
            trainer.enable_progress_bar = False
            trainer.enable_model_summary = False
            
            # 训练模型
            trainer.fit(model, datamodule)
            
            # 获取最佳验证分数
            best_score = model.best_val_score
            
            # 保存试验结果
            trial_results = {
                'trial_number': trial.number,
                'best_score': best_score,
                'best_thresholds': model.best_thresholds,
                'hyperparameters': trial.params,
                'config': OmegaConf.to_container(trial_config, resolve=True)
            }
            
            results_path = trial_output_dir / "trial_results.json"
            with open(results_path, 'w') as f:
                json.dump(trial_results, f, indent=2, default=str)
            
            return best_score
            
        except optuna.TrialPruned:
            # 试验被剪枝
            raise
        except Exception as e:
            # 其他异常，返回较低的分数
            print(f"Trial {trial.number} failed with error: {e}")
            return 0.0


def create_study(config: DictConfig, study_name: str, storage_url: Optional[str] = None) -> optuna.Study:
    """创建Optuna研究"""
    
    # 采样器
    if config.hpo.sampler.name == 'tpe':
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=config.hpo.sampler.n_startup_trials,
            n_ei_candidates=config.hpo.sampler.n_ei_candidates,
            seed=config.experiment.seed
        )
    elif config.hpo.sampler.name == 'random':
        sampler = optuna.samplers.RandomSampler(seed=config.experiment.seed)
    elif config.hpo.sampler.name == 'cmaes':
        sampler = optuna.samplers.CmaEsSampler(seed=config.experiment.seed)
    else:  # tpe
        sampler = optuna.samplers.TPESampler(seed=config.experiment.seed)
    
    # 剪枝器
    if config.hpo.pruner.name == 'median':
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=config.hpo.pruner.n_startup_trials,
            n_warmup_steps=config.hpo.pruner.n_warmup_steps
        )
    elif config.hpo.pruner.name == 'hyperband':
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=config.hpo.pruner.min_resource,
            max_resource=config.hpo.pruner.max_resource
        )
    else:
        pruner = optuna.pruners.MedianPruner()
    
    # 创建研究
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction='maximize',  # 最大化验证分数
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )
    
    return study


def run_optimization(config: DictConfig, output_dir: Path) -> optuna.Study:
    """运行超参数优化"""
    
    # 创建输出目录
    hpo_output_dir = output_dir / "hpo"
    hpo_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建研究名称
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"{config.project_name}_hpo_{timestamp}"
    
    # 存储URL（可选）
    storage_url = None
    if config.hpo.storage.enable:
        storage_url = f"sqlite:///{hpo_output_dir / 'optuna_study.db'}"
    
    # 创建研究
    study = create_study(config, study_name, storage_url)
    
    # 创建目标函数
    objective = OptunaObjective(config, hpo_output_dir)
    
    # 运行优化
    print(f"开始超参数优化: {study_name}")
    print(f"试验数量: {config.hpo.n_trials}")
    print(f"并行作业: {config.hpo.n_jobs}")
    
    study.optimize(
        objective,
        n_trials=config.hpo.n_trials,
        n_jobs=config.hpo.n_jobs,
        timeout=config.hpo.timeout,
        show_progress_bar=True
    )
    
    # 保存结果
    print("\n=== 优化完成 ===")
    print(f"最佳试验: {study.best_trial.number}")
    print(f"最佳分数: {study.best_value:.4f}")
    print("最佳超参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # 保存最佳超参数
    best_params_path = hpo_output_dir / "best_hyperparameters.yaml"
    OmegaConf.save(study.best_params, best_params_path)
    
    # 保存完整的研究结果
    study_results = {
        'study_name': study_name,
        'best_trial_number': study.best_trial.number,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials),
        'trials': []
    }
    
    for trial in study.trials:
        trial_info = {
            'number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'state': trial.state.name,
            'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
            'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None
        }
        study_results['trials'].append(trial_info)
    
    study_results_path = hpo_output_dir / "study_results.json"
    with open(study_results_path, 'w') as f:
        json.dump(study_results, f, indent=2, default=str)
    
    # 生成优化历史图
    try:
        import matplotlib.pyplot as plt
        
        # 优化历史
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 试验值历史
        trial_numbers = [t.number for t in study.trials if t.value is not None]
        trial_values = [t.value for t in study.trials if t.value is not None]
        
        ax1.plot(trial_numbers, trial_values, 'o-', alpha=0.7)
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('Objective Value')
        ax1.set_title('Optimization History')
        ax1.grid(True, alpha=0.3)
        
        # 参数重要性（如果有足够的试验）
        if len(study.trials) >= 10:
            try:
                importance = optuna.importance.get_param_importances(study)
                params = list(importance.keys())
                values = list(importance.values())
                
                ax2.barh(params, values)
                ax2.set_xlabel('Importance')
                ax2.set_title('Parameter Importance')
            except:
                ax2.text(0.5, 0.5, 'Parameter importance\nnot available', 
                        ha='center', va='center', transform=ax2.transAxes)
        else:
            ax2.text(0.5, 0.5, 'Not enough trials for\nparameter importance', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        plt.savefig(hpo_output_dir / 'optimization_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except ImportError:
        print("Matplotlib not available, skipping visualization.")
    
    return study


def apply_best_hyperparameters(config: DictConfig, best_params: Dict[str, Any]) -> DictConfig:
    """将最佳超参数应用到配置中"""
    optimized_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    
    # 应用最佳超参数
    param_mapping = {
        'window_size': 'data.window_size',
        'd_model': 'model.time_encoder.d_model',
        'time_layers': 'model.time_encoder.num_layers',
        'n_heads': 'model.time_encoder.n_heads',
        'freq_layers': 'model.freq_encoder.small_transformer_layers',
        'freq_proj_dim': 'model.freq_encoder.proj_dim',
        'n_fft': 'data.features.freq_domain.stft.n_fft',
        'hop_length': 'data.features.freq_domain.stft.hop_length',
        'alpha': 'loss.weights.classification',
        'beta_final': 'loss.weights.consistency_schedule.final',
        'warmup_epochs': 'loss.weights.consistency_schedule.warmup_epochs',
        'dropout': 'model.time_encoder.dropout',
        'ramp_weight': 'loss.priors.weights.ramp',
        'range_weight': 'loss.priors.weights.range',
        'learning_rate': 'training.optimizer.lr',
        # 推理/后处理参数
        'hysteresis_ratio': 'inference.hysteresis_ratio',
        'default_power_threshold': 'inference.default_power_threshold',
        'power_threshold_method': 'inference.power_threshold_method',
        'power_threshold_percentile': 'inference.power_threshold_percentile',
        'voting_window': 'inference.voting.window',
        'voting_threshold': 'inference.voting.threshold',
        'cls_threshold_margin': 'inference.cls_threshold_margin'
    }
    
    # 允许修改配置结构
    OmegaConf.set_struct(optimized_config, False)
    
    for param_name, param_value in best_params.items():
        if param_name in param_mapping:
            config_path = param_mapping[param_name]
            # 使用OmegaConf.update方式设置嵌套配置
            keys = config_path.split('.')
            current = optimized_config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = param_value
    
    # 恢复结构保护
    OmegaConf.set_struct(optimized_config, True)
    
    return optimized_config


@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def main(config: DictConfig) -> None:
    """主HPO函数"""
    
    # 创建输出目录
    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行超参数优化
    study = run_optimization(config, output_dir)
    
    # 应用最佳超参数并保存优化后的配置
    optimized_config = apply_best_hyperparameters(config, study.best_params)
    optimized_config_path = output_dir / "hpo" / "optimized_config.yaml"
    OmegaConf.save(optimized_config, optimized_config_path)
    
    print(f"\n优化后的配置保存至: {optimized_config_path}")
    print("可以使用此配置进行最终训练。")


if __name__ == "__main__":
    main()