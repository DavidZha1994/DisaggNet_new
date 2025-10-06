"""训练脚本"""

import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor,
    DeviceStatsMonitor, RichProgressBar, GradientAccumulationScheduler
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.strategies import DDPStrategy
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

# 启用TF32优化以提升性能
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

from src.data.datamodule import NILMDataModule  # 使用新的工业级数据模块
from src.models.fusion_transformer import FusionTransformer, TemperatureScaling
from src.losses.losses import MultiTaskLoss
from src.utils.metrics import NILMMetrics, ConsistencyMetrics, DelayMetrics
from src.models.priors import PriorKnowledgeIntegrator
from src.utils.conformal_prediction import MultiTaskConformalPredictor
from src.utils.conformal_evaluation import ConformalEvaluator


class NILMLightningModule(pl.LightningModule):
    """NILM PyTorch Lightning模块"""
    
    def __init__(self, config: DictConfig, device_info: Dict, device_names: List[str]):
        super().__init__()
        
        self.config = config
        self.device_info = device_info
        self.device_names = device_names
        self.n_devices = len(device_names)
        
        # 保存超参数
        self.save_hyperparameters({
            'config': OmegaConf.to_container(config, resolve=True),
            'device_info': device_info,
            'device_names': device_names
        })
        
        # 初始化模型
        self.model = FusionTransformer(config.model, self.n_devices)
        
        # 初始化损失函数
        self.loss_fn = MultiTaskLoss(config.loss, device_info, self.n_devices)
        
        # 初始化评估指标
        self.metrics = NILMMetrics(device_names, config.evaluation.threshold_method)
        
        # 温度缩放（用于校准）
        if config.model.calibration.enable:
            self.temperature_scaling = TemperatureScaling(self.n_devices)
        else:
            self.temperature_scaling = None
        
        # Conformal Prediction标定器
        if config.get('conformal_prediction', {}).get('enable', False):
            self.conformal_predictor = MultiTaskConformalPredictor(
                alpha=config.conformal_prediction.alpha,
                device_names=device_names,
                regression_method=config.conformal_prediction.regression_method,
                classification_method=config.conformal_prediction.classification_method
            )
            # 创建评估器
            self.conformal_evaluator = ConformalEvaluator(
                device_names=device_names,
                alpha=config.conformal_prediction.alpha
            )
        else:
            self.conformal_predictor = None
            self.conformal_evaluator = None
        
        # 验证集最佳指标
        self.best_val_score = 0.0
        self.best_thresholds = {}
        
        # 训练状态
        self.automatic_optimization = True
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        time_features = batch['time_features']  # (batch_size, window_size, n_time_features)
        freq_features = batch.get('freq_features', None)  # (batch_size, n_time_frames, n_freq_bins)
        time_positional = batch.get('time_positional', None)  # (batch_size, window_size, time_dim)
        aux_features = batch.get('aux_features', None)  # (batch_size, n_aux_features)
        
        return self.model(time_features, freq_features, time_positional, aux_features=aux_features)
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor], 
                     predictions: Tuple[torch.Tensor, torch.Tensor],
                     stage: str = 'train') -> Dict[str, torch.Tensor]:
        """计算损失"""
        pred_power, pred_states = predictions
        target_power = batch['target_power']  # (batch_size, n_devices)
        target_states = batch['target_states']  # (batch_size, n_devices)
        
        # 历史功率（用于先验损失）
        historical_power = batch.get('historical_power', None)
        window_length = batch.get('window_length', self.config.data.window_size)
        
        # 计算损失
        losses = self.loss_fn(
            predictions=(pred_power, pred_states),
            targets=(target_power, target_states),
            historical_power=historical_power,
            window_length=window_length
        )
        
        # 记录损失
        for loss_name, loss_value in losses.items():
            self.log(f'{stage}_{loss_name}', loss_value, 
                    on_step=(stage == 'train'), on_epoch=True, 
                    prog_bar=(loss_name == 'total'), sync_dist=True)
        
        return losses
    
    def _compute_metrics(self, batch: Dict[str, torch.Tensor],
                        predictions: Tuple[torch.Tensor, torch.Tensor],
                        stage: str = 'val') -> Dict[str, float]:
        """计算评估指标"""
        pred_power, pred_states = predictions
        target_power = batch['target_power']
        target_states = batch['target_states']
        
        # 应用温度缩放（仅在验证/测试阶段进行校准）
        logits = pred_states
        if self.temperature_scaling is not None and stage in ['val', 'test']:
            logits = self.temperature_scaling(logits)
        # 应用sigmoid到分类输出
        pred_proba = torch.sigmoid(logits)
        
        # 计算所有指标
        metrics = self.metrics.compute_all_metrics(
            y_pred_power=pred_power,
            y_pred_proba=pred_proba,
            y_true_power=target_power,
            y_true_states=target_states,
            optimize_thresholds=(stage == 'val')
        )
        
        # 记录主要指标（统一为层级式标签，按 epoch 写入）
        main_metrics = ['mae', 'nde', 'sae', 'teca', 'f1', 'mcc', 'pr_auc', 'roc_auc', 'score']
        for metric_name in main_metrics:
            if metric_name in metrics:
                self.log(f'{stage}/{metric_name}', metrics[metric_name],
                        on_epoch=True, on_step=False,
                        prog_bar=(metric_name in ['score', 'f1']), sync_dist=True)
        
        # 一致性指标
        consistency_metrics = {
            'power_balance_error': ConsistencyMetrics.power_balance_error(pred_power, target_power),
            'state_power_consistency': ConsistencyMetrics.state_power_consistency(pred_power, pred_proba),
            'temporal_consistency': ConsistencyMetrics.temporal_consistency(pred_power)
        }
        
        for metric_name, metric_value in consistency_metrics.items():
            # 统一使用层级式标签，按 epoch 记录，减少乱序
            self.log(f'{stage}/{metric_name}', metric_value, on_epoch=True, sync_dist=True)
        
        return metrics
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """训练步骤"""
        # 数据验证和NaN检测
        self._validate_batch_data(batch, 'train', batch_idx)
        
        predictions = self(batch)
        
        # 检测预测结果中的NaN/Inf
        self._validate_predictions(predictions, 'train', batch_idx)
        
        losses = self._compute_loss(batch, predictions, 'train')
        
        # 检测损失中的NaN/Inf
        self._validate_losses(losses, 'train', batch_idx)
        
        # 记录学习率（按 epoch），避免与 step 级曲线混写造成乱序
        self.log('train/lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=False, on_epoch=True)
        
        # 每隔一定步数计算训练指标
        if batch_idx % self.config.training.log_every_n_steps == 0:
            with torch.no_grad():
                self._compute_metrics(batch, predictions, 'train')
                
                # 记录梯度范数 - 增强梯度监控
                if self.config.debug.track_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=float('inf'))
                    # 统一使用 epoch 级记录
                    self.log('train/grad_norm', grad_norm, on_step=False, on_epoch=True)
                    
                    # 检查梯度异常
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        self.logger.experiment.add_scalar('train/grad_anomaly', 1.0, self.global_step)
                        print(f"Warning: Invalid gradient norm at step {self.global_step}: {grad_norm}")
        
        return losses['total']
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """验证步骤"""
        # 数据验证和NaN检测
        self._validate_batch_data(batch, 'val', batch_idx)
        
        predictions = self(batch)
        self._validate_predictions(predictions, 'val', batch_idx)
        
        losses = self._compute_loss(batch, predictions, 'val')
        self._validate_losses(losses, 'val', batch_idx)
        metrics = self._compute_metrics(batch, predictions, 'val')
        
        # 显式按 epoch 记录验证损失与分数，保证每个 epoch 有序写入
        self.log('val/loss', losses['total'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/score', metrics['score'], on_step=False, on_epoch=True, prog_bar=True)

        return {
            'val_loss': losses['total'],
            'val_score': metrics['score'],
            'predictions': predictions,
            'targets': (batch['target_power'], batch['target_states'])
        }
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """测试步骤"""
        # 数据验证和NaN检测
        self._validate_batch_data(batch, 'test', batch_idx)
        
        predictions = self(batch)
        self._validate_predictions(predictions, 'test', batch_idx)
        
        losses = self._compute_loss(batch, predictions, 'test')
        self._validate_losses(losses, 'test', batch_idx)
        metrics = self._compute_metrics(batch, predictions, 'test')
        
        # 显式按 epoch 记录测试损失与分数（若启用测试），保证写入一致性
        self.log('test/loss', losses['total'], on_step=False, on_epoch=True)
        self.log('test/score', metrics['score'], on_step=False, on_epoch=True)

        return {
            'test_loss': losses['total'],
            'test_score': metrics['score'],
            'predictions': predictions,
            'targets': (batch['target_power'], batch['target_states'])
        }
    
    def on_validation_epoch_end(self) -> None:
        """验证轮次结束"""
        # 获取当前验证分数（兼容旧键名）
        current_score = self.trainer.callback_metrics.get('val/score', None)
        if current_score is None:
            current_score = self.trainer.callback_metrics.get('val_score', 0.0)
        # 将 Tensor 转为 float，避免后续配置或日志处理中出现 OmegaConf 类型不支持
        try:
            if hasattr(current_score, 'item'):
                current_score = float(current_score.item())
            else:
                current_score = float(current_score)
        except Exception:
            current_score = 0.0
        
        # 更新最佳分数和阈值
        if current_score > float(self.best_val_score):
            self.best_val_score = float(current_score)
            self.best_thresholds = self.metrics.thresholds.copy()
            # 记录最佳验证分数（层级命名）
            self.log('val/best_score', self.best_val_score, on_epoch=True)
            
            # 记录最佳阈值
            for device_name, threshold in self.best_thresholds.items():
                self.log(f'best_threshold/{device_name}', threshold, on_epoch=True)

            # 持久化保存最佳阈值
            try:
                thresholds_path = Path(self.config.paths.output_dir) / 'best_thresholds.json'
                self.metrics.save_thresholds(str(thresholds_path))
                print(f"最佳阈值已保存至: {thresholds_path}")
            except Exception as e:
                print(f"保存最佳阈值失败: {e}")
        
        # Conformal Prediction标定
        if self.conformal_predictor is not None and self.trainer.state.stage == 'validate':
            # 收集验证集的预测和真实值
            val_outputs = self.trainer.predict_loop.predictions
            if val_outputs:
                all_predictions = []
                all_targets = []
                
                for output in val_outputs:
                    if 'predictions' in output and 'targets' in output:
                        pred_power, pred_states = output['predictions']
                        target_power, target_states = output['targets']
                        
                        all_predictions.append((pred_power, pred_states))
                        all_targets.append((target_power, target_states))
                
                if all_predictions:
                    # 合并所有批次的数据
                    pred_powers = torch.cat([p[0] for p in all_predictions], dim=0)
                    pred_states = torch.cat([p[1] for p in all_predictions], dim=0)
                    target_powers = torch.cat([t[0] for t in all_targets], dim=0)
                    target_states = torch.cat([t[1] for t in all_targets], dim=0)
                    
                    # 执行conformal标定
                    self.conformal_predictor.calibrate(
                        predictions=(pred_powers, pred_states),
                        targets=(target_powers, target_states)
                    )
                    
                    # 评估conformal prediction性能
                    if self.conformal_evaluator is not None:
                        eval_results = self.conformal_evaluator.evaluate_regression(
                            predictions=pred_powers,
                            targets=target_powers,
                            intervals=self.conformal_predictor.predict_intervals(pred_powers)
                        )
                        
                        # 记录回归评估指标
                        for device_name, metrics in eval_results.items():
                            for metric_name, value in metrics.items():
                                # 归入验证命名空间，并使用层级式标签
                                self.log(f'val/conformal/regression/{device_name}/{metric_name}', value, on_epoch=True)
                        
                        # 评估分类校准
                        classification_results = self.conformal_evaluator.evaluate_classification(
                            predictions=pred_states,
                            targets=target_states,
                            calibrated_probs=self.conformal_predictor.predict_calibrated_probs(pred_states)
                        )
                        
                        # 记录分类评估指标
                        for device_name, metrics in classification_results.items():
                            for metric_name, value in metrics.items():
                                self.log(f'val/conformal/classification/{device_name}/{metric_name}', value, on_epoch=True)
                    
                    # 记录标定信息（归入验证命名空间）
                    self.log('val/conformal/calibrated', 1.0, on_epoch=True)
                    print(f"Conformal prediction calibrated at epoch {self.current_epoch}")
                    
                    # 生成评估报告（每10个epoch一次）
                    if self.conformal_evaluator is not None and self.current_epoch % 10 == 0:
                        report_path = f"conformal_report_epoch_{self.current_epoch}.html"
                        self.conformal_evaluator.generate_report(
                            predictions=(pred_powers, pred_states),
                            targets=(target_powers, target_states),
                            intervals=self.conformal_predictor.predict_intervals(pred_powers),
                            calibrated_probs=self.conformal_predictor.predict_calibrated_probs(pred_states),
                            output_path=report_path
                        )
                        print(f"Conformal evaluation report saved to {report_path}")
        
        # 记录最佳验证分数（归入验证命名空间）
        self.log('val/best_score', self.best_val_score, on_epoch=True)
        
        # 记录模型参数统计
        for name, param in self.named_parameters():
            if param.grad is not None:
                self.logger.experiment.add_histogram(f'params/{name}', param, self.current_epoch)
                self.logger.experiment.add_histogram(f'grads/{name}', param.grad, self.current_epoch)
        
        # 更新损失函数的epoch
        self.loss_fn.update_epoch(self.current_epoch)
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """配置优化器和学习率调度器"""
        # 优化器 - 使用更保守的学习率
        if self.config.training.optimizer.name == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=min(self.config.training.optimizer.lr, 1e-4),  # 限制最大学习率
                weight_decay=self.config.training.optimizer.weight_decay,
                betas=self.config.training.optimizer.betas,
                eps=1e-8  # 数值稳定性
            )
        elif self.config.training.optimizer.name == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=min(self.config.training.optimizer.lr, 1e-4),  # 限制最大学习率
                weight_decay=self.config.training.optimizer.weight_decay,
                eps=1e-8  # 数值稳定性
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer.name}")
        
        # 学习率调度器
        scheduler_config = self.config.training.scheduler
        
        if scheduler_config.name == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config.T_max,
                eta_min=scheduler_config.eta_min
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        elif scheduler_config.name == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=max(scheduler_config.factor, 0.3),  # 更大的衰减因子
                patience=min(scheduler_config.patience, 3),  # 更短的耐心
                min_lr=1e-7  # 最小学习率
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/score',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        else:
            return optimizer
    
    def _validate_batch_data(self, batch: Dict[str, torch.Tensor], stage: str, batch_idx: int) -> None:
        """验证批次数据的有效性"""
        for key, tensor in batch.items():
            # 只对tensor类型的数据进行验证
            if isinstance(tensor, torch.Tensor) and not torch.isfinite(tensor).all():
                nan_count = torch.isnan(tensor).sum().item()
                inf_count = torch.isinf(tensor).sum().item()
                self.logger.experiment.add_scalar(f'{stage}/data_nan_count/{key}', nan_count, self.global_step)
                self.logger.experiment.add_scalar(f'{stage}/data_inf_count/{key}', inf_count, self.global_step)
                
                if self.config.debug.strict_validation:
                    raise ValueError(f"发现无效数据在 {stage} batch {batch_idx}, key '{key}': "
                                   f"NaN: {nan_count}, Inf: {inf_count}")
                else:
                    print(f"警告: {stage} batch {batch_idx} 中 '{key}' 包含无效值: NaN: {nan_count}, Inf: {inf_count}")
    
    def _validate_predictions(self, predictions: Tuple[torch.Tensor, torch.Tensor], 
                            stage: str, batch_idx: int) -> None:
        """验证预测结果的有效性"""
        pred_power, pred_states = predictions
        
        # 检查功率预测
        if not torch.isfinite(pred_power).all():
            nan_count = torch.isnan(pred_power).sum().item()
            inf_count = torch.isinf(pred_power).sum().item()
            self.logger.experiment.add_scalar(f'{stage}/pred_power_nan_count', nan_count, self.global_step)
            self.logger.experiment.add_scalar(f'{stage}/pred_power_inf_count', inf_count, self.global_step)
            
            if self.config.debug.strict_validation:
                raise ValueError(f"功率预测包含无效值在 {stage} batch {batch_idx}: NaN: {nan_count}, Inf: {inf_count}")
        
        # 检查状态预测
        if not torch.isfinite(pred_states).all():
            nan_count = torch.isnan(pred_states).sum().item()
            inf_count = torch.isinf(pred_states).sum().item()
            self.logger.experiment.add_scalar(f'{stage}/pred_states_nan_count', nan_count, self.global_step)
            self.logger.experiment.add_scalar(f'{stage}/pred_states_inf_count', inf_count, self.global_step)
            
            if self.config.debug.strict_validation:
                raise ValueError(f"状态预测包含无效值在 {stage} batch {batch_idx}: NaN: {nan_count}, Inf: {inf_count}")
    
    def _validate_losses(self, losses: Dict[str, torch.Tensor], stage: str, batch_idx: int) -> None:
        """验证损失值的有效性"""
        for loss_name, loss_value in losses.items():
            # 确保loss_value是tensor
            if not isinstance(loss_value, torch.Tensor):
                print(f"警告: {stage} batch {batch_idx} 中损失 '{loss_name}' 不是tensor类型: {type(loss_value)}")
                continue
                
            # 确保是标量tensor
            if loss_value.dim() > 0:
                loss_value = loss_value.mean()
                
            if not torch.isfinite(loss_value):
                self.logger.experiment.add_scalar(f'{stage}/loss_invalid/{loss_name}', 1.0, self.global_step)
                
                if self.config.debug.strict_validation:
                    raise ValueError(f"损失 '{loss_name}' 包含无效值在 {stage} batch {batch_idx}: {loss_value}")
                else:
                    print(f"警告: {stage} batch {batch_idx} 中损失 '{loss_name}' 无效: {loss_value}")
    
    def configure_callbacks(self) -> List[pl.Callback]:
        """配置回调函数"""
        callbacks = []
        
        # 模型检查点
        # 兼容新的层级式指标命名：将旧的 'val_score' 映射为 'val/score'
        ckpt_monitor = getattr(self.config.training.checkpoint, 'monitor', 'val/score')
        if ckpt_monitor == 'val_score':
            ckpt_monitor = 'val/score'
        # 使用安全的文件名（不依赖指标占位，避免层级名导致格式错误）
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.training.checkpoint.dirpath,
            filename='epoch{epoch:02d}',
            monitor=ckpt_monitor,
            mode=self.config.training.checkpoint.mode,
            save_top_k=self.config.training.checkpoint.save_top_k,
            save_last=True,
            verbose=True
        )
        callbacks.append(checkpoint_callback)
        
        # 早停
        if self.config.training.early_stopping.enable:
            es_monitor = getattr(self.config.training.early_stopping, 'monitor', 'val/score')
            if es_monitor == 'val_score':
                es_monitor = 'val/score'
            early_stopping = EarlyStopping(
                monitor=es_monitor,
                patience=self.config.training.early_stopping.patience,
                mode=self.config.training.early_stopping.mode,
                verbose=True
            )
            callbacks.append(early_stopping)
        
        # 学习率监控
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor)
        
        # 设备状态监控
        if self.config.training.monitor_device_stats:
            device_stats = DeviceStatsMonitor()
            callbacks.append(device_stats)
        
        # 进度条 - 使用默认进度条避免冲突
        # progress_bar = RichProgressBar()
        # callbacks.append(progress_bar)
        
        return callbacks


def create_trainer(config: DictConfig, logger: Optional[pl.loggers.Logger] = None) -> pl.Trainer:
    """创建训练器"""
    
    # 设备配置（自动检测：macOS->MPS，Linux/Windows->CUDA，否则CPU）
    if getattr(config.training, 'accelerator', 'auto') == 'auto':
        is_mac = (sys.platform == 'darwin')
        mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        if is_mac and mps_available:
            accelerator = 'mps'
            devices = 1
        elif torch.cuda.is_available():
            accelerator = 'gpu'
            # 使用全部可用 GPU 以最大化吞吐（如 5090 等现代卡）
            devices = getattr(config.training, 'devices', 'auto')
            if devices in (None, 0):
                devices = 'auto'
        else:
            accelerator = 'cpu'
            devices = 1
    else:
        accelerator = config.training.accelerator
        devices = getattr(config.training, 'devices', 1)
    
    # 策略配置
    strategy = 'auto'
    # 当使用多 GPU 时启用 DDP 以最佳性能
    try:
        multi_gpu = (accelerator == 'gpu') and (
            (isinstance(devices, int) and devices > 1) or devices == 'auto'
        ) and torch.cuda.device_count() > 1
    except Exception:
        multi_gpu = False
    if multi_gpu:
        strategy = DDPStrategy(find_unused_parameters=False)
    
    # 精度配置 - 自动选择（GPU优先BF16，其次FP16；MPS使用FP16混合）
    if hasattr(config.training, 'precision'):
        precision = config.training.precision
    else:
        if accelerator == 'gpu':
            precision = 'bf16-mixed' if torch.cuda.is_bf16_supported() else '16-mixed'
        elif accelerator == 'mps':
            precision = '16-mixed'
        else:
            precision = '32'
    
    # 创建训练器
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        max_epochs=config.training.max_epochs,
        min_epochs=config.training.min_epochs,
        gradient_clip_val=min(config.training.gradient_clip_val, 1.0),  # 限制梯度裁剪值
        gradient_clip_algorithm='norm',  # 使用L2范数裁剪
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        check_val_every_n_epoch=config.training.check_val_every_n_epoch,
        log_every_n_steps=config.training.log_every_n_steps,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=logger,
        deterministic=config.reproducibility.deterministic,
        benchmark=config.reproducibility.benchmark,
        # 性能优化
        sync_batchnorm=True if (accelerator == 'gpu' and devices > 1) else False
    )
    
    return trainer


def setup_logging(config: DictConfig, experiment_name: str) -> TensorBoardLogger:
    """设置日志记录"""
    # 使用自动版本递增，避免复用同一版本目录导致多次运行混写
    version = getattr(config.logging, 'version', None)
    if isinstance(version, str) and version.strip().lower() in {'stable', 'default', ''}:
        version = None

    logger = TensorBoardLogger(
        save_dir=config.logging.save_dir,
        name=experiment_name,
        version=version,
        default_hp_metric=False
    )
    
    # 记录超参数到TensorBoard
    hparams = {
        'learning_rate': config.training.optimizer.lr,
        'batch_size': config.data.batch_size,
        'max_epochs': config.training.max_epochs,
        'model_d_model': config.model.d_model,
        'model_n_heads': config.model.n_heads,
        'model_num_layers': config.model.num_layers,
        'dropout': config.model.dropout,
        'precision': config.training.precision if hasattr(config.training, 'precision') else '16-mixed',
        'optimizer_type': config.training.optimizer.name,
        'scheduler_type': config.training.scheduler.name if hasattr(config.training, 'scheduler') else 'none'
    }
    
    # 记录超参数
    logger.log_hyperparams(hparams)
    
    return logger


def load_device_info(config: DictConfig) -> Tuple[Dict, List[str]]:
    """加载设备信息"""
    # 这里应该从数据中推断设备信息
    # 暂时使用配置中的默认值
    device_names = config.data.device_names
    n_devices = len(device_names)
    
    device_info = {}
    for i, name in enumerate(device_names):
        device_info[i] = {
            'name': name,
            'type': 'unknown',  # 可以从数据中推断
            'min_power': 0.0,
            'max_power': 1000.0,
            'weight': 1.0,
            'pos_weight': 1.0
        }
    
    return device_info, device_names


def main(config: DictConfig) -> None:
    """主训练函数"""
    
    # 优化GPU性能：设置float32矩阵乘法精度
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')  # 使用medium精度以平衡性能和准确性
        print("已设置float32矩阵乘法精度为'medium'以优化GPU性能")
    
    # 设置随机种子
    # 兼容新的复现性配置：优先使用 reproducibility.seed，其次回退到顶层 seed
    if hasattr(config, 'reproducibility') and hasattr(config.reproducibility, 'seed'):
        _seed = config.reproducibility.seed
    elif hasattr(config, 'seed'):
        _seed = config.seed
    else:
        _seed = 42
    pl.seed_everything(_seed, workers=True)
    
    # 创建输出目录
    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载设备信息
    device_info, device_names = load_device_info(config)
    
    # 设置日志记录
    experiment_name = f"{config.project_name}_{config.experiment.name}"
    logger = setup_logging(config, experiment_name)
    
    # 创建数据模块
    datamodule = NILMDataModule(config)
    datamodule.setup()
    
    # 从数据统计中自动注入先验：pos_weight（用于BCE）与 init_p（用于分类头偏置）
    try:
        pos_weight_vec = getattr(datamodule, 'get_pos_weight', None)
        pos_weight_vec = pos_weight_vec() if callable(pos_weight_vec) else getattr(datamodule, 'pos_weight_vec', None)
        prior_p_vec = getattr(datamodule, 'get_prior_p', None)
        prior_p_vec = prior_p_vec() if callable(prior_p_vec) else getattr(datamodule, 'prior_p_vec', None)

        # 写入 device_info 供损失函数使用
        if pos_weight_vec is not None:
            try:
                for i in range(len(device_names)):
                    device_info[i]['pos_weight'] = float(pos_weight_vec[i].item())
                print(f"已自动注入 pos_weight 到 device_info: {pos_weight_vec.tolist()}")
            except Exception as e:
                print(f"注入 pos_weight 失败: {e}")

        # 写入模型配置以初始化分类头偏置
        if prior_p_vec is not None:
            try:
                p_list = [float(p) for p in prior_p_vec.tolist()]
                if hasattr(config, 'model') and hasattr(config.model, 'heads') and hasattr(config.model.heads, 'classification'):
                    config.model.heads.classification.init_p = p_list
                    print(f"已自动注入 init_p 到模型配置: {p_list}")
                else:
                    print("警告: 模型配置缺少 heads.classification，无法注入 init_p")
            except Exception as e:
                print(f"注入 init_p 失败: {e}")
    except Exception as e:
        print(f"读取并注入统计先验失败: {e}")

    # 基于训练数据统计，自动注入 pos_weight（用于分类BCE）与 init_p（用于分类头偏置初始化）
    try:
        # 注入 pos_weight 到 device_info，供 MultiTaskLoss 注册为 buffer 使用
        pos_weight_tensor = datamodule.get_pos_weight()
        if pos_weight_tensor is not None:
            pos_list = pos_weight_tensor.detach().cpu().tolist()
            for i, pw in enumerate(pos_list):
                if i in device_info:
                    device_info[i]['pos_weight'] = float(pw)
            # 确保开启 pos_weight 配置
            if hasattr(config, 'loss') and hasattr(config.loss, 'classification') and hasattr(config.loss.classification, 'pos_weight'):
                setattr(config.loss.classification.pos_weight, 'enable', True)

        # 注入先验阳性概率 init_p 到模型配置，用于分类头偏置初始化
        prior_p_tensor = datamodule.get_prior_p()
        if prior_p_tensor is not None:
            p_list = prior_p_tensor.detach().cpu().tolist()
            # 写入到模型头配置：model.heads.classification.init_p
            if hasattr(config, 'model') and hasattr(config.model, 'heads') and hasattr(config.model.heads, 'classification'):
                setattr(config.model.heads.classification, 'init_p', [float(p) for p in p_list])
    except Exception as e:
        print(f"自动注入 pos_weight/init_p 失败: {e}")

    # 从数据统计自动注入 pos_weight（分类损失）与 init_p（分类头偏置先验）
    try:
        pos_weight_vec = datamodule.get_pos_weight()
        prior_p_vec = datamodule.get_prior_p()

        # 注入到 device_info 供 MultiTaskLoss 使用
        if pos_weight_vec is not None:
            for i in range(len(device_names)):
                try:
                    # 支持 tensor 或 numpy/列表
                    val = float(pos_weight_vec[i].item() if hasattr(pos_weight_vec[i], 'item') else pos_weight_vec[i])
                    device_info[i]['pos_weight'] = val
                except Exception:
                    pass

        # 注入到模型配置，作为分类头的初始概率偏置
        if prior_p_vec is not None:
            prior_list = []
            for i in range(len(device_names)):
                try:
                    p = float(prior_p_vec[i].item() if hasattr(prior_p_vec[i], 'item') else prior_p_vec[i])
                except Exception:
                    p = 0.1
                # 合理范围保护
                if p <= 0.0:
                    p = 1e-3
                if p >= 1.0:
                    p = 1.0 - 1e-3
                prior_list.append(p)
            # 将先验概率写入配置供 FusionTransformer 的 MultiTaskHead 初始化偏置
            if hasattr(config, 'heads') and hasattr(config.heads, 'classification'):
                config.heads.classification.init_p = prior_list
            else:
                # 若配置中缺少分类头字段，则创建所需字段
                from omegaconf import OmegaConf
                missing = {
                    'heads': {
                        'classification': {
                            'init_p': prior_list
                        }
                    }
                }
                config.merge_with(OmegaConf.create(missing))

        # 打印确认信息
        print("自动注入的 pos_weight:", [device_info[i].get('pos_weight', 1.0) for i in range(len(device_names))])
        if hasattr(config, 'heads') and hasattr(config.heads, 'classification'):
            print("自动注入的 init_p:", getattr(config.heads.classification, 'init_p', None))
    except Exception as e:
        print(f"自动注入 pos_weight/init_p 失败: {e}")
    
    # 创建模型
    model = NILMLightningModule(config, device_info, device_names)
    
    # 创建训练器
    trainer = create_trainer(config, logger)
    
    # 记录模型图到TensorBoard（如果启用）
    if hasattr(config.debug, 'log_model_graph') and config.debug.log_model_graph:
        try:
            # 创建示例输入
            sample_batch = next(iter(datamodule.train_dataloader()))
            logger.experiment.add_graph(model, sample_batch)
            print("Model graph logged to TensorBoard")
        except Exception as e:
            print(f"Failed to log model graph: {e}")
    
    # 训练模型
    print(f"开始训练实验: {experiment_name}")
    print(f"设备: {trainer.accelerator} ({trainer.num_devices} devices)")
    print(f"精度: {trainer.precision}")
    print(f"最大轮数: {config.training.max_epochs}")
    
    trainer.fit(model, datamodule)
    
    # 测试模型
    test_results = None
    if config.evaluation.test_after_training:
        print("开始测试...")
        test_results = trainer.test(model, datamodule)
        
        # 记录最终测试结果到TensorBoard
        if test_results:
            for metric_name, metric_value in test_results[0].items():
                logger.experiment.add_scalar(f'final_test/{metric_name}', metric_value, 0)
    
    # 保存模型摘要
    model_summary = ModelSummary(model, max_depth=2)
    summary_path = output_dir / 'model_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(str(model_summary))
    
    # 保存最终结果
    results = {
        'best_val_score': model.best_val_score,
        'best_thresholds': model.best_thresholds,
        'test_results': test_results,
        'best_model_path': trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else None,
        'model_summary': str(model_summary),
        'config': OmegaConf.to_container(config, resolve=True)
    }
    
    results_path = output_dir / f"{experiment_name}_results.yaml"
    OmegaConf.save(results, results_path)
    
    print(f"训练完成! 最佳验证分数: {model.best_val_score:.4f}")
    print(f"结果保存至: {results_path}")
    print(f"模型摘要保存至: {summary_path}")
    if trainer.checkpoint_callback:
        print(f"最佳模型保存至: {trainer.checkpoint_callback.best_model_path}")


# 注意：此脚本不应直接执行，请使用统一入口 main.py
# 示例：python main.py train --config-name=default