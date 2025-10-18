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
import json

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

# 启用TF32优化以提升性能
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

from src.data.datamodule import NILMDataModule  # 使用新的工业级数据模块
from src.models.fusion_transformer import FusionTransformer, TemperatureScaling
from src.models.submeter_encoder import SubmeterEncoder
from src.losses.losses import create_loss_function, RECOMMENDED_LOSS_CONFIGS
from src.utils.metrics import NILMMetrics, ConsistencyMetrics, DelayMetrics
from src.models.priors import PriorKnowledgeIntegrator
from src.utils.conformal_prediction import MultiTaskConformalPredictor
from src.utils.conformal_evaluation import ConformalEvaluator
from src.utils.prototypes import PrototypeLibrary


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
        
        # 初始化损失函数 - 使用推荐配置
        loss_config = RECOMMENDED_LOSS_CONFIGS.get('balanced', {})
        # 如果config中有自定义损失配置，则覆盖默认值
        if hasattr(config, 'loss') and config.loss:
            loss_config.update(OmegaConf.to_container(config.loss, resolve=True))
        
        self.loss_fn = create_loss_function(loss_config)
        
        # 显式禁用损失权重调度，避免分类权重被自动开启
        try:
            self.loss_fn.enable_scheduling = False
        except Exception:
            pass
        
        # 打印损失权重配置用于验证
        print(f"损失函数权重配置: 分类={self.loss_fn.classification_weight}, "
              f"回归={self.loss_fn.regression_weight}, "
              f"守恒={self.loss_fn.conservation_weight}, "
              f"一致性={self.loss_fn.consistency_weight}")
        
        # 当前仅做seq2seq功率回归，强制关闭分类相关学习
        self.loss_fn.classification_weight = 0.0
        
        # 基于损失权重设置分类开关（分类权重<=0则视为禁用）
        try:
            self.classification_enabled = bool(getattr(self.loss_fn, 'classification_weight', 0.0) > 0.0)
        except Exception:
            self.classification_enabled = True
        
        # 初始化评估指标
        self.metrics = NILMMetrics(device_names, config.evaluation.threshold_method)
        
        # 温度缩放（用于校准）
        if config.model.calibration.enable:
            self.temperature_scaling = TemperatureScaling(self.n_devices)
        else:
            self.temperature_scaling = None

        # 训练期度量学习（蒸馏/对比学习）开关与组件，仅训练使用
        ml_conf = getattr(getattr(config, 'aux_training', None), 'metric_learning', None)
        self.metric_learning_enable = bool(getattr(ml_conf, 'enable', False))
        self.metric_margin = float(getattr(ml_conf, 'margin', 0.2) if ml_conf is not None else 0.2)
        self.metric_weight = float(getattr(ml_conf, 'weight', 0.2) if ml_conf is not None else 0.2)
        self.metric_use_power = bool(getattr(ml_conf, 'use_power', True) if ml_conf is not None else True)
        if self.metric_learning_enable:
            embed_dim = int(getattr(getattr(config.model.time_encoder, 'd_model', None), 'value', config.model.time_encoder.d_model)) if hasattr(config.model.time_encoder, 'd_model') else config.model.time_encoder.d_model
            self.submeter_encoder = SubmeterEncoder(n_devices=self.n_devices, embed_dim=embed_dim, hidden_dim=64)
            # 原型库：用于流式统计设备嵌入的均值/协方差并计算 Mahalanobis 距离
            self.prototype_library = PrototypeLibrary(n_devices=self.n_devices, embed_dim=embed_dim)
            # 训练期每设备距离日志缓冲，用于 epoch 末统计与可视化
            from collections import deque
            self._distance_log_buffer = [deque(maxlen=5000) for _ in range(self.n_devices)]
        else:
            self.submeter_encoder = None
            self.prototype_library = None
            self._distance_log_buffer = None
        
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

    def _safe_log(self, name: str, value: Any, **kwargs) -> None:
        """在未附加 Trainer 的环境中安全地进行日志记录。
        当 `self.trainer` 尚未注册时，直接跳过以避免 PyTorch Lightning 警告。
        """
        if getattr(self, 'trainer', None) is not None:
            self.log(name, value, **kwargs)
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        time_features = batch['time_features']  # (batch_size, window_size, n_time_features)
        freq_features = batch.get('freq_features', None)  # (batch_size, n_time_frames, n_freq_bins)
        time_positional = batch.get('time_positional', None)  # (batch_size, window_size, time_dim)
        aux_features = batch.get('aux_features', None)  # (batch_size, n_aux_features)
        
        return self.model(time_features, freq_features, time_positional, aux_features=aux_features, time_valid_mask=batch.get('time_valid_mask', None), freq_valid_mask=batch.get('freq_valid_mask', None))

    def forward_with_embeddings(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """前向传播（返回设备嵌入）。仅在模型支持时使用。"""
        time_features = batch['time_features']
        freq_features = batch.get('freq_features', None)
        time_positional = batch.get('time_positional', None)
        aux_features = batch.get('aux_features', None)
        out = self.model.forward_with_embeddings(time_features, freq_features, time_positional, aux_features, time_valid_mask=batch.get('time_valid_mask', None), freq_valid_mask=batch.get('freq_valid_mask', None))
        return out
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor], 
                     predictions: Tuple[torch.Tensor, torch.Tensor],
                     stage: str = 'train',
                     sample_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """计算损失"""
        # 兼容可能包含unknown的返回
        if isinstance(predictions, tuple) and len(predictions) == 3:
            pred_power, pred_states, unknown_pred = predictions
        else:
            pred_power, pred_states = predictions
            unknown_pred = None
        target_power = batch['target_power']  # (batch_size, n_devices)
        target_states = batch['target_states']  # (batch_size, n_devices)
        total_power = batch.get('total_power', None)  # (batch_size, 1) or None
        
        # 历史功率（用于先验损失）
        historical_power = batch.get('historical_power', None)
        window_length = batch.get('window_length', self.config.data.window_size)
        
        # 计算损失（返回 总损失 与 明细）
        total_loss, loss_details = self.loss_fn(
            pred_power=pred_power,
            pred_switch=pred_states,
            target_power=target_power,
            target_switch=target_states,
            total_power=total_power,
            unknown_pred=unknown_pred,
            sample_weights=sample_weights
        )
        # 将总损失写入明细，保持后续逻辑兼容
        loss_details['total'] = total_loss
        
        # 记录损失（在无 Trainer 时安全跳过）
        for loss_name, loss_value in loss_details.items():
            self._safe_log(
                f"{stage}/loss/{loss_name}", loss_value,
                on_step=(stage == 'train'), on_epoch=True,
                prog_bar=(loss_name == 'total'), sync_dist=True
            )
        
        return loss_details
    
    def _compute_metrics(self, batch: Dict[str, torch.Tensor],
                        predictions: Tuple[torch.Tensor, torch.Tensor],
                        stage: str = 'val') -> Dict[str, float]:
        """计算评估指标"""
        if isinstance(predictions, tuple):
            pred_power, pred_states = predictions[0], predictions[1]
        else:
            pred_power, pred_states = predictions
        target_power = batch['target_power']
        target_states = batch['target_states']
        
        # 应用温度缩放（分类启用且仅在验证/测试阶段进行校准）
        if self.classification_enabled:
            logits = pred_states
            if self.temperature_scaling is not None:
                logits = self.temperature_scaling(logits)
            # 应用sigmoid到分类输出
            pred_proba = torch.sigmoid(logits)
        else:
            # 占位，避免下游张量形状依赖；分类指标会被门控跳过
            pred_proba = torch.zeros_like(pred_states)
        
        # 计算所有窗口级指标（传入分类开关）
        metrics = self.metrics.compute_all_metrics(
            y_pred_power=pred_power,
            y_pred_proba=pred_proba,
            y_true_power=target_power,
            y_true_states=target_states,
            optimize_thresholds=(stage == 'val'),
            classification_enabled=self.classification_enabled,
        )
        
        # 记录主要指标（统一为层级式标签，按 epoch 写入）
        main_metrics = ['mae', 'nde', 'sae', 'teca', 'f1', 'mcc', 'pr_auc', 'roc_auc', 'score']
        for metric_name in main_metrics:
            if metric_name in metrics:
                # 分层标签：回归 / 分类 / 综合分数
                if metric_name in ['mae', 'nde', 'sae', 'teca']:
                    tag = f'{stage}/metrics/regression/{metric_name}'
                elif metric_name in ['f1', 'mcc', 'pr_auc', 'roc_auc']:
                    tag = f'{stage}/metrics/classification/{metric_name}'
                elif metric_name == 'score':
                    tag = f'{stage}/metrics/score'
                    # 兼容旧监控键，保留顶层 'stage/score'
                    self._safe_log(
                        f'{stage}/score', metrics[metric_name],
                        on_epoch=True, on_step=False,
                        prog_bar=True, sync_dist=True
                    )
                else:
                    tag = f'{stage}/{metric_name}'

                self._safe_log(
                    tag, metrics[metric_name],
                    on_epoch=True, on_step=False,
                    prog_bar=(metric_name in ['score', 'f1']), sync_dist=True
                )
        
        # 一致性指标
        consistency_metrics = {
            'power_balance_error': ConsistencyMetrics.power_balance_error(pred_power, target_power),
            'temporal_consistency': ConsistencyMetrics.temporal_consistency(pred_power)
        }
        if self.classification_enabled:
            consistency_metrics['state_power_consistency'] = ConsistencyMetrics.state_power_consistency(pred_power, pred_proba)
        for metric_name, metric_value in consistency_metrics.items():
            self._safe_log(f'{stage}/metrics/consistency/{metric_name}', metric_value, on_epoch=True, sync_dist=True)
        
        # 序列级指标（seq-MAE/seq-RMSE/DTW），仅当 batch 含有 target_seq 时计算
        if 'target_seq' in batch and batch['target_seq'] is not None:
            target_seq = batch['target_seq']
            try:
                with torch.no_grad():
                    seq_out = self.model.forward_seq(
                        batch.get('time_features'),
                        batch.get('freq_features'),
                        batch.get('time_positional'),
                        batch.get('aux_features'),
                        time_valid_mask=batch.get('time_valid_mask'),
                        freq_valid_mask=batch.get('freq_valid_mask')
                    )
                pred_seq = seq_out[0] if isinstance(seq_out, tuple) else seq_out
                # MAE/RMSE（按 B、T、K 汇总）
                seq_mae = torch.nanmean(torch.abs(pred_seq - target_seq)).item()
                seq_rmse = torch.sqrt(torch.nanmean((pred_seq - target_seq) ** 2)).item()
                
                # DTW：对每个样本/设备计算 L1 代价的DTW并取平均；为稳健性做下采样
                import numpy as np
                def _downsample_np(x: np.ndarray, max_len: int = 256) -> np.ndarray:
                    T = x.shape[0]
                    if T <= max_len:
                        return x
                    stride = int(np.ceil(T / max_len))
                    return x[::stride]
                def _dtw_l1(a: np.ndarray, b: np.ndarray) -> float:
                    a = a.astype(np.float64)
                    b = b.astype(np.float64)
                    m, n = len(a), len(b)
                    cost = np.full((m + 1, n + 1), np.inf, dtype=np.float64)
                    cost[0, 0] = 0.0
                    for i in range(1, m + 1):
                        ai = a[i - 1]
                        for j in range(1, n + 1):
                            bj = b[j - 1]
                            d = abs(ai - bj)
                            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
                    return float(cost[m, n])
                pred_np = pred_seq.detach().cpu().numpy()
                target_np = target_seq.detach().cpu().numpy()
                B = pred_np.shape[0]
                K = pred_np.shape[2] if pred_np.ndim == 3 else 1
                dtw_vals = []
                for b in range(B):
                    for k in range(K):
                        p = pred_np[b, :, k] if K > 1 else (pred_np[b, :, 0] if pred_np.ndim == 3 else pred_np[b, :])
                        t = target_np[b, :, k] if K > 1 else (target_np[b, :, 0] if target_np.ndim == 3 else target_np[b, :])
                        p = _downsample_np(p)
                        t = _downsample_np(t)
                        dtw_vals.append(_dtw_l1(p, t))
                seq_dtw = float(np.mean(dtw_vals)) if dtw_vals else float('nan')
                
                # 记录与返回
                self._safe_log(f'{stage}/metrics/sequence/mae', seq_mae, on_epoch=True, sync_dist=True)
                self._safe_log(f'{stage}/metrics/sequence/rmse', seq_rmse, on_epoch=True, sync_dist=True)
                self._safe_log(f'{stage}/metrics/sequence/dtw', seq_dtw, on_epoch=True, sync_dist=True)
                metrics.update({'seq_mae': seq_mae, 'seq_rmse': seq_rmse, 'seq_dtw': seq_dtw})
            except Exception:
                # 若序列前向或计算失败，不阻断整体评估
                pass
        
        return metrics
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """训练步骤"""
        # 数据验证和NaN检测
        self._validate_batch_data(batch, 'train', batch_idx)

        # 前向传播：若启用度量学习，则同时返回设备嵌入
        if self.metric_learning_enable and self.classification_enabled:
            out = self.model.forward_with_embeddings(
                batch['time_features'],
                batch.get('freq_features'),
                batch.get('time_positional'),
                batch.get('aux_features'),
                time_valid_mask=batch.get('time_valid_mask'),
                freq_valid_mask=batch.get('freq_valid_mask')
            )
            # 兼容是否包含unknown的返回
            if isinstance(out, tuple) and len(out) == 4:
                pred_power, pred_states, unknown_pred, pred_embeddings = out
                predictions = (pred_power, pred_states, unknown_pred)
            else:
                pred_power, pred_states, pred_embeddings = out
                predictions = (pred_power, pred_states)
        else:
            # 主 forward 已在模型中兼容 unknown 返回，这里保持两元组
            predictions = self(batch)
        self._validate_predictions(predictions, 'train', batch_idx)
        
        # 窗口屏蔽/降权(仅训练)
        sample_weights = None
        # 训练步：根据配置动态计算样本权重
        if hasattr(self.config, 'training') and hasattr(self.config.training, 'masking'):
            mask_cfg = self.config.training.masking
            if getattr(mask_cfg, 'window_strategy', 'none') in ('drop', 'downweight'):
                # 确定用于计算丢弃/权重的有效率
                ratios = []
                if 'time_valid_ratio' in batch:
                    ratios.append(batch['time_valid_ratio'])
                if getattr(mask_cfg, 'combine_modalities', False) and 'freq_valid_ratio' in batch:
                    ratios.append(batch['freq_valid_ratio'])

                if ratios:
                    # 组合多个有效率：默认取最小值，确保最差的模态满足要求
                    combined_ratio = torch.stack(ratios).min(dim=0).values
                    
                    # 确定对应模态的阈值
                    thresholds = []
                    if 'time_valid_ratio' in batch:
                        thresholds.append(getattr(mask_cfg, 'min_valid_ratio_time', 0.8))
                    if getattr(mask_cfg, 'combine_modalities', False) and 'freq_valid_ratio' in batch:
                        thresholds.append(getattr(mask_cfg, 'min_valid_ratio_freq', 0.8))
                    
                    min_valid_ratio = max(thresholds) if thresholds else 0.8

                    if getattr(mask_cfg, 'window_strategy', 'none') == 'drop':
                        keep_mask = (combined_ratio >= min_valid_ratio)
                        # 通过权重的方式实现丢弃（避免真正过滤导致维度不一致）
                        sample_weights = keep_mask.float()
                    elif getattr(mask_cfg, 'window_strategy', 'none') == 'downweight':
                        # 小于阈值的样本权重从1线性下降到epsilon
                        w = torch.ones_like(combined_ratio)
                        denom = max(1e-6, 1.0 - float(min_valid_ratio))
                        epsilon = float(getattr(mask_cfg, 'epsilon', 0.2))
                        slope = (1.0 - epsilon) / denom
                        power = float(getattr(mask_cfg, 'downweight_power', 1.0))
                        weights_unclamped = epsilon + slope * (combined_ratio - min_valid_ratio)
                        weights_unclamped = torch.pow(weights_unclamped.clamp(min=epsilon), power)

                        sample_weights = torch.where(
                            combined_ratio >= min_valid_ratio, 
                            w, 
                            weights_unclamped
                        )

        losses = self._compute_loss(batch, predictions, 'train', sample_weights=sample_weights)

        # 序列→序列回归损失（替代窗口均值回归）
        if 'target_seq' in batch and batch['target_seq'] is not None:
            seq_out = self.model.forward_seq(
                batch['time_features'],
                batch.get('freq_features'),
                batch.get('time_positional'),
                batch.get('aux_features'),
                time_valid_mask=batch.get('time_valid_mask'),
                freq_valid_mask=batch.get('freq_valid_mask')
            )
            # 解包序列输出
            if isinstance(seq_out, tuple):
                pred_seq = seq_out[0]
            else:
                pred_seq = seq_out
            target_seq = batch['target_seq']
            # 屏蔽无效标签，稳定序列损失
            valid = torch.isfinite(pred_seq) & torch.isfinite(target_seq)
            element_loss = torch.nn.functional.smooth_l1_loss(pred_seq, target_seq, reduction='none')
            element_loss = torch.where(valid, element_loss, torch.zeros_like(element_loss))
            denom = valid.float().sum().clamp_min(1.0)
            seq_loss = element_loss.sum() / denom
            # 禁用窗口级回归损失以满足纯序列监督
            try:
                self.loss_fn.regression_weight = 0.0
            except Exception:
                pass
            losses['seq_regression'] = seq_loss
            losses['total'] = losses['total'] + losses['seq_regression']
            self._safe_log('train/loss/sequence', seq_loss, on_step=True, on_epoch=True, prog_bar=True)

        # 度量学习辅助损失（仅训练期，且不影响推理）
        if self.metric_learning_enable and self.classification_enabled:
            aux_loss = self._compute_aux_metric_loss(
                batch,
                pred_embeddings,
            )
            losses['aux_metric'] = aux_loss
            losses['total'] = losses['total'] + self.metric_weight * aux_loss

            # 原型库更新与距离日志（仅训练期）
            try:
                self._update_prototypes_and_log(pred_embeddings, batch.get('target_states'))
            except Exception as e:
                # 在调试阶段避免因日志失败中断训练
                print(f"Prototype update/logging failed at step {batch_idx}: {e}")
        
        # 检测损失中的NaN/Inf
        self._validate_losses(losses, 'train', batch_idx)
        
        # 记录学习率（按 epoch），在未附加 Trainer 的单元测试场景下跳过
        if getattr(self, 'trainer', None) is not None and getattr(self.trainer, 'optimizers', None):
            self._safe_log('train/metrics/optimization/lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=False, on_epoch=True)
        
        # 每隔一定步数计算训练指标
        if batch_idx % self.config.training.log_every_n_steps == 0:
            with torch.no_grad():
                self._compute_metrics(batch, predictions, 'train')
                
                # 记录梯度范数 - 增强梯度监控
                if self.config.debug.track_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=float('inf'))
                    # 统一使用 epoch 级记录（在无 Trainer 时安全跳过）
                    self._safe_log('train/metrics/optimization/grad_norm', grad_norm, on_step=False, on_epoch=True)
                    
                    # 检查梯度异常
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        self._safe_log('train/metrics/optimization/grad_anomaly', 1.0, on_step=True, on_epoch=False)
                        print(f"Warning: Invalid gradient norm at step {self.global_step}: {grad_norm}")
        
        return losses['total']

    @torch.no_grad()
    def _update_prototypes_and_log(self, pred_embeddings: torch.Tensor, target_states: Optional[torch.Tensor]) -> None:
        """更新原型统计并记录 Mahalanobis 距离摘要。

        - 更新：仅在设备处于开启状态时统计（若提供 target_states）。
        - 日志：当某设备样本数达到阈值后，记录距离的均值与最大值（epoch 级）。
        """
        if self.prototype_library is None:
            return
        # 更新原型统计
        self.prototype_library.update(pred_embeddings, states=target_states)

        # 计算距离并记录摘要（按 epoch）
        distances = self.prototype_library.mahalanobis(pred_embeddings)  # (B, N)
        if distances is None:
            return
        # 只统计已就绪设备的距离（min_count 达到阈值）
        ready_mask = torch.tensor([
            self.prototype_library.is_ready(i) for i in range(self.n_devices)
        ], device=distances.device)
        if ready_mask.any():
            # 选择已就绪列
            sel = distances[:, ready_mask]
            mean_d = sel.mean()
            max_d = sel.max()
            self._safe_log('train/mahalanobis_mean', mean_d, on_step=False, on_epoch=True, sync_dist=True)
            self._safe_log('train/mahalanobis_max', max_d, on_step=False, on_epoch=True, sync_dist=True)

        # 累积每设备距离样本用于 epoch 末统计与可视化
        if self._distance_log_buffer is not None:
            with torch.no_grad():
                B, N = distances.shape
                for i in range(N):
                    if self.prototype_library.is_ready(i):
                        vals = distances[:, i].detach().cpu().tolist()
                        buf = self._distance_log_buffer[i]
                        for v in vals:
                            buf.append(v)

    def _compute_aux_metric_loss(self, batch: Dict[str, torch.Tensor], pred_embeddings: torch.Tensor) -> torch.Tensor:
        """计算度量学习辅助损失，使主干设备头嵌入接近对应分表原型。

        - 正样本：设备开启 (y_on=1) 时，预测嵌入与其原型的余弦距离最小化。
        - 负样本：与其他设备原型的相似度低于 margin（hinge）。
        """
        assert self.submeter_encoder is not None, "SubmeterEncoder is required when metric learning is enabled"

        target_states: torch.Tensor = batch['target_states']  # (B, N)
        target_power: torch.Tensor = batch.get('target_power', None)  # (B, N) or None

        # 原型编码（训练期）
        prototypes = self.submeter_encoder(
            target_power if self.metric_use_power else None,
            target_states
        )  # (B, N, D)

        # 归一化后的余弦相似度
        pred = pred_embeddings  # (B, N, D)
        pred = torch.nn.functional.normalize(pred, dim=-1)
        pos_sim = (pred * prototypes).sum(dim=-1)  # (B, N)

        # 正样本损失：maximizing similarity -> minimize (1 - sim)
        pos_mask = (target_states > 0).float()
        pos_loss = ((1.0 - pos_sim) * pos_mask).sum() / (pos_mask.sum() + 1e-6)

        # 负样本：与其他设备原型相似度不应高于 margin
        B, N, D = pred.size()
        # 扩展计算 (B, N, N) 相似度矩阵：pred_i · proto_j
        pred_exp = pred.unsqueeze(2).expand(B, N, N, D)
        proto_exp = prototypes.unsqueeze(1).expand(B, N, N, D)
        sim_matrix = (pred_exp * proto_exp).sum(dim=-1)  # (B, N, N)

        # 屏蔽对角（自身设备）
        eye = torch.eye(N, device=sim_matrix.device).unsqueeze(0)
        neg_sim = sim_matrix * (1.0 - eye)

        # 仅在设备开启时应用负样本约束
        pos_mask_exp = pos_mask.unsqueeze(2).expand(B, N, N)
        # hinge: relu(sim - margin)
        neg_margin = torch.relu(neg_sim - self.metric_margin) * pos_mask_exp
        neg_loss = neg_margin.sum() / (pos_mask_exp.sum() + 1e-6)

        aux_loss = pos_loss + neg_loss
        # 记录到日志（未加权，在无 Trainer 时安全跳过）
        self._safe_log('train/aux_metric_pos', pos_loss, on_step=False, on_epoch=True)
        self._safe_log('train/aux_metric_neg', neg_loss, on_step=False, on_epoch=True)
        self._safe_log('train/aux_metric_total', aux_loss, on_step=False, on_epoch=True)

        return aux_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """验证步骤"""
        # 数据验证和NaN检测
        self._validate_batch_data(batch, 'val', batch_idx)
        
        predictions = self(batch)
        self._validate_predictions(predictions, 'val', batch_idx)
        
        # 验证步：可选样本权重（与训练一致的策略）
        sample_weights = None
        if hasattr(self.config, 'training') and hasattr(self.config.training, 'masking'):
            mask_cfg = self.config.training.masking
            if getattr(mask_cfg, 'window_strategy', 'none') in ('drop', 'downweight'):
                ratios = []
                if 'time_valid_ratio' in batch:
                    ratios.append(batch['time_valid_ratio'])
                if getattr(mask_cfg, 'combine_modalities', False) and 'freq_valid_ratio' in batch:
                    ratios.append(batch['freq_valid_ratio'])
                if ratios:
                    combined_ratio = torch.stack(ratios).min(dim=0).values
                    thresholds = []
                    if 'time_valid_ratio' in batch:
                        thresholds.append(getattr(mask_cfg, 'min_valid_ratio_time', 0.8))
                    if getattr(mask_cfg, 'combine_modalities', False) and 'freq_valid_ratio' in batch:
                        thresholds.append(getattr(mask_cfg, 'min_valid_ratio_freq', 0.8))
                    min_valid_ratio = max(thresholds) if thresholds else 0.8
                    if getattr(mask_cfg, 'window_strategy', 'none') == 'drop':
                        sample_weights = (combined_ratio >= min_valid_ratio).float()
                    else:
                        w = torch.ones_like(combined_ratio)
                        denom = max(1e-6, 1.0 - float(min_valid_ratio))
                        epsilon = float(getattr(mask_cfg, 'epsilon', 0.2))
                        slope = (1.0 - epsilon) / denom
                        power = float(getattr(mask_cfg, 'downweight_power', 1.0))
                        weights_unclamped = epsilon + slope * (combined_ratio - min_valid_ratio)
                        weights_unclamped = torch.pow(weights_unclamped.clamp(min=epsilon), power)
                        sample_weights = torch.where(combined_ratio >= min_valid_ratio, w, weights_unclamped)
        
        losses = self._compute_loss(batch, predictions, 'val', sample_weights=sample_weights)
        # 验证期的序列→序列损失（若提供序列标签）
        if 'target_seq' in batch and batch['target_seq'] is not None:
            seq_out = self.model.forward_seq(
                batch['time_features'],
                batch.get('freq_features'),
                batch.get('time_positional'),
                batch.get('aux_features'),
                time_valid_mask=batch.get('time_valid_mask'),
                freq_valid_mask=batch.get('freq_valid_mask')
            )
            pred_seq = seq_out[0] if isinstance(seq_out, tuple) else seq_out
            # 屏蔽无效标签，稳定序列损失
            target_seq = batch['target_seq']
            valid = torch.isfinite(pred_seq) & torch.isfinite(target_seq)
            element_loss = torch.nn.functional.smooth_l1_loss(pred_seq, target_seq, reduction='none')
            element_loss = torch.where(valid, element_loss, torch.zeros_like(element_loss))
            denom = valid.float().sum().clamp_min(1.0)
            seq_loss = element_loss.sum() / denom
            self._safe_log('val/seq_regression', seq_loss, on_step=False, on_epoch=True)
            losses['total'] = losses['total'] + seq_loss
        # 在添加序列损失后再进行损失有效性校验
        self._validate_losses(losses, 'val', batch_idx)
        metrics = self._compute_metrics(batch, predictions, 'val')
        
        # 显式按 epoch 记录验证损失与分数（在无 Trainer 时安全跳过）
        self._safe_log('val/loss', losses['total'], on_step=False, on_epoch=True, prog_bar=True)
        self._safe_log('val/score', metrics['score'], on_step=False, on_epoch=True, prog_bar=True)

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
        # 测试期的序列→序列损失（若提供序列标签）
        if 'target_seq' in batch and batch['target_seq'] is not None:
            seq_out = self.model.forward_seq(
                batch['time_features'],
                batch.get('freq_features'),
                batch.get('time_positional'),
                batch.get('aux_features'),
                time_valid_mask=batch.get('time_valid_mask'),
                freq_valid_mask=batch.get('freq_valid_mask')
            )
            pred_seq = seq_out[0] if isinstance(seq_out, tuple) else seq_out
            # 屏蔽无效标签，稳定序列损失
            target_seq = batch['target_seq']
            valid = torch.isfinite(pred_seq) & torch.isfinite(target_seq)
            element_loss = torch.nn.functional.smooth_l1_loss(pred_seq, target_seq, reduction='none')
            element_loss = torch.where(valid, element_loss, torch.zeros_like(element_loss))
            denom = valid.float().sum().clamp_min(1.0)
            seq_loss = element_loss.sum() / denom
            self._safe_log('test/loss/sequence', seq_loss, on_step=False, on_epoch=True)
            losses['total'] = losses['total'] + seq_loss
        # 在添加序列损失后再进行损失有效性校验
        self._validate_losses(losses, 'test', batch_idx)
        metrics = self._compute_metrics(batch, predictions, 'test')
        
        # 显式按 epoch 记录测试损失与分数（在无 Trainer 的场景安全跳过）
        self._safe_log('test/loss/total', losses['total'], on_step=False, on_epoch=True)
        # 同时保留旧键以兼容现有监控
        self._safe_log('test/loss', losses['total'], on_step=False, on_epoch=True)
        self._safe_log('test/metrics/score', metrics['score'], on_step=False, on_epoch=True)
        self._safe_log('test/score', metrics['score'], on_step=False, on_epoch=True)

        return {
            'test_loss': losses['total'],
            'test_score': metrics['score'],
            'predictions': predictions,
            'targets': (batch['target_power'], batch['target_states'])
        }

    def on_train_epoch_end(self) -> None:
        """训练轮次结束：记录每设备原型样本数与距离分布统计，并进行可视化。"""
        if self.prototype_library is None or self._distance_log_buffer is None or getattr(self, 'logger', None) is None:
            return
        try:
            import numpy as np
            for i, device_name in enumerate(self.device_names):
                c = int(self.prototype_library.count[i].item())
                self._safe_log(f'train/prototype_count/{device_name}', c, on_step=False, on_epoch=True)
                vals = list(self._distance_log_buffer[i])
                if len(vals) == 0:
                    continue
                arr = np.array(vals, dtype=np.float32)
                mean_v = float(arr.mean())
                p50 = float(np.percentile(arr, 50))
                p90 = float(np.percentile(arr, 90))
                p95 = float(np.percentile(arr, 95))
                self._safe_log(f'train/distance_mean/{device_name}', mean_v, on_step=False, on_epoch=True)
                self._safe_log(f'train/distance_p50/{device_name}', p50, on_step=False, on_epoch=True)
                self._safe_log(f'train/distance_p90/{device_name}', p90, on_step=False, on_epoch=True)
                self._safe_log(f'train/distance_p95/{device_name}', p95, on_step=False, on_epoch=True)
                # 可视化直方图
                try:
                    self.logger.experiment.add_histogram(f'train/distance_hist/{device_name}', torch.tensor(arr), self.current_epoch)
                except Exception:
                    pass
            # 清空缓冲以便下一轮
            for buf in self._distance_log_buffer:
                buf.clear()
        except Exception as e:
            print(f"on_train_epoch_end distance logging failed: {e}")

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """保存检查点时持久化原型库状态。"""
        try:
            if self.prototype_library is not None:
                checkpoint['prototype_library_state'] = self.prototype_library.state_dict()
        except Exception as e:
            print(f"保存原型库状态失败: {e}")

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """加载检查点时恢复原型库状态。"""
        try:
            state = checkpoint.get('prototype_library_state', None)
            if state is not None and self.prototype_library is not None:
                self.prototype_library.load_state_dict(state)
        except Exception as e:
            print(f"加载原型库状态失败: {e}")
    
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
            # 记录最佳验证分数（层级命名，在无 Trainer 时安全跳过）
            self._safe_log('val/best_score', self.best_val_score, on_epoch=True)
            
            # 记录最佳阈值
            for device_name, threshold in self.best_thresholds.items():
                self._safe_log(f'best_threshold/{device_name}', threshold, on_epoch=True)

            # 持久化保存最佳阈值
            try:
                thresholds_path = Path(self.config.paths.output_dir) / 'best_thresholds.json'
                self.metrics.save_thresholds(str(thresholds_path))
                print(f"最佳阈值已保存至: {thresholds_path}")
            except Exception as e:
                print(f"保存最佳阈值失败: {e}")
        
        # Conformal Prediction标定器
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
                        
                        # 评估回归区间（从多任务预测器聚合区间）
                        mt_results = self.conformal_predictor.predict_with_intervals((pred_powers, pred_states))
                        lower_bounds = []
                        upper_bounds = []
                        for device_name in self.device_names:
                            lower_bounds.append(mt_results[device_name]['power']['lower_bound'])
                            upper_bounds.append(mt_results[device_name]['power']['upper_bound'])
                        reg_intervals = {
                            'lower': torch.stack(lower_bounds, dim=1),
                            'upper': torch.stack(upper_bounds, dim=1),
                        }
                        reg_eval = self.conformal_evaluator.evaluate_regression_intervals(
                            true_values=target_powers,
                            predicted_intervals=reg_intervals,
                            device_names=self.device_names
                        )
                        # 记录回归评估指标
                        for device_name in self.device_names:
                            device_metrics = reg_eval.get(device_name, {})
                            for metric_name, value in device_metrics.items():
                                # 归入验证命名空间，并使用层级式标签
                                self._safe_log(f'val/conformal/regression/{device_name}/{metric_name}', value, on_epoch=True)
                        
                        # 汇总评估结果用于报告
                        evaluation_results = {'regression': reg_eval}
                        
                        # 评估分类校准（受分类开关控制）
                        if self.classification_enabled:
                            cls_logits = pred_states
                            if self.temperature_scaling is not None:
                                cls_logits = self.temperature_scaling(cls_logits)
                            pred_proba = torch.sigmoid(cls_logits)
                            
                            # 从分类预测器提取每设备的校准阈值（quantile）
                            thresholds = []
                            for device_name in self.device_names:
                                thr = self.conformal_predictor.classification_predictors[device_name].quantile
                                if isinstance(thr, torch.Tensor):
                                    thresholds.append(thr if thr.dim() == 0 else thr.squeeze())
                                else:
                                    thresholds.append(torch.tensor(thr, device=pred_proba.device))
                            calibrated_thresholds = torch.stack(thresholds)
                            
                            cls_eval = self.conformal_evaluator.evaluate_classification_calibration(
                                true_labels=target_states,
                                predicted_probs=pred_proba,
                                calibrated_thresholds=calibrated_thresholds,
                                device_names=self.device_names
                            )
                            
                            # 记录分类评估指标
                            for device_name in self.device_names:
                                device_metrics = cls_eval.get(device_name, {})
                                for metric_name, value in device_metrics.items():
                                    self._safe_log(f'val/conformal/classification/{device_name}/{metric_name}', value, on_epoch=True)
                            
                            evaluation_results['classification'] = cls_eval
                        
                        # 记录标定信息（归入验证命名空间）
                        self._safe_log('val/conformal/calibrated', 1.0, on_epoch=True)
                        print(f"Conformal prediction calibrated at epoch {self.current_epoch}")
                        
                        # 生成评估报告（每10个epoch一次）
                        if self.conformal_evaluator is not None and self.current_epoch % 10 == 0:
                            report_dir = f"conformal_report_epoch_{self.current_epoch}"
                            self.conformal_evaluator.generate_visualization_report(evaluation_results, save_path=report_dir)
                            print(f"Conformal evaluation report saved to {report_dir}")
        
        # 记录最佳验证分数（归入验证命名空间，安全记录）
        self._safe_log('val/best_score', self.best_val_score, on_epoch=True)
        
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
        suppress_invalid = getattr(getattr(self.config, 'debug', {}), 'suppress_invalid_warnings', True)
        for key, tensor in batch.items():
            # 只对tensor类型的数据进行验证
            if isinstance(tensor, torch.Tensor) and not torch.isfinite(tensor).all():
                nan_count = torch.isnan(tensor).sum().item()
                inf_count = torch.isinf(tensor).sum().item()
                self._safe_log(f'{stage}/data_nan_count/{key}', float(nan_count), on_step=True, on_epoch=False)
                self._safe_log(f'{stage}/data_inf_count/{key}', float(inf_count), on_step=True, on_epoch=False)

                # 若是已掩蔽的目标字段，则在非严格模式下抑制告警输出（仍记录到 TensorBoard）
                masked = False
                if key == 'target_power' and isinstance(batch.get('target_power_valid_mask'), torch.Tensor):
                    masked = True
                elif key == 'target_seq' and isinstance(batch.get('target_seq_valid_mask'), torch.Tensor):
                    masked = True
                
                if self.config.debug.strict_validation:
                    raise ValueError(f"发现无效数据在 {stage} batch {batch_idx}, key '{key}': "
                                   f"NaN: {nan_count}, Inf: {inf_count}")
                else:
                    if masked and suppress_invalid:
                        # 跳过打印，避免训练日志噪声
                        continue
                    print(f"警告: {stage} batch {batch_idx} 中 '{key}' 包含无效值: NaN: {nan_count}, Inf: {inf_count}")
    
    def _validate_predictions(self, predictions: Tuple[torch.Tensor, ...], 
                            stage: str, batch_idx: int) -> None:
        """验证预测结果的有效性，兼容 (power, states) 或 (power, states, unknown)。"""
        if not isinstance(predictions, tuple) or len(predictions) < 2:
            raise ValueError(f"{stage} batch {batch_idx} 预测结果格式异常: 期望至少包含功率与状态")
        pred_power, pred_states = predictions[0], predictions[1]
        
        # 检查功率预测
        if not torch.isfinite(pred_power).all():
            nan_count = torch.isnan(pred_power).sum().item()
            inf_count = torch.isinf(pred_power).sum().item()
            self._safe_log(f'{stage}/pred_power_nan_count', float(nan_count), on_step=True, on_epoch=False)
            self._safe_log(f'{stage}/pred_power_inf_count', float(inf_count), on_step=True, on_epoch=False)
            
            if self.config.debug.strict_validation:
                raise ValueError(f"功率预测包含无效值在 {stage} batch {batch_idx}: NaN: {nan_count}, Inf: {inf_count}")
        
        # 检查状态预测（仅在分类启用时）
        if self.classification_enabled:
            if not torch.isfinite(pred_states).all():
                nan_count = torch.isnan(pred_states).sum().item()
                inf_count = torch.isinf(pred_states).sum().item()
                self._safe_log(f'{stage}/pred_states_nan_count', float(nan_count), on_step=True, on_epoch=False)
                self._safe_log(f'{stage}/pred_states_inf_count', float(inf_count), on_step=True, on_epoch=False)
                
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
                self._safe_log(f'{stage}/loss_invalid/{loss_name}', 1.0, on_step=True, on_epoch=False)
                
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
    
    # 新增：环境变量强制CPU，避免macOS MPS OOM
    force_cpu = os.environ.get('DISAGGNET_FORCE_CPU', '0') == '1'
    
    # 设备配置（自动检测：macOS->MPS，Linux/Windows->CUDA，否则CPU）
    if force_cpu:
        accelerator = 'cpu'
        devices = 1
    elif getattr(config.training, 'accelerator', 'auto') == 'auto':
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
    """加载设备信息。
    优先从 Data/prepared/device_name_to_id.json 推断设备名称；若不可用则回退到配置中的 data.device_names。
    """
    # 推断 prepared 数据目录
    try:
        prepared_dir = Path(getattr(config.paths, 'prepared_dir'))
    except Exception:
        prepared_dir = Path('Data/prepared')
    mapping_path = prepared_dir / 'device_name_to_id.json'

    device_names: List[str] = []
    # 若存在映射文件，按 id 顺序解析设备名称
    if mapping_path.exists():
        try:
            with open(mapping_path, 'r') as f:
                mapping = json.load(f)

            def _to_int(x):
                try:
                    return int(x)
                except Exception:
                    return None

            sample_key = next(iter(mapping.keys()))
            sample_val = mapping[sample_key]
            key_is_int_like = _to_int(sample_key) is not None
            val_is_int_like = _to_int(sample_val) is not None

            if key_is_int_like and not val_is_int_like:
                # id(str)->name(str)
                pairs = sorted(((int(k), v) for k, v in mapping.items()), key=lambda kv: kv[0])
                device_names = [name for _, name in pairs]
            elif not key_is_int_like and val_is_int_like:
                # name->id
                pairs = sorted(((v, k) for k, v in mapping.items()), key=lambda kv: kv[0])
                device_names = [name for _, name in pairs]
            else:
                # 回退：按键排序
                device_names = sorted(list(mapping.keys()))
        except Exception as e:
            print(f"[警告] 解析 {mapping_path} 失败，回退到配置 data.device_names：{e}")

    # 若未能从映射获取，或设备列表为空，则使用配置中的 device_names
    if not device_names:
        try:
            cfg_names = list(getattr(getattr(config, 'data', {}), 'device_names', []) or [])
        except Exception:
            cfg_names = []
        device_names = cfg_names

    # 若仍为空，构造占位名称
    if not device_names:
        device_names = ['device_1']
        print('[警告] 未能推断设备名称，使用占位 device_1')

    # 构造设备信息字典（默认值，后续会由数据模块注入 pos_weight 等）
    device_info: Dict[int, Dict[str, Any]] = {}
    for i, name in enumerate(device_names):
        device_info[i] = {
            'name': name,
            'type': 'unknown',
            'min_power': 0.0,
            'max_power': 1000.0,
            'weight': 1.0,
            'pos_weight': 1.0,
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