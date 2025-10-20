"""训练脚本"""

import os
import sys
# macOS OpenMP workaround: avoid multiple runtime init errors
if sys.platform == 'darwin':
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
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
import matplotlib
matplotlib.use('Agg')

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

# 启用TF32优化以提升性能
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

from src.data.datamodule import NILMDataModule  # 使用新的工业级数据模块
from src.models.fusion_transformer import FusionTransformer
from src.models.submeter_encoder import SubmeterEncoder
from src.losses.losses import create_loss_function, RECOMMENDED_LOSS_CONFIGS

from src.models.priors import PriorKnowledgeIntegrator
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
        # 明确暴露分类开关属性以配合测试和下游逻辑
        self.classification_enabled = False

        # 训练期度量学习（蒸馏/对比学习）开关与组件，仅训练使用
        ml_conf = getattr(getattr(config, 'aux_training', None), 'metric_learning', None)
        self.metric_learning_enable = bool(getattr(ml_conf, 'enable', False))
        self.metric_margin = max(0.0, float(getattr(ml_conf, 'margin', 0.2) if ml_conf is not None else 0.2))
        self.metric_weight = max(0.0, float(getattr(ml_conf, 'weight', 0.2) if ml_conf is not None else 0.2))
        self.metric_use_power = bool(getattr(ml_conf, 'use_power', False) if ml_conf is not None else False)
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
        
        # Conformal Prediction 标定器（序列回归不使用）
        self.conformal_predictor = None
        self.conformal_evaluator = None
        
        # 验证集最佳指标（与ModelCheckpoint的模式对齐）
        ckpt_cfg = getattr(getattr(config, 'training', None), 'checkpoint', None)
        mode = str(getattr(ckpt_cfg, 'mode', 'max')).lower() if ckpt_cfg is not None else 'max'
        self.best_val_score = float('-inf') if mode == 'max' else float('inf')
        # 兼容下游摘要与推理：在仅序列回归场景下也提供阈值占位
        self.best_thresholds = {}

        # 训练状态
        self.automatic_optimization = True

        # 新增：事件关注与可视化配置，以及序列例子缓冲
        ev_cfg = getattr(getattr(config, 'training', None), 'event_focus', None)
        self.event_focus_enable = bool(getattr(ev_cfg, 'enable', True) if ev_cfg is not None else True)
        self.event_weight_factor = float(getattr(ev_cfg, 'weight', 2.0) if ev_cfg is not None else 2.0)
        vis_cfg = getattr(getattr(config, 'training', None), 'visualization', None)
        self.plot_event_only = bool(getattr(vis_cfg, 'plot_event_only', True) if vis_cfg is not None else True)
        self.max_plots_per_epoch = int(getattr(vis_cfg, 'max_plots_per_epoch', 16) if vis_cfg is not None else 16)
        # 新增：绘图总开关（默认禁用），避免 Windows 上阻塞
        self.enable_visualization = bool(getattr(vis_cfg, 'enable', False) if vis_cfg is not None else False)
        # 指标零窗口权重（降低全零窗口影响）
        eval_cfg = getattr(getattr(config, 'evaluation', None), 'zero_window_weight', None)
        self.metrics_zero_weight = float(eval_cfg if eval_cfg is not None else 0.2)
        self._sequence_examples: List[Dict[str, Any]] = []

    def _safe_log(self, name: str, value: Any, **kwargs) -> None:
        """在未附加 Trainer 的环境中安全地进行日志记录。
        当未附加 Trainer 时，直接跳过以避免 PyTorch Lightning 报错。
        """
        # 避免触发 Lightning 的 `trainer` 属性访问（未附加时会抛错）
        if getattr(self, '_trainer', None) is not None:
            try:
                self.log(name, value, **kwargs)
            except Exception:
                pass
        
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
    
    def _compute_metrics(self, batch: Dict[str, torch.Tensor], preds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, stage: str = 'val') -> Dict[str, float]:
        """计算评估指标。

        - 若提供 `preds=(pred_power, pred_states)`，计算窗口级 NILM 指标（MAE/NDE/SAE/TECA），并根据 `classification_enabled` 决定是否计算分类指标。
        - 若未提供，则计算序列级指标（MAE/RMSE/DTW），与当前 seq2seq 训练保持一致。
        """
        metrics: Dict[str, float] = {}

        # 分支A：窗口级 NILM 指标（测试/旧逻辑兼容）
        if preds is not None and isinstance(preds, tuple) and len(preds) >= 2:
            try:
                pred_power, pred_states = preds[0], preds[1]
                y_true_power = batch.get('target_power', None)
                y_true_states = batch.get('target_states', None)
                if y_true_power is None or y_true_states is None:
                    metrics.setdefault('score', float('nan'))
                    return metrics
                from src.utils.metrics import NILMMetrics
                nilm = NILMMetrics(self.device_names)
                nilm_results = nilm.compute_all_metrics(
                    pred_power, pred_states, y_true_power, y_true_states,
                    optimize_thresholds=False,
                    classification_enabled=bool(getattr(self, 'classification_enabled', False))
                )
                metrics.update(nilm_results)
                # 安全记录一个通用分数
                if 'score' in metrics:
                    self._safe_log(f'{stage}/score', metrics['score'], on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
                return metrics
            except Exception:
                # 回退到序列指标
                pass

        # 分支B：序列级指标（默认）
        target_seq = batch.get('target_seq', None)
        if target_seq is None:
            metrics.setdefault('score', float('nan'))
            return metrics
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
            # MAE/RMSE（按 B、T、K 汇总，使用有效掩码）
            valid = torch.isfinite(pred_seq) & torch.isfinite(target_seq)
            vm = batch.get('target_seq_valid_mask', None)
            try:
                if isinstance(vm, torch.Tensor):
                    if vm.dim() == pred_seq.dim():
                        valid = valid & (vm > 0)
                    elif vm.dim() + 1 == pred_seq.dim():
                        valid = valid & (vm.unsqueeze(-1) > 0)
            except Exception:
                pass

            element_mae = torch.abs(pred_seq - target_seq)
            element_mse = (pred_seq - target_seq) ** 2
            denom = valid.float().sum().clamp_min(1.0)
            element_mae_masked = torch.where(valid, element_mae, torch.zeros_like(element_mae))
            element_mse_masked = torch.where(valid, element_mse, torch.zeros_like(element_mse))
            seq_mae = element_mae_masked.sum() / denom
            seq_rmse = torch.sqrt(element_mse_masked.sum() / denom)
            seq_mae = float(seq_mae.item())
            seq_rmse = float(seq_rmse.item())

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
                    # 替换无效值以保证DTW稳健
                    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
                    t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
                    p = _downsample_np(p)
                    t = _downsample_np(t)
                    dtw_vals.append(_dtw_l1(p, t))
            seq_dtw = float(np.mean(dtw_vals)) if dtw_vals else float('nan')

            # 记录与返回
            self._safe_log(f'{stage}/metrics/sequence/mae', seq_mae, on_epoch=True, sync_dist=True)
            self._safe_log(f'{stage}/metrics/sequence/rmse', seq_rmse, on_epoch=True, sync_dist=True)
            self._safe_log(f'{stage}/metrics/sequence/dtw', seq_dtw, on_epoch=True, sync_dist=True)
            metrics.update({'seq_mae': seq_mae, 'seq_rmse': seq_rmse, 'seq_dtw': seq_dtw})

            # 通用 score（以 -MAE 表示，值越高越好）
            if np.isfinite(seq_mae):
                metrics['score'] = -float(seq_mae)
                self._safe_log(
                    f'{stage}/score', metrics['score'],
                    on_epoch=True, on_step=False,
                    prog_bar=True, sync_dist=True
                )
            else:
                metrics['score'] = float('nan')
                self._safe_log(
                    f'{stage}/score', float('nan'),
                    on_epoch=True, on_step=False,
                    prog_bar=True, sync_dist=True
                )
        except Exception:
            # 若序列前向或计算失败，不阻断整体评估
            metrics.setdefault('score', float('nan'))

        return metrics
    
    def _collect_sequence_examples(self, pred_seq: torch.Tensor, target_seq: torch.Tensor, batch: Dict[str, torch.Tensor], stage: str = 'val') -> None:
        """收集用于可视化的序列样本，支持仅包含事件窗口，并加入主表原始波形。"""
        try:
            if not isinstance(pred_seq, torch.Tensor) or not isinstance(target_seq, torch.Tensor):
                return
            B = pred_seq.size(0)
            import numpy as np
            if self.plot_event_only and ('has_events' in batch):
                mask = batch['has_events'].detach().cpu().numpy().astype(bool)
            else:
                mask = np.ones(B, dtype=bool)
            for i in range(B):
                if len(self._sequence_examples) >= self.max_plots_per_epoch:
                    break
                if not mask[i]:
                    continue
                ts = batch.get('timestamps', None)
                # 构造时间戳，仅在长度与序列长度一致时保留，否则置为 None
                timestamps = None
                try:
                    if isinstance(ts, torch.Tensor):
                        ts_i = ts[i].detach().cpu().to(torch.float64)
                        if ts_i.dim() == 0:
                            ts_i = ts_i.view(1)
                        L_i = pred_seq[i].size(0)
                        if int(ts_i.numel()) == int(L_i):
                            timestamps = ts_i
                except Exception:
                    timestamps = None
                # 清理 NaN/Inf，保证绘图可见
                p = pred_seq[i].detach().cpu().to(torch.float32)
                t = target_seq[i].detach().cpu().to(torch.float32)
                p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
                t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
                # 主表原始波形（来自 datamodule 提供的 mains_seq）
                m = None
                try:
                    ms = batch.get('mains_seq', None)
                    if isinstance(ms, torch.Tensor):
                        mi = ms[i].detach().cpu().to(torch.float32)
                        if mi.dim() == 1 and int(mi.numel()) == int(p.size(0)):
                            m = torch.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0)
                except Exception:
                    m = None
                example = {
                    'pred': p,
                    'target': t,
                    'timestamps': timestamps,
                    'mains': m,
                }
                self._sequence_examples.append(example)
        except Exception:
            pass

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """训练步骤（混合式：序列回归为主 + 原型度量为辅）。"""
        # 数据验证和NaN检测
        self._validate_batch_data(batch, 'train', batch_idx)

        # 序列前向（主任务）
        seq_out = self.model.forward_seq(
            batch.get('time_features'),
            batch.get('freq_features'),
            batch.get('time_positional'),
            batch.get('aux_features'),
            time_valid_mask=batch.get('time_valid_mask'),
            freq_valid_mask=batch.get('freq_valid_mask')
        )
        pred_seq = seq_out[0] if isinstance(seq_out, tuple) else seq_out
        target_seq = batch.get('target_seq', None)
        if target_seq is None:
            # 缺少监督时，返回0损失以避免中断（同时记录）
            try:
                dev = pred_seq.device
            except Exception:
                dev = torch.device('cpu')
            self._safe_log('train/loss/sequence', float('nan'), on_step=True, on_epoch=True, prog_bar=True)
            return torch.tensor(0.0, device=dev)

        # 计算序列平滑L1损失，屏蔽无效元素（含标签有效掩码）
        valid = torch.isfinite(pred_seq) & torch.isfinite(target_seq)
        vm = batch.get('target_seq_valid_mask', None)
        try:
            if isinstance(vm, torch.Tensor):
                if vm.dim() == pred_seq.dim():
                    valid = valid & (vm > 0)
                elif vm.dim() + 1 == pred_seq.dim():
                    valid = valid & (vm.unsqueeze(-1) > 0)
        except Exception:
            pass
        element_loss = torch.nn.functional.smooth_l1_loss(pred_seq, target_seq, reduction='none')
        element_loss = torch.where(valid, element_loss, torch.zeros_like(element_loss))
        denom = valid.float().sum().clamp_min(1.0)
        seq_loss = element_loss.sum() / denom
        # 逐设备序列损失（按epoch记录）
        try:
            per_dev_denom = valid.float().sum(dim=(0,1)).clamp_min(1.0)
            per_dev_sum = element_loss.sum(dim=(0,1))
            per_dev_loss = per_dev_sum / per_dev_denom
            for i, d_loss in enumerate(per_dev_loss):
                name = self.device_names[i] if i < len(self.device_names) else f'device_{i+1}'
                self._safe_log(f'train/loss/sequence/device/{name}', d_loss, on_step=False, on_epoch=True)
        except Exception:
            pass

        total_loss = seq_loss

        # 辅助损失：原型度量（可选）
        if getattr(self, 'metric_learning_enable', False):
            try:
                # 获取设备嵌入（注意：会再次执行编码与融合，但改动最小，后续可合并）
                reg, cls, unk, pred_embeddings = self.forward_with_embeddings(batch)
                aux_loss = self._compute_aux_metric_loss(batch, pred_embeddings)
                self._safe_log('train/loss/metric', aux_loss, on_step=True, on_epoch=True, prog_bar=False)
                # 在线更新原型库并记录距离摘要
                self._update_prototypes_and_log(pred_embeddings, batch.get('target_states', None))
                # 总损失融合
                total_loss = total_loss + self.metric_weight * aux_loss
            except Exception as e:
                print(f"metric learning skipped at step {batch_idx}: {e}")

        losses = {'seq_regression': seq_loss, 'total': total_loss}
        self._validate_losses(losses, 'train', batch_idx)

        # 记录学习率（按 epoch），在未附加 Trainer 的单元测试场景下跳过
        if getattr(self, '_trainer', None) is not None and getattr(self._trainer, 'optimizers', None):
            self._safe_log('train/metrics/optimization/lr', self._trainer.optimizers[0].param_groups[0]['lr'], on_step=False, on_epoch=True)

        # 记录损失
        self._safe_log('train/loss/sequence', seq_loss, on_step=True, on_epoch=True, prog_bar=True)
        if getattr(self, 'metric_learning_enable', False):
            self._safe_log('train/loss/total', total_loss, on_step=True, on_epoch=True, prog_bar=False)

        # 每隔一定步数记录训练指标与梯度范数
        if batch_idx % self.config.training.log_every_n_steps == 0:
            with torch.no_grad():
                _ = self._compute_metrics(batch, stage='train')
                if self.config.debug.track_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=float('inf'))
                    self._safe_log('train/metrics/optimization/grad_norm', grad_norm, on_step=False, on_epoch=True)
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

        # 归一化原型与预测嵌入，确保余弦相似度在[-1,1]
        pred = torch.nn.functional.normalize(pred_embeddings, dim=-1)
        prototypes = torch.nn.functional.normalize(prototypes, dim=-1)
        pos_sim = torch.clamp((pred * prototypes).sum(dim=-1), min=-1.0, max=1.0)  # (B, N)

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
        """验证步骤（纯 seq2seq）：仅进行序列前向、序列损失与序列指标。"""
        # 数据验证和NaN检测
        self._validate_batch_data(batch, 'val', batch_idx)

        # 序列前向
        seq_out = self.model.forward_seq(
            batch.get('time_features'),
            batch.get('freq_features'),
            batch.get('time_positional'),
            batch.get('aux_features'),
            time_valid_mask=batch.get('time_valid_mask'),
            freq_valid_mask=batch.get('freq_valid_mask')
        )
        pred_seq = seq_out[0] if isinstance(seq_out, tuple) else seq_out
        target_seq = batch.get('target_seq', None)

        # 计算序列损失（若提供标签），应用标签有效掩码
        if target_seq is not None:
            valid = torch.isfinite(pred_seq) & torch.isfinite(target_seq)
            vm = batch.get('target_seq_valid_mask', None)
            try:
                if isinstance(vm, torch.Tensor):
                    if vm.dim() == pred_seq.dim():
                        valid = valid & (vm > 0)
                    elif vm.dim() + 1 == pred_seq.dim():
                        valid = valid & (vm.unsqueeze(-1) > 0)
            except Exception:
                pass
            element_loss = torch.nn.functional.smooth_l1_loss(pred_seq, target_seq, reduction='none')
            element_loss = torch.where(valid, element_loss, torch.zeros_like(element_loss))
            denom = valid.float().sum().clamp_min(1.0)
            seq_loss = element_loss.sum() / denom
            # 逐设备序列损失（按epoch记录）
            try:
                per_dev_denom = valid.float().sum(dim=(0,1)).clamp_min(1.0)
                per_dev_sum = element_loss.sum(dim=(0,1))
                per_dev_loss = per_dev_sum / per_dev_denom
                for i, d_loss in enumerate(per_dev_loss):
                    name = self.device_names[i] if i < len(self.device_names) else f'device_{i+1}'
                    self._safe_log(f'test/loss/sequence/device/{name}', d_loss, on_step=False, on_epoch=True)
            except Exception:
                pass
            # 逐设备序列损失（按epoch记录）
            try:
                per_dev_denom = valid.float().sum(dim=(0,1)).clamp_min(1.0)
                per_dev_sum = element_loss.sum(dim=(0,1))
                per_dev_loss = per_dev_sum / per_dev_denom
                for i, d_loss in enumerate(per_dev_loss):
                    name = self.device_names[i] if i < len(self.device_names) else f'device_{i+1}'
                    self._safe_log(f'val/loss/sequence/device/{name}', d_loss, on_step=False, on_epoch=True)
            except Exception:
                pass
            self._safe_log('val/loss/sequence', seq_loss, on_step=False, on_epoch=True)
        else:
            seq_loss = torch.tensor(0.0, device=pred_seq.device)

        # 指标与可视化样本收集
        try:
            if target_seq is not None:
                self._collect_sequence_examples(pred_seq, target_seq, batch, stage='val')
        except Exception:
            pass
        metrics = self._compute_metrics(batch, stage='val')

        # 明确记录验证损失与分数（按 epoch）
        self._safe_log('val/loss', seq_loss, on_step=False, on_epoch=True, prog_bar=True)
        self._safe_log('val/score', metrics['score'], on_step=False, on_epoch=True, prog_bar=True)

        return {
            'val_loss': seq_loss,
            'val_score': metrics['score'],
            'predictions': pred_seq,
            'targets': batch.get('target_seq', None)
        }
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """测试步骤（纯 seq2seq）：仅进行序列前向、序列损失与序列指标。"""
        # 数据验证和NaN检测
        self._validate_batch_data(batch, 'test', batch_idx)

        # 序列前向
        seq_out = self.model.forward_seq(
            batch.get('time_features'),
            batch.get('freq_features'),
            batch.get('time_positional'),
            batch.get('aux_features'),
            time_valid_mask=batch.get('time_valid_mask'),
            freq_valid_mask=batch.get('freq_valid_mask')
        )
        pred_seq = seq_out[0] if isinstance(seq_out, tuple) else seq_out
        target_seq = batch.get('target_seq', None)

        # 计算序列损失（若提供标签），应用标签有效掩码
        if target_seq is not None:
            valid = torch.isfinite(pred_seq) & torch.isfinite(target_seq)
            vm = batch.get('target_seq_valid_mask', None)
            try:
                if isinstance(vm, torch.Tensor):
                    if vm.dim() == pred_seq.dim():
                        valid = valid & (vm > 0)
                    elif vm.dim() + 1 == pred_seq.dim():
                        valid = valid & (vm.unsqueeze(-1) > 0)
            except Exception:
                pass
            element_loss = torch.nn.functional.smooth_l1_loss(pred_seq, target_seq, reduction='none')
            element_loss = torch.where(valid, element_loss, torch.zeros_like(element_loss))
            denom = valid.float().sum().clamp_min(1.0)
            seq_loss = element_loss.sum() / denom
            # 逐设备序列损失（按epoch记录）
            try:
                per_dev_denom = valid.float().sum(dim=(0,1)).clamp_min(1.0)
                per_dev_sum = element_loss.sum(dim=(0,1))
                per_dev_loss = per_dev_sum / per_dev_denom
                for i, d_loss in enumerate(per_dev_loss):
                    name = self.device_names[i] if i < len(self.device_names) else f'device_{i+1}'
                    self._safe_log(f'test/loss/sequence/device/{name}', d_loss, on_step=False, on_epoch=True)
            except Exception:
                pass
        else:
            seq_loss = torch.tensor(0.0, device=pred_seq.device)

        metrics = self._compute_metrics(batch, stage='test')

        # 显式按 epoch 记录测试损失与分数
        self._safe_log('test/loss/total', seq_loss, on_step=False, on_epoch=True)
        self._safe_log('test/loss', seq_loss, on_step=False, on_epoch=True)
        self._safe_log('test/metrics/score', metrics['score'], on_step=False, on_epoch=True)
        self._safe_log('test/score', metrics['score'], on_step=False, on_epoch=True)

        return {
            'test_loss': seq_loss,
            'test_score': metrics['score'],
            'predictions': pred_seq,
            'targets': batch.get('target_seq', None)
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
        # 若未启用可视化，仅跳过可视化，不影响指标更新
        # 序列对比可视化：绘制预测 vs 标签，并叠加主表原始波形
        try:
            if getattr(self, 'logger', None) is not None and len(self._sequence_examples) > 0:
                import matplotlib.pyplot as plt
                import numpy as np
                from datetime import datetime
                import matplotlib.dates as mdates
                for idx, ex in enumerate(self._sequence_examples[:self.max_plots_per_epoch]):
                    pred = ex['pred']
                    target = ex['target']
                    L = pred.size(0)
                    K = pred.size(1) if pred.dim() == 2 else 1
                    fig = plt.figure(figsize=(12, 3 + 1.5*K))
                    for k in range(K):
                        ax = fig.add_subplot(K, 1, k+1)
                        ts = ex.get('timestamps', None)
                        if ts is not None and hasattr(ts, 'numel') and int(ts.numel()) == int(L):
                            ts_np = ts.view(-1).detach().cpu().numpy()
                            x = [datetime.fromtimestamp(float(v)) for v in ts_np]
                            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
                            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                        else:
                            x = np.arange(L)
                        y_pred = pred[:, k].numpy() if K > 1 else (pred[:, 0].numpy() if pred.dim() == 2 else pred.numpy())
                        y_true = target[:, k].numpy() if K > 1 else (target[:, 0].numpy() if target.dim() == 2 else target.numpy())
                        mains = ex.get('mains', None)
                        if mains is not None and int(mains.numel()) == int(L):
                            ax.plot(x, mains.numpy(), label='mains', color='tab:orange', linewidth=1.0, alpha=0.6)
                        ax.plot(x, y_true, label='label', color='black', linewidth=1.2)
                        ax.plot(x, y_pred, label='pred', color='tab:blue', linewidth=1.2, alpha=0.8)
                        ax.set_title(f'Device {k}')
                        ax.legend(loc='upper right')
                    fig.autofmt_xdate()
                    self.logger.experiment.add_figure(f'val/visualization/sequence/sample_{idx}', fig, self.current_epoch)
                    plt.close(fig)
                self._sequence_examples.clear()
        except Exception as e:
            print(f"sequence visualization failed: {e}")
        # 获取当前验证监控值（根据配置兼容不同键）
        ckpt_monitor = getattr(self.config.training.checkpoint, 'monitor', 'val/score')
        if ckpt_monitor == 'val_score':
            ckpt_monitor = 'val/score'
        current_val = self.trainer.callback_metrics.get(ckpt_monitor, None)
        if current_val is None:
            # 回退（优先val/loss）
            current_val = self.trainer.callback_metrics.get('val/loss', None)
        if current_val is None:
            current_val = self.trainer.callback_metrics.get('val/score', 0.0)
        # 将 Tensor 转为 float，避免后续配置或日志处理中出现 OmegaConf 类型不支持
        try:
            if hasattr(current_val, 'item'):
                current_val = float(current_val.item())
            else:
                current_val = float(current_val)
        except Exception:
            current_val = float('nan')
        
        # 根据模式更新最佳值（支持min/max）
        mode = str(getattr(self.config.training.checkpoint, 'mode', 'max')).lower()
        if np.isfinite(current_val):
            if mode == 'min':
                is_better = current_val < float(self.best_val_score)
            else:
                is_better = current_val > float(self.best_val_score)
            if is_better:
                self.best_val_score = float(current_val)
                # 记录最佳验证分数（层级命名，在无 Trainer 时安全跳过）
                self._safe_log('val/best_score', self.best_val_score, on_epoch=True)
        
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
            # 让调度器监控与回调一致的指标与模式
            es_cfg = getattr(self.config.training, 'early_stopping', None)
            ckpt_cfg = getattr(self.config.training, 'checkpoint', None)
            monitor_key = str(getattr(es_cfg, 'monitor', None) or getattr(ckpt_cfg, 'monitor', 'val/loss'))
            monitor_key = 'val/score' if monitor_key == 'val_score' else monitor_key
            monitor_mode = str(getattr(es_cfg, 'mode', None) or getattr(ckpt_cfg, 'mode', 'min')).lower()
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=monitor_mode,
                factor=max(getattr(scheduler_config, 'factor', 0.3), 0.3),  # 更大的衰减因子
                patience=min(getattr(scheduler_config, 'patience', 3), 3),  # 更短的耐心
                min_lr=1e-7  # 最小学习率
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': monitor_key,
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
                if key == 'target_seq' and isinstance(batch.get('target_seq_valid_mask'), torch.Tensor):
                    masked = True
                
                if self.config.debug.strict_validation:
                    raise ValueError(f"发现无效数据在 {stage} batch {batch_idx}, key '{key}': "
                                   f"NaN: {nan_count}, Inf: {inf_count}")
                else:
                    if masked and suppress_invalid:
                        # 跳过打印，避免训练日志噪声
                        continue
                    print(f"警告: {stage} batch {batch_idx} 中 '{key}' 包含无效值: NaN: {nan_count}, Inf: {inf_count}")
    
    def _validate_predictions(self, pred_seq: torch.Tensor, 
                            stage: str, batch_idx: int) -> None:
        """验证序列预测的有效性，仅检查曲线值。"""
        if not torch.isfinite(pred_seq).all():
            nan_count = torch.isnan(pred_seq).sum().item()
            inf_count = torch.isinf(pred_seq).sum().item()
            self._safe_log(f'{stage}/pred_seq_nan_count', float(nan_count), on_step=True, on_epoch=False)
            self._safe_log(f'{stage}/pred_seq_inf_count', float(inf_count), on_step=True, on_epoch=False)
            if getattr(getattr(self.config, 'debug', None), 'strict_validation', False):
                raise ValueError(f"序列预测包含无效值在 {stage} batch {batch_idx}: NaN: {nan_count}, Inf: {inf_count}")
    
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
    
    # 精度配置 - 环境感知与兼容修正（GPU优先BF16，其次FP16；MPS使用FP16混合；CPU用FP32）
    requested_precision = getattr(getattr(config, 'training', None), 'precision', None)
    if accelerator == 'gpu':
        auto_precision = 'bf16-mixed' if torch.cuda.is_bf16_supported() else '16-mixed'
        precision = auto_precision if (requested_precision in (None, 'auto')) else requested_precision
    elif accelerator == 'mps':
        precision = '16-mixed' if (requested_precision in (None, 'auto', 'bf16-mixed')) else requested_precision
    else:
        precision = '32'
    
    # 创建训练器
    num_cuda_devices = torch.cuda.device_count() if accelerator == 'gpu' else 0
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
        sync_batchnorm=True if (accelerator == 'gpu' and num_cuda_devices > 1) else False
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
    优先使用配置中的 data.device_names；若未提供，则尝试从 Data/prepared/device_name_to_id.json 推断。
    """
    # 推断 prepared 数据目录
    try:
        prepared_dir = Path(getattr(config.paths, 'prepared_dir'))
    except Exception:
        prepared_dir = Path('Data/prepared')

    # 若配置显式提供了设备名称，则直接使用（用于测试/最小配置场景）
    try:
        cfg_names = list(getattr(getattr(config, 'data', {}), 'device_names', []) or [])
    except Exception:
        cfg_names = []
    if cfg_names:
        device_names: List[str] = cfg_names
    else:
        mapping_path = prepared_dir / 'device_name_to_id.json'
        device_names: List[str] = []
        # 若存在映射文件，按 id 顺序解析设备名称
        if mapping_path.exists():
            try:
                with open(mapping_path, 'r', encoding='utf-8') as f:
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
                print(f"[警告] 解析 {mapping_path} 失败，回退到默认设备名称：{e}")

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
    
    # 统一：从数据统计一次性注入 pos_weight（BCE）与 init_p（分类头偏置）
    try:
        pos_weight_vec = datamodule.get_pos_weight()
        prior_p_vec = datamodule.get_prior_p()

        if pos_weight_vec is not None:
            pos_list = pos_weight_vec.detach().cpu().tolist()
            for i, pw in enumerate(pos_list):
                if i in device_info:
                    device_info[i]['pos_weight'] = float(pw)
            # 启用 pos_weight
            if hasattr(config, 'loss') and hasattr(config.loss, 'classification') and hasattr(config.loss.classification, 'pos_weight'):
                setattr(config.loss.classification.pos_weight, 'enable', True)

        if prior_p_vec is not None:
            p_list = [float(p) for p in prior_p_vec.detach().cpu().tolist()]
            if hasattr(config, 'model') and hasattr(config.model, 'heads') and hasattr(config.model.heads, 'classification'):
                setattr(config.model.heads.classification, 'init_p', p_list)
    except Exception as e:
        print(f"统计先验注入失败: {e}")
    
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
    with open(summary_path, 'w', encoding='utf-8') as f:
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