"""训练脚本 (已优化：适配简化版回归模型)"""

import os
import sys
if sys.platform == 'darwin':
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


from .datamodule.datamodule import NILMDataModule
from .models.fusion_transformer import FusionTransformer
from .losses.losses import create_loss_function, RECOMMENDED_LOSS_CONFIGS
from .utils.metrics import NILMMetrics
from .utils.prototypes import PrototypeLibrary
from .utils.viz import save_validation_interactive_plot

# 启用TF32优化（放在所有导入之后，避免 E402）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")

class DeferredEarlyStopping(EarlyStopping):
    def __init__(self, start_epoch: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.start_epoch = int(start_epoch or 0)
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        try:
            if int(getattr(trainer, "current_epoch", 0)) < self.start_epoch:
                return
        except Exception:
            pass
        return super().on_validation_epoch_end(trainer, pl_module)
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        try:
            if int(getattr(trainer, "current_epoch", 0)) < self.start_epoch:
                return
        except Exception:
            pass
        return super().on_validation_end(trainer, pl_module)


class NILMLightningModule(pl.LightningModule):
    """NILM PyTorch Lightning模块 (简化版)"""

    def __init__(self, config: DictConfig, device_info: Dict, device_names: List[str]):
        super().__init__()

        self.config = config
        self.device_info = device_info
        self.device_names = device_names
        self.n_devices = len(device_names)
        
        # Task Mode: seq2seq vs seq2point
        self.task = str(getattr(config, 'task', 'seq2seq')).lower()
        print(f"[Info] Task Mode: {self.task}")

        # 保存超参数
        self.save_hyperparameters({
            'config': OmegaConf.to_container(config, resolve=True),
            'device_info': device_info,
            'device_names': device_names
        })

        # 初始化模型
        self.model = FusionTransformer(config.model, self.n_devices)

        # 初始化损失函数 (使用简化配置)
        loss_config = RECOMMENDED_LOSS_CONFIGS.get('balanced', {})
        if hasattr(config, 'loss') and config.loss:
            loss_config.update(OmegaConf.to_container(config.loss, resolve=True))
        self.loss_fn = create_loss_function(loss_config)

        # 彻底禁用分类
        self.classification_enabled = False

        # 评估器
        self.nilm_metrics = NILMMetrics(self.device_names, threshold_method='optimal')

        # 尺度策略：每设备尺度（训练集 P95），由 DataModule 提供
        self.power_scale: Optional[torch.Tensor] = None
        self.max_power: Optional[torch.Tensor] = None
        self.per_device_boost: Optional[torch.Tensor] = None

        # 可视化配置
        self.best_val_loss = float('inf')
        self.best_thresholds = {}
        vis_cfg = getattr(getattr(config, 'training', None), 'visualization', None)
        self.enable_visualization = bool(getattr(vis_cfg, 'enable', False) if vis_cfg is not None else False)
        self.max_plots_per_epoch = int(getattr(vis_cfg, 'max_plots_per_epoch', 8))
        self.vis_output_dir = str(getattr(vis_cfg, 'save_dir', Path('outputs') / 'viz'))
        self.enable_interactive = bool(getattr(vis_cfg, 'interactive', False) if vis_cfg is not None else False)
        self._val_buffers: List[Dict[str, Any]] = []
        # 度量学习（原型库）最小实现
        try:
            ml_cfg = getattr(getattr(config, 'aux_training', None), 'metric_learning', None)
            self.metric_learning_enable = bool(getattr(ml_cfg, 'enable', False)) if ml_cfg is not None else False
        except Exception:
            self.metric_learning_enable = False
        self.prototype_library = PrototypeLibrary(n_devices=self.n_devices, embed_dim=config.model.time_encoder.d_model) if self.metric_learning_enable else None
        # 记录基线损失权重（用于课程式调整）
        self._base_loss_weights = {
            'peak_focus_weight': float(getattr(getattr(config, 'loss', None), 'peak_focus_weight', getattr(self.loss_fn, 'peak_focus_weight', 0.0))),
            'edge_focus_weight': float(getattr(getattr(config, 'loss', None), 'edge_focus_weight', getattr(self.loss_fn, 'edge_focus_weight', 0.0))),
            'derivative_loss_weight': float(getattr(getattr(config, 'loss', None), 'derivative_loss_weight', getattr(self.loss_fn, 'derivative_loss_weight', 0.0))),
            'conservation_weight': float(getattr(getattr(config, 'loss', None), 'conservation_weight', getattr(self.loss_fn, 'conservation_weight', 0.0))),
            'off_penalty_weight': float(getattr(getattr(config, 'loss', None), 'off_penalty_weight', getattr(self.loss_fn, 'off_penalty_weight', 0.0))),
            'active_boost': float(getattr(getattr(config, 'loss', None), 'active_boost', getattr(self.loss_fn, 'active_boost', 2.0))),
        }

    def _safe_log(self, name: str, value: Any, **kwargs) -> None:
        """安全日志记录"""
        if getattr(self, '_trainer', None) is not None:
            try:
                self.log(name, value, **kwargs)
            except Exception:
                pass

    def _ensure_power_scale(self, device: torch.device, k: int) -> torch.Tensor:
        """获取形状 (1,1,K) 的尺度张量"""
        if (self.power_scale is None) or (self.power_scale.numel() != k):
            try:
                dm = getattr(self.trainer, 'datamodule', None)
                if dm is not None and hasattr(dm, 'power_scale_vec') and isinstance(dm.power_scale_vec, torch.Tensor):
                    self.power_scale = dm.power_scale_vec.detach().clone().float()
                else:
                    self.power_scale = torch.ones(k, dtype=torch.float32)
            except Exception:
                self.power_scale = torch.ones(k, dtype=torch.float32)
        scale = torch.clamp(self.power_scale, min=1.0).to(device)
        return scale.view(1, 1, -1)

    def _ensure_max_power(self, device: torch.device, k: int) -> torch.Tensor:
        if (self.max_power is None) or (self.max_power.numel() != k):
            try:
                dm = getattr(self.trainer, 'datamodule', None)
                if dm is not None and hasattr(dm, 'max_power_vec') and isinstance(dm.max_power_vec, torch.Tensor):
                    self.max_power = dm.max_power_vec.detach().clone().float()
                else:
                    self.max_power = torch.ones(k, dtype=torch.float32)
            except Exception:
                self.max_power = torch.ones(k, dtype=torch.float32)
        return torch.clamp(self.max_power, min=1.0).to(device).view(1, 1, -1)

    def on_fit_start(self) -> None:
        """训练开始时载入尺度"""
        try:
            dm = getattr(self.trainer, 'datamodule', None)
            if dm is not None and hasattr(dm, 'power_scale_vec'):
                self.power_scale = dm.power_scale_vec.detach().clone().float()
                print(f"[Info] Per-device P95 scales loaded: {self.power_scale.tolist()}")
            if dm is not None and hasattr(dm, 'max_power_vec'):
                self.max_power = dm.max_power_vec.detach().clone().float()
                print(f"[Info] Per-device max powers loaded: {self.max_power.tolist()}")
            # 自适应稀疏设备损失增强：基于数据的激活稀疏度，限幅且不依赖手动参数
            try:
                rb = getattr(dm, 'rarity_boost_vec', None)
                if isinstance(rb, torch.Tensor) and rb.numel() == self.n_devices:
                    base = float(getattr(self.loss_fn, 'active_boost', 2.0))
                    boost = torch.clamp(rb.detach().clone().float(), min=1.0, max=4.0) * base
                    self.per_device_boost = boost
                    print(f"[Info] Adaptive per-device boost: {self.per_device_boost.tolist()}")
            except Exception:
                pass
            try:
                K = int(self.n_devices)
                amp = torch.ones(K, dtype=torch.float32)
                evt = torch.ones(K, dtype=torch.float32)
                var = torch.ones(K, dtype=torch.float32)
                off_w = torch.ones(K, dtype=torch.float32)
                excl_w = torch.ones(K, dtype=torch.float32)
                self.loss_fn.per_device_amplitude_scale = amp
                self.loss_fn.per_device_event_scale = evt
                self.loss_fn.per_device_variance_scale = var
                self.loss_fn.per_device_off_scale = off_w
                self.loss_fn.exclusive_device_weight = excl_w
                print(f"[LossCfg] per_device_amplitude_scale={amp.tolist()} per_device_event_scale={evt.tolist()} per_device_variance_scale={var.tolist()} per_device_off_scale={off_w.tolist()} exclusive_device_weight={excl_w.tolist()}")
            except Exception as e:
                try:
                    print("[LossCfg] Failed to set per-device loss scales:", str(e))
                except Exception:
                    pass
        except Exception as e:
            print(f"[Warning] Failed to load scales: {e}")
        # 初始应用课程式权重
        try:
            self._apply_loss_weight_schedule(epoch=int(getattr(self, 'current_epoch', 0)))
        except Exception:
            pass

    def _linear_ramp(self, base: float, target: float, start_epoch: int, ramp_epochs: int, epoch: int) -> float:
        if ramp_epochs <= 0:
            return base if epoch < start_epoch else target
        if epoch < start_epoch:
            return base
        if epoch >= start_epoch + ramp_epochs:
            return target
        a = (epoch - start_epoch) / float(max(ramp_epochs, 1))
        return base + a * (target - base)

    def _apply_loss_weight_schedule(self, epoch: int) -> None:
        """按配置对损失权重进行线性ramp"""
        cfg = getattr(self, 'config', None)
        lc = getattr(cfg, 'loss', None)
        if lc is None:
            return
        # 读取目标权重与起始/时长
        peak_target = float(getattr(lc, 'peak_target_weight', getattr(self.loss_fn, 'peak_focus_weight', 0.0)))
        peak_start = int(getattr(lc, 'peak_ramp_start', 0))
        peak_epochs = int(getattr(lc, 'peak_ramp_epochs', 0))
        edge_target = float(getattr(lc, 'edge_target_weight', getattr(self.loss_fn, 'edge_focus_weight', 0.0)))
        edge_start = int(getattr(lc, 'edge_ramp_start', 0))
        edge_epochs = int(getattr(lc, 'edge_ramp_epochs', 0))
        der_target = float(getattr(lc, 'derivative_target_weight', getattr(self.loss_fn, 'derivative_loss_weight', 0.0)))
        der_start = int(getattr(lc, 'derivative_ramp_start', 0))
        der_epochs = int(getattr(lc, 'derivative_ramp_epochs', 0))
        cons_start = int(getattr(lc, 'cons_start_epoch', 0))
        cons_epochs = int(getattr(lc, 'cons_ramp_epochs', 0))
        cons_target = float(getattr(lc, 'conservation_weight', getattr(self.loss_fn, 'conservation_weight', 0.0)))
        off_start = int(getattr(lc, 'off_start_epoch', 0))
        off_epochs = int(getattr(lc, 'off_ramp_epochs', 0))
        off_target = float(getattr(lc, 'off_penalty_weight', getattr(self.loss_fn, 'off_penalty_weight', 0.0)))
        ab_start = int(getattr(lc, 'active_boost_start_epoch', 0))
        ab_epochs = int(getattr(lc, 'active_boost_ramp_epochs', 0))
        ab_target = float(getattr(lc, 'active_boost', getattr(self.loss_fn, 'active_boost', 2.0)))

        # 基线
        base = self._base_loss_weights
        # 计算ramp值
        self.loss_fn.peak_focus_weight = self._linear_ramp(base['peak_focus_weight'], peak_target, peak_start, peak_epochs, epoch)
        self.loss_fn.edge_focus_weight = self._linear_ramp(base['edge_focus_weight'], edge_target, edge_start, edge_epochs, epoch)
        self.loss_fn.derivative_loss_weight = self._linear_ramp(base['derivative_loss_weight'], der_target, der_start, der_epochs, epoch)
        self.loss_fn.conservation_weight = self._linear_ramp(base['conservation_weight'], cons_target, cons_start, cons_epochs, epoch)
        self.loss_fn.off_penalty_weight = self._linear_ramp(base['off_penalty_weight'], off_target, off_start, off_epochs, epoch)
        self.loss_fn.active_boost = self._linear_ramp(base['active_boost'], ab_target, ab_start, ab_epochs, epoch)
        # 日志
        self._safe_log('train/weights/peak', self.loss_fn.peak_focus_weight, on_epoch=True)
        self._safe_log('train/weights/edge', self.loss_fn.edge_focus_weight, on_epoch=True)
        self._safe_log('train/weights/derivative', self.loss_fn.derivative_loss_weight, on_epoch=True)
        self._safe_log('train/weights/conservation', self.loss_fn.conservation_weight, on_epoch=True)
        self._safe_log('train/weights/off_penalty', self.loss_fn.off_penalty_weight, on_epoch=True)
        self._safe_log('train/weights/active_boost', self.loss_fn.active_boost, on_epoch=True)

    def on_train_epoch_start(self) -> None:
        """每个epoch开始时应用课程式权重"""
        try:
            ep = int(getattr(self, 'current_epoch', 0))
            self._apply_loss_weight_schedule(epoch=ep)
        except Exception:
            pass

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播 (Inference)"""
        if self.task == 'seq2point':
            time_features = batch['time_features']
            if hasattr(self.model, 'forward_point_with_unknown'):
                pred_point, _ = self.model.forward_point_with_unknown(
                    time_features,
                    freq_features=batch.get('freq_features'),
                    time_positional=batch.get('time_positional'),
                    aux_features=batch.get('aux_features'),
                    time_valid_mask=batch.get('time_valid_mask'),
                    freq_valid_mask=batch.get('freq_valid_mask'),
                    aux_valid_mask=batch.get('aux_valid_mask'),
                    external_scale=None
                )
            else:
                pred_point = self.model.forward_point(
                    time_features,
                    freq_features=batch.get('freq_features'),
                    time_positional=batch.get('time_positional'),
                    aux_features=batch.get('aux_features'),
                    time_valid_mask=batch.get('time_valid_mask'),
                    freq_valid_mask=batch.get('freq_valid_mask'),
                    aux_valid_mask=batch.get('aux_valid_mask'),
                    external_scale=None
                )
            device = pred_point.device
            mp = self._ensure_max_power(device, self.n_devices).view(1, -1)
            k_common = min(pred_point.size(1), mp.size(1))
            pred_point = pred_point[:, :k_common]
            mp = mp[:, :k_common]
            pred_norm = torch.clamp(pred_point / mp, 0.0, 1.0)
            pred_power_watts = pred_norm * mp
            pred_states = torch.zeros_like(pred_power_watts)
            return pred_power_watts, pred_states

        time_features = batch['time_features']
        out = self.model.forward_seq(
            time_features,
            freq_features=batch.get('freq_features'),
            time_positional=batch.get('time_positional'),
            aux_features=batch.get('aux_features'),
            time_valid_mask=batch.get('time_valid_mask'),
            freq_valid_mask=batch.get('freq_valid_mask'),
            aux_valid_mask=batch.get('aux_valid_mask'),
            external_scale=None
        )

        pred_seq = out[0] if isinstance(out, tuple) else out
        per_dev_mask = batch.get('target_seq_per_device_valid_mask', None)
        if isinstance(per_dev_mask, torch.Tensor):
            m = per_dev_mask > 0
        else:
            m = batch.get('target_seq_valid_mask', None)
        device = pred_seq.device
        mp = self._ensure_max_power(device, self.n_devices)
        pred_seq_norm = torch.clamp(pred_seq / mp, 0.0, 1.0)
        if isinstance(m, torch.Tensor):
            mf = (m > 0).float()
            num = (pred_seq_norm * mf).sum(dim=1)
            den = mf.sum(dim=1).clamp_min(1.0)
            pred_power_norm = num / den
        else:
            pred_power_norm = pred_seq_norm.mean(dim=1)
        pred_power_watts = pred_power_norm * mp.view(1, -1)
        pred_states = torch.zeros_like(pred_power_watts)
        return pred_power_watts, pred_states

    def _forward_seq2point(self, batch: Dict[str, torch.Tensor], stage: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Seq2Point 模式的前向与损失计算"""
        tf = batch.get('time_features')
        device = tf.device
        if hasattr(self.model, 'forward_point_with_unknown'):
            pred_point, unknown_point = self.model.forward_point_with_unknown(
                tf,
                freq_features=batch.get('freq_features'),
                time_positional=batch.get('time_positional'),
                aux_features=batch.get('aux_features'),
                time_valid_mask=batch.get('time_valid_mask'),
                freq_valid_mask=batch.get('freq_valid_mask'),
                aux_valid_mask=batch.get('aux_valid_mask'),
                external_scale=None
            )
        else:
            pred_point = self.model.forward_point(
                tf,
                freq_features=batch.get('freq_features'),
                time_positional=batch.get('time_positional'),
                aux_features=batch.get('aux_features'),
                time_valid_mask=batch.get('time_valid_mask'),
                freq_valid_mask=batch.get('freq_valid_mask'),
                aux_valid_mask=batch.get('aux_valid_mask'),
                external_scale=None
            )
            unknown_point = None
        
        target_point = batch.get('target_point')
        if target_point is None:
            # Fallback (e.g. inference without labels)
            return pred_point, torch.tensor(0.0, device=device)
            
        K = min(pred_point.size(1), target_point.size(1), self.n_devices)
        pred_point = pred_point[:, :K]
        target_point = target_point[:, :K]
        pred_center_seq = pred_point.unsqueeze(1)
        target_center_seq = target_point.unsqueeze(1)

        status_seq = batch.get('status_seq')
        center_status = None
        if isinstance(status_seq, torch.Tensor) and status_seq.dim() == 3:
            Ls = int(status_seq.size(1))
            c_idx = Ls // 2
            center_status = status_seq[:, c_idx:c_idx + 1, :K]

        per_dev_mask = batch.get('target_seq_per_device_valid_mask')
        center_valid = None
        if isinstance(per_dev_mask, torch.Tensor) and per_dev_mask.dim() == 3:
            Lm = int(per_dev_mask.size(1))
            c_idx_m = Lm // 2
            center_valid = per_dev_mask[:, c_idx_m:c_idx_m + 1, :K]

        pscale = self._ensure_power_scale(device, self.n_devices)
        pscale_c = pscale[:, :, :K]

        loss_k = self.loss_fn.regression_seq_loss_per_device(
            pred_center_seq,
            target_center_seq,
            center_status,
            center_valid,
            pscale_c
        )
        loss_point = loss_k.mean()
        loss_unknown = torch.tensor(0.0, device=device)
        if isinstance(unknown_point, torch.Tensor) and getattr(self.loss_fn, 'unknown_weight', 0.0) > 0.0:
            mains_seq = batch.get('mains_seq')
            target_seq = batch.get('target_seq')
            if isinstance(mains_seq, torch.Tensor) and isinstance(target_seq, torch.Tensor) and target_seq.dim() == 3:
                Ls = int(target_seq.size(1))
                c_idx = Ls // 2
                mains_center = mains_seq[:, c_idx:c_idx + 1]
                pred_seq_win = pred_point[:, :K].unsqueeze(1)
                loss_unknown = self.loss_fn.unknown_residual_loss(
                    mains_seq=mains_center,
                    pred_seq=pred_seq_win,
                    unknown_win=unknown_point,
                    status_seq=None
                )
        total_loss = self.loss_fn.regression_weight * loss_point + self.loss_fn.unknown_weight * loss_unknown
        
        self._safe_log(f'{stage}/loss/total', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self._safe_log(f'{stage}/loss/point_regression', loss_point, on_step=False, on_epoch=True)
        
        return pred_point, total_loss

    def _forward_and_compute_loss(self, batch: Dict[str, torch.Tensor], stage: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算联合损失"""
        if self.task == 'seq2point':
            return self._forward_seq2point(batch, stage)
            
        tf = batch.get('time_features')
        device = tf.device
        mp = self._ensure_max_power(device, self.n_devices)

        out = self.model.forward_seq(
            tf,
            freq_features=batch.get('freq_features'),
            time_positional=batch.get('time_positional'),
            aux_features=batch.get('aux_features'),
            time_valid_mask=batch.get('time_valid_mask'),
            freq_valid_mask=batch.get('freq_valid_mask'),
            aux_valid_mask=batch.get('aux_valid_mask'),
            external_scale=None
        )

        if isinstance(out, tuple):
            pred_seq = out[0]
            reg_win_pred = out[1]
            cls_seq_pred = out[4] if len(out) >= 5 else None
            unknown_win_pred = out[3] if len(out) >= 4 else None
        else:
            pred_seq = out
            reg_win_pred = None
            cls_seq_pred = None
            unknown_win_pred = None

        target_seq = batch.get('target_seq')
        if target_seq is None:
            return pred_seq, torch.tensor(0.0, device=device)

        # 设备维对齐
        try:
            Kp = int(pred_seq.size(-1))
            Kt = int(target_seq.size(-1))
            if Kp != Kt:
                Kc = min(Kp, Kt)
                pred_seq = pred_seq[:, :, :Kc]
                target_seq = target_seq[:, :, :Kc]
                if isinstance(reg_win_pred, torch.Tensor) and int(reg_win_pred.size(-1)) != Kc:
                    reg_win_pred = reg_win_pred[:, :Kc]
                try:
                    if isinstance(out, tuple):
                        cls_seq_pred = out[4] if len(out) >= 5 else None
                    if isinstance(cls_seq_pred, torch.Tensor) and int(cls_seq_pred.size(-1)) != Kc:
                        cls_seq_pred = cls_seq_pred[:, :, :Kc]
                except Exception:
                    pass
        except Exception:
            pass

        per_dev_mask = batch.get('target_seq_per_device_valid_mask')
        if isinstance(per_dev_mask, torch.Tensor):
            valid = per_dev_mask > 0
        else:
            fallback = batch.get('target_seq_valid_mask')
            if isinstance(fallback, torch.Tensor) and fallback.dim() + 1 == pred_seq.dim():
                fallback = fallback.unsqueeze(-1)
            valid = (fallback > 0) if isinstance(fallback, torch.Tensor) else torch.ones_like(pred_seq, dtype=torch.bool)

        # 匹配 mp 设备维
        mp_c = mp[:, :, :pred_seq.size(-1)]
        # 融合分配门控（如可用）：seq_pred_alloc = gate * mains；按配置进行线性混合
        blend_alpha = float(getattr(getattr(self.config, 'loss', None), 'allocation_blend_alpha', 0.0) or 0.0)
        if isinstance(cls_seq_pred, torch.Tensor) and isinstance(batch.get('mains_seq'), torch.Tensor) and blend_alpha > 0.0:
            ms = batch['mains_seq']
            if ms.dim() == 1:
                ms = ms.view(1, -1)
            seq_pred_alloc = cls_seq_pred * ms.unsqueeze(-1)
            pred_seq_watts_raw = (1.0 - blend_alpha) * pred_seq + blend_alpha * seq_pred_alloc
        else:
            pred_seq_watts_raw = pred_seq
        pred_seq_norm = torch.clamp(pred_seq_watts_raw / mp_c, 0.0, 1.0)
        target_seq_norm = torch.clamp(target_seq / mp_c, 0.0, 1.0)
        # 每设备尺度（P95），用于相对阈值与归一化惩罚
        pscale = self._ensure_power_scale(device, self.n_devices)
        pscale_c = pscale[:, :, :pred_seq.size(-1)]
        loss_reg = self.loss_fn.regression_seq_loss(
            pred_seq_norm, target_seq_norm,
            status_seq=batch.get('status_seq'),
            valid_mask=valid,
            power_scale=pscale_c,
            per_device_boost=self.per_device_boost
        )

        # 辅助窗口级监督（回归头）：用目标序列的时间均值作为窗口目标
        loss_win = torch.tensor(0.0, device=device)
        if isinstance(reg_win_pred, torch.Tensor):
            m_float = valid.float()
            num = (target_seq_norm * m_float).sum(dim=1)
            den = m_float.sum(dim=1).clamp_min(1.0)
            win_target_norm = num / den
            reg_win_norm = reg_win_pred / mp_c.view(1, -1)
            hl = torch.nn.HuberLoss(reduction='mean', delta=self.loss_fn.huber_delta)
            loss_win = hl(reg_win_norm, win_target_norm)

        pred_seq_watts = pred_seq_norm * mp_c
        loss_cons = self.loss_fn.conservation_loss(
            batch.get('mains_seq'), pred_seq_watts,
            valid_mask=None
        )
        excl_pen = self.loss_fn.device_exclusive_penalty(pred_seq_norm, valid)
        sparse_pen = self.loss_fn.sparsity_gate_penalty(cls_seq_pred, valid)
        alloc_dist = self.loss_fn.allocation_distribution_loss(cls_seq_pred, target_seq_norm, valid)
        ev_pen = self.loss_fn.event_count_penalty(pred_seq_norm, batch.get('status_seq'), valid)
        amp_pen = self.loss_fn.active_amplitude_loss(pred_seq_norm, target_seq_norm, batch.get('status_seq'), valid)
        var_pen = self.loss_fn.shape_variance_loss(pred_seq_norm, target_seq_norm, valid)
        loss_unknown = torch.tensor(0.0, device=device)
        if isinstance(unknown_win_pred, torch.Tensor) and getattr(self.loss_fn, 'unknown_weight', 0.0) > 0.0:
            loss_unknown = self.loss_fn.unknown_residual_loss(
                mains_seq=batch.get('mains_seq'),
                pred_seq=pred_seq_watts,
                unknown_win=unknown_win_pred,
                status_seq=batch.get('status_seq')
            )
        total_loss = self.loss_fn.regression_weight * (loss_reg + loss_win) + loss_cons + excl_pen + sparse_pen + alloc_dist + ev_pen + amp_pen + var_pen + self.loss_fn.unknown_weight * loss_unknown

        self._safe_log(f'{stage}/loss/total', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self._safe_log(f'{stage}/loss/regression', loss_reg, on_step=False, on_epoch=True)
        self._safe_log(f'{stage}/loss/conservation', loss_cons, on_step=False, on_epoch=True)
        try:
            self._safe_log(f'{stage}/loss/exclusive', excl_pen, on_epoch=True)
            self._safe_log(f'{stage}/loss/sparsity', sparse_pen, on_epoch=True)
            self._safe_log(f'{stage}/loss/allocation_kl', alloc_dist, on_epoch=True)
            self._safe_log(f'{stage}/loss/event_count', ev_pen, on_epoch=True)
            self._safe_log(f'{stage}/loss/active_amplitude', amp_pen, on_epoch=True)
            self._safe_log(f'{stage}/loss/shape_variance', var_pen, on_epoch=True)
        except Exception:
            pass

        return pred_seq, total_loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        pred_seq, loss = self._forward_and_compute_loss(batch, 'train')
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        pred_seq, loss = self._forward_and_compute_loss(batch, 'val')
        self._compute_metrics(batch, (pred_seq, None), stage='val')
        if self.enable_interactive:
            try:
                ts_mat = batch.get('timestamps')
                target_seq = batch.get('target_seq')
                target_point = batch.get('target_point')
                mains_seq = batch.get('mains_seq')
                time_valid_mask = batch.get('time_valid_mask')
                if self.task == 'seq2point':
                    unknown_point = None
                    if hasattr(self.model, 'forward_point_with_unknown'):
                        try:
                            tf_viz = batch.get('time_features')
                            if tf_viz is not None:
                                _, unknown_point = self.model.forward_point_with_unknown(
                                    tf_viz,
                                    freq_features=batch.get('freq_features'),
                                    time_positional=batch.get('time_positional'),
                                    aux_features=batch.get('aux_features'),
                                    time_valid_mask=batch.get('time_valid_mask'),
                                    freq_valid_mask=batch.get('freq_valid_mask'),
                                    aux_valid_mask=batch.get('aux_valid_mask'),
                                    external_scale=None
                                )
                        except Exception:
                            unknown_point = None
                    if isinstance(pred_seq, torch.Tensor) and isinstance(target_point, torch.Tensor):
                        b = int(pred_seq.size(0))
                        for i in range(b):
                            device = pred_seq.device
                            mp3 = self._ensure_max_power(device, self.n_devices)
                            mp2 = mp3.view(1, -1)
                            p_point = pred_seq[i].view(1, -1)
                            p_norm = torch.clamp(p_point / mp2, 0.0, 1.0)
                            if isinstance(target_seq, torch.Tensor) and target_seq.dim() == 3:
                                L = int(target_seq.size(1))
                            elif isinstance(mains_seq, torch.Tensor) and mains_seq.dim() >= 1:
                                L = int(mains_seq.size(1))
                            else:
                                L = 1
                            p_seq = p_norm.expand(L, -1)
                            p_w = (p_seq * mp2).detach().cpu().numpy().astype('float32')
                            if isinstance(target_seq, torch.Tensor) and target_seq.dim() == 3:
                                try:
                                    tn = target_seq[i] / mp2
                                    t_w = (torch.clamp(tn, 0.0, 1.0) * mp2).detach().cpu().numpy().astype('float32')
                                except Exception:
                                    t_w = (torch.clamp(target_seq[i], 0.0, 1.0) * mp2).detach().cpu().numpy().astype('float32')
                            else:
                                t_point = target_point[i].view(1, -1)
                                t_norm = torch.clamp(t_point / mp2, 0.0, 1.0)
                                t_seq = t_norm.expand(L, -1)
                                t_w = (t_seq * mp2).detach().cpu().numpy().astype('float32')
                            m = None
                            try:
                                if isinstance(mains_seq, torch.Tensor):
                                    mm = mains_seq[i].detach().cpu()
                                    if mm.dim() > 1:
                                        mm = mm.squeeze()
                                    m = mm
                            except Exception:
                                m = None
                            if isinstance(ts_mat, torch.Tensor) and ts_mat.dim() == 2:
                                start_ts = float(ts_mat[i, 0].detach().cpu().item())
                                step = float(getattr(getattr(self.trainer, 'datamodule', None), 'resample_seconds', 5.0))
                            else:
                                start_ts = float(torch.tensor(0.0).item())
                                step = float(getattr(getattr(self.trainer, 'datamodule', None), 'resample_seconds', 5.0))
                            vmask = None
                            try:
                                if isinstance(time_valid_mask, torch.Tensor):
                                    vmask = time_valid_mask[i].detach().cpu()
                                    if vmask.dim() > 1:
                                        vmask = vmask.squeeze()
                            except Exception:
                                vmask = None
                            buf = {
                                'pred': torch.tensor(p_w),
                                'true': torch.tensor(t_w),
                                'mains': m,
                                'valid': vmask,
                                'start': start_ts,
                                'step': step,
                            }
                            if isinstance(unknown_point, torch.Tensor) and unknown_point.dim() >= 1 and i < int(unknown_point.size(0)):
                                try:
                                    u_scalar = float(unknown_point[i].view(-1)[0].detach().cpu().item())
                                    if L > 0:
                                        u_seq = torch.full((L,), u_scalar, dtype=torch.float32)
                                        buf['unknown'] = u_seq
                                except Exception:
                                    pass
                            self._val_buffers.append(buf)
                else:
                    unknown_win_pred = None
                    if getattr(self.model, 'include_unknown', False) and hasattr(self.model, 'forward_seq'):
                        try:
                            tf_viz = batch.get('time_features')
                            if tf_viz is not None:
                                out_viz = self.model.forward_seq(
                                    tf_viz,
                                    freq_features=batch.get('freq_features'),
                                    time_positional=batch.get('time_positional'),
                                    aux_features=batch.get('aux_features'),
                                    time_valid_mask=batch.get('time_valid_mask'),
                                    freq_valid_mask=batch.get('freq_valid_mask'),
                                    aux_valid_mask=batch.get('aux_valid_mask'),
                                    external_scale=None
                                )
                                if isinstance(out_viz, tuple) and len(out_viz) >= 4:
                                    unknown_win_pred = out_viz[3]
                        except Exception:
                            unknown_win_pred = None
                    if isinstance(pred_seq, torch.Tensor) and isinstance(target_seq, torch.Tensor):
                        b = int(pred_seq.size(0))
                        for i in range(b):
                            device = pred_seq.device
                            mp3 = self._ensure_max_power(device, self.n_devices)
                            mp2 = mp3.view(1, -1)
                            p_norm = torch.clamp(pred_seq[i] / mp2, 0.0, 1.0)
                            try:
                                tn = target_seq[i] / mp2
                                t_w = (torch.clamp(tn, 0.0, 1.0) * mp2).detach().cpu().numpy().astype('float32')
                            except Exception:
                                t_w = (torch.clamp(target_seq[i], 0.0, 1.0) * mp2).detach().cpu().numpy().astype('float32')
                            p_w = (p_norm * mp2).detach().cpu().numpy().astype('float32')
                            m = None
                            try:
                                if isinstance(mains_seq, torch.Tensor):
                                    mm = mains_seq[i].detach().cpu()
                                    if mm.dim() > 1:
                                        mm = mm.squeeze()
                                    m = mm
                            except Exception:
                                m = None
                            if isinstance(ts_mat, torch.Tensor) and ts_mat.dim() == 2:
                                start_ts = float(ts_mat[i, 0].detach().cpu().item())
                                step = float(getattr(getattr(self.trainer, 'datamodule', None), 'resample_seconds', 5.0))
                            else:
                                start_ts = float(torch.tensor(0.0).item())
                                step = float(getattr(getattr(self.trainer, 'datamodule', None), 'resample_seconds', 5.0))
                            vmask = None
                            try:
                                if isinstance(time_valid_mask, torch.Tensor):
                                    vmask = time_valid_mask[i].detach().cpu()
                                    if vmask.dim() > 1:
                                        vmask = vmask.squeeze()
                            except Exception:
                                vmask = None
                            buf = {
                                'pred': torch.tensor(p_w),
                                'true': torch.tensor(t_w),
                                'mains': m,
                                'valid': vmask,
                                'start': start_ts,
                                'step': step,
                            }
                            if isinstance(unknown_win_pred, torch.Tensor) and unknown_win_pred.dim() >= 1 and i < int(unknown_win_pred.size(0)):
                                try:
                                    u_scalar = float(unknown_win_pred[i].view(-1)[0].detach().cpu().item())
                                    L = int(p_w.shape[0])
                                    if L > 0:
                                        u_seq = torch.full((L,), u_scalar, dtype=torch.float32)
                                        buf['unknown'] = u_seq
                                except Exception:
                                    pass
                            self._val_buffers.append(buf)
            except Exception:
                pass
        return {'val_loss': loss}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        pred_seq, loss = self._forward_and_compute_loss(batch, 'test')
        self._compute_metrics(batch, (pred_seq, None), stage='test')
        return {'test_loss': loss}

    def on_validation_epoch_end(self) -> None:
        if not self.enable_interactive:
            return
        bufs = list(self._val_buffers)
        self._val_buffers.clear()
        if len(bufs) == 0:
            return
        _tr = getattr(self, '_trainer', None)
        dm = getattr(_tr, 'datamodule', None)
        dataset_name = str(getattr(self.config, 'dataset', '') or '')
        fold_id = int(getattr(dm, 'fold_id', 0)) if dm is not None else 0
        ep = int(getattr(self, 'current_epoch', 0))
        try:
            mp = self._ensure_max_power(torch.device('cpu'), self.n_devices).view(-1).detach().cpu().numpy()
            fp = save_validation_interactive_plot(
                buffers=bufs,
                device_names=self.device_names,
                vis_output_dir=self.vis_output_dir,
                dataset_name=dataset_name,
                fold_id=fold_id,
                epoch=ep,
                max_power=mp,
            )
            try:
                import numpy as _np
                total_len = int(_np.sum([int(x['pred'].size(0)) for x in bufs]))
                K = int(bufs[0]['pred'].size(-1))
                print(f"[Viz] 保存交互式验证图: {str(fp)} (len={total_len}, devices={K})")
            except Exception:
                pass
        except Exception as e:
            try:
                print("[Viz] 交互式验证图保存失败:", str(e))
            except Exception:
                pass

    def _compute_metrics(self, batch: Dict[str, torch.Tensor], preds: Tuple, stage: str) -> Dict:
        """计算指标 (MAE, NDE, SAE)"""
        try:
            pred0 = preds[0]
            device = pred0.device
            mp = self._ensure_max_power(device, self.n_devices)

            if self.task == 'seq2point':
                y_true_point = batch.get('target_point')
                if y_true_point is None:
                    return {}
                if pred0.dim() == 1:
                    pred0 = pred0.view(1, -1)
                if y_true_point.dim() == 1:
                    y_true_point = y_true_point.view(1, -1)
                K_common = min(pred0.size(1), y_true_point.size(1))
                pred_power_watts = pred0[:, :K_common]
                y_true = y_true_point[:, :K_common]
            else:
                if pred0.dim() == 3:
                    per_dev_mask = batch.get('target_seq_per_device_valid_mask')
                    if isinstance(per_dev_mask, torch.Tensor):
                        valid = per_dev_mask > 0
                    else:
                        valid = batch.get('target_seq_valid_mask')
                        if isinstance(valid, torch.Tensor) and valid.dim() + 1 == pred0.dim():
                            valid = valid.unsqueeze(-1)
                    valid_f = (valid > 0).float() if isinstance(valid, torch.Tensor) else None
                    pred_seq_norm = torch.clamp(pred0 / mp, 0.0, 1.0)
                    if isinstance(valid_f, torch.Tensor):
                        num = (pred_seq_norm * valid_f).sum(dim=1)
                        den = valid_f.sum(dim=1).clamp_min(1.0)
                        pred_power_norm = num / den
                    else:
                        pred_power_norm = pred_seq_norm.mean(dim=1)
                    pred_power_watts = pred_power_norm * mp.view(1, -1)
                else:
                    pred_power_watts = pred0
                y_true = batch.get('target_power')
                if y_true is None:
                    return {}

                K_common = min(pred_power_watts.size(1), y_true.size(1))
                pred_power_watts = pred_power_watts[:, :K_common]
                y_true = y_true[:, :K_common]

            # 仅回归指标
            dummy_cls_pred = torch.zeros_like(y_true)
            dummy_cls_true = torch.zeros_like(y_true)
            res = self.nilm_metrics.compute_all_metrics(
                y_pred_power=pred_power_watts,
                y_pred_proba=dummy_cls_pred,
                y_true_power=y_true,
                y_true_states=dummy_cls_true,
                optimize_thresholds=False,
                classification_enabled=False,
                sample_weights=None,
            )
            ov = res.get('overall', {}) if isinstance(res, dict) else {}
            for mk in ('mae', 'nde', 'sae', 'teca', 'score'):
                if mk in ov:
                    self._safe_log(f'{stage}/metrics/{mk}', ov[mk], on_epoch=True)

            return ov
        except Exception:
            return {}

    def configure_optimizers(self):
        """配置优化器"""
        opt_cfg = getattr(self.config.training, 'optimizer', None)
        lr = float(getattr(opt_cfg, 'lr', 1e-4)) if opt_cfg is not None else 1e-4
        wd = float(getattr(opt_cfg, 'weight_decay', 0.0)) if opt_cfg is not None else 0.0
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        sch_cfg = getattr(self.config.training, 'scheduler', None)
        name = str(getattr(sch_cfg, 'name', 'plateau')).lower() if sch_cfg is not None else 'plateau'
        if name == 'cosine':
            T_max = int(getattr(sch_cfg, 'T_max', getattr(self.config.training, 'max_epochs', 25)))
            eta_min = float(getattr(sch_cfg, 'eta_min', 1e-6))
            warmup_steps = int(getattr(sch_cfg, 'warmup_steps', 0))
            sched_main = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
            if warmup_steps > 0:
                sched_warm = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, total_iters=warmup_steps)
                scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[sched_warm, sched_main], milestones=[warmup_steps])
            else:
                scheduler = sched_main
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss/total",
                    "interval": "epoch"
                },
            }

    def forward_with_embeddings(self, batch: Dict[str, torch.Tensor]):
        tf = batch.get('time_features')
        out = self.model.forward_with_embeddings(
            time_features=tf,
            freq_features=batch.get('freq_features'),
            time_positional=batch.get('time_positional'),
            aux_features=batch.get('aux_features'),
            time_valid_mask=batch.get('time_valid_mask'),
            freq_valid_mask=batch.get('freq_valid_mask'),
            aux_valid_mask=batch.get('aux_valid_mask'),
        )
        reg, cls, unk, emb = out
        if cls is None:
            B = tf.size(0)
            cls = torch.zeros(B, self.n_devices, device=tf.device, dtype=reg.dtype)
        return reg, cls, unk, emb

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        try:
            if self.metric_learning_enable and self.prototype_library is not None:
                checkpoint['prototype_library_state'] = self.prototype_library.state_dict()
        except Exception:
            pass

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        try:
            state = checkpoint.get('prototype_library_state', None)
            if self.metric_learning_enable and self.prototype_library is not None and state is not None:
                self.prototype_library.load_state_dict(state)
        except Exception:
            pass


# ------------------------------------------------------------------------
# Main Helper Functions (Keep largely same but clean)
# ------------------------------------------------------------------------

def _load_device_names_from_mapping_for_config(config: DictConfig) -> Optional[List[str]]:
    try:
        cfg_paths = getattr(config, "paths", None)
        prepared = getattr(cfg_paths, "prepared_dir", None) if cfg_paths is not None else None
        dataset_name = str(getattr(config, "dataset", "") or "").strip().lower()
        if prepared:
            base = Path(prepared)
        else:
            base = Path("Data/prepared")
        if dataset_name:
            parts_lower = [p.lower() for p in base.parts]
            if dataset_name not in parts_lower:
                base = base / dataset_name
        candidates = [base, base.parent, base.parent.parent]
        mapping = None
        for d in candidates:
            if d is None:
                continue
            fp = Path(d) / "device_name_to_id.json"
            if fp.exists():
                with open(fp, "r", encoding="utf-8") as f:
                    mapping = json.load(f)
                break
        if not mapping:
            return None
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
            pairs = sorted(((int(k), v) for k, v in mapping.items()), key=lambda kv: kv[0])
            names = [name for _, name in pairs]
        elif not key_is_int_like and val_is_int_like:
            pairs = sorted(((v, k) for k, v in mapping.items()), key=lambda kv: kv[0])
            names = [name for _, name in pairs]
        else:
            names = sorted(list(mapping.keys()))
        return [str(n) for n in names]
    except Exception:
        return None


def load_device_info(config: DictConfig) -> Tuple[Dict, List[str]]:
    names_from_mapping = _load_device_names_from_mapping_for_config(config)
    if isinstance(names_from_mapping, list) and len(names_from_mapping) > 0:
        names = names_from_mapping
    else:
        try:
            names = list(config.data.device_names)
        except Exception:
            names = ["device_1"]

    info = {}
    for i, n in enumerate(names):
        info[i] = {'name': n, 'max_power': 3000.0}
    return info, names


def main(config: DictConfig) -> None:
    """主入口"""
    pl.seed_everything(config.get('seed', 42))

    datamodule = NILMDataModule(config)
    try:
        datamodule.setup('fit')
    except Exception:
        pass

    # 优先使用数据集真实设备映射名称
    try:
        if hasattr(datamodule, 'device_names') and datamodule.device_names:
            device_names = list(datamodule.device_names)
        else:
            _info, device_names = load_device_info(config)
    except Exception:
        _info, device_names = load_device_info(config)
    device_info, _ = load_device_info(config)

    task_name = str(getattr(config, 'task', 'seq2seq'))
    logger = TensorBoardLogger("outputs", name=f"nilm_experiment_{task_name}")

    model = NILMLightningModule(config, device_info, device_names)

    import platform
    if platform.system() == "Darwin":
        current_precision = str(getattr(config.training, "precision", ""))
        if ("bf16" in current_precision) or ("16" in current_precision):
            print("[Auto-Config] macOS (MPS) 调试模式：自动将精度回退到 32-true 以保证稳定性")
            try:
                config.training.precision = "32-true"
            except Exception:
                pass

    trainer = pl.Trainer(
        accelerator=getattr(config.training, 'accelerator', 'auto'),
        devices=getattr(config.training, 'devices', 1),
        precision=getattr(config.training, 'precision', 32),
        min_epochs=int(getattr(config.training, 'min_epochs', 1)),
        max_epochs=config.training.max_epochs,
        logger=logger,
        callbacks=(lambda cfg: (
            (lambda monitor, mode, patience, min_delta, enable_es, start_epoch: (
                [ModelCheckpoint(monitor=monitor, mode=mode, save_top_k=1, filename='best-model')] +
                ([DeferredEarlyStopping(monitor=monitor, patience=patience, mode=mode, min_delta=min_delta, start_epoch=start_epoch)] if enable_es else []) +
                [LearningRateMonitor(logging_interval='epoch')]
            ))(
                str(getattr(getattr(cfg.training, 'early_stopping', None), 'monitor', 'val/loss/total')),
                str(getattr(getattr(cfg.training, 'early_stopping', None), 'mode', 'min')),
                int(getattr(getattr(cfg.training, 'early_stopping', None), 'patience', 5)),
                float(getattr(getattr(cfg.training, 'early_stopping', None), 'min_delta', 0.0)),
                bool(getattr(getattr(cfg.training, 'early_stopping', None), 'enable', True)),
                int(getattr(getattr(cfg.training, 'early_stopping', None), 'start_epoch', 0))
            )
        ))(config),
        check_val_every_n_epoch=int(getattr(config.training, 'check_val_every_n_epoch', 1)),
        log_every_n_steps=int(getattr(config.training, 'log_every_n_steps', 10))
    )

    trainer.fit(model, datamodule)

    if config.evaluation.test_after_training:
        trainer.test(model, datamodule)


if __name__ == "__main__":
    pass


def setup_logging(config: DictConfig, output_dir: Path) -> TensorBoardLogger:
    logger = TensorBoardLogger(str(output_dir), name="nilm_experiment")
    return logger


def create_trainer(config: DictConfig, logger: TensorBoardLogger) -> pl.Trainer:
    monitor = str(getattr(getattr(config.training, 'early_stopping', None), 'monitor', 'val/loss/total'))
    mode = str(getattr(getattr(config.training, 'early_stopping', None), 'mode', 'min'))
    patience = int(getattr(getattr(config.training, 'early_stopping', None), 'patience', 5))
    min_delta = float(getattr(getattr(config.training, 'early_stopping', None), 'min_delta', 0.0))
    enable_es = bool(getattr(getattr(config.training, 'early_stopping', None), 'enable', True))
    start_epoch = int(getattr(getattr(config.training, 'early_stopping', None), 'start_epoch', 0))
    mc = ModelCheckpoint(monitor=monitor, mode=mode, save_top_k=1, filename='best-model')
    lrmon = LearningRateMonitor(logging_interval='epoch')
    callbacks = [mc, lrmon]
    if enable_es:
        callbacks.append(DeferredEarlyStopping(monitor=monitor, patience=patience, mode=mode, min_delta=min_delta, start_epoch=start_epoch))
    trainer = pl.Trainer(
        accelerator=getattr(config.training, 'accelerator', 'auto'),
        devices=getattr(config.training, 'devices', 1),
        precision=getattr(config.training, 'precision', 32),
        min_epochs=int(getattr(config.training, 'min_epochs', 1)),
        max_epochs=config.training.max_epochs,
        logger=logger,
        callbacks=callbacks,
        check_val_every_n_epoch=int(getattr(config.training, 'check_val_every_n_epoch', 1)),
        log_every_n_steps=int(getattr(config.training, 'log_every_n_steps', 10)),
        enable_checkpointing=True,
    )
    return trainer
