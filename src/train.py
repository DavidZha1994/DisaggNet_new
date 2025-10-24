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
from src.losses.losses import create_loss_function, RECOMMENDED_LOSS_CONFIGS


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

        # 归一化损失配置（按设备相对刻度）
        loss_cfg = getattr(self.config, 'loss', None)
        self.normalize_per_device: bool = bool(getattr(loss_cfg, 'normalize_per_device', True)) if loss_cfg is not None else True
        # Huber 相对转折点（相对满量程）
        self.huber_beta_rel: float = float(getattr(loss_cfg, 'huber_beta_rel', 0.05)) if loss_cfg is not None else 0.05
        # 相对误差项系数与稳定项 epsilon
        self.rel_loss_weight: float = float(getattr(loss_cfg, 'rel_loss_weight', 0.5)) if loss_cfg is not None else 0.5
        self.rel_eps: float = float(getattr(loss_cfg, 'rel_eps', 0.05)) if loss_cfg is not None else 0.05
        # 仅在激活时刻加相对误差：用相对阈值（推荐等于 rel_eps）
        self.active_threshold_rel: float = float(getattr(loss_cfg, 'active_threshold_rel', self.rel_eps)) if loss_cfg is not None else self.rel_eps
        # 每设备尺度（训练集 P95），由 DataModule 提供
        self.power_scale: Optional[torch.Tensor] = None

        # 校验/可视化与训练状态初始化
        self.best_thresholds = {}
        self.best_val_loss = float('inf')
        self.automatic_optimization = True
        vis_cfg = getattr(getattr(config, 'training', None), 'visualization', None)
        self.enable_visualization = bool(getattr(vis_cfg, 'enable', False) if vis_cfg is not None else False)
        self.max_plots_per_epoch = int(getattr(vis_cfg, 'max_plots_per_epoch', 8) if vis_cfg is not None else 8)
        # 新增：事件筛选与保存目录
        self.plot_event_only = bool(getattr(vis_cfg, 'plot_event_only', False) if vis_cfg is not None else False)
        self.active_threshold_kw = float(getattr(vis_cfg, 'active_threshold_kw', 0.1) if vis_cfg is not None else 0.1)
        from pathlib import Path as _P
        self.vis_output_dir = str(getattr(vis_cfg, 'save_dir', _P('reports') / 'viz')) if vis_cfg is not None else str(_P('reports') / 'viz')
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
        
    def _isfinite(self, x: torch.Tensor) -> torch.Tensor:
        """设备安全有限性检查：在 MPS 上转 CPU 计算避免后端崩溃"""
        if not isinstance(x, torch.Tensor):
            raise TypeError("_isfinite 只接受 torch.Tensor")
        if x.device.type == 'mps':
            return torch.isfinite(x.detach().to('cpu')).to(x.device)
        try:
            return torch.isfinite(x)
        except Exception:
            return torch.isfinite(x.detach().to('cpu')).to(x.device)
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        time_features = batch['time_features']  # (batch_size, window_size, n_time_features)
        freq_features = batch.get('freq_features', None)  # (batch_size, n_time_frames, n_freq_bins)
        time_positional = batch.get('time_positional', None)  # (batch_size, window_size, time_dim)
        aux_features = batch.get('aux_features', None)  # (batch_size, n_aux_features)
        
        return self.model(time_features, freq_features, time_positional, aux_features=aux_features, time_valid_mask=batch.get('time_valid_mask', None), freq_valid_mask=batch.get('freq_valid_mask', None), aux_valid_mask=batch.get('aux_valid_mask', None))

    # --- 新增：按设备尺度工具 ---
    def _ensure_power_scale(self, device: torch.device, k: int) -> torch.Tensor:
        """获取形状 (1,1,K) 的尺度张量；从 datamodule 复制或用单位尺度回退。"""
        if (self.power_scale is None) or (self.power_scale.numel() != k):
            # 从 DataModule 获取
            try:
                dm = getattr(self.trainer, 'datamodule', None)
                if dm is not None and hasattr(dm, 'power_scale_vec') and isinstance(dm.power_scale_vec, torch.Tensor):
                    self.power_scale = dm.power_scale_vec.detach().clone().float()
                else:
                    self.power_scale = torch.ones(k, dtype=torch.float32)
            except Exception:
                self.power_scale = torch.ones(k, dtype=torch.float32)
        # 安全下限，避免除零
        scale = torch.clamp(self.power_scale, min=1e-6).to(device)
        return scale.view(1, 1, -1)

    def on_fit_start(self) -> None:
        """在训练开始时，从 DataModule 读取 per-device 尺度。"""
        try:
            dm = getattr(self.trainer, 'datamodule', None)
            if dm is not None and hasattr(dm, 'power_scale_vec') and isinstance(dm.power_scale_vec, torch.Tensor):
                self.power_scale = dm.power_scale_vec.detach().clone().float()
                print(f"[信息] 已载入每设备 P95 尺度：{self.power_scale.tolist()}")
            else:
                print("[警告] DataModule 未提供 power_scale_vec，回退为单位尺度。")
                self.power_scale = None
        except Exception as e:
            print(f"[警告] 读取 power_scale 失败：{e}")
            self.power_scale = None

    def _compute_normalized_seq_loss(self, pred_seq: torch.Tensor, target_seq: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """在相对刻度(0-1)上计算 Huber(beta=0.05) + 相对误差项。"""
        B, L, K = pred_seq.size(0), pred_seq.size(1), pred_seq.size(2)
        scale = self._ensure_power_scale(pred_seq.device, K)  # (1,1,K)
        # 归一化
        pred_n = pred_seq / scale
        target_n = target_seq / scale
        # Huber（SmoothL1）在相对刻度
        beta = float(self.huber_beta_rel)
        resid = torch.abs(pred_n - target_n)
        huber_el = torch.where(
            resid < beta,
            0.5 * resid ** 2 / beta,  # 常见定义 variant：确保在 resid==beta 处一阶连续
            resid - 0.5 * beta
        )
        huber_el = torch.where(valid, huber_el, torch.zeros_like(huber_el))
        huber_loss = huber_el.sum() / valid.float().sum().clamp_min(1.0)
        # 相对误差项（仅在激活时刻）
        act_mask = (target_n > float(self.active_threshold_rel))
        act_mask = act_mask & valid
        denom_rel = (target_n + float(self.rel_eps))
        rel_err = torch.abs(pred_n - target_n) / denom_rel
        rel_err = torch.where(act_mask, rel_err, torch.zeros_like(rel_err))
        rel_loss = rel_err.sum() / act_mask.float().sum().clamp_min(1.0)
        return huber_loss + float(self.rel_loss_weight) * rel_loss

    def _forward_and_compute_loss(self, batch: Dict[str, torch.Tensor], stage: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """通用的前向传播和损失计算"""
        # 序列前向
        seq_out = self.model.forward_seq(
            batch.get('time_features'),
            batch.get('freq_features'),
            batch.get('time_positional'),
            batch.get('aux_features'),
            time_valid_mask=batch.get('time_valid_mask'),
            freq_valid_mask=batch.get('freq_valid_mask'),
            aux_valid_mask=batch.get('aux_valid_mask')
        )
        pred_seq = seq_out[0] if isinstance(seq_out, tuple) else seq_out
        
        # 可选的预测清理
        if os.environ.get('DISAGGNET_SANITIZE_PRED', '0') == '1':
            pred_seq = torch.nan_to_num(pred_seq, nan=0.0, posinf=0.0, neginf=0.0)
        
        target_seq = batch.get('target_seq', None)
        if target_seq is None:
            return pred_seq, torch.tensor(0.0, device=pred_seq.device)
        
        # 有效掩码
        valid = self._isfinite(pred_seq) & self._isfinite(target_seq)
        vm = batch.get('target_seq_valid_mask', None)
        if isinstance(vm, torch.Tensor):
            if vm.dim() + 1 == pred_seq.dim():
                vm = vm.unsqueeze(-1)
            mask_vm = (vm > 0)
            if os.environ.get('DISAGGNET_MASK_RELAX', '0') == '1':
                valid = mask_vm
            else:
                valid = valid & mask_vm
        # 若无有效点，尝试放宽到仅使用标签掩码，并清理预测
        try:
            valid_count = int(valid.float().sum().item())
            if valid_count == 0 and isinstance(vm, torch.Tensor):
                if vm.dim() + 1 == pred_seq.dim():
                    vm = vm.unsqueeze(-1)
                valid = (vm > 0)
                # 清理预测，防止非有限值导致后续全部无效
                pred_seq = torch.nan_to_num(pred_seq, nan=0.0, posinf=0.0, neginf=0.0)
                self._safe_log(f'{stage}/debug/mask_fallback', 1.0, on_step=True, on_epoch=False)
        except Exception:
            pass
        
        # 使用相对刻度损失
        if self.normalize_per_device:
            seq_loss = self._compute_normalized_seq_loss(pred_seq, target_seq, valid)
        else:
            # 回退：原始 kW 域 Huber
            delta = getattr(self.loss_fn.huber_loss, 'delta', 1.0) if hasattr(self.loss_fn, 'huber_loss') else 1.0
            residual = torch.abs(pred_seq - target_seq)
            element_loss = torch.where(
                residual < delta,
                0.5 * residual ** 2,
                delta * (residual - 0.5 * delta)
            )
            element_loss = torch.where(valid, element_loss, torch.zeros_like(element_loss))
            seq_loss = element_loss.sum() / valid.float().sum().clamp_min(1.0)
        
        return pred_seq, seq_loss

    def _compute_metrics(self, batch: Dict[str, torch.Tensor], stage: str = 'val') -> Dict[str, float]:
        """计算序列级评估指标（MAE/RMSE），默认在 kW 域；若启用归一化，额外记录相对指标。"""
        metrics: Dict[str, float] = {}
        target_seq = batch.get('target_seq', None)
        if target_seq is None:
            metrics['score'] = float('nan')
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
                pred_seq = torch.nan_to_num(pred_seq, nan=0.0, posinf=0.0, neginf=0.0)

                valid = self._isfinite(pred_seq) & self._isfinite(target_seq)
                vm = batch.get('target_seq_valid_mask', None)
                if isinstance(vm, torch.Tensor):
                    if vm.dim() + 1 == pred_seq.dim():
                        vm = vm.unsqueeze(-1)
                    valid = valid & (vm > 0)

                denom = valid.float().sum().clamp_min(1.0)
                mae_el = torch.abs(pred_seq - target_seq)
                mse_el = (pred_seq - target_seq) ** 2
                mae = torch.where(valid, mae_el, torch.zeros_like(mae_el)).sum() / denom
                rmse = torch.sqrt(torch.where(valid, mse_el, torch.zeros_like(mse_el)).sum() / denom)

                self._safe_log(f'{stage}/metrics/sequence/mae', mae, on_epoch=True, sync_dist=True)
                self._safe_log(f'{stage}/metrics/sequence/rmse', rmse, on_epoch=True, sync_dist=True)
                metrics['seq_mae'] = float(mae.item())
                metrics['seq_rmse'] = float(rmse.item())

                # 若启用归一化，同时记录相对指标
                if self.normalize_per_device:
                    K = pred_seq.size(2)
                    scale = self._ensure_power_scale(pred_seq.device, K)
                    pred_n = pred_seq / scale
                    target_n = target_seq / scale
                    denom_n = valid.float().sum().clamp_min(1.0)
                    mae_n = torch.where(valid, torch.abs(pred_n - target_n), torch.zeros_like(pred_n)).sum() / denom_n
                    rmse_n = torch.sqrt(torch.where(valid, (pred_n - target_n)**2, torch.zeros_like(pred_n)).sum() / denom_n)
                    self._safe_log(f'{stage}/metrics/sequence/mae_rel', mae_n, on_epoch=True, sync_dist=True)
                    self._safe_log(f'{stage}/metrics/sequence/rmse_rel', rmse_n, on_epoch=True, sync_dist=True)

                # 评分：-MAE（kW 域），越大越好
                metrics['score'] = -metrics['seq_mae']
        except Exception:
            metrics.setdefault('score', float('nan'))
        return metrics

    def _collect_sequence_examples(self, pred_seq: torch.Tensor, target_seq: torch.Tensor, batch: Dict[str, torch.Tensor]) -> None:
        """收集用于可视化的序列样本"""
        try:
            if not self.enable_visualization or len(self._sequence_examples) >= self.max_plots_per_epoch:
                return
            
            B = pred_seq.size(0)
            for i in range(min(B, self.max_plots_per_epoch - len(self._sequence_examples))):
                # 使用联合掩码：isfinite(pred) & isfinite(target) 与 target_seq_valid_mask
                p = pred_seq[i].detach().cpu().float()
                t = target_seq[i].detach().cpu().float()
                valid = self._isfinite(p) & self._isfinite(t)
                vm = batch.get('target_seq_valid_mask', None)
                if isinstance(vm, torch.Tensor):
                    vm_i = vm[i]
                    if vm_i.dim() + 1 == p.dim():
                        vm_i = vm_i.unsqueeze(-1)
                    valid = (vm_i > 0)
                    mask_vm = (vm_i > 0)
                    if os.environ.get('DISAGGNET_MASK_RELAX', '0') == '1':
                        valid = mask_vm
                    else:
                        valid = valid & mask_vm
                # 若无有效点，回退到仅使用标签掩码
                try:
                    if int(valid.float().sum().item()) == 0 and isinstance(vm, torch.Tensor):
                        vm_i2 = vm[i]
                        if vm_i2.dim() + 1 == p.dim():
                            vm_i2 = vm_i2.unsqueeze(-1)
                        valid = (vm_i2 > 0)
                except Exception:
                    pass
                # 事件筛选：仅在存在事件时收集
                try:
                    event_present = True
                    if self.plot_event_only:
                        # 基于相对刻度的事件阈值（优先使用每设备尺度）
                        if isinstance(self.power_scale, torch.Tensor):
                            scale = self.power_scale.detach().cpu().float().view(1, 1, -1)  # (1,1,K)
                            t_n = t / scale
                            event_present = bool((t_n > float(self.active_threshold_rel)).any().item())
                        else:
                            # 回退：使用绝对kW阈值
                            event_present = bool((t > float(self.active_threshold_kw)).any().item())
                    if not event_present:
                        continue
                except Exception:
                    pass
                # 对无效位置置为 NaN，避免零线伪象
                p = p.clone(); t = t.clone()
                p[~valid] = torch.nan
                t[~valid] = torch.nan
                
                # 可选：采集总功率（mains）曲线（若存在）
                mains_curve = None
                tf = batch.get('time_features', None)
                if isinstance(tf, torch.Tensor):
                    tf_i = tf[i].detach().cpu().float()  # (L, C)
                    mains_idx = None
                    try:
                        dm = getattr(self.trainer, 'datamodule', None)
                        feat_names = getattr(dm, 'feature_names', []) if dm is not None else []
                        if isinstance(feat_names, list) and feat_names:
                            for cand in ['P_kW', 'P_active', 'P']:  # 常见有功功率命名
                                if cand in feat_names:
                                    mains_idx = feat_names.index(cand)
                                    break
                        # 回退：若未知，尝试使用第0列作为mains
                        if mains_idx is None and tf_i.size(1) > 0:
                            mains_idx = 0
                    except Exception:
                        mains_idx = 0 if tf_i.size(1) > 0 else None
                    if mains_idx is not None and mains_idx < tf_i.size(1):
                        mains_curve = tf_i[:, mains_idx]
                
                example = {'pred': p, 'target': t, 'valid': valid, 'mains': mains_curve}
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
        if os.environ.get('DISAGGNET_SANITIZE_PRED', '0') == '1':
            try:
                pred_seq = torch.nan_to_num(pred_seq, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                pass
        target_seq = batch.get('target_seq', None)
        if target_seq is None:
            # 缺少监督时，返回0损失以避免中断（同时记录）
            try:
                dev = pred_seq.device
            except Exception:
                dev = torch.device('cpu')
            self._safe_log('train/loss/sequence', float('nan'), on_step=True, on_epoch=True, prog_bar=True)
            return torch.tensor(0.0, device=dev)

        # 计算掩码
        valid = self._isfinite(pred_seq) & self._isfinite(target_seq)
        vm = batch.get('target_seq_valid_mask', None)
        try:
            if isinstance(vm, torch.Tensor):
                if vm.dim() + 1 == pred_seq.dim():
                    vm = vm.unsqueeze(-1)
                mask_vm = (vm > 0)
                # 环境变量可放宽掩码，仅使用标签掩码
                if os.environ.get('DISAGGNET_MASK_RELAX', '0') == '1':
                    valid = mask_vm
                else:
                    valid = valid & mask_vm
        except Exception:
            pass
        # 调试：统计有效元素数量
        valid_count = int(valid.float().sum().item())
        self._safe_log('train/debug/valid_count', float(valid_count), on_step=True, on_epoch=False)
        # 若无有效点，回退到仅使用标签掩码，并清理预测
        try:
            if valid_count == 0:
                vm2 = batch.get('target_seq_valid_mask', None)
                if isinstance(vm2, torch.Tensor):
                    if vm2.dim() + 1 == pred_seq.dim():
                        vm2 = vm2.unsqueeze(-1)
                    valid = (vm2 > 0)
                    pred_seq = torch.nan_to_num(pred_seq, nan=0.0, posinf=0.0, neginf=0.0)
                    self._safe_log('train/debug/mask_fallback', 1.0, on_step=True, on_epoch=False)
        except Exception:
            pass

        # 序列损失：相对刻度 + 相对误差项
        if self.normalize_per_device:
            seq_loss = self._compute_normalized_seq_loss(pred_seq, target_seq, valid)
            # 逐设备（按相对刻度）
            try:
                B, L, K = pred_seq.size(0), pred_seq.size(1), pred_seq.size(2)
                scale = self._ensure_power_scale(pred_seq.device, K)
                pred_n = pred_seq / scale
                target_n = target_seq / scale
                beta = float(self.huber_beta_rel)
                resid = torch.abs(pred_n - target_n)
                el = torch.where(resid < beta, 0.5 * resid ** 2 / beta, resid - 0.5 * beta)
                el = torch.where(valid, el, torch.zeros_like(el))
                per_dev_denom = valid.float().sum(dim=(0,1)).clamp_min(1.0)
                per_dev_sum = el.sum(dim=(0,1))
                per_dev_loss = per_dev_sum / per_dev_denom
                for i, d_loss in enumerate(per_dev_loss):
                    name = self.device_names[i] if i < len(self.device_names) else f'device_{i+1}'
                    self._safe_log(f'train/loss/sequence/device/{name}', d_loss, on_step=False, on_epoch=True)
            except Exception:
                pass
        else:
            # 回退：原始 kW 域 Huber
            delta = getattr(self.loss_fn.huber_loss, 'delta', 1.0) if hasattr(self.loss_fn, 'huber_loss') else 1.0
            residual = torch.abs(pred_seq - target_seq)
            element_loss = torch.where(
                residual < delta,
                0.5 * residual ** 2,
                delta * (residual - 0.5 * delta)
            )
            element_loss = torch.where(valid, element_loss, torch.zeros_like(element_loss))
            denom = valid.float().sum().clamp_min(1.0)
            seq_loss = element_loss.sum() / denom
            # 逐设备记录
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

        # 能量守恒损失（窗口级）：设备总和应接近总功率（仍在 kW 域）
        try:
            total_power = batch.get('total_power', None)
            if isinstance(total_power, torch.Tensor) and total_power.dim() == 2 and total_power.size(1) == 1:
                sum_per_t = pred_seq.sum(dim=2)  # (B, L)
                pred_total_mean = sum_per_t.mean(dim=1, keepdim=True)  # (B, 1)
                # 掩蔽无效项，避免 NaN/Inf 传播
                valid_total = self._isfinite(pred_total_mean) & self._isfinite(total_power)
                rel_err = torch.where(
                    valid_total,
                    torch.abs(pred_total_mean - total_power) / (total_power.abs() + 1e-6),
                    torch.zeros_like(total_power)
                )
                # 对有效样本做平均
                denom = valid_total.float().sum().clamp_min(1.0)
                conservation_loss = rel_err.sum() / denom
                self._safe_log('train/loss/conservation', conservation_loss, on_step=True, on_epoch=True)
                # 使用损失配置中的权重（若可用），默认0
                cons_w = float(getattr(self.loss_fn, 'conservation_weight', 0.0))
                total_loss = total_loss + cons_w * conservation_loss
        except Exception:
            pass

        losses = {'seq_regression': seq_loss, 'total': total_loss}
        self._validate_losses(losses, 'train', batch_idx)

        # 记录学习率（按 epoch）
        if getattr(self, '_trainer', None) is not None and getattr(self._trainer, 'optimizers', None):
            self._safe_log('train/metrics/optimization/lr', self._trainer.optimizers[0].param_groups[0]['lr'], on_step=False, on_epoch=True)

        # 记录损失
        self._safe_log('train/loss/sequence', seq_loss, on_step=True, on_epoch=True, prog_bar=True)
        self._safe_log('train/loss/total', total_loss, on_step=True, on_epoch=True, prog_bar=False)

        # 收集训练可视化样本（事件筛选在收集函数中控制）
        try:
            if batch.get('target_seq', None) is not None:
                self._collect_sequence_examples(pred_seq, batch.get('target_seq'), batch)
        except Exception:
            pass

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


    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """验证步骤"""
        self._validate_batch_data(batch, 'val', batch_idx)
        
        pred_seq, seq_loss = self._forward_and_compute_loss(batch, 'val')
        target_seq = batch.get('target_seq', None)
        
        # 可视化样本收集
        if target_seq is not None:
            self._collect_sequence_examples(pred_seq, target_seq, batch)
        
        # 计算指标
        metrics = self._compute_metrics(batch, stage='val')
        
        # 记录损失
        self._safe_log('val/loss', seq_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return {
            'val_loss': seq_loss,
            'predictions': pred_seq,
            'targets': target_seq
        }
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """测试步骤"""
        self._validate_batch_data(batch, 'test', batch_idx)
        
        pred_seq, seq_loss = self._forward_and_compute_loss(batch, 'test')
        target_seq = batch.get('target_seq', None)
        
        # 计算指标
        metrics = self._compute_metrics(batch, stage='test')
        
        # 记录损失和分数
        self._safe_log('test/loss', seq_loss, on_step=False, on_epoch=True)
        self._safe_log('test/score', metrics['score'], on_step=False, on_epoch=True)
        
        return {
            'test_loss': seq_loss,
            'test_score': metrics['score'],
            'predictions': pred_seq,
            'targets': target_seq
        }


    
    def on_validation_epoch_end(self) -> None:
        """验证轮次结束"""
        # 简单的可视化：保存到文件而不是TensorBoard
        if self.enable_visualization and len(self._sequence_examples) > 0:
            try:
                self._save_sequence_examples('val')
                self._sequence_examples.clear()
            except Exception as e:
                try:
                    print(f"Visualization failed: {e}")
                except Exception:
                    pass
        # 获取当前验证监控值（统一为 val/loss，兼容旧键）
        current_val = self.trainer.callback_metrics.get('val/loss', None)
        if current_val is None:
            current_val = float('inf')
        # 将 Tensor 转为 float，避免后续配置或日志处理中出现 OmegaConf 类型不支持
        try:
            if hasattr(current_val, 'item'):
                current_val = float(current_val.item())
            else:
                current_val = float(current_val)
        except Exception:
            current_val = float('inf')
        
        # 若当前值非有限，安全回退为 +inf（与最小化目标一致）
        if not np.isfinite(current_val):
            current_val = float('inf')
        
        # 更新最佳验证损失（min 模式）
        mode = str(getattr(self.config.training.checkpoint, 'mode', 'min')).lower()
        if np.isfinite(current_val):
            is_better = current_val < float(getattr(self, 'best_val_loss', float('inf')))
            if is_better:
                self.best_val_loss = float(current_val)
                # 记录最佳验证损失（层级命名，在无 Trainer 时安全跳过）
                self._safe_log('val/best_loss', self.best_val_loss, on_epoch=True)
        
        # 记录最佳验证损失（归入验证命名空间，安全记录）
        self._safe_log('val/best_loss', getattr(self, 'best_val_loss', float('inf')), on_epoch=True)
        
        # 记录模型参数统计
        for name, param in self.named_parameters():
            if param.grad is not None:
                self.logger.experiment.add_histogram(f'params/{name}', param, self.current_epoch)
                self.logger.experiment.add_histogram(f'grads/{name}', param.grad, self.current_epoch)
        
        # 更新损失函数的epoch
        self.loss_fn.update_epoch(self.current_epoch)

    def on_train_epoch_end(self) -> None:
        """训练轮次结束：保存可视化样本到文件"""
        try:
            if self.enable_visualization and len(self._sequence_examples) > 0:
                self._save_sequence_examples('train')
                self._sequence_examples.clear()
        except Exception:
            pass

    def _save_sequence_examples(self, stage: str) -> None:
        """将收集的样本保存为PNG到指定目录"""
        try:
            import matplotlib.pyplot as plt
            from pathlib import Path
            base_dir = Path(self.vis_output_dir) / stage / f"epoch_{self.current_epoch:02d}"
            base_dir.mkdir(parents=True, exist_ok=True)
            for idx, ex in enumerate(self._sequence_examples[:self.max_plots_per_epoch]):
                pred = ex.get('pred'); target = ex.get('target'); valid = ex.get('valid'); mains = ex.get('mains', None)
                if valid is None:
                    valid = self._isfinite(pred) & self._isfinite(target)
                # 图1：设备预测 vs 目标（按设备维）
                fig1, ax1 = plt.subplots(figsize=(10, 4))
                try:
                    p_np = pred.clone(); t_np = target.clone()
                    mask = ~valid if isinstance(valid, torch.Tensor) else ~torch.tensor(valid, dtype=torch.bool)
                    p_np[mask] = torch.nan; t_np[mask] = torch.nan
                    # 将(K设备)在时间维上叠加或分别作图：这里叠加对比更直观
                    ax1.plot(t_np.numpy(), label='target')
                    ax1.plot(p_np.numpy(), label='pred')
                    ax1.set_title(f'{stage} sample {idx} devices')
                    ax1.legend()
                    fig1.tight_layout()
                    fig1.savefig(base_dir / f'sample_{idx}_devices.png')
                finally:
                    plt.close(fig1)
                # 图2：总功率对比（若有mains）
                if mains is not None:
                    fig2, ax2 = plt.subplots(figsize=(10, 4))
                    try:
                        # 预测总和：按设备求和
                        pred_total = pred.sum(dim=1) if pred.dim() == 2 else pred
                        true_total = target.sum(dim=1) if target.dim() == 2 else target
                        ax2.plot(true_total.numpy(), label='true_total')
                        ax2.plot(pred_total.numpy(), label='pred_total')
                        ax2.plot(mains.numpy(), label='mains')
                        ax2.set_title(f'{stage} sample {idx} total power')
                        ax2.legend()
                        fig2.tight_layout()
                        fig2.savefig(base_dir / f'sample_{idx}_total.png')
                    finally:
                        plt.close(fig2)
        except Exception as e:
            try:
                print(f"[Viz Save] failed: {e}")
            except Exception:
                pass


    
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
            monitor_key = 'val/loss'
            monitor_mode = 'min'
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
        """验证批次数据的有效性，并智能清理辅助特征的 NaN/Inf"""
        suppress_invalid = getattr(getattr(self.config, 'debug', {}), 'suppress_invalid_warnings', True)
        
        for key, tensor in batch.items():
            # 只对tensor类型的数据进行验证
            if not isinstance(tensor, torch.Tensor):
                continue
                
            # 检查是否有无效值
            finite_mask = self._isfinite(tensor)
            if finite_mask.all():
                continue
                
            nan_count = torch.isnan(tensor).sum().item()
            inf_count = torch.isinf(tensor).sum().item()
            total_elements = tensor.numel()
            invalid_ratio = (nan_count + inf_count) / total_elements
            
            # 记录统计信息
            self._safe_log(f'{stage}/data_quality/{key}_nan_count', float(nan_count), on_step=True, on_epoch=False)
            self._safe_log(f'{stage}/data_quality/{key}_inf_count', float(inf_count), on_step=True, on_epoch=False)
            self._safe_log(f'{stage}/data_quality/{key}_invalid_ratio', float(invalid_ratio), on_step=True, on_epoch=False)

            # 特殊处理不同类型的数据
            if key == 'aux_features':
                # 只记录辅助特征的数据质量，不修改原始数据
                # 数据清理现在通过掩码在模型层面处理
                self._safe_log(f'{stage}/data_quality/aux_features_invalid_count', float(nan_count + inf_count), on_step=True, on_epoch=False)
                
                # 如果无效值比例过高，发出警告
                if invalid_ratio > 0.1:  # 超过10%的值无效
                    print(f"警告: {stage} batch {batch_idx} 中 'aux_features' 无效值比例过高: {invalid_ratio:.2%} (NaN: {nan_count}, Inf: {inf_count})")
                elif not suppress_invalid:
                    print(f"信息: {stage} batch {batch_idx} 中 'aux_features' 包含无效值（将通过掩码处理）: NaN: {nan_count}, Inf: {inf_count}")
                continue
                
            elif key == 'target_seq':
                # 目标序列有掩码处理，只记录不清理
                if isinstance(batch.get('target_seq_valid_mask'), torch.Tensor):
                    if not suppress_invalid:
                        print(f"信息: {stage} batch {batch_idx} 中 'target_seq' 包含无效值（已有掩码处理）: NaN: {nan_count}, Inf: {inf_count}")
                    continue
                    
            elif key in ['time_features', 'freq_features']:
                # 时域和频域特征的保守清理
                cleaned = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
                batch[key] = cleaned
                self._safe_log(f'{stage}/data_cleaning/{key}_cleaned', float(nan_count + inf_count), on_step=True, on_epoch=False)
                
            # 严格验证模式下抛出错误
            if self.config.debug.strict_validation and invalid_ratio > 0.05:  # 超过5%无效值
                raise ValueError(f"发现过多无效数据在 {stage} batch {batch_idx}, key '{key}': "
                               f"NaN: {nan_count}, Inf: {inf_count}, 无效比例: {invalid_ratio:.2%}")
            
            # 一般情况下的警告
            if not suppress_invalid and invalid_ratio > 0.01:  # 超过1%无效值
                print(f"警告: {stage} batch {batch_idx} 中 '{key}' 包含无效值: NaN: {nan_count}, Inf: {inf_count} ({invalid_ratio:.2%})")
    
    def _clean_aux_features_smart(self, aux_features: torch.Tensor, nan_count: int, inf_count: int) -> torch.Tensor:
        """智能清理辅助特征"""
        if aux_features.numel() == 0:
            return aux_features
            
        # 获取特征维度
        if aux_features.dim() == 1:
            # 单个样本 [n_features]
            features = aux_features.unsqueeze(0)
            squeeze_output = True
        else:
            # 批次 [batch_size, n_features]
            features = aux_features
            squeeze_output = False
            
        batch_size, n_features = features.shape
        cleaned = features.clone()
        
        # 逐特征处理
        for feat_idx in range(n_features):
            feat_col = cleaned[:, feat_idx]
            
            # 检查这个特征的无效值
            nan_mask = torch.isnan(feat_col)
            inf_mask = torch.isinf(feat_col)
            
            if not (nan_mask.any() or inf_mask.any()):
                continue
                
            # 获取有效值用于统计
            valid_mask = ~(nan_mask | inf_mask)
            valid_values = feat_col[valid_mask]
            
            if valid_values.numel() == 0:
                # 全部无效，填充为0
                feat_col.fill_(0.0)
            else:
                # 有部分有效值，使用中位数填充NaN
                if nan_mask.any():
                    if valid_values.numel() == 1:
                        median_val = valid_values[0]
                    else:
                        median_val = torch.median(valid_values)
                    feat_col[nan_mask] = median_val
                    
                # 处理Inf值：裁剪到有效值范围
                if inf_mask.any():
                    min_val = torch.min(valid_values)
                    max_val = torch.max(valid_values)
                    
                    # 扩展范围以避免过度裁剪
                    range_val = max_val - min_val
                    if range_val > 0:
                        extended_min = min_val - 0.1 * range_val
                        extended_max = max_val + 0.1 * range_val
                    else:
                        extended_min = min_val - 1.0
                        extended_max = max_val + 1.0
                        
                    feat_col = torch.clamp(feat_col, extended_min, extended_max)
                    
            cleaned[:, feat_idx] = feat_col
            
        # 最终安全检查
        cleaned = torch.nan_to_num(cleaned, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if squeeze_output:
            cleaned = cleaned.squeeze(0)
            
        return cleaned
    
    def _validate_predictions(self, pred_seq: torch.Tensor, 
                            stage: str, batch_idx: int) -> None:
        """验证序列预测的有效性，仅检查曲线值。"""
        if not self._isfinite(pred_seq).all():
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
                
            if not bool(self._isfinite(loss_value)):
                self._safe_log(f'{stage}/loss_invalid/{loss_name}', 1.0, on_step=True, on_epoch=False)
                
                if self.config.debug.strict_validation:
                    raise ValueError(f"损失 '{loss_name}' 包含无效值在 {stage} batch {batch_idx}: {loss_value}")
                else:
                    print(f"警告: {stage} batch {batch_idx} 中损失 '{loss_name}' 无效: {loss_value}")
    
    def configure_callbacks(self) -> List[pl.Callback]:
        """配置回调函数"""
        callbacks = []
        
        # 模型检查点
        ckpt_monitor = 'val/loss'
        # 使用安全的文件名（不依赖指标占位，避免层级名导致格式错误）
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.training.checkpoint.dirpath,
            filename='epoch{epoch:02d}',
            monitor=ckpt_monitor,
            mode='min',
            save_top_k=self.config.training.checkpoint.save_top_k,
            save_last=True,
            verbose=True
        )
        callbacks.append(checkpoint_callback)
        
        # 早停
        if self.config.training.early_stopping.enable:
            es_monitor = 'val/loss'
            early_stopping = EarlyStopping(
                monitor=es_monitor,
                patience=self.config.training.early_stopping.patience,
                mode='min',
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
        sync_batchnorm=True if (accelerator == 'gpu' and num_cuda_devices > 1) else False,
        # 避免 macOS MPS 在初始验证阶段触发 NDArray 缓冲错误
        num_sanity_val_steps=0 if accelerator == 'mps' else getattr(getattr(config, 'training', None), 'num_sanity_val_steps', 2)
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
    
    # 保存最终结果（统一使用基于损失的最佳指标）
    best_loss = float(getattr(model, 'best_val_loss', float('inf')))
    results = {
        'best_val_loss': best_loss,
        'best_thresholds': model.best_thresholds,
        'test_results': test_results,
        'best_model_path': trainer.checkpoint_callback.best_model_path if getattr(trainer, 'checkpoint_callback', None) else None,
        'model_summary': str(model_summary),
        'config': OmegaConf.to_container(config, resolve=True)
    }
    
    results_path = output_dir / f"{experiment_name}_results.yaml"
    OmegaConf.save(results, results_path)
    
    print(f"训练完成! 最佳验证损失: {best_loss:.4f}")
    print(f"结果保存至: {results_path}")
    print(f"模型摘要保存至: {summary_path}")
    if getattr(trainer, 'checkpoint_callback', None):
        print(f"最佳模型保存至: {trainer.checkpoint_callback.best_model_path}")


# 注意：此脚本不应直接执行，请使用统一入口 main.py
# 示例：python main.py train --config-name=default