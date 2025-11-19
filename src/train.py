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
from src.utils.metrics import NILMMetrics
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
        
        # 初始化简洁联合损失
        loss_config = RECOMMENDED_LOSS_CONFIGS.get('balanced', {})
        if hasattr(config, 'loss') and config.loss:
            loss_config.update(OmegaConf.to_container(config.loss, resolve=True))
        self.loss_fn = create_loss_function(loss_config)
        loss_cfg_container = OmegaConf.to_container(getattr(config, 'loss', {}), resolve=True) if hasattr(config, 'loss') else {}
        cls_w = float(loss_cfg_container.get('classification_weight', 1.0)) if isinstance(loss_cfg_container, dict) else 1.0
        self.classification_enabled = bool(cls_w > 0.0)
        self._base_loss_weights = {
            'regression_weight': float(loss_cfg_container.get('regression_weight', getattr(self.loss_fn, 'regression_weight', 1.0))) if isinstance(loss_cfg_container, dict) else getattr(self.loss_fn, 'regression_weight', 1.0),
            'classification_weight': float(loss_cfg_container.get('classification_weight', getattr(self.loss_fn, 'classification_weight', 1.0))) if isinstance(loss_cfg_container, dict) else getattr(self.loss_fn, 'classification_weight', 1.0),
            'unknown_weight': float(loss_cfg_container.get('unknown_weight', getattr(self.loss_fn, 'unknown_weight', 0.0))) if isinstance(loss_cfg_container, dict) else getattr(self.loss_fn, 'unknown_weight', 0.0),
            'consistency_weight': float(loss_cfg_container.get('consistency_weight', getattr(self.loss_fn, 'consistency_weight', 0.0))) if isinstance(loss_cfg_container, dict) else getattr(self.loss_fn, 'consistency_weight', 0.0),
            'conservation_weight': float(loss_cfg_container.get('conservation_weight', getattr(self.loss_fn, 'conservation_weight', 0.0))) if isinstance(loss_cfg_container, dict) else getattr(self.loss_fn, 'conservation_weight', 0.0),
            'off_penalty_weight': float(loss_cfg_container.get('off_penalty_weight', getattr(self.loss_fn, 'off_penalty_weight', 0.0))) if isinstance(loss_cfg_container, dict) else getattr(self.loss_fn, 'off_penalty_weight', 0.0),
            'peak_focus_weight': float(loss_cfg_container.get('peak_focus_weight', getattr(self.loss_fn, 'peak_focus_weight', 0.0))) if isinstance(loss_cfg_container, dict) else getattr(self.loss_fn, 'peak_focus_weight', 0.0),
            'edge_focus_weight': float(loss_cfg_container.get('edge_focus_weight', getattr(self.loss_fn, 'edge_focus_weight', 0.0))) if isinstance(loss_cfg_container, dict) else getattr(self.loss_fn, 'edge_focus_weight', 0.0),
            'derivative_loss_weight': float(loss_cfg_container.get('derivative_loss_weight', getattr(self.loss_fn, 'derivative_loss_weight', 0.0))) if isinstance(loss_cfg_container, dict) else getattr(self.loss_fn, 'derivative_loss_weight', 0.0),
            'shape_loss_weight': float(loss_cfg_container.get('shape_loss_weight', getattr(self.loss_fn, 'shape_loss_weight', 0.0))) if isinstance(loss_cfg_container, dict) else getattr(self.loss_fn, 'shape_loss_weight', 0.0),
            'active_boost_weight': float(loss_cfg_container.get('active_boost_weight', getattr(self.loss_fn, 'active_boost_weight', 2.0))) if isinstance(loss_cfg_container, dict) else getattr(self.loss_fn, 'active_boost_weight', 2.0),
        }

        # 评估器（用于窗口级指标计算）
        eval_cfg = getattr(self.config, 'evaluation', None)
        thr_method = str(getattr(eval_cfg, 'threshold_method', 'optimal')) if eval_cfg is not None else 'optimal'
        self.nilm_metrics = NILMMetrics(self.device_names, threshold_method=thr_method)

        # —— Metric Learning / Prototype Library ——
        # 兼容最小配置：当 aux_training.metric_learning.enable 为 True 时启用
        try:
            ml_cfg = getattr(getattr(config, 'aux_training', None), 'metric_learning', None)
        except Exception:
            ml_cfg = None
        self.metric_learning_enable: bool = bool(getattr(ml_cfg, 'enable', False)) if ml_cfg is not None else False
        self.metric_learning_use_power: bool = bool(getattr(ml_cfg, 'use_power', False)) if ml_cfg is not None else False
        # 嵌入维度：优先从模型的 time_encoder 读取，其次从配置读取，最后回退到 d_model
        try:
            embed_dim = int(getattr(getattr(self.model, 'time_encoder', None), 'd_model', None))
        except Exception:
            embed_dim = None
        if embed_dim is None:
            try:
                embed_dim = int(getattr(getattr(getattr(config, 'model', None), 'time_encoder', None), 'd_model', None))
            except Exception:
                embed_dim = None
        if embed_dim is None:
            embed_dim = int(getattr(getattr(config, 'model', None), 'd_model', 32))

        self.prototype_library: Optional[PrototypeLibrary]
        if self.metric_learning_enable:
            self.prototype_library = PrototypeLibrary(self.n_devices, embed_dim)
        else:
            self.prototype_library = None

        # 归一化损失配置（按设备相对刻度）
        loss_cfg = getattr(self.config, 'loss', None)
        # 归一化损失配置
        self.normalize_per_device: bool = bool(getattr(loss_cfg, 'normalize_per_device', True)) if loss_cfg is not None else True
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
        # 验证集拼接缓存（按批累积，轮次结束保存）
        self._val_concat_store: Optional[Dict[str, List[torch.Tensor]]] = None

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
        """前向传播（用于指标计算）：
        与训练中的序列损失保持一致的缩放策略。

        逻辑：
        - 使用 forward_seq 生成序列级回归与窗口级分类预测；
        - 对序列回归输出按时间做掩码均值，得到窗口级功率（与 target_power 对齐）；
        - forward 返回 (pred_power_window, pred_states_window)。
        """
        time_features = batch['time_features']  # (B, L, C)
        freq_features = batch.get('freq_features', None)
        time_positional = batch.get('time_positional', None)
        aux_features = batch.get('aux_features', None)

        # 保证与训练一致的反标准化：优先使用每设备功率尺度
        try:
            device = time_features.device if isinstance(time_features, torch.Tensor) else torch.device('cpu')
            ext_scale = self._ensure_power_scale(device, self.n_devices)  # (1,1,K)
        except Exception:
            ext_scale = None

        # 产生序列预测与窗口级分类预测
        seq_pred, reg_win, cls_win, _unk, cls_seq = self.model.forward_seq(
            time_features,
            freq_features=freq_features,
            time_positional=time_positional,
            aux_features=aux_features,
            time_valid_mask=batch.get('time_valid_mask', None),
            freq_valid_mask=batch.get('freq_valid_mask', None),
            aux_valid_mask=batch.get('aux_valid_mask', None),
            external_scale=ext_scale
        )

        # 将序列回归输出聚合为窗口级功率，与 target_power 维度一致
        # 优先使用目标序列的有效掩码，其次使用时间有效掩码
        mask_seq = batch.get('target_seq_valid_mask', None)
        if isinstance(mask_seq, torch.Tensor):
            m = mask_seq
            # 若掩码缺设备维，扩展到与 seq_pred 相同形状
            if m.dim() + 1 == seq_pred.dim():
                m = m.unsqueeze(-1)
            m = (m > 0).to(seq_pred.dtype)
            num = (seq_pred * m).sum(dim=1)
            den = m.sum(dim=1).clamp_min(1.0)
            pred_power = num / den
        else:
            time_mask = batch.get('time_valid_mask', None)
            if isinstance(time_mask, torch.Tensor):
                tm = (time_mask > 0).to(seq_pred.dtype)  # (B, L)
                tm = tm.unsqueeze(-1).expand(-1, -1, seq_pred.size(-1))
                num = (seq_pred * tm).sum(dim=1)
                den = tm.sum(dim=1).clamp_min(1.0)
                pred_power = num / den
            else:
                pred_power = torch.nanmean(seq_pred, dim=1)

        # 窗口级分类：若启用硬二值，直接二值化用于指标与展示
        if bool(getattr(self.loss_fn, 'classification_hard', False)):
            thr = float(getattr(self.loss_fn, 'hard_threshold', 0.5))
            pred_states = (cls_win >= thr).float()
        else:
            pred_states = cls_win

        return pred_power, pred_states

    def forward_with_embeddings(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """窗口级前向，返回 (reg, cls, unknown?, embeddings)。
        与模型接口保持一致，便于测试与原型库更新。
        """
        time_features = batch['time_features']
        freq_features = batch.get('freq_features', None)
        time_positional = batch.get('time_positional', None)
        aux_features = batch.get('aux_features', None)
        return self.model.forward_with_embeddings(
            time_features,
            freq_features=freq_features,
            time_positional=time_positional,
            aux_features=aux_features,
            time_valid_mask=batch.get('time_valid_mask', None),
            freq_valid_mask=batch.get('freq_valid_mask', None),
            aux_valid_mask=batch.get('aux_valid_mask', None)
        )

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
        # 设置分类 pos_weight（若提供）
        try:
            dm = getattr(self.trainer, 'datamodule', None)
            if dm is not None and hasattr(dm, 'get_pos_weight'):
                self.loss_fn.set_pos_weight(dm.get_pos_weight())
        except Exception:
            pass
        # 传入每设备阳性比例 prior_p（用于稀有设备活跃期加权）
        try:
            dm = getattr(self.trainer, 'datamodule', None)
            if dm is not None and hasattr(dm, 'get_prior_p'):
                self.loss_fn.set_prior_p(dm.get_prior_p())
        except Exception:
            pass

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
        """通用的前向传播与联合损失计算（序列回归 + 序列分类 + 守恒）。"""
        # 序列前向
        # 为序列输出提供每设备尺度（训练集P95），确保单位统一到瓦特
        ext_scale = None
        try:
            tf = batch.get('time_features')
            device = tf.device if isinstance(tf, torch.Tensor) else getattr(self, 'device', torch.device('cpu'))
            ext_scale = self._ensure_power_scale(device, self.n_devices)  # (1,1,K)
        except Exception:
            ext_scale = None

        seq_out = self.model.forward_seq(
            batch.get('time_features'),
            batch.get('freq_features'),
            batch.get('time_positional'),
            batch.get('aux_features'),
            time_valid_mask=batch.get('time_valid_mask'),
            freq_valid_mask=batch.get('freq_valid_mask'),
            aux_valid_mask=batch.get('aux_valid_mask'),
            external_scale=ext_scale
        )
        # 解析输出
        if isinstance(seq_out, tuple) and len(seq_out) >= 5:
            pred_seq, reg_win, cls_win, unk_win, cls_seq = seq_out
        elif isinstance(seq_out, tuple) and len(seq_out) >= 4:
            pred_seq, reg_win, cls_win, unk_win = seq_out
            cls_seq = None
        else:
            pred_seq = seq_out
            reg_win, cls_win, unk_win, cls_seq = None, None, None, None
        # 分类序列：若启用硬二值训练，生成二值序列用于可视化
        try:
            if bool(getattr(self.loss_fn, 'classification_hard', False)):
                thr = float(getattr(self.loss_fn, 'hard_threshold', 0.5))
                cls_seq_bin = (cls_seq >= thr).float()
            else:
                cls_seq_bin = (cls_seq >= 0.5).float()
            self._last_cls_seq = cls_seq_bin
        except Exception:
            try:
                self._last_cls_seq = cls_seq
            except Exception:
                pass
        
        # 可选的预测清理
        if os.environ.get('DISAGGNET_SANITIZE_PRED', '0') == '1':
            pred_seq = torch.nan_to_num(pred_seq, nan=0.0, posinf=0.0, neginf=0.0)
        
        target_seq = batch.get('target_seq', None)
        status_seq = batch.get('status_seq', None)
        if target_seq is None:
            return pred_seq, torch.tensor(0.0, device=pred_seq.device)
        # 有效掩码
        vm = batch.get('target_seq_valid_mask', None)
        if isinstance(vm, torch.Tensor) and vm.dim() + 1 == pred_seq.dim():
            vm = vm.unsqueeze(-1)
        valid = vm if isinstance(vm, torch.Tensor) else (self._isfinite(pred_seq) & self._isfinite(target_seq))
        # 并入时域有效掩码，仅在有效时间步参与损失
        tvm = batch.get('time_valid_mask', None)
        if isinstance(tvm, torch.Tensor):
            if tvm.dim() + 1 == pred_seq.dim():
                tvm = tvm.unsqueeze(-1)
            valid = valid & (tvm > 0)

        # 序列回归损失
        seq_reg_loss = self.loss_fn.regression_seq_loss(pred_seq, target_seq, status_seq, valid, self.power_scale)

        # 序列分类损失（若提供标签与分类序列）
        seq_cls_loss = torch.tensor(0.0, device=pred_seq.device)
        if isinstance(status_seq, torch.Tensor) and isinstance(cls_seq, torch.Tensor):
            svm = batch.get('status_seq_valid_mask', None)
            if isinstance(svm, torch.Tensor) and svm.dim() + 1 == cls_seq.dim():
                svm = svm.unsqueeze(-1)
            # 将直通估计的处理集中在损失函数内部，避免双重STE减弱梯度
            seq_cls_loss = self.loss_fn.classification_seq_loss(cls_seq, status_seq, svm)

        # 守恒损失（窗口级）
        cons_loss = self.loss_fn.conservation_loss(
            batch.get('mains_seq', None),
            pred_seq,
            batch.get('target_seq', None),
            self.power_scale
        )
        # 序列→窗口一致性（将时间均值与窗口级预测对齐）
        consw_loss = torch.tensor(0.0, device=pred_seq.device)
        try:
            consw_loss = self.loss_fn.consistency_window_loss(pred_seq, reg_win, valid, self.power_scale)
        except Exception:
            pass
        unk_loss = torch.tensor(0.0, device=pred_seq.device)
        try:
            if isinstance(batch.get('mains_seq', None), torch.Tensor) and (unk_win is not None):
                unk_loss = self.loss_fn.unknown_residual_loss(batch.get('mains_seq'), pred_seq, unk_win, batch.get('status_seq', None))
        except Exception:
            pass

        # 课程式权重（可选）：若配置提供 warmup/boost，按当前 epoch 调整权重
        w_reg = float(getattr(self.loss_fn, 'regression_weight', 1.0))
        w_cls = float(getattr(self.loss_fn, 'classification_weight', 1.0))
        w_cons = float(getattr(self.loss_fn, 'conservation_weight', 0.0))
        w_unk = float(getattr(self.loss_fn, 'unknown_weight', 0.0))
        try:
            ep = int(getattr(self, 'current_epoch', 0))
            cur_cfg = getattr(self.config, 'loss', None)
            if cur_cfg is not None:
                reg_warm = int(getattr(cur_cfg, 'reg_warmup_epochs', 0))
                reg_warm_scale = float(getattr(cur_cfg, 'reg_warmup_scale', 1.5))
                cls_start = int(getattr(cur_cfg, 'cls_start_epoch', 0))
                unk_start = int(getattr(cur_cfg, 'unknown_start_epoch', 0))
                unk_ramp = int(getattr(cur_cfg, 'unknown_ramp_epochs', 0))
                if ep < reg_warm:
                    w_reg = w_reg * reg_warm_scale
                if ep < cls_start:
                    w_cls = 0.0
                if ep < unk_start:
                    w_unk = 0.0
                elif unk_ramp > 0:
                    w_unk = w_unk * min(1.0, (ep - unk_start + 1) / float(unk_ramp))
        except Exception:
            pass
        total = w_reg * seq_reg_loss 
        total = total + w_cls * seq_cls_loss 
        total = total + w_cons * cons_loss
        total = total + float(getattr(self.loss_fn, 'consistency_weight', 0.0)) * consw_loss
        total = total + w_unk * unk_loss

        # —— 训练期：更新设备原型库（用于后续异常检测） ——
        try:
            if stage == 'train' and self.metric_learning_enable and (self.prototype_library is not None):
                with torch.no_grad():
                    _, _, _, emb = self.forward_with_embeddings(batch)
                    tp = batch.get('target_power', None)
                    if isinstance(tp, torch.Tensor):
                        sv = self._ensure_power_scale(emb.device, self.n_devices).view(1, 1, -1).squeeze(1)
                        sv = sv.expand(tp.size(0), -1)
                        thr = float(getattr(getattr(getattr(self.config, 'aux_training', None), 'metric_learning', None), 'power_threshold_rel', 0.05))
                        mask = (tp / sv > thr).float()
                        self.prototype_library.update(emb.detach(), mask.detach())
                    else:
                        self.prototype_library.update(emb.detach(), None)
        except Exception:
            pass

        # 记录
        self._safe_log(f'{stage}/loss/regression_seq', seq_reg_loss, on_step=True, on_epoch=True)
        try:
            neg_ratio = (pred_seq < 0).float().mean()
            self._safe_log(f'{stage}/stats/neg_ratio', neg_ratio, on_step=False, on_epoch=True)
            self._safe_log(f'{stage}/stats/seq_mean', torch.nanmean(pred_seq), on_step=False, on_epoch=True)
            self._safe_log(f'{stage}/stats/seq_std', torch.nanstd(pred_seq), on_step=False, on_epoch=True)
            if reg_win is not None:
                self._safe_log(f'{stage}/stats/reg_win_mean', torch.nanmean(reg_win), on_step=False, on_epoch=True)
                self._safe_log(f'{stage}/stats/reg_win_std', torch.nanstd(reg_win), on_step=False, on_epoch=True)
                vm2 = valid.to(torch.float32) if isinstance(valid, torch.Tensor) else torch.ones_like(pred_seq[...,0])
                num = (pred_seq * vm2.unsqueeze(-1)).sum(dim=1)
                den = vm2.sum(dim=1).clamp_min(1.0)
                mean_seq = num / den
                gap = torch.abs(mean_seq - reg_win).mean()
                self._safe_log(f'{stage}/stats/consistency_gap', gap, on_step=False, on_epoch=True)
        except Exception:
            pass
        if isinstance(status_seq, torch.Tensor) and isinstance(cls_seq, torch.Tensor):
            self._safe_log(f'{stage}/loss/classification_seq', seq_cls_loss, on_step=True, on_epoch=True)
        if self.loss_fn.conservation_weight > 0:
            self._safe_log(f'{stage}/loss/conservation', cons_loss, on_step=False, on_epoch=True)
        if float(getattr(self.loss_fn, 'consistency_weight', 0.0)) > 0:
            self._safe_log(f'{stage}/loss/consistency_window', consw_loss, on_step=False, on_epoch=True)
        if self.loss_fn.unknown_weight > 0:
            self._safe_log(f'{stage}/loss/unknown', unk_loss, on_step=False, on_epoch=True)

        # 分设备损失记录（按 epoch 聚合，避免日志过密）
        try:
            reg_k = self.loss_fn.regression_seq_loss_per_device(pred_seq, target_seq, status_seq, valid, self.power_scale)
            if isinstance(reg_k, torch.Tensor) and reg_k.numel() == self.n_devices:
                for i in range(self.n_devices):
                    name = self.device_names[i] if i < len(self.device_names) else f'device_{i+1}'
                    self._safe_log(f"{stage}/loss_per_device/regression/{name}", reg_k[i], on_step=False, on_epoch=True)
            if isinstance(status_seq, torch.Tensor) and isinstance(cls_seq, torch.Tensor):
                svm = batch.get('status_seq_valid_mask', None)
                if isinstance(svm, torch.Tensor) and svm.dim() + 1 == cls_seq.dim():
                    svm = svm.unsqueeze(-1)
                cls_k = self.loss_fn.classification_seq_loss_per_device(cls_seq, status_seq, svm)
                if isinstance(cls_k, torch.Tensor) and cls_k.numel() == self.n_devices:
                    for i in range(self.n_devices):
                        name = self.device_names[i] if i < len(self.device_names) else f'device_{i+1}'
                        self._safe_log(f"{stage}/loss_per_device/classification/{name}", cls_k[i], on_step=False, on_epoch=True)
        except Exception:
            pass

        return pred_seq, total

    def _compute_metrics(self, batch: Dict[str, torch.Tensor], preds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, stage: str = 'val') -> Dict[str, float]:
        """窗口级 NILM 指标计算（MAE/NDE/SAE/TECA 等）。
        兼容调用方式：
        - _compute_metrics(batch, stage='val')  // 内部前向产生预测
        - _compute_metrics(batch, (pred_power, pred_states), stage='val')
        """
        try:
            with torch.no_grad():
                # 提取或生成预测
                if preds is None:
                    fwd = self(batch)
                    if isinstance(fwd, tuple) and len(fwd) >= 2:
                        pred_power, pred_states = fwd[0], fwd[1]
                    else:
                        pred_power, pred_states = fwd
                else:
                    pred_power, pred_states = preds

                # 目标
                y_true_power = batch.get('target_power', None)
                y_true_states = batch.get('target_states', None)
                if y_true_power is None or y_true_states is None:
                    return {'mae': float('nan'), 'nde': float('nan'), 'sae': float('nan'), 'teca': float('nan'), 'score': float('nan')}

                # 形状对齐（防止设备维不一致导致的异常）
                try:
                    Bp, Kp = int(pred_power.size(0)), int(pred_power.size(1))
                    Bt, Kt = int(y_true_power.size(0)), int(y_true_power.size(1))
                    Bs, Ks = int(y_true_states.size(0)), int(y_true_states.size(1))
                    # 对设备维进行裁剪到公共最小值
                    K_common = min(Kp, Kt, Ks)
                    if K_common <= 0:
                        return {'mae': float('nan'), 'nde': float('nan'), 'sae': float('nan'), 'teca': float('nan'), 'score': float('nan')}
                    pred_power = pred_power[:, :K_common]
                    pred_states = pred_states[:, :K_common]
                    y_true_power = y_true_power[:, :K_common]
                    y_true_states = y_true_states[:, :K_common]
                except Exception:
                    pass

                # 计算所有指标（关闭阈值优化以加速单批评估）
                metrics_all = self.nilm_metrics.compute_all_metrics(
                    y_pred_power=pred_power,
                    y_pred_proba=pred_states,
                    y_true_power=y_true_power,
                    y_true_states=y_true_states,
                    optimize_thresholds=False,
                    classification_enabled=bool(self.classification_enabled)
                )

                # 安全记录关键指标
                for key in ['mae', 'nde', 'sae', 'teca', 'score']:
                    val = metrics_all.get(key, None)
                    if isinstance(val, (int, float)):
                        self._safe_log(f'{stage}/metrics/{key}', float(val), on_epoch=True, sync_dist=True)
                # 分设备指标（若提供）。常见键：mae_per_device/nde_per_device/teca_per_device/f1_per_device/mcc_per_device/pr_auc_per_device/roc_auc_per_device
                try:
                    per_keys = [
                        'mae_per_device', 'nde_per_device', 'teca_per_device',
                        'f1_per_device', 'mcc_per_device', 'pr_auc_per_device', 'roc_auc_per_device'
                    ]
                    for kname in per_keys:
                        arr = metrics_all.get(kname, None)
                        if isinstance(arr, (list, tuple)):
                            for i, v in enumerate(arr):
                                if isinstance(v, (int, float)):
                                    name = self.device_names[i] if i < len(self.device_names) else f'device_{i+1}'
                                    self._safe_log(f"{stage}/metrics/per_device/{kname.replace('_per_device','')}/{name}", float(v), on_epoch=True, sync_dist=True)
                except Exception:
                    pass
                return {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics_all.items()}
        except Exception:
            # 返回包含期望键的占位结果，避免测试因缺键失败
            return {'mae': float('nan'), 'nde': float('nan'), 'sae': float('nan'), 'teca': float('nan'), 'score': float('nan')}

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
        """训练步骤（联合损失）。"""
        # 数据验证
        self._validate_batch_data(batch, 'train', batch_idx)

        # 统一计算损失
        pred_seq, total_loss = self._forward_and_compute_loss(batch, 'train')

        losses = {'total': total_loss}
        self._validate_losses(losses, 'train', batch_idx)

        # 记录学习率（按 epoch）
        if getattr(self, '_trainer', None) is not None and getattr(self._trainer, 'optimizers', None):
            self._safe_log('train/metrics/optimization/lr', self._trainer.optimizers[0].param_groups[0]['lr'], on_step=False, on_epoch=True)

        # 记录损失
        self._safe_log('train/loss/total', total_loss, on_step=True, on_epoch=True, prog_bar=True)

        # 收集训练可视化样本（事件筛选在收集函数中控制）
        try:
            if batch.get('target_seq', None) is not None:
                self._collect_sequence_examples(pred_seq, batch.get('target_seq'), batch)
        except Exception:
            pass

        # 每隔一定步数记录训练指标与梯度范数
        try:
            log_every = int(getattr(getattr(self.config, 'training', None), 'log_every_n_steps', 50))
        except Exception:
            log_every = 50
        if log_every > 0 and (batch_idx % log_every == 0):
            with torch.no_grad():
                _ = self._compute_metrics(batch, stage='train')
                if int(getattr(getattr(self.config, 'debug', None), 'track_grad_norm', 0)) > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=float('inf'))
                    self._safe_log('train/metrics/optimization/grad_norm', grad_norm, on_step=False, on_epoch=True)
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        self._safe_log('train/metrics/optimization/grad_anomaly', 1.0, on_step=True, on_epoch=False)
                        print(f"Warning: Invalid gradient norm at step {self.global_step}: {grad_norm}")

        return total_loss


    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """验证步骤"""
        self._validate_batch_data(batch, 'val', batch_idx)
        
        pred_seq, seq_loss = self._forward_and_compute_loss(batch, 'val')
        target_seq = batch.get('target_seq', None)
        
        # 可视化样本收集
        if target_seq is not None:
            self._collect_sequence_examples(pred_seq, target_seq, batch)

        # 累积验证批次用于轮次结束时拼接与保存
        try:
            if self._val_concat_store is None:
                self._val_concat_store = {
                    'predictions': [],
                    'targets': [],
                    'time_features': [],
                    'raw_windows': [],
                    'timestamps': [],
                    'valid_mask': [],
                    'pred_states': [],
                    'true_states': []
                }
            self._val_concat_store['predictions'].append(pred_seq.detach().to('cpu'))
            if isinstance(target_seq, torch.Tensor):
                self._val_concat_store['targets'].append(target_seq.detach().to('cpu'))
            # 分类序列与标签（若有）
            try:
                if isinstance(self._last_cls_seq, torch.Tensor):
                    self._val_concat_store['pred_states'].append(self._last_cls_seq.detach().to('cpu'))
                stat = batch.get('status_seq', None)
                if isinstance(stat, torch.Tensor):
                    self._val_concat_store['true_states'].append(stat.detach().to('cpu'))
            except Exception:
                pass
            tf = batch.get('time_features', None)
            if isinstance(tf, torch.Tensor):
                self._val_concat_store['time_features'].append(tf.detach().to('cpu'))
            rw = batch.get('raw_windows', None)
            if isinstance(rw, torch.Tensor):
                self._val_concat_store['raw_windows'].append(rw.detach().to('cpu'))
            ts = batch.get('timestamps', None)
            if isinstance(ts, torch.Tensor):
                self._val_concat_store['timestamps'].append(ts.detach().to('cpu'))
            vm = batch.get('target_seq_valid_mask', None)
            if isinstance(vm, torch.Tensor):
                self._val_concat_store['valid_mask'].append(vm.detach().to('cpu'))
        except Exception:
            pass
        
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
        # 生成并保存验证集拼接的交互式HTML（Plotly）
        try:
            if self._val_concat_store is not None and len(self._val_concat_store.get('predictions', [])) > 0:
                rank = int(getattr(self.trainer, 'global_rank', 0))
                def _safe_cat(lst: List[torch.Tensor]) -> Optional[torch.Tensor]:
                    try:
                        return torch.cat(lst, dim=0) if isinstance(lst, list) and len(lst) > 0 else None
                    except Exception:
                        return None
                preds = _safe_cat(self._val_concat_store.get('predictions', []))
                targs = _safe_cat(self._val_concat_store.get('targets', []))
                tf = _safe_cat(self._val_concat_store.get('time_features', []))
                vm = _safe_cat(self._val_concat_store.get('valid_mask', []))
                # 数据集与fold信息
                try:
                    dataset_name = Path(getattr(self.config.paths, 'prepared_dir', 'Data/prepared')).name
                except Exception:
                    dataset_name = 'prepared'
                try:
                    dm = getattr(self.trainer, 'datamodule', None)
                    fold_id = int(getattr(dm, 'fold_id', 0)) if dm is not None else 0
                except Exception:
                    fold_id = 0
                # 保存交互式HTML
                self._save_val_interactive_html(preds, targs, tf, vm, dataset_name, fold_id, rank)
                # 清空本轮缓存
                self._val_concat_store = None
        except Exception as e:
            try:
                print(f"[警告] 验证交互式可视化保存失败：{e}")
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
        
        # 旧版钩子：损失函数无需按 epoch 更新，删除过时调用

    def on_train_epoch_end(self) -> None:
        """训练轮次结束：保存可视化样本到文件"""
        try:
            if self.enable_visualization and len(self._sequence_examples) > 0:
                self._save_sequence_examples('train')
                self._sequence_examples.clear()
        except Exception:
            pass

    def on_train_epoch_start(self) -> None:
        try:
            dm = getattr(self.trainer, 'datamodule', None)
            imb = getattr(self.config, 'imbalance_handling', None)
            if dm is not None and imb is not None:
                early = int(getattr(imb, 'early_oversample_epochs', 0))
                base = str(getattr(imb, 'sampling_strategy', 'mixed'))
                base_boost = float(getattr(imb, 'event_boost', 2.0))
                early_boost = float(getattr(imb, 'early_event_boost', base_boost))
                if int(self.current_epoch) < early:
                    dm.set_sampling_strategy('oversample', event_boost=early_boost, pos_count_inverse_enable=False)
                else:
                    dm.set_sampling_strategy(base, event_boost=base_boost, pos_count_inverse_enable=bool(getattr(imb, 'pos_count_inverse_enable', False)))
                try:
                    self.trainer.reset_train_dataloader()
                except Exception:
                    pass
        except Exception:
            pass

        try:
            loss_cfg = getattr(self.config, 'loss', None)
            if loss_cfg is not None:
                warm_ep = int(getattr(loss_cfg, 'reg_warmup_epochs', 0))
                warm_scale = float(getattr(loss_cfg, 'reg_warmup_scale', 1.0))
                if int(self.current_epoch) < warm_ep and warm_scale > 1.0:
                    self.loss_fn.regression_weight = float(self._base_loss_weights['regression_weight']) * warm_scale
                else:
                    self.loss_fn.regression_weight = float(self._base_loss_weights['regression_weight'])

                cls_start = int(getattr(loss_cfg, 'cls_start_epoch', 0))
                if int(self.current_epoch) < cls_start:
                    self.loss_fn.classification_weight = 0.0
                else:
                    self.loss_fn.classification_weight = float(self._base_loss_weights['classification_weight'])

                unk_start = int(getattr(loss_cfg, 'unknown_start_epoch', 0))
                unk_ramp = int(getattr(loss_cfg, 'unknown_ramp_epochs', 0))
                base_unk = float(self._base_loss_weights['unknown_weight'])
                if int(self.current_epoch) < unk_start:
                    self.loss_fn.unknown_weight = 0.0
                else:
                    if unk_ramp > 0:
                        t = min(max(int(self.current_epoch) - unk_start + 1, 0), unk_ramp)
                        frac = float(t) / float(unk_ramp)
                        self.loss_fn.unknown_weight = base_unk * frac
                    else:
                        self.loss_fn.unknown_weight = base_unk

                self.loss_fn.consistency_weight = float(self._base_loss_weights['consistency_weight'])

                # 能量守恒权重：在 cons_start_epoch 后线性爬升到基础权重
                cons_start = int(getattr(loss_cfg, 'cons_start_epoch', 0))
                cons_ramp = int(getattr(loss_cfg, 'cons_ramp_epochs', 0))
                base_cons = float(self._base_loss_weights['conservation_weight'])
                if int(self.current_epoch) < cons_start:
                    self.loss_fn.conservation_weight = 0.0
                else:
                    if cons_ramp > 0:
                        t = min(max(int(self.current_epoch) - cons_start + 1, 0), cons_ramp)
                        frac = float(t) / float(cons_ramp)
                        self.loss_fn.conservation_weight = base_cons * frac
                    else:
                        self.loss_fn.conservation_weight = base_cons

                # 关闭抑制项权重：在 off_start_epoch 后线性爬升，避免早期压制振幅
                off_start = int(getattr(loss_cfg, 'off_start_epoch', 0))
                off_ramp = int(getattr(loss_cfg, 'off_ramp_epochs', 0))
                base_off = float(self._base_loss_weights['off_penalty_weight'])
                if int(self.current_epoch) < off_start:
                    self.loss_fn.off_penalty_weight = 0.0
                else:
                    if off_ramp > 0:
                        t = min(max(int(self.current_epoch) - off_start + 1, 0), off_ramp)
                        frac = float(t) / float(off_ramp)
                        self.loss_fn.off_penalty_weight = base_off * frac
                    else:
                        self.loss_fn.off_penalty_weight = base_off

                # 事件期聚焦权重课程式提升（peak/edge/derivative/shape）
                def _ramp(cur_ep: int, start: int, ramp: int, base_w: float, target_w: float) -> float:
                    if cur_ep < start:
                        return base_w
                    if ramp <= 0:
                        return target_w
                    t = min(max(cur_ep - start + 1, 0), ramp)
                    frac = float(t) / float(ramp)
                    return base_w + (target_w - base_w) * frac

                cur_ep = int(self.current_epoch)
                # peak
                p_start = int(getattr(loss_cfg, 'peak_ramp_start', 0))
                p_ramp = int(getattr(loss_cfg, 'peak_ramp_epochs', 0))
                p_target = float(getattr(loss_cfg, 'peak_target_weight', self._base_loss_weights['peak_focus_weight']))
                self.loss_fn.peak_focus_weight = _ramp(cur_ep, p_start, p_ramp, float(self._base_loss_weights['peak_focus_weight']), p_target)
                # edge
                e_start = int(getattr(loss_cfg, 'edge_ramp_start', 0))
                e_ramp = int(getattr(loss_cfg, 'edge_ramp_epochs', 0))
                e_target = float(getattr(loss_cfg, 'edge_target_weight', self._base_loss_weights['edge_focus_weight']))
                self.loss_fn.edge_focus_weight = _ramp(cur_ep, e_start, e_ramp, float(self._base_loss_weights['edge_focus_weight']), e_target)
                # derivative
                d_start = int(getattr(loss_cfg, 'derivative_ramp_start', 0))
                d_ramp = int(getattr(loss_cfg, 'derivative_ramp_epochs', 0))
                d_target = float(getattr(loss_cfg, 'derivative_target_weight', self._base_loss_weights['derivative_loss_weight']))
                self.loss_fn.derivative_loss_weight = _ramp(cur_ep, d_start, d_ramp, float(self._base_loss_weights['derivative_loss_weight']), d_target)
                # shape
                s_start = int(getattr(loss_cfg, 'shape_ramp_start', 0))
                s_ramp = int(getattr(loss_cfg, 'shape_ramp_epochs', 0))
                s_target = float(getattr(loss_cfg, 'shape_target_weight', self._base_loss_weights['shape_loss_weight']))
                self.loss_fn.shape_loss_weight = _ramp(cur_ep, s_start, s_ramp, float(self._base_loss_weights['shape_loss_weight']), s_target)
                # active boost (提升事件期损失贡献)
                ab_target = float(getattr(loss_cfg, 'active_boost_target', self._base_loss_weights['active_boost_weight']))
                ab_start = int(getattr(loss_cfg, 'active_boost_start_epoch', 0))
                ab_ramp = int(getattr(loss_cfg, 'active_boost_ramp_epochs', 0))
                self.loss_fn.active_boost_weight = _ramp(cur_ep, ab_start, ab_ramp, float(self._base_loss_weights['active_boost_weight']), ab_target)
                self.classification_enabled = bool(self.loss_fn.classification_weight > 0.0)
                self._safe_log('train/schedule/regression_weight', self.loss_fn.regression_weight, on_step=False, on_epoch=True)
                self._safe_log('train/schedule/classification_weight', self.loss_fn.classification_weight, on_step=False, on_epoch=True)
                self._safe_log('train/schedule/unknown_weight', self.loss_fn.unknown_weight, on_step=False, on_epoch=True)
                self._safe_log('train/schedule/conservation_weight', self.loss_fn.conservation_weight, on_step=False, on_epoch=True)
                self._safe_log('train/schedule/off_penalty_weight', getattr(self.loss_fn, 'off_penalty_weight', 0.0), on_step=False, on_epoch=True)
                self._safe_log('train/schedule/peak_focus_weight', getattr(self.loss_fn, 'peak_focus_weight', 0.0), on_step=False, on_epoch=True)
                self._safe_log('train/schedule/edge_focus_weight', getattr(self.loss_fn, 'edge_focus_weight', 0.0), on_step=False, on_epoch=True)
                self._safe_log('train/schedule/derivative_loss_weight', getattr(self.loss_fn, 'derivative_loss_weight', 0.0), on_step=False, on_epoch=True)
                self._safe_log('train/schedule/shape_loss_weight', getattr(self.loss_fn, 'shape_loss_weight', 0.0), on_step=False, on_epoch=True)
                self._safe_log('train/schedule/active_boost_weight', getattr(self.loss_fn, 'active_boost_weight', 0.0), on_step=False, on_epoch=True)
        except Exception:
            pass

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """保存检查点时持久化原型库（若启用度量学习）。"""
        try:
            if self.metric_learning_enable and self.prototype_library is not None:
                state = {k: v.detach().cpu() for k, v in self.prototype_library.state_dict().items()}
                checkpoint['prototype_library_state'] = state
        except Exception:
            pass

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """加载检查点时恢复原型库状态（若存在）。"""
        try:
            state = checkpoint.get('prototype_library_state', None)
            if state is not None:
                # 确保库已创建
                if self.prototype_library is None:
                    # 使用当前模型的 d_model 初始化
                    embed_dim = int(getattr(getattr(self.model, 'time_encoder', None), 'd_model', getattr(getattr(self.config, 'model', None), 'd_model', 32)))
                    self.prototype_library = PrototypeLibrary(self.n_devices, embed_dim)
                # 将张量移到当前设备（必要时）
                device = getattr(self, 'device', torch.device('cpu'))
                state_tensor = {k: (t.to(device) if isinstance(t, torch.Tensor) else t) for k, t in state.items()}
                self.prototype_library.load_state_dict(state_tensor)
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

    def _save_val_interactive_html(
        self,
        preds: Optional[torch.Tensor],
        targs: Optional[torch.Tensor],
        time_features: Optional[torch.Tensor],
        valid_mask: Optional[torch.Tensor],
        dataset_name: str,
        fold_id: int,
        rank: int
    ) -> None:
        """将当前验证轮次的拼接结果保存为可交互HTML。

        - 若提供 `time_features`，尝试提取 mains 曲线在上方子图展示。
        - 下方子图绘制每个设备的 `target` 与 `pred` 序列（按窗口顺序拼接）。
        """
        try:
            if preds is None or targs is None:
                return
            B, L, K = int(preds.size(0)), int(preds.size(1)), int(preds.size(2))
            # 展平为连续时间轴（按验证集窗口顺序），非真实时间，仅用于可视化拼接
            p_np = preds.detach().cpu().float().reshape(B * L, K).numpy()
            t_np = targs.detach().cpu().float().reshape(B * L, K).numpy()
            # 分类：拼接预测与真实
            pred_states_np, true_states_np = None, None
            try:
                store = getattr(self, '_val_concat_store', None)
                if isinstance(store, dict):
                    ps_list = store.get('pred_states', [])
                    ts_list = store.get('true_states', [])
                    if isinstance(ps_list, list) and len(ps_list) > 0:
                        pred_states_np = torch.cat(ps_list, dim=0).detach().cpu().float().reshape(B * L, K).numpy()
                    if isinstance(ts_list, list) and len(ts_list) > 0:
                        true_states_np = torch.cat(ts_list, dim=0).detach().cpu().float().reshape(B * L, K).numpy()
            except Exception:
                pred_states_np, true_states_np = None, None

            # mains（若有）：优先使用 raw_windows 第0通道，否则回退到 time_features 推断
            mains_series = None
            try:
                store = getattr(self, '_val_concat_store', None)
                if isinstance(store, dict):
                    raw_list = store.get('raw_windows', [])
                    if isinstance(raw_list, list) and len(raw_list) > 0:
                        rw = torch.cat(raw_list, dim=0).detach().cpu().float()  # (B, L, C)
                        # 单位判断：若 datamodule 提供原始通道名，且第0通道为 P_kW，则转为 W
                        dm = getattr(self.trainer, 'datamodule', None)
                        raw_names = getattr(dm, 'raw_channel_names', []) if dm is not None else []
                        mains_arr = rw[:, :, 0].reshape(B * L)
                        try:
                            if isinstance(raw_names, list) and len(raw_names) > 0:
                                ch0 = str(raw_names[0]).lower()
                                if ch0.endswith('p_kw') or ch0 == 'p_kw':
                                    mains_arr = mains_arr * 1000.0
                        except Exception:
                            pass
                        mains_series = mains_arr.numpy()
            except Exception:
                mains_series = None

            if mains_series is None and isinstance(time_features, torch.Tensor):
                try:
                    tf = time_features.detach().cpu().float()  # (B, L, C)
                    C = int(tf.size(2)) if tf.dim() == 3 else 0
                    if C > 0:
                        # 通过特征名推断 mains 索引
                        mains_idx = None
                        mains_name = None
                        try:
                            dm = getattr(self.trainer, 'datamodule', None)
                            feat_names = getattr(dm, 'feature_names', []) if dm is not None else []
                            if isinstance(feat_names, list) and feat_names:
                                for cand in ['P_W', 'P_kW', 'P_active', 'P']:
                                    if cand in feat_names:
                                        mains_idx = feat_names.index(cand)
                                        mains_name = cand
                                        break
                            if mains_idx is None:
                                mains_idx = 0
                        except Exception:
                            mains_idx = 0
                        mains_arr = tf[:, :, mains_idx].reshape(B * L)
                        try:
                            if mains_name == 'P_kW':
                                mains_arr = mains_arr * 1000.0
                        except Exception:
                            pass
                        mains_series = mains_arr.numpy()
                except Exception:
                    mains_series = None

            # 输出目录：统一到 self.vis_output_dir / 'val_interactive'
            from pathlib import Path as _P
            base_out = _P(self.vis_output_dir) / 'val_interactive' / dataset_name / f'fold_{int(fold_id)}' / f'rank_{int(rank)}'
            base_out.mkdir(parents=True, exist_ok=True)
            out_path = base_out / f'epoch_{int(self.current_epoch):03d}.html'

            # 生成 Plotly 图
            try:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
            except Exception as e:
                print(f"[警告] 导入 Plotly 失败，无法生成交互式HTML：{e}")
                return

            # 子图布局：上方 mains（若有），下方按设备逐个子图对比真实 vs 回归
            # 设备功率每个子图 + 可选一个开关状态子图
            has_states = (pred_states_np is not None) and (true_states_np is not None)
            total_rows = (1 + K + (K if has_states else 0)) if mains_series is not None else (K + (K if has_states else 0))
            titles = []
            if mains_series is not None:
                titles.append('输入总功率 (W)')
            for k in range(K):
                dev_name = self.device_names[k] if k < len(self.device_names) else f'device_{k+1}'
                titles.append(f'{dev_name}（真实 vs 回归）')
            if has_states:
                for k in range(K):
                    dev_name = self.device_names[k] if k < len(self.device_names) else f'device_{k+1}'
                    titles.append(f'{dev_name}（状态：真实 vs 预测）')
            fig = make_subplots(rows=total_rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=titles)

            x = np.arange(p_np.shape[0])
            # Row 1: mains
            current_row = 1
            if mains_series is not None:
                fig.add_trace(
                    go.Scatter(x=x, y=mains_series, name='mains_real', mode='lines', line=dict(color='#888')),
                    row=current_row, col=1
                )
                fig.update_yaxes(title_text='W (mains)', row=current_row, col=1)
                current_row += 1

            # Each device in its own subplot: target vs pred
            dev_limit = K
            for k in range(dev_limit):
                dev_name = self.device_names[k] if k < len(self.device_names) else f'device_{k+1}'
                fig.add_trace(
                    go.Scatter(x=x, y=t_np[:, k], name=f'{dev_name}_真实', mode='lines', line=dict(color='#2ca02c')),
                    row=current_row, col=1
                )
                fig.add_trace(
                    go.Scatter(x=x, y=p_np[:, k], name=f'{dev_name}_回归', mode='lines', line=dict(color='#ff7f0e')),
                    row=current_row, col=1
                )
                fig.update_yaxes(title_text='W', row=current_row, col=1)
                current_row += 1

            # Optional: states per device
            if has_states:
                for k in range(dev_limit):
                    dev_name = self.device_names[k] if k < len(self.device_names) else f'device_{k+1}'
                    # 使用阶梯线表示状态
                    fig.add_trace(
                        go.Scatter(x=x, y=true_states_np[:, k], name=f'{dev_name}_状态_真实', mode='lines', line=dict(color='#1f77b4')),
                        row=current_row, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=x, y=pred_states_np[:, k], name=f'{dev_name}_状态_预测', mode='lines', line=dict(color='#d62728')),
                        row=current_row, col=1
                    )
                    fig.update_yaxes(title_text='state', range=[-0.05, 1.05], row=current_row, col=1)
                    current_row += 1

            # 布局与交互
            # 动态高度：顶部 mains 约 240px，每设备约 200px
            base_h = 240 if mains_series is not None else 0
            per_dev_h = 200
            per_state_h = 110 if has_states else 0
            total_h = base_h + K * (per_dev_h + per_state_h)
            fig.update_layout(
                title=f'验证拼接对比图 (dataset={dataset_name}, fold={fold_id}, epoch={int(self.current_epoch)})',
                legend=dict(orientation='h', yanchor='bottom', y=-0.25),
                height=max(600, min(2200, total_h))
            )
            # 范围滑条在最底部轴显示
            fig.update_xaxes(rangeslider=dict(visible=True), row=total_rows, col=1)

            # 写入HTML
            try:
                import plotly
                plotly.offline.plot(fig, filename=str(out_path), auto_open=False, include_plotlyjs='cdn')
                print(f"[信息] 验证交互式可视化已保存：{out_path}")
            except Exception as e:
                print(f"[警告] 保存交互式HTML失败：{e}")
        except Exception:
            pass


    
    def configure_optimizers(self) -> Dict[str, Any]:
        """配置优化器和学习率调度器"""
        # 优化器 - 使用更保守的学习率
        if getattr(getattr(getattr(self.config, 'training', None), 'optimizer', None), 'name', 'adamw') == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=float(getattr(getattr(getattr(self.config, 'training', None), 'optimizer', None), 'lr', 1e-3)),
                weight_decay=float(getattr(getattr(getattr(self.config, 'training', None), 'optimizer', None), 'weight_decay', 0.01)),
                betas=getattr(getattr(getattr(self.config, 'training', None), 'optimizer', None), 'betas', (0.9, 0.999)),
                eps=1e-8
            )
        elif getattr(getattr(getattr(self.config, 'training', None), 'optimizer', None), 'name', 'adamw') == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=float(getattr(getattr(getattr(self.config, 'training', None), 'optimizer', None), 'lr', 1e-3)),
                weight_decay=float(getattr(getattr(getattr(self.config, 'training', None), 'optimizer', None), 'weight_decay', 0.01)),
                eps=1e-8
            )
        else:
            name = getattr(getattr(getattr(self.config, 'training', None), 'optimizer', None), 'name', 'adamw')
            if name not in ('adamw', 'adam'):
                raise ValueError(f"Unknown optimizer: {name}")
        
        # 学习率调度器
        scheduler_config = getattr(getattr(self.config, 'training', None), 'scheduler', None)
        
        if getattr(scheduler_config, 'name', 'none') == 'cosine':
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
        elif getattr(scheduler_config, 'name', 'none') == 'reduce_on_plateau':
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
        try:
            dirpath = getattr(getattr(getattr(self.config, 'training', None), 'checkpoint', None), 'dirpath', str(Path('outputs')/ 'checkpoints'))
            save_top_k = int(getattr(getattr(getattr(self.config, 'training', None), 'checkpoint', None), 'save_top_k', 1))
        except Exception:
            dirpath = str(Path('outputs') / 'checkpoints')
            save_top_k = 1
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            filename='epoch{epoch:02d}',
            monitor=ckpt_monitor,
            mode='min',
            save_top_k=save_top_k,
            save_last=True,
            verbose=True
        )
        callbacks.append(checkpoint_callback)
        
        # 早停
        try:
            es_cfg = getattr(getattr(self.config, 'training', None), 'early_stopping', None)
            enable_es = bool(getattr(es_cfg, 'enable', False))
        except Exception:
            enable_es = False
        if enable_es:
            try:
                patience = int(getattr(es_cfg, 'patience', 10))
            except Exception:
                patience = 10
            es_monitor = 'val/loss'
            early_stopping = EarlyStopping(
                monitor=es_monitor,
                patience=patience,
                mode='min',
                verbose=True
            )
            callbacks.append(early_stopping)
        
        # 学习率监控
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor)
        
        # 设备状态监控
        if bool(getattr(getattr(self.config, 'training', None), 'monitor_device_stats', False)):
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
        max_epochs=getattr(getattr(config, 'training', None), 'max_epochs', 10),
        min_epochs=getattr(getattr(config, 'training', None), 'min_epochs', 1),
        gradient_clip_val=min(getattr(getattr(config, 'training', None), 'gradient_clip_val', 0.5), 1.0),  # 限制梯度裁剪值
        gradient_clip_algorithm='norm',  # 使用L2范数裁剪
        accumulate_grad_batches=getattr(getattr(config, 'training', None), 'accumulate_grad_batches', 1),
        check_val_every_n_epoch=getattr(getattr(config, 'training', None), 'check_val_every_n_epoch', 1),
        log_every_n_steps=getattr(getattr(config, 'training', None), 'log_every_n_steps', 50),
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

    try:
        logger = TensorBoardLogger(
            save_dir=config.logging.save_dir,
            name=experiment_name,
            version=version,
            default_hp_metric=False
        )
    except Exception as e:
        try:
            from pytorch_lightning.loggers import CSVLogger
            logger = CSVLogger(save_dir=config.logging.save_dir, name=experiment_name, version=version)
            print(f"[警告] TensorBoard 不可用，已回退到 CSVLogger：{e}")
        except Exception as e2:
            print(f"[警告] 所有日志器初始化失败，将禁用训练日志：{e2}")
            class _NullLogger:
                def __init__(self):
                    self.experiment = type('E', (), {'add_scalar': lambda *args, **kwargs: None, 'add_histogram': lambda *args, **kwargs: None})()
                def log_hyperparams(self, *args, **kwargs):
                    pass
            logger = _NullLogger()
    
    # 记录超参数到TensorBoard（容错）
    try:
        lr_val = float(getattr(getattr(getattr(config, 'training', None), 'optimizer', None), 'lr', 1e-4))
    except Exception:
        lr_val = 1e-4
    try:
        max_epochs = int(getattr(getattr(config, 'training', None), 'max_epochs', 10))
    except Exception:
        max_epochs = 10
    try:
        opt_name = str(getattr(getattr(getattr(config, 'training', None), 'optimizer', None), 'name', 'adamw'))
    except Exception:
        opt_name = 'adamw'
    try:
        prec = getattr(getattr(config, 'training', None), 'precision', '16-mixed')
    except Exception:
        prec = '16-mixed'
    try:
        sched_name = str(getattr(getattr(getattr(config, 'training', None), 'scheduler', None), 'name', 'none'))
    except Exception:
        sched_name = 'none'
    hparams = {
        'learning_rate': lr_val,
        'batch_size': int(getattr(getattr(config, 'data', None), 'batch_size', 32)),
        'max_epochs': max_epochs,
        'model_d_model': getattr(getattr(config, 'model', None), 'd_model', 32),
        'model_n_heads': getattr(getattr(config, 'model', None), 'n_heads', getattr(getattr(config, 'model', None), 'num_heads', 4)),
        'model_num_layers': getattr(getattr(config, 'model', None), 'num_layers', 4),
        'dropout': getattr(getattr(config, 'model', None), 'dropout', 0.1),
        'precision': prec,
        'optimizer_type': opt_name,
        'scheduler_type': sched_name,
    }
    
    # 记录超参数
    try:
        logger.log_hyperparams(hparams)
    except Exception:
        pass
    
    return logger


def _detect_dataset_dir(prepared_dir: Path) -> Tuple[Path, str]:
    """在 prepared_dir 下自动检测数据集子目录。

    返回 (数据集目录, 数据集名称)。
    规则：
    - 若 prepared_dir 下直接存在 fold_0，则认为 prepared_dir 已是数据集目录；名称取其末级目录名。
    - 否则在 prepared_dir 下查找包含 fold_0 的子目录；优先选择存在 device_name_to_id.json 的目录；若无则选第一个匹配目录。
    - 若未找到匹配，回退到 prepared_dir 本身，名称为其末级名。
    """
    try:
        if (prepared_dir / 'fold_0').exists():
            return prepared_dir, prepared_dir.name
        candidates = []
        for p in prepared_dir.iterdir():
            if p.is_dir() and (p / 'fold_0').exists():
                candidates.append(p)
        if not candidates:
            return prepared_dir, prepared_dir.name
        # 优先含有设备映射文件的目录
        with_map = [p for p in candidates if (p / 'device_name_to_id.json').exists()]
        chosen = with_map[0] if with_map else candidates[0]
        return chosen, chosen.name
    except Exception:
        return prepared_dir, prepared_dir.name


def load_device_info(config: DictConfig) -> Tuple[Dict, List[str]]:
    """加载设备信息。
    若 prepared_dir 下存在设备映射文件，则优先使用该映射；否则回退到配置中的 data.device_names。
    """
    # 推断 prepared 数据目录
    try:
        prepared_dir = Path(getattr(config.paths, 'prepared_dir'))
    except Exception:
        prepared_dir = Path('Data/prepared')

    mapping_path = prepared_dir / 'device_name_to_id.json'
    device_names: List[str] = []
    # 优先使用映射文件
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
            print(f"[警告] 解析 {mapping_path} 失败：{e}")

    # 回退到配置中的 data.device_names
    if not device_names:
        try:
            cfg_names = list(getattr(getattr(config, 'data', {}), 'device_names', []) or [])
        except Exception:
            cfg_names = []
        device_names = cfg_names if cfg_names else ['device_1']
        if not cfg_names:
            print('[警告] 未能从配置或映射推断设备名称，使用占位 device_1')

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

    # 优先使用配置中的 dataset 参数；若未提供则自动检测
    try:
        base_prepared = Path(getattr(config.paths, 'prepared_dir', 'Data/prepared'))
        cfg_dataset = getattr(config, 'dataset', None)
        if isinstance(cfg_dataset, str) and cfg_dataset.strip():
            dataset_name = cfg_dataset.strip()
            dataset_dir = base_prepared / dataset_name
            if not dataset_dir.exists():
                print(f"[警告] 指定的数据集目录不存在：{dataset_dir}，将尝试自动检测。")
                dataset_dir, dataset_name = _detect_dataset_dir(base_prepared)
        else:
            dataset_dir, dataset_name = _detect_dataset_dir(base_prepared)
        # 更新配置中的 prepared_dir
        setattr(config.paths, 'prepared_dir', str(dataset_dir))
        print(f"[信息] 使用数据集目录：{dataset_dir}（dataset={dataset_name}）")
        # 为该数据集创建独立的检查点目录
        try:
            ckpt_base = Path(getattr(config.training.checkpoint, 'dirpath', output_dir / 'checkpoints'))
            ckpt_dir = ckpt_base if ckpt_base.is_absolute() else (output_dir / 'checkpoints')
            ckpt_dir = ckpt_dir / dataset_name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            setattr(config.training.checkpoint, 'dirpath', str(ckpt_dir))
        except Exception as e:
            print(f"[警告] 设置检查点目录失败：{e}")
    except Exception as e:
        print(f"[警告] 数据集目录处理失败：{e}")

    # 加载设备信息
    device_info, device_names = load_device_info(config)

    # 设置日志记录
    try:
        # 将数据集名称并入实验名，便于区分不同数据集的运行
        dataset_name = Path(getattr(config.paths, 'prepared_dir', 'Data/prepared')).name
    except Exception:
        dataset_name = 'prepared'
    experiment_name = f"{config.project_name}_{dataset_name}_{config.experiment.name}"
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