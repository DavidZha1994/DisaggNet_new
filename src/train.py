"""训练脚本 (已优化：适配简化版回归模型)"""

import os
import sys
if sys.platform == 'darwin':
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


from .data.datamodule import NILMDataModule
from .models.fusion_transformer import FusionTransformer
from .losses.losses import create_loss_function, RECOMMENDED_LOSS_CONFIGS
from .utils.metrics import NILMMetrics
from .utils.viz import save_validation_interactive_plot

# 启用TF32优化（放在所有导入之后，避免 E402）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")


class NILMLightningModule(pl.LightningModule):
    """NILM PyTorch Lightning模块 (简化版)"""

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

        # 可视化配置
        self.best_val_loss = float('inf')
        self.best_thresholds = {}
        vis_cfg = getattr(getattr(config, 'training', None), 'visualization', None)
        self.enable_visualization = bool(getattr(vis_cfg, 'enable', False) if vis_cfg is not None else False)
        self.max_plots_per_epoch = int(getattr(vis_cfg, 'max_plots_per_epoch', 8))
        self.vis_output_dir = str(getattr(vis_cfg, 'save_dir', Path('outputs') / 'viz'))
        self.enable_interactive = bool(getattr(vis_cfg, 'interactive', False) if vis_cfg is not None else False)
        self._val_buffers: List[Dict[str, Any]] = []

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
        except Exception as e:
            print(f"[Warning] Failed to load scales: {e}")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, None]:
        """前向传播 (Inference)"""
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

        mask_seq = batch.get('target_seq_valid_mask', None)
        device = pred_seq.device
        mp = self._ensure_max_power(device, self.n_devices)
        pred_seq_norm = torch.clamp(pred_seq / mp, 0.0, 1.0)
        if isinstance(mask_seq, torch.Tensor):
            m = mask_seq
            if m.dim() + 1 == pred_seq_norm.dim():
                m = m.unsqueeze(-1)
            m = (m > 0).float()
            num = (pred_seq_norm * m).sum(dim=1)
            den = m.sum(dim=1).clamp_min(1.0)
            pred_power_norm = num / den
        else:
            pred_power_norm = pred_seq_norm.mean(dim=1)
        pred_power_watts = pred_power_norm * mp.view(1, -1)
        return pred_power_watts, None

    def _forward_and_compute_loss(self, batch: Dict[str, torch.Tensor], stage: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算联合损失"""
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
        else:
            pred_seq = out

        target_seq = batch.get('target_seq')
        if target_seq is None:
            return pred_seq, torch.tensor(0.0, device=device)

        valid = batch.get('target_seq_valid_mask')
        if isinstance(valid, torch.Tensor) and valid.dim() + 1 == pred_seq.dim():
            valid = valid.unsqueeze(-1)
        valid = (valid > 0) if valid is not None else torch.ones_like(pred_seq, dtype=torch.bool)

        pred_seq_norm = torch.clamp(pred_seq / mp, 0.0, 1.0)
        loss_reg = self.loss_fn.regression_seq_loss(
            pred_seq_norm, target_seq,
            status_seq=batch.get('status_seq'),
            valid_mask=valid,
            power_scale=None
        )

        pred_seq_watts = pred_seq_norm * mp
        loss_cons = self.loss_fn.conservation_loss(
            batch.get('mains_seq'), pred_seq_watts
        )

        total_loss = loss_reg + loss_cons

        self._safe_log(f'{stage}/loss/total', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self._safe_log(f'{stage}/loss/regression', loss_reg, on_step=False, on_epoch=True)
        self._safe_log(f'{stage}/loss/conservation', loss_cons, on_step=False, on_epoch=True)

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
                mains_seq = batch.get('mains_seq')
                if isinstance(pred_seq, torch.Tensor) and isinstance(target_seq, torch.Tensor):
                    b = int(pred_seq.size(0))
                    for i in range(b):
                        device = pred_seq.device
                        mp3 = self._ensure_max_power(device, self.n_devices)  # (1,1,K)
                        mp2 = mp3.view(1, -1)  # (1,K)
                        # 归一化到 [0,1] 后再按最大功率反归一化为瓦特
                        p_norm = torch.clamp(pred_seq[i] / mp2, 0.0, 1.0)
                        t_norm = target_seq[i]
                        p_w = (p_norm * mp2).detach().cpu().numpy().astype('float32')  # (L,K)
                        t_w = (torch.clamp(t_norm, 0.0, 1.0) * mp2).detach().cpu().numpy().astype('float32')  # (L,K)
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
                        self._val_buffers.append({'pred': torch.tensor(p_w), 'true': torch.tensor(t_w), 'mains': m, 'start': start_ts, 'step': step})
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

            if pred0.dim() == 3:
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
                # 已是 (B, K) 的窗口级功率（应为瓦特）
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

            for k, v in res.items():
                self._safe_log(f'{stage}/metrics/{k}', v, on_epoch=True)

            return res
        except Exception:
            return {}

    def configure_optimizers(self):
        """配置优化器"""
        opt_cfg = self.config.training.optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss/total",
                "interval": "epoch"
            },
        }


# ------------------------------------------------------------------------
# Main Helper Functions (Keep largely same but clean)
# ------------------------------------------------------------------------

def load_device_info(config: DictConfig) -> Tuple[Dict, List[str]]:
    """加载设备信息"""
    try:
        names = list(config.data.device_names)
    except Exception:
        names = ['device_1']

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

    logger = TensorBoardLogger("outputs", name="nilm_experiment")

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
        max_epochs=config.training.max_epochs,
        logger=logger,
        callbacks=[
            ModelCheckpoint(monitor='val/loss/total', mode='min', save_top_k=1, filename='best-model'),
            EarlyStopping(monitor='val/loss/total', patience=5, mode='min'),
            LearningRateMonitor(logging_interval='epoch')
        ],
        check_val_every_n_epoch=1,
        log_every_n_steps=10
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
    mc = ModelCheckpoint(monitor='val/loss/total', mode='min', save_top_k=1, filename='best-model')
    es = EarlyStopping(monitor='val/loss/total', patience=5, mode='min')
    lrmon = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(
        accelerator=getattr(config.training, 'accelerator', 'auto'),
        devices=getattr(config.training, 'devices', 1),
        precision=getattr(config.training, 'precision', 32),
        max_epochs=config.training.max_epochs,
        logger=logger,
        callbacks=[mc, es, lrmon],
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        enable_checkpointing=True,
    )
    return trainer
