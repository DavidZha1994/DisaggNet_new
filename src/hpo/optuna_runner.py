import os
import sys
from typing import Optional, Dict, Any

import torch
import optuna
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from src.train import NILMLightningModule, load_device_info
from src.data.datamodule import NILMDataModule
from src.train import DeferredEarlyStopping

try:
    from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback as _PLPrunerBase
except Exception:
    from optuna.integration import PyTorchLightningPruningCallback as _PLPrunerBase


class OptunaPruningCallback(_PLPrunerBase, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def _clone_config(config: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(config, resolve=True))


def _load_space(space_config_path: Optional[str]) -> DictConfig:
    if space_config_path and os.path.exists(space_config_path):
        return OmegaConf.load(space_config_path)
    default_path = os.path.join("configs", "hpo", "space_optimized_stable.yaml")
    if os.path.exists(default_path):
        return OmegaConf.load(default_path)
    return OmegaConf.create({})


def _apply_trial(cfg: DictConfig, trial: optuna.Trial, space: Optional[DictConfig] = None) -> DictConfig:
    if not hasattr(cfg, "training"):
        cfg.training = OmegaConf.create({})
    if not hasattr(cfg.training, "optimizer"):
        cfg.training.optimizer = OmegaConf.create({})
    if not hasattr(cfg.training, "scheduler"):
        cfg.training.scheduler = OmegaConf.create({})
    if not hasattr(cfg, "data"):
        cfg.data = OmegaConf.create({})
    if not hasattr(cfg, "model"):
        cfg.model = OmegaConf.create({})
    if not hasattr(cfg.model, "time_encoder"):
        cfg.model.time_encoder = OmegaConf.create({})
    if not hasattr(cfg.model, "freq_encoder"):
        cfg.model.freq_encoder = OmegaConf.create({})
    if not hasattr(cfg.model, "fusion"):
        cfg.model.fusion = OmegaConf.create({})
    if not hasattr(cfg.model, "heads"):
        cfg.model.heads = OmegaConf.create({})
    if not hasattr(cfg.model.heads, "regression"):
        cfg.model.heads.regression = OmegaConf.create({})
    if not hasattr(cfg.training, "early_stopping"):
        cfg.training.early_stopping = OmegaConf.create({})
    if not hasattr(cfg, "evaluation"):
        cfg.evaluation = OmegaConf.create({})
    if not hasattr(cfg.training, "visualization"):
        cfg.training.visualization = OmegaConf.create({})

    if space and hasattr(space, "search_space") and hasattr(space.search_space, "training") and hasattr(space.search_space.training, "max_epochs"):
        me = space.search_space.training.max_epochs
        if isinstance(me, dict) and me.get("type") == "int":
            low = int(me.get("low", 18))
            high = int(me.get("high", 30))
            cfg.training.max_epochs = int(trial.suggest_int("max_epochs", low, high))
        else:
            cfg.training.max_epochs = int(getattr(cfg.training, "max_epochs", 25))
    else:
        cfg.training.max_epochs = int(min(int(getattr(cfg.training, "max_epochs", 25)), 25))
    cfg.training.min_epochs = int(getattr(cfg.training, "min_epochs", 1))
    cfg.training.check_val_every_n_epoch = int(getattr(cfg.training, "check_val_every_n_epoch", 1))
    cfg.training.log_every_n_steps = int(getattr(cfg.training, "log_every_n_steps", 10))
    cfg.training.early_stopping.enable = True
    cfg.training.early_stopping.start_epoch = 0
    cfg.training.early_stopping.patience = int(min(int(getattr(cfg.training.early_stopping, "patience", 5)), 3))
    cfg.training.early_stopping.min_delta = float(getattr(cfg.training.early_stopping, "min_delta", 0.0))
    cfg.training.early_stopping.monitor = str(getattr(cfg.training.early_stopping, "monitor", "val/loss/total"))
    cfg.training.visualization.enable = True
    cfg.training.visualization.interactive = True
    cfg.evaluation.test_after_training = False

    if space and hasattr(space, "search_space"):
        ss = space.search_space
        lr_def = getattr(getattr(getattr(ss, "training", None), "optimizer", None), "lr", None)
        wd_def = getattr(getattr(getattr(ss, "training", None), "optimizer", None), "weight_decay", None)
        clip_def = getattr(getattr(ss, "training", None), "gradient_clip_val", None)
        accum_def = getattr(getattr(ss, "training", None), "accumulate_grad_batches", None)
        bs_def = getattr(getattr(ss, "data", None), "batch_size", None)
        te_layers_def = getattr(getattr(getattr(ss, "model", None), "time_encoder", None), "num_layers", None)
        te_dropout_def = getattr(getattr(getattr(ss, "model", None), "time_encoder", None), "dropout", None)
        te_d_model_def = getattr(getattr(getattr(ss, "model", None), "time_encoder", None), "d_model", None)
        te_n_heads_def = getattr(getattr(getattr(ss, "model", None), "time_encoder", None), "n_heads", None)
        fusion_bidir_def = getattr(getattr(getattr(ss, "model", None), "fusion", None), "bidirectional", None)
        head_hidden_def = getattr(getattr(getattr(getattr(ss, "model", None), "heads", None), "regression", None), "hidden", None)
        reg_sp_def = getattr(getattr(getattr(getattr(ss, "model", None), "heads", None), "regression", None), "use_softplus", None)
        seq_sp_def = getattr(getattr(getattr(getattr(ss, "model", None), "heads", None), "regression", None), "seq_use_softplus", None)
        freq_proj_def = getattr(getattr(getattr(ss, "model", None), "freq_encoder", None), "proj_dim", None)
    else:
        lr_def = {"type": "float_log", "low": 1e-5, "high": 3e-3}
        wd_def = {"type": "float_log", "low": 1e-8, "high": 5e-3}
        clip_def = {"type": "float", "low": 0.0, "high": 1.0}
        accum_def = {"type": "choice", "values": [1, 2, 4]}
        bs_def = {"type": "choice", "values": [32, 64, 128]}
        te_layers_def = {"type": "int", "low": 2, "high": 4}
        te_dropout_def = {"type": "float", "low": 0.0, "high": 0.3}
        te_d_model_def = {"type": "choice", "values": [128, 256, 384]}
        te_n_heads_def = {"type": "choice", "values": [8]}
        fusion_bidir_def = {"type": "choice", "values": [False, True]}
        head_hidden_def = {"type": "choice", "values": [192, 256, 384]}
        reg_sp_def = {"type": "choice", "values": [False, True]}
        seq_sp_def = {"type": "choice", "values": [False, True]}
        freq_proj_def = {"type": "choice", "values": [192, 256, 384]}

    def _suggest(name, spec):
        t = spec.get("type")
        if t == "float_log":
            return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=True)
        if t == "float":
            return trial.suggest_float(name, float(spec["low"]), float(spec["high"]))
        if t == "int":
            return trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
        if t == "choice":
            return trial.suggest_categorical(name, list(spec["values"]))
        if t == "fixed":
            return spec["value"]
        return None

    lr = _suggest("lr", lr_def)
    wd = _suggest("weight_decay", wd_def)
    clip = _suggest("gradient_clip_val", clip_def)
    accum = _suggest("accumulate_grad_batches", accum_def)
    bs = _suggest("batch_size", bs_def)
    te_layers = _suggest("time_layers", te_layers_def)
    te_dropout = _suggest("time_dropout", te_dropout_def)
    te_d_model = _suggest("time_d_model", te_d_model_def)
    te_n_heads = _suggest("time_n_heads", te_n_heads_def)
    fusion_bidir = _suggest("fusion_bidirectional", fusion_bidir_def)
    head_hidden = _suggest("head_hidden", head_hidden_def)
    freq_proj = _suggest("freq_proj_dim", freq_proj_def)
    reg_softplus = _suggest("reg_use_softplus", reg_sp_def)
    seq_softplus = _suggest("seq_use_softplus", seq_sp_def)

    cfg.training.optimizer.name = "adamw"
    cfg.training.optimizer.lr = float(lr)
    cfg.training.optimizer.weight_decay = float(wd)
    cfg.training.gradient_clip_val = float(clip)
    cfg.training.accumulate_grad_batches = int(accum)
    cfg.data.batch_size = int(bs)
    cfg.model.time_encoder.d_model = int(te_d_model)
    cfg.model.time_encoder.n_heads = int(te_n_heads)
    cfg.model.time_encoder.num_layers = int(te_layers)
    cfg.model.time_encoder.dropout = float(te_dropout)
    cfg.model.time_encoder.input_conv_embed = bool(getattr(cfg.model.time_encoder, "input_conv_embed", True))
    cfg.model.time_encoder.causal_mask = bool(getattr(cfg.model.time_encoder, "causal_mask", True))
    cfg.model.freq_encoder.enable = bool(getattr(cfg.model.freq_encoder, "enable", True))
    cfg.model.freq_encoder.proj_dim = int(freq_proj)
    cfg.model.fusion.type = "cross_attention"
    cfg.model.fusion.bidirectional = bool(fusion_bidir)
    cfg.model.heads.regression.hidden = int(head_hidden)
    cfg.model.heads.regression.use_softplus = bool(reg_softplus)
    cfg.model.heads.regression.seq_use_softplus = bool(seq_softplus)
    if sys.platform == "darwin":
        prec = str(getattr(cfg.training, "precision", "32-true"))
        if ("bf16" in prec) or ("16" in prec):
            cfg.training.precision = "32-true"
    if hasattr(space, "stability") and bool(getattr(space.stability, "use_efficient_attention", True)):
        pass
    try:
        study = "hpo"
        if hasattr(space, "optuna") and hasattr(space.optuna, "study_name"):
            study = str(getattr(space.optuna, "study_name"))
        vis_dir = os.path.join("outputs", "viz", "hpo", study, f"trial_{trial.number}")
        cfg.training.visualization.save_dir = vis_dir
    except Exception:
        pass
    return cfg


def _build_trainer(cfg: DictConfig, logger: TensorBoardLogger, trial: Optional[optuna.Trial]) -> pl.Trainer:
    monitor = str(getattr(getattr(cfg.training, "early_stopping", None), "monitor", "val/loss/total"))
    mode = str(getattr(getattr(cfg.training, "early_stopping", None), "mode", "min"))
    patience = int(getattr(getattr(cfg.training, "early_stopping", None), "patience", 3))
    min_delta = float(getattr(getattr(cfg.training, "early_stopping", None), "min_delta", 0.0))
    start_epoch = int(getattr(getattr(cfg.training, "early_stopping", None), "start_epoch", 0))
    callbacks = [
        ModelCheckpoint(monitor=monitor, mode=mode, save_top_k=1, filename="best-model"),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if bool(getattr(getattr(cfg.training, "early_stopping", None), "enable", True)):
        callbacks.append(DeferredEarlyStopping(monitor=monitor, patience=patience, mode=mode, min_delta=min_delta, start_epoch=start_epoch))
    if trial is not None:
        callbacks.append(OptunaPruningCallback(trial, monitor))
    trainer = pl.Trainer(
        accelerator=getattr(cfg.training, "accelerator", "auto"),
        devices=getattr(cfg.training, "devices", 1),
        precision=getattr(cfg.training, "precision", 32),
        min_epochs=int(getattr(cfg.training, "min_epochs", 1)),
        max_epochs=int(getattr(cfg.training, "max_epochs", 10)),
        logger=logger,
        callbacks=callbacks,
        check_val_every_n_epoch=int(getattr(cfg.training, "check_val_every_n_epoch", 1)),
        log_every_n_steps=int(getattr(cfg.training, "log_every_n_steps", 10)),
        enable_checkpointing=True,
    )
    return trainer


def _objective_factory(base_config: DictConfig, space: Optional[DictConfig]):
    def objective(trial: optuna.Trial) -> float:
        cfg = _clone_config(base_config)
        cfg = _apply_trial(cfg, trial, space)
        pl.seed_everything(int(getattr(cfg, "seed", getattr(getattr(cfg, "reproducibility", None), "seed", 42))))
        dm = NILMDataModule(cfg)
        try:
            dm.setup("fit")
        except Exception:
            pass
        device_info, device_names = load_device_info(cfg)
        if space and hasattr(space, "logging"):
            save_dir = str(getattr(space.logging, "save_dir", "logs/tensorboard/hpo"))
            name = str(getattr(space.logging, "name", "hpo"))
        else:
            save_dir = "logs/tensorboard/hpo"
            name = "hpo"
        version = f"trial_{trial.number}"
        logger = TensorBoardLogger(save_dir, name=name, version=version)
        module = NILMLightningModule(cfg, device_info, device_names)
        trainer = _build_trainer(cfg, logger, trial)
        trainer.fit(module, dm)
        metrics = getattr(trainer, "callback_metrics", {}) or {}
        monitor = str(getattr(getattr(cfg.training, "early_stopping", None), "monitor", "val/loss/total"))
        val = None
        if monitor in metrics:
            val = metrics[monitor]
        elif "val/loss/total" in metrics:
            val = metrics["val/loss/total"]
        elif "val_loss" in metrics:
            val = metrics["val_loss"]
        if isinstance(val, torch.Tensor):
            return float(val.detach().cpu().item())
        if isinstance(val, (int, float)):
            return float(val)
        return float("inf")
    return objective


def run_optimization(config: DictConfig, n_trials: int = 20, timeout: Optional[int] = None, study_name: Optional[str] = None, storage: Optional[str] = None, space_config: Optional[str] = None) -> Dict[str, Any]:
    if sys.platform == "darwin":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    space = _load_space(space_config)
    direction = "minimize"
    warm = 2
    if hasattr(space, "optuna") and hasattr(space.optuna, "warmup_steps"):
        warm = int(getattr(space.optuna, "warmup_steps", 2))
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=warm)
    if storage is None and hasattr(space, "optuna") and hasattr(space.optuna, "storage"):
        storage = str(getattr(space.optuna, "storage"))
    if study_name is None and hasattr(space, "optuna") and hasattr(space.optuna, "study_name"):
        study_name = str(getattr(space.optuna, "study_name"))
    objective = _objective_factory(config, space)
    if storage:
        os.makedirs(os.path.dirname(storage.replace("sqlite:///", "")), exist_ok=True)
    study = optuna.create_study(direction=direction, study_name=study_name, storage=storage, load_if_exists=True, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    best = {"value": study.best_value, "params": study.best_trial.params, "number": study.best_trial.number}
    return best
