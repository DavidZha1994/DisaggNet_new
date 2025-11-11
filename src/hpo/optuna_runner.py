import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import yaml
import optuna
from omegaconf import DictConfig, OmegaConf
from copy import deepcopy

# 延迟导入训练主函数以避免循环引用问题
from src.train import main as train_main

logger = logging.getLogger(__name__)


def _apply_trial_params_to_config(trial: optuna.Trial, cfg: DictConfig) -> Dict[str, Any]:
    """根据试验建议更新配置，并返回建议参数字典。

    仅选择与稳定训练相关、对性能影响显著且在 macOS MPS 下安全的超参数。
    """
    params: Dict[str, Any] = {}

    # 优化器超参数
    params["training.optimizer.lr"] = trial.suggest_float("training.optimizer.lr", 5e-6, 5e-4, log=True)
    cfg.training.optimizer.lr = params["training.optimizer.lr"]

    params["training.optimizer.weight_decay"] = trial.suggest_float("training.optimizer.weight_decay", 1e-6, 2e-2, log=True)
    cfg.training.optimizer.weight_decay = params["training.optimizer.weight_decay"]

    # 优化器类型
    params["training.optimizer.name"] = trial.suggest_categorical("training.optimizer.name", ["adam", "adamw"])
    cfg.training.optimizer.name = params["training.optimizer.name"]

    # 梯度与累计
    params["training.gradient_clip_val"] = trial.suggest_float("training.gradient_clip_val", 0.5, 2.0)
    cfg.training.gradient_clip_val = params["training.gradient_clip_val"]

    params["training.accumulate_grad_batches"] = trial.suggest_categorical("training.accumulate_grad_batches", [1, 2])
    cfg.training.accumulate_grad_batches = params["training.accumulate_grad_batches"]

    # 批大小（受内存限制）
    params["data.batch_size"] = trial.suggest_categorical("data.batch_size", [8, 12, 16])
    cfg.data.batch_size = params["data.batch_size"]

    # 模型结构（MPS 友好范围）
    params["model.d_model"] = trial.suggest_categorical("model.d_model", [192, 256])
    cfg.model.d_model = params["model.d_model"]
    if hasattr(cfg.model, "time_encoder"):
        cfg.model.time_encoder.d_model = params["model.d_model"]

    params["model.n_heads"] = trial.suggest_categorical("model.n_heads", [4, 8])
    cfg.model.n_heads = params["model.n_heads"]
    if hasattr(cfg.model, "time_encoder"):
        cfg.model.time_encoder.n_heads = params["model.n_heads"]

    params["model.dropout"] = trial.suggest_float("model.dropout", 0.1, 0.3)
    cfg.model.dropout = params["model.dropout"]
    if hasattr(cfg.model, "time_encoder"):
        cfg.model.time_encoder.dropout = params["model.dropout"]

    # 层数
    params["model.num_layers"] = trial.suggest_int("model.num_layers", 4, 10)
    cfg.model.num_layers = params["model.num_layers"]
    if hasattr(cfg.model, "time_encoder"):
        cfg.model.time_encoder.num_layers = params["model.num_layers"]

    # 频域编码器
    params["model.freq_encoder.enable"] = trial.suggest_categorical("model.freq_encoder.enable", [True, False])
    cfg.model.freq_encoder.enable = params["model.freq_encoder.enable"]
    if cfg.model.freq_encoder.enable:
        params["model.freq_encoder.proj_dim"] = trial.suggest_categorical("model.freq_encoder.proj_dim", [96, 128, 160])
        cfg.model.freq_encoder.proj_dim = params["model.freq_encoder.proj_dim"]

        params["model.freq_encoder.small_transformer_layers"] = trial.suggest_int("model.freq_encoder.small_transformer_layers", 0, 2)
        # 已移除小型Transformer层，保持为0
        cfg.model.freq_encoder.small_transformer_layers = 0

        params["model.freq_encoder.dropout"] = trial.suggest_float("model.freq_encoder.dropout", 0.05, 0.2)
        cfg.model.freq_encoder.dropout = params["model.freq_encoder.dropout"]

    # 回归头
    params["model.heads.regression.hidden"] = trial.suggest_categorical("model.heads.regression.hidden", [64, 96, 128])
    cfg.model.heads.regression.hidden = params["model.heads.regression.hidden"]

    params["model.heads.regression.dropout"] = trial.suggest_float("model.heads.regression.dropout", 0.05, 0.25)
    cfg.model.heads.regression.dropout = params["model.heads.regression.dropout"]

    # 调度器及预热
    params["training.scheduler.name"] = trial.suggest_categorical("training.scheduler.name", ["cosine", "onecycle", "none"])
    cfg.training.scheduler.name = params["training.scheduler.name"]
    params["training.scheduler.warmup_steps"] = trial.suggest_int("training.scheduler.warmup_steps", 0, 2000)
    cfg.training.scheduler.warmup_steps = params["training.scheduler.warmup_steps"]

    # —— 损失函数相关（与 NILMLoss 对齐）——
    if not hasattr(cfg, "loss"):
        cfg.loss = OmegaConf.create({})
    params["loss.huber_delta"] = trial.suggest_float("loss.huber_delta", 0.5, 2.0)
    cfg.loss.huber_delta = params["loss.huber_delta"]

    params["loss.active_threshold_rel"] = trial.suggest_float("loss.active_threshold_rel", 0.01, 0.10)
    cfg.loss.active_threshold_rel = params["loss.active_threshold_rel"]

    params["loss.off_penalty_weight"] = trial.suggest_float("loss.off_penalty_weight", 0.0, 0.5)
    cfg.loss.off_penalty_weight = params["loss.off_penalty_weight"]

    # 仅在 NILMLoss 中可选使用
    params["loss.rel_loss_weight"] = trial.suggest_float("loss.rel_loss_weight", 1.0, 2.5)
    cfg.loss.rel_loss_weight = params["loss.rel_loss_weight"]

    params["loss.classification_loss_type"] = trial.suggest_categorical("loss.classification_loss_type", ["focal", "bce"])
    cfg.loss.classification_loss_type = params["loss.classification_loss_type"]
    if cfg.loss.classification_loss_type == "focal":
        params["loss.focal_alpha"] = trial.suggest_float("loss.focal_alpha", 0.25, 0.75)
        cfg.loss.focal_alpha = params["loss.focal_alpha"]
        params["loss.focal_gamma"] = trial.suggest_float("loss.focal_gamma", 1.0, 3.0)
        cfg.loss.focal_gamma = params["loss.focal_gamma"]

    # —— 评估阈值方法 ——
    if not hasattr(cfg, "evaluation"):
        cfg.evaluation = OmegaConf.create({})
    params["evaluation.threshold_method"] = trial.suggest_categorical("evaluation.threshold_method", ["f1", "youden", "optimal"])
    cfg.evaluation.threshold_method = params["evaluation.threshold_method"]

    return params


def _objective_builder(base_config: DictConfig, hpo_dir: Path, trial_epochs: int = 1):
    """创建 Optuna 目标函数，基于 best_val_loss 进行最小化。"""

    def objective(trial: optuna.Trial) -> float:
        # 深拷贝配置，避免污染原配置
        cfg: DictConfig = OmegaConf.create(OmegaConf.to_container(base_config, resolve=True))

        # 将试验编号注入实验名，便于输出文件检索
        if not hasattr(cfg, "experiment"):
            cfg.experiment = OmegaConf.create({"name": f"optuna_trial_{trial.number}"})
        else:
            cfg.experiment.name = f"optuna_trial_{trial.number}"

        # 控制试验轮数与日志/可视化，提升试验效率
        try:
            cfg.training.max_epochs = min(getattr(cfg.training, "max_epochs", 100), max(1, trial_epochs))
            if hasattr(cfg.training, "min_epochs"):
                cfg.training.min_epochs = 1
            cfg.training.check_val_every_n_epoch = 1
            if hasattr(cfg.training, "visualization"):
                cfg.training.visualization.interactive = False
                cfg.training.visualization.enable = False
        except Exception:
            pass

        # 应用试验参数
        params = _apply_trial_params_to_config(trial, cfg)

        # 保存试验配置以便复现
        trial_cfg_dir = hpo_dir / "trial_configs"
        trial_cfg_dir.mkdir(parents=True, exist_ok=True)
        with open(trial_cfg_dir / f"trial_{trial.number}.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(OmegaConf.to_container(cfg, resolve=True), f, allow_unicode=True)

        # 运行训练
        try:
            train_main(cfg)
        except Exception as e:
            logger.error(f"试验 {trial.number} 训练失败: {e}")
            # 返回较大损失，避免影响搜索过程
            return float("inf")

        # 检索输出结果文件（形如 outputs/DisaggNet_<dataset>_optuna_trial_N_results.yaml）
        outputs_dir = Path(getattr(cfg.paths, "output_dir", "outputs"))
        result_file: Optional[Path] = None
        pattern = f"{cfg.project_name}_*_optuna_trial_{trial.number}_results.yaml"
        for p in outputs_dir.glob(pattern):
            result_file = p
            break

        if result_file is None:
            # 兜底：在整个 outputs 下搜索
            matches = list(outputs_dir.glob(f"**/*optuna_trial_{trial.number}_results.yaml"))
            if matches:
                result_file = matches[0]

        if result_file is None or not result_file.exists():
            logger.warning(f"未找到试验 {trial.number} 的结果文件，返回高损失")
            return float("inf")

        with open(result_file, "r", encoding="utf-8") as f:
            results = yaml.safe_load(f) or {}

        best_val = results.get("best_val_loss")
        if best_val is None:
            # 兼容可能的字段名
            best_val = results.get("best_loss") or results.get("val_loss")
        if best_val is None:
            logger.warning(f"试验 {trial.number} 结果文件缺少 best_val_loss 字段，返回高损失")
            return float("inf")

        # 记录试验结果与参数
        trial_result_dir = hpo_dir / "trial_results"
        trial_result_dir.mkdir(parents=True, exist_ok=True)
        with open(trial_result_dir / f"trial_{trial.number}_result.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump({"params": params, "best_val_loss": float(best_val)}, f, allow_unicode=True)

        return float(best_val)

    return objective


def _merge_best_params(base_cfg: DictConfig, best_params: Dict[str, Any]) -> DictConfig:
    """将最佳参数应用到基础配置，返回新的配置。"""
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    for k, v in best_params.items():
        try:
            # 支持嵌套键（以点号分割）
            keys = k.split(".")
            target = cfg
            for key in keys[:-1]:
                if not hasattr(target, key):
                    setattr(target, key, OmegaConf.create({}))
                target = getattr(target, key)
            setattr(target, keys[-1], v)
        except Exception:
            pass
    return cfg


def run_optimization(config: DictConfig, n_trials: int = 50, timeout: Optional[int] = None, study_name: Optional[str] = None) -> Dict[str, Any]:
    """运行 Optuna 超参数优化，返回优化摘要。

    - 使用 TPE 采样器最小化验证损失（best_val_loss）。
    - 每个试验运行较少 epoch（默认 1）以快速探索。
    - 输出：最佳参数与合并配置保存至 outputs/hpo。
    """

    # 输出目录与 HPO 目录
    outputs_dir = Path(getattr(config.paths, "output_dir", "outputs"))
    hpo_dir = outputs_dir / "hpo"
    hpo_dir.mkdir(parents=True, exist_ok=True)

    # 创建/加载 study（持久化到 SQLite）
    study_name = study_name or f"{getattr(config, 'project_name', 'DisaggNet')}_optuna"
    storage = f"sqlite:///{hpo_dir / 'optuna_studies.db'}"
    sampler = optuna.samplers.TPESampler(multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=2, n_startup_trials=2)

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    # 构建目标函数（快速试验：1 epoch）
    objective = _objective_builder(config, hpo_dir, trial_epochs=1)

    logger.info(f"开始 Optuna 超参优化：trials={n_trials}, study='{study_name}'")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    if not study.best_trial:
        logger.warning("Optuna 未获得有效试验结果")
        return {"success": False, "message": "No valid trials"}

    best_params = study.best_trial.params
    best_value = study.best_value

    # 仅保存最佳参数，不生成新的配置文件（基于现有 optimized_stable.yaml 调参）
    best_params_path = hpo_dir / "best_params.yaml"
    with open(best_params_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(best_params, f, allow_unicode=True)

    summary = {
        "success": True,
        "study_name": study_name,
        "best_value": float(best_value),
        "best_params_path": str(best_params_path),
        "best_params": best_params,
    }

    logger.info(f"Optuna 完成：best_val_loss={best_value:.6f}, 保存至 {best_params_path}")
    return summary