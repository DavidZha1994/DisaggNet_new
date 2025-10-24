import os
import sys
from pathlib import Path
import pytorch_lightning as pl
from omegaconf import OmegaConf

# Ensure project root in path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.train import NILMLightningModule, setup_logging, load_device_info
from src.data.datamodule import NILMDataModule


def main():
    # Load config
    cfg_path = project_root / "configs" / "optimized_stable.yaml"
    config = OmegaConf.load(str(cfg_path))

    # Ensure visualization enabled
    try:
        if not getattr(getattr(config, 'training', None), 'visualization', None):
            config.training.visualization = OmegaConf.create({"enable": True, "max_plots_per_epoch": 8})
        else:
            config.training.visualization.enable = True
    except Exception:
        pass

    # Prepare output and logger
    output_dir = Path(getattr(getattr(config, 'paths', None), 'output_dir', project_root / 'output'))
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_name = f"{getattr(config, 'project_name', 'DisaggNet')}_{getattr(getattr(config, 'experiment', None), 'name', 'quickval')}"
    logger = setup_logging(config, experiment_name)

    # Data and model
    device_info, device_names = load_device_info(config)
    datamodule = NILMDataModule(config)
    datamodule.setup()
    model = NILMLightningModule(config, device_info, device_names)

    # Trainer: limit validation to 2 batches, disable sanity steps
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=1,
        limit_val_batches=2,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        deterministic=getattr(getattr(config, 'reproducibility', None), 'deterministic', True),
        benchmark=getattr(getattr(config, 'reproducibility', None), 'benchmark', False),
    )

    print("Running quick validation (limit_val_batches=2)...")
    trainer.validate(model, datamodule=datamodule)
    print("Quick validation done. Check TensorBoard Images tab for 'val/sample_*' and 'val/sample_total_*'.")


if __name__ == "__main__":
    main()