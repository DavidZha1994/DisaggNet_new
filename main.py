#!/usr/bin/env python3
"""
DisaggNet 统一训练入口 - 主程序
整合所有训练功能模块，提供统一的命令行接口

功能包括：
- 基础训练
- 超参数优化 (HPO)
- Walk-Forward验证
- 模型评估和推理
- 实验管理

使用方法：
    python main.py train --config-name=default
    python main.py eval --checkpoint outputs/checkpoints/best_model.pth
python main.py infer --checkpoint outputs/checkpoints/best_model.pth --input data/test.csv
    python main.py hpo --n-trials=50
    python main.py walk-forward --n-folds=5

注意：所有训练相关操作都应通过此统一入口执行，不要直接运行单独的脚本文件。
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# 导入项目模块
from src.train import main as train_main
from src.eval import main as eval_main
from src.infer import main as infer_main
from src.walk_forward import run_walk_forward_validation as walk_forward_main
# 延迟导入 HPO 组件，避免在未使用时因可选依赖报错
optuna_main = None

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnifiedTrainingSystem:
    """统一训练系统"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.configs_dir = self.project_root / "configs"
        self.checkpoints_dir = self.project_root / "checkpoints"
        self.logs_dir = self.project_root / "logs"
        
        # 确保目录存在
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
    
    def setup_environment(self, config_name: str = "optimized_stable") -> DictConfig:
        """设置训练环境（合并 base 与目标配置）"""
        # 加载配置
        config_path = self.configs_dir / f"{config_name}.yaml"
        if not config_path.exists():
            logger.warning(f"配置文件 {config_path} 不存在，使用默认配置")
            config_path = self.configs_dir / "default.yaml"

        # 手工合并 base.yaml 与目标配置，避免仅 OmegaConf.load 无法解析 Hydra defaults
        base_config_path = self.configs_dir / "base.yaml"
        base_cfg = OmegaConf.load(base_config_path) if base_config_path.exists() else OmegaConf.create({})
        user_cfg = OmegaConf.load(config_path)
        config = OmegaConf.merge(base_cfg, user_cfg)
        
        # 设置随机种子（优先使用 reproducibility.seed，其次回退到顶层 seed，再默认 42）
        if hasattr(config, 'reproducibility') and hasattr(config.reproducibility, 'seed'):
            seed = config.reproducibility.seed
        elif hasattr(config, 'seed'):
            seed = config.seed
        else:
            seed = 42
            
        pl.seed_everything(seed, workers=True)
        
        # 设置PyTorch
        torch.backends.cudnn.deterministic = getattr(config.reproducibility, 'deterministic', True)
        torch.backends.cudnn.benchmark = getattr(config.reproducibility, 'benchmark', False)
        
        # 启用异常检测（如果配置）
        if hasattr(config, 'stability') and hasattr(config.stability, 'numerical'):
            if getattr(config.stability.numerical, 'detect_anomaly', False):
                torch.autograd.set_detect_anomaly(True)
                logger.info("已启用PyTorch异常检测")
        
        logger.info(f"环境设置完成，使用配置: {config_name}")
        return config
    
    def train_basic(self, config_name: str = "optimized_stable", **kwargs) -> None:
        """基础训练"""
        logger.info(f"开始基础训练，配置: {config_name}")
        
        try:
            config = self.setup_environment(config_name)
            
            # 更新配置参数（支持嵌套项覆盖）
            for key, value in kwargs.items():
                try:
                    if key == 'epochs':
                        config.training.max_epochs = int(value)
                        # 保证最小轮数不超过最大轮数
                        if hasattr(config.training, 'min_epochs'):
                            config.training.min_epochs = min(int(getattr(config.training, 'min_epochs', 1)), int(value))
                        logger.info(f"更新配置参数: training.max_epochs = {value}")
                    elif key == 'batch_size':
                        config.data.batch_size = int(value)
                        logger.info(f"更新配置参数: data.batch_size = {value}")
                    elif key == 'lr':
                        config.training.optimizer.lr = float(value)
                        logger.info(f"更新配置参数: training.optimizer.lr = {value}")
                    else:
                        # 顶层兜底
                        if hasattr(config, key):
                            setattr(config, key, value)
                            logger.info(f"更新配置参数: {key} = {value}")
                except Exception as e:
                    logger.warning(f"更新配置参数失败 {key}={value}: {e}")
            
            # 调用训练函数
            train_main(config)
            logger.info("基础训练完成")
            
        except Exception as e:
            logger.error(f"基础训练失败: {e}")
            raise
    
    def optimize_hyperparameters(self, 
                                config_name: str = "optimized_stable",
                                n_trials: int = 50,
                                timeout: Optional[int] = None,
                                **kwargs) -> None:
        """超参数优化"""
        logger.info(f"开始超参数优化，试验次数: {n_trials}")
        config = self.setup_environment(config_name)
        
        # 构建参数
        args = [
            "--config", str(self.configs_dir / f"{config_name}.yaml"),
            "--n-trials", str(n_trials)
        ]
        
        if timeout:
            args.extend(["--timeout", str(timeout)])
        
        # 添加额外参数
        for key, value in kwargs.items():
            if value is not None:
                args.extend([f"--{key}", str(value)])
        
        # 调用HPO
        optuna_main(config, n_trials=n_trials, timeout=timeout)
    
    def walk_forward_validation(self, 
                               config_name: str = "optimized_stable",
                               n_folds: int = 5,
                               **kwargs) -> None:
        """Walk-Forward验证"""
        logger.info(f"开始Walk-Forward验证，折叠数: {n_folds}")
        config = self.setup_environment(config_name)
        
        # 构建参数
        args = [
            "--config", str(self.configs_dir / f"{config_name}.yaml"),
            "--n-folds", str(n_folds)
        ]
        
        # 添加额外参数
        for key, value in kwargs.items():
            if value is not None:
                args.extend([f"--{key}", str(value)])
        
        # 调用Walk-Forward验证
        walk_forward_main(config, n_folds=n_folds)
    
    def evaluate_model(self, 
                      checkpoint_path: str,
                      config_name: str = "optimized_stable",
                      output_dir: Optional[str] = None,
                      **kwargs) -> None:
        """模型评估"""
        logger.info(f"开始模型评估，模型: {checkpoint_path}")
        
        try:
            config = self.setup_environment(config_name)
            
            # 设置输出目录
            if output_dir:
                output_path = Path(output_dir)
            else:
                output_path = Path(config.paths.output_dir) / "evaluation"
            
            # 调用评估函数
            results = eval_main(config, checkpoint_path, output_path)
            logger.info("模型评估完成")
            return results
            
        except Exception as e:
            logger.error(f"模型评估失败: {e}")
            raise
    
    def inference(self, 
                 checkpoint_path: str,
                 data_path: str,
                 config_name: str = "optimized_stable",
                 output_dir: Optional[str] = None,
                 **kwargs) -> None:
        """模型推理"""
        logger.info(f"开始模型推理，模型: {checkpoint_path}, 数据: {data_path}")
        
        try:
            config = self.setup_environment(config_name)
            
            # 设置输出目录
            if output_dir:
                output_path = Path(output_dir)
            else:
                output_path = Path(config.paths.output_dir) / "inference"
            
            # 调用推理函数
            results = infer_main(config, checkpoint_path, data_path, output_path)
            logger.info("模型推理完成")
            return results
            
        except Exception as e:
            logger.error(f"模型推理失败: {e}")
            raise
    

    
    def stability_check(self, config_name: str = "optimized_stable") -> Dict[str, Any]:
        """训练稳定性检查"""
        logger.info("开始训练稳定性检查...")
        config = self.setup_environment(config_name)
        
        results = {
            "config_valid": True,
            "environment_ready": True,
            "stability_features": [],
            "recommendations": []
        }
        
        # 检查配置稳定性特性
        if hasattr(config, 'stability'):
            results["stability_features"].append("稳定性配置已启用")
            
            if hasattr(config.stability, 'numerical'):
                if getattr(config.stability.numerical, 'detect_anomaly', False):
                    results["stability_features"].append("异常检测已启用")
                if getattr(config.stability.numerical, 'gradient_clip_val', 0) > 0:
                    results["stability_features"].append("梯度裁剪已配置")
        
        # 检查精度设置
        if hasattr(config, 'compute'):
            precision = getattr(config.compute, 'precision', '16-mixed')
            if precision == '32':
                results["stability_features"].append("使用FP32精度")
            elif precision in ['16-mixed', 'bf16-mixed']:
                results["recommendations"].append("建议使用FP32精度提高稳定性")
        
        # 检查学习率
        if hasattr(config, 'training') and hasattr(config.training, 'optimizer'):
            lr = getattr(config.training.optimizer, 'lr', 1e-3)
            if lr > 1e-3:
                results["recommendations"].append("学习率可能过高，建议降低到1e-4以下")
        
        logger.info(f"稳定性检查完成: {results}")
        return results


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="DisaggNet统一训练入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基础训练
  python main.py --mode train --config configs/optimized_stable.yaml
  
  # 超参数优化
  python main.py --mode hpo --config configs/optimized_stable.yaml --trials 50
  
  # Walk-Forward验证
  python main.py --mode walk_forward --config configs/optimized_stable.yaml --n_splits 5
  
  # 模型评估
  python main.py --mode eval --checkpoint outputs/checkpoints/best_model.pth
  
  # 模型推理
  python main.py --mode infer --checkpoint outputs/checkpoints/best_model.pth --input data/test.csv
  
  # 稳定性检查
  python main.py --mode stability_check --config configs/optimized_stable.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 基础训练
    train_parser = subparsers.add_parser("train", help="基础训练")
    train_parser.add_argument("--config", default="optimized_stable", help="配置文件名")
    train_parser.add_argument("--epochs", type=int, help="训练轮数")
    train_parser.add_argument("--batch-size", type=int, help="批次大小")
    train_parser.add_argument("--lr", type=float, help="学习率")
    
    # 超参数优化
    hpo_parser = subparsers.add_parser("hpo", help="超参数优化")
    hpo_parser.add_argument("--config", default="optimized_stable", help="配置文件名")
    hpo_parser.add_argument("--n-trials", type=int, default=50, help="试验次数")
    hpo_parser.add_argument("--timeout", type=int, help="超时时间(秒)")
    hpo_parser.add_argument("--study-name", help="研究名称")
    
    # Walk-Forward验证
    wf_parser = subparsers.add_parser("walk-forward", help="Walk-Forward验证")
    wf_parser.add_argument("--config", default="optimized_stable", help="配置文件名")
    wf_parser.add_argument("--n-folds", type=int, default=5, help="折叠数")
    wf_parser.add_argument("--train-days", type=int, help="训练天数")
    wf_parser.add_argument("--val-days", type=int, help="验证天数")
    
    # 模型评估
    eval_parser = subparsers.add_parser("eval", help="模型评估")
    eval_parser.add_argument("--checkpoint", required=True, help="模型检查点路径")
    eval_parser.add_argument("--config", default="optimized_stable", help="配置文件名")
    eval_parser.add_argument("--output-dir", help="输出目录")
    
    # 模型推理
    infer_parser = subparsers.add_parser("infer", help="模型推理")
    infer_parser.add_argument("--checkpoint", required=True, help="模型检查点路径")
    infer_parser.add_argument("--data", required=True, help="数据路径")
    infer_parser.add_argument("--config", default="optimized_stable", help="配置文件名")
    infer_parser.add_argument("--output", help="输出文件路径")
    
    # 稳定性检查
    stability_parser = subparsers.add_parser("stability-check", help="训练稳定性检查")
    stability_parser.add_argument("--config", default="optimized_stable", help="配置文件名")
    
    return parser


def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 创建训练系统
    training_system = UnifiedTrainingSystem()
    
    try:
        if args.command == "train":
            # 基础训练
            kwargs = {k.replace('-', '_'): v for k, v in vars(args).items() 
                     if v is not None and k not in ['command', 'config']}
            training_system.train_basic(args.config, **kwargs)
            
        elif args.command == "hpo":
            # 超参数优化
            # 仅在需要时导入 optuna 相关代码
            from src.hpo.optuna_runner import run_optimization as optuna_main
            kwargs = {k.replace('-', '_'): v for k, v in vars(args).items() 
                     if v is not None and k not in ['command', 'config', 'n_trials']}
            training_system.optimize_hyperparameters(
                args.config, args.n_trials, **kwargs
            )
            
        elif args.command == "walk-forward":
            # Walk-Forward验证
            kwargs = {k.replace('-', '_'): v for k, v in vars(args).items() 
                     if v is not None and k not in ['command', 'config', 'n_folds']}
            training_system.walk_forward_validation(
                args.config, args.n_folds, **kwargs
            )
            
        elif args.command == "eval":
            # 模型评估
            kwargs = {k.replace('-', '_'): v for k, v in vars(args).items() 
                     if v is not None and k not in ['command', 'config', 'checkpoint']}
            training_system.evaluate_model(
                args.checkpoint, args.config, **kwargs
            )
            
        elif args.command == "infer":
            # 模型推理
            kwargs = {k.replace('-', '_'): v for k, v in vars(args).items() 
                     if v is not None and k not in ['command', 'config', 'checkpoint', 'data']}
            training_system.inference(
                args.checkpoint, args.data, args.config, **kwargs
            )
            
        elif args.command == "stability-check":
            # 稳定性检查
            results = training_system.stability_check(args.config)
            
            print("\n=== 训练稳定性检查结果 ===")
            print(f"配置有效性: {'✓' if results['config_valid'] else '✗'}")
            print(f"环境就绪: {'✓' if results['environment_ready'] else '✗'}")
            
            if results['stability_features']:
                print("\n已启用的稳定性特性:")
                for feature in results['stability_features']:
                    print(f"  ✓ {feature}")
            
            if results['recommendations']:
                print("\n建议改进:")
                for rec in results['recommendations']:
                    print(f"  ⚠ {rec}")
            
            print("\n检查完成!")
        
        else:
            parser.print_help()
            
    except Exception as e:
        logger.error(f"执行命令 '{args.command}' 时发生错误: {e}")
        raise


if __name__ == "__main__":
    main()