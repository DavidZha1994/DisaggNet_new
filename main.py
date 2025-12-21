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
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

# macOS: 统一 OpenMP 运行时，避免重复初始化错误
if sys.platform == 'darwin':
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('PYTHONUTF8', '1')
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
# Windows 控制台切换到 UTF-8 代码页
if os.name == 'nt':
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:
        pass


# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# 项目模块在方法内部按需导入，避免 E402

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    encoding='utf-8'
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
    
    def setup_environment(self, config_ref: str = "configs/default.yaml") -> DictConfig:
        """设置训练环境（仅加载单一配置文件；支持路径或名称）"""
        # 解析配置引用：支持绝对/相对路径或简短名称
        candidates = []
        try:
            is_path_like = (config_ref.endswith('.yaml') or os.sep in config_ref or '/' in config_ref)
        except Exception:
            is_path_like = False

        if is_path_like:
            cfg_path = Path(config_ref)
            if not cfg_path.is_absolute():
                cfg_path = self.project_root / cfg_path
            candidates.append(cfg_path)
        else:
            # 优先在 configs/training 下查找，再回退到 configs 根目录
            candidates.append(self.configs_dir / "training" / f"{config_ref}.yaml")
            candidates.append(self.configs_dir / f"{config_ref}.yaml")

        # 回退候选：默认配置（优先 training/default.yaml）
        candidates.append(self.configs_dir / "training" / "default.yaml")
        candidates.append(self.configs_dir / "default.yaml")

        # 选择首个存在的配置文件
        chosen = None
        for path in candidates:
            if path.exists():
                chosen = path
                break

        if chosen is None:
            # 无法找到任何配置文件，抛出明确错误
            search_hint = (
                "未找到可用配置。尝试过: "
                + ", ".join(str(p) for p in candidates)
            )
            logger.error(search_hint)
            raise FileNotFoundError(search_hint)

        # 仅加载单一配置文件（不再合并 base.yaml）
        logger.info(f"环境设置完成，使用配置: {chosen}")
        config = OmegaConf.load(chosen)
        
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
        
        return config
    
    def train_basic(self, config_ref: str = "configs/default.yaml", **kwargs) -> None:
        """训练"""
        logger.info(f"开始训练，配置: {config_ref}")
        
        try:
            config = self.setup_environment(config_ref)
            # 兼容缺失 training 节点的配置：动态填充最小必需项
            if not hasattr(config, 'training') or getattr(config, 'training') is None:
                from omegaconf import OmegaConf as _OC
                config.training = _OC.create({
                    'max_epochs': 10,
                    'min_epochs': 1,
                    'optimizer': {'name': 'adamw', 'lr': 1e-4},
                    'checkpoint': {'dirpath': 'outputs/checkpoints', 'filename': 'epoch={epoch:02d}', 'monitor': 'val/loss', 'mode': 'min', 'save_top_k': 1},
                })
            
            # 更新配置参数（支持嵌套项覆盖）
            for key, value in kwargs.items():
                try:
                    if key == 'epochs':
                        if hasattr(config, 'training'):
                            config.training.max_epochs = int(value)
                        # 保证最小轮数不超过最大轮数
                        if hasattr(config, 'training') and hasattr(config.training, 'min_epochs'):
                            config.training.min_epochs = min(int(getattr(config.training, 'min_epochs', 1)), int(value))
                        logger.info(f"更新配置参数: training.max_epochs = {value}")
                    elif key == 'batch_size':
                        config.data.batch_size = int(value)
                        logger.info(f"更新配置参数: data.batch_size = {value}")
                    elif key == 'lr':
                        if hasattr(config, 'training') and hasattr(config.training, 'optimizer'):
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
            from src.train import main as train_main
            train_main(config)
            logger.info("训练完成")
            
        except Exception as e:
            logger.error(f"训练失败: {e}")
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
        
        # 调用HPO（在此处进行局部导入，避免全局 None 覆盖）
        try:
            from src.hpo.optuna_runner import run_optimization
            run_optimization(
                config,
                n_trials=n_trials,
                timeout=timeout,
                study_name=kwargs.get("study_name"),
                storage=kwargs.get("storage"),
                space_config=kwargs.get("space_config"),
            )
        except Exception as e:
            logger.error(f"超参数优化调用失败: {e}")
            raise
    
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
        from src.walk_forward import run_walk_forward_validation as walk_forward_main
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
            from src.eval import main as eval_main
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
            
            # 延迟导入推理模块，避免非推理路径导入失败
            from src.infer import main as infer_main
            
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
    train_parser.add_argument("--config", default="configs/default.yaml", help="配置文件路径或名称")
    train_parser.add_argument("--epochs", type=int, help="训练轮数")
    train_parser.add_argument("--batch-size", type=int, help="批次大小")
    train_parser.add_argument("--lr", type=float, help="学习率")
    train_parser.add_argument("--dataset", type=str, help="数据集名称（如 UKDALE/REFIT）")
    
    # 超参数优化
    hpo_parser = subparsers.add_parser("hpo", help="超参数优化")
    hpo_parser.add_argument("--config", default="optimized_stable", help="配置文件名")
    hpo_parser.add_argument("--n-trials", type=int, default=50, help="试验次数")
    hpo_parser.add_argument("--timeout", type=int, help="超时时间(秒)")
    hpo_parser.add_argument("--study-name", help="研究名称")
    hpo_parser.add_argument("--storage", help="Optuna存储(例如 sqlite:///outputs/hpo/optuna.db)")
    hpo_parser.add_argument("--space-config", help="搜索空间配置文件路径")
    
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
