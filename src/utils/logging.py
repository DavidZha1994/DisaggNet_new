"""日志和可视化工具"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import pickle
from datetime import datetime
import logging
from omegaconf import DictConfig
import warnings
warnings.filterwarnings('ignore')

# TensorBoard相关
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

# Matplotlib相关
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    sns = None


class ExperimentLogger:
    """实验日志记录器"""
    
    def __init__(self, log_dir: Path, experiment_name: str, config: DictConfig):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.config = config
        
        # 创建日志目录
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化TensorBoard
        if TENSORBOARD_AVAILABLE:
            self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
        else:
            self.tb_writer = None
        
        # 初始化文件日志
        self._setup_file_logging()
        
        # 保存配置
        self._save_config()
        
        # 指标历史
        self.metrics_history = {
            'train': [],
            'val': [],
            'test': []
        }
        
        # 图表计数器
        self.plot_counter = 0
    
    def _setup_file_logging(self):
        """设置文件日志"""
        log_file = self.log_dir / f"{self.experiment_name}.log"
        
        # 创建logger
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # 避免重复添加handler
        if not self.logger.handlers:
            # 文件handler
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            
            # 控制台handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 格式化器
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # 添加handler
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def _save_config(self):
        """保存实验配置"""
        from omegaconf import OmegaConf
        
        config_file = self.log_dir / "config.yaml"
        OmegaConf.save(self.config, config_file)
        
        # 同时保存JSON格式
        config_json = self.log_dir / "config.json"
        with open(config_json, 'w') as f:
            json.dump(OmegaConf.to_container(self.config, resolve=True), f, indent=2)
    
    def log_metrics(self, metrics: Dict[str, float], step: int, split: str = 'train'):
        """记录指标"""
        
        # TensorBoard记录
        if self.tb_writer:
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(f"{split}/{metric_name}", value, step)
        
        # 文件记录
        self.logger.info(f"Step {step} - {split.upper()} - {metrics}")
        
        # 历史记录
        metric_record = {
            'step': step,
            'metrics': metrics.copy(),
            'timestamp': datetime.now().isoformat()
        }
        self.metrics_history[split].append(metric_record)
    
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """记录超参数"""
        if self.tb_writer:
            # 过滤可序列化的超参数
            serializable_hparams = {}
            for key, value in hparams.items():
                if isinstance(value, (int, float, str, bool)):
                    serializable_hparams[key] = value
                elif isinstance(value, (list, tuple)) and len(value) > 0:
                    if isinstance(value[0], (int, float)):
                        serializable_hparams[key] = str(value)
            
            self.tb_writer.add_hparams(serializable_hparams, metrics)
    
    def log_model_graph(self, model, input_sample):
        """记录模型图"""
        if self.tb_writer:
            try:
                self.tb_writer.add_graph(model, input_sample)
            except Exception as e:
                self.logger.warning(f"Failed to log model graph: {e}")
    
    def log_predictions(self, predictions: np.ndarray, targets: np.ndarray, 
                       step: int, split: str = 'val', max_samples: int = 100):
        """记录预测结果"""
        
        # 限制样本数量
        n_samples = min(len(predictions), max_samples)
        pred_sample = predictions[:n_samples]
        target_sample = targets[:n_samples]
        
        if self.tb_writer:
            # 记录预测散点图
            for device_idx in range(pred_sample.shape[1]):
                device_name = f"device_{device_idx}"
                
                # 创建散点图数据
                pred_values = pred_sample[:, device_idx]
                true_values = target_sample[:, device_idx]
                
                # 记录到TensorBoard
                self.tb_writer.add_scalar(
                    f"{split}/prediction_correlation_{device_name}",
                    np.corrcoef(pred_values, true_values)[0, 1] if len(pred_values) > 1 else 0,
                    step
                )
    
    def log_confusion_matrix(self, cm: np.ndarray, class_names: List[str], 
                           step: int, split: str = 'val', device_name: str = 'device'):
        """记录混淆矩阵"""
        if self.tb_writer and MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # 绘制混淆矩阵
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=class_names, yticklabels=class_names)
            ax.set_title(f'{device_name} Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            
            # 记录到TensorBoard
            self.tb_writer.add_figure(f"{split}/confusion_matrix_{device_name}", fig, step)
            plt.close(fig)
    
    def log_power_comparison(self, predictions: np.ndarray, targets: np.ndarray, 
                           mains: np.ndarray, step: int, split: str = 'val', 
                           device_names: List[str] = None):
        """记录功率对比图"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        if device_names is None:
            device_names = [f"device_{i}" for i in range(predictions.shape[1])]
        
        # 选择一个时间窗口进行可视化
        window_size = min(200, len(predictions))
        start_idx = max(0, len(predictions) - window_size)
        
        pred_window = predictions[start_idx:start_idx + window_size]
        target_window = targets[start_idx:start_idx + window_size]
        mains_window = mains[start_idx:start_idx + window_size]
        
        # 创建子图
        n_devices = len(device_names)
        fig, axes = plt.subplots(n_devices + 1, 1, figsize=(12, 3 * (n_devices + 1)))
        
        if n_devices == 0:
            axes = [axes]
        
        # 主线功率
        time_steps = range(window_size)
        axes[0].plot(time_steps, mains_window[:, 0] if mains_window.ndim > 1 else mains_window, 
                    label='Mains Power', color='black', linewidth=2)
        axes[0].plot(time_steps, np.sum(pred_window, axis=1), 
                    label='Sum of Predicted', color='red', linestyle='--', alpha=0.7)
        axes[0].set_title('Mains vs Sum of Devices')
        axes[0].set_ylabel('Power (W)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 各设备功率
        for i, device_name in enumerate(device_names):
            if i + 1 < len(axes):
                ax = axes[i + 1]
                ax.plot(time_steps, target_window[:, i], label='True', alpha=0.8)
                ax.plot(time_steps, pred_window[:, i], label='Predicted', alpha=0.8)
                ax.set_title(f'{device_name} Power')
                ax.set_ylabel('Power (W)')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time Steps')
        
        plt.tight_layout()
        
        # 记录到TensorBoard
        if self.tb_writer:
            self.tb_writer.add_figure(f"{split}/power_comparison", fig, step)
        
        # 保存到文件
        plot_path = self.log_dir / f"power_comparison_{split}_step_{step}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def log_training_progress(self, epoch: int, train_loss: float, val_loss: float, 
                            val_score: float, learning_rate: float):
        """记录训练进度"""
        
        progress_info = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_score': val_score,
            'learning_rate': learning_rate,
            'timestamp': datetime.now().isoformat()
        }
        
        # 文件记录
        self.logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                        f"val_loss={val_loss:.4f}, val_score={val_score:.4f}, "
                        f"lr={learning_rate:.6f}")
        
        # TensorBoard记录
        if self.tb_writer:
            self.tb_writer.add_scalar('training/train_loss', train_loss, epoch)
            self.tb_writer.add_scalar('training/val_loss', val_loss, epoch)
            self.tb_writer.add_scalar('training/val_score', val_score, epoch)
            self.tb_writer.add_scalar('training/learning_rate', learning_rate, epoch)
    
    def save_metrics_history(self):
        """保存指标历史"""
        history_file = self.log_dir / "metrics_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2, default=str)
    
    def generate_training_summary(self) -> str:
        """生成训练总结报告"""
        
        summary_lines = []
        summary_lines.append(f"# 训练总结报告 - {self.experiment_name}")
        summary_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append("")
        
        # 训练历史统计
        if self.metrics_history['train']:
            train_history = self.metrics_history['train']
            summary_lines.append("## 训练历史")
            summary_lines.append(f"总训练步数: {len(train_history)}")
            
            # 最终指标
            if train_history:
                final_metrics = train_history[-1]['metrics']
                summary_lines.append("### 最终训练指标")
                for metric, value in final_metrics.items():
                    if isinstance(value, (int, float)):
                        summary_lines.append(f"- {metric}: {value:.4f}")
                summary_lines.append("")
        
        # 验证历史统计
        if self.metrics_history['val']:
            val_history = self.metrics_history['val']
            summary_lines.append("## 验证历史")
            summary_lines.append(f"总验证步数: {len(val_history)}")
            
            # 最佳指标
            if val_history:
                # 假设有score指标
                scores = [h['metrics'].get('score', 0) for h in val_history]
                if scores:
                    best_idx = np.argmax(scores)
                    best_metrics = val_history[best_idx]['metrics']
                    summary_lines.append("### 最佳验证指标")
                    summary_lines.append(f"最佳步数: {val_history[best_idx]['step']}")
                    for metric, value in best_metrics.items():
                        if isinstance(value, (int, float)):
                            summary_lines.append(f"- {metric}: {value:.4f}")
                summary_lines.append("")
        
        # 测试结果
        if self.metrics_history['test']:
            test_history = self.metrics_history['test']
            summary_lines.append("## 测试结果")
            if test_history:
                final_test = test_history[-1]['metrics']
                for metric, value in final_test.items():
                    if isinstance(value, (int, float)):
                        summary_lines.append(f"- {metric}: {value:.4f}")
            summary_lines.append("")
        
        # 保存总结
        summary_content = "\n".join(summary_lines)
        summary_file = self.log_dir / "training_summary.md"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        return str(summary_file)
    
    def close(self):
        """关闭日志记录器"""
        # 保存指标历史
        self.save_metrics_history()
        
        # 生成训练总结
        self.generate_training_summary()
        
        # 关闭TensorBoard
        if self.tb_writer:
            self.tb_writer.close()
        
        # 关闭文件日志
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


class VisualizationManager:
    """可视化管理器"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置matplotlib样式
        if MATPLOTLIB_AVAILABLE:
            plt.style.use('default')
            if sns:
                sns.set_palette("husl")
    
    def plot_training_curves(self, metrics_history: Dict[str, List], 
                           save_path: Optional[str] = None) -> str:
        """绘制训练曲线"""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 提取数据
        train_steps = [h['step'] for h in metrics_history.get('train', [])]
        val_steps = [h['step'] for h in metrics_history.get('val', [])]
        
        # 损失曲线
        if train_steps:
            train_losses = [h['metrics'].get('loss', 0) for h in metrics_history['train']]
            axes[0, 0].plot(train_steps, train_losses, label='Train Loss', alpha=0.7)
        
        if val_steps:
            val_losses = [h['metrics'].get('loss', 0) for h in metrics_history['val']]
            axes[0, 0].plot(val_steps, val_losses, label='Val Loss', alpha=0.7)
        
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 分数曲线
        if val_steps:
            val_scores = [h['metrics'].get('score', 0) for h in metrics_history['val']]
            axes[0, 1].plot(val_steps, val_scores, label='Val Score', color='green', alpha=0.7)
        
        axes[0, 1].set_title('Validation Score')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 学习率曲线
        if train_steps:
            learning_rates = [h['metrics'].get('learning_rate', 0) for h in metrics_history['train']]
            axes[1, 0].plot(train_steps, learning_rates, label='Learning Rate', color='orange', alpha=0.7)
        
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 梯度范数（如果有）
        if train_steps:
            grad_norms = [h['metrics'].get('grad_norm', 0) for h in metrics_history['train']]
            if any(g > 0 for g in grad_norms):
                axes[1, 1].plot(train_steps, grad_norms, label='Gradient Norm', color='red', alpha=0.7)
                axes[1, 1].set_title('Gradient Norm')
                axes[1, 1].set_xlabel('Steps')
                axes[1, 1].set_ylabel('Norm')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'No gradient norm data', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            save_path = self.output_dir / "training_curves.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return str(save_path)
    
    def plot_device_performance(self, device_metrics: Dict[str, Dict], 
                              save_path: Optional[str] = None) -> str:
        """绘制各设备性能对比"""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        devices = list(device_metrics.keys())
        metrics = ['mae', 'rmse', 'f1', 'precision', 'recall']
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                ax = axes[i]
                
                values = []
                labels = []
                
                for device in devices:
                    if metric in device_metrics[device]:
                        values.append(device_metrics[device][metric])
                        labels.append(device)
                
                if values:
                    bars = ax.bar(labels, values, alpha=0.7)
                    ax.set_title(f'{metric.upper()} by Device')
                    ax.set_ylabel(metric.upper())
                    ax.tick_params(axis='x', rotation=45)
                    
                    # 添加数值标签
                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.3f}', ha='center', va='bottom')
                
                ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            save_path = self.output_dir / "device_performance.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return str(save_path)


def create_experiment_logger(config: DictConfig, experiment_name: str) -> ExperimentLogger:
    """创建实验日志记录器"""
    
    # 创建日志目录
    log_dir = Path(config.logging.save_dir) / experiment_name
    
    return ExperimentLogger(log_dir, experiment_name, config)


def create_visualization_manager(output_dir: Path) -> VisualizationManager:
    """创建可视化管理器"""
    return VisualizationManager(output_dir)