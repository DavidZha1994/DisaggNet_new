"""模型评估脚本"""

import os
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any
from omegaconf import DictConfig
import json
from datetime import datetime
import warnings
from .train import NILMLightningModule, load_device_info
from .datamodule.datamodule import NILMDataModule
from .utils.metrics import NILMMetrics, ConsistencyMetrics, DelayMetrics

warnings.filterwarnings('ignore')


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, config: DictConfig, model_path: str):
        self.config = config
        self.model_path = model_path
        
        # 加载设备信息
        self.device_info, self.device_names = load_device_info(config)
        
        # 加载模型
        self.model = self._load_model()
        
        # 初始化指标计算器
        self.nilm_metrics = NILMMetrics(self.device_names)
        self.consistency_metrics = ConsistencyMetrics()
        self.delay_metrics = DelayMetrics()
    
    def _load_model(self) -> NILMLightningModule:
        """加载训练好的模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # 加载模型
        model = NILMLightningModule.load_from_checkpoint(
            self.model_path,
            config=self.config,
            device_info=self.device_info,
            device_names=self.device_names,
            map_location='cpu'
        )
        
        model.eval()
        return model
    
    def evaluate_on_dataloader(self, dataloader: torch.utils.data.DataLoader, 
                              split_name: str = 'test') -> Dict[str, Any]:
        """在数据加载器上评估模型"""
        
        print(f"开始评估 {split_name} 集...")
        
        device = next(self.model.parameters()).device
        
        all_pred_power = []
        all_pred_states = []  # logits
        all_true_power = []
        all_true_states = []
        all_timestamps = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # 取真实标签
                y_true_power = batch.get('target_power')
                y_true_states = batch.get('target_states')
                if y_true_power is None or y_true_states is None:
                    raise KeyError("batch需要包含 'target_power' 与 'target_states' 字段")
                y_true_power = y_true_power.to(device)
                y_true_states = y_true_states.to(device)
                
                # 模型预测
                outputs = self.model(batch)
                # 模块forward返回 (pred_power, pred_states)
                pred_power, pred_states = outputs

                # 收集结果
                all_pred_power.append(pred_power.detach().cpu())
                all_pred_states.append(pred_states.detach().cpu())
                all_true_power.append(y_true_power.detach().cpu())
                all_true_states.append(y_true_states.detach().cpu())
                
                if 'timestamps' in batch:
                    all_timestamps.extend(batch['timestamps'])
                
                # 打印进度
                if batch_idx % 50 == 0:
                    print(f"  处理批次: {batch_idx}/{len(dataloader)}")
        
        # 合并所有批次
        pred_power = torch.cat(all_pred_power, dim=0)
        logits = torch.cat(all_pred_states, dim=0)
        # 温度缩放（若可用）
        if hasattr(self.model, 'temperature_scaling') and self.model.temperature_scaling is not None:
            logits = self.model.temperature_scaling(logits)
        pred_probs = torch.sigmoid(logits)
        
        y_true_power = torch.cat(all_true_power, dim=0)
        y_true_states = torch.cat(all_true_states, dim=0)
        targets = y_true_states.numpy()
        predictions = pred_power.numpy()
        
        print(
            f"评估数据形状: predictions={predictions.shape}, targets={targets.shape}"
        )
        
        # 计算指标与状态策略对比
        results = self._compute_all_metrics(
            predictions,
            targets,
            y_true_power.numpy(),
            pred_probs.numpy(),
            all_timestamps
        )

        # 统一计算两种状态策略
        states_comparison = self._compute_state_strategies(predictions, pred_probs.numpy())
        results['state_strategies'] = states_comparison
        
        return results
    
    def _compute_all_metrics(self, predictions: np.ndarray, targets: np.ndarray, 
                           true_power: np.ndarray, pred_proba: np.ndarray,
                           timestamps: List = None) -> Dict[str, Any]:
        """计算所有评估指标"""
        
        results = {}
        
        # 获取最佳阈值（分类或功率）
        raw_thresholds = getattr(self.model, 'best_thresholds', None)
        class_thresholds: Optional[Dict[str, float]] = None
        power_thresholds: Optional[Dict[str, float]] = None

        if isinstance(raw_thresholds, dict) and len(raw_thresholds) > 0:
            if max(map(float, raw_thresholds.values())) <= 1.0 + 1e-8:
                class_thresholds = {name: float(raw_thresholds.get(name, 0.5)) for name in self.device_names}
            else:
                power_thresholds = {name: float(raw_thresholds.get(name, 10.0)) for name in self.device_names}
        else:
            # 退回分类默认阈值
            class_thresholds = {name: 0.5 for name in self.device_names}

        # 功率滞回阈值默认值
        if power_thresholds is None:
            default_power_thr = float(self.config.inference.get('default_power_threshold', 10.0))
            power_thresholds = {name: default_power_thr for name in self.device_names}
        
        print(f"分类阈值: {class_thresholds}")
        print(f"功率滞回阈值: {power_thresholds}")
        
        # NILM指标
        print("计算NILM指标...")
        nilm_results = self.nilm_metrics.compute_all_metrics(
            torch.from_numpy(predictions),
            torch.from_numpy(pred_proba),
            torch.from_numpy(true_power),
            torch.from_numpy(targets),
            optimize_thresholds=False
        )
        results['nilm_metrics'] = nilm_results
        
        # 一致性指标
        print("计算一致性指标...")
        # 一致性指标：功率平衡误差（预测总功率 vs 真实总功率）
        consistency_results = {
            'power_balance_error': self.consistency_metrics.power_balance_error(
                torch.from_numpy(predictions), torch.from_numpy(true_power)
            )
        }
        results['consistency_metrics'] = consistency_results
        
        # 延迟指标
        print("计算延迟指标...")
        # 延迟指标：使用检测延迟并汇总平均
        per_device_delays = self.delay_metrics.detection_delay(
            torch.from_numpy(pred_proba), torch.from_numpy(targets)
        )
        finite_delays = [d for d in per_device_delays.values() if np.isfinite(d)]
        avg_delay = float(np.mean(finite_delays)) if finite_delays else float('inf')
        delay_results = {
            'per_device': per_device_delays,
            'avg_detection_delay': avg_delay
        }
        results['delay_metrics'] = delay_results
        
        # 计算综合分数
        overall_score = self._compute_overall_score(nilm_results, consistency_results, delay_results)
        results['overall_score'] = overall_score
        
        return results
    
    def _compute_overall_score(self, nilm_results: Dict, consistency_results: Dict, 
                              delay_results: Dict) -> float:
        """计算综合评分"""
        
        scores = []
        
        # NILM分数 (权重: 0.6)
        if 'overall' in nilm_results and 'score' in nilm_results['overall']:
            nilm_score = nilm_results['overall']['score']
            scores.append(('nilm', nilm_score, 0.6))
        
        # 一致性分数 (权重: 0.3)
        if 'power_balance_error' in consistency_results:
            # 将误差转换为分数 (误差越小分数越高)
            consistency_error = consistency_results['power_balance_error']
            consistency_score = max(0, 1 - consistency_error)
            scores.append(('consistency', consistency_score, 0.3))
        
        # 延迟分数 (权重: 0.1)
        if 'avg_detection_delay' in delay_results:
            # 将延迟转换为分数 (延迟越小分数越高)
            avg_delay = delay_results['avg_detection_delay']
            delay_score = max(0, 1 - avg_delay / 100)  # 假设100为最大可接受延迟
            scores.append(('delay', delay_score, 0.1))
        
        # 计算加权平均
        if scores:
            weighted_sum = sum(score * weight for _, score, weight in scores)
            total_weight = sum(weight for _, _, weight in scores)
            overall_score = weighted_sum / total_weight
        else:
            overall_score = 0.0
        
        return overall_score

    def _compute_state_strategies(self, pred_power: np.ndarray, pred_probs: np.ndarray) -> Dict[str, Any]:
        """统一计算并展示两种状态策略：功率滞回与分类概率阈值"""
        # 阈值准备（与 _compute_all_metrics 同步逻辑）
        raw_thresholds = getattr(self.model, 'best_thresholds', None)
        if isinstance(raw_thresholds, dict) and len(raw_thresholds) > 0:
            if max(map(float, raw_thresholds.values())) <= 1.0 + 1e-8:
                class_thresholds = {name: float(raw_thresholds.get(name, 0.5)) for name in self.device_names}
                power_thresholds = None
            else:
                class_thresholds = {name: 0.5 for name in self.device_names}
                power_thresholds = {name: float(raw_thresholds.get(name, 10.0)) for name in self.device_names}
        else:
            class_thresholds = {name: 0.5 for name in self.device_names}
            power_thresholds = None

        # 功率阈值估计（数据驱动）
        if power_thresholds is None:
            method = str(self.config.inference.get('power_threshold_method', 'fixed'))
            if method == 'otsu':
                power_thresholds = self._estimate_power_thresholds_otsu(pred_power)
            elif method == 'percentile':
                pctl = float(self.config.inference.get('power_threshold_percentile', 90.0))
                power_thresholds = self._estimate_power_thresholds_percentile(pred_power, pctl)
            else:
                default_power_thr = float(self.config.inference.get('default_power_threshold', 10.0))
                power_thresholds = {name: default_power_thr for name in self.device_names}

        hysteresis_ratio = float(self.config.inference.get('hysteresis_ratio', 0.1))

        # 计算功率滞回状态
        hysteresis_states = self._apply_hysteresis(pred_power, power_thresholds, hysteresis_ratio)
        # 计算概率阈值状态
        threshold_states = self._apply_class_thresholds(pred_probs, class_thresholds)

        # 差异率（每设备）
        disagreement = {}
        for idx, name in enumerate(self.device_names):
            a = hysteresis_states[:, idx]
            b = threshold_states[:, idx]
            disagreement[name] = float(np.mean(a != b))

        # 开启比率（每设备）
        on_ratio_hys = {name: float(np.mean(hysteresis_states[:, i])) for i, name in enumerate(self.device_names)}
        on_ratio_thr = {name: float(np.mean(threshold_states[:, i])) for i, name in enumerate(self.device_names)}

        return {
            'power_thresholds': power_thresholds,
            'class_thresholds': class_thresholds,
            'hysteresis_states_shape': list(hysteresis_states.shape),
            'threshold_states_shape': list(threshold_states.shape),
            'disagreement_ratio': disagreement,
            'on_ratio_hysteresis': on_ratio_hys,
            'on_ratio_threshold': on_ratio_thr
        }

    def _apply_hysteresis(self, power: np.ndarray, power_thresholds: Dict[str, float], hysteresis_ratio: float) -> np.ndarray:
        """按设备对功率应用滞回，生成二值状态 [N, D]"""
        N, D = power.shape
        states = np.zeros((N, D), dtype=bool)
        on_thr = np.array([power_thresholds[name] for name in self.device_names], dtype=float)
        off_thr = on_thr * (1.0 - hysteresis_ratio)
        current = np.zeros(D, dtype=bool)
        for t in range(N):
            p = power[t]
            # 上升沿
            current = np.where(p >= on_thr, True, current)
            # 下降沿
            current = np.where(p <= off_thr, False, current)
            states[t] = current
        return states

    def _apply_class_thresholds(self, probs: np.ndarray, class_thresholds: Dict[str, float]) -> np.ndarray:
        """按设备分类阈值进行二值化 [N, D]"""
        N, D = probs.shape
        states = np.zeros((N, D), dtype=bool)
        thr = np.array([class_thresholds[name] for name in self.device_names], dtype=float)
        states = probs >= thr[None, :]
        return states

    def _estimate_power_thresholds_percentile(self, power: np.ndarray, percentile: float) -> Dict[str, float]:
        """使用分位数估计每个设备的功率阈值"""
        thresholds = {}
        pct = np.clip(percentile, 1.0, 99.0)
        for i, name in enumerate(self.device_names):
            arr = power[:, i]
            # 去除负值并裁剪异常大值以稳健
            arr = arr[arr >= 0]
            if arr.size == 0:
                thresholds[name] = float(self.config.inference.get('default_power_threshold', 10.0))
                continue
            thr = np.percentile(arr, pct)
            thresholds[name] = float(thr)
        return thresholds

    def _estimate_power_thresholds_otsu(self, power: np.ndarray) -> Dict[str, float]:
        """使用Otsu方法从功率分布估计每个设备的阈值"""
        thresholds = {}
        for i, name in enumerate(self.device_names):
            arr = power[:, i]
            arr = arr[arr >= 0]
            if arr.size < 10:
                thresholds[name] = float(self.config.inference.get('default_power_threshold', 10.0))
                continue
            # 构建直方图
            bins = 128
            hist, bin_edges = np.histogram(arr, bins=bins, range=(0, float(arr.max())))
            hist = hist.astype(np.float64)
            total = hist.sum()
            if total <= 0:
                thresholds[name] = float(self.config.inference.get('default_power_threshold', 10.0))
                continue
            bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            # 累积分布
            weight_bg = np.cumsum(hist)
            weight_fg = total - weight_bg
            mean_bg = np.cumsum(hist * bin_mids) / np.maximum(weight_bg, 1e-12)
            mean_total = (hist * bin_mids).sum() / total
            mean_fg = (mean_total * total - mean_bg * weight_bg) / np.maximum(weight_fg, 1e-12)
            # 类间方差
            sigma_b2 = (weight_bg / total) * (weight_fg / total) * (mean_bg - mean_fg) ** 2
            # 选择最大方差对应的阈值
            idx = int(np.nanargmax(sigma_b2))
            thr = bin_mids[idx]
            if not np.isfinite(thr) or thr <= 0:
                thr = float(self.config.inference.get('default_power_threshold', 10.0))
            thresholds[name] = float(thr)
        return thresholds
    
    def generate_evaluation_report(self, results: Dict[str, Any], output_dir: Path, 
                                 split_name: str = 'test') -> str:
        """生成评估报告"""
        
        report_lines = []
        report_lines.append(f"# {split_name.upper()} 集评估报告")
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"模型路径: {self.model_path}")
        report_lines.append("")
        
        # 综合分数
        if 'overall_score' in results:
            report_lines.append(f"## 综合分数: {results['overall_score']:.4f}")
            report_lines.append("")
        
        # NILM指标
        if 'nilm_metrics' in results:
            report_lines.append("## NILM指标")
            nilm_metrics = results['nilm_metrics']
            
            # 整体指标
            if 'overall' in nilm_metrics:
                overall = nilm_metrics['overall']
                report_lines.append("### 整体指标")
                for metric, value in overall.items():
                    if isinstance(value, (int, float)):
                        report_lines.append(f"- {metric}: {value:.4f}")
                report_lines.append("")
            
            # 各设备指标
            report_lines.append("### 各设备指标")
            for device in self.device_names:
                if device in nilm_metrics:
                    device_metrics = nilm_metrics[device]
                    report_lines.append(f"#### {device}")
                    for metric, value in device_metrics.items():
                        if isinstance(value, (int, float)):
                            report_lines.append(f"- {metric}: {value:.4f}")
                    report_lines.append("")
        
        # 一致性指标
        if 'consistency_metrics' in results:
            report_lines.append("## 一致性指标")
            consistency = results['consistency_metrics']
            for metric, value in consistency.items():
                if isinstance(value, (int, float)):
                    report_lines.append(f"- {metric}: {value:.4f}")
            report_lines.append("")
        
        # 延迟指标
        if 'delay_metrics' in results:
            report_lines.append("## 延迟指标")
            delay = results['delay_metrics']
            for metric, value in delay.items():
                if isinstance(value, (int, float)):
                    report_lines.append(f"- {metric}: {value:.4f}")
            report_lines.append("")
        
        # 保存报告
        report_content = "\n".join(report_lines)
        report_path = output_dir / f"{split_name}_evaluation_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"评估报告保存至: {report_path}")
        return str(report_path)
    
    def generate_visualizations(self, predictions: np.ndarray, targets: np.ndarray, 
                              mains: np.ndarray, output_dir: Path, split_name: str = 'test'):
        """生成可视化图表"""
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 设置样式
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 创建可视化目录
            viz_dir = output_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            # 1. 功率预测对比图
            self._plot_power_comparison(predictions, targets, viz_dir, split_name)
            
            # 2. 误差分布图
            self._plot_error_distribution(predictions, targets, viz_dir, split_name)
            
            # 3. 一致性分析图
            self._plot_consistency_analysis(predictions, mains, viz_dir, split_name)
            
            # 4. 设备状态混淆矩阵
            self._plot_confusion_matrices(predictions, targets, viz_dir, split_name)
            
            print(f"可视化图表保存至: {viz_dir}")
            
        except ImportError:
            print("Matplotlib/Seaborn not available, skipping visualizations.")
    
    def _plot_power_comparison(self, predictions: np.ndarray, targets: np.ndarray, 
                              output_dir: Path, split_name: str):
        """绘制功率预测对比图"""
        import matplotlib.pyplot as plt
        
        n_devices = len(self.device_names)
        fig, axes = plt.subplots(2, (n_devices + 1) // 2, figsize=(15, 8))
        axes = axes.flatten() if n_devices > 1 else [axes]
        
        for i, device in enumerate(self.device_names):
            if i < len(axes):
                ax = axes[i]
                
                # 选择一个时间窗口进行可视化
                start_idx = 0
                end_idx = min(1000, len(predictions))
                
                time_steps = range(start_idx, end_idx)
                pred_power = predictions[start_idx:end_idx, i]
                true_power = targets[start_idx:end_idx, i]
                
                ax.plot(time_steps, true_power, label='True', alpha=0.7)
                ax.plot(time_steps, pred_power, label='Predicted', alpha=0.7)
                ax.set_title(f'{device} Power Prediction')
                ax.set_xlabel('Time Steps')
                ax.set_ylabel('Power (W)')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(n_devices, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{split_name}_power_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_error_distribution(self, predictions: np.ndarray, targets: np.ndarray, 
                               output_dir: Path, split_name: str):
        """绘制误差分布图"""
        import matplotlib.pyplot as plt
        
        errors = predictions - targets
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 整体误差分布
        axes[0, 0].hist(errors.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Overall Error Distribution')
        axes[0, 0].set_xlabel('Error (W)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 各设备误差分布
        for i, device in enumerate(self.device_names[:3]):  # 最多显示3个设备
            row = (i + 1) // 2
            col = (i + 1) % 2
            if row < 2:
                axes[row, col].hist(errors[:, i], bins=30, alpha=0.7, edgecolor='black')
                axes[row, col].set_title(f'{device} Error Distribution')
                axes[row, col].set_xlabel('Error (W)')
                axes[row, col].set_ylabel('Frequency')
                axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{split_name}_error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_consistency_analysis(self, predictions: np.ndarray, mains: np.ndarray, 
                                 output_dir: Path, split_name: str):
        """绘制一致性分析图"""
        import matplotlib.pyplot as plt
        
        # 计算总设备功率
        total_device_power = np.sum(predictions, axis=1)
        mains_power = mains[:, 0] if mains.ndim > 1 else mains
        
        # 选择一个时间窗口
        start_idx = 0
        end_idx = min(1000, len(total_device_power))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 功率对比
        time_steps = range(start_idx, end_idx)
        ax1.plot(time_steps, mains_power[start_idx:end_idx], label='Mains Power', alpha=0.7)
        ax1.plot(time_steps, total_device_power[start_idx:end_idx], label='Sum of Devices', alpha=0.7)
        ax1.set_title('Power Consistency Analysis')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Power (W)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 一致性误差
        consistency_error = np.abs(mains_power - total_device_power)
        ax2.plot(time_steps, consistency_error[start_idx:end_idx], color='red', alpha=0.7)
        ax2.set_title('Consistency Error')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Absolute Error (W)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{split_name}_consistency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrices(self, predictions: np.ndarray, targets: np.ndarray, 
                               output_dir: Path, split_name: str):
        """绘制混淆矩阵"""
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        # 获取阈值
        thresholds = getattr(self.model, 'best_thresholds', None)
        if thresholds is None:
            thresholds = {device: 10.0 for device in self.device_names}
        
        n_devices = len(self.device_names)
        fig, axes = plt.subplots(2, (n_devices + 1) // 2, figsize=(15, 8))
        axes = axes.flatten() if n_devices > 1 else [axes]
        
        for i, device in enumerate(self.device_names):
            if i < len(axes):
                ax = axes[i]
                
                # 转换为二分类
                threshold = thresholds.get(device, 10.0)
                pred_binary = (predictions[:, i] > threshold).astype(int)
                true_binary = (targets[:, i] > threshold).astype(int)
                
                # 计算混淆矩阵
                cm = confusion_matrix(true_binary, pred_binary)
                
                # 绘制热图
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=['Off', 'On'], yticklabels=['Off', 'On'])
                ax.set_title(f'{device} Confusion Matrix')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
        
        # 隐藏多余的子图
        for i in range(n_devices, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{split_name}_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate_model(self, datamodule: NILMDataModule, output_dir: Path) -> Dict[str, Any]:
        """完整的模型评估流程"""
        
        print("=== 开始模型评估 ===")
        
        # 创建输出目录
        eval_output_dir = output_dir / "evaluation"
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = {}
        
        # 评估测试集
        if hasattr(datamodule, 'test_dataloader'):
            test_loader = datamodule.test_dataloader()
            if test_loader is not None:
                print("\n--- 评估测试集 ---")
                test_results = self.evaluate_on_dataloader(test_loader, 'test')
                all_results['test'] = test_results
                
                # 生成测试集报告
                self.generate_evaluation_report(test_results, eval_output_dir, 'test')
        
        # 评估验证集
        if hasattr(datamodule, 'val_dataloader'):
            val_loader = datamodule.val_dataloader()
            if val_loader is not None:
                print("\n--- 评估验证集 ---")
                val_results = self.evaluate_on_dataloader(val_loader, 'val')
                all_results['val'] = val_results
                
                # 生成验证集报告
                self.generate_evaluation_report(val_results, eval_output_dir, 'val')
        
        # 保存完整结果
        results_path = eval_output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print("\n=== 评估完成 ===")
        print(f"结果保存至: {eval_output_dir}")
        
        return all_results


def evaluate_model(config: DictConfig, model_path: str, output_dir: Path) -> Dict[str, Any]:
    """评估模型的主函数"""
    
    # 创建评估器
    evaluator = ModelEvaluator(config, model_path)
    
    # 创建数据模块
    datamodule = NILMDataModule(config)
    datamodule.setup()
    
    # 运行评估
    results = evaluator.evaluate_model(datamodule, output_dir)
    
    return results


def main(config: DictConfig, model_path: str = None, output_dir: Path = None) -> Dict[str, Any]:
    """主评估函数"""
    # 模型路径
    if model_path is None:
        model_path = config.get('eval_model_path', 'outputs/best_model.ckpt')
    
    # 输出目录
    if output_dir is None:
        output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行评估
    results = evaluate_model(config, model_path, output_dir)
    
    # 打印主要结果
    if 'test' in results and 'overall_score' in results['test']:
        print(f"\n测试集综合分数: {results['test']['overall_score']:.4f}")
    
    if 'val' in results and 'overall_score' in results['val']:
        print(f"验证集综合分数: {results['val']['overall_score']:.4f}")
    
    return results


# 注意：此脚本不应直接执行，请使用统一入口 main.py
# 示例：python main.py eval --checkpoint-path=path/to/model.ckpt
