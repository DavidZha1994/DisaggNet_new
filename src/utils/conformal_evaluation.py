"""
Conformal Prediction 评估和可视化工具
用于评估覆盖率、区间宽度等指标，并生成可视化报告
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import torch
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ConformalEvaluator:
    """Conformal Prediction评估器"""
    
    def __init__(self, device_names: List[str], alpha: float = 0.1):
        self.device_names = device_names
        self.alpha = alpha
        self.expected_coverage = 1 - alpha
        
        # 存储评估结果
        self.evaluation_results = {}
        self.coverage_history = {device: [] for device in device_names}
        self.interval_width_history = {device: [] for device in device_names}
        
    def evaluate_regression_intervals(self, 
                                    true_values: torch.Tensor,
                                    predicted_intervals: Dict[str, torch.Tensor],
                                    device_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """评估回归区间预测的性能"""
        
        if device_names is None:
            device_names = self.device_names
            
        results = {}
        
        for i, device in enumerate(device_names):
            device_results = {}
            
            # 提取该设备的真实值和区间
            true_vals = true_values[:, i].cpu().numpy()
            lower_bounds = predicted_intervals['lower'][:, i].cpu().numpy()
            upper_bounds = predicted_intervals['upper'][:, i].cpu().numpy()
            
            # 计算覆盖率
            coverage = self._compute_coverage(true_vals, lower_bounds, upper_bounds)
            device_results['coverage'] = coverage
            
            # 计算平均区间宽度
            interval_widths = upper_bounds - lower_bounds
            device_results['mean_interval_width'] = np.mean(interval_widths)
            device_results['median_interval_width'] = np.median(interval_widths)
            device_results['std_interval_width'] = np.std(interval_widths)
            
            # 计算效率指标（区间宽度的相对大小）
            mean_true_value = np.mean(np.abs(true_vals))
            if mean_true_value > 0:
                device_results['relative_interval_width'] = device_results['mean_interval_width'] / mean_true_value
            else:
                device_results['relative_interval_width'] = float('inf')
            
            # 计算覆盖率偏差
            device_results['coverage_deviation'] = abs(coverage - self.expected_coverage)
            
            # 条件覆盖率分析
            device_results['conditional_coverage'] = self._compute_conditional_coverage(
                true_vals, lower_bounds, upper_bounds
            )
            
            # 更新历史记录
            self.coverage_history[device].append(coverage)
            self.interval_width_history[device].append(device_results['mean_interval_width'])
            
            results[device] = device_results
        
        # 计算整体指标
        overall_coverage = np.mean([results[device]['coverage'] for device in device_names])
        overall_width = np.mean([results[device]['mean_interval_width'] for device in device_names])
        
        results['overall'] = {
            'coverage': overall_coverage,
            'mean_interval_width': overall_width,
            'coverage_deviation': abs(overall_coverage - self.expected_coverage)
        }
        
        return results
    
    def evaluate_classification_calibration(self,
                                          true_labels: torch.Tensor,
                                          predicted_probs: torch.Tensor,
                                          calibrated_thresholds: torch.Tensor,
                                          device_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """评估分类校准的性能"""
        
        if device_names is None:
            device_names = self.device_names
            
        results = {}
        
        for i, device in enumerate(device_names):
            device_results = {}
            
            # 提取该设备的数据
            true_labels_device = true_labels[:, i].cpu().numpy()
            pred_probs_device = predicted_probs[:, i].cpu().numpy()
            threshold = calibrated_thresholds[i].item()
            
            # 计算校准后的预测
            calibrated_preds = (pred_probs_device >= threshold).astype(int)
            
            # 计算准确率
            accuracy = np.mean(calibrated_preds == true_labels_device)
            device_results['accuracy'] = accuracy
            
            # 计算精确率和召回率
            tp = np.sum((calibrated_preds == 1) & (true_labels_device == 1))
            fp = np.sum((calibrated_preds == 1) & (true_labels_device == 0))
            fn = np.sum((calibrated_preds == 0) & (true_labels_device == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            device_results['precision'] = precision
            device_results['recall'] = recall
            device_results['f1_score'] = f1
            device_results['calibrated_threshold'] = threshold
            
            # 计算校准误差（可靠性图）
            calibration_error = self._compute_calibration_error(
                true_labels_device, pred_probs_device
            )
            device_results['calibration_error'] = calibration_error
            
            # 计算Brier分数
            brier_score = np.mean((pred_probs_device - true_labels_device) ** 2)
            device_results['brier_score'] = brier_score
            
            results[device] = device_results
        
        # 计算整体指标
        overall_accuracy = np.mean([results[device]['accuracy'] for device in device_names])
        overall_f1 = np.mean([results[device]['f1_score'] for device in device_names])
        overall_calibration_error = np.mean([results[device]['calibration_error'] for device in device_names])
        
        results['overall'] = {
            'accuracy': overall_accuracy,
            'f1_score': overall_f1,
            'calibration_error': overall_calibration_error
        }
        
        return results
    
    def _compute_coverage(self, true_values: np.ndarray, 
                         lower_bounds: np.ndarray, 
                         upper_bounds: np.ndarray) -> float:
        """计算覆盖率"""
        covered = (true_values >= lower_bounds) & (true_values <= upper_bounds)
        return np.mean(covered)
    
    def _compute_conditional_coverage(self, true_values: np.ndarray,
                                    lower_bounds: np.ndarray,
                                    upper_bounds: np.ndarray,
                                    n_bins: int = 10) -> Dict[str, Any]:
        """计算条件覆盖率（基于预测值的分位数）"""
        
        # 计算预测值（区间中点）
        predicted_values = (lower_bounds + upper_bounds) / 2
        
        # 将预测值分为若干个区间
        quantiles = np.linspace(0, 1, n_bins + 1)
        pred_quantiles = np.quantile(predicted_values, quantiles)
        
        conditional_coverage = []
        bin_sizes = []
        
        for i in range(n_bins):
            # 找到在当前分位数区间内的样本
            mask = (predicted_values >= pred_quantiles[i]) & (predicted_values < pred_quantiles[i + 1])
            if i == n_bins - 1:  # 最后一个区间包含右端点
                mask = (predicted_values >= pred_quantiles[i]) & (predicted_values <= pred_quantiles[i + 1])
            
            if np.sum(mask) > 0:
                # 计算该区间内的覆盖率
                bin_coverage = self._compute_coverage(
                    true_values[mask], lower_bounds[mask], upper_bounds[mask]
                )
                conditional_coverage.append(bin_coverage)
                bin_sizes.append(np.sum(mask))
            else:
                conditional_coverage.append(0.0)
                bin_sizes.append(0)
        
        return {
            'conditional_coverage': conditional_coverage,
            'bin_sizes': bin_sizes,
            'quantile_boundaries': pred_quantiles.tolist()
        }
    
    def _compute_calibration_error(self, true_labels: np.ndarray, 
                                 predicted_probs: np.ndarray,
                                 n_bins: int = 10) -> float:
        """计算校准误差（Expected Calibration Error, ECE）"""
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        total_samples = len(predicted_probs)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # 找到在当前置信度区间内的样本
            in_bin = (predicted_probs > bin_lower) & (predicted_probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # 计算该区间内的准确率和平均置信度
                accuracy_in_bin = true_labels[in_bin].mean()
                avg_confidence_in_bin = predicted_probs[in_bin].mean()
                
                # 累加加权校准误差
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def generate_visualization_report(self, 
                                    evaluation_results: Dict[str, Any],
                                    save_path: str) -> None:
        """生成可视化评估报告"""
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 设置绘图风格
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. 覆盖率对比图
        self._plot_coverage_comparison(evaluation_results, save_path / "coverage_comparison.png")
        
        # 2. 区间宽度分布图
        self._plot_interval_width_distribution(evaluation_results, save_path / "interval_width_distribution.png")
        
        # 3. 覆盖率历史趋势图
        self._plot_coverage_history(save_path / "coverage_history.png")
        
        # 4. 条件覆盖率图
        self._plot_conditional_coverage(evaluation_results, save_path / "conditional_coverage.png")
        
        # 5. 校准图（可靠性图）
        if 'classification' in str(evaluation_results):
            self._plot_reliability_diagram(evaluation_results, save_path / "reliability_diagram.png")
        
        # 6. 生成汇总报告
        self._generate_summary_report(evaluation_results, save_path / "summary_report.json")
        
        print(f"可视化报告已保存到: {save_path}")
    
    def _plot_coverage_comparison(self, results: Dict[str, Any], save_path: Path):
        """绘制覆盖率对比图"""
        
        devices = [d for d in results.keys() if d != 'overall']
        coverages = [results[device]['coverage'] for device in devices]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(devices, coverages, alpha=0.7)
        plt.axhline(y=self.expected_coverage, color='red', linestyle='--', 
                   label=f'Expected Coverage ({self.expected_coverage:.1%})')
        
        # 添加数值标签
        for bar, coverage in zip(bars, coverages):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{coverage:.3f}', ha='center', va='bottom')
        
        plt.xlabel('Device')
        plt.ylabel('Coverage Rate')
        plt.title('Conformal Prediction Coverage Rate by Device')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_interval_width_distribution(self, results: Dict[str, Any], save_path: Path):
        """绘制区间宽度分布图"""
        
        devices = [d for d in results.keys() if d != 'overall']
        widths = [results[device]['mean_interval_width'] for device in devices]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(devices, widths, alpha=0.7, color='orange')
        
        # 添加数值标签
        for bar, width in zip(bars, widths):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(widths) * 0.01,
                    f'{width:.2f}', ha='center', va='bottom')
        
        plt.xlabel('Device')
        plt.ylabel('Mean Interval Width')
        plt.title('Mean Prediction Interval Width by Device')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_coverage_history(self, save_path: Path):
        """绘制覆盖率历史趋势图"""
        
        plt.figure(figsize=(12, 8))
        
        for device in self.device_names:
            if len(self.coverage_history[device]) > 1:
                plt.plot(self.coverage_history[device], label=device, marker='o', markersize=4)
        
        plt.axhline(y=self.expected_coverage, color='red', linestyle='--', 
                   label=f'Expected Coverage ({self.expected_coverage:.1%})')
        
        plt.xlabel('Evaluation Round')
        plt.ylabel('Coverage Rate')
        plt.title('Coverage Rate History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_conditional_coverage(self, results: Dict[str, Any], save_path: Path):
        """绘制条件覆盖率图"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        devices = [d for d in results.keys() if d != 'overall'][:4]  # 最多显示4个设备
        
        for i, device in enumerate(devices):
            if 'conditional_coverage' in results[device]:
                cond_cov = results[device]['conditional_coverage']
                bin_sizes = cond_cov['conditional_coverage']
                
                axes[i].bar(range(len(bin_sizes)), bin_sizes, alpha=0.7)
                axes[i].axhline(y=self.expected_coverage, color='red', linestyle='--')
                axes[i].set_title(f'{device} - Conditional Coverage')
                axes[i].set_xlabel('Prediction Quantile Bin')
                axes[i].set_ylabel('Coverage Rate')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_reliability_diagram(self, results: Dict[str, Any], save_path: Path):
        """绘制可靠性图（校准图）"""
        
        # 这里需要原始的预测概率和真实标签数据
        # 由于当前结果中没有这些数据，这里提供一个框架
        plt.figure(figsize=(10, 8))
        
        # 绘制完美校准线
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Reliability Diagram (Calibration Plot)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_summary_report(self, results: Dict[str, Any], save_path: Path):
        """生成汇总报告"""
        
        summary = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'expected_coverage': self.expected_coverage,
            'alpha': self.alpha,
            'device_count': len(self.device_names),
            'results': results,
            'recommendations': self._generate_recommendations(results)
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """基于评估结果生成建议"""
        
        recommendations = []
        
        # 检查整体覆盖率
        if 'overall' in results:
            overall_coverage = results['overall']['coverage']
            coverage_deviation = results['overall']['coverage_deviation']
            
            if coverage_deviation > 0.05:  # 偏差超过5%
                if overall_coverage < self.expected_coverage:
                    recommendations.append(
                        f"整体覆盖率({overall_coverage:.3f})低于期望值({self.expected_coverage:.3f})，"
                        "建议增加alpha值或检查标定数据质量"
                    )
                else:
                    recommendations.append(
                        f"整体覆盖率({overall_coverage:.3f})高于期望值({self.expected_coverage:.3f})，"
                        "区间可能过于保守，建议减少alpha值"
                    )
        
        # 检查各设备的覆盖率
        devices = [d for d in results.keys() if d != 'overall']
        for device in devices:
            if 'coverage_deviation' in results[device]:
                deviation = results[device]['coverage_deviation']
                if deviation > 0.1:  # 单设备偏差超过10%
                    recommendations.append(
                        f"设备{device}的覆盖率偏差较大({deviation:.3f})，"
                        "建议检查该设备的数据分布或使用设备特定的标定"
                    )
        
        # 检查区间宽度
        if 'overall' in results and 'mean_interval_width' in results['overall']:
            overall_width = results['overall']['mean_interval_width']
            if overall_width > 100:  # 假设功率单位为瓦特
                recommendations.append(
                    f"平均区间宽度({overall_width:.2f})较大，可能影响实用性，"
                    "建议优化模型或使用更精确的conformal方法"
                )
        
        if not recommendations:
            recommendations.append("Conformal prediction性能良好，覆盖率和区间宽度都在合理范围内")
        
        return recommendations


def create_conformal_evaluator(device_names: List[str], alpha: float = 0.1) -> ConformalEvaluator:
    """创建Conformal Prediction评估器的工厂函数"""
    return ConformalEvaluator(device_names, alpha)