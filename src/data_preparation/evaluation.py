"""
评估模块
Evaluation Module

实现时序数据的评估系统，包括：
- 事件级指标计算
- 模型性能评估
- 报告生成
- 可视化分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, roc_curve,
    confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class EventEvaluator:
    """事件评估器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.evaluation_config = config['evaluation']
        self.metrics_config = self.evaluation_config['metrics']
        self.visualization_config = self.evaluation_config['visualization']
        
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           y_prob: Optional[np.ndarray] = None,
                           metadata: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        评估预测结果
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_prob: 预测概率
            metadata: 元数据
            
        Returns:
            evaluation_results: 评估结果字典
        """
        logger.info("开始评估预测结果")
        
        results = {}
        
        # 基础分类指标
        results['basic_metrics'] = self._calculate_basic_metrics(y_true, y_pred, y_prob)
        
        # 事件级指标
        if metadata is not None:
            results['event_metrics'] = self._calculate_event_metrics(y_true, y_pred, metadata)
        
        # 设备级指标
        if metadata is not None and 'device_name' in metadata.columns:
            results['device_metrics'] = self._calculate_device_metrics(y_true, y_pred, metadata)
        
        # 时间级指标
        if metadata is not None and 'start_ts' in metadata.columns:
            results['temporal_metrics'] = self._calculate_temporal_metrics(y_true, y_pred, metadata)
        
        # 校准指标
        if y_prob is not None:
            results['calibration_metrics'] = self._calculate_calibration_metrics(y_true, y_prob)
        
        logger.info("预测结果评估完成")
        return results
    
    def _calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """计算基础分类指标"""
        metrics = {}
        
        # 准确率
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # 精确率、召回率、F1分数
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # 各类别的详细指标
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        for i, cls in enumerate(unique_classes):
            if i < len(precision_per_class):
                metrics[f'precision_class_{cls}'] = precision_per_class[i]
                metrics[f'recall_class_{cls}'] = recall_per_class[i]
                metrics[f'f1_class_{cls}'] = f1_per_class[i]
        
        # ROC-AUC（如果有概率预测）
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) == 2:  # 二分类
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
                else:  # 多分类
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
            except ValueError as e:
                logger.warning(f"无法计算ROC-AUC: {e}")
                metrics['roc_auc'] = np.nan
        
        # PR-AUC
        if y_prob is not None and len(np.unique(y_true)) == 2:
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
            metrics['pr_auc'] = np.trapz(precision_curve, recall_curve)
        
        return metrics
    
    def _calculate_event_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                metadata: pd.DataFrame) -> Dict[str, Any]:
        """计算事件级指标"""
        event_metrics = {}
        
        # 事件检测指标
        true_events = self._extract_events(y_true, metadata)
        pred_events = self._extract_events(y_pred, metadata)
        
        # 事件级精确率和召回率
        event_precision, event_recall, event_f1 = self._calculate_event_precision_recall(
            true_events, pred_events
        )
        
        event_metrics['event_precision'] = event_precision
        event_metrics['event_recall'] = event_recall
        event_metrics['event_f1'] = event_f1
        
        # 事件数量统计
        event_metrics['true_event_count'] = len(true_events)
        event_metrics['pred_event_count'] = len(pred_events)
        
        # 漏检和误检分析
        missed_events, false_events = self._analyze_detection_errors(true_events, pred_events)
        event_metrics['missed_events'] = len(missed_events)
        event_metrics['false_events'] = len(false_events)
        
        return event_metrics
    
    def _extract_events(self, labels: np.ndarray, metadata: pd.DataFrame) -> List[Dict]:
        """从标签中提取事件"""
        events = []
        
        # 找到事件窗口
        event_indices = np.where(labels == 1)[0]
        
        if len(event_indices) == 0:
            return events
        
        # 合并连续的事件窗口
        event_groups = []
        current_group = [event_indices[0]]
        
        for i in range(1, len(event_indices)):
            if event_indices[i] - event_indices[i-1] <= 1:  # 连续或相邻
                current_group.append(event_indices[i])
            else:
                event_groups.append(current_group)
                current_group = [event_indices[i]]
        
        event_groups.append(current_group)
        
        # 创建事件记录
        for group in event_groups:
            start_idx = group[0]
            end_idx = group[-1]
            
            event = {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'duration': end_idx - start_idx + 1,
                'start_ts': metadata.iloc[start_idx]['start_ts'] if 'start_ts' in metadata.columns else None,
                'end_ts': metadata.iloc[end_idx]['end_ts'] if 'end_ts' in metadata.columns else None,
                'device_name': metadata.iloc[start_idx]['device_name'] if 'device_name' in metadata.columns else None
            }
            events.append(event)
        
        return events
    
    def _calculate_event_precision_recall(self, true_events: List[Dict], 
                                         pred_events: List[Dict]) -> Tuple[float, float, float]:
        """计算事件级精确率和召回率"""
        if len(pred_events) == 0:
            precision = 1.0 if len(true_events) == 0 else 0.0
            recall = 1.0 if len(true_events) == 0 else 0.0
            f1 = 1.0 if len(true_events) == 0 else 0.0
            return precision, recall, f1
        
        if len(true_events) == 0:
            return 0.0, 1.0, 0.0
        
        # 匹配事件
        matched_true = set()
        matched_pred = set()
        
        for i, pred_event in enumerate(pred_events):
            for j, true_event in enumerate(true_events):
                if j in matched_true:
                    continue
                
                # 检查时间重叠
                if self._events_overlap(pred_event, true_event):
                    matched_true.add(j)
                    matched_pred.add(i)
                    break
        
        # 计算指标
        precision = len(matched_pred) / len(pred_events)
        recall = len(matched_true) / len(true_events)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    def _events_overlap(self, event1: Dict, event2: Dict) -> bool:
        """检查两个事件是否重叠"""
        # 使用时间戳比较（如果可用）
        if event1.get('start_ts') and event2.get('start_ts'):
            return not (event1['end_ts'] < event2['start_ts'] or event2['end_ts'] < event1['start_ts'])
        
        # 使用索引比较
        return not (event1['end_idx'] < event2['start_idx'] or event2['end_idx'] < event1['start_idx'])
    
    def _analyze_detection_errors(self, true_events: List[Dict], 
                                 pred_events: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """分析检测错误"""
        matched_true = set()
        matched_pred = set()
        
        # 找到匹配的事件
        for i, pred_event in enumerate(pred_events):
            for j, true_event in enumerate(true_events):
                if j in matched_true:
                    continue
                
                if self._events_overlap(pred_event, true_event):
                    matched_true.add(j)
                    matched_pred.add(i)
                    break
        
        # 漏检事件
        missed_events = [true_events[i] for i in range(len(true_events)) if i not in matched_true]
        
        # 误检事件
        false_events = [pred_events[i] for i in range(len(pred_events)) if i not in matched_pred]
        
        return missed_events, false_events
    
    def _calculate_device_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 metadata: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """计算设备级指标"""
        device_metrics = {}
        
        unique_devices = metadata['device_name'].unique()
        
        for device in unique_devices:
            device_mask = metadata['device_name'] == device
            device_y_true = y_true[device_mask]
            device_y_pred = y_pred[device_mask]
            
            if len(device_y_true) > 0:
                device_metrics[device] = self._calculate_basic_metrics(device_y_true, device_y_pred)
        
        return device_metrics
    
    def _calculate_temporal_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   metadata: pd.DataFrame) -> Dict[str, Any]:
        """计算时间级指标"""
        temporal_metrics = {}
        
        # 按小时统计
        metadata['hour'] = pd.to_datetime(metadata['start_ts'], unit='s').dt.hour
        hourly_metrics = {}
        
        for hour in range(24):
            hour_mask = metadata['hour'] == hour
            if np.sum(hour_mask) > 0:
                hour_y_true = y_true[hour_mask]
                hour_y_pred = y_pred[hour_mask]
                hourly_metrics[hour] = self._calculate_basic_metrics(hour_y_true, hour_y_pred)
        
        temporal_metrics['hourly'] = hourly_metrics
        
        # 按星期几统计
        metadata['weekday'] = pd.to_datetime(metadata['start_ts'], unit='s').dt.weekday
        weekday_metrics = {}
        
        for weekday in range(7):
            weekday_mask = metadata['weekday'] == weekday
            if np.sum(weekday_mask) > 0:
                weekday_y_true = y_true[weekday_mask]
                weekday_y_pred = y_pred[weekday_mask]
                weekday_metrics[weekday] = self._calculate_basic_metrics(weekday_y_true, weekday_y_pred)
        
        temporal_metrics['weekday'] = weekday_metrics
        
        return temporal_metrics
    
    def _calculate_calibration_metrics(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """计算校准指标"""
        calibration_metrics = {}
        
        try:
            # 校准曲线
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=10
            )
            
            # Brier分数
            brier_score = np.mean((y_prob - y_true) ** 2)
            
            # 可靠性图数据
            calibration_metrics['fraction_of_positives'] = fraction_of_positives.tolist()
            calibration_metrics['mean_predicted_value'] = mean_predicted_value.tolist()
            calibration_metrics['brier_score'] = brier_score
            
            # 校准误差
            calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            calibration_metrics['calibration_error'] = calibration_error
            
        except Exception as e:
            logger.warning(f"校准指标计算失败: {e}")
            calibration_metrics['error'] = str(e)
        
        return calibration_metrics
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any], 
                                  output_path: str, fold_id: Optional[int] = None):
        """
        生成评估报告
        
        Args:
            evaluation_results: 评估结果
            output_path: 输出路径
            fold_id: 折ID
        """
        logger.info(f"生成评估报告: {output_path}")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'fold_id': fold_id,
            'evaluation_results': evaluation_results
        }
        
        # 保存JSON报告
        json_path = output_path.replace('.html', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成HTML报告
        self._generate_html_report(report, output_path)
        
        logger.info(f"评估报告已保存: {output_path}")
    
    def _generate_html_report(self, report: Dict[str, Any], output_path: str):
        """生成HTML报告"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>模型评估报告</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metrics-table {{ border-collapse: collapse; width: 100%; }}
                .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .metrics-table th {{ background-color: #f2f2f2; }}
                .highlight {{ background-color: #ffffcc; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>模型评估报告</h1>
                <p>生成时间: {report['timestamp']}</p>
                {f"<p>交叉验证折: {report['fold_id']}</p>" if report.get('fold_id') is not None else ""}
            </div>
            
            {self._generate_metrics_section(report['evaluation_results'])}
            
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_metrics_section(self, results: Dict[str, Any]) -> str:
        """生成指标部分的HTML"""
        html = ""
        
        # 基础指标
        if 'basic_metrics' in results:
            html += self._generate_basic_metrics_html(results['basic_metrics'])
        
        # 事件级指标
        if 'event_metrics' in results:
            html += self._generate_event_metrics_html(results['event_metrics'])
        
        # 设备级指标
        if 'device_metrics' in results:
            html += self._generate_device_metrics_html(results['device_metrics'])
        
        return html
    
    def _generate_basic_metrics_html(self, metrics: Dict[str, float]) -> str:
        """生成基础指标HTML"""
        html = """
        <div class="section">
            <h2>基础分类指标</h2>
            <table class="metrics-table">
                <tr><th>指标</th><th>值</th></tr>
        """
        
        key_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
        
        for metric in key_metrics:
            if metric in metrics:
                value = metrics[metric]
                if not np.isnan(value):
                    html += f"<tr><td>{metric.upper()}</td><td>{value:.4f}</td></tr>"
        
        html += "</table></div>"
        return html
    
    def _generate_event_metrics_html(self, metrics: Dict[str, Any]) -> str:
        """生成事件级指标HTML"""
        html = """
        <div class="section">
            <h2>事件级指标</h2>
            <table class="metrics-table">
                <tr><th>指标</th><th>值</th></tr>
        """
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                val_str = f"{value:.4f}" if isinstance(value, float) else str(value)
                html += f"<tr><td>{key}</td><td>{val_str}</td></tr>"
        
        html += "</table></div>"
        return html
    
    def _generate_device_metrics_html(self, metrics: Dict[str, Dict[str, float]]) -> str:
        """生成设备级指标HTML"""
        html = """
        <div class="section">
            <h2>设备级指标</h2>
            <table class="metrics-table">
                <tr><th>设备</th><th>准确率</th><th>精确率</th><th>召回率</th><th>F1分数</th></tr>
        """
        
        for device, device_metrics in metrics.items():
            accuracy = device_metrics.get('accuracy', 0)
            precision = device_metrics.get('precision', 0)
            recall = device_metrics.get('recall', 0)
            f1 = device_metrics.get('f1', 0)
            
            html += f"""
            <tr>
                <td>{device}</td>
                <td>{accuracy:.4f}</td>
                <td>{precision:.4f}</td>
                <td>{recall:.4f}</td>
                <td>{f1:.4f}</td>
            </tr>
            """
        
        html += "</table></div>"
        return html
    
    def create_visualization_plots(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  y_prob: Optional[np.ndarray] = None,
                                  metadata: Optional[pd.DataFrame] = None,
                                  output_dir: str = "plots"):
        """
        创建可视化图表
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_prob: 预测概率
            metadata: 元数据
            output_dir: 输出目录
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 混淆矩阵
        self._plot_confusion_matrix(y_true, y_pred, f"{output_dir}/confusion_matrix.png")
        
        # ROC曲线
        if y_prob is not None and len(np.unique(y_true)) == 2:
            self._plot_roc_curve(y_true, y_prob, f"{output_dir}/roc_curve.png")
        
        # PR曲线
        if y_prob is not None and len(np.unique(y_true)) == 2:
            self._plot_pr_curve(y_true, y_prob, f"{output_dir}/pr_curve.png")
        
        # 校准曲线
        if y_prob is not None:
            self._plot_calibration_curve(y_true, y_prob, f"{output_dir}/calibration_curve.png")
        
        # 时间序列预测结果
        if metadata is not None and 'start_ts' in metadata.columns:
            self._plot_prediction_timeline(y_true, y_pred, metadata, f"{output_dir}/prediction_timeline.png")
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, output_path: str):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, output_path: str):
        """绘制ROC曲线"""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC曲线 (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='随机分类器')
        plt.xlabel('假正率')
        plt.ylabel('真正率')
        plt.title('ROC曲线')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pr_curve(self, y_true: np.ndarray, y_prob: np.ndarray, output_path: str):
        """绘制PR曲线"""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = np.trapz(precision, recall)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR曲线 (AUC = {pr_auc:.3f})')
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('精确率-召回率曲线')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_calibration_curve(self, y_true: np.ndarray, y_prob: np.ndarray, output_path: str):
        """绘制校准曲线"""
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10)
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="模型")
        plt.plot([0, 1], [0, 1], "k:", label="完美校准")
        plt.xlabel('平均预测概率')
        plt.ylabel('正例比例')
        plt.title('校准曲线')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_prediction_timeline(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 metadata: pd.DataFrame, output_path: str):
        """绘制预测时间线"""
        timestamps = pd.to_datetime(metadata['start_ts'], unit='s')
        
        plt.figure(figsize=(15, 6))
        
        # 真实标签
        true_events = timestamps[y_true == 1]
        plt.scatter(true_events, np.ones(len(true_events)), 
                   c='green', marker='o', s=50, alpha=0.7, label='真实事件')
        
        # 预测标签
        pred_events = timestamps[y_pred == 1]
        plt.scatter(pred_events, np.ones(len(pred_events)) * 0.5, 
                   c='red', marker='x', s=50, alpha=0.7, label='预测事件')
        
        plt.ylim(0, 1.5)
        plt.xlabel('时间')
        plt.ylabel('事件')
        plt.title('事件检测时间线')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()