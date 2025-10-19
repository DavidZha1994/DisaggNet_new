#!/usr/bin/env python3
"""
数据准备流程使用示例
Data Preparation Pipeline Usage Examples

展示如何使用时序数据准备与交叉验证系统
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_preparation import HIPEDataPreparationPipeline

def create_sample_data():
    """创建示例数据"""
    print("创建示例时序数据...")
    
    # 设置参数
    devices = ['device_001', 'device_002', 'device_003']
    start_time = datetime(2024, 1, 1)
    sampling_interval = 5  # 5秒采样间隔
    
    all_data = []
    
    for device in devices:
        print(f"  生成设备 {device} 的数据...")
        
        # 为每个设备生成多个连续段
        segments_per_device = 3
        
        for segment_idx in range(segments_per_device):
            # 每段数据长度 (2-6小时)
            segment_duration_hours = np.random.uniform(2, 6)
            segment_samples = int(segment_duration_hours * 3600 / sampling_interval)
            
            # 段开始时间 (加入一些随机间隔)
            if segment_idx == 0:
                segment_start = start_time
            else:
                # 添加大间隔 (5-30分钟)
                gap_minutes = np.random.uniform(5, 30)
                segment_start = segment_start + timedelta(minutes=gap_minutes)
            
            # 生成时间戳
            timestamps = [
                segment_start + timedelta(seconds=i * sampling_interval)
                for i in range(segment_samples)
            ]
            
            # 生成模拟的电力数据
            t = np.arange(segment_samples)
            
            # 基础功率 (带趋势和周期性)
            base_power = 1000 + 200 * np.sin(2 * np.pi * t / (12 * 3600 / sampling_interval))  # 12小时周期
            base_power += 50 * np.sin(2 * np.pi * t / (3600 / sampling_interval))  # 1小时周期
            base_power += np.random.normal(0, 20, segment_samples)  # 噪声
            
            # 电压 (三相)
            voltage_base = 220
            V1_V = voltage_base + np.random.normal(0, 5, segment_samples)
            V2_V = voltage_base + np.random.normal(0, 5, segment_samples)
            V3_V = voltage_base + np.random.normal(0, 5, segment_samples)
            
            # 电流 (基于功率计算，加入一些变化)
            I1_A = base_power / (V1_V * np.sqrt(3)) + np.random.normal(0, 0.1, segment_samples)
            I2_A = base_power / (V2_V * np.sqrt(3)) + np.random.normal(0, 0.1, segment_samples)
            I3_A = base_power / (V3_V * np.sqrt(3)) + np.random.normal(0, 0.1, segment_samples)
            
            # 频率
            frequency = 50 + np.random.normal(0, 0.1, segment_samples)
            
            # THD
            THD_V = np.random.uniform(1, 5, segment_samples)
            THD_I = np.random.uniform(2, 8, segment_samples)
            
            # 生成一些事件 (开关状态变化)
            switch_state = np.zeros(segment_samples)
            
            # 随机生成一些开关事件
            num_events = np.random.poisson(3)  # 平均每段3个事件
            if num_events > 0:
                event_positions = np.random.choice(segment_samples, min(num_events, segment_samples//10), replace=False)
                for pos in event_positions:
                    # 事件持续时间 (5-30分钟)
                    event_duration = np.random.randint(60, 360)  # 5-30分钟，以采样点计
                    end_pos = min(pos + event_duration, segment_samples)
                    switch_state[pos:end_pos] = 1
                    
                    # 事件期间功率变化
                    power_change = np.random.uniform(0.5, 1.5)  # 功率变化倍数
                    base_power[pos:end_pos] *= power_change
            
            # 创建数据框
            segment_data = pd.DataFrame({
                'ts_utc': [int(ts.timestamp()) for ts in timestamps],
                'device_name': device,
                'P_W': base_power,
                'V1_V': V1_V,
                'V2_V': V2_V,
                'V3_V': V3_V,
                'I1_A': I1_A,
                'I2_A': I2_A,
                'I3_A': I3_A,
                'frequency': frequency,
                'THD_V': THD_V,
                'THD_I': THD_I,
                'switch_state': switch_state
            })
            
            all_data.append(segment_data)
            
            # 更新下一段的开始时间
            segment_start = timestamps[-1] + timedelta(seconds=sampling_interval)
    
    # 合并所有数据
    full_data = pd.concat(all_data, ignore_index=True)
    
    # 按时间排序
    full_data = full_data.sort_values('ts_utc').reset_index(drop=True)
    
    print(f"示例数据创建完成:")
    print(f"  总样本数: {len(full_data)}")
    print(f"  设备数: {len(full_data['device_name'].unique())}")
    print(f"  时间范围: {datetime.fromtimestamp(full_data['ts_utc'].min())} 到 {datetime.fromtimestamp(full_data['ts_utc'].max())}")
    print(f"  事件总数: {full_data['switch_state'].sum()}")
    
    return full_data

def example_1_basic_usage():
    """示例1: 基本使用流程"""
    print("\n" + "="*60)
    print("示例1: 基本使用流程")
    print("="*60)
    
    # 创建示例数据
    data = create_sample_data()
    
    # 保存示例数据
    data_path = "Data/sample_data.csv"
    os.makedirs("Data", exist_ok=True)
    data.to_csv(data_path, index=False)
    print(f"示例数据已保存到: {data_path}")
    
    # 创建流程实例
    config_path = "config/prep_config.yaml"
    if not os.path.exists(config_path):
        print(f"错误: 配置文件 {config_path} 不存在")
        print("请先运行数据准备流程创建配置文件")
        return
    
    pipeline = HIPEDataPreparationPipeline(config_path)
    
    # 设置输出目录
    output_dir = "Data/prepared_sample"
    pipeline.output_dir = output_dir
    pipeline.config['data_storage']['output_directory'] = output_dir
    
    # 运行完整流程
    print("\n运行数据准备流程...")
    results = pipeline.run_full_pipeline(data_path)
    
    print(f"\n流程运行完成! 状态: {results['status']}")
    print(f"输出目录: {output_dir}")
    
    # 显示摘要
    summary = pipeline.get_pipeline_summary()
    if 'segments_summary' in summary:
        seg_summary = summary['segments_summary']
        print(f"\n数据摘要:")
        print(f"  总段数: {seg_summary['total_segments']}")
        print(f"  总样本数: {seg_summary['total_samples']}")
        print(f"  设备数: {len(seg_summary['devices'])}")

def example_2_load_and_use_data():
    """示例2: 加载和使用处理后的数据"""
    print("\n" + "="*60)
    print("示例2: 加载和使用处理后的数据")
    print("="*60)
    
    config_path = "config/prep_config.yaml"
    if not os.path.exists(config_path):
        print(f"错误: 配置文件 {config_path} 不存在")
        return
    
    pipeline = HIPEDataPreparationPipeline(config_path)
    pipeline.output_dir = "Data/prepared_sample"
    
    # 检查是否有处理后的数据
    if not os.path.exists(pipeline.output_dir):
        print(f"错误: 处理后的数据目录不存在: {pipeline.output_dir}")
        print("请先运行示例1生成数据")
        return
    
    try:
        # 加载第一折的数据
        fold_data = pipeline.load_processed_data(fold_idx=0)
        
        print("成功加载第0折数据:")
        print(f"  训练特征形状: {fold_data['train_features'].shape}")
        print(f"  验证特征形状: {fold_data['val_features'].shape}")
        print(f"  训练标签形状: {fold_data['train_labels'].shape}")
        print(f"  验证标签形状: {fold_data['val_labels'].shape}")
        print(f"  特征数量: {len(fold_data['feature_names'])}")
        
        # 显示标签分布
        train_label_dist = np.bincount(fold_data['train_labels'].astype(int))
        val_label_dist = np.bincount(fold_data['val_labels'].astype(int))
        
        print(f"\n标签分布:")
        print(f"  训练集: {dict(enumerate(train_label_dist))}")
        print(f"  验证集: {dict(enumerate(val_label_dist))}")
        
        # 显示一些特征名称
        print(f"\n前10个特征名称:")
        for i, name in enumerate(fold_data['feature_names'][:10]):
            print(f"  {i}: {name}")
        
        return fold_data
        
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

def example_3_model_training_simulation():
    """示例3: 模拟模型训练和评估"""
    print("\n" + "="*60)
    print("示例3: 模拟模型训练和评估")
    print("="*60)
    
    # 加载数据
    fold_data = example_2_load_and_use_data()
    if fold_data is None:
        return
    
    # 模拟一个简单的模型预测
    print("\n模拟模型训练和预测...")
    
    # 使用简单的逻辑回归作为示例
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    
    # 训练模型
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(fold_data['train_features'], fold_data['train_labels'])
    
    # 预测
    y_pred = model.predict(fold_data['val_features'])
    y_prob = model.predict_proba(fold_data['val_features'])[:, 1]
    
    print("模型预测完成")
    print(f"预测标签形状: {y_pred.shape}")
    print(f"预测概率形状: {y_prob.shape}")
    
    # 使用流程的评估功能
    config_path = "config/prep_config.yaml"
    pipeline = HIPEDataPreparationPipeline(config_path)
    pipeline.output_dir = "Data/prepared_sample"
    
    try:
        evaluation_results = pipeline.evaluate_fold(fold_idx=0, y_pred=y_pred, y_prob=y_prob)
        
        print("\n评估结果:")
        if 'basic_metrics' in evaluation_results:
            basic_metrics = evaluation_results['basic_metrics']
            print(f"  准确率: {basic_metrics.get('accuracy', 0):.4f}")
            print(f"  精确率: {basic_metrics.get('precision', 0):.4f}")
            print(f"  召回率: {basic_metrics.get('recall', 0):.4f}")
            print(f"  F1分数: {basic_metrics.get('f1', 0):.4f}")
            if 'roc_auc' in basic_metrics:
                print(f"  ROC-AUC: {basic_metrics['roc_auc']:.4f}")
            if 'pr_auc' in basic_metrics:
                print(f"  PR-AUC: {basic_metrics['pr_auc']:.4f}")
        
        if 'event_metrics' in evaluation_results:
            event_metrics = evaluation_results['event_metrics']
            print(f"\n事件级指标:")
            print(f"  事件精确率: {event_metrics.get('event_precision', 0):.4f}")
            print(f"  事件召回率: {event_metrics.get('event_recall', 0):.4f}")
            print(f"  事件F1分数: {event_metrics.get('event_f1', 0):.4f}")
            print(f"  真实事件数: {event_metrics.get('true_event_count', 0)}")
            print(f"  预测事件数: {event_metrics.get('pred_event_count', 0)}")
        
        print(f"\n评估报告和可视化图表已保存到:")
        print(f"  {pipeline.output_dir}/fold_0/evaluation_report.html")
        print(f"  {pipeline.output_dir}/fold_0/plots/")
        
    except Exception as e:
        print(f"评估失败: {e}")
        import traceback
        traceback.print_exc()

def example_4_cross_validation():
    """示例4: 完整的交叉验证流程"""
    print("\n" + "="*60)
    print("示例4: 完整的交叉验证流程")
    print("="*60)
    
    config_path = "config/prep_config.yaml"
    if not os.path.exists(config_path):
        print(f"错误: 配置文件 {config_path} 不存在")
        return
    
    pipeline = HIPEDataPreparationPipeline(config_path)
    pipeline.output_dir = "Data/prepared_sample"
    
    # 检查有多少折
    cv_splits_path = os.path.join(pipeline.output_dir, 'cv_splits.pkl')
    if not os.path.exists(cv_splits_path):
        print(f"错误: 交叉验证分割文件不存在: {cv_splits_path}")
        print("请先运行示例1生成数据")
        return
    
    import pickle
    with open(cv_splits_path, 'rb') as f:
        cv_splits = pickle.load(f)
    
    print(f"发现 {len(cv_splits)} 个交叉验证折")
    
    # 对每一折进行训练和评估
    from sklearn.linear_model import LogisticRegression
    
    all_results = []
    
    for fold_idx in range(len(cv_splits)):
        print(f"\n处理第 {fold_idx + 1} 折...")
        
        try:
            # 加载数据
            fold_data = pipeline.load_processed_data(fold_idx)
            
            # 训练模型
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(fold_data['train_features'], fold_data['train_labels'])
            
            # 预测
            y_pred = model.predict(fold_data['val_features'])
            y_prob = model.predict_proba(fold_data['val_features'])[:, 1]
            
            # 评估
            evaluation_results = pipeline.evaluate_fold(fold_idx, y_pred, y_prob)
            all_results.append(evaluation_results)
            
            # 显示基本指标
            if 'basic_metrics' in evaluation_results:
                basic_metrics = evaluation_results['basic_metrics']
                print(f"  准确率: {basic_metrics.get('accuracy', 0):.4f}")
                print(f"  F1分数: {basic_metrics.get('f1', 0):.4f}")
                if 'roc_auc' in basic_metrics:
                    print(f"  ROC-AUC: {basic_metrics['roc_auc']:.4f}")
        
        except Exception as e:
            print(f"  第 {fold_idx + 1} 折处理失败: {e}")
    
    # 计算平均性能
    if all_results:
        print(f"\n交叉验证平均结果:")
        
        # 计算基本指标的平均值
        metrics_to_average = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
        
        for metric in metrics_to_average:
            values = []
            for result in all_results:
                if 'basic_metrics' in result and metric in result['basic_metrics']:
                    value = result['basic_metrics'][metric]
                    if not np.isnan(value):
                        values.append(value)
            
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"  {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")

def main():
    """主函数"""
    print("时序数据准备与交叉验证系统 - 使用示例")
    print("="*60)
    
    # 检查配置文件
    config_path = "prep_config.yaml"
    if not os.path.exists(config_path):
        print(f"警告: 配置文件 {config_path} 不存在")
        print("某些示例可能无法运行")
        print("请先创建配置文件或运行主流程")
    
    try:
        # 运行示例
        example_1_basic_usage()
        example_2_load_and_use_data()
        example_3_model_training_simulation()
        example_4_cross_validation()
        
        print("\n" + "="*60)
        print("所有示例运行完成！")
        print("="*60)
        print("生成的文件:")
        print("  Data/sample_data.csv - 示例原始数据")
        print("  Data/prepared_sample/ - 处理后的数据")
        print("    ├── segments_meta.csv - 段元数据")
        print("    ├── cv_splits.pkl - 交叉验证分割")
        print("    ├── labels.pkl - 标签数据")
        print("    ├── fold_0/, fold_1/, ... - 各折的数据和结果")
        print("    └── pipeline_results.json - 流程结果")
        
    except KeyboardInterrupt:
        print("\n用户中断操作")
    except Exception as e:
        print(f"\n示例运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()