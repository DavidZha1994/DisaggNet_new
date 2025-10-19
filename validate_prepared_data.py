#!/usr/bin/env python3
"""
验证准备好的数据
"""
import json
import os
from src.data_preparation.hipe_pipeline import HIPEDataPreparationPipeline

def validate_prepared_data():
    """验证准备好的数据"""
    data_dir = "Data/prepared"
    
    print("=" * 60)
    print("验证准备好的数据")
    print("=" * 60)
    
    # 1. 检查基本文件
    required_files = ['cv_splits.pkl', 'labels.pkl', 'segments_meta.csv', 'pipeline_results.json']
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"✓ {file}: {size:.2f} MB")
        else:
            print(f"✗ {file}: 缺失")
    
    # 2. 检查fold目录
    fold_dirs = [d for d in os.listdir(data_dir) if d.startswith('fold_')]
    print(f"\n找到 {len(fold_dirs)} 个fold目录:")
    
    for fold_dir in sorted(fold_dirs):
        fold_path = os.path.join(data_dir, fold_dir)
        print(f"\n{fold_dir}:")
        
        # 检查fold文件
        fold_files = os.listdir(fold_path)
        for file in fold_files:
            file_path = os.path.join(fold_path, file)
            size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"  - {file}: {size:.2f} MB")
    
    # 3. 加载并验证数据
    print("\n" + "=" * 60)
    print("数据验证")
    print("=" * 60)
    
    try:
        # 加载pipeline结果
        with open(os.path.join(data_dir, 'pipeline_results.json'), 'r') as f:
            results = json.load(f)
        
        print(f"Pipeline状态: {results['status']}")
        print(f"开始时间: {results['start_time']}")
        print(f"结束时间: {results['end_time']}")
        
        # 步骤执行情况
        steps = results['steps']
        print(f"\n步骤执行情况:")
        for step_name, step_info in steps.items():
            print(f"  - {step_name}: {step_info['status']}")
            if 'details' in step_info:
                for key, value in step_info['details'].items():
                    print(f"    * {key}: {value}")
        
        # 从步骤中提取数据摘要
        if 'segmentation' in steps:
            seg_details = steps['segmentation'].get('details', {})
            print(f"\n数据摘要:")
            if 'segments_count' in seg_details:
                print(f"  - 总段数: {seg_details['segments_count']}")
            if 'total_samples' in seg_details:
                print(f"  - 总样本数: {seg_details['total_samples']:,}")
        
        if 'windowing' in steps:
            win_details = steps['windowing'].get('details', {})
            if 'total_windows' in win_details:
                print(f"  - 总窗口数: {win_details['total_windows']:,}")
            if 'window_length' in win_details:
                print(f"  - 窗口长度: {win_details['window_length']}")
        
        if 'cross_validation' in steps:
            cv_details = steps['cross_validation'].get('details', {})
            if 'n_folds' in cv_details:
                print(f"  - 交叉验证折数: {cv_details['n_folds']}")
        
        # 4. 验证fold数据
        print(f"\n" + "=" * 60)
        print("Fold数据验证")
        print("=" * 60)
        
        pipeline = HIPEDataPreparationPipeline('config/prep_config.yaml')
        
        for i in range(len(fold_dirs)):
            try:
                fold_data = pipeline.load_processed_data(i)
                
                train_features = fold_data['train_features']
                val_features = fold_data['val_features']
                train_indices = fold_data['train_indices']
                val_indices = fold_data['val_indices']
                
                print(f"\nFold {i}:")
                print(f"  - 训练特征形状: {train_features.shape}")
                print(f"  - 验证特征形状: {val_features.shape}")
                print(f"  - 训练索引数量: {len(train_indices)}")
                print(f"  - 验证索引数量: {len(val_indices)}")
                print(f"  - 特征数量: {len(fold_data['feature_names'])}")
                
                # 检查数据类型和范围
                print(f"  - 训练特征范围: [{train_features.min():.3f}, {train_features.max():.3f}]")
                print(f"  - 验证特征范围: [{val_features.min():.3f}, {val_features.max():.3f}]")
                
            except Exception as e:
                print(f"✗ Fold {i} 加载失败: {e}")
        
        print(f"\n" + "=" * 60)
        print("验证完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"验证过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    validate_prepared_data()
