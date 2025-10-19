#!/usr/bin/env python3
"""
PyTorch Lightning训练性能诊断脚本
检查数据加载、模型前向传播、损失计算等各个环节的性能
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
import platform
import psutil
from omegaconf import DictConfig, OmegaConf

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from src.data.datamodule import NILMDataModule, PreparedDataset
from src.models.fusion_transformer import FusionTransformer
from src.losses.losses import create_loss_function, RECOMMENDED_LOSS_CONFIGS

def print_system_info():
    """打印系统信息"""
    print("=" * 60)
    print("系统信息")
    print("=" * 60)
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"CPU核心数: {os.cpu_count()}")
    print(f"内存: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print()

def create_test_config():
    """创建测试配置"""
    config = OmegaConf.create({
        'data': {
            'batch_size': 32,
            'num_workers': 4,
            'pin_memory': True,
            'window_size': 256,
            'cv': {'fold_id': 0}
        },
        'paths': {
            'prepared_dir': 'Data/prepared'
        },
        'model': {
            'd_model': 128,
            'n_heads': 8,
            'num_layers': 4,
            'dropout': 0.1,
            'time_encoder': {
                'd_model': 128,
                'n_heads': 8,
                'num_layers': 4,
                'dropout': 0.1,
                'input_conv_embed': False,
                'causal_mask': True
            },
            'freq_encoder': {
                'enable': True,
                'proj_dim': 64,
                'conv1d_kernel': 3,
                'small_transformer_layers': 0,
                'dropout': 0.1
            },
            'fusion': {
                'type': 'cross_attention',
                'gated': True
            },
            'aux_encoder': {
                'enable': False,
                'hidden': 64,
                'dropout': 0.1
            },
            'heads': {
                'regression': {
                    'hidden': 64,
                    'dropout': 0.1
                },
                'classification': {
                    'init_p': None
                }
            },
            'calibration': {
                'enable': False
            }
        },
        'evaluation': {
            'threshold_method': 'adaptive'
        }
    })
    return config

def test_data_loading_performance(config: DictConfig):
    """测试数据加载性能"""
    print("=" * 60)
    print("数据加载性能测试")
    print("=" * 60)
    
    try:
        # 创建数据模块
        datamodule = NILMDataModule(config)
        datamodule.setup()
        
        print(f"数据集大小: 训练={len(datamodule.train_dataset)}, 验证={len(datamodule.val_dataset)}")
        print(f"设备数量: {len(datamodule.device_names)}")
        print(f"设备名称: {datamodule.device_names}")
        
        # 测试单个样本加载
        start_time = time.time()
        sample = datamodule.train_dataset[0]
        single_load_time = time.time() - start_time
        print(f"单个样本加载时间: {single_load_time*1000:.2f} ms")
        
        # 打印样本形状
        print("样本数据形状:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} ({value.dtype})")
            else:
                print(f"  {key}: {type(value)}")
        
        # 测试DataLoader性能
        train_loader = datamodule.train_dataloader()
        print(f"DataLoader配置: batch_size={train_loader.batch_size}, num_workers={train_loader.num_workers}")
        
        # 测试批次加载时间
        print("\n批次加载性能测试:")
        batch_times = []
        for i, batch in enumerate(train_loader):
            start_time = time.time()
            # 模拟简单的数据处理
            _ = batch['time_features'].mean()
            batch_time = time.time() - start_time
            batch_times.append(batch_time)
            
            if i < 5:
                print(f"  批次 {i+1}: {batch_time*1000:.2f} ms")
            
            if i >= 10:  # 只测试前10个批次
                break
        
        avg_batch_time = np.mean(batch_times)
        print(f"平均批次加载时间: {avg_batch_time*1000:.2f} ms")
        print(f"预估每epoch时间: {avg_batch_time * len(train_loader) / 60:.2f} 分钟")
        
    except Exception as e:
        print(f"数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_model_performance(config: DictConfig, device_names: list):
    """测试模型前向传播性能"""
    print("\n" + "=" * 60)
    print("模型前向传播性能测试")
    print("=" * 60)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 创建模型
        n_devices = len(device_names)
        model = FusionTransformer(config.model, n_devices)
        model = model.to(device)
        model.eval()
        
        # 创建测试数据
        batch_size = config.data.batch_size
        window_size = config.data.window_size
        n_time_features = 10  # 假设10个时间特征
        
        test_batch = {
            'time_features': torch.randn(batch_size, window_size, n_time_features).to(device),
            'aux_features': torch.randn(batch_size, 20).to(device),  # 假设20个辅助特征
            'time_valid_mask': torch.ones(batch_size, window_size).to(device)
        }
        
        print(f"测试批次形状:")
        for key, value in test_batch.items():
            print(f"  {key}: {value.shape}")
        
        # 预热
        with torch.no_grad():
            for _ in range(3):
                _ = model(
                    time_features=test_batch['time_features'],
                    aux_features=test_batch['aux_features'],
                    time_valid_mask=test_batch['time_valid_mask']
                )
        
        # 测试前向传播时间
        forward_times = []
        with torch.no_grad():
            for i in range(20):
                start_time = time.time()
                outputs = model(
                    time_features=test_batch['time_features'],
                    aux_features=test_batch['aux_features'],
                    time_valid_mask=test_batch['time_valid_mask']
                )
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                forward_time = time.time() - start_time
                forward_times.append(forward_time)
                
                if i < 5:
                    print(f"  前向传播 {i+1}: {forward_time*1000:.2f} ms")
        
        avg_forward_time = np.mean(forward_times)
        print(f"平均前向传播时间: {avg_forward_time*1000:.2f} ms")
        
        # 打印输出形状
        print(f"模型输出形状:")
        if isinstance(outputs, tuple):
            for i, output in enumerate(outputs):
                if output is not None:
                    print(f"  输出 {i+1}: {output.shape}")
                else:
                    print(f"  输出 {i+1}: None")
        else:
            print(f"  输出: {outputs.shape}")
        
    except Exception as e:
        print(f"模型性能测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_loss_computation_performance(config: DictConfig, device_names: list):
    """测试损失计算性能"""
    print("\n" + "=" * 60)
    print("损失计算性能测试")
    print("=" * 60)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建损失函数
        loss_config = RECOMMENDED_LOSS_CONFIGS.get('balanced', {})
        loss_fn = create_loss_function(loss_config)
        
        # 创建测试数据
        batch_size = config.data.batch_size
        n_devices = len(device_names)
        
        pred_power = torch.randn(batch_size, n_devices).to(device)
        pred_states = torch.randn(batch_size, n_devices).to(device)
        target_power = torch.abs(torch.randn(batch_size, n_devices)).to(device)
        target_states = torch.randint(0, 2, (batch_size, n_devices)).float().to(device)
        total_power = torch.abs(torch.randn(batch_size, 1)).to(device)
        
        print(f"损失函数配置:")
        print(f"  分类权重: {loss_fn.classification_weight}")
        print(f"  回归权重: {loss_fn.regression_weight}")
        print(f"  守恒权重: {loss_fn.conservation_weight}")
        print(f"  一致性权重: {loss_fn.consistency_weight}")
        
        # 测试损失计算时间
        loss_times = []
        for i in range(20):
            start_time = time.time()
            total_loss, loss_details = loss_fn(
                pred_power=pred_power,
                pred_switch=pred_states,
                target_power=target_power,
                target_switch=target_states,
                total_power=total_power
            )
            if device.type == 'cuda':
                torch.cuda.synchronize()
            loss_time = time.time() - start_time
            loss_times.append(loss_time)
            
            if i < 5:
                print(f"  损失计算 {i+1}: {loss_time*1000:.2f} ms, 总损失: {total_loss.item():.4f}")
        
        avg_loss_time = np.mean(loss_times)
        print(f"平均损失计算时间: {avg_loss_time*1000:.2f} ms")
        
        # 打印损失详情
        print(f"损失详情:")
        for key, value in loss_details.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.4f}")
        
    except Exception as e:
        print(f"损失计算测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_memory_usage():
    """测试内存使用情况"""
    print("\n" + "=" * 60)
    print("内存使用情况")
    print("=" * 60)
    
    # CPU内存
    memory = psutil.virtual_memory()
    print(f"CPU内存使用: {memory.percent:.1f}% ({memory.used / (1024**3):.1f} GB / {memory.total / (1024**3):.1f} GB)")
    
    # GPU内存
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"GPU {i} 内存: 已分配 {allocated:.1f} GB, 已保留 {reserved:.1f} GB, 总计 {total:.1f} GB")

def main():
    """主函数"""
    print_system_info()
    
    # 创建测试配置
    config = create_test_config()
    
    # 检查数据目录是否存在
    data_dir = Path(config.paths.prepared_dir)
    if not data_dir.exists():
        print(f"错误: 数据目录不存在: {data_dir}")
        print("请先运行数据准备脚本生成训练数据")
        return
    
    # 测试数据加载性能
    test_data_loading_performance(config)
    
    # 获取设备名称
    device_names = ['device_1', 'device_2', 'device_3']  # 默认设备名称
    try:
        datamodule = NILMDataModule(config)
        datamodule.setup()
        device_names = datamodule.device_names
    except:
        pass
    
    # 测试模型性能
    test_model_performance(config, device_names)
    
    # 测试损失计算性能
    test_loss_computation_performance(config, device_names)
    
    # 测试内存使用
    test_memory_usage()
    
    print("\n" + "=" * 60)
    print("性能诊断完成")
    print("=" * 60)

if __name__ == "__main__":
    main()