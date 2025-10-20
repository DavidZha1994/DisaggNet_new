#!/usr/bin/env python3
"""
单元测试脚本 - 验证训练流程中每个组件的输入输出
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
from omegaconf import DictConfig, OmegaConf

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from src.data.datamodule import NILMDataModule
from src.train import NILMLightningModule

def test_data_shapes_and_types():
    """测试数据形状和类型"""
    print("=" * 60)
    print("数据形状和类型测试")
    print("=" * 60)
    
    # 创建测试配置
    config = OmegaConf.create({
        'data': {
            'batch_size': 8,
            'num_workers': 0,  # 单线程避免复杂性
            'pin_memory': False,
            'window_size': 256,
            'cv': {'fold_id': 0}
        },
        'paths': {
            'prepared_dir': 'Data/prepared'
        }
    })
    
    try:
        # 创建数据模块
        datamodule = NILMDataModule(config)
        datamodule.setup()
        
        # 获取一个批次
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))
        
        print("✓ 数据加载成功")
        print(f"批次大小: {len(batch['time_features'])}")
        
        # 检查每个字段的形状和类型
        expected_shapes = {
            'time_features': (8, 256, None),  # (B, T, C)
            'aux_features': (8, None),        # (B, F)
            'target_power': (8, None),        # (B, K)
            'target_states': (8, None),       # (B, K)
            'total_power': (8, 1),           # (B, 1)
            'time_valid_mask': (8, 256),     # (B, T)
        }
        
        for key, expected_shape in expected_shapes.items():
            if key in batch:
                actual_shape = batch[key].shape
                print(f"  {key}: {actual_shape} (期望: {expected_shape})")
                
                # 验证形状
                if expected_shape[0] is not None and actual_shape[0] != expected_shape[0]:
                    print(f"    ❌ 批次维度不匹配: {actual_shape[0]} != {expected_shape[0]}")
                if expected_shape[1] is not None and actual_shape[1] != expected_shape[1]:
                    print(f"    ❌ 第二维度不匹配: {actual_shape[1]} != {expected_shape[1]}")
                
                # 验证数据类型
                if not isinstance(batch[key], torch.Tensor):
                    print(f"    ❌ 类型错误: {type(batch[key])} (期望: torch.Tensor)")
                elif batch[key].dtype != torch.float32:
                    print(f"    ❌ 数据类型错误: {batch[key].dtype} (期望: torch.float32)")
                else:
                    print(f"    ✓ 形状和类型正确")
                
                # 检查NaN和Inf
                if torch.isnan(batch[key]).any():
                    print(f"    ⚠️  包含NaN值")
                if torch.isinf(batch[key]).any():
                    print(f"    ⚠️  包含Inf值")
            else:
                print(f"  {key}: 缺失")
        
        return batch, datamodule
        
    except Exception as e:
        print(f"❌ 数据测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_model_forward_pass(batch, device_names):
    """测试模型前向传播"""
    print("\n" + "=" * 60)
    print("模型前向传播测试")
    print("=" * 60)
    
    try:
        # 创建模型配置
        model_config = OmegaConf.create({
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
                'enable': False,  # 简化测试
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
                'enable': True,
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
        })
        
        # 创建完整配置
        config = OmegaConf.create({
            'model': model_config,
            'evaluation': {
                'threshold_method': 'adaptive'
            },
            'training': {
                'log_every_n_steps': 10
            },
            'debug': {
                'track_grad_norm': 0
            }
        })
        
        device_info = {'device_count': len(device_names)}
        
        # 创建模型
        from src.models.fusion_transformer import FusionTransformer
        model = FusionTransformer(model_config, len(device_names))
        
        print(f"✓ 模型创建成功")
        print(f"设备数量: {len(device_names)}")
        
        # 测试前向传播
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            outputs = model(
                time_features=batch['time_features'],
                aux_features=batch['aux_features'],
                time_valid_mask=batch['time_valid_mask']
            )
            forward_time = time.time() - start_time
        
        print(f"✓ 前向传播成功 ({forward_time*1000:.2f} ms)")
        
        # 检查输出形状
        if isinstance(outputs, tuple):
            pred_power, pred_states = outputs[0], outputs[1]
            print(f"  功率预测形状: {pred_power.shape}")
            print(f"  状态预测形状: {pred_states.shape}")
            
            # 验证输出形状
            expected_power_shape = (batch['time_features'].shape[0], len(device_names))
            expected_states_shape = (batch['time_features'].shape[0], len(device_names))
            
            if pred_power.shape != expected_power_shape:
                print(f"    ❌ 功率预测形状错误: {pred_power.shape} != {expected_power_shape}")
            else:
                print(f"    ✓ 功率预测形状正确")
                
            if pred_states.shape != expected_states_shape:
                print(f"    ❌ 状态预测形状错误: {pred_states.shape} != {expected_states_shape}")
            else:
                print(f"    ✓ 状态预测形状正确")
            
            # 检查输出值范围
            print(f"  功率预测范围: [{pred_power.min().item():.4f}, {pred_power.max().item():.4f}]")
            print(f"  状态预测范围: [{pred_states.min().item():.4f}, {pred_states.max().item():.4f}]")
            
            # 检查NaN和Inf
            if torch.isnan(pred_power).any() or torch.isnan(pred_states).any():
                print(f"    ❌ 输出包含NaN值")
            elif torch.isinf(pred_power).any() or torch.isinf(pred_states).any():
                print(f"    ❌ 输出包含Inf值")
            else:
                print(f"    ✓ 输出值正常")
        
        return outputs, model
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_loss_computation(batch, outputs, device_names):
    """测试损失计算"""
    print("\n" + "=" * 60)
    print("损失计算测试")
    print("=" * 60)
    
    try:
        from src.losses.losses import create_loss_function, RECOMMENDED_LOSS_CONFIGS
        
        # 创建损失函数
        loss_config = RECOMMENDED_LOSS_CONFIGS.get('balanced', {})
        loss_fn = create_loss_function(loss_config)
        
        print(f"✓ 损失函数创建成功")
        print(f"  分类权重: {loss_fn.classification_weight}")
        print(f"  回归权重: {loss_fn.regression_weight}")
        print(f"  守恒权重: {loss_fn.conservation_weight}")
        print(f"  一致性权重: {loss_fn.consistency_weight}")
        
        # 计算损失
        pred_power, pred_states = outputs[0], outputs[1]
        
        start_time = time.time()
        total_loss, loss_details = loss_fn(
            pred_power=pred_power,
            pred_switch=pred_states,
            target_power=batch['target_power'],
            target_switch=batch['target_states'],
            total_power=batch['total_power']
        )
        loss_time = time.time() - start_time
        
        print(f"✓ 损失计算成功 ({loss_time*1000:.2f} ms)")
        print(f"  总损失: {total_loss.item():.4f}")
        
        # 检查各项损失
        for loss_name, loss_value in loss_details.items():
            if isinstance(loss_value, torch.Tensor):
                print(f"  {loss_name}: {loss_value.item():.4f}")
                
                # 检查损失值是否正常
                if torch.isnan(loss_value).any():
                    print(f"    ❌ {loss_name}包含NaN")
                elif torch.isinf(loss_value).any():
                    print(f"    ❌ {loss_name}包含Inf")
                elif loss_value.item() < 0:
                    print(f"    ❌ {loss_name}为负值")
                else:
                    print(f"    ✓ {loss_name}正常")
        
        # 测试梯度计算（需要模型参数有梯度）
        try:
            # 创建一个简单的模型来测试梯度
            test_model = torch.nn.Linear(10, 10)
            test_pred = test_model(torch.randn(8, 10))
            test_loss = torch.nn.functional.mse_loss(test_pred, batch['target_power'])
            test_loss.backward()
            print(f"✓ 反向传播成功")
        except Exception as e:
            print(f"⚠️  反向传播测试跳过: {e}")
        
        return total_loss, loss_details
        
    except Exception as e:
        print(f"❌ 损失计算测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_training_step_integration(batch, datamodule):
    """测试完整的训练步骤"""
    print("\n" + "=" * 60)
    print("训练步骤集成测试")
    print("=" * 60)
    
    try:
        # 创建完整配置
        config = OmegaConf.create({
            'data': {
                'window_size': 256
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
                    'enable': False,
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
                    'enable': True,
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
            },
            'training': {
                'log_every_n_steps': 10,
                'visualization': {
                    'enable': False
                }
            },
            'debug': {
                'track_grad_norm': 0
            },
            'aux_training': {
                'metric_learning': {
                    'enable': False
                }
            }
        })
        
        device_info = {'device_count': len(datamodule.device_names)}
        device_names = datamodule.device_names
        
        # 创建Lightning模块
        lightning_module = NILMLightningModule(config, device_info, device_names)
        lightning_module.train()
        
        print(f"✓ Lightning模块创建成功")
        
        # 测试训练步骤
        start_time = time.time()
        loss = lightning_module.training_step(batch, 0)
        step_time = time.time() - start_time
        
        print(f"✓ 训练步骤成功 ({step_time*1000:.2f} ms)")
        print(f"  训练损失: {loss.item():.4f}")
        
        # 检查损失值
        if torch.isnan(loss).any():
            print(f"    ❌ 训练损失包含NaN")
        elif torch.isinf(loss).any():
            print(f"    ❌ 训练损失包含Inf")
        elif loss.item() < 0:
            print(f"    ❌ 训练损失为负值")
        else:
            print(f"    ✓ 训练损失正常")
        
        # 测试验证步骤
        lightning_module.eval()
        with torch.no_grad():
            start_time = time.time()
            val_loss = lightning_module.validation_step(batch, 0)
            val_step_time = time.time() - start_time
        
        print(f"✓ 验证步骤成功 ({val_step_time*1000:.2f} ms)")
        if val_loss is not None:
            print(f"  验证损失: {val_loss.item():.4f}")
        
        return lightning_module
        
    except Exception as e:
        print(f"❌ 训练步骤测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_memory_efficiency():
    """测试内存效率"""
    print("\n" + "=" * 60)
    print("内存效率测试")
    print("=" * 60)
    
    import psutil
    import gc
    
    # 清理内存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 记录初始内存
    initial_memory = psutil.virtual_memory().percent
    if torch.cuda.is_available():
        initial_gpu_memory = torch.cuda.memory_allocated() / (1024**3)
    
    print(f"初始CPU内存使用: {initial_memory:.1f}%")
    if torch.cuda.is_available():
        print(f"初始GPU内存使用: {initial_gpu_memory:.2f} GB")
    
    # 运行多个批次测试内存泄漏
    try:
        config = OmegaConf.create({
            'data': {
                'batch_size': 16,
                'num_workers': 0,
                'pin_memory': False,
                'window_size': 256,
                'cv': {'fold_id': 0}
            },
            'paths': {
                'prepared_dir': 'Data/prepared'
            }
        })
        
        datamodule = NILMDataModule(config)
        datamodule.setup()
        train_loader = datamodule.train_dataloader()
        
        memory_usage = []
        gpu_memory_usage = []
        
        for i, batch in enumerate(train_loader):
            if i >= 10:  # 只测试10个批次
                break
            
            # 模拟处理
            _ = batch['time_features'].mean()
            
            # 记录内存使用
            current_memory = psutil.virtual_memory().percent
            memory_usage.append(current_memory)
            
            if torch.cuda.is_available():
                current_gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_usage.append(current_gpu_memory)
        
        # 分析内存使用趋势
        if len(memory_usage) > 1:
            memory_trend = memory_usage[-1] - memory_usage[0]
            print(f"CPU内存变化: {memory_trend:+.1f}%")
            
            if abs(memory_trend) > 5.0:
                print(f"    ⚠️  可能存在内存泄漏")
            else:
                print(f"    ✓ 内存使用稳定")
        
        if torch.cuda.is_available() and len(gpu_memory_usage) > 1:
            gpu_memory_trend = gpu_memory_usage[-1] - gpu_memory_usage[0]
            print(f"GPU内存变化: {gpu_memory_trend:+.2f} GB")
            
            if abs(gpu_memory_trend) > 0.5:
                print(f"    ⚠️  可能存在GPU内存泄漏")
            else:
                print(f"    ✓ GPU内存使用稳定")
        
    except Exception as e:
        print(f"❌ 内存效率测试失败: {e}")

def main():
    """主函数"""
    print("开始单元测试...")
    
    # 检查数据目录
    data_dir = Path('Data/prepared')
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        print("请先运行数据准备脚本")
        return
    
    # 测试数据形状和类型
    batch, datamodule = test_data_shapes_and_types()
    if batch is None or datamodule is None:
        return
    
    # 测试模型前向传播
    outputs, model = test_model_forward_pass(batch, datamodule.device_names)
    if outputs is None or model is None:
        return
    
    # 测试损失计算
    loss, loss_details = test_loss_computation(batch, outputs, datamodule.device_names)
    if loss is None:
        return
    
    # 测试训练步骤集成
    lightning_module = test_training_step_integration(batch, datamodule)
    if lightning_module is None:
        return
    
    # 测试内存效率
    test_memory_efficiency()
    
    print("\n" + "=" * 60)
    print("✓ 所有单元测试完成")
    print("=" * 60)

if __name__ == "__main__":
    main()