#!/usr/bin/env python3
"""
数据准备流程运行脚本
Data Preparation Pipeline Runner

使用示例:
python run_data_preparation.py --config config/prep_config.yaml --data Data/processed_data.csv --output Data/prepared_data
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_preparation import DataPreparationPipeline

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='运行时序数据准备与交叉验证流程',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法
  python run_data_preparation.py --config config/prep_config.yaml --data Data/processed_data.csv
  
  # 指定输出目录
  python run_data_preparation.py --config config/prep_config.yaml --data Data/processed_data.csv --output Data/prepared_data
  
  # 只显示摘要
  python run_data_preparation.py --config config/prep_config.yaml --summary-only --output Data/prepared_data
        """
    )
    
    parser.add_argument(
        '--config', 
        required=True, 
        help='配置文件路径 (YAML格式)'
    )
    
    parser.add_argument(
        '--data', 
        help='输入数据路径 (CSV或Parquet格式)'
    )
    
    parser.add_argument(
        '--output', 
        help='输出目录 (可选，会覆盖配置文件中的设置)'
    )
    
    parser.add_argument(
        '--summary-only', 
        action='store_true',
        help='只显示已有流程的摘要，不运行新的流程'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # 检查配置文件
        if not os.path.exists(args.config):
            logger.error(f"配置文件不存在: {args.config}")
            return 1
        
        # 创建流程实例
        logger.info(f"加载配置文件: {args.config}")
        pipeline = DataPreparationPipeline(args.config)
        
        # 如果指定了输出目录，更新配置
        if args.output:
            pipeline.output_dir = args.output
            pipeline.config['data_storage']['output_directory'] = args.output
            os.makedirs(args.output, exist_ok=True)
            logger.info(f"输出目录设置为: {args.output}")
        
        # 如果只要摘要
        if args.summary_only:
            logger.info("显示流程摘要")
            summary = pipeline.get_pipeline_summary()
            print_summary(summary)
            return 0
        
        # 检查输入数据
        if not args.data:
            logger.error("必须指定输入数据路径 (--data)")
            return 1
        
        if not os.path.exists(args.data):
            logger.error(f"输入数据文件不存在: {args.data}")
            return 1
        
        # 运行流程
        logger.info(f"开始运行数据准备流程")
        logger.info(f"输入数据: {args.data}")
        logger.info(f"输出目录: {pipeline.output_dir}")
        
        start_time = datetime.now()
        results = pipeline.run_full_pipeline(args.data)
        end_time = datetime.now()
        
        duration = end_time - start_time
        logger.info(f"数据准备流程运行完成！耗时: {duration}")
        
        # 打印结果摘要
        print("\n" + "="*60)
        print("数据准备流程运行成功！")
        print("="*60)
        print(f"输出目录: {pipeline.output_dir}")
        print(f"运行时间: {duration}")
        
        # 打印详细摘要
        summary = pipeline.get_pipeline_summary()
        print_summary(summary)
        
        # 提示下一步
        print("\n" + "="*60)
        print("下一步操作:")
        print("="*60)
        print("1. 查看生成的数据和报告:")
        print(f"   ls -la {pipeline.output_dir}")
        print("\n2. 加载处理后的数据进行模型训练:")
        print("   from src.data_preparation import DataPreparationPipeline")
        print(f"   pipeline = DataPreparationPipeline('{args.config}')")
        print("   fold_data = pipeline.load_processed_data(fold_idx=0)")
        print("\n3. 查看配置文件了解所有参数:")
        print(f"   cat {args.config}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("用户中断操作")
        return 1
    except Exception as e:
        logger.error(f"数据准备流程运行失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

def print_summary(summary):
    """打印流程摘要"""
    print("\n" + "-"*40)
    print("流程摘要")
    print("-"*40)
    
    print(f"状态: {summary.get('status', 'unknown')}")
    
    if 'start_time' in summary:
        print(f"开始时间: {summary['start_time']}")
    if 'end_time' in summary:
        print(f"结束时间: {summary['end_time']}")
    
    # 步骤信息
    if 'steps' in summary:
        print("\n步骤执行情况:")
        for step_name, step_info in summary['steps'].items():
            status = step_info.get('status', 'unknown')
            print(f"  {step_name}: {status}")
            
            # 显示一些关键信息
            if step_name == 'data_loading' and 'data_shape' in step_info:
                print(f"    数据形状: {step_info['data_shape']}")
            elif step_name == 'segmentation' and 'num_segments' in step_info:
                print(f"    段数: {step_info['num_segments']}")
                print(f"    总样本数: {step_info['total_samples']}")
            elif step_name == 'windowing' and 'num_windows' in step_info:
                print(f"    窗口数: {step_info['num_windows']}")
                print(f"    窗口长度: {step_info['window_length']}")
            elif step_name == 'cross_validation' and 'num_folds' in step_info:
                print(f"    折数: {step_info['num_folds']}")
    
    # 段摘要
    if 'segments_summary' in summary:
        seg_summary = summary['segments_summary']
        print(f"\n数据摘要:")
        print(f"  总段数: {seg_summary['total_segments']}")
        print(f"  总样本数: {seg_summary['total_samples']}")
        print(f"  设备数: {len(seg_summary['devices'])}")
        print(f"  设备列表: {', '.join(seg_summary['devices'])}")
        
        if 'time_range' in seg_summary:
            time_range = seg_summary['time_range']
            start_time = datetime.fromtimestamp(time_range['start'])
            end_time = datetime.fromtimestamp(time_range['end'])
            print(f"  时间范围: {start_time} 到 {end_time}")
    
    # 错误信息
    if 'error' in summary:
        print(f"\n错误信息: {summary['error']}")

if __name__ == "__main__":
    exit(main())