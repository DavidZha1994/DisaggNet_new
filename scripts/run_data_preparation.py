#!/usr/bin/env python3
"""
统一数据准备入口脚本

支持数据集：HIPE / UKDALE / REFIT

简化用法（无需多配置文件）：
  # 运行 UKDALE/REFIT（无需 --config/--data）
  python run_data_preparation.py --dataset UKDALE --output Data/prepared
  python run_data_preparation.py --dataset REFIT  --output Data/prepared

  # 运行 HIPE（需要配置与输入数据）
  python run_data_preparation.py --dataset HIPE --config configs/pipeline/prep_config.yaml --data Data/processed.csv --output Data/prepared/hipe
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# 添加项目根目录到 Python 路径（脚本位于子目录时仍能导入 src 包）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(PROJECT_ROOT) == "scripts":
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 注意：为避免不必要的依赖加载，在需要时再进行模块导入


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
  # UKDALE/REFIT
  python run_data_preparation.py --dataset UKDALE --output Data/prepared
  python run_data_preparation.py --dataset REFIT  --output Data/prepared

  # HIPE
  python run_data_preparation.py --dataset HIPE --config configs/pipeline/prep_config.yaml --data Data/processed.csv --output Data/prepared/hipe
        """
    )
    parser.add_argument('--dataset', type=str, choices=['HIPE', 'UKDALE', 'REFIT'], default='UKDALE', help='数据集名称')
    parser.add_argument('--config', help='配置文件路径 (仅 HIPE 需要)')
    parser.add_argument('--data', help='输入数据路径 (仅 HIPE 需要，CSV/Parquet)')
    parser.add_argument('--output', help='输出目录 (可选，会覆盖配置中的设置)')
    
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
        if args.dataset.upper() == 'HIPE':
            # HIPE 模式：需要配置与输入数据
            from src.data_preparation.hipe_pipeline import HIPEDataPreparationPipeline
            if not args.config:
                logger.error("HIPE 模式必须指定配置文件路径 (--config)")
                return 1
            if not os.path.exists(args.config):
                logger.error(f"配置文件不存在: {args.config}")
                return 1
            pipeline = HIPEDataPreparationPipeline(args.config)
            if args.output:
                pipeline.output_dir = args.output
                pipeline.config.setdefault('data_storage', {})['output_directory'] = args.output
                os.makedirs(args.output, exist_ok=True)
                logger.info(f"输出目录设置为: {args.output}")
            if args.summary_only:
                logger.info("显示流程摘要")
                summary = pipeline.get_pipeline_summary()
                print_summary(summary)
                return 0
            if not args.data:
                logger.error("HIPE 模式必须指定输入数据路径 (--data)")
                return 1
            if not os.path.exists(args.data):
                logger.error(f"输入数据文件不存在: {args.data}")
                return 1
            logger.info("开始运行 HIPE 数据准备流程")
            start_time = datetime.now()
            pipeline.run_full_pipeline(args.data)
            end_time = datetime.now()
            duration = end_time - start_time
            logger.info(f"HIPE 数据准备完成！耗时: {duration}")
            print("\n" + "=" * 60)
            print("HIPE 数据准备运行成功！")
            print("=" * 60)
            print(f"输出目录: {pipeline.output_dir}")
            print(f"运行时间: {duration}")
            summary = pipeline.get_pipeline_summary()
            print_summary(summary)
            return 0
        else:
            # UKDALE/REFIT 模式：无需配置与输入数据，仅依据数据集名称与默认配置
            from src.data_preparation.refit_ukdale_pipeline import REFITUKDALEPipeline
            expes = os.path.join("configs", "expes.yaml")
            datasets = os.path.join("configs", "datasets.yaml")
            prep = os.path.join("configs", "pipeline", "prep_config.yaml")
            pipeline = REFITUKDALEPipeline(expes, datasets, prep, dataset_override=args.dataset.upper())
            if args.output:
                out_root = args.output.rstrip(os.sep)
                base_name = os.path.basename(out_root)
                if base_name.lower() in ("ukdale", "refit"):
                    out_root = os.path.dirname(out_root) or "."
                pipeline.output_dir = out_root
                os.makedirs(out_root, exist_ok=True)
                logger.info(f"输出目录设置为: {out_root}")
            logger.info(f"开始运行 {args.dataset.upper()} 数据准备流程")
            start_time = datetime.now()
            summary = pipeline.run_full_pipeline()
            end_time = datetime.now()
            duration = end_time - start_time
            logger.info(f"{args.dataset.upper()} 数据准备完成！耗时: {duration}")
            print("\n" + "=" * 60)
            print(f"{args.dataset.upper()} 数据准备运行成功！")
            print("=" * 60)
            print(f"输出目录: {pipeline.output_dir}")
            print(f"运行时间: {duration}")
            print_summary(summary)
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
    print("\n" + "-" * 40)
    print("流程摘要")
    print("-" * 40)

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
        print("\n数据摘要:")
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
