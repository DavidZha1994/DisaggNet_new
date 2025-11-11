#!/usr/bin/env python3
"""
运行 REFIT/UKDALE 数据准备管线（简化：只需传数据集名称）

示例：
  python scripts/run_refit_ukdale_pipeline.py --dataset UKDALE --output Data/prepared
  python scripts/run_refit_ukdale_pipeline.py --dataset REFIT  --output Data/prepared

可选：仍支持显式传入配置文件路径，但不再必须。
"""

import os
import sys
import argparse
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data_preparation.refit_ukdale_pipeline import REFITUKDALEPipeline


def main():
    parser = argparse.ArgumentParser(description="Run REFIT/UKDALE data preparation pipeline")
    parser.add_argument("--dataset", type=str, choices=["UKDALE", "REFIT"], default="UKDALE", help="Dataset name")
    parser.add_argument("--output", type=str, default=os.path.join("Data", "prepared"), help="Output directory")
    # 兼容旧参数（可选，不必传）
    parser.add_argument("--expes", type=str, default=os.path.join("configs", "expes.yaml"), help="Path to expes.yaml")
    parser.add_argument("--datasets", type=str, default=os.path.join("configs", "datasets.yaml"), help="Path to datasets.yaml")
    parser.add_argument("--prep", type=str, default=os.path.join("configs", "pipeline", "prep_config.yaml"), help="Path to prep_config.yaml")
    args = parser.parse_args()

    pipe = REFITUKDALEPipeline(args.expes, args.datasets, args.prep, dataset_override=args.dataset)
    pipe.output_dir = args.output
    os.makedirs(pipe.output_dir, exist_ok=True)

    start = datetime.now()
    summary = pipe.run_full_pipeline()
    end = datetime.now()

    print("\n=== REFIT/UKDALE 数据准备完成 ===")
    print(f"输出目录: {pipe.output_dir}")
    print(f"耗时: {(end - start).total_seconds():.1f}s")
    print("摘要:")
    for k, v in summary.items():
        print(f" - {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())