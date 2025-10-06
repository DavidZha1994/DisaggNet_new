"""  
数据准备模块
Data Preparation Module

包含时序数据准备的所有功能：
- 数据分段 (DataSegmenter)
- 窗口生成 (WindowGenerator)
- 交叉验证 (WalkForwardCV)
- 特征工程 (FeatureEngineer)
- 标签处理 (LabelHandler)
- 评估系统 (EventEvaluator)
- 主流程 (DataPreparationPipeline)
"""

from .segmentation import DataSegmenter
from .windowing import WindowGenerator
from .cross_validation import WalkForwardCV
from .feature_engineering import FeatureEngineer
from .label_handling import LabelHandler
from .evaluation import EventEvaluator
from .pipeline import DataPreparationPipeline

__all__ = [
    'DataSegmenter',
    'WindowGenerator', 
    'WalkForwardCV',
    'FeatureEngineer',
    'LabelHandler',
    'EventEvaluator',
    'DataPreparationPipeline'
]