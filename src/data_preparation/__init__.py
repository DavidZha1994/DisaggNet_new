"""  
数据准备模块
Data Preparation Module

包含时序数据准备的所有功能：
- 数据分段 (DataSegmenter)
- 窗口生成 (WindowGenerator)
- 交叉验证 (WalkForwardCV)
- 特征工程 (FeatureEngineer)
- HIPE数据准备流程 (HIPEDataPreparationPipeline)
"""


from .cross_validation import WalkForwardCV

# 延迟/安全导入 HIPE 管线，避免可选依赖缺失导致模块导入失败
try:
    from .hipe_pipeline import HIPEDataPreparationPipeline  # type: ignore
except Exception:
    HIPEDataPreparationPipeline = None  # 可选

__all__ = [
    'WalkForwardCV',
    'HIPEDataPreparationPipeline'
]