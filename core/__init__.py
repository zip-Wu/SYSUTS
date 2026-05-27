"""
core - 核心接口
=================

提供 EOP 预报工具包的核心抽象接口。

- BaseModel: 本科生唯一需要实现的模型接口
- ForecastEvaluator: 预报评估器（MAE/RMSE）

作者: 吴梓鹏
创建: 2025-05-25
"""

from .model import BaseModel
from .evaluator import ForecastEvaluator

__all__ = [
    "BaseModel",
    "ForecastEvaluator",
]
