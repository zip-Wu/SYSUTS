"""
correction - 角动量修正模块（可选）
====================================

实现"基础预报 + 角动量补偿"架构，利用 AAM（大气角动量）
和 OAM（海洋角动量）数据对主模型预报进行次级校正。

- PolarCorrector: 极移（PMX/PMY）角动量修正器
- UT1Corrector: UT1-UTC 角动量修正器（继承 PolarCorrector）

作者: 吴梓鹏
创建: 2025-05-25
"""

from .polar_corrector import PolarCorrector
from .ut1_corrector import UT1Corrector

__all__ = [
    "PolarCorrector",
    "UT1Corrector",
]
