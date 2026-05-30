"""
correction - 角动量修正模块（"补充包"）
=========================================

独立的可插拔模块，演示如何在不修改主框架的前提下，
接入辅助方法（AAM/OAM 角动量补偿）来改善预报精度。

- PolarCorrector: 极移（PMX/PMY）角动量修正器
- UT1Corrector: UT1-UTC 角动量修正器（继承 PolarCorrector）
- LiouvilleAxialPINN: 修正器内部使用的微型 PINN 网络

作者: 吴梓鹏
创建: 2025-05-25
"""

from .polar_corrector import PolarCorrector
from .ut1_corrector import UT1Corrector
from .pinn_corrector import LiouvilleAxialPINN

__all__ = [
    "PolarCorrector",
    "UT1Corrector",
    "LiouvilleAxialPINN",
]
