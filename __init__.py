"""
SYSUTS - 地球自转参数（EOP）预报工具包
========================================

支持极移（PMX/PMY）和 UT1-UTC 的端到端预报。

封装了所有 EOP 物理预处理（LS 分解、固体潮汐改正、闰秒处理、
差分变换等），留出深度学习和统计模型接口供后续研究者接入。

架构:
    core/             核心接口 —— BaseModel（本科生唯一需要关心的抽象类）
                      和 ForecastEvaluator（预报评估器）
    preprocessing/    EOP 专用预处理 —— 数据加载、LS 分解、潮汐改正、
                      闰秒处理、差分变换、AAM/OAM 数据加载
    models/           预报模型 —— WZPNet（Linear-skipGRU）、AR 统计模型、
                      角动量修正 PINN 网络、模型模板
    correction/       角动量修正 —— 可选模块，对基础预报进行 AAM/OAM 补偿
    experiments/      实验入口 —— 验证脚本和演示脚本

使用方式:
    # 运行完整验证
    python experiments/verify.py --all
    python experiments/verify_correction.py --all

    # 接入自定义模型
    from SYSUTS.core.model import BaseModel

作者: 吴梓鹏
版本: 0.3.0
"""

__version__ = "0.3.0"
__author__ = "吴梓鹏"

from .core.model import BaseModel
from .core.evaluator import ForecastEvaluator

__all__ = [
    "BaseModel",
    "ForecastEvaluator",
]
