"""
models - 预报模型库
====================

包含所有可用的预报模型。本科生只需继承 BaseModel 并实现
fit() / predict() 接口，即可将自己的模型接入 SYSUTS 流水线。

可用模型:
    - WZPNetModel: 混合神经网络（AR + skipGRU + GRU + LSTM + Transformer）
    - ARModel: 自回归统计模型
    - ModelTrainer: WZPNet 的训练器（内部使用）
    - LiouvilleAxialPINN: 角动量修正网络（内部使用）

作者: 吴梓鹏
创建: 2025-05-25
"""

from .wzpnet import WZPNetModel
from .trainer import ModelTrainer
from .ar_model import ARModel
from .pinn_corrector import LiouvilleAxialPINN

__all__ = [
    "WZPNetModel",
    "ModelTrainer",
    "ARModel",
    "LiouvilleAxialPINN",
]
