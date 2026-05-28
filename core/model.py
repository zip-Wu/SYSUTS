"""
core.model - 预报模型抽象基类
==============================

定义了所有预报模型的统一接口。
若仅更换模型结构，唯一需要关心的文件——只需继承 BaseModel 并实现
fit() 和 predict() 方法，即可将自己的创新模型接入
SYSUTS 的 EOP 预报流水线。

接口约定:
    - fit(X, y):    训练模型，X 为输入序列，y 为目标序列
    - predict(X):   单步预报，X 为历史序列，返回预报序列
    - save(path):   保存模型（可选）
    - load(path):   加载模型（可选）

作者: 吴梓鹏
创建: 2025-05-25
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseModel(ABC):
    """
    预报模型抽象基类。
    
    这是 SYSUTS 框架最核心的接口。
    无论是深度学习模型（WZPNet、LSTM、Transformer）
    还是统计模型（AR、ARIMA），都必须实现此接口。
    
    接口约定:
        - 输入: np.ndarray（经分解后的残差序列）
        - 输出: np.ndarray（与 pred_len 等长的预报序列）
        
    示例:
        >>> # 使用深度学习模型
        >>> model = WZPNetModel(seq_len=200, seq_out=20, ...)
        >>> model.fit(train_residual)
        >>> forecast = model.predict(train_residual, pred_len=360)
        
        >>> # 使用统计模型
        >>> model = ARModel(order=5)
        >>> model.fit(train_residual)
        >>> forecast = model.predict(train_residual, pred_len=360)
    """
    
    def __init__(self, name: str = "BaseModel", **kwargs):
        """
        Args:
            name: 模型名称
            **kwargs: 模型特定参数
        """
        self.name = name
        self._is_fitted = False
    
    @abstractmethod
    def fit(self, train_data: np.ndarray) -> None:
        """
        训练模型（或加载预训练权重）。
        
        Args:
            train_data: 训练序列，形状 (n_samples,)
            
        Note:
            - 训练完成后必须设置 self._is_fitted = True
            - 如果模型支持加载预训练权重，可在此实现
        """
        ...
    
    @abstractmethod
    def predict(self, train_data: np.ndarray, pred_len: int) -> np.ndarray:
        """
        生成预报序列。
        
        Args:
            train_data: 用于推理的历史序列，形状 (n_samples,)
            pred_len: 预报长度（步数）
            
        Returns:
            np.ndarray: 预报序列，形状 (pred_len,)
            
        Raises:
            RuntimeError: 如果模型未训练（_is_fitted=False）
        """
        ...
    
    def save(self, path: str | Path) -> None:
        """
        保存模型（可选实现）。
        
        Args:
            path: 保存路径
            
        Raises:
            NotImplementedError: 如果子类未实现此方法
        """
        raise NotImplementedError(f"{self.name} 未实现 save() 方法")
    
    def load(self, path: str | Path) -> None:
        """
        加载模型（可选实现）。
        
        Args:
            path: 模型文件路径
            
        Raises:
            NotImplementedError: 如果子类未实现此方法
        """
        raise NotImplementedError(f"{self.name} 未实现 load() 方法")
    
    def check_fitted(self) -> None:
        """
        检查模型是否已训练。
        
        Raises:
            RuntimeError: 如果模型未训练
        """
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} 必须先调用 fit() 才能进行预测")
