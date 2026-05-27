"""
ARModel - 自回归模型（AR）

使用 statsmodels 的 AutoReg 实现，符合 SYSUTS 的 BaseModel 接口。
用于 LS+AR 预报流程。
"""

import numpy as np

from ..core.model import BaseModel


class ARModel(BaseModel):
    """
    自回归模型（AR），基于 statsmodels.AutoReg。

    符合 BaseModel 接口，可直接替换 WZPNetModel 插入任何 Pipeline。

    示例:
        >>> model = ARModel(lag_order=365)
        >>> model.fit(residual)
        >>> forecast = model.predict(residual, pred_len=360)
    """

    def __init__(self, lag_order: int = 365, **kwargs):
        """
        Args:
            lag_order: AR 模型滞后阶数（使用多少个历史观测值）
        """
        super().__init__(name="AR")
        self.lag_order = lag_order
        self._ar_result = None   # statsmodels fit 结果
        self._train_len = 0      # 训练集长度，predict 时需要用来确定 start/end

    def fit(self, train_data: np.ndarray) -> None:
        """
        拟合 AR 模型。

        Args:
            train_data: 训练序列（一维 numpy 数组，LS 残差）
        """
        from statsmodels.tsa.ar_model import AutoReg

        model = AutoReg(train_data, lags=self.lag_order)
        self._ar_result = model.fit()
        self._train_len = len(train_data)
        self._is_fitted = True

    def predict(self, train_data: np.ndarray, pred_len: int) -> np.ndarray:
        """
        生成 AR 递推预报。

        Args:
            train_data: 同 fit 时的训练序列（仅用于确定长度，结果不受影响）
            pred_len: 预报步数

        Returns:
            np.ndarray: 长度为 pred_len 的残差预报序列
        """
        self.check_fitted()

        start = self._train_len
        end = self._train_len + pred_len - 1   # statsmodels predict 区间是闭区间

        forecast = self._ar_result.predict(start=start, end=end, dynamic=True)
        return np.asarray(forecast, dtype=np.float64)
