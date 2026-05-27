"""
preprocessing.diff - 差分与逆差分变换
=======================================

提供多阶差分和逆差分的纯函数实现。
主要用于 UT1-UTC 预报：去除潮汐后对序列做一阶差分，
使序列平稳化；预报完成后再通过累积求和恢复原始尺度。

作者: 吴梓鹏
创建: 2025-05-25
"""

import numpy as np


def diff_series(data: np.ndarray, order: int = 1) -> tuple[np.ndarray, list]:
    """
    对序列进行 n 阶差分。

    每次差分会使序列长度减 1，n 阶差分后长度为 len(data) - n。
    返回差分后的序列和每阶的最后一个值（用于逆差分）。

    Args:
        data: 输入序列，形状 (n_samples,)
        order: 差分阶数，默认为 1

    Returns:
        (diff_data, last_values):
            diff_data: 差分后的序列，形状 (len(data) - order,)
            last_values: 每阶差分的最后一个值，用于逆差分恢复

    示例:
        >>> x = np.array([1.0, 3.0, 6.0, 10.0])
        >>> diff_x, last_vals = diff_series(x, order=1)
        >>> diff_x
        array([2., 3., 4.])
        >>> inverse_diff(diff_x, last_vals)
        array([3., 6., 10.])
    """
    if order <= 0:
        return data.copy(), []

    last_values = []
    current = data.copy()
    for _ in range(order):
        last_values.append(current[-1])
        current = np.diff(current)

    return current, last_values


def inverse_diff(
    diff_forecast: np.ndarray,
    last_values: list
) -> np.ndarray:
    """
    对差分预报进行逆差分恢复。

    从差分域的预报值恢复到原始尺度。
    恢复过程与 diff_series 的差分过程严格互逆。

    Args:
        diff_forecast: 差分域的预报值，形状 (n,)
        last_values: diff_series 返回的每阶最后一个值列表

    Returns:
        np.ndarray: 恢复后的原始尺度序列

    示例:
        >>> x = np.array([1.0, 3.0, 6.0, 10.0])
        >>> diff_x, last_vals = diff_series(x)
        >>> forecast = np.array([5.0, 6.0, 7.0])  # 差分域预报
        >>> inverse_diff(forecast, last_vals)
        array([15., 21., 28.])  # 逐天累积：10+5=15, 15+6=21, 21+7=28
    """
    result = diff_forecast.copy()
    for last_val in reversed(last_values):
        result = np.cumsum(np.concatenate([[last_val], result]))
        result = result[1:]  # 去掉插入的初始值
    return result
