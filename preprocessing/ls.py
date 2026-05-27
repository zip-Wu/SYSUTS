"""
preprocessing.ls - 最小二乘（LS）趋势分解
==========================================

对时间序列进行最小二乘趋势分解，提取确定性的长期趋势和周期项。
极移和 UT1-UTC 使用不同的周期项集合，通过 periods 参数区分。

极移周期项（PMX/PMY）:
    - Chandler 摆动: 435 天
    - 周年项: 365.24 天
    - 半年项: 182.62 天
    - 其他周期: 450 天

UT1-UTC 周期项（差分域）:
    - 6793.464 天（~19 年）
    - 3396.732 天（~9.3 年）
    - 365.24 天（年）
    - 182.62 天（半年）
    - 121.747 天（~4 月）

每项包含 sin 和 cos 两个基函数，另加常数项和线性趋势项。

算法: 正规方程求解（B @ B^T @ params = B @ data）

作者: 吴梓鹏
创建: 2025-05-25
"""

from dataclasses import dataclass

import numpy as np


# ============================================================================
# 周期项定义
# ============================================================================

# 极移周期项 (PMX/PMY)
POLAR_PERIODS = [435.0, 365.24, 182.62, 450.0]

# UT1-UTC 周期项（差分域）
UT1_PERIODS = [6793.464, 3396.732, 365.24, 182.62, 121.747]


@dataclass
class LSResult:
    """
    LS 分解结果。

    Attributes:
        fit: 拟合值（与训练数据等长）
        forecast: 趋势预报值
        residual: 残差（训练数据 - 拟合值）
        params: LS 拟合系数
    """
    fit: np.ndarray
    forecast: np.ndarray
    residual: np.ndarray
    params: np.ndarray


# ============================================================================
# 核心求解函数
# ============================================================================

def _build_design_matrix(
    x: np.ndarray,
    periods: list[float],
    use_linear: bool = True
) -> np.ndarray:
    """
    构建 LS 设计矩阵。

    基函数顺序: 常数项, 线性趋势, cos(2πx/p1), sin(2πx/p1), ... for each p in periods

    Args:
        x: 时间坐标（从 0 开始的整数索引）
        periods: 周期列表（天）
        use_linear: 是否包含线性趋势项

    Returns:
        np.ndarray: 设计矩阵，形状 (n_basis, n_points)，每行为一个基函数
    """
    components = []
    # 常数项
    components.append(np.ones_like(x))
    # 线性趋势
    if use_linear:
        components.append(x)
    # 周期项（sin + cos）
    for period in periods:
        components.append(np.cos(2 * np.pi * x / period))
        components.append(np.sin(2 * np.pi * x / period))
    return np.array(components)


def _solve_ls(
    data: np.ndarray,
    design: np.ndarray,
    weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    最小二乘求解。

    Args:
        data: 观测数据，形状 (n,)
        design: 设计矩阵 B，形状 (n_basis, n)
        weights: 权重向量（可选）

    Returns:
        (fit, params):
            fit: 拟合值，形状 (n,)
            params: 拟合系数，形状 (n_basis,)
    """
    if weights is not None:
        W = np.diag(weights)
        G_inv = np.linalg.inv(design @ W @ design.T)
        params = G_inv @ design @ W @ data
    else:
        G_inv = np.linalg.inv(design @ design.T)
        params = G_inv @ design @ data

    fit = design.T @ params
    return fit, params


# ============================================================================
# 公开接口
# ============================================================================

def ls_decompose(
    data: np.ndarray,
    pred_len: int,
    periods: list[float] | None = None,
    use_linear: bool = True,
    weights: np.ndarray | None = None
) -> LSResult:
    """
    对序列进行 LS 趋势分解并外推预报。

    通用接口，适用于极移（PMX/PMY）和其他 EOP 参数的 LS 分解。
    通过 periods 参数控制使用的周期项。

    Args:
        data: 训练序列，形状 (n_train,)
        pred_len: 预报长度（步数）
        periods: 周期项列表（天），默认为极移周期项
        use_linear: 是否包含线性趋势项
        weights: LS 拟合权重（可选）

    Returns:
        LSResult: 包含 fit、forecast、residual、params

    示例:
        >>> # 极移 LS 分解
        >>> result = ls_decompose(pmx_data, pred_len=360, periods=POLAR_PERIODS)
        >>> trend = result.forecast
        >>> residual = result.residual
    """
    if periods is None:
        periods = POLAR_PERIODS

    n_train = len(data)

    # 训练期设计矩阵
    x_train = np.arange(n_train)
    B_train = _build_design_matrix(x_train, periods, use_linear)

    # LS 拟合
    fit, params = _solve_ls(data, B_train, weights)

    # 预报期趋势外推
    x_forecast = np.arange(n_train, n_train + pred_len)
    B_forecast = _build_design_matrix(x_forecast, periods, use_linear)
    forecast = B_forecast.T @ params

    # 训练期残差
    residual = data - fit

    return LSResult(fit=fit, forecast=forecast, residual=residual, params=params)


def ls_decompose_ut1(
    data: np.ndarray,
    pred_len: int,
    use_linear: bool = True,
    weights: np.ndarray | None = None
) -> LSResult:
    """
    对差分后的 UT1-UTC 序列进行 LS 趋势分解并外推预报。

    工作于差分域（UT1Rdiff），使用 UT1 专用的 11 个周期项。

    与原始 Jupyter Notebook (Linear.ipynb) 的 LS_func_UT1 完全对齐。

    Args:
        data: 差分后的 UT1 训练序列（UT1Rdiff），形状 (n_train,)
        pred_len: 预报长度（步数）
        use_linear: 是否包含线性趋势项
        weights: LS 拟合权重（可选）

    Returns:
        LSResult: 包含 fit、forecast、residual、params

    示例:
        >>> result = ls_decompose_ut1(ut1rdiff_data, pred_len=360)
        >>> trend = result.forecast
        >>> residual = result.residual
    """
    return ls_decompose(
        data=data,
        pred_len=pred_len,
        periods=UT1_PERIODS,
        use_linear=use_linear,
        weights=weights,
    )
