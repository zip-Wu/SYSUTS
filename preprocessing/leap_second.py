"""
preprocessing.leap_second - 闰秒处理
=====================================

UT1-UTC 到 UT1-TAI 的转换需要扣除历史上所有注入的闰秒。
闰秒注入后，UT1-UTC 会在特定修正儒略日（MJD）发生 -1s 跳变。
去除闰秒后得到连续时间序列 UT1-TAI，适合时间序列建模。

参考: IERS Bulletin C 闰秒公告

作者: 吴梓鹏
创建: 2025-05-25
"""

import numpy as np
import pandas as pd


# ============================================================================
# 闰秒注入 MJD 列表（共 27 个闰秒，截至 2023 年）
# ============================================================================

LEAP_SECONDS_MJD = [
    41499, 41683, 42048, 42413, 42778, 43144, 43509,
    43874, 44239, 44786, 45151, 45516, 46247, 47161,
    47892, 48257, 48804, 49169, 49534, 50083, 50630,
    51179, 53736, 54832, 56109, 57204, 57754
]


def remove_leap_seconds(df: pd.DataFrame) -> pd.DataFrame:
    """
    去除闰秒：UT1-UTC → UT1-TAI（连续时间序列）。

    算法: 对 DataFrame 中的 UT1_UTC 列，在每个闰秒注入日（含当日）
    之后的所有值减去累计闰秒数，得到连续的 UT1-TAI。

    Args:
        df: DataFrame，必须包含 'MJD' 列（修正儒略日）和
            'UT1_UTC' 列（UT1-UTC，秒）。

    Returns:
        新增 'UT1-TAI' 列的 DataFrame（不修改原 DataFrame）。

    示例:
        >>> df = pd.DataFrame({'MJD': [50082, 50083, 50084],
        ...                    'UT1_UTC': [-0.1, -0.2, -1.3]})
        >>> result = remove_leap_seconds(df)
        >>> result['UT1-TAI'].values  # 50083 是闰秒注入日（含当日起减 1s）
    """
    df = df.copy()
    df['MJD'] = df['MJD'].astype(float)
    df['UT1-TAI'] = df['UT1_UTC'].astype(float)

    for leap_mjd in LEAP_SECONDS_MJD:
        mask = df['MJD'] >= leap_mjd
        df.loc[mask, 'UT1-TAI'] -= 1.0

    return df
