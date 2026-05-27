"""
preprocessing - EOP 专用预处理模块
====================================

封装所有地球自转参数（EOP）预报所需的物理预处理，包括：
- 数据加载与解析（IERS EOP C04 格式）
- 最小二乘（LS）趋势分解（极移／UT1-UTC）
- 差分与逆差分变换
- 闰秒处理（UT1-UTC → UT1-TAI）
- 固体潮汐改正（RG_ZONT2，62 项待谐潮汐）
- AAM/OAM 角动量数据加载

作者: 吴梓鹏
创建: 2025-05-25
"""

from .dataset import EOPDataset
from .ls import ls_decompose, ls_decompose_ut1
from .diff import diff_series, inverse_diff
from .leap_second import LEAP_SECONDS_MJD, remove_leap_seconds
from .tide import remove_tide_ut1, RG_ZONT2
from .aam_oam import load_AAM_data, load_OAM_data

__all__ = [
    "EOPDataset",
    "ls_decompose",
    "ls_decompose_ut1",
    "diff_series",
    "inverse_diff",
    "LEAP_SECONDS_MJD",
    "remove_leap_seconds",
    "remove_tide_ut1",
    "RG_ZONT2",
    "load_AAM_data",
    "load_OAM_data",
]
