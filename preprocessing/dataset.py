"""
preprocessing.dataset - EOP 数据集加载器
=========================================

解析 IERS 标准 EOP C04 格式数据文件，提取极移（PMX/PMY）
和 UT1-UTC 时间序列，支持按日期范围切片和训练/测试集分割。

数据格式约定（与原始 Jupyter Notebook 完全一致）：
    - 跳过前 6 行文件头
    - 第 7 行起为数据行
    - 列映射: x(") → PMX, y(") → PMY, UT1-UTC(s) → UT1_UTC

作者: 吴梓鹏
创建: 2025-05-25
"""

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


class EOPDataset:
    """
    地球自转参数（EOP）数据集。

    支持解析 IERS 提供的 EOP C04/14 等格式数据。

    Attributes:
        AVAILABLE_TARGETS: 可用的目标变量
            - "PMX": 极移 X 分量（角秒）
            - "PMY": 极移 Y 分量（角秒）
            - "UT1_UTC": UT1-UTC 差（秒）

    示例:
        >>> dataset = EOPDataset("./data/data_origin.txt")
        >>> df = dataset.load()
        >>> pmx_series = dataset.get_series(df, target="PMX")
        >>> train, test = dataset.get_train_test_split(df, "PMX", "2020-01-02", 20)
    """

    AVAILABLE_TARGETS = ["PMX", "PMY", "UT1_UTC"]

    def __init__(self, data_path: str | Path):
        """
        Args:
            data_path: EOP 数据文件的路径
        """
        self.data_path = Path(data_path)
        self._data: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        """
        加载并解析 IERS 格式的 EOP 数据。

        Returns:
            pd.DataFrame: 包含 date 索引和 PMX / PMY / UT1_UTC / MJD 列的数据框
        """
        labels = self._parse_header()
        data = self._parse_data(labels)

        df = pd.DataFrame(data, columns=labels)
        df['YR'] = df['YR'].astype(int)

        date = pd.to_datetime(
            df['YR'].astype(str) + '-' + df['MM'] + '-' + df['DD']
        )
        df.insert(0, 'date', date)

        normal_data = df[['date', 'MJD', 'x(")', 'y(")', 'UT1-UTC(s)']].copy()
        normal_data.columns = ['date', 'MJD', 'PMX', 'PMY', 'UT1_UTC']
        normal_data.set_index('date', inplace=True)

        for col in ['MJD', 'PMX', 'PMY', 'UT1_UTC']:
            normal_data[col] = pd.to_numeric(normal_data[col], errors='coerce')

        self._data = normal_data
        return normal_data

    def _parse_header(self) -> list[str]:
        """解析 IERS 文件头（第 6 行），提取列标签。"""
        line_num = 0
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_num += 1
                if line_num == 6:
                    parts = line.strip().split('  ')
                    labels = [p.strip() for p in parts if p.strip()]
                    labels[0] = 'YR'
                    break
        return labels

    def _parse_data(self, labels: list[str]) -> list[list]:
        """从第 7 行起解析数据行。"""
        data = []
        line_num = 0
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_num += 1
                if line_num < 7:
                    continue
                parts = line.strip().split()
                if len(parts) >= len(labels):
                    data.append(parts[:len(labels)])
        return data

    def get_series(
        self,
        df: pd.DataFrame,
        target: str,
        start: datetime | str | None = None,
        end: datetime | str | None = None
    ) -> np.ndarray:
        """
        按日期范围提取目标序列。

        Args:
            df: load() 返回的 DataFrame
            target: 目标变量（"PMX" / "PMY" / "UT1_UTC"）
            start: 起始日期（可选，含当日）
            end: 结束日期（可选，含当日）

        Returns:
            np.ndarray: 一维目标序列
        """
        if target not in self.AVAILABLE_TARGETS:
            raise ValueError(
                f"未知目标变量: {target}. 可用: {self.AVAILABLE_TARGETS}"
            )

        mask = pd.Series(True, index=df.index)
        if start is not None:
            mask &= df.index >= pd.to_datetime(start)
        if end is not None:
            mask &= df.index <= pd.to_datetime(end)

        return df.loc[mask, target].values

    def get_available_date_range(self, df: pd.DataFrame) -> tuple[datetime, datetime]:
        """返回数据集的日期范围。"""
        return (df.index.min(), df.index.max())

    def get_train_test_split(
        self,
        df: pd.DataFrame,
        target: str,
        forecast_date: str,
        train_len_years: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        按预报日期分割训练集和测试集。

        与原始 Jupyter Notebook (tset_PMX.ipynb) 的分割逻辑完全一致：
            - 训练集: [forecast_date - train_len_years, forecast_date]（闭区间，含两端）
            - 真值:   [forecast_date + 1 天, forecast_date + 400 天]（调用方截 pred_len）

        Args:
            df: load() 返回的 DataFrame
            target: 目标变量（"PMX" / "PMY" / "UT1_UTC"）
            forecast_date: 预报日期（字符串，YYYY-MM-DD 格式）
            train_len_years: 训练数据长度（年）

        Returns:
            (train_series, test_series): 两个 numpy 数组
        """
        forecast_date = pd.Timestamp(forecast_date)
        train_end = forecast_date
        train_start = train_end - pd.DateOffset(years=train_len_years)

        # 训练集：闭区间 [train_start, train_end]
        train_mask = (df.index >= train_start) & (df.index <= train_end)

        # 真值：forecast_date 之后第 1 天起，取 400 天（调用方截取 pred_len）
        test_start = train_end + pd.Timedelta(days=1)
        test_end = train_end + pd.Timedelta(days=400)
        test_mask = (df.index >= test_start) & (df.index <= test_end)

        train_series = df.loc[train_mask, target].values.astype(float)
        test_series = df.loc[test_mask, target].values.astype(float)

        return train_series, test_series

    def get_df_slice(
        self,
        df: pd.DataFrame,
        start: datetime | str | None = None,
        end: datetime | str | None = None
    ) -> pd.DataFrame:
        """
        获取 DataFrame 的日期切片（含所有列，包括 MJD）。

        UT1-UTC 的预处理（潮汐改正）需要 MJD 列，
        因此保留完整 DataFrame 而非仅提取数值序列。

        Args:
            df: load() 返回的 DataFrame
            start: 起始日期（可选，含当日）
            end: 结束日期（可选，含当日）

        Returns:
            pd.DataFrame: 指定日期范围内的完整 DataFrame（含 MJD 列）
        """
        mask = pd.Series(True, index=df.index)
        if start is not None:
            mask &= df.index >= pd.to_datetime(start)
        if end is not None:
            mask &= df.index <= pd.to_datetime(end)
        return df.loc[mask].copy()
