"""
preprocessing.aam_oam - AAM/OAM 角动量数据加载
================================================

提供大气角动量（AAM）和海洋角动量（OAM）的 .asc 文件读取功能。
用于角动量修正模块的数据输入。

文件格式约定:
    - AAM: 跳过前 40 行文件头
    - OAM: 跳过前 42 行文件头
    - 列格式: [date_str, hour, mjd, X_mass, Y_mass, Z_mass, X_motion, Y_motion, Z_motion]

数据来源: IERS Special Bureau for the Atmosphere / Oceans

作者: 吴梓鹏
创建: 2025-05-25
"""

import os

import numpy as np
import pandas as pd


def load_AAM_data(folder: str) -> pd.DataFrame:
    """
    加载大气角动量（AAM）数据。

    读取指定目录下所有 .asc 文件，跳过前 40 行文件头，
    合并为按日期索引的 DataFrame。

    Args:
        folder: 包含 .asc 文件的目录路径

    Returns:
        pd.DataFrame: 按日期索引，包含 hour / mjd / X_mass / Y_mass / Z_mass
            / X_motion / Y_motion / Z_motion 列

    示例:
        >>> aam = load_AAM_data("./data/aam_oam/aam/")
        >>> print(aam.columns)
        Index(['hour', 'mjd', 'X_mass', 'Y_mass', 'Z_mass',
               'X_motion', 'Y_motion', 'Z_motion'], dtype='object')
    """
    dfs = []
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith('.asc'):
            continue
        filepath = os.path.join(folder, filename)
        with open(filepath, 'r') as f:
            lines = f.readlines()
        # AAM 跳过 40 行文件头
        data = lines[40:]
        data = [line.strip().split() for line in data if line.strip()]
        data = [
            [
                f"{int(line[0]):04d}-{int(line[1]):02d}-{int(line[2]):02d}",
                int(line[3]),
                float(line[4]),
            ] + [float(x) for x in line[5:]]
            for line in data if line
        ]
        label = ['date', 'hour', 'mjd',
                 'X_mass', 'Y_mass', 'Z_mass',
                 'X_motion', 'Y_motion', 'Z_motion']
        df = pd.DataFrame(data, columns=label)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        dfs.append(df)

    df_all = pd.concat(dfs)
    df_all.sort_index(inplace=True)
    return df_all


def load_OAM_data(folder: str) -> pd.DataFrame:
    """
    加载海洋角动量（OAM）数据。

    读取指定目录下所有 .asc 文件，跳过前 42 行文件头，
    合并为按日期索引的 DataFrame。

    Args:
        folder: 包含 .asc 文件的目录路径

    Returns:
        pd.DataFrame: 按日期索引，列同 AAM（hour / mjd / X_mass / Y_mass / Z_mass
            / X_motion / Y_motion / Z_motion）

    示例:
        >>> oam = load_OAM_data("./data/aam_oam/oam/")
    """
    dfs = []
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith('.asc'):
            continue
        filepath = os.path.join(folder, filename)
        with open(filepath, 'r') as f:
            lines = f.readlines()
        # ★ OAM 跳过 42 行文件头（AAM 是 40 行）
        data = lines[42:]
        data = [line.strip().split() for line in data if line.strip()]
        data = [
            [
                f"{int(line[0]):04d}-{int(line[1]):02d}-{int(line[2]):02d}",
                int(line[3]),
                float(line[4]),
            ] + [float(x) for x in line[5:]]
            for line in data if line
        ]
        label = ['date', 'hour', 'mjd',
                 'X_mass', 'Y_mass', 'Z_mass',
                 'X_motion', 'Y_motion', 'Z_motion']
        df = pd.DataFrame(data, columns=label)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        dfs.append(df)

    df_all = pd.concat(dfs)
    df_all.sort_index(inplace=True)
    return df_all
