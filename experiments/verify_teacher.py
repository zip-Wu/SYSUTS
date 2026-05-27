#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_teacher.py — 老师演示脚本（无需 GPU，只加载预训练模型）
=============================================================
前提: 已在服务器上运行过 verify.py --all，模型保存于 saved_models/wzpnet/。

工作流:
    服务器:  python experiments/verify.py --all
            → 训练 36 个模型 (3目标×12种子)
            → 保存在 saved_models/wzpnet/

    老师电脑: 复制整个 SYSUTS 目录
            → python experiments/verify_teacher.py
            → 加载已有模型 → 评估 209 组 → 打印 MAE 汇总表

用法:
    python experiments/verify_teacher.py

作者: 吴梓鹏, 2025-05-26
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

_PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from SYSUTS.preprocessing.dataset import EOPDataset
from SYSUTS.preprocessing.ls import ls_decompose, ls_decompose_ut1, POLAR_PERIODS
from SYSUTS.preprocessing.leap_second import remove_leap_seconds
from SYSUTS.preprocessing.tide import remove_tide_ut1, RG_ZONT2
from SYSUTS.core.evaluator import ForecastEvaluator
from SYSUTS.models.wzpnet import WZPNetModel

# ── 参数 ────────────────────────────────────────────────────
ALL_SEEDS = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
PRED_LEN = 360
TRAIN_START, TRAIN_END = "2000-01-02", "2020-01-02"
FORECAST_START, FORECAST_END, STEP = "2020-01-02", "2023-12-28", 7

POLAR_CONFIG = {
    "seq_len": 200, "seq_out": 20, "dropout": 0.0,
    "use_ar": True, "ar_seq": 200,
    "use_gru": False, "gru_seq": 0, "gru_layer": 1, "gru_hidden": 64,
    "use_skip": True, "skip_seq": 200, "skip_stride": 3,
    "skip_layer": 1, "skip_hidden": 64,
    "use_lstm": False, "lstm_seq": 0, "lstm_layer": 1, "lstm_hidden": 64,
    "use_transformer": False, "trans_seq": 0, "trans_layers": 2,
    "trans_nhead": 4, "trans_d_model": 64, "trans_dim_feedforward": 256,
    "num_epochs": 2000, "batch_size": 200, "learning_rate": 0.01,
    "val_num": 50, "shuffle": False,
}

UT1_CONFIG = {
    "seq_len": 100, "seq_out": 1, "dropout": 0.0,
    "use_ar": True, "ar_seq": 100,
    "use_gru": False, "gru_seq": 0, "gru_layer": 1, "gru_hidden": 64,
    "use_skip": True, "skip_seq": 100, "skip_stride": 3,
    "skip_layer": 1, "skip_hidden": 64,
    "use_lstm": False, "lstm_seq": 0, "lstm_layer": 1, "lstm_hidden": 64,
    "use_transformer": False, "trans_seq": 0, "trans_layers": 2,
    "trans_nhead": 4, "trans_d_model": 64, "trans_dim_feedforward": 256,
    "num_epochs": 2000, "batch_size": 200, "learning_rate": 0.01,
    "val_num": 50, "shuffle": True,
}


# ── 工具 ────────────────────────────────────────────────────

def gen_dates(start, end, step):
    d, dates = datetime.strptime(start, "%Y-%m-%d"), []
    while d <= datetime.strptime(end, "%Y-%m-%d"):
        dates.append(d.strftime("%Y-%m-%d")); d += timedelta(days=step)
    return dates


# ── 极移评估（仅推理，不训练）──────────────────────────────

def eval_polar(dataset, df, target, config, seeds, dates):
    evaluator = ForecastEvaluator(unit_conversion=1000)
    all_seeds_mae = []

    for seed in seeds:
        model = WZPNetModel(**config, seed=seed, target=target,
                            train_date="20200102")
        # ★ 检查预训练模型是否存在
        if not model._get_model_path().exists():
            raise RuntimeError(
                f"\n  模型文件不存在: {model._get_model_path()}\n"
                f"  请先在服务器上运行: python experiments/verify.py --all\n"
                f"  然后将整个 SYSUTS 目录复制到本机。")
        model.fit(np.zeros(1))  # 自动加载已有模型

        forecasts, truths = [], []
        for forecast_date in dates:
            fd = pd.Timestamp(forecast_date)
            train_s = df.loc[fd - pd.DateOffset(years=20):fd, target].values.astype(float)
            ls_r = ls_decompose(train_s, PRED_LEN, POLAR_PERIODS, use_linear=True)
            res_fc = model.predict(ls_r.residual, PRED_LEN)
            forecast = ls_r.forecast + res_fc
            truth = df.loc[fd + pd.Timedelta(days=1):
                           fd + pd.Timedelta(days=PRED_LEN), target].values.astype(float)
            forecasts.append(forecast)
            truths.append(truth[:PRED_LEN])

        all_seeds_mae.append(
            evaluator.compute_ae_by_step(forecasts, truths, PRED_LEN))
    return np.array(all_seeds_mae).mean(axis=0)


# ── UT1 评估（仅推理）─────────────────────────────────────────

def eval_ut1(dataset, df, config, seeds, dates):
    df_tai_full = remove_leap_seconds(df)
    evaluator = ForecastEvaluator(unit_conversion=1.0)
    all_seeds_mae = []

    for seed in seeds:
        model = WZPNetModel(**config, seed=seed, target="UT1",
                            train_date="20200102")
        if not model._get_model_path().exists():
            raise RuntimeError(
                f"\n  模型文件不存在: {model._get_model_path()}\n"
                f"  请先在服务器上运行: python experiments/verify.py --all\n"
                f"  然后将整个 SYSUTS 目录复制到本机。")
        model.fit(np.zeros(1))

        forecasts, truths = [], []
        for forecast_date in dates:
            fd = pd.Timestamp(forecast_date)
            df_s = dataset.get_df_slice(df, fd - pd.DateOffset(years=20), fd)
            ut1r = (remove_leap_seconds(df_s)['UT1-TAI'].astype(float).values
                    - remove_tide_ut1(df_s))
            ls_s = ls_decompose_ut1(np.diff(ut1r), PRED_LEN, use_linear=True)
            res_fc = model.predict(ls_s.residual, PRED_LEN)
            ut1r_fc = np.cumsum(ls_s.forecast + res_fc) + ut1r[-1]
            mjd_last = df_s['MJD'].iloc[-1]
            tide_test = RG_ZONT2(np.arange(mjd_last+1, mjd_last+1+PRED_LEN), 'UT1')
            forecast = ut1r_fc + tide_test
            truth = (df_tai_full.loc[fd + pd.Timedelta(days=1):
                                     fd + pd.Timedelta(days=PRED_LEN), 'UT1-TAI']
                     .values.astype(float) * 1000.0)
            forecasts.append(forecast * 1000.0)
            truths.append(truth[:PRED_LEN])

        all_seeds_mae.append(
            evaluator.compute_ae_by_step(forecasts, truths, PRED_LEN))
    return np.array(all_seeds_mae).mean(axis=0)


# ── 主函数 ──────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  SYSUTS EOP 预报 —— 老师演示模式")
    print("  (加载服务器预训练模型，仅推理，无需 GPU)")
    print("=" * 60)

    dataset = EOPDataset("./data/data_origin.txt")
    df = dataset.load()
    dates = gen_dates(FORECAST_START, FORECAST_END, STEP)
    print(f"\n  数据: {len(df)} 行, {df.index.min().date()} ~ {df.index.max().date()}")
    print(f"  起报点: {len(dates)} 组 ({dates[0]} ~ {dates[-1]})")
    print(f"  种子数: {len(ALL_SEEDS)}")

    # 逐个目标评估
    results = {}
    for target, config in [("PMX", POLAR_CONFIG), ("PMY", POLAR_CONFIG)]:
        print(f"\n  [{target}] 加载模型并评估...", end=" ", flush=True)
        mae = eval_polar(dataset, df, target, config, ALL_SEEDS, dates)
        results[target] = mae
        print(f"Day1={mae[0]:.2f} mas")

    print(f"\n  [UT1] 加载模型并评估...", end=" ", flush=True)
    results["UT1"] = eval_ut1(dataset, df, UT1_CONFIG, ALL_SEEDS, dates)
    print(f"Day1={results['UT1'][0]:.4f} ms")

    # 汇总表
    KEY_DAYS = [1, 2, 3, 4, 5, 7, 10, 15, 30, 60, 90, 180, 360]
    print(f"\n  MAE 汇总 (12 种子 × 209 起报点)")
    print(f"  {'='*65}")
    print(f"  {'目标':<6}", end="")
    for d in KEY_DAYS:
        print(f" {d:>4}", end="")
    for target in ["PMX", "PMY", "UT1"]:
        mae = results[target]
        print(f"\n  {target:<6}", end="")
        for d in KEY_DAYS:
            print(f" {mae[d-1]:>4.2f}", end="")
    print(f"\n  {'='*65}")

    # 保存
    out = Path("./results/verify_teacher")
    out.mkdir(exist_ok=True)
    for tgt in ["PMX", "PMY", "UT1"]:
        np.save(out / f"mae_{tgt}.npy", results[tgt])
        pd.DataFrame({'day': np.arange(1, PRED_LEN+1),
                      'mae': results[tgt]}).to_csv(out / f"mae_{tgt}.csv", index=False)
    print(f"\n  结果已保存: {out}/")


if __name__ == "__main__":
    main()
