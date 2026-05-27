#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_correction.py - SYSUTS 角动量修正验证
==============================================
在 Linear-skipGRU 基础预报之上，应用 AAM/OAM 角动量修正（LiouvilleAxialPINN），
评估修正后的 MAE。

核心流程:
    1. 基础预报: 同 verify.py（固定训练集，模型一次训练，209组预报）
    2. 角动量修正: 对每组预报，PolarCorrector/UT1Corrector 生成修正量
    3. 基础预报 + 修正量 = 混合预报 → 与真值对比

用法:
    python experiments/verify_correction.py --target PMX --quick
    python experiments/verify_correction.py --all

数据路径:
    AAM: ./data/aam_oam/aam/  (默认)
    OAM: ./data/aam_oam/oam/  (默认)

作者: 吴梓鹏, 2025-05-25
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(_PROJECT_ROOT))

from SYSUTS.preprocessing.dataset import EOPDataset
from SYSUTS.preprocessing.ls import ls_decompose, ls_decompose_ut1, POLAR_PERIODS
from SYSUTS.preprocessing.leap_second import remove_leap_seconds
from SYSUTS.preprocessing.tide import remove_tide_ut1, RG_ZONT2
from SYSUTS.core.evaluator import ForecastEvaluator
from SYSUTS.models.wzpnet import WZPNetModel
from SYSUTS.correction.polar_corrector import PolarCorrector
from SYSUTS.correction.ut1_corrector import UT1Corrector


# ── 超参数 ──────────────────────────────────────────────────
ALL_SEEDS = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
PRED_LEN = 360
TRAIN_START, TRAIN_END = "2000-01-02", "2020-01-02"
FORECAST_START, FORECAST_END, STEP_DAYS = "2020-01-02", "2023-12-28", 7

POLAR_MODEL_CONFIG = {
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

UT1_MODEL_CONFIG = {
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

# 角动量修正参数
CORRECTION_PARAMS = dict(var_lag=15, var_window=1000, corr_len=5,
                         epochs=150, day1_weight=10.0, damping_factor=0.98)


# ── 工具 ────────────────────────────────────────────────────

def gen_dates(start, end, step):
    d, dates = datetime.strptime(start, "%Y-%m-%d"), []
    while d <= datetime.strptime(end, "%Y-%m-%d"):
        dates.append(d.strftime("%Y-%m-%d")); d += timedelta(days=step)
    return dates


KEY_DAYS = [1, 2, 3, 4, 5, 7, 10, 15, 30, 60, 90, 180, 360]


def _print_corr_table(target, base_mae, corr_mae, unit):
    """打印基础 vs 修正 MAE 对比表。"""
    impr = (base_mae - corr_mae) / base_mae * 100
    print(f"\n  [{target}] MAE 对比 ({unit})")
    print(f"  {'='*60}")
    print(f"  {'':<8}", end="")
    for d in KEY_DAYS:
        print(f" {d:>5}", end="")
    print(f"\n  {'基础':<8}", end="")
    for d in KEY_DAYS:
        print(f" {base_mae[d-1]:>5.2f}", end="")
    print(f"\n  {'修正':<8}", end="")
    for d in KEY_DAYS:
        print(f" {corr_mae[d-1]:>5.2f}", end="")
    print(f"\n  {'改善%':<8}", end="")
    for d in KEY_DAYS:
        print(f" {impr[d-1]:>+4.0f}", end="")
    print(f"\n  {'='*60}")


# ── 极移修正验证 ────────────────────────────────────────────

def verify_polar_correction(target, data_path, aam_dir, oam_dir,
                            output_dir, quick=False):
    dates = gen_dates(FORECAST_START, FORECAST_END, STEP_DAYS)
    seeds = ALL_SEEDS[:1] if quick else ALL_SEEDS
    if quick: dates = dates[:3]

    print(f"\n  [{target}+Corr]  {len(dates)} 起报点 × {len(seeds)} 种子")
    print(f"    AAM: {aam_dir}   OAM: {oam_dir}")

    dataset = EOPDataset(data_path)
    df = dataset.load()
    evaluator = ForecastEvaluator(unit_conversion=1000)
    seq_len = POLAR_MODEL_CONFIG["seq_len"]

    # ★ 角动量修正器（全局加载一次 AAM/OAM）
    corrector = PolarCorrector(target=target, aam_dir=aam_dir,
                               oam_dir=oam_dir, **CORRECTION_PARAMS)

    all_base_mae, all_corr_mae = [], []
    for si, seed in enumerate(seeds):
        print(f"    Seed {si+1}/{len(seeds)} (seed={seed})", end="", flush=True)

        # 训练基础模型（一次）
        fixed_train = df.loc[TRAIN_START:TRAIN_END, target].values.astype(float)
        fixed_ls = ls_decompose(fixed_train, PRED_LEN, POLAR_PERIODS, use_linear=True)
        model = WZPNetModel(**POLAR_MODEL_CONFIG, seed=seed, target=target,
                            train_date="20200102")
        model.fit(fixed_ls.residual)  # 加载 verify.py 已训练的模型

        # 评估（209组，每组分别做基础预报和修正预报）
        base_forecasts, corr_forecasts, truths = [], [], []
        for forecast_date in dates:
            fd = pd.Timestamp(forecast_date)
            train_s = df.loc[fd - pd.DateOffset(years=20):fd, target].values.astype(float)
            ls_r = ls_decompose(train_s, PRED_LEN, POLAR_PERIODS, use_linear=True)
            res_fc = model.predict(ls_r.residual, PRED_LEN)

            # 基础预报
            base_fc = ls_r.forecast + res_fc

            # 角动量修正
            try:
                corr_fc = corrector.correct(
                    base_model_nn=model._model,
                    ls_residual=ls_r.residual,
                    base_forecast=base_fc,
                    forecast_date=forecast_date,
                    seq_len=seq_len,
                    pred_len=PRED_LEN,
                )
            except Exception:
                # 修正失败时退回基础预报
                corr_fc = base_fc

            truth = df.loc[fd + pd.Timedelta(days=1):
                           fd + pd.Timedelta(days=PRED_LEN), target].values.astype(float)
            base_forecasts.append(base_fc)
            corr_forecasts.append(corr_fc)
            truths.append(truth[:PRED_LEN])

        base_mae = evaluator.compute_ae_by_step(base_forecasts, truths, PRED_LEN)
        corr_mae = evaluator.compute_ae_by_step(corr_forecasts, truths, PRED_LEN)
        all_base_mae.append(base_mae)
        all_corr_mae.append(corr_mae)
        impr = (base_mae[0] - corr_mae[0]) / base_mae[0] * 100
        print(f"  -> base={base_mae[0]:.2f} corr={corr_mae[0]:.2f} mas "
              f"({impr:+.1f}%)")

    # 多种子平均
    final_base = np.array(all_base_mae).mean(axis=0)
    final_corr = np.array(all_corr_mae).mean(axis=0)

    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for tag, mae in [("base", final_base), ("hybrid", final_corr)]:
        pd.DataFrame({'day': np.arange(1, PRED_LEN+1), 'mae': mae}).to_csv(
            out_dir / f"mae_{target}_{tag}.csv", index=False)
        np.save(out_dir / f"mae_{target}_{tag}.npy", mae)

    _print_corr_table(target, final_base, final_corr, "mas")
    return final_base, final_corr


# ── UT1 修正验证 ────────────────────────────────────────────

def verify_ut1_correction(data_path, aam_dir, oam_dir, output_dir, quick=False):
    dates = gen_dates(FORECAST_START, FORECAST_END, STEP_DAYS)
    seeds = ALL_SEEDS[:1] if quick else ALL_SEEDS
    if quick: dates = dates[:3]

    print(f"\n  [UT1+Corr]  {len(dates)} 起报点 × {len(seeds)} 种子")
    print(f"    AAM: {aam_dir}   OAM: {oam_dir}")

    dataset = EOPDataset(data_path)
    df = dataset.load()
    df_tai_full = remove_leap_seconds(df)
    evaluator = ForecastEvaluator(unit_conversion=1.0)
    seq_len = UT1_MODEL_CONFIG["seq_len"]

    # ★ 角动量修正器
    corrector = UT1Corrector(aam_dir=aam_dir, oam_dir=oam_dir,
                             **CORRECTION_PARAMS)

    all_base_mae, all_corr_mae = [], []
    for si, seed in enumerate(seeds):
        print(f"    Seed {si+1}/{len(seeds)} (seed={seed})", end="", flush=True)

        # 训练基础模型（一次）
        df_fix = dataset.get_df_slice(df, pd.Timestamp(TRAIN_START), pd.Timestamp(TRAIN_END))
        ut1r_fix = (remove_leap_seconds(df_fix)['UT1-TAI'].astype(float).values
                    - remove_tide_ut1(df_fix))
        fixed_ls = ls_decompose_ut1(np.diff(ut1r_fix), PRED_LEN, use_linear=True)
        model = WZPNetModel(**UT1_MODEL_CONFIG, seed=seed, target="UT1",
                            train_date="20200102")
        model.fit(fixed_ls.residual)  # 加载 verify.py 已训练的模型

        # 评估
        base_forecasts, corr_forecasts, truths = [], [], []
        for forecast_date in dates:
            fd = pd.Timestamp(forecast_date)
            df_s = dataset.get_df_slice(df, fd - pd.DateOffset(years=20), fd)
            ut1r = (remove_leap_seconds(df_s)['UT1-TAI'].astype(float).values
                    - remove_tide_ut1(df_s))
            ut1rdiff = np.diff(ut1r)
            ls_s = ls_decompose_ut1(ut1rdiff, PRED_LEN, use_linear=True)
            res_fc = model.predict(ls_s.residual, PRED_LEN)
            combined_diff = ls_s.forecast + res_fc

            # 基础预报（逆差分+潮汐）
            ut1r_base = np.cumsum(combined_diff) + ut1r[-1]
            mjd_last = df_s['MJD'].iloc[-1]
            tide_test = RG_ZONT2(np.arange(mjd_last+1, mjd_last+1+PRED_LEN), 'UT1')
            base_fc = ut1r_base + tide_test

            # 角动量修正（在差分域训练 PINN，内部做逆差分+潮汐）
            try:
                corr_fc = corrector.correct(
                    base_model_nn=model._model,
                    ls_residual=ls_s.residual,
                    base_forecast=combined_diff,
                    forecast_date=forecast_date,
                    seq_len=seq_len,
                    pred_len=PRED_LEN,
                    last_value=ut1r[-1],
                    tide_test=tide_test,
                )
            except Exception:
                corr_fc = base_fc

            truth = (df_tai_full.loc[fd + pd.Timedelta(days=1):
                                     fd + pd.Timedelta(days=PRED_LEN), 'UT1-TAI']
                     .values.astype(float) * 1000.0)
            base_forecasts.append(base_fc * 1000.0)
            corr_forecasts.append(corr_fc * 1000.0)
            truths.append(truth[:PRED_LEN])

        base_mae = evaluator.compute_ae_by_step(base_forecasts, truths, PRED_LEN)
        corr_mae = evaluator.compute_ae_by_step(corr_forecasts, truths, PRED_LEN)
        all_base_mae.append(base_mae)
        all_corr_mae.append(corr_mae)
        impr = (base_mae[0] - corr_mae[0]) / base_mae[0] * 100
        print(f"  -> base={base_mae[0]:.4f} corr={corr_mae[0]:.4f} ms "
              f"({impr:+.1f}%)")

    final_base = np.array(all_base_mae).mean(axis=0)
    final_corr = np.array(all_corr_mae).mean(axis=0)

    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for tag, mae in [("base", final_base), ("hybrid", final_corr)]:
        pd.DataFrame({'day': np.arange(1, PRED_LEN+1), 'mae': mae}).to_csv(
            out_dir / f"mae_UT1_{tag}.csv", index=False)
        np.save(out_dir / f"mae_UT1_{tag}.npy", mae)

    _print_corr_table("UT1", final_base, final_corr, "ms")
    return final_base, final_corr


# ── 主函数 ──────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="SYSUTS 角动量修正验证")
    p.add_argument("--target", "-t", choices=["PMX","PMY","UT1"])
    p.add_argument("--all", action="store_true")
    p.add_argument("--output", "-o", default="./results/verify_correction/")
    p.add_argument("--quick", "-q", action="store_true")
    p.add_argument("--data", "-d", default="./data/data_origin.txt")
    p.add_argument("--aam", default="./data/aam_oam/aam/",
                   help="AAM 数据目录（默认 ./data/aam_oam/aam/）")
    p.add_argument("--oam", default="./data/aam_oam/oam/",
                   help="OAM 数据目录（默认 ./data/aam_oam/oam/）")
    args = p.parse_args()

    if not args.target and not args.all:
        p.print_help()
        print("\n示例:")
        print("  python experiments/verify_correction.py --target PMX --quick")
        print("  python experiments/verify_correction.py --all")
        return

    targets = ["PMX","PMY","UT1"] if args.all else [args.target]
    for t in targets:
        try:
            if t == "UT1":
                verify_ut1_correction(args.data, args.aam, args.oam,
                                      args.output, quick=args.quick)
            else:
                verify_polar_correction(t, args.data, args.aam, args.oam,
                                        args.output, quick=args.quick)
        except Exception as e:
            print(f"\n  [ERROR] {t}: {e}")
            import traceback; traceback.print_exc()
    print("\n完成。")


if __name__ == "__main__":
    main()
