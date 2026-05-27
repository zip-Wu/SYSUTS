#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify.py - SYSUTS Linear-skipGRU 基础预报验证
================================================
固定训练集 [2000-01-02, 2020-01-02]，训练一次模型，
在 2020~2023 年共 209 组起报点上评估 MAE。

用法:
    python experiments/verify.py --target PMX --quick   # 快速测试
    python experiments/verify.py --all                  # 完整验证
    python experiments/verify.py --compare ./results/verify/ -c ./参考结果/

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


# ── 工具 ────────────────────────────────────────────────────

def gen_dates(start, end, step):
    d, dates = datetime.strptime(start, "%Y-%m-%d"), []
    while d <= datetime.strptime(end, "%Y-%m-%d"):
        dates.append(d.strftime("%Y-%m-%d")); d += timedelta(days=step)
    return dates


KEY_DAYS = [1, 2, 3, 4, 5, 7, 10, 15, 30, 60, 90, 180, 360]


def _print_mae_table(mae, unit="mas"):
    """打印 13 个关键节点的 MAE 表。"""
    print(f"  {'='*55}")
    print(f"  {'Day':<6}", end="")
    for d in KEY_DAYS:
        print(f" {d:>4}", end="")
    print(f"\n  {'MAE':<6}", end="")
    for d in KEY_DAYS:
        print(f" {mae[d-1]:>4.2f}", end="")
    print(f"\n  {'='*55}")


# ── 极移验证 ────────────────────────────────────────────────

def verify_polar(target, data_path, output_dir, quick=False):
    dates = gen_dates(FORECAST_START, FORECAST_END, STEP_DAYS)
    seeds = ALL_SEEDS[:1] if quick else ALL_SEEDS
    if quick: dates = dates[:3]

    print(f"\n  [{target}]  {len(dates)} 起报点 × {len(seeds)} 种子")

    dataset = EOPDataset(data_path)
    df = dataset.load()
    evaluator = ForecastEvaluator(unit_conversion=1000)  # arcsec → mas

    all_seeds_mae = []
    for si, seed in enumerate(seeds):
        print(f"    Seed {si+1}/{len(seeds)} (seed={seed})", end="", flush=True)

        # 训练（一次）
        fixed_train = df.loc[TRAIN_START:TRAIN_END, target].values.astype(float)
        fixed_ls = ls_decompose(fixed_train, PRED_LEN, POLAR_PERIODS, use_linear=True)
        model = WZPNetModel(**POLAR_MODEL_CONFIG, seed=seed, target=target,
                            train_date="20200102")
        model.fit(fixed_ls.residual, force_train=True)

        # 评估（209组）
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

        mae = evaluator.compute_ae_by_step(forecasts, truths, PRED_LEN)
        all_seeds_mae.append(mae)
        print(f"  -> day1={mae[0]:.2f} mas")

    final_mae = np.array(all_seeds_mae).mean(axis=0)
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'day': np.arange(1, PRED_LEN+1), 'mae': final_mae}).to_csv(
        out_dir / f"mae_{target}.csv", index=False)
    np.save(out_dir / f"mae_{target}.npy", final_mae)
    print(f"\n  [{target}] MAE ({len(seeds)} 种子平均, {len(dates)} 起报点) mas")
    _print_mae_table(final_mae, "mas")
    return final_mae


# ── UT1 验证 ────────────────────────────────────────────────

def verify_ut1(data_path, output_dir, quick=False):
    dates = gen_dates(FORECAST_START, FORECAST_END, STEP_DAYS)
    seeds = ALL_SEEDS[:1] if quick else ALL_SEEDS
    if quick: dates = dates[:3]

    print(f"\n  [UT1]  {len(dates)} 起报点 × {len(seeds)} 种子")

    dataset = EOPDataset(data_path)
    df = dataset.load()
    df_tai_full = remove_leap_seconds(df)
    evaluator = ForecastEvaluator(unit_conversion=1.0)

    all_seeds_mae = []
    for si, seed in enumerate(seeds):
        print(f"    Seed {si+1}/{len(seeds)} (seed={seed})", end="", flush=True)

        # 训练（一次）
        df_fix = dataset.get_df_slice(df, pd.Timestamp(TRAIN_START), pd.Timestamp(TRAIN_END))
        ut1r_fix = (remove_leap_seconds(df_fix)['UT1-TAI'].astype(float).values
                    - remove_tide_ut1(df_fix))
        fixed_ls = ls_decompose_ut1(np.diff(ut1r_fix), PRED_LEN, use_linear=True)
        model = WZPNetModel(**UT1_MODEL_CONFIG, seed=seed, target="UT1",
                            train_date="20200102")
        model.fit(fixed_ls.residual, force_train=True)

        # 评估（209组）
        forecasts, truths = [], []
        for forecast_date in dates:
            fd = pd.Timestamp(forecast_date)
            df_s = dataset.get_df_slice(df, fd - pd.DateOffset(years=20), fd)
            ut1_tai = remove_leap_seconds(df_s)['UT1-TAI'].astype(float).values
            ut1r = ut1_tai - remove_tide_ut1(df_s)
            ut1rdiff = np.diff(ut1r)
            ls_s = ls_decompose_ut1(ut1rdiff, PRED_LEN, use_linear=True)
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

        mae = evaluator.compute_ae_by_step(forecasts, truths, PRED_LEN)
        all_seeds_mae.append(mae)
        print(f"  -> day1={mae[0]:.4f} ms")

    final_mae = np.array(all_seeds_mae).mean(axis=0)
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'day': np.arange(1, PRED_LEN+1), 'mae': final_mae}).to_csv(
        out_dir / "mae_UT1.csv", index=False)
    np.save(out_dir / "mae_UT1.npy", final_mae)
    print(f"\n  [UT1] MAE ({len(seeds)} 种子平均, {len(dates)} 起报点) ms")
    _print_mae_table(final_mae, "ms")
    return final_mae


# ── 对比 ────────────────────────────────────────────────────

def compare_with_reference(output_dir, ref_dir):
    print(f"\n{'='*50}\n  MAE 对比\n{'='*50}")
    targets = {"PMX": ("mas", "mae_PMX_framework_base.npy"),
               "PMY": ("mas", "mae_PMY_framework_base.npy"),
               "UT1": ("ms",  "mae_UT1_framework_base.npy")}
    for tgt, (unit, fn) in targets.items():
        rp, np_ = Path(ref_dir)/fn, Path(output_dir)/f"mae_{tgt}.npy"
        if not rp.exists() or not np_.exists(): continue
        ref, new = np.load(rp), np.load(np_)
        d = np.abs(ref - new)
        ok = np.max(d) < max(0.01*np.mean(np.abs(ref)), 0.1)
        print(f"  {tgt}: max diff={np.max(d):.4f} {unit}  "
              f"{'OK' if ok else 'FAIL'}")


# ── 主函数 ──────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="SYSUTS Linear-skipGRU 验证")
    p.add_argument("--target", "-t", choices=["PMX","PMY","UT1"])
    p.add_argument("--all", action="store_true")
    p.add_argument("--output", "-o", default="./results/verify/")
    p.add_argument("--quick", "-q", action="store_true")
    p.add_argument("--data", "-d", default="./data/data_origin.txt")
    p.add_argument("--compare", "-c", default=None)
    args = p.parse_args()

    if args.compare:
        compare_with_reference(args.output, args.compare); return

    if not args.target and not args.all:
        p.print_help(); return

    targets = ["PMX","PMY","UT1"] if args.all else [args.target]
    for t in targets:
        try:
            if t == "UT1":
                verify_ut1(args.data, args.output, quick=args.quick)
            else:
                verify_polar(t, args.data, args.output, quick=args.quick)
        except Exception as e:
            print(f"\n  [ERROR] {t}: {e}")
            import traceback; traceback.print_exc()
    print("\n完成。")


if __name__ == "__main__":
    main()
