#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_template.py — 验证模型模板能否正常接入框架
=================================================
使用 models/model_template.py 中的 MyModel，按 verify.py 同样的
209 组起报点评估 PMX（1 种子），输出完整 MAE 供对比。

用法:
    python experiments/test_template.py

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
from SYSUTS.preprocessing.ls import ls_decompose, POLAR_PERIODS
from SYSUTS.core.evaluator import ForecastEvaluator
from SYSUTS.models.model_template import MyModel

TRAIN_START, TRAIN_END = "2000-01-02", "2020-01-02"
FORECAST_START, FORECAST_END, STEP = "2020-01-02", "2023-12-28", 7
PRED_LEN, SEED = 360, 25

# 生成起报日期
d = datetime.strptime(FORECAST_START, "%Y-%m-%d")
dates = []
while d <= datetime.strptime(FORECAST_END, "%Y-%m-%d"):
    dates.append(d.strftime("%Y-%m-%d")); d += timedelta(days=STEP)

print("=" * 60)
print("  SYSUTS 模型模板接入验证 (PMX, 209 组, 1 种子)")
print("=" * 60)

dataset = EOPDataset("./data/data_origin.txt")
df = dataset.load()
evaluator = ForecastEvaluator(unit_conversion=1000)

# ── 训练（一次）──
fixed_train = df.loc[TRAIN_START:TRAIN_END, "PMX"].values.astype(float)
fixed_ls = ls_decompose(fixed_train, PRED_LEN, POLAR_PERIODS, use_linear=True)

print(f"\n  训练数据: {len(fixed_train)} 点, seed={SEED}")
print(f"  LS 残差: [{fixed_ls.residual.min():.4f}, {fixed_ls.residual.max():.4f}]")

model = MyModel(seq_len=200, seq_out=20, hidden_size=64, num_epochs=200)
model.fit(fixed_ls.residual)

# ── 评估（209 组）──
forecasts, truths = [], []
for forecast_date in dates:
    fd = pd.Timestamp(forecast_date)
    train_s = df.loc[fd - pd.DateOffset(years=20):fd, "PMX"].values.astype(float)
    ls_r = ls_decompose(train_s, PRED_LEN, POLAR_PERIODS, use_linear=True)
    res_fc = model.predict(ls_r.residual, PRED_LEN)
    forecast = ls_r.forecast + res_fc
    truth = df.loc[fd + pd.Timedelta(days=1):
                    fd + pd.Timedelta(days=PRED_LEN), "PMX"].values.astype(float)
    forecasts.append(forecast)
    truths.append(truth[:PRED_LEN])

mae = evaluator.compute_ae_by_step(forecasts, truths, PRED_LEN)

print(f"\n  PMX MAE (1种子, 209组)")
print(f"  {'Day':<8} {'MAE (mas)':>10}")
print(f"  {'-'*20}")
for day in [1, 2, 3, 4, 5, 7, 10, 15, 30, 60, 90, 180, 360]:
    print(f"  {day:<8} {mae[day-1]:>10.2f}")

out = Path("./results/test_template")
out.mkdir(parents=True, exist_ok=True)
np.save(out / "mae_PMX.npy", mae)
pd.DataFrame({'day': np.arange(1, PRED_LEN+1), 'mae': mae}).to_csv(
    out / "mae_PMX.csv", index=False)

print(f"\n  结果已保存: {out}/")
print(f"  可与 results/verify/mae_PMX.npy 对比（验证需跑 verify.py --target PMX）")
print("=" * 60)
