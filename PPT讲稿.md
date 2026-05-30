# SYSUTS 组会 PPT 讲解稿

> 按文件夹顺序，每页讲一个 `.py` 文件。共 22 页。

---

## 第 1 页：项目概览 — `__init__.py`

**文件**: `SYSUTS/__init__.py`（根目录）

**用途**: 包的入口，定义版本号和顶层导出。

**导出**:
- `BaseModel` — 所有预报模型的抽象基类
- `ForecastEvaluator` — 预报评估器

**架构总览**:
```
SYSUTS/
├── core/             核心接口
├── preprocessing/    EOP 物理预处理（全封装）
├── models/           预报模型库
├── correction/       角动量修正（可选）
├── experiments/      实验验证脚本
└── data/             数据文件
```

**讲解要点**: 整个框架只暴露两个类给外部。本科生的任务就是实现一个 `BaseModel` 子类。

---

## 第 2 页：`core/__init__.py`

**导出**: `BaseModel`, `ForecastEvaluator`

**讲解要点**: `core` 是整个框架的"宪法"——定义所有模型必须遵守的接口。

---

## 第 3 页：`core/model.py` — 模型抽象基类 ★

**类**: `BaseModel(ABC)`

**必须实现的方法**:

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `fit(train_data)` | `(n_samples,)` | 无 | 训练模型 |
| `predict(train_data, pred_len)` | `(n_samples,), int` | `(pred_len,)` | 生成预报 |

**可选实现**:
- `save(path)` — 保存模型
- `load(path)` — 加载模型
- `check_fitted()` — 检查是否已训练（抛 RuntimeError）

**核心约定**:
- 输入是 LS 分解后的**残差**，不是原始 EOP
- 输出是残差预报值
- `fit()` 后必须设 `self._is_fitted = True`

**讲解要点**: 这是本科生唯一需要看的文件。只需实现两个方法就能接入框架。

---

## 第 4 页：`core/evaluator.py` — 预报评估器

**类**: `ForecastEvaluator`

**构造**: `unit_conversion` = 极移 1000 (arcsec→mas)，UT1 1.0

**核心方法**: `compute_ae_by_step(forecasts, truths, pred_len) → (360,)`

- 对 209 个起报点每天计算平均 MAE
- 输出 360 维数组，`MAE[i]` = 第 i+1 天预报误差

**其他方法**: `compute_metrics`, `aggregate_multi_seed`

**讲解要点**: 209 组起报 × 12 种子求平均 = 标准评估流程。每个起报点的 360 天预报分别和真值对比。

---

## 第 5 页：`preprocessing/__init__.py`

**导出 11 个组件**:

| 类别 | 组件 |
|------|------|
| 数据 | `EOPDataset`, `load_AAM_data`, `load_OAM_data` |
| LS 分解 | `ls_decompose`, `ls_decompose_ut1` |
| 潮汐 | `RG_ZONT2`, `remove_tide_ut1` |
| 闰秒 | `LEAP_SECONDS_MJD`, `remove_leap_seconds` |
| 差分 | `diff_series`, `inverse_diff` |

**讲解要点**: 这就是"我替本科生做完的 EOP 物理"。

---

## 第 6 页：`preprocessing/dataset.py` — 数据加载

**类**: `EOPDataset`

**关键方法**:

| 方法 | 作用 |
|------|------|
| `load()` | 解析 IERS C04 文件 → DataFrame（date/MJD/PMX/PMY/UT1_UTC） |
| `get_df_slice(df, start, end)` | 按日期切片（UT1 用，保留 MJD 列） |

**数据约定**: 跳过 6 行文件头，列重命名 `x(")` → PMX，`UT1-UTC(s)` → UT1_UTC

**讲解要点**: IERS C04 是全球统一的 EOP 数据标准格式。数据从 1962 年到 2025 年约 23000 行。

---

## 第 7 页：`preprocessing/ls.py` — LS 趋势分解 ★

**返回**: `LSResult(fit, forecast, residual, params)`

**极移周期项**（10 个基函数）:
```
常数 + 线性 + [435天(Chandler), 365.24天, 182.62天, 450天] × (sin + cos)
```

**UT1 周期项**（12 个基函数）:
```
常数 + 线性 + [~19年, ~9.3年, 365.24天, 182.62天, ~121天] × (sin + cos)
```

**核心算法**: `params = (B·Bᵀ)⁻¹·B·data`，`forecast = B_forecastᵀ·params`

**讲解要点**: LS 是"分离-预报-合成"策略的第一步。Chandler 摆动（435 天）是地球自转最重要的周期信号。UT1 的 LS 在差分域进行。

---

## 第 8 页：`preprocessing/tide.py` — 62 项潮汐改正

**函数**: `RG_ZONT2(MJD, EOP='UT1') → 潮汐改正量（秒）`

**来源**: IERS Conventions 2003，直接翻译自 GAMIT 软件 Fortran 代码 zont2.f

**关键项**: 第 62 项（-1617.2681）是 18.6 年交点潮，振幅最大

**讲解要点**: 固体潮汐改正只在 UT1 预报中需要。它必须在预处理阶段扣除，预报完再恢复。62 项是 IERS 规定的标准模型。

---

## 第 9 页：`preprocessing/leap_second.py` — 闰秒处理

**函数**: `remove_leap_seconds(df) → df（新增 UT1-TAI 列）`

**原理**: UTC 不连续（有闰秒跳变）→ TAI 连续 → 适合时间序列建模

**常量**: `LEAP_SECONDS_MJD` = 1972 年以来 27 个闰秒注入日的 MJD 列表

**讲解要点**: UT1-UTC 不能直接建模（序列有跳变），必须先转换为连续的 UT1-TAI。

---

## 第 10 页：`preprocessing/diff.py` — 差分变换

**函数**:
- `diff_series(data, order=1) → (diff_data, last_values)`
- `inverse_diff(diff_forecast, last_values) → 原始尺度`

**原理**: 一阶差分 `x_t' = x_t - x_{t-1}` 使序列平稳化

**讲解要点**: `last_values` 记录了每次差分的最后一个值——这是逆差分能精确恢复的关键。

---

## 第 11 页：`preprocessing/aam_oam.py` — 角动量数据加载

**函数**:
- `load_AAM_data(folder)` — 跳过 40 行头
- `load_OAM_data(folder)` — 跳过 42 行头

**列**: `hour, mjd, X_mass, Y_mass, Z_mass, X_motion, Y_motion, Z_motion`

**讲解要点**: PMX/PMY 用 X/Y 分量，UT1 用 Z 分量。mass = 气压项，motion = 风/洋流项。数据 3 小时一组，取 `hour==0` 为日值。

---

## 第 12 页：`models/__init__.py`

**导出**: `WZPNetModel`, `ModelTrainer`, `ARModel`, `LiouvilleAxialPINN`

**讲解要点**: 四个类各司其职——WZPNetModel 是主模型，ARModel 是统计基线，ModelTrainer 辅助训练，LiouvilleAxialPINN 做角动量修正。

---

## 第 13 页：`models/wzpnet.py` — WZPNet 模型 ★

**两个类**:
```
WZPNet(nn.Module)       ← 裸网络（只有 forward）
WZPNetModel(BaseModel)  ← 完整模型（fit + predict + save + load）
```

**WZPNet 架构** — 五个并行分支，输出直接相加:
```
AR ──→ Linear(seq_ar, seq_out)
GRU ──→ GRU → Linear
skipGRU ──→ skipGRU → Linear        ┐
LSTM ──→ LSTM → Linear              ├→ 求和
Transformer ──→ Positional + Encoder ┘
```

**默认配置**: 只用 AR + skipGRU（GRU/LSTM/Transformer 默认关闭）

**skipGRU 机制**（核心创新）: 把长序列按 stride 重组 → GRU 可以跳过相邻点，聚焦长周期模式

**模型管理**: 自动命名（config_hash MD5 前 8 位）→ 参数变更自动重训

**关键差异**: 极移 seq_len=200, seq_out=20, shuffle=False；UT1 seq_len=100, seq_out=1, shuffle=True

**讲解要点**: 这是论文的核心模型。skipGRU 的设计动机是 PMX 序列有 435 天的 Chandler 周期——普通 GRU 200 步很难捕获。

---

## 第 14 页：`models/trainer.py` — 训练器

**类**: `ModelTrainer`

**构造**: `model, num_epochs=2000, batch_size=200, lr=0.01, val_num=50, shuffle`

**学习率调度**（与原始 notebook 一致）:
```
相邻 epoch 的 val_loss 比较:
  if 连续 2 轮不改善 → lr /= 1.05
```

**优化器**: AdamW | **损失**: MSELoss | **验证集**: 序列末尾 50 个样本

**讲解要点**: 没有早停，只降学习率。验证集划分方法是从序列末尾取 `val_num` 个样本——因为 EOP 是时间序列，不能用随机切分。

---

## 第 15 页：`models/ar_model.py` — AR 统计基线

**类**: `ARModel(BaseModel)`

**构造**: `lag_order=365`（一年历史）

**实现**: 底层调用 `statsmodels.tsa.ar_model.AutoReg`

**讲解要点**: 最简单的统计基线模型。用于证明深度学习模型的优越性。

---

## 第 16 页：`models/pinn_corrector.py` — 角动量 PINN

**类**: `LiouvilleAxialPINN(nn.Module)` — 不是 BaseModel！

**三引擎架构**:
```
Day 1:    纯线性 (physics_day1)  ← 瞬时角动量守恒
Day 2~5:  线性物理基准 + 门控非线性补偿 (gate × residual)
Day 6+:   指数衰减外推
```

**Loss**: `10×MSE(Day1) + MSE(Day2~5) + 0.1×TV_smooth + |gate|`

**讲解要点**: 这是"物理+数据"融合的典范——Day1 死保物理规律（线性），后续逐渐允许非线性补偿，长期外推有物理托底。

---

## 第 17 页：`models/model_template.py` — 本科生模板 ★

**两个类**（与 wzpnet.py 同结构）:
```
MyNetwork(nn.Module)  ← 示例 GRU 网络
MyModel(BaseModel)    ← 完整接口（fit + predict + save + load）
```

**用法**: 复制 → 改类名 → 改网络 → 在 `test_template.py` 中测试

**讲解要点**: 这是给本科生练手的。只需修改 `MyNetwork`（网络架构）和 `MyModel.fit()`（训练逻辑），其余不用动。

---

## 第 18 页：`correction/__init__.py`

**导出**: `PolarCorrector`, `UT1Corrector`

---

## 第 19 页：`correction/polar_corrector.py` — 极移修正器

**类**: `PolarCorrector` — 独立接口，不继承 BaseModel

**核心方法**: `correct(base_model_nn, ls_residual, base_forecast, forecast_date, seq_len, pred_len)`

**六步修正流程** (Steps B~F):
```
B: 提取 var_window=1000 天单步残差历史
C: 构建多步误差矩阵 (1000×5)
D: 构建滑动窗口训练集 (var_lag=15)
E: 训练 LiouvilleAxialPINN（每起报点独立，150 epochs）
F: 前 5 天网络预测 + 后 355 天指数衰减
```

**讲解要点**: 每个起报点独立训练一个微型 PINN（不缓存）——因为修正量和起报点强相关。AAM/OAM 数据全局加载一次，所有起报点共享。

---

## 第 20 页：`correction/ut1_corrector.py` — UT1 修正器

**类**: `UT1Corrector(PolarCorrector)` — 继承极移修正器

**三个关键差异**:

| | 极移 | UT1 |
|------|------|------|
| AAM/OAM 分量 | X/Y | **Z**（轴向） |
| 标准化方式 | 合并标准化 | **分别标准化**（残差+物理分开） |
| 后处理 | 无 | **逆差分 + 恢复潮汐** |

**讲解要点**: 分别标准化是 UT1 的关键技巧——残差量级 ~1e-4，物理信号量级 ~1e1，合并标准化会淹没残差信息。

---

## 第 21 页：`experiments/verify.py` — 基础预报验证 ★

**运行**:
```bash
python experiments/verify.py --all          # 完整（3目标×12种子）
python experiments/verify.py --target PMX --quick  # 快速测试
```

**实验参数**: 训练集固定 [2000-2020]，209 起报点 (2020-2023, 每 7 天)

**核心函数**:
- `verify_polar(target, ...)` — 极移流程: LS → 模型训练(一次) → 209组 LS+预测
- `verify_ut1(...)` — UT1 流程: 闰秒→潮汐→差分→LS→模型→逆差分→潮汐

**输出**: `results/verify/mae_{PMX,PMY,UT1}.npy` + `.csv`

**讲解要点**: 这是所有实验结果的来源。模型只训练一次，但 LS 随起报点滑动。

---

## 第 22 页：`experiments/verify_correction.py` — 角动量修正验证

**运行**:
```bash
python experiments/verify_correction.py --all  # 依赖 verify.py 的模型文件
```

**核心函数**:
- `verify_polar_correction(target, ...)` — 基础预报 + PolarCorrector
- `verify_ut1_correction(...)` — 基础预报 + UT1Corrector

**输出**: `results/verify_correction/mae_{target}_{base,hybrid}.npy`

**讲解要点**: 输出基础 vs 修正 MAE 对比 + 改善百分比。修正失败退回基础预报。

---

## 第 23 页：`experiments/verify_teacher.py` — 老师演示

**运行**: `python experiments/verify_teacher.py`（CPU 即可）

**流程**: 检查模型文件存在 → 加载 → 209 组推理 → 打印三目标 MAE 汇总

**使用场景**: 服务器训练完 → 拷到老师电脑 → 一条命令展示结果

**讲解要点**: 无需 GPU，不训练。模型文件不存在时会给出明确提示。

---

## 第 24 页：`experiments/test_template.py` — 模板接入测试

**运行**: `python experiments/test_template.py`（30 秒）

**测试**: 用 `MyModel`(简单 GRU) 跑完整 PMX 209 组验证，输出 13 节点 MAE

**用途**: 验证本科生自己的模型能否正确接入框架

**讲解要点**: 本科生改好模型后跑这个脚本——不报错且输出有意义 MAE 即接入成功。

---

## 第 25 页：总结

**项目做了什么**:
- 封装了 EOP 预报的全部物理预处理（LS、潮汐、闰秒、差分）
- 实现了 Linear-skipGRU 主模型（12 种子 × 209 起报点验证通过）
- 实现了 AAM/OAM 角动量修正（基于 Liouville PINN）
- 留出了标准化的模型接入接口（BaseModel）

**本科生需要做什么**:
1. 复制 `model_template.py`
2. 实现 `fit()` 和 `predict()`
3. 运行 `test_template.py` 验证
4. 对比 `results/verify/` 中的基准 MAE

**三条数据流**:

```
极移:  原始 PMX → LS → 模型(残差) → 合成 → MAE
UT1:   UT1-UTC → 闰秒 → 潮汐 → 差分 → LS → 模型 → 逆差分 → 潮汐 → MAE
修正:   基础预报 → + PolarCorrector/UT1Corrector → 混合预报 → MAE
```
