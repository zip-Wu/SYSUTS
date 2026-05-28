# SYSUTS — 地球自转参数（EOP）预报工具包

> A research toolkit for Earth Orientation Parameters forecasting.
> Supports polar motion (PMX/PMY) and UT1-UTC prediction.
> All EOP physics preprocessing is fully encapsulated — researchers
> only need to implement a model class and plug it in.
>
> **作者**: 吴梓鹏 | **版本**: v0.3.0 | **用途**: 本科毕业设计或时间序列科学研究

---

## 一、15 秒速览

本工具包的目标：让预毕业本科生或研究生把精力集中在**设计神经网络模型**上，不用操心 EOP 物理预处理。

```
┌─────────────────────────────────────────────────┐
│              本科生只需关心这一步                  │
│  ┌──────────────┐                               │
│  │  你的新模型    │  ← 实现 fit() + predict()      │
│  └──────┬───────┘                               │
│         │ 输入: LS 残差 (numpy 1D)                │
│         │ 输出: 残差预报 (numpy 1D, 360天)         │
│         ▼                                       │
│  ┌────────────────────────────────────────┐     │
│  │          SYSUTS 框架（不用动）            │     │
│  │  LS 趋势分解 · 潮汐改正 · 闰秒处理          │     │
│  │  差分/逆差分 · 预报评估 · 角动量修正         │     │
│  └────────────────────────────────────────┘     │
└─────────────────────────────────────────────────┘
```

---

## 二、代码包结构

```
SYSUTS/
├── core/                    核心接口
│   ├── model.py                 BaseModel 抽象类（残差模型的统一接口）
│   └── evaluator.py             ForecastEvaluator（MAE/RMSE 评估）
│
├── preprocessing/            EOP 物理预处理（全封装，不用动）
│   ├── dataset.py               EOPDataset（IERS 数据解析、训练/测试切分）
│   ├── ls.py                    LS 趋势分解（极移4周期、UT1 5周期）
│   ├── tide.py                  RG_ZONT2 潮汐改正（62 项）
│   ├── leap_second.py           闰秒处理
│   ├── diff.py                  差分/逆差分
│   └── aam_oam.py               AAM/OAM 角动量数据加载
│
├── models/                   模型库（放你的新模型）
│   ├── wzpnet.py                WZPNetModel（Linear-skipGRU，当前主模型）
│   ├── ar_model.py              ARModel（自回归基线）
│   ├── trainer.py               ModelTrainer（训练循环，WZPNetModel 内部使用）
│   ├── pinn_corrector.py        LiouvilleAxialPINN（角动量修正网络）
│   └── model_template.py        ★ 模板 —— 复制它开始写你的模型
│
├── correction/              角动量修正（可选模块，使用独立接口）
│   ├── polar_corrector.py
│   └── ut1_corrector.py
│
├── experiments/             实验脚本
│   ├── verify.py                基础预报验证（PMX/PMY/UT1，209 起报点）
│   ├── verify_correction.py     角动量修正验证
│   ├── verify_teacher.py        老师演示（仅推理，无需 GPU）
│   └── test_template.py         模板模型接入测试
│
├── results/                 实验结果（运行后自动生成）
└── data/                    数据
    ├── data_origin.txt           IERS EOP C04 数据
    └── aam_oam/                  角动量 .asc 文件
```

---

## 三、两类模型的接口设计

本项目包含**两种不同角色的模型**，使用不同的接口：

### 3.1 残差预报模型 → 实现 BaseModel

用于替换核心预报网络（如 WZPNet → 你自己的网络）。

```python
from SYSUTS.core.model import BaseModel
import numpy as np

class MyModel(BaseModel):
    """残差预报模型 —— 实现 fit() 和 predict() 即可接入。"""

    def fit(self, train_data: np.ndarray) -> None:
        """train_data: LS 残差 (n_samples,)。训练完成后设 self._is_fitted = True。"""
        ...

    def predict(self, train_data: np.ndarray, pred_len: int) -> np.ndarray:
        """返回残差预报 (pred_len,)。"""
        ...
```

**适用场景**：框架替你完成了 LS 分解、潮汐改正等全部 EOP 物理预处理，你只需对残差序列建模。这也是本科生毕设的主要工作。

| 已实现 | 文件 | 说明 |
|--------|------|------|
| `WZPNetModel` | `models/wzpnet.py` | AR + skipGRU 混合网络 |
| `ARModel` | `models/ar_model.py` | statsmodels 自回归 |
| `MyModel`（模板） | `models/model_template.py` | 简单 GRU 示例 |

### 3.2 角动量修正器 → 独立接口

`PolarCorrector` / `UT1Corrector` **不实现 BaseModel**——它们有自己的 `correct()` 接口，因为：
- 每个起报点独立训练一个微型 PINN（209 次训练 vs 残差模型只训练 1 次）
- 输入包含多变量（AAM/OAM 角动量特征 + 基础预报残差）
- 输出是修正量而非直接预报值

**BaseModel 不是万能接口**。如果你的研究方向需要多变量输入或非标准预报流程，可以定义自己的接口协议。

---

## 四、数据流详解

### 4.1 极移（PMX / PMY）

```
原始 PMX 序列 (arcsec, 2000-2020, 20年)
         │
    ┌────▼────┐  LS 分解（4周期: 435d Chandler, 365.24d, 182.62d, 450d）
    │ LS 分解  │  → preprocessing/ls.py → ls_decompose()
    └────┬────┘
         │
    ┌────┴────┐
    ▼         ▼
 趋势预报    残差序列 (N)
 (360天)        │
    │      ┌────▼────┐
    │      │ 你的模型  │  ★ 接入点: fit(residual) → predict(residual, 360)
    │      └────┬────┘
    │           ▼
    │      残差预报 (360)
    │           │
    └─────┬─────┘
          ▼
    趋势 + 残差 = 最终预报 (arcsec) → ×1000 → MAE (mas)
```

### 4.2 UT1-UTC

```
UT1-UTC(s) → 去闰秒 → 去潮汐(RG_ZONT2) → 一阶差分 → LS分解(5周期)
                                                          │
                                             趋势预报 + 残差 → 你的模型
                                                          │
                                             差分域预报 → 逆差分 → 恢复潮汐
                                                          │
                                             最终 UT1-TAI (秒) → → MAE (ms)
```

**对应代码**: 直接看 `experiments/verify.py` 中的 `verify_polar()` 和 `verify_ut1()` 函数——它们是数据流的最直观实现。

---

## 五、本科生接入实操

### 步骤 1：复制模板

```bash
cp models/model_template.py models/my_model.py
```

修改 `my_model.py` 中的 `MyNetwork`（网络架构）和 `MyModel.fit()` / `MyModel.predict()`。

### 步骤 2：快速验证

编辑 `experiments/test_template.py`，把 `from ...model_template import MyModel` 改成你的模型。运行：

```bash
python experiments/test_template.py
```

输出 PMX 13 个关键节点的 MAE。不报错即接入成功。

### 步骤 3：完整对比

```bash
# ① 跑基准模型
python experiments/verify.py --all

# ② 编辑 verify.py，把 WZPNetModel 替换为你的模型
# ③ 对比 results/verify/ 中新旧 MAE
```

---

## 六、运行命令速查

```bash
# 模板测试（30 秒）
python experiments/test_template.py

# 基础验证（GPU 数小时）
python experiments/verify.py --target PMX --quick
python experiments/verify.py --all

# 角动量修正（依赖基础验证的模型文件）
python experiments/verify_correction.py --target PMX --quick
python experiments/verify_correction.py --all

# 老师演示（CPU，仅推理）
python experiments/verify_teacher.py
```

---

## 七、查阅指南

| 你想知道... | 看这个文件 |
|-------------|-----------|
| 模型接口规范 | `core/model.py` |
| 极移/UT1 的 LS 周期项 | `preprocessing/ls.py` → `POLAR_PERIODS` / `UT1_PERIODS` |
| 潮汐改正 62 项是什么 | `preprocessing/tide.py` |
| WZPNet 网络架构 | `models/wzpnet.py` → `WZPNet` 类 |
| 训练循环怎么写的 | `models/trainer.py` |
| 极移完整预报流程 | `experiments/verify.py` → `verify_polar()` |
| UT1 完整预报流程 | `experiments/verify.py` → `verify_ut1()` |
| 角动量修正怎么做的 | `correction/polar_corrector.py` → `correct()` |

---

## 八、常见问题

**Q: 为什么模型输入是残差不是原始 EOP？**
LS 分解提取了确定性长期趋势，模型只需学习残差中的非线性模式。这是 EOP 预报标准做法。

**Q: 我可以不用 BaseModel 做多变量输入吗？**
可以。BaseModel 是为"LS 残差 → 残差预报"这一特定环节设计的。如果你的模型需要原始 EOP 或多变量输入，直接在 `verify.py` 中修改预处理流程即可，不需要继承 BaseModel。

**Q: save/load 必须实现吗？**
不必须。但实现了可以利用模型缓存（同参数不重复训练）。

---

## 引用

```
@misc{SYSUTS2025,
  author = {Wu Zipeng},
  title  = {SYSUTS: Earth Orientation Parameters Forecasting Toolkit},
  year   = {2025},
  note   = {Undergraduate thesis, Sun Yat-sen University}
}
```
