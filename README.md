# SYSUTS — Earth Orientation Parameters Forecasting Toolkit

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A research toolkit for Earth Orientation Parameters (EOP) forecasting,
supporting **polar motion (PMX/PMY)** and **UT1-UTC** prediction.

> **Author**: Wu Zipeng (吴梓鹏)
> **Version**: 0.3.0
> **Context**: Undergraduate thesis — Sun Yat-sen University

---

## Architecture

```
SYSUTS/
├── core/                 Core interfaces (BaseModel + ForecastEvaluator)
├── preprocessing/        EOP physics preprocessing (LS, tide, leap seconds, diff)
├── models/               Forecast models (WZPNet, AR, model template)
├── correction/           AAM/OAM angular momentum correction
├── experiments/          Verification & teacher demo scripts
├── data/                 IERS EOP C04 data + AAM/OAM .asc files
└── results/              Output directory (generated at runtime)
```

### Design Philosophy

All EOP physics preprocessing (LS decomposition, solid Earth tide correction
via RG_ZONT2, leap second handling, differencing) is **fully encapsulated**.
Researchers only need to implement a model class inheriting `BaseModel`
and plug it into the pipeline:

```python
from SYSUTS.core.model import BaseModel
import numpy as np

class MyModel(BaseModel):
    def fit(self, train_data: np.ndarray) -> None:
        """Train on LS residual. train_data shape: (n_samples,)"""
        ...

    def predict(self, train_data: np.ndarray, pred_len: int) -> np.ndarray:
        """Predict residual. Returns shape (pred_len,)"""
        ...
```

---

## Quick Start

### Requirements

```bash
conda env create -f environment.yml   # or: pip install torch numpy pandas scikit-learn statsmodels
```

### Verification

```bash
# Step 1: Template model test (30 seconds)
python experiments/test_template.py

# Step 2: Full verification (GPU recommended)
python experiments/verify.py --all

# Step 3: Angular momentum correction (requires Step 2)
python experiments/verify_correction.py --all

# Step 4: Teacher demo (inference only, no GPU needed)
python experiments/verify_teacher.py
```

### Run on a new model

1. Copy `models/model_template.py` → `models/my_model.py`
2. Implement `fit()` and `predict()`
3. Test: `python experiments/test_template.py` (swap `MyModel` for your class)

---

## Models Included

| Model | File | Description |
|-------|------|-------------|
| WZPNet | `models/wzpnet.py` | AR + skipGRU hybrid (main model) |
| AR | `models/ar_model.py` | Statsmodels AutoReg baseline |
| Template | `models/model_template.py` | Minimal GRU example for new models |

---

## Key References

- **LS decomposition**: IERS conventions — Chandler (435d), annual (365.24d),
  semi-annual (182.62d) periods for polar motion; 5 periods for UT1
- **Tide correction**: RG_ZONT2 — 62-term zonal tide model
  (IERS Conventions 2003 / GAMIT zont2.f)
- **AAM/OAM**: Atmospheric & Oceanic Angular Momentum from IERS Special Bureaus

---

## Citation

If you use this toolkit in your research, please cite:

```
@misc{SYSUTS2025,
  author = {Wu Zipeng},
  title  = {SYSUTS: Earth Orientation Parameters Forecasting Toolkit},
  year   = {2025},
  note   = {Undergraduate thesis, Sun Yat-sen University}
}
```
