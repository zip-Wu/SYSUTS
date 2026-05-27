"""
correction.polar_corrector - 极移角动量修正器（PMX / PMY 共用）
==============================================================

实现"基础预报 + 角动量补偿"架构：

    主模型(WZPNet skipGRU)预报的 LS 残差
        │
    ┌───┴────────────────────────────┐
    │  单步残差历史 + AAM/OAM 差分特征  │ ← 构建 X
    │  多步真实误差矩阵                │ ← 构建 Y
    └───┬────────────────────────────┘
        ▼
    LiouvilleAxialPINN 训练 (每个起报点独立，不缓存)
        │
    Day1~5: 网络输出
    Day6~360: 指数衰减向0
        │
        ▼
    基础预报 + 修正量 = 最终混合预报

与原始 Jupyter Notebook 的对应关系：
    Step B → _extract_single_step_residuals()
    Step C → _build_multi_step_targets()
    Step D → _build_sliding_windows()
    Step E → _train_pinn()
    Step F → _predict_correction()

作者: 吴梓鹏
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from ..models.pinn_corrector import LiouvilleAxialPINN


class PolarCorrector:
    """
    极移角动量修正器。

    核心特性：
    - AAM/OAM 数据在 __init__ 时全局加载一次，所有起报点共享
    - 每个 (seed, forecast_date) 独立训练一个 tiny PINN（不缓存）
    - 严格复现原始 PMX/PMY智能化修正.ipynb 的全部逻辑

    Args:
        target:           "PMX" 或 "PMY"
        aam_dir:          AAM .asc 文件目录
        oam_dir:          OAM .asc 文件目录
        var_lag:          滑动窗口回看长度（默认15）
        var_window:       历史窗口大小（默认1000）
        corr_len:         直接预测天数（默认5）
        epochs:           PINN 训练轮数（默认150）
        day1_weight:      Loss 中 Day1 权重系数（默认10.0）
        damping_factor:   长期衰减乘数（默认0.98）
        lr:               PINN 学习率（默认0.01）
        seed:             随机种子（用于 PINN 初始化）
        device:           计算设备
    """

    # ★ 路径隔离：相对于当前工作目录，而非 __file__
    DEFAULT_SAVE_DIR = Path.cwd() / "saved_models" / "corrector"

    def __init__(
        self,
        target: str,
        aam_dir: str,
        oam_dir: str,
        var_lag: int = 15,
        var_window: int = 1000,
        corr_len: int = 5,
        epochs: int = 150,
        day1_weight: float = 10.0,
        damping_factor: float = 0.98,
        lr: float = 0.01,
        seed: int | None = None,
        device: str | None = None,
    ):
        self.target = target.upper()
        self.aam_dir = aam_dir
        self.oam_dir = oam_dir
        self.var_lag = var_lag
        self.var_window = var_window
        self.corr_len = corr_len
        self.epochs = epochs
        self.day1_weight = day1_weight
        self.damping_factor = damping_factor
        self.lr = lr
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ★ 全局加载 AAM/OAM 数据（只加载一次，所有 date 共享）
        print(f"[Correction] 加载角动量数据...")
        self.df_excitation = self._load_and_preprocess(aam_dir, oam_dir)
        print(f"[Correction] 角动量数据加载完成: shape={self.df_excitation.shape}")

    # =========================================================================
    # 数据加载
    # =========================================================================

    def _load_and_preprocess(self, aam_dir: str, oam_dir: str) -> pd.DataFrame:
        """加载 AAM + OAM → 选择目标分量列 → 拼接 → 一阶差分。"""
        from ..preprocessing.aam_oam import load_AAM_data, load_OAM_data

        start_date = '2000-1-1'
        end_date = '2025-1-1'

        df_aam = load_AAM_data(aam_dir)[start_date:end_date]
        df_aam = df_aam[df_aam['hour'] == 0]

        df_oam = load_OAM_data(oam_dir)[start_date:end_date]
        df_oam = df_oam[df_oam['hour'] == 0]

        if self.target == "PMX":
            cols_aam = ['X_mass', 'X_motion', 'Y_mass', 'Y_motion']
            prefix_aam = 'AAM_'
            prefix_oam = 'OAM_'
        elif self.target == "PMY":
            cols_aam = ['X_mass', 'X_motion', 'Y_mass', 'Y_motion']
            prefix_aam = 'AAM_'
            prefix_oam = 'OAM_'
        else:
            raise ValueError(f"PolarCorrector 不支持目标: {self.target}")

        df_aam_selected = df_aam[cols_aam].copy()
        df_aam_selected.columns = [prefix_aam + c for c in cols_aam]

        df_oam_selected = df_oam[cols_aam].copy()
        df_oam_selected.columns = [prefix_oam + c for c in cols_aam]

        df_all = pd.concat([df_aam_selected, df_oam_selected], axis=1).dropna()
        df_all = df_all.diff().fillna(0)

        return df_all

    # =========================================================================
    # 核心修正接口
    # =========================================================================

    def correct(
        self,
        base_model_nn,              # 已训练好的 WZPNet 实例 (nn.Module)
        ls_residual: np.ndarray,     # 训练集 LS 残差
        base_forecast: np.ndarray,   # 基础模型预报 (pred_len,)
        forecast_date: str,         # 起报日期
        seq_len: int,               # 主模型的 seq_len
        pred_len: int = 360,        # 总预报长度
    ) -> np.ndarray:
        """
        执行单次起报点的完整角动量修正流程。

        严格复现原始 notebook 中一个 (seed, num) 循环体内的全部逻辑（Step B ~ F）。

        Args:
            base_model_nn:  WZPNet 神经网络实例（已加载权重）
            ls_residual:   LS 分解后的残差序列（numpy 1D array）
            base_forecast:  基础模型输出的预报值 (pred_len,)
            forecast_date:  起报日期字符串
            seq_len:        主模型输入序列长度
            pred_len:       总预报长度

        Returns:
            np.ndarray: 修正后的预报值 (pred_len,) = base_forecast + var_correct
        """
        x = ls_residual  # 别名，与原 notebook 变量名一致

        # ------------------------------------------------------------------
        # Step B: 提取单步残差历史
        # ------------------------------------------------------------------
        start_idx = len(x) - self.var_window
        hist_inputs = []
        hist_targets = []
        for k in range(self.var_window):
            curr_idx = start_idx + k
            hist_inputs.append(x[curr_idx - seq_len : curr_idx])
            hist_targets.append(x[curr_idx])

        # 批量推理：用主模型对历史窗口做单步预测
        hist_inputs_tensor = torch.tensor(
            np.array(hist_inputs), dtype=torch.float32
        ).unsqueeze(-1).to(self.device)

        with torch.no_grad():
            preds_hist = base_model_nn(hist_inputs_tensor)[:, 0].cpu().numpy()

        res_hist = np.array(hist_targets) - preds_hist  # (var_window,)

        # AAM/OAM 特征切片
        feat_slice = self.df_excitation.loc[:forecast_date].iloc[-self.var_window:]
        train_data = np.column_stack([res_hist, feat_slice.values])

        # ------------------------------------------------------------------
        # Step C: 构建多步真实误差矩阵
        # ------------------------------------------------------------------
        res_hist_multi = np.zeros((self.var_window, self.corr_len))
        for k in range(self.var_lag, self.var_window - self.corr_len + 1):
            curr_idx = start_idx + k
            curr_input = torch.tensor(
                x[curr_idx - seq_len : curr_idx], dtype=torch.float32
            ).reshape(1, seq_len, 1).to(self.device)

            curr_forecast_list = []
            loops = (self.corr_len + base_model_nn.seq_out - 1) // base_model_nn.seq_out
            with torch.no_grad():
                for j in range(loops):
                    out = base_model_nn(curr_input)
                    curr_forecast_list.append(out.view(-1).cpu())
                    curr_input = torch.cat(
                        (curr_input[:, base_model_nn.seq_out:, :], out.unsqueeze(-1)), dim=1
                    )

            curr_forecast = np.concatenate(curr_forecast_list)[:self.corr_len]
            true_future = x[curr_idx : curr_idx + self.corr_len]
            res_hist_multi[k, :] = true_future - curr_forecast

        # ------------------------------------------------------------------
        # Step D: 构建训练集（滑动窗口）
        # ------------------------------------------------------------------
        X_train_c = []
        Y_train_c = []
        n_samples = len(train_data) - self.var_lag - self.corr_len + 1
        for w in range(n_samples):
            X_train_c.append(train_data[w : w + self.var_lag].flatten())
            Y_train_c.append(res_hist_multi[w + self.var_lag, :])

        X_train_c = np.array(X_train_c)
        Y_train_c = np.array(Y_train_c)

        # 标准化 X
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_train_c)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        Y_tensor = torch.tensor(Y_train_c * 1000.0, dtype=torch.float32).to(self.device)

        # ------------------------------------------------------------------
        # Step E: 训练 LiouvilleAxialPINN
        # ------------------------------------------------------------------
        pinn_model = self._train_pinn(X_tensor, Y_tensor)

        # ------------------------------------------------------------------
        # Step F: 预测与物理外推托底
        # ------------------------------------------------------------------
        var_correct = np.zeros(pred_len)
        pinn_model.eval()
        with torch.no_grad():
            curr_X = train_data[-self.var_lag:].flatten().reshape(1, -1)
            curr_X_scaled = scaler_X.transform(curr_X)
            input_tensor = torch.tensor(curr_X_scaled, dtype=torch.float32).to(self.device)

            forecast_window, damping_factor = pinn_model(input_tensor)

            # 前 corr_len 天使用网络直接预测
            pred_trajectory = forecast_window.cpu().numpy()[0] / 1000.0
            var_correct[:self.corr_len] = pred_trajectory

        # 超出 corr_len 的天数：指数衰减向0
        last_lambda = damping_factor.item() * self.damping_factor
        for step in range(self.corr_len, pred_len):
            var_correct[step] = var_correct[step - 1] * last_lambda

        # 最终混合预报
        final_forecast = base_forecast + var_correct
        return final_forecast

    # =========================================================================
    # PINN 训练
    # =========================================================================

    def _train_pinn(
        self, X_tensor: torch.Tensor, Y_tensor: torch.Tensor
    ) -> LiouvilleAxialPINN:
        """
        训练 LiouvilleAxialPINN 模型。

        Loss 与原 notebook 完全一致：
            loss = day1_weight * MSE(Day1) + MSE(Day2~5) + 0.1*TV_smooth + |gate|

        Returns:
            训练完成的 LiouvilleAxialPINN 模型
        """
        input_dim = X_tensor.shape[1]
        c_model = LiouvilleAxialPINN(input_dim=input_dim, out_steps=self.corr_len).to(self.device)

        if self.seed is not None:
            torch.manual_seed(self.seed)

        optimizer = optim.Adam(c_model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        c_model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            forecast_window, _ = c_model(X_tensor)

            # 数据驱动的 Loss（死保第一天）
            loss_data_day1 = criterion(forecast_window[:, 0], Y_tensor[:, 0])
            loss_data_others = criterion(forecast_window[:, 1:], Y_tensor[:, 1:])

            # 物理正则化
            loss_physics = torch.abs(c_model.gate[0]) * 0.5
            # TV Loss：惩罚突变
            loss_smooth = torch.mean(torch.abs(forecast_window[:, 1:] - forecast_window[:, :-1]))

            # 总 Loss
            loss = (
                self.day1_weight * loss_data_day1
                + loss_data_others
                + 0.1 * loss_smooth
                + loss_physics
            )

            loss.backward()
            optimizer.step()

        return c_model
