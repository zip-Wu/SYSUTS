"""
correction.ut1_corrector - UT1-UTC 角动量修正器
=================================================

继承 PolarCorrector 的核心逻辑，针对 UT1-UTC 的特殊处理：

与 PMX/PMY 的关键差异：
    1. AAM/OAM 使用 Z 分量（而非 X/Y）
    2. 残差和物理特征**分别标准化**再拼接
       （防止残差噪声淹没物理信号）
    3. 修正量在 UT1Rdiff 差分域，需：
       - 逆差分 (cumsum + last_value)
       - 恢复潮汐 (+ tide_test)

作者: 吴梓鹏
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .polar_corrector import PolarCorrector
from ..preprocessing.aam_oam import load_AAM_data, load_OAM_data


class UT1Corrector(PolarCorrector):
    """
    UT1-UTC 角动量修正器。

    继承 PolarCorrector，覆写以下方法以适配 UT1 特殊流程：
        - _load_and_preprocess(): 使用 Z 分量的 AAM/OAM
        - correct():              分别标准化残差和物理特征 + 逆差分 + 恢复潮汐

    注意：correct() 返回的是**最终域**的预报值（秒），可直接与真值比较。
    """

    def __init__(self, *args, **kwargs):
        # 强制 target 为 UT1（PolarCorrector 内部用此选择列）
        if 'target' not in kwargs:
            kwargs['target'] = 'UT1'
        super().__init__(*args, **kwargs)

    # =========================================================================
    # 覆写：数据加载 —— 使用 Z 分量
    # =========================================================================

    def _load_and_preprocess(self, aam_dir: str, oam_dir: str) -> pd.DataFrame:
        """加载 AAM/OAM → 选择 Z 分量 → 拼接 → 一阶差分。"""
        start_date = '2000-1-1'
        end_date = '2025-1-1'

        df_aam = load_AAM_data(aam_dir)[start_date:end_date]
        df_aam = df_aam[df_aam['hour'] == 0]
        df_aam = df_aam[['Z_mass', 'Z_motion']]
        df_aam.columns = ['AAM_Z_mass', 'AAM_Z_motion']

        df_oam = load_OAM_data(oam_dir)[start_date:end_date]
        df_oam = df_oam[df_oam['hour'] == 0]
        df_oam = df_oam[['Z_mass', 'Z_motion']]
        df_oam.columns = ['OAM_Z_mass', 'OAM_Z_motion']

        df_all = pd.concat([df_aam, df_oam], axis=1).dropna()
        df_all_diff = df_all.diff().fillna(0)

        return df_all_diff

    # =========================================================================
    # 覆写：完整 correct 流程（含 UT1 特有后处理）
    # =========================================================================

    def correct(
        self,
        base_model_nn,
        ls_residual: np.ndarray,
        base_forecast: np.ndarray,
        forecast_date: str,
        seq_len: int,
        pred_len: int = 360,
        # UT1 额外参数
        last_value: float | None = None,     # 差分前最后一个值（用于逆差分）
        tide_test: np.ndarray | None = None,  # 预报期潮汐改正量
    ) -> np.ndarray:
        """
        执行单次起报点的 UT1 角动量修正。

        与 PolarCorrector.correct() 的差异：
        1. 分别标准化残差和物理信号
        2. 输出经过逆差分 + 潮汐恢复的结果

        Args:
            base_model_nn:  WZPNet 实例
            ls_residual:   UT1Rdiff 域的 LS 残差序列
            base_forecast:  基础模型预报（UT1Rdiff 域）
            forecast_date:  起报日期
            seq_len:        主模型 seq_len（UT1=100）
            pred_len:       总预报长度
            last_value:     差分前最后一个 UT1R 观测值（用于逆差分）
            tide_test:      预报期潮汐改正量数组 (pred_len,)

        Returns:
            np.ndarray: 最终 UT1-UTC 预报值（秒域，已恢复潮汐）
        """
        x = ls_residual  # UT1Rdiff 域

        # ------------------------------------------------------------------
        # Step B: 单步残差历史
        # ------------------------------------------------------------------
        start_idx = len(x) - self.var_window
        hist_inputs = []
        hist_targets = []
        for k in range(self.var_window):
            curr_idx = start_idx + k
            hist_inputs.append(x[curr_idx - seq_len : curr_idx])
            hist_targets.append(x[curr_idx])

        hist_inputs_tensor = torch.tensor(
            np.array(hist_inputs), dtype=torch.float32
        ).unsqueeze(-1).to(self.device)
        with torch.no_grad():
            preds_hist = base_model_nn(hist_inputs_tensor)[:, 0].cpu().numpy()

        res_hist = np.array(hist_targets) - preds_hist

        # AAM/OAM 特征切片（使用差分后的数据）
        feat_slice = self.df_excitation.loc[:forecast_date].iloc[-self.var_window:]

        # ★★★ UT1 特有：分别标准化残差和物理信号 ★★★
        scaler_res = StandardScaler()
        scaler_phys = StandardScaler()

        res_scaled = scaler_res.fit_transform(res_hist.reshape(-1, 1))
        phys_scaled = scaler_phys.fit_transform(feat_slice.values)

        train_data = np.column_stack([res_scaled, phys_scaled])

        # ------------------------------------------------------------------
        # Step C: 多步误差矩阵 Y
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
        # Step D: 滑动窗口训练集
        # ------------------------------------------------------------------
        X_train_c = []
        Y_train_c = []
        n_samples = len(train_data) - self.var_lag - self.corr_len + 1
        for w in range(n_samples):
            X_train_c.append(train_data[w : w + self.var_lag].flatten())
            Y_train_c.append(res_hist_multi[w + self.var_lag, :])

        X_train_c = np.array(X_train_c)
        Y_train_c = np.array(Y_train_c)

        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_train_c)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        Y_tensor = torch.tensor(Y_train_c * 1000.0, dtype=torch.float32).to(self.device)

        # ------------------------------------------------------------------
        # Step E: 训练 PINN
        # ------------------------------------------------------------------
        pinn_model = self._train_pinn(X_tensor, Y_tensor)

        # ------------------------------------------------------------------
        # Step F: 预测修正量 + 物理衰减
        # ------------------------------------------------------------------
        var_correct_diff = np.zeros(pred_len)  # 在差分域
        pinn_model.eval()
        with torch.no_grad():
            curr_X = train_data[-self.var_lag:].flatten().reshape(1, -1)
            curr_X_scaled = scaler_X.transform(curr_X)
            input_tensor = torch.tensor(curr_X_scaled, dtype=torch.float32).to(self.device)

            forecast_window, damping_factor = pinn_model(input_tensor)
            pred_trajectory = forecast_window.cpu().numpy()[0] / 1000.0
            var_correct_diff[:self.corr_len] = pred_trajectory

        # 指数衰减
        last_lambda = damping_factor.item() * self.damping_factor
        for step in range(self.corr_len, pred_len):
            var_correct_diff[step] = var_correct_diff[step - 1] * last_lambda

        # ------------------------------------------------------------------
        # UT1 特有后处理：逆差分 + 恢复潮汐
        # ------------------------------------------------------------------
        if last_value is None:
            raise ValueError("UT1 修正需要提供 last_value 参数")
        if tide_test is None:
            raise ValueError("UT1 修正需要提供 tide_test 参数")

        # 差分域混合预报
        hybrid_diff = base_forecast + var_correct_diff

        # 逆差分：cumsum + 最后一个观测值
        ut1r_hybrid = np.cumsum(hybrid_diff) + last_value

        # 恢复潮汐
        final_ut1 = ut1r_hybrid + tide_test

        return final_ut1
