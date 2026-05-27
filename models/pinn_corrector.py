"""
LiouvilleAxialPINN - 角动量补偿神经网络

架构设计（与原始 notebook 的 Liouville_Axial_PINN_V2 完全一致）：

    Day 1:   physics_day1(x)                    → 纯线性，梯度纯净
    Day 2~5: physics_others(x) + gate * residual_others(x)  → 门控非线性
    Day 6+:  指数衰减向0（由外部 damping_factor × 0.98 控制）

Loss = 10 * MSE(Day1) + MSE(Day2~5) + 0.1 * TV_smooth + |gate|

作者: 吴梓鹏
"""

import torch
import torch.nn as nn


class LiouvilleAxialPINN(nn.Module):
    """
    Liouville 轴向 PINN 模型 —— 用于角动量修正的轻量神经网络。

    三者通用（PMX / PMY / UT1），仅 input_dim 不同。

    Args:
        input_dim:     输入特征维度（VAR_LAG × (残差列数 + AAM/OAM列数)）
        out_steps:     输出天数（默认 5，对应 corr_len）
        hidden_dim:    非线性分支隐藏维度（默认 32）
    """

    def __init__(self, input_dim: int, out_steps: int = 5, hidden_dim: int = 32):
        super().__init__()

        # ==================== 1. 初始边界条件引擎 (Day 1 专属) ====================
        # 严格遵循角动量守恒的瞬态线性响应。不受任何隐藏层梯度的污染！
        self.physics_day1 = nn.Linear(input_dim, 1)

        # ==================== 2. 演化物理基准引擎 (Day 2 到 Day N) ====================
        # 继续保持线性映射，作为后续天数的物理主轴
        self.physics_others = nn.Linear(input_dim, out_steps - 1)

        # ==================== 3. 混沌补偿引擎 (仅针对 Day 2 到 Day N) ====================
        # 大气的非线性混沌效应需要时间累积，因此非线性引擎只作用于后续天数
        self.residual_others = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),          # PINNs 推荐使用平滑的 Tanh
            nn.Linear(hidden_dim, out_steps - 1),
        )

        # 可学习的物理门控（初始只给很小的非线性权限）
        # 使用 1-D 张量（PyTorch 2.x 要求 nn.Parameter 至少 1 维才能用 [0] 索引）
        self.gate = nn.Parameter(torch.tensor([0.01]))

        # ==================== 4. 外推阻尼器 ====================
        self.fc_damping = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播。

        Args:
            x: [batch, input_dim] 输入特征

        Returns:
            forecast: [batch, out_steps] 修正量预报
            damping:  [batch, 1]      衰减因子（Sigmoid 输出，0~1）
        """
        # 1. 绝对纯净的 Day 1 预测（硬约束）
        day1_pred = self.physics_day1(x)

        # 2. 后续天数的物理基准 + 非线性补偿
        others_base = self.physics_others(x)
        others_res = self.residual_others(x)
        others_pred = others_base + self.gate * others_res

        # 3. 拼接输出
        final_forecast = torch.cat([day1_pred, others_pred], dim=1)

        # 4. 衰减因子
        damping_factor = self.fc_damping(x)

        return final_forecast, damping_factor
