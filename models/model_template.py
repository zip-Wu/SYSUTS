"""
models.model_template — 模型接入模板
===========================================

复制此文件，重命名为你的模型名，修改 fit() 和 predict() 即可。
不需要改动框架任何其他文件。

示例用法:
    # 在 verify.py 中替换一行即可测试:
    # model = MyModel(hidden_size=128)   # 替换原来的 WZPNetModel(...)

接口约定:
    fit(train_data):    输入 LS 残差 (n_samples,)，训练你的模型
    predict(train_data, pred_len):
                        输入同一残差序列，输出 (pred_len,) 的残差预报

作者: 吴梓鹏
版本: 模板，供后续研究参考
"""

import numpy as np
import torch
import torch.nn as nn
from SYSUTS.core.model import BaseModel


# ═══════════════════════════════════════════════════════════════
# 第 1 步: 定义你的神经网络架构
# ═══════════════════════════════════════════════════════════════

class MyNetwork(nn.Module):
    """
    示例网络: 一个简单的 MLP + GRU 混合模型。

    替换为你自己的架构 —— CNN, LSTM, Transformer, 任何 PyTorch 模块。
    """

    def __init__(self, seq_len: int = 200, seq_out: int = 20,
                 hidden_size: int = 128, dropout: float = 0.0):
        super().__init__()
        self.seq_len = seq_len
        self.seq_out = seq_out

        # GRU 层: 处理时序依赖
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        # 输出层: hidden_size → seq_out
        self.fc = nn.Linear(hidden_size, seq_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, 1)
        返回: (batch, seq_out)
        """
        out, _ = self.gru(x)          # (batch, seq_len, hidden)
        out = out[:, -1, :]           # 取最后时刻的隐状态
        out = self.fc(out)            # (batch, seq_out)
        return out


# ═══════════════════════════════════════════════════════════════
# 第 2 步: 实现 BaseModel 接口
# ═══════════════════════════════════════════════════════════════

class MyModel(BaseModel):
    """
    你的创新模型。

    必须实现的方法:
        fit(train_data)     — 训练模型
        predict(train_data, pred_len) — 生成预报

    可选实现的方法:
        save(path)           — 保存模型权重
        load(path)           — 加载模型权重
    """

    def __init__(
        self,
        seq_len: int = 200,           # 输入序列长度
        seq_out: int = 20,            # 单次输出长度
        hidden_size: int = 128,       # 隐藏层大小
        dropout: float = 0.0,         # dropout 率
        num_epochs: int = 500,        # 训练轮数
        batch_size: int = 200,        # 批次大小
        learning_rate: float = 0.01,  # 学习率
        val_num: int = 50,            # 验证集大小
        shuffle: bool = False,        # DataLoader 是否 shuffle
        device: str | None = None,    # 计算设备
        **kwargs,                     # 兼容框架传入的额外参数
    ):
        super().__init__(name="MyModel")
        self.seq_len = seq_len
        self.seq_out = seq_out
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.val_num = val_num
        self.shuffle = shuffle
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 延迟初始化网络 —— 在 fit() 中创建
        self._network: nn.Module | None = None
        self._count_no_improve = 0

    # ── 核心接口 ──────────────────────────────────────────────

    def fit(self, train_data: np.ndarray) -> None:
        """
        训练模型。

        Args:
            train_data: LS 残差序列，形状 (n_samples,)
        """
        n = len(train_data)

        # 1. 创建网络
        self._network = MyNetwork(
            seq_len=self.seq_len,
            seq_out=self.seq_out,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
        ).to(self.device)

        # 2. 构建训练样本（滑动窗口）
        X_list, y_list = [], []
        for i in range(n - self.seq_len - self.seq_out + 1):
            X_list.append(train_data[i : i + self.seq_len])
            y_list.append(train_data[i + self.seq_len : i + self.seq_len + self.seq_out])

        if len(X_list) == 0:
            raise ValueError(f"训练数据 ({n}) 不足以构造 (seq_len={self.seq_len}, "
                             f"seq_out={self.seq_out}) 的样本")

        X = np.array(X_list).reshape(-1, self.seq_len, 1)
        y = np.array(y_list)

        # 3. 划分训练/验证集
        split = len(X) - self.val_num
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # 4. 转 Tensor
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)

        # 5. 训练循环
        optimizer = torch.optim.Adam(self._network.parameters(),
                                     lr=self.learning_rate)
        criterion = nn.MSELoss()
        best_val_loss = float('inf')

        self._network.train()
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            pred = self._network(X_train_t)
            loss = criterion(pred, y_train_t)
            loss.backward()
            optimizer.step()

            # 验证
            if (epoch + 1) % 100 == 0:
                self._network.eval()
                with torch.no_grad():
                    val_pred = self._network(X_val_t)
                    val_loss = criterion(val_pred, y_val_t).item()
                self._network.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                # 学习率衰减（连续2轮不改善）
                if epoch > 0 and val_loss >= best_val_loss:
                    self._count_no_improve += 1
                else:
                    self._count_no_improve = 0

                if self._count_no_improve > 1:
                    for g in optimizer.param_groups:
                        g['lr'] /= 1.05

        self._is_fitted = True

    def predict(self, train_data: np.ndarray, pred_len: int) -> np.ndarray:
        """
        生成预报。

        Args:
            train_data: 同 fit() 的残差序列
            pred_len: 预报长度（360 天）

        Returns:
            np.ndarray: 形状 (pred_len,) 的残差预报
        """
        self.check_fitted()
        self._network.eval()

        # 取最后 seq_len 个点作为起始输入
        current = train_data[-self.seq_len:].copy()

        forecasts = []
        loops = (pred_len + self.seq_out - 1) // self.seq_out

        with torch.no_grad():
            for _ in range(loops):
                inp = torch.FloatTensor(current[-self.seq_len:]).reshape(1, -1, 1)
                inp = inp.to(self.device)
                out = self._network(inp).cpu().numpy()[0]  # (seq_out,)
                forecasts.append(out)

                # 滑动窗口：用预报值更新输入
                current = np.concatenate([current[-self.seq_len + self.seq_out:], out])

        return np.concatenate(forecasts)[:pred_len]

    # ── 可选接口 ──────────────────────────────────────────────

    def save(self, path: str) -> None:
        """保存模型权重。"""
        torch.save(self._network.state_dict(), path)

    def load(self, path: str) -> None:
        """加载模型权重。"""
        self._network = MyNetwork(
            seq_len=self.seq_len,
            seq_out=self.seq_out,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
        ).to(self.device)
        self._network.load_state_dict(torch.load(path, map_location=self.device))
        self._is_fitted = True
