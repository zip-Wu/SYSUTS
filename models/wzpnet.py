"""
WZPNet - 混合神经网络模型

作者: 吴梓鹏
创建时间: 2026.4.2

模型结构:
    - AR层: 线性自回归
    - GRU层: 门控循环单元（可选）
    - skipGRU层: 跳跃采样GRU（可选）

三个分支并行，输出相加得到最终预测。

模型管理:
    - 模型命名: {model_type}_{target}_{train_date}_{seed}.pt
    - 保存路径: SYSUTS/saved_models/wzpnet/
    - 元数据: 同目录下的 {model_name}.json
"""

from datetime import datetime
from pathlib import Path
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from ..core.model import BaseModel


class WZPNet(nn.Module):
    """
    WZPNet神经网络架构。

    支持以下分支的灵活开关与组合（直接相加）：
        - AR      (Linear)    : seq_ar > 0
        - GRU                 : seq_gru > 0
        - skipGRU             : skip_num > 0 and skip_len > 0
        - LSTM                : seq_lstm > 0
        - Transformer         : seq_transformer > 0

    各分支参数全部为 0 时表示禁用该分支，可任意组合使用。
    """

    def __init__(
        self,
        seq_out: int,
        dropout: float = 0.0,
        # AR 分支
        seq_ar: int = 0,
        # GRU 分支
        seq_gru: int = 0,
        gru_layer: int = 1,
        gru_hidden: int = 64,
        # skipGRU 分支
        skip_num: int = 0,
        skip_len: int = 0,
        skip_layer: int = 1,
        skip_hidden: int = 64,
        # LSTM 分支
        seq_lstm: int = 0,
        lstm_layer: int = 1,
        lstm_hidden: int = 64,
        # Transformer 分支
        seq_transformer: int = 0,
        trans_layers: int = 2,
        trans_nhead: int = 4,
        trans_d_model: int = 64,
        trans_dim_feedforward: int = 256,
    ):
        """
        Args:
            seq_out: 输出序列长度
            dropout: Dropout概率
            seq_ar: AR层输入长度（0表示禁用）
            seq_gru: GRU层输入长度（0表示禁用）
            gru_layer: GRU层数
            gru_hidden: GRU隐藏层维度
            skip_num: skipGRU采样点数（0表示禁用）
            skip_len: skipGRU跳跃步长
            skip_layer: skipGRU层数
            skip_hidden: skipGRU隐藏层维度
            seq_lstm: LSTM层输入长度（0表示禁用）
            lstm_layer: LSTM层数
            lstm_hidden: LSTM隐藏层维度
            seq_transformer: Transformer输入长度（0表示禁用）
            trans_layers: Transformer编码器层数
            trans_nhead: 多头注意力头数
            trans_d_model: Transformer模型维度
            trans_dim_feedforward: FFN维度
        """
        super().__init__()
        self.seq_out = seq_out
        self.seq_ar = seq_ar
        self.seq_gru = seq_gru
        self.skip_len = skip_len
        self.seq_lstm = seq_lstm
        self.seq_transformer = seq_transformer
        self.trans_d_model = trans_d_model
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

        # 可学习权重（预留，当前未使用）
        self.w1 = nn.Parameter(torch.tensor(1.0))
        self.w2 = nn.Parameter(torch.tensor(1.0))
        self.w3 = nn.Parameter(torch.tensor(1.0))
        self.w4 = nn.Parameter(torch.tensor(1.0))

        # AR层（Linear 直接映射）
        if seq_ar > 0:
            self.ar_linear = nn.Linear(seq_ar, seq_out, bias=True)

        # GRU层
        if seq_gru > 0:
            self.gru = nn.GRU(1, gru_hidden, gru_layer, batch_first=True)
            self.gru_linear = nn.Linear(gru_hidden, seq_out, bias=True)

        # skipGRU层
        if skip_num > 0 and skip_len > 0:
            self.skip_num = skip_num
            self.skip_hidden = skip_hidden
            self.skip_gru = nn.GRU(1, skip_hidden, skip_layer, batch_first=True)
            self.skip_linear = nn.Linear(skip_len * skip_hidden, seq_out, bias=True)

        # LSTM层
        if seq_lstm > 0:
            self.lstm = nn.LSTM(1, lstm_hidden, lstm_layer, batch_first=True)
            self.lstm_linear = nn.Linear(lstm_hidden, seq_out, bias=True)

        # Transformer层
        if seq_transformer > 0:
            import math
            self.trans_input_proj = nn.Linear(1, trans_d_model)
            # 位置编码（固定式，不参与训练）
            pe = torch.zeros(seq_transformer, trans_d_model)
            position = torch.arange(0, seq_transformer, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, trans_d_model, 2).float()
                * (-math.log(10000.0) / trans_d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            if trans_d_model % 2 == 1:
                pe[:, 1::2] = torch.cos(position * div_term[:-1])
            else:
                pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('trans_pe', pe.unsqueeze(0))  # [1, seq, d_model]
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=trans_d_model,
                nhead=trans_nhead,
                dim_feedforward=trans_dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=trans_layers)
            self.trans_fc = nn.Linear(trans_d_model, seq_out, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入序列 [batch, seq_len, 1]

        Returns:
            torch.Tensor: 输出 [batch, seq_out]
        """
        batch = x.size(0)
        y = torch.zeros(batch, self.seq_out).to(x.device)

        # AR分支
        if self.seq_ar > 0:
            y = y + self._ar_forward(x)

        # GRU分支
        if self.seq_gru > 0:
            y = y + self._gru_forward(x)

        # skipGRU分支
        if self.skip_len > 0:
            y = y + self._skip_forward(x)

        # LSTM分支
        if self.seq_lstm > 0:
            y = y + self._lstm_forward(x)

        # Transformer分支
        if self.seq_transformer > 0:
            y = y + self._transformer_forward(x)

        return y

    def _ar_forward(self, x: torch.Tensor) -> torch.Tensor:
        """AR层前向传播。"""
        x = torch.squeeze(x, -1)  # [batch, seq_len]
        x = x[:, -self.seq_ar:]   # [batch, seq_ar]
        return self.ar_linear(x)  # [batch, seq_out]

    def _gru_forward(self, x: torch.Tensor) -> torch.Tensor:
        """GRU层前向传播。"""
        x = x[:, -self.seq_gru:, :]  # [batch, seq_gru, 1]
        _, h = self.gru(x)           # [num_layers, batch, hidden]
        h = h[-1, :, :]              # [batch, hidden]
        if self.dropout:
            h = self.dropout(h)
        return self.gru_linear(h)    # [batch, seq_out]

    def _skip_forward(self, x: torch.Tensor) -> torch.Tensor:
        """skipGRU层前向传播。"""
        batch = x.size(0)
        skip_total = self.skip_num * self.skip_len

        x = x[:, -skip_total:, :].contiguous()  # [batch, skip_total, 1]
        x = x.view(batch, self.skip_num, self.skip_len)  # [batch, skip_num, skip_len]
        x = x.permute(0, 2, 1).contiguous()  # [batch, skip_len, skip_num]
        x = x.view(batch * self.skip_len, self.skip_num)  # [batch*skip_len, skip_num]
        x = torch.unsqueeze(x, -1)  # [batch*skip_len, skip_num, 1]

        _, h = self.skip_gru(x)  # [num_layers, batch*skip_len, hidden]
        h = h[-1, :, :]  # [batch*skip_len, hidden]
        h = h.view(batch, self.skip_len * self.skip_hidden)  # [batch, skip_len*hidden]

        if self.dropout:
            h = self.dropout(h)

        return self.skip_linear(h)  # [batch, seq_out]

    def _lstm_forward(self, x: torch.Tensor) -> torch.Tensor:
        """LSTM层前向传播。"""
        x = x[:, -self.seq_lstm:, :]   # [batch, seq_lstm, 1]
        _, (h, _) = self.lstm(x)        # h: [num_layers, batch, hidden]
        h = h[-1, :, :]                 # [batch, hidden]
        if self.dropout:
            h = self.dropout(h)
        return self.lstm_linear(h)      # [batch, seq_out]

    def _transformer_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transformer层前向传播。"""
        x = x[:, -self.seq_transformer:, :]            # [batch, seq_transformer, 1]
        x = self.trans_input_proj(x)                   # [batch, seq_transformer, d_model]
        x = x + self.trans_pe[:, :x.size(1), :].to(x.dtype)
        x = self.transformer_encoder(x)                # [batch, seq_transformer, d_model]
        last = x[:, -1, :]                             # [batch, d_model]（取最后时间步）
        if self.dropout:
            last = self.dropout(last)
        return self.trans_fc(last)                     # [batch, seq_out]


class WZPNetModel(BaseModel):
    """
    WZPNet预报模型（符合BaseModel接口）。

    将WZPNet神经网络包装为符合SYSUTS框架的预报模型。

    使用模式:
        1. train_first: 在第一个起报点训练，保存模型
        2. load: 直接加载已有模型，不训练

    模型命名: {model_type}_{target}_{train_date}_{seed}.pt
    """
    # ★ 路径隔离：相对于当前工作目录，而非 __file__
    # 这样无论 SYSUTS 放在哪个目录、从哪个目录运行，模型都保存在运行位置的 ./saved_models/wzpnet/
    DEFAULT_SAVE_DIR = Path.cwd() / "saved_models" / "wzpnet"

    def __init__(
        self,
        seq_len: int = 200,
        seq_out: int = 20,
        dropout: float = 0.0,
        # AR参数
        use_ar: bool = True,
        ar_seq: int | None = None,
        # GRU参数
        use_gru: bool = False,
        gru_seq: int = 0,
        gru_layer: int = 1,
        gru_hidden: int = 64,
        # skipGRU参数
        use_skip: bool = True,
        skip_seq: int | None = None,
        skip_stride: int = 3,
        skip_layer: int = 1,
        skip_hidden: int = 64,
        # LSTM参数
        use_lstm: bool = False,
        lstm_seq: int = 0,
        lstm_layer: int = 1,
        lstm_hidden: int = 64,
        # Transformer参数
        use_transformer: bool = False,
        trans_seq: int = 0,
        trans_layers: int = 2,
        trans_nhead: int = 4,
        trans_d_model: int = 64,
        trans_dim_feedforward: int = 256,
        # 训练参数
        num_epochs: int = 2000,
        batch_size: int = 200,
        learning_rate: float = 0.01,
        val_num: int = 50,
        shuffle: bool = False,  # DataLoader shuffle——极移用 False，UT1 用 True
        device: str | None = None,
        seed: int | None = None,
        # 模型管理参数
        train_mode: str = "train_first",  # "train_first" | "load"
        model_path: str | Path | None = None,
        model_name: str | None = None,
        target: str = "PMX",
        train_date: str | None = None,
        save_dir: str | Path | None = None,
    ):
        """
        Args:
            seq_len: 输入序列长度
            seq_out: 单次输出长度
            dropout: Dropout概率
            use_ar: 是否使用AR（Linear）层
            ar_seq: AR层输入长度（默认等于seq_len）
            use_gru: 是否使用GRU层
            gru_seq: GRU层输入长度
            gru_layer: GRU层数
            gru_hidden: GRU隐藏层维度
            use_skip: 是否使用skipGRU层
            skip_seq: skipGRU总输入长度（默认等于seq_len）
            skip_stride: skipGRU跳跃步长
            skip_layer: skipGRU层数
            skip_hidden: skipGRU隐藏层维度
            use_lstm: 是否使用LSTM层
            lstm_seq: LSTM层输入长度（默认等于seq_len）
            lstm_layer: LSTM层数
            lstm_hidden: LSTM隐藏层维度
            use_transformer: 是否使用Transformer层
            trans_seq: Transformer输入长度（默认等于seq_len）
            trans_layers: Transformer编码器层数
            trans_nhead: 多头注意力头数
            trans_d_model: Transformer模型维度
            trans_dim_feedforward: FFN维度
            num_epochs: 训练轮数
            batch_size: 批大小
            learning_rate: 学习率
            val_num: 验证集样本数
            shuffle: DataLoader 是否打乱顺序（极移用 False，UT1 用 True）
            device: 计算设备（None则自动选择）
            seed: 随机种子
            train_mode: 训练模式（"train_first" | "load"）
            model_path: 预训练模型路径（指定时直接加载）
            model_name: 自定义模型名称（不含.pt）
            target: 目标变量名（PMX/PMY），用于自动命名
            train_date: 训练日期（YYYY-MM-DD），用于自动命名
            save_dir: 模型保存目录
        """
        super().__init__(name="WZPNet")

        self.seq_len = seq_len
        self.seq_out = seq_out
        self.dropout = dropout

        # AR配置
        self.use_ar = use_ar
        self.ar_seq = ar_seq or (seq_len if use_ar else 0)

        # GRU配置
        self.use_gru = use_gru
        self.gru_seq = gru_seq if use_gru else 0
        self.gru_layer = gru_layer
        self.gru_hidden = gru_hidden

        # skipGRU配置
        self.use_skip = use_skip
        self.skip_seq = skip_seq or (seq_len if use_skip else 0)
        self.skip_stride = skip_stride
        self.skip_layer = skip_layer
        self.skip_hidden = skip_hidden

        # LSTM配置
        self.use_lstm = use_lstm
        self.lstm_seq = lstm_seq or (seq_len if use_lstm else 0)
        self.lstm_layer = lstm_layer
        self.lstm_hidden = lstm_hidden

        # Transformer配置
        self.use_transformer = use_transformer
        self.trans_seq = trans_seq or (seq_len if use_transformer else 0)
        self.trans_layers = trans_layers
        self.trans_nhead = trans_nhead
        self.trans_d_model = trans_d_model
        self.trans_dim_feedforward = trans_dim_feedforward

        # 训练配置
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.val_num = val_num
        self.shuffle = shuffle  # 极移用 False，UT1 用 True

        # 设备
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 随机种子
        self.seed = seed
        if seed is not None:
            self._set_seed(seed)

        # 模型管理
        self.train_mode = train_mode
        self._model_path = model_path
        self.target = target
        self.train_date = train_date
        self.save_dir = Path(save_dir) if save_dir else self.DEFAULT_SAVE_DIR

        # 自动生成模型名称
        self.model_name = model_name or self._generate_model_name()
        self._model: WZPNet | None = None

        # 检查是否已有模型
        self._is_trained = self._check_existing_model()

    def _generate_model_name(self) -> str:
        """自动生成模型文件名。

        文件名格式: wzpnet_{target}_{date}_{seed}_{config_hash}.pt

        其中 config_hash = 所有影响训练结果的超参数的哈希值。
        当任意超参数（网络结构、训练策略等）发生变化时，hash不同 → 文件名不同 →
        自然触发重新训练，无需手动 force_retrain。
        """
        import hashlib

        date_str = self.train_date.replace("-", "") if self.train_date else "unknown"
        seed_str = str(self.seed) if self.seed is not None else "noseed"

        # 所有影响模型训练结果的超参数（按字母序排列，保证确定性）
        # ★ target 必须参与 hash，否则 PMX/PMY 使用相同超参数时会生成相同文件名 ★
        config = {
            "target": self.target,
            "seq_len": self.seq_len,
            "seq_out": self.seq_out,
            "use_ar": self.use_ar,
            "ar_seq": self.ar_seq,
            "use_gru": self.use_gru,
            "gru_seq": self.gru_seq,
            "gru_layer": self.gru_layer,
            "gru_hidden": self.gru_hidden,
            "use_skip": self.use_skip,
            "skip_seq": self.skip_seq,
            "skip_stride": self.skip_stride,
            "skip_layer": self.skip_layer,
            "skip_hidden": self.skip_hidden,
            "use_lstm": self.use_lstm,
            "lstm_seq": self.lstm_seq,
            "lstm_layer": self.lstm_layer,
            "lstm_hidden": self.lstm_hidden,
            "use_transformer": self.use_transformer,
            "trans_seq": self.trans_seq,
            "trans_layers": self.trans_layers,
            "trans_nhead": self.trans_nhead,
            "trans_d_model": self.trans_d_model,
            "trans_dim_feedforward": self.trans_dim_feedforward,
            "dropout": self.dropout,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.lr,
            "val_num": self.val_num,
            "shuffle": self.shuffle,
            "seed": self.seed,
        }
        # 取 MD5 前8位，够用且文件名不会过长
        config_hash = hashlib.md5(
            str(sorted(config.items())).encode()
        ).hexdigest()[:8]

        return f"wzpnet_{self.target.lower()}_{date_str}_{seed_str}_{config_hash}.pt"

    def _get_model_path(self) -> Path:
        """获取模型保存/加载路径。"""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if self._model_path:
            return Path(self._model_path)
        return self.save_dir / self.model_name

    def _get_metadata_path(self) -> Path:
        """获取元数据文件路径。"""
        model_name = Path(self.model_name).stem  # 去掉 .pt
        return self.save_dir / f"{model_name}.json"

    def _check_existing_model(self) -> bool:
        """检查模型文件是否存在。"""
        return self._get_model_path().exists()

    def _save_metadata(self):
        """保存模型元数据到JSON。"""
        metadata = {
            "model_name": self.model_name,
            "model_type": "wzpnet",
            "target": self.target,
            "train_date": self.train_date,
            "seed": self.seed,
            "config": {
                "seq_len": self.seq_len,
                "seq_out": self.seq_out,
                "dropout": self.dropout,
                "use_ar": self.use_ar,
                "ar_seq": self.ar_seq,
                "use_gru": self.use_gru,
                "gru_seq": self.gru_seq,
                "gru_layer": self.gru_layer,
                "gru_hidden": self.gru_hidden,
                "use_skip": self.use_skip,
                "skip_seq": self.skip_seq,
                "skip_stride": self.skip_stride,
                "skip_layer": self.skip_layer,
                "skip_hidden": self.skip_hidden,
                "use_lstm": self.use_lstm,
                "lstm_seq": self.lstm_seq,
                "lstm_layer": self.lstm_layer,
                "lstm_hidden": self.lstm_hidden,
                "use_transformer": self.use_transformer,
                "trans_seq": self.trans_seq,
                "trans_layers": self.trans_layers,
                "trans_nhead": self.trans_nhead,
                "trans_d_model": self.trans_d_model,
                "trans_dim_feedforward": self.trans_dim_feedforward,
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.lr,
                "val_num": self.val_num,
            },
            "train_mode": self.train_mode,
            "created_at": datetime.now().isoformat(),
        }
        metadata_path = self._get_metadata_path()
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"元数据已保存: {metadata_path}")

    def _load_metadata(self) -> dict | None:
        """加载模型元数据。"""
        metadata_path = self._get_metadata_path()
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    @classmethod
    def from_saved(cls, model_path: str | Path, **kwargs) -> "WZPNetModel":
        """
        从已有模型加载创建实例。

        Args:
            model_path: 模型文件路径
            **kwargs: 其他配置参数（会被元数据覆盖）

        Returns:
            WZPNetModel实例（已加载模型）
        """
        model_path = Path(model_path)
        model_name = model_path.stem  # 去掉 .pt

        # 尝试加载元数据
        metadata_path = model_path.parent / f"{model_name}.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            # 使用元数据中的配置
            config = metadata.get("config", {})
            config["model_path"] = str(model_path)
            config["model_name"] = model_path.name
            config["train_mode"] = "load"
            config.update(kwargs)  # kwargs 覆盖元数据
            return cls(**config)

        # 没有元数据，只使用提供的参数
        return cls(model_path=str(model_path), train_mode="load", **kwargs)
    
    def _set_seed(self, seed: int) -> None:
        """设置随机种子，确保结果可重复性。
        
        注意：完全的可重复性需要在以下条件下保证：
        1. 使用相同的PyTorch版本
        2. 使用相同的cuDNN版本
        3. 使用相同的硬件（CPU/GPU）
        4. 不移动模型到不同的设备
        
        重要：与原始 test_PMX 保持一致，不设置 random.seed()
              （原始代码中该行被注释掉）
        """
        # 注意：random.seed(seed) 在原始 test_PMX 中被注释掉了！
        # import random
        # random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # 关键：强制使用确定性算法
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        # 设置环境变量确保Python hash的确定性
        import os
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    def _build_model(self) -> WZPNet:
        """构建WZPNet模型。"""
        skip_num = self.skip_seq // self.skip_stride if self.use_skip else 0
        skip_len = self.skip_stride if self.use_skip else 0  # use_skip=False 时必须为 0，否则 forward() 会误触发

        return WZPNet(
            seq_out=self.seq_out,
            dropout=self.dropout,
            # AR分支
            seq_ar=self.ar_seq,
            # GRU分支
            seq_gru=self.gru_seq,
            gru_layer=self.gru_layer,
            gru_hidden=self.gru_hidden,
            # skipGRU分支
            skip_num=skip_num,
            skip_len=skip_len,
            skip_layer=self.skip_layer,
            skip_hidden=self.skip_hidden,
            # LSTM分支
            seq_lstm=self.lstm_seq,
            lstm_layer=self.lstm_layer,
            lstm_hidden=self.lstm_hidden,
            # Transformer分支
            seq_transformer=self.trans_seq,
            trans_layers=self.trans_layers,
            trans_nhead=self.trans_nhead,
            trans_d_model=self.trans_d_model,
            trans_dim_feedforward=self.trans_dim_feedforward,
        ).to(self.device)
    
    def fit(self, train_data: np.ndarray, force_train: bool = False) -> None:
        """
        训练或加载模型。

        Args:
            train_data: 训练序列（残差）
            force_train: 是否强制重新训练（忽略已有模型文件和内存状态）

        Note:
            - train_mode="train_first": 如果已有模型则加载，否则训练并保存
            - train_mode="load": 直接加载已有模型（必须存在）
            - force_train=True: 强制重新训练
            - 模型一旦加载/训练完毕（_is_fitted=True），后续调用 fit() 会直接跳过，
              无需每次重复文件IO。这与原始 test_PMX 的行为一致：只有 i==0 时训练，
              后续所有起报点复用内存中的同一份模型权重。
        """
        # ★★★ 核心优化：已加载/训练过的模型直接跳过，不重复文件IO ★★★
        # 这与原始 test_PMX 的逻辑完全一致：
        #   i==0: 训练模型（Hybrid_model 带 num_epoch 参数）
        #   i>0 : 加载已有模型（Hybrid_model 带 model_path 参数）
        #         后续所有起报点共用同一份权重，不需要每次重读文件
        if self._is_fitted and not force_train:
            return

        from .trainer import ModelTrainer

        # 检查是否已有模型文件
        model_path = self._get_model_path()
        has_existing = model_path.exists()

        if self.train_mode == "load":
            # 必须加载已有模型
            if not has_existing:
                raise FileNotFoundError(
                    f"train_mode='load' 但模型不存在: {model_path}\n"
                    f"请先使用 train_mode='train_first' 训练模型。"
                )
            self._model = self._build_model()
            self.load(model_path)
            print(f"✅ 已加载模型: {model_path}")
            self._is_fitted = True
            return

        # train_mode == "train_first"
        if has_existing and not force_train:
            # 已有模型文件，加载到内存（仅首次调用会执行到这里）
            self._model = self._build_model()
            self.load(model_path)
            print(f"✅ 已加载已有模型: {model_path}")
            self._is_fitted = True
            return

        # 需要训练：先获取文件锁，防止多进程同时训练导致文件损坏
        lock_acquired = False
        lock_path = model_path.with_suffix('.lock')
        max_retries = 120  # 最多等 120 秒
        retry_interval = 1  # 每秒检查一次

        for attempt in range(max_retries):
            try:
                # os.O_CREAT | os.O_EXCL：原子操作，文件已存在则抛 FileExistsError
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                lock_acquired = True
                break
            except FileExistsError:
                if attempt == 0:
                    print(f"  检测到其他进程正在训练，等待完成... ({lock_path.name})")
                time.sleep(retry_interval)

        if not lock_acquired:
            raise TimeoutError(
                f"无法获取模型锁（{lock_path}），等待超时。"
                f"请确认没有其他进程正在训练同一模型。"
            )

        # 获取锁后，再次检查模型文件——可能其他进程刚训练完
        if model_path.exists():
            self._model = self._build_model()
            self.load(model_path)
            os.remove(lock_path)
            print(f"✅ 已加载已有模型（等待其他进程完成）: {model_path}")
            self._is_fitted = True
            return

        print(f"开始训练模型 (epochs={self.num_epochs})...")
        self._model = self._build_model()

        # 创建训练器并训练
        trainer = ModelTrainer(
            model=self._model,
            device=self.device,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            learning_rate=self.lr,
            val_num=self.val_num,
            save_path=str(model_path),  # 训练后自动保存
            shuffle=self.shuffle,
        )

        trainer.fit(train_data, self.seq_len, self.seq_out)

        # 保存元数据
        self._save_metadata()

        print(f"✅ 模型已训练并保存: {model_path}")

        # 释放锁
        if lock_acquired and lock_path.exists():
            os.remove(lock_path)
        self._is_fitted = True

    def reset(self) -> None:
        """
        重置模型状态，强制下次 fit() 时重新训练。
        
        同时删除已保存的模型文件（如果存在），确保从头训练。
        用于 FORCE_RETRAIN 场景，替代直接访问私有属性。
        
        示例:
            >>> model.reset()
            >>> model.fit(train_data)  # 将重新训练，而不是加载旧模型
        """
        model_path = self._get_model_path()
        if model_path.exists():
            os.remove(model_path)
            print(f"  已删除旧模型文件: {model_path}")
        metadata_path = self._get_metadata_path()
        if metadata_path.exists():
            os.remove(metadata_path)
        self._model = None
        self._is_fitted = False

    def predict(self, train_data: np.ndarray, pred_len: int) -> np.ndarray:
        """
        生成预报。
        
        Args:
            train_data: 历史序列
            pred_len: 预报长度
            
        Returns:
            np.ndarray: 预报序列
        """
        self.check_fitted()
        
        self._model.eval()
        forecast = []
        
        with torch.no_grad():
            # 初始输入
            test_input = torch.tensor(
                train_data[-self.seq_len:],
                dtype=torch.float32
            ).reshape(1, self.seq_len, 1).to(self.device)
            
            # 迭代预测
            num_iterations = (pred_len + self.seq_out - 1) // self.seq_out
            
            for _ in range(num_iterations):
                output = self._model(test_input)
                forecast.append(output.view(-1).cpu().numpy())
                
                # 滑动窗口更新
                test_input = torch.cat(
                    (test_input[:, self.seq_out:, :], output.unsqueeze(-1)),
                    dim=1
                )
        
        # 截取到所需长度
        forecast = np.concatenate(forecast)[:pred_len]
        return forecast
    
    def save(self, path: str | Path) -> None:
        """保存模型。"""
        if self._model is None:
            raise RuntimeError("模型未训练，无法保存")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), path)
    
    def load(self, path: str | Path) -> None:
        """加载模型。"""
        if self._model is None:
            self._model = self._build_model()
        
        self._model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self._is_fitted = True
