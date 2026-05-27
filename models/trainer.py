"""
models.trainer - 神经网络训练器
================================

封装 WZPNet 的训练逻辑，支持：
- 数据打包和 DataLoader 创建
- 训练和验证循环
- 早停和学习率调整
- 最佳模型保存

作者: 吴梓鹏
创建: 2025-05-25
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset


class ModelTrainer:
    """
    神经网络训练器。
    
    负责WZPNet的训练流程管理。
    
    示例:
        >>> trainer = ModelTrainer(model=wzpnet, device="cuda")
        >>> trainer.fit(train_data, seq_len=200, seq_out=20)
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str,
        num_epochs: int = 2000,
        batch_size: int = 200,
        learning_rate: float = 0.01,
        val_num: int = 50,
        patience: int = 2,  # 原始代码逻辑: 连续 2 个 epoch 验证损失没改善才降学习率
        lr_decay_factor: float = 1.05,
        save_path: str = "./saved_model/best_model.pt",
        print_every: int = 200,  # 每多少个epoch打印一次
        shuffle: bool = False,   # DataLoader shuffle——极移用False，UT1用True
    ):
        """
        Args:
            model: 神经网络模型
            device: 计算设备
            num_epochs: 训练轮数
            batch_size: 批大小
            learning_rate: 初始学习率
            val_num: 验证集样本数
            patience: 学习率调整 patience
            lr_decay_factor: 学习率衰减因子
            save_path: 最佳模型保存路径
        """
        self.model = model
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.val_num = val_num
        self.patience = patience
        self.lr_decay_factor = lr_decay_factor
        self.save_path = save_path
        self.print_every = print_every
        self.shuffle = shuffle

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    def _create_dataloader(
        self,
        data: np.ndarray,
        seq_len: int,
        seq_out: int
    ) -> DataLoader | tuple[DataLoader, DataLoader]:
        """
        创建DataLoader。
        
        Args:
            data: 训练数据
            seq_len: 输入序列长度
            seq_out: 输出序列长度
            
        Returns:
            DataLoader或(train_loader, val_loader)
        """
        # 创建样本
        inout_seq = []
        for i in range(len(data) - seq_len - seq_out + 1):
            train_seq = data[i:i + seq_len]
            train_label = data[i + seq_len:i + seq_len + seq_out]
            inout_seq.append((train_seq, train_label))
        
        # 转换为Tensor
        trains = torch.stack([
            torch.tensor(seq[:, None], dtype=torch.float32)
            for seq, _ in inout_seq
        ])  # [n_samples, seq_len, 1]
        
        labels = torch.stack([
            torch.tensor(label, dtype=torch.float32)
            for _, label in inout_seq
        ])  # [n_samples, seq_out]
        
        dataset = TensorDataset(trains, labels)
        
        # 无需验证集
        if self.val_num == 0:
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        # 划分训练集和验证集（从末尾取验证集）
        train_num = len(dataset) - self.val_num
        train_dataset = Subset(dataset, range(train_num))
        val_dataset = Subset(dataset, range(train_num, train_num + self.val_num))
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=self.shuffle)
        
        return train_loader, val_loader
    
    def fit(
        self,
        train_data: np.ndarray,
        seq_len: int,
        seq_out: int
    ) -> dict:
        """
        训练模型。
        
        Args:
            train_data: 训练数据
            seq_len: 输入序列长度
            seq_out: 输出序列长度
            
        Returns:
            dict: 训练历史记录
        """
        # 创建数据加载器
        if self.val_num > 0:
            train_loader, val_loader = self._create_dataloader(
                train_data, seq_len, seq_out
            )
        else:
            train_loader = self._create_dataloader(train_data, seq_len, seq_out)
            val_loader = None
        
        # 训练状态（与原始 test_PMX 保持一致）
        temp_loss = 10.0       # 上一个epoch的验证损失
        temp_train = 10.0      # 历史最佳训练损失（仅用于记录）
        best_model_state = None
        count = 0              # 连续未改善计数器
        
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        for epoch in range(self.num_epochs):
            # 训练阶段
            self.model.train()
            train_losses = []
            
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)
            
            # 验证阶段
            if val_loader is not None:
                self.model.eval()
                val_losses = []
                
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                        val_losses.append(loss.item())
                
                avg_val_loss = np.mean(val_losses)
                history['val_loss'].append(avg_val_loss)
            else:
                avg_val_loss = avg_train_loss
            
            # 打印进度（与原始代码一致：每200个epoch打印）
            if epoch % self.print_every == 0:
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], '
                      f'训练集上的 MSE 为：{avg_train_loss}')
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], '
                      f'验证集上的 MSE 为：{avg_val_loss}')
            
            # ★★★ 与原始 test_PMX 完全一致的学习率调整逻辑 ★★★
            # 原始代码逻辑:
            #   if val_loss < temp_loss: temp_loss=val_loss, save model, count=0
            #   else: count++, temp_loss=val_loss
            #   if count > 1: lr /= 1.05, count=0
            # 
            # 关键点: count 从 0 开始，> 1 意味着连续 2 个 epoch val_loss 都上升时降学习率
            # 注意: temp_loss 每个 epoch 都更新（无论是否改善），这是与"历史最佳"策略的本质区别
            
            if avg_val_loss < temp_loss:
                temp_loss = avg_val_loss           # 更新基准为当前值
                best_model_state = self.model.state_dict().copy()
                count = 0
            else:
                count += 1
                temp_loss = avg_val_loss          # ⚠️ 关键！也更新temp_loss（比较的是相邻两个epoch）
            
            if count > 1:                          # ★★★ 原始代码是 > 1 不是 >= 或 > patience ★★★
                prev_lr = self.optimizer.param_groups[0]['lr']
                self.optimizer.param_groups[0]['lr'] = prev_lr / self.lr_decay_factor
                count = 0
        
        # 加载最佳模型并保存（与原始 test_PMX 一致）
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            import os
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            torch.save(best_model_state, self.save_path)
            print(f"训练已完成，当前模型保存至'{self.save_path}'")
        
        return history
