"""
core.evaluator - 预报评估器
===========================

提供统一的误差统计接口，支持：
- MAE（平均绝对误差）
- RMSE（均方根误差）
- MAPE（平均绝对百分比误差）
- 多步预报误差分析（compute_ae_by_step）
- 多随机种子聚合

作者: 吴梓鹏
创建: 2025-05-25
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class MetricsResult:
    """
    误差指标结果数据类。
    
    Attributes:
        mae: 平均绝对误差
        rmse: 均方根误差
        mape: 平均绝对百分比误差（可选）
        mae_by_step: 各预报步长的MAE（可选）
        rmse_by_step: 各预报步长的RMSE（可选）
    """
    mae: float
    rmse: float
    mape: float | None = None
    mae_by_step: np.ndarray | None = None
    rmse_by_step: np.ndarray | None = None
    
    def to_dict(self) -> dict:
        """转换为字典格式。"""
        result = {
            'mae': self.mae,
            'rmse': self.rmse,
        }
        if self.mape is not None:
            result['mape'] = self.mape
        return result


class ForecastEvaluator:
    """
    预报评估器。
    
    用于计算预报值与真实值之间的各种误差指标。
    支持单步评估和多步滚动评估。
    
    示例:
        >>> evaluator = ForecastEvaluator()
        >>> metrics = evaluator.compute_metrics(forecast, truth)
        >>> print(f"MAE: {metrics.mae:.4f}")
        
        >>> # 多步预报误差分析
        >>> step_metrics = evaluator.compute_metrics_by_step(
        ...     forecasts_list, truths_list, pred_len=360
        ... )
    """
    
    def __init__(self, unit_conversion: float = 1.0):
        """
        Args:
            unit_conversion: 单位转换系数（如mas转换用1000）
        """
        self.unit_conversion = unit_conversion
    
    def compute_metrics(
        self,
        forecast: np.ndarray,
        truth: np.ndarray,
        compute_mape: bool = False
    ) -> MetricsResult:
        """
        计算基本误差指标。
        
        Args:
            forecast: 预报序列
            truth: 真实序列
            compute_mape: 是否计算MAPE（默认False，避免除零问题）
            
        Returns:
            MetricsResult: 误差指标结果
        """
        # 确保长度一致
        min_len = min(len(forecast), len(truth))
        forecast = forecast[:min_len]
        truth = truth[:min_len]
        
        # 计算误差
        error = forecast - truth
        abs_error = np.abs(error)
        
        # 应用单位转换
        mae = np.mean(abs_error) * self.unit_conversion
        rmse = np.sqrt(np.mean(error ** 2)) * self.unit_conversion
        
        # MAPE（可选）
        mape = None
        if compute_mape:
            # 避免除零
            mask = truth != 0
            if np.any(mask):
                mape = np.mean(abs_error[mask] / np.abs(truth[mask])) * 100
        
        return MetricsResult(mae=mae, rmse=rmse, mape=mape)
    
    def compute_metrics_by_step(
        self,
        forecasts: list[np.ndarray],
        truths: list[np.ndarray],
        pred_len: int
    ) -> MetricsResult:
        """
        计算各预报步长的误差（用于分析误差随预报时长变化）。
        
        Args:
            forecasts: 多个起报点的预报结果列表
            truths: 对应的真实值列表
            pred_len: 预报长度
            
        Returns:
            MetricsResult: 包含各步长误差的结果
        """
        # 收集所有起报点在各步长的误差
        errors_by_step = [[] for _ in range(pred_len)]
        
        for forecast, truth in zip(forecasts, truths):
            min_len = min(len(forecast), len(truth), pred_len)
            for i in range(min_len):
                errors_by_step[i].append(forecast[i] - truth[i])
        
        # 计算各步长的MAE和RMSE
        mae_by_step = np.array([
            np.mean(np.abs(errors)) * self.unit_conversion 
            if errors else np.nan
            for errors in errors_by_step
        ])
        
        rmse_by_step = np.array([
            np.sqrt(np.mean(np.array(errors) ** 2)) * self.unit_conversion
            if errors else np.nan
            for errors in errors_by_step
        ])
        
        # 整体平均
        all_errors = np.concatenate([
            np.array(errors) for errors in errors_by_step if errors
        ])
        mae = np.mean(np.abs(all_errors)) * self.unit_conversion
        rmse = np.sqrt(np.mean(all_errors ** 2)) * self.unit_conversion
        
        return MetricsResult(
            mae=mae,
            rmse=rmse,
            mae_by_step=mae_by_step,
            rmse_by_step=rmse_by_step
        )
    
    def compute_ae_by_step(
        self,
        forecasts: list[np.ndarray],
        truths: list[np.ndarray],
        pred_len: int
    ) -> np.ndarray:
        """
        计算每个起报点每天的绝对误差，然后对所有起报点取平均。
        
        这是你最关心的输出格式：
        - 输入: N个起报点的预报和真实值
        - 输出: (pred_len,) 的MAE数组，MAE[i] 表示第 i+1 天的平均MAE
        
        Args:
            forecasts: 多个起报点的预报结果列表
            truths: 对应的真实值列表
            pred_len: 预报长度
            
        Returns:
            np.ndarray: 形状为 (pred_len,) 的MAE数组，MAE[i] 是第i+1天的平均MAE
        """
        # 收集每个起报点在每个步长的绝对误差
        ae_by_step = [[] for _ in range(pred_len)]
        
        for forecast, truth in zip(forecasts, truths):
            min_len = min(len(forecast), len(truth), pred_len)
            for i in range(min_len):
                ae = np.abs(forecast[i] - truth[i]) * self.unit_conversion
                ae_by_step[i].append(ae)
        
        # 对每个步长的所有起报点取平均
        mae_by_day = np.array([
            np.mean(ae_list) if ae_list else np.nan
            for ae_list in ae_by_step
        ])
        
        return mae_by_day
    
    def aggregate_multi_seed(
        self,
        results_by_seed: list[MetricsResult]
    ) -> MetricsResult:
        """
        聚合多随机种子的结果（取平均）。
        
        Args:
            results_by_seed: 各随机种子的评估结果列表
            
        Returns:
            MetricsResult: 聚合后的结果
        """
        maes = [r.mae for r in results_by_seed]
        rmses = [r.rmse for r in results_by_seed]
        
        result = MetricsResult(
            mae=np.mean(maes),
            rmse=np.mean(rmses)
        )
        
        # 如果有分步误差，也聚合
        if results_by_seed[0].mae_by_step is not None:
            mae_by_steps = [r.mae_by_step for r in results_by_seed]
            rmse_by_steps = [r.rmse_by_step for r in results_by_seed]
            result.mae_by_step = np.mean(mae_by_steps, axis=0)
            result.rmse_by_step = np.mean(rmse_by_steps, axis=0)
        
        return result
    
    def print_report(self, metrics: MetricsResult, title: str = "预报误差报告") -> None:
        """
        打印误差报告。
        
        Args:
            metrics: 误差指标结果
            title: 报告标题
        """
        print(f"\n{'='*50}")
        print(f"{title}")
        print(f"{'='*50}")
        print(f"MAE:  {metrics.mae:.4f}")
        print(f"RMSE: {metrics.rmse:.4f}")
        if metrics.mape is not None:
            print(f"MAPE: {metrics.mape:.2f}%")
        print(f"{'='*50}\n")
