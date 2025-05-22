from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional, Callable
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from datetime import datetime
import os

# ==================== БАЗОВИЙ КЛАС МОДЕЛІ ====================
class BaseDeepModel(nn.Module, ABC):
    """Базовий клас для всіх моделей глибокого навчання"""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямий прохід через модель"""
        pass

    @abstractmethod
    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """Ініціалізація прихованого стану"""
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """Отримання інформації про модель"""
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'output_dim': self.output_dim,
            'dropout': self.dropout,
            'total_parameters': self.count_parameters()
        }

    def count_parameters(self) -> int:
        """Підрахунок кількості параметрів моделі"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)