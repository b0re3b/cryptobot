from typing import Dict, Any
import torch
import torch.nn as nn
from ML.base import BaseDeepModel

# ==================== GRU МОДЕЛЬ ====================
class GRUModel(BaseDeepModel):
    """GRU модель для прогнозування часових рядів криптовалут"""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 output_dim: int, dropout: float = 0.2):
        super().__init__(input_dim, hidden_dim, num_layers, output_dim, dropout)

        # GRU шари
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Dropout шар
        self.dropout_layer = nn.Dropout(dropout)

        # Повнозв'язний шар для виходу
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямий прохід через GRU"""
        # x shape: (batch_size, seq_len, input_dim)
        pass

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """Ініціалізація прихованого стану GRU"""
        pass

    def get_gru_specific_info(self) -> Dict[str, Any]:
        """GRU-специфічна інформація"""
        return {
            'model_type': 'GRU',
            'bidirectional': False,
            'has_cell_state': False
        }