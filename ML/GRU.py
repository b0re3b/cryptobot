from typing import Dict, Any
import torch
import torch.nn as nn
from ML.base import BaseDeepModel


# ==================== GRU МОДЕЛЬ ====================
class GRUModel(BaseDeepModel):
    """GRU модель для прогнозування часових рядів криптовалют"""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 output_dim: int, dropout: float = 0.2, bidirectional: bool = False):
        super().__init__(input_dim, hidden_dim, num_layers, output_dim, dropout)

        self.bidirectional = bidirectional

        # GRU шари
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Розмір виходу GRU залежить від bidirectional
        gru_output_size = hidden_dim * 2 if bidirectional else hidden_dim

        # Dropout шар
        self.dropout_layer = nn.Dropout(dropout)

        # Повнозв'язний шар для виходу
        self.fc = nn.Linear(gru_output_size, output_dim)

        # Ініціалізація ваг
        self._init_weights()

    def _init_weights(self):
        """Ініціалізація ваг моделі"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямий прохід через GRU"""
        # x shape: (batch_size, seq_len, input_dim)
        batch_size = x.size(0)

        # Ініціалізація прихованого стану
        h0 = self.init_hidden(batch_size)

        # Прохід через GRU
        gru_out, _ = self.gru(x, h0)

        # Використовуємо останній вихід послідовності
        # gru_out shape: (batch_size, seq_len, hidden_dim * num_directions)
        last_output = gru_out[:, -1, :]  # (batch_size, hidden_dim * num_directions)

        # Застосовуємо dropout
        dropped_out = self.dropout_layer(last_output)

        # Фінальний повнозв'язний шар
        output = self.fc(dropped_out)  # (batch_size, output_dim)

        return output

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """Ініціалізація прихованого стану GRU"""
        device = next(self.parameters()).device
        num_directions = 2 if self.bidirectional else 1

        h0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_dim,
            device=device
        )

        return h0

    def get_model_type(self) -> str:
        return "GRU"

    def get_gru_specific_info(self) -> Dict[str, Any]:
        """GRU-специфічна інформація"""
        return {
            'model_type': 'GRU',
            'bidirectional': self.bidirectional,
            'has_cell_state': False,
            'num_directions': 2 if self.bidirectional else 1,
            'gru_parameters': sum(p.numel() for p in self.gru.parameters()),
            'fc_parameters': sum(p.numel() for p in self.fc.parameters())
        }
