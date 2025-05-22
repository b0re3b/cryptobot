from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from ML.base import BaseDeepModel


# ==================== LSTM МОДЕЛЬ ====================
class LSTMModel(BaseDeepModel):
    """LSTM модель для прогнозування часових рядів криптовалют"""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 output_dim: int, dropout: float = 0.2):
        super().__init__(input_dim, hidden_dim, num_layers, output_dim, dropout)

        # LSTM шари
        self.lstm = nn.LSTM(
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
        """Прямий прохід через LSTM"""
        # x shape: (batch_size, seq_len, input_dim)
        batch_size = x.size(0)

        # Ініціалізація прихованого стану
        h0, c0 = self.init_hidden(batch_size)

        # Прохід через LSTM
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))

        # Беремо останній вихід з послідовності
        # lstm_out shape: (batch_size, seq_len, hidden_dim)
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim)

        # Застосовуємо dropout
        dropped_output = self.dropout_layer(last_output)

        # Прохід через повнозв'язний шар
        output = self.fc(dropped_output)  # (batch_size, output_dim)

        return output

    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ініціалізація прихованого стану LSTM"""
        device = next(self.parameters()).device

        # Ініціалізація прихованого стану (h0) та стану комірки (c0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)

        return h0, c0

    def get_lstm_specific_info(self) -> Dict[str, Any]:
        """LSTM-специфічна інформація"""
        return {
            'model_type': 'LSTM',
            'bidirectional': False,
            'has_cell_state': True,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'output_dim': self.output_dim,
            'dropout': self.dropout
        }