from ML.base import BaseDeepModel
from typing import Dict, Any
import torch
import torch.nn as nn

class TransformerModel(BaseDeepModel):
    """Transformer модель для прогнозування часових рядів"""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 output_dim: int, dropout: float = 0.2, n_heads: int = 8):
        super().__init__(input_dim, hidden_dim, num_layers, output_dim, dropout)

        self.n_heads = n_heads

        # Перевіряємо, чи hidden_dim ділиться на n_heads
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) повинен ділитися на n_heads ({n_heads})")

        # Позиційне кодування
        self.pos_encoding = nn.Parameter(torch.randn(1000, hidden_dim))

        # Вхідна проекція
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Dropout шар
        self.dropout_layer = nn.Dropout(dropout)

        # Вихідний шар
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямий прохід через Transformer"""
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape

        # Проекція входу
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_dim)

        # Додаємо позиційне кодування
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)

        # Прохід через transformer
        transformer_out = self.transformer(x)  # (batch_size, seq_len, hidden_dim)

        # Використовуємо останній вихід
        last_output = transformer_out[:, -1, :]  # (batch_size, hidden_dim)

        # Dropout
        dropped_output = self.dropout_layer(last_output)

        # Вихідний шар
        output = self.fc(dropped_output)  # (batch_size, output_dim)

        return output

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """Для Transformer не потрібен прихований стан"""
        return torch.tensor([])

    def get_model_type(self) -> str:
        return "Transformer"

    def get_transformer_specific_info(self) -> Dict[str, Any]:
        """Transformer-специфічна інформація"""
        return {
            'model_type': 'Transformer',
            'n_heads': self.n_heads,
            'has_cell_state': False,
            'uses_attention': True,
            'transformer_parameters': sum(p.numel() for p in self.transformer.parameters()),
            'fc_parameters': sum(p.numel() for p in self.fc.parameters())
        }