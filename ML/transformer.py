from ML.base import BaseDeepModel
from typing import Dict, Any, Union
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class ModelConfig:
    input_dim: int
    hidden_dim: int = 64
    num_layers: int = 2
    output_dim: int = 1
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    sequence_length: int = 60

    # Transformer specific parameters
    num_heads: int = 8
    dim_feedforward: int = 256

    @classmethod
    def get_sequence_length_for_timeframe(cls, timeframe: str) -> int:
        """
        Возвращает рекомендуемую длину последовательности для таймфрейма
        """
        sequence_mapping = {
            '1m': 60,
            '1h': 24,
            '4h': 60,
            '1d': 30,
            '1w': 12
        }
        return sequence_mapping.get(timeframe, 60)


class TransformerModel(BaseDeepModel):
    """Transformer модель для прогнозування часових рядів"""

    def __init__(self, config: Union[ModelConfig, None] = None,
                 input_dim: int = None, hidden_dim: int = None, num_layers: int = None,
                 output_dim: int = None, dropout: float = None, n_heads: int = None,
                 dim_feedforward: int = None):

        # Якщо передано config, використовуємо його параметри
        if config is not None:
            input_dim = config.input_dim
            hidden_dim = config.hidden_dim
            num_layers = config.num_layers
            output_dim = config.output_dim
            dropout = config.dropout
            n_heads = config.num_heads
            dim_feedforward = config.dim_feedforward
        else:
            # Використовуємо значення за замовчуванням, якщо не передано
            input_dim = input_dim or 1
            hidden_dim = hidden_dim or 64
            num_layers = num_layers or 2
            output_dim = output_dim or 1
            dropout = dropout if dropout is not None else 0.2
            n_heads = n_heads or 8
            dim_feedforward = dim_feedforward or 256

        super().__init__(input_dim, hidden_dim, num_layers, output_dim, dropout)

        self.config = config
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward

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
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Dropout шар
        self.dropout_layer = nn.Dropout(dropout)

        # Вихідний шар
        self.fc = nn.Linear(hidden_dim, output_dim)

    @classmethod
    def from_config(cls, config: ModelConfig):
        """Створення моделі з конфігурації"""
        return cls(config=config)

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

    def get_config(self) -> ModelConfig:
        """Повертає конфігурацію моделі"""
        return self.config

    def get_training_params(self) -> Dict[str, Any]:
        """Повертає параметри для тренування з конфігурації"""
        if self.config is None:
            return {}

        return {
            'learning_rate': self.config.learning_rate,
            'batch_size': self.config.batch_size,
            'epochs': self.config.epochs,
            'sequence_length': self.config.sequence_length
        }

    def get_transformer_specific_info(self) -> Dict[str, Any]:
        """Transformer-специфічна інформація"""
        info = {
            'model_type': 'Transformer',
            'n_heads': self.n_heads,
            'dim_feedforward': self.dim_feedforward,
            'has_cell_state': False,
            'uses_attention': True,
            'transformer_parameters': sum(p.numel() for p in self.transformer.parameters()),
            'fc_parameters': sum(p.numel() for p in self.fc.parameters()),
            'input_projection_parameters': sum(p.numel() for p in self.input_projection.parameters()),
            'pos_encoding_size': self.pos_encoding.numel()
        }

        # Додаємо інформацію з конфігурації, якщо вона є
        if self.config:
            info.update({
                'config_learning_rate': self.config.learning_rate,
                'config_batch_size': self.config.batch_size,
                'config_epochs': self.config.epochs,
                'config_sequence_length': self.config.sequence_length,
                'config_num_heads': self.config.num_heads,
                'config_dim_feedforward': self.config.dim_feedforward
            })

        return info

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Отримання ваг уваги (для аналізу)"""
        # Цей метод може бути корисним для інтерпретації моделі
        with torch.no_grad():
            batch_size, seq_len, _ = x.shape
            x = self.input_projection(x)
            x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)

            # Для отримання ваг уваги потрібно модифікувати transformer
            # Це спрощена версія - в реальності потрібен доступ до внутрішніх шарів
            return torch.ones(batch_size, self.n_heads, seq_len, seq_len)