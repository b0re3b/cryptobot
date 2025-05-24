from ML.base import BaseDeepModel
from typing import Dict, Any, Union, List, Tuple
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


class TransformerEncoderLayerWithAttention(nn.Module):
    """Custom TransformerEncoderLayer that returns attention weights"""

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, return_attention=False):
        # Self-attention
        attn_output, attn_weights = self.self_attn(src, src, src, need_weights=return_attention)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        # Feed forward
        ff_output = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        if return_attention:
            return src, attn_weights
        return src


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

        # Створюємо custom transformer layers для отримання ваг уваги
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayerWithAttention(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # Dropout шар
        self.dropout_layer = nn.Dropout(dropout)

        # Вихідний шар
        self.fc = nn.Linear(hidden_dim, output_dim)

    @classmethod
    def from_config(cls, config: ModelConfig):
        """Створення моделі з конфігурації"""
        return cls(config=config)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Union[
        torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Прямий прохід через Transformer"""
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape

        # Проекція входу
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_dim)

        # Додаємо позиційне кодування
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)

        # Прохід через transformer layers
        attention_weights = []
        for layer in self.transformer_layers:
            if return_attention:
                x, attn_weights = layer(x, return_attention=True)
                attention_weights.append(attn_weights)
            else:
                x = layer(x, return_attention=False)

        # Використовуємо останній вихід
        last_output = x[:, -1, :]  # (batch_size, hidden_dim)

        # Dropout
        dropped_output = self.dropout_layer(last_output)

        # Вихідний шар
        output = self.fc(dropped_output)  # (batch_size, output_dim)

        if return_attention:
            return output, attention_weights
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
            'transformer_parameters': sum(p.numel() for p in self.transformer_layers.parameters()),
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

    def get_attention_weights(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Отримання ваг уваги для всіх шарів transformer

        Args:
            x: Вхідний тензор (batch_size, seq_len, input_dim)

        Returns:
            List[torch.Tensor]: Список ваг уваги для кожного шару
                               Кожен тензор має розмір (batch_size, n_heads, seq_len, seq_len)
        """
        self.eval()  # Переключаємо в режим оцінки
        with torch.no_grad():
            _, attention_weights = self.forward(x, return_attention=True)
            return attention_weights

    def visualize_attention(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        Візуалізація ваг уваги для конкретного шару

        Args:
            x: Вхідний тензор
            layer_idx: Індекс шару (-1 для останнього шару)

        Returns:
            torch.Tensor: Ваги уваги для вказаного шару
        """
        attention_weights = self.get_attention_weights(x)
        if not attention_weights:
            raise ValueError("Не вдалося отримати ваги уваги")

        return attention_weights[layer_idx]

    def get_averaged_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Отримання усереднених ваг уваги по всіх головах та шарах

        Args:
            x: Вхідний тензор

        Returns:
            torch.Tensor: Усереднені ваги уваги (batch_size, seq_len, seq_len)
        """
        attention_weights = self.get_attention_weights(x)
        if not attention_weights:
            raise ValueError("Не вдалося отримати ваги уваги")

        # Усереднюємо по всіх шарах та головах
        all_layers_attention = torch.stack(attention_weights)  # (num_layers, batch_size, n_heads, seq_len, seq_len)
        averaged_attention = all_layers_attention.mean(dim=[0, 2])  # (batch_size, seq_len, seq_len)

        return averaged_attention
