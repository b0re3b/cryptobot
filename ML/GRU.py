from typing import Dict, Any, Union
import torch
import torch.nn as nn
from dataclasses import dataclass
from ML.base import BaseDeepModel


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


# ==================== GRU МОДЕЛЬ ====================
class GRUModel(BaseDeepModel):
    """
    GRU модель для прогнозування часових рядів криптовалют.

    Параметри
    ---------
    config : ModelConfig | None
        Об'єкт конфігурації моделі. Якщо передано, ігнорує інші параметри конструктора.
    input_dim : int | None
        Розмір вхідного шару.
    hidden_dim : int | None
        Розмір прихованого шару GRU.
    num_layers : int | None
        Кількість шарів GRU.
    output_dim : int | None
        Розмір вихідного шару.
    dropout : float | None
        Вірогідність дроп-ауту.
    bidirectional : bool
        Чи є GRU двонапрямленою мережею.

    Методи
    -------
    from_config(config, bidirectional=False)
        Створює екземпляр моделі на основі конфігурації.
    _init_weights()
        Ініціалізує ваги моделі.
    forward(x)
        Прямий прохід через GRU модель.
    init_hidden(batch_size)
        Ініціалізує прихований стан GRU.
    get_model_type()
        Повертає тип моделі.
    get_config()
        Повертає конфігурацію моделі.
    get_training_params()
        Повертає параметри навчання з конфігурації.
    get_gru_specific_info()
        Повертає специфічну інформацію по GRU моделі.
    """

    def __init__(self, config: Union[ModelConfig, None] = None,
                 input_dim: int = None, hidden_dim: int = None, num_layers: int = None,
                 output_dim: int = None, dropout: float = None, bidirectional: bool = False):

        # Якщо передано config, використовуємо його параметри
        if config is not None:
            input_dim = config.input_dim
            hidden_dim = config.hidden_dim
            num_layers = config.num_layers
            output_dim = config.output_dim
            dropout = config.dropout
        else:
            # Використовуємо значення за замовчуванням, якщо не передано
            input_dim = input_dim or 1
            hidden_dim = hidden_dim or 64
            num_layers = num_layers or 2
            output_dim = output_dim or 1
            dropout = dropout if dropout is not None else 0.2

        super().__init__(input_dim, hidden_dim, num_layers, output_dim, dropout)

        self.config = config
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

    @classmethod
    def from_config(cls, config: ModelConfig, bidirectional: bool = False):
        """Створення моделі з конфігурації"""
        return cls(config=config, bidirectional=bidirectional)

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

    def get_gru_specific_info(self) -> Dict[str, Any]:
        """GRU-специфічна інформація"""
        info = {
            'model_type': 'GRU',
            'bidirectional': self.bidirectional,
            'has_cell_state': False,
            'num_directions': 2 if self.bidirectional else 1,
            'gru_parameters': sum(p.numel() for p in self.gru.parameters()),
            'fc_parameters': sum(p.numel() for p in self.fc.parameters())
        }

        # Додаємо інформацію з конфігурації, якщо вона є
        if self.config:
            info.update({
                'config_learning_rate': self.config.learning_rate,
                'config_batch_size': self.config.batch_size,
                'config_epochs': self.config.epochs,
                'config_sequence_length': self.config.sequence_length
            })

        return info