from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Union
import torch
import torch.nn as nn
from datetime import datetime


class BaseDeepModel(nn.Module, ABC):
    """
    Абстрактний базовий клас для моделей глибокого навчання.

    Цей клас визначає загальну структуру моделей на базі PyTorch, які можна
    адаптувати для задач передбачення часових рядів або інших задач, що потребують
    рекурентних або глибоких нейронних мереж. Містить стандартні методи для:
    - ініціалізації параметрів,
    - збереження/завантаження моделі,
    - управління параметрами,
    - отримання характеристик моделі.

    Атрибути
    --------
    input_dim : int
        Розмір вхідного шару.
    hidden_dim : int
        Розмір прихованого шару (наприклад, у LSTM/GRU).
    num_layers : int
        Кількість шарів у мережі.
    output_dim : int
        Розмір вихідного шару.
    dropout : float
        Ймовірність відсічення нейронів (Dropout).
    training_history : list
        Історія навчання (список метрик або втрат).
    validation_history : list
        Історія валідації (список метрик або втрат).

    Методи
    -------
    forward(x)
        Пряме проходження через модель (абстрактний метод).
    init_hidden(batch_size)
        Ініціалізація прихованих станів моделі (абстрактний метод).
    get_model_type()
        Повертає тип моделі (наприклад, "LSTM", "GRU") (абстрактний метод).
    get_model_info()
        Повертає словник з параметрами та метаінформацією про модель.
    count_parameters()
        Повертає загальну кількість параметрів у моделі.
    count_trainable_parameters()
        Повертає кількість параметрів, які підлягають навчанню.
    freeze_layers(layer_names)
        Заморожує всі або вказані шари моделі (вимикає їх оновлення).
    unfreeze_layers(layer_names)
        Розморожує всі або вказані шари моделі (дозволяє оновлення).
    save_model(filepath, additional_info)
        Зберігає модель у файл разом із історією тренування, конфігурацією та метаданими.
    load_model(filepath, **kwargs)
        Завантажує модель з файлу, відновлює стан та гіперпараметри.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout

        # Метрики для відстеження
        self.training_history = []
        self.validation_history = []

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямий прохід через модель"""
        pass

    @abstractmethod
    def init_hidden(self, batch_size: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Ініціалізація прихованого стану (для RNN моделей)"""
        pass

    @abstractmethod
    def get_model_type(self) -> str:
        """Повертає тип моделі"""
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """Отримання інформації про модель"""
        return {
            'model_type': self.get_model_type(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'output_dim': self.output_dim,
            'dropout': self.dropout,
            'total_parameters': self.count_parameters(),
            'trainable_parameters': self.count_trainable_parameters()
        }

    def count_parameters(self) -> int:
        """Підрахунок загальної кількості параметрів моделі"""
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        """Підрахунок кількості параметрів, що навчаються"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_layers(self, layer_names: List[str] = None):
        """Заморожування шарів моделі"""
        if layer_names is None:
            # Заморожуємо всі параметри
            for param in self.parameters():
                param.requires_grad = False
        else:
            # Заморожуємо конкретні шари
            for name, param in self.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = False

    def unfreeze_layers(self, layer_names: List[str] = None):
        """Розморожування шарів моделі"""
        if layer_names is None:
            # Розморожуємо всі параметри
            for param in self.parameters():
                param.requires_grad = True
        else:
            # Розморожуємо конкретні шари
            for name, param in self.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = True

    def save_model(self, filepath: str, additional_info: Dict[str, Any] = None):
        """Збереження моделі"""
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_info': self.get_model_info(),
            'training_history': self.training_history,
            'validation_history': self.validation_history,
            'timestamp': datetime.now().isoformat()
        }

        if additional_info:
            save_dict.update(additional_info)

        torch.save(save_dict, filepath)

    @classmethod
    def load_model(cls, filepath: str, **kwargs):
        """Завантаження моделі"""
        checkpoint = torch.load(filepath, map_location='cpu')
        model_info = checkpoint['model_info']

        # Створюємо екземпляр моделі
        model = cls(
            input_dim=model_info['input_dim'],
            hidden_dim=model_info['hidden_dim'],
            num_layers=model_info['num_layers'],
            output_dim=model_info['output_dim'],
            dropout=model_info.get('dropout', 0.2),
            **kwargs
        )

        # Завантажуємо стан моделі
        model.load_state_dict(checkpoint['model_state_dict'])
        model.training_history = checkpoint.get('training_history', [])
        model.validation_history = checkpoint.get('validation_history', [])

        return model, checkpoint