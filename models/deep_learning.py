import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import os
import logging
from datetime import datetime

# Імпорт з інших модулів проекту
from data.db import DatabaseManager (
    get_btc_lstm_sequence, get_eth_lstm_sequence, get_sol_lstm_sequence,
    save_prediction,  # Збереження прогнозів моделі
    save_ml_model_metrics,  # Збереження метрик ефективності моделі
    save_ml_model,  # Збереження інформації про модель
    save_ml_sequence_data,  # Збереження послідовностей для LSTM/GRU
    save_technical_indicator,  # Збереження технічних індикаторів
    update_prediction_actual_value  # Оновлення прогнозів з фактичними значеннями
)
from analysis.trend_detection import prepare_ml_trend_features  # Для отримання ознак тренду
from analysis.volatility_analysis import prepare_volatility_features_for_ml  # Для отримання ознак волатильності
from cyclefeatures.crypto_cycles import prepare_cycle_ml_features  # Для отримання циклічних ознак
from featureengineering.feature_engineering import prepare_features_pipeline  # Для підготовки всіх ознак
from utils.logger import get_logger
from utils.config import get_config

logger = get_logger(__name__)
config = get_config()


class LSTMModel(nn.Module):
    """LSTM модель для прогнозування часових рядів криптовалют"""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int, dropout: float = 0.2):
        """Ініціалізація архітектури LSTM моделі"""
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM шар для обробки послідовностей
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        # Повноз'єднаний шар для прогнозування
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Ініціалізація прихованого стану
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Прямий прохід через LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Використовуємо лише вихід останнього часового кроку
        out = self.fc(out[:, -1, :])
        return out


class GRUModel(nn.Module):
    """GRU модель для прогнозування часових рядів криптовалют"""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int, dropout: float = 0.2):
        """Ініціалізація архітектури GRU моделі"""
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GRU шар для обробки послідовностей
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        # Повноз'єднаний шар для прогнозування
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Ініціалізація прихованого стану
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Прямий прохід через GRU
        out, _ = self.gru(x, h0)

        # Використовуємо лише вихід останнього часового кроку
        out = self.fc(out[:, -1, :])
        return out


class DeepLearning:
    """
    Клас для роботи з глибокими нейронними мережами для прогнозування криптовалют
    Підтримує LSTM та GRU моделі для BTC, ETH та SOL на різних таймфреймах
    """

    SYMBOLS = ['BTC', 'ETH', 'SOL']
    TIMEFRAMES = ['1m', '1h', '4h', '1d', '1w']

    def __init__(self):
        """
        Ініціалізація класу DeepLearning
        Створення структур для зберігання моделей, їх конфігурацій та метрик
        """
        self.models = {}  # Словник для зберігання навчених моделей
        self.model_configs = {}  # Конфігурації моделей
        self.model_metrics = {}  # Метрики ефективності моделей
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Шляхи для збереження моделей
        self.models_dir = os.path.join(config.get('paths', 'models_dir'), 'deep_learning')
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def _get_data_loader(self, symbol: str, timeframe: str, model_type: str) -> Callable:
        """
        Отримати функцію для завантаження даних в залежності від символу

        Args:
            symbol: Символ криптовалюти ('BTC', 'ETH', 'SOL')
            timeframe: Часовий інтервал ('1m', '1h', '4h', '1d', '1w')
            model_type: Тип моделі ('lstm', 'gru')

        Returns:
            Callable: Функція для завантаження даних
        """
        # Використовуємо методи з модуля data.db для отримання даних послідовностей
        if symbol == 'BTC':
            return lambda: get_btc_lstm_sequence(timeframe)
        elif symbol == 'ETH':
            return lambda: get_eth_lstm_sequence(timeframe)
        elif symbol == 'SOL':
            return lambda: get_sol_lstm_sequence(timeframe)
        else:
            raise ValueError(f"Непідтримуваний символ: {symbol}")

    def _prepare_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Підготовка ознак для навчання моделі

        Args:
            df: DataFrame з даними
            symbol: Символ криптовалюти

        Returns:
            pd.DataFrame: DataFrame з підготовленими ознаками
        """
        # Використовуємо методи з різних модулів для підготовки ознак
        # 1. Отримуємо ознаки тренду
        trend_features = prepare_ml_trend_features(df, symbol)

        # 2. Отримуємо ознаки волатильності
        volatility_features = prepare_volatility_features_for_ml(df, symbol)

        # 3. Отримуємо циклічні ознаки
        cycle_features = prepare_cycle_ml_features(df, symbol)

        # 4. Об'єднуємо всі ознаки за допомогою пайплайну
        final_features = prepare_features_pipeline(
            df,
            trend_features=trend_features,
            volatility_features=volatility_features,
            cycle_features=cycle_features,
            symbol=symbol
        )

        # Після підготовки ознак, зберігаємо результати в БД за допомогою відповідних методів
        # save_technical_indicator можна використати для збереження технічних індикаторів
        # save_ml_sequence_data для збереження підготовлених послідовностей

        return final_features

    def _build_model(self, model_type: str, input_dim: int, hidden_dim: int = 64,
                     num_layers: int = 2, output_dim: int = 1) -> nn.Module:
        """
        Створення моделі відповідного типу

        Args:
            model_type: Тип моделі ('lstm' або 'gru')
            input_dim: Розмірність вхідних даних
            hidden_dim: Розмірність прихованого шару
            num_layers: Кількість шарів
            output_dim: Розмірність вихідних даних

        Returns:
            nn.Module: Створена модель
        """
        if model_type.lower() == 'lstm':
            return LSTMModel(input_dim, hidden_dim, num_layers, output_dim).to(self.device)
        elif model_type.lower() == 'gru':
            return GRUModel(input_dim, hidden_dim, num_layers, output_dim).to(self.device)
        else:
            raise ValueError(f"Непідтримуваний тип моделі: {model_type}")

    def train_model(self, symbol: str, timeframe: str, model_type: str,
                    epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001,
                    hidden_dim: int = 64, num_layers: int = 2,
                    validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Навчання моделі для вказаного символу та таймфрейму

        Args:
            symbol: Символ криптовалюти ('BTC', 'ETH', 'SOL')
            timeframe: Часовий інтервал ('1m', '1h', '4h', '1d', '1w')
            model_type: Тип моделі ('lstm' або 'gru')
            epochs: Кількість епох навчання
            batch_size: Розмір батчу
            learning_rate: Швидкість навчання
            hidden_dim: Розмірність прихованого шару
            num_layers: Кількість шарів
            validation_split: Частка даних для валідації

        Returns:
            Dict: Історія навчання та метрики
        """
        # Перевірка коректності вхідних параметрів
        self._validate_inputs(symbol, timeframe, model_type)

        # Завантаження даних та підготовка ознак
        data_loader = self._get_data_loader(symbol, timeframe, model_type)
        df = data_loader()
        processed_data = self._prepare_features(df, symbol)

        # Підготовка даних для навчання (нормалізація, розділення на тренувальну та валідаційну вибірки)
        # ...

        # Створення моделі
        input_dim = len(processed_data.columns) - 1  # Не враховуємо цільовий стовпець
        model = self._build_model(model_type, input_dim, hidden_dim, num_layers)

        # Налаштування оптимізатора та функції втрат
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Навчання моделі
        # ...

        # Після навчання зберігаємо модель та метрики
        model_key = self._create_model_key(symbol, timeframe, model_type)
        self.models[model_key] = model

        # Зберігаємо конфігурацію моделі
        self.model_configs[model_key] = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'output_dim': 1,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }

        # Обчислюємо та зберігаємо метрики
        metrics = self.evaluate_model(symbol, timeframe, model_type)
        self.model_metrics[model_key] = metrics

        # Зберігаємо модель та метрики в БД
        # Використовуємо методи з модуля data.db
        save_ml_model(symbol, timeframe, model_type, self.model_configs[model_key])
        save_ml_model_metrics(symbol, timeframe, model_type, metrics)

        return {
            'config': self.model_configs[model_key],
            'metrics': metrics
        }

    def online_learning(self, symbol: str, timeframe: str, model_type: str,
                        new_data: pd.DataFrame, epochs: int = 10,
                        learning_rate: float = 0.0005) -> Dict[str, Any]:
        """
        Онлайн-навчання існуючої моделі на нових даних

        Args:
            symbol: Символ криптовалюти ('BTC', 'ETH', 'SOL')
            timeframe: Часовий інтервал ('1m', '1h', '4h', '1d', '1w')
            model_type: Тип моделі ('lstm' або 'gru')
            new_data: Нові дані для навчання
            epochs: Кількість епох навчання
            learning_rate: Швидкість навчання

        Returns:
            Dict: Результати донавчання
        """
        # Перевірка наявності моделі
        model_key = self._create_model_key(symbol, timeframe, model_type)
        if model_key not in self.models:
            if not self.load_model(symbol, timeframe, model_type):
                raise ValueError(f"Модель {model_key} не знайдена")

        # Підготовка нових даних
        processed_data = self._prepare_features(new_data, symbol)

        # Підготовка даних для навчання
        # ...

        # Донавчання моделі
        model = self.models[model_key]
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # ...

        # Оновлення метрик
        metrics = self.evaluate_model(symbol, timeframe, model_type, test_data=new_data)
        self.model_metrics[model_key] = metrics

        # Зберігаємо оновлену модель та метрики
        save_ml_model(symbol, timeframe, model_type, self.model_configs[model_key])
        save_ml_model_metrics(symbol, timeframe, model_type, metrics)

        return {
            'metrics': metrics
        }

    def predict(self, symbol: str, timeframe: str, model_type: str,
                steps_ahead: int = 1, input_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Прогнозування на основі навченої моделі

        Args:
            symbol: Символ криптовалюти ('BTC', 'ETH', 'SOL')
            timeframe: Часовий інтервал ('1m', '1h', '4h', '1d', '1w')
            model_type: Тип моделі ('lstm' або 'gru')
            steps_ahead: Кількість кроків для прогнозування вперед
            input_data: Вхідні дані для прогнозування (якщо None, використовуються останні дані)

        Returns:
            np.ndarray: Прогнозовані значення
        """
        # Перевірка наявності моделі
        model_key = self._create_model_key(symbol, timeframe, model_type)
        if model_key not in self.models:
            if not self.load_model(symbol, timeframe, model_type):
                raise ValueError(f"Модель {model_key} не знайдена")

        # Якщо вхідні дані не надані, завантажуємо останні дані
        if input_data is None:
            data_loader = self._get_data_loader(symbol, timeframe, model_type)
            input_data = data_loader()

        # Підготовка даних для прогнозування
        processed_data = self._prepare_features(input_data, symbol)

        # Прогнозування
        model = self.models[model_key]
        model.eval()

        with torch.no_grad():
            # Підготовка вхідних даних для моделі
            # ...

            # Прогнозування на кілька кроків вперед
            predictions = []
            # ...

        # Зберігаємо прогнози в БД
        timestamp = datetime.now()
        for i, pred in enumerate(predictions):
            # Використовуємо метод save_prediction для збереження прогнозу
            save_prediction(
                symbol=symbol,
                timeframe=timeframe,
                model_type=model_type,
                prediction_value=float(pred),
                prediction_timestamp=timestamp,
                steps_ahead=i + 1
            )

        return np.array(predictions)

    def get_model_metrics(self, symbol: str, timeframe: str, model_type: str) -> Dict[str, float]:
        """
        Отримання метрик ефективності моделі

        Args:
            symbol: Символ криптовалюти ('BTC', 'ETH', 'SOL')
            timeframe: Часовий інтервал ('1m', '1h', '4h', '1d', '1w')
            model_type: Тип моделі ('lstm' або 'gru')

        Returns:
            Dict: Метрики ефективності моделі
        """
        model_key = self._create_model_key(symbol, timeframe, model_type)
        if model_key in self.model_metrics:
            return self.model_metrics[model_key]
        else:
            # Якщо метрики не знайдено, спробуємо завантажити їх
            # Тут можна додати метод для завантаження метрик з БД
            # ...
            return {}

    def save_model(self, symbol: str, timeframe: str, model_type: str) -> str:
        """
        Збереження навченої моделі на диск

        Args:
            symbol: Символ криптовалюти ('BTC', 'ETH', 'SOL')
            timeframe: Часовий інтервал ('1m', '1h', '4h', '1d', '1w')
            model_type: Тип моделі ('lstm' або 'gru')

        Returns:
            str: Шлях до збереженої моделі
        """
        model_key = self._create_model_key(symbol, timeframe, model_type)
        if model_key not in self.models:
            raise ValueError(f"Модель {model_key} не знайдена")

        # Створюємо директорію для моделі, якщо вона не існує
        model_dir = os.path.join(self.models_dir, symbol, timeframe)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Шлях до файлу моделі
        model_path = os.path.join(model_dir, f"{model_type.lower()}_model.pth")

        # Зберігаємо модель
        torch.save({
            'model_state_dict': self.models[model_key].state_dict(),
            'config': self.model_configs[model_key],
            'metrics': self.model_metrics.get(model_key, {})
        }, model_path)

        # Зберігаємо модель в БД
        save_ml_model(
            symbol=symbol,
            timeframe=timeframe,
            model_type=model_type,
            model_config=self.model_configs[model_key],
            model_path=model_path
        )

        return model_path

    def load_model(self, symbol: str, timeframe: str, model_type: str) -> bool:
        """
        Завантаження моделі з диску

        Args:
            symbol: Символ криптовалюти ('BTC', 'ETH', 'SOL')
            timeframe: Часовий інтервал ('1m', '1h', '4h', '1d', '1w')
            model_type: Тип моделі ('lstm' або 'gru')

        Returns:
            bool: True, якщо модель успішно завантажена
        """
        model_key = self._create_model_key(symbol, timeframe, model_type)

        # Шлях до файлу моделі
        model_dir = os.path.join(self.models_dir, symbol, timeframe)
        model_path = os.path.join(model_dir, f"{model_type.lower()}_model.pth")

        if not os.path.exists(model_path):
            logger.warning(f"Модель {model_key} не знайдена за шляхом {model_path}")
            return False

        try:
            # Завантаження моделі
            checkpoint = torch.load(model_path, map_location=self.device)

            # Отримання конфігурації та створення моделі
            config = checkpoint['config']
            model = self._build_model(
                model_type,
                config['input_dim'],
                config['hidden_dim'],
                config['num_layers'],
                config['output_dim']
            )

            # Завантаження ваг моделі
            model.load_state_dict(checkpoint['model_state_dict'])

            # Збереження моделі та її конфігурації
            self.models[model_key] = model
            self.model_configs[model_key] = config
            self.model_metrics[model_key] = checkpoint.get('metrics', {})

            logger.info(f"Модель {model_key} успішно завантажена")
            return True

        except Exception as e:
            logger.error(f"Помилка при завантаженні моделі {model_key}: {str(e)}")
            return False

    def evaluate_model(self, symbol: str, timeframe: str, model_type: str,
                       test_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Оцінка ефективності моделі на тестових даних

        Args:
            symbol: Символ криптовалюти ('BTC', 'ETH', 'SOL')
            timeframe: Часовий інтервал ('1m', '1h', '4h', '1d', '1w')
            model_type: Тип моделі ('lstm' або 'gru')
            test_data: Тестові дані (якщо None, використовуються збережені тестові дані)

        Returns:
            Dict: Метрики ефективності моделі
        """
        # Перевірка наявності моделі
        model_key = self._create_model_key(symbol, timeframe, model_type)
        if model_key not in self.models:
            if not self.load_model(symbol, timeframe, model_type):
                raise ValueError(f"Модель {model_key} не знайдена")

        # Якщо тестові дані не надані, завантажуємо останні дані
        if test_data is None:
            data_loader = self._get_data_loader(symbol, timeframe, model_type)
            test_data = data_loader()

        # Підготовка даних для оцінки
        processed_data = self._prepare_features(test_data, symbol)

        # Оцінка моделі
        model = self.models[model_key]
        model.eval()

        # Обчислення метрик (MSE, RMSE, MAE, MAPE тощо)
        metrics = {}
        # ...

        # Зберігаємо метрики в БД
        save_ml_model_metrics(symbol, timeframe, model_type, metrics)

        # Оновлюємо фактичні значення для прогнозів
        # Використовуємо метод update_prediction_actual_value для оновлення прогнозів

        return metrics

    def compare_models(self, symbol: str, timeframe: str) -> Dict[str, Dict[str, float]]:
        """
        Порівняння ефективності LSTM та GRU моделей для вказаного символу та таймфрейму

        Args:
            symbol: Символ криптовалюти ('BTC', 'ETH', 'SOL')
            timeframe: Часовий інтервал ('1m', '1h', '4h', '1d', '1w')

        Returns:
            Dict: Порівняльні метрики моделей
        """
        # Перевірка наявності моделей
        models_to_compare = ['lstm', 'gru']
        comparison_results = {}

        for model_type in models_to_compare:
            model_key = self._create_model_key(symbol, timeframe, model_type)
            if model_key not in self.models:
                if not self.load_model(symbol, timeframe, model_type):
                    logger.warning(f"Модель {model_key} не знайдена і не буде включена в порівняння")
                    continue

            # Оцінка моделі
            metrics = self.evaluate_model(symbol, timeframe, model_type)
            comparison_results[model_type] = metrics

        # Можна порівняти результати і визначити кращу модель
        # ...

        return comparison_results

    def generate_multi_step_forecast(self, symbol: str, timeframe: str, model_type: str,
                                     steps: int = 10) -> pd.DataFrame:
        """
        Генерація багатокрокового прогнозу

        Args:
            symbol: Символ криптовалюти ('BTC', 'ETH', 'SOL')
            timeframe: Часовий інтервал ('1m', '1h', '4h', '1d', '1w')
            model_type: Тип моделі ('lstm' або 'gru')
            steps: Кількість кроків для прогнозування

        Returns:
            pd.DataFrame: Прогнозовані значення
        """
        # Прогнозування на кілька кроків вперед
        predictions = self.predict(symbol, timeframe, model_type, steps_ahead=steps)

        # Створення DataFrame з прогнозами
        forecast_df = pd.DataFrame({
            'step': range(1, steps + 1),
            'prediction': predictions
        })

        # Зберігаємо прогнози в БД
        timestamp = datetime.now()
        for i, pred in enumerate(predictions):
            # Використовуємо метод save_prediction для збереження прогнозу
            save_prediction(
                symbol=symbol,
                timeframe=timeframe,
                model_type=model_type,
                prediction_value=float(pred),
                prediction_timestamp=timestamp,
                steps_ahead=i + 1
            )

        return forecast_df

    def _create_model_key(self, symbol: str, timeframe: str, model_type: str) -> str:
        """
        Створення ключа для доступу до моделі у словнику

        Args:
            symbol: Символ криптовалюти ('BTC', 'ETH', 'SOL')
            timeframe: Часовий інтервал ('1m', '1h', '4h', '1d', '1w')
            model_type: Тип моделі ('lstm' або 'gru')

        Returns:
            str: Ключ моделі
        """
        return f"{symbol}_{timeframe}_{model_type}"
    
    def _validate_inputs(self, symbol: str, timeframe: str, model_type: str) -> bool:
        """
        Перевірка правильності вхідних параметрів
        
        Args:
            symbol: Символ криптовалюти
            timeframe: Часовий інтервал
            model_type: Тип моделі
            
        Returns:
            bool: True, якщо всі параметри правильні
        """
        pass


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import os
import logging
from datetime import datetime
from abc import ABC, abstractmethod

# Імпорт з інших модулів проекту
from data.db import (
    get_btc_lstm_sequence, get_eth_lstm_sequence, get_sol_lstm_sequence,
    save_prediction,  # Збереження прогнозів моделі
    save_ml_model_metrics,  # Збереження метрик ефективності моделі
    save_ml_model,  # Збереження інформації про модель
    save_ml_sequence_data,  # Збереження послідовностей для LSTM/GRU
    save_technical_indicator,  # Збереження технічних індикаторів
    update_prediction_actual_value  # Оновлення прогнозів з фактичними значеннями
)
from analysis.trend_detection import prepare_ml_trend_features  # Для отримання ознак тренду
from analysis.volatility_analysis import prepare_volatility_features_for_ml  # Для отримання ознак волатильності
from cyclefeatures.crypto_cycles import prepare_cycle_ml_features  # Для отримання циклічних ознак
from featureengineering.feature_engineering import prepare_features_pipeline  # Для підготовки всіх ознак
from utils.logger import get_logger
from utils.config import get_config

logger = get_logger(__name__)
config = get_config()


class BaseDeepModel(nn.Module, ABC):
    """Базовий абстрактний клас для deep learning моделей"""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int, dropout: float = 0.2):
        """Базовий конструктор для нейронних моделей"""
        super(BaseDeepModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout

    @abstractmethod
    def forward(self, x):
        """Абстрактний метод для прямого проходу"""
        pass


class LSTMModel(BaseDeepModel):
    """LSTM модель для прогнозування часових рядів криптовалют"""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int, dropout: float = 0.2):
        """Ініціалізація архітектури LSTM моделі"""
        super(LSTMModel, self).__init__(input_dim, hidden_dim, num_layers, output_dim, dropout)

        # LSTM шар для обробки послідовностей
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        # Повноз'єднаний шар для прогнозування
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Ініціалізація прихованого стану
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Прямий прохід через LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Використовуємо лише вихід останнього часового кроку
        out = self.fc(out[:, -1, :])
        return out


class GRUModel(BaseDeepModel):
    """GRU модель для прогнозування часових рядів криптовалют"""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int, dropout: float = 0.2):
        """Ініціалізація архітектури GRU моделі"""
        super(GRUModel, self).__init__(input_dim, hidden_dim, num_layers, output_dim, dropout)

        # GRU шар для обробки послідовностей
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        # Повноз'єднаний шар для прогнозування
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Ініціалізація прихованого стану
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Прямий прохід через GRU
        out, _ = self.gru(x, h0)

        # Використовуємо лише вихід останнього часового кроку
        out = self.fc(out[:, -1, :])
        return out


class DataLoader:
    """Клас для завантаження та підготовки даних для моделей"""

    SYMBOLS = ['BTC', 'ETH', 'SOL']
    TIMEFRAMES = ['1m', '1h', '4h', '1d', '1w']

    @staticmethod
    def get_data_loader(symbol: str, timeframe: str) -> Callable:
        """
        Отримати функцію для завантаження даних в залежності від символу

        Args:
            symbol: Символ криптовалюти ('BTC', 'ETH', 'SOL')
            timeframe: Часовий інтервал ('1m', '1h', '4h', '1d', '1w')

        Returns:
            Callable: Функція для завантаження даних
        """
        # Використовуємо методи з модуля data.db для отримання даних послідовностей
        if symbol == 'BTC':
            return lambda: get_btc_lstm_sequence(timeframe)
        elif symbol == 'ETH':
            return lambda: get_eth_lstm_sequence(timeframe)
        elif symbol == 'SOL':
            return lambda: get_sol_lstm_sequence(timeframe)
        else:
            raise ValueError(f"Непідтримуваний символ: {symbol}")

    @staticmethod
    def prepare_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Підготовка ознак для навчання моделі

        Args:
            df: DataFrame з даними
            symbol: Символ криптовалюти

        Returns:
            pd.DataFrame: DataFrame з підготовленими ознаками
        """
        # Використовуємо методи з різних модулів для підготовки ознак
        # 1. Отримуємо ознаки тренду
        trend_features = prepare_ml_trend_features(df, symbol)

        # 2. Отримуємо ознаки волатильності
        volatility_features = prepare_volatility_features_for_ml(df, symbol)

        # 3. Отримуємо циклічні ознаки
        cycle_features = prepare_cycle_ml_features(df, symbol)

        # 4. Об'єднуємо всі ознаки за допомогою пайплайну
        final_features = prepare_features_pipeline(
            df,
            trend_features=trend_features,
            volatility_features=volatility_features,
            cycle_features=cycle_features,
            symbol=symbol
        )

        return final_features

    @staticmethod
    def preprocess_data_for_model(data: pd.DataFrame, validation_split: float = 0.2):
        """
        Підготовка даних для навчання: нормалізація, розділення на тренувальну та валідаційну вибірки

        Args:
            data: DataFrame з підготовленими ознаками
            validation_split: Частка даних для валідації

        Returns:
            tuple: (X_train, y_train, X_val, y_val) - підготовлені дані для навчання та валідації
        """
        # Реалізація логіки препроцесингу даних
        # ...

        # Тут має бути логіка розділення даних, нормалізації тощо
        # Заглушка для демонстрації
        return None, None, None, None


class ModelFactory:
    """Фабрика для створення моделей різних типів"""

    @staticmethod
    def create_model(model_type: str, input_dim: int, hidden_dim: int = 64,
                     num_layers: int = 2, output_dim: int = 1) -> BaseDeepModel:
        """
        Створення моделі відповідного типу

        Args:
            model_type: Тип моделі ('lstm' або 'gru')
            input_dim: Розмірність вхідних даних
            hidden_dim: Розмірність прихованого шару
            num_layers: Кількість шарів
            output_dim: Розмірність вихідних даних

        Returns:
            BaseDeepModel: Створена модель
        """
        if model_type.lower() == 'lstm':
            return LSTMModel(input_dim, hidden_dim, num_layers, output_dim)
        elif model_type.lower() == 'gru':
            return GRUModel(input_dim, hidden_dim, num_layers, output_dim)
        else:
            raise ValueError(f"Непідтримуваний тип моделі: {model_type}")


class ModelTrainer:
    """Клас для навчання та оцінки моделей"""

    def __init__(self, device=None):
        """
        Ініціалізація тренера моделей

        Args:
            device: Пристрій для навчання (якщо None, визначається автоматично)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def train(self, model: BaseDeepModel, train_data, validation_data,
              epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001) -> Dict[str, Any]:
        """
        Навчання моделі на даних

        Args:
            model: Модель для навчання
            train_data: Дані для навчання (X_train, y_train)
            validation_data: Дані для валідації (X_val, y_val)
            epochs: Кількість епох навчання
            batch_size: Розмір батчу
            learning_rate: Швидкість навчання

        Returns:
            Dict: Історія навчання та результати
        """
        model = model.to(self.device)

        # Налаштування оптимізатора та функції втрат
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Історія навчання
        history = {
            'train_loss': [],
            'val_loss': []
        }

        # Тут має бути логіка навчання моделі
        # ...

        return history

    def evaluate(self, model: BaseDeepModel, test_data) -> Dict[str, float]:
        """
        Оцінка ефективності моделі на тестових даних

        Args:
            model: Навчена модель
            test_data: Тестові дані (X_test, y_test)

        Returns:
            Dict: Метрики ефективності моделі
        """
        model = model.to(self.device)
        model.eval()

        # Тут має бути логіка оцінки моделі
        # ...

        # Заглушка для демонстрації
        return {
            'mse': 0.0,
            'rmse': 0.0,
            'mae': 0.0,
            'mape': 0.0
        }


class ModelPersistence:
    """Клас для збереження та завантаження моделей"""

    def __init__(self, models_dir: str):
        """
        Ініціалізація менеджера збереження моделей

        Args:
            models_dir: Директорія для збереження моделей
        """
        self.models_dir = models_dir
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def save_model(self, model: BaseDeepModel, symbol: str, timeframe: str, model_type: str,
                   config: Dict[str, Any], metrics: Dict[str, float]) -> str:
        """
        Збереження навченої моделі на диск

        Args:
            model: Навчена модель
            symbol: Символ криптовалюти
            timeframe: Часовий інтервал
            model_type: Тип моделі
            config: Конфігурація моделі
            metrics: Метрики ефективності моделі

        Returns:
            str: Шлях до збереженої моделі
        """
        # Створюємо директорію для моделі, якщо вона не існує
        model_dir = os.path.join(self.models_dir, symbol, timeframe)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Шлях до файлу моделі
        model_path = os.path.join(model_dir, f"{model_type.lower()}_model.pth")

        # Зберігаємо модель
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'metrics': metrics
        }, model_path)

        # Зберігаємо модель в БД
        save_ml_model(
            symbol=symbol,
            timeframe=timeframe,
            model_type=model_type,
            model_config=config,
            model_path=model_path
        )

        return model_path

    def load_model(self, symbol: str, timeframe: str, model_type: str, device) -> tuple:
        """
        Завантаження моделі з диску

        Args:
            symbol: Символ криптовалюти
            timeframe: Часовий інтервал
            model_type: Тип моделі
            device: Пристрій для завантаження моделі

        Returns:
            tuple: (model, config, metrics) - завантажена модель, її конфігурація та метрики
        """
        # Шлях до файлу моделі
        model_dir = os.path.join(self.models_dir, symbol, timeframe)
        model_path = os.path.join(model_dir, f"{model_type.lower()}_model.pth")

        if not os.path.exists(model_path):
            logger.warning(f"Модель {symbol}_{timeframe}_{model_type} не знайдена за шляхом {model_path}")
            return None, None, None

        try:
            # Завантаження моделі
            checkpoint = torch.load(model_path, map_location=device)

            # Отримання конфігурації та створення моделі
            config = checkpoint['config']
            model = ModelFactory.create_model(
                model_type,
                config['input_dim'],
                config['hidden_dim'],
                config['num_layers'],
                config['output_dim']
            ).to(device)

            # Завантаження ваг моделі
            model.load_state_dict(checkpoint['model_state_dict'])

            # Отримання метрик
            metrics = checkpoint.get('metrics', {})

            logger.info(f"Модель {symbol}_{timeframe}_{model_type} успішно завантажена")
            return model, config, metrics

        except Exception as e:
            logger.error(f"Помилка при завантаженні моделі {symbol}_{timeframe}_{model_type}: {str(e)}")
            return None, None, None


class Predictor:
    """Клас для прогнозування на основі навчених моделей"""

    def __init__(self, device=None):
        """
        Ініціалізація предиктора

        Args:
            device: Пристрій для прогнозування (якщо None, визначається автоматично)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self, model: BaseDeepModel, input_data, steps_ahead: int = 1) -> np.ndarray:
        """
        Прогнозування на основі навченої моделі

        Args:
            model: Навчена модель
            input_data: Вхідні дані для прогнозування
            steps_ahead: Кількість кроків для прогнозування вперед

        Returns:
            np.ndarray: Прогнозовані значення
        """
        model = model.to(self.device)
        model.eval()

        with torch.no_grad():
            # Тут має бути логіка прогнозування
            # ...

            # Заглушка для демонстрації
            predictions = np.zeros(steps_ahead)

        return predictions

    def save_predictions(self, predictions: np.ndarray, symbol: str, timeframe: str, model_type: str):
        """
        Збереження прогнозів в базу даних

        Args:
            predictions: Прогнозовані значення
            symbol: Символ криптовалюти
            timeframe: Часовий інтервал
            model_type: Тип моделі
        """
        timestamp = datetime.now()
        for i, pred in enumerate(predictions):
            # Використовуємо метод save_prediction для збереження прогнозу
            save_prediction(
                symbol=symbol,
                timeframe=timeframe,
                model_type=model_type,
                prediction_value=float(pred),
                prediction_timestamp=timestamp,
                steps_ahead=i + 1
            )


class DeepLearning:
    """
    Фасадний клас для роботи з глибокими нейронними мережами для прогнозування криптовалют
    Підтримує LSTM та GRU моделі для BTC, ETH та SOL на різних таймфреймах
    """

    SYMBOLS = ['BTC', 'ETH', 'SOL']
    TIMEFRAMES = ['1m', '1h', '4h', '1d', '1w']

    def __init__(self):
        """Ініціалізація класу DeepLearning"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Шляхи для збереження моделей
        self.models_dir = os.path.join(config.get('paths', 'models_dir'), 'deep_learning')

        # Ініціалізація допоміжних класів
        self.data_loader = DataLoader()
        self.model_trainer = ModelTrainer(self.device)
        self.model_persistence = ModelPersistence(self.models_dir)
        self.predictor = Predictor(self.device)

        # Словники для кешування моделей та їх конфігурацій
        self.models = {}  # Словник для зберігання навчених моделей
        self.model_configs = {}  # Конфігурації моделей
        self.model_metrics = {}  # Метрики ефективності моделей

    def _validate_inputs(self, symbol: str, timeframe: str, model_type: str) -> bool:
        """
        Перевірка правильності вхідних параметрів

        Args:
            symbol: Символ криптовалюти
            timeframe: Часовий інтервал
            model_type: Тип моделі

        Returns:
            bool: True, якщо всі параметри правильні
        """
        if symbol not in self.SYMBOLS:
            raise ValueError(f"Непідтримуваний символ: {symbol}. Доступні символи: {self.SYMBOLS}")

        if timeframe not in self.TIMEFRAMES:
            raise ValueError(f"Непідтримуваний таймфрейм: {timeframe}. Доступні таймфрейми: {self.TIMEFRAMES}")

        if model_type.lower() not in ['lstm', 'gru']:
            raise ValueError(f"Непідтримуваний тип моделі: {model_type}. Доступні типи: 'lstm', 'gru'")

        return True

    def _create_model_key(self, symbol: str, timeframe: str, model_type: str) -> str:
        """
        Створення ключа для доступу до моделі у словнику

        Args:
            symbol: Символ криптовалюти ('BTC', 'ETH', 'SOL')
            timeframe: Часовий інтервал ('1m', '1h', '4h', '1d', '1w')
            model_type: Тип моделі ('lstm' або 'gru')

        Returns:
            str: Ключ моделі
        """
        return f"{symbol}_{timeframe}_{model_type}"

    def train_model(self, symbol: str, timeframe: str, model_type: str,
                    epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001,
                    hidden_dim: int = 64, num_layers: int = 2,
                    validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Навчання моделі для вказаного символу та таймфрейму

        Args:
            symbol: Символ криптовалюти ('BTC', 'ETH', 'SOL')
            timeframe: Часовий інтервал ('1m', '1h', '4h', '1d', '1w')
            model_type: Тип моделі ('lstm' або 'gru')
            epochs: Кількість епох навчання
            batch_size: Розмір батчу
            learning_rate: Швидкість навчання
            hidden_dim: Розмірність прихованого шару
            num_layers: Кількість шарів
            validation_split: Частка даних для валідації

        Returns:
            Dict: Історія навчання та метрики
        """
        # Перевірка коректності вхідних параметрів
        self._validate_inputs(symbol, timeframe, model_type)

        # Завантаження даних та підготовка ознак
        data_loader_fn = DataLoader.get_data_loader(symbol, timeframe)
        df = data_loader_fn()
        processed_data = DataLoader.prepare_features(df, symbol)

        # Підготовка даних для навчання
        X_train, y_train, X_val, y_val = DataLoader.preprocess_data_for_model(
            processed_data, validation_split=validation_split
        )

        # Створення моделі
        input_dim = len(processed_data.columns) - 1  # Не враховуємо цільовий стовпець
        model = ModelFactory.create_model(model_type, input_dim, hidden_dim, num_layers)

        # Навчання моделі
        training_history = self.model_trainer.train(
            model,
            (X_train, y_train),
            (X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )

        # Оцінка моделі
        metrics = self.model_trainer.evaluate(model, (X_val, y_val))

        # Зберігаємо модель, конфігурацію та метрики
        model_key = self._create_model_key(symbol, timeframe, model_type)
        self.models[model_key] = model

        # Зберігаємо конфігурацію моделі
        self.model_configs[model_key] = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'output_dim': 1,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }

        # Зберігаємо метрики
        self.model_metrics[model_key] = metrics

        # Зберігаємо модель на диск і в БД
        self.model_persistence.save_model(
            model,
            symbol,
            timeframe,
            model_type,
            self.model_configs[model_key],
            metrics
        )

        # Зберігаємо метрики в БД
        save_ml_model_metrics(symbol, timeframe, model_type, metrics)

        return {
            'config': self.model_configs[model_key],
            'metrics': metrics,
            'history': training_history
        }

    def online_learning(self, symbol: str, timeframe: str, model_type: str,
                        new_data: pd.DataFrame, epochs: int = 10,
                        learning_rate: float = 0.0005) -> Dict[str, Any]:
        """
        Онлайн-навчання існуючої моделі на нових даних

        Args:
            symbol: Символ криптовалюти ('BTC', 'ETH', 'SOL')
            timeframe: Часовий інтервал ('1m', '1h', '4h', '1d', '1w')
            model_type: Тип моделі ('lstm' або 'gru')
            new_data: Нові дані для навчання
            epochs: Кількість епох навчання
            learning_rate: Швидкість навчання

        Returns:
            Dict: Результати донавчання
        """
        # Перевірка наявності моделі
        model_key = self._create_model_key(symbol, timeframe, model_type)
        if model_key not in self.models:
            if not self.load_model(symbol, timeframe, model_type):
                raise ValueError(f"Модель {model_key} не знайдена")

        # Підготовка нових даних
        processed_data = DataLoader.prepare_features(new_data, symbol)
        X_train, y_train, X_val, y_val = DataLoader.preprocess_data_for_model(
            processed_data, validation_split=0.2
        )

        # Донавчання моделі
        model = self.models[model_key]
        training_history = self.model_trainer.train(
            model,
            (X_train, y_train),
            (X_val, y_val),
            epochs=epochs,
            batch_size=self.model_configs[model_key]['batch_size'],
            learning_rate=learning_rate
        )

        # Оновлення метрик
        metrics = self.model_trainer.evaluate(model, (X_val, y_val))
        self.model_metrics[model_key] = metrics

        # Зберігаємо оновлену модель та метрики
        self.model_persistence.save_model(
            model,
            symbol,
            timeframe,
            model_type,
            self.model_configs[model_key],
            metrics
        )
        save_ml_model_metrics(symbol, timeframe, model_type, metrics)

        return {
            'metrics': metrics,
            'history': training_history
        }