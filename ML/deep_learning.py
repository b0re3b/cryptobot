from dataclasses import dataclass, field

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
from data.db import DatabaseManager

(
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
from utils.logger import CryptoLogger
from ML.DataPreprocessor import DataPreprocessor
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

@dataclass
class CryptoConfig:
    symbols: List[str] = field(default_factory=lambda: ['BTC', 'ETH', 'SOL'])
    timeframes: List[str] = field(default_factory=lambda: ['1m', '1h', '4h', '1d', '1w'])
    model_types: List[str] = field(default_factory=lambda: ['lstm', 'gru'])

class DeepLearning:
    """
    Клас для роботи з глибокими нейронними мережами для прогнозування криптовалют
    Підтримує LSTM та GRU моделі для BTC, ETH та SOL на різних таймфреймах
    """



    def __init__(self, models_dir: str = "models/deep_learning"):
        """
        Ініціалізація класу DeepLearning
        Створення структур для зберігання моделей, їх конфігурацій та метрик
        """
        self.logger = CryptoLogger('deep_learning')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ініціалізація компонентів
        self.data_preprocessor = DataPreprocessor()
        self.model_trainer = ModelTrainer(self.device)

        # Словники для зберігання навчених моделей та їх конфігурацій
        self.models = {}  # {model_key: model}
        self.model_configs = {}  # {model_key: config}
        self.model_metrics = {}  # {model_key: metrics}
        self.db_manager = DatabaseManager()
        # Шляхи для збереження моделей
        self.models_dir = models_dir
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)



    # ==================== ПРОГНОЗУВАННЯ ====================

    def predict(self, symbol: str, timeframe: str, model_type: str,
                steps_ahead: int = 1, input_data: Optional[pd.DataFrame] = None,
                confidence_interval: bool = False) -> Dict[str, Any]:
        """
        Прогнозування на основі навченої моделі

        Args:
            symbol: Символ криптовалюти ('BTC', 'ETH', 'SOL')
            timeframe: Часовий інтервал ('1m', '1h', '4h', '1d', '1w')
            model_type: Тип моделі ('lstm' або 'gru')
            steps_ahead: Кількість кроків для прогнозування вперед
            input_data: Вхідні дані для прогнозування (якщо None, використовуються останні дані)
            confidence_interval: Чи повертати довірчий інтервал

        Returns:
            Dict: Прогнозовані значення та метрики
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

        return {
            'predictions': np.array(predictions),
            'timestamp': timestamp,
            'confidence_interval': None if not confidence_interval else {}
        }

    def predict_multiple_steps(self, symbol: str, timeframe: str, model_type: str,
                               steps: int = 10, input_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Багатокроковий прогноз

        Args:
            symbol: Символ криптовалюти ('BTC', 'ETH', 'SOL')
            timeframe: Часовий інтервал ('1m', '1h', '4h', '1d', '1w')
            model_type: Тип моделі ('lstm' або 'gru')
            steps: Кількість кроків для прогнозування
            input_data: Вхідні дані для прогнозування

        Returns:
            pd.DataFrame: Прогнозовані значення
        """
        # Прогнозування на кілька кроків вперед
        result = self.predict(symbol, timeframe, model_type, steps_ahead=steps, input_data=input_data)
        predictions = result['predictions']

        # Створення DataFrame з прогнозами
        forecast_df = pd.DataFrame({
            'step': range(1, steps + 1),
            'prediction': predictions
        })

        return forecast_df

    def predict_all_symbols(self, timeframe: str, model_type: str,
                            steps_ahead: int = 1) -> Dict[str, np.ndarray]:
        """Прогнозування для всіх символів"""
        pass

    def ensemble_predict(self, symbol: str, timeframe: str, steps_ahead: int = 1,
                         weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Ансамблевий прогноз (комбінація LSTM та GRU)"""
        pass

    # ==================== ОЦІНКА МОДЕЛЕЙ ====================



    def compare_models(self, symbol: str, timeframe: str,
                       test_data: Optional[pd.DataFrame] = None) -> Dict[str, Dict[str, float]]:
        """
        Порівняння ефективності LSTM та GRU моделей для вказаного символу та таймфрейму

        Args:
            symbol: Символ криптовалюти ('BTC', 'ETH', 'SOL')
            timeframe: Часовий інтервал ('1m', '1h', '4h', '1d', '1w')
            test_data: Тестові дані для порівняння

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
                    self.logger.warning(f"Модель {model_key} не знайдена і не буде включена в порівняння")
                    continue

            # Оцінка моделі
            metrics = self.evaluate_model(symbol, timeframe, model_type, test_data)
            comparison_results[model_type] = metrics

        # Можна порівняти результати і визначити кращу модель
        # ...

        return comparison_results



    def model_performance_report(self, symbol: Optional[str] = None,
                                 timeframe: Optional[str] = None) -> pd.DataFrame:
        """Звіт про ефективність моделей"""
        pass

    # ==================== УПРАВЛІННЯ МОДЕЛЯМИ ====================



    def delete_model(self, symbol: str, timeframe: str, model_type: str) -> bool:
        """Видалення моделі"""
        pass

    def get_model_info(self, symbol: str, timeframe: str, model_type: str) -> Dict[str, Any]:
        """Отримання інформації про модель"""
        pass

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

    def list_trained_models(self) -> List[Dict[str, str]]:
        """Список навчених моделей"""
        pass

    def update_model_metrics(self, symbol: str, timeframe: str, model_type: str,
                             new_metrics: Dict[str, float]) -> None:
        """Оновлення метрик моделі"""
        pass

    # ==================== МОНІТОРИНГ ТА ЛОГУВАННЯ ====================

    def get_training_history(self, symbol: str, timeframe: str, model_type: str) -> Dict[str, List[float]]:
        """Отримання історії навчання"""
        pass

    def plot_training_history(self, symbol: str, timeframe: str, model_type: str) -> None:
        """Візуалізація історії навчання"""
        pass

    # ==================== ЕКСПОРТ ТА ІНТЕГРАЦІЯ ====================

    def export_model_for_production(self, symbol: str, timeframe: str, model_type: str,
                                    export_format: str = 'torch') -> str:
        """Експорт моделі для продакшн"""
        pass

    def load_pretrained_model(self, model_path: str, symbol: str,
                              timeframe: str, model_type: str) -> bool:
        """Завантаження попередньо навченої моделі"""
        pass

    def benchmark_models(self, benchmark_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Бенчмарк моделей на тестових даних"""
        pass

    # ==================== ДОПОМІЖНІ МЕТОДИ ====================

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

        if model_type.lower() not in self.MODEL_TYPES:
            raise ValueError(f"Непідтримуваний тип моделі: {model_type}. Доступні типи: {self.MODEL_TYPES}")

        return True

    def _prepare_training_data(self, symbol: str, timeframe: str,
                               sequence_length: int, validation_split: float) -> Tuple:
        """Підготовка даних для навчання"""
        pass
