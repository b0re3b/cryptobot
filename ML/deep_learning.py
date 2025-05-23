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

from ML import LSTMModel, GRUModel, ModelTrainer
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
        self.model_trainer = ModelTrainer()
        self.lstm = LSTMModel()
        self.gru = GRUModel()
        # Словники для зберігання навчених моделей та їх конфігурацій
        self.models = {}  # {model_key: model}
        self.model_configs = {}  # {model_key: config}
        self.model_metrics = {}  # {model_key: metrics}
        self.db_manager = DatabaseManager()
        # Шляхи для збереження моделей
        self.models_dir = models_dir
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def train_model(self, symbol: str, timeframe: str, model_type: str,
                    data: pd.DataFrame, input_dim: int, **training_params) -> Dict[str, Any]:
        """
        Навчання моделі з використанням ModelTrainer.

        Args:
            symbol: Символ криптовалюти
            timeframe: Таймфрейм
            model_type: Тип моделі
            data: Дані для навчання
            input_dim: Розмірність вхідних даних
            **training_params: Додаткові параметри навчання

        Returns:
            Результати навчання
        """
        return self.model_trainer.train_model(
            symbol=symbol,
            timeframe=timeframe,
            model_type=model_type,
            data=data,
            input_dim=input_dim,
            **training_params
        )

    # ==================== ПРОГНОЗУВАННЯ ====================

    def predict(self, symbol: str, timeframe: str, model_type: str,
                input_data: Optional[pd.DataFrame] = None, steps_ahead: int = 1) -> Dict[str, Any]:
        """
        Прогнозування за допомогою навченої моделі.

        Args:
            symbol: Символ криптовалюти
            timeframe: Таймфрейм
            model_type: Тип моделі
            input_data: Вхідні дані (якщо None, завантажуються останні дані)
            steps_ahead: Кількість кроків прогнозу

        Returns:
            Словник з прогнозами та додатковою інформацією
        """
        # Завантаження моделі, якщо вона ще не завантажена
        model_key = self.model_trainer._create_model_key(symbol, timeframe, model_type)
        if model_key not in self.model_trainer.models:
            if not self.model_trainer.load_model(symbol, timeframe, model_type):
                raise ValueError(f"Модель {model_key} не знайдена")

        # Отримання даних, якщо вони не надані
        if input_data is None:
            data_loader = self.data_preprocessor.get_data_loader(symbol, timeframe, model_type)
            input_data = data_loader()

        # Підготовка даних
        processed_data = self.data_preprocessor.prepare_features(input_data, symbol)
        X = torch.tensor(processed_data.drop(columns=["target"]).float().to(self.device)

        # Прогнозування
        model = self.model_trainer.models[model_key]
        model.eval()

        with torch.no_grad():
            predictions = []
        current_input = X[-1:].unsqueeze(0)  # Беремо останню послідовність

        for _ in range(steps_ahead):
            pred = model(current_input)
        predictions.append(pred.item())
        # Оновлюємо вхідні дані для багатокрокового прогнозу
        if steps_ahead > 1:
            current_input = torch.cat([current_input[:, 1:], pred.unsqueeze(0).unsqueeze(0)], dim=1)

        # Збереження прогнозу
        timestamp = datetime.now()
        for i, pred in enumerate(predictions):
            self.db_manager.save_prediction(
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
            'model_key': model_key
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





    def evaluate_model(self, symbol: str, timeframe: str, model_type: str,
                       test_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Оцінка моделі на тестових даних.

        Args:
            symbol: Символ криптовалюти
            timeframe: Таймфрейм
            model_type: Тип моделі
            test_data: Тестові дані

        Returns:
            Словник з метриками
        """
        # Завантаження моделі
        model_key = self.model_trainer._create_model_key(symbol, timeframe, model_type)
        if model_key not in self.model_trainer.models:
            if not self.model_trainer.load_model(symbol, timeframe, model_type):
                raise ValueError(f"Модель {model_key} не знайдена")

        # Отримання даних, якщо вони не надані
        if test_data is None:
            data_loader = self.data_preprocessor.get_data_loader(symbol, timeframe, model_type)
            test_data = data_loader()

        # Підготовка даних
        processed_data = self.data_preprocessor.prepare_features(test_data, symbol)
        X = torch.tensor(processed_data.drop(columns=["target"]).float()
        y = torch.tensor(processed_data["target"]).float()

        # Оцінка
        return self.model_trainer.evaluate(self.model_trainer.models[model_key], (X, y))

    def model_performance_report(self, symbol: Optional[str] = None,
                                 timeframe: Optional[str] = None) -> pd.DataFrame:
        """Звіт про ефективність моделей"""
        pass

    # ==================== УПРАВЛІННЯ МОДЕЛЯМИ ====================



    def delete_model(self, symbol: str, timeframe: str, model_type: str) -> bool:
        """Видалення моделі"""
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
        model_key = self.model_trainer._create_model_key(symbol, timeframe, model_type)
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

    def online_learning(self, symbol: str, timeframe: str, model_type: str,
                        new_data: pd.DataFrame, epochs: int = 10) -> Dict[str, Any]:
        """
        Онлайн-дообучення моделі на нових даних.

        Args:
            symbol: Символ криптовалюти
            timeframe: Таймфрейм
            model_type: Тип моделі
            new_data: Нові дані для дообучення
            epochs: Кількість епох

        Returns:
            Результати дообучення
        """
        # Визначення input_dim
        input_dim = new_data.shape[1] - 1 if 'target' in new_data.columns else new_data.shape[1]

        return self.model_trainer.online_learning(
            symbol=symbol,
            timeframe=timeframe,
            model_type=model_type,
            new_data=new_data,
            input_dim=input_dim,
            epochs=epochs
        )

    def compare_models(self, symbol: str, timeframe: str,
                       test_data: Optional[pd.DataFrame] = None) -> Dict[str, Dict[str, float]]:
        """
        Порівняння моделей для заданого символу та таймфрейму.

        Args:
            symbol: Символ криптовалюти
            timeframe: Таймфрейм
            test_data: Тестові дані

        Returns:
            Словник з метриками для кожної моделі
        """
        results = {}

        for model_type in self.crypto_config.model_types:
            try:
                metrics = self.evaluate_model(symbol, timeframe, model_type, test_data)
                results[model_type] = metrics
            except Exception as e:
                self.logger.warning(f"Не вдалося оцінити модель {model_type}: {str(e)}")
                continue

        return results

    def get_model_info(self, symbol: str, timeframe: str, model_type: str) -> Dict[str, Any]:
        """
        Отримання інформації про модель.

        Args:
            symbol: Символ криптовалюти
            timeframe: Таймфрейм
            model_type: Тип моделі

        Returns:
            Інформація про модель
        """
        model_key = self.model_trainer._create_model_key(symbol, timeframe, model_type)
        if model_key not in self.model_trainer.models:
            if not self.model_trainer.load_model(symbol, timeframe, model_type):
                raise ValueError(f"Модель {model_key} не знайдена")

        model = self.model_trainer.models[model_key]

        # Базова інформація
        info = model.get_model_info()

        # Додаткова інформація в залежності від типу моделі
        if model_type.lower() == 'lstm':
            info.update(model.get_lstm_specific_info())
        elif model_type.lower() == 'gru':
            info.update(model.get_gru_specific_info())
        elif model_type.lower() == 'transformer':
            info.update(model.get_transformer_specific_info())

        return info

    def train_all_models(self, symbols: Optional[List[str]] = None,
                         timeframes: Optional[List[str]] = None,
                         model_types: Optional[List[str]] = None,
                         **training_params) -> Dict[str, Dict[str, Any]]:
        """
        Навчання всіх моделей для вказаних параметрів.

        Args:
            symbols: Список символів
            timeframes: Список таймфреймів
            model_types: Список типів моделей
            **training_params: Параметри навчання

        Returns:
            Результати навчання для всіх моделей
        """
        return self.model_trainer.train_all_models(
            symbols=symbols,
            timeframes=timeframes,
            model_types=model_types,
            **training_params
        )

    def cross_validate_model(self, symbol: str, timeframe: str, model_type: str,
                             k_folds: int = 5, **model_params) -> Dict[str, List[float]]:
        """
        Крос-валідація моделі.

        Args:
            symbol: Символ криптовалюти
            timeframe: Таймфрейм
            model_type: Тип моделі
            k_folds: Кількість фолдів
            **model_params: Параметри моделі

        Returns:
            Результати крос-валідації
        """
        return self.model_trainer.cross_validate_model(
            symbol=symbol,
            timeframe=timeframe,
            model_type=model_type,
            k_folds=k_folds,
            **model_params
        )

    def get_training_summary(self) -> Dict[str, Any]:
        """
        Отримання зведення про навчені моделі.

        Returns:
            Зведена інформація
        """
        return self.model_trainer.get_training_summary()