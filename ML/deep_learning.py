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
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

    SYMBOLS = ['BTC', 'ETH', 'SOL']
    TIMEFRAMES = ['1m', '1h', '4h', '1d', '1w']
    MODEL_TYPES = ['lstm', 'gru']

    def __init__(self, models_dir: str = "models/deep_learning"):
        """
        Ініціалізація класу DeepLearning
        Створення структур для зберігання моделей, їх конфігурацій та метрик
        """
        self.logger = CryptoLogger('deep_learning')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.crypto_config = CryptoConfig()

        # Ініціалізація компонентів
        self.data_preprocessor = DataPreprocessor()
        self.model_trainer = ModelTrainer()
        self.lstm = LSTMModel()
        self.gru = GRUModel()

        # Словники для зберігання навчених моделей та їх конфігурацій
        self.models = {}  # {model_key: model}
        self.model_configs = {}  # {model_key: config}
        self.model_metrics = {}  # {model_key: metrics}
        self.training_history = {}  # {model_key: history}

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
        self._validate_inputs(symbol, timeframe, model_type)

        try:
            result = self.model_trainer.train_model(
                symbol=symbol,
                timeframe=timeframe,
                model_type=model_type,
                data=data,
                input_dim=input_dim,
                **training_params
            )

            # Збереження результатів
            model_key = self.model_trainer._create_model_key(symbol, timeframe, model_type)
            self.model_metrics[model_key] = result.get('metrics', {})
            self.training_history[model_key] = result.get('history', {})

            self.logger.info(f"Модель {model_key} успішно навчена")
            return result

        except Exception as e:
            self.logger.error(f"Помилка навчання моделі {symbol}-{timeframe}-{model_type}: {str(e)}")
            raise

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
        self._validate_inputs(symbol, timeframe, model_type)

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
        X = torch.tensor(processed_data.drop(columns=["target"]).values, dtype=torch.float32).to(self.device)

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
            'prediction': predictions,
            'timestamp': [result['timestamp']] * steps
        })

        return forecast_df

    def predict_all_symbols(self, timeframe: str, model_type: str,
                            steps_ahead: int = 1) -> Dict[str, np.ndarray]:
        """Прогнозування для всіх символів"""
        predictions = {}

        for symbol in self.SYMBOLS:
            try:
                result = self.predict(symbol, timeframe, model_type, steps_ahead=steps_ahead)
                predictions[symbol] = result['predictions']
                self.logger.info(f"Прогноз для {symbol} створено успішно")
            except Exception as e:
                self.logger.error(f"Помилка прогнозування для {symbol}: {str(e)}")
                predictions[symbol] = np.array([np.nan] * steps_ahead)

        return predictions

    def ensemble_predict(self, symbol: str, timeframe: str, steps_ahead: int = 1,
                         weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Ансамблевий прогноз (комбінація LSTM та GRU)"""
        if weights is None:
            weights = {'lstm': 0.5, 'gru': 0.5}

        predictions = {}
        ensemble_pred = np.zeros(steps_ahead)

        for model_type in self.MODEL_TYPES:
            try:
                result = self.predict(symbol, timeframe, model_type, steps_ahead=steps_ahead)
                predictions[model_type] = result['predictions']
                ensemble_pred += weights.get(model_type, 0) * result['predictions']
            except Exception as e:
                self.logger.warning(f"Модель {model_type} недоступна: {str(e)}")
                continue

        return {
            'ensemble_predictions': ensemble_pred,
            'individual_predictions': predictions,
            'weights': weights,
            'timestamp': datetime.now()
        }

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
        self._validate_inputs(symbol, timeframe, model_type)

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
        X = torch.tensor(processed_data.drop(columns=["target"]).values, dtype=torch.float32)
        y = torch.tensor(processed_data["target"].values, dtype=torch.float32)

        # Оцінка
        return self.model_trainer.evaluate(self.model_trainer.models[model_key], (X, y))

    def model_performance_report(self, symbol: Optional[str] = None,
                                 timeframe: Optional[str] = None) -> pd.DataFrame:
        """Звіт про ефективність моделей"""
        performance_data = []

        symbols = [symbol] if symbol else self.SYMBOLS
        timeframes = [timeframe] if timeframe else self.TIMEFRAMES

        for sym in symbols:
            for tf in timeframes:
                for model_type in self.MODEL_TYPES:
                    try:
                        metrics = self.get_model_metrics(sym, tf, model_type)
                        if metrics:
                            performance_data.append({
                                'symbol': sym,
                                'timeframe': tf,
                                'model_type': model_type,
                                **metrics
                            })
                    except Exception as e:
                        self.logger.warning(f"Не вдалося отримати метрики для {sym}-{tf}-{model_type}: {str(e)}")
                        continue

        return pd.DataFrame(performance_data)

    # ==================== УПРАВЛІННЯ МОДЕЛЯМИ ====================

    def delete_model(self, symbol: str, timeframe: str, model_type: str) -> bool:
        """Видалення моделі"""
        self._validate_inputs(symbol, timeframe, model_type)

        try:
            model_key = self.model_trainer._create_model_key(symbol, timeframe, model_type)

            # Видалення з пам'яті
            if model_key in self.model_trainer.models:
                del self.model_trainer.models[model_key]
            if model_key in self.model_metrics:
                del self.model_metrics[model_key]
            if model_key in self.training_history:
                del self.training_history[model_key]

            # Видалення файлів
            model_file = os.path.join(self.models_dir, f"{model_key}.pt")
            if os.path.exists(model_file):
                os.remove(model_file)

            config_file = os.path.join(self.models_dir, f"{model_key}_config.json")
            if os.path.exists(config_file):
                os.remove(config_file)

            self.logger.info(f"Модель {model_key} успішно видалена")
            return True

        except Exception as e:
            self.logger.error(f"Помилка видалення моделі {symbol}-{timeframe}-{model_type}: {str(e)}")
            return False

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
            # Спроба завантажити метрики з файлу
            metrics_file = os.path.join(self.models_dir, f"{model_key}_metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    self.model_metrics[model_key] = metrics
                    return metrics
            return {}

    def list_trained_models(self) -> List[Dict[str, str]]:
        """Список навчених моделей"""
        models_list = []

        # Моделі в пам'яті
        for model_key in self.model_trainer.models.keys():
            parts = model_key.split('_')
            if len(parts) >= 3:
                models_list.append({
                    'symbol': parts[0],
                    'timeframe': parts[1],
                    'model_type': parts[2],
                    'status': 'loaded',
                    'model_key': model_key
                })

        # Моделі у файлах
        if os.path.exists(self.models_dir):
            for filename in os.listdir(self.models_dir):
                if filename.endswith('.pt'):
                    model_key = filename[:-3]  # Видаляємо .pt
                    if model_key not in [m['model_key'] for m in models_list]:
                        parts = model_key.split('_')
                        if len(parts) >= 3:
                            models_list.append({
                                'symbol': parts[0],
                                'timeframe': parts[1],
                                'model_type': parts[2],
                                'status': 'saved',
                                'model_key': model_key
                            })

        return models_list

    def update_model_metrics(self, symbol: str, timeframe: str, model_type: str,
                             new_metrics: Dict[str, float]) -> None:
        """Оновлення метрик моделі"""
        model_key = self.model_trainer._create_model_key(symbol, timeframe, model_type)

        if model_key not in self.model_metrics:
            self.model_metrics[model_key] = {}

        self.model_metrics[model_key].update(new_metrics)

        # Збереження в файл
        metrics_file = os.path.join(self.models_dir, f"{model_key}_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.model_metrics[model_key], f, indent=2)

    # ==================== МОНІТОРИНГ ТА ЛОГУВАННЯ ====================

    def get_training_history(self, symbol: str, timeframe: str, model_type: str) -> Dict[str, List[float]]:
        """Отримання історії навчання"""
        model_key = self.model_trainer._create_model_key(symbol, timeframe, model_type)

        if model_key in self.training_history:
            return self.training_history[model_key]
        else:
            # Спроба завантажити з файлу
            history_file = os.path.join(self.models_dir, f"{model_key}_history.pkl")
            if os.path.exists(history_file):
                with open(history_file, 'rb') as f:
                    history = pickle.load(f)
                    self.training_history[model_key] = history
                    return history
            return {}

    def plot_training_history(self, symbol: str, timeframe: str, model_type: str) -> None:
        """Візуалізація історії навчання"""
        history = self.get_training_history(symbol, timeframe, model_type)

        if not history:
            self.logger.warning(f"Історія навчання для {symbol}-{timeframe}-{model_type} не знайдена")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        if 'train_loss' in history:
            axes[0, 0].plot(history['train_loss'], label='Train Loss')
            if 'val_loss' in history:
                axes[0, 0].plot(history['val_loss'], label='Validation Loss')
            axes[0, 0].set_title('Model Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()

        # Accuracy (якщо є)
        if 'train_acc' in history:
            axes[0, 1].plot(history['train_acc'], label='Train Accuracy')
            if 'val_acc' in history:
                axes[0, 1].plot(history['val_acc'], label='Validation Accuracy')
            axes[0, 1].set_title('Model Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()

        # Learning Rate
        if 'lr' in history:
            axes[1, 0].plot(history['lr'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')

        # Gradient Norm (якщо є)
        if 'grad_norm' in history:
            axes[1, 1].plot(history['grad_norm'])
            axes[1, 1].set_title('Gradient Norm')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Gradient Norm')

        plt.tight_layout()
        plt.show()

    # ==================== ЕКСПОРТ ТА ІНТЕГРАЦІЯ ====================

    def export_model_for_production(self, symbol: str, timeframe: str, model_type: str,
                                    export_format: str = 'torch') -> str:
        """Експорт моделі для продакшн"""
        self._validate_inputs(symbol, timeframe, model_type)

        model_key = self.model_trainer._create_model_key(symbol, timeframe, model_type)
        if model_key not in self.model_trainer.models:
            if not self.model_trainer.load_model(symbol, timeframe, model_type):
                raise ValueError(f"Модель {model_key} не знайдена")

        model = self.model_trainer.models[model_key]
        export_dir = os.path.join(self.models_dir, 'production')
        os.makedirs(export_dir, exist_ok=True)

        if export_format.lower() == 'torch':
            export_path = os.path.join(export_dir, f"{model_key}_production.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_class': model.__class__.__name__,
                'config': self.model_configs.get(model_key, {}),
                'metrics': self.model_metrics.get(model_key, {}),
                'export_timestamp': datetime.now().isoformat()
            }, export_path)

        elif export_format.lower() == 'onnx':
            try:
                import torch.onnx
                export_path = os.path.join(export_dir, f"{model_key}_production.onnx")
                dummy_input = torch.randn(1, model.sequence_length, model.input_dim)
                torch.onnx.export(model, dummy_input, export_path,
                                  export_params=True, opset_version=11,
                                  input_names=['input'], output_names=['output'])
            except ImportError:
                raise ImportError("ONNX не встановлено. Встановіть: pip install onnx")

        else:
            raise ValueError(f"Непідтримуваний формат експорту: {export_format}")

        self.logger.info(f"Модель {model_key} експортована в {export_path}")
        return export_path

    def load_pretrained_model(self, model_path: str, symbol: str,
                              timeframe: str, model_type: str) -> bool:
        """Завантаження попередньо навченої моделі"""
        try:
            self._validate_inputs(symbol, timeframe, model_type)

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Файл моделі не знайдено: {model_path}")

            model_key = self.model_trainer._create_model_key(symbol, timeframe, model_type)

            # Завантаження моделі
            checkpoint = torch.load(model_path, map_location=self.device)

            # Створення моделі відповідного типу
            if model_type.lower() == 'lstm':
                model = self.lstm
            elif model_type.lower() == 'gru':
                model = self.gru
            else:
                raise ValueError(f"Непідтримуваний тип моделі: {model_type}")

            # Завантаження стану моделі
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()

            # Збереження в пам'яті
            self.model_trainer.models[model_key] = model
            self.model_configs[model_key] = checkpoint.get('config', {})
            self.model_metrics[model_key] = checkpoint.get('metrics', {})

            self.logger.info(f"Попередньо навчена модель {model_key} завантажена успішно")
            return True

        except Exception as e:
            self.logger.error(f"Помилка завантаження попередньо навченої моделі: {str(e)}")
            return False

    def benchmark_models(self, benchmark_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Бенчмарк моделей на тестових даних"""
        results = {}

        for symbol in self.SYMBOLS:
            for timeframe in self.TIMEFRAMES:
                for model_type in self.MODEL_TYPES:
                    try:
                        metrics = self.evaluate_model(symbol, timeframe, model_type, benchmark_data)
                        model_key = f"{symbol}_{timeframe}_{model_type}"
                        results[model_key] = metrics
                    except Exception as e:
                        self.logger.warning(f"Не вдалося протестувати {symbol}-{timeframe}-{model_type}: {str(e)}")
                        continue

        return results

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
        try:
            # Отримання даних
            data_loader = self.data_preprocessor.get_data_loader(symbol, timeframe, 'lstm')
            data = data_loader()

            # Підготовка ознак
            processed_data = self.data_preprocessor.prepare_features(data, symbol)

            # Розділення на тренувальні та валідаційні дані
            split_idx = int(len(processed_data) * (1 - validation_split))

            train_data = processed_data[:split_idx]
            val_data = processed_data[split_idx:]

            return train_data, val_data

        except Exception as e:
            self.logger.error(f"Помилка підготовки даних для {symbol}-{timeframe}: {str(e)}")
            raise

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

        # ==================== HYPERPARAMETER OPTIMIZATION ====================

        def hyperparameter_optimization(self, symbol: str, timeframe: str, model_type: str,
                                        param_space: Dict[str, List], optimization_method: str = 'grid_search',
                                        cv_folds: int = 3, max_iterations: int = 50) -> Dict[str, Any]:
            """
            Оптимізація гіперпараметрів моделі.

            Args:
                symbol: Символ криптовалюти
                timeframe: Таймфрейм
                model_type: Тип моделі
                param_space: Простір пошуку параметрів
                optimization_method: Метод оптимізації ('grid_search', 'random_search', 'bayesian')
                cv_folds: Кількість фолдів для крос-валідації
                max_iterations: Максимальна кількість ітерацій

            Returns:
                Результати оптимізації
            """
            self._validate_inputs(symbol, timeframe, model_type)

            try:
                # Отримання даних
                data_loader = self.data_preprocessor.get_data_loader(symbol, timeframe, model_type)
                data = data_loader()
                processed_data = self.data_preprocessor.prepare_features(data, symbol)

                best_params = {}
                best_score = float('inf')
                optimization_history = []

                if optimization_method == 'grid_search':
                    best_params, best_score, optimization_history = self._grid_search_optimization(
                        symbol, timeframe, model_type, processed_data, param_space, cv_folds
                    )
                elif optimization_method == 'random_search':
                    best_params, best_score, optimization_history = self._random_search_optimization(
                        symbol, timeframe, model_type, processed_data, param_space, max_iterations, cv_folds
                    )
                elif optimization_method == 'bayesian':
                    best_params, best_score, optimization_history = self._bayesian_optimization(
                        symbol, timeframe, model_type, processed_data, param_space, max_iterations, cv_folds
                    )
                else:
                    raise ValueError(f"Непідтримуваний метод оптимізації: {optimization_method}")

                # Збереження результатів
                optimization_results = {
                    'best_params': best_params,
                    'best_score': best_score,
                    'optimization_history': optimization_history,
                    'method': optimization_method,
                    'timestamp': datetime.now().isoformat()
                }

                # Збереження в файл
                results_file = os.path.join(self.models_dir, f"{symbol}_{timeframe}_{model_type}_optimization.json")
                with open(results_file, 'w') as f:
                    json.dump(optimization_results, f, indent=2, default=str)

                self.logger.info(f"Оптимізація гіперпараметрів для {symbol}-{timeframe}-{model_type} завершена")
                return optimization_results

            except Exception as e:
                self.logger.error(f"Помилка оптимізації гіперпараметрів: {str(e)}")
                raise

        def _grid_search_optimization(self, symbol: str, timeframe: str, model_type: str,
                                      data: pd.DataFrame, param_space: Dict, cv_folds: int) -> Tuple[Dict, float, List]:
            """Grid Search оптимізація"""
            from itertools import product

            param_combinations = list(product(*param_space.values()))
            param_names = list(param_space.keys())

            best_params = {}
            best_score = float('inf')
            history = []

            for combination in param_combinations:
                params = dict(zip(param_names, combination))

                try:
                    # Крос-валідація з поточними параметрами
                    cv_results = self.cross_validate_model(symbol, timeframe, model_type, cv_folds, **params)
                    avg_loss = np.mean(cv_results.get('val_loss', [float('inf')]))

                    history.append({
                        'params': params.copy(),
                        'score': avg_loss,
                        'cv_results': cv_results
                    })

                    if avg_loss < best_score:
                        best_score = avg_loss
                        best_params = params.copy()

                except Exception as e:
                    self.logger.warning(f"Помилка з параметрами {params}: {str(e)}")
                    continue

            return best_params, best_score, history

        def _random_search_optimization(self, symbol: str, timeframe: str, model_type: str,
                                        data: pd.DataFrame, param_space: Dict, max_iterations: int, cv_folds: int) -> \
        Tuple[Dict, float, List]:
            """Random Search оптимізація"""
            import random

            best_params = {}
            best_score = float('inf')
            history = []

            for i in range(max_iterations):
                # Випадковий вибір параметрів
                params = {}
                for param_name, param_values in param_space.items():
                    params[param_name] = random.choice(param_values)

                try:
                    # Крос-валідація з поточними параметрами
                    cv_results = self.cross_validate_model(symbol, timeframe, model_type, cv_folds, **params)
                    avg_loss = np.mean(cv_results.get('val_loss', [float('inf')]))

                    history.append({
                        'iteration': i + 1,
                        'params': params.copy(),
                        'score': avg_loss,
                        'cv_results': cv_results
                    })

                    if avg_loss < best_score:
                        best_score = avg_loss
                        best_params = params.copy()

                except Exception as e:
                    self.logger.warning(f"Помилка з параметрами {params}: {str(e)}")
                    continue

            return best_params, best_score, history

        def _bayesian_optimization(self, symbol: str, timeframe: str, model_type: str,
                                   data: pd.DataFrame, param_space: Dict, max_iterations: int, cv_folds: int) -> Tuple[
            Dict, float, List]:
            """Bayesian Optimization (спрощена версія)"""
            # Спрощена версія без зовнішніх залежностей
            # В реальному проекті можна використовувати optuna або skopt

            best_params = {}
            best_score = float('inf')
            history = []

            # Початкові випадкові точки
            initial_points = min(5, max_iterations // 2)

            # Випадкові початкові точки
            for i in range(initial_points):
                params = {}
                for param_name, param_values in param_space.items():
                    params[param_name] = np.random.choice(param_values)

                try:
                    cv_results = self.cross_validate_model(symbol, timeframe, model_type, cv_folds, **params)
                    avg_loss = np.mean(cv_results.get('val_loss', [float('inf')]))

                    history.append({
                        'iteration': i + 1,
                        'params': params.copy(),
                        'score': avg_loss,
                        'cv_results': cv_results,
                        'acquisition_type': 'random'
                    })

                    if avg_loss < best_score:
                        best_score = avg_loss
                        best_params = params.copy()

                except Exception as e:
                    self.logger.warning(f"Помилка з параметрами {params}: {str(e)}")
                    continue

            # Решта ітерацій з простою евристикою
            for i in range(initial_points, max_iterations):
                # Проста евристика: варіація навколо кращих параметрів
                params = self._vary_best_params(best_params, param_space)

                try:
                    cv_results = self.cross_validate_model(symbol, timeframe, model_type, cv_folds, **params)
                    avg_loss = np.mean(cv_results.get('val_loss', [float('inf')]))

                    history.append({
                        'iteration': i + 1,
                        'params': params.copy(),
                        'score': avg_loss,
                        'cv_results': cv_results,
                        'acquisition_type': 'exploitation'
                    })

                    if avg_loss < best_score:
                        best_score = avg_loss
                        best_params = params.copy()

                except Exception as e:
                    self.logger.warning(f"Помилка з параметрами {params}: {str(e)}")
                    continue

            return best_params, best_score, history

        def _vary_best_params(self, best_params: Dict, param_space: Dict) -> Dict:
            """Варіація кращих параметрів для Bayesian optimization"""
            varied_params = best_params.copy()

            # Варіюємо один або два параметри
            params_to_vary = np.random.choice(list(param_space.keys()),
                                              size=min(2, len(param_space)),
                                              replace=False)

            for param_name in params_to_vary:
                if param_name in param_space:
                    varied_params[param_name] = np.random.choice(param_space[param_name])

            return varied_params

        # ==================== ADVANCED ANALYSIS ====================

        def feature_importance_analysis(self, symbol: str, timeframe: str, model_type: str) -> Dict[str, float]:
            """
            Аналіз важливості ознак для моделі.

            Args:
                symbol: Символ криптовалюти
                timeframe: Таймфрейм
                model_type: Тип моделі

            Returns:
                Словник з важливістю кожної ознаки
            """
            self._validate_inputs(symbol, timeframe, model_type)

            # Завантаження моделі
            model_key = self.model_trainer._create_model_key(symbol, timeframe, model_type)
            if model_key not in self.model_trainer.models:
                if not self.model_trainer.load_model(symbol, timeframe, model_type):
                    raise ValueError(f"Модель {model_key} не знайдена")

            model = self.model_trainer.models[model_key]

            # Отримання даних
            data_loader = self.data_preprocessor.get_data_loader(symbol, timeframe, model_type)
            data = data_loader()
            processed_data = self.data_preprocessor.prepare_features(data, symbol)

            # Аналіз важливості методом пермутації
            feature_names = processed_data.drop(columns=["target"]).columns.tolist()
            importance_scores = self._permutation_importance(model, processed_data, feature_names)

            return dict(zip(feature_names, importance_scores))

        def _permutation_importance(self, model, data: pd.DataFrame, feature_names: List[str]) -> List[float]:
            """Розрахунок важливості ознак методом пермутації"""
            X = torch.tensor(data.drop(columns=["target"]).values, dtype=torch.float32).to(self.device)
            y = torch.tensor(data["target"].values, dtype=torch.float32).to(self.device)

            # Базова точність
            model.eval()
            with torch.no_grad():
                base_pred = model(X)
                base_loss = nn.MSELoss()(base_pred.squeeze(), y).item()

            importance_scores = []

            for i, feature_name in enumerate(feature_names):
                # Створюємо копію даних
                X_permuted = X.clone()

                # Перемішуємо значення ознаки
                perm_indices = torch.randperm(X_permuted.size(0))
                X_permuted[:, i] = X_permuted[perm_indices, i]

                # Розраховуємо втрату з перемішаною ознакою
                with torch.no_grad():
                    perm_pred = model(X_permuted)
                    perm_loss = nn.MSELoss()(perm_pred.squeeze(), y).item()

                # Важливість = зростання втрати
                importance = perm_loss - base_loss
                importance_scores.append(max(0, importance))  # Мінімум 0

            return importance_scores

        def model_interpretation(self, symbol: str, timeframe: str, model_type: str) -> Dict[str, Any]:
            """
            Комплексна інтерпретація моделі.

            Args:
                symbol: Символ криптовалюти
                timeframe: Таймфрейм
                model_type: Тип моделі

            Returns:
                Словник з результатами інтерпретації
            """
            interpretation_results = {}

            try:
                # Важливість ознак
                interpretation_results['feature_importance'] = self.feature_importance_analysis(symbol, timeframe,
                                                                                                model_type)

                # Метрики моделі
                interpretation_results['model_metrics'] = self.get_model_metrics(symbol, timeframe, model_type)

                # Інформація про модель
                interpretation_results['model_info'] = self.get_model_info(symbol, timeframe, model_type)

                # Аналіз помилок
                interpretation_results['error_analysis'] = self._error_analysis(symbol, timeframe, model_type)

                # Стабільність прогнозів
                interpretation_results['prediction_stability'] = self._prediction_stability_analysis(symbol, timeframe,
                                                                                                     model_type)

                return interpretation_results

            except Exception as e:
                self.logger.error(f"Помилка інтерпретації моделі {symbol}-{timeframe}-{model_type}: {str(e)}")
                raise

        def _error_analysis(self, symbol: str, timeframe: str, model_type: str) -> Dict[str, Any]:
            """Аналіз помилок моделі"""
            # Отримання даних для аналізу
            data_loader = self.data_preprocessor.get_data_loader(symbol, timeframe, model_type)
            data = data_loader()
            processed_data = self.data_preprocessor.prepare_features(data, symbol)

            # Прогнози моделі
            X = torch.tensor(processed_data.drop(columns=["target"]).values, dtype=torch.float32).to(self.device)
            y_true = processed_data["target"].values

            model_key = self.model_trainer._create_model_key(symbol, timeframe, model_type)
            model = self.model_trainer.models[model_key]

            model.eval()
            with torch.no_grad():
                y_pred = model(X).cpu().numpy().flatten()

            # Розрахунок помилок
            errors = y_pred - y_true

            return {
                'mean_error': float(np.mean(errors)),
                'std_error': float(np.std(errors)),
                'max_error': float(np.max(np.abs(errors))),
                'error_percentiles': {
                    '25%': float(np.percentile(errors, 25)),
                    '50%': float(np.percentile(errors, 50)),
                    '75%': float(np.percentile(errors, 75)),
                    '90%': float(np.percentile(errors, 90)),
                    '95%': float(np.percentile(errors, 95))
                },
                'error_distribution': {
                    'skewness': float(self._calculate_skewness(errors)),
                    'kurtosis': float(self._calculate_kurtosis(errors))
                }
            }

        def _prediction_stability_analysis(self, symbol: str, timeframe: str, model_type: str,
                                           n_runs: int = 10) -> Dict[str, float]:
            """Аналіз стабільності прогнозів"""
            # Отримання тестових даних
            data_loader = self.data_preprocessor.get_data_loader(symbol, timeframe, model_type)
            test_data = data_loader()

            predictions = []

            # Виконуємо кілька прогнозів
            for _ in range(n_runs):
                try:
                    result = self.predict(symbol, timeframe, model_type, test_data, steps_ahead=1)
                    predictions.append(result['predictions'][0])
                except Exception:
                    continue

            if not predictions:
                return {'stability_score': 0.0, 'prediction_variance': float('inf')}

            predictions = np.array(predictions)

            return {
                'stability_score': float(1.0 / (1.0 + np.std(predictions))),  # Вища стабільність = менше варіацій
                'prediction_variance': float(np.var(predictions)),
                'prediction_range': float(np.max(predictions) - np.min(predictions)),
                'coefficient_of_variation': float(np.std(predictions) / np.mean(predictions)) if np.mean(
                    predictions) != 0 else float('inf')
            }

        def _calculate_skewness(self, data: np.ndarray) -> float:
            """Розрахунок асиметрії"""
            mean_val = np.mean(data)
            std_val = np.std(data)
            if std_val == 0:
                return 0.0
            return np.mean(((data - mean_val) / std_val) ** 3)

        def _calculate_kurtosis(self, data: np.ndarray) -> float:
            """Розрахунок ексцесу"""
            mean_val = np.mean(data)
            std_val = np.std(data)
            if std_val == 0:
                return 0.0
            return np.mean(((data - mean_val) / std_val) ** 4) - 3.0

        # ==================== VISUALIZATION ====================

        def plot_model_comparison(self, symbol: str, timeframe: str,
                                  test_data: Optional[pd.DataFrame] = None, save_path: Optional[str] = None) -> None:
            """
            Візуалізація порівняння моделей.

            Args:
                symbol: Символ криптовалюти
                timeframe: Таймфрейм
                test_data: Тестові дані
                save_path: Шлях для збереження графіку
            """
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(20, 12))

            # Порівняння метрик
            comparison_results = self.compare_models(symbol, timeframe, test_data)

            if not comparison_results:
                self.logger.warning(f"Немає навчених моделей для {symbol}-{timeframe}")
                return

            # 1. Порівняння MSE
            model_names = list(comparison_results.keys())
            mse_values = [comparison_results[model].get('mse', 0) for model in model_names]

            axes[0, 0].bar(model_names, mse_values, color=['skyblue', 'lightcoral'])
            axes[0, 0].set_title(f'MSE Comparison - {symbol} {timeframe}')
            axes[0, 0].set_ylabel('MSE')
            axes[0, 0].tick_params(axis='x', rotation=45)

            # 2. Порівняння MAE
            mae_values = [comparison_results[model].get('mae', 0) for model in model_names]

            axes[0, 1].bar(model_names, mae_values, color=['lightgreen', 'orange'])
            axes[0, 1].set_title(f'MAE Comparison - {symbol} {timeframe}')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].tick_params(axis='x', rotation=45)

            # 3. Порівняння R²
            r2_values = [comparison_results[model].get('r2', 0) for model in model_names]

            axes[1, 0].bar(model_names, r2_values, color=['gold', 'purple'])
            axes[1, 0].set_title(f'R² Comparison - {symbol} {timeframe}')
            axes[1, 0].set_ylabel('R² Score')
            axes[1, 0].tick_params(axis='x', rotation=45)

            # 4. Загальна оцінка (комбінована метрика)
            combined_scores = []
            for model in model_names:
                metrics = comparison_results[model]
                # Нормалізована комбінована оцінка (менше MSE і MAE, більше R² = краще)
                score = metrics.get('r2', 0) - (metrics.get('mse', 1) + metrics.get('mae', 1)) / 2
                combined_scores.append(score)

            colors = ['green' if score > 0 else 'red' for score in combined_scores]
            axes[1, 1].bar(model_names, combined_scores, color=colors)
            axes[1, 1].set_title(f'Combined Score - {symbol} {timeframe}')
            axes[1, 1].set_ylabel('Combined Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Графік збережено: {save_path}")

            plt.show()

        def plot_feature_importance(self, symbol: str, timeframe: str, model_type: str,
                                    top_n: int = 15, save_path: Optional[str] = None) -> None:
            """
            Візуалізація важливості ознак.

            Args:
                symbol: Символ криптовалюти
                timeframe: Таймфрейм
                model_type: Тип моделі
                top_n: Кількість топ-ознак для показу
                save_path: Шлях для збереження графіку
            """
            feature_importance = self.feature_importance_analysis(symbol, timeframe, model_type)

            # Сортування за важливістю
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]

            features, importance_values = zip(*sorted_features)

            plt.figure(figsize=(12, 8))
            colors = plt.cm.viridis(np.linspace(0, 1, len(features)))

            bars = plt.barh(range(len(features)), importance_values, color=colors)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance Score')
            plt.title(f'Top {top_n} Feature Importance - {symbol} {timeframe} {model_type.upper()}')
            plt.gca().invert_yaxis()

            # Додавання значень на стовпчики
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + max(importance_values) * 0.01, bar.get_y() + bar.get_height() / 2,
                         f'{width:.4f}', ha='left', va='center', fontsize=9)

            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Графік важливості ознак збережено: {save_path}")

            plt.show()

        def plot_prediction_vs_actual(self, symbol: str, timeframe: str, model_type: str,
                                      test_data: Optional[pd.DataFrame] = None,
                                      n_points: int = 100, save_path: Optional[str] = None) -> None:
            """
            Візуалізація прогнозів проти фактичних значень.

            Args:
                symbol: Символ криптовалюти
                timeframe: Таймфрейм
                model_type: Тип моделі
                test_data: Тестові дані
                n_points: Кількість точок для відображення
                save_path: Шлях для збереження графіку
            """
            # Отримання даних
            if test_data is None:
                data_loader = self.data_preprocessor.get_data_loader(symbol, timeframe, model_type)
                test_data = data_loader()

            processed_data = self.data_preprocessor.prepare_features(test_data, symbol)

            # Вибір останніх n_points
            if len(processed_data) > n_points:
                processed_data = processed_data.tail(n_points)

            # Прогнози
            X = torch.tensor(processed_data.drop(columns=["target"]).values, dtype=torch.float32).to(self.device)
            y_true = processed_data["target"].values

            model_key = self.model_trainer._create_model_key(symbol, timeframe, model_type)
            if model_key not in self.model_trainer.models:
                if not self.model_trainer.load_model(symbol, timeframe, model_type):
                    raise ValueError(f"Модель {model_key} не знайдена")

            model = self.model_trainer.models[model_key]
            model.eval()

            with torch.no_grad():
                y_pred = model(X).cpu().numpy().flatten()

            # Візуалізація
            plt.figure(figsize=(15, 8))

            x_axis = range(len(y_true))

            plt.plot(x_axis, y_true, label='Actual', color='blue', alpha=0.7, linewidth=2)
            plt.plot(x_axis, y_pred, label='Predicted', color='red', alpha=0.7, linewidth=2)

            plt.fill_between(x_axis, y_true, y_pred, alpha=0.2, color='gray', label='Prediction Error')

            plt.xlabel('Time Steps')
            plt.ylabel('Price')
            plt.title(f'Prediction vs Actual - {symbol} {timeframe} {model_type.upper()}')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Додавання метрик на графік
            mse = np.mean((y_pred - y_true) ** 2)
            mae = np.mean(np.abs(y_pred - y_true))
            r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

            textstr = f'MSE: {mse:.6f}\nMAE: {mae:.6f}\nR²: {r2:.4f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                     verticalalignment='top', bbox=props)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Графік прогнозів збережено: {save_path}")

            plt.show()

        # ==================== UTILITY METHODS ====================

        def cleanup_old_models(self, days_old: int = 30) -> int:
            """
            Очищення старих моделей.

            Args:
                days_old: Видалити моделі старше вказаної кількості днів

            Returns:
                Кількість видалених моделей
            """
            if not os.path.exists(self.models_dir):
                return 0

            deleted_count = 0
            cutoff_time = datetime.now().timestamp() - (days_old * 24 * 3600)

            for filename in os.listdir(self.models_dir):
                file_path = os.path.join(self.models_dir, filename)

                if os.path.isfile(file_path):
                    file_time = os.path.getmtime(file_path)

                    if file_time < cutoff_time:
                        try:
                            os.remove(file_path)
                            deleted_count += 1
                            self.logger.info(f"Видалено старий файл: {filename}")
                        except Exception as e:
                            self.logger.warning(f"Не вдалося видалити файл {filename}: {str(e)}")

            return deleted_count

        def get_system_info(self) -> Dict[str, Any]:
            """Отримання інформації про систему"""
            return {
                'device': str(self.device),
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'pytorch_version': torch.__version__,
                'models_directory': self.models_dir,
                'loaded_models_count': len(self.model_trainer.models),
                'total_trained_models': len(self.list_trained_models()),
                'supported_symbols': self.SYMBOLS,
                'supported_timeframes': self.TIMEFRAMES,
                'supported_model_types': self.MODEL_TYPES
            }