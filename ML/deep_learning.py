from dataclasses import dataclass, field
import torch.onnx
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple,  Optional, Any
import os
from datetime import datetime, timedelta
import json
import pickle
import matplotlib.pyplot as plt
from numpy import floating

from ML.LSTM import LSTMModel
from ML.GRU import GRUModel
from ML.transformer import TransformerModel
from ML.ModelTrainer import ModelTrainer
from data.db import DatabaseManager
from timeseriesmodels import ModelEvaluator
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
    sequence_length: int = 60


@dataclass
class CryptoConfig:
    symbols: List[str] = field(default_factory=lambda: ['BTC', 'ETH', 'SOL'])
    timeframes: List[str] = field(default_factory=lambda: ['4h'])
    model_types: List[str] = field(default_factory=lambda: ['lstm', 'gru', 'transformer'])


class DeepLearning:


    SYMBOLS = ['BTC', 'ETH', 'SOL']
    TIMEFRAMES = ['1m', '1h', '4h', '1d', '1w']
    MODEL_TYPES = ['lstm', 'gru', 'transformer']  # Додано transformer

    def __init__(self, models_dir: str = "models/deep_learning"):

        self.logger = CryptoLogger('deep_learning')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.crypto_config = CryptoConfig()
        self.model_config = ModelConfig(input_dim=13)
        # Ініціалізація компонентів
        self.data_preprocessor = DataPreprocessor()
        self.model_trainer = ModelTrainer()
        self.model_evaluator = ModelEvaluator()
        # Ініціалізація моделей з правильними параметрами
        self.lstm = LSTMModel()  # Будуть створені при потребі
        self.gru = GRUModel()
        self.transformer = TransformerModel()

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
                    data: pd.DataFrame, input_dim: int,
                    config: Optional[ModelConfig] = None,
                    validation_split: float = 0.2,
                    patience: int = 10, model_data=None, **kwargs) -> Dict[str, Any]:
        return self.model_trainer.train_model(
            symbol=symbol,
            timeframe=timeframe,
            model_type=model_type,
            data=data,
            input_dim=input_dim,
            config=config,
            validation_split=validation_split,
            patience=patience,
            model_data=model_data,
            **kwargs
        )

    # ==================== ПРОГНОЗУВАННЯ ====================
    def predict(self, symbol: str, timeframe: str, model_type: str,
                input_data: Optional[pd.DataFrame] = None, steps_ahead: int = 1) -> Dict[str, Any]:
        """
        Generate predictions using a trained model with improved error handling and data validation.
        """
        self._validate_inputs(symbol, timeframe, model_type)
        model_key = f"{symbol}_{timeframe}_{model_type}"

        # Ensure model is loaded - improved validation
        if model_key not in self.model_trainer.models:
            if not self.model_trainer.load_model(symbol, timeframe, model_type):
                raise ValueError(f"Model {model_key} not found and could not be loaded")

        # Get model and config from trainer - more reliable approach
        model = self.model_trainer.models[model_key]
        config = self.model_trainer.model_configs.get(model_key)

        if config is None:
            raise ValueError(f"Configuration for model {model_key} not found")

        # Load input data if not provided
        if input_data is None:
            try:
                data_loader = self.data_preprocessor.get_data_loader(symbol, timeframe, model_type)
                input_data = data_loader()
            except Exception as e:
                raise ValueError(f"Failed to load input data: {str(e)}")

        # Store original data for potential inverse transformations
        original_data = input_data.copy()

        # Ensure we only keep numeric columns
        numeric_data = input_data.select_dtypes(include=['number'])
        if len(numeric_data.columns) < len(input_data.columns):
            dropped_cols = set(input_data.columns) - set(numeric_data.columns)
            self.logger.warning(f"Dropped non-numeric columns: {dropped_cols}")

        if numeric_data.empty:
            raise ValueError("No numeric columns found in input data")

        # Get the correct input dimension from model config
        expected_dim = config.input_dim
        if numeric_data.shape[1] > expected_dim:
            # Select only the first 'expected_dim' features
            numeric_data = numeric_data.iloc[:, :expected_dim]
            self.logger.warning(f"Selected first {expected_dim} features from {input_data.shape[1]} available")
        elif numeric_data.shape[1] < expected_dim:
            raise ValueError(f"Input data has {numeric_data.shape[1]} features, but model expects {expected_dim}")

        # Ensure we have enough data for the sequence length
        sequence_length = config.sequence_length
        if len(numeric_data) < sequence_length:
            raise ValueError(f"Input data has {len(numeric_data)} rows, but model requires {sequence_length}")

        # Convert to tensor with proper shape and validation
        try:
            # Take the last sequence_length rows for prediction
            input_array = numeric_data.iloc[-sequence_length:].values.astype(np.float32)

            # Check for NaN or infinite values
            if np.isnan(input_array).any() or np.isinf(input_array).any():
                self.logger.warning("Input data contains NaN or infinite values, filling with zeros")
                input_array = np.nan_to_num(input_array, nan=0.0, posinf=0.0, neginf=0.0)

            # Create tensor with proper dimensions: (batch_size, sequence_length, features)
            X = torch.tensor(input_array, dtype=torch.float32).to(self.device)
            X = X.unsqueeze(0)  # Add batch dimension

            self.logger.info(f"Input tensor shape: {X.shape}, expected: (1, {sequence_length}, {expected_dim})")

        except Exception as e:
            raise ValueError(f"Failed to convert input data to tensor: {str(e)}")

        # Final dimension validation
        if X.shape != (1, sequence_length, expected_dim):
            raise ValueError(
                f"Input tensor shape {X.shape} doesn't match expected shape (1, {sequence_length}, {expected_dim})")

        # Generate predictions
        model.eval()
        raw_predictions = []

        try:
            with torch.no_grad():
                current_input = X.clone()

                for step in range(steps_ahead):
                    # Make prediction
                    pred = model(current_input)

                    # Handle different output shapes
                    if pred.dim() > 1:
                        pred_value = pred.squeeze().item()
                    else:
                        pred_value = pred.item()

                    raw_predictions.append(pred_value)

                    self.logger.debug(f"Step {step + 1}: Raw prediction = {pred_value}")

                    # For multi-step predictions, update the input sequence
                    if steps_ahead > 1 and step < steps_ahead - 1:
                        # Create new input with the prediction
                        new_input = torch.zeros_like(current_input[:, -1:, :])
                        new_input[0, 0, 0] = pred_value  # Assume first feature is the target

                        # Shift the sequence window
                        current_input = torch.cat([current_input[:, 1:, :], new_input], dim=1)

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

        # Transform predictions back to original scale
        try:
            raw_predictions_array = np.array(raw_predictions)

            # Attempt inverse transformation
            try:
                predictions = self.data_preprocessor.inverse_transform_predictions(
                    raw_predictions_array,
                    symbol,
                    timeframe
                )
                self.logger.info(
                    f"Прогнози перетворені у оригінальний масштаб: {raw_predictions_array} -> {predictions}")
            except Exception as e:
                self.logger.warning(f"Не вдалося перетворити прогнози у оригінальний масштаб: {str(e)}")
                predictions = raw_predictions_array

        except Exception as e:
            self.logger.error(f"Failed to process predictions: {str(e)}")
            predictions = raw_predictions

        # Calculate confidence and save predictions
        prediction_timestamp = datetime.now()

        try:
            model_id = self.db_manager._get_model_id(symbol, timeframe, model_type)
            confidence = self._calculate_prediction_confidence(model_key, numeric_data)

            # Save each prediction
            for i, pred in enumerate(predictions):
                try:
                    target_timestamp = self._calculate_target_timestamp(prediction_timestamp, timeframe, i + 1)
                    confidence_low, confidence_high = self._calculate_confidence_intervals(pred, confidence)

                    # Transform confidence intervals if possible
                    try:
                        if hasattr(self.data_preprocessor, 'inverse_transform_confidence_intervals'):
                            confidence_low, confidence_high = self.data_preprocessor.inverse_transform_confidence_intervals(
                                confidence_low, confidence_high, original_data, symbol
                            )
                    except Exception as e:
                        self.logger.warning(f"Не вдалося перетворити інтервали довіри: {str(e)}")

                    prediction_data = {
                        'model_id': model_id,
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'prediction_timestamp': prediction_timestamp,
                        'target_timestamp': target_timestamp,
                        'predicted_value': float(pred),
                        'confidence_interval_low': float(confidence_low),
                        'confidence_interval_high': float(confidence_high)
                    }

                    prediction_id = self.db_manager.save_prediction(prediction_data, symbol)
                    self.logger.info(f"Збережено прогноз з ID: {prediction_id}")

                except Exception as e:
                    self.logger.warning(f"Не вдалося зберегти прогноз {i}: {str(e)}")

        except Exception as e:
            self.logger.error(f"Failed to save predictions: {str(e)}")
            # Continue execution even if saving fails

        return {
            'predictions': np.array(predictions),
            'raw_predictions': np.array(raw_predictions),
            'timestamp': prediction_timestamp,
            'model_key': model_key,
            'confidence': confidence if 'confidence' in locals() else 0.0,
            'input_shape': X.shape if 'X' in locals() else None
        }

    def _calculate_target_timestamp(self, prediction_timestamp: datetime, timeframe: str, steps_ahead: int) -> datetime:
        """Розраховує цільовий час для прогнозу"""
        timeframe_deltas = {
            '1m': timedelta(minutes=1),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '1h': timedelta(hours=1),
            '4h': timedelta(hours=4),
            '1d': timedelta(days=1)
        }

        delta = timeframe_deltas.get(timeframe, timedelta(hours=1))
        return prediction_timestamp + (delta * steps_ahead)

    @staticmethod
    def _calculate_confidence_intervals(prediction: float, confidence: float) -> tuple:
        """Розраховує інтервали довіри для прогнозу"""
        # Простий розрахунок на основі довіри (можна покращити)
        margin = prediction * (1 - confidence) * 0.1  # 10% від різниці до повної довіри
        return (prediction - margin, prediction + margin)

    def _calculate_prediction_confidence(self, model_key: str, data: pd.DataFrame) -> float:
        """Розрахунок довіри до прогнозу на основі метрик моделі"""
        if model_key in self.model_metrics:
            metrics = self.model_metrics[model_key]
            # Простий розрахунок довіри на основі R²
            r2 = metrics.get('r2_score', 0)
            return max(0, min(1, r2))  # Обмежуємо від 0 до 1
        return 0.5  # Середня довіра, якщо метрики недоступні

    def predict_multiple_steps(self, symbol: str, timeframe: str, model_type: str,
                               steps: int = 10, input_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:

        # Прогнозування на кілька кроків вперед
        result = self.predict(symbol, timeframe, model_type, steps_ahead=steps, input_data=input_data)
        predictions = result['predictions']

        # Створення DataFrame з прогнозами
        forecast_df = pd.DataFrame({
            'step': range(1, steps + 1),
            'prediction': predictions,
            'timestamp': [result['timestamp']] * steps,
            'confidence': [result['confidence']] * steps
        })

        return forecast_df

    def predict_all_symbols(self, timeframe: str, model_type: str,
                            steps_ahead: int = 1) -> Dict[str, np.ndarray]:
        """Прогнозування для всіх символів з використанням одного типу моделі"""
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
                         model_types: Optional[List[str]] = None,
                         weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Ансамблевий прогноз (комбінація різних типів моделей)"""
        if model_types is None:
            model_types = self.MODEL_TYPES

        if weights is None:
            # Рівномірний розподіл ваг між доступними моделями
            weight_per_model = 1.0 / len(model_types)
            weights = {model_type: weight_per_model for model_type in model_types}

        predictions = {}
        ensemble_pred = np.zeros(steps_ahead)
        total_weight = 0

        for model_type in model_types:
            try:
                result = self.predict(symbol, timeframe, model_type, steps_ahead=steps_ahead)
                predictions[model_type] = result['predictions']
                weight = weights.get(model_type, 0)
                ensemble_pred += weight * result['predictions']
                total_weight += weight
            except Exception as e:
                self.logger.warning(f"Модель {model_type} недоступна: {str(e)}")
                continue

        # Нормалізація, якщо не всі моделі доступні
        if total_weight > 0:
            ensemble_pred /= total_weight

        return {
            'ensemble_predictions': ensemble_pred,
            'individual_predictions': predictions,
            'weights': weights,
            'timestamp': datetime.now()
        }

    # ==================== ОЦІНКА МОДЕЛЕЙ ====================

    def evaluate_model(self, model_key: str, test_data: pd.Series, use_rolling_validation: bool = True,
                       window_size: int = 100, step: int = 20, forecast_horizon: int = 10,
                       apply_inverse_transforms: bool = False) -> Dict:
        return self.model_evaluator.evaluate_model(model_key, test_data, use_rolling_validation, window_size, step)

    def model_performance_report(self, symbol: Optional[str] = None,
                                 timeframe: Optional[str] = None,
                                 model_type: Optional[str] = None) -> pd.DataFrame:
        """Звіт про ефективність моделей"""
        performance_data = []

        symbols = [symbol] if symbol else self.SYMBOLS
        timeframes = [timeframe] if timeframe else self.TIMEFRAMES
        model_types = [model_type] if model_type else self.MODEL_TYPES

        for sym in symbols:
            for tf in timeframes:
                for mt in model_types:
                    try:
                        metrics = self.get_model_metrics(sym, tf, mt)
                        if metrics:
                            performance_data.append({
                                'symbol': sym,
                                'timeframe': tf,
                                'model_type': mt,
                                **metrics
                            })
                    except Exception as e:
                        self.logger.warning(f"Не вдалося отримати метрики для {sym}-{tf}-{mt}: {str(e)}")
                        continue

        return pd.DataFrame(performance_data)

    # ==================== УПРАВЛІННЯ МОДЕЛЯМИ ====================

    def delete_model(self, symbol: str, timeframe: str, model_type: str) -> bool:
        """Видалення моделі"""
        self._validate_inputs(symbol, timeframe, model_type)

        try:
            model_key = self.model_trainer._create_model_key(symbol, timeframe, model_type)

            # Видалення з пам'яті
            if model_key in self.models:
                del self.models[model_key]
            if model_key in self.model_configs:
                del self.model_configs[model_key]
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

            metrics_file = os.path.join(self.models_dir, f"{model_key}_metrics.json")
            if os.path.exists(metrics_file):
                os.remove(metrics_file)

            self.logger.info(f"Модель {model_key} успішно видалена")
            return True

        except Exception as e:
            self.logger.error(f"Помилка видалення моделі {symbol}-{timeframe}-{model_type}: {str(e)}")
            return False

    def get_model_metrics(self, symbol: str, timeframe: str, model_type: str) -> Dict[str, float]:

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
        for model_key in self.models.keys():
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

        if symbol not in self.SYMBOLS:
            raise ValueError(f"Непідтримуваний символ: {symbol}. Доступні символи: {self.SYMBOLS}")

        if timeframe not in self.TIMEFRAMES:
            raise ValueError(f"Непідтримуваний таймфрейм: {timeframe}. Доступні таймфрейми: {self.TIMEFRAMES}")

        if model_type.lower() not in self.MODEL_TYPES:
            raise ValueError(f"Непідтримуваний тип моделі: {model_type}. Доступні типи: {self.MODEL_TYPES}")

        return True


    def online_learning(self, symbol: str, timeframe: str, model_type: str,
                        new_data: pd.DataFrame, epochs: int = 10) -> Dict[str, Any]:

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

        return self.model_trainer.train_all_models(
            symbols=symbols,
            timeframes=timeframes,
            model_types=model_types,
            **training_params
        )

    def cross_validate_model(self, symbol: str, timeframe: str, model_type: str,
                             k_folds: int = 5, **model_params) -> Dict[str, List[float]]:

        return self.model_trainer.cross_validate_model(
            symbol=symbol,
            timeframe=timeframe,
            model_type=model_type,
            k_folds=k_folds,
            **model_params
        )

    def get_training_summary(self) -> Dict[str, Any]:

        return self.model_trainer.get_training_summary()

        # ==================== HYPERPARAMETER OPTIMIZATION ====================

    def hyperparameter_optimization(self, symbol: str, timeframe: str, model_type: str,
                                    param_space: Dict[str, List], optimization_method: str = 'grid_search',
                                    cv_folds: int = 3, max_iterations: int = 50,
                                    validation_split: float = 0.2,
                                    target_column: str = 'target_close_1') -> Dict[str, Any]:
        """
        Оптимізація гіперпараметрів для моделі

        Args:
            symbol: Символ активу
            timeframe: Таймфрейм
            model_type: Тип моделі
            param_space: Простір параметрів для оптимізації
            optimization_method: Метод оптимізації ('grid_search', 'random_search', 'bayesian')
            cv_folds: Кількість фолдів для крос-валідації
            max_iterations: Максимальна кількість ітерацій для випадкового/байєсівського пошуку
            validation_split: Розмір валідаційної вибірки
            target_column: Назва цільового стовпця

        Returns:
            Dict з результатами оптимізації
        """
        self._validate_inputs(symbol, timeframe, model_type)

        try:
            # Спочатку отримуємо сирі дані для оптимізації (як DataFrame)
            self.logger.info(f"Завантаження сирих даних для оптимізації {symbol}-{timeframe}-{model_type}")

            data_loader = self.data_preprocessor.get_data_loader(symbol, timeframe, model_type)
            raw_data = data_loader()

            if raw_data is None or len(raw_data) == 0:
                raise ValueError(f"Не вдалося завантажити дані для {symbol}-{timeframe}")

            # Конвертація в DataFrame якщо потрібно
            if isinstance(raw_data, list):
                processed_data = pd.DataFrame(raw_data)
            elif isinstance(raw_data, pd.DataFrame):
                processed_data = raw_data.copy()
            else:
                raise ValueError(f"Невідомий тип даних: {type(raw_data)}")

            self.logger.info(f"Дані для оптимізації підготовлено: {processed_data.shape}")

            # Також підготовляємо конфігурацію моделі для збереження
            try:
                X_train, X_val, y_train, y_val, model_config = self.data_preprocessor.prepare_data_with_config(
                    symbol=symbol,
                    timeframe=timeframe,
                    model_type=model_type,
                    validation_split=validation_split,
                    target_column=target_column
                )
                self.logger.info(f"Конфігурація моделі створена: input_dim={model_config.input_dim}")
            except Exception as config_error:
                self.logger.warning(f"Не вдалося створити конфігурацію моделі: {config_error}")
                model_config = None

            # Ініціалізація результатів
            best_params = {}
            best_score = float('inf')
            optimization_history = []

            # Вибір методу оптимізації
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

            # Створення результатів оптимізації
            optimization_results = {
                'best_params': best_params,
                'best_score': best_score,
                'optimization_history': optimization_history,
                'method': optimization_method,
                'data_info': {
                    'total_samples': len(processed_data),
                    'features': len(processed_data.columns),
                    'validation_split': validation_split,
                    'target_column': target_column
                },
                'timestamp': datetime.now().isoformat()
            }

            # Додаємо інформацію про конфігурацію моделі, якщо вона доступна
            if model_config is not None:
                optimization_results['model_config'] = {
                    'input_dim': model_config.input_dim,
                    'hidden_dim': model_config.hidden_dim,
                    'num_layers': model_config.num_layers,
                    'output_dim': model_config.output_dim,
                    'dropout': model_config.dropout,
                    'learning_rate': model_config.learning_rate,
                    'batch_size': model_config.batch_size,
                    'epochs': model_config.epochs,
                    'sequence_length': model_config.sequence_length
                }

            # Збереження результатів у файл
            results_file = os.path.join(
                self.models_dir,
                f"{symbol}_{timeframe}_{model_type}_optimization.json"
            )

            try:
                os.makedirs(self.models_dir, exist_ok=True)
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(optimization_results, f, indent=2, default=str, ensure_ascii=False)

                self.logger.info(f"Результати оптимізації збережено в: {results_file}")
            except Exception as save_error:
                self.logger.warning(f"Не вдалося зберегти результати: {save_error}")

            self.logger.info(
                f"Оптимізація гіперпараметрів для {symbol}-{timeframe}-{model_type} завершена. "
                f"Найкращий результат: {best_score:.6f}"
            )

            return optimization_results

        except Exception as e:
            self.logger.error(f"Помилка оптимізації гіперпараметрів для {symbol}-{timeframe}-{model_type}: {str(e)}")
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
        """Аналіз важливості ознак для моделі"""

        self._validate_inputs(symbol, timeframe, model_type)

        # Завантаження моделі
        model_key = self.model_trainer._create_model_key(symbol, timeframe, model_type)
        if model_key not in self.model_trainer.models:
            if not self.model_trainer.load_model(symbol, timeframe, model_type):
                raise ValueError(f"Модель {model_key} не знайдена")

        model = self.model_trainer.models[model_key]

        # Отримання оброблених даних для аналізу
        try:
            # Завантажуємо сирі дані
            data_loader = self.data_preprocessor.get_data_loader(symbol, timeframe, model_type)
            raw_data = data_loader()

            if raw_data is None or len(raw_data) == 0:
                raise ValueError(f"Не вдалося завантажити дані для {symbol}-{timeframe}")

            # Конвертуємо в DataFrame якщо потрібно
            if isinstance(raw_data, list):
                processed_data = pd.DataFrame(raw_data)
            elif isinstance(raw_data, pd.DataFrame):
                processed_data = raw_data.copy()
            else:
                raise ValueError(f"Невідомий тип даних: {type(raw_data)}")

            # Перевіряємо наявність цільової змінної
            if "target" not in processed_data.columns:
                # Якщо немає стандартного 'target', шукаємо target_close_1 або інші варіанти
                target_candidates = [col for col in processed_data.columns if col.startswith('target')]
                if target_candidates:
                    # Перейменовуємо першу знайдену цільову змінну на 'target'
                    processed_data = processed_data.rename(columns={target_candidates[0]: 'target'})
                else:
                    raise ValueError("Цільова змінна не знайдена в даних")

            self.logger.info(f"Підготовлено дані для аналізу важливості: {processed_data.shape}")

        except Exception as e:
            self.logger.error(f"Помилка підготовки даних для аналізу важливості: {e}")
            raise

        # Аналіз важливості методом пермутації
        try:
            feature_names = processed_data.drop(columns=["target"]).columns.tolist()
            importance_scores = self._permutation_importance(model, processed_data, feature_names)

            self.logger.info(f"Завершено аналіз важливості для {len(feature_names)} ознак")
            return dict(zip(feature_names, importance_scores))

        except Exception as e:
            self.logger.error(f"Помилка аналізу важливості ознак: {e}")
            raise

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

        # Завантаження моделі
        model_key = self.model_trainer._create_model_key(symbol, timeframe, model_type)
        if model_key not in self.model_trainer.models:
            if not self.model_trainer.load_model(symbol, timeframe, model_type):
                raise ValueError(f"Модель {model_key} не знайдена")

        model = self.model_trainer.models[model_key]

        # Отримання даних для аналізу
        try:
            # Завантажуємо сирі дані
            data_loader = self.data_preprocessor.get_data_loader(symbol, timeframe, model_type)
            raw_data = data_loader()

            if raw_data is None or len(raw_data) == 0:
                raise ValueError(f"Не вдалося завантажити дані для {symbol}-{timeframe}")

            # Конвертуємо в DataFrame якщо потрібно
            if isinstance(raw_data, list):
                processed_data = pd.DataFrame(raw_data)
            elif isinstance(raw_data, pd.DataFrame):
                processed_data = raw_data.copy()
            else:
                raise ValueError(f"Невідомий тип даних: {type(raw_data)}")

            # Перевіряємо наявність цільової змінної
            if "target" not in processed_data.columns:
                # Якщо немає стандартного 'target', шукаємо target_close_1 або інші варіанти
                target_candidates = [col for col in processed_data.columns if col.startswith('target')]
                if target_candidates:
                    # Перейменовуємо першу знайдену цільову змінну на 'target'
                    processed_data = processed_data.rename(columns={target_candidates[0]: 'target'})
                else:
                    raise ValueError("Цільова змінна не знайдена в даних")

            self.logger.info(f"Підготовлено дані для аналізу помилок: {processed_data.shape}")

        except Exception as e:
            self.logger.error(f"Помилка підготовки даних для аналізу помилок: {e}")
            raise

        # Прогнози моделі
        try:
            X = torch.tensor(
                processed_data.drop(columns=["target"]).values,
                dtype=torch.float32
            ).to(self.device)
            y_true = processed_data["target"].values

            model.eval()
            with torch.no_grad():
                y_pred = model(X).cpu().numpy().flatten()

            # Розрахунок помилок
            errors = y_pred - y_true

            result = {
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

            self.logger.info(f"Завершено аналіз помилок для {symbol}-{timeframe}-{model_type}")
            return result

        except Exception as e:
            self.logger.error(f"Помилка аналізу помилок моделі: {e}")
            raise

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

    def _calculate_skewness(self, data: np.ndarray) -> float | floating[Any]:
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

        # Отримання даних
        if test_data is None:
            data_loader = self.data_preprocessor.get_data_loader(symbol, timeframe, model_type)
            test_data = data_loader()

        # Якщо у test_data є стовпець target, використовуємо його
        if 'target' in test_data.columns or 'target_close_1' in test_data.columns:
            # Визначаємо назву цільового стовпця
            target_col = 'target' if 'target' in test_data.columns else 'target_close_1'

            # Вибір останніх n_points
            if len(test_data) > n_points:
                processed_data = test_data.tail(n_points)
            else:
                processed_data = test_data

            # Підготовка даних
            X = torch.tensor(processed_data.drop(columns=[target_col]).values,
                             dtype=torch.float32).to(self.device)
            y_true = processed_data[target_col].values
        else:
            # Використовуємо prepare_data_with_config для підготовки даних
            X_train, X_val, y_train, y_val, config = self.data_preprocessor.prepare_data_with_config(
                symbol, timeframe, model_type
            )

            # Об'єднуємо тренувальні та валідаційні дані
            X_full = torch.cat([X_train, X_val], dim=0)
            y_full = torch.cat([y_train, y_val], dim=0)

            # Вибір останніх n_points
            if len(y_full) > n_points:
                X = X_full[-n_points:].to(self.device)
                y_true = y_full[-n_points:].cpu().numpy()
            else:
                X = X_full.to(self.device)
                y_true = y_full.cpu().numpy()

        # Прогнози
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

    def full_training_pipeline(self,
                               symbols: Optional[List[str]] = None,
                               timeframes: Optional[List[str]] = None,
                               model_type: Optional[str] = None,
                               validation_split: float = 0.2,
                               epochs: int = 100,
                               batch_size: int = 32) -> Dict[str, Dict[str, Any]]:
        """
        Complete training pipeline from data loading to model evaluation.

        Args:
            symbols: List of cryptocurrency symbols to train on
            timeframes: List of timeframes to train on
            model_type: Model type to train
            validation_split: Ratio of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training

        Returns:
            Dictionary containing training results for each model
        """
        results = {}

        # Use default values if not specified
        symbols = symbols or self.SYMBOLS
        timeframes = timeframes or self.TIMEFRAMES
        model_type = model_type or self.MODEL_TYPES[0]  # Use first model type as default

        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    model_key = f"{symbol}_{timeframe}_{model_type}"
                    self.logger.info(f"Starting training for {model_key}")

                    # 1. Data Loading
                    data_loader = self.data_preprocessor.get_data_loader(symbol, timeframe, model_type)
                    raw_data = data_loader()

                    # 2. Feature Engineering
                    features = self.data_preprocessor.prepare_data_with_config(raw_data, symbol, model_type)
                    input_dim = features.shape[1] - 1  # Subtract target column

                    # 3. Model Training
                    train_result = self.train_model(
                        symbol=symbol,
                        timeframe=timeframe,
                        model_type=model_type,
                        data=features,
                        input_dim=input_dim,
                        validation_split=validation_split,
                        epochs=epochs,
                        batch_size=batch_size
                    )

                    # 4. Model Evaluation
                    metrics = self.evaluate_model(symbol, timeframe, model_type, features)

                    # 5. Save Results
                    results[model_key] = {
                        'training_result': train_result,
                        'metrics': metrics
                    }

                    self.logger.info(f"Completed training for {model_key} with RMSE: {metrics.get('RMSE')}")

                except Exception as e:
                    self.logger.error(f"Error training {symbol}-{timeframe}-{model_type}: {str(e)}")
                    results[f"{symbol}_{timeframe}_{model_type}"] = {'error': str(e)}
                    continue

        return results

    def prediction_pipeline(self,
                                symbol: str,
                                timeframe: str,
                                model_type: str,
                                steps_ahead: int = 1,
                                input_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
            """
            Complete prediction pipeline from data loading to prediction storage.

            Args:
                symbol: Cryptocurrency symbol
                timeframe: Timeframe for prediction
                model_type: Type of model to use
                steps_ahead: Number of steps to predict ahead
                input_data: Optional input data (uses latest data if None)

            Returns:
                Dictionary containing predictions and metadata
            """
            try:
                # 1. Load or prepare data
                if input_data is None:
                    data_loader = self.data_preprocessor.get_data_loader(symbol, timeframe, model_type)
                    input_data = data_loader()

                # 2. Make prediction
                prediction_result = self.predict(
                    symbol=symbol,
                    timeframe=timeframe,
                    model_type=model_type,
                    input_data=input_data,
                    steps_ahead=steps_ahead
                )

                self.logger.info(f"Prediction completed for {symbol}-{timeframe}-{model_type}")

                # 3. Format results
                formatted_predictions = []
                for i, pred in enumerate(prediction_result['predictions']):
                    formatted_predictions.append({
                        'step': i + 1,
                        'prediction': float(pred),
                        'confidence': prediction_result['confidence']
                    })

                return {
                    'status': 'success',
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'model_type': model_type,
                    'predictions': formatted_predictions,
                    'timestamp': datetime.now().isoformat()
                }

            except Exception as e:
                self.logger.error(f"Prediction failed for {symbol}-{timeframe}-{model_type}: {str(e)}")
                return {
                    'status': 'error',
                    'error': str(e),
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'model_type': model_type
                }

    def online_learning_pipeline(self,model_key: str,
                                     symbol: str,
                                     timeframe: str,
                                     model_type: str,
                                     new_data: pd.DataFrame,
                                     epochs: int = 10) -> Dict[str, Any]:
            """
            Online learning pipeline to update models with new data.

            Args:
                symbol: Cryptocurrency symbol
                timeframe: Timeframe for the model
                model_type: Type of model to update
                new_data: New data for online learning
                epochs: Number of epochs for online training

            Returns:
                Dictionary containing online learning results
            """
            try:
                # 1. Prepare features
                processed_data = self.data_preprocessor.prepare_data_with_config(new_data, symbol, model_type)

                # 2. Perform online learning
                result = self.online_learning(
                    symbol=symbol,
                    timeframe=timeframe,
                    model_type=model_type,
                    new_data=processed_data,
                    epochs=epochs
                )

                self.logger.info(f"Online learning completed for {symbol}-{timeframe}-{model_type}")

                return {
                    'status': 'success',
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'model_type': model_type,
                    'metrics': result['metrics'],
                    'training_duration': result['training_duration']
                }

            except Exception as e:
                self.logger.error(f"Online learning failed for {symbol}-{timeframe}-{model_type}: {str(e)}")
                return {
                    'status': 'error',
                    'error': str(e),
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'model_type': model_type
                }

    def model_evaluation_pipeline(self,model_key: str,
                                      symbol: str,
                                      timeframe: str,
                                      model_type: str,
                                      test_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
            """
            Complete model evaluation pipeline.

            Args:
                symbol: Cryptocurrency symbol
                timeframe: Timeframe for the model
                model_type: Type of model to evaluate
                test_data: Optional test data (uses latest data if None)

            Returns:
                Dictionary containing evaluation metrics
            """
            try:
                # 1. Load or prepare test data
                if test_data is None:
                    data_loader = self.data_preprocessor.get_data_loader(symbol, timeframe, model_type)
                    test_data = data_loader()

                # 2. Evaluate model
                metrics = self.evaluate_model(
                    model_key =model_key,
                    test_data=test_data
                )

                self.logger.info(f"Evaluation completed for {symbol}-{timeframe}-{model_type}")

                return {
                    'status': 'success',
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'model_type': model_type,
                    'metrics': metrics
                }

            except Exception as e:
                self.logger.error(f"Evaluation failed for {symbol}-{timeframe}-{model_type}: {str(e)}")
                return {
                    'status': 'error',
                    'error': str(e),
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'model_type': model_type
                }

    def model_interpretation_pipeline(self,
                                          symbol: str,
                                          timeframe: str,
                                          model_type: str) -> Dict[str, Any]:
            """
            Model interpretation pipeline including feature importance and error analysis.

            Args:
                symbol: Cryptocurrency symbol
                timeframe: Timeframe for the model
                model_type: Type of model to interpret

            Returns:
                Dictionary containing interpretation results
            """
            try:
                # 1. Get feature importance
                feature_importance = self.feature_importance_analysis(
                    symbol=symbol,
                    timeframe=timeframe,
                    model_type=model_type
                )

                # 2. Get model interpretation
                interpretation = self.model_interpretation(
                    symbol=symbol,
                    timeframe=timeframe,
                    model_type=model_type
                )

                self.logger.info(f"Interpretation completed for {symbol}-{timeframe}-{model_type}")

                return {
                    'status': 'success',
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'model_type': model_type,
                    'feature_importance': feature_importance,
                    'interpretation': interpretation
                }

            except Exception as e:
                self.logger.error(f"Interpretation failed for {symbol}-{timeframe}-{model_type}: {str(e)}")
                return {
                    'status': 'error',
                    'error': str(e),
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'model_type': model_type
                }

    def ensemble_prediction_pipeline(self,
                                         symbol: str,
                                         timeframe: str,
                                         steps_ahead: int = 1,
                                         weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
            """
            Ensemble prediction pipeline combining predictions from multiple models.

            Args:
                symbol: Cryptocurrency symbol
                timeframe: Timeframe for prediction
                steps_ahead: Number of steps to predict ahead
                weights: Optional weights for each model type

            Returns:
                Dictionary containing ensemble predictions
            """
            try:
                # 1. Get ensemble prediction
                ensemble_result = self.ensemble_predict(
                    symbol=symbol,
                    timeframe=timeframe,
                    steps_ahead=steps_ahead,
                    weights=weights
                )

                self.logger.info(f"Ensemble prediction completed for {symbol}-{timeframe}")

                # 2. Format results
                formatted_predictions = []
                for i in range(steps_ahead):
                    formatted_predictions.append({
                        'step': i + 1,
                        'ensemble_prediction': float(ensemble_result['ensemble_predictions'][i]),
                        'individual_predictions': {
                            model_type: float(preds[i])
                            for model_type, preds in ensemble_result['individual_predictions'].items()
                        }
                    })

                return {
                    'status': 'success',
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'predictions': formatted_predictions,
                    'weights': ensemble_result['weights'],
                    'timestamp': datetime.now().isoformat()
                }

            except Exception as e:
                self.logger.error(f"Ensemble prediction failed for {symbol}-{timeframe}: {str(e)}")
                return {
                    'status': 'error',
                    'error': str(e),
                    'symbol': symbol,
                    'timeframe': timeframe
                }


def main():
    """
    Головна функція для виконання повного циклу роботи з моделями:
    - Ініціалізація
    - Навчання всіх моделей
    - Прогнозування
    - Оцінка
    - Візуалізація
    - Управління моделями
    """
    # Ініціалізація
    dl_pipeline = DeepLearning()
    logger = dl_pipeline.logger

    # 1. Вивід інформації про систему
    logger.info("=== Початок роботи Deep Learning Pipeline ===")
    system_info = dl_pipeline.get_system_info()
    logger.info(f"Інформація про систему:\n{json.dumps(system_info, indent=2)}")

    # 2. Навчання всіх моделей для всіх символів і таймфреймів
    logger.info("\n=== Навчання моделей ===")
    training_results = dl_pipeline.train_all_models(
        symbols=CryptoConfig().symbols,
        timeframes=CryptoConfig().timeframes,
        model_types=CryptoConfig().model_types,
        epochs=50,
        batch_size=64,
        save_models=True
    )

    # 3. Вивід списку навчених моделей
    trained_models = dl_pipeline.list_trained_models()
    logger.info(f"Навчені моделі:\n{pd.DataFrame(trained_models).to_string()}")

    # 4. Прогнозування для всіх моделей
    logger.info("\n=== Прогнозування ===")
    for symbol in CryptoConfig().symbols:
        for timeframe in CryptoConfig().timeframes:
            for model_type in CryptoConfig().model_types:
                try:
                    # Одноетапний прогноз
                    prediction = dl_pipeline.predict(
                        symbol=symbol,
                        timeframe=timeframe,
                        model_type=model_type,
                        steps_ahead=3
                    )
                    logger.info(f"{symbol}-{timeframe}-{model_type} прогноз: {prediction['predictions']}")

                    # Багатоетапний прогноз
                    multi_step = dl_pipeline.predict_multiple_steps(
                        symbol=symbol,
                        timeframe=timeframe,
                        model_type=model_type,
                        steps=5
                    )
                    logger.info(f"{symbol}-{timeframe}-{model_type} багатоетапний прогноз:\n{multi_step}")

                except Exception as e:
                    logger.error(f"Помилка прогнозування для {symbol}-{timeframe}-{model_type}: {str(e)}")

    # 5. Ансамблеві прогнози
    logger.info("\n=== Ансамблеві прогнози ===")
    for symbol in CryptoConfig().symbols:
        for timeframe in CryptoConfig().timeframes:
            try:
                ensemble_pred = dl_pipeline.ensemble_prediction_pipeline(
                    symbol=symbol,
                    timeframe=timeframe,
                    steps_ahead=3,
                    weights={'lstm': 0.4, 'gru': 0.3, 'transformer': 0.3}
                )
                logger.info(f"{symbol}-{timeframe} ансамблевий прогноз: {ensemble_pred}")
            except Exception as e:
                logger.error(f"Помилка ансамблевого прогнозу для {symbol}-{timeframe}: {str(e)}")

    # 6. Оцінка моделей
    logger.info("\n=== Оцінка моделей ===")
    evaluation_results = {}
    for symbol in CryptoConfig().symbols:
        for timeframe in CryptoConfig().timeframes:
            for model_type in CryptoConfig().model_types:
                try:
                    model_key = f"{symbol}_{timeframe}_{model_type}"
                    metrics = dl_pipeline.evaluate_model(
                        model_key=model_key,
                        test_data=None
                    )
                    evaluation_results[model_key] = metrics
                    logger.info(f"{model_key} метрики: {metrics}")
                except Exception as e:
                    logger.error(f"Помилка оцінки для {symbol}-{timeframe}-{model_type}: {str(e)}")

    # 7. Порівняння моделей
    logger.info("\n=== Порівняння моделей ===")
    for symbol in CryptoConfig().symbols:
        for timeframe in CryptoConfig().timeframes:
            try:
                comparison = dl_pipeline.compare_models(symbol, timeframe)
                logger.info(f"{symbol}-{timeframe} порівняння моделей:\n{pd.DataFrame(comparison).T.to_string()}")
            except Exception as e:
                logger.error(f"Помилка порівняння для {symbol}-{timeframe}: {str(e)}")

    # 8. Аналіз важливості ознак
    logger.info("\n=== Аналіз важливості ознак ===")
    for symbol in CryptoConfig().symbols:
        for timeframe in CryptoConfig().timeframes:
            for model_type in CryptoConfig().model_types:
                try:
                    feature_imp = dl_pipeline.feature_importance_analysis(symbol, timeframe, model_type)
                    top_features = dict(sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)[:5])
                    logger.info(f"{symbol}-{timeframe}-{model_type} топ-5 ознак: {top_features}")
                except Exception as e:
                    logger.error(f"Помилка аналізу ознак для {symbol}-{timeframe}-{model_type}: {str(e)}")

    # 9. Інтерпретація моделей
    logger.info("\n=== Інтерпретація моделей ===")
    for symbol in CryptoConfig().symbols:
        for timeframe in CryptoConfig().timeframes:
            for model_type in CryptoConfig().model_types:
                try:
                    interpretation = dl_pipeline.model_interpretation(symbol, timeframe, model_type)
                    logger.info(f"{symbol}-{timeframe}-{model_type} інтерпретація завершена")
                except Exception as e:
                    logger.error(f"Помилка інтерпретації для {symbol}-{timeframe}-{model_type}: {str(e)}")

    # 10. Оптимізація гіперпараметрів
    logger.info("\n=== Оптимізація гіперпараметрів ===")
    param_space = {
        'hidden_dim': [32, 64, 128],
        'num_layers': [1, 2],
        'learning_rate': [0.001, 0.01],
        'dropout': [0.1, 0.2]
    }

    for symbol in CryptoConfig().symbols:
        for timeframe in CryptoConfig().timeframes:
            for model_type in CryptoConfig().model_types:
                try:
                    opt_result = dl_pipeline.hyperparameter_optimization(
                        symbol=symbol,
                        timeframe=timeframe,
                        model_type=model_type,
                        param_space=param_space,
                        optimization_method='random_search',
                        max_iterations=10
                    )
                    logger.info(f"{symbol}-{timeframe}-{model_type} оптимальні параметри: {opt_result['best_params']}")
                except Exception as e:
                    logger.error(f"Помилка оптимізації для {symbol}-{timeframe}-{model_type}: {str(e)}")

    # 11. Візуалізація
    logger.info("\n=== Генерація візуалізацій ===")
    os.makedirs("visualizations", exist_ok=True)
    for symbol in CryptoConfig().symbols:
        for timeframe in CryptoConfig().timeframes:
            for model_type in CryptoConfig().model_types:
                try:
                    # Порівняння моделей
                    comp_path = f"visualizations/{symbol}_{timeframe}_comparison.png"
                    dl_pipeline.plot_model_comparison(symbol, timeframe, save_path=comp_path)

                    # Важливість ознак
                    feat_path = f"visualizations/{symbol}_{timeframe}_{model_type}_features.png"
                    dl_pipeline.plot_feature_importance(
                        symbol, timeframe, model_type,
                        top_n=10, save_path=feat_path
                    )

                    # Прогнози vs фактичні значення
                    pred_path = f"visualizations/{symbol}_{timeframe}_{model_type}_predictions.png"
                    dl_pipeline.plot_prediction_vs_actual(
                        symbol, timeframe, model_type,
                        n_points=50, save_path=pred_path
                    )

                    # Історія навчання
                    hist_path = f"visualizations/{symbol}_{timeframe}_{model_type}_history.png"
                    dl_pipeline.plot_training_history(symbol, timeframe, model_type)

                    logger.info(f"Візуалізації для {symbol}-{timeframe}-{model_type} збережено")
                except Exception as e:
                    logger.error(f"Помилка візуалізації для {symbol}-{timeframe}-{model_type}: {str(e)}")

    # 12. Управління моделями
    logger.info("\n=== Управління моделями ===")
    for symbol in CryptoConfig().symbols:
        for timeframe in CryptoConfig().timeframes:
            for model_type in CryptoConfig().model_types:
                try:
                    # Інформація про модель
                    model_info = dl_pipeline.get_model_info(symbol, timeframe, model_type)
                    logger.info(f"{symbol}-{timeframe}-{model_type} інформація отримана")

                    # Експорт моделі
                    export_path = dl_pipeline.export_model_for_production(
                        symbol=symbol,
                        timeframe=timeframe,
                        model_type=model_type,
                        export_format='onnx'
                    )
                    logger.info(f"{symbol}-{timeframe}-{model_type} експортовано до {export_path}")
                except Exception as e:
                    logger.error(f"Помилка управління для {symbol}-{timeframe}-{model_type}: {str(e)}")

    # 13. Очищення старих моделей
    deleted_count = dl_pipeline.cleanup_old_models(days_old=7)
    logger.info(f"Видалено {deleted_count} старих моделей")

    # 14. Звіт про продуктивність
    logger.info("\n=== Звіт про продуктивність ===")
    perf_report = dl_pipeline.model_performance_report()
    logger.info(f"Звіт про продуктивність:\n{perf_report.to_string()}")

    logger.info("=== Deep Learning Pipeline завершено успішно ===")


if __name__ == "__main__":
    main()