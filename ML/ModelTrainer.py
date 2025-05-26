import os
from dataclasses import dataclass, field
from datetime import time, datetime
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from ML.DataPreprocessor import DataPreprocessor
from ML.base import BaseDeepModel
from data.db import DatabaseManager
from featureengineering.feature_engineering import FeatureEngineering
from utils.logger import CryptoLogger


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


@dataclass
class CryptoConfig:
    symbols: List[str] = field(default_factory=lambda: ['BTC', 'ETH', 'SOL'])
    timeframes: List[str] = field(default_factory=lambda: ['1m', '1h', '4h', '1d', '1w'])
    model_types: List[str] = field(default_factory=lambda: ['lstm', 'gru', 'transformer'])


class ModelTrainer:


    def __init__(self, device: Optional[torch.device] = None, models_dir: str = "models"):

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: Dict[str, BaseDeepModel] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.model_metrics: Dict[str, Dict[str, float]] = {}
        self.best_models: Dict[str, BaseDeepModel] = {}
        self.training_history: Dict[str, List[float]] = {}
        self.models_dir = models_dir
        self.db_manager = DatabaseManager()
        # Ініціалізація компонентів
        self.feature_engineering = FeatureEngineering()
        self.processor = DataPreprocessor()

        # Налаштування логування
        self.logger = CryptoLogger('trainer')


        # Створення директорії для моделей
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def create_model_config(self, symbol: str, timeframe: str, model_type: str,
                            input_dim: int, **kwargs) -> ModelConfig:

        # Базова конфігурація
        config = ModelConfig(
            input_dim=input_dim,
            sequence_length=ModelConfig.get_sequence_length_for_timeframe(timeframe)
        )

        # Специфічні налаштування для різних типів моделей
        if model_type.lower() == 'transformer':
            config.hidden_dim = kwargs.get('hidden_dim', 128)  # Трансформер потребує більше параметрів
            config.num_heads = kwargs.get('num_heads', 8)
            config.dim_feedforward = kwargs.get('dim_feedforward', 256)
            config.num_layers = kwargs.get('num_layers', 3)
            config.learning_rate = kwargs.get('learning_rate', 0.0005)  # Менша швидкість для трансформера
        elif model_type.lower() in ['lstm', 'gru']:
            config.hidden_dim = kwargs.get('hidden_dim', 64)
            config.num_layers = kwargs.get('num_layers', 2)
            config.learning_rate = kwargs.get('learning_rate', 0.001)

        # Перевизначення будь-яких параметрів
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    def train_model(self, symbol: str, timeframe: str, model_type: str,
                    data: pd.DataFrame, input_dim: int,
                    config: Optional[ModelConfig] = None,
                    validation_split: float = 0.2,
                    patience: int = 10, model_data=None, **kwargs) -> Dict[str, Any]:

        self.logger.info(f"Початок навчання моделі {symbol}_{timeframe}_{model_type}")

        # Час початку навчання
        training_start_time = time()

        # Створення або використання наданої конфігурації
        if config is None:
            config = self.create_model_config(symbol, timeframe, model_type, input_dim, **kwargs)

        # Підготовка даних з урахуванням sequence_length
        X, y = self._prepare_sequence_data(data, config.sequence_length)

        split = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        train_tensor = (torch.tensor(X_train).float(), torch.tensor(y_train).float())
        val_tensor = (torch.tensor(X_val).float(), torch.tensor(y_val).float())

        # Створення та навчання моделі
        model = self._build_model_from_config(model_type, config).to(self.device)
        history = self._train_loop(model, train_tensor, val_tensor,
                                   config.epochs, config.batch_size,
                                   config.learning_rate, patience)

        # Час завершення навчання
        training_end_time = time()
        training_duration = training_end_time - training_start_time

        # Збереження моделі та конфігурації
        model_key = self._create_model_key(symbol, timeframe, model_type)
        self.models[model_key] = model
        self.model_configs[model_key] = config

        # Оцінка моделі
        metrics = self.evaluate(model, val_tensor)
        self.model_metrics[model_key] = metrics
        self.training_history[model_key] = history

        # Збереження в БД
        model_id = None
        try:
            # Спочатку зберігаємо модель і отримуємо її ID
            if model_data is None:
                model_data = self._create_model_data_dict(symbol, timeframe, model_type, config)
            model_id = self.db_manager.save_ml_model(model_data)

            # Потім зберігаємо метрики з правильним model_id
            metrics_data = {
                'model_id': model_id,
                'mse': metrics.get('MSE', metrics.get('mse')),
                'rmse': metrics.get('RMSE', metrics.get('rmse')),
                'mae': metrics.get('MAE', metrics.get('mae')),
                'r2_score': metrics.get('R2', metrics.get('r2_score', metrics.get('r2'))),
                'test_date': datetime.now(),
                'training_duration_seconds': int(training_duration),
                'epochs_completed': len(history.get('train_loss', []))
            }

            metrics_id = self.db_manager.save_ml_model_metrics(metrics_data)
            self.logger.info(
                f"Модель {model_key} успішно збережена в БД. Model ID: {model_id}, Metrics ID: {metrics_id}")

        except Exception as e:
            self.logger.error(f"Помилка при збереженні в БД: {str(e)}")
            # Можна додати rollback логіку якщо потрібно

        self.logger.info(
            f"Навчання моделі {model_key} завершено. RMSE: {metrics.get('RMSE', metrics.get('rmse', 'N/A'))}")

        return {
            'config': config.__dict__,
            'metrics': metrics,
            'history': history,
            'model_id': model_id,
            'training_duration': training_duration
        }

    # Додатковий допоміжний метод для створення словника метрик
    def _prepare_metrics_data(self, model_id: int, metrics: Dict[str, Any],
                              training_duration: float, epochs_completed: int) -> Dict[str, Any]:

        return {
            'model_id': model_id,
            'mse': metrics.get('MSE', metrics.get('mse')),
            'rmse': metrics.get('RMSE', metrics.get('rmse')),
            'mae': metrics.get('MAE', metrics.get('mae')),
            'r2_score': metrics.get('R2', metrics.get('r2_score', metrics.get('r2'))),
            'test_date': datetime.now(),
            'training_duration_seconds': int(training_duration),
            'epochs_completed': epochs_completed
        }

    def _prepare_sequence_data(self, data: pd.DataFrame, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:

        features = data.drop(columns=["target"]).values
        targets = data["target"].values

        # Створення послідовностей
        X, y = [], []
        for i in range(sequence_length, len(features)):
            X.append(features[i - sequence_length:i])
            y.append(targets[i])

        return np.array(X), np.array(y)

    def _build_model_from_config(self, model_type: str, config: ModelConfig) -> BaseDeepModel:

        try:
            if model_type.lower() == 'lstm':
                from ML.LSTM import LSTMModel
                return LSTMModel(
                    input_dim=config.input_dim,
                    hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers,
                    output_dim=config.output_dim,
                    dropout=config.dropout
                )
            elif model_type.lower() == 'gru':
                from ML.GRU import GRUModel
                return GRUModel(
                    input_dim=config.input_dim,
                    hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers,
                    output_dim=config.output_dim,
                    dropout=config.dropout
                )
            elif model_type.lower() == 'transformer':
                from ML.transformer import TransformerModel
                return TransformerModel(
                    input_dim=config.input_dim,
                    hidden_dim=config.hidden_dim,
                    n_heads=config.num_heads,
                    num_layers=config.num_layers,
                    dim_feedforward=config.dim_feedforward,
                    output_dim=config.output_dim,
                    dropout=config.dropout,
                )
            else:
                raise ValueError(f"Невідомий тип моделі: {model_type}")
        except ImportError as e:
            self.logger.error(f"Помилка імпорту моделі {model_type}: {str(e)}")
            raise

    def _create_model_data_dict(self, symbol: str, timeframe: str, model_type: str,
                                config: ModelConfig) -> Dict[str, Any]:

        model_path = os.path.join(self.models_dir, symbol, timeframe, f"{model_type.lower()}_model.pth")

        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'model_type': model_type,
            'model_version': '1.0',
            'model_path': model_path,
            'input_features': list(range(config.input_dim)),  # Список індексів фічів
            'hidden_dim': config.hidden_dim,
            'num_layers': config.num_layers,
            'sequence_length': config.sequence_length,
            'num_heads': getattr(config, 'num_heads', None),
            'dim_feedforward': getattr(config, 'dim_feedforward', None),
            'active': True
        }

    def _train_loop(self, model: BaseDeepModel, train_data: Tuple[torch.Tensor, torch.Tensor],
                    val_data: Tuple[torch.Tensor, torch.Tensor], epochs: int,
                    batch_size: int, learning_rate: float, patience: int) -> List[float]:

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*train_data),
            batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*val_data),
            batch_size=batch_size
        )

        optimizer, criterion = self.setup_optimizer_and_loss(model, learning_rate)
        val_losses = []
        best_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        for epoch in range(epochs):
            # Тренування
            model.train()
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)

            # Валідація
            model.eval()
            val_loss = self.validate_epoch(model, val_loader, criterion)
            val_losses.append(val_loss)

            # Early stopping logic
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                self.logger.info(f"Early stopping на епосі {epoch + 1}")
                break

            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Епоха {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Завантаження найкращих ваг
        if best_model_state:
            model.load_state_dict(best_model_state)

        return val_losses

    def train_epoch(self, model: BaseDeepModel, train_loader, optimizer, criterion) -> float:

        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            optimizer.zero_grad()
            output = model(X_batch).squeeze()
            loss = criterion(output, y_batch)
            loss.backward()

            # Gradient clipping для стабільності (особливо важливо для Transformer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def validate_epoch(self, model: BaseDeepModel, val_loader, criterion) -> float:

        total_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                output = model(X_batch).squeeze()
                loss = criterion(output, y_batch)
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def evaluate(self, model: BaseDeepModel, test_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:

        model.eval()
        X_test, y_true = test_data
        X_test, y_true = X_test.to(self.device), y_true.to(self.device)

        with torch.no_grad():
            y_pred = model(X_test).squeeze().cpu().numpy()

        y_true = y_true.cpu().numpy()
        return self.calculate_metrics(y_true, y_pred)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:

        # Обробка випадків з нульовими значеннями для MAPE
        epsilon = 1e-8
        y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)

        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

        # Додаткові метрики
        r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

        return {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'MAPE': float(mape),
            'R2': float(r2)
        }

    def setup_optimizer_and_loss(self, model: BaseDeepModel, learning_rate: float):

        # Для Transformer моделей можна використовувати AdamW
        if hasattr(model, 'model_type') and model.model_type == 'transformer':
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        else:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        criterion = nn.MSELoss()
        return optimizer, criterion

    def online_learning(self, symbol: str, timeframe: str, model_type: str,
                        new_data: pd.DataFrame, input_dim: int = None,
                        config: Optional[ModelConfig] = None,
                        epochs: int = 10, learning_rate: float = 0.0005,
                        model_data=None) -> Dict[str, Any]:

        model_key = self._create_model_key(symbol, timeframe, model_type)

        # Завантаження моделі, якщо вона не в пам'яті
        if model_key not in self.models:
            if not self.load_model(symbol, timeframe, model_type):
                raise ValueError(f"Модель {model_key} не знайдена")

        model = self.models[model_key]
        model_config = self.model_configs[model_key]

        self.logger.info(f"Початок онлайн навчання для {model_key}")

        # Час початку онлайн навчання
        training_start_time = time()

        # Підготовка нових даних з урахуванням sequence_length
        X, y = self._prepare_sequence_data(new_data, model_config.sequence_length)
        train_tensor = (torch.tensor(X).float(), torch.tensor(y).float())

        # Дообучення з меншою кількістю епох та швидкістю навчання
        history = self._train_loop(model, train_tensor, train_tensor, epochs, 32, learning_rate, patience=5)

        # Час завершення навчання
        training_end_time = time()
        training_duration = training_end_time - training_start_time

        # Оцінка оновленої моделі
        metrics = self.evaluate(model, train_tensor)
        self.model_metrics[model_key] = metrics

        # Збереження оновленої моделі
        model_id = None
        try:
            # Спочатку зберігаємо/оновлюємо модель і отримуємо її ID
            if model_data is None:
                model_data = self._create_model_data_dict(symbol, timeframe, model_type, model_config)
            model_id = self.db_manager.save_ml_model(model_data)

            # Потім зберігаємо метрики онлайн навчання
            metrics_data = {
                'model_id': model_id,
                'mse': metrics.get('MSE', metrics.get('mse')),
                'rmse': metrics.get('RMSE', metrics.get('rmse')),
                'mae': metrics.get('MAE', metrics.get('mae')),
                'r2_score': metrics.get('R2', metrics.get('r2_score', metrics.get('r2'))),
                'test_date': datetime.now(),
                'training_duration_seconds': int(training_duration),
                'epochs_completed': epochs
            }

            metrics_id = self.db_manager.save_ml_model_metrics(metrics_data)
            self.logger.info(
                f"Онлайн навчання для {model_key} завершено. Model ID: {model_id}, Metrics ID: {metrics_id}")

        except Exception as e:
            self.logger.error(f"Помилка при збереженні після онлайн навчання: {str(e)}")

        return {
            'metrics': metrics,
            'history': history,
            'model_id': model_id,
            'training_duration': training_duration
        }
    def _build_model(self, model_type: str, input_dim: int, hidden_dim: int, num_layers: int) -> BaseDeepModel:

        config = ModelConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        return self._build_model_from_config(model_type, config)

    def _create_model_key(self, symbol: str, timeframe: str, model_type: str) -> str:

        return f"{symbol}_{timeframe}_{model_type}"

    def save_model(self, symbol: str, timeframe: str, model_type: str) -> str:

        model_key = self._create_model_key(symbol, timeframe, model_type)
        if model_key not in self.models:
            raise ValueError(f"Модель {model_key} не знайдена")

        # Створюємо директорію для моделі
        model_dir = os.path.join(self.models_dir, symbol, timeframe)
        os.makedirs(model_dir, exist_ok=True)

        # Шлях до файлу моделі
        model_path = os.path.join(model_dir, f"{model_type.lower()}_model.pth")

        try:
            # Зберігаємо модель
            torch.save({
                'model_state_dict': self.models[model_key].state_dict(),
                'config': self.model_configs[model_key].__dict__,
                'metrics': self.model_metrics.get(model_key, {}),
                'training_history': self.training_history.get(model_key, [])
            }, model_path)

            # Підготовлюємо дані для збереження в БД
            config = self.model_configs[model_key]
            model_data = self._create_model_data_dict(symbol, timeframe, model_type, config)

            # Зберігаємо інформацію про модель в БД
            model_id = self.db_manager.save_ml_model(model_data)

            self.logger.info(f"Модель {model_key} збережена за шляхом: {model_path} з ID: {model_id}")
            return model_path

        except Exception as e:
            self.logger.error(f"Помилка при збереженні моделі {model_key}: {str(e)}")
            raise

    def load_model(self, symbol: str, timeframe: str, model_type: str) -> bool:

        model_key = self._create_model_key(symbol, timeframe, model_type)

        # Шлях до файлу моделі
        model_dir = os.path.join(self.models_dir, symbol, timeframe)
        model_path = os.path.join(model_dir, f"{model_type.lower()}_model.pth")

        if not os.path.exists(model_path):
            self.logger.warning(f"Модель {model_key} не знайдена за шляхом {model_path}")
            return False

        try:
            # Завантаження моделі
            checkpoint = torch.load(model_path, map_location=self.device)

            # Відновлення конфігурації
            config_dict = checkpoint['config']
            config = ModelConfig(**config_dict)

            # Створення моделі
            model = self._build_model_from_config(model_type, config)

            # Завантаження ваг моделі
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)

            # Збереження моделі та її конфігурації
            self.models[model_key] = model
            self.model_configs[model_key] = config
            self.model_metrics[model_key] = checkpoint.get('metrics', {})
            self.training_history[model_key] = checkpoint.get('training_history', [])

            self.logger.info(f"Модель {model_key} успішно завантажена")
            return True

        except Exception as e:
            self.logger.error(f"Помилка при завантаженні моделі {model_key}: {str(e)}")
            return False

    # ==================== НАВЧАННЯ МОДЕЛЕЙ ====================

    def train_all_models(self, symbols: Optional[List[str]] = None,
                         timeframes: Optional[List[str]] = None,
                         model_types: Optional[List[str]] = None,
                         **training_params) -> Dict[str, Dict[str, Any]]:

        config = CryptoConfig()
        symbols = symbols or config.symbols
        timeframes = timeframes or config.timeframes
        model_types = model_types or config.model_types

        results = {}
        total_models = len(symbols) * len(timeframes) * len(model_types)
        current_model = 0

        self.logger.info(f"Початок навчання {total_models} моделей")

        for symbol in symbols:
            for timeframe in timeframes:
                for model_type in model_types:
                    current_model += 1
                    model_key = self._create_model_key(symbol, timeframe, model_type)

                    try:
                        self.logger.info(f"Навчання моделі {current_model}/{total_models}: {model_key}")

                        # Завантаження та підготовка даних
                        data = self.processor.get_data_loader(symbol, timeframe, model_type)
                        if data is None or len(data) < 100:
                            self.logger.warning(f"Недостатньо даних для {model_key}")
                            continue

                        # Підготовка фічів
                        processed_data = self.processor.prepare_features(data, symbol)
                        input_dim = processed_data.shape[1] - 1  # -1 для target колонки

                        # Навчання моделі
                        result = self.train_model(
                            symbol=symbol,
                            timeframe=timeframe,
                            model_type=model_type,
                            data=processed_data,
                            input_dim=input_dim,
                            **training_params
                        )

                        results[model_key] = result
                        self.logger.info(f"Модель {model_key} навчена успішно")

                    except Exception as e:
                        self.logger.error(f"Помилка навчання моделі {model_key}: {str(e)}")
                        results[model_key] = {'error': str(e)}

        self.logger.info(
            f"Навчання завершено. Успішно: {len([r for r in results.values() if 'error' not in r])}/{total_models}")
        return results

    def batch_online_learning(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:

        results = {}
        successful_updates = 0

        self.logger.info(f"Початок пакетного онлайн навчання для {len(updates)} моделей")

        for i, update in enumerate(updates):
            try:
                symbol = update['symbol']
                timeframe = update['timeframe']
                model_type = update['model_type']
                data = update['data']

                model_key = self._create_model_key(symbol, timeframe, model_type)

                # Визначення input_dim
                input_dim = data.shape[1] - 1 if 'target' in data.columns else data.shape[1]

                # Онлайн навчання
                result = self.online_learning(
                    symbol=symbol,
                    timeframe=timeframe,
                    model_type=model_type,
                    new_data=data,
                    input_dim=input_dim,
                    epochs=update.get('epochs', 10),
                    learning_rate=update.get('learning_rate', 0.0005)
                )

                results[model_key] = result
                successful_updates += 1

                self.logger.info(f"Онлайн навчання {i + 1}/{len(updates)} ({model_key}) - успішно")

            except Exception as e:
                model_key = f"{update.get('symbol', 'unknown')}_{update.get('timeframe', 'unknown')}_{update.get('model_type', 'unknown')}"
                self.logger.error(f"Помилка онлайн навчання {model_key}: {str(e)}")
                results[model_key] = {'error': str(e)}

        summary = {
            'total_updates': len(updates),
            'successful_updates': successful_updates,
            'failed_updates': len(updates) - successful_updates,
            'results': results
        }

        self.logger.info(f"Пакетне онлайн навчання завершено: {successful_updates}/{len(updates)} успішно")
        return summary

    def cross_validate_model(self, symbol: str, timeframe: str, model_type: str,
                             k_folds: int = 5, **model_params) -> Dict[str, List[float]]:

        self.logger.info(f"Початок крос-валідації для {symbol}_{timeframe}_{model_type}")

        # Завантаження даних
        data = self.processor.get_data_loader(symbol, timeframe, model_type)
        if data is None:
            raise ValueError(f"Не вдалося завантажити дані для {symbol}_{timeframe}")

        processed_data = self.processor.prepare_features(data, symbol)
        X = processed_data.drop(columns=["target"]).values
        y = processed_data["target"].values

        # Ініціалізація крос-валідації
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        # Результати для кожного фолда
        fold_results = {
            'MSE': [],
            'RMSE': [],
            'MAE': [],
            'MAPE': [],
            'R2': []
        }

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            self.logger.info(f"Обробка фолда {fold + 1}/{k_folds}")

            # Розділення даних
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Створення та навчання моделі
            input_dim = X_train.shape[1]
            model = self._build_model(
                model_type,
                input_dim,
                model_params.get('hidden_dim', 64),
                model_params.get('num_layers', 2)
            ).to(self.device)

            # Підготовка даних для навчання
            train_tensor = (torch.tensor(X_train).float(), torch.tensor(y_train).float())
            val_tensor = (torch.tensor(X_val).float(), torch.tensor(y_val).float())

            # Навчання моделі
            self._train_loop(
                model, train_tensor, val_tensor,
                epochs=model_params.get('epochs', 100),
                batch_size=model_params.get('batch_size', 32),
                learning_rate=model_params.get('learning_rate', 0.001),
                patience=model_params.get('patience', 10)
            )

            # Оцінка моделі
            metrics = self.evaluate(model, val_tensor)

            # Збереження результатів
            for metric_name, value in metrics.items():
                fold_results[metric_name].append(value)

        # Обчислення середніх значень та стандартних відхилень
        summary = {}
        for metric_name, values in fold_results.items():
            summary[f'{metric_name}_mean'] = np.mean(values)
            summary[f'{metric_name}_std'] = np.std(values)
            summary[f'{metric_name}_values'] = values

        self.logger.info(
            f"Крос-валідація завершена. Середній RMSE: {summary['RMSE_mean']:.6f} ± {summary['RMSE_std']:.6f}")

        return summary

    def get_training_summary(self) -> Dict[str, Any]:

        total_models = len(self.models)

        if total_models == 0:
            return {'total_models': 0, 'message': 'Немає навчених моделей'}

        # Статистика по символам
        symbol_stats = {}
        timeframe_stats = {}
        model_type_stats = {}

        for model_key in self.models.keys():
            parts = model_key.split('_')
            if len(parts) >= 3:
                symbol = parts[0]
                timeframe = parts[1]
                model_type = parts[2]

                symbol_stats[symbol] = symbol_stats.get(symbol, 0) + 1
                timeframe_stats[timeframe] = timeframe_stats.get(timeframe, 0) + 1
                model_type_stats[model_type] = model_type_stats.get(model_type, 0) + 1

        # Статистика по метриках
        rmse_values = []
        mae_values = []
        r2_values = []

        for metrics in self.model_metrics.values():
            if 'RMSE' in metrics:
                rmse_values.append(metrics['RMSE'])
            if 'MAE' in metrics:
                mae_values.append(metrics['MAE'])
            if 'R2' in metrics:
                r2_values.append(metrics['R2'])

        summary = {
            'total_models': total_models,
            'symbol_distribution': symbol_stats,
            'timeframe_distribution': timeframe_stats,
            'model_type_distribution': model_type_stats,
            'metrics_summary': {
                'rmse': {
                    'mean': np.mean(rmse_values) if rmse_values else None,
                    'std': np.std(rmse_values) if rmse_values else None,
                    'min': np.min(rmse_values) if rmse_values else None,
                    'max': np.max(rmse_values) if rmse_values else None
                },
                'mae': {
                    'mean': np.mean(mae_values) if mae_values else None,
                    'std': np.std(mae_values) if mae_values else None,
                    'min': np.min(mae_values) if mae_values else None,
                    'max': np.max(mae_values) if mae_values else None
                },
                'r2': {
                    'mean': np.mean(r2_values) if r2_values else None,
                    'std': np.std(r2_values) if r2_values else None,
                    'min': np.min(r2_values) if r2_values else None,
                    'max': np.max(r2_values) if r2_values else None
                }
            }
        }

        return summary

