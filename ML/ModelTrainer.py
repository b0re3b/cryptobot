import os
from dataclasses import dataclass, field
from datetime import time, datetime
from typing import Dict, Any, List, Optional, Tuple, Generator
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from ML.DataPreprocessor import DataPreprocessor
from ML.base import BaseDeepModel
from data.db import DatabaseManager
from utils.logger import CryptoLogger


@dataclass
class ChunkConfig:
    chunk_size: int = 10000
    overlap_size: int = 100
    max_chunks_in_memory: int = 5


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
        Повертає рекомендовану довжину послідовності для таймфрейму
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
        self.processor = DataPreprocessor()

        # Конфігурація для роботи з чанками
        self.chunk_config = ChunkConfig()

        # Налаштування логування
        self.logger = CryptoLogger('trainer')

        # Створення директорії для моделей
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def _create_data_chunks(self, data: pd.DataFrame) -> Generator[pd.DataFrame, None, None]:
        """
        Створює чанки даних з перекриттям для безперервного навчання
        """
        total_rows = len(data)
        self.logger.info(f"Розділення {total_rows} рядків на чанки розміром {self.chunk_config.chunk_size}")

        chunk_count = 0
        start_idx = 0

        while start_idx < total_rows:
            end_idx = min(start_idx + self.chunk_config.chunk_size, total_rows)

            # Додаємо перекриття для збереження контексту послідовностей
            if start_idx > 0:
                actual_start = max(0, start_idx - self.chunk_config.overlap_size)
            else:
                actual_start = start_idx

            chunk = data.iloc[actual_start:end_idx].copy()
            chunk_count += 1

            self.logger.info(f"Створено чанк {chunk_count}: рядки {actual_start}-{end_idx - 1} ({len(chunk)} рядків)")

            yield chunk

            # Переходимо до наступного чанку
            start_idx = end_idx

            # Якщо залишилося менше overlap_size рядків, завершуємо
            if total_rows - start_idx < self.chunk_config.overlap_size:
                break

    def _prepare_chunk_sequences(self, chunk: pd.DataFrame, sequence_length: int,
                                 previous_chunk: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Підготовляє послідовності з чанку з урахуванням попереднього чанку для контексту
        """
        # Об'єднуємо з попереднім чанком для контексту, якщо він є
        if previous_chunk is not None:
            # Беремо останні sequence_length рядків з попереднього чанку
            context_rows = min(sequence_length, len(previous_chunk))
            context_data = previous_chunk.tail(context_rows)
            combined_data = pd.concat([context_data, chunk], ignore_index=True)
        else:
            combined_data = chunk

        features = combined_data.drop(columns=["close_scaled"]).values
        targets = combined_data["close_scaled"].values

        # Створюємо послідовності
        X, y = [], []
        start_idx = sequence_length if previous_chunk is not None else sequence_length

        for i in range(start_idx, len(features)):
            X.append(features[i - sequence_length:i])
            y.append(targets[i])

        self.logger.debug(f"З чанку розміром {len(chunk)} створено {len(X)} послідовностей")

        return np.array(X), np.array(y)


    def _train_chunk(self, model: BaseDeepModel, chunk_data: Tuple[torch.Tensor, torch.Tensor],
                     optimizer, criterion, batch_size: int) -> float:
        """
        Навчання моделі на одному чанку даних
        """
        model.train()
        X_chunk, y_chunk = chunk_data
        X_chunk, y_chunk = X_chunk.to(self.device), y_chunk.to(self.device)

        # Створюємо DataLoader для чанку
        chunk_dataset = torch.utils.data.TensorDataset(X_chunk, y_chunk)
        chunk_loader = torch.utils.data.DataLoader(
            chunk_dataset, batch_size=batch_size, shuffle=True
        )

        total_loss = 0
        batch_count = 0

        for X_batch, y_batch in chunk_loader:
            optimizer.zero_grad()
            output = model(X_batch).squeeze()
            loss = criterion(output, y_batch)
            loss.backward()

            # Gradient clipping для стабільності
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

        return total_loss / batch_count if batch_count > 0 else 0

    def _validate_chunk(self, model: BaseDeepModel, val_data: Tuple[torch.Tensor, torch.Tensor],
                        criterion) -> float:
        """
        Валідація моделі на валідаційних даних
        """
        model.eval()
        X_val, y_val = val_data
        X_val, y_val = X_val.to(self.device), y_val.to(self.device)

        with torch.no_grad():
            output = model(X_val).squeeze()
            loss = criterion(output, y_val)

        return loss.item()

    def train_model(self, symbol: str, timeframe: str, model_type: str,
                    data: pd.DataFrame, input_dim: int,
                    config: Optional[ModelConfig] = None,
                    validation_split: float = 0.2,
                    patience: int = 10, model_data=None, **kwargs) -> Dict[str, Any]:
        """
        Оновлений метод без використання prepare_features
        """
        self.logger.info(f"Початок навчання моделі {symbol}_{timeframe}_{model_type}")

        # Створюємо або отримуємо конфігурацію моделі
        if config is None:
            config = self.create_model_config(symbol, timeframe, model_type, input_dim, **kwargs)

        # Валідація вхідних даних
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Очікується DataFrame як вхідні дані")

        if 'target' not in data.columns:
            raise ValueError("DataFrame повинен містити колонку 'target'")

        # Підготовка даних
        X = data.drop(columns=['close_scaled']).values
        y = data['close_scaled'].values

        # Розділення на тренувальний та валідаційний набори
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Конвертація в тензори
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        # Створення моделі
        model = self._build_model_from_config(model_type, config).to(self.device)

        # Навчання моделі
        history = self._train_loop(
            model=model,
            train_data=(X_train_tensor, y_train_tensor),
            val_data=(X_val_tensor, y_val_tensor),
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            patience=patience
        )

        # Оцінка моделі
        metrics = self.evaluate(model, (X_val_tensor, y_val_tensor))

        # Збереження моделі
        model_key = self._create_model_key(symbol, timeframe, model_type)
        self.models[model_key] = model
        self.model_configs[model_key] = config
        self.model_metrics[model_key] = metrics
        self.training_history[model_key] = history

        return {
            'config': config.__dict__,
            'metrics': metrics,
            'history': history
        }
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

    def _prepare_sequence_data(self, data: pd.DataFrame, sequence_length: int,
                                       batch_size: int = 10000) -> Tuple[np.ndarray, np.ndarray]:

        features = data.drop(columns=["target"]).values
        targets = data["target"].values

        total_sequences = len(features) - sequence_length
        if total_sequences <= 0:
            return np.array([]), np.array([])

        # Розраховуємо кількість батчів
        num_batches = (total_sequences + batch_size - 1) // batch_size

        X_batches = []
        y_batches = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_sequences)

            X_batch = []
            y_batch = []

            for i in range(start_idx, end_idx):
                seq_start = i
                seq_end = i + sequence_length
                X_batch.append(features[seq_start:seq_end])
                y_batch.append(targets[seq_end])

            X_batches.append(np.array(X_batch))
            y_batches.append(np.array(y_batch))

            # Логування прогресу
            if batch_idx % 10 == 0:
                self.logger.debug(f"Оброблено батч {batch_idx + 1}/{num_batches}")

        # Об'єднуємо всі батчі
        X = np.vstack(X_batches) if X_batches else np.array([])
        y = np.concatenate(y_batches) if y_batches else np.array([])

        self.logger.info(f"Створено {len(X)} послідовностей з {len(data)} рядків даних")
        return X, y

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

    def train_epoch(self, model: BaseDeepModel, train_loader: torch.utils.data.DataLoader,
                    optimizer, criterion) -> float:
        """
        Навчання моделі на одній епосі з використанням DataLoader
        """
        model.train()
        total_loss = 0
        batch_count = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            optimizer.zero_grad()
            output = model(X_batch).squeeze()
            loss = criterion(output, y_batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

            # Очищення пам'яті після кожного батчу
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return total_loss / batch_count if batch_count > 0 else 0

    def validate_epoch(self, model: BaseDeepModel, val_loader: torch.utils.data.DataLoader,
                       criterion) -> float:
        """
        Валідація моделі на одній епосі з використанням DataLoader
        """
        model.eval()
        total_loss = 0
        batch_count = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                output = model(X_batch).squeeze()
                loss = criterion(output, y_batch)
                total_loss += loss.item()
                batch_count += 1

                # Очищення пам'яті після кожного батчу
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return total_loss / batch_count if batch_count > 0 else 0

    def _train_loop(self, model: BaseDeepModel, train_data: Tuple[torch.Tensor, torch.Tensor],
                    val_data: Tuple[torch.Tensor, torch.Tensor], epochs: int,
                    batch_size: int, learning_rate: float, patience: int) -> Dict[str, List[float]]:
        """
        Виправлений основний цикл навчання
        """
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*train_data),
            batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*val_data),
            batch_size=batch_size
        )

        optimizer, criterion = self.setup_optimizer_and_loss(model, learning_rate)

        # Історія навчання
        history = {
            'train_loss': [],
            'val_loss': []
        }

        best_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        for epoch in range(epochs):
            # Тренування - передаємо train_loader
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)

            # Валідація - передаємо val_loader
            val_loss = self.validate_epoch(model, val_loader, criterion)

            # Додаємо до історії
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

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

        return history


    def evaluate_batched(self, model: BaseDeepModel, test_data: Tuple[torch.Tensor, torch.Tensor],
                         batch_size: int = 1000) -> Dict[str, float]:
        """
        Оцінка моделі по батчах для великих наборів даних
        """
        model.eval()
        X_test, y_true = test_data

        # Якщо дані невеликі, використовуємо звичайну оцінку
        if len(X_test) <= batch_size:
            return self.evaluate(model, test_data)

        self.logger.info(f"Початок батчевої оцінки на {len(X_test)} зразках")

        all_predictions = []

        # Створюємо DataLoader для батчевої обробки
        test_dataset = torch.utils.data.TensorDataset(X_test, y_true)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

        with torch.no_grad():
            for batch_idx, (X_batch, y_batch) in enumerate(test_loader):
                X_batch = X_batch.to(self.device)

                # Отримання прогнозів для батчу
                y_pred_batch = model(X_batch).squeeze().cpu().numpy()
                all_predictions.append(y_pred_batch)

                # Очищення пам'яті GPU після кожного батчу
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if batch_idx % 50 == 0:
                    self.logger.debug(f"Оброблено батч {batch_idx + 1}/{len(test_loader)}")

        # Об'єднання всіх прогнозів
        y_pred = np.concatenate(all_predictions)
        y_true_np = y_true.cpu().numpy()

        metrics = self.calculate_metrics(y_true_np, y_pred)
        self.logger.info(f"Батчева оцінка завершена. RMSE: {metrics.get('RMSE', 'N/A'):.6f}")

        return metrics

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
                        epochs: int = 10, learning_rate: float = 0.0005,
                        model_data=None, **kwargs) -> Dict[str, Any]:
        """
        Оновлений метод онлайн навчання без використання prepare_features
        """
        model_key = self._create_model_key(symbol, timeframe, model_type)

        # Завантаження моделі, якщо вона не в пам'яті
        if model_key not in self.models:
            if not self.load_model(symbol, timeframe, model_type):
                raise ValueError(f"Модель {model_key} не знайдена")

        model = self.models[model_key]
        model_config = self.model_configs[model_key]

        # Валідація вхідних даних
        if not isinstance(new_data, pd.DataFrame):
            raise ValueError("Очікується DataFrame як вхідні дані")

        if 'target' not in new_data.columns:
            raise ValueError("DataFrame повинен містити колонку 'target'")

        # Підготовка даних
        X = new_data.drop(columns=['target']).values
        y = new_data['target'].values

        # Конвертація в тензори
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        # Налаштування оптимізатора
        optimizer, criterion = self.setup_optimizer_and_loss(model, learning_rate)

        # Історія навчання
        history = {'train_loss': []}

        # Навчання
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_tensor).squeeze()
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

            history['train_loss'].append(loss.item())

            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Епоха {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

        # Оцінка моделі
        metrics = self.evaluate(model, (X_tensor, y_tensor))
        self.model_metrics[model_key] = metrics

        return {
            'metrics': metrics,
            'history': history
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
        """
        Оновлений метод без використання prepare_features
        """
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

                        # Отримання даних
                        data_loader = self.processor.get_data_loader(symbol, timeframe, model_type)
                        data = data_loader()

                        if data.empty:
                            raise ValueError("Порожній DataFrame")

                        # Визначення розмірності вхідних даних
                        input_dim = data.shape[1] - 1  # Враховуємо, що остання колонка - target

                        # Навчання моделі
                        result = self.train_model(
                            symbol=symbol,
                            timeframe=timeframe,
                            model_type=model_type,
                            data=data,
                            input_dim=input_dim,
                            **training_params
                        )

                        results[model_key] = result
                        self.logger.info(f"Модель {model_key} навчена успішно. RMSE: {result['metrics']['RMSE']:.6f}")

                    except Exception as e:
                        self.logger.error(f"Помилка навчання моделі {model_key}: {str(e)}")
                        results[model_key] = {'error': str(e)}

        successful_models = len([r for r in results.values() if 'error' not in r])
        self.logger.info(f"Навчання завершено. Успішно: {successful_models}/{total_models}")

        return results


    def cross_validate_model(self, symbol: str, timeframe: str, model_type: str,
                             k_folds: int = 5, batch_size: int = 1000,
                             **model_params) -> Dict[str, List[float]]:
        """
        Крос-валідація з батчевою обробкою для великих наборів даних
        """
        self.logger.info(f"Початок батчевої крос-валідації для {symbol}_{timeframe}_{model_type}")

        # Завантаження та підготовка даних
        data = self.processor.get_data_loader(symbol, timeframe, model_type)
        if data is None:
            raise ValueError(f"Не вдалося завантажити дані для {symbol}_{timeframe}")

        features = self.processor.prepare_features(data, symbol)
        if not isinstance(features, dict) or 'X' not in features or 'y' not in features:
            raise ValueError("Очікувався словник з ключами 'X' та 'y' від prepare_features")

        X_data = features['X']
        y_data = features['y']

        input_dim = X_data.shape[1]
        config = self.create_model_config(symbol, timeframe, model_type, input_dim, **model_params)

        # Підготовка послідовностей з батчевою обробкою
        X, y = self._prepare_sequence_data(X_data, y_data, config.sequence_length, batch_size)

        if len(X) == 0:
            raise ValueError("Недостатньо даних для створення послідовностей")

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

            # Створення моделі
            model = self._build_model_from_config(model_type, config).to(self.device)

            # Підготовка тензорів
            train_tensor = (torch.tensor(X_train).float(), torch.tensor(y_train).float())
            val_tensor = (torch.tensor(X_val).float(), torch.tensor(y_val).float())

            # Навчання моделі з батчевою обробкою
            optimizer, criterion = self.setup_optimizer_and_loss(model, config.learning_rate)

            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(config.epochs):
                # Навчання
                model.train()
                train_loss = self.train_epoch(model, train_tensor, optimizer, criterion, batch_size)

                # Валідація
                model.eval()
                val_loss = self.validate_epoch(model, val_tensor, criterion, batch_size)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= model_params.get('patience', 10):
                    self.logger.debug(f"Early stopping на епосі {epoch + 1} для фолда {fold + 1}")
                    break

            # Оцінка
            metrics = self.evaluate_batched(model, val_tensor, batch_size)

            for metric_name, value in metrics.items():
                if metric_name in fold_results:
                    fold_results[metric_name].append(value)

            del model, train_tensor, val_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Обчислення середніх значень і відхилень
        summary = {}
        for metric_name, values in fold_results.items():
            if values:
                summary[f'{metric_name}_mean'] = np.mean(values)
                summary[f'{metric_name}_std'] = np.std(values)
                summary[f'{metric_name}_values'] = values

        self.logger.info(
            f"Батчева крос-валідація завершена. Середній RMSE: {summary.get('RMSE_mean', 'N/A'):.6f} ± {summary.get('RMSE_std', 'N/A'):.6f}")

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
