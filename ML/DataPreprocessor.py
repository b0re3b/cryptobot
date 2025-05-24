from dataclasses import field, dataclass
from typing import Dict, Tuple, Callable, List, Any
import pandas as pd
import numpy as np
import torch
from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from analysis import VolatilityAnalysis
from analysis.trend_detection import TrendDetection
from cyclefeatures import CryptoCycles
from data.db import DatabaseManager
from featureengineering.feature_engineering import FeatureEngineering


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
    timeframes: List[str] = field(default_factory=lambda: ['1m', '1h', '4h', '1d', '1w'])
    model_types: List[str] = field(default_factory=lambda: ['lstm', 'gru'])


class DataPreprocessor:
    """Клас для завантаження, підготовки, нормалізації та обробки даних для моделей"""

    # Константи для розмірів послідовностей по таймфреймам
    TIMEFRAME_SEQUENCES = {
        '1m': 60,
        '1h': 24,
        '4h': 60,
        '1d': 30,
        '1w': 12
    }

    def __init__(self):
        self.scalers = {}  # Словник для зберігання скейлерів
        self.feature_configs = {}  # Конфігурації ознак
        self.model_configs = {}  # Конфігурації моделей
        self.trend = TrendDetection()
        self.vol = VolatilityAnalysis()
        self.cycle = CryptoCycles()
        self.indicators = FeatureEngineering()
        self.db_manager = DatabaseManager()

    def get_sequence_length_for_timeframe(self, timeframe: str) -> int:

        return self.TIMEFRAME_SEQUENCES.get(timeframe, 60)

    def create_model_config(self, input_dim: int, timeframe: str, **kwargs) -> ModelConfig:

        sequence_length = self.get_sequence_length_for_timeframe(timeframe)

        config_params = {
            'input_dim': input_dim,
            'sequence_length': sequence_length,
            **kwargs
        }

        return ModelConfig(**config_params)

    def get_model_config(self, symbol: str, timeframe: str, model_type: str) -> ModelConfig:

        key = f"{symbol}_{timeframe}_{model_type}"
        return self.model_configs.get(key)

    def set_model_config(self, symbol: str, timeframe: str, model_type: str, config: ModelConfig):

        key = f"{symbol}_{timeframe}_{model_type}"
        self.model_configs[key] = config

    def get_data_loader(self, symbol: str, timeframe: str, model_type: str) -> Callable:

        if symbol == 'BTC':
            return lambda: self.db_manager.get_btc_lstm_sequence(timeframe)
        elif symbol == 'ETH':
            return lambda: self.db_manager.get_eth_lstm_sequence(timeframe)
        elif symbol == 'SOL':
            return lambda: self.db_manager.get_sol_lstm_sequence(timeframe)
        else:
            raise ValueError(f"Непідтримуваний символ: {symbol}")

    def prepare_features(self, df: pd.DataFrame, symbol: str) -> tuple[DataFrame, Series] | DataFrame | Any:

        try:
            # Список для зберігання DataFrames перед об'єднанням
            feature_dataframes = []

            # Отримання різних типів ознак
            print(f"Підготовка трендових ознак для {symbol}...")
            trend_features = self.trend.prepare_ml_trend_features(df)
            if not trend_features.empty:
                feature_dataframes.append(trend_features)

            print(f"Підготовка ознак волатильності для {symbol}...")
            volatility_features = self.vol.prepare_volatility_features_for_ml(df, symbol)
            if not volatility_features.empty:
                feature_dataframes.append(volatility_features)

            print(f"Підготовка циклічних ознак для {symbol}...")
            cycle_features = self.cycle.prepare_cycle_ml_features(df, symbol)
            if not cycle_features.empty:
                feature_dataframes.append(cycle_features)

            print(f"Підготовка технічних індикаторів для {symbol}...")
            indicator_features = self.indicators.prepare_features_pipeline(df)
            if not indicator_features.empty:
                feature_dataframes.append(indicator_features)

            # Перевірка наявності ознак для об'єднання
            if not feature_dataframes:
                raise ValueError(f"Не вдалося створити жодних ознак для {symbol}")

            # Оптимізоване об'єднання з copy=False та ignore_index=False для кращої продуктивності
            print(f"Об'єднання {len(feature_dataframes)} наборів ознак...")
            final_features = pd.concat(
                feature_dataframes,
                axis=1,
                copy=False,  # Не копіювати дані без потреби
                sort=False  # Не сортувати колонки для швидкості
            )

            # Ефективне видалення дублікатів колонок
            if final_features.columns.duplicated().any():
                print("Видалення дублікатних колонок...")
                # Отримуємо унікальні колонки, зберігаючи порядок
                unique_columns = final_features.columns[~final_features.columns.duplicated()]
                final_features = final_features[unique_columns]

            # Додаткова валідація результату
            if final_features.empty:
                raise ValueError(f"Результуючий DataFrame порожній для {symbol}")

            print(f"Успішно підготовлено {final_features.shape[1]} ознак для {symbol}")
            return final_features

        except Exception as e:
            print(f"Помилка при підготовці ознак для {symbol}: {e}")
            # Повертаємо базовий набір ознак як fallback
            try:
                print(f"Спроба створити базові ознаки для {symbol}...")
                basic_features = self.indicators.prepare_features_pipeline(df)
                if not basic_features.empty:
                    return basic_features
            except Exception as fallback_error:
                print(f"Помилка fallback для {symbol}: {fallback_error}")

            raise

    def preprocess_data_for_model(self, data: pd.DataFrame, symbol: str, timeframe: str,
                                  model_type: str = 'lstm', validation_split: float = 0.2,
                                  config: ModelConfig = None) -> Tuple[torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, ModelConfig]:
        """
        Препроцесинг даних для моделі з використанням ModelConfig

        Args:
            data: DataFrame з ознаками
            symbol: Символ криптовалюти
            timeframe: Часовий інтервал
            model_type: Тип моделі
            validation_split: Розмір валідаційної вибірки
            config: Опціональна конфігурація моделі

        Returns:
            Кортеж (X_train, y_train, X_val, y_val, model_config)
        """
        # Перевірка наявності цільової змінної
        if "target" not in data.columns:
            raise ValueError("Цільова змінна 'target' не знайдена в даних")

        # Видалення рядків з NaN значеннями
        data_clean = data.dropna()

        if len(data_clean) == 0:
            raise ValueError("Після видалення NaN значень не залишилось даних")

        # Розділення на ознаки та цільову змінну
        feature_columns = data_clean.drop(columns=["target"])
        data_values = feature_columns.values
        target_values = data_clean["target"].values

        # Створення або отримання конфігурації моделі
        if config is None:
            sequence_length = self.get_sequence_length_for_timeframe(timeframe)
            input_dim = data_values.shape[1]
            config = self.create_model_config(input_dim, timeframe)
            # Збереження конфігурації
            self.set_model_config(symbol, timeframe, model_type, config)
        else:
            # Оновлення input_dim якщо потрібно
            if config.input_dim != data_values.shape[1]:
                config.input_dim = data_values.shape[1]

        # Нормалізація даних
        scaler = StandardScaler()
        data_values_scaled = scaler.fit_transform(data_values)

        # Збереження скейлера
        scaler_key = f"{symbol}_{timeframe}"
        self.scalers[scaler_key] = scaler

        # Створення послідовностей з використанням sequence_length з конфігурації
        X, y = self.create_sequences(data_values_scaled, target_values, config.sequence_length)

        # Розділення на тренувальну та валідаційну вибірки
        X_train, X_val, y_train, y_val = self.split_data(X, y, validation_split)

        # Конвертація в тензори PyTorch
        return (torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32),
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32),
                config)

    def create_sequences(self, data: np.ndarray, target: np.ndarray,
                         seq_length: int) -> Tuple[np.ndarray, np.ndarray]:

        X, y = [], []

        # Перевірка достатності даних
        if len(data) <= seq_length:
            raise ValueError(f"Недостатньо даних для створення послідовностей. "
                             f"Потрібно мінімум {seq_length + 1} записів, є {len(data)}")

        # Створення послідовностей
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(target[i + seq_length])

        return np.array(X), np.array(y)

    def prepare_data_with_config(self, symbol: str, timeframe: str, model_type: str,
                                 validation_split: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, ModelConfig]:

        # Завантаження даних
        data_loader = self.get_data_loader(symbol, timeframe, model_type)
        raw_data = data_loader()

        # Підготовка ознак
        features = self.prepare_features(raw_data, symbol)

        # Препроцесинг даних з створенням конфігурації
        return self.preprocess_data_for_model(
            features, symbol, timeframe, model_type, validation_split
        )

    def get_optimal_config_for_timeframe(self, timeframe: str, input_dim: int,
                                         **custom_params) -> ModelConfig:

        # Базові параметри залежно від таймфрейму
        timeframe_configs = {
            '1m': {'hidden_dim': 128, 'num_layers': 3, 'dropout': 0.3, 'learning_rate': 0.0005},
            '1h': {'hidden_dim': 64, 'num_layers': 2, 'dropout': 0.2, 'learning_rate': 0.001},
            '4h': {'hidden_dim': 96, 'num_layers': 2, 'dropout': 0.25, 'learning_rate': 0.0008},
            '1d': {'hidden_dim': 48, 'num_layers': 2, 'dropout': 0.15, 'learning_rate': 0.001},
            '1w': {'hidden_dim': 32, 'num_layers': 1, 'dropout': 0.1, 'learning_rate': 0.0015}
        }

        base_config = timeframe_configs.get(timeframe, {})
        base_config.update(custom_params)

        return self.create_model_config(input_dim, timeframe, **base_config)

    def denormalize_predictions(self, predictions: np.ndarray, symbol: str,
                                timeframe: str) -> np.ndarray:

        key = f"{symbol}_{timeframe}"
        scaler = self.scalers.get(key)

        if scaler is None:
            raise ValueError(f"Скейлер для {key} не знайдено. "
                             f"Доступні скейлери: {list(self.scalers.keys())}")

        # Перевірка розмірності
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        return scaler.inverse_transform(predictions).flatten()

    def split_data(self, X: np.ndarray, y: np.ndarray,
                   validation_split: float) -> Tuple[np.ndarray, np.ndarray,
    np.ndarray, np.ndarray]:

        # Перевірка валідності параметрів
        if not 0 < validation_split < 1:
            raise ValueError(f"validation_split має бути між 0 та 1, отримано: {validation_split}")

        if len(X) != len(y):
            raise ValueError(f"Розміри X ({len(X)}) та y ({len(y)}) не співпадають")

        # Обчислення індексу розділення
        split_index = int(len(X) * (1 - validation_split))

        # Перевірка мінімального розміру вибірок
        if split_index < 1 or len(X) - split_index < 1:
            raise ValueError("Недостатньо даних для створення тренувальної та валідаційної вибірок")

        return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

    def normalize_features(self, data: pd.DataFrame, symbol: str, timeframe: str,
                           scaler_type: str = 'standard') -> pd.DataFrame:

        key = f"{symbol}_{timeframe}"

        # Вибір типу скейлера
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Непідтримуваний тип скейлера: {scaler_type}")

        # Нормалізація даних
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data_normalized = data.copy()

        if len(numeric_columns) > 0:
            data_normalized[numeric_columns] = scaler.fit_transform(data[numeric_columns])
            # Збереження скейлера
            self.scalers[key] = scaler
        else:
            print(f"Попередження: не знайдено числових колонок для нормалізації в {key}")

        return data_normalized

    def get_feature_importance(self, symbol: str) -> Dict:

        return self.feature_configs.get(symbol, {})

    def set_feature_config(self, symbol: str, config: Dict):

        self.feature_configs[symbol] = config

    def validate_data(self, data: pd.DataFrame) -> bool:

        if data.empty:
            raise ValueError("DataFrame порожній")

        if data.isnull().all().any():
            print("Попередження: знайдені колонки з усіма NaN значеннями")

        return True

    def get_scaler(self, symbol: str, timeframe: str):

        key = f"{symbol}_{timeframe}"
        return self.scalers.get(key)