from dataclasses import field, dataclass
from typing import Dict, Tuple, Callable, List
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from analysis import VolatilityAnalysis
from analysis.trend_detection import TrendDetection
from cyclefeatures import CryptoCycles
from data.db import DatabaseManager
from featureengineering.feature_engineering import FeatureEngineering


@dataclass
class CryptoConfig:
    symbols: List[str] = field(default_factory=lambda: ['BTC', 'ETH', 'SOL'])
    timeframes: List[str] = field(default_factory=lambda: ['1m', '1h', '4h', '1d', '1w'])
    model_types: List[str] = field(default_factory=lambda: ['lstm', 'gru'])


class DataPreprocessor:
    """Клас для завантаження, підготовки, нормалізації та обробки даних для моделей"""

    def __init__(self):
        self.scalers = {}  # Словник для зберігання скейлерів
        self.feature_configs = {}  # Конфігурації ознак
        self.trend = TrendDetection()
        self.vol = VolatilityAnalysis()
        self.cycle = CryptoCycles()
        self.indicators = FeatureEngineering()
        self.db_manager = DatabaseManager

    def get_data_loader(self, symbol: str, timeframe: str, model_type: str) -> Callable:
        """
        Отримати функцію для завантаження даних

        Args:
            symbol: Символ криптовалюти ('BTC', 'ETH', 'SOL')
            timeframe: Часовий інтервал ('1m', '1h', '4h', '1d', '1w')
            model_type: Тип моделі ('lstm', 'gru')

        Returns:
            Callable: Функція для завантаження даних
        """
        if symbol == 'BTC':
            return lambda: self.db_manager.get_btc_lstm_sequence(timeframe)
        elif symbol == 'ETH':
            return lambda: self.db_manager.get_eth_lstm_sequence(timeframe)
        elif symbol == 'SOL':
            return lambda: self.db_manager.get_sol_lstm_sequence(timeframe)
        else:
            raise ValueError(f"Непідтримуваний символ: {symbol}")

    def prepare_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Підготовка ознак для навчання моделі

        Args:
            df: Вхідний DataFrame
            symbol: Символ криптовалюти

        Returns:
            DataFrame з ознаками
        """
        # Отримання різних типів ознак
        trend_features = self.trend.prepare_ml_trend_features(df, symbol)
        volatility_features = self.vol.prepare_volatility_features_for_ml(df, symbol)
        cycle_features = self.cycle.prepare_cycle_ml_features(df, symbol)

        # Об'єднання всіх ознак через конвейер
        final_features = self.indicators.prepare_features_pipeline(
            df,
            trend_features=trend_features,
            volatility_features=volatility_features,
            cycle_features=cycle_features,
            symbol=symbol
        )

        return final_features

    def preprocess_data_for_model(self, data: pd.DataFrame, sequence_length: int = 60,
                                  validation_split: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor]:
        """
        Препроцесинг даних для моделі

        Args:
            data: DataFrame з ознаками
            sequence_length: Довжина послідовностей
            validation_split: Розмір валідаційної вибірки

        Returns:
            Кортеж тензорів (X_train, y_train, X_val, y_val)
        """
        # Перевірка наявності цільової змінної
        if "target" not in data.columns:
            raise ValueError("Цільова змінна 'target' не знайдена в даних")

        # Розділення на ознаки та цільову змінну
        data_values = data.drop(columns=["target"]).values
        target_values = data["target"].values

        # Нормалізація даних
        scaler = StandardScaler()
        data_values_scaled = scaler.fit_transform(data_values)

        # Створення послідовностей
        X, y = self.create_sequences(data_values_scaled, target_values, sequence_length)

        # Розділення на тренувальну та валідаційну вибірки
        X_train, X_val, y_train, y_val = self.split_data(X, y, validation_split)

        # Конвертація в тензори PyTorch
        return (torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32),
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32))

    def create_sequences(self, data: np.ndarray, target: np.ndarray,
                         seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Створення послідовностей для моделей RNN

        Args:
            data: Масив ознак
            target: Цільові значення
            seq_length: Довжина послідовності

        Returns:
            Кортеж масивів (X, y)
        """
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

    def denormalize_predictions(self, predictions: np.ndarray, symbol: str,
                                timeframe: str) -> np.ndarray:
        """
        Денормалізація прогнозів

        Args:
            predictions: Масив прогнозів
            symbol: Символ криптовалюти
            timeframe: Таймфрейм

        Returns:
            Денормалізований масив
        """
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
        """
        Розділення даних на тренувальну та валідаційну вибірки

        Args:
            X: Ознаки
            y: Ціль
            validation_split: Частка валідації

        Returns:
            X_train, X_val, y_train, y_val
        """
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
        """
        Нормалізація ознак з збереженням скейлера

        Args:
            data: DataFrame з ознаками
            symbol: Символ криптовалюти
            timeframe: Таймфрейм
            scaler_type: Тип скейлера ('standard' або 'minmax')

        Returns:
            Нормалізований DataFrame
        """
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
        data_normalized[numeric_columns] = scaler.fit_transform(data[numeric_columns])

        # Збереження скейлера
        self.scalers[key] = scaler

        return data_normalized

    def get_feature_importance(self, symbol: str) -> Dict:
        """
        Отримання конфігурації важливості ознак

        Args:
            symbol: Символ криптовалюти

        Returns:
            Словник з конфігурацією ознак
        """
        return self.feature_configs.get(symbol, {})

    def set_feature_config(self, symbol: str, config: Dict):
        """
        Встановлення конфігурації ознак для символу

        Args:
            symbol: Символ криптовалюти
            config: Конфігурація ознак
        """
        self.feature_configs[symbol] = config