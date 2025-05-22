from typing import Dict, Tuple, Callable
import pandas as pd
import numpy as np
import torch

class DataPreprocessor:
    """Клас для завантаження, підготовки, нормалізації та обробки даних для моделей"""

    SYMBOLS = ['BTC', 'ETH', 'SOL']
    TIMEFRAMES = ['1m', '1h', '4h', '1d', '1w']

    def __init__(self):
        self.scalers = {}  # Словник для зберігання скейлерів
        self.feature_configs = {}  # Конфігурації ознак

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
            return lambda: get_btc_lstm_sequence(timeframe)
        elif symbol == 'ETH':
            return lambda: get_eth_lstm_sequence(timeframe)
        elif symbol == 'SOL':
            return lambda: get_sol_lstm_sequence(timeframe)
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
        trend_features = prepare_ml_trend_features(df, symbol)
        volatility_features = prepare_volatility_features_for_ml(df, symbol)
        cycle_features = prepare_cycle_ml_features(df, symbol)

        final_features = prepare_features_pipeline(
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
        data_values = data.drop(columns=["target"]).values
        target_values = data["target"].values

        X, y = self.create_sequences(data_values, target_values, sequence_length)
        X_train, X_val, y_train, y_val = self.split_data(X, y, validation_split)

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
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(target[i+seq_length])
        return np.array(X), np.array(y)

    def normalize_data(self, data: pd.DataFrame, symbol: str,
                       timeframe: str) -> pd.DataFrame:
        """
        Нормалізація даних (мін-макс або стандартна)

        Args:
            data: DataFrame з числовими ознаками
            symbol: Символ криптовалюти
            timeframe: Таймфрейм

        Returns:
            Нормалізований DataFrame
        """
        from sklearn.preprocessing import MinMaxScaler

        key = f"{symbol}_{timeframe}"
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data.values)
        self.scalers[key] = scaler

        return pd.DataFrame(scaled, index=data.index, columns=data.columns)

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
            raise ValueError(f"Скейлер для {key} не знайдено")
        return scaler.inverse_transform(predictions)

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
        split_index = int(len(X) * (1 - validation_split))
        return X[:split_index], X[split_index:], y[:split_index], y[split_index:]
