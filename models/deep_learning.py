import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Union, Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


class DeepLearningModels:
    """
    Клас для створення та тренування глибоких нейронних мереж для прогнозування криптовалют.
    """

    def __init__(self, log_level=logging.INFO):
        """
        Ініціалізація класу глибокого навчання.

        Args:
            log_level: Рівень логування
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.models = {}
        self.scalers = {}

    def prepare_sequences(self, data: pd.DataFrame, target_col: str,
                          lookback: int = 30, forecast_horizon: int = 1,
                          scaler_key: str = 'default') -> Tuple[np.ndarray, np.ndarray]:
        """
        Підготовка послідовностей для тренування LSTM/GRU.

        Args:
            data: Дані для тренування
            target_col: Стовпець-ціль
            lookback: Кількість попередніх часових кроків
            forecast_horizon: Горизонт прогнозування
            scaler_key: Ключ для збереження скейлера

        Returns:
            X, y: вхідні послідовності та відповідні цільові значення
        """
        # Масштабування даних
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        self.scalers[scaler_key] = scaler

        X, y = [], []
        for i in range(lookback, len(scaled_data) - forecast_horizon + 1):
            X.append(scaled_data[i - lookback:i])

            # Для прогнозу використовуємо target_col
            target_index = data.columns.get_loc(target_col)
            target_values = scaled_data[i:i + forecast_horizon, target_index]
            y.append(target_values)

        return np.array(X), np.array(y)

    def create_lstm_model(self, input_shape: Tuple[int, int],
                          output_size: int = 1,
                          layers: List[int] = [128, 64, 32],
                          dropout_rate: float = 0.2) -> Sequential:
        """
        Створення моделі LSTM.

        Args:
            input_shape: Форма вхідних даних (timesteps, features)
            output_size: Розмір вихідного шару
            layers: Список з кількістю нейронів для кожного шару LSTM
            dropout_rate: Коефіцієнт виключення нейронів

        Returns:
            Модель Keras
        """
        model = Sequential()

        # Додавання шарів LSTM
        for i, units in enumerate(layers):
            return_sequences = i < len(layers) - 1  # Повертати послідовності для всіх шарів крім останнього
            if i == 0:
                model.add(LSTM(units, return_sequences=return_sequences, input_shape=input_shape))
            else:
                model.add(LSTM(units, return_sequences=return_sequences))

            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))

        # Вихідний шар
        model.add(Dense(output_size))

        return model

    def create_gru_model(self, input_shape: Tuple[int, int],
                         output_size: int = 1,
                         layers: List[int] = [128, 64, 32],
                         dropout_rate: float = 0.2) -> Sequential:
        """
        Створення моделі GRU.

        Args:
            input_shape: Форма вхідних даних (timesteps, features)
            output_size: Розмір вихідного шару
            layers: Список з кількістю нейронів для кожного шару GRU
            dropout_rate: Коефіцієнт виключення нейронів

        Returns:
            Модель Keras
        """
        model = Sequential()

        # Додавання шарів GRU
        for i, units in enumerate(layers):
            return_sequences = i < len(layers) - 1  # Повертати послідовності для всіх шарів крім останнього
            if i == 0:
                model.add(GRU(units, return_sequences=return_sequences, input_shape=input_shape))
            else:
                model.add(GRU(units, return_sequences=return_sequences))

            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))

        # Вихідний шар
        model.add(Dense(output_size))

        return model

    def train_model(self, model, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray = None, y_val: np.ndarray = None,
                    model_key: str = 'default',
                    epochs: int = 100, batch_size: int = 32,
                    patience: int = 10, checkpoint_path: str = None) -> Dict:
        """
        Тренування моделі.

        Args:
            model: Модель для тренування
            X_train, y_train: Тренувальні дані
            X_val, y_val: Валідаційні дані
            model_key: Ключ для збереження моделі
            epochs: Кількість епох
            batch_size: Розмір батчу
            patience: Кількість епох для раннього зупинення
            checkpoint_path: Шлях для збереження чекпоінтів

        Returns:
            Історія тренування
        """
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        ]

        if checkpoint_path:
            callbacks.append(ModelCheckpoint(
                checkpoint_path, save_best_only=True, monitor='val_loss'
            ))

        # Компіляція моделі
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Тренування
        if X_val is not None and y_val is not None:
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        else:
            history = model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )

        self.models[model_key] = model

        return {
            'model': model,
            'history': history.history
        }

    def predict(self, model_key: str, X: np.ndarray, scaler_key: str = 'default',
                target_col_index: int = 0) -> np.ndarray:
        """
        Виконання прогнозу на основі моделі.

        Args:
            model_key: Ключ моделі
            X: Вхідні дані
            scaler_key: Ключ скейлера
            target_col_index: Індекс цільового стовпця для зворотнього масштабування

        Returns:
            Прогнозні значення
        """
        if model_key not in self.models:
            self.logger.error(f"Модель {model_key} не знайдена")
            return None

        if scaler_key not in self.scalers:
            self.logger.error(f"Скейлер {scaler_key} не знайдений")
            return None

        # Прогнозування масштабованих значень
        scaled_predictions = self.models[model_key].predict(X)

        # Підготовка структури для зворотнього масштабування
        dummy = np.zeros((len(scaled_predictions), self.scalers[scaler_key].scale_.shape[0]))
        dummy[:, target_col_index] = scaled_predictions.reshape(-1)

        # Зворотне масштабування
        predictions = self.scalers[scaler_key].inverse_transform(dummy)[:, target_col_index]

        return predictions

    def evaluate(self, model_key: str, X_test: np.ndarray, y_test: np.ndarray,
                 scaler_key: str = 'default', target_col_index: int = 0) -> Dict:
        """
        Оцінка точності моделі.

        Args:
            model_key: Ключ моделі
            X_test, y_test: Тестові дані
            scaler_key: Ключ скейлера
            target_col_index: Індекс цільового стовпця

        Returns:
            Метрики точності
        """
        predictions = self.predict(model_key, X_test, scaler_key, target_col_index)

        # Підготовка реальних значень для порівняння
        dummy = np.zeros((len(y_test), self.scalers[scaler_key].scale_.shape[0]))
        dummy[:, target_col_index] = y_test.reshape(-1)
        actual = self.scalers[scaler_key].inverse_transform(dummy)[:, target_col_index]

        # Розрахунок метрик
        mse = mean_squared_error(actual, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predictions)
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }

    def save_model(self, model_key: str, path: str) -> bool:
        """
        Збереження моделі на диск.

        Args:
            model_key: Ключ моделі
            path: Шлях для збереження

        Returns:
            Успішність операції
        """
        if model_key not in self.models:
            self.logger.error(f"Модель {model_key} не знайдена")
            return False

        try:
            self.models[model_key].save(path)
            return True
        except Exception as e:
            self.logger.error(f"Помилка збереження моделі: {e}")
            return False

    def load_model(self, model_key: str, path: str) -> bool:
        """
        Завантаження моделі з диску.

        Args:
            model_key: Ключ для збереження моделі
            path: Шлях до файлу моделі

        Returns:
            Успішність операції
        """
        try:
            model = load_model(path)
            self.models[model_key] = model
            return True
        except Exception as e:
            self.logger.error(f"Помилка завантаження моделі: {e}")
            return False

    def save_scaler(self, scaler_key: str, path: str) -> bool:
        """
        Збереження скейлера.

        Args:
            scaler_key: Ключ скейлера
            path: Шлях для збереження

        Returns:
            Успішність операції
        """
        if scaler_key not in self.scalers:
            self.logger.error(f"Скейлер {scaler_key} не знайдений")
            return False

        import joblib
        try:
            joblib.dump(self.scalers[scaler_key], path)
            return True
        except Exception as e:
            self.logger.error(f"Помилка збереження скейлера: {e}")
            return False

    def load_scaler(self, scaler_key: str, path: str) -> bool:
        """
        Завантаження скейлера з диску.

        Args:
            scaler_key: Ключ для збереження скейлера
            path: Шлях до файлу скейлера

        Returns:
            Успішність операції
        """
        import joblib
        try:
            scaler = joblib.load(path)
            self.scalers[scaler_key] = scaler
            return True
        except Exception as e:
            self.logger.error(f"Помилка завантаження скейлера: {e}")
            return False