from dataclasses import field, dataclass
from typing import Dict, Tuple, Callable, List, Optional, Any
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


@dataclass
class CryptoConfig:
    symbols: List[str] = field(default_factory=lambda: ['BTC', 'ETH', 'SOL'])
    timeframes: List[str] = field(default_factory=lambda: ['1m', '1h', '4h', '1d', '1w'])
    model_types: List[str] = field(default_factory=lambda: ['lstm', 'gru'])


class DataPreprocessor:
    """Клас для обробки попередньо підготовлених та відскейлених даних для моделей"""

    # Константи для розмірів послідовностей по таймфреймам
    TIMEFRAME_SEQUENCES = {
        '1m': 60,
        '1h': 24,
        '4h': 60,
        '1d': 30,
        '1w': 12
    }

    # Мапінг таймфреймів на числові значення для бази даних
    TIMEFRAME_MAPPING = {
        '1m': {'timeframe': 1, 'unit': 'minute'},
        '1h': {'timeframe': 1, 'unit': 'hour'},
        '4h': {'timeframe': 4, 'unit': 'hour'},
        '1d': {'timeframe': 1, 'unit': 'day'},
        '1w': {'timeframe': 1, 'unit': 'week'}
    }

    def __init__(self):
        self.scalers = {}  # Словник для зберігання скейлерів (для зворотного перетворення)
        self.feature_configs = {}  # Конфігурації ознак
        self.model_configs = {}  # Конфігурації моделей
        self.trend = TrendDetection()
        self.vol = VolatilityAnalysis()
        self.cycle = CryptoCycles()
        self.indicators = FeatureEngineering()
        self.db_manager = DatabaseManager()
        self.logger = CryptoLogger('DataPreprocessor')

        # Флажки для відстеження стану даних
        self.data_is_scaled = True
        self.sequences_prepared = True
        self.time_features_prepared = True

        self.logger.info("DataPreprocessor ініціалізовано для роботи з підготовленими даними")

    def parse_timeframe(self, timeframe: str) -> Dict[str, Any]:
        """
        Парсинг рядка таймфрейму на компоненти для бази даних

        Args:
            timeframe: Рядок таймфрейму (наприклад, '1h', '4h', '1d')

        Returns:
            Словник з компонентами таймфрейму
        """
        self.logger.debug(f"Парсинг таймфрейму: {timeframe}")

        if timeframe in self.TIMEFRAME_MAPPING:
            result = self.TIMEFRAME_MAPPING[timeframe].copy()
            self.logger.debug(f"Знайдено мапінг для {timeframe}: {result}")
            return result

        # Якщо таймфрейм не в мапінгу, спробуємо парсити вручну
        import re
        match = re.match(r'(\d+)([mhdw])', timeframe.lower())
        if not match:
            self.logger.error(f"Невідомий формат таймфрейму: {timeframe}")
            raise ValueError(f"Невідомий формат таймфрейму: {timeframe}")

        number, unit_char = match.groups()
        unit_mapping = {
            'm': 'minute',
            'h': 'hour',
            'd': 'day',
            'w': 'week'
        }

        result = {
            'timeframe': int(number),
            'unit': unit_mapping[unit_char]
        }

        self.logger.debug(f"Розпарсено таймфрейм {timeframe}: {result}")
        return result

    def get_sequence_length_for_timeframe(self, timeframe: str) -> int:
        """Отримання довжини послідовності для конкретного таймфрейму"""
        seq_length = self.TIMEFRAME_SEQUENCES.get(timeframe, 60)
        self.logger.debug(f"Довжина послідовності для {timeframe}: {seq_length}")
        return seq_length

    def create_model_config(self, input_dim: int, timeframe: str, **kwargs) -> ModelConfig:
        """Створення конфігурації моделі"""
        self.logger.debug(f"Створення конфігурації моделі для таймфрейму {timeframe} з input_dim={input_dim}")

        sequence_length = self.get_sequence_length_for_timeframe(timeframe)

        config_params = {
            'input_dim': input_dim,
            'sequence_length': sequence_length,
            **kwargs
        }

        config = ModelConfig(**config_params)
        self.logger.info(f"Створено конфігурацію моделі: hidden_dim={config.hidden_dim}, "
                         f"num_layers={config.num_layers}, sequence_length={config.sequence_length}")
        return config

    def get_model_config(self, symbol: str, timeframe: str, model_type: str) -> Optional[ModelConfig]:
        """Отримання конфігурації моделі"""
        key = f"{symbol}_{timeframe}_{model_type}"
        config = self.model_configs.get(key)

        if config:
            self.logger.debug(f"Знайдено конфігурацію для {key}")
        else:
            self.logger.warning(f"Конфігурація для {key} не знайдена")

        return config

    def set_model_config(self, symbol: str, timeframe: str, model_type: str, config: ModelConfig):
        """Збереження конфігурації моделі"""
        key = f"{symbol}_{timeframe}_{model_type}"
        self.model_configs[key] = config
        self.logger.info(f"Збережено конфігурацію моделі для {key}")

    def get_data_loader(self, symbol: str, timeframe: str, model_type: str) -> Callable:
        """
        Отримання функції завантаження попередньо підготовлених даних.

        :param symbol: Символ криптовалюти (BTC, ETH, SOL)
        :param timeframe: Таймфрейм у вигляді рядка (наприклад, '1d', '15m')
        :param model_type: Тип моделі
        :return: Callable — функція, що при виклику повертає підготовлені дані
        """
        try:
            symbol = symbol.upper()
            self.logger.info(f"Підготовка завантажувача даних для {symbol} з таймфреймом {timeframe} "
                             f"та моделлю {model_type}")

            if not hasattr(self, 'db_manager') or self.db_manager is None:
                self.logger.error("db_manager не ініціалізований у класі")
                raise ValueError("db_manager не встановлений")

            # Мапування символів на методи завантаження
            symbol_methods = {
                'BTC': self.db_manager.get_btc_lstm_sequence,
                'ETH': self.db_manager.get_eth_lstm_sequence,
                'SOL': self.db_manager.get_sol_lstm_sequence
            }

            if symbol not in symbol_methods:
                self.logger.error(f"Непідтримуваний символ: {symbol}")
                raise ValueError(f"Непідтримуваний символ: {symbol}")

            # Парсимо timeframe у зрозумілий формат
            timeframe_params = self.parse_timeframe(timeframe)
            self.logger.debug(f"Параметри таймфрейму: {timeframe_params}")

            method = symbol_methods[symbol]

            def data_loader():
                try:
                    self.logger.debug(f"Завантаження даних для {symbol} з параметрами: {timeframe_params}")

                    data = method(
                        timeframe_value=timeframe_params['timeframe'],
                        timeframe_unit=timeframe_params['unit']
                    )

                    if data is not None and not data.empty:
                        self.logger.info(f"Успішно завантажено {len(data)} записів для {symbol}-{timeframe}")
                        self.logger.debug(f"Колонки у завантажених даних: {list(data.columns)}")
                    else:
                        self.logger.warning(f"Завантажені дані для {symbol}-{timeframe} порожні")

                    return data

                except Exception as e:
                    self.logger.error(f"Помилка завантаження даних у data_loader для {symbol}: {str(e)}")
                    return None

            return data_loader

        except Exception as e:
            self.logger.error(f"Помилка підготовки завантажувача даних для {symbol}: {str(e)}")
            raise

    def get_data_with_fallback(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Завантаження підготовлених даних з fallback механізмом
        """
        try:
            self.logger.info(f"Спроба завантаження даних для {symbol}-{timeframe}")

            # Спробуємо основний метод
            data_loader = self.get_data_loader(symbol, timeframe, 'lstm')
            data = data_loader()

            if data is not None and not data.empty:
                self.logger.info(f"Успішно завантажено дані для {symbol}-{timeframe}: "
                                 f"форма {data.shape}")
                return data
            else:
                self.logger.warning(f"Основний метод повернув порожні дані для {symbol}-{timeframe}")
                raise ValueError("Порожні дані з основного методу")

        except Exception as e:
            self.logger.error(f"Помилка завантаження даних для {symbol}-{timeframe}: {e}")

            # Fallback: спробуємо альтернативний метод завантаження
            try:
                self.logger.info(f"Спроба fallback завантаження для {symbol}-{timeframe}")
                return self._fallback_data_loader(symbol, timeframe)
            except Exception as fallback_error:
                self.logger.error(f"Fallback також не спрацював для {symbol}-{timeframe}: {fallback_error}")
                raise e

    def _fallback_data_loader(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Альтернативний метод завантаження даних
        """
        self.logger.warning(f"Використання fallback методу для {symbol}-{timeframe}")
        # Тут можна реалізувати альтернативний спосіб завантаження даних
        # Наприклад, з файлу або API
        raise NotImplementedError("Fallback метод не реалізований")

    def validate_prepared_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """
        Валідація попередньо підготовлених даних

        Args:
            df: DataFrame з підготовленими даними
            symbol: Символ для логування

        Returns:
            bool: True якщо дані валідні
        """
        self.logger.info(f"Валідація підготовлених даних для {symbol}")

        if df.empty:
            self.logger.error(f"DataFrame порожній для {symbol}")
            raise ValueError(f"DataFrame порожній для {symbol}")

        # Перевірка наявності цільової змінної
        if "target" not in df.columns:
            self.logger.error(f"Цільова змінна 'target' не знайдена для {symbol}")
            raise ValueError(f"Цільова змінна 'target' не знайдена для {symbol}")

        # Перевірка часових ознак
        time_features = [col for col in df.columns if 'sin' in col or 'cos' in col]
        if time_features:
            self.logger.info(f"Знайдено {len(time_features)} часових ознак для {symbol}: {time_features[:5]}...")
        else:
            self.logger.warning(f"Часові ознаки не знайдені для {symbol}")

        # Перевірка послідовностей (якщо є колонки типу sequence_*)
        sequence_features = [col for col in df.columns if 'sequence' in col.lower()]
        if sequence_features:
            self.logger.info(f"Знайдено {len(sequence_features)} ознак послідовностей для {symbol}")
        else:
            self.logger.info(f"Ознаки послідовностей не виявлені (можливо, дані у звичайному форматі) для {symbol}")

        # Перевірка масштабування (евристично)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            mean_abs_values = df[numeric_cols].abs().mean().mean()
            if mean_abs_values > 10:
                self.logger.warning(f"Дані можуть бути не відскейлені для {symbol} "
                                    f"(середнє абсолютне значення: {mean_abs_values:.2f})")
            else:
                self.logger.info(f"Дані, схоже, відскейлені для {symbol} "
                                 f"(середнє абсолютне значення: {mean_abs_values:.2f})")

        # Статистика по NaN
        nan_counts = df.isnull().sum()
        total_nans = nan_counts.sum()
        if total_nans > 0:
            self.logger.warning(
                f"Знайдено {total_nans} NaN значень у {len(nan_counts[nan_counts > 0])} колонках для {symbol}")
            # Логування топ-колонок з найбільшою кількістю NaN
            top_nan_cols = nan_counts[nan_counts > 0].head(5)
            for col, count in top_nan_cols.items():
                self.logger.debug(f"  {col}: {count} NaN значень")
        else:
            self.logger.info(f"NaN значення не знайдені для {symbol}")

        self.logger.info(f"Валідація завершена для {symbol}: форма даних {df.shape}")
        return True

    def prepare_features(self, df: pd.DataFrame, symbol: str) -> tuple[DataFrame, Series] | DataFrame | Any:
        """Підготовка ознак для моделі"""
        try:
            # Список для зберігання DataFrames перед об'єднанням
            feature_dataframes = []

            # Отримання різних типів ознак
            print(f"Підготовка трендових ознак для {symbol}...")
            try:
                trend_features = self.trend.prepare_ml_trend_features(df)
                if not trend_features.empty:
                    feature_dataframes.append(trend_features)
            except Exception as e:
                print(f"Помилка при створенні трендових ознак: {e}")

            print(f"Підготовка ознак волатильності для {symbol}...")
            try:
                volatility_features = self.vol.prepare_volatility_features_for_ml(df, symbol)
                if not volatility_features.empty:
                    feature_dataframes.append(volatility_features)
            except Exception as e:
                print(f"Помилка при створенні ознак волатильності: {e}")

            print(f"Підготовка циклічних ознак для {symbol}...")
            try:
                cycle_features = self.cycle.prepare_cycle_ml_features(df, symbol)
                if not cycle_features.empty:
                    feature_dataframes.append(cycle_features)
            except Exception as e:
                print(f"Помилка при створенні циклічних ознак: {e}")

            print(f"Підготовка технічних індикаторів для {symbol}...")
            try:
                indicator_features = self.indicators.prepare_features_pipeline(df)
                if not indicator_features.empty:
                    feature_dataframes.append(indicator_features)
            except Exception as e:
                print(f"Помилка при створенні технічних індикаторів: {e}")

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

    def validate_sequences_format(self, data: np.ndarray, symbol: str, timeframe: str) -> bool:
        """
        Валідація формату послідовностей

        Args:
            data: Масив даних
            symbol: Символ для логування
            timeframe: Таймфрейм для логування

        Returns:
            bool: True якщо формат правильний
        """
        self.logger.debug(f"Валідація формату послідовностей для {symbol}-{timeframe}")

        if data.ndim != 3:
            self.logger.error(f"Неправильна розмірність даних для {symbol}-{timeframe}: "
                              f"очікується 3D, отримано {data.ndim}D з формою {data.shape}")
            return False

        expected_seq_length = self.get_sequence_length_for_timeframe(timeframe)
        actual_seq_length = data.shape[1]

        if actual_seq_length != expected_seq_length:
            self.logger.warning(f"Довжина послідовності для {symbol}-{timeframe} "
                                f"({actual_seq_length}) відрізняється від очікуваної ({expected_seq_length})")

        self.logger.info(f"Формат послідовностей валідний для {symbol}-{timeframe}: {data.shape}")
        return True

    def preprocess_prepared_data_for_model(self, data: pd.DataFrame, symbol: str, timeframe: str,
                                           model_type: str = 'lstm', validation_split: float = 0.2,
                                           config: Optional[ModelConfig] = None) -> Tuple[torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, ModelConfig]:
        """
        Обробка попередньо підготовлених та відскейлених даних для моделі

        Args:
            data: DataFrame з попередньо підготовленими ознаками
            symbol: Символ криптовалюти
            timeframe: Часовий інтервал
            model_type: Тип моделі
            validation_split: Розмір валідаційної вибірки
            config: Опціональна конфігурація моделі

        Returns:
            Кортеж (X_train, y_train, X_val, y_val, model_config)
        """
        self.logger.info(f"Початок обробки підготовлених даних для {symbol}-{timeframe}-{model_type}")

        # Валідація та базова обробка
        if "target" not in data.columns:
            self.logger.error(f"Цільова змінна 'target' не знайдена для {symbol}-{timeframe}")
            raise ValueError("Цільова змінна 'target' не знайдена в даних")

        # Видалення рядків з NaN значеннями тільки якщо це критично
        initial_len = len(data)
        data_clean = data.dropna()

        if len(data_clean) == 0:
            self.logger.error(f"Після видалення NaN не залишилось даних для {symbol}-{timeframe}")
            raise ValueError("Після видалення NaN значень не залишилось даних")

        if len(data_clean) < initial_len:
            self.logger.warning(f"Видалено {initial_len - len(data_clean)} рядків з NaN "
                                f"для {symbol}-{timeframe}")

        # Розділення на ознаки та цільову змінну
        feature_columns = data_clean.drop(columns=["target"])
        target_values = data_clean["target"].values

        self.logger.info(f"Розмір ознак: {feature_columns.shape}, розмір target: {target_values.shape}")

        # Створення або отримання конфігурації моделі
        if config is None:
            input_dim = feature_columns.shape[1]
            config = self.create_model_config(input_dim, timeframe)
            self.set_model_config(symbol, timeframe, model_type, config)
            self.logger.info(f"Створено нову конфігурацію моделі для {symbol}-{timeframe}-{model_type}")
        else:
            # Оновлення input_dim якщо потрібно
            if config.input_dim != feature_columns.shape[1]:
                old_dim = config.input_dim
                config.input_dim = feature_columns.shape[1]
                self.logger.info(f"Оновлено input_dim з {old_dim} на {config.input_dim} "
                                 f"для {symbol}-{timeframe}-{model_type}")

        # Перевірка чи дані вже у форматі послідовностей
        data_values = feature_columns.values

        if data_values.ndim == 3:
            # Дані вже у форматі послідовностей (samples, sequence_length, features)
            self.logger.info(f"Дані вже у форматі послідовностей для {symbol}-{timeframe}: {data_values.shape}")
            self.validate_sequences_format(data_values, symbol, timeframe)
            X = data_values
            y = target_values
        else:
            # Потрібно створити послідовності
            self.logger.info(f"Створення послідовностей з 2D даних для {symbol}-{timeframe}")
            X, y = self.create_sequences(data_values, target_values, config.sequence_length)
            self.logger.info(f"Створено послідовності: X {X.shape}, y {y.shape}")

        # Збереження "скейлера" для можливості зворотного перетворення
        # (хоча дані вже відскейлені, зберігаємо для сумісності)
        scaler_key = f"{symbol}_{timeframe}"
        if scaler_key not in self.scalers:
            # Створюємо фіктивний скейлер для сумісності
            dummy_scaler = StandardScaler()
            # Підганяємо його під поточні дані (хоча вони вже відскейлені)
            if data_values.ndim == 2:
                dummy_scaler.fit(data_values)
            else:
                # Для 3D даних беремо перший семпл
                dummy_scaler.fit(data_values.reshape(-1, data_values.shape[-1]))
            self.scalers[scaler_key] = dummy_scaler
            self.logger.debug(f"Створено фіктивний скейлер для {scaler_key}")

        # Розділення на тренувальну та валідаційну вибірки
        X_train, X_val, y_train, y_val = self.split_data(X, y, validation_split)

        self.logger.info(f"Розділення даних для {symbol}-{timeframe}: "
                         f"train {X_train.shape}, val {X_val.shape}")

        # Конвертація в тензори PyTorch
        self.logger.debug(f"Конвертація в PyTorch тензори для {symbol}-{timeframe}")

        return (torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32),
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32),
                config)

    def create_sequences(self, data: np.ndarray, target: np.ndarray,
                         seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Створення послідовностей для RNN моделей (якщо потрібно)"""
        self.logger.debug(f"Створення послідовностей: вхід {data.shape}, довжина {seq_length}")

        X, y = [], []

        # Перевірка достатності даних
        if len(data) <= seq_length:
            self.logger.error(f"Недостатньо даних для створення послідовностей. "
                              f"Потрібно мінімум {seq_length + 1} записів, є {len(data)}")
            raise ValueError(f"Недостатньо даних для створення послідовностей. "
                             f"Потрібно мінімум {seq_length + 1} записів, є {len(data)}")

        # Створення послідовностей
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(target[i + seq_length])

        X_array = np.array(X)
        y_array = np.array(y)

        self.logger.debug(f"Створено послідовності: X {X_array.shape}, y {y_array.shape}")
        return X_array, y_array

    def prepare_data_with_config(self, symbol: str, timeframe: str, model_type: str,
                                 validation_split: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, ModelConfig]:
        """Підготовка попередньо оброблених даних з автоматичним створенням конфігурації"""
        self.logger.info(f"Підготовка даних з конфігурацією для {symbol}-{timeframe}-{model_type}")

        # Завантаження попередньо підготовлених даних
        try:
            raw_data = self.get_data_with_fallback(symbol, timeframe)
            self.logger.info(f"Завантажено сирі дані для {symbol}-{timeframe}: {raw_data.shape}")
        except Exception as e:
            self.logger.error(f"Критична помилка завантаження даних для {symbol}-{timeframe}: {e}")
            raise

        # Мінімальна обробка попередньо підготовлених ознак
        try:
            features = self.prepare_features(raw_data, symbol)
            self.logger.info(f"Підготовлено ознаки для {symbol}: {features.shape}")
        except Exception as e:
            self.logger.error(f"Помилка підготовки ознак для {symbol}: {e}")
            raise

        # Обробка для моделі
        try:
            result = self.preprocess_prepared_data_for_model(
                features, symbol, timeframe, model_type, validation_split
            )
            self.logger.info(f"Успішно підготовлено дані для моделі {symbol}-{timeframe}-{model_type}")
            return result
        except Exception as e:
            self.logger.error(f"Помилка препроцесингу для моделі {symbol}-{timeframe}-{model_type}: {e}")
            raise

    def get_optimal_config_for_timeframe(self, timeframe: str, input_dim: int,
                                         **custom_params) -> ModelConfig:
        """Отримання оптимальної конфігурації для конкретного таймфрейму"""
        self.logger.debug(f"Створення оптимальної конфігурації для {timeframe} з input_dim={input_dim}")

        # Базові параметри залежно від таймфрейму
        timeframe_configs = {
            '1m': {'hidden_dim': 128, 'num_layers': 3, 'dropout': 0.3, 'learning_rate': 0.0005},
            '1h': {'hidden_dim': 64, 'num_layers': 2, 'dropout': 0.2, 'learning_rate': 0.001},
            '4h': {'hidden_dim': 96, 'num_layers': 2, 'dropout': 0.25, 'learning_rate': 0.001},
            '1d': {'hidden_dim': 128, 'num_layers': 3, 'dropout': 0.2, 'learning_rate': 0.0008},
            '1w': {'hidden_dim': 64, 'num_layers': 2, 'dropout': 0.15, 'learning_rate': 0.0012}
        }

        # Отримання базової конфігурації
        base_config = timeframe_configs.get(timeframe, {
            'hidden_dim': 64, 'num_layers': 2, 'dropout': 0.2, 'learning_rate': 0.001
        })

        # Додаткові параметри
        additional_params = {
            'batch_size': 32 if timeframe in ['1m', '1h'] else 64,
            'epochs': 150 if timeframe == '1m' else 100,
            'output_dim': 1
        }

        # Об'єднання всіх параметрів
        config_params = {
            'input_dim': input_dim,
            'sequence_length': self.get_sequence_length_for_timeframe(timeframe),
            **base_config,
            **additional_params,
            **custom_params  # Кастомні параметри мають найвищий пріоритет
        }

        config = ModelConfig(**config_params)
        self.logger.info(f"Створено оптимальну конфігурацію для {timeframe}: "
                         f"hidden_dim={config.hidden_dim}, num_layers={config.num_layers}, "
                         f"sequence_length={config.sequence_length}")
        return config

    def split_data(self, X: np.ndarray, y: np.ndarray,
                   validation_split: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Розділення даних на тренувальну та валідаційну вибірки з урахуванням часової послідовності"""
        self.logger.debug(f"Розділення даних: X {X.shape}, y {y.shape}, split={validation_split}")

        # Для часових рядів використовуємо послідовне розділення (не рандомне)
        split_idx = int(len(X) * (1 - validation_split))

        X_train = X[:split_idx]
        X_val = X[split_idx:]
        y_train = y[:split_idx]
        y_val = y[split_idx:]

        self.logger.info(f"Дані розділені: train {X_train.shape}, val {X_val.shape}")
        return X_train, X_val, y_train, y_val

    def inverse_transform_predictions(self, predictions: np.ndarray,
                                      symbol: str, timeframe: str) -> np.ndarray:
        """Зворотнє перетворення прогнозів до оригінального масштабу"""
        scaler_key = f"{symbol}_{timeframe}"

        if scaler_key not in self.scalers:
            self.logger.warning(f"Скейлер для {scaler_key} не знайдений. Повертаємо прогнози без перетворення")
            return predictions

        scaler = self.scalers[scaler_key]

        try:
            # Якщо прогнози 1D, робимо їх 2D для inverse_transform
            if predictions.ndim == 1:
                predictions_2d = predictions.reshape(-1, 1)
                transformed = scaler.inverse_transform(predictions_2d)
                return transformed.flatten()
            else:
                return scaler.inverse_transform(predictions)

        except Exception as e:
            self.logger.error(f"Помилка зворотного перетворення для {scaler_key}: {e}")
            return predictions

    def get_feature_importance_info(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Отримання інформації про важливість ознак"""
        key = f"{symbol}_{timeframe}"

        return {
            'total_features': self.feature_configs.get(f"{key}_count", 0),
            'feature_types': self.feature_configs.get(f"{key}_types", {}),
            'time_features': self.feature_configs.get(f"{key}_time_features", []),
            'scaled': self.data_is_scaled,
            'sequences_ready': self.sequences_prepared
        }

    def validate_model_input(self, X: torch.Tensor, config: ModelConfig,
                             symbol: str, timeframe: str) -> bool:
        """Валідація вхідних даних для моделі"""
        self.logger.debug(f"Валідація вхідних даних моделі для {symbol}-{timeframe}")

        # Перевірка розмірності
        if X.dim() != 3:
            self.logger.error(f"Неправильна розмірність вхідних даних: {X.dim()}, очікується 3")
            return False

        # Перевірка sequence_length
        if X.shape[1] != config.sequence_length:
            self.logger.error(f"Неправильна довжина послідовності: {X.shape[1]}, "
                              f"очікується {config.sequence_length}")
            return False

        # Перевірка input_dim
        if X.shape[2] != config.input_dim:
            self.logger.error(f"Неправильна розмірність ознак: {X.shape[2]}, "
                              f"очікується {config.input_dim}")
            return False

        # Перевірка на NaN та Inf
        if torch.isnan(X).any() or torch.isinf(X).any():
            self.logger.error(f"Знайдено NaN або Inf значення у вхідних даних для {symbol}-{timeframe}")
            return False

        self.logger.debug(f"Валідація пройшла успішно для {symbol}-{timeframe}: {X.shape}")
        return True

    def get_data_summary(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Отримання зведеної інформації про дані"""
        try:
            data = self.get_data_with_fallback(symbol, timeframe)

            summary = {
                'symbol': symbol,
                'timeframe': timeframe,
                'total_records': len(data),
                'date_range': {
                    'start': data.index.min() if hasattr(data.index, 'min') else 'Unknown',
                    'end': data.index.max() if hasattr(data.index, 'max') else 'Unknown'
                },
                'features': {
                    'total_columns': len(data.columns),
                    'numeric_columns': len(data.select_dtypes(include=[np.number]).columns),
                    'has_target': 'target' in data.columns
                },
                'data_quality': {
                    'null_values': data.isnull().sum().sum(),
                    'null_percentage': (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
                },
                'preprocessing_status': {
                    'scaled': self.data_is_scaled,
                    'sequences_prepared': self.sequences_prepared,
                    'time_features_prepared': self.time_features_prepared
                }
            }

            self.logger.info(f"Створено зведення даних для {symbol}-{timeframe}")
            return summary

        except Exception as e:
            self.logger.error(f"Помилка створення зведення для {symbol}-{timeframe}: {e}")
            return {'error': str(e)}

    def cleanup_scalers(self, symbols: List[str] = None, timeframes: List[str] = None):
        """Очищення скейлерів для вивільнення пам'яті"""
        if symbols is None and timeframes is None:
            # Очищення всіх скейлерів
            cleared_count = len(self.scalers)
            self.scalers.clear()
            self.logger.info(f"Очищено {cleared_count} скейлерів")
        else:
            # Вибіркове очищення
            keys_to_remove = []
            for key in self.scalers.keys():
                parts = key.split('_')
                if len(parts) >= 2:
                    symbol, timeframe = parts[0], parts[1]
                    if (symbols is None or symbol in symbols) and \
                            (timeframes is None or timeframe in timeframes):
                        keys_to_remove.append(key)

            for key in keys_to_remove:
                del self.scalers[key]

            self.logger.info(f"Очищено {len(keys_to_remove)} скейлерів")

    def get_memory_usage_info(self) -> Dict[str, Any]:
        """Отримання інформації про використання пам'яті"""
        import sys

        scalers_memory = sys.getsizeof(self.scalers)
        configs_memory = sys.getsizeof(self.model_configs) + sys.getsizeof(self.feature_configs)

        return {
            'scalers_count': len(self.scalers),
            'model_configs_count': len(self.model_configs),
            'feature_configs_count': len(self.feature_configs),
            'approximate_memory_mb': (scalers_memory + configs_memory) / (1024 * 1024),
            'scalers_keys': list(self.scalers.keys())[:10]  # Перші 10 ключів
        }