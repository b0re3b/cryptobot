from dataclasses import field, dataclass
from typing import Dict, Tuple, Callable, List, Optional, Any
import pandas as pd
import numpy as np
import torch
from pandas import DataFrame, Series
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
        self.db_manager = DatabaseManager()
        self.logger = CryptoLogger('DataPreprocessor')
        self.indicators = FeatureEngineering()
        self.cycle = CryptoCycles()
        self.vol = VolatilityAnalysis()
        self.trend = TrendDetection()
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

            method = symbol_methods[symbol]

            def data_loader():
                try:
                    self.logger.debug(f"Завантаження даних для {symbol} з таймфреймом {timeframe}")

                    # Викликаємо метод напряму через db_manager з timeframe як позиційним аргументом
                    if symbol == 'BTC':
                        data = self.db_manager.get_btc_lstm_sequence(timeframe)
                    elif symbol == 'ETH':
                        data = self.db_manager.get_eth_lstm_sequence(timeframe)
                    elif symbol == 'SOL':
                        data = self.db_manager.get_sol_lstm_sequence(timeframe)
                    else:
                        raise ValueError(f"Непідтримуваний символ: {symbol}")

                    if data is not None and len(data) > 0:
                        self.logger.info(f"Успішно завантажено {len(data)} записів для {symbol}-{timeframe}")
                        # Якщо data - це список словників, можна перевірити ключі першого елемента
                        if isinstance(data, list) and len(data) > 0:
                            self.logger.debug(f"Ключі у завантажених даних: {list(data[0].keys())}")
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
        Валідація попередньо підготовлених даних з новою структурою колонок

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

        # Список очікуваних колонок
        expected_columns = [
            'id', 'timeframe', 'sequence_id', 'sequence_position', 'open_time',
            'open_scaled', 'high_scaled', 'low_scaled', 'close_scaled', 'volume_scaled',
            'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
            'month_sin', 'month_cos', 'day_of_month_sin', 'day_of_month_cos',
            'target_close_1', 'target_close_5', 'target_close_10',
            'sequence_length', 'scaling_metadata', 'created_at', 'updated_at'
        ]

        # Перевірка наявності основних колонок
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            self.logger.warning(f"Відсутні колонки для {symbol}: {missing_columns}")

        # Перевірка наявності цільових змінних
        target_columns = ['target_close_1', 'target_close_5', 'target_close_10']
        available_targets = [col for col in target_columns if col in df.columns]
        if not available_targets:
            self.logger.error(f"Жодної цільової змінної не знайдено для {symbol}")
            raise ValueError(f"Жодної цільової змінної не знайдено для {symbol}")
        else:
            self.logger.info(f"Знайдено цільові змінні для {symbol}: {available_targets}")

        # Перевірка часових ознак
        time_features = ['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
                         'month_sin', 'month_cos', 'day_of_month_sin', 'day_of_month_cos']
        available_time_features = [col for col in time_features if col in df.columns]
        self.logger.info(f"Знайдено {len(available_time_features)} часових ознак для {symbol}")

        # Перевірка scaled ознак
        scaled_features = ['open_scaled', 'high_scaled', 'low_scaled', 'close_scaled', 'volume_scaled']
        available_scaled_features = [col for col in scaled_features if col in df.columns]
        self.logger.info(f"Знайдено {len(available_scaled_features)} scaled ознак для {symbol}")

        # Перевірка sequence_id та sequence_position
        if 'sequence_id' in df.columns and 'sequence_position' in df.columns:
            unique_sequences = df['sequence_id'].nunique()
            max_position = df['sequence_position'].max()
            self.logger.info(f"Знайдено {unique_sequences} унікальних послідовностей "
                             f"з максимальною позицією {max_position} для {symbol}")
        else:
            self.logger.warning(f"Колонки sequence_id або sequence_position відсутні для {symbol}")

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

    def extract_sequences_from_prepared_data(self, df: pd.DataFrame, target_column: str = 'target_close_1') -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Витягування послідовностей з підготовлених даних використовуючи sequence_id та sequence_position

        Args:
            df: DataFrame з підготовленими даними
            target_column: Назва цільової колонки

        Returns:
            Кортеж (X, y) де X - послідовності ознак, y - цільові значення
        """
        self.logger.info(f"Витягування послідовностей з підготовлених даних")

        # Перевірка наявності необхідних колонок
        if 'sequence_id' not in df.columns or 'sequence_position' not in df.columns:
            self.logger.error("Колонки sequence_id або sequence_position відсутні")
            raise ValueError("Колонки sequence_id або sequence_position відсутні")

        if target_column not in df.columns:
            self.logger.error(f"Цільова колонка {target_column} відсутня")
            raise ValueError(f"Цільова колонка {target_column} відсутня")

        # Визначення ознак для моделі (виключаємо службові колонки)
        exclude_columns = [
            'id', 'timeframe', 'sequence_id', 'sequence_position', 'open_time',
            'target_close_1', 'target_close_5', 'target_close_10',
            'sequence_length', 'scaling_metadata', 'created_at', 'updated_at'
        ]

        feature_columns = [col for col in df.columns if col not in exclude_columns]
        self.logger.info(f"Використовуємо {len(feature_columns)} ознак: {feature_columns}")

        # Сортування даних по sequence_id та sequence_position
        df_sorted = df.sort_values(['sequence_id', 'sequence_position'])

        # Групування по sequence_id
        sequences = []
        targets = []

        for sequence_id, group in df_sorted.groupby('sequence_id'):
            # Перевірка послідовності позицій
            positions = group['sequence_position'].values
            if not np.array_equal(positions, np.arange(len(positions))):
                self.logger.warning(f"Непослідовні позиції в sequence_id {sequence_id}")
                continue

            # Витягування ознак та цільового значення
            sequence_features = group[feature_columns].values
            # Берем цільове значення з останньої позиції послідовності
            target_value = group[target_column].iloc[-1]

            if not np.isnan(target_value):
                sequences.append(sequence_features)
                targets.append(target_value)

        if not sequences:
            self.logger.error("Не вдалося витягти жодної валідної послідовності")
            raise ValueError("Не вдалося витягати жодної валідної послідовності")

        # Перетворення в numpy arrays
        X = np.array(sequences)
        y = np.array(targets)

        self.logger.info(f"Витягнуто {len(sequences)} послідовностей: X {X.shape}, y {y.shape}")
        return X, y

    def preprocess_prepared_data_for_model(self, data: pd.DataFrame, symbol: str, timeframe: str,
                                           model_type: str = 'lstm', validation_split: float = 0.2,
                                           target_column: str = 'target_close_1',
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
            target_column: Назва цільової колонки
            config: Опціональна конфігурація моделі

        Returns:
            Кортеж (X_train, y_train, X_val, y_val, model_config)
        """
        self.logger.info(f"Початок обробки підготовлених даних для {symbol}-{timeframe}-{model_type}")

        # Валідація підготовлених даних
        self.validate_prepared_data(data, symbol)

        # Витягування послідовностей з підготовлених даних
        X, y = self.extract_sequences_from_prepared_data(data, target_column)

        # Створення або отримання конфігурації моделі
        if config is None:
            input_dim = X.shape[2]  # Кількість ознак
            actual_seq_length = X.shape[1]  # Фактична довжина послідовності

            config = self.create_model_config(input_dim, timeframe)
            # Оновлюємо sequence_length до фактичного значення
            config.sequence_length = actual_seq_length

            self.set_model_config(symbol, timeframe, model_type, config)
            self.logger.info(f"Створено нову конфігурацію моделі для {symbol}-{timeframe}-{model_type}")
        else:
            # Оновлення конфігурації якщо потрібно
            if config.input_dim != X.shape[2]:
                old_dim = config.input_dim
                config.input_dim = X.shape[2]
                self.logger.info(f"Оновлено input_dim з {old_dim} на {config.input_dim}")

            if config.sequence_length != X.shape[1]:
                old_len = config.sequence_length
                config.sequence_length = X.shape[1]
                self.logger.info(f"Оновлено sequence_length з {old_len} на {config.sequence_length}")

        # Перевірка на NaN та Inf
        if np.isnan(X).any() or np.isinf(X).any():
            self.logger.error(f"Знайдено NaN або Inf значення у ознаках для {symbol}-{timeframe}")
            # Заміна NaN та Inf на 0 (або інше значення за замовчуванням)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            self.logger.warning(f"NaN та Inf значення замінено на 0 для {symbol}-{timeframe}")

        if np.isnan(y).any() or np.isinf(y).any():
            self.logger.error(f"Знайдено NaN або Inf значення у цільових значеннях для {symbol}-{timeframe}")
            raise ValueError("NaN або Inf значення у цільових змінних неприпустимі")

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

    def prepare_data_with_config(self, symbol: str, timeframe: str, model_type: str,
                                 validation_split: float = 0.2,
                                 target_column: str = 'target_close_1') -> Tuple[torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, ModelConfig]:
        """Підготовка попередньо оброблених даних з автоматичним створенням конфігурації"""
        self.logger.info(f"Підготовка даних з конфігурацією для {symbol}-{timeframe}-{model_type}")

        # Завантаження попередньо підготовлених даних
        try:
            prepared_data = self.get_data_with_fallback(symbol, timeframe)
            self.logger.info(f"Завантажено підготовлені дані для {symbol}-{timeframe}: {prepared_data.shape}")
        except Exception as e:
            self.logger.error(f"Критична помилка завантаження даних для {symbol}-{timeframe}: {e}")
            raise

        # Обробка для моделі
        try:
            result = self.preprocess_prepared_data_for_model(
                prepared_data, symbol, timeframe, model_type, validation_split, target_column
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
                    'has_target': any('target' in col for col in data.columns)
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

    def prepare_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Підготовка ознак для моделі з використанням всіх доступних модулів"""
        try:
            self.logger.info(f"Початок підготовки ознак для {symbol}")
            # Список для зберігання DataFrames перед об'єднанням
            feature_dataframes = []

            # Базові OHLCV ознаки (з оригінального DataFrame)
            self.logger.debug(f"Додавання базових OHLCV ознак для {symbol}")
            base_features = df[['open', 'high', 'low', 'close', 'volume']].copy()
            feature_dataframes.append(base_features)

            # Отримання трендових ознак
            self.logger.debug(f"Підготовка трендових ознак для {symbol}")
            try:
                trend_features = self.trend.prepare_ml_trend_features(df)
                if trend_features is not None and not trend_features.empty:
                    feature_dataframes.append(trend_features)
                    self.logger.info(f"Додано {trend_features.shape[1]} трендових ознак")
                else:
                    self.logger.warning(f"Трендові ознаки для {symbol} порожні")
            except Exception as e:
                self.logger.error(f"Помилка при створенні трендових ознак для {symbol}: {e}")

            # Отримання ознак волатильності
            self.logger.debug(f"Підготовка ознак волатильності для {symbol}")
            try:
                volatility_features = self.vol.prepare_volatility_features_for_ml(df, symbol)
                if volatility_features is not None and not volatility_features.empty:
                    feature_dataframes.append(volatility_features)
                    self.logger.info(f"Додано {volatility_features.shape[1]} ознак волатильності")
                else:
                    self.logger.warning(f"Ознаки волатильності для {symbol} порожні")
            except Exception as e:
                self.logger.error(f"Помилка при створенні ознак волатильності для {symbol}: {e}")

            # Отримання циклічних ознак
            self.logger.debug(f"Підготовка циклічних ознак для {symbol}")
            try:
                cycle_features = self.cycle.prepare_cycle_ml_features(df, symbol)
                if cycle_features is not None and not cycle_features.empty:
                    feature_dataframes.append(cycle_features)
                    self.logger.info(f"Додано {cycle_features.shape[1]} циклічних ознак")
                else:
                    self.logger.warning(f"Циклічні ознаки для {symbol} порожні")
            except Exception as e:
                self.logger.error(f"Помилка при створенні циклічних ознак для {symbol}: {e}")

            # Отримання технічних індикаторів
            self.logger.debug(f"Підготовка технічних індикаторів для {symbol}")
            try:
                indicator_features = self.indicators.prepare_features_pipeline(df)
                if indicator_features is not None and not indicator_features.empty:
                    feature_dataframes.append(indicator_features)
                    self.logger.info(f"Додано {indicator_features.shape[1]} технічних індикаторів")
                else:
                    self.logger.warning(f"Технічні індикатори для {symbol} порожні")
            except Exception as e:
                self.logger.error(f"Помилка при створенні технічних індикаторів для {symbol}: {e}")

            # Перевірка наявності ознак для об'єднання
            if not feature_dataframes:
                raise ValueError(f"Не вдалося створити жодних ознак для {symbol}")

            # Оптимізоване об'єднання з обробкою індексів
            self.logger.debug(f"Об'єднання {len(feature_dataframes)} наборів ознак для {symbol}")

            # Вирівнюємо всі DataFrame по індексу першого (базового)
            base_index = feature_dataframes[0].index
            aligned_dataframes = []

            for i, features_df in enumerate(feature_dataframes):
                try:
                    # Вирівнюємо по базовому індексу
                    aligned_df = features_df.reindex(base_index, method='ffill')
                    aligned_dataframes.append(aligned_df)
                except Exception as e:
                    self.logger.warning(f"Помилка вирівнювання DataFrame {i} для {symbol}: {e}")
                    # Спробуємо додати як є
                    aligned_dataframes.append(features_df)

            # Об'єднання з обробкою помилок
            try:
                final_features = pd.concat(
                    aligned_dataframes,
                    axis=1,
                    join='outer',  # Зовнішнє об'єднання для збереження всіх індексів
                    copy=False,
                    sort=False
                )
            except Exception as e:
                self.logger.error(f"Помилка при concat для {symbol}: {e}")
                # Fallback - використовуємо тільки базові ознаки
                final_features = feature_dataframes[0].copy()

            # Ефективне видалення дублікатів колонок
            if final_features.columns.duplicated().any():
                self.logger.debug(f"Видалення дублікатних колонок для {symbol}")
                # Отримуємо унікальні колонки, зберігаючи порядок
                unique_columns = final_features.columns[~final_features.columns.duplicated()]
                final_features = final_features[unique_columns]

            # Обробка NaN значень
            initial_nan_count = final_features.isnull().sum().sum()
            if initial_nan_count > 0:
                self.logger.warning(f"Знайдено {initial_nan_count} NaN значень для {symbol}")
                # Заповнення NaN методом forward fill, потім backward fill
                final_features = final_features.fillna(method='ffill').fillna(method='bfill')

                # Якщо все ще є NaN, заповнюємо нулями
                remaining_nan = final_features.isnull().sum().sum()
                if remaining_nan > 0:
                    self.logger.warning(f"Заповнення залишкових {remaining_nan} NaN нулями для {symbol}")
                    final_features = final_features.fillna(0)

            # Додаткова валідація результату
            if final_features.empty:
                raise ValueError(f"Результуючий DataFrame порожній для {symbol}")

            # Видалення колонок з постійними значеннями (нульова дисперсія)
            constant_columns = []
            for col in final_features.columns:
                if final_features[col].nunique() <= 1:
                    constant_columns.append(col)

            if constant_columns:
                self.logger.warning(f"Видалення {len(constant_columns)} колонок з постійними значеннями для {symbol}")
                final_features = final_features.drop(columns=constant_columns)

            # Збереження інформації про ознаки
            feature_info = {
                'total_features': final_features.shape[1],
                'feature_names': list(final_features.columns),
                'nan_count': final_features.isnull().sum().sum(),
                'constant_columns_removed': len(constant_columns)
            }

            key = f"{symbol}_features"
            self.feature_configs[key] = feature_info

            self.logger.info(f"Успішно підготовлено {final_features.shape[1]} ознак для {symbol}")
            self.logger.debug(f"Форма фінального DataFrame: {final_features.shape}")

            return final_features

        except Exception as e:
            self.logger.error(f"Критична помилка при підготовці ознак для {symbol}: {e}")
            # Повертаємо базовий набір ознак як fallback
            try:
                self.logger.warning(f"Спроба створити базові ознаки для {symbol}")
                basic_features = self.indicators.prepare_features_pipeline(df)
                if basic_features is not None and not basic_features.empty:
                    self.logger.info(f"Повернуто базові ознаки для {symbol}: {basic_features.shape}")
                    return basic_features
                else:
                    # Останній fallback - тільки OHLCV
                    self.logger.warning(f"Повертаємо тільки OHLCV для {symbol}")
                    return df[['open', 'high', 'low', 'close', 'volume']].copy()
            except Exception as fallback_error:
                self.logger.error(f"Помилка fallback для {symbol}: {fallback_error}")
                # Критичний fallback - тільки ціна закриття
                return pd.DataFrame({'close': df['close']})

    def create_enhanced_sequences_with_features(self, df: pd.DataFrame, symbol: str,
                                                sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Створення послідовностей з розширеними ознаками для навчання моделі
        """
        self.logger.info(f"Створення розширених послідовностей для {symbol}")

        try:
            # Підготовка розширених ознак
            enhanced_features = self.prepare_features(df, symbol)

            # Створення цільової змінної (прогноз ціни закриття на наступний період)
            target = enhanced_features['close'].shift(-1).dropna()

            # Видалення останнього рядка з ознак (для якого немає цілі)
            features_aligned = enhanced_features.iloc[:-1]

            # Перевірка вирівнювання
            if len(features_aligned) != len(target):
                min_len = min(len(features_aligned), len(target))
                features_aligned = features_aligned.iloc[:min_len]
                target = target.iloc[:min_len]
                self.logger.warning(f"Вирівняно дані до {min_len} записів для {symbol}")

            # Створення послідовностей
            sequences = []
            targets = []

            for i in range(sequence_length, len(features_aligned)):
                # Послідовність ознак
                sequence = features_aligned.iloc[i - sequence_length:i].values
                sequences.append(sequence)

                # Цільове значення
                targets.append(target.iloc[i])

            if not sequences:
                raise ValueError(f"Не вдалося створити послідовності для {symbol}")

            X = np.array(sequences)
            y = np.array(targets)

            self.logger.info(f"Створено {len(sequences)} послідовностей для {symbol}: "
                             f"X shape: {X.shape}, y shape: {y.shape}")

            return X, y

        except Exception as e:
            self.logger.error(f"Помилка створення розширених послідовностей для {symbol}: {e}")
            raise
