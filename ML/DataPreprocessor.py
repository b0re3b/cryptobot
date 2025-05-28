from dataclasses import field, dataclass
from typing import Tuple, Callable, List, Optional

import numpy as np
import pandas as pd
import torch

from analysis import VolatilityAnalysis
from cyclefeatures import CryptoCycles
from data.db import DatabaseManager
from featureengineering.feature_engineering import FeatureEngineering
from trends.trend_detection import TrendDetection
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
    timeframes: List[str] = field(default_factory=lambda: ['4h'])
    model_types: List[str] = field(default_factory=lambda: ['lstm', 'gru','transformer'])


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

    def set_model_config(self, symbol: str, timeframe: str, model_type: str, config: ModelConfig):
        """Збереження конфігурації моделі"""
        key = f"{symbol}_{timeframe}_{model_type}"
        self.model_configs[key] = config
        self.logger.info(f"Збережено конфігурацію моделі для {key}")

    def get_data_loader(self, symbol: str, timeframe: str, model_type: str) -> Callable:
        """Create data loader that returns a DataFrame with target column"""
        try:
            self.logger.info(f"Preparing data loader for {symbol} with timeframe {timeframe} "
                             f"and model {model_type}")

            if not hasattr(self, 'db_manager') or self.db_manager is None:
                self.logger.error("db_manager not initialized in class")
                raise ValueError("db_manager is not set")

            symbol_methods = {
                'BTC': self.db_manager.get_btc_lstm_sequence,
                'ETH': self.db_manager.get_eth_lstm_sequence,
                'SOL': self.db_manager.get_sol_lstm_sequence
            }

            if symbol not in symbol_methods:
                self.logger.error(f"Unsupported symbol: {symbol}")
                raise ValueError(f"Unsupported symbol: {symbol}")

            def data_loader():
                try:
                    self.logger.debug(f"Loading data for {symbol} with timeframe {timeframe}")

                    # Get raw data
                    if symbol == 'BTC':
                        raw_data = self.db_manager.get_btc_lstm_sequence(timeframe)
                    elif symbol == 'ETH':
                        raw_data = self.db_manager.get_eth_lstm_sequence(timeframe)
                    elif symbol == 'SOL':
                        raw_data = self.db_manager.get_sol_lstm_sequence(timeframe)
                    else:
                        raise ValueError(f"Unsupported symbol: {symbol}")

                    if raw_data is None or len(raw_data) == 0:
                        self.logger.warning(f"Loaded data for {symbol}-{timeframe} is empty")
                        return pd.DataFrame()

                    # Convert to DataFrame if needed
                    if isinstance(raw_data, list):
                        df = pd.DataFrame(raw_data)
                    elif isinstance(raw_data, pd.DataFrame):
                        df = raw_data.copy()
                    else:
                        raise ValueError(f"Unknown data type: {type(raw_data)}")

                    # *** КРИТИЧНЕ ВИПРАВЛЕННЯ: Очистка та конвертація типів даних ***

                    # Список числових колонок які повинні бути float
                    numeric_columns = [
                        'open_scaled', 'high_scaled', 'low_scaled', 'close_scaled', 'volume_scaled',
                        'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
                        'month_sin', 'month_cos', 'day_of_month_sin', 'day_of_month_cos',
                        'target_close_1', 'target_close_5', 'target_close_10'
                    ]

                    # Додаємо всі колонки які закінчуються на _scaled
                    additional_numeric = [col for col in df.columns if col.endswith('_scaled')]
                    numeric_columns.extend(additional_numeric)

                    # Видаляємо дублікати
                    numeric_columns = list(set(numeric_columns))

                    # Конвертація числових колонок
                    for col in numeric_columns:
                        if col in df.columns:
                            try:
                                # Замінюємо None на NaN
                                df[col] = df[col].replace([None, 'None', 'null'], np.nan)

                                # Конвертуємо в числовий тип
                                df[col] = pd.to_numeric(df[col], errors='coerce')

                                # Логування якщо є проблеми
                                if df[col].dtype == 'object':
                                    unique_values = df[col].unique()[:10]  # Перші 10 унікальних значень
                                    self.logger.warning(
                                        f"Колонка {col} все ще має object dtype. Унікальні значення: {unique_values}")

                            except Exception as e:
                                self.logger.error(f"Помилка конвертації колонки {col}: {e}")

                    # Ensure we have a target column - create from next period's close price if not exists
                    if 'target' not in df.columns and 'close' in df.columns:
                        df['target'] = df['close'].shift(-1)
                        df.dropna(subset=['target'], inplace=True)
                        self.logger.info(f"Created target column from next period close prices")

                    # *** ДОДАТКОВА ПЕРЕВІРКА ТИПІВ ДАНИХ ***

                    # Перевірка що всі числові колонки дійсно числові
                    problematic_columns = []
                    for col in numeric_columns:
                        if col in df.columns and df[col].dtype == 'object':
                            problematic_columns.append(col)

                    if problematic_columns:
                        self.logger.error(
                            f"КРИТИЧНА ПОМИЛКА: Наступні колонки все ще мають object dtype: {problematic_columns}")

                        # Детальна діагностика
                        for col in problematic_columns:
                            sample_values = df[col].dropna().head(10).tolist()
                            self.logger.error(f"Приклади значень з {col}: {sample_values}")

                        raise ValueError(f"Не вдалося конвертувати колонки в числовий тип: {problematic_columns}")

                    self.logger.info(f"Successfully loaded {len(df)} records for {symbol}-{timeframe}")

                    # Логування типів даних для діагностики
                    numeric_dtypes = {col: str(df[col].dtype) for col in numeric_columns if col in df.columns}
                    self.logger.debug(f"Типи даних числових колонок: {numeric_dtypes}")

                    return df

                except Exception as e:
                    self.logger.error(f"Error loading data in data_loader for {symbol}: {str(e)}")
                    return pd.DataFrame()

            return data_loader

        except Exception as e:
            self.logger.error(f"Error preparing data loader for {symbol}: {str(e)}")
            raise


    def validate_prepared_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """Валідація підготовлених даних"""
        self.logger.info(f"Валідація підготовлених даних для {symbol}")

        # Перевірка що це DataFrame
        if not isinstance(df, pd.DataFrame):
            self.logger.error(f"Дані для {symbol} не є DataFrame: {type(df)}")
            raise ValueError(f"Очікується DataFrame, отримано {type(df)}")

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

        # Валідація таймфрейму в даних
        if 'timeframe' in df.columns:
            unique_timeframes = df['timeframe'].unique()
            self.logger.info(f"Унікальні таймфрейми в даних для {symbol}: {unique_timeframes}")

        # НЕ ЛОГУЄМО NaN для lag-колонок, оскільки це нормально
        self._validate_nan_values(df, symbol)

        self.logger.info(f"Валідація завершена для {symbol}: форма даних {df.shape}")
        return True

    def _validate_nan_values(self, df: pd.DataFrame, symbol: str):
        """Валідація NaN значень з урахуванням lag-колонок"""
        # Виключаємо lag-колонки з перевірки NaN, оскільки для них це нормально
        lag_columns = [col for col in df.columns if '_lag_' in col]
        non_lag_columns = [col for col in df.columns if '_lag_' not in col]

        if non_lag_columns:
            non_lag_df = df[non_lag_columns]
            nan_counts = non_lag_df.isnull().sum()
            total_nans = nan_counts.sum()

            if total_nans > 0:
                self.logger.warning(f"Знайдено {total_nans} NaN значень у non-lag колонках для {symbol}")
                # Логування топ-колонок з найбільшою кількістю NaN
                top_nan_cols = nan_counts[nan_counts > 0].head(5)
                for col, count in top_nan_cols.items():
                    self.logger.debug(f"  {col}: {count} NaN значень")
            else:
                self.logger.info(f"NaN значення в non-lag колонках не знайдені для {symbol}")

        if lag_columns:
            self.logger.debug(f"Знайдено {len(lag_columns)} lag-колонок для {symbol} (NaN в них є нормальним)")

    def extract_sequences_from_prepared_data(self, df: pd.DataFrame, target_column: str = 'target_close_1') -> Tuple[
        np.ndarray, np.ndarray]:
        """Витягування послідовностей з підготовлених даних"""
        self.logger.info(f"Витягування послідовностей з підготовлених даних")

        # Перевірка що це DataFrame
        if not isinstance(df, pd.DataFrame):
            self.logger.error(f"Дані не є DataFrame: {type(df)}")
            raise ValueError(f"Очікується DataFrame, отримано {type(df)}")

        # Перевірка наявності необхідних колонок
        if 'sequence_id' not in df.columns or 'sequence_position' not in df.columns:
            self.logger.error("Колонки sequence_id або sequence_position відсутні")
            raise ValueError("Колонки sequence_id або sequence_position відсутні")

        if target_column not in df.columns:
            self.logger.error(f"Цільова колонка {target_column} відсутня")
            raise ValueError(f"Цільова колонка {target_column} відсутня")

        exclude_columns = [
            'id', 'timeframe', 'sequence_id', 'sequence_position', 'open_time',
            'target_close_1', 'target_close_5', 'target_close_10',
            'sequence_length', 'scaling_metadata', 'created_at', 'updated_at'
        ]

        # Додаємо всі lag-колонки до виключених
        lag_columns = [col for col in df.columns if '_lag_' in col]
        exclude_columns.extend(lag_columns)

        feature_columns = [col for col in df.columns if col not in exclude_columns]
        self.logger.info(f"Використовуємо {len(feature_columns)} ознак: {feature_columns[:10]}...")

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

            # *** КРИТИЧНЕ ВИПРАВЛЕННЯ: Перевірка та конвертація типів ***

            # Перевірка типу даних
            if sequence_features.dtype == 'object':
                self.logger.warning(f"Знайдено object dtype в sequence_id {sequence_id}, спроба конвертації")
                try:
                    # Спроба конвертації в float64
                    sequence_features = sequence_features.astype(np.float64)
                    self.logger.debug(f"Успішно конвертовано sequence_id {sequence_id} в float64")
                except (ValueError, TypeError) as e:
                    self.logger.error(f"Не вдалося конвертувати sequence_id {sequence_id}: {e}")
                    # Детальна діагностика
                    for i, col in enumerate(feature_columns):
                        col_data = group[col].values
                        if col_data.dtype == 'object':
                            self.logger.error(f"Колонка {col} має object dtype: {col_data[:5]}")
                    continue

            # Перевірка на правильність форми
            if len(sequence_features.shape) != 2:
                self.logger.error(
                    f"Неправильна форма sequence_features для sequence_id {sequence_id}: {sequence_features.shape}")
                continue

            # Перевірка на NaN в ознаках (після виключення lag-колонок)
            if np.isnan(sequence_features).any():
                nan_count = np.isnan(sequence_features).sum()
                self.logger.warning(f"NaN значення ({nan_count}) в ознаках для sequence_id {sequence_id}, пропускаємо")
                continue

            # Перевірка на Inf значення
            if np.isinf(sequence_features).any():
                inf_count = np.isinf(sequence_features).sum()
                self.logger.warning(f"Inf значення ({inf_count}) в ознаках для sequence_id {sequence_id}, пропускаємо")
                continue

            # Берем цільове значення з останньої позиції послідовності
            target_value = group[target_column].iloc[-1]

            # Конвертація цільового значення
            try:
                target_value = float(target_value)
            except (ValueError, TypeError):
                self.logger.warning(
                    f"Неможливо конвертувати target_value в float для sequence_id {sequence_id}: {target_value}")
                continue

            if not np.isnan(target_value) and not np.isinf(target_value):
                sequences.append(sequence_features)
                targets.append(target_value)

        if not sequences:
            self.logger.error("Не вдалося витягти жодної валідної послідовності")
            raise ValueError("Не вдалося витягати жодної валідної послідовності")

        # *** КРИТИЧНЕ ВИПРАВЛЕННЯ: Гарантія правильного dtype ***

        # Перетворення в numpy arrays з явним dtype
        try:
            X = np.array(sequences, dtype=np.float64)
            y = np.array(targets, dtype=np.float64)

            # Додаткова перевірка
            if X.dtype == 'object':
                self.logger.error("КРИТИЧНА ПОМИЛКА: X все ще має object dtype після конвертації")
                # Спроба форсованої конвертації
                X = np.stack(sequences).astype(np.float64)

            if y.dtype == 'object':
                self.logger.error("КРИТИЧНА ПОМИЛКА: y все ще має object dtype після конвертації")
                y = np.array(targets, dtype=np.float64)

        except Exception as e:
            self.logger.error(f"Критична помилка створення numpy arrays: {e}")

            # Діагностика проблеми
            self.logger.error(f"Кількість послідовностей: {len(sequences)}")
            if sequences:
                self.logger.error(f"Форма першої послідовності: {sequences[0].shape}")
                self.logger.error(f"Dtype першої послідовності: {sequences[0].dtype}")

                # Перевірка всіх послідовностей на однакову форму
                shapes = [seq.shape for seq in sequences]
                dtypes = [seq.dtype for seq in sequences]

                unique_shapes = set(shapes)
                unique_dtypes = set(dtypes)

                self.logger.error(f"Унікальні форми: {unique_shapes}")
                self.logger.error(f"Унікальні dtypes: {unique_dtypes}")

                if len(unique_shapes) > 1:
                    self.logger.error("ПРОБЛЕМА: Послідовності мають різні форми!")
                if len(unique_dtypes) > 1:
                    self.logger.error("ПРОБЛЕМА: Послідовності мають різні типи даних!")

            raise

        # Фінальна валідація
        if X.dtype == 'object' or y.dtype == 'object':
            self.logger.error(f"КРИТИЧНА ПОМИЛКА: Залишився object dtype - X: {X.dtype}, y: {y.dtype}")
            raise ValueError("Не вдалося конвертувати дані в числовий тип")

        self.logger.info(f"Витягнуто {len(sequences)} послідовностей: X {X.shape} ({X.dtype}), y {y.shape} ({y.dtype})")
        return X, y

    def preprocess_prepared_data_for_model(self, data: pd.DataFrame, symbol: str, timeframe: str,
                                           model_type: str, validation_split: float = 0.2,
                                           target_column: str = 'target_close_1',
                                           config: Optional[ModelConfig] = None) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ModelConfig]:
        """Обробка підготовлених даних для моделі"""
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
            self.logger.error(f"КРИТИЧНА ПОМИЛКА: Знайдено NaN або Inf значення у ознаках для {symbol}-{timeframe}")
            self.logger.error("Це не повинно відбуватися після правильного виключення lag-колонок")
            raise ValueError("NaN або Inf значення у ознаках після виключення lag-колонок")

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
                                 target_column: str = 'target_close_1') -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ModelConfig]:
        """Підготовка попередньо оброблених даних з автоматичним створенням конфігурації"""
        self.logger.info(f"Підготовка даних з конфігурацією для {symbol}-{timeframe}-{model_type}")

        # Завантаження попередньо підготовлених даних
        try:
            data_loader = self.get_data_loader(symbol, timeframe, model_type)
            data = data_loader()

            if data is None or len(data) == 0:
                raise ValueError(f"Не вдалося завантажити дані для {symbol}-{timeframe}")

            # Конвертація в DataFrame якщо потрібно
            if isinstance(data, list):
                prepared_data = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                prepared_data = data.copy()
            else:
                raise ValueError(f"Невідомий тип даних: {type(data)}")

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


    # def prepare_features(self, data, symbol: str) -> DataFrame | tuple[DataFrame, Series] | Any:
    #     """Підготовка ознак для моделі з використанням всіх доступних модулів та підтримкою chunked даних"""
    #     try:
    #         self.logger.info(f"Початок підготовки ознак для {symbol}")
    #
    #         # Перевірка типу вхідних даних та конвертація в DataFrame
    #         if isinstance(data, list):
    #             self.logger.info(f"Конвертація списку словників в DataFrame для {symbol}")
    #             if not data:
    #                 raise ValueError(f"Порожній список даних для {symbol}")
    #
    #             if not isinstance(data[0], dict):
    #                 raise ValueError(f"Елементи списку повинні бути словниками для {symbol}")
    #
    #             df = pd.DataFrame(data)
    #             self.logger.info(f"Успішно конвертовано {len(data)} записів в DataFrame для {symbol}")
    #
    #         elif isinstance(data, pd.DataFrame):
    #             df = data.copy()
    #             self.logger.debug(f"Використовується існуючий DataFrame для {symbol}")
    #
    #         elif isinstance(data, Generator):
    #             # Обробка генератора чанків
    #             self.logger.info(f"Обробка генератора чанків для {symbol}")
    #             chunks = []
    #             for chunk in data:
    #                 if isinstance(chunk, pd.DataFrame) and not chunk.empty:
    #                     chunks.append(chunk)
    #
    #             if not chunks:
    #                 raise ValueError(f"Не отримано жодного валідного чанку для {symbol}")
    #
    #             df = pd.concat(chunks, ignore_index=False)
    #             self.logger.info(f"Об'єднано {len(chunks)} чанків для {symbol}, загальна форма: {df.shape}")
    #
    #         else:
    #             self.logger.error(f"Невідомий тип вхідних даних для {symbol}: {type(data)}")
    #             raise ValueError(f"Очікується DataFrame, список словників або генератор, отримано {type(data)}")
    #
    #         # Перевірка що DataFrame не порожній
    #         if df.empty:
    #             raise ValueError(f"DataFrame порожній для {symbol}")
    #
    #         # Логування інформації про DataFrame
    #         self.logger.debug(f"DataFrame для {symbol}: форма {df.shape}, колонки {list(df.columns)}")
    #
    #         # Список для зберігання DataFrames перед об'єднанням
    #         feature_dataframes = []
    #
    #         # Базові OHLCV ознаки (з оригінального DataFrame)
    #         self.logger.debug(f"Додавання базових OHLCV ознак для {symbol}")
    #
    #         # Перевірка наявності базових колонок (можуть бути з суфіксами _scaled)
    #         potential_base_columns = ['open', 'high', 'low', 'close', 'volume',
    #                                   'open_scaled', 'high_scaled', 'low_scaled', 'close_scaled', 'volume_scaled']
    #         available_base_columns = [col for col in potential_base_columns if col in df.columns]
    #
    #         if not available_base_columns:
    #             self.logger.error(f"Жодної базової колонки не знайдено для {symbol}")
    #             self.logger.debug(f"Доступні колонки: {list(df.columns)}")
    #             raise ValueError(f"Відсутні необхідні базові колонки для {symbol}")
    #
    #         # Використовуємо доступні базові колонки
    #         base_features = df[available_base_columns].copy()
    #         feature_dataframes.append(base_features)
    #         self.logger.info(f"Додано {len(available_base_columns)} базових ознак для {symbol}")
    #
    #         # Для аналізу потрібні unscaled дані
    #         analysis_df = df.copy()
    #         scaled_columns = [col for col in available_base_columns if col.endswith('_scaled')]
    #         unscaled_map = {
    #             'open_scaled': 'open',
    #             'high_scaled': 'high',
    #             'low_scaled': 'low',
    #             'close_scaled': 'close',
    #             'volume_scaled': 'volume'
    #         }
    #
    #         # Створимо unscaled версії якщо їх немає (для технічного аналізу)
    #         for scaled_col in scaled_columns:
    #             unscaled_col = unscaled_map.get(scaled_col)
    #             if unscaled_col and unscaled_col not in analysis_df.columns:
    #                 analysis_df[unscaled_col] = analysis_df[scaled_col]
    #                 self.logger.debug(f"Створено {unscaled_col} з {scaled_col} для аналізу")
    #
    #         # Отримання трендових ознак
    #         self.logger.debug(f"Підготовка трендових ознак для {symbol}")
    #         try:
    #             trend_result = self.trend.prepare_ml_trend_features(analysis_df)
    #
    #             if trend_result is not None and len(trend_result) == 3:
    #                 X, y, regimes = trend_result
    #                 if X is not None and len(X) > 0:
    #                     trend_features_array = X[:, -1, :] if len(X.shape) == 3 else X
    #                     trend_feature_names = [
    #                         'close_norm', 'volume_norm', 'adx_norm', 'di_plus_norm', 'di_minus_norm',
    #                         'rsi_norm', 'macd_norm', 'macd_signal_norm', 'trend_strength_norm',
    #                         'speed_20_norm', 'volatility_20_norm'
    #                     ]
    #
    #                     if trend_features_array.shape[1] > len(trend_feature_names):
    #                         trend_feature_names.extend([f'trend_feature_{i}' for i in
    #                                                     range(len(trend_feature_names), trend_features_array.shape[1])])
    #                     elif trend_features_array.shape[1] < len(trend_feature_names):
    #                         trend_feature_names = trend_feature_names[:trend_features_array.shape[1]]
    #
    #                     start_idx = len(analysis_df) - len(trend_features_array)
    #                     trend_index = analysis_df.index[start_idx:]
    #
    #                     trend_features = pd.DataFrame(
    #                         trend_features_array,
    #                         index=trend_index,
    #                         columns=trend_feature_names
    #                     )
    #
    #                     if regimes is not None and len(regimes) == len(trend_features):
    #                         trend_features['market_regime'] = regimes
    #
    #                     feature_dataframes.append(trend_features)
    #                     self.logger.info(f"Додано {trend_features.shape[1]} трендових ознак")
    #                 else:
    #                     self.logger.warning(f"Трендові ознаки для {symbol} порожні")
    #             else:
    #                 self.logger.warning(f"Неправильний формат результату трендових ознак для {symbol}")
    #         except Exception as e:
    #             self.logger.error(f"Помилка при створенні трендових ознак для {symbol}: {e}")
    #
    #         # Отримання ознак волатільності
    #         self.logger.debug(f"Підготовка ознак волатільності для {symbol}")
    #         try:
    #             volatility_features = self.vol.prepare_volatility_features_for_ml(analysis_df, symbol)
    #             if volatility_features is not None and not volatility_features.empty:
    #                 if isinstance(volatility_features, pd.DataFrame):
    #                     feature_dataframes.append(volatility_features)
    #                     self.logger.info(f"Додано {volatility_features.shape[1]} ознак волатільності")
    #                 else:
    #                     self.logger.warning(
    #                         f"Ознаки волатільності не є DataFrame для {symbol}: {type(volatility_features)}")
    #             else:
    #                 self.logger.warning(f"Ознаки волатільності для {symbol} порожні")
    #         except Exception as e:
    #             self.logger.error(f"Помилка при створенні ознак волатільності для {symbol}: {e}")
    #
    #         # Отримання технічних індикаторів
    #         self.logger.debug(f"Підготовка технічних індикаторів для {symbol}")
    #         try:
    #             indicator_features = self.indicators.prepare_features_pipeline(analysis_df)
    #             if indicator_features is not None and not indicator_features.empty:
    #                 if isinstance(indicator_features, pd.DataFrame):
    #                     feature_dataframes.append(indicator_features)
    #                     self.logger.info(f"Додано {indicator_features.shape[1]} технічних індикаторів")
    #                 else:
    #                     self.logger.warning(
    #                         f"Технічні індикатори не є DataFrame для {symbol}: {type(indicator_features)}")
    #             else:
    #                 self.logger.warning(f"Технічні індикатори для {symbol} порожні")
    #         except Exception as e:
    #             self.logger.error(f"Помилка при створенні технічних індикаторів для {symbol}: {e}")
    #
    #         # Перевірка наявності ознак для об'єднання
    #         if not feature_dataframes:
    #             raise ValueError(f"Не вдалося створити жодних ознак для {symbol}")
    #
    #         # Оптимізоване об'єднання з обробкою індексів
    #         self.logger.debug(f"Об'єднання {len(feature_dataframes)} наборів ознак для {symbol}")
    #
    #         # Знаходимо спільний індекс для всіх DataFrames
    #         common_index = feature_dataframes[0].index
    #         for df_features in feature_dataframes[1:]:
    #             common_index = common_index.intersection(df_features.index)
    #
    #         if len(common_index) == 0:
    #             self.logger.warning(f"Немає спільного індексу між DataFrames для {symbol}, використовуємо outer join")
    #             common_index = None
    #
    #         # Вирівнюємо всі DataFrame
    #         aligned_dataframes = []
    #         for i, features_df in enumerate(feature_dataframes):
    #             try:
    #                 if not isinstance(features_df, pd.DataFrame):
    #                     self.logger.warning(f"DataFrame {i} не є DataFrame для {symbol}: {type(features_df)}")
    #                     continue
    #
    #                 if common_index is not None:
    #                     aligned_df = features_df.reindex(common_index, method='ffill')
    #                 else:
    #                     aligned_df = features_df.copy()
    #
    #                 aligned_dataframes.append(aligned_df)
    #             except Exception as e:
    #                 self.logger.warning(f"Помилка вирівнювання DataFrame {i} для {symbol}: {e}")
    #                 if isinstance(features_df, pd.DataFrame):
    #                     aligned_dataframes.append(features_df)
    #
    #         # Перевірка що є що об'єднувати
    #         if not aligned_dataframes:
    #             self.logger.error(f"Немає валідних DataFrame для об'єднання для {symbol}")
    #             raise ValueError(f"Немає валідних DataFrame для об'єднання для {symbol}")
    #
    #         # Об'єднання з обробкою помилок
    #         try:
    #             final_features = pd.concat(
    #                 aligned_dataframes,
    #                 axis=1,
    #                 join='outer',
    #                 copy=False,
    #                 sort=False
    #             )
    #         except Exception as e:
    #             self.logger.error(f"Помилка при concat для {symbol}: {e}")
    #             final_features = feature_dataframes[0].copy()
    #
    #         # Ефективне видалення дублікатів колонок
    #         if final_features.columns.duplicated().any():
    #             unique_columns = final_features.columns[~final_features.columns.duplicated()]
    #             final_features = final_features[unique_columns]
    #
    #         # Обробка NaN значень
    #         initial_nan_count = final_features.isnull().sum().sum()
    #         if initial_nan_count > 0:
    #             self.logger.warning(f"Знайдено {initial_nan_count} NaN значень для {symbol}")
    #             final_features = final_features.fillna(method='ffill').fillna(method='bfill')
    #             remaining_nan = final_features.isnull().sum().sum()
    #             if remaining_nan > 0:
    #                 self.logger.warning(f"Заповнення залишкових {remaining_nan} NaN нулями для {symbol}")
    #                 final_features = final_features.fillna(0)
    #
    #         # Додаткова валідація результату
    #         if final_features.empty:
    #             raise ValueError(f"Результуючий DataFrame порожній для {symbol}")
    #
    #         # Видалення колонок з постійними значеннями
    #         constant_columns = []
    #         for col in final_features.columns:
    #             if final_features[col].nunique() <= 1:
    #                 constant_columns.append(col)
    #
    #         if constant_columns:
    #             self.logger.warning(f"Видалення {len(constant_columns)} колонок з постійними значеннями для {symbol}")
    #             final_features = final_features.drop(columns=constant_columns)
    #
    #         # Збереження інформації про ознаки
    #         feature_info = {
    #             'total_features': final_features.shape[1],
    #             'feature_names': list(final_features.columns),
    #             'nan_count': final_features.isnull().sum().sum(),
    #             'constant_columns_removed': len(constant_columns)
    #         }
    #
    #         key = f"{symbol}_features"
    #         self.feature_configs[key] = feature_info
    #
    #         self.logger.info(f"Успішно підготовлено {final_features.shape[1]} ознак для {symbol}")
    #         self.logger.debug(f"Форма фінального DataFrame: {final_features.shape}")
    #
    #         return final_features
    #
    #     except Exception as e:
    #         self.logger.error(f"Критична помилка при підготовці ознак для {symbol}: {e}")
    #         # Повертаємо базовий набір ознак як fallback
    #         try:
    #             self.logger.warning(f"Спроба створити базові ознаки для {symbol}")
    #
    #             if isinstance(data, list):
    #                 if not data:
    #                     raise ValueError(f"Порожній список даних для fallback {symbol}")
    #                 df = pd.DataFrame(data)
    #             elif isinstance(data, pd.DataFrame):
    #                 df = data.copy()
    #             elif isinstance(data, Generator):
    #                 chunks = [chunk for chunk in data if isinstance(chunk, pd.DataFrame) and not chunk.empty]
    #                 df = pd.concat(chunks, ignore_index=False) if chunks else pd.DataFrame()
    #             else:
    #                 raise ValueError(f"Невідомий тип даних для fallback {symbol}: {type(data)}")
    #
    #             if not df.empty:
    #                 basic_columns = ['open', 'high', 'low', 'close', 'volume',
    #                                  'open_scaled', 'high_scaled', 'low_scaled', 'close_scaled', 'volume_scaled']
    #                 available_basic = [col for col in basic_columns if col in df.columns]
    #
    #                 if available_basic:
    #                     basic_features = df[available_basic].copy()
    #                     self.logger.info(f"Повернуто базові ознаки для {symbol}: {basic_features.shape}")
    #                     return basic_features
    #                 else:
    #                     numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    #                     if numeric_columns:
    #                         return df[numeric_columns].copy()
    #                     else:
    #                         raise ValueError(f"Жодної числової колонки не знайдено для {symbol}")
    #             else:
    #                 raise ValueError(f"DataFrame невалідний для {symbol}")
    #         except Exception as fallback_error:
    #             self.logger.error(f"Помилка fallback для {symbol}: {fallback_error}")
    #             raise e

