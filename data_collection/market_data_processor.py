import traceback
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import gc
from datetime import datetime, time

from pandas import DataFrame

from data_collection.AnomalyDetector import AnomalyDetector
from data_collection.DataCleaner import DataCleaner
from data_collection.DataResampler import DataResampler
from data_collection.DataStorageManager import DataStorageManager
from data.db import DatabaseManager


class MarketDataProcessor:

    VALID_TIMEFRAMES = ['1m','1h', '4h', '1d', '1w']
    BASE_TIMEFRAMES = ['1m', '1h', '1d']
    DERIVED_TIMEFRAMES = ['4h', '1w']
    VOLUME_PROFILE_TIMEFRAMES = ['1d', '1w']

    def __init__(self, log_level=logging.INFO, use_multiprocessing=True, chunk_size=100000):
        # Налаштування логування
        self.log_level = log_level
        logging.basicConfig(
            level=self.log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Ініціалізація MarketDataProcessor...")

        # Налаштування оптимізацій
        self.use_multiprocessing = use_multiprocessing
        self.chunk_size = chunk_size
        self.num_workers = max(1, mp.cpu_count() - 1)  # Залишаємо один потік вільним
        self.logger.info(
            f"Налаштування: use_multiprocessing={use_multiprocessing}, chunk_size={chunk_size}, workers={self.num_workers}")

        # Ініціалізація залежних класів
        self.data_cleaner = DataCleaner(logger=self.logger)
        self.data_resampler = DataResampler(logger=self.logger)
        self.data_storage = DataStorageManager(logger=self.logger)
        self.anomaly_detector = AnomalyDetector(logger=self.logger)
        self.db_manager = DatabaseManager()
        self.supported_symbols = self.db_manager.supported_symbols
        self.logger.info(f"Підтримувані символи: {', '.join(self.supported_symbols)}")



        self.ready = True
        self.filtered_data = None
        self.orderbook_statistics = None
        self.logger.info("MarketDataProcessor успішно ініціалізовано")

    def _validate_timeframe(self, timeframe: str) -> bool:
        if timeframe not in self.VALID_TIMEFRAMES:
            self.logger.error(
                f"Невірний таймфрейм: {timeframe}. Допустимі таймфрейми: {', '.join(self.VALID_TIMEFRAMES)}")
            return False
        return True

    def _get_source_timeframe(self, target_timeframe: str) -> Optional[str]:
        if target_timeframe not in self.DERIVED_TIMEFRAMES:
            return None

        # Визначаємо оптимальний вихідний таймфрейм
        if target_timeframe == '4h':
            return '1h'
        elif target_timeframe == '1w':
            return '1d'

        return None

    def _validate_datetime_format(self, date_str: Optional[str]) -> bool:
        if date_str is None:
            return True

        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            try:
                datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                return True
            except ValueError:
                self.logger.error(
                    f"Невірний формат дати: {date_str}. Використовуйте 'YYYY-MM-DD' або 'YYYY-MM-DD HH:MM:SS'")
                return False

    def align_time_series(self, data_list: List[pd.DataFrame],
                          reference_index: int = 0) -> List[pd.DataFrame]:

        if not data_list:
            self.logger.warning("Порожній список DataFrame для вирівнювання")
            return []

        if reference_index < 0 or reference_index >= len(data_list):
            self.logger.error(f"Невірний reference_index: {reference_index}. Має бути від 0 до {len(data_list) - 1}")
            reference_index = 0

        processed_data_list = []

        for i, df in enumerate(data_list):
            if df is None or df.empty:
                self.logger.warning(f"DataFrame {i} є порожнім або None")
                processed_data_list.append(pd.DataFrame())
                continue

            df_copy = df.copy()

            # Перевірка та конвертація до DatetimeIndex
            if not isinstance(df_copy.index, pd.DatetimeIndex):
                self.logger.warning(f"DataFrame {i} не має часового індексу. Спроба конвертувати.")
                try:
                    # Векторизована перевірка часових колонок
                    time_cols = df_copy.columns[
                        df_copy.columns.str.lower().str.contains('|'.join(['time', 'date', 'timestamp']))]

                    if len(time_cols) > 0:
                        df_copy[time_cols[0]] = pd.to_datetime(df_copy[time_cols[0]], errors='coerce')
                        df_copy.set_index(time_cols[0], inplace=True)
                        # Векторизована фільтрація невалідних дат
                        df_copy = df_copy.loc[df_copy.index.notna()]
                    else:
                        self.logger.error(f"Неможливо конвертувати DataFrame {i}: не знайдено часову колонку")
                        processed_data_list.append(pd.DataFrame())
                        continue
                except Exception as e:
                    self.logger.error(f"Помилка при конвертації індексу для DataFrame {i}: {str(e)}")
                    processed_data_list.append(pd.DataFrame())
                    continue

            if not df_copy.index.is_monotonic_increasing:
                df_copy = df_copy.sort_index()

            processed_data_list.append(df_copy)

        reference_df = processed_data_list[reference_index]
        if reference_df is None or reference_df.empty:
            self.logger.error("Еталонний DataFrame є порожнім")
            return processed_data_list

        all_start_times = pd.Series([df.index.min() for df in processed_data_list if not df.empty])
        all_end_times = pd.Series([df.index.max() for df in processed_data_list if not df.empty])

        if all_start_times.empty or all_end_times.empty:
            self.logger.error("Неможливо визначити спільний часовий діапазон")
            return processed_data_list

        common_start = all_start_times.max()
        common_end = all_end_times.min()

        self.logger.info(f"Визначено спільний часовий діапазон: {common_start} - {common_end}")

        if common_start > common_end:
            self.logger.error("Немає спільного часового діапазону між DataFrame")
            return processed_data_list

        try:
            reference_freq = pd.infer_freq(reference_df.index)

            if not reference_freq:
                self.logger.warning("Не вдалося визначити частоту reference DataFrame. Визначення вручну.")
                time_diffs = reference_df.index.to_series().diff().dropna()
                if not time_diffs.empty:
                    median_diff = time_diffs.median()
                    # Конвертація до рядка частоти pandas
                    seconds_mapping = {
                        60: '1min',
                        300: '5min',
                        900: '15min',
                        1800: '30min',
                        3600: '1H',
                        14400: '4H'
                    }

                    if median_diff.days == 1:
                        reference_freq = '1D'
                    else:
                        total_seconds = median_diff.total_seconds()
                        reference_freq = seconds_mapping.get(total_seconds, f"{int(total_seconds)}S")

                    self.logger.info(f"Визначено частоту: {reference_freq}")
                else:
                    self.logger.error("Не вдалося визначити частоту. Повертаємо оригінальні DataFrame")
                    return processed_data_list


            reference_subset = reference_df[(reference_df.index >= common_start) & (reference_df.index <= common_end)]
            common_index = reference_subset.index

            # Якщо частота визначена, перестворимо індекс для забезпечення регулярності
            if reference_freq:
                try:
                    common_index = pd.date_range(start=common_start, end=common_end, freq=reference_freq)
                except pd.errors.OutOfBoundsDatetime:
                    self.logger.warning("Помилка створення date_range. Використання оригінального індексу.")

            # Паралельне вирівнювання всіх DataFrame до спільного індексу
            def _align_one_df(df_info):
                i, df = df_info
                if df.empty:
                    return i, df

                self.logger.debug(f"Вирівнювання DataFrame {i} до спільного індексу")

                # Якщо це еталонний DataFrame
                if i == reference_index:

                    df_aligned = df[(df.index >= common_start) & (df.index <= common_end)]
                    if len(df_aligned.index) != len(common_index):
                        self.logger.debug(f"Перестворення індексу для еталонного DataFrame {i}")
                        df_aligned = df_aligned.reindex(common_index)
                else:
                    df_aligned = df.reindex(common_index)

                # Векторизована інтерполяція числових даних
                numeric_cols = df_aligned.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    df_aligned[numeric_cols] = df_aligned[numeric_cols].interpolate(method='time')
                    df_aligned[numeric_cols] = df_aligned[numeric_cols].fillna(method='ffill').fillna(method='bfill')

                missing_values = df_aligned.isna().sum().sum()
                if missing_values > 0:
                    self.logger.warning(
                        f"Після вирівнювання DataFrame {i} залишилося {missing_values} відсутніх значень")

                return i, df_aligned

            if self.use_multiprocessing and len(processed_data_list) > 1:
                # Паралельне вирівнювання
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    results = list(executor.map(
                        _align_one_df,
                        [(i, df) for i, df in enumerate(processed_data_list)]
                    ))
                # Відновлення початкового порядку
                results.sort(key=lambda x: x[0])
                aligned_data_list = [df for _, df in results]
            else:
                # Послідовне вирівнювання
                aligned_data_list = []
                for i, df in enumerate(processed_data_list):
                    _, aligned_df = _align_one_df((i, df))
                    aligned_data_list.append(aligned_df)

            return aligned_data_list

        except Exception as e:
            self.logger.error(f"Помилка при вирівнюванні часових рядів: {str(e)}")
            self.logger.error(traceback.format_exc())
            return processed_data_list

    # --- Methods delegated to DataCleaner ---

    def remove_duplicate_timestamps(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        return self.data_cleaner.remove_duplicate_timestamps(data)

    def clean_data(self, data: pd.DataFrame, remove_outliers: bool = True,
                   fill_missing: bool = True, **kwargs) -> pd.DataFrame:

        if data.empty:
            return data

        return self.data_cleaner.clean_data(
            data,
            remove_outliers=remove_outliers,
            fill_missing=fill_missing,
            **kwargs
        )

    def handle_missing_values(self, data: pd.DataFrame, method: str = 'interpolate',
                              fetch_missing: bool = False, symbol: Optional[str] = None,
                              timeframe: Optional[str] = None) -> pd.DataFrame:

        return self.data_cleaner.handle_missing_values(
            data,
            method=method,
            fetch_missing=fetch_missing,
            symbol=symbol,
            timeframe=timeframe,
        )

    def normalize_data(self, data: pd.DataFrame, method: str = 'z-score',
                       exclude_columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict]:

        return self.data_cleaner.normalize_data(
            data,
            method=method,
            exclude_columns=exclude_columns,
            **kwargs
        )

    def add_time_features_safely(self, data: pd.DataFrame, tz: str = 'Europe/Kiev') -> pd.DataFrame:

        return self.data_cleaner.add_time_features_safely(data, tz=tz)

    def filter_by_time_range(self, data: pd.DataFrame, start_time: str = None, end_time: str = None) -> pd.DataFrame:

        return self.data_cleaner.filter_by_time_range(data, start_time=start_time, end_time=end_time)

    def validate_data_integrity(self, data: pd.DataFrame) -> dict[str, Any]:

        return self.data_cleaner.validate_data_integrity(data)

    # --- Methods delegated to AnomalyDetector ---
    def detect_outliers(self, data: pd.DataFrame, method: str = 'iqr',
                        threshold: float = 1.5, **kwargs) -> tuple[DataFrame, list]:

        return self.anomaly_detector.detect_outliers(
            data,
            method=method,
            threshold=threshold,
            **kwargs
        )

    # --- Methods delegated to DataResampler ---
    def auto_resample(self, data: pd.DataFrame, target_interval: str = None) -> pd.DataFrame:

        return self.data_resampler.auto_resample(data, target_interval=target_interval)

    def resample_data(self, data: pd.DataFrame, target_interval: str,
                      required_columns: List[str] = None,
                      auto_detect: bool = True,
                      check_interval_compatibility: bool = True) -> pd.DataFrame:
        """
        Resample time series data to a target interval.

        Args:
            data: Input DataFrame with DatetimeIndex
            target_interval: Target interval for resampling ('1m', '4h', '1d', '1w', etc.)
            required_columns: List of columns that must be present in data
            auto_detect: Whether to automatically detect the source interval
            check_interval_compatibility: Whether to check compatibility between source and target intervals

        Returns:
            DataFrame with resampled data
        """
        if data is None or data.empty:
            self.logger.error("Порожній DataFrame для ресемплінгу")
            return pd.DataFrame()

        # Ensure we have a datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            self.logger.warning("Індекс не є DatetimeIndex. Спроба конвертації.")
            try:
                # Look for time-related columns if the index isn't datetime
                time_cols = data.columns[
                    data.columns.str.lower().str.contains('|'.join(['time', 'date', 'timestamp']))]

                if len(time_cols) > 0:
                    data = data.set_index(time_cols[0])
                    data.index = pd.to_datetime(data.index)
                else:
                    self.logger.error("Неможливо конвертувати до DatetimeIndex: не знайдено часову колонку")
                    return pd.DataFrame()
            except Exception as e:
                self.logger.error(f"Помилка при конвертації індексу: {str(e)}")
                return pd.DataFrame()

        # Ensure target_interval is valid
        if not self._validate_timeframe(target_interval):
            self.logger.error(f"Невірний цільовий таймфрейм: {target_interval}")
            return pd.DataFrame()

        # Validate required columns
        if required_columns is None:
            # Default columns for OHLCV data
            required_columns = ['open', 'high', 'low', 'close', 'volume']

        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            self.logger.error(f"Відсутні необхідні колонки: {', '.join(missing_columns)}")
            return pd.DataFrame()

        # Detect source interval if auto_detect is True
        source_interval = None
        if auto_detect:
            # Try to infer frequency from the data
            try:
                inferred_freq = pd.infer_freq(data.index)
                if inferred_freq:
                    self.logger.info(f"Визначено частоту: {inferred_freq}")
                    source_interval = self._pandas_freq_to_timeframe(inferred_freq)
                else:
                    # If can't infer, calculate the median time difference
                    time_diffs = data.index.to_series().diff().dropna()
                    if not time_diffs.empty:
                        median_diff = time_diffs.median()
                        seconds = median_diff.total_seconds()

                        # Map seconds to timeframes
                        if seconds <= 60:
                            source_interval = '1m'
                        elif 3500 <= seconds <= 3700:  # ~1 hour
                            source_interval = '1h'
                        elif 14000 <= seconds <= 15000:  # ~4 hours
                            source_interval = '4h'
                        elif 85000 <= seconds <= 87000:  # ~1 day
                            source_interval = '1d'
                        elif 600000 <= seconds <= 610000:  # ~1 week
                            source_interval = '1w'
                        else:
                            self.logger.warning(f"Невідомий інтервал: {seconds} секунд")

                self.logger.info(f"Визначено вихідний таймфрейм: {source_interval}")
            except Exception as e:
                self.logger.error(f"Помилка при визначенні вихідного таймфрейму: {str(e)}")

        # Check compatibility between source and target intervals
        if check_interval_compatibility and source_interval:
            if not self._is_compatible_timeframe(source_interval, target_interval):
                self.logger.error(
                    f"Несумісні таймфрейми: {source_interval} -> {target_interval}. Рекомендовані: 1m->5m, 1h->4h, 1d->1w")
                return pd.DataFrame()

        # Map timeframe to pandas resampling rule
        resampling_rule = self._timeframe_to_pandas_rule(target_interval)
        if not resampling_rule:
            self.logger.error(f"Неможливо конвертувати таймфрейм {target_interval} у правило ресемплінгу pandas")
            return pd.DataFrame()

        self.logger.info(f"Початок ресемплінгу з правилом: {resampling_rule}")

        try:
            # Make a copy to avoid modifying the original data
            df = data.copy()

            # Ensure the index is sorted
            if not df.index.is_monotonic_increasing:
                df = df.sort_index()

            # Define aggregation functions for OHLCV data
            agg_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }

            # Add aggregation rules for additional columns if they exist
            for col in df.columns:
                if col not in agg_dict:
                    # Try to determine column type and set appropriate aggregation
                    if pd.api.types.is_numeric_dtype(df[col]):
                        if 'volume' in col.lower() or 'amount' in col.lower() or 'qty' in col.lower():
                            agg_dict[col] = 'sum'
                        elif 'price' in col.lower() or 'rate' in col.lower():
                            agg_dict[col] = 'mean'
                        else:
                            # Default for numeric columns
                            agg_dict[col] = 'mean'
                    else:
                        # Default for non-numeric columns
                        agg_dict[col] = 'last'

            # Filter agg_dict to only include columns that are actually in the DataFrame
            agg_dict = {col: agg for col, agg in agg_dict.items() if col in df.columns}

            # Handle specific cases for optimal resampling
            if target_interval == '4h' and source_interval == '1h':
                # For 1h to 4h, we need to ensure proper alignment to 00:00, 04:00, 08:00, 12:00, 16:00, 20:00
                # Calculate the offset to ensure alignment
                first_timestamp = df.index[0]
                hours_offset = first_timestamp.hour % 4
                if hours_offset > 0:
                    offset = f"{4 - hours_offset}H"
                    self.logger.info(f"Застосовуємо зміщення {offset} для правильного вирівнювання 4-годинних свічок")
                    resampler = df.resample(rule=resampling_rule, offset=offset)
                else:
                    resampler = df.resample(rule=resampling_rule)
            elif target_interval == '1w' and source_interval == '1d':
                # For 1d to 1w, align to Monday (start of week)
                resampler = df.resample(rule=resampling_rule, label='left')
            else:
                # Default resampling
                resampler = df.resample(rule=resampling_rule)

            resampled = resampler.agg(agg_dict)

            # Handle missing values that might arise during resampling
            numeric_cols = resampled.select_dtypes(include=[np.number]).columns
            resampled[numeric_cols] = resampled[numeric_cols].interpolate(method='linear', limit=3)

            # Fill any remaining NaN values
            resampled = resampled.fillna(method='ffill')

            # Final check for NaN values
            na_count = resampled.isna().sum().sum()
            if na_count > 0:
                self.logger.warning(f"Після ресемплінгу залишилося {na_count} відсутніх значень")

            self.logger.info(f"Ресемплінг завершено: {len(data)} рядків -> {len(resampled)} рядків")
            return resampled

        except Exception as e:
            self.logger.error(f"Помилка при ресемплінгу: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()

    def _pandas_freq_to_timeframe(self, freq: str) -> Optional[str]:
        """Convert pandas frequency string to timeframe notation"""
        if not freq:
            return None

        freq = freq.upper().lstrip('-')  # <-- ось ключове виправлення

        # Далі йде вже існуюча логіка
        if freq == 'T' or freq == 'MIN':
            return '1m'
        elif freq == 'H':
            return '1h'
        elif freq == '4H':
            return '4h'
        elif freq == 'D':
            return '1d'
        elif freq == 'W':
            return '1w'
        elif freq.endswith('MIN') or freq.endswith('T'):
            try:
                minutes = int(freq.rstrip('MINT'))
                return f"{minutes}m"
            except ValueError:
                return None
        elif freq.endswith('H'):
            try:
                hours = int(freq.rstrip('H'))
                return f"{hours}h"
            except ValueError:
                return None
        elif freq.endswith('D'):
            try:
                days = int(freq.rstrip('D'))
                return f"{days}d"
            except ValueError:
                return None
        elif freq.endswith('W'):
            try:
                weeks = int(freq.rstrip('W'))
                return f"{weeks}w"
            except ValueError:
                return None

        return None

    def _timeframe_to_pandas_rule(self, timeframe: str) -> Optional[str]:
        """Convert timeframe notation to pandas resampling rule"""
        if not timeframe:
            return None

        timeframe = timeframe.lower()

        # Extract number and unit
        if len(timeframe) < 2:
            return None

        try:
            number = int(timeframe[:-1])
            unit = timeframe[-1]

            if unit == 'm':
                return f"{number}min"
            elif unit == 'h':
                return f"{number}H"
            elif unit == 'd':
                return f"{number}D"
            elif unit == 'w':
                return f"{number}W"
            else:
                return None
        except ValueError:
            return None

    def _is_compatible_timeframe(self, source: str, target: str) -> bool:
        """Check if source and target timeframes are compatible for resampling"""
        # Define compatible paths
        compatible_paths = {
            '1m': ['5m', '15m', '30m', '1h', '4h', '1d'],
            '5m': ['15m', '30m', '1h', '4h', '1d'],
            '15m': ['30m', '1h', '4h', '1d'],
            '30m': ['1h', '4h', '1d'],
            '1h': ['4h', '1d', '1w'],
            '4h': ['1d', '1w'],
            '1d': ['1w']
        }

        if source not in compatible_paths:
            return False

        return target in compatible_paths[source]

    def make_stationary(self, data: pd.DataFrame, method: str = 'diff') -> pd.DataFrame:

        return self.data_resampler.make_stationary(data, method=method)

    def prepare_arima_data(self, data: pd.DataFrame, symbol: str, **kwargs) -> pd.DataFrame:

        return self.data_resampler.prepare_arima_data(data, symbol=symbol, **kwargs)

    def prepare_lstm_data(self, data: pd.DataFrame, symbol: str, timeframe: str, **kwargs) -> DataFrame:

        return self.data_resampler.prepare_lstm_data(data, symbol=symbol, timeframe=timeframe, **kwargs)


    def load_data(self, data_source: str, symbol: str, timeframe: str, data_type: str = 'candles',
                  **kwargs) -> pd.DataFrame:

        return self.data_storage.load_data(
            data_source=data_source,
            symbol=symbol,
            timeframe=timeframe,
            data_type=data_type,
            **kwargs
        )


    def save_lstm_sequence(self, symbol: str, data_points: List[Dict[str, Any]], **kwargs) -> List[int]:
        if symbol == 'BTC':
            return self.data_storage.save_btc_lstm_sequence(data_points)
        elif symbol == 'ETH':
            return self.data_storage.save_eth_lstm_sequence(data_points)
        elif symbol == 'SOL':
            return self.data_storage.save_sol_lstm_sequence(data_points)
        else:
            self.logger.error(f"Непідтримуваний символ: {symbol}")
            return []

    def save_arima_data(self, symbol: str, data_points: List[Dict[str, Any]], **kwargs) -> List[int]:
        if symbol == 'BTC':
            return self.data_storage.save_btc_arima_data(data_points)
        elif symbol == 'ETH':
            return self.data_storage.save_eth_arima_data(data_points)
        elif symbol == 'SOL':
            return self.data_storage.save_sol_arima_data(data_points)
        else:
            self.logger.error(f"Непідтримуваний символ: {symbol}")
            return []

    def load_lstm_sequence(self, symbol: str, timeframe: str, sequence_id: Optional[int] = None, **kwargs) -> List[
        Dict[str, Any]]:

        if symbol == 'BTC':
            return self.data_storage.get_btc_lstm_sequence(timeframe, sequence_id)
        elif symbol == 'ETH':
            return self.data_storage.get_eth_lstm_sequence(timeframe, sequence_id)
        elif symbol == 'SOL':
            return self.data_storage.get_sol_lstm_sequence(timeframe, sequence_id)
        else:
            self.logger.error(f"Непідтримуваний символ: {symbol}")
            return []

    def load_arima_data(self, symbol: str, timeframe: str, data_id: Optional[int] = None, **kwargs) -> List[
        Dict[str, Any]]:

        if symbol == 'BTC':
            return self.data_storage.get_btc_arima_data(timeframe, data_id)
        elif symbol == 'ETH':
            return self.data_storage.get_eth_arima_data(timeframe, data_id)
        elif symbol == 'SOL':
            return self.data_storage.get_sol_arima_data(timeframe, data_id)
        else:
            self.logger.error(f"Непідтримуваний символ: {symbol}")
            return []

    def preprocess_pipeline(self, data: pd.DataFrame,
                            steps: Optional[List[Dict]] = None,
                            symbol: Optional[str] = None,
                            interval: Optional[str] = None) -> pd.DataFrame:

        if data is None or data.empty:
            self.logger.warning("Отримано порожній DataFrame для обробки в конвеєрі")
            return pd.DataFrame()

        # Перевірка наявності основних колонок для свічок
        expected_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in expected_columns if col not in data.columns]
        if missing_columns:
            self.logger.warning(f"Відсутні необхідні колонки: {', '.join(missing_columns)}")

        # Перевірка таймфрейму, якщо вказано
        if interval and not self._validate_timeframe(interval):
            self.logger.warning(f"Невірний таймфрейм: {interval}. Обробка продовжується з застереженням.")

        # Встановлення типових кроків обробки, якщо не вказано
        if steps is None:
            steps = [
                {'name': 'remove_duplicate_timestamps', 'params': {}},
                {'name': 'clean_data', 'params': {'remove_outliers': True, 'fill_missing': True}},
                {'name': 'handle_missing_values', 'params': {
                    'method': 'interpolate',
                    'fetch_missing': True
                }}
            ]

        self.logger.info(f"Початок виконання конвеєра обробки даних для {'символу ' + symbol if symbol else 'даних'} "
                         f"({'таймфрейм ' + interval if interval else 'без вказаного таймфрейму'}) "
                         f"з {len(steps)} кроками")

        # Зберігаємо початкові розміри даних для порівняння
        initial_rows = len(data)
        initial_cols = len(data.columns)
        self.logger.info(f"Початкові дані: {initial_rows} рядків, {initial_cols} колонок")

        # Перевірка індексу
        if not isinstance(data.index, pd.DatetimeIndex):
            self.logger.warning("Індекс не є DatetimeIndex. Можливі проблеми з часовими операціями.")

        result = data.copy()

        # Виконання кожного кроку конвеєра
        for step_idx, step in enumerate(steps, 1):
            step_name = step.get('name')
            step_params = step.get('params', {})

            start_time = time()  # Додаємо вимірювання часу виконання

            if not hasattr(self, step_name):
                self.logger.warning(f"Крок {step_idx}: Метод '{step_name}' не існує. Пропускаємо.")
                continue

            try:
                self.logger.info(f"Крок {step_idx}: Виконання '{step_name}' з параметрами {step_params}")
                method = getattr(self, step_name)

                # Зберігаємо розмір даних перед виконанням кроку
                before_rows = len(result)
                before_cols = len(result.columns)

                # Додаємо symbol та interval якщо метод підтримує їх
                if step_name == 'handle_missing_values':
                    step_params['symbol'] = symbol
                    step_params['timeframe'] = interval

                # Для методів, які повертають кортеж (результат, додаткова інформація)
                if step_name in ['normalize_data', 'detect_outliers', 'detect_zscore_outliers',
                                 'detect_iqr_outliers', 'detect_isolation_forest_outliers',
                                 'validate_data_integrity', 'detect_outliers_essemble']:
                    result, additional_info = method(result, **step_params)
                    # Логування додаткової інформації, якщо вона є
                    if additional_info and isinstance(additional_info, dict):
                        self.logger.debug(f"Додаткова інформація з кроку '{step_name}': ")
                else:
                    result = method(result, **step_params)

                # Перевірка результату
                if result is None or result.empty:
                    self.logger.error(f"Крок {step_idx}: '{step_name}' повернув порожні дані. Зупинка конвеєра.")
                    return pd.DataFrame()

                # Аналіз змін після кроку
                after_rows = len(result)
                after_cols = len(result.columns)
                rows_diff = after_rows - before_rows
                cols_diff = after_cols - before_cols


                # Логування з інформацією про зміни
                self.logger.info(
                    f"Рядків: {before_rows} → {after_rows} ({rows_diff:+d}), "
                    f"Колонок: {before_cols} → {after_cols} ({cols_diff:+d})"
                )

                # Додаткова перевірка на значні зміни в даних
                if abs(rows_diff) > before_rows * 0.3:  # Якщо зміна більше 30%
                    self.logger.warning(
                        f"Крок {step_idx}: '{step_name}' призвів до значної зміни кількості рядків: {rows_diff:+d} ({rows_diff / before_rows * 100:.1f}%)"
                    )

            except Exception as e:
                self.logger.error(f"Помилка на кроці {step_idx}: '{step_name}': {str(e)}")
                self.logger.error(traceback.format_exc())
                # Продовжуємо виконання конвеєра, незважаючи на помилку в одному кроці

        # Перевірка цілісності індексу після всіх перетворень
        if not isinstance(result.index, pd.DatetimeIndex):
            self.logger.warning("Після обробки індекс не є DatetimeIndex. Спроба конвертації...")
            try:
                result.index = pd.to_datetime(result.index)
            except Exception as e:
                self.logger.error(f"Не вдалося конвертувати індекс: {str(e)}")

        # Перевірка сортування індексу
        if not result.index.is_monotonic_increasing:
            self.logger.warning("Індекс не відсортований. Сортуємо за часом...")
            result = result.sort_index()

        # Перевірка на дублікати індексу
        duplicates = result.index.duplicated().sum()
        if duplicates > 0:
            self.logger.warning(
                f"Знайдено {duplicates} дублікатів у індексі. Рекомендується викликати remove_duplicate_timestamps.")

        # Перевірка на пропуски
        na_count = result.isna().sum().sum()
        if na_count > 0:
            na_percent = na_count / (result.shape[0] * result.shape[1]) * 100
            self.logger.warning(f"В даних залишилось {na_count} пропущених значень ({na_percent:.2f}%).")

        # Підсумок
        final_rows = len(result)
        final_cols = len(result.columns)
        rows_diff = final_rows - initial_rows
        cols_diff = final_cols - initial_cols

        self.logger.info(
            f"Конвеєр обробки даних завершено. "
            f"Початково: {initial_rows} рядків, {initial_cols} колонок. "
            f"Результат: {final_rows} рядків, {final_cols} колонок. "
            f"Зміна: {rows_diff:+d} рядків, {cols_diff:+d} колонок."
        )

        return result

    def process_market_data(self, symbol: str, timeframe: str, start_date: Optional[str] = None,
                            end_date: Optional[str] = None, save_results: bool = True,
                            auto_detect: bool = True,
                            check_interval_compatibility: bool = True) -> DataFrame | dict[Any, Any]:

        self.logger.info(f"Початок комплексної обробки даних для {symbol} ({timeframe})")
        results = {}

        # Перевірка символу
        if symbol not in self.supported_symbols:
            self.logger.error(f"Символ {symbol} не підтримується. Підтримуються: {', '.join(self.supported_symbols)}")
            return results

        # Перевірка таймфрейму
        if not self._validate_timeframe(timeframe):
            self.logger.error(
                f"Таймфрейм {timeframe} не підтримується. Підтримуються: {', '.join(self.VALID_TIMEFRAMES)}")
            return results

        # Перевірка формату дат
        if not self._validate_datetime_format(start_date) or not self._validate_datetime_format(end_date):
            self.logger.error(f"Невірний формат дати. Використовуйте 'YYYY-MM-DD' або 'YYYY-MM-DD HH:MM:SS'")
            return results

        # Визначення базових та похідних таймфреймів
        base_timeframes = ['1m', '1h', '1d']
        target_interval = ['4h', '1w']

        # 1. Завантаження даних

        if timeframe in base_timeframes:
            # Для базових таймфреймів завантажуємо дані з БД
            self.logger.info(f"Завантаження базових даних для {symbol} ({timeframe})")
            raw_data = self.load_data(
                data_source='database',
                symbol=symbol,
                timeframe=timeframe,
                data_type='candles'
            )

            self.logger.info(f"Завантаження даних виконано ")

            if raw_data is None or raw_data.empty:
                self.logger.warning(f"Дані не знайдено для {symbol} {timeframe}")
                return results

            self.logger.info(f"Завантажено {len(raw_data)} рядків для {symbol} ({timeframe})")

        elif timeframe in target_interval:
            # Для похідних таймфреймів визначаємо вихідний таймфрейм для ресемплінгу
            source_timeframe = self._get_source_timeframe(timeframe)
            if not source_timeframe:
                self.logger.error(f"Не вдалося визначити вихідний таймфрейм для {timeframe}")
                return results

            self.logger.info(f"Створення {timeframe} даних через ресемплінг з {source_timeframe}")

            # Завантажуємо дані вихідного таймфрейму
            source_data = self.load_data(
                data_source='database',
                symbol=symbol,
                timeframe=source_timeframe,
                data_type='candles'
            )

            self.logger.info(f"Завантаження даних виконано")

            if source_data is None or source_data.empty:
                self.logger.warning(f"Базові дані не знайдено для {symbol} {source_timeframe}")
                return results

            self.logger.info(f"Завантажено {len(source_data)} рядків базових даних для ресемплінгу")

            # Виконуємо ресемплінг до цільового таймфрейму
            raw_data = self.resample_data(source_data, target_interval=timeframe,auto_detect=auto_detect,
            check_interval_compatibility=check_interval_compatibility)

            self.logger.info(f"Ресемплінг до {timeframe} виконано ")

            if raw_data is None or raw_data.empty:
                self.logger.warning(f"Не вдалося створити дані для {symbol} {timeframe} через ресемплінг")
                return results

            self.logger.info(f"Після ресемплінгу отримано {len(raw_data)} рядків для {symbol} ({timeframe})")
        else:
            self.logger.error(f"Непідтримуваний таймфрейм: {timeframe}")
            return results

        # Зберігаємо сирі дані в результаті
        results['raw_data'] = raw_data

        # Перевірка формату сирих даних

        # 2. Фільтрація за часовим діапазоном
        if start_date or end_date:
            self.logger.info(f"Фільтрація даних за періодом: {start_date or 'початок'} - {end_date or 'кінець'}")
            before_filter_rows = len(raw_data)
            raw_data = self.filter_by_time_range(raw_data, start_time=start_date, end_time=end_date)
            after_filter_rows = len(raw_data)

            self.logger.info(
                f"Відфільтровано {before_filter_rows - after_filter_rows} рядків. Залишилось {after_filter_rows} рядків.")

            if raw_data is None or raw_data.empty:
                self.logger.warning(f"Після фільтрації за часом дані відсутні")
                return results

        # 3. Обробка відсутніх значень
        self.logger.info(f"Обробка відсутніх значень")
        filled_data = self.handle_missing_values(
            raw_data,
            symbol=symbol,
            timeframe=timeframe,
            fetch_missing=True
        )

        self.logger.info(f"Обробка відсутніх значень виконана ")

        if filled_data is None or filled_data.empty:
            self.logger.warning(f"Після обробки відсутніх значень дані порожні")
            return results

        # 4. Повна обробка через конвеєр
        self.logger.info(f"Запуск повного конвеєра обробки")
        processed_data = self.preprocess_pipeline(filled_data, symbol=symbol, interval=timeframe)

        self.logger.info(f"Конвеєр обробки виконано ")

        if processed_data is None or processed_data.empty:
            self.logger.warning(f"Після обробки даних результат порожній")
            return results

        results['processed_data'] = processed_data

        # 5. Додавання часових ознак
        self.logger.info(f"Додавання часових ознак")
        processed_data = self.add_time_features_safely(processed_data, tz='Europe/Kiev')

        self.logger.info(f"Додавання часових ознак виконано ")

        # 6. Виявлення та обробка аномалій
        self.logger.info(f"Виявлення аномалій")
        processed_data, outliers_info = self.detect_outliers(processed_data)

        self.logger.info(f"Виявлення аномалій виконано ")

        # 7. Підготовка даних для моделей ARIMA і LSTM
        model_data_timeframes = ['1m', '4h', '1d', '1w']
        if timeframe in model_data_timeframes:
            # ARIMA
            self.logger.info(f"Підготовка даних для ARIMA моделі")
            arima_data = self.prepare_arima_data(processed_data, symbol=symbol, timeframe=timeframe)


            if arima_data is not None and not arima_data.empty:
                results['arima_data'] = arima_data
                self.logger.info(f"Підготовлено {len(arima_data)} записів ARIMA даних")

                if save_results:
                    try:
                        self.logger.info(f"Збереження ARIMA даних")
                        arima_data_points = arima_data.reset_index().to_dict('records')

                        for record in arima_data_points:
                            record['open_time'] = record.get('open_time', record.get('timestamp', record.get('index',
                                                                                                             pd.Timestamp.now())))
                            record['original_close'] = record.get('original_close', record.get('close', None))

                        # Виклик уніфікованого методу
                        arima_ids = self.save_arima_data(symbol, arima_data_points)

                        if arima_ids:
                            self.logger.info(f"ARIMA дані збережено")
                        else:
                            self.logger.warning(f"Не вдалося зберегти ARIMA дані для {symbol}")
                    except Exception as e:
                        self.logger.error(f"Помилка збереження ARIMA даних для {symbol}: {str(e)}")
                        self.logger.error(traceback.format_exc())
            else:
                self.logger.warning(f"Не вдалося підготувати ARIMA дані")

            # LSTM
            try:
                self.logger.info(f"Підготовка даних для LSTM моделі")
                lstm_df = self.prepare_lstm_data(processed_data, symbol=symbol, timeframe=timeframe)


                if lstm_df is not None and not lstm_df.empty:
                    results['lstm_data'] = lstm_df
                    self.logger.info(f"Підготовлено {len(lstm_df)} записів LSTM даних")

                    if save_results:
                        try:
                            self.logger.info(f"Збереження LSTM даних")
                            lstm_data_points = lstm_df.reset_index().to_dict('records')

                            timestamp_str = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
                            for i, record in enumerate(lstm_data_points):
                                record.setdefault('sequence_position', i + 1)
                                record.setdefault('sequence_id', f"{symbol}_{timeframe}_{timestamp_str}_{i}")

                            required_fields = ['sequence_position', 'sequence_id']
                            for record in lstm_data_points:
                                missing = [f for f in required_fields if f not in record]
                                if missing:
                                    raise ValueError(f"Відсутні обов'язкові поля: {missing}")

                            # Виклик уніфікованого методу
                            sequence_ids = self.save_lstm_sequence(symbol, lstm_data_points)

                            if sequence_ids:
                                self.logger.info(
                                    f"LSTM послідовності збережено ")
                            else:
                                self.logger.warning(f"Не вдалося зберегти LSTM послідовності для {symbol}")
                        except Exception as e:
                            self.logger.error(f"Помилка збереження LSTM послідовностей для {symbol}: {str(e)}")
                            self.logger.error(traceback.format_exc())
                else:
                    self.logger.warning(f"Не вдалося підготувати LSTM дані")

            except Exception as e:
                self.logger.error(f"Помилка підготовки LSTM даних: {str(e)}")
                self.logger.error(traceback.format_exc())

        self.logger.info(
            f"Комплексна обробка даних для {symbol} ({timeframe}) завершена ")
        return results

    def validate_market_data(self, data: pd.DataFrame) -> Tuple[bool, Dict]:

        if data.empty:
            return False, {"error": "Порожній DataFrame"}

        results = {}

        # Перевірка часового індексу
        if not isinstance(data.index, pd.DatetimeIndex):
            return False, {"error": "DataFrame не має DatetimeIndex"}

        # Базові перевірки якості
        results["data_shape"] = {"rows": len(data), "columns": len(data.columns)}
        results["duplicated_rows"] = int(data.duplicated().sum())
        results["null_values"] = int(data.isna().sum().sum())

        # Перевірка OHLCV даних
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        results["missing_columns"] = missing_columns

        if missing_columns:
            results["validation_passed"] = False
            return False, results

        # Перевірка правильності цін
        price_issues = []

        # High не може бути нижче Low
        high_lt_low = (data['high'] < data['low']).sum()
        if high_lt_low > 0:
            price_issues.append(f"High < Low знайдено в {high_lt_low} рядках")

        # Close має бути між High і Low
        close_issues = ((data['close'] > data['high']) | (data['close'] < data['low'])).sum()
        if close_issues > 0:
            price_issues.append(f"Close не між High і Low в {close_issues} рядках")

        # Open має бути між High і Low
        open_issues = ((data['open'] > data['high']) | (data['open'] < data['low'])).sum()
        if open_issues > 0:
            price_issues.append(f"Open не між High і Low в {open_issues} рядках")

        results["price_issues"] = price_issues

        # Перевірка на нульові або від'ємні ціни/об'єми
        zero_prices = ((data[['open', 'high', 'low', 'close']] <= 0).sum()).sum()
        negative_volume = (data['volume'] < 0).sum()

        results["zero_prices"] = int(zero_prices)
        results["negative_volume"] = int(negative_volume)

        # Перевірка часових проміжків
        time_diffs = data.index.to_series().diff().dropna()
        if not time_diffs.empty:
            regular_diff = time_diffs.value_counts().index[0]
            irregular_count = len(time_diffs[time_diffs != regular_diff])
            results["irregular_intervals"] = int(irregular_count)
            results["missing_intervals"] = int(irregular_count > 0)

        # Загальний результат валідації
        validation_failed = (
                len(missing_columns) > 0 or
                len(price_issues) > 0 or
                zero_prices > 0 or
                negative_volume > 0
        )

        results["validation_passed"] = not validation_failed

        return not validation_failed, results

    def combine_market_datasets(self, datasets: Dict[str, pd.DataFrame],
                                reference_key: str = None) -> pd.DataFrame:

        if not datasets:
            self.logger.warning("Порожній словник наборів даних для об'єднання")
            return pd.DataFrame()

        # Перетворення словника на список для подальшої обробки
        data_list = list(datasets.values())
        keys_list = list(datasets.keys())

        # Визначення індексу еталонного набору
        reference_index = 0
        if reference_key and reference_key in datasets:
            reference_index = keys_list.index(reference_key)

        # Вирівнювання часових рядів
        aligned_data = self.align_time_series(data_list, reference_index=reference_index)

        # Перевірка результатів вирівнювання
        if not aligned_data or all(df.empty for df in aligned_data):
            self.logger.error("Не вдалося вирівняти часові ряди")
            return pd.DataFrame()

        # Підготовка до об'єднання - додавання префіксів до колонок
        renamed_dfs = []

        for i, (key, df) in enumerate(zip(keys_list, aligned_data)):
            if df.empty:
                continue

            df_copy = df.copy()

            # Не додаємо префікс до індексу
            if df_copy.index.name:
                index_name = df_copy.index.name
            else:
                index_name = 'timestamp'
                df_copy.index.name = index_name

            # Перейменовуємо колонки з префіксом
            rename_dict = {col: f"{key}_{col}" for col in df_copy.columns}
            df_copy = df_copy.rename(columns=rename_dict)

            renamed_dfs.append(df_copy)

        if not renamed_dfs:
            self.logger.error("Після вирівнювання та перейменування немає даних для об'єднання")
            return pd.DataFrame()

        # Об'єднання даних
        result = renamed_dfs[0]

        for df in renamed_dfs[1:]:
            if df.empty:
                continue
            result = result.join(df, how='outer')

        self.logger.info(f"Об'єднання завершено. Результат: {len(result)} рядків, {len(result.columns)} колонок")

        return result


def main():
    EU_TIMEZONE = 'Europe/Kiev'
    SYMBOLS = ['ETH']

    # Всі таймфрейми
    ALL_TIMEFRAMES = ['1m','1h', '4h', '1d', '1w']

    # Базові таймфрейми, які вже існують в базі даних
    BASE_TIMEFRAMES = ['1m','1h', '1d']

    # Похідні таймфрейми, які будуть створені через ресемплінг
    DERIVED_TIMEFRAMES = ['4h', '1w']

    # Ініціалізація процесора
    processor = MarketDataProcessor(log_level=logging.INFO)

    # Словник для зберігання результатів обробки
    processed_results = {}

    # Спочатку обробляємо базові таймфрейми
    print("\n=== Обробка базових таймфреймів ===")
    for symbol in SYMBOLS:
        for timeframe in BASE_TIMEFRAMES:
            print(f"\nОбробка {symbol} ({timeframe})...")

            try:
                results = processor.process_market_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    save_results=True,
                )

                if not results:
                    print(f"Не вдалося обробити дані для {symbol} {timeframe}")
                    continue

                # Зберігаємо результати для можливого використання при похідних таймфреймах
                processed_results[f"{symbol}_{timeframe}"] = results

                # Виводимо підсумок результатів
                for key, data in results.items():
                    if isinstance(data, pd.DataFrame) and not data.empty:
                        print(f" - {key}: {len(data)} рядків, {len(data.columns)} колонок")

                print(f"Обробка {symbol} ({timeframe}) завершена успішно")

            except Exception as e:
                print(f"Помилка при обробці {symbol} ({timeframe}): {str(e)}")
                traceback.print_exc()

    # Після обробки базових таймфреймів обробляємо похідні
    print("\n=== Обробка похідних таймфреймів ===")
    for symbol in SYMBOLS:
        for timeframe in DERIVED_TIMEFRAMES:
            print(f"\nОбробка {symbol} ({timeframe})...")

            source_timeframe = None
            if timeframe == '4h':
                source_timeframe = '1h'
            elif timeframe == '1w':
                source_timeframe = '1d'

            print(f"Буде використано ресемплінг із {source_timeframe} до {timeframe}")

            try:
                results = processor.process_market_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    save_results=True,
                    auto_detect=True,
                    check_interval_compatibility=True
                )

                if not results:
                    print(f"Не вдалося обробити дані для {symbol} {timeframe}")
                    continue

                # Зберігаємо результати
                processed_results[f"{symbol}_{timeframe}"] = results

                # Виводимо підсумок результатів
                for key, data in results.items():
                    if isinstance(data, pd.DataFrame) and not data.empty:
                        print(f" - {key}: {len(data)} рядків, {len(data.columns)} колонок")

                print(f"Обробка {symbol} ({timeframe}) завершена успішно")

            except Exception as e:
                print(f"Помилка при обробці {symbol} ({timeframe}): {str(e)}")
                traceback.print_exc()

    print("\nВсі операції обробки даних завершено")


if __name__ == "__main__":
    main()