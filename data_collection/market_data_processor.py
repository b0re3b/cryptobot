import os
import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Tuple, Union, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import hashlib
import json
from functools import lru_cache
import pytz
from utils.config import db_connection
from data.db import DatabaseManager


class MarketDataProcessor:

    def __init__(self,  log_level=logging.INFO):
        self.log_level = log_level
        self.db_connection = db_connection
        self.db_manager = DatabaseManager()
        self.supported_symbols = self.db_manager.supported_symbols
        logging.basicConfig(level=self.log_level)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Ініціалізація класу...")
        self.ready = True

    def save_klines_to_db(self, df: pd.DataFrame, symbol: str, interval: str):
        if df.empty:
            self.logger.warning("Спроба зберегти порожні свічки")
            return

        def convert_numpy_types(obj):
            import numpy as np
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (np.generic, np.bool_)):
                return obj.item()
            else:
                return obj

        for _, row in df.iterrows():
            try:
                kline_data = {
                    'interval': interval,
                    'open_time': row.name,
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row.get('volume', 0),
                    'close_time': row.get('close_time', row.name),
                    'quote_asset_volume': row.get('quote_asset_volume', 0),
                    'number_of_trades': row.get('number_of_trades', 0),
                    'taker_buy_base_volume': row.get('taker_buy_base_volume', 0),
                    'taker_buy_quote_volume': row.get('taker_buy_quote_volume', 0),
                    'is_closed': bool(row.get('is_closed', True)),
                }

                # Конвертація до базових типів Python
                kline_data = convert_numpy_types(kline_data)

                self.db_manager.insert_kline(symbol, kline_data)

            except Exception as e:
                self.logger.error(f"Помилка при збереженні свічки для {symbol}: {e}")

    def save_processed_klines_to_db(self, df: pd.DataFrame, symbol: str, interval: str):
        if df.empty:
            self.logger.warning("Спроба зберегти порожні оброблені свічки")
            return

        for _, row in df.iterrows():
            try:
                processed_data = {
                    'interval': interval,
                    'open_time': row.name,
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume'],
                    'price_zscore': row.get('price_zscore'),
                    'volume_zscore': row.get('volume_zscore'),
                    'volatility': row.get('volatility'),
                    'trend': row.get('trend'),
                    'hour': row.get('hour'),
                    'day_of_week': row.get('weekday'),
                    'is_weekend': bool(row.get('is_weekend')),  # очікується bool
                    'session': row.get('session', 'unknown'),
                    'is_anomaly': row.get('is_anomaly', False),
                    'has_missing': row.get('has_missing', False)
                }
                self.db_manager.insert_kline_processed(symbol, processed_data)
            except Exception as e:
                self.logger.error(f"Помилка при збереженні обробленої свічки: {e}")

    def save_volume_profile_to_db(self, df: pd.DataFrame, symbol: str, interval: str):
        if df.empty:
            self.logger.warning("Спроба зберегти порожній профіль об'єму")
            return

        for _, row in df.iterrows():
            try:
                profile_data = {
                    'interval': interval,
                    'time_bucket': row.get('period') or row.name,
                    'price_bin_start': row.get('bin_lower'),
                    'price_bin_end': row.get('bin_upper'),
                    'volume': row['volume']
                }
                self.db_manager.insert_volume_profile(symbol, profile_data)
            except Exception as e:
                self.logger.error(f"Помилка при збереженні профілю об'єму: {e}")

    def _load_from_database(self, symbol: str, interval: str,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            data_type: str = 'candles') -> pd.DataFrame:

        self.logger.info(f"Завантаження {data_type} даних з бази даних для {symbol} {interval}")

        try:
            data = None
            if data_type == 'candles':
                data = self.db_manager.get_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=start_date,
                    end_time=end_date
                )
            elif data_type == 'orderbook':
                data = self.db_manager.get_orderbook(
                    symbol=symbol,
                    start_time=start_date,
                    end_time=end_date
                )
            else:
                raise ValueError(f"Непідтримуваний тип даних: {data_type}")

            if data is None:
                self.logger.warning("База даних повернула None")
                return pd.DataFrame()

            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)

            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data.set_index('timestamp', inplace=True)

            return data

        except Exception as e:
            self.logger.error(f"Помилка при завантаженні з бази даних: {str(e)}")
            raise

    def load_data(self, data_source: str, symbol: str, interval: str,
                  start_date: Optional[Union[str, datetime]] = None,
                  end_date: Optional[Union[str, datetime]] = None,
                  file_path: Optional[str] = None,
                  data_type: str = 'candles') -> pd.DataFrame:

        start_date_dt = pd.to_datetime(start_date) if start_date else None
        end_date_dt = pd.to_datetime(end_date) if end_date else None

        self.logger.info(f"Завантаження даних з {data_source}: {symbol}, {interval}, {data_type}")

        try:
            if data_source == 'database':
                data = self._load_from_database(
                    symbol,
                    interval,
                    start_date_dt,
                    end_date_dt,
                    data_type
                )
            elif data_source == 'csv':
                if not file_path:
                    raise ValueError("Для джерела 'csv' необхідно вказати шлях до файлу (file_path)")

                self.logger.info(f"Завантаження даних з CSV файлу: {file_path}")
                data = pd.read_csv(file_path)

                if 'timestamp' in data.columns or 'date' in data.columns or 'time' in data.columns:
                    time_col = next((col for col in ['timestamp', 'date', 'time'] if col in data.columns), None)
                    if time_col:
                        data[time_col] = pd.to_datetime(data[time_col])
                        data.set_index(time_col, inplace=True)
                    else:
                        self.logger.warning("Не знайдено часову колонку в CSV файлі")

                if start_date_dt and isinstance(data.index, pd.DatetimeIndex):
                    data = data[data.index >= start_date_dt]
                if end_date_dt and isinstance(data.index, pd.DatetimeIndex):
                    data = data[data.index <= end_date_dt]

            else:
                raise ValueError(f"Непідтримуване джерело даних: {data_source}")

            if data is None or data.empty:
                self.logger.warning(f"Отримано порожній набір даних від {data_source}")
                return pd.DataFrame()

            return data

        except Exception as e:
            self.logger.error(f"Помилка при завантаженні даних: {str(e)}")
            raise

    def clean_data(self, data: pd.DataFrame, remove_outliers: bool = True,
                   fill_missing: bool = True) -> pd.DataFrame:

        if data.empty:
            self.logger.warning("Отримано порожній DataFrame для очищення")
            return data

        self.logger.info(f"Початок очищення даних: {data.shape[0]} рядків, {data.shape[1]} стовпців")
        result = data.copy()

        if not isinstance(result.index, pd.DatetimeIndex):
            try:
                time_cols = [col for col in result.columns if
                             any(x in col.lower() for x in ['time', 'date', 'timestamp'])]
                if time_cols:
                    time_col = time_cols[0]
                    self.logger.info(f"Конвертування колонки {time_col} в індекс часу")
                    result[time_col] = pd.to_datetime(result[time_col])
                    result.set_index(time_col, inplace=True)
                else:
                    self.logger.warning("Не знайдено колонку з часом, індекс залишається незмінним")
            except Exception as e:
                self.logger.error(f"Помилка при конвертуванні індексу: {str(e)}")

        if result.index.duplicated().any():
            dup_count = result.index.duplicated().sum()
            self.logger.info(f"Знайдено {dup_count} дублікатів індексу, видалення...")
            result = result[~result.index.duplicated(keep='first')]

        result = result.sort_index()

        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors='coerce')

        if remove_outliers:
            self.logger.info("Видалення аномальних значень...")
            price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in result.columns]
            for col in price_cols:
                # Перевірка на порожній DataFrame або серію
                if result[col].empty or result[col].isna().all():
                    continue

                Q1 = result[col].quantile(0.25)
                Q3 = result[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR

                outliers = (result[col] < lower_bound) | (result[col] > upper_bound)
                if outliers.any():
                    outlier_count = outliers.sum()
                    self.logger.info(f"Знайдено {outlier_count} аномалій в колонці {col}")
                    result.loc[outliers, col] = np.nan

        if fill_missing and result.isna().any().any():
            self.logger.info("Заповнення відсутніх значень...")
            price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in result.columns]
            if price_cols and isinstance(result.index, pd.DatetimeIndex):
                result[price_cols] = result[price_cols].interpolate(method='time')

            if 'volume' in result.columns and result['volume'].isna().any():
                result['volume'] = result['volume'].fillna(0)

            numeric_cols = result.select_dtypes(include=[np.number]).columns
            other_numeric = [col for col in numeric_cols if col not in price_cols + ['volume']]
            if other_numeric:
                if isinstance(result.index, pd.DatetimeIndex):
                    result[other_numeric] = result[other_numeric].interpolate(method='time')
                else:
                    result[other_numeric] = result[other_numeric].interpolate(method='linear')

            result = result.fillna(method='ffill').fillna(method='bfill')

        price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in result.columns]
        if len(price_cols) == 4:
            invalid_hl = result['high'] < result['low']
            if invalid_hl.any():
                invalid_count = invalid_hl.sum()
                self.logger.warning(f"Знайдено {invalid_count} рядків, де high < low")

                temp = result.loc[invalid_hl, 'high'].copy()
                result.loc[invalid_hl, 'high'] = result.loc[invalid_hl, 'low']
                result.loc[invalid_hl, 'low'] = temp

        self.logger.info(f"Очищення даних завершено: {result.shape[0]} рядків, {result.shape[1]} стовпців")
        return result

    def resample_data(self, data: pd.DataFrame, target_interval: str) -> pd.DataFrame:

        if data.empty:
            self.logger.warning("Отримано порожній DataFrame для ресемплінгу")
            return data

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Дані повинні мати DatetimeIndex для ресемплінгу")

        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            self.logger.warning(f"Відсутні необхідні колонки: {missing_cols}")
            return data

        pandas_interval = self._convert_interval_to_pandas_format(target_interval)
        self.logger.info(f"Ресемплінг даних до інтервалу: {target_interval} (pandas формат: {pandas_interval})")

        if len(data) > 1:
            current_interval = pd.Timedelta(data.index[1] - data.index[0])
            estimated_target_interval = self._parse_interval(target_interval)

            if estimated_target_interval < current_interval:
                self.logger.warning(f"Цільовий інтервал ({target_interval}) менший за поточний інтервал даних. "
                                    f"Даунсемплінг неможливий без додаткових даних.")
                return data

        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }

        if 'volume' in data.columns:
            agg_dict['volume'] = 'sum'

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in agg_dict:
                if any(x in col.lower() for x in ['count', 'number', 'trades']):
                    agg_dict[col] = 'sum'
                else:
                    agg_dict[col] = 'mean'

        try:
            resampled = data.resample(pandas_interval).agg(agg_dict)

            if resampled.isna().any().any():
                self.logger.info("Заповнення відсутніх значень після ресемплінгу...")
                price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in resampled.columns]
                resampled[price_cols] = resampled[price_cols].fillna(method='ffill')

                if 'volume' in resampled.columns:
                    resampled['volume'] = resampled['volume'].fillna(0)

            self.logger.info(f"Ресемплінг успішно завершено: {resampled.shape[0]} рядків")
            return resampled

        except Exception as e:
            self.logger.error(f"Помилка при ресемплінгу даних: {str(e)}")
            raise

    def _convert_interval_to_pandas_format(self, interval: str) -> str:

        interval_map = {
            's': 'S',
            'm': 'T',
            'h': 'H',
            'd': 'D',
            'w': 'W',
            'M': 'M',
        }

        if not interval or not isinstance(interval, str):
            raise ValueError(f"Неправильний формат інтервалу: {interval}")

        import re
        match = re.match(r'(\d+)([smhdwM])', interval)
        if not match:
            raise ValueError(f"Неправильний формат інтервалу: {interval}")

        number, unit = match.groups()

        if unit in interval_map:
            return f"{number}{interval_map[unit]}"
        else:
            raise ValueError(f"Непідтримувана одиниця часу: {unit}")

    def _parse_interval(self, interval: str) -> pd.Timedelta:

        interval_map = {
            's': 'seconds',
            'm': 'minutes',
            'h': 'hours',
            'd': 'days',
            'w': 'weeks',
        }

        import re
        match = re.match(r'(\d+)([smhdwM])', interval)
        if not match:
            raise ValueError(f"Неправильний формат інтервалу: {interval}")

        number, unit = match.groups()

        if unit == 'M':
            return pd.Timedelta(days=int(number) * 30.44)

        return pd.Timedelta(**{interval_map[unit]: int(number)})

    def detect_outliers(self, data: pd.DataFrame, method: str = 'zscore',
                        threshold: float = 3) -> Tuple[pd.DataFrame, List]:

        if data.empty:
            self.logger.warning("Отримано порожній DataFrame для виявлення аномалій")
            return pd.DataFrame(), []

        self.logger.info(f"Початок виявлення аномалій методом {method} з порогом {threshold}")

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            self.logger.warning("У DataFrame відсутні числові колонки для аналізу аномалій")
            return pd.DataFrame(), []

        outliers_df = pd.DataFrame(index=data.index)
        all_outlier_indices = set()

        if method == 'zscore':
            for col in numeric_cols:
                # Перевірка на стандартне відхилення == 0
                std = data[col].std()
                if std == 0 or pd.isna(std):
                    self.logger.warning(f"Колонка {col} має нульове стандартне відхилення або NaN")
                    continue

                z_scores = np.abs((data[col] - data[col].mean()) / std)
                outliers = z_scores > threshold
                outliers_df[f'{col}_outlier'] = outliers

                if outliers.any():
                    self.logger.info(f"Знайдено {outliers.sum()} аномалій у колонці {col} (zscore)")
                    all_outlier_indices.update(data.index[outliers])

        elif method == 'iqr':
            for col in numeric_cols:
                # Перевірка на достатню кількість даних для обчислення квартилів
                if len(data[col].dropna()) < 4:
                    self.logger.warning(f"Недостатньо даних у колонці {col} для IQR методу")
                    continue

                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1

                if IQR == 0 or pd.isna(IQR):
                    self.logger.warning(f"Колонка {col} має нульовий IQR або NaN")
                    continue

                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
                outliers_df[f'{col}_outlier'] = outliers

                if outliers.any():
                    self.logger.info(f"Знайдено {outliers.sum()} аномалій у колонці {col} (IQR)")
                    all_outlier_indices.update(data.index[outliers])

        elif method == 'isolation_forest':
            try:
                from sklearn.ensemble import IsolationForest

                # Перевірка наявності достатньої кількості даних
                if len(data) < 10:
                    self.logger.warning("Недостатньо даних для Isolation Forest")
                    return pd.DataFrame(), []

                X = data[numeric_cols].fillna(data[numeric_cols].mean())

                # Перевірка на наявність NaN після заповнення
                if X.isna().any().any():
                    self.logger.warning("Залишились NaN після заповнення. Вони будуть замінені на 0")
                    X = X.fillna(0)

                model = IsolationForest(contamination=min(0.1, 1 / threshold), random_state=42)
                predictions = model.fit_predict(X)

                outliers = predictions == -1

                outliers_df['isolation_forest_outlier'] = outliers

                if outliers.any():
                    self.logger.info(f"Знайдено {outliers.sum()} аномалій методом Isolation Forest")
                    all_outlier_indices.update(data.index[outliers])

            except ImportError:
                self.logger.error("Для використання методу 'isolation_forest' необхідно встановити scikit-learn")
                return pd.DataFrame(), []

        else:
            self.logger.error(f"Непідтримуваний метод виявлення аномалій: {method}")
            return pd.DataFrame(), []

        if not outliers_df.empty:
            outliers_df['is_outlier'] = outliers_df.any(axis=1)

        outlier_indices = list(all_outlier_indices)

        self.logger.info(f"Виявлення аномалій завершено. Знайдено {len(outlier_indices)} аномалій у всіх колонках")
        return outliers_df, outlier_indices

    def handle_missing_values(self, data: pd.DataFrame, method: str = 'interpolate',
                              symbol: str = None, interval: str = None,
                              fetch_missing: bool = False) -> pd.DataFrame:
        if data.empty:
            self.logger.warning("Отримано порожній DataFrame для обробки відсутніх значень")
            return data

        result = data.copy()

        missing_values = result.isna().sum()
        total_missing = missing_values.sum()

        if total_missing == 0:
            self.logger.info("Відсутні значення не знайдено")
            return result

        self.logger.info(f"Знайдено {total_missing} відсутніх значень:")
        for col in result.columns:
            if missing_values[col] > 0:
                self.logger.info(f"  - {col}: {missing_values[col]} ({missing_values[col] / len(result) * 100:.2f}%)")

        if isinstance(result.index, pd.DatetimeIndex) and fetch_missing:
            time_diff = result.index.to_series().diff()
            expected_diff = None

            if len(time_diff) > 5:
                expected_diff = time_diff.dropna().median()

            if expected_diff is not None and symbol and interval and method == 'binance':
                missing_periods = self._detect_missing_periods(result, expected_diff)
                if missing_periods:
                    result = self._fetch_missing_data_from_binance(result, missing_periods, symbol, interval)

        filled_values = 0
        numeric_cols = result.select_dtypes(include=[np.number]).columns

        if method == 'interpolate':
            self.logger.info("Застосування методу інтерполяції")

            price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in result.columns]
            if price_cols and isinstance(result.index, pd.DatetimeIndex):
                # Підраховуємо відсутні значення до заповнення
                before_fill_prices = result[price_cols].count().sum()
                result[price_cols] = result[price_cols].interpolate(method='time')
                # Підраховуємо заповнені значення
                filled_values += result[price_cols].count().sum() - before_fill_prices

            other_numeric = [col for col in numeric_cols if col not in price_cols]
            if other_numeric:
                before_fill = result[other_numeric].count().sum()

                if isinstance(result.index, pd.DatetimeIndex):
                    result[other_numeric] = result[other_numeric].interpolate(method='time')
                else:
                    result[other_numeric] = result[other_numeric].interpolate(method='linear')

                filled_values += result[other_numeric].count().sum() - before_fill

            # Використовуємо ffill і bfill для залишкових NaN
            before_fill_total = result.count().sum()
            result = result.fillna(method='ffill').fillna(method='bfill')
            filled_values += result.count().sum() - before_fill_total

        elif method == 'ffill':
            self.logger.info("Застосування методу заповнення попереднім значенням (forward fill)")
            before_fill = result.count().sum()
            result = result.fillna(method='ffill')
            after_forward = result.count().sum()

            # Використовуємо backward fill для перших рядків, які не могли бути заповнені
            result = result.fillna(method='bfill')
            filled_values = result.count().sum() - before_fill

        elif method == 'mean':
            self.logger.info("Застосування методу заповнення середнім значенням")
            for col in numeric_cols:
                if missing_values[col] > 0:
                    # Перевірка на порожність або всі NaN
                    if result[col].dropna().empty:
                        self.logger.warning(f"Колонка {col} не містить значень для обчислення середнього")
                        continue

                    col_mean = result[col].mean()
                    if pd.isna(col_mean):
                        self.logger.warning(f"Середнє значення колонки {col} є NaN")
                        continue

                    missing_before = result[col].isna().sum()
                    result[col] = result[col].fillna(col_mean)
                    filled_values += missing_before - result[col].isna().sum()

        elif method == 'median':
            self.logger.info("Застосування методу заповнення медіанним значенням")
            for col in numeric_cols:
                if missing_values[col] > 0:
                    # Перевірка на порожність або всі NaN
                    if result[col].dropna().empty:
                        self.logger.warning(f"Колонка {col} не містить значень для обчислення медіани")
                        continue

                    col_median = result[col].median()
                    if pd.isna(col_median):
                        self.logger.warning(f"Медіанне значення колонки {col} є NaN")
                        continue

                    missing_before = result[col].isna().sum()
                    result[col] = result[col].fillna(col_median)
                    filled_values += missing_before - result[col].isna().sum()

        remaining_missing = result.isna().sum().sum()
        if remaining_missing > 0:
            self.logger.warning(f"Залишилося {remaining_missing} незаповнених значень після обробки")

        self.logger.info(f"Заповнено {filled_values} відсутніх значень методом '{method}'")
        return result

    def _detect_missing_periods(self, data: pd.DataFrame, expected_diff: pd.Timedelta) -> List[
        Tuple[datetime, datetime]]:

        if not isinstance(data.index, pd.DatetimeIndex) or data.empty:
            return []

        if expected_diff is None:
            self.logger.warning("expected_diff є None, неможливо визначити пропущені періоди")
            return []

        sorted_index = data.index.sort_values()

        time_diff = sorted_index.to_series().diff()
        # Використовуємо більш безпечне порівняння
        large_gaps = time_diff[time_diff > expected_diff * 1.5]

        missing_periods = []
        for timestamp, gap in large_gaps.items():
            prev_timestamp = timestamp - gap

            # Запобігаємо потенційному переповненню при обчисленні missing_steps
            try:
                missing_steps = max(0, int(gap / expected_diff) - 1)
                if missing_steps > 0:
                    self.logger.info(
                        f"Виявлено проміжок: {prev_timestamp} - {timestamp} ({missing_steps} пропущених записів)")
                    missing_periods.append((prev_timestamp, timestamp))
            except (OverflowError, ZeroDivisionError) as e:
                self.logger.error(f"Помилка при обчисленні missing_steps: {str(e)}")

        return missing_periods

    def _fetch_missing_data_from_binance(self, data: pd.DataFrame,
                                         missing_periods: List[Tuple[datetime, datetime]],
                                         symbol: str, interval: str) -> pd.DataFrame:

        if data is None or data.empty:
            self.logger.warning("Отримано порожній DataFrame для заповнення даними")
            return pd.DataFrame()

        # Перевірка валідності параметрів
        if not symbol or not interval:
            self.logger.error("Невалідний symbol або interval")
            return data

        try:
            from binance.client import Client
            # Отримуємо ключі API з конфігураційного файлу або змінних середовища
            api_key = self.config.get('BINANCE_API_KEY', os.environ.get('BINANCE_API_KEY'))
            api_secret = self.config.get('BINANCE_API_SECRET', os.environ.get('BINANCE_API_SECRET'))

            if not api_key or not api_secret:
                self.logger.error("Не знайдено ключі API Binance в конфігурації або змінних середовища")
                return data

            client = Client(api_key, api_secret)

            filled_data = data.copy()

            valid_intervals = ['1m',  '1h',  '4h', '1d',]
            if interval not in valid_intervals:
                self.logger.error(f"Невалідний інтервал: {interval}. Дозволені значення: {valid_intervals}")
                return data

            for start_time, end_time in missing_periods:
                try:
                    self.logger.info(f"Отримання даних з Binance для {symbol} від {start_time} до {end_time}")

                    start_ms = int(start_time.timestamp() * 1000)
                    end_ms = int(end_time.timestamp() * 1000)

                    klines = client.get_historical_klines(
                        symbol=symbol,
                        interval=interval,
                        start_str=start_ms,
                        end_str=end_ms
                    )

                    if not klines:
                        self.logger.warning(f"Дані не отримано з Binance для проміжку {start_time} - {end_time}")
                        continue

                    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                               'close_time', 'quote_asset_volume', 'trades',
                               'taker_buy_base', 'taker_buy_quote', 'ignored']

                    binance_df = pd.DataFrame(klines, columns=columns)
                    binance_df['timestamp'] = pd.to_datetime(binance_df['timestamp'], unit='ms')

                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if col in binance_df.columns:
                            binance_df[col] = pd.to_numeric(binance_df[col])

                    binance_df.set_index('timestamp', inplace=True)

                    common_cols = [col for col in binance_df.columns if col in filled_data.columns]
                    if not common_cols:
                        self.logger.warning("Немає спільних колонок між DataFrame та даними Binance")
                        continue

                    binance_df = binance_df[common_cols]

                    filled_data = pd.concat([filled_data, binance_df])
                    filled_data = filled_data[~filled_data.index.duplicated(keep='first')]
                    filled_data = filled_data.sort_index()

                    self.logger.info(f"Додано {len(binance_df)} записів з Binance")

                except Exception as e:
                    self.logger.error(f"Помилка при отриманні даних з Binance: {str(e)}")

            return filled_data

        except ImportError:
            self.logger.error(
                "Не вдалося імпортувати модуль binance.")
            return data

    def normalize_data(self, data: pd.DataFrame, method: str = 'z-score',
                       columns: List[str] = None, exclude_columns: List[str] = None) -> Tuple[pd.DataFrame, Dict]:

        if data is None or data.empty:
            self.logger.warning("Отримано порожній DataFrame для нормалізації")
            return data, None

        result = data.copy()

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        if columns is not None:
            normalize_cols = [col for col in columns if col in numeric_cols]
            if not normalize_cols:
                self.logger.warning("Жодна з указаних колонок не є числовою")
                return result, None
        else:
            normalize_cols = numeric_cols

        if exclude_columns is not None:
            normalize_cols = [col for col in normalize_cols if col not in exclude_columns]

        if not normalize_cols:
            self.logger.warning("Немає числових колонок для нормалізації")
            return result, None

        self.logger.info(f"Нормалізація {len(normalize_cols)} колонок методом {method}")

        X = result[normalize_cols].values

        scaler = None
        if method == 'z-score':
            scaler = StandardScaler()
            self.logger.info("Застосування StandardScaler (z-score нормалізація)")
        elif method == 'min-max':
            scaler = MinMaxScaler()
            self.logger.info("Застосування MinMaxScaler (min-max нормалізація)")
        elif method == 'robust':
            scaler = RobustScaler()
            self.logger.info("Застосування RobustScaler (робастна нормалізація)")
        else:
            self.logger.error(f"Непідтримуваний метод нормалізації: {method}")
            return result, None

        try:
            # Обробка NaN значень
            if np.isnan(X).any():
                self.logger.warning("Знайдено NaN значення в даних. Заміна на середні значення колонок")
                for i, col in enumerate(normalize_cols):
                    # Перевірка чи вся колонка складається з NaN
                    if np.all(np.isnan(X[:, i])):
                        self.logger.warning(f"Колонка {col} містить лише NaN значення. Заміна на 0.")
                        X[:, i] = 0
                    else:
                        col_mean = np.nanmean(X[:, i])
                        X[:, i] = np.nan_to_num(X[:, i], nan=col_mean)

            X_scaled = scaler.fit_transform(X)

            # Перевірка на нескінченні значення після трансформації
            if not np.isfinite(X_scaled).all():
                self.logger.warning("Знайдено нескінченні значення після нормалізації. Заміна на 0.")
                X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)

            for i, col in enumerate(normalize_cols):
                result[col] = X_scaled[:, i]

            self.logger.info(f"Успішно нормалізовано колонки: {normalize_cols}")

            scaler_meta = {
                'method': method,
                'columns': normalize_cols,
                'scaler': scaler
            }

            return result, scaler_meta

        except Exception as e:
            self.logger.error(f"Помилка при нормалізації даних: {str(e)}")
            return data, None

    def align_time_series(self, data_list: List[pd.DataFrame],
                          reference_index: int = 0) -> List[pd.DataFrame]:

        if not data_list:
            self.logger.warning("Порожній список DataFrame для вирівнювання")
            return []

        if reference_index < 0 or reference_index >= len(data_list):
            self.logger.error(f"Невірний reference_index: {reference_index}. Має бути від 0 до {len(data_list) - 1}")
            reference_index = 0

        # Перевірка та конвертація індексів до DatetimeIndex
        for i, df in enumerate(data_list):
            if df is None or df.empty:
                self.logger.warning(f"DataFrame {i} є порожнім або None")
                data_list[i] = pd.DataFrame()  # Замінюємо на порожній DataFrame
                continue

            if not isinstance(df.index, pd.DatetimeIndex):
                self.logger.warning(f"DataFrame {i} не має часового індексу. Спроба конвертувати.")
                try:
                    time_cols = [col for col in df.columns if
                                 any(x in col.lower() for x in ['time', 'date', 'timestamp'])]
                    if time_cols:
                        df[time_cols[0]] = pd.to_datetime(df[time_cols[0]])
                        df.set_index(time_cols[0], inplace=True)
                        data_list[i] = df
                    else:
                        self.logger.error(f"Неможливо конвертувати DataFrame {i}: не знайдено часову колонку")
                        return []
                except Exception as e:
                    self.logger.error(f"Помилка при конвертації індексу для DataFrame {i}: {str(e)}")
                    return []

        # Перевірка еталонного DataFrame
        reference_df = data_list[reference_index]
        if reference_df is None or reference_df.empty:
            self.logger.error("Еталонний DataFrame є порожнім")
            return data_list

        # Визначення частоти часового ряду
        try:
            reference_freq = pd.infer_freq(reference_df.index)

            if not reference_freq:
                self.logger.warning("Не вдалося визначити частоту reference DataFrame. Спроба визначити вручну.")
                if len(reference_df.index) > 1:
                    time_diff = reference_df.index.to_series().diff().dropna()
                    if not time_diff.empty:
                        reference_freq = time_diff.median()
                        self.logger.info(f"Визначено медіанний інтервал: {reference_freq}")
                    else:
                        self.logger.error("Не вдалося визначити інтервал з різниці часових міток")
                        return data_list
                else:
                    self.logger.error("Недостатньо точок для визначення частоти reference DataFrame")
                    return data_list
        except Exception as e:
            self.logger.error(f"Помилка при визначенні частоти reference DataFrame: {str(e)}")
            return data_list

        aligned_data_list = [reference_df]

        for i, df in enumerate(data_list):
            if i == reference_index:
                continue

            if df is None or df.empty:
                self.logger.warning(f"Пропускаємо порожній DataFrame {i}")
                aligned_data_list.append(df)
                continue

            self.logger.info(f"Вирівнювання DataFrame {i} з reference DataFrame")

            if df.index.equals(reference_df.index):
                aligned_data_list.append(df)
                continue

            try:
                start_time = max(df.index.min(), reference_df.index.min())
                end_time = min(df.index.max(), reference_df.index.max())

                reference_subset = reference_df.loc[(reference_df.index >= start_time) &
                                                    (reference_df.index <= end_time)]

                # Безпечний спосіб reindex
                aligned_df = df.reindex(reference_subset.index, method=None)

                numeric_cols = aligned_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    aligned_df[numeric_cols] = aligned_df[numeric_cols].interpolate(method='time')

                aligned_data_list.append(aligned_df)

                missing_values = aligned_df.isna().sum().sum()
                if missing_values > 0:
                    self.logger.warning(
                        f"Після вирівнювання DataFrame {i} залишилося {missing_values} відсутніх значень")

            except Exception as e:
                self.logger.error(f"Помилка при вирівнюванні DataFrame {i}: {str(e)}")
                self.logger.error(f"Деталі помилки: {traceback.format_exc()}")
                aligned_data_list.append(df)  # Додаємо оригінал при помилці

        return aligned_data_list

    def validate_data_integrity(self, data: pd.DataFrame, price_jump_threshold: float = 0.2,
                                volume_anomaly_threshold: float = 5) -> Dict[str, List]:

        if data is None or data.empty:
            self.logger.warning("Отримано порожній DataFrame для перевірки цілісності")
            return {"empty_data": []}

        issues = {}

        expected_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in expected_cols if col not in data.columns]
        if missing_cols:
            issues["missing_columns"] = missing_cols
            self.logger.warning(f"Відсутні колонки: {missing_cols}")

        if not isinstance(data.index, pd.DatetimeIndex):
            issues["not_datetime_index"] = True
            self.logger.warning("Індекс не є DatetimeIndex")
        else:
            # Перевірка часових проміжків
            if len(data.index) > 1:
                time_diff = data.index.to_series().diff().dropna()
                if not time_diff.empty:
                    median_diff = time_diff.median()

                    # Перевірка на великі проміжки
                    if median_diff.total_seconds() > 0:  # Уникаємо ділення на нуль
                        large_gaps = time_diff[time_diff > 2 * median_diff]
                        if not large_gaps.empty:
                            gap_locations = large_gaps.index.tolist()
                            issues["time_gaps"] = gap_locations
                            self.logger.warning(f"Знайдено {len(gap_locations)} аномальних проміжків у часових мітках")

                    # Перевірка на дублікати часових міток
                    duplicates = data.index.duplicated()
                    if duplicates.any():
                        dup_indices = data.index[duplicates].tolist()
                        issues["duplicate_timestamps"] = dup_indices
                        self.logger.warning(f"Знайдено {len(dup_indices)} дублікатів часових міток")

        # Перевірка цінових аномалій
        price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in data.columns]
        if len(price_cols) == 4:
            # Перевірка high < low
            invalid_hl = data['high'] < data['low']
            if invalid_hl.any():
                invalid_hl_indices = data.index[invalid_hl].tolist()
                issues["high_lower_than_low"] = invalid_hl_indices
                self.logger.warning(f"Знайдено {len(invalid_hl_indices)} записів де high < low")

            # Перевірка від'ємних цін
            for col in price_cols:
                negative_prices = data[col] < 0
                if negative_prices.any():
                    neg_price_indices = data.index[negative_prices].tolist()
                    issues[f"negative_{col}"] = neg_price_indices
                    self.logger.warning(f"Знайдено {len(neg_price_indices)} записів з від'ємними значеннями у {col}")

            # Перевірка різких стрибків цін
            for col in price_cols:
                pct_change = data[col].pct_change().abs()
                price_jumps = pct_change > price_jump_threshold
                if price_jumps.any():
                    jump_indices = data.index[price_jumps].tolist()
                    issues[f"price_jumps_{col}"] = jump_indices
                    self.logger.warning(f"Знайдено {len(jump_indices)} різких змін у колонці {col}")

        # Перевірка об'єму
        if 'volume' in data.columns:
            # Перевірка від'ємного об'єму
            negative_volume = data['volume'] < 0
            if negative_volume.any():
                neg_vol_indices = data.index[negative_volume].tolist()
                issues["negative_volume"] = neg_vol_indices
                self.logger.warning(f"Знайдено {len(neg_vol_indices)} записів з від'ємним об'ємом")

            # Перевірка аномального об'єму
            try:
                volume_std = data['volume'].std()
                if volume_std > 0:  # Уникаємо ділення на нуль
                    volume_zscore = np.abs((data['volume'] - data['volume'].mean()) / volume_std)
                    volume_anomalies = volume_zscore > volume_anomaly_threshold
                    if volume_anomalies.any():
                        vol_anomaly_indices = data.index[volume_anomalies].tolist()
                        issues["volume_anomalies"] = vol_anomaly_indices
                        self.logger.warning(f"Знайдено {len(vol_anomaly_indices)} записів з аномальним об'ємом")
            except Exception as e:
                self.logger.error(f"Помилка при аналізі аномалій об'єму: {str(e)}")

        # Перевірка на NaN значення
        na_counts = data.isna().sum()
        cols_with_na = na_counts[na_counts > 0].index.tolist()
        if cols_with_na:
            issues["columns_with_na"] = {col: data.index[data[col].isna()].tolist() for col in cols_with_na}
            self.logger.warning(f"Знайдено відсутні значення у колонках: {cols_with_na}")

        # Перевірка на нескінченні значення
        try:
            inf_counts = np.isinf(data.select_dtypes(include=[np.number])).sum()
            cols_with_inf = inf_counts[inf_counts > 0].index.tolist()
            if cols_with_inf:
                issues["columns_with_inf"] = {col: data.index[np.isinf(data[col])].tolist() for col in cols_with_inf}
                self.logger.warning(f"Знайдено нескінченні значення у колонках: {cols_with_inf}")
        except Exception as e:
            self.logger.error(f"Помилка при перевірці нескінченних значень: {str(e)}")

        return issues

    def aggregate_volume_profile(self, data: pd.DataFrame, bins: int = 10,
                                 price_col: str = 'close', volume_col: str = 'volume',
                                 time_period: Optional[str] = None) -> pd.DataFrame:

        if data is None or data.empty:
            self.logger.warning("Отримано порожній DataFrame для профілю об'єму")
            return pd.DataFrame()

        if price_col not in data.columns:
            self.logger.error(f"Колонка {price_col} відсутня у DataFrame")
            return pd.DataFrame()

        if volume_col not in data.columns:
            self.logger.error(f"Колонка {volume_col} відсутня у DataFrame")
            return pd.DataFrame()

        self.logger.info(f"Створення профілю об'єму з {bins} ціновими рівнями")

        # Перевірка можливості створення часового профілю
        if time_period:
            if not isinstance(data.index, pd.DatetimeIndex):
                self.logger.warning("Індекс не є DatetimeIndex. Часовий профіль не може бути створений.")
                # Створюємо простий профіль об'єму замість часового
                return self._create_volume_profile(data, bins, price_col, volume_col)

            self.logger.info(f"Створення часового профілю об'єму з періодом {time_period}")
            period_groups = data.groupby(pd.Grouper(freq=time_period))

            result_dfs = []

            for period, group in period_groups:
                if group.empty:
                    continue

                period_profile = self._create_volume_profile(group, bins, price_col, volume_col)
                if not period_profile.empty:
                    period_profile['period'] = period
                    result_dfs.append(period_profile)

            if result_dfs:
                return pd.concat(result_dfs)
            else:
                self.logger.warning("Не вдалося створити часовий профіль об'єму")
                return pd.DataFrame()
        else:
            return self._create_volume_profile(data, bins, price_col, volume_col)

    def _create_volume_profile(self, data: pd.DataFrame, bins: int,
                               price_col: str, volume_col: str) -> pd.DataFrame:

        price_min = data[price_col].min()
        price_max = data[price_col].max()

        if price_min == price_max:
            self.logger.warning("Мінімальна та максимальна ціни однакові. Неможливо створити профіль об'єму.")
            return pd.DataFrame()

        effective_bins = min(bins, int((price_max - price_min) * 100) + 1)
        if effective_bins < bins:
            self.logger.warning(f"Зменшено кількість бінів з {bins} до {effective_bins} через малий діапазон цін")
            bins = effective_bins

        if bins <= 1:
            self.logger.warning("Недостатньо бінів для створення профілю об'єму")
            return pd.DataFrame()

        try:
            bin_edges = np.linspace(price_min, price_max, bins + 1)
            bin_width = (price_max - price_min) / bins

            bin_labels = list(range(bins))
            data = data.copy()  # гарантія, що не змінюємо оригінал

            # Виправлено SettingWithCopyWarning
            data.loc[:, 'price_bin'] = pd.cut(
                data[price_col],
                bins=bin_edges,
                labels=bin_labels,
                include_lowest=True
            )

            # Виправлено FutureWarning — додано observed=False
            volume_profile = data.groupby('price_bin', observed=False).agg({
                volume_col: 'sum',
                price_col: ['count', 'min', 'max']
            })

            if volume_profile.empty:
                self.logger.warning("Отримано порожній профіль об'єму після групування")
                return pd.DataFrame()

            volume_profile.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in volume_profile.columns]
            volume_profile = volume_profile.rename(columns={
                f'{volume_col}_sum': 'volume',
                f'{price_col}_count': 'count',
                f'{price_col}_min': 'price_min',
                f'{price_col}_max': 'price_max'
            })

            total_volume = volume_profile['volume'].sum()
            if total_volume > 0:
                volume_profile['volume_percent'] = (volume_profile['volume'] / total_volume * 100).round(2)
            else:
                volume_profile['volume_percent'] = 0

            volume_profile['price_mid'] = (volume_profile['price_min'] + volume_profile['price_max']) / 2

            volume_profile['bin_lower'] = [bin_edges[i] for i in volume_profile.index]
            volume_profile['bin_upper'] = [bin_edges[i + 1] for i in volume_profile.index]

            volume_profile = volume_profile.reset_index()
            volume_profile = volume_profile.sort_values('price_bin', ascending=False)

            if 'price_bin' in volume_profile.columns:
                volume_profile = volume_profile.drop('price_bin', axis=1)

            return volume_profile

        except Exception as e:
            self.logger.error(f"Помилка при створенні профілю об'єму: {str(e)}")
            return pd.DataFrame()

    def add_time_features(self, data: pd.DataFrame, cyclical: bool = True,
                          add_sessions: bool = False, tz: str = 'Europe/Kiev') -> pd.DataFrame:

        if data.empty:
            self.logger.warning("Отримано порожній DataFrame для додавання часових ознак")
            return data

        result = data.copy()

        if not isinstance(result.index, pd.DatetimeIndex):
            self.logger.warning("Індекс не є DatetimeIndex. Спроба конвертувати.")
            try:
                time_cols = [col for col in result.columns if
                             any(x in col.lower() for x in ['time', 'date', 'timestamp'])]
                if time_cols:
                    time_col = time_cols[0]
                    result[time_col] = pd.to_datetime(result[time_col])
                    result.set_index(time_col, inplace=True)
                else:
                    self.logger.error("Неможливо додати часові ознаки: не знайдено часову колонку")
                    return data
            except Exception as e:
                self.logger.error(f"Помилка при конвертації індексу: {str(e)}")
                return data

        self.logger.info("Додавання часових ознак")

        if result.index.tz is None:
            self.logger.info(f"Встановлення часового поясу {tz}")
            try:
                result.index = result.index.tz_localize(tz)
            except Exception as e:
                self.logger.warning(
                    f"Помилка при локалізації часового поясу: {str(e)}. Продовжуємо без часового поясу.")
        elif result.index.tz.zone != tz:
            self.logger.info(f"Конвертація часового поясу з {result.index.tz.zone} в {tz}")
            try:
                result.index = result.index.tz_convert(tz)
            except Exception as e:
                self.logger.warning(
                    f"Помилка при конвертації часового поясу: {str(e)}. Продовжуємо з поточним часовим поясом.")

        result['hour'] = result.index.hour
        result['day'] = result.index.day
        result['weekday'] = result.index.weekday
        result['week'] = result.index.isocalendar().week
        result['month'] = result.index.month
        result['quarter'] = result.index.quarter
        result['year'] = result.index.year
        result['dayofyear'] = result.index.dayofyear

        result['is_weekend'] = result['weekday'].isin([5, 6]).astype(int)
        result['is_month_start'] = result.index.is_month_start.astype(int)
        result['is_month_end'] = result.index.is_month_end.astype(int)
        result['is_quarter_start'] = result.index.is_quarter_start.astype(int)
        result['is_quarter_end'] = result.index.is_quarter_end.astype(int)
        result['is_year_start'] = result.index.is_year_start.astype(int)
        result['is_year_end'] = result.index.is_year_end.astype(int)

        if cyclical:
            self.logger.info("Додавання циклічних ознак")

            result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
            result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)

            days_in_month = result.index.days_in_month
            result['day_sin'] = np.sin(2 * np.pi * result['day'] / days_in_month)
            result['day_cos'] = np.cos(2 * np.pi * result['day'] / days_in_month)

            result['weekday_sin'] = np.sin(2 * np.pi * result['weekday'] / 7)
            result['weekday_cos'] = np.cos(2 * np.pi * result['weekday'] / 7)

            result['week_sin'] = np.sin(2 * np.pi * result['week'] / 52)
            result['week_cos'] = np.cos(2 * np.pi * result['week'] / 52)

            result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
            result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)

            result['quarter_sin'] = np.sin(2 * np.pi * result['quarter'] / 4)
            result['quarter_cos'] = np.cos(2 * np.pi * result['quarter'] / 4)

        if add_sessions:
            self.logger.info("Додавання індикаторів торгових сесій")

            result['asian_session'] = ((result['hour'] >= 0) & (result['hour'] < 9)).astype(int)

            result['european_session'] = ((result['hour'] >= 8) & (result['hour'] < 17)).astype(int)

            result['american_session'] = ((result['hour'] >= 13) & (result['hour'] < 22)).astype(int)

            result['asia_europe_overlap'] = ((result['hour'] >= 8) & (result['hour'] < 9)).astype(int)
            result['europe_america_overlap'] = ((result['hour'] >= 13) & (result['hour'] < 17)).astype(int)

            result['inactive_hours'] = ((result['hour'] >= 22) | (result['hour'] < 0)).astype(int)

        self.logger.info(f"Успішно додано {len(result.columns) - len(data.columns)} часових ознак")
        return result


    def remove_duplicate_timestamps(self, data: pd.DataFrame) -> pd.DataFrame:

        if data.empty:
            self.logger.warning("Отримано порожній DataFrame для обробки дублікатів")
            return data

        if not isinstance(data.index, pd.DatetimeIndex):
            self.logger.warning("Індекс не є DatetimeIndex. Спроба конвертувати.")
            try:
                time_cols = [col for col in data.columns if
                             any(x in col.lower() for x in ['time', 'date', 'timestamp'])]
                if time_cols:
                    time_col = time_cols[0]
                    data[time_col] = pd.to_datetime(data[time_col])
                    data.set_index(time_col, inplace=True)
                else:
                    self.logger.error("Неможливо виявити дублікати: не знайдено часову колонку")
                    return data
            except Exception as e:
                self.logger.error(f"Помилка при конвертації індексу: {str(e)}")
                return data

        duplicates = data.index.duplicated()
        duplicates_count = duplicates.sum()

        if duplicates_count == 0:
            self.logger.info("Дублікати часових міток не знайдено")
            return data

        self.logger.info(f"Знайдено {duplicates_count} дублікатів часових міток")

        result = data[~duplicates]

        self.logger.info(f"Видалено {duplicates_count} дублікатів. Залишилось {len(result)} записів.")
        return result

    def filter_by_time_range(self, data: pd.DataFrame,
                             start_time: Optional[Union[str, datetime]] = None,
                             end_time: Optional[Union[str, datetime]] = None) -> pd.DataFrame:

        if data.empty:
            self.logger.warning("Отримано порожній DataFrame для фільтрації")
            return data

        if not isinstance(data.index, pd.DatetimeIndex):
            self.logger.warning("Індекс не є DatetimeIndex. Спроба конвертувати.")
            try:
                time_cols = [col for col in data.columns if
                             any(x in col.lower() for x in ['time', 'date', 'timestamp'])]
                if time_cols:
                    time_col = time_cols[0]
                    data[time_col] = pd.to_datetime(data[time_col])
                    data.set_index(time_col, inplace=True)
                else:
                    self.logger.error("Неможливо фільтрувати за часом: не знайдено часову колонку")
                    return data
            except Exception as e:
                self.logger.error(f"Помилка при конвертації індексу: {str(e)}")
                return data

        result = data.copy()
        initial_count = len(result)

        if start_time is not None:
            try:
                start_dt = pd.to_datetime(start_time)
                result = result[result.index >= start_dt]
                self.logger.info(f"Фільтрація за початковим часом: {start_dt}")
            except Exception as e:
                self.logger.error(f"Помилка при конвертації початкового часу: {str(e)}")

        if end_time is not None:
            try:
                end_dt = pd.to_datetime(end_time)
                result = result[result.index <= end_dt]
                self.logger.info(f"Фільтрація за кінцевим часом: {end_dt}")
            except Exception as e:
                self.logger.error(f"Помилка при конвертації кінцевого часу: {str(e)}")

        final_count = len(result)

        if start_time is not None and end_time is not None:
            start_dt = pd.to_datetime(start_time)
            end_dt = pd.to_datetime(end_time)
            if start_dt > end_dt:
                self.logger.warning(f"Початковий час ({start_dt}) пізніше кінцевого часу ({end_dt})")

        self.logger.info(f"Відфільтровано {initial_count - final_count} записів. Залишилось {final_count} записів.")
        return result

    def save_processed_data(self, data: pd.DataFrame, filename: str, db_connection=None) -> str:

        if data.empty:
            self.logger.warning("Спроба зберегти порожній DataFrame")
            return ""

        # Збереження в базу даних, якщо надано з'єднання
        if db_connection:
            try:
                table_name = os.path.basename(filename).split('.')[0]
                data.to_sql(table_name, db_connection, if_exists='replace', index=True)
                self.logger.info(f"Дані збережено в базу даних, таблиця: {table_name}")
                return table_name
            except Exception as e:
                self.logger.error(f"Помилка при збереженні в базу даних: {str(e)}")
                return ""

        # Забезпечення формату CSV
        if '.' in filename and filename.split('.')[-1].lower() != 'csv':
            filename = f"{filename.split('.')[0]}.csv"
            self.logger.warning(f"Змінено формат файлу на CSV: {filename}")

        if not filename.endswith('.csv'):
            filename = f"{filename}.csv"

        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            self.logger.info(f"Створено директорію: {directory}")

        try:
            data.to_csv(filename)
            self.logger.info(f"Дані збережено у CSV форматі: {filename}")
            return os.path.abspath(filename)
        except Exception as e:
            self.logger.error(f"Помилка при збереженні даних: {str(e)}")
            return ""

    def load_processed_data(self, filename: str) -> pd.DataFrame:

        if not os.path.exists(filename):
            self.logger.error(f"Файл не знайдено: {filename}")
            return pd.DataFrame()

        file_extension = filename.split('.')[-1].lower() if '.' in filename else ''

        try:
            if file_extension == 'csv':
                data = pd.read_csv(filename)
                self.logger.info(f"Дані завантажено з CSV файлу: {filename}")

                time_cols = [col for col in data.columns if
                             any(x in col.lower() for x in ['time', 'date', 'timestamp'])]
                if time_cols:
                    time_col = time_cols[0]
                    data[time_col] = pd.to_datetime(data[time_col])
                    data.set_index(time_col, inplace=True)
                    self.logger.info(f"Встановлено індекс за колонкою {time_col}")
            else:
                self.logger.error(f"Підтримується лише формат CSV, отримано: {file_extension}")
                return pd.DataFrame()

            if not isinstance(data.index, pd.DatetimeIndex) and len(data) > 0:
                self.logger.warning("Завантажені дані не мають DatetimeIndex. Спроба конвертувати.")
                try:
                    time_cols = [col for col in data.columns if
                                 any(x in col.lower() for x in ['time', 'date', 'timestamp'])]
                    if time_cols:
                        time_col = time_cols[0]
                        data[time_col] = pd.to_datetime(data[time_col])
                        data.set_index(time_col, inplace=True)
                        self.logger.info(f"Встановлено індекс за колонкою {time_col}")
                except Exception as e:
                    self.logger.warning(f"Не вдалося встановити DatetimeIndex: {str(e)}")

            return data

        except Exception as e:
            self.logger.error(f"Помилка при завантаженні даних: {str(e)}")
            return pd.DataFrame()

    def merge_datasets(self, datasets: List[pd.DataFrame],
                       merge_on: str = 'timestamp') -> pd.DataFrame:

        if not datasets:
            self.logger.warning("Порожній список наборів даних для об'єднання")
            return pd.DataFrame()

        if len(datasets) == 1:
            return datasets[0].copy()

        self.logger.info(f"Початок об'єднання {len(datasets)} наборів даних")

        all_have_merge_on = all(merge_on in df.columns or df.index.name == merge_on for df in datasets)

        if not all_have_merge_on:
            if merge_on == 'timestamp':
                self.logger.info("Перевірка, чи всі DataFrame мають DatetimeIndex")
                all_have_datetime_index = all(isinstance(df.index, pd.DatetimeIndex) for df in datasets)

                if all_have_datetime_index:
                    for i in range(len(datasets)):
                        if datasets[i].index.name is None:
                            datasets[i].index.name = 'timestamp'

                    all_have_merge_on = True

            if not all_have_merge_on:
                self.logger.error(f"Не всі набори даних містять '{merge_on}' для об'єднання")
                return pd.DataFrame()

        datasets_copy = []
        for i, df in enumerate(datasets):
            df_copy = df.copy()

            if merge_on in df_copy.columns:
                df_copy.set_index(merge_on, inplace=True)
                self.logger.info(f"DataFrame {i} перетворено: колонка '{merge_on}' стала індексом")
            elif df_copy.index.name != merge_on:
                df_copy.index.name = merge_on
                self.logger.info(f"DataFrame {i}: індекс перейменовано на '{merge_on}'")

            datasets_copy.append(df_copy)

        result = datasets_copy[0]
        total_columns = len(result.columns)

        for i, df in enumerate(datasets_copy[1:], 2):
            rename_dict = {}
            for col in df.columns:
                if col in result.columns:
                    rename_dict[col] = f"{col}_{i}"

            if rename_dict:
                self.logger.info(f"Перейменування колонок у DataFrame {i}: {rename_dict}")
                df = df.rename(columns=rename_dict)

            result = result.join(df, how='outer')
            total_columns += len(df.columns)

        self.logger.info(f"Об'єднання завершено. Результат: {len(result)} рядків, {len(result.columns)} колонок")
        self.logger.info(f"З {total_columns} вхідних колонок, {total_columns - len(result.columns)} були дублікатами")

        return result

    def preprocess_pipeline(self, data: pd.DataFrame, steps: Optional[List[Dict]] = None) -> pd.DataFrame:

        if data.empty:
            self.logger.warning("Отримано порожній DataFrame для обробки в конвеєрі")
            return data

        if steps is None:
            steps = [
                {'name': 'remove_duplicate_timestamps', 'params': {}},
                {'name': 'clean_data', 'params': {'remove_outliers': True, 'fill_missing': True}},
                {'name': 'handle_missing_values', 'params': {'method': 'interpolate'}}
            ]

        self.logger.info(f"Початок виконання конвеєра обробки даних з {len(steps)} кроками")

        result = data.copy()

        for step_idx, step in enumerate(steps, 1):
            step_name = step.get('name')
            step_params = step.get('params', {})

            if not hasattr(self, step_name):
                self.logger.warning(f"Крок {step_idx}: Метод '{step_name}' не існує. Пропускаємо.")
                continue

            try:
                self.logger.info(f"Крок {step_idx}: Виконання '{step_name}' з параметрами {step_params}")

                method = getattr(self, step_name)

                if step_name == 'normalize_data':
                    result, _ = method(result, **step_params)
                elif step_name == 'detect_outliers':
                    outliers_df, _ = method(result, **step_params)
                    self.logger.info(f"Виявлено аномалії, але дані не змінено")
                else:
                    result = method(result, **step_params)

                self.logger.info(
                    f"Крок {step_idx}: '{step_name}' завершено. Результат: {len(result)} рядків, {len(result.columns)} колонок")

            except Exception as e:
                self.logger.error(f"Помилка на кроці {step_idx}: '{step_name}': {str(e)}")

        self.logger.info(
            f"Конвеєр обробки даних завершено. Початково: {len(data)} рядків, {len(data.columns)} колонок. "
            f"Результат: {len(result)} рядків, {len(result.columns)} колонок.")

        return result
def main():
    # Конфігурація
    EU_TIMEZONE = 'Europe/Kiev'
    SYMBOLS = ['BTC', 'ETH', 'SOL']
    INTERVALS = ['1d', '1h', '4h']

    data_source_paths = {
        'csv': {
            'BTC': {
                '1d': '/Users/bogdanresetko/Desktop/kursova/data/crypto_data/BTCUSDT_1d.csv',
                '1h': '/Users/bogdanresetko/Desktop/kursova//data/crypto_data/BTCUSDT_1h.csv',
                '4h': '/Users/bogdanresetko/Desktop/kursova//data/crypto_data/BTCUSDT_4h.csv'
            },
            'ETH': {
                '1d': '/Users/bogdanresetko/Desktop/kursova//data/crypto_data/ETHUSDT_1d.csv',
                '1h': '/Users/bogdanresetko/Desktop/kursova//data/crypto_data/ETHUSDT_1h.csv',
                '4h': '/Users/bogdanresetko/Desktop/kursova//data/crypto_data/ETHUSDT_4h.csv'
            },
            'SOL': {
                '1d': '/Users/bogdanresetko/Desktop/kursova//data/crypto_data/SOLUSDT_1d.csv',
                '1h': '/Users/bogdanresetko/Desktop/kursova//data/crypto_data/SOLUSDT_1h.csv',
                '4h': '/Users/bogdanresetko/Desktop/kursova//data/crypto_data/SOLUSDT_4h.csv'
            }
        }
    }

    processor = MarketDataProcessor()

    for symbol in SYMBOLS:
        for interval in INTERVALS:
            print(f"\n🔄 Обробка {symbol} {interval}...")

            data = processor.load_data(
                data_source='database',
                symbol=symbol,
                interval=interval,
                data_type='candles'
            )

            if data.empty:
                file_path = data_source_paths['csv'].get(symbol, {}).get(interval)
                if not file_path:
                    print(f"⚠️ Немає CSV-файлу для {symbol} {interval}")
                    continue

                print(f"📁 Завантаження з CSV: {file_path}")
                data = processor.load_data(
                    data_source='csv',
                    symbol=symbol,
                    interval=interval,
                    file_path=file_path,
                    data_type='candles'
                )

                if data.empty:
                    print(f"⚠️ Дані не знайдено для {symbol} {interval}")
                    continue

                processor.save_klines_to_db(data, symbol, interval)
                print("📥 Збережено свічки в базу даних")

            print(f"✔️ Завантажено {len(data)} рядків")

            # Обробка
            processed_data = processor.preprocess_pipeline(data)

            if interval != '1d':
                processed_data = processor.resample_data(processed_data, target_interval='1d')

            processed_data = processor.add_time_features(processed_data, tz=EU_TIMEZONE)

            # Збереження оброблених даних
            processor.save_processed_klines_to_db(processed_data, symbol, '1d')
            print("✅ Оброблені дані збережено в БД")

            # Побудова та збереження профілю об'єму
            volume_profile = processor.aggregate_volume_profile(
                processed_data, bins=12, time_period='1W'
            )
            if not volume_profile.empty:
                processor.save_volume_profile_to_db(volume_profile, symbol, '1d')
                print("📊 Профіль об'єму збережено")

if __name__ == "__main__":
    main()
