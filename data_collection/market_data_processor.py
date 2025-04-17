import os
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
import data.db as db

class MarketDataProcessor:

    def __init__(self, cache_dir=None, log_level=logging.INFO):
        self.cache_dir = cache_dir
        self.log_level = log_level
        self.db_connection = db_connection

        logging.basicConfig(level=self.log_level)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Ініціалізація класу...")

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.logger.info(f"Директорію для кешу створено: {self.cache_dir}")

        self.cache_index = {}
        self._load_cache_index()
        self.ready = True

    def _get_cache_path(self, cache_key: str) -> str:
        return os.path.join(self.cache_dir, f"{cache_key}.parquet")

    def save_to_cache(self, cache_key: str, data: pd.DataFrame, metadata: Dict = None) -> bool:
        cache_path = self._get_cache_path(cache_key)
        try:
            data.to_parquet(cache_path)

            self.cache_index[cache_key] = {
                "created_at": datetime.now().isoformat(),
                "rows": len(data),
                "columns": list(data.columns),
                **(metadata or {})
            }

            self._save_cache_index()
            return True
        except Exception as e:
            print(f"Помилка збереження в кеш: {e}")
            return False

    def _load_from_database(self, symbol: str, interval: str,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            data_type: str = 'candles') -> pd.DataFrame:

        self.logger.info(f"Завантаження {data_type} даних з бази даних для {symbol} {interval}")

        try:
            if data_type == 'candles':
                data = db.get_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=start_date,
                    end_time=end_date
                )
            elif data_type == 'orderbook':
                data = db.get_orderbook(
                    symbol=symbol,
                    start_time=start_date,
                    end_time=end_date
                )
            else:
                raise ValueError(f"Непідтримуваний тип даних: {data_type}")

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

        cache_key = self.create_cache_key(
            data_source, symbol, interval, start_date, end_date, data_type
        )

        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.parquet")
            if os.path.exists(cache_file):
                self.logger.info(f"Завантаження даних з кешу: {cache_key}")
                return pd.read_parquet(cache_file)

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
                    time_col = next(col for col in ['timestamp', 'date', 'time'] if col in data.columns)
                    data[time_col] = pd.to_datetime(data[time_col])
                    data.set_index(time_col, inplace=True)

                if start_date_dt:
                    data = data[data.index >= start_date_dt]
                if end_date_dt:
                    data = data[data.index <= end_date_dt]

            else:
                raise ValueError(f"Непідтримуване джерело даних: {data_source}")

            if data is None or data.empty:
                self.logger.warning(f"Отримано порожній набір даних від {data_source}")
                return pd.DataFrame()

            if self.cache_dir:
                self.save_to_cache(cache_key, data, metadata={
                    'source': data_source,
                    'symbol': symbol,
                    'interval': interval,
                    'data_type': data_type,
                    'start_date': start_date_dt.isoformat() if start_date_dt else None,
                    'end_date': end_date_dt.isoformat() if end_date_dt else None,
                    'file_path': file_path if data_source == 'csv' else None
                })

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
            if price_cols:
                result[price_cols] = result[price_cols].interpolate(method='time')

            if 'volume' in result.columns and result['volume'].isna().any():
                result['volume'] = result['volume'].fillna(0)

            numeric_cols = result.select_dtypes(include=[np.number]).columns
            other_numeric = [col for col in numeric_cols if col not in price_cols + ['volume']]
            if other_numeric:
                result[other_numeric] = result[other_numeric].interpolate(method='time')

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
        match = re.match(r'(\d+)([smhdw])', interval)
        if not match:
            raise ValueError(f"Неправильний формат інтервалу: {interval}")

        number, unit = match.groups()

        if unit == 'M':
            return pd.Timedelta(days=int(number) * 30)

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
                if data[col].std() == 0:
                    continue

                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outliers = z_scores > threshold
                outliers_df[f'{col}_outlier'] = outliers

                if outliers.any():
                    self.logger.info(f"Знайдено {outliers.sum()} аномалій у колонці {col} (zscore)")
                    all_outlier_indices.update(data.index[outliers])

        elif method == 'iqr':
            for col in numeric_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1

                if IQR == 0:
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

                X = data[numeric_cols].fillna(data[numeric_cols].mean())

                model = IsolationForest(contamination=1 / threshold, random_state=42)
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
                result[price_cols] = result[price_cols].interpolate(method='time')
                filled_values += result[price_cols].count().sum() - (
                            len(result) * len(price_cols) - missing_values[price_cols].sum())

            other_numeric = [col for col in numeric_cols if col not in price_cols]
            if other_numeric:
                before_fill = result[other_numeric].count().sum()
                result[other_numeric] = result[other_numeric].interpolate(method='linear')
                filled_values += result[other_numeric].count().sum() - before_fill

            result = result.fillna(method='ffill').fillna(method='bfill')

        elif method == 'ffill':
            self.logger.info("Застосування методу заповнення попереднім значенням (forward fill)")
            before_fill = result.count().sum()
            result = result.fillna(method='ffill')
            filled_values = result.count().sum() - before_fill

        elif method == 'mean':
            self.logger.info("Застосування методу заповнення середнім значенням")
            for col in numeric_cols:
                if missing_values[col] > 0:
                    col_mean = result[col].mean()
                    missing_before = result[col].isna().sum()
                    result[col] = result[col].fillna(col_mean)
                    filled_values += missing_before - result[col].isna().sum()

        elif method == 'median':
            self.logger.info("Застосування методу заповнення медіанним значенням")
            for col in numeric_cols:
                if missing_values[col] > 0:
                    col_median = result[col].median()
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

        if not isinstance(data.index, pd.DatetimeIndex):
            return []

        sorted_index = data.index.sort_values()

        time_diff = sorted_index.to_series().diff()
        large_gaps = time_diff[time_diff > expected_diff * 1.5]

        missing_periods = []
        for timestamp, gap in large_gaps.items():
            prev_timestamp = timestamp - gap

            missing_steps = int(gap / expected_diff) - 1
            if missing_steps > 0:
                self.logger.info(
                    f"Виявлено проміжок: {prev_timestamp} - {timestamp} ({missing_steps} пропущених записів)")
                missing_periods.append((prev_timestamp, timestamp))

        return missing_periods

    def _fetch_missing_data_from_binance(self, data: pd.DataFrame,
                                         missing_periods: List[Tuple[datetime, datetime]],
                                         symbol: str, interval: str) -> pd.DataFrame:
        try:
            from binance.client import Client

            client = Client("", "")

            filled_data = data.copy()

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
                "Не вдалося імпортувати модуль binance. Встановіть його за допомогою 'pip install python-binance'")
            return data

    def normalize_data(self, data: pd.DataFrame, method: str = 'z-score',
                       columns: List[str] = None, exclude_columns: List[str] = None) -> Tuple[pd.DataFrame, object]:
        if data.empty:
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

        original_index = result.index

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
            if np.isnan(X).any():
                self.logger.warning("Знайдено NaN значення в даних. Заміна на середні значення колонок")
                for i, col in enumerate(normalize_cols):
                    col_mean = np.nanmean(X[:, i])
                    X[:, i] = np.nan_to_num(X[:, i], nan=col_mean)

            X_scaled = scaler.fit_transform(X)

            for i, col in enumerate(normalize_cols):
                result[col] = X_scaled[:, i]

            inf_mask = np.isinf(X_scaled)
            if inf_mask.any():
                inf_count = np.sum(inf_mask)
                self.logger.warning(f"Знайдено {inf_count} нескінченних значень після нормалізації. Заміна на 0.")
                for i, col in enumerate(normalize_cols):
                    result[col] = result[col].replace([np.inf, -np.inf], 0)

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

        for i, df in enumerate(data_list):
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

        reference_df = data_list[reference_index]
        reference_freq = pd.infer_freq(reference_df.index)

        if not reference_freq:
            self.logger.warning("Не вдалося визначити частоту reference DataFrame. Спроба визначити вручну.")
            if len(reference_df.index) > 1:
                time_diff = reference_df.index.to_series().diff().dropna()
                reference_freq = time_diff.median()
                self.logger.info(f"Визначено медіанний інтервал: {reference_freq}")
            else:
                self.logger.error("Недостатньо точок для визначення частоти reference DataFrame")
                return data_list

        aligned_data_list = [reference_df]

        for i, df in enumerate(data_list):
            if i == reference_index:
                continue

            self.logger.info(f"Вирівнювання DataFrame {i} з reference DataFrame")

            if df.index.equals(reference_df.index):
                aligned_data_list.append(df)
                continue

            start_time = max(df.index.min(), reference_df.index.min())
            end_time = min(df.index.max(), reference_df.index.max())

            try:
                reference_subset = reference_df.loc[(reference_df.index >= start_time) &
                                                    (reference_df.index <= end_time)]

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
                aligned_data_list.append(df)  # Додаємо оригінал при помилці

        return aligned_data_list

    def validate_data_integrity(self, data: pd.DataFrame) -> Dict[str, List]:

        if data.empty:
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
            time_diff = data.index.to_series().diff()
            if len(time_diff) > 1:
                median_diff = time_diff.median()

                large_gaps = time_diff[time_diff > 2 * median_diff]
                if not large_gaps.empty:
                    gap_locations = large_gaps.index.tolist()
                    issues["time_gaps"] = gap_locations
                    self.logger.warning(f"Знайдено {len(gap_locations)} аномальних проміжків у часових мітках")

                duplicates = data.index.duplicated()
                if duplicates.any():
                    dup_indices = data.index[duplicates].tolist()
                    issues["duplicate_timestamps"] = dup_indices
                    self.logger.warning(f"Знайдено {len(dup_indices)} дублікатів часових міток")

        price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in data.columns]
        if len(price_cols) == 4:
            invalid_hl = data['high'] < data['low']
            if invalid_hl.any():
                invalid_hl_indices = data.index[invalid_hl].tolist()
                issues["high_lower_than_low"] = invalid_hl_indices
                self.logger.warning(f"Знайдено {len(invalid_hl_indices)} записів де high < low")

            for col in price_cols:
                negative_prices = data[col] < 0
                if negative_prices.any():
                    neg_price_indices = data.index[negative_prices].tolist()
                    issues[f"negative_{col}"] = neg_price_indices
                    self.logger.warning(f"Знайдено {len(neg_price_indices)} записів з від'ємними значеннями у {col}")

            for col in price_cols:
                pct_change = data[col].pct_change().abs()
                price_jumps = pct_change > 0.2
                if price_jumps.any():
                    jump_indices = data.index[price_jumps].tolist()
                    issues[f"price_jumps_{col}"] = jump_indices
                    self.logger.warning(f"Знайдено {len(jump_indices)} різких змін у колонці {col}")

        if 'volume' in data.columns:
            negative_volume = data['volume'] < 0
            if negative_volume.any():
                neg_vol_indices = data.index[negative_volume].tolist()
                issues["negative_volume"] = neg_vol_indices
                self.logger.warning(f"Знайдено {len(neg_vol_indices)} записів з від'ємним об'ємом")

            volume_zscore = np.abs((data['volume'] - data['volume'].mean()) / data['volume'].std())
            volume_anomalies = volume_zscore > 5  # Z-score > 5 вважаємо аномальним
            if volume_anomalies.any():
                vol_anomaly_indices = data.index[volume_anomalies].tolist()
                issues["volume_anomalies"] = vol_anomaly_indices
                self.logger.warning(f"Знайдено {len(vol_anomaly_indices)} записів з аномальним об'ємом")

        na_counts = data.isna().sum()
        cols_with_na = na_counts[na_counts > 0].index.tolist()
        if cols_with_na:
            issues["columns_with_na"] = {col: data.index[data[col].isna()].tolist() for col in cols_with_na}
            self.logger.warning(f"Знайдено відсутні значення у колонках: {cols_with_na}")

        # Перевірка на нескінченні значення
        inf_counts = np.isinf(data.select_dtypes(include=[np.number])).sum()
        cols_with_inf = inf_counts[inf_counts > 0].index.tolist()
        if cols_with_inf:
            issues["columns_with_inf"] = {col: data.index[np.isinf(data[col])].tolist() for col in cols_with_inf}
            self.logger.warning(f"Знайдено нескінченні значення у колонках: {cols_with_inf}")

        return issues

    def aggregate_volume_profile(self, data: pd.DataFrame, bins: int = 10,
                                 price_col: str = 'close', volume_col: str = 'volume',
                                 time_period: Optional[str] = None) -> pd.DataFrame:

        if data.empty:
            self.logger.warning("Отримано порожній DataFrame для профілю об'єму")
            return pd.DataFrame()

        if price_col not in data.columns:
            self.logger.error(f"Колонка {price_col} відсутня у DataFrame")
            return pd.DataFrame()

        if volume_col not in data.columns:
            self.logger.error(f"Колонка {volume_col} відсутня у DataFrame")
            return pd.DataFrame()

        self.logger.info(f"Створення профілю об'єму з {bins} ціновими рівнями")

        if time_period:
            if not isinstance(data.index, pd.DatetimeIndex):
                self.logger.warning("Індекс не є DatetimeIndex. Часовий профіль не може бути створений.")
                time_period = None

        if time_period:
            self.logger.info(f"Створення часового профілю об'єму з періодом {time_period}")
            period_groups = data.groupby(pd.Grouper(freq=time_period))

            result_dfs = []

            for period, group in period_groups:
                if group.empty:
                    continue

                period_profile = self._create_volume_profile(group, bins, price_col, volume_col)
                period_profile['period'] = period
                result_dfs.append(period_profile)

            if result_dfs:
                return pd.concat(result_dfs)
            else:
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

        bin_edges = np.linspace(price_min, price_max, bins + 1)
        bin_width = (price_max - price_min) / bins

        data['price_bin'] = pd.cut(data[price_col], bins=bin_edges, labels=False, include_lowest=True)

        volume_profile = data.groupby('price_bin').agg({
            volume_col: 'sum',
            price_col: ['count', 'min', 'max']
        })

        volume_profile.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in volume_profile.columns]

        volume_profile = volume_profile.rename(columns={
            f'{volume_col}_sum': 'volume',
            f'{price_col}_count': 'count',
            f'{price_col}_min': 'price_min',
            f'{price_col}_max': 'price_max'
        })

        total_volume = volume_profile['volume'].sum()
        volume_profile['volume_percent'] = (volume_profile['volume'] / total_volume * 100).round(2)

        volume_profile['price_mid'] = (volume_profile['price_min'] + volume_profile['price_max']) / 2

        volume_profile['bin_lower'] = [bin_edges[i] for i in volume_profile.index]
        volume_profile['bin_upper'] = [bin_edges[i + 1] for i in volume_profile.index]

        volume_profile = volume_profile.reset_index()

        volume_profile = volume_profile.sort_values('price_bin', ascending=False)

        if 'price_bin' in volume_profile.columns:
            volume_profile = volume_profile.drop('price_bin', axis=1)

        return volume_profile

    def add_time_features(self, data: pd.DataFrame, cyclical: bool = True,
                          add_sessions: bool = False, tz: str = 'UTC') -> pd.DataFrame:

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

    def create_cache_key(self, source: str, symbol: str, interval: str,
                         start_date: Union[str, datetime, None],
                         end_date: Union[str, datetime, None]) -> str:

        params={
            'symbol': symbol,
            'interval': interval,
            'start_date': start_date,
            'end_date': end_date
        }
        cache_dict ={
            'source': source,
            **params
        }
        for key, value in cache_dict.items():
            if isinstance(value, datetime):
                cache_dict[key] = value.strftime('%Y-%m-%d %H:%M:%S')
            elif value is None:
                cache_dict[key] = None

        json_string = json.dumps(cache_dict, sort_keys=True)
        return hashlib.md5(json_string.encode()).hexdigest()

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

        # Зберігаємо перші входження для кожної часової мітки
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

    def save_processed_data(self, data: pd.DataFrame, filename: str) -> str:

        if data.empty:
            self.logger.warning("Спроба зберегти порожній DataFrame")
            return ""

        # Визначення формату на основі розширення
        file_extension = filename.split('.')[-1].lower() if '.' in filename else 'parquet'

        # Створення директорії, якщо вона не існує
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            self.logger.info(f"Створено директорію: {directory}")

        try:
            if file_extension == 'csv':
                data.to_csv(filename)
                self.logger.info(f"Дані збережено у CSV форматі: {filename}")
            elif file_extension == 'parquet':
                data.to_parquet(filename)
                self.logger.info(f"Дані збережено у Parquet форматі: {filename}")
            elif file_extension in ['h5', 'hdf', 'hdf5']:
                data.to_hdf(filename, key='data', mode='w')
                self.logger.info(f"Дані збережено у HDF5 форматі: {filename}")
            elif file_extension == 'json':
                data.to_json(filename)
                self.logger.info(f"Дані збережено у JSON форматі: {filename}")
            elif file_extension == 'pickle' or file_extension == 'pkl':
                data.to_pickle(filename)
                self.logger.info(f"Дані збережено у Pickle форматі: {filename}")
            else:
                default_filename = f"{filename}.parquet"
                data.to_parquet(default_filename)
                self.logger.warning(
                    f"Невідомий формат файлу: {file_extension}. Збережено як Parquet: {default_filename}")
                return os.path.abspath(default_filename)

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

                # Спроба визначити та перетворити часовий індекс
                time_cols = [col for col in data.columns if
                             any(x in col.lower() for x in ['time', 'date', 'timestamp'])]
                if time_cols:
                    time_col = time_cols[0]
                    data[time_col] = pd.to_datetime(data[time_col])
                    data.set_index(time_col, inplace=True)
                    self.logger.info(f"Встановлено індекс за колонкою {time_col}")

            elif file_extension == 'parquet':
                data = pd.read_parquet(filename)
                self.logger.info(f"Дані завантажено з Parquet файлу: {filename}")

            elif file_extension in ['h5', 'hdf', 'hdf5']:
                data = pd.read_hdf(filename, key='data')
                self.logger.info(f"Дані завантажено з HDF5 файлу: {filename}")

            elif file_extension == 'json':
                data = pd.read_json(filename)
                self.logger.info(f"Дані завантажено з JSON файлу: {filename}")

            elif file_extension == 'pickle' or file_extension == 'pkl':
                data = pd.read_pickle(filename)
                self.logger.info(f"Дані завантажено з Pickle файлу: {filename}")

            else:
                self.logger.error(f"Невідомий формат файлу: {file_extension}")
                return pd.DataFrame()

            # Перевірка DatetimeIndex
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

        # Перевірка чи всі datasets мають вказаний merge_on
        all_have_merge_on = all(merge_on in df.columns or df.index.name == merge_on for df in datasets)

        if not all_have_merge_on:
            # Якщо merge_on - це "timestamp", але у деяких datasets індекс безіменний
            if merge_on == 'timestamp':
                self.logger.info("Перевірка, чи всі DataFrame мають DatetimeIndex")
                all_have_datetime_index = all(isinstance(df.index, pd.DatetimeIndex) for df in datasets)

                if all_have_datetime_index:
                    # Перейменування індексів
                    for i in range(len(datasets)):
                        if datasets[i].index.name is None:
                            datasets[i].index.name = 'timestamp'

                    all_have_merge_on = True

            if not all_have_merge_on:
                self.logger.error(f"Не всі набори даних містять '{merge_on}' для об'єднання")
                return pd.DataFrame()

        # Приведення до єдиного формату (індекс або колонка)
        datasets_copy = []
        for i, df in enumerate(datasets):
            df_copy = df.copy()

            # Якщо merge_on - колонка, а не індекс
            if merge_on in df_copy.columns:
                df_copy.set_index(merge_on, inplace=True)
                self.logger.info(f"DataFrame {i} перетворено: колонка '{merge_on}' стала індексом")
            elif df_copy.index.name != merge_on:
                df_copy.index.name = merge_on
                self.logger.info(f"DataFrame {i}: індекс перейменовано на '{merge_on}'")

            datasets_copy.append(df_copy)

        # Об'єднання наборів даних
        result = datasets_copy[0]
        total_columns = len(result.columns)

        for i, df in enumerate(datasets_copy[1:], 2):
            # Перейменування колонок у випадку конфлікту імен
            rename_dict = {}
            for col in df.columns:
                if col in result.columns:
                    # Додаємо суфікс з номером DataFrame тільки для дублікатів колонок
                    rename_dict[col] = f"{col}_{i}"

            if rename_dict:
                self.logger.info(f"Перейменування колонок у DataFrame {i}: {rename_dict}")
                df = df.rename(columns=rename_dict)

            # Об'єднання
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
            # Стандартний конвеєр обробки
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

                # Виклик методу з параметрами та вхідними даними
                if step_name == 'normalize_data':
                    # normalize_data повертає кортеж (DataFrame, scaler_meta)
                    result, _ = method(result, **step_params)
                elif step_name == 'detect_outliers':
                    # detect_outliers повертає кортеж (DataFrame з outlier flags, список індексів outliers)
                    outliers_df, _ = method(result, **step_params)
                    # Тут ми не змінюємо result, але можемо зробити щось із outliers_df
                    self.logger.info(f"Виявлено аномалії, але дані не змінено")
                else:
                    # Інші методи повертають оброблений DataFrame
                    result = method(result, **step_params)

                # Логування результатів кроку
                self.logger.info(
                    f"Крок {step_idx}: '{step_name}' завершено. Результат: {len(result)} рядків, {len(result.columns)} колонок")

            except Exception as e:
                self.logger.error(f"Помилка на кроці {step_idx}: '{step_name}': {str(e)}")
                # Продовжуємо з наступним кроком

        self.logger.info(
            f"Конвеєр обробки даних завершено. Початково: {len(data)} рядків, {len(data.columns)} колонок. "
            f"Результат: {len(result)} рядків, {len(result.columns)} колонок.")

        return result

    def _load_cache_index(self):

        if not self.cache_dir:
            return

        cache_index_path = os.path.join(self.cache_dir, "cache_index.json")
        if os.path.exists(cache_index_path):
            try:
                with open(cache_index_path, 'r') as f:
                    self.cache_index = json.load(f)
                self.logger.info(f"Завантажено індекс кешу: {len(self.cache_index)} записів")
            except Exception as e:
                self.logger.error(f"Помилка при завантаженні індексу кешу: {str(e)}")
                self.cache_index = {}
        else:
            self.logger.info("Індекс кешу не знайдено. Створено порожній індекс.")
            self.cache_index = {}

    def _save_cache_index(self):

        if not self.cache_dir:
            return

        cache_index_path = os.path.join(self.cache_dir, "cache_index.json")
        try:
            with open(cache_index_path, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
            self.logger.info(f"Збережено індекс кешу: {len(self.cache_index)} записів")
        except Exception as e:
            self.logger.error(f"Помилка при збереженні індексу кешу: {str(e)}")


def main():
    from market_data_processor import MarketDataProcessor

    data_source = {
        'csv': {
            'BTC': {
                '1d': './data/crypto_data/BTCUSDT_1d.csv',
                '1h': './data/crypto_data/BTCUSDT_1h.csv',
                '4h': './data/crypto_data/BTCUSDT_4h.csv'
            },
            'ETH': {
                '1d': './data/crypto_data/ETHUSDT_1d.csv',
                '1h': './data/crypto_data/ETHUSDT_1h.csv',
                '4h': './data/crypto_data/ETHUSDT_4h.csv'
            },
            'SOL': {
                '1d': './data/crypto_data/SOLUSDT_1d.csv',
                '1h': './data/crypto_data/SOLUSDT_1h.csv',
                '4h': './data/crypto_data/SOLUSDT_4h.csv'
            }
        }
    }

    symbols = ['BTC', 'ETH', 'SOL']
    intervals = ['1d', '1h', '4h']

    processor = MarketDataProcessor(cache_dir="./cache")

    for symbol in symbols:
        for interval in intervals:
            file_path = data_source['csv'][symbol].get(interval)
            if not file_path:
                continue

            print(f"\n Завантаження {symbol} {interval} з {file_path}")
            try:
                data = processor.load_data(
                    data_source='csv',
                    symbol=symbol,
                    interval=interval,
                    file_path=file_path,
                    data_type='candles'
                )

                print(f"✔️ Дані завантажено: {len(data)} рядків")

                data = processor.preprocess_pipeline(data)

                if interval != '1d':
                    data = processor.resample_data(data, target_interval='1d')

                data = processor.add_time_features(data)

                save_path = f"./processed/{symbol}_{interval}_processed.parquet"
                processor.save_processed_data(data, save_path)
                print(f" Дані збережено: {save_path}")

            except Exception as e:
                print(f" Помилка обробки {symbol} {interval}: {e}")


if __name__ == "__main__":
    main()
