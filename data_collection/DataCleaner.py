from datetime import datetime
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import pandas as pd
import pytz
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from data_collection.AnomalyDetector import AnomalyDetector
from data_collection.DataResampler import DataResampler
from utils.config import BINANCE_API_KEY, BINANCE_API_SECRET


class DataCleaner:
    def __init__(self, logger):
        self.logger = logger
        self.anomaly_detector = AnomalyDetector(logger=self.logger)

    def clean_data(self, data: pd.DataFrame, remove_outliers: bool = True,
                   fill_missing: bool = True, normalize: bool = True,
                   norm_method: str = 'z-score', resample: bool = True,
                   target_interval: str = None, add_time_features: bool = True,
                   cyclical: bool = True, add_sessions: bool = False) -> pd.DataFrame:

        if data is None or data.empty:
            self.logger.warning("Отримано порожній DataFrame для очищення")
            return data if data is not None else pd.DataFrame()

        self.logger.info(f"Початок очищення даних: {data.shape[0]} рядків, {data.shape[1]} стовпців")

        # Виконуємо перевірку цілісності даних перед очищенням
        integrity_issues = AnomalyDetector.validate_data_integrity(data)
        if integrity_issues:
            issue_count = sum(len(issues) if isinstance(issues, list) or isinstance(issues, dict) else 1
                              for issues in integrity_issues.values())
            self.logger.warning(f"Знайдено {issue_count} проблем з цілісністю даних")

        result = data.copy()

        if not isinstance(result.index, pd.DatetimeIndex):
            try:
                time_cols = [col for col in result.columns if
                             any(x in col.lower() for x in ['time', 'date', 'timestamp'])]
                if time_cols:
                    time_col = time_cols[0]
                    self.logger.info(f"Конвертування колонки {time_col} в індекс часу")
                    result[time_col] = pd.to_datetime(result[time_col], errors='coerce')
                    result.set_index(time_col, inplace=True)
                else:
                    self.logger.warning("Не знайдено колонку з часом, індекс залишається незмінним")
            except Exception as e:
                self.logger.error(f"Помилка при конвертуванні індексу: {str(e)}")

        if isinstance(result.index, pd.DatetimeIndex) and result.index.duplicated().any():
            dup_count = result.index.duplicated().sum()
            self.logger.info(f"Знайдено {dup_count} дублікатів індексу, видалення...")
            result = result[~result.index.duplicated(keep='first')]

        result = result.sort_index()

        # Додаємо часові ознаки, якщо потрібно
        if add_time_features and isinstance(result.index, pd.DatetimeIndex):
            self.logger.info("Додавання часових ознак...")
            result = self.add_time_features(
                data=result,
                cyclical=cyclical,
                add_sessions=add_sessions
            )

        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in result.columns:
                # Перевіряємо, чи не є колонка вже числового типу
                if not pd.api.types.is_numeric_dtype(result[col]):
                    self.logger.info(f"Конвертування колонки {col} в числовий тип")
                    result[col] = pd.to_numeric(result[col], errors='coerce')

        if remove_outliers:
            self.logger.info("Видалення аномальних значень...")
            price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in result.columns]
            for col in price_cols:
                # Перевірка на порожній DataFrame або серію
                if col not in result.columns or result[col].empty or result[col].isna().all():
                    continue

                Q1 = result[col].quantile(0.25)
                Q3 = result[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR

                outliers = (result[col] < lower_bound) | (result[col] > upper_bound)
                if outliers.any():
                    outlier_count = outliers.sum()
                    outlier_indexes = result.index[outliers].tolist()
                    self.logger.info(f"Знайдено {outlier_count} аномалій в колонці {col}")
                    self.logger.debug(
                        f"Індекси перших 10 аномалій: {outlier_indexes[:10]}{'...' if len(outlier_indexes) > 10 else ''}")
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

            result = result.ffill().bfill()

        price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in result.columns]
        if len(price_cols) == 4:
            invalid_hl = result['high'] < result['low']
            if invalid_hl.any():
                invalid_count = invalid_hl.sum()
                invalid_indexes = result.index[invalid_hl].tolist()
                self.logger.warning(f"Знайдено {invalid_count} рядків, де high < low")
                self.logger.debug(
                    f"Індекси проблемних рядків: {invalid_indexes[:10]}{'...' if len(invalid_indexes) > 10 else ''}")

                # Swap high and low values
                temp = result.loc[invalid_hl, 'high'].copy()
                result.loc[invalid_hl, 'high'] = result.loc[invalid_hl, 'low']
                result.loc[invalid_hl, 'low'] = temp

        # Виконуємо ресемплінг даних, якщо потрібно
        if resample and target_interval and isinstance(result.index, pd.DatetimeIndex):
            try:
                self.logger.info(f"Виконання ресемплінгу даних до інтервалу {target_interval}...")
                result = DataResampler.resample_data(result, target_interval)
            except Exception as e:
                self.logger.error(f"Помилка при ресемплінгу даних: {str(e)}")

        # Додаємо нормалізацію даних, якщо потрібно
        if normalize:
            self.logger.info(f"Виконання нормалізації даних методом {norm_method}...")
            price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in result.columns]

            # Нормалізуємо цінові колонки
            if price_cols:
                result, price_scaler = self.normalize_data(
                    data=result,
                    method=norm_method,
                    columns=price_cols
                )

                if price_scaler is None:
                    self.logger.warning("Не вдалося нормалізувати цінові колонки")

            # Окремо нормалізуємо об'єм, якщо він присутній
            if 'volume' in result.columns:
                result, volume_scaler = self.normalize_data(
                    data=result,
                    method='min-max',  # Для об'єму краще використовувати min-max нормалізацію
                    columns=['volume']
                )

                if volume_scaler is None:
                    self.logger.warning("Не вдалося нормалізувати колонку об'єму")

        # Перевіряємо цілісність даних після очищення
        clean_integrity_issues = AnomalyDetector.validate_data_integrity(result)
        if clean_integrity_issues:
            issue_count = sum(len(issues) if isinstance(issues, list) or isinstance(issues, dict) else 1
                              for issues in clean_integrity_issues.values())
            self.logger.info(f"Після очищення залишилось {issue_count} проблем з цілісністю даних")

        self.logger.info(f"Очищення даних завершено: {result.shape[0]} рядків, {result.shape[1]} стовпців")
        return result

    def handle_missing_values(self, data: pd.DataFrame, method: str = 'interpolate',
                              symbol: str = None, timeframe: str = None,
                              fetch_missing: bool = True) -> pd.DataFrame:
        if data is None or data.empty:
            self.logger.warning("Отримано порожній DataFrame для обробки відсутніх значень")
            return data if data is not None else pd.DataFrame()

        integrity_issues = self.anomaly_detector.validate_data_integrity(data)
        if integrity_issues:
            issue_count = sum(len(issues) if isinstance(issues, list) or isinstance(issues, dict) else 1
                              for issues in integrity_issues.values())
            self.logger.warning(f"Перед обробкою відсутніх значень знайдено {issue_count} проблем з цілісністю даних")
            # Перевіряємо, чи є проблеми з відсутніми значеннями серед виявлених проблем
            if "columns_with_na" in integrity_issues:
                na_cols = list(integrity_issues["columns_with_na"].keys())
                self.logger.info(f"Виявлені колонки з відсутніми значеннями: {na_cols}")

        result = data.copy()
        missing_values = result.isna().sum()
        total_missing = missing_values.sum()

        if total_missing == 0:
            self.logger.info("Відсутні значення не знайдено")
            return result

        self.logger.info(
            f"Знайдено {total_missing} відсутніх значень у {len(missing_values[missing_values > 0])} колонках")

        #  Підтягування з Binance
        if isinstance(result.index, pd.DatetimeIndex) and fetch_missing and symbol and timeframe:
            time_diff = result.index.to_series().diff()
            expected_diff = time_diff.dropna().median() if len(time_diff) > 5 else None

            if expected_diff:
                missing_periods = self._detect_missing_periods(result, expected_diff)
                if missing_periods:
                    self.logger.info(f"Знайдено {len(missing_periods)} прогалин. Підтягуємо з Binance...")
                    filled = self._fetch_missing_data_from_binance(result, missing_periods, symbol, timeframe)
                    if not filled.empty:
                        result = pd.concat([result, filled])
                        result = result[~result.index.duplicated(keep='last')].sort_index()

        filled_values = 0
        numeric_cols = result.select_dtypes(include=[np.number]).columns

        if method == 'interpolate':
            self.logger.info("Застосування методу лінійної інтерполяції")
            price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in result.columns]
            other_cols = [col for col in numeric_cols if col not in price_cols]
            before_fill = result.count().sum()

            if price_cols:
                if isinstance(result.index, pd.DatetimeIndex):
                    result[price_cols] = result[price_cols].interpolate(method='time')
                else:
                    result[price_cols] = result[price_cols].interpolate(method='linear')

            if other_cols:
                result[other_cols] = result[other_cols].interpolate().ffill().bfill()

            result = result.ffill().bfill()
            filled_values = result.count().sum() - before_fill

        elif method == 'ffill':
            self.logger.info("Застосування методу forward/backward fill")
            before_fill = result.count().sum()
            result = result.ffill().bfill()
            filled_values = result.count().sum() - before_fill

        elif method == 'mean':
            self.logger.info("Застосування методу заповнення середнім значенням")
            for col in numeric_cols:
                if col in result.columns and missing_values.get(col, 0) > 0:
                    if result[col].dropna().empty:
                        self.logger.warning(f"Колонка {col} не містить значень для обчислення середнього")
                        continue
                    col_mean = result[col].mean()
                    if pd.notna(col_mean):
                        before = result[col].isna().sum()
                        result[col] = result[col].fillna(col_mean)
                        filled_values += before - result[col].isna().sum()

        elif method == 'median':
            self.logger.info("Застосування методу заповнення медіанним значенням")
            for col in numeric_cols:
                if col in result.columns and missing_values.get(col, 0) > 0:
                    if result[col].dropna().empty:
                        self.logger.warning(f"Колонка {col} не містить значень для обчислення медіани")
                        continue
                    col_median = result[col].median()
                    if pd.notna(col_median):
                        before = result[col].isna().sum()
                        result[col] = result[col].fillna(col_median)
                        filled_values += before - result[col].isna().sum()

        else:
            self.logger.warning(f"Метод заповнення '{method}' не підтримується")

        remaining_missing = result.isna().sum().sum()
        if remaining_missing > 0:
            self.logger.warning(f"Залишилося {remaining_missing} незаповнених значень після обробки")

        self.logger.info(f"Заповнено {filled_values} відсутніх значень методом '{method}'")

        clean_integrity_issues = AnomalyDetector.validate_data_integrity(result)
        if clean_integrity_issues:
            issue_count = sum(len(issues) if isinstance(issues, list) or isinstance(issues, dict) else 1
                              for issues in clean_integrity_issues.values())
            self.logger.info(f"Після обробки відсутніх значень залишилось {issue_count} проблем з цілісністю даних")

            # Перевіряємо, чи залишились проблеми з відсутніми значеннями
            if "columns_with_na" in clean_integrity_issues:
                na_cols = list(clean_integrity_issues["columns_with_na"].keys())
                self.logger.warning(f"Після обробки все ще є колонки з відсутніми значеннями: {na_cols}")
        else:
            self.logger.info("Після обробки відсутніх значень проблем з цілісністю даних не виявлено")

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
        if data is None or data.empty or not missing_periods:
            self.logger.warning("Отримано порожній DataFrame або немає missing_periods для заповнення даними")
            return pd.DataFrame()

        if not symbol or not interval:
            self.logger.error("Невалідний symbol або interval")
            return pd.DataFrame()

        try:
            from binance.client import Client
            api_key = BINANCE_API_KEY
            api_secret = BINANCE_API_SECRET

            if not api_key or not api_secret:
                self.logger.error("Не знайдено ключі API Binance")
                return pd.DataFrame()

            client = Client(api_key, api_secret)

            valid_intervals = ['1m', '1h', '4h', '1d']
            if interval not in valid_intervals:
                self.logger.error(f"Невалідний інтервал: {interval}")
                return pd.DataFrame()

            new_data_frames = []

            for start_time, end_time in missing_periods:
                try:
                    self.logger.info(f" Отримання даних з Binance: {symbol}, {interval}, {start_time} - {end_time}")
                    start_ms = int(start_time.timestamp() * 1000)
                    end_ms = int(end_time.timestamp() * 1000)
                    self.logger.info(f"Запит до Binance: {start_time} -> {start_ms} мс, {end_time} -> {end_ms} мс")

                    klines = client.get_historical_klines(
                        symbol=symbol,
                        interval=interval,
                        start_str=start_ms,
                        end_str=end_ms
                    )

                    if not klines:
                        self.logger.warning(f" Порожній результат з Binance: {start_time} - {end_time}")
                        continue

                    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                               'close_time', 'quote_asset_volume', 'number_of_trades',
                               'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore']
                    binance_df = pd.DataFrame(klines, columns=columns[:min(len(columns), len(klines[0]) if klines else 0)])

                    binance_df['timestamp'] = pd.to_datetime(binance_df['timestamp'], unit='ms')
                    binance_df.set_index('timestamp', inplace=True)

                    # Конвертація числових значень
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if col in binance_df.columns:
                            binance_df[col] = pd.to_numeric(binance_df[col], errors='coerce')

                    binance_df['is_closed'] = True

                    # Вибираємо лише ті колонки, які є в обох DataFrame
                    common_cols = data.columns.intersection(binance_df.columns)
                    if common_cols.empty:
                        self.logger.warning("⚠️ Немає спільних колонок для об'єднання")
                        continue

                    new_data = binance_df[common_cols]
                    new_data_frames.append(new_data)

                    self.logger.info(f"✅ Отримано {len(new_data)} нових записів")

                except Exception as e:
                    self.logger.error(f" Помилка при запиті Binance: {e}")

            if not new_data_frames:
                return pd.DataFrame()

            combined_new = pd.concat(new_data_frames)
            self.logger.info(f" Загалом додано {len(combined_new)} нових рядків після об'єднання")

            return combined_new

        except ImportError:
            self.logger.error(" Модуль binance не встановлено.")
            return pd.DataFrame()

    def normalize_data(self, data: pd.DataFrame, method: str = 'z-score',
                       columns: List[str] = None, exclude_columns: List[str] = None) -> Tuple[pd.DataFrame, Optional[Dict]]:

        if data is None or data.empty:
            self.logger.warning("Отримано порожній DataFrame для нормалізації")
            return data if data is not None else pd.DataFrame(), None

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


    def add_time_features(self, data: pd.DataFrame, cyclical: bool = True,
                          add_sessions: bool = False, tz: str = 'Europe/Kiev') -> pd.DataFrame:

        if data is None or data.empty:
            self.logger.warning("Отримано порожній DataFrame для додавання часових ознак")
            return data if data is not None else pd.DataFrame()

        result = data.copy()

        # Перевірка та конвертація індексу в DatetimeIndex
        if not isinstance(result.index, pd.DatetimeIndex):
            self.logger.warning("Індекс не є DatetimeIndex. Спроба конвертувати.")
            try:
                time_cols = [col for col in result.columns if
                             any(x in col.lower() for x in ['time', 'date', 'timestamp'])]
                if time_cols:
                    time_col = time_cols[0]
                    result[time_col] = pd.to_datetime(result[time_col], errors='coerce')
                    result.set_index(time_col, inplace=True)

                    # Перевірка на NaT після конвертації
                    if result.index.isna().any():
                        nat_count = result.index.isna().sum()
                        self.logger.warning(f"Знайдено {nat_count} NaT значень в індексі після конвертації")
                        result = result[~result.index.isna()]

                else:
                    self.logger.error("Неможливо додати часові ознаки: не знайдено часову колонку")
                    return data
            except Exception as e:
                self.logger.error(f"Помилка при конвертації індексу: {str(e)}")
                return data

        self.logger.info("Додавання часових ознак")

        # Коректна обробка часових поясів
        try:
            if result.index.tz is None:
                self.logger.info(f"Встановлення часового поясу {tz}")
                try:
                    result.index = result.index.tz_localize(tz)
                except pytz.exceptions.NonExistentTimeError:
                    # Обробка випадків при переході на літній час
                    self.logger.warning(
                        "Виявлено час, що не існує при переході на літній час. Використання нестрогої локалізації.")
                    result.index = result.index.tz_localize(tz, nonexistent='shift_forward')
                except pytz.exceptions.AmbiguousTimeError:
                    # Обробка випадків при переході на зимовий час
                    self.logger.warning(
                        "Виявлено неоднозначний час при переході на зимовий час. Використання першого варіанту.")
                    result.index = result.index.tz_localize(tz, ambiguous='infer')
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
        except Exception as e:
            self.logger.error(f"Загальна помилка при обробці часового поясу: {str(e)}")

        # Базові часові ознаки
        result['hour'] = result.index.hour
        result['day'] = result.index.day
        result['weekday'] = result.index.weekday

        # Безпечне отримання номера тижня
        try:
            if hasattr(result.index, 'isocalendar') and callable(result.index.isocalendar):
                isocal = result.index.isocalendar()
                if isinstance(isocal, pd.DataFrame):  # pandas >= 1.1.0
                    result['week'] = isocal['week']
                else:  # старіші версії pandas
                    result['week'] = [x[1] for x in isocal]
            else:
                # Альтернативний метод для старіших версій
                result['week'] = result.index.to_series().apply(lambda x: x.isocalendar()[1])
        except Exception as e:
            self.logger.warning(f"Помилка при отриманні номера тижня: {str(e)}. Використовуємо альтернативний метод.")
            # Запасний варіант
            result['week'] = result.index.week if hasattr(result.index, 'week') else result.index.to_series().dt.week

        result['month'] = result.index.month
        result['quarter'] = result.index.quarter
        result['year'] = result.index.year
        result['dayofyear'] = result.index.dayofyear

        # Бінарні ознаки
        result['is_weekend'] = result['weekday'].isin([5, 6]).astype(int)
        result['is_month_start'] = result.index.is_month_start.astype(int)
        result['is_month_end'] = result.index.is_month_end.astype(int)
        result['is_quarter_start'] = result.index.is_quarter_start.astype(int)
        result['is_quarter_end'] = result.index.is_quarter_end.astype(int)
        result['is_year_start'] = result.index.is_year_start.astype(int)
        result['is_year_end'] = result.index.is_year_end.astype(int)

        # Циклічні ознаки
        if cyclical:
            self.logger.info("Додавання циклічних ознак")

            result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
            result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)

            # Безпечне обчислення кількості днів у місяці
            try:
                days_in_month = result.index.days_in_month
            except AttributeError:
                self.logger.warning("Атрибут days_in_month не знайдений. Використовуємо стандартне значення 30.")
                days_in_month = pd.Series([30] * len(result), index=result.index)

            # Перевірка на нульові значення у знаменнику
            days_in_month = days_in_month.replace(0, 30)

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

        # Торгові сесії
        if add_sessions:
            self.logger.info("Додавання індикаторів торгових сесій")

            # Азійська сесія: 00:00-09:00
            result['asian_session'] = ((result['hour'] >= 0) & (result['hour'] < 9)).astype(int)

            # Європейська сесія: 08:00-17:00
            result['european_session'] = ((result['hour'] >= 8) & (result['hour'] < 17)).astype(int)

            # Американська сесія: 13:00-22:00
            result['american_session'] = ((result['hour'] >= 13) & (result['hour'] < 22)).astype(int)

            # Перекриття сесій
            result['asia_europe_overlap'] = ((result['hour'] >= 8) & (result['hour'] < 9)).astype(int)
            result['europe_america_overlap'] = ((result['hour'] >= 13) & (result['hour'] < 17)).astype(int)

            # Виправлено: неактивні години (22:00-00:00)
            result['inactive_hours'] = (result['hour'] >= 22).astype(int)

        self.logger.info(f"Успішно додано {len(result.columns) - len(data.columns)} часових ознак")
        return result

    def remove_duplicate_timestamps(self, data: pd.DataFrame) -> pd.DataFrame:

        if data is None or data.empty:
            self.logger.warning("Отримано порожній DataFrame для обробки дублікатів")
            return data if data is not None else pd.DataFrame()

        original_shape = data.shape

        if not isinstance(data.index, pd.DatetimeIndex):
            self.logger.warning("Індекс не є DatetimeIndex. Спроба конвертувати.")
            try:
                time_cols = [col for col in data.columns if
                             any(x in col.lower() for x in ['time', 'date', 'timestamp'])]
                if time_cols:
                    time_col = time_cols[0]
                    data[time_col] = pd.to_datetime(data[time_col], errors='coerce')

                    # Перевірка на NaT
                    if data[time_col].isna().any():
                        self.logger.warning(f"Знайдено NaT значення в колонці {time_col}")
                        data = data[~data[time_col].isna()]

                    data.set_index(time_col, inplace=True)
                else:
                    self.logger.error("Неможливо виявити дублікати: не знайдено часову колонку")
                    return data
            except Exception as e:
                self.logger.error(f"Помилка при конвертації індексу: {str(e)}")
                return data

        # Перевірка на наявність дублікатів
        duplicates = data.index.duplicated()
        duplicates_count = duplicates.sum()

        if duplicates_count == 0:
            self.logger.info("Дублікати часових міток не знайдено")
            return data

        self.logger.info(f"Знайдено {duplicates_count} дублікатів часових міток")

        # Зберігаємо унікальні індекси, видаляємо дублікати і сортуємо
        result = data[~duplicates].sort_index()

        # Логування результатів
        self.logger.info(
            f"Видалено {duplicates_count} дублікатів. Вхідний розмір {original_shape}, залишилось {result.shape} записів.")

        return result

    def filter_by_time_range(self, data: pd.DataFrame,
                             start_time: Optional[Union[str, datetime]] = None,
                             end_time: Optional[Union[str, datetime]] = None) -> pd.DataFrame:

        if data is None or data.empty:
            self.logger.warning("Отримано порожній DataFrame для фільтрації")
            return data if data is not None else pd.DataFrame()

        if not isinstance(data.index, pd.DatetimeIndex):
            self.logger.warning("Індекс не є DatetimeIndex. Спроба конвертувати.")
            try:
                time_cols = [col for col in data.columns if
                             any(x in col.lower() for x in ['time', 'date', 'timestamp'])]
                if time_cols:
                    time_col = time_cols[0]
                    data[time_col] = pd.to_datetime(data[time_col], errors='coerce')

                    # Перевірка на NaT
                    if data[time_col].isna().any():
                        nat_count = data[time_col].isna().sum()
                        self.logger.warning(f"Знайдено {nat_count} NaT значень в колонці {time_col}")
                        data = data[~data[time_col].isna()]

                    data.set_index(time_col, inplace=True)
                else:
                    self.logger.error("Неможливо фільтрувати за часом: не знайдено часову колонку")
                    return data
            except Exception as e:
                self.logger.error(f"Помилка при конвертації індексу: {str(e)}")
                return data

        result = data.copy()
        initial_count = len(result)

        start_dt = None
        end_dt = None

        # Обробка початкового часу
        if start_time is not None:
            try:
                start_dt = pd.to_datetime(start_time)

                # Коректна обробка часових поясів
                if result.index.tz is not None and start_dt.tz is None:
                    self.logger.info(f"Додавання часового поясу {result.index.tz} до початкового часу")
                    start_dt = start_dt.tz_localize(result.index.tz)
                elif result.index.tz is None and start_dt.tz is not None:
                    self.logger.info(f"Видалення часового поясу з початкового часу")
                    start_dt = start_dt.tz_localize(None)
                elif result.index.tz is not None and start_dt.tz is not None and result.index.tz != start_dt.tz:
                    self.logger.info(f"Конвертація часового поясу з {start_dt.tz} в {result.index.tz}")
                    start_dt = start_dt.tz_convert(result.index.tz)

                result = result[result.index >= start_dt]
                self.logger.info(f"Фільтрація за початковим часом: {start_dt}")
            except Exception as e:
                self.logger.error(f"Помилка при конвертації початкового часу: {str(e)}")

        # Обробка кінцевого часу
        if end_time is not None:
            try:
                end_dt = pd.to_datetime(end_time)

                # Коректна обробка часових поясів
                if result.index.tz is not None and end_dt.tz is None:
                    self.logger.info(f"Додавання часового поясу {result.index.tz} до кінцевого часу")
                    end_dt = end_dt.tz_localize(result.index.tz)
                elif result.index.tz is None and end_dt.tz is not None:
                    self.logger.info(f"Видалення часового поясу з кінцевого часу")
                    end_dt = end_dt.tz_localize(None)
                elif result.index.tz is not None and end_dt.tz is not None and result.index.tz != end_dt.tz:
                    self.logger.info(f"Конвертація часового поясу з {end_dt.tz} в {result.index.tz}")
                    end_dt = end_dt.tz_convert(result.index.tz)

                result = result[result.index <= end_dt]
                self.logger.info(f"Фільтрація за кінцевим часом: {end_dt}")
            except Exception as e:
                self.logger.error(f"Помилка при конвертації кінцевого часу: {str(e)}")

        final_count = len(result)

        # Перевірка логічної відповідності початкової та кінцевої дати
        if start_dt is not None and end_dt is not None:
            if start_dt > end_dt:
                self.logger.warning(
                    f"Початковий час ({start_dt}) пізніше кінцевого часу ({end_dt}). Результат може бути порожнім.")
                if len(result) == 0:
                    self.logger.warning("Після фільтрації отримано порожній DataFrame")

        self.logger.info(f"Відфільтровано {initial_count - final_count} записів. Залишилось {final_count} записів.")
        return result