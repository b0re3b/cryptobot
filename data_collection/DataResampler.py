import json

import numpy as np
import pandas as pd
import dask.dataframe as dd
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List
from functools import lru_cache

class DataResampler:
    def __init__(self, logger, chunk_size=500_000, scaling_sample_size= 1_000_000):
        self.logger = logger
        self.scalers = {}
        self.original_data_map = {}
        self.chunk_size = chunk_size
        self.scaling_sample_size = scaling_sample_size
        self.find_column = lru_cache(maxsize=128)(self._find_column_original)
        self.cache = {}
    def _find_column_original(self, df, column_name):
        """Знаходить колонку незалежно від регістру з обробкою конфліктів"""
        exact_match = [col for col in df.columns if col == column_name]
        if exact_match:
            return exact_match[0]

        case_insensitive_matches = [col for col in df.columns if col.lower() == column_name.lower()]
        if case_insensitive_matches:
            if len(case_insensitive_matches) > 1:
                self.logger.warning(
                    f"Знайдено кілька варіантів для колонки '{column_name}': {case_insensitive_matches}. Використовуємо перший.")
            return case_insensitive_matches[0]

        return None

    def detect_interval(self, data: pd.DataFrame) -> str:

        if not isinstance(data.index, pd.DatetimeIndex):
            self.logger.error("DataFrame повинен мати DatetimeIndex для визначення інтервалу")
            return None

        if len(data) < 2:
            self.logger.error("Потрібно щонайменше 2 точки даних для визначення інтервалу")
            return None

        # Обчислюємо різницю в часі між сусідніми індексами
        time_diffs = data.index.to_series().diff().dropna()

        # Визначаємо найбільш поширену різницю (моду)
        from collections import Counter
        counter = Counter(time_diffs)
        most_common_diff = counter.most_common(1)[0][0]

        # Перетворюємо в загальні одиниці виміру (секунди)
        diff_seconds = most_common_diff.total_seconds()

        # Визначаємо інтервал на основі кількості секунд
        if diff_seconds < 60:
            interval = f"{int(diff_seconds)}s"
        elif diff_seconds < 3600:
            minutes = int(diff_seconds / 60)
            interval = f"{minutes}m"
        elif diff_seconds < 86400:
            hours = int(diff_seconds / 3600)
            interval = f"{hours}h"
        elif diff_seconds < 604800:
            days = int(diff_seconds / 86400)
            interval = f"{days}d"
        elif diff_seconds < 2592000:
            weeks = int(diff_seconds / 604800)
            interval = f"{weeks}w"
        else:
            months = max(1, int(diff_seconds / 2592000))
            interval = f"{months}M"

        self.logger.info(f"Визначений інтервал даних: {interval}")

        # Зберігаємо результат для подальшого використання
        self.original_data_map['detected_interval'] = interval
        return interval

    def auto_resample(self, data: pd.DataFrame, target_interval: str = None,
                      scaling_factor: int = None) -> pd.DataFrame:

        if not isinstance(data.index, pd.DatetimeIndex):
            self.logger.error("DataFrame повинен мати DatetimeIndex для ресемплінгу")
            return data

        # Визначаємо поточний інтервал
        current_interval = self.detect_interval(data)
        if not current_interval:
            self.logger.error("Не вдалося визначити поточний інтервал даних")
            return data

        # Якщо цільовий інтервал не задано, використовуємо scaling_factor
        if target_interval is None:
            if scaling_factor is None:
                # За замовчуванням множимо інтервал на 4
                scaling_factor = 4
                self.logger.info(f"Використовується масштабуючий коефіцієнт за замовчуванням: {scaling_factor}")

            import re
            match = re.match(r'(\d+)([smhdwM])', current_interval)
            if not match:
                self.logger.error(f"Неправильний формат поточного інтервалу: {current_interval}")
                return data

            number, unit = match.groups()
            number = int(number) * scaling_factor
            target_interval = f"{number}{unit}"
            self.logger.info(f"Обчислений цільовий інтервал: {target_interval}")

        # Здійснюємо ресемплінг
        self.logger.info(f"Виконується зміна інтервалу: {current_interval} -> {target_interval}")
        resampled_data = self.resample_data(data, target_interval)

        # Зберігаємо інформацію про трансформацію
        self.original_data_map['auto_resample_info'] = {
            'original_interval': current_interval,
            'target_interval': target_interval,
            'scaling_factor': scaling_factor,
            'original_shape': data.shape,
            'resampled_shape': resampled_data.shape
        }

        return resampled_data

    def suggest_intervals(self, data: pd.DataFrame, max_suggestions: int = 5) -> list:

        current_interval = self.detect_interval(data)
        if not current_interval:
            return []

        import re
        match = re.match(r'(\d+)([smhdwM])', current_interval)
        if not match:
            return []

        number, unit = match.groups()
        number = int(number)

        # Стандартні множники для різних одиниць
        standard_multipliers = {
            's': [5, 10, 15, 30, 60],
            'm': [5, 10, 15, 30, 60],
            'h': [2, 3, 4, 6, 8, 12, 24],
            'd': [2, 3, 5, 7, 10, 14, 30],
            'w': [2, 3, 4],
            'M': [2, 3, 6, 12]
        }

        multipliers = standard_multipliers.get(unit, [2, 3, 4, 5])

        # Формуємо список рекомендацій
        suggestions = []
        for multiplier in multipliers:
            new_value = number * multiplier

            if unit == 's' and new_value >= 60:
                suggestions.append(f"{new_value // 60}m")
            elif unit == 'm' and new_value >= 60:
                suggestions.append(f"{new_value // 60}h")
            elif unit == 'h' and new_value >= 24:
                suggestions.append(f"{new_value // 24}d")
            elif unit == 'd' and new_value >= 7 and new_value % 7 == 0:
                suggestions.append(f"{new_value // 7}w")
            elif unit == 'd' and new_value >= 30:
                suggestions.append(f"{new_value // 30}M")
            else:
                suggestions.append(f"{new_value}{unit}")

        return suggestions[:max_suggestions]

    def _optimize_aggregation_dict(self, data: pd.DataFrame, store_column_map: bool = False) -> Dict:
        """
        Формує словник агрегацій на основі типу колонок і їх назв (уніфікована логіка).
        """
        agg_dict = {}
        columns_lower_map = {col.lower(): col for col in data.columns}
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        # Основні фінансові колонки
        standard_aggs = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'trades': 'sum',
            'taker_buy_volume': 'sum',
            'taker_sell_volume': 'sum',
            'taker_buy_base_volume': 'sum',
            'taker_buy_quote_volume': 'sum',
            'quote_volume': 'sum',
            'quote_asset_volume': 'sum',
            'number_of_trades': 'sum',
            'vwap': 'mean',
            'funding_rate': 'mean',
        }

        for base_col_lower, agg_method in standard_aggs.items():
            if base_col_lower in columns_lower_map:
                actual_col = columns_lower_map[base_col_lower]
                agg_dict[actual_col] = agg_method
                if store_column_map:
                    self.original_data_map[f"{base_col_lower}_column"] = actual_col

        # Обробка решти числових колонок
        for col in numeric_cols:
            if col not in agg_dict:
                col_lower = col.lower()
                if any(x in col_lower for x in ['count', 'number', 'trades', 'qty', 'quantity', 'amount', 'volume']):
                    agg_dict[col] = 'sum'
                elif any(x in col_lower for x in ['id', 'code', 'identifier']):
                    agg_dict[col] = 'last'
                elif any(x in col_lower for x in ['price', 'rate', 'fee', 'vwap']):
                    agg_dict[col] = 'mean'
                else:
                    agg_dict[col] = 'mean'

        # Обробка нечислових колонок
        non_numeric_cols = [col for col in data.columns if col not in numeric_cols]
        for col in non_numeric_cols:
            if col not in agg_dict:
                agg_dict[col] = 'last'

        return agg_dict

    def resample_data(self, data: pd.DataFrame, target_interval: str,
                      required_columns: List[str] = None) -> pd.DataFrame:
        """
        Оптимізована версія методу ресемплінгу з підтримкою великих обсягів даних
        """
        self.logger.info(f"Наявні колонки в resample_data: {list(data.columns)}")

        if data.empty:
            self.logger.warning("Отримано порожній DataFrame для ресемплінгу")
            return data

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Дані повинні мати DatetimeIndex для ресемплінгу")

        # Збереження списку оригінальних колонок
        original_columns = set(data.columns)
        self.logger.info(f"Початкові колонки: {original_columns}")

        # Зберігаємо оригінальні дані
        self.original_data_map['original_data'] = data.copy()

        # Перевірка необхідних колонок
        if required_columns is None:
            required_columns = ['open', 'high', 'low', 'close', 'volume']

        # Перевірка наявності колонок (незалежно від регістру)
        data_columns_lower = {col.lower(): col for col in data.columns}
        missing_cols = [
            col for col in required_columns
            if col.lower() not in data_columns_lower
        ]

        if missing_cols:
            self.logger.error(f"Відсутні необхідні колонки: {missing_cols}")
            if len(missing_cols) == len(required_columns):
                self.logger.error("Неможливо виконати ресемплінг без необхідних колонок даних")
                return data
            else:
                self.logger.warning("Ресемплінг буде виконано, але результати можуть бути неповними")

        # Перевірка інтервалу часу
        try:
            pandas_interval = self.convert_interval_to_pandas_format(target_interval)
            self.logger.info(f"Ресемплінг даних до інтервалу: {target_interval} (pandas формат: {pandas_interval})")
        except ValueError as e:
            self.logger.error(f"Неправильний формат інтервалу: {str(e)}")
            return data

        # Оптимізована підготовка агрегацій
        agg_dict = self._optimize_aggregation_dict(data)

        # Оптимізована обробка великих наборів даних
        batch_size = 500_000  # Налаштуйте під ваші обмеження пам'яті
        total_rows = len(data)

        if total_rows <= batch_size:
            # Якщо дані менші за batch_size, обробляємо весь DataFrame
            try:
                resampled = data.resample(pandas_interval).agg(agg_dict)

                # Заповнення відсутніх значень з оптимізацією
                resampled = self._fill_missing_values(resampled)

                self.original_data_map['resampled_data'] = resampled.copy()
                return resampled
            except Exception as e:
                self.logger.error(f"Помилка при ресемплінгу: {str(e)}")
                return data

        # Batch-обробка для великих наборів даних
        result_batches = []
        for start in range(0, total_rows, batch_size):
            end = min(start + batch_size, total_rows)
            batch = data.iloc[start:end]

            try:
                # Ресемплінг батчу
                resampled_batch = batch.resample(pandas_interval).agg(agg_dict)
                result_batches.append(resampled_batch)
            except Exception as e:
                self.logger.error(f"Помилка при обробці батчу: {str(e)}")
                continue

        # Об'єднання результатів
        try:
            resampled = pd.concat(result_batches, ignore_index=False)

            # Заповнення відсутніх значень
            resampled = self._fill_missing_values(resampled)

            self.original_data_map['resampled_data'] = resampled.copy()

            self.logger.info(
                f"Ресемплінг успішно завершено: {resampled.shape[0]} рядків, {len(resampled.columns)} колонок")
            return resampled

        except Exception as e:
            self.logger.error(f"Помилка при об'єднанні батчів: {str(e)}")
            return data


    def _fill_missing_values(self, df: pd.DataFrame, fill_method: str = 'auto',
                             max_gap: int = 5, interpolate_prices: bool = True) -> pd.DataFrame:

        if df.empty:
            return df

        # Створимо копію для запобігання модифікації оригінального DataFrame
        df = df.copy()

        # Створимо маску пропущених значень для подальшого аналізу
        missing_mask = df.isna()

        # Підрахуємо кількість пропущених значень до заповнення для логування
        initial_missing = missing_mask.sum().sum()

        # Мапінг колонок до нижнього регістру для спрощення пошуку
        columns_lower = {col.lower(): col for col in df.columns}

        # Визначення колонок за категоріями
        price_cols = []
        volume_cols = []
        trades_cols = []
        other_numeric_cols = []
        non_numeric_cols = []

        # Визначаємо цінові колонки
        for col_pattern in ['open', 'high', 'low', 'close', 'vwap']:
            if col_pattern in columns_lower:
                price_cols.append(columns_lower[col_pattern])

        # Визначаємо колонки об'єму
        for col_pattern in ['volume', 'taker_buy_volume', 'taker_sell_volume', 'quote_volume',
                            'taker_buy_base_volume', 'taker_buy_quote_volume', 'quote_asset_volume']:
            if col_pattern in columns_lower:
                volume_cols.append(columns_lower[col_pattern])

        # Колонки для кількості угод
        for col_pattern in ['trades', 'number_of_trades']:
            if col_pattern in columns_lower:
                trades_cols.append(columns_lower[col_pattern])

        # Виділяємо всі інші числові колонки
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        remaining_numeric = [col for col in numeric_cols if col not in price_cols and
                             col not in volume_cols and col not in trades_cols]

        # Виділяємо не-числові колонки
        non_numeric_cols = [col for col in df.columns if col not in numeric_cols]

        # Функція для безпечного forward fill з обмеженням на max_gap
        def safe_ffill(series, limit=max_gap):
            return series.fillna(method='ffill', limit=limit)

        # Функція для безпечного backward fill з обмеженням на max_gap
        def safe_bfill(series, limit=max_gap):
            return series.fillna(method='bfill', limit=limit)

        # Заповнення цінових колонок
        if price_cols:
            if fill_method in ['auto', 'interpolate'] and interpolate_prices:
                # Спочатку спробуємо лінійну інтерполяцію для малих проміжків
                df[price_cols] = df[price_cols].interpolate(method='linear', limit=max_gap, limit_area='inside')

                # Для проміжків більших за max_gap або на краях використовуємо обмежений ffill/bfill
                df[price_cols] = df[price_cols].apply(safe_ffill)
                df[price_cols] = df[price_cols].apply(safe_bfill)
            elif fill_method in ['auto', 'ffill']:
                # Використовуємо безпечний ffill з обмеженням
                df[price_cols] = df[price_cols].apply(safe_ffill)
                df[price_cols] = df[price_cols].apply(safe_bfill)

        # Заповнення колонок об'єму
        if volume_cols:
            if fill_method in ['auto', 'zero']:
                # Для об'єму логічно використовувати нулі за відсутності даних
                df[volume_cols] = df[volume_cols].fillna(0)
            elif fill_method == 'ffill':
                df[volume_cols] = df[volume_cols].apply(safe_ffill)
                df[volume_cols] = df[volume_cols].fillna(0)  # Залишкові NaN як нулі

        # Заповнення колонок кількості угод
        if trades_cols:
            if fill_method in ['auto', 'zero']:
                df[trades_cols] = df[trades_cols].fillna(0)
            elif fill_method == 'ffill':
                df[trades_cols] = df[trades_cols].apply(safe_ffill)
                df[trades_cols] = df[trades_cols].fillna(0)

        # Заповнення інших числових колонок
        for col in remaining_numeric:
            col_lower = col.lower()
            if fill_method == 'auto':
                if any(x in col_lower for x in ['count', 'number', 'trades', 'qty', 'quantity', 'amount']):
                    # Лічильники заповнюємо нулями
                    df[col] = df[col].fillna(0)
                elif any(x in col_lower for x in ['funding', 'rate', 'fee']):
                    # Ставки заповнюємо обмеженим ffill або нулями
                    df[col] = safe_ffill(df[col]).fillna(0)
                else:
                    # Решта - обмежений ffill/bfill і нулі
                    df[col] = safe_ffill(df[col])
                    df[col] = safe_bfill(df[col])
                    df[col] = df[col].fillna(0)
            elif fill_method == 'ffill':
                df[col] = safe_ffill(df[col])
                df[col] = safe_bfill(df[col])
            elif fill_method == 'zero':
                df[col] = df[col].fillna(0)
            elif fill_method == 'interpolate':
                df[col] = df[col].interpolate(method='linear', limit=max_gap, limit_area='inside')
                df[col] = safe_ffill(df[col])
                df[col] = df[col].fillna(0)

        # Заповнення не-числових колонок (з обмеженням на max_gap)
        if non_numeric_cols and fill_method in ['auto', 'ffill', 'bfill']:
            for col in non_numeric_cols:
                df[col] = safe_ffill(df[col])
                df[col] = safe_bfill(df[col])

        # Підрахуємо кількість пропущених значень після заповнення для логування
        remaining_missing = df.isna().sum().sum()

        # Виявлення довгих проміжків пропусків, які можуть свідчити про проблеми з даними
        if initial_missing > 0:
            long_gaps = {}
            for col in df.columns:
                # Знаходимо проміжки NaN довші за max_gap
                gaps = missing_mask[col].astype(int).groupby(
                    (missing_mask[col] != missing_mask[col].shift()).cumsum()
                ).sum()
                long_gaps_count = (gaps[gaps > max_gap]).count()
                if long_gaps_count > 0:
                    long_gaps[col] = long_gaps_count

        return df

    def convert_interval_to_pandas_format(self, timeframe: str) -> str:

        if not timeframe or not isinstance(timeframe, str):
            raise ValueError(f"Неправильний формат інтервалу: {timeframe}")

        interval_map = {
            's': 'S',  # секунди
            'm': 'T',  # хвилини (в pandas використовується 'T' для хвилин)
            'h': 'H',  # години
            'd': 'D',  # дні
            'w': 'W',  # тижні
            'M': 'M',  # місяці
        }

        import re
        match = re.match(r'(\d+)([smhdwM])', timeframe)
        if not match:
            raise ValueError(f"Неправильний формат інтервалу: {timeframe}")

        number, unit = match.groups()

        # Перевірка, чи число додатне
        if int(number) <= 0:
            raise ValueError(f"Інтервал повинен бути додатнім: {timeframe}")

        # Перевірка, чи підтримується одиниця часу
        if unit not in interval_map:
            raise ValueError(f"Непідтримувана одиниця часу: {unit}")

        # Конвертуємо у pandas формат
        pandas_interval = f"{number}{interval_map[unit]}"
        self.logger.info(f"Перетворено інтервал '{timeframe}' у pandas формат '{pandas_interval}'")

        return pandas_interval

    @lru_cache(maxsize=128)
    def _cached_convert_interval(self, timeframe: str) -> str:
        """
        Кешована версія конвертації інтервалу з підтримкою різних форматів
        """
        import re
        interval_map = {
            's': 'S', 'm': 'T', 'h': 'H',
            'd': 'D', 'w': 'W', 'M': 'M'
        }

        match = re.match(r'(\d+)([smhdwM])', timeframe)
        if not match:
            raise ValueError(f"Неправильний формат інтервалу: {timeframe}")

        number, unit = match.groups()

        if int(number) <= 0:
            raise ValueError(f"Інтервал повинен бути додатнім: {timeframe}")

        if unit not in interval_map:
            raise ValueError(f"Непідтримувана одиниця часу: {unit}")

        return f"{number}{interval_map[unit]}"

    def parse_interval(self, timeframe: str) -> pd.Timedelta:

        interval_map = {
            's': 'seconds',
            'm': 'minutes',
            'h': 'hours',
            'd': 'days',
            'w': 'weeks',
        }

        import re
        match = re.match(r'(\d+)([smhdwM])', timeframe)
        if not match:
            raise ValueError(f"Неправильний формат інтервалу: {timeframe}")

        number, unit = match.groups()

        # Перевірка, чи число додатне
        if int(number) <= 0:
            raise ValueError(f"Інтервал повинен бути додатнім: {timeframe}")

        if unit == 'M':
            return pd.Timedelta(days=int(number) * 30.44)

        return pd.Timedelta(**{interval_map[unit]: int(number)})

    def create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:

        df = data.copy()

        # Зберігаємо оригінальні дані перед маніпуляціями
        self.original_data_map['data_before_time_features'] = df.copy()

        # Перевірка на DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            self.logger.warning("DataFrame не має DatetimeIndex. Часові ознаки не можуть бути створені.")
            return df

        # Часові компоненти
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year

        # Додаємо is_weekend флаг (менш важливо для криптовалют, але залишаємо для повноти)
        df['is_weekend'] = df['day_of_week'].isin([5, 6])  # 5=Saturday, 6=Sunday

        # Для криптовалют поняття торгової сесії менш важливе, але можна залишити
        # з точки зору активності різних регіонів
        df['session'] = 'Unknown'
        df.loc[(df['hour'] >= 0) & (df['hour'] < 8), 'session'] = 'Asia'
        df.loc[(df['hour'] >= 8) & (df['hour'] < 16), 'session'] = 'Europe'
        df.loc[(df['hour'] >= 16) & (df['hour'] < 24), 'session'] = 'US'

        # Додаємо ознаку "час доби" для криптовалют
        df['time_of_day'] = 'Unknown'
        df.loc[(df['hour'] >= 0) & (df['hour'] < 6), 'time_of_day'] = 'Night'
        df.loc[(df['hour'] >= 6) & (df['hour'] < 12), 'time_of_day'] = 'Morning'
        df.loc[(df['hour'] >= 12) & (df['hour'] < 18), 'time_of_day'] = 'Afternoon'
        df.loc[(df['hour'] >= 18) & (df['hour'] < 24), 'time_of_day'] = 'Evening'

        # Циклічні ознаки для часових компонентів
        df['hour_sin'] = np.sin(df['hour'] * (2 * np.pi / 24))
        df['hour_cos'] = np.cos(df['hour'] * (2 * np.pi / 24))
        df['day_of_week_sin'] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
        df['day_of_week_cos'] = np.cos(df['day_of_week'] * (2 * np.pi / 7))
        df['month_sin'] = np.sin((df['month'] - 1) * (2 * np.pi / 12))
        df['month_cos'] = np.cos((df['month'] - 1) * (2 * np.pi / 12))
        df['day_of_month_sin'] = np.sin((df['day_of_month'] - 1) * (2 * np.pi / 31))
        df['day_of_month_cos'] = np.cos((df['day_of_month'] - 1) * (2 * np.pi / 31))

        return df

    def make_stationary(self, data: pd.DataFrame, columns=None, method='diff',
                        order=1, seasonal_order=None) -> pd.DataFrame:
        self.logger.info(f"Наявні колонки в make_stationary: {list(data.columns)}")
        if data.empty:
            self.logger.warning("make_stationary: Отримано порожній DataFrame")
            return data

        if columns is None:
            columns = ['close']

        df = data.copy()
        result_df = df.copy()  # Створюємо окрему копію для результату, щоб не втратити оригінальні дані

        # Зберігаємо посилання на оригінальний DataFrame
        self.original_data_map['data_before_stationary'] = df.copy()

        # Створимо словник для відображення колонок за нижнім регістром
        column_map = {col.lower(): col for col in df.columns}

        # Для кожної колонки в переліку
        for col_name in columns:
            # Знаходимо відповідну колонку, незалежно від регістру
            col = None
            if col_name in df.columns:
                col = col_name
            elif col_name.lower() in column_map:
                col = column_map[col_name.lower()]

            if col is None:
                self.logger.warning(f"make_stationary: Колонка '{col_name}' відсутня у DataFrame і буде пропущена")
                continue

            # Зберігаємо оригінальні дані для цієї колонки
            col_key = f"{col}_original"
            self.original_data_map[col_key] = df[col].copy()
            self.logger.info(f"make_stationary: Збережено оригінальні дані для колонки '{col}'")

            # Базове диференціювання
            if method == 'diff' or method == 'all':
                diff_col = f'{col}_diff'
                result_df[diff_col] = df[col].diff(order)
                self.logger.info(f"make_stationary: Створено колонку {diff_col}")

                # Додаємо диференціювання 2-го порядку
                diff2_col = f'{col}_diff2'
                result_df[diff2_col] = df[col].diff().diff()
                self.logger.info(f"make_stationary: Створено колонку {diff2_col}")

                # Додаємо сезонне диференціювання якщо потрібно
                if seasonal_order and seasonal_order > 0:
                    seasonal_diff_col = f'{col}_seasonal_diff'
                    result_df[seasonal_diff_col] = df[col].diff(seasonal_order)
                    self.logger.info(f"make_stationary: Створено колонку {seasonal_diff_col}")

                    # Комбіноване сезонне + звичайне диференціювання
                    combo_diff_col = f'{col}_combo_diff'
                    result_df[combo_diff_col] = df[col].diff(seasonal_order).diff()
                    self.logger.info(f"make_stationary: Створено колонку {combo_diff_col}")

            # Логарифмічне перетворення
            if method == 'log' or method == 'all':
                # Перевірка на відсутність нульових чи від'ємних значень
                if (df[col] > 0).all():
                    log_col = f'{col}_log'
                    result_df[log_col] = np.log(df[col])
                    self.logger.info(f"make_stationary: Створено колонку {log_col}")

                    # Логарифм + диференціювання
                    log_diff_col = f'{col}_log_diff'
                    result_df[log_diff_col] = result_df[log_col].diff(order)
                    self.logger.info(f"make_stationary: Створено колонку {log_diff_col}")
                else:
                    self.logger.warning(
                        f"make_stationary: Колонка {col} містить нульові або від'ємні значення. "
                        f"Логарифмічне перетворення пропущено.")

            # Відсоткова зміна
            if method == 'pct_change' or method == 'all':
                pct_col = f'{col}_pct'
                result_df[pct_col] = df[col].pct_change(order)
                self.logger.info(f"make_stationary: Створено колонку {pct_col}")

            # Додаємо різницю між high та low (волатильність) якщо це колонка close
            high_col = self.find_column(df, 'high')
            low_col = self.find_column(df, 'low')

            if col.lower() == 'close' and high_col and low_col:
                high_low_range_col = 'high_low_range'
                result_df[high_low_range_col] = df[high_col] - df[low_col]
                self.logger.info(f"make_stationary: Створено колонку {high_low_range_col}")

                high_low_range_pct_col = 'high_low_range_pct'
                # Виправлення: перевірка на ділення на нуль
                mask = df[col] != 0  # Створюємо маску для ненульових значень
                result_df[high_low_range_pct_col] = np.nan  # Ініціалізуємо колонку як NaN
                # Застосовуємо ділення тільки там, де знаменник не дорівнює нулю
                result_df.loc[mask, high_low_range_pct_col] = result_df.loc[mask, high_low_range_col] / df.loc[mask, col]
                self.logger.info(f"make_stationary: Створено колонку {high_low_range_pct_col}")
                self.logger.warning(f"make_stationary: Значення, де {col} = 0, замінені на NaN в {high_low_range_pct_col}")

            # Для об'єму додаємо логарифм, що часто корисно для криптовалют
            if col.lower() == 'volume' and (df[col] > 0).all():
                vol_log_col = f'{col}_log'
                result_df[vol_log_col] = np.log(df[col])
                self.logger.info(f"make_stationary: Створено колонку {vol_log_col}")

                vol_log_diff_col = f'{col}_log_diff'
                result_df[vol_log_diff_col] = result_df[vol_log_col].diff(order)
                self.logger.info(f"make_stationary: Створено колонку {vol_log_diff_col}")

        # Зберігаємо повний DataFrame до видалення NaN
        self.original_data_map['stationary_with_na'] = result_df.copy()

        # Перевіряємо наявність NaN
        na_count = result_df.isna().sum().sum()
        if na_count > 0:
            self.logger.info(f"make_stationary: У результуючому DataFrame виявлено {na_count} NaN значень")

            # Замість повного видалення NaN, можемо замінити їх на відповідні значення
            # Для збереження розмірності даних
            for col in result_df.columns:
                if result_df[col].isna().any():
                    if col.endswith(('_diff', '_diff2', '_seasonal_diff', '_combo_diff', '_log_diff', '_pct')):
                        # Для диференційованих даних заповнюємо NaN нулями
                        result_df[col] = result_df[col].fillna(0)
                    elif col.endswith('_log'):
                        # Для логарифмічних даних використовуємо ffill/bfill
                        result_df[col] = result_df[col].fillna(method='ffill').fillna(method='bfill')
                    elif col == 'high_low_range_pct':  # Додаємо особливу обробку для high_low_range_pct
                        # Для high_low_range_pct заповнюємо NaN середнім значенням або нулем
                        if result_df[col].notna().any():  # Якщо є хоч якісь не-NaN значення
                            mean_val = result_df[col].mean()
                            result_df[col] = result_df[col].fillna(mean_val)
                            self.logger.info(f"make_stationary: NaN в {col} замінені на середнє значення {mean_val}")
                        else:
                            result_df[col] = result_df[col].fillna(0)
                            self.logger.info(f"make_stationary: NaN в {col} замінені на 0")

            # Перевіряємо, чи всі NaN були замінені
            remaining_na = result_df.isna().sum().sum()
            if remaining_na > 0:
                self.logger.warning(f"make_stationary: Залишилось {remaining_na} NaN значень після заповнення")

                # Якщо залишились NaN, зберігаємо оригінальні дані й видаляємо рядки з NaN
                rows_before = len(result_df)
                cleaned_df = result_df.dropna()
                rows_after = len(cleaned_df)

                if rows_before > rows_after:
                    self.logger.warning(f"make_stationary: Видалено {rows_before - rows_after} рядків з NaN "
                                        f"після диференціювання")
                    # Зберігаємо індекси видалених рядків для можливого відновлення
                    dropped_indices = set(result_df.index) - set(cleaned_df.index)
                    self.original_data_map['dropped_indices'] = list(dropped_indices)

                result_df = cleaned_df

        # Зберігаємо фінальний результат для подальшого використання
        self.original_data_map['stationary_result'] = result_df.copy()
        self.logger.info(f"make_stationary: Створено стаціонарні дані з {len(result_df)} рядками та "
                         f"{len(result_df.columns)} колонками")

        return result_df

    def check_stationarity(self, data: pd.DataFrame, column='close_diff', sample_size=10000,
                           parallel=True, confidence_level=0.05) -> dict:

        self.logger.info(f"Наявні колонки в check_stationarity: {list(data.columns)}")
        from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
        import numpy as np
        import concurrent.futures

        results = {}

        # Перевірка наявності даних
        if data.empty:
            error_msg = "check_stationarity: Отримано порожній DataFrame для перевірки стаціонарності"
            self.logger.error(error_msg)
            return {'error': error_msg, 'is_stationary': False}

        # Перевіряємо наявність колонки, незалежно від регістру
        column_to_use = self.find_column(data, column)

        if column_to_use is None:
            error_msg = f"check_stationarity: Колонка '{column}' відсутня у DataFrame. Доступні колонки: {list(data.columns)}"
            self.logger.error(error_msg)
            return {'error': error_msg, 'is_stationary': False}

        # Перевірка на відсутні значення
        clean_data = data[column_to_use].dropna()
        if len(clean_data) < 2:
            error_msg = f"check_stationarity: Недостатньо даних для перевірки стаціонарності після видалення NaN значень"
            self.logger.error(error_msg)
            return {'error': error_msg, 'is_stationary': False}

        # Використання вибірки для великих наборів даних
        data_size = len(clean_data)
        self.logger.info(f"check_stationarity: Розмір даних - {data_size} точок")

        # Якщо даних більше ніж sample_size, беремо вибірку
        if data_size > sample_size:
            # Стратегія вибірки: беремо рівномірно розподілені точки з усього ряду
            step = max(1, data_size // sample_size)
            sampled_data = clean_data.iloc[::step].copy()
            self.logger.info(f"check_stationarity: Використовуємо вибірку {len(sampled_data)} точок (крок {step})")
        else:
            sampled_data = clean_data.copy()

        # Функції для виконання тестів
        def run_adf_test():
            try:
                # Використовуємо менше лагів для прискорення
                max_lags = min(int(np.ceil(12 * (len(sampled_data) / 100) ** (1 / 4))), 20)
                adf_result = adfuller(sampled_data, maxlag=max_lags)
                return {
                    'test_statistic': adf_result[0],
                    'p-value': adf_result[1],
                    'is_stationary': adf_result[1] < confidence_level,
                    'critical_values': adf_result[4],
                    'used_lags': max_lags
                }
            except Exception as e:
                return {'error': f"Помилка при виконанні ADF тесту: {str(e)}", 'is_stationary': False}

        def run_kpss_test():
            try:
                # Використовуємо менше лагів для прискорення
                max_lags = min(int(np.ceil(12 * (len(sampled_data) / 100) ** (1 / 4))), 20)
                kpss_result = kpss(sampled_data, nlags=max_lags)
                return {
                    'test_statistic': kpss_result[0],
                    'p-value': kpss_result[1],
                    'is_stationary': kpss_result[1] > confidence_level,
                    'critical_values': kpss_result[3],
                    'used_lags': max_lags
                }
            except Exception as e:
                return {'error': f"Помилка при виконанні KPSS тесту: {str(e)}", 'is_stationary': False}

        def run_acf_pacf_analysis():
            try:
                # Визначаємо розумну кількість лагів
                # Для 4 млн точок, навіть sample_size може бути надто великим для ACF/PACF
                acf_pacf_sample = sampled_data
                if len(sampled_data) > 2000:
                    # Для ACF/PACF використовуємо ще меншу вибірку
                    step = max(1, len(sampled_data) // 2000)
                    acf_pacf_sample = sampled_data.iloc[::step].copy()
                    self.logger.info(f"ACF/PACF: Використовуємо зменшену вибірку {len(acf_pacf_sample)} точок")

                # Обмежуємо максимальну кількість лагів
                max_lags = min(40, int(len(acf_pacf_sample) * 0.1))

                # Використовуємо fft=False для менших вибірок, щоб зменшити споживання пам'яті
                use_fft = len(acf_pacf_sample) > 1000

                acf_values = acf(acf_pacf_sample, nlags=max_lags, fft=use_fft)
                pacf_values = pacf(acf_pacf_sample, nlags=max_lags, method="ywm")  # метод Юла-Уокера - швидший

                # Знаходження значимих лагів (з 95% довірчим інтервалом)
                n = len(acf_pacf_sample)
                confidence_interval = 1.96 / np.sqrt(n)

                significant_acf_lags = [i for i, v in enumerate(acf_values) if abs(v) > confidence_interval and i > 0]
                significant_pacf_lags = [i for i, v in enumerate(pacf_values) if abs(v) > confidence_interval and i > 0]

                # Якщо немає значимих лагів, використовуємо лаг 1
                suggested_p = min(significant_pacf_lags) if significant_pacf_lags else 1
                suggested_q = min(significant_acf_lags) if significant_acf_lags else 1

                return {
                    'significant_acf_lags': significant_acf_lags[:5],  # обмежуємо вивід для економії пам'яті
                    'significant_pacf_lags': significant_pacf_lags[:5],
                    'suggested_p': suggested_p,
                    'suggested_q': suggested_q,
                    'sample_size': len(acf_pacf_sample)
                }
            except Exception as e:
                return {'error': f"Помилка при розрахунку ACF/PACF: {str(e)}"}

        # Виконання тестів - паралельно або послідовно
        if parallel and data_size > 5000:
            self.logger.info("check_stationarity: Використовуємо паралельну обробку для тестів")
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    'adf_test': executor.submit(run_adf_test),
                    'kpss_test': executor.submit(run_kpss_test),
                    'acf_pacf': executor.submit(run_acf_pacf_analysis)
                }

                for key, future in futures.items():
                    try:
                        results[key] = future.result()
                        if key == 'adf_test':
                            self.logger.info(
                                f"check_stationarity: ADF тест завершено, p-value={results[key].get('p-value', 'N/A')}")
                        elif key == 'kpss_test':
                            self.logger.info(
                                f"check_stationarity: KPSS тест завершено, p-value={results[key].get('p-value', 'N/A')}")
                    except Exception as e:
                        results[key] = {'error': f"Помилка при виконанні {key}: {str(e)}"}
                        self.logger.error(f"check_stationarity: {results[key]['error']}")
        else:
            # Послідовне виконання
            self.logger.info("check_stationarity: Виконуємо тести послідовно")
            results['adf_test'] = run_adf_test()
            if 'p-value' in results['adf_test']:
                self.logger.info(
                    f"check_stationarity: ADF тест завершено, p-value={results['adf_test']['p-value']:.4f}")

            results['kpss_test'] = run_kpss_test()
            if 'p-value' in results['kpss_test']:
                self.logger.info(
                    f"check_stationarity: KPSS тест завершено, p-value={results['kpss_test']['p-value']:.4f}")

            results['acf_pacf'] = run_acf_pacf_analysis()
            if 'suggested_p' in results['acf_pacf']:
                self.logger.info(
                    f"check_stationarity: ACF/PACF аналіз завершено, p={results['acf_pacf']['suggested_p']}, q={results['acf_pacf']['suggested_q']}")

        # Загальний висновок про стаціонарність
        adf_stationary = results.get('adf_test', {}).get('is_stationary', False)
        kpss_stationary = results.get('kpss_test', {}).get('is_stationary', True)

        results['is_stationary'] = adf_stationary and kpss_stationary
        results['sample_info'] = {
            'original_size': data_size,
            'sample_size': len(sampled_data),
            'sampling_rate': f"1/{data_size // len(sampled_data)}" if data_size > len(sampled_data) else "1/1"
        }

        self.logger.info(f"check_stationarity: Загальний висновок про стаціонарність: {results['is_stationary']}")

        return results

    def prepare_arima_data(
            self,
            data: pd.DataFrame | dd.DataFrame,
            symbol: str,
            timeframe: str
    ) -> pd.DataFrame:
        try:
            # Convert Dask DataFrame to pandas if necessary
            if hasattr(data, 'compute'):
                data = data.compute()

            # Ensure we have datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                if 'open_time' in data.columns:
                    data.set_index('open_time', inplace=True)
                else:
                    self.logger.error("No datetime index or open_time column found")
                    return pd.DataFrame()

            # 1. Select close price column
            close_columns = ['close', 'price', 'last', 'last_price']
            close_column = next((col for col in close_columns if col in data.columns), None)

            if not close_column:
                self.logger.error("No close price column found")
                return pd.DataFrame()

            # Create a copy to avoid modifying the original
            # Виправлений рядок: створюємо DataFrame з правильним індексом замість reset_index
            result_df = pd.DataFrame(index=pd.RangeIndex(len(data.index)))

            # Basic info
            result_df['timeframe'] = timeframe
            result_df['open_time'] = data.index  # Store timestamp as a column
            result_df['original_close'] = data[close_column]

            # 2. Compute various stationary transformations
            # First difference
            result_df['close_diff'] = data[close_column].diff()

            # Second difference
            result_df['close_diff2'] = result_df['close_diff'].diff()

            # Log transformation
            result_df['close_log'] = np.log(data[close_column])

            # Log difference (log return)
            result_df['close_log_diff'] = result_df['close_log'].diff()

            # Percentage change
            result_df['close_pct_change'] = data[close_column].pct_change()

            # Seasonal differencing (assuming daily data with weekly seasonality)
            season_period = self._determine_seasonal_period(timeframe)
            result_df['close_seasonal_diff'] = data[close_column].diff(season_period)

            # Combined differencing (first difference + seasonal)
            result_df['close_combo_diff'] = result_df['close_seasonal_diff'].diff()

            # 3. Perform stationarity tests on log returns (most commonly used transformation)
            log_returns = result_df['close_log_diff'].dropna()

            if len(log_returns) > 10:
                # Augmented Dickey-Fuller test
                adf_result = adfuller(log_returns)
                result_df['adf_pvalue'] = adf_result[1]

                # KPSS test
                try:
                    from statsmodels.tsa.stattools import kpss
                    kpss_result = kpss(log_returns)
                    result_df['kpss_pvalue'] = kpss_result[1]
                except:
                    result_df['kpss_pvalue'] = None
                    self.logger.warning("KPSS test failed, consider installing statsmodels")

                # Determine stationarity based on both tests
                result_df['is_stationary'] = (
                        (result_df['adf_pvalue'] <= 0.05) &
                        ((result_df['kpss_pvalue'] > 0.05) if 'kpss_pvalue' in result_df else True)
                )

                # 4. Calculate ACF/PACF for model configuration
                try:
                    from statsmodels.tsa.stattools import acf, pacf
                    acf_values = acf(log_returns, nlags=20)
                    pacf_values = pacf(log_returns, nlags=20)

                    # Find significant lags (using 95% confidence interval)
                    significant_lags = []
                    confidence_level = 1.96 / np.sqrt(len(log_returns))

                    for i, (acf_val, pacf_val) in enumerate(zip(acf_values, pacf_values)):
                        if i > 0 and (abs(acf_val) > confidence_level or abs(pacf_val) > confidence_level):
                            significant_lags.append(i)

                    result_df['significant_lags'] = json.dumps(significant_lags)
                except:
                    result_df['significant_lags'] = '[]'
                    self.logger.warning("ACF/PACF calculation failed")

                # 5. Fit a basic ARIMA model to get additional metrics
                try:
                    from statsmodels.tsa.arima.model import ARIMA
                    # Use a simple ARIMA(1,1,1) as default
                    model = ARIMA(data[close_column], order=(1, 1, 1))
                    model_fit = model.fit()

                    result_df['residual_variance'] = model_fit.resid.var()
                    result_df['aic_score'] = model_fit.aic
                    result_df['bic_score'] = model_fit.bic
                except:
                    result_df['residual_variance'] = None
                    result_df['aic_score'] = None
                    result_df['bic_score'] = None
                    self.logger.warning("ARIMA model fitting failed")

                self.logger.info(f"Prepared ARIMA data for {symbol} ({timeframe})")
                return result_df.reset_index(drop=True)  # Ensure no duplicate index/column
            else:
                self.logger.warning(f"Insufficient data for stationarity tests for {symbol}")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error preparing ARIMA data: {e}")
            return pd.DataFrame()

    def _determine_seasonal_period(self, timeframe: str) -> int:
            """
            Determine the appropriate seasonal period based on timeframe.

            :param timeframe: Trading timeframe string (e.g., '1d', '1h', '15m')
            :return: Integer representing the seasonal period
            """
            # Extract the numeric part and unit from timeframe
            import re
            match = re.match(r'(\d+)([a-zA-Z]+)', timeframe)
            if not match:
                return 7  # Default to weekly seasonality

            value, unit = match.groups()
            value = int(value)

            # Set seasonal periods based on common patterns
            if unit.lower() in ['m', 'min', 'minute']:
                if value <= 5:
                    return 288  # Daily seasonality for 5m or less (288 5-min periods in a day)
                elif value <= 15:
                    return 96  # Daily seasonality for 15m (96 15-min periods in a day)
                elif value <= 60:
                    return 24  # Daily seasonality for hourly data (24 hours in a day)
            elif unit.lower() in ['h', 'hour']:
                return 24  # Daily seasonality for hourly data
            elif unit.lower() in ['d', 'day']:
                return 7  # Weekly seasonality for daily data
            elif unit.lower() in ['w', 'week']:
                return 4  # Monthly seasonality for weekly data

            # Default to weekly seasonality
            return 7

    def prepare_lstm_data(
            self,
            data: pd.DataFrame | dd.DataFrame,
            symbol: str,
            timeframe: str,
            sequence_length: int = 60,
            target_horizons: list = [1, 5, 10]
    ) -> pd.DataFrame:
        try:
            self.logger.info(f"Preparing LSTM data for database storage: {symbol}, {timeframe}")

            # Convert Dask DataFrame to pandas if necessary
            if hasattr(data, 'compute'):
                data = data.compute()

            if data.empty:
                self.logger.warning("Empty DataFrame provided")
                return pd.DataFrame()

            # Ensure we have a DatetimeIndex
            if not isinstance(data.index, pd.DatetimeIndex):
                self.logger.error("Data must have a DatetimeIndex")
                return pd.DataFrame()

            # 1. Select required feature columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']

            # Check for missing columns (case-insensitive)
            data_columns_lower = {col.lower(): col for col in data.columns}
            missing_cols = [col for col in required_columns if col.lower() not in data_columns_lower]

            if missing_cols:
                self.logger.error(f"Missing required columns: {missing_cols}")
                return pd.DataFrame()

            # Map to actual column names in the DataFrame
            feature_mapping = {col: data_columns_lower[col.lower()] for col in required_columns}

            # Create a copy with standardized column names
            df = data.copy()
            for std_name, actual_name in feature_mapping.items():
                if std_name != actual_name:
                    df[std_name] = df[actual_name]

            # 2. Add time features from the create_time_features method
            df = self.create_time_features(df)

            # 3. Scale the features
            scaler = MinMaxScaler(feature_range=(0, 1))

            # Select a sample for fitting the scaler if the dataset is very large
            if len(df) > self.scaling_sample_size:
                sample_indices = np.random.choice(df.index, size=self.scaling_sample_size, replace=False)
                sample_df = df.loc[sample_indices, required_columns]
                scaler.fit(sample_df)
            else:
                scaler.fit(df[required_columns])

            # Scale the required columns
            scaled_data = scaler.transform(df[required_columns])
            scaled_df = pd.DataFrame(
                scaled_data,
                columns=[f"{col}_scaled" for col in required_columns],
                index=df.index
            )

            # Combine with original DataFrame
            result_df = pd.concat([df, scaled_df], axis=1)

            # 4. Create target values for different horizons
            for horizon in target_horizons:
                result_df[f'target_close_{horizon}'] = result_df['close'].shift(-horizon)

            # 5. Create sequence IDs and positions
            valid_end_idx = len(result_df) - max(target_horizons)

            # Generate a reasonable number of sequences
            step = 1
            if valid_end_idx > 10000:  # If we have a lot of data
                step = valid_end_idx // 10000  # Limit to ~10K sequences
                self.logger.info(f"Large dataset detected, using step size: {step} for sequence generation")

            sequence_data = []
            for seq_id, start_idx in enumerate(range(0, valid_end_idx - sequence_length, step)):
                for pos in range(sequence_length):
                    idx = start_idx + pos
                    row = result_df.iloc[idx].copy()

                    # Only include rows where we have target values for all horizons
                    if idx + max(target_horizons) < len(result_df):
                        sequence_data.append({
                            'timeframe': timeframe,
                            'sequence_id': seq_id,
                            'sequence_position': pos,
                            'open_time': result_df.index[idx],

                            # Scaled features
                            'open_scaled': float(row['open_scaled']),
                            'high_scaled': float(row['high_scaled']),
                            'low_scaled': float(row['low_scaled']),
                            'close_scaled': float(row['close_scaled']),
                            'volume_scaled': float(row['volume_scaled']),

                            # Time features
                            'hour_sin': float(row['hour_sin']),
                            'hour_cos': float(row['hour_cos']),
                            'day_of_week_sin': float(row['day_of_week_sin']),
                            'day_of_week_cos': float(row['day_of_week_cos']),
                            'month_sin': float(row['month_sin']),
                            'month_cos': float(row['month_cos']),
                            'day_of_month_sin': float(row['day_of_month_sin']),
                            'day_of_month_cos': float(row['day_of_month_cos']),

                            # Target values
                            'target_close_1': float(row['target_close_1']) if 1 in target_horizons else None,
                            'target_close_5': float(row['target_close_5']) if 5 in target_horizons else None,
                            'target_close_10': float(row['target_close_10']) if 10 in target_horizons else None,

                            # Metadata
                            'sequence_length': sequence_length,
                            'scaling_metadata': json.dumps({
                                'feature_range': scaler.feature_range,
                                'data_min': scaler.data_min_.tolist(),
                                'data_max': scaler.data_max_.tolist(),
                                'columns': required_columns
                            })
                        })

            final_df = pd.DataFrame(sequence_data)

            self.logger.info(f"Prepared {len(final_df)} rows of LSTM data for database storage")
            self.logger.info(f"Created {final_df['sequence_id'].nunique()} unique sequences")

            # Store the scaler in cache for later use
            self.scalers[f'{symbol}_{timeframe}_lstm_scaler'] = scaler

            return final_df

        except Exception as e:
            self.logger.error(f"Error preparing LSTM data for database: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()