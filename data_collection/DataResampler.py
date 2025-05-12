import numpy as np
import pandas as pd
import dask.dataframe as dd
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
from pmdarima.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Union, Optional
import statsmodels.api as sm
import concurrent.futures
from functools import lru_cache
import numba

class DataResampler:
    def __init__(self, logger, chunk_size=500_000, scaling_sample_size= 1_000_000):
        self.logger = logger
        self.scalers = {}
        self.original_data_map = {}
        self.chunk_size = chunk_size
        self.scaling_sample_size = scaling_sample_size
        # Кешування методів пошуку колонок
        self.find_column = lru_cache(maxsize=128)(self._find_column_original)

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

    def _optimize_aggregation_dict(self, data: pd.DataFrame) -> Dict:
        """
        Оптимізована версія створення словника агрегацій з паралельною обробкою
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        agg_dict = {}

        # Базові пріоритетні колонки
        priority_columns = {
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }

        # Додаткові специфічні колонки
        crypto_specific_columns = {
            'trades': 'sum', 'taker_buy_volume': 'sum',
            'taker_sell_volume': 'sum', 'quote_volume': 'sum',
            'vwap': 'mean', 'funding_rate': 'mean'
        }

        # Пошук колонок з використанням прискорених методів
        columns_lower_map = {col.lower(): col for col in data.columns}

        # Додавання пріоритетних колонок
        for base_col_lower, agg_method in {**priority_columns, **crypto_specific_columns}.items():
            if base_col_lower in columns_lower_map:
                actual_col = columns_lower_map[base_col_lower]
                agg_dict[actual_col] = agg_method

        # Обробка решти числових колонок
        for col in numeric_cols:
            if col not in agg_dict:
                col_lower = col.lower()
                if any(x in col_lower for x in ['count', 'number', 'trades', 'qty', 'quantity', 'amount', 'volume']):
                    agg_dict[col] = 'sum'
                elif any(x in col_lower for x in ['price', 'rate', 'fee', 'vwap']):
                    agg_dict[col] = 'mean'
                else:
                    agg_dict[col] = 'last'

        # Додавання не-числових колонок
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

    def _create_aggregation_dict(self, data: pd.DataFrame) -> Dict:
        self.logger.info(f"Наявні колонки в _create_aggregation_dict: {list(data.columns)}")

        agg_dict = {}

        # Мапінг колонок до нижнього регістру для спрощення пошуку
        columns_lower_map = {col.lower(): col for col in data.columns}

        # Базові OHLCV колонки та їх методи агрегації
        base_columns = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }

        # Додаткові специфічні колонки для криптовалютних даних
        crypto_specific_columns = {
            'trades': 'sum',  # Кількість угод
            'taker_buy_volume': 'sum',  # Об'єм покупок taker
            'taker_sell_volume': 'sum',  # Об'єм продажів taker
            'taker_buy_base_volume': 'sum',  # Об'єм у базовій валюті покупок через taker
            'taker_buy_quote_volume': 'sum',  # Об'єм у котированій валюті покупок через taker
            'quote_volume': 'sum',  # Об'єм у котированій валюті
            'quote_asset_volume': 'sum',  # Об'єм у котированій валюті (альтернативна назва)
            'number_of_trades': 'sum',  # Кількість угод (альтернативна назва)
            'vwap': 'mean',  # Volume Weighted Average Price
            'funding_rate': 'mean'  # Funding Rate для ф'ючерсів
        }

        # Об'єднуємо базові і специфічні для криптовалют колонки
        all_columns = {**base_columns, **crypto_specific_columns}

        # Додаємо базові колонки зі словника (незалежно від регістру)
        for base_col_lower, agg_method in all_columns.items():
            if base_col_lower in columns_lower_map:
                actual_col = columns_lower_map[base_col_lower]
                agg_dict[actual_col] = agg_method
                # Зберігаємо відображення для пізнішого доступу
                self.original_data_map[f"{base_col_lower}_column"] = actual_col

        # Обробка всіх числових колонок, які ще не додані
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in agg_dict:  # Якщо колонку ще не додано
                col_lower = col.lower()
                if any(x in col_lower for x in ['count', 'number', 'trades', 'qty', 'quantity', 'amount', 'volume']):
                    agg_dict[col] = 'sum'
                elif any(x in col_lower for x in ['id', 'code', 'identifier']):
                    agg_dict[col] = 'last'  # Для ідентифікаторів беремо останнє значення
                elif any(x in col_lower for x in ['price', 'rate', 'fee', 'vwap']):
                    agg_dict[col] = 'mean'  # Для цін та ставок використовуємо середнє
                else:
                    agg_dict[col] = 'mean'  # Для всіх інших числових - середнє

        # Обробка всіх не-числових колонок (категорійних, текстових тощо)
        non_numeric_cols = [col for col in data.columns if col not in numeric_cols]
        for col in non_numeric_cols:
            if col not in agg_dict:
                agg_dict[col] = 'last'  # За замовчуванням беремо останнє значення

        # Перевірка, чи всі колонки включено
        for col in data.columns:
            if col not in agg_dict:
                self.logger.warning(f"Колонка '{col}' має нестандартний тип і буде агрегована методом 'last'")
                agg_dict[col] = 'last'

        return agg_dict

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
        """
        Prepare data for ARIMA time series analysis.

        :param data: Input DataFrame (pandas or Dask)
        :param symbol: Trading symbol
        :param timeframe: Trading timeframe
        :return: Prepared DataFrame for ARIMA analysis
        """
        try:
            # Convert Dask DataFrame to pandas if necessary
            if hasattr(data, 'compute'):
                data = data.compute()

            # 1. Select close price column
            close_columns = ['close', 'price', 'last', 'last_price']
            close_column = next((col for col in close_columns if col in data.columns), None)

            if not close_column:
                self.logger.error("No close price column found")
                return pd.DataFrame()

            # 2. Compute differences and log returns
            data = data.copy()
            data['diff'] = data[close_column].diff()
            data['log_return'] = np.log(data[close_column] / data[close_column].shift(1))

            # 3. Check stationarity
            log_returns = data['log_return'].dropna()

            if len(log_returns) > 10:
                # Perform Augmented Dickey-Fuller test
                adf_result = adfuller(log_returns)

                # 4. Prepare result DataFrame
                result_df = pd.DataFrame({
                    'symbol': [symbol],
                    'timeframe': [timeframe],
                    'close_price': [data[close_column]],
                    'log_return': [data['log_return']],
                    'diff': [data['diff']],
                    'adf_statistic': [adf_result[0]],
                    'adf_pvalue': [adf_result[1]],
                    'is_stationary': [adf_result[1] <= 0.05]
                })

                self.logger.info(f"Prepared ARIMA data for {symbol}")
                return result_df
            else:
                self.logger.warning(f"Insufficient data for stationarity test for {symbol}")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error preparing ARIMA data: {e}")
            return pd.DataFrame()

    def prepare_lstm_data(
            self,
            data: pd.DataFrame | dd.DataFrame,
            symbol: str,
            timeframe: str,
            sequence_length: int = 60
    ) -> pd.DataFrame:
        """
        Prepare data for LSTM time series analysis.

        :param data: Input DataFrame (pandas or Dask)
        :param symbol: Trading symbol
        :param timeframe: Trading timeframe
        :param sequence_length: Length of input sequences
        :return: Prepared DataFrame for LSTM analysis
        """
        try:
            # Convert Dask DataFrame to pandas if necessary
            if hasattr(data, 'compute'):
                data = data.compute()

            # 1. Select and prepare feature columns
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'trades', 'quote_volume'
            ]
            feature_columns = [col for col in feature_columns if col in data.columns]

            if not feature_columns:
                self.logger.error("No feature columns found for analysis")
                return pd.DataFrame()

            # 2. Sampling for scaling
            sample_data = data[feature_columns].sample(
                frac=0.2,
                random_state=42
            )

            # 3. Scaling using scikit-learn
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(sample_data)

            # 4. Create sequences
            def create_sequences(df):
                X, y = [], []
                for i in range(len(df) - sequence_length):
                    X.append(df.iloc[i:i + sequence_length][feature_columns].values)
                    y.append(df.iloc[i + sequence_length]['close'])
                return np.array(X), np.array(y)

            # 5. Split into train/test sets
            X, y = create_sequences(sample_data)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # 6. Prepare results
            result_df = pd.DataFrame({
                'symbol': [symbol],
                'timeframe': [timeframe],
                'X_train_shape': [X_train.shape],
                'X_test_shape': [X_test.shape],
                'sequence_length': [sequence_length],
                'features': [feature_columns]
            })

            # Store additional metadata in cache
            self.cache[f'{symbol}_{timeframe}_lstm_data'] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'scaler': scaler
            }

            self.logger.info(f"Prepared LSTM data for {symbol}")
            return result_df

        except Exception as e:
            self.logger.error(f"Error preparing LSTM data: {e}")
            return pd.DataFrame()