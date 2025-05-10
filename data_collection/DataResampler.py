import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Union, Optional
import statsmodels.api as sm


class DataResampler:
    def __init__(self, logger):
        self.logger = logger
        self.scalers = {}
        self.original_data_map = {}

    def find_column(self, df, column_name):
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

    def resample_data(self, data: pd.DataFrame, target_interval: str,
                      required_columns: List[str] = None) -> pd.DataFrame:

        if data.empty:
            self.logger.warning("Отримано порожній DataFrame для ресемплінгу")
            return data

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Дані повинні мати DatetimeIndex для ресемплінгу")

        # Збереження списку оригінальних колонок для перевірки після ресемплінгу
        original_columns = set(data.columns)
        self.logger.info(f"Початкові колонки: {original_columns}")

        # Зберігаємо оригінальні дані, щоб не втратити доступ до них
        self.original_data_map['original_data'] = data.copy()

        # Перевірка необхідних колонок, з урахуванням специфіки криптовалют
        if required_columns is None:
            required_columns = ['open', 'high', 'low', 'close', 'volume']

        # Перевірка наявності колонок (незалежно від регістру)
        data_columns_lower = {col.lower(): col for col in data.columns}
        missing_cols = []

        for required_col in required_columns:
            if required_col.lower() not in data_columns_lower:
                missing_cols.append(required_col)

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

        if len(data) > 1:
            current_interval = pd.Timedelta(data.index[1] - data.index[0])
            estimated_target_interval = self.parse_interval(target_interval)

            if estimated_target_interval < current_interval:
                self.logger.warning(f"Цільовий інтервал ({target_interval}) менший за поточний інтервал даних. "
                                    f"Даунсемплінг неможливий без додаткових даних.")
                return data

        # Створення словника агрегацій для всіх типів колонок
        agg_dict = self._create_aggregation_dict(data)

        try:
            # Виконання ресемплінгу
            self.logger.info(f"Починаємо ресемплінг з інтервалом {pandas_interval}")
            resampled = data.resample(pandas_interval).agg(agg_dict)

            # Перевірка результату ресемплінгу
            if resampled.empty:
                self.logger.warning(f"Результат ресемплінгу порожній. Можливо, проблема з інтервалом {pandas_interval}")
                return data

            # Зберігаємо ресемпльовані дані для подальшого доступу
            self.original_data_map['resampled_data'] = resampled.copy()

            # Заповнення відсутніх значень для всіх колонок після ресемплінгу
            if resampled.isna().any().any():
                self.logger.info("Заповнення відсутніх значень після ресемплінгу...")
                resampled = self._fill_missing_values(resampled)

            # Перевірка збереження всіх колонок
            resampled_columns = set(resampled.columns)
            missing_after_resample = original_columns - resampled_columns

            if missing_after_resample:
                self.logger.warning(f"Після ресемплінгу відсутні колонки: {missing_after_resample}")
                # Відновлення відсутніх колонок з NaN значеннями
                for col in missing_after_resample:
                    resampled[col] = np.nan

            self.logger.info(
                f"Ресемплінг успішно завершено: {resampled.shape[0]} рядків, {len(resampled.columns)} колонок")
            return resampled

        except Exception as e:
            self.logger.error(f"Помилка при ресемплінгу даних: {str(e)}")
            self.logger.error(f"Спробуйте інший формат інтервалу або перевірте вхідні дані")
            # Повертаємо вихідні дані замість помилки
            return data

    def _create_aggregation_dict(self, data: pd.DataFrame) -> Dict:

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

    def _fill_missing_values(self, df: pd.DataFrame, fill_method: str = 'auto') -> pd.DataFrame:

        if df.empty:
            return df

        # Мапінг колонок до нижнього регістру для спрощення пошуку
        columns_lower = {col.lower(): col for col in df.columns}

        # Визначення колонок за категоріями
        price_cols = []
        volume_cols = []
        trades_cols = []
        other_numeric_cols = []
        non_numeric_cols = []

        # Визначаємо цінові колонки та колонки об'єму
        for col_pattern in ['open', 'high', 'low', 'close', 'vwap']:
            if col_pattern in columns_lower:
                price_cols.append(columns_lower[col_pattern])

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

        # Заповнення цінових колонок
        if price_cols and fill_method in ['auto', 'ffill']:
            df[price_cols] = df[price_cols].fillna(method='ffill')
            # Заповнення початкових NaN значень (якщо є)
            df[price_cols] = df[price_cols].fillna(method='bfill')

        # Заповнення колонок об'єму
        if volume_cols:
            if fill_method in ['auto', 'zero']:
                df[volume_cols] = df[volume_cols].fillna(0)
            elif fill_method == 'ffill':
                df[volume_cols] = df[volume_cols].fillna(method='ffill')
                df[volume_cols] = df[volume_cols].fillna(0)  # Залишкові NaN як нулі

        # Заповнення колонок кількості угод
        if trades_cols:
            if fill_method in ['auto', 'zero']:
                df[trades_cols] = df[trades_cols].fillna(0)
            elif fill_method == 'ffill':
                df[trades_cols] = df[trades_cols].fillna(method='ffill')
                df[trades_cols] = df[trades_cols].fillna(0)

        # Заповнення інших числових колонок
        for col in remaining_numeric:
            col_lower = col.lower()
            if fill_method == 'auto':
                if any(x in col_lower for x in ['count', 'number', 'trades', 'qty', 'quantity', 'amount']):
                    df[col] = df[col].fillna(0)  # Лічильники заповнюємо нулями
                elif any(x in col_lower for x in ['funding', 'rate', 'fee']):
                    df[col] = df[col].fillna(method='ffill').fillna(0)  # Ставки заповнюємо попередніми або нулями
                else:
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
            elif fill_method == 'ffill':
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            elif fill_method == 'zero':
                df[col] = df[col].fillna(0)

        # Заповнення не-числових колонок
        if non_numeric_cols and fill_method in ['auto', 'ffill', 'bfill']:
            df[non_numeric_cols] = df[non_numeric_cols].fillna(method='ffill').fillna(method='bfill')

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
                result_df[high_low_range_pct_col] = result_df[high_low_range_col] / df[col]
                self.logger.info(f"make_stationary: Створено колонку {high_low_range_pct_col}")

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

            # Видалення рядків з NaN після диференціювання (ВАЖЛИВО: це змінює кількість рядків)
            rows_before = len(result_df)
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

    def check_stationarity(self, data: pd.DataFrame, column='close_diff') -> dict:

        from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf

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
        if data[column_to_use].isna().any():
            self.logger.warning(f"check_stationarity: Виявлено NaN значення у колонці {column_to_use}")
            clean_data = data[column_to_use].dropna()
            if len(clean_data) < 2:
                error_msg = f"check_stationarity: Недостатньо даних для перевірки стаціонарності після видалення NaN значень"
                self.logger.error(error_msg)
                return {'error': error_msg, 'is_stationary': False}
        else:
            clean_data = data[column_to_use]

        self.logger.info(
            f"check_stationarity: Перевірка стаціонарності для колонки {column_to_use} ({len(clean_data)} точок)")

        # Тест Дікі-Фуллера (нульова гіпотеза: ряд нестаціонарний)
        try:
            adf_result = adfuller(clean_data)
            results['adf_test'] = {
                'test_statistic': adf_result[0],
                'p-value': adf_result[1],
                'is_stationary': adf_result[1] < 0.05,
                'critical_values': adf_result[4]
            }
            self.logger.info(f"check_stationarity: ADF тест завершено, p-value={adf_result[1]:.4f}")
        except Exception as e:
            error_msg = f"check_stationarity: Помилка при виконанні ADF тесту: {str(e)}"
            self.logger.error(error_msg)
            results['adf_test'] = {
                'error': error_msg,
                'is_stationary': False
            }

        # KPSS тест (нульова гіпотеза: ряд стаціонарний)
        try:
            kpss_result = kpss(clean_data)
            results['kpss_test'] = {
                'test_statistic': kpss_result[0],
                'p-value': kpss_result[1],
                'is_stationary': kpss_result[1] > 0.05,
                'critical_values': kpss_result[3]
            }
            self.logger.info(f"check_stationarity: KPSS тест завершено, p-value={kpss_result[1]:.4f}")
        except Exception as e:
            error_msg = f"check_stationarity: Помилка при виконанні KPSS тесту: {str(e)}"
            self.logger.warning(error_msg)
            results['kpss_test'] = {
                'error': error_msg,
                'is_stationary': False
            }

        # ACF and PACF для визначення параметрів моделі ARIMA
        try:
            # Максимальну кількість лагів обмежуємо розумним значенням
            max_lags = min(40, int(len(clean_data) * 0.3))

            acf_values = acf(clean_data, nlags=max_lags, fft=True)
            pacf_values = pacf(clean_data, nlags=max_lags)

            # Знаходження значимих лагів (з 95% довірчим інтервалом)
            n = len(clean_data)
            confidence_interval = 1.96 / np.sqrt(n)

            significant_acf_lags = [i for i, v in enumerate(acf_values) if abs(v) > confidence_interval and i > 0]
            significant_pacf_lags = [i for i, v in enumerate(pacf_values) if abs(v) > confidence_interval and i > 0]

            # Якщо немає значимих лагів, використовуємо лаг 1
            suggested_p = min(significant_pacf_lags) if significant_pacf_lags else 1
            suggested_q = min(significant_acf_lags) if significant_acf_lags else 1

            results['acf_pacf'] = {
                'significant_acf_lags': significant_acf_lags,
                'significant_pacf_lags': significant_pacf_lags,
                'suggested_p': suggested_p,
                'suggested_q': suggested_q
            }
            self.logger.info(f"check_stationarity: ACF/PACF аналіз завершено, p={suggested_p}, q={suggested_q}")
        except Exception as e:
            error_msg = f"check_stationarity: Помилка при розрахунку ACF/PACF: {str(e)}"
            self.logger.warning(error_msg)
            results['acf_pacf'] = {
                'error': error_msg
            }

        # Загальний висновок про стаціонарність
        adf_stationary = results.get('adf_test', {}).get('is_stationary', False)
        kpss_stationary = results.get('kpss_test', {}).get('is_stationary', True)

        results['is_stationary'] = adf_stationary and kpss_stationary
        self.logger.info(f"check_stationarity: Загальний висновок про стаціонарність: {results['is_stationary']}")

        return results

    def prepare_arima_data(self, data: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:

        if data.empty:
            self.logger.warning("prepare_arima_data: Отримано порожній DataFrame для підготовки ARIMA даних")
            return pd.DataFrame()

        # Зберігаємо вхідні дані
        self.original_data_map[f'arima_input_{symbol}_{timeframe}'] = data.copy()

        # Перевірка наявності необхідної колонки 'close' з різними варіантами написання
        close_column = self.find_column(data, 'close')
        if not close_column:
            # Пробуємо інші можливі назви для колонки з ціною закриття
            for variant in ['price', 'last', 'last_price', 'close_price']:
                close_column = self.find_column(data, variant)
                if close_column:
                    break

        if not close_column:
            self.logger.error(
                f"prepare_arima_data: Не знайдено колонку з ціною закриття. Доступні колонки: {list(data.columns)}")
            return pd.DataFrame()

        self.logger.info(f"prepare_arima_data: Підготовка ARIMA даних для {symbol} на інтервалі {timeframe}, "
                         f"використовується колонка {close_column}")

        # Копія даних для подальшої обробки
        working_data = data.copy()

        # 1. Перевірка стаціонарності вихідних даних
        self.logger.info(f"prepare_arima_data: Перевірка стаціонарності вихідних даних для {close_column}")
        original_stationarity = self.check_stationarity(working_data, close_column)

        # 2. Перетворення для стаціонарності
        self.logger.info(f"prepare_arima_data: Застосування перетворень для забезпечення стаціонарності")
        stationary_data = self.make_stationary(working_data, columns=[close_column], method='all')

        # Зберігаємо перетворені дані
        self.original_data_map[f'arima_stationary_{symbol}_{timeframe}'] = stationary_data.copy()

        # Базові трансформації для ARIMA
        # Додаємо логарифмічні прибутки, якщо всі значення додатні
        if (working_data[close_column] > 0).all():
            log_return_col = f'{close_column}_log_return'
            stationary_data[log_return_col] = np.log(
                working_data[close_column] / working_data[close_column].shift(1))
            self.logger.info(f"prepare_arima_data: Створено колонку {log_return_col}")

        # 3. Перевірка стаціонарності після різних перетворень
        # Визначаємо назви колонок для перетворених даних
        diff_column = f"{close_column}_diff"
        log_diff_column = f"{close_column}_log_diff"
        log_return_column = f"{close_column}_log_return"

        # Перевіряємо стаціонарність трансформованих даних
        transform_tests = {}

        if diff_column in stationary_data.columns:
            self.logger.info(f"prepare_arima_data: Перевірка стаціонарності для {diff_column}")
            transform_tests['diff'] = self.check_stationarity(stationary_data, diff_column)

        if log_diff_column in stationary_data.columns:
            self.logger.info(f"prepare_arima_data: Перевірка стаціонарності для {log_diff_column}")
            transform_tests['log_diff'] = self.check_stationarity(stationary_data, log_diff_column)

        if log_return_column in stationary_data.columns:
            self.logger.info(f"prepare_arima_data: Перевірка стаціонарності для {log_return_column}")
            transform_tests['log_return'] = self.check_stationarity(stationary_data, log_return_column)

        # 4. Підготовка даних для результату
        arima_data = pd.DataFrame()
        arima_data['open_time'] = working_data.index
        arima_data['original_close'] = working_data[close_column]

        # Додаємо трансформовані дані
        for suffix in ['diff', 'diff2', 'log', 'log_diff', 'pct', 'seasonal_diff', 'combo_diff', 'log_return']:
            col_name = f"{close_column}_{suffix}"
            if col_name in stationary_data.columns:
                arima_data[f"close_{suffix}"] = stationary_data[col_name]
                self.logger.info(f"prepare_arima_data: Додано колонку {col_name} до результату")

        # 5. Вибираємо найкращу трансформацію на основі тестів на стаціонарність
        best_transform = None
        best_adf = 1.0
        best_stationarity = None

        for transform_name, stationarity_test in transform_tests.items():
            if not stationarity_test:
                continue

            current_adf = stationarity_test['adf_test'].get('p-value', 1.0)
            if current_adf < best_adf:
                best_adf = current_adf
                best_transform = f"close_{transform_name}"
                best_stationarity = stationarity_test
                self.logger.info(
                    f"prepare_arima_data: Знайдено кращу трансформацію: {best_transform} з p-value={best_adf:.4f}")

        # Додаємо інформацію про найкращу трансформацію
        if best_stationarity:
            arima_data['adf_pvalue'] = best_adf
            arima_data['kpss_pvalue'] = best_stationarity.get('kpss_test', {}).get('p-value', None)
            arima_data['is_stationary'] = best_stationarity['is_stationary']
            arima_data['best_transform'] = best_transform

            # Серіалізуємо значимі лаги для ACF/PACF
            if 'acf_pacf' in best_stationarity and 'error' not in best_stationarity['acf_pacf']:
                arima_data['significant_lags'] = json.dumps({
                    'acf': best_stationarity['acf_pacf']['significant_acf_lags'],
                    'pacf': best_stationarity['acf_pacf']['significant_pacf_lags']
                })
                self.logger.info("prepare_arima_data: Додано інформацію про значимі лаги")

        # Додаємо метадані
        arima_data['timeframe'] = timeframe
        arima_data['symbol'] = symbol

        # 6. Спроба підбору параметрів ARIMA для найкращого стаціонарного перетворення
        try:
            if best_transform and best_transform in stationary_data.columns and best_stationarity:
                # Вибираємо параметри p, d, q на основі результатів ACF/PACF
                p = best_stationarity.get('acf_pacf', {}).get('suggested_p', 1)

                # Визначаємо порядок інтегрування (d)
                if 'log_return' in best_transform or 'pct' in best_transform:
                    d = 0  # Логарифмічні прибутки часто вже стаціонарні
                else:
                    d = 1  # Для інших трансформацій використовуємо стандартне диференціювання

                q = best_stationarity.get('acf_pacf', {}).get('suggested_q', 1)

                self.logger.info(f"prepare_arima_data: Обрано параметри ARIMA: p={p}, d={d}, q={q}")

                # Підготовка даних для моделі (видалення NaN значень)
                model_data = stationary_data[best_transform].dropna()

                if len(model_data) >= p + q + d + 2:  # Мінімальна необхідна довжина для ARIMA
                    # Підганяємо ARIMA модель
                    self.logger.info(f"prepare_arima_data: Створення ARIMA моделі з параметрами ({p},{d},{q})")
                    model = sm.tsa.ARIMA(model_data, order=(p, d, q))
                    model_fit = model.fit()

                    # Додаємо метрики моделі
                    arima_data['aic_score'] = model_fit.aic
                    arima_data['bic_score'] = model_fit.bic
                    arima_data['residual_variance'] = model_fit.resid.var()
                    arima_data['arima_params'] = json.dumps({'p': p, 'd': d, 'q': q, 'transform': best_transform})
                    self.logger.info(f"prepare_arima_data: ARIMA модель успішно створена з AIC={model_fit.aic:.2f}")

                    # Додаємо інформацію про сезонність, якщо це годинні або щоденні дані
                    if 'h' in timeframe or 'd' in timeframe:
                        # Для годинних даних - можлива денна сезонність (24 години)
                        # Для денних даних - можлива тижнева сезонність (7 днів)
                        seasonality_period = 24 if 'h' in timeframe else 7
                        arima_data['suggested_seasonality'] = seasonality_period
                        arima_data['suggested_sarima_params'] = json.dumps(
                            {'P': 1, 'D': 1, 'Q': 1, 'S': seasonality_period})
                        self.logger.info(
                            f"prepare_arima_data: Додано пропозиції щодо сезонності (період={seasonality_period})")
                else:
                    self.logger.warning(
                        f"prepare_arima_data: Недостатньо даних для підбору ARIMA моделі: {len(model_data)} точок, "
                        f"потрібно мінімум {p + q + d + 2}")
                    arima_data['arima_error'] = "Недостатньо даних для підбору моделі"
            else:
                self.logger.warning("prepare_arima_data: Не вдалося визначити найкращу трансформацію для ARIMA моделі")
                arima_data['arima_error'] = "Не вдалося визначити оптимальну трансформацію"

        except Exception as e:
            self.logger.warning(f"prepare_arima_data: Помилка при підборі ARIMA моделі: {str(e)}")
            arima_data['arima_error'] = str(e)

        self.logger.info(
            f"prepare_arima_data: Підготовка ARIMA даних завершена, створено DataFrame з {len(arima_data)} рядками")
        return arima_data

    def prepare_lstm_data(self, data: pd.DataFrame, symbol: str, timeframe: str,
                          sequence_length: int = 60, forecast_horizons: List[int] = [1, 5, 10]) -> pd.DataFrame:

        if data is None or data.empty:
            self.logger.warning("Отримано порожній DataFrame для підготовки LSTM даних")
            return pd.DataFrame()

        # Зберігаємо оригінальні дані
        self.original_data_map[f'lstm_input_{symbol}_{timeframe}'] = data.copy()

        self.logger.info(f"Підготовка LSTM даних для {symbol} на інтервалі {timeframe}. "
                         f"Початковий DataFrame: {data.shape[0]} рядків, {data.shape[1]} колонок")

        # 1. Додаємо часові ознаки
        try:
            df = self.create_time_features(data)
            self.logger.info(
                f"Часові ознаки додано. DataFrame після додавання: {df.shape[0]} рядків, {df.shape[1]} колонок")
        except Exception as e:
            self.logger.error(f"Помилка при створенні часових ознак: {str(e)}")
            df = data.copy()  # Використовуємо оригінальні дані, якщо додати часові ознаки не вдалося

        # 2. Визначення колонок для обробки (з урахуванням можливих відмінностей у регістрі)
        column_map = {col.lower(): col for col in df.columns}
        self.logger.debug(f"Створено мапінг колонок за нижнім регістром: {column_map}")

        # Базові ринкові дані з різними варіантами назв колонок
        basic_feature_cols = {
            'open': ['open', 'open_price'],
            'high': ['high', 'high_price', 'max_price'],
            'low': ['low', 'low_price', 'min_price'],
            'close': ['close', 'close_price', 'price', 'last', 'last_price'],
            'volume': ['volume', 'base_volume', 'quantity', 'amount']
        }

        feature_cols = []
        found_features = {}  # Для логування знайдених колонок

        # Шукаємо відповідні колонки у різних варіантах
        for base_name, variants in basic_feature_cols.items():
            found = False
            for variant in variants:
                variant_lower = variant.lower()
                if variant_lower in column_map:
                    actual_col_name = column_map[variant_lower]
                    feature_cols.append(actual_col_name)
                    found_features[base_name] = actual_col_name
                    found = True
                    break
            if not found:
                self.logger.warning(f"Колонку '{base_name}' не знайдено в жодному з варіантів {variants}")

        # Логуємо знайдені колонки
        self.logger.info(f"Знайдені базові колонки даних: {found_features}")

        # Додаткові колонки, специфічні для криптовалют
        crypto_specific_cols = ['trades', 'quote_volume', 'taker_buy_volume', 'taker_sell_volume',
                                'number_of_trades', 'taker_buy_base_volume', 'taker_buy_quote_volume']

        additional_features = {}
        for col in crypto_specific_cols:
            col_lower = col.lower()
            if col_lower in column_map:
                actual_col_name = column_map[col_lower]
                feature_cols.append(actual_col_name)
                additional_features[col] = actual_col_name

        if additional_features:
            self.logger.info(f"Знайдені додаткові колонки: {additional_features}")

        if not feature_cols:
            self.logger.error("Не знайдено жодної з необхідних колонок для аналізу")
            return pd.DataFrame()

        # Перевіряємо наявність колонок у DataFrame
        for col in feature_cols:
            if col not in df.columns:
                self.logger.error(f"Критична помилка: колонка '{col}' відсутня в DataFrame")
                return pd.DataFrame()

        # Зберігаємо знайдені колонки в original_data_map для подальшого доступу
        self.original_data_map[f'lstm_feature_cols_{symbol}_{timeframe}'] = feature_cols

        # Часові ознаки
        time_features = [
            'hour_sin', 'hour_cos',
            'day_of_week_sin', 'day_of_week_cos',
            'month_sin', 'month_cos',
            'day_of_month_sin', 'day_of_month_cos'
        ]
        available_time_features = [col for col in time_features if col in df.columns]

        self.logger.info(f"Доступні часові ознаки: {available_time_features}")

        # 3. Перевірка наявності NaN значень перед масштабуванням
        na_counts = df[feature_cols].isna().sum()
        if na_counts.any():
            self.logger.warning(f"Знайдено відсутні значення в колонках: {na_counts[na_counts > 0].to_dict()}")

        # 4. Масштабування даних
        try:
            # Копіюємо дані для безпечного масштабування
            scaling_df = df[feature_cols].copy()

            # Заповнюємо відсутні значення перед масштабуванням
            scaling_df = scaling_df.fillna(method='ffill').fillna(method='bfill').fillna(0)

            # Зберігаємо оригінальні дані перед масштабуванням
            self.original_data_map[f'lstm_before_scaling_{symbol}_{timeframe}'] = scaling_df.copy()

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_values = scaler.fit_transform(scaling_df)

            # Перевірка на валідність масштабованих значень
            if np.isnan(scaled_values).any():
                self.logger.error("Масштабовані дані містять NaN значення. Перевірте вхідні дані.")
                return pd.DataFrame()

            # Створюємо датафрейм з масштабованими значеннями
            scaled_cols = [f"{col}_scaled" for col in feature_cols]
            scaled_features = pd.DataFrame(
                scaled_values,
                columns=scaled_cols,
                index=df.index
            )

            self.logger.info(f"Дані успішно масштабовані. Створено {len(scaled_cols)} масштабованих колонок.")

            # Зберігаємо скалер для подальшого використання
            scaler_key = f"{symbol}_{timeframe}"
            self.scalers[scaler_key] = scaler

            # Зберігаємо параметри скалера для відновлення даних
            scaling_metadata = {
                'feature_names': feature_cols,
                'feature_min_': scaler.min_.tolist(),
                'feature_scale_': scaler.scale_.tolist()
            }

            self.logger.debug(f"Метадані масштабування збережено для {scaler_key}")

        except Exception as e:
            self.logger.error(f"Помилка при масштабуванні даних: {str(e)}")
            return pd.DataFrame()

        # 5. Перевірка достатньої кількості даних
        min_required_length = sequence_length + max(forecast_horizons)
        if len(df) < min_required_length:
            self.logger.warning(f"Недостатньо даних для створення послідовностей: {len(df)} точок, "
                                f"потрібно мінімум {min_required_length}")
            return pd.DataFrame()

        # 6. Визначаємо цільову колонку для прогнозування
        target_col = None
        for variant in ['close', 'price', 'last', 'close_price', 'last_price']:
            variant_lower = variant.lower()
            if variant_lower in column_map:
                target_col = column_map[variant_lower]
                break

        if target_col is None:
            self.logger.error("Не знайдено цільову колонку для прогнозування")
            return pd.DataFrame()

        self.logger.info(f"Цільова колонка для прогнозування: '{target_col}'")

        # 7. Створення послідовностей для обробки
        sequences_data = []
        sequence_counter = 0

        # Обмежуємо індекси, щоб уникнути виходу за межі
        max_idx = len(df) - max(forecast_horizons)
        valid_range = len(df) - sequence_length - max(forecast_horizons) + 1

        self.logger.info(f"Початок створення послідовностей. Можлива кількість послідовностей: {valid_range}")

        try:
            for i in range(valid_range):
                sequence_id = sequence_counter
                sequence_counter += 1

                for pos in range(sequence_length):
                    idx = i + pos
                    if idx >= len(df):
                        self.logger.error(f"Критична помилка індексування: idx={idx}, len(df)={len(df)}")
                        continue

                    row = {
                        'sequence_id': sequence_id,
                        'sequence_position': pos,
                        'open_time': df.index[idx],
                        'timeframe': timeframe,
                        'symbol': symbol,
                        'sequence_length': sequence_length
                    }

                    # Додаємо масштабовані ознаки
                    for col in scaled_features.columns:
                        try:
                            row[col] = scaled_features.iloc[idx][col]
                        except Exception as e:
                            self.logger.error(
                                f"Помилка при доступі до масштабованої колонки {col} в індексі {idx}: {str(e)}")
                            row[col] = np.nan

                    # Додаємо оригінальні немасштабовані значення для референсу
                    for col in feature_cols:
                        try:
                            row[f"{col}_original"] = df[col].iloc[idx]
                        except Exception as e:
                            self.logger.error(
                                f"Помилка при доступі до оригінальної колонки {col} в індексі {idx}: {str(e)}")
                            row[f"{col}_original"] = np.nan

                    # Додаємо цільові значення для різних горизонтів прогнозування
                    for horizon in forecast_horizons:
                        target_idx = i + pos + horizon
                        if target_idx < len(df):
                            try:
                                # Абсолютна ціна закриття
                                row[f'target_close_{horizon}'] = df[target_col].iloc[target_idx]

                                # Тільки для останнього елемента в послідовності додаємо додаткові цільові змінні
                                if pos == sequence_length - 1:
                                    current_price = df[target_col].iloc[idx]
                                    target_price = df[target_col].iloc[target_idx]

                                    if current_price is not None and current_price != 0:
                                        # Відносна зміна ціни
                                        pct_change = (target_price - current_price) / current_price
                                        row[f'target_close_pct_{horizon}'] = pct_change

                                        # Логарифмічна прибутковість
                                        if current_price > 0 and target_price > 0:
                                            log_return = np.log(target_price / current_price)
                                            row[f'target_close_log_return_{horizon}'] = log_return

                                        # Бінарний напрямок руху (1 - зростання, 0 - падіння)
                                        row[f'target_direction_{horizon}'] = 1 if target_price > current_price else 0
                            except Exception as e:
                                self.logger.error(
                                    f"Помилка при розрахунку цільових змінних для горизонту {horizon}: {str(e)}")
                                row[f'target_close_{horizon}'] = np.nan

                    # Додаємо часові ознаки без масштабування (вони вже нормалізовані)
                    for feature in available_time_features:
                        try:
                            row[feature] = df[feature].iloc[idx]
                        except Exception as e:
                            self.logger.error(
                                f"Помилка при доступі до часової ознаки {feature} в індексі {idx}: {str(e)}")
                            row[feature] = np.nan

                    # Додаємо метадані масштабування лише для першого елемента кожної послідовності
                    if pos == 0:
                        row['scaling_metadata'] = json.dumps(scaling_metadata)

                    sequences_data.append(row)

                # Логуємо прогрес кожні 100 послідовностей
                if sequence_id % 100 == 0 and sequence_id > 0:
                    self.logger.debug(f"Створено {sequence_id} послідовностей...")

        except Exception as e:
            self.logger.error(f"Помилка при створенні послідовностей: {str(e)}")
            if sequences_data:  # Якщо вже є якісь дані, продовжуємо з ними
                self.logger.info(f"Продовжуємо з {len(sequences_data)} вже створеними записами")
            else:
                return pd.DataFrame()

        # Створюємо DataFrame з даними послідовностей
        if sequences_data:
            try:
                sequences_df = pd.DataFrame(sequences_data)
                self.logger.info(
                    f"Створено DataFrame з послідовностями: {len(sequences_df)} рядків, {len(sequences_df.columns)} колонок")

                # Зберігаємо фінальний результат
                self.original_data_map[f'lstm_sequences_{symbol}_{timeframe}'] = sequences_df.copy()

                return sequences_df
            except Exception as e:
                self.logger.error(f"Помилка при створенні DataFrame з послідовностей: {str(e)}")
                return pd.DataFrame()
        else:
            self.logger.warning("Не вдалося створити жодної послідовності даних")
            return pd.DataFrame()