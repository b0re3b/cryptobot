import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
from typing import Dict,List
import statsmodels.api as sm


class DataResampler:
    def __init__(self, logger):
        self.logger = logger
        self.scalers = {}
        self.original_data_map = {}

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

        # Перевірка необхідних колонок
        if required_columns is None:
            required_columns = ['open', 'high', 'low', 'close']

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
            resampled = data.resample(pandas_interval).agg(agg_dict)

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
            raise

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

        # Додаємо базові колонки зі словника (незалежно від регістру)
        for base_col_lower, agg_method in base_columns.items():
            if base_col_lower in columns_lower_map:
                actual_col = columns_lower_map[base_col_lower]
                agg_dict[actual_col] = agg_method

        # Обробка всіх числових колонок, які ще не додані
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in agg_dict:  # Якщо колонку ще не додано
                col_lower = col.lower()
                if any(x in col_lower for x in ['count', 'number', 'trades', 'qty', 'quantity', 'amount']):
                    agg_dict[col] = 'sum'
                elif any(x in col_lower for x in ['id', 'code', 'identifier']):
                    agg_dict[col] = 'last'  # Для ідентифікаторів беремо останнє значення
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
        other_numeric_cols = []
        non_numeric_cols = []

        # Визначаємо цінові колонки та колонки об'єму
        for col_pattern in ['open', 'high', 'low', 'close']:
            if col_pattern in columns_lower:
                price_cols.append(columns_lower[col_pattern])

        for col_pattern in ['volume']:
            if col_pattern in columns_lower:
                volume_cols.append(columns_lower[col_pattern])

        # Виділяємо всі інші числові колонки
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        remaining_numeric = [col for col in numeric_cols if col not in price_cols and col not in volume_cols]

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

        # Заповнення інших числових колонок
        for col in remaining_numeric:
            col_lower = col.lower()
            if fill_method == 'auto':
                if any(x in col_lower for x in ['count', 'number', 'trades', 'qty', 'quantity', 'amount']):
                    df[col] = df[col].fillna(0)  # Лічильники заповнюємо нулями
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
            's': 'S',
            'm': 'T',
            'h': 'H',
            'd': 'D',
            'w': 'W',
            'M': 'M',
        }

        import re
        match = re.match(r'(\d+)([smhdwM])', timeframe)
        if not match:
            raise ValueError(f"Неправильний формат інтервалу: {timeframe}")

        number, unit = match.groups()

        # Перевірка, чи число додатне
        if int(number) <= 0:
            raise ValueError(f"Інтервал повинен бути додатнім: {timeframe}")

        if unit in interval_map:
            return f"{number}{interval_map[unit]}"
        else:
            raise ValueError(f"Непідтримувана одиниця часу: {unit}")

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

        # Додаємо is_weekend флаг
        df['is_weekend'] = df['day_of_week'].isin([5, 6])  # 5=Saturday, 6=Sunday

        # Додаємо торгову сесію
        df['session'] = 'Unknown'
        df.loc[(df['hour'] >= 0) & (df['hour'] < 8), 'session'] = 'Asia'
        df.loc[(df['hour'] >= 8) & (df['hour'] < 16), 'session'] = 'Europe'
        df.loc[(df['hour'] >= 16) & (df['hour'] < 24), 'session'] = 'US'

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

        if columns is None:
            columns = ['close']

        df = data.copy()

        # Зберігаємо оригінальні дані для кожної колонки окремо
        for col in columns:
            col_key = f"{col}_original"
            if col in df.columns:
                self.original_data_map[col_key] = df[col].copy()
            else:
                # Пошук колонки у різних регістрах
                found = False
                for alt_col in df.columns:
                    if alt_col.lower() == col.lower():
                        self.original_data_map[col_key] = df[alt_col].copy()
                        found = True
                        break

                if not found:
                    self.logger.warning(f"Колонка '{col}' відсутня у DataFrame і буде пропущена")
                    continue

        # Обробка колонок і створення стаціонарних версій
        for col in columns:
            if col not in df.columns:
                # Пошук колонки у різних регістрах
                found = False
                for alt_col in df.columns:
                    if alt_col.lower() == col.lower():
                        col = alt_col
                        found = True
                        break

                if not found:
                    self.logger.warning(f"Колонка '{col}' відсутня у DataFrame і буде пропущена")
                    continue

            # Базове диференціювання
            if method == 'diff' or method == 'all':
                df[f'{col}_diff'] = df[col].diff(order)

                # Додаємо диференціювання 2-го порядку
                df[f'{col}_diff2'] = df[col].diff().diff()

                # Додаємо сезонне диференціювання якщо потрібно
                if seasonal_order:
                    df[f'{col}_seasonal_diff'] = df[col].diff(seasonal_order)
                    # Комбіноване сезонне + звичайне диференціювання
                    df[f'{col}_combo_diff'] = df[col].diff(seasonal_order).diff()

            # Логарифмічне перетворення
            if method == 'log' or method == 'all':
                # Перевірка на відсутність нульових чи від'ємних значень
                if (df[col] > 0).all():
                    df[f'{col}_log'] = np.log(df[col])
                    # Логарифм + диференціювання
                    df[f'{col}_log_diff'] = df[f'{col}_log'].diff(order)
                else:
                    self.logger.warning(
                        f"Колонка {col} містить нульові або від'ємні значення. Логарифмічне перетворення пропущено.")

            # Відсоткова зміна
            if method == 'pct_change' or method == 'all':
                df[f'{col}_pct'] = df[col].pct_change(order)

        # Видалення рядків з NaN після диференціювання
        rows_before = len(df)
        df = df.dropna()
        rows_after = len(df)

        if rows_before > rows_after:
            self.logger.info(f"Видалено {rows_before - rows_after} рядків з NaN після диференціювання")

        return df

    def check_stationarity(self, data: pd.DataFrame, column='close_diff') -> dict:

        from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf

        results = {}

        # Перевірка наявності даних
        if data.empty:
            error_msg = "Отримано порожній DataFrame для перевірки стаціонарності"
            self.logger.error(error_msg)
            return {'error': error_msg, 'is_stationary': False}

        # Перевірка наявності колонки (з урахуванням регістру)
        column_to_use = None
        if column in data.columns:
            column_to_use = column
        else:
            # Пошук колонки у різних регістрах
            for col in data.columns:
                if col.lower() == column.lower():
                    column_to_use = col
                    break

        if column_to_use is None:
            error_msg = f"Колонка '{column}' відсутня у DataFrame. Доступні колонки: {list(data.columns)}"
            self.logger.error(error_msg)
            return {'error': error_msg, 'is_stationary': False}

        # Перевірка на відсутні значення
        if data[column_to_use].isna().any():
            clean_data = data[column_to_use].dropna()
            if len(clean_data) < 2:
                error_msg = f"Недостатньо даних для перевірки стаціонарності після видалення NaN значень"
                self.logger.error(error_msg)
                return {'error': error_msg, 'is_stationary': False}
        else:
            clean_data = data[column_to_use]

        # Тест Дікі-Фуллера (нульова гіпотеза: ряд нестаціонарний)
        try:
            adf_result = adfuller(clean_data)
            results['adf_test'] = {
                'test_statistic': adf_result[0],
                'p-value': adf_result[1],
                'is_stationary': adf_result[1] < 0.05,
                'critical_values': adf_result[4]
            }
        except Exception as e:
            error_msg = f"Помилка при виконанні ADF тесту: {str(e)}"
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
        except Exception as e:
            error_msg = f"Помилка при виконанні KPSS тесту: {str(e)}"
            self.logger.warning(error_msg)
            results['kpss_test'] = {
                'error': error_msg,
                'is_stationary': False
            }

        # ACF and PACF для визначення параметрів моделі ARIMA
        try:
            acf_values = acf(clean_data, nlags=40)
            pacf_values = pacf(clean_data, nlags=40)

            # Знаходження значимих лагів (з 95% довірчим інтервалом)
            n = len(clean_data)
            confidence_interval = 1.96 / np.sqrt(n)

            significant_acf_lags = [i for i, v in enumerate(acf_values) if abs(v) > confidence_interval and i > 0]
            significant_pacf_lags = [i for i, v in enumerate(pacf_values) if abs(v) > confidence_interval and i > 0]

            results['acf_pacf'] = {
                'significant_acf_lags': significant_acf_lags,
                'significant_pacf_lags': significant_pacf_lags,
                'suggested_p': min(significant_pacf_lags) if significant_pacf_lags else 0,
                'suggested_q': min(significant_acf_lags) if significant_acf_lags else 0
            }
        except Exception as e:
            error_msg = f"Помилка при розрахунку ACF/PACF: {str(e)}"
            self.logger.warning(error_msg)
            results['acf_pacf'] = {
                'error': error_msg
            }

        # Загальний висновок про стаціонарність
        results['is_stationary'] = results.get('adf_test', {}).get('is_stationary', False) and \
                                   (results.get('kpss_test', {}).get('is_stationary', False) or
                                    'error' in results.get('kpss_test', {}))

        # Збереження інформації про дані
        results['data_info'] = {
            'column': column_to_use,
            'data_length': len(clean_data),
            'has_nan': data[column_to_use].isna().any()
        }

        return results
    def prepare_arima_data(self, data: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:

        if data.empty:
            self.logger.warning("Отримано порожній DataFrame для підготовки ARIMA даних")
            return pd.DataFrame()

        # Нормалізація імен колонок до нижнього регістру для пошуку
        column_map = {col.lower(): col for col in data.columns}

        # Перевірка наявності необхідної колонки 'close'
        if 'close' not in column_map:
            self.logger.error(f"Колонка 'close' відсутня у DataFrame. Доступні колонки: {list(data.columns)}")
            return pd.DataFrame()

        close_column = column_map['close']

        self.logger.info(f"Підготовка ARIMA даних для {symbol} на інтервалі {timeframe}")

        # Копія даних для подальшої обробки
        working_data = data.copy()

        # 1. Перевірка стаціонарності вихідних даних
        original_stationarity = self.check_stationarity(working_data, close_column)

        # 2. Перетворення для стаціонарності
        stationary_data = self.make_stationary(working_data, columns=[close_column], method='all')

        # 3. Перевірка стаціонарності після різних перетворень
        diff_column = f"{close_column}_diff"
        log_diff_column = f"{close_column}_log_diff"

        diff_stationarity = self.check_stationarity(stationary_data,
                                                    diff_column) if diff_column in stationary_data.columns else None
        log_diff_stationarity = self.check_stationarity(stationary_data,
                                                        log_diff_column) if log_diff_column in stationary_data.columns else None

        # 4. Підготовка даних для результату
        arima_data = pd.DataFrame()
        arima_data['open_time'] = working_data.index
        arima_data['original_close'] = working_data[close_column]

        # Додаємо трансформовані дані
        for suffix in ['diff', 'diff2', 'log', 'log_diff', 'pct', 'seasonal_diff', 'combo_diff']:
            col_name = f"{close_column}_{suffix}"
            if col_name in stationary_data.columns:
                arima_data[f"close_{suffix}"] = stationary_data[col_name]

        # 5. Додаємо результати тестів на стаціонарність (якщо доступні)
        if diff_stationarity:
            arima_data['adf_pvalue'] = diff_stationarity['adf_test']['p-value']
            arima_data['kpss_pvalue'] = diff_stationarity.get('kpss_test', {}).get('p-value', None)
            arima_data['is_stationary'] = diff_stationarity['is_stationary']

            # Серіалізуємо значимі лаги для ACF/PACF
            if 'acf_pacf' in diff_stationarity and 'error' not in diff_stationarity['acf_pacf']:
                arima_data['significant_lags'] = json.dumps({
                    'acf': diff_stationarity['acf_pacf']['significant_acf_lags'],
                    'pacf': diff_stationarity['acf_pacf']['significant_pacf_lags']
                })

        # Додаємо метадані
        arima_data['timeframe'] = timeframe
        arima_data['symbol'] = symbol

        # 6. Спроба підбору параметрів ARIMA для найкращого стаціонарного перетворення
        try:
            # Вибираємо найкраще перетворення на основі ADF p-value
            best_transform = 'close_diff'
            best_adf = diff_stationarity['adf_test']['p-value'] if diff_stationarity else 1.0

            if log_diff_stationarity and log_diff_stationarity['adf_test']['p-value'] < best_adf:
                best_transform = 'close_log_diff'
                best_adf = log_diff_stationarity['adf_test']['p-value']

            # Вибираємо параметри p, d, q на основі результатів ACF/PACF
            best_stationarity = diff_stationarity if best_transform == 'close_diff' else log_diff_stationarity

            p = best_stationarity.get('acf_pacf', {}).get('suggested_p', 1) if best_stationarity else 1
            d = 1  # Ми вже диференціювали ряд
            q = best_stationarity.get('acf_pacf', {}).get('suggested_q', 1) if best_stationarity else 1

            # Обмежуємо значення параметрів для уникнення надто складних моделей
            p = min(p, 5)
            q = min(q, 5)

            # Підготовка даних для моделі (видалення NaN значень)
            model_data = stationary_data[best_transform].dropna()

            if len(model_data) >= p + q + 2:  # Мінімальна необхідна довжина для ARIMA
                # Підганяємо ARIMA модель
                model = sm.tsa.ARIMA(model_data, order=(p, 0, q))
                model_fit = model.fit()

                # Додаємо метрики моделі
                arima_data['aic_score'] = model_fit.aic
                arima_data['bic_score'] = model_fit.bic
                arima_data['residual_variance'] = model_fit.resid.var()
                arima_data['arima_params'] = json.dumps({'p': p, 'd': d, 'q': q, 'transform': best_transform})

                self.logger.info(f"ARIMA модель успішно створена для {symbol} з параметрами ({p},{d},{q})")
            else:
                self.logger.warning(
                    f"Недостатньо даних для підбору ARIMA моделі: {len(model_data)} точок, потрібно мінімум {p + q + 2}")
                arima_data['arima_error'] = "Недостатньо даних для підбору моделі"
        except Exception as e:
            self.logger.warning(f"Помилка при підборі ARIMA моделі: {str(e)}")
            arima_data['arima_error'] = str(e)

        return arima_data

    def prepare_lstm_data(self, data: pd.DataFrame, symbol: str, timeframe: str,
                          sequence_length: int = 60, forecast_horizons: List[int] = [1, 5, 10]) -> pd.DataFrame:

        if data.empty:
            self.logger.warning("Отримано порожній DataFrame для підготовки LSTM даних")
            return pd.DataFrame()

        self.logger.info(f"Підготовка LSTM даних для {symbol} на інтервалі {timeframe}")

        # 1. Додаємо часові ознаки
        df = self.create_time_features(data)

        # 2. Визначення колонок для обробки (з урахуванням можливих відмінностей у регістрі)
        column_map = {col.lower(): col for col in df.columns}

        # Базові ринкові дані
        feature_cols = []
        for basic_col in ['open', 'high', 'low', 'close', 'volume']:
            if basic_col in column_map:
                feature_cols.append(column_map[basic_col])

        if not feature_cols:
            self.logger.error("Не знайдено жодної з необхідних колонок (open, high, low, close, volume)")
            return pd.DataFrame()

        # Часові ознаки
        time_features = [
            'hour_sin', 'hour_cos',
            'day_of_week_sin', 'day_of_week_cos',
            'month_sin', 'month_cos',
            'day_of_month_sin', 'day_of_month_cos'
        ]
        available_time_features = [col for col in time_features if col in df.columns]

        # 3. Масштабування даних
        try:
            # Заповнюємо відсутні значення перед масштабуванням
            features_df = df[feature_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_values = scaler.fit_transform(features_df)

            # Створюємо датафрейм з масштабованими значеннями
            scaled_features = pd.DataFrame(
                scaled_values,
                columns=[f"{col}_scaled" for col in feature_cols],
                index=df.index
            )

            # Зберігаємо скалер для подальшого використання
            scaler_key = f"{symbol}_{timeframe}"
            self.scalers[scaler_key] = scaler

            # Зберігаємо параметри скалера для відновлення даних
            scaling_metadata = {
                'feature_names': feature_cols,
                'feature_min_': scaler.min_.tolist(),
                'feature_scale_': scaler.scale_.tolist()
            }

        except Exception as e:
            self.logger.error(f"Помилка при масштабуванні даних: {str(e)}")
            return pd.DataFrame()

        # 4. Перевірка достатньої кількості даних
        if len(df) < sequence_length + max(forecast_horizons):
            self.logger.warning(f"Недостатньо даних для створення послідовностей: {len(df)} точок, "
                                f"потрібно мінімум {sequence_length + max(forecast_horizons)}")
            return pd.DataFrame()

        # 5. Створення послідовностей для обробки
        sequences_data = []
        sequence_counter = 0

        for i in range(len(df) - sequence_length - max(forecast_horizons) + 1):
            sequence_id = sequence_counter
            sequence_counter += 1

            for pos in range(sequence_length):
                idx = i + pos
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
                    row[col] = scaled_features.iloc[idx][col]

                # Додаємо оригінальні немасштабовані значення для референсу
                for col in feature_cols:
                    row[f"{col}_original"] = df[col].iloc[idx]

                # Додаємо цільові значення для різних горизонтів прогнозування
                for horizon in forecast_horizons:
                    target_idx = i + pos + horizon
                    if target_idx < len(df):
                        for target_col in ['close']:
                            if target_col in column_map:
                                col_name = column_map[target_col]
                                row[f'target_{target_col}_{horizon}'] = df[col_name].iloc[target_idx]

                                # Додаємо також відносну зміну ціни для кожного горизонту
                                if pos == sequence_length - 1:  # Тільки для останнього елемента в послідовності
                                    current_price = df[col_name].iloc[idx]
                                    target_price = df[col_name].iloc[target_idx]

                                    if current_price != 0:
                                        pct_change = (target_price - current_price) / current_price
                                        row[f'target_{target_col}_pct_{horizon}'] = pct_change

                # Додаємо часові ознаки без масштабування (вони вже нормалізовані)
                for feature in available_time_features:
                    row[feature] = df[feature].iloc[idx]

                # Додаємо метадані масштабування лише для першого елемента кожної послідовності
                if pos == 0:
                    row['scaling_metadata'] = json.dumps(scaling_metadata)

                sequences_data.append(row)

        # Створюємо DataFrame з даними послідовностей
        sequences_df = pd.DataFrame(sequences_data)

        self.logger.info(f"Створено {sequence_counter} послідовностей для {symbol} на інтервалі {timeframe}")

        return sequences_df