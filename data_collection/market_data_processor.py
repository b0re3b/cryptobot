import traceback
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any

from pandas import DataFrame

from data_collection.AnomalyDetector import AnomalyDetector
from data_collection.DataCleaner import DataCleaner
from data_collection.DataResampler import DataResampler
from data_collection.DataStorageManager import DataStorageManager
from data.db import DatabaseManager


class MarketDataProcessor:


    def __init__(self, log_level=logging.INFO):

        self.log_level = log_level
        logging.basicConfig(level=self.log_level)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Ініціалізація класу...")

        # Initialize dependency classes
        self.data_cleaner = DataCleaner(logger=self.logger)
        self.data_resampler = DataResampler(logger=self.logger)
        self.data_storage = DataStorageManager(logger=self.logger)
        self.anomaly_detector = AnomalyDetector(logger=self.logger)
        self.db_manager = DatabaseManager()
        self.supported_symbols = self.db_manager.supported_symbols

        self.ready = True
        self.filtered_data = None
        self.orderbook_statistics = None

    def align_time_series(self, data_list: List[pd.DataFrame],
                          reference_index: int = 0) -> List[pd.DataFrame]:

        if not data_list:
            self.logger.warning("Порожній список DataFrame для вирівнювання")
            return []

        if reference_index < 0 or reference_index >= len(data_list):
            self.logger.error(f"Невірний reference_index: {reference_index}. Має бути від 0 до {len(data_list) - 1}")
            reference_index = 0

        # Підготовка DataFrames
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
                    time_cols = [col for col in df_copy.columns if
                                 any(x in col.lower() for x in ['time', 'date', 'timestamp'])]
                    if time_cols:
                        df_copy[time_cols[0]] = pd.to_datetime(df_copy[time_cols[0]], errors='coerce')
                        df_copy.set_index(time_cols[0], inplace=True)
                        df_copy = df_copy.loc[df_copy.index.notna()]  # Видалення рядків з невалідними датами
                    else:
                        self.logger.error(f"Неможливо конвертувати DataFrame {i}: не знайдено часову колонку")
                        processed_data_list.append(pd.DataFrame())
                        continue
                except Exception as e:
                    self.logger.error(f"Помилка при конвертації індексу для DataFrame {i}: {str(e)}")
                    processed_data_list.append(pd.DataFrame())
                    continue

            # Сортування за часовим індексом, якщо потрібно
            if not df_copy.index.is_monotonic_increasing:
                df_copy = df_copy.sort_index()

            processed_data_list.append(df_copy)

        # Перевірка еталонного DataFrame
        reference_df = processed_data_list[reference_index]
        if reference_df is None or reference_df.empty:
            self.logger.error("Еталонний DataFrame є порожнім")
            return processed_data_list

        # Знаходження спільного часового діапазону
        all_start_times = [df.index.min() for df in processed_data_list if not df.empty]
        all_end_times = [df.index.max() for df in processed_data_list if not df.empty]

        if not all_start_times or not all_end_times:
            self.logger.error("Неможливо визначити спільний часовий діапазон")
            return processed_data_list

        common_start = max(all_start_times)
        common_end = min(all_end_times)

        self.logger.info(f"Визначено спільний часовий діапазон: {common_start} - {common_end}")

        if common_start > common_end:
            self.logger.error("Немає спільного часового діапазону між DataFrame")
            return processed_data_list

        # Створення спільної часової сітки
        try:
            # Спроба визначити частоту еталонного DataFrame
            reference_freq = pd.infer_freq(reference_df.index)

            if not reference_freq:
                self.logger.warning("Не вдалося визначити частоту reference DataFrame. Визначення вручну.")
                # Розрахунок медіанної різниці часових міток
                time_diffs = reference_df.index.to_series().diff().dropna()
                if not time_diffs.empty:
                    median_diff = time_diffs.median()
                    # Конвертація до рядка частоти pandas
                    if median_diff.seconds == 60:
                        reference_freq = '1min'
                    elif median_diff.seconds == 300:
                        reference_freq = '5min'
                    elif median_diff.seconds == 900:
                        reference_freq = '15min'
                    elif median_diff.seconds == 1800:
                        reference_freq = '30min'
                    elif median_diff.seconds == 3600:
                        reference_freq = '1H'
                    elif median_diff.seconds == 14400:
                        reference_freq = '4H'
                    elif median_diff.days == 1:
                        reference_freq = '1D'
                    else:
                        # Використовуємо кількість секунд як частоту
                        total_seconds = median_diff.total_seconds()
                        reference_freq = f"{int(total_seconds)}S"

                    self.logger.info(f"Визначено частоту: {reference_freq}")
                else:
                    self.logger.error("Не вдалося визначити частоту. Повертаємо оригінальні DataFrame")
                    return processed_data_list

            # Створення нового індексу з використанням визначеної частоти
            reference_subset = reference_df.loc[(reference_df.index >= common_start) &
                                                (reference_df.index <= common_end)]
            common_index = reference_subset.index

            # Якщо частота визначена, перестворимо індекс для забезпечення регулярності
            if reference_freq:
                try:
                    common_index = pd.date_range(start=common_start, end=common_end, freq=reference_freq)
                except pd.errors.OutOfBoundsDatetime:
                    self.logger.warning("Помилка створення date_range. Використання оригінального індексу.")

            aligned_data_list = []

            # Вирівнювання всіх DataFrame до спільного індексу
            for i, df in enumerate(processed_data_list):
                if df.empty:
                    aligned_data_list.append(df)
                    continue

                self.logger.info(f"Вирівнювання DataFrame {i} до спільного індексу")

                # Якщо це еталонний DataFrame
                if i == reference_index:
                    # Використовуємо оригінальний індекс, обмежений спільним діапазоном
                    df_aligned = df.loc[(df.index >= common_start) & (df.index <= common_end)]
                    # Перевірка, чи потрібно перестворити індекс з визначеною частотою
                    if len(df_aligned.index) != len(common_index):
                        self.logger.info(f"Перестворення індексу для еталонного DataFrame {i}")
                        df_aligned = df_aligned.reindex(common_index)
                else:
                    # Для інших DataFrame - вирівнюємо до спільного індексу
                    df_aligned = df.reindex(common_index)

                # Інтерполяція числових даних
                numeric_cols = df_aligned.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    for col in numeric_cols:
                        # Обробка кожної колонки окремо
                        if df_aligned[col].isna().sum() > 0:
                            try:
                                # Спочатку спроба інтерполяції за часом
                                df_aligned[col] = df_aligned[col].interpolate(method='time')
                                # Якщо залишились NA на краях, використовуємо заповнення вперед/назад
                                if df_aligned[col].isna().sum() > 0:
                                    df_aligned[col] = df_aligned[col].fillna(method='ffill').fillna(method='bfill')
                            except Exception as e:
                                self.logger.warning(f"Помилка інтерполяції колонки {col}: {str(e)}")
                                # Спробуємо простіший метод
                                df_aligned[col] = df_aligned[col].interpolate(method='linear').fillna(
                                    method='ffill').fillna(method='bfill')

                # Перевірка та звіт про відсутні значення
                missing_values = df_aligned.isna().sum().sum()
                if missing_values > 0:
                    self.logger.warning(
                        f"Після вирівнювання DataFrame {i} залишилося {missing_values} відсутніх значень")

                aligned_data_list.append(df_aligned)

            return aligned_data_list

        except Exception as e:
            self.logger.error(f"Помилка при вирівнюванні часових рядів: {str(e)}")
            self.logger.error(traceback.format_exc())
            return processed_data_list

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
            try:
                # Перевірка та локалізація часового індексу
                if data.index.tz is None:
                    self.logger.info("Локалізація часового індексу до UTC для групування")
                    try:
                        # Безпечна локалізація
                        data = data.copy()
                        data.index = data.index.tz_localize('UTC', ambiguous='NaT')
                        # Прибираємо рядки з NaT
                        mask = data.index.isna()
                        if mask.any():
                            self.logger.warning(f"Знайдено {mask.sum()} неоднозначних часових міток. Видаляємо.")
                            data = data[~mask]
                    except pd.errors.OutOfBoundsDatetime:
                        self.logger.warning("Помилка локалізації часового індексу. Продовжуємо без локалізації.")

                # Безпечне групування з перевіркою на порожні групи
                period_groups = data.groupby(pd.Grouper(freq=time_period, dropna=True))

                result_dfs = []

                for period, group in period_groups:
                    if group.empty:
                        continue

                    # Перевірка наявності даних у групі
                    if len(group) < 2:
                        self.logger.info(f"Пропуск періоду {period}: недостатньо даних")
                        continue

                    period_profile = self._create_volume_profile(group, bins, price_col, volume_col)
                    if not period_profile.empty:
                        period_profile['period'] = period
                        result_dfs.append(period_profile)

                if result_dfs:
                    # Сортування за періодом перед об'єднанням
                    result = pd.concat(result_dfs)
                    # Перетворення колонки period на DatetimeIndex, якщо це можливо
                    if 'period' in result.columns and not result['period'].isna().any():
                        result['period'] = pd.to_datetime(result['period'])
                        # Сортування за періодом
                        result = result.sort_values('period')
                    return result
                else:
                    self.logger.warning("Не вдалося створити часовий профіль об'єму")
                    # Спробуємо створити загальний профіль як запасний варіант
                    return self._create_volume_profile(data, bins, price_col, volume_col)

            except Exception as e:
                self.logger.error(f"Помилка при створенні часового профілю об'єму: {str(e)}")
                self.logger.error(traceback.format_exc())
                # У випадку помилки спробуємо створити звичайний профіль
                return self._create_volume_profile(data, bins, price_col, volume_col)
        else:
            return self._create_volume_profile(data, bins, price_col, volume_col)

    def _create_volume_profile(self, data: pd.DataFrame, bins: int,
                               price_col: str, volume_col: str) -> pd.DataFrame:

        price_min = data[price_col].min()
        price_max = data[price_col].max()

        # Додаємо невеликий буфер, якщо ціни майже ідентичні
        price_range = price_max - price_min
        if price_range == 0:
            # Якщо всі ціни однакові, створюємо штучний діапазон
            price_min = price_max * 0.99  # на 1% менше
            price_max = price_max * 1.01  # на 1% більше
            price_range = price_max - price_min

        # Обчислення ефективної кількості бінів
        min_bin_width = price_range * 0.001  # мінімальна ширина біну 0.1% від діапазону
        effective_bins = max(min(bins, int(price_range / min_bin_width)), 2)

        if effective_bins < 2:
            self.logger.warning(f"Неможливо створити профіль об'єму. Діапазон цін занадто малий.")
            return pd.DataFrame()

        self.logger.info(f"Створення профілю об'єму з {effective_bins} ціновими рівнями")

        try:
            # Створення бінів з урахуванням нового ефективного діапазону
            bin_edges = np.linspace(price_min, price_max, effective_bins + 1)
            bin_width = (price_max - price_min) / effective_bins

            bin_labels = list(range(effective_bins))
            data = data.copy()

            # Виправлено SettingWithCopyWarning
            data.loc[:, 'price_bin'] = pd.cut(
                data[price_col],
                bins=bin_edges,
                labels=bin_labels,
                include_lowest=True
            )

            # Виправлено FutureWarning — додано observed=True
            volume_profile = data.groupby('price_bin', observed=True).agg({
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

    def add_time_features_safely(self, data: pd.DataFrame, tz: str = 'UTC') -> pd.DataFrame:

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

    def resample_data(self, data: pd.DataFrame, target_interval: str) -> pd.DataFrame:

        return self.data_resampler.resample_data(data, target_interval=target_interval)

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

    def save_volume_profile_to_db(self, data: pd.DataFrame, symbol: str, timeframe: str) -> bool:
        if data.empty:
            self.logger.warning("Спроба зберегти порожній профіль об'єму")
            return False

        try:
            for _, row in data.iterrows():
                time_bucket = row.get('period') if pd.notna(row.get('period')) else row.name

                profile_data = {
                    'interval': timeframe,
                    'time_bucket': time_bucket,
                    'price_bin_start': float(row.get('bin_lower')),
                    'price_bin_end': float(row.get('bin_upper')),
                    'volume': float(row['volume'])
                }

                # Конвертуємо numpy типи
                profile_data = {k: v.item() if isinstance(v, np.generic) else v for k, v in profile_data.items()}

                self.db_manager.insert_volume_profile(symbol, profile_data)

            return True
        except Exception as e:
            self.logger.error(f"Помилка при збереженні профілю об'єму: {e}")
            return False
    def save_btc_lstm_sequence(self, data: pd.DataFrame, symbol: str, timeframe: str, **kwargs) -> bool:

        return self.data_storage.save_lstm_sequence(data, symbol, timeframe)

    def save_btc_arima_data(self, data: pd.DataFrame, timeframe:str, symbol: str, **kwargs) -> bool:

        return self.data_storage.save_arima_data(data, symbol, timeframe)

    def load_lstm_sequence(self, symbol: str, **kwargs) -> pd.DataFrame:

        return self.data_storage.load_lstm_sequence(symbol, **kwargs)

    def load_arima_data(self, symbol: str, **kwargs) -> pd.DataFrame:

        return self.data_storage.load_arima_data(symbol, **kwargs)



    def preprocess_pipeline(self, data: pd.DataFrame,
                            steps: Optional[List[Dict]] = None,
                            symbol: Optional[str] = None,
                            interval: Optional[str] = None) -> pd.DataFrame:

        if data.empty:
            self.logger.warning("Отримано порожній DataFrame для обробки в конвеєрі")
            return data

        if steps is None:
            steps = [
                {'name': 'remove_duplicate_timestamps', 'params': {}},
                {'name': 'clean_data', 'params': {'remove_outliers': True, 'fill_missing': True}},
                {'name': 'handle_missing_values', 'params': {
                    'method': 'interpolate',
                    'fetch_missing': True
                }}
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

                # Додаємо symbol та interval якщо метод підтримує їх
                if step_name == 'handle_missing_values':
                    step_params['symbol'] = symbol
                    step_params['timeframe'] = interval

                if step_name in ['normalize_data', 'detect_outliers', 'detect_zscore_outliers',
                                 'detect_iqr_outliers', 'detect_isolation_forest_outliers',
                                 'validate_data_integrity', 'detect_outliers_essemble']:
                    result, _ = method(result, **step_params)
                else:
                    result = method(result, **step_params)

                self.logger.info(
                    f"Крок {step_idx}: '{step_name}' завершено. Результат: {len(result)} рядків, {len(result.columns)} колонок")

            except Exception as e:
                self.logger.error(f"Помилка на кроці {step_idx}: '{step_name}': {str(e)}")
                self.logger.error(traceback.format_exc())

        self.logger.info(
            f"Конвеєр обробки даних завершено. Початково: {len(data)} рядків, {len(data.columns)} колонок. "
            f"Результат: {len(result)} рядків, {len(result.columns)} колонок.")

        return result

    def process_market_data(self, symbol: str, timeframe: str, start_date: Optional[str] = None,
                            end_date: Optional[str] = None, save_results: bool = True) -> Dict[str, pd.DataFrame]:

        self.logger.info(f"Початок комплексної обробки даних для {symbol} ({timeframe})")
        results = {}

        if symbol not in self.supported_symbols:
            self.logger.error(f"Символ {symbol} не підтримується")
            return results

        # Визначення базових та похідних таймфреймів
        base_timeframes = ['1m', '1h', '1d']
        derived_timeframes = ['4h', '1w']

        # 1. Завантаження даних
        if timeframe in base_timeframes:
            raw_data = self.load_data(
                data_source='database',
                symbol=symbol,
                timeframe=timeframe,
                data_type='candles'
            )

            if raw_data.empty:
                self.logger.warning(f"Дані не знайдено для {symbol} {timeframe}")
                return results

        elif timeframe in derived_timeframes:
            source_timeframe = None
            if timeframe == '4h':
                source_timeframe = '1h'
            elif timeframe == '1w':
                source_timeframe = '1d'

            self.logger.info(f"Створення {timeframe} даних через ресемплінг з {source_timeframe}")

            source_data = self.load_data(
                data_source='database',
                symbol=symbol,
                timeframe=source_timeframe,
                data_type='candles'
            )

            if source_data.empty:
                self.logger.warning(f"Базові дані не знайдено для {symbol} {source_timeframe}")
                return results

            raw_data = self.resample_data(source_data, target_interval=timeframe)

            if raw_data.empty:
                self.logger.warning(f"Не вдалося створити дані для {symbol} {timeframe} через ресемплінг")
                return results
        else:
            self.logger.error(f"Непідтримуваний таймфрейм: {timeframe}")
            return results

        results['raw_data'] = raw_data

        # 2. Фільтрація за часовим діапазоном
        if start_date or end_date:
            raw_data = self.filter_by_time_range(raw_data, start_time=start_date, end_time=end_date)
            if raw_data.empty:
                self.logger.warning(f"Після фільтрації за часом дані відсутні")
                return results

        # 3. Обробка відсутніх значень
        filled_data = self.handle_missing_values(
            raw_data,
            symbol=symbol,
            timeframe=timeframe,
            fetch_missing=True
        )

        # 4. Повна обробка через конвеєр
        processed_data = self.preprocess_pipeline(filled_data, symbol=symbol, interval=timeframe)

        if processed_data.empty:
            self.logger.warning(f"Після обробки даних результат порожній")
            return results

        results['processed_data'] = processed_data

        # 5. Додавання часових ознак
        processed_data = self.add_time_features_safely(processed_data, tz='UTC')

        # 6. Виявлення та обробка аномалій
        processed_data, outliers_info = self.detect_outliers(processed_data)

        # 8. Створення профілю об'єму
        volume_profile = self.aggregate_volume_profile(
            processed_data,
            bins=20,
            price_col='close',
            volume_col='volume',
            time_period='1D'
        )

        if not volume_profile.empty:
            results['volume_profile'] = volume_profile

            if save_results:
                self.logger.info(f"Спроба збереження профілю об'єму: symbol={symbol}, timeframe={timeframe}")

                success = self.save_volume_profile_to_db(volume_profile, symbol, timeframe)
                if success:
                    self.logger.info(f"Профіль об'єму для {symbol} {timeframe} успішно збережено")
                else:
                    self.logger.error(f"Помилка збереження профілю об'єму для {symbol} {timeframe}")

        # 9. Підготовка даних для моделей ARIMA і LSTM
        if timeframe in ['1m', '4h', '1d', '1w']:
            # ARIMA
            arima_data = self.prepare_arima_data(processed_data, symbol=symbol, timeframe=timeframe)
            if not arima_data.empty:
                results['arima_data'] = arima_data
                if save_results:
                    method_name = f"save_{symbol}_arima_data"
                    if hasattr(self.data_storage, method_name):
                        getattr(self.data_storage, method_name)(arima_data)
                        self.logger.info(f"Дані ARIMA для {symbol} успішно збережено")
                    else:
                        self.save_btc_arima_data(processed_data, timeframe, symbol)
                        self.logger.info(f"Дані ARIMA для {symbol} успішно збережено (загальний метод)")

            try:
                lstm_df = self.prepare_lstm_data(processed_data, symbol=symbol, timeframe=timeframe)
                if not lstm_df.empty:
                    results['lstm_data'] = lstm_df

                    if save_results:
                        method_name = f"save_{symbol}_lstm_sequence"
                        if hasattr(self.data_storage, method_name):
                            getattr(self.data_storage, method_name)(lstm_df)
                            self.logger.info(f"Послідовності LSTM для {symbol} успішно збережено")
                        else:
                            self.save_btc_lstm_sequence(processed_data, timeframe, symbol)
                            self.logger.info(f"Послідовності LSTM для {symbol} успішно збережено (загальний метод)")
            except Exception as e:
                self.logger.error(f"Помилка при підготовці LSTM даних: {str(e)}")

        self.logger.info(f"Комплексна обробка даних для {symbol} ({timeframe}) завершена успішно")
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
    SYMBOLS = ['BTC']

    # Визначення базових та похідних таймфреймів
    BASE_TIMEFRAMES = ['1m', '1h', '1d']  # Таймфрейми, які зберігаються безпосередньо в БД
    DERIVED_TIMEFRAMES = ['4h', '1w']  # Таймфрейми, які потребують ресемплінгу

    processor = MarketDataProcessor(log_level=logging.INFO)

    # Спочатку обробляємо базові таймфрейми
    print("\n=== Обробка базових таймфреймів ===")
    for symbol in SYMBOLS:
        for timeframe in BASE_TIMEFRAMES:
            print(f"\nОбробка {symbol} ({timeframe})...")

            try:
                results = processor.process_market_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    save_results=True
                )

                if not results:
                    print(f"Не вдалося обробити дані для {symbol} {timeframe}")
                    continue

                # Print summary of results
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

            try:
                results = processor.process_market_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    save_results=True
                )

                if not results:
                    print(f"Не вдалося обробити дані для {symbol} {timeframe}")
                    continue

                # Print summary of results
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