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

        # Ініціалізація кешу результатів
        self.cache_enabled = True
        self._result_cache = {}

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
        if target_timeframe in ['5m', '15m', '30m']:
            return '1m'
        elif target_timeframe == '4h':
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

    def _create_volume_profile(self, data: pd.DataFrame, bins: int,
                               price_col: str, volume_col: str) -> pd.DataFrame:
        try:
            # Перевірка наявності необхідних колонок
            if price_col not in data.columns:
                self.logger.error(f"Колонка ціни '{price_col}' відсутня в даних")
                return pd.DataFrame()

            if volume_col not in data.columns:
                self.logger.error(f"Колонка об'єму '{volume_col}' відсутня в даних")
                return pd.DataFrame()

            # Перевірка типів даних
            if not pd.api.types.is_numeric_dtype(data[price_col]):
                self.logger.error(f"Колонка '{price_col}' має бути числового типу")
                return pd.DataFrame()

            if not pd.api.types.is_numeric_dtype(data[volume_col]):
                self.logger.error(f"Колонка '{volume_col}' має бути числового типу")
                return pd.DataFrame()

            # Перевірка на NaN або нульові значення
            null_prices = data[price_col].isna().sum()
            null_volumes = data[volume_col].isna().sum()

            if null_prices > 0:
                self.logger.warning(f"Знайдено {null_prices} null значень у колонці '{price_col}'")
                data = data.dropna(subset=[price_col])

            if null_volumes > 0:
                self.logger.warning(f"Знайдено {null_volumes} null значень у колонці '{volume_col}'")
                data = data.dropna(subset=[volume_col])

            if data.empty:
                self.logger.error("Після видалення null значень DataFrame порожній")
                return pd.DataFrame()

            # Перевірка валідності цінового діапазону
            price_min = data[price_col].min()
            price_max = data[price_col].max()

            # Додаємо захист від однакових або близьких цін
            price_range = price_max - price_min
            if np.isclose(price_range, 0) or price_range < 1e-10:
                # Створюємо штучний діапазон для запобігання помилок
                price_mean = data[price_col].mean()
                price_min = price_mean * 0.99  # на 1% менше
                price_max = price_mean * 1.01  # на 1% більше
                price_range = price_max - price_min
                self.logger.warning(
                    f"Діапазон цін занадто малий. Створено штучний діапазон: {price_min:.4f} - {price_max:.4f}")

            # Обчислення ефективної кількості бінів із захистом від помилок
            min_bin_width = max(price_range * 0.001, 1e-8)  # мінімальна ширина біну з нижньою межею
            effective_bins = max(min(bins, int(price_range / min_bin_width)), 2)

            if effective_bins < 2:
                self.logger.warning(f"Неможливо створити профіль об'єму. Встановлено мінімум 2 біни.")
                effective_bins = 2

            self.logger.info(f"Створення профілю об'єму з {effective_bins} ціновими рівнями")

            # Створення бінів з урахуванням діапазону
            bin_edges = np.linspace(price_min, price_max, effective_bins + 1)
            bin_width = (price_max - price_min) / effective_bins

            bin_labels = np.arange(effective_bins)
            data = data.copy()
            # Безпечне створення бінів з обробкою крайніх випадків
            try:
                data['price_bin'] = pd.cut(
                    data[price_col],
                    bins=bin_edges,
                    labels=bin_labels,
                    include_lowest=True
                )
            except Exception as e:
                self.logger.error(f"Помилка створення цінових бінів: {str(e)}")
                # Альтернативний підхід у разі помилки - квантилі
                data['price_bin'] = pd.qcut(
                    data[price_col],
                    q=effective_bins,
                    labels=bin_labels,
                    duplicates='drop'
                )

            # Групування з обробкою помилок
            try:
                volume_profile = data.groupby('price_bin', observed=True).agg({
                    volume_col: 'sum',
                    price_col: ['count', 'min', 'max']
                })

                if volume_profile.empty:
                    self.logger.warning("Отримано порожній профіль об'єму після групування")
                    return pd.DataFrame()

                # Перейменування колонок
                volume_profile.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in volume_profile.columns]
                volume_profile = volume_profile.rename(columns={
                    f'{volume_col}_sum': 'volume',
                    f'{price_col}_count': 'count',
                    f'{price_col}_min': 'price_min',
                    f'{price_col}_max': 'price_max'
                })

                # Безпечний розрахунок відсотків об'єму
                total_volume = volume_profile['volume'].sum()
                volume_profile['volume_percent'] = np.where(
                    np.isclose(total_volume, 0),  # Захист від ділення на нуль
                    0,
                    (volume_profile['volume'] / total_volume * 100).round(2)
                )

                # Розрахунок середньої ціни
                volume_profile['price_mid'] = (volume_profile['price_min'] + volume_profile['price_max']) / 2

                # Створення меж бінів
                volume_profile['bin_lower'] = [bin_edges[i] for i in volume_profile.index]
                volume_profile['bin_upper'] = [bin_edges[i + 1] for i in volume_profile.index]

                # Скидання індексу та сортування
                volume_profile = volume_profile.reset_index()
                volume_profile = volume_profile.sort_values('price_bin', ascending=False)

                if 'price_bin' in volume_profile.columns:
                    volume_profile = volume_profile.drop('price_bin', axis=1)

                return volume_profile

            except Exception as e:
                self.logger.error(f"Помилка при створенні профілю об'єму: {str(e)}")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Критична помилка при створенні профілю об'єму: {str(e)}")
            return pd.DataFrame()

    def aggregate_volume_profile(self, data: pd.DataFrame, bins: int = 20,
                                 price_col: str = 'close', volume_col: str = 'volume',
                                 time_period: Optional[str] = None) -> pd.DataFrame:
        # Add this validation at the start
        if not isinstance(data.index, pd.DatetimeIndex):
            self.logger.error("Input data must have DatetimeIndex")
            return pd.DataFrame()

        # Валідація вхідних даних
        if data is None or data.empty:
            self.logger.warning("Отримано порожній DataFrame для профілю об'єму")
            return pd.DataFrame()

        # Перевірка наявності необхідних колонок
        required_cols = [price_col, volume_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            self.logger.error(f"Відсутні необхідні колонки: {', '.join(missing_cols)}")
            return pd.DataFrame()

        # Валідація типів даних для запобігання помилок
        if not pd.api.types.is_numeric_dtype(data[price_col]):
            self.logger.error(f"Колонка '{price_col}' має бути числового типу")
            return pd.DataFrame()

        if not pd.api.types.is_numeric_dtype(data[volume_col]):
            self.logger.error(f"Колонка '{volume_col}' має бути числового типу")
            return pd.DataFrame()

        self.logger.info(f"Створення профілю об'єму з {bins} ціновими рівнями")

        # Копія даних для запобігання змін у вхідному DataFrame
        working_data = data.copy()

        # Перевірка можливості створення часового профілю
        if time_period:
            # Безпечна перевірка та конвертація часового індексу
            try:
                if not isinstance(working_data.index, pd.DatetimeIndex):
                    self.logger.warning("Індекс не є DatetimeIndex. Спроба конвертації...")
                    # Пошук часових колонок
                    time_cols = [col for col in working_data.columns if any(
                        time_str in col.lower() for time_str in ['time', 'date', 'timestamp'])]

                    if time_cols:
                        # Конвертація часової колонки в DatetimeIndex
                        working_data[time_cols[0]] = pd.to_datetime(working_data[time_cols[0]], errors='coerce')
                        working_data.set_index(time_cols[0], inplace=True)
                        # Видалення рядків з невалідними датами
                        working_data = working_data[working_data.index.notna()]
                    else:
                        self.logger.error("Не знайдено часової колонки для конвертації в DatetimeIndex")
                        return pd.DataFrame()

                # Перевірка, чи є дані в DataFrame після конвертації
                if working_data.empty:
                    self.logger.error("Дані порожні після конвертації до DatetimeIndex")
                    return pd.DataFrame()

                # Локалізація часового індексу для коректного групування
                if working_data.index.tz is None:
                    working_data.index = working_data.index.tz_localize('Europe/Kiev', ambiguous='NaT',
                                                                        nonexistent='shift_forward')
                    working_data = working_data[~working_data.index.isna()]

                # Перевірка, чи є дані в DataFrame після локалізації
                if working_data.empty:
                    self.logger.error("Дані порожні після локалізації часового індексу")
                    return pd.DataFrame()

                # Перевірка правильності time_period формату
                try:
                    # Проста перевірка на валідність частоти для pandas
                    test_range = pd.date_range(
                        start=working_data.index.min(),
                        periods=2,
                        freq=time_period
                    )
                    self.logger.info(f"Використання часового періоду: {time_period}")
                except Exception as e:
                    self.logger.error(f"Невірний часовий період '{time_period}': {str(e)}")
                    self.logger.info("Спроба створення загального профілю...")
                    return self._create_volume_profile(working_data, bins, price_col, volume_col)

                # Групування по часовому періоду з обробкою помилок
                period_groups = working_data.groupby(pd.Grouper(freq=time_period, dropna=True))

                result_dfs = []
                for period, group in period_groups:
                    if group.empty or len(group) < 2:
                        continue

                    period_profile = self._create_volume_profile(group, bins, price_col, volume_col)
                    if not period_profile.empty:
                        period_profile['period'] = period
                        result_dfs.append(period_profile)

                if result_dfs:
                    result = pd.concat(result_dfs)
                    # Перетворення та сортування результату
                    if 'period' in result.columns and not result['period'].isna().any():
                        result['period'] = pd.to_datetime(result['period'])
                        result = result.sort_values('period')

                    self.logger.info(f"Створено часовий профіль об'єму з {len(result)} записами")
                    return result
                else:
                    self.logger.warning("Не вдалося створити часовий профіль. Створюємо загальний профіль...")

            except Exception as e:
                self.logger.error(f"Помилка при створенні часового профілю: {str(e)}")
                self.logger.info("Спроба створення загального профілю...")

        # Створення загального профілю об'єму
        self.logger.info("Створення загального профілю об'єму...")
        return self._create_volume_profile(working_data, bins, price_col, volume_col)


    def merge_datasets(self, datasets: List[pd.DataFrame],
                       merge_on: str = 'timestamp',
                       chunk_size: Optional[int] = None) -> pd.DataFrame:

        global cache_key
        if not datasets:
            self.logger.warning("Порожній список наборів даних для об'єднання")
            return pd.DataFrame()

        if len(datasets) == 1:
            return datasets[0].copy()

        # Використовуємо власний chunk_size або значення з класу
        if chunk_size is None:
            chunk_size = self.chunk_size

        # Визначення загального розміру даних для вибору стратегії
        total_rows = sum(len(df) for df in datasets if df is not None and not df.empty)
        total_cols = sum(len(df.columns) for df in datasets if df is not None and not df.empty)

        self.logger.info(f"Початок об'єднання {len(datasets)} наборів даних "
                         f"(всього ~{total_rows} рядків, ~{total_cols} колонок)")

        # Перевірка чи потрібна обробка чанками
        needs_chunking = total_rows > chunk_size

        # Генеруємо кеш-ключ для збереження результату
        if self.cache_enabled:
            cache_key = f"merge_datasets_{hash(tuple([id(df) for df in datasets]))}"
            if cache_key in self._result_cache:
                self.logger.info(f"Повернення кешованого результату об'єднання наборів даних")
                return self._result_cache[cache_key].copy()

        # Попередня перевірка структури даних для оптимізації
        all_have_merge_on = all(merge_on in df.columns or
                                (isinstance(df.index, pd.Index) and df.index.name == merge_on)
                                for df in datasets if df is not None and not df.empty)

        if not all_have_merge_on:
            if merge_on == 'timestamp':
                self.logger.info("Перевірка, чи всі DataFrame мають DatetimeIndex")

                # Використання генератора замість циклу
                all_have_datetime_index = all(isinstance(df.index, pd.DatetimeIndex)
                                              for df in datasets if df is not None and not df.empty)

                if all_have_datetime_index:
                    # Векторизоване перейменування індексів
                    for df in datasets:
                        if df is not None and not df.empty and df.index.name is None:
                            df.index.name = 'timestamp'

                    all_have_merge_on = True

            if not all_have_merge_on:
                self.logger.error(f"Не всі набори даних містять '{merge_on}' для об'єднання")
                return pd.DataFrame()

        # Якщо потрібна обробка по чанках і дані великі
        if needs_chunking:
            return self._merge_datasets_in_chunks(datasets, merge_on, chunk_size)

        # Стандартна обробка для менших наборів даних
        try:
            # Підготовка даних - векторизовані операції
            datasets_copy = []

            for i, df in enumerate(datasets):
                if df is None or df.empty:
                    continue

                df_copy = df.copy()

                # Встановлюємо merge_on як індекс, якщо потрібно
                if merge_on in df_copy.columns:
                    df_copy.set_index(merge_on, inplace=True)
                    self.logger.debug(f"DataFrame {i} перетворено: колонка '{merge_on}' стала індексом")
                elif df_copy.index.name != merge_on:
                    df_copy.index.name = merge_on
                    self.logger.debug(f"DataFrame {i}: індекс перейменовано на '{merge_on}'")

                datasets_copy.append(df_copy)

            if not datasets_copy:
                self.logger.warning("Після підготовки не залишилось даних для об'єднання")
                return pd.DataFrame()

            # Об'єднання даних ефективним способом
            result = datasets_copy[0]
            columns_count = len(result.columns)

            # Створюємо словники для перейменування заздалегідь
            renaming_actions = []

            for i, df in enumerate(datasets_copy[1:], 2):
                # Векторизоване виявлення дублікатів колонок
                duplicate_cols = np.intersect1d(result.columns, df.columns)

                if len(duplicate_cols) > 0:
                    # Створюємо словник перейменування для всіх дублікатів одразу
                    rename_dict = {col: f"{col}_{i}" for col in duplicate_cols}
                    renaming_actions.append((i, df, rename_dict))
                else:
                    renaming_actions.append((i, df, {}))

            # Виконуємо перейменування та об'єднання
            for i, df, rename_dict in renaming_actions:
                if rename_dict:
                    self.logger.debug(f"Перейменування {len(rename_dict)} колонок у DataFrame {i}")
                    df = df.rename(columns=rename_dict)

                # Векторизоване об'єднання
                result = result.join(df, how='outer')
                columns_count += len(df.columns)

            duplicate_columns = columns_count - len(result.columns)
            self.logger.info(f"Об'єднання завершено. Результат: {len(result)} рядків, {len(result.columns)} колонок")
            self.logger.info(f"З {columns_count} вхідних колонок, {duplicate_columns} були дублікатами")


            # Кешування результату, якщо увімкнено
            if self.cache_enabled:
                self._result_cache[cache_key] = result.copy()

            return result

        except Exception as e:
            self.logger.error(f"Помилка при об'єднанні наборів даних: {str(e)}")
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()

    def _merge_datasets_in_chunks(self, datasets: List[pd.DataFrame],
                                  merge_on: str,
                                  chunk_size: int) -> pd.DataFrame:

        self.logger.info(f"Об'єднання наборів даних по чанках (розмір чанка: {chunk_size})")

        # Підготовка даних - встановлення індексу
        prepared_datasets = []
        for i, df in enumerate(datasets):
            if df is None or df.empty:
                continue

            df_copy = df.copy()

            if merge_on in df_copy.columns:
                df_copy.set_index(merge_on, inplace=True)
            elif df_copy.index.name != merge_on:
                df_copy.index.name = merge_on

            prepared_datasets.append(df_copy)

        if not prepared_datasets:
            return pd.DataFrame()

        # Визначення загального індексу для чанкування
        all_indices = pd.Index([])
        for df in prepared_datasets:
            all_indices = all_indices.union(df.index)
        all_indices = all_indices.sort_values()

        # Розбиття на чанки
        chunks = []
        for i in range(0, len(all_indices), chunk_size):
            chunk_indices = all_indices[i:i + chunk_size]
            chunk_results = []

            self.logger.debug(f"Обробка чанка {i // chunk_size + 1}/{(len(all_indices) - 1) // chunk_size + 1}")

            for j, df in enumerate(prepared_datasets):
                # Вибір тільки рядків, що входять у поточний чанк
                chunk_df = df[df.index.isin(chunk_indices)]
                if not chunk_df.empty:
                    chunk_results.append(chunk_df)

            # Рекурсивний виклик для обробки чанка (без чанкування)
            if chunk_results:
                merged_chunk = self.merge_datasets(chunk_results, merge_on=merge_on, chunk_size=None)
                if not merged_chunk.empty:
                    chunks.append(merged_chunk)

            # Очищення пам'яті після обробки чанка
            gc.collect()

        # Об'єднання всіх чанків
        if not chunks:
            return pd.DataFrame()

        try:
            result = pd.concat(chunks)
            return result
        except Exception as e:
            self.logger.error(f"Помилка при об'єднанні чанків: {str(e)}")
            return pd.DataFrame()

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
        """Зберігає профіль об'єму в базу даних"""

        if data is None or data.empty:
            self.logger.warning("Спроба зберегти порожній профіль об'єму")
            return False

        try:
            saved_count = 0
            error_count = 0

            for _, row in data.iterrows():
                try:
                    # Визначення часового сегменту
                    time_bucket = None
                    if 'period' in row and pd.notna(row['period']):
                        time_bucket = row['period']
                    elif isinstance(row.name, pd.Timestamp):
                        time_bucket = row.name
                    else:
                        # Використовуємо поточний час як запасний варіант
                        time_bucket = pd.Timestamp.now()

                    # Гарантуємо, що time_bucket є Timestamp об'єктом
                    if not isinstance(time_bucket, pd.Timestamp):
                        time_bucket = pd.to_datetime(time_bucket)

                    # Підготовка і конвертація даних для збереження
                    profile_data = {
                        'timeframe': str(timeframe),
                        'time_bucket': time_bucket,
                        'price_bin_start': float(row.get('bin_lower', 0)),
                        'price_bin_end': float(row.get('bin_upper', 0)),
                        'volume': float(row.get('volume', 0)),
                        'count': int(row.get('count', 0)) if 'count' in row else None,
                        'volume_percent': float(row.get('volume_percent', 0)) if 'volume_percent' in row else None
                    }

                    # Безпечна конвертація numpy типів
                    for key, value in profile_data.items():
                        if isinstance(value, np.generic):
                            if np.issubdtype(value.dtype, np.integer):
                                profile_data[key] = int(value)
                            elif np.issubdtype(value.dtype, np.floating):
                                profile_data[key] = float(value)
                            else:
                                profile_data[key] = value.item()

                    # Видалення None значень для запобігання помилок БД
                    profile_data = {k: v for k, v in profile_data.items() if v is not None}

                    # Збереження в БД
                    self.db_manager.insert_volume_profile(symbol, profile_data)
                    saved_count += 1

                except Exception as e:
                    error_count += 1
                    self.logger.error(f"Помилка при збереженні запису профілю об'єму: {str(e)}")
                    # Продовжуємо зберігати інші записи
                    continue

            success_rate = saved_count / (saved_count + error_count) if (saved_count + error_count) > 0 else 0
            self.logger.info(
                f"Збережено {saved_count} записів профілю об'єму, помилок: {error_count}, успішність: {success_rate:.1%}")

            return success_rate > 0.5  # Повертаємо True, якщо збережено більше половини записів

        except Exception as e:
            self.logger.error(f"Критична помилка при збереженні профілю об'єму: {str(e)}")
            return False

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
                            create_volume_profile: bool = True) -> DataFrame | dict[Any, Any]:

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
        derived_timeframes = ['4h', '1w']

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

        elif timeframe in derived_timeframes:
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
            resampling_start_time = time()
            raw_data = self.resample_data(source_data, target_interval=timeframe)

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


        # 7. Створення профілю об'єму лише для 1d та 1w таймфреймів, якщо потрібно
        volume_profile_allowed_timeframes = ['1d', '1w']
        if create_volume_profile and timeframe in volume_profile_allowed_timeframes:
            self.logger.info(f"Створення профілю об'єму для {symbol} ({timeframe})")

            try:
                    # Виклик методу з правильними аргументами
                    volume_profile = self.aggregate_volume_profile(
                        data=processed_data,
                        bins=20,
                        price_col='close',
                        volume_col='volume',
                        time_period='1W'
                    )

                    self.logger.info(f"Створення профілю об'єму виконано ")

                    if volume_profile is not None and not volume_profile.empty:
                        self.logger.info(f"Створено профіль об'єму з {len(volume_profile)} записами")
                        results['volume_profile'] = volume_profile

                        if save_results:
                            self.logger.info(f"Збереження профілю об'єму в БД")
                            success = self.save_volume_profile_to_db(volume_profile, symbol, timeframe)

                            if success:
                                self.logger.info(f"Профіль об'єму збережено ")
                            else:
                                self.logger.error(f"Помилка збереження профілю об'єму")
                    else:
                        self.logger.warning(f"Отримано порожній профіль об'єму")

            except Exception as e:
                self.logger.error(f"Помилка при створенні профілю об'єму: {str(e)}")
                self.logger.error(traceback.format_exc())

        # 8. Підготовка даних для моделей ARIMA і LSTM
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
    SYMBOLS = ['SOL']

    # Визначення всіх таймфреймів
    ALL_TIMEFRAMES = ['1h', '4h', '1d', '1w']

    # Базові таймфрейми, які вже існують в базі даних
    BASE_TIMEFRAMES = [ '1h', '1d']

    # Похідні таймфрейми, які будуть створені через ресемплінг
    target_interval = ['4h', '1w']

    # Таймфрейми, для яких створюємо volume профіль
    VOLUME_PROFILE_TIMEFRAMES = ['1d', '1w']

    processor = MarketDataProcessor(log_level=logging.INFO)

    # Спочатку обробляємо базові таймфрейми
    print("\n=== Обробка базових таймфреймів ===")

    # Для використання volume профілю
    volume_profiles = {}

    for symbol in SYMBOLS:
        for timeframe in BASE_TIMEFRAMES:
            print(f"\nОбробка {symbol} ({timeframe})...")

            try:
                # Визначаємо, чи потрібно створювати volume профіль для цього таймфрейму
                create_volume_profile = timeframe in VOLUME_PROFILE_TIMEFRAMES

                results = processor.process_market_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    save_results=True,
                    create_volume_profile=create_volume_profile
                )

                if not results:
                    print(f"Не вдалося обробити дані для {symbol} {timeframe}")
                    continue

                # Зберігаємо volume профіль для подальшого використання
                if create_volume_profile and 'volume_profile' in results:
                    volume_profiles[(symbol, timeframe)] = results['volume_profile']

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
        for timeframe in target_interval:
            print(f"\nОбробка {symbol} ({timeframe})...")

            try:
                # Визначаємо, чи потрібно створювати volume профіль для цього таймфрейму
                create_volume_profile = timeframe in VOLUME_PROFILE_TIMEFRAMES and (symbol,
                                                                                    timeframe) not in volume_profiles

                results = processor.process_market_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    save_results=True,
                    create_volume_profile=create_volume_profile
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
