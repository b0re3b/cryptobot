import traceback
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any, Callable
from functools import lru_cache
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import os
import gc
import psutil
from joblib import Memory

from pandas import DataFrame, Series

from data_collection.AnomalyDetector import AnomalyDetector
from data_collection.DataCleaner import DataCleaner
from data_collection.DataResampler import DataResampler
from data_collection.DataStorageManager import DataStorageManager
from data.db import DatabaseManager


class MarketDataProcessor:
    """
    Клас для обробки ринкових даних з оптимізаціями для швидкості та ефективності пам'яті.
    """
    # Налаштування кешування
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
    memory = Memory(cache_dir, verbose=0)

    def __init__(self, log_level=logging.INFO, use_multiprocessing=True,
                 cache_enabled=True, chunk_size=100000):

        self.log_level = log_level
        logging.basicConfig(level=self.log_level)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Ініціалізація класу...")

        # Налаштування оптимізацій
        self.use_multiprocessing = use_multiprocessing
        self.cache_enabled = cache_enabled
        self.chunk_size = chunk_size
        self.num_workers = max(1, mp.cpu_count() - 1)  # Залишаємо один потік вільним

        # Ініціалізація залежних класів
        self.data_cleaner = DataCleaner(logger=self.logger)
        self.data_resampler = DataResampler(logger=self.logger)
        self.data_storage = DataStorageManager(logger=self.logger)
        self.anomaly_detector = AnomalyDetector(logger=self.logger)
        self.db_manager = DatabaseManager()
        self.supported_symbols = self.db_manager.supported_symbols

        # Кеш для проміжних результатів обробки даних
        self._result_cache = {}

        self.ready = True
        self.filtered_data = None
        self.orderbook_statistics = None

    def _get_memory_usage(self):
        """Отримати поточне використання пам'яті."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB

    def _log_memory_usage(self, message=""):
        """Логувати поточне використання пам'яті."""
        memory_mb = self._get_memory_usage()
        self.logger.info(f"{message} Використання пам'яті: {memory_mb:.2f} MB")

    def _process_in_chunks(self, data: pd.DataFrame,
                           process_func: Callable,
                           **kwargs) -> pd.DataFrame:

        if data is None or data.empty:
            return data

        # Якщо дані менші за розмір чанка, обробляємо цілим
        if len(data) <= self.chunk_size:
            return process_func(data, **kwargs)

        self.logger.info(f"Обробка даних по чанкам: {len(data)} рядків, розмір чанка {self.chunk_size}")
        self._log_memory_usage("До обробки по чанкам:")

        chunks = []
        chunk_indices = list(range(0, len(data), self.chunk_size))

        for i, start_idx in enumerate(chunk_indices):
            end_idx = min(start_idx + self.chunk_size, len(data))
            chunk = data.iloc[start_idx:end_idx].copy()

            self.logger.debug(f"Обробка чанка {i + 1}/{len(chunk_indices)}: рядки {start_idx}-{end_idx}")

            try:
                processed_chunk = process_func(chunk, **kwargs)
                if processed_chunk is not None and not processed_chunk.empty:
                    chunks.append(processed_chunk)
            except Exception as e:
                self.logger.error(f"Помилка при обробці чанка {i + 1}: {str(e)}")

            # Явно очищуємо пам'ять
            del chunk
            gc.collect()

        if not chunks:
            return pd.DataFrame()

        # Об'єднуємо результати
        try:
            result = pd.concat(chunks, axis=0)
            self._log_memory_usage("Після обробки по чанкам:")
            return result
        except Exception as e:
            self.logger.error(f"Помилка при об'єднанні результатів: {str(e)}")
            return pd.DataFrame()

    def _parallel_process(self, data_list: List[pd.DataFrame],
                          func: Callable,
                          **kwargs) -> List[pd.DataFrame]:

        if not self.use_multiprocessing or len(data_list) <= 1:
            return [func(df, **kwargs) for df in data_list]

        self.logger.info(f"Паралельна обробка {len(data_list)} наборів даних на {self.num_workers} ядрах")

        def _process_one(df_idx_pair):
            idx, df = df_idx_pair
            try:
                if df is None or df.empty:
                    return idx, pd.DataFrame()
                result = func(df, **kwargs)
                return idx, result
            except Exception as e:
                self.logger.error(f"Помилка в паралельній обробці #{idx}: {str(e)}")
                return idx, pd.DataFrame()

        # Створення списку індекс-датафрейм пар для збереження порядку
        indexed_data = list(enumerate(data_list))

        # Вибір типу виконавця в залежності від операції
        # ProcessPoolExecutor для CPU-bound операцій, ThreadPoolExecutor для I/O-bound
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(_process_one, indexed_data))

        # Відновлення оригінального порядку результатів
        sorted_results = sorted(results, key=lambda x: x[0])
        return [result for _, result in sorted_results]

    @lru_cache(maxsize=32)
    def _get_cached_data(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:

        return self.load_data(
            data_source='database',
            symbol=symbol,
            timeframe=timeframe,
            data_type='candles',
            start_date=start_date,
            end_date=end_date
        )

    def align_time_series(self, data_list: List[pd.DataFrame],
                          reference_index: int = 0) -> List[pd.DataFrame]:

        if not data_list:
            self.logger.warning("Порожній список DataFrame для вирівнювання")
            return []

        if reference_index < 0 or reference_index >= len(data_list):
            self.logger.error(f"Невірний reference_index: {reference_index}. Має бути від 0 до {len(data_list) - 1}")
            reference_index = 0

        # Підготовка DataFrames - векторизована обробка
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

            # Сортування за часовим індексом, якщо потрібно
            if not df_copy.index.is_monotonic_increasing:
                df_copy = df_copy.sort_index()

            processed_data_list.append(df_copy)

        # Перевірка еталонного DataFrame
        reference_df = processed_data_list[reference_index]
        if reference_df is None or reference_df.empty:
            self.logger.error("Еталонний DataFrame є порожнім")
            return processed_data_list

        # Векторизований пошук спільного часового діапазону
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

        # Створення спільної часової сітки
        try:
            # Спроба визначити частоту еталонного DataFrame
            reference_freq = pd.infer_freq(reference_df.index)

            if not reference_freq:
                self.logger.warning("Не вдалося визначити частоту reference DataFrame. Визначення вручну.")
                # Векторизований розрахунок медіанної різниці часових міток
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

            # Створення нового індексу з використанням визначеної частоти
            # Векторизована фільтрація для reference_subset
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
                    # Використовуємо оригінальний індекс, обмежений спільним діапазоном
                    # Векторизована фільтрація
                    df_aligned = df[(df.index >= common_start) & (df.index <= common_end)]
                    # Перевірка, чи потрібно перестворити індекс з визначеною частотою
                    if len(df_aligned.index) != len(common_index):
                        self.logger.debug(f"Перестворення індексу для еталонного DataFrame {i}")
                        df_aligned = df_aligned.reindex(common_index)
                else:
                    # Для інших DataFrame - вирівнюємо до спільного індексу
                    df_aligned = df.reindex(common_index)

                # Векторизована інтерполяція числових даних
                numeric_cols = df_aligned.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    # Інтерполяція всіх числових колонок одразу
                    df_aligned[numeric_cols] = df_aligned[numeric_cols].interpolate(method='time')
                    # Заповнення NaN, що залишились
                    df_aligned[numeric_cols] = df_aligned[numeric_cols].fillna(method='ffill').fillna(method='bfill')

                # Перевірка та звіт про відсутні значення
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

    @memory.cache
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
                        # Векторизована локалізація
                        data.index = pd.DatetimeIndex(data.index).tz_localize('UTC', ambiguous='NaT')
                        # Прибираємо рядки з NaT - векторизована операція
                        data = data[~data.index.isna()]
                    except pd.errors.OutOfBoundsDatetime:
                        self.logger.warning("Помилка локалізації часового індексу. Продовжуємо без локалізації.")

                # Безпечне групування з перевіркою на порожні групи
                period_groups = data.groupby(pd.Grouper(freq=time_period, dropna=True))

                # Обробка даних по чанках, якщо вони великі
                def _process_groups():
                    result_dfs = []
                    for period, group in period_groups:
                        if group.empty or len(group) < 2:
                            continue

                        period_profile = self._create_volume_profile(group, bins, price_col, volume_col)
                        if not period_profile.empty:
                            period_profile['period'] = period
                            result_dfs.append(period_profile)
                    return result_dfs

                if len(data) > self.chunk_size:
                    # Обробка груп у чанках
                    result_dfs = self._process_in_chunks(data, _process_groups)
                else:
                    result_dfs = _process_groups()

                if result_dfs:
                    # Об'єднання результатів
                    result = pd.concat(result_dfs)

                    # Перетворення колонки period на DatetimeIndex, якщо це можливо
                    if 'period' in result.columns and not result['period'].isna().any():
                        # Векторизована конвертація
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

        # Векторизоване знаходження мін/макс замість покроковій ітерації
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

            bin_labels = np.arange(effective_bins)  # Використання numpy для оптимізації
            data = data.copy()

            # Векторизоване створення бінів
            data.loc[:, 'price_bin'] = pd.cut(
                data[price_col],
                bins=bin_edges,
                labels=bin_labels,
                include_lowest=True
            )

            # Векторизоване групування з параметром observed=True
            volume_profile = data.groupby('price_bin', observed=True).agg({
                volume_col: 'sum',
                price_col: ['count', 'min', 'max']
            })

            if volume_profile.empty:
                self.logger.warning("Отримано порожній профіль об'єму після групування")
                return pd.DataFrame()

            # Оптимізоване перейменування колонок
            volume_profile.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in volume_profile.columns]
            volume_profile = volume_profile.rename(columns={
                f'{volume_col}_sum': 'volume',
                f'{price_col}_count': 'count',
                f'{price_col}_min': 'price_min',
                f'{price_col}_max': 'price_max'
            })

            # Векторизована обробка відсотків об'єму
            total_volume = volume_profile['volume'].sum()
            volume_profile['volume_percent'] = np.where(
                total_volume > 0,
                (volume_profile['volume'] / total_volume * 100).round(2),
                0
            )

            # Векторизований розрахунок середньої ціни
            volume_profile['price_mid'] = (volume_profile['price_min'] + volume_profile['price_max']) / 2

            # Векторизоване створення меж бінів
            volume_profile['bin_lower'] = [bin_edges[i] for i in volume_profile.index]
            volume_profile['bin_upper'] = [bin_edges[i + 1] for i in volume_profile.index]

            volume_profile = volume_profile.reset_index()
            # Оптимізоване сортування
            volume_profile = volume_profile.sort_values('price_bin', ascending=False)

            if 'price_bin' in volume_profile.columns:
                volume_profile = volume_profile.drop('price_bin', axis=1)

            return volume_profile

        except Exception as e:
            self.logger.error(f"Помилка при створенні профілю об'єму: {str(e)}")
            return pd.DataFrame()

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
        self._log_memory_usage("До об'єднання наборів даних:")

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

            self._log_memory_usage("Після об'єднання наборів даних:")

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

    # Додавання утилітних методів для кешування та управління пам'яттю
    def clear_cache(self):
        """Очищує всі внутрішні кеші."""
        self._result_cache.clear()
        self.memory.clear()
        gc.collect()
        self.logger.info("Кеш очищено")
        self._log_memory_usage("Після очищення кешу:")

    def optimize_memory(self, data: pd.DataFrame) -> pd.DataFrame:

        if data is None or data.empty:
            return data

        start_mem = data.memory_usage(deep=True).sum() / 1024 ** 2
        self.logger.debug(f"Початковий розмір DataFrame: {start_mem:.2f} MB")

        # Копіювання для запобігання зміни оригіналу
        result = data.copy()

        # Оптимізація числових колонок
        numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        for col in result.select_dtypes(include=numerics).columns:
            col_dtype = result[col].dtype

            # Цілі числа
            if np.issubdtype(col_dtype, np.integer):
                c_min = result[col].min()
                c_max = result[col].max()

                # Визначення оптимального цілочисельного типу на основі діапазону
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    result[col] = result[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    result[col] = result[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    result[col] = result[col].astype(np.int32)
                else:
                    result[col] = result[col].astype(np.int64)

            # Числа з плаваючою точкою
            elif np.issubdtype(col_dtype, np.floating):
                # Перетворення в float32, якщо не призводить до втрати точності
                # Для фінансових даних часто потрібна висока точність, тому перевіряємо різницю
                f32_col = result[col].astype(np.float32)
                if (result[col] - f32_col).abs().max() < 1e-6:
                    result[col] = f32_col

        # Оптимізація категоріальних колонок
        for col in result.select_dtypes(include=['object']).columns:
            num_unique = result[col].nunique()
            num_total = len(result[col])

            # Якщо колонка має невелику кількість унікальних значень відносно загальної кількості
            if num_unique / num_total < 0.5:  # менше 50% унікальних значень
                result[col] = result[col].astype('category')

        end_mem = result.memory_usage(deep=True).sum() / 1024 ** 2
        savings = (1 - end_mem / start_mem) * 100
        self.logger.info(f"Оптимізований розмір DataFrame: {end_mem:.2f} MB, економія: {savings:.1f}%")

        return result

    def parallel_apply(self, data: pd.DataFrame, func: Callable,
                       column: Optional[str] = None,
                       axis: int = 0) -> Series | DataFrame:

        if data is None or data.empty:
            return pd.Series() if column else pd.DataFrame()

        # Визначення обробника даних
        if column is not None:
            data_to_process = data[column]
        else:
            data_to_process = data

        # Розділення даних на частини для паралельної обробки
        splits = np.array_split(data_to_process, self.num_workers)

        # Функція для обробки однієї частини
        def process_chunk(chunk):
            return chunk.apply(func, axis=axis)

        # Паралельне застосування
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(process_chunk, splits))

        # Об'єднання результатів
        if isinstance(results[0], pd.Series):
            return pd.concat(results)
        else:
            return pd.concat(results)
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

    def save_sol_lstm_sequence(self, data_points: List[Dict[str, Any]], **kwargs) -> List[int]:

        return self.data_storage.save_sol_lstm_sequence(data_points)

    def save_sol_arima_data(self, data_points: List[Dict[str, Any]], **kwargs) -> List[int]:

        return self.data_storage.save_sol_arima_data(data_points)

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
                            end_date: Optional[str] = None, save_results: bool = True) -> DataFrame | dict[Any, Any]:

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
        try:
            # Перевірка колонок
            if 'close' not in processed_data.columns or 'volume' not in processed_data.columns:
                self.logger.error("Відсутні колонки 'close' або 'volume'")
                return pd.DataFrame()

            # Перевірка індексу
            if not isinstance(processed_data.index, pd.DatetimeIndex):
                self.logger.error("Індекс не є DatetimeIndex")
                return pd.DataFrame()

            # Логування для зневадження
            self.logger.debug(f"Дані для профілю об'єму:\n{processed_data.head()}")
            self.logger.debug(f"Колонки: {processed_data.columns}")

            # Виклик методу з правильними аргументами
            volume_profile = self.aggregate_volume_profile(
                data=processed_data,
                bins=20,
                price_col='close',  # Лапки додані
                volume_col='volume',  # Лапки додані
                time_period='1W'
            )

            if not volume_profile.empty:
                results['volume_profile'] = volume_profile
                if save_results:
                    success = self.save_volume_profile_to_db(volume_profile, symbol, timeframe)
                    if success:
                        self.logger.info("Профіль об'єму збережено")
                    else:
                        self.logger.error("Помилка збереження профілю об'єму")
        except Exception as e:
            self.logger.error(f"Помилка при створенні профілю об'єму: {str(e)}")

        # 9. Підготовка даних для моделей ARIMA і LSTM
        if timeframe in ['1m', '4h', '1d', '1w']:
            # ARIMA
            arima_data = self.prepare_arima_data(processed_data, symbol=symbol, timeframe=timeframe)
            if not arima_data.empty:
                results['arima_data'] = arima_data

                if save_results:
                    try:
                        # Convert DataFrame to list of dicts and ensure required fields exist
                        arima_data_points = arima_data.reset_index().to_dict('records')

                        # Add missing required fields if they don't exist
                        for record in arima_data_points:
                            if 'open_time' not in record:
                                record['open_time'] = record.get('timestamp', record.get('index', pd.Timestamp.now()))
                            if 'original_close' not in record:
                                record['original_close'] = record.get('close', None)

                        # Try specific method first, then fall back to general
                        method_name = f"save_{symbol.lower()}_arima_data"
                        if hasattr(self.data_storage, method_name):
                            arima_ids = getattr(self.data_storage, method_name)(arima_data_points)
                        else:
                            arima_ids = self.data_storage.save_sol_arima_data(
                                data_points=arima_data_points,
                            )

                        if arima_ids:
                            self.logger.info(f"ARIMA data saved for {symbol}, IDs: {arima_ids}")
                        else:
                            self.logger.warning(f"Failed to save ARIMA data for {symbol}")
                    except Exception as e:
                        self.logger.error(f"Error saving ARIMA data for {symbol}: {str(e)}")

            # LSTM data preparation and saving
            try:
                lstm_df = self.prepare_lstm_data(processed_data, symbol=symbol, timeframe=timeframe)
                if not lstm_df.empty:
                    results['lstm_data'] = lstm_df

                    if save_results:
                        try:
                            # Convert DataFrame and add required fields
                            lstm_data_points = lstm_df.reset_index().to_dict('records')

                            # Add sequence_position if missing (sequential numbering)
                            for i, record in enumerate(lstm_data_points):
                                if 'sequence_position' not in record:
                                    record['sequence_position'] = i + 1  # 1-based indexing
                                if 'sequence_id' not in record:
                                    record[
                                        'sequence_id'] = f"{symbol}_{timeframe}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}_{i}"

                            # Validate required fields
                            required_fields = ['sequence_position', 'sequence_id']
                            for record in lstm_data_points:
                                missing = [f for f in required_fields if f not in record]
                                if missing:
                                    raise ValueError(f"Missing required fields: {missing}")

                            # Save using appropriate method
                            method_name = f"save_{symbol.lower()}_lstm_sequence"
                            if hasattr(self.data_storage, method_name):
                                sequence_ids = getattr(self.data_storage, method_name)(lstm_data_points)
                            else:
                                sequence_ids = self.data_storage.save_sol_lstm_sequence(lstm_data_points)

                            if sequence_ids:
                                self.logger.info(f"Saved LSTM sequences for {symbol}, IDs: {sequence_ids}")
                            else:
                                self.logger.warning(f"Failed to save LSTM sequences for {symbol}")
                        except Exception as e:
                            self.logger.error(f"Error saving LSTM sequences for {symbol}: {str(e)}")
            except Exception as e:
                self.logger.error(f"Error preparing LSTM data: {str(e)}")

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
    BASE_TIMEFRAMES = ['1h']  # Таймфрейми, які зберігаються безпосередньо в БД
    DERIVED_TIMEFRAMES = ['4h']  # Таймфрейми, які потребують ресемплінгу

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