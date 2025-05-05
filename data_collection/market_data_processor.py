import traceback
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple

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


    def remove_duplicate_timestamps(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        return self.data_cleaner.remove_duplicate_timestamps(data)

    def clean_data(self, data: pd.DataFrame, remove_outliers: bool = True,
                   fill_missing: bool = True, **kwargs) -> pd.DataFrame:

        if data.empty:
            return data

        result = data.copy()

        # Видалення викидів
        if remove_outliers:
            # Use the specialized class method
            result, _ = self.anomaly_detector.detect_outliers(result)

        # Заповнення пропусків
        if fill_missing:
            result = self.data_cleaner.handle_missing_values(result)

        return result

    def handle_missing_values(self, data: pd.DataFrame, method: str = 'interpolate',
                              fetch_missing: bool = False, symbol: Optional[str] = None,
                              timeframe: Optional[str] = None, **kwargs) -> pd.DataFrame:

        return self.data_cleaner.handle_missing_values(
            data,
            method=method,
            fetch_missing=fetch_missing,
            symbol=symbol,
            timeframe=timeframe
        )

    def normalize_data(self, data: pd.DataFrame, method: str = 'z-score',
                       exclude_columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict]:

        return self.data_cleaner.normalize_data(
            data,
            method=method,
            exclude_columns=exclude_columns
        )

    def detect_outliers(self, data: pd.DataFrame, method: str = 'iqr',
                        threshold: float = 1.5, **kwargs) -> Tuple[pd.DataFrame, Dict]:

        return self.anomaly_detector.detect_outliers(
            data,
            method=method,
            threshold=threshold
        )

    def resample_data(self, data: pd.DataFrame, target_interval: str) -> pd.DataFrame:

        return self.data_resampler.resample_data(data, target_interval=target_interval)

    def make_stationary(self, data: pd.DataFrame, method: str = 'diff') -> pd.DataFrame:

        return self.data_resampler.make_stationary(data, method=method)

    def prepare_arima_data(self, data: pd.DataFrame, symbol: str, **kwargs) -> pd.DataFrame:

        return self.data_resampler.prepare_arima_data(data, symbol=symbol, **kwargs)

    def prepare_lstm_data(self, data: pd.DataFrame, symbol: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:

        return self.data_resampler.prepare_lstm_data(data, symbol=symbol, **kwargs)

    def add_time_features(self, data: pd.DataFrame, tz: str = 'UTC') -> pd.DataFrame:

        return self.data_cleaner.add_time_features(data, tz=tz)

    def filter_by_time_range(self, data: pd.DataFrame, start_date: str = None, end_date: str = None) -> pd.DataFrame:

        return self.data_cleaner.filter_by_time_range(data, start_date=start_date, end_date=end_date)

    def validate_data_integrity(self, data: pd.DataFrame) -> Tuple[bool, Dict]:

        return self.anomaly_detector.validate_data_integrity(data)



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

        return self.data_storage.save_volume_profile_to_db(data, symbol, timeframe)

    def save_lstm_sequence(self, data: pd.DataFrame, symbol: str, **kwargs) -> bool:

        return self.data_storage.save_lstm_sequence(data, symbol, **kwargs)

    def save_arima_data(self, data: pd.DataFrame, symbol: str, **kwargs) -> bool:

        return self.data_storage.save_arima_data(data, symbol, **kwargs)

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
                    step_params['interval'] = interval

                if step_name in ['normalize_data', 'detect_outliers', 'validate_data_integrity']:
                    result, _ = method(result, **step_params)
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
    EU_TIMEZONE = 'Europe/Kiev'
    SYMBOLS = ['BTC', 'ETH', 'SOL']
    TIMEFRAMES = ['5m', '1h']
    processor = MarketDataProcessor()

    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            print(f"\n Обробка {symbol} ({timeframe})...")

            # 1. Завантаження даних з БД
            raw_data = processor.load_data(
                data_source='database',
                symbol=symbol,
                timeframe=timeframe,
                data_type='candles'
            )

            if raw_data.empty:
                print(f" Дані не знайдено для {symbol} {timeframe}")
                continue

            # 2. Обробка відсутніх значень + підтягування з Binance (якщо треба)
            filled_data = processor.handle_missing_values(
                raw_data,
                symbol=symbol,
                interval=timeframe,
                fetch_missing=True
            )



            # 4. Попередня обробка (пайплайн)
            processed = processor.preprocess_pipeline(filled_data, symbol=symbol, interval=timeframe)

            if processed.empty:
                print(f" Обробка не дала результатів для {symbol} {timeframe}")
                continue

            processed = processor.add_time_features(processed, tz=EU_TIMEZONE)
            print(f" Оброблені свічки ({timeframe}) збережено")

            # 5. Ресемплінг і збереження даних для моделей
            if timeframe == '1m':
                resampled_30m = processor.resample_data(processed, target_interval='30m')
                resampled_30m = processor.add_time_features(resampled_30m, tz=EU_TIMEZONE)

                # Збереження 30-хвилинних даних

                # Збереження для LSTM та ARIMA
                processor.save_lstm_sequence(resampled_30m, symbol, timeframe='30m')
                processor.save_arima_data(resampled_30m, symbol, timeframe='30m')
                print(f" 30-хвилинні дані збережено для LSTM та ARIMA")

            if timeframe == '1h':
                # Ресемплінг до 4h
                resampled_4h = processor.resample_data(processed, target_interval='4h')
                resampled_4h = processor.add_time_features(resampled_4h, tz=EU_TIMEZONE)

                # Збереження 4-годинних даних


                # Збереження для LSTM та ARIMA
                processor.save_lstm_sequence(resampled_4h, symbol, timeframe='4h')
                processor.save_arima_data(resampled_4h, symbol, timeframe='4h')
                print(f" 4-годинні дані збережено для LSTM та ARIMA")

                # Ресемплінг до 1d
                resampled_1d = processor.resample_data(processed, target_interval='1d')
                resampled_1d = processor.add_time_features(resampled_1d, tz=EU_TIMEZONE)

                # Збереження денних даних


                # Збереження для LSTM та ARIMA
                processor.save_lstm_sequence(resampled_1d, symbol, timeframe='1d')
                processor.save_arima_data(resampled_1d, symbol, timeframe='1d')
                print(f" Денні дані збережено для LSTM та ARIMA")

                # 6. Профіль об'єму
                volume_profile = processor.aggregate_volume_profile(resampled_1d, bins=12, time_period='1W')
                if not volume_profile.empty:
                    processor.save_volume_profile_to_db(volume_profile, symbol, '1d')
                    print(f" Профіль об'єму збережено")

            # Збереження оригінального таймфрейму для LSTM та ARIMA
            processor.save_lstm_sequence(processed, symbol, timeframe=timeframe)
            processor.save_arima_data(processed, symbol, timeframe=timeframe)
            print(f" Дані {timeframe} збережено для LSTM та ARIMA")


if __name__ == "__main__":
    main()