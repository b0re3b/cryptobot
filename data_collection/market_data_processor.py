import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional
from data_collection.AnomalyDetector import AnomalyDetector
from data_collection.DataCleaner import DataCleaner
from data_collection.DataResampler import DataResampler
from data_collection.DataStorageManager import DataStorageManager
from data_collection.OrderBookProcessor import OrderBookProcessor
from utils.config import db_connection
from data.db import DatabaseManager

class MarketDataProcessor:

    def __init__(self,  log_level=logging.INFO):
        self.log_level = log_level
        self.db_connection = db_connection
        self.db_manager = DatabaseManager()
        self.supported_symbols = self.db_manager.supported_symbols
        logging.basicConfig(level=self.log_level)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Ініціалізація класу...")
        self.ready = True
        self.storage_manager = DataStorageManager(self.db_manager, self.logger)
        self.orderbook_processor = OrderBookProcessor(self.db_manager, self.logger)
        self.data_resampler = DataResampler(self.logger)
        self.data_cleaner = DataCleaner(self.logger)
        self.anomaly_detector = AnomalyDetector(self.logger)

    def preprocess_orderbook_pipeline(self, symbol: str,
                                      start_time: Optional[datetime] = None,
                                      end_time: Optional[datetime] = None,
                                      add_time_features: bool = False,
                                      cyclical: bool = True,
                                      add_sessions: bool = True) -> pd.DataFrame:
        """Повний конвеєр обробки даних ордербука."""
        # Завантаження даних
        raw_data = self.load_orderbook_data(symbol, start_time, end_time)

        # Перевіряємо цілісність сирих даних
        raw_integrity_issues = self.validate_data_integrity(raw_data)
        if raw_integrity_issues:
            issue_count = sum(len(issues) if isinstance(issues, list) or isinstance(issues, dict) else 1
                              for issues in raw_integrity_issues.values())
            self.logger.warning(f"Знайдено {issue_count} проблем з цілісністю сирих даних ордербука")

        # Виявлення пропущених періодів
        if not raw_data.empty and isinstance(raw_data.index, pd.DatetimeIndex):
            expected_diff = pd.Timedelta(minutes=1)  # Припускаємо хвилинні дані
            missing_periods = self._detect_missing_periods(raw_data, expected_diff)

            if missing_periods:
                self.logger.info(f"Знайдено {len(missing_periods)} пропущених періодів")
                fetched_data = self.fetch_missing_orderbook_data(symbol, missing_periods)

                if not fetched_data.empty:
                    # Зберігаємо отримані дані
                    for _, row in fetched_data.iterrows():
                        # Виправлено формат даних
                        orderbook_data = {
                            'bids': [[row['bid_price'], row['bid_qty']]],
                            'asks': [[row['ask_price'], row['ask_qty']]]
                        }
                        self.save_orderbook_to_db(orderbook_data, symbol, row.name)

                    # Оновлюємо raw_data
                    raw_data = pd.concat([raw_data, fetched_data])
                    raw_data = raw_data[~raw_data.index.duplicated(keep='last')].sort_index()

        # Обробка даних
        processed_data = self.process_orderbook_data(raw_data)

        # Перевірка цілісності даних після первинної обробки
        processed_integrity_issues = self.validate_data_integrity(processed_data)
        if processed_integrity_issues:
            issue_count = sum(len(issues) if isinstance(issues, list) or isinstance(issues, dict) else 1
                              for issues in processed_integrity_issues.values())
            self.logger.info(f"Після процесу обробки залишилось {issue_count} проблем з цілісністю даних")

        # Нормалізація даних перед виявленням аномалій
        normalized_data, scaler_meta = self.normalize_data(
            processed_data,
            method='z-score',
            exclude_columns=['timestamp', 'symbol']  # Виключаємо нечислові колонки
        )

        # Якщо нормалізація пройшла успішно, використовуємо нормалізовані дані
        if scaler_meta is not None:
            processed_data = normalized_data
            self.logger.info(f"Дані нормалізовано методом {scaler_meta['method']}")

            # Перевірка цілісності даних після нормалізації
            norm_integrity_issues = self.validate_data_integrity(processed_data)
            if norm_integrity_issues:
                issue_count = sum(len(issues) if isinstance(issues, list) or isinstance(issues, dict) else 1
                                  for issues in norm_integrity_issues.values())
                self.logger.info(f"Після нормалізації залишилось {issue_count} проблем з цілісністю даних")

                # Перевіряємо на наявність нескінченних або NaN значень після нормалізації
                if "columns_with_inf" in norm_integrity_issues or "columns_with_na" in norm_integrity_issues:
                    self.logger.warning("Нормалізація створила проблеми з нескінченними або відсутніми значеннями")

        # Додавання часових ознак перед виявленням аномалій
        if add_time_features:
            if isinstance(processed_data.index, pd.DatetimeIndex):
                self.logger.info("Додавання часових ознак до даних ордербука перед виявленням аномалій...")
                processed_data = self.add_time_features(
                    data=processed_data,
                    cyclical=cyclical,
                    add_sessions=add_sessions
                )
                self.logger.info(f"Додано часові ознаки. Нова кількість колонок: {processed_data.shape[1]}")
            else:
                self.logger.warning("Неможливо додати часові ознаки: індекс не є DatetimeIndex")

        # Виявлення аномалій
        anomalies = self.detect_orderbook_anomalies(processed_data)
        processed_data = pd.concat([processed_data, anomalies], axis=1)

        # Ресемплінг до більшого інтервалу (якщо потрібно)
        if len(processed_data) > 1000:  # Ресемплінг тільки для великих наборів
            processed_data = self.resample_orderbook_data(processed_data, '5min')

            # Перевірка цілісності даних після ресемплінгу
            resample_integrity_issues = self.validate_data_integrity(processed_data)
            if resample_integrity_issues:
                issue_count = sum(len(issues) if isinstance(issues, list) or isinstance(issues, dict) else 1
                                  for issues in resample_integrity_issues.values())
                self.logger.info(f"Після ресемплінгу залишилось {issue_count} проблем з цілісністю даних")

        # Додаємо фільтрацію аномалій, якщо потрібно
        if 'is_anomaly' in processed_data.columns:
            # Зберігаємо відфільтровані дані в атрибуті класу для можливого використання
            self.filtered_data = processed_data[~processed_data['is_anomaly']]

            # Перевірка цілісності відфільтрованих даних
            if not self.filtered_data.empty:
                filtered_integrity_issues = self.validate_data_integrity(self.filtered_data)
                if filtered_integrity_issues:
                    issue_count = sum(len(issues) if isinstance(issues, list) or isinstance(issues, dict) else 1
                                      for issues in filtered_integrity_issues.values())
                    self.logger.info(
                        f"У відфільтрованих даних (без аномалій) залишилось {issue_count} проблем з цілісністю")
                else:
                    self.logger.info("У відфільтрованих даних (без аномалій) проблем з цілісністю не виявлено")

        # Фінальна перевірка цілісності даних перед поверненням результату
        final_integrity_issues = self.validate_data_integrity(processed_data)
        if final_integrity_issues:
            issue_count = sum(len(issues) if isinstance(issues, list) or isinstance(issues, dict) else 1
                              for issues in final_integrity_issues.values())
            self.logger.warning(f"У фінальному наборі даних залишилось {issue_count} проблем з цілісністю")

            # Деталізація проблем, які залишились
            issue_types = list(final_integrity_issues.keys())
            self.logger.info(f"Типи проблем у фінальному наборі даних: {issue_types}")
        else:
            self.logger.info("Фінальний набір даних не має проблем з цілісністю")

        # Розрахунок статистики ордербука
        statistics = self.get_orderbook_statistics(processed_data)
        self.logger.info(f"Розраховано статистику ордербука: {len(statistics)} показників")

        # Зберігаємо статистику для подальшого використання
        self.orderbook_statistics = statistics

        self.logger.info(f"Завершено препроцесинг даних ордербука для {symbol}, рядків: {len(processed_data)}")
        return processed_data

    def update_orderbook_data(self, symbol: str):
        """Оновлює дані ордербука до поточного моменту."""
        # Отримуємо останній доступний запис з використанням limit
        last_entry = self.load_orderbook_data(symbol, limit=1)

        if last_entry.empty:
            start_time = datetime.now() - timedelta(days=7)  # За замовчуванням - останні 7 днів
        else:
            start_time = last_entry.index[-1] + timedelta(seconds=1)  # +1 секунда, щоб уникнути дублювання

        end_time = datetime.now()

        # Логування інформації про оновлення
        self.logger.info(f"Оновлення даних ордербука для {symbol} від {start_time} до {end_time}")

        # Отримуємо нові дані
        new_data = self.fetch_missing_orderbook_data(
            symbol,
            [(start_time, end_time)]
        )

        if not new_data.empty:
            # Зберігаємо нові дані з покращеною обробкою
            for _, row in new_data.iterrows():
                # Виправлено формат даних
                orderbook_data = {
                    'bids': [[row['bid_price'], row['bid_qty']]],
                    'asks': [[row['ask_price'], row['ask_qty']]]
                }
                self.save_orderbook_to_db(orderbook_data, symbol, row.name)

            self.logger.info(f"Оновлено {len(new_data)} записів ордербука для {symbol}")
        else:
            self.logger.info(f"Нових даних для {symbol} не знайдено")

        return new_data












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

            # Виправлено FutureWarning — додано observed=False
            volume_profile = data.groupby('price_bin', observed=False).agg({
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

                if step_name == 'normalize_data':
                    result, _ = method(result, **step_params)
                elif step_name == 'detect_outliers':
                    outliers_df, _ = method(result, **step_params)
                    self.logger.info(f"Виявлено аномалії, але дані не змінено")
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
    INTERVALS = ['5m', '1h']
    processor = MarketDataProcessor()

    for symbol in SYMBOLS:
        for interval in INTERVALS:
            print(f"\n Обробка {symbol} ({interval})...")

            # 1. Завантаження даних з БД
            raw_data = processor.load_data(
                data_source='database',
                symbol=symbol,
                interval=interval,
                data_type='candles'
            )

            if raw_data.empty:
                print(f" Дані не знайдено для {symbol} {interval}")
                continue

            # 2. Обробка відсутніх значень + підтягування з Binance (якщо треба)
            filled_data = processor.handle_missing_values(
                raw_data,
                symbol=symbol,
                interval=interval,
                fetch_missing=True
            )

            # 3. Збереження сирих даних
            processor.save_klines_to_db(filled_data, symbol, interval)
            print(f" Сирі свічки ({interval}) збережено")

            # 4. Попередня обробка (пайплайн)
            processed = processor.preprocess_pipeline(filled_data, symbol=symbol, interval=interval)

            if processed.empty:
                print(f" Обробка не дала результатів для {symbol} {interval}")
                continue

            processed = processor.add_time_features(processed, tz=EU_TIMEZONE)
            processor.save_processed_klines_to_db(processed, symbol, interval)
            print(f" Оброблені свічки ({interval}) збережено")

            # 5. Ресемплінг
            if interval == '5m':
                resampled_30m = processor.resample_data(processed, target_interval='30m')
                resampled_30m = processor.add_time_features(resampled_30m, tz=EU_TIMEZONE)
                processor.save_processed_klines_to_db(resampled_30m, symbol, '30m')
                print(f" 30-хвилинні свічки збережено")

            if interval == '1h':
                resampled_1d = processor.resample_data(processed, target_interval='1d')
                resampled_1d = processor.add_time_features(resampled_1d, tz=EU_TIMEZONE)
                processor.save_processed_klines_to_db(resampled_1d, symbol, '1d')
                print(f" Денні свічки збережено")

                # 6. Профіль об'єму
                volume_profile = processor.aggregate_volume_profile(resampled_1d, bins=12, time_period='1W')
                if not volume_profile.empty:
                    processor.save_volume_profile_to_db(volume_profile, symbol, '1d')
                    print(f" Профіль об'єму збережено")

        # 7. Ордербук
        print(f"\n Ордербук для {symbol}...")
        processed_orderbook = processor.preprocess_orderbook_pipeline(
            symbol=symbol,
            add_time_features=True,
            add_sessions=True
        )

        if not processed_orderbook.empty:
            processor.save_processed_orderbook_to_db(symbol, processed_orderbook)
            print(f" Оброблені дані ордербука збережено")
        else:
            print(f" Не вдалося обробити ордербук для {symbol}")


if __name__ == "__main__":
    main()
