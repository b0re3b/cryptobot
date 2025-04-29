from datetime import datetime, time
from typing import Dict,List,Optional,Tuple

import numpy as np
import pandas as pd


class OrderBookProcessor:

    def __init__(self, db_manager, logger):
        self.db_manager = db_manager
        self.logger = logger

    def save_orderbook_to_db(self, orderbook_data: Dict, symbol: str, timestamp: datetime):
        """Зберігає дані ордербука в базу даних."""
        try:
            if not orderbook_data or 'bids' not in orderbook_data or 'asks' not in orderbook_data:
                self.logger.warning(f"Невірний формат ордербука для {symbol}")
                return

            # Конвертуємо numpy типи до Python native
            def convert_numpy_types(obj):
                if isinstance(obj, np.generic):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_numpy_types(x) for x in obj]
                return obj

            orderbook_data = convert_numpy_types(orderbook_data)

            # Додаємо timestamp до даних
            orderbook_data['timestamp'] = timestamp
            orderbook_data['symbol'] = symbol

            self.db_manager.insert_orderbook(symbol, orderbook_data)
            self.logger.info(f"Збережено ордербук для {symbol} на {timestamp}")

        except Exception as e:
            self.logger.error(f"Помилка при збереженні ордербука: {e}")

    def load_orderbook_data(self, symbol: str,
                            start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None,
                            limit: Optional[int] = None) -> pd.DataFrame:

        try:
            data = self.db_manager.get_orderbook(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                limit=limit  # Передаємо limit в db_manager
            )

            if not data:
                self.logger.warning(f"Ордербук для {symbol} не знайдено")
                return pd.DataFrame()

            df = pd.DataFrame(data)

            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)

            return df

        except Exception as e:
            self.logger.error(f"Помилка при завантаженні ордербука: {e}")
            return pd.DataFrame()

    def fetch_missing_orderbook_data(self, symbol: str,
                                     missing_periods: List[Tuple[datetime, datetime]]) -> pd.DataFrame:
        """Отримує відсутні дані ордербука з Binance API."""
        try:

            if not hasattr(self, 'binance_client'):
                from binance.client import Client
                from utils.config import get_binance_keys  # Імпортуємо з конфіг файлу

                api_key, api_secret = get_binance_keys()

                if not api_key or not api_secret:
                    self.logger.error("Не знайдено API ключі Binance")
                    return pd.DataFrame()

                self.binance_client = Client(api_key, api_secret)

            all_data = []

            for start, end in missing_periods:
                try:
                    # Конвертуємо в мілісекунди
                    start_ms = int(start.timestamp() * 1000)
                    end_ms = int(end.timestamp() * 1000)

                    # Додаємо повторні спроби при мережевих помилках
                    max_retries = 3
                    retry_delay = 2  # секунди

                    for attempt in range(max_retries):
                        try:
                            # Отримуємо дані з Binance
                            depth_data = self.binance_client.get_aggregate_trades(
                                symbol=symbol,
                                startTime=start_ms,
                                endTime=end_ms
                            )
                            break  # Виходимо з циклу при успіху
                        except Exception as retry_error:
                            if attempt < max_retries - 1:
                                self.logger.warning(
                                    f"Спроба {attempt + 1} невдала, повторна спроба через {retry_delay}с: {retry_error}")
                                time.sleep(retry_delay)
                                retry_delay *= 2  # Експоненційне збільшення затримки
                            else:
                                raise  # Перекидаємо винятки після всіх спроб

                    if not depth_data:
                        continue

                    # Обробка даних
                    processed = self._process_raw_orderbook(depth_data, symbol)
                    all_data.append(processed)

                except Exception as e:
                    self.logger.error(f"Помилка при запиті до Binance: {e}")
                    continue

            if not all_data:
                return pd.DataFrame()

            return pd.concat(all_data) if len(all_data) > 1 else all_data[0]

        except ImportError:
            self.logger.error("Модуль python-binance не встановлено")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Критична помилка при отриманні даних: {e}")
            return pd.DataFrame()

    def _process_raw_orderbook(self, raw_data: List, symbol: str) -> pd.DataFrame:

        if not raw_data:
            return pd.DataFrame()

        try:
            # Створюємо DataFrame одразу, щоб уникнути проміжних копій
            df = pd.DataFrame(raw_data)

            # Перетворюємо timestamp
            df['timestamp'] = pd.to_datetime(df['T'], unit='ms')

            # Перейменовуємо і перетворюємо колонки
            df_processed = pd.DataFrame({
                'symbol': symbol,
                'timestamp': df['timestamp'],
                'price': df['p'].astype(float),
                'quantity': df['q'].astype(float),
                'is_buyer_maker': df['m']
            })

            # Розділення на bids і asks на основі is_buyer_maker
            df_processed['bid_price'] = np.where(df_processed['is_buyer_maker'], df_processed['price'], np.nan)
            df_processed['bid_qty'] = np.where(df_processed['is_buyer_maker'], df_processed['quantity'], np.nan)
            df_processed['ask_price'] = np.where(~df_processed['is_buyer_maker'], df_processed['price'], np.nan)
            df_processed['ask_qty'] = np.where(~df_processed['is_buyer_maker'], df_processed['quantity'], np.nan)

            # Видаляємо проміжні колонки
            df_processed.drop(['price', 'quantity'], axis=1, inplace=True)

            # Встановлюємо індекс
            df_processed.set_index('timestamp', inplace=True)

            return df_processed

        except Exception as e:
            self.logger.warning(f"Помилка обробки даних ордербука: {e}")
            return pd.DataFrame()

    def process_orderbook_data(self, data: pd.DataFrame, add_time_features: bool = False,
                               cyclical: bool = True, add_sessions: bool = False) -> pd.DataFrame:
        """Обробляє дані ордербука (нормалізація, розрахунок метрик)."""
        if data.empty:
            return data

        # Перевіряємо цілісність даних перед обробкою
        integrity_issues = self.validate_data_integrity(data)
        if integrity_issues:
            issue_count = sum(len(issues) if isinstance(issues, list) or isinstance(issues, dict) else 1
                              for issues in integrity_issues.values())
            self.logger.warning(f"Перед обробкою ордербука знайдено {issue_count} проблем з цілісністю даних")

            # Перевіряємо наявність ключових колонок для ордербука
            if "missing_columns" in integrity_issues:
                missing_cols = integrity_issues["missing_columns"]
                orderbook_key_cols = [col for col in ['bid_price', 'ask_price', 'bid_qty', 'ask_qty'] if
                                      col in missing_cols]
                if orderbook_key_cols:
                    self.logger.error(f"Відсутні критичні колонки для ордербука: {orderbook_key_cols}")

        processed = data.copy()

        # Розрахунок спреду - використовуємо прогресивну перевірку наявності колонок
        if all(col in processed.columns for col in ['bid_price', 'ask_price']):
            # Перевіряємо на NaN значення
            valid_rows = ~(processed['bid_price'].isna() | processed['ask_price'].isna())
            if valid_rows.any():
                processed.loc[valid_rows, 'spread'] = processed.loc[valid_rows, 'ask_price'] - processed.loc[
                    valid_rows, 'bid_price']
                processed.loc[valid_rows, 'mid_price'] = (processed.loc[valid_rows, 'ask_price'] + processed.loc[
                    valid_rows, 'bid_price']) / 2

        # Розрахунок об'ємів
        if all(col in processed.columns for col in ['bid_qty', 'ask_qty']):
            # Заповнюємо NaN для кращої обробки
            bid_qty = processed['bid_qty'].fillna(0)
            ask_qty = processed['ask_qty'].fillna(0)

            processed['total_volume'] = bid_qty + ask_qty

            # Уникаємо ділення на нуль
            valid_volume = processed['total_volume'] > 0
            if valid_volume.any():
                processed.loc[valid_volume, 'volume_imbalance'] = (bid_qty.loc[valid_volume] - ask_qty.loc[
                    valid_volume]) / processed.loc[valid_volume, 'total_volume']

        # Додаткові метрики
        if 'mid_price' in processed.columns:
            processed['price_change'] = processed['mid_price'].pct_change()

            # Додамо волатильність
            if len(processed) >= 10:  # Достатньо даних для розрахунку волатильності
                processed['volatility'] = processed['mid_price'].rolling(10).std() / processed['mid_price']

        # Додаємо часові ознаки, якщо потрібно
        if add_time_features:
            if isinstance(processed.index, pd.DatetimeIndex):
                self.logger.info("Додавання часових ознак до даних ордербука...")
                processed = self.add_time_features(
                    data=processed,
                    cyclical=cyclical,
                    add_sessions=add_sessions
                )
            else:
                self.logger.warning("Неможливо додати часові ознаки: індекс не є DatetimeIndex")

        # Нормалізація числових даних
        # Виключаємо часові або ідентифікаційні колонки з нормалізації
        exclude_cols = ['timestamp', 'datetime', 'date', 'time', 'id', 'symbol', 'pair']
        # Визначаємо колонки для нормалізації - всі числові крім виключених
        numeric_cols = processed.select_dtypes(include=[np.number]).columns.tolist()
        normalize_cols = [col for col in numeric_cols if col not in exclude_cols]

        # Застосовуємо робастну нормалізацію, яка менш чутлива до викидів
        processed, scaler_meta = self.normalize_data(
            data=processed,
            method='robust',
            columns=normalize_cols,
            exclude_columns=None
        )

        # Перевіряємо цілісність даних після обробки
        post_integrity_issues = self.validate_data_integrity(processed)
        if post_integrity_issues:
            issue_count = sum(len(issues) if isinstance(issues, list) or isinstance(issues, dict) else 1
                              for issues in post_integrity_issues.values())
            self.logger.info(f"Після обробки ордербука залишилось {issue_count} проблем з цілісністю даних")

            # Перевіряємо, чи виникли нові проблеми після обробки
            new_issues = set(post_integrity_issues.keys()) - set(integrity_issues.keys() if integrity_issues else [])
            if new_issues:
                self.logger.warning(f"Після обробки ордербука виникли нові типи проблем: {new_issues}")

            # Особлива увага до аномалій у розрахованих значеннях
            if "price_jumps_mid_price" in post_integrity_issues or "price_jumps_spread" in post_integrity_issues:
                self.logger.warning("Виявлено аномалії у розрахованих цінових метриках")
        else:
            self.logger.info("Після обробки ордербука проблем з цілісністю даних не виявлено")

        return processed

    def detect_orderbook_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Виявляє аномалії в даних ордербука з покращеними методами."""
        if data.empty:
            return pd.DataFrame()

        anomalies = pd.DataFrame(index=data.index)

        # Виявлення аномальних спредів з більш надійним методом
        if 'spread' in data.columns:
            # Використовуємо медіану і MAD замість середнього і стандартного відхилення
            spread_median = data['spread'].median()
            spread_mad = (data['spread'] - spread_median).abs().median() * 1.4826  # константа для нормального розподілу

            if spread_mad > 0:
                robust_z_scores = (data['spread'] - spread_median) / spread_mad
                anomalies['spread_anomaly'] = np.abs(robust_z_scores) > 3.5  # трохи вищий поріг

        # Виявлення аномальних об'ємів
        if 'total_volume' in data.columns:
            # Використання логарифмічного перетворення для об'ємів
            log_volume = np.log1p(data['total_volume'])  # log(1+x) для уникнення log(0)
            volume_median = log_volume.median()
            volume_mad = (log_volume - volume_median).abs().median() * 1.4826

            if volume_mad > 0:
                volume_scores = (log_volume - volume_median) / volume_mad
                anomalies['volume_anomaly'] = np.abs(volume_scores) > 3.5

        # Виявлення різких змін ціни з адаптивним порогом
        if 'price_change' in data.columns:
            if len(data) >= 20:
                # Адаптивний поріг на основі історичної волатильності
                rolling_std = data['price_change'].rolling(20).std()
                valid_std = rolling_std > 0

                if valid_std.any():
                    adaptive_threshold = 3 * rolling_std
                    anomalies.loc[valid_std, 'price_jump'] = np.abs(
                        data.loc[valid_std, 'price_change']) > adaptive_threshold
            else:
                # Фіксований поріг для малої кількості даних
                anomalies['price_jump'] = np.abs(data['price_change']) > 0.05

        # Додаємо виявлення аномалій в обсязі торгів
        if 'total_volume' in data.columns and len(data) > 20:
            # Використовуємо ковзне вікно для аналізу тренду об'єму
            rolling_volume = data['total_volume'].rolling(10).mean()
            volume_ratio = data['total_volume'] / rolling_volume

            # Заповнюємо NaN для перших рядків
            volume_ratio = volume_ratio.fillna(1)

            # Виявлення незвичайних сплесків або падінь об'єму
            anomalies['volume_spike'] = volume_ratio > 3  # об'єм втричі більший за середній
            anomalies['volume_drop'] = volume_ratio < 0.3  # об'єм втричі менший за середній

        anomalies['is_anomaly'] = anomalies.any(axis=1)
        return anomalies

    def resample_orderbook_data(self, data: pd.DataFrame, interval: str) -> pd.DataFrame:
        """Ресемплінг даних ордербука до заданого інтервалу."""
        if data.empty:
            return data

        if not isinstance(data.index, pd.DatetimeIndex):
            self.logger.warning("Для ресемплінгу потрібен DatetimeIndex")
            return data

        pandas_interval = self._convert_interval_to_pandas_format(interval)

        # Агрегація даних з оптимізованими правилами
        agg_rules = {
            'bid_price': 'ohlc',
            'ask_price': 'ohlc',
            'bid_qty': 'sum',
            'ask_qty': 'sum',
            'spread': ['mean', 'min', 'max'],  # Розширена агрегація для спреду
            'mid_price': 'ohlc',
            'total_volume': 'sum',
            'volume_imbalance': 'mean'
        }

        # Фільтруємо тільки наявні колонки
        agg_rules = {k: v for k, v in agg_rules.items() if k in data.columns}

        # Проводимо ресемплінг з кешуванням
        resampled = data.resample(pandas_interval).agg(agg_rules)

        # Виправлення мульті-індексу після агрегації з кращими іменами
        if isinstance(resampled.columns, pd.MultiIndex):
            # Використовуємо більш зрозумілий формат іменування
            new_columns = []
            for col in resampled.columns:
                # Приклад перейменування: ('bid_price', 'open') -> 'bid_price_open'
                if isinstance(col, tuple) and len(col) == 2:
                    metric, agg = col
                    # Використовуємо повні назви для відкриття/закриття
                    if agg == 'o':
                        agg = 'open'
                    elif agg == 'h':
                        agg = 'high'
                    elif agg == 'l':
                        agg = 'low'
                    elif agg == 'c':
                        agg = 'close'
                    new_columns.append(f"{metric}_{agg}")
                else:
                    new_columns.append("_".join(col).strip())

            resampled.columns = new_columns

        # Додаємо додаткові обчислювані метрики після ресемплінгу
        if all(col in resampled.columns for col in ['mid_price_open', 'mid_price_close']):
            resampled['price_change_pct'] = (resampled['mid_price_close'] - resampled['mid_price_open']) / resampled[
                'mid_price_open'] * 100

        return resampled

    def get_orderbook_statistics(self, data: pd.DataFrame) -> Dict:
        """Розраховує статистику ордербука."""
        if data.empty:
            return {}

        stats = {}

        # Базова статистика
        for col in data.select_dtypes(include=[np.number]).columns:
            stats[f"{col}_mean"] = data[col].mean()
            stats[f"{col}_median"] = data[col].median()
            stats[f"{col}_std"] = data[col].std()

        # Додаткова статистика для спреду
        if 'spread' in data.columns:
            stats['avg_spread_bps'] = (data['spread'] / data['mid_price']).mean() * 10000  # в базисних пунктах
            stats['max_spread_bps'] = (data['spread'] / data['mid_price']).max() * 10000

        # Статистика об'ємів
        if 'total_volume' in data.columns:
            stats['total_traded_volume'] = data['total_volume'].sum()
            stats['avg_trade_size'] = data['total_volume'].mean()

        # Статистика аномалій
        if 'is_anomaly' in data.columns:
            stats['anomaly_rate'] = data['is_anomaly'].mean() * 100  # відсоток аномалій

        return stats

    def save_processed_orderbook_to_db(self, symbol: str, processed_data: pd.DataFrame) -> bool:
        """Зберігає оброблені дані ордербука в БД"""
        if processed_data.empty:
            self.logger.warning(f"Порожній DataFrame для збереження")
            return False

        formatted_data = self.prepare_data_for_db(processed_data)

        success_count = 0
        for entry in formatted_data:
            result = self.db_manager.insert_orderbook_processed(symbol, entry)
            if result:
                success_count += 1

        self.logger.info(f"Збережено {success_count}/{len(formatted_data)} записів для {symbol}")
        return success_count > 0