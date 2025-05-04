import os
from typing import Optional, Union, Dict, List
import pandas as pd
import numpy as np
from datetime import datetime


class DataStorageManager:
    def __init__(self, db_manager, logger):
        self.db_manager = db_manager
        self.logger = logger

    def save_klines_to_db(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Зберігає дані свічок у базу даних"""
        if df.empty:
            self.logger.warning("Спроба зберегти порожні свічки")
            return

        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (np.generic, np.bool_)):
                return obj.item()
            else:
                return obj

        # Додано перевірку на мінімальну дату
        MIN_VALID_DATE = datetime(2017, 1, 1)

        for _, row in df.iterrows():
            try:
                open_time = row.name
                if isinstance(open_time, (int, float)):
                    # Додано перевірку на валідність timestamp перед конвертацією
                    if open_time > 0:
                        open_time = pd.to_datetime(open_time, unit='ms')
                    else:
                        continue  # Пропускаємо невалідні timestamp

                # Пропускаємо дати до 2000 року
                if open_time < MIN_VALID_DATE:
                    continue

                close_time = row.get('close_time', open_time)
                if isinstance(close_time, (int, float)):
                    if close_time > 0:
                        close_time = pd.to_datetime(close_time, unit='ms')
                    else:
                        close_time = open_time

                if close_time < MIN_VALID_DATE:
                    close_time = open_time

                kline_data = {
                    'timeframe': timeframe,
                    'open_time': open_time,
                    'open': float(row['open']),  # Явне перетворення на float для безпеки
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row.get('volume', 0)),
                    'close_time': close_time,
                    'quote_asset_volume': float(row.get('quote_asset_volume', 0)),
                    'number_of_trades': int(row.get('number_of_trades', 0)),
                    'taker_buy_base_volume': float(row.get('taker_buy_base_volume', 0)),
                    'taker_buy_quote_volume': float(row.get('taker_buy_quote_volume', 0)),
                    'is_closed': bool(row.get('is_closed', True)),
                }

                kline_data = convert_numpy_types(kline_data)
                self.db_manager.insert_kline(symbol, kline_data)

            except Exception as e:
                self.logger.error(f"❌ Помилка при збереженні свічки для {symbol}: {e}")

    def save_processed_klines_to_db(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Зберігає оброблені дані свічок у базу даних"""
        if df.empty:
            self.logger.warning("Спроба зберегти порожні оброблені свічки")
            return

        MIN_VALID_DATE = datetime(2017, 1, 1)

        for _, row in df.iterrows():
            try:
                open_time = row.name
                if isinstance(open_time, (int, float)):
                    if open_time > 0:
                        open_time = pd.to_datetime(open_time, unit='ms')
                    else:
                        continue

                if open_time < MIN_VALID_DATE:
                    continue

                processed_data = {
                    'timeframe': timeframe,
                    'open_time': open_time,
                    'open': float(row['open']),  # Явне перетворення типів
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume']),
                    'price_zscore': float(row.get('price_zscore')) if pd.notna(row.get('price_zscore')) else None,
                    'volume_zscore': float(row.get('volume_zscore')) if pd.notna(row.get('volume_zscore')) else None,
                    'volatility': float(row.get('volatility')) if pd.notna(row.get('volatility')) else None,
                    'trend': row.get('trend'),
                    'hour': int(row.get('hour')) if pd.notna(row.get('hour')) else None,
                    'day_of_week': int(row.get('weekday')) if pd.notna(row.get('weekday')) else None,
                    'is_weekend': bool(row.get('is_weekend')),
                    'session': row.get('session', 'unknown'),
                    'is_anomaly': bool(row.get('is_anomaly', False)),
                    'has_missing': bool(row.get('has_missing', False))
                }

                # Конвертуємо numpy типи
                processed_data = {k: v.item() if isinstance(v, np.generic) else v for k, v in processed_data.items()}

                self.db_manager.insert_kline_processed(symbol, processed_data)

            except Exception as e:
                self.logger.error(f"❌ Помилка при збереженні обробленої свічки для {symbol}: {e}")

    def save_volume_profile_to_db(self, df: pd.DataFrame, symbol: str, interval: str):
        """Зберігає профіль об'єму в базу даних"""
        if df.empty:
            self.logger.warning("Спроба зберегти порожній профіль об'єму")
            return

        for _, row in df.iterrows():
            try:
                time_bucket = row.get('period') if pd.notna(row.get('period')) else row.name

                profile_data = {
                    'interval': interval,
                    'time_bucket': time_bucket,
                    'price_bin_start': float(row.get('bin_lower')),
                    'price_bin_end': float(row.get('bin_upper')),
                    'volume': float(row['volume'])
                }

                # Конвертуємо numpy типи
                profile_data = {k: v.item() if isinstance(v, np.generic) else v for k, v in profile_data.items()}

                self.db_manager.insert_volume_profile(symbol, profile_data)
            except Exception as e:
                self.logger.error(f"Помилка при збереженні профілю об'єму: {e}")

    def _load_from_database(self, symbol: str, timeframe: str,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            data_type: str = 'candles') -> pd.DataFrame:
        """Завантажує дані з бази даних"""
        self.logger.info(f"Завантаження {data_type} даних з бази даних для {symbol} {timeframe}")

        try:
            data = None
            if data_type == 'candles':
                data = self.db_manager.get_klines(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_date,
                    end_time=end_date
                )

            elif data_type == 'processed_candles':
                data = self.db_manager.get_processed_klines(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_date,
                    end_time=end_date
                )
            elif data_type == 'volume_profile':
                data = self.db_manager.get_volume_profile(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_date,
                    end_time=end_date
                )
            else:
                raise ValueError(f"Непідтримуваний тип даних: {data_type}")

            if data is None:
                self.logger.warning("База даних повернула None")
                return pd.DataFrame()

            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)

            if data.empty:
                self.logger.warning(f"Отримано порожній набір даних для {symbol} {timeframe}")
                return pd.DataFrame()

            # Стандартизація часового індексу
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data.set_index('timestamp', inplace=True)
            elif 'open_time' in data.columns:
                data['open_time'] = pd.to_datetime(data['open_time'])
                data.set_index('open_time', inplace=True)
            elif 'time_bucket' in data.columns:
                data['time_bucket'] = pd.to_datetime(data['time_bucket'])
                data.set_index('time_bucket', inplace=True)

            return data

        except Exception as e:
            self.logger.error(f"Помилка при завантаженні з бази даних: {str(e)}")
            raise

    def load_data(self, data_source: str, symbol: str, timeframe: str,
                  start_date: Optional[Union[str, datetime]] = None,
                  end_date: Optional[Union[str, datetime]] = None,
                  data_type: str = 'candles') -> pd.DataFrame:
        """Завантажує дані з вказаного джерела"""
        start_date_dt = pd.to_datetime(start_date) if start_date else None
        end_date_dt = pd.to_datetime(end_date) if end_date else None

        self.logger.info(f"Завантаження даних з {data_source}: {symbol}, {timeframe}, {data_type}")

        try:
            if data_source == 'database':
                data = self._load_from_database(
                    symbol,
                    timeframe,
                    start_date_dt,
                    end_date_dt,
                    data_type
                )
            else:
                raise ValueError(f"Непідтримуване джерело даних: {data_source}")

            if data is None or data.empty:
                self.logger.warning(f"Отримано порожній набір даних від {data_source}")
                return pd.DataFrame()

            return data

        except Exception as e:
            self.logger.error(f"Помилка при завантаженні даних: {str(e)}")
            raise

    def prepare_data_for_db(self, processed_data: pd.DataFrame) -> List[Dict]:
        """Перетворює оброблені дані ордербука у формат для збереження в БД"""
        result = []

        for timestamp, row in processed_data.iterrows():
            try:
                db_entry = {
                    'timestamp': timestamp,
                    'spread': float(row.get('spread')) if pd.notna(row.get('spread')) else None,
                    'imbalance': float(row.get('volume_imbalance')) if pd.notna(row.get('volume_imbalance')) else None,
                    'bid_volume': float(row.get('bid_qty')) if pd.notna(row.get('bid_qty')) else None,
                    'ask_volume': float(row.get('ask_qty')) if pd.notna(row.get('ask_qty')) else None,
                    'average_bid_price': float(row.get('bid_price')) if pd.notna(row.get('bid_price')) else None,
                    'average_ask_price': float(row.get('ask_price')) if pd.notna(row.get('ask_price')) else None,
                    'volatility_estimate': float(row.get('volatility')) if pd.notna(row.get('volatility')) else None,
                    'is_anomaly': bool(row.get('is_anomaly', False))
                }

                # Конвертуємо numpy типи
                db_entry = {k: v.item() if isinstance(v, np.generic) else v for k, v in db_entry.items()}

                result.append(db_entry)
            except Exception as e:
                self.logger.error(f"Помилка при підготовці даних ордербука: {e}")

        return result

    def save_processed_data(self, data: pd.DataFrame, table_name: str, db_connection=None) -> str:

        if data.empty:
            self.logger.warning("Спроба зберегти порожній DataFrame")
            return ""

        # Отримуємо назву таблиці з параметра або шляху файлу
        if os.path.basename(table_name):
            actual_table_name = os.path.basename(table_name).split('.')[0]
        else:
            actual_table_name = table_name

        # Перевіряємо, чи є назва таблиці
        if not actual_table_name:
            self.logger.error("Не вказано назву таблиці")
            return ""

        # Збереження в базу даних
        if db_connection:
            try:
                # Конвертуємо numpy типи перед збереженням
                for col in data.select_dtypes(include=[np.number]).columns:
                    data[col] = data[col].astype(float)

                data.to_sql(actual_table_name, db_connection, if_exists='replace', index=True)
                self.logger.info(f"Дані збережено в базу даних, таблиця: {actual_table_name}")
                return actual_table_name
            except Exception as e:
                self.logger.error(f"Помилка при збереженні в базу даних: {str(e)}")
                return ""
        else:
            try:
                # Якщо не передано з'єднання, використовуємо наш DB Manager
                processed_data = []
                for idx, row in data.iterrows():
                    record = {'timestamp': idx}
                    for col, val in row.items():
                        # Перетворюємо numpy типи
                        if isinstance(val, np.generic):
                            record[col] = val.item()
                        else:
                            record[col] = val
                    processed_data.append(record)

                # Викликаємо метод збереження в БД через наш менеджер
                self.db_manager.insert_processed_data(actual_table_name, processed_data)
                self.logger.info(f"Дані збережено в базу даних через DB Manager, таблиця: {actual_table_name}")
                return actual_table_name
            except Exception as e:
                self.logger.error(f"Помилка при збереженні в базу даних через DB Manager: {str(e)}")
                return ""

    def load_processed_data(self, table_name: str, db_manager) -> pd.DataFrame:

        # Очищуємо назву таблиці від шляху або розширення файлу, якщо вони є
        if os.path.basename(table_name):
            actual_table_name = os.path.basename(table_name).split('.')[0]
        else:
            actual_table_name = table_name

        try:
            # Якщо передано пряме з'єднання з БД, використовуємо його
            if db_manager:
                self.logger.info(f"Завантаження даних з таблиці {actual_table_name}")
                try:
                    data = pd.read_sql_table(actual_table_name, db_manager)

                    # Встановлюємо часовий індекс, якщо є підходящі колонки
                    time_cols = [col for col in data.columns if
                                 any(x in col.lower() for x in ['time', 'date', 'timestamp'])]
                    if time_cols:
                        time_col = time_cols[0]
                        data[time_col] = pd.to_datetime(data[time_col])
                        data.set_index(time_col, inplace=True)
                        self.logger.info(f"Встановлено індекс за колонкою {time_col}")

                    return data
                except Exception as e:
                    self.logger.error(f"Помилка при завантаженні з бази даних: {str(e)}")
                    return pd.DataFrame()
            else:
                # В іншому випадку використовуємо наш менеджер БД
                self.logger.info(f"Завантаження даних з таблиці {actual_table_name} через DB Manager")
                data = self.db_manager.get_processed_data(actual_table_name)

                if data is None:
                    self.logger.warning(f"Не знайдено даних у таблиці {actual_table_name}")
                    return pd.DataFrame()

                if not isinstance(data, pd.DataFrame):
                    data = pd.DataFrame(data)

                # Встановлюємо часовий індекс
                if 'timestamp' in data.columns:
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                    data.set_index('timestamp', inplace=True)

                return data
        except Exception as e:
            self.logger.error(f"Помилка при завантаженні даних: {str(e)}")
            return pd.DataFrame()