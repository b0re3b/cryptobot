import os
from typing import Optional, Union, Dict, List
import pandas as pd
import numpy as np
from datetime import datetime

class DataStorageManager:
    def __init__(self, db_manager, logger):
        self.db_manager = db_manager
        self.logger = logger
    def save_klines_to_db(self, df: pd.DataFrame, symbol: str, interval: str):

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
        MIN_VALID_DATE = datetime(2000, 1, 1)

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
                    'interval': interval,
                    'open_time': open_time,
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row.get('volume', 0),
                    'close_time': close_time,
                    'quote_asset_volume': row.get('quote_asset_volume', 0),
                    'number_of_trades': row.get('number_of_trades', 0),
                    'taker_buy_base_volume': row.get('taker_buy_base_volume', 0),
                    'taker_buy_quote_volume': row.get('taker_buy_quote_volume', 0),
                    'is_closed': bool(row.get('is_closed', True)),
                }

                kline_data = convert_numpy_types(kline_data)
                self.db_manager.insert_kline(symbol, kline_data)

            except Exception as e:
                self.logger.error(f"❌ Помилка при збереженні свічки для {symbol}: {e}")


    def save_processed_klines_to_db(self, df: pd.DataFrame, symbol: str, interval: str):
        if df.empty:
            self.logger.warning("Спроба зберегти порожні оброблені свічки")
            return

        MIN_VALID_DATE = datetime(2000, 1, 1)

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
                    'interval': interval,
                    'open_time': open_time,
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume'],
                    'price_zscore': row.get('price_zscore'),
                    'volume_zscore': row.get('volume_zscore'),
                    'volatility': row.get('volatility'),
                    'trend': row.get('trend'),
                    'hour': row.get('hour'),
                    'day_of_week': row.get('weekday'),
                    'is_weekend': bool(row.get('is_weekend')),
                    'session': row.get('session', 'unknown'),
                    'is_anomaly': row.get('is_anomaly', False),
                    'has_missing': row.get('has_missing', False)
                }

                self.db_manager.insert_kline_processed(symbol, processed_data)

            except Exception as e:
                self.logger.error(f"❌ Помилка при збереженні обробленої свічки для {symbol}: {e}")


    def save_volume_profile_to_db(self, df: pd.DataFrame, symbol: str, interval: str):
        if df.empty:
            self.logger.warning("Спроба зберегти порожній профіль об'єму")
            return

        for _, row in df.iterrows():
            try:
                profile_data = {
                    'interval': interval,
                    'time_bucket': row.get('period') or row.name,
                    'price_bin_start': row.get('bin_lower'),
                    'price_bin_end': row.get('bin_upper'),
                    'volume': row['volume']
                }
                self.db_manager.insert_volume_profile(symbol, profile_data)
            except Exception as e:
                self.logger.error(f"Помилка при збереженні профілю об'єму: {e}")

    def _load_from_database(self, symbol: str, interval: str,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            data_type: str = 'candles') -> pd.DataFrame:

        self.logger.info(f"Завантаження {data_type} даних з бази даних для {symbol} {interval}")

        try:
            data = None
            if data_type == 'candles':
                data = self.db_manager.get_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=start_date,
                    end_time=end_date
                )
            elif data_type == 'orderbook':
                data = self.db_manager.get_orderbook(
                    symbol=symbol,
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

            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data.set_index('timestamp', inplace=True)

            return data

        except Exception as e:
            self.logger.error(f"Помилка при завантаженні з бази даних: {str(e)}")
            raise

    def load_data(self, data_source: str, symbol: str, interval: str,
                  start_date: Optional[Union[str, datetime]] = None,
                  end_date: Optional[Union[str, datetime]] = None,
                  file_path: Optional[str] = None,
                  data_type: str = 'candles') -> pd.DataFrame:

        start_date_dt = pd.to_datetime(start_date) if start_date else None
        end_date_dt = pd.to_datetime(end_date) if end_date else None

        self.logger.info(f"Завантаження даних з {data_source}: {symbol}, {interval}, {data_type}")

        try:
            if data_source == 'database':
                data = self._load_from_database(
                    symbol,
                    interval,
                    start_date_dt,
                    end_date_dt,
                    data_type
                )
            elif data_source == 'csv':
                if not file_path:
                    raise ValueError("Для джерела 'csv' необхідно вказати шлях до файлу (file_path)")

                self.logger.info(f"Завантаження даних з CSV файлу: {file_path}")
                data = pd.read_csv(file_path)

                if 'timestamp' in data.columns or 'date' in data.columns or 'time' in data.columns:
                    time_col = next((col for col in ['timestamp', 'date', 'time'] if col in data.columns), None)
                    if time_col:
                        data[time_col] = pd.to_datetime(data[time_col])
                        data.set_index(time_col, inplace=True)
                    else:
                        self.logger.warning("Не знайдено часову колонку в CSV файлі")

                if start_date_dt and isinstance(data.index, pd.DatetimeIndex):
                    data = data[data.index >= start_date_dt]
                if end_date_dt and isinstance(data.index, pd.DatetimeIndex):
                    data = data[data.index <= end_date_dt]

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
            db_entry = {
                'timestamp': timestamp,
                'spread': row.get('spread', None),
                'imbalance': row.get('volume_imbalance', None),  # Мапінг volume_imbalance -> imbalance
                'bid_volume': row.get('bid_qty', None),  # Мапінг bid_qty -> bid_volume
                'ask_volume': row.get('ask_qty', None),  # Мапінг ask_qty -> ask_volume
                'average_bid_price': row.get('bid_price', None),  # Спрощений мапінг
                'average_ask_price': row.get('ask_price', None),  # Спрощений мапінг
                'volatility_estimate': row.get('volatility', None),
                'is_anomaly': row.get('is_anomaly', False)
            }
            result.append(db_entry)

        return result



    def save_processed_data(self, data: pd.DataFrame, filename: str, db_connection=None) -> str:

        if data.empty:
            self.logger.warning("Спроба зберегти порожній DataFrame")
            return ""

        # Збереження в базу даних, якщо надано з'єднання
        if db_connection:
            try:
                table_name = os.path.basename(filename).split('.')[0]
                data.to_sql(table_name, db_connection, if_exists='replace', index=True)
                self.logger.info(f"Дані збережено в базу даних, таблиця: {table_name}")
                return table_name
            except Exception as e:
                self.logger.error(f"Помилка при збереженні в базу даних: {str(e)}")
                return ""

        # Забезпечення формату CSV
        if '.' in filename and filename.split('.')[-1].lower() != 'csv':
            filename = f"{filename.split('.')[0]}.csv"
            self.logger.warning(f"Змінено формат файлу на CSV: {filename}")

        if not filename.endswith('.csv'):
            filename = f"{filename}.csv"

        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            self.logger.info(f"Створено директорію: {directory}")

        try:
            data.to_csv(filename)
            self.logger.info(f"Дані збережено у CSV форматі: {filename}")
            return os.path.abspath(filename)
        except Exception as e:
            self.logger.error(f"Помилка при збереженні даних: {str(e)}")
            return ""

    def load_processed_data(self, filename: str) -> pd.DataFrame:

        if not os.path.exists(filename):
            self.logger.error(f"Файл не знайдено: {filename}")
            return pd.DataFrame()

        file_extension = filename.split('.')[-1].lower() if '.' in filename else ''

        try:
            if file_extension == 'csv':
                data = pd.read_csv(filename)
                self.logger.info(f"Дані завантажено з CSV файлу: {filename}")

                time_cols = [col for col in data.columns if
                             any(x in col.lower() for x in ['time', 'date', 'timestamp'])]
                if time_cols:
                    time_col = time_cols[0]
                    data[time_col] = pd.to_datetime(data[time_col])
                    data.set_index(time_col, inplace=True)
                    self.logger.info(f"Встановлено індекс за колонкою {time_col}")
            else:
                self.logger.error(f"Підтримується лише формат CSV, отримано: {file_extension}")
                return pd.DataFrame()

            if not isinstance(data.index, pd.DatetimeIndex) and len(data) > 0:
                self.logger.warning("Завантажені дані не мають DatetimeIndex. Спроба конвертувати.")
                try:
                    time_cols = [col for col in data.columns if
                                 any(x in col.lower() for x in ['time', 'date', 'timestamp'])]
                    if time_cols:
                        time_col = time_cols[0]
                        data[time_col] = pd.to_datetime(data[time_col])
                        data.set_index(time_col, inplace=True)
                        self.logger.info(f"Встановлено індекс за колонкою {time_col}")
                except Exception as e:
                    self.logger.warning(f"Не вдалося встановити DatetimeIndex: {str(e)}")

            return data

        except Exception as e:
            self.logger.error(f"Помилка при завантаженні даних: {str(e)}")
            return pd.DataFrame()