from typing import Optional, Union, Dict, List
import pandas as pd
import numpy as np
from datetime import datetime
from data.db import DatabaseManager


class DataStorageManager:
    def __init__(self, logger):
        self.db_manager = DatabaseManager()
        self.logger = logger

        # Визначення списку підтримуваних криптовалют та моделей
        self.supported_cryptos = ['BTC', 'ETH', 'SOL']
        self.supported_models = ['LSTM', 'ARIMA']
        self.supported_timeframes = ['1m', '1h', '1d', '4h', '1w']

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

    # Покращені методи для збереження даних моделей для кожної криптовалюти з урахуванням часових проміжків
    def save_lstm_sequence(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Зберігає послідовності LSTM для конкретної криптовалюти та часового проміжку"""
        if df.empty:
            self.logger.warning(f"Спроба зберегти порожні LSTM послідовності для {symbol} ({timeframe})")
            return False

        # Конвертуємо символ до верхнього регістру для стандартизації
        symbol = symbol.upper()

        # Перевірка підтримуваних криптовалют та часових проміжків
        if symbol not in self.supported_cryptos:
            self.logger.error(f"Непідтримувана криптовалюта: {symbol}")
            return False

        if timeframe not in self.supported_timeframes:
            self.logger.error(f"Непідтримуваний часовий проміжок: {timeframe}")
            return False

        try:
            # Вибір відповідного методу збереження з урахуванням часового проміжку
            if symbol == 'BTC':
                return self.save_btc_lstm_sequence(df, timeframe)
            elif symbol == 'ETH':
                return self.save_eth_lstm_sequence(df, timeframe)
            elif symbol == 'SOL':
                return self.save_sol_lstm_sequence(df, timeframe)
            else:
                self.logger.error(f"Немає відповідного методу для збереження LSTM послідовностей для {symbol}")
                return False
        except Exception as e:
            self.logger.error(f"Помилка при виборі методу збереження LSTM послідовностей для {symbol}: {str(e)}")
            return False

    def save_arima_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Зберігає дані ARIMA для конкретної криптовалюти та часового проміжку"""
        if df.empty:
            self.logger.warning(f"Спроба зберегти порожні ARIMA дані для {symbol} ({timeframe})")
            return False

        # Конвертуємо символ до верхнього регістру для стандартизації
        symbol = symbol.upper()

        # Перевірка підтримуваних криптовалют та часових проміжків
        if symbol not in self.supported_cryptos:
            self.logger.error(f"Непідтримувана криптовалюта: {symbol}")
            return False

        if timeframe not in self.supported_timeframes:
            self.logger.error(f"Непідтримуваний часовий проміжок: {timeframe}")
            return False

        try:
            # Вибір відповідного методу збереження з урахуванням часового проміжку
            if symbol == 'BTC':
                return self.save_btc_arima_data(df, timeframe)
            elif symbol == 'ETH':
                return self.save_eth_arima_data(df, timeframe)
            elif symbol == 'SOL':
                return self.save_sol_arima_data(df, timeframe)
            else:
                self.logger.error(f"Немає відповідного методу для збереження ARIMA даних для {symbol}")
                return False
        except Exception as e:
            self.logger.error(f"Помилка при виборі методу збереження ARIMA даних для {symbol}: {str(e)}")
            return False

    def save_btc_lstm_sequence(self, df: pd.DataFrame, timeframe: str):
        """Зберігає послідовності LSTM для Bitcoin з урахуванням часового проміжку"""
        table_name = f"btc_lstm_sequence_{timeframe}"

        try:
            # Конвертуємо numpy типи перед збереженням
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df[col].astype(float)

            # Підготовка даних для збереження
            processed_data = []
            for idx, row in df.iterrows():
                record = {
                    'timestamp': idx if not isinstance(idx, (int, float)) else pd.to_datetime(idx, unit='ms'),
                    'timeframe': timeframe  # Додаємо часовий проміжок як поле
                }
                for col, val in row.items():
                    # Перетворюємо numpy типи
                    if isinstance(val, np.generic):
                        record[col] = val.item()
                    else:
                        record[col] = val
                processed_data.append(record)

            # Збереження через менеджер БД
            self.db_manager.save_btc_lstm_sequence(processed_data, timeframe)
            self.logger.info(f"LSTM послідовності для BTC ({timeframe}) збережено в таблицю {table_name}")
            return True
        except Exception as e:
            self.logger.error(f"Помилка при збереженні LSTM послідовностей для BTC ({timeframe}): {str(e)}")
            return False

    def save_eth_lstm_sequence(self, df: pd.DataFrame, timeframe: str):
        """Зберігає послідовності LSTM для Ethereum з урахуванням часового проміжку"""
        table_name = f"eth_lstm_sequence_{timeframe}"

        try:
            # Конвертуємо numpy типи перед збереженням
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df[col].astype(float)

            # Підготовка даних для збереження
            processed_data = []
            for idx, row in df.iterrows():
                record = {
                    'timestamp': idx if not isinstance(idx, (int, float)) else pd.to_datetime(idx, unit='ms'),
                    'timeframe': timeframe  # Додаємо часовий проміжок як поле
                }
                for col, val in row.items():
                    # Перетворюємо numpy типи
                    if isinstance(val, np.generic):
                        record[col] = val.item()
                    else:
                        record[col] = val
                processed_data.append(record)

            # Збереження через менеджер БД
            self.db_manager.save_eth_lstm_sequence(processed_data, timeframe)
            self.logger.info(f"LSTM послідовності для ETH ({timeframe}) збережено в таблицю {table_name}")
            return True
        except Exception as e:
            self.logger.error(f"Помилка при збереженні LSTM послідовностей для ETH ({timeframe}): {str(e)}")
            return False

    def save_sol_lstm_sequence(self, df: pd.DataFrame, timeframe: str):
        """Зберігає послідовності LSTM для Solana з урахуванням часового проміжку"""
        table_name = f"sol_lstm_sequence_{timeframe}"

        try:
            # Конвертуємо numpy типи перед збереженням
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df[col].astype(float)

            # Підготовка даних для збереження
            processed_data = []
            for idx, row in df.iterrows():
                record = {
                    'timestamp': idx if not isinstance(idx, (int, float)) else pd.to_datetime(idx, unit='ms'),
                    'timeframe': timeframe  # Додаємо часовий проміжок як поле
                }
                for col, val in row.items():
                    # Перетворюємо numpy типи
                    if isinstance(val, np.generic):
                        record[col] = val.item()
                    else:
                        record[col] = val
                processed_data.append(record)

            # Збереження через менеджер БД
            self.db_manager.save_sol_lstm_sequence(processed_data, timeframe)
            self.logger.info(f"LSTM послідовності для SOL ({timeframe}) збережено в таблицю {table_name}")
            return True
        except Exception as e:
            self.logger.error(f"Помилка при збереженні LSTM послідовностей для SOL ({timeframe}): {str(e)}")
            return False

    def save_btc_arima_data(self, df: pd.DataFrame, timeframe: str):
        """Зберігає дані ARIMA для Bitcoin з урахуванням часового проміжку"""
        table_name = f"btc_arima_data_{timeframe}"

        try:
            # Конвертуємо numpy типи перед збереженням
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df[col].astype(float)

            # Підготовка даних для збереження
            processed_data = []
            for idx, row in df.iterrows():
                record = {
                    'timestamp': idx if not isinstance(idx, (int, float)) else pd.to_datetime(idx, unit='ms'),
                    'timeframe': timeframe  # Додаємо часовий проміжок як поле
                }
                for col, val in row.items():
                    # Перетворюємо numpy типи
                    if isinstance(val, np.generic):
                        record[col] = val.item()
                    else:
                        record[col] = val
                processed_data.append(record)

            # Збереження через менеджер БД
            self.db_manager.save_btc_arima_data(processed_data, timeframe)
            self.logger.info(f"ARIMA дані для BTC ({timeframe}) збережено в таблицю {table_name}")
            return True
        except Exception as e:
            self.logger.error(f"Помилка при збереженні ARIMA даних для BTC ({timeframe}): {str(e)}")
            return False

    def save_eth_arima_data(self, df: pd.DataFrame, timeframe: str):
        """Зберігає дані ARIMA для Ethereum з урахуванням часового проміжку"""
        table_name = f"eth_arima_data_{timeframe}"

        try:
            # Конвертуємо numpy типи перед збереженням
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df[col].astype(float)

            # Підготовка даних для збереження
            processed_data = []
            for idx, row in df.iterrows():
                record = {
                    'timestamp': idx if not isinstance(idx, (int, float)) else pd.to_datetime(idx, unit='ms'),
                    'timeframe': timeframe  # Додаємо часовий проміжок як поле
                }
                for col, val in row.items():
                    # Перетворюємо numpy типи
                    if isinstance(val, np.generic):
                        record[col] = val.item()
                    else:
                        record[col] = val
                processed_data.append(record)

            # Збереження через менеджер БД
            self.db_manager.save_eth_arima_data(processed_data, timeframe)
            self.logger.info(f"ARIMA дані для ETH ({timeframe}) збережено в таблицю {table_name}")
            return True
        except Exception as e:
            self.logger.error(f"Помилка при збереженні ARIMA даних для ETH ({timeframe}): {str(e)}")
            return False

    def save_sol_arima_data(self, df: pd.DataFrame, timeframe: str):
        """Зберігає дані ARIMA для Solana з урахуванням часового проміжку"""
        table_name = f"sol_arima_data_{timeframe}"

        try:
            # Конвертуємо numpy типи перед збереженням
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df[col].astype(float)

            # Підготовка даних для збереження
            processed_data = []
            for idx, row in df.iterrows():
                record = {
                    'timestamp': idx if not isinstance(idx, (int, float)) else pd.to_datetime(idx, unit='ms'),
                    'timeframe': timeframe  # Додаємо часовий проміжок як поле
                }
                for col, val in row.items():
                    # Перетворюємо numpy типи
                    if isinstance(val, np.generic):
                        record[col] = val.item()
                    else:
                        record[col] = val
                processed_data.append(record)

            # Збереження через менеджер БД
            self.db_manager.save_sol_arima_data(processed_data, timeframe)
            self.logger.info(f"ARIMA дані для SOL ({timeframe}) збережено в таблицю {table_name}")
            return True
        except Exception as e:
            self.logger.error(f"Помилка при збереженні ARIMA даних для SOL ({timeframe}): {str(e)}")
            return False

    # Аналогічним чином оновлюємо методи завантаження даних
    def load_lstm_sequence(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Завантажує послідовності LSTM для конкретної криптовалюти та часового проміжку"""
        symbol = symbol.upper()

        if symbol not in self.supported_cryptos:
            self.logger.error(f"Непідтримувана криптовалюта: {symbol}")
            return pd.DataFrame()

        if timeframe not in self.supported_timeframes:
            self.logger.error(f"Непідтримуваний часовий проміжок: {timeframe}")
            return pd.DataFrame()

        try:
            # Вибір відповідного методу завантаження
            if symbol == 'BTC':
                return self.load_btc_lstm_sequence(timeframe)
            elif symbol == 'ETH':
                return self.load_eth_lstm_sequence(timeframe)
            elif symbol == 'SOL':
                return self.load_sol_lstm_sequence(timeframe)
            else:
                self.logger.error(f"Немає відповідного методу для завантаження LSTM послідовностей для {symbol}")
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Помилка при виборі методу завантаження LSTM послідовностей для {symbol}: {str(e)}")
            return pd.DataFrame()

    def load_arima_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Завантажує дані ARIMA для конкретної криптовалюти та часового проміжку"""
        symbol = symbol.upper()

        if symbol not in self.supported_cryptos:
            self.logger.error(f"Непідтримувана криптовалюта: {symbol}")
            return pd.DataFrame()

        if timeframe not in self.supported_timeframes:
            self.logger.error(f"Непідтримуваний часовий проміжок: {timeframe}")
            return pd.DataFrame()

        try:
            # Вибір відповідного методу завантаження
            if symbol == 'BTC':
                return self.load_btc_arima_data(timeframe)
            elif symbol == 'ETH':
                return self.load_eth_arima_data(timeframe)
            elif symbol == 'SOL':
                return self.load_sol_arima_data(timeframe)
            else:
                self.logger.error(f"Немає відповідного методу для завантаження ARIMA даних для {symbol}")
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Помилка при виборі методу завантаження ARIMA даних для {symbol}: {str(e)}")
            return pd.DataFrame()

    def load_btc_lstm_sequence(self, timeframe: str) -> pd.DataFrame:
        """Завантажує послідовності LSTM для Bitcoin для конкретного часового проміжку"""
        table_name = f"btc_lstm_sequence_{timeframe}"

        try:
            self.logger.info(f"Завантаження LSTM послідовностей для BTC ({timeframe}) з таблиці {table_name}")
            data = self.db_manager.get_btc_lstm_sequence(timeframe)

            if data is None:
                self.logger.warning(f"Не знайдено LSTM послідовності для BTC ({timeframe})")
                return pd.DataFrame()

            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)

            # Встановлюємо часовий індекс
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data.set_index('timestamp', inplace=True)

            return data
        except Exception as e:
            self.logger.error(f"Помилка при завантаженні LSTM послідовностей для BTC ({timeframe}): {str(e)}")
            return pd.DataFrame()

    def load_eth_lstm_sequence(self, timeframe: str) -> pd.DataFrame:
        """Завантажує послідовності LSTM для Ethereum для конкретного часового проміжку"""
        table_name = f"eth_lstm_sequence_{timeframe}"

        try:
            self.logger.info(f"Завантаження LSTM послідовностей для ETH ({timeframe}) з таблиці {table_name}")
            data = self.db_manager.get_eth_lstm_sequence(timeframe)

            if data is None:
                self.logger.warning(f"Не знайдено LSTM послідовності для ETH ({timeframe})")
                return pd.DataFrame()

            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)

            # Встановлюємо часовий індекс
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data.set_index('timestamp', inplace=True)

            return data
        except Exception as e:
            self.logger.error(f"Помилка при завантаженні LSTM послідовностей для ETH ({timeframe}): {str(e)}")
            return pd.DataFrame()

    def load_sol_lstm_sequence(self, timeframe: str) -> pd.DataFrame:
        """Завантажує послідовності LSTM для Solana для конкретного часового проміжку"""
        table_name = f"sol_lstm_sequence_{timeframe}"

        try:
            self.logger.info(f"Завантаження LSTM послідовностей для SOL ({timeframe}) з таблиці {table_name}")
            data = self.db_manager.get_sol_lstm_sequence(timeframe)

            if data is None:
                self.logger.warning(f"Не знайдено LSTM послідовності для SOL ({timeframe})")
                return pd.DataFrame()

            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)

            # Встановлюємо часовий індекс
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data.set_index('timestamp', inplace=True)

            return data
        except Exception as e:
            self.logger.error(f"Помилка при завантаженні LSTM послідовностей для SOL ({timeframe}): {str(e)}")
            return pd.DataFrame()

    def load_btc_arima_data(self, timeframe: str) -> pd.DataFrame:
        """Завантажує дані ARIMA для Bitcoin для конкретного часового проміжку"""
        table_name = f"btc_arima_data_{timeframe}"

        try:
            self.logger.info(f"Завантаження ARIMA даних для BTC ({timeframe}) з таблиці {table_name}")
            data = self.db_manager.get_all_btc_arima_data(timeframe)

            if data is None:
                self.logger.warning(f"Не знайдено ARIMA дані для BTC ({timeframe})")
                return pd.DataFrame()

            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)

            # Встановлюємо часовий індекс
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data.set_index('timestamp', inplace=True)

            return data
        except Exception as e:
            self.logger.error(f"Помилка при завантаженні ARIMA даних для BTC ({timeframe}): {str(e)}")
            return pd.DataFrame()

    def load_eth_arima_data(self, timeframe: str) -> pd.DataFrame:
        """Завантажує дані ARIMA для Ethereum для конкретного часового проміжку"""
        table_name = f"eth_arima_data_{timeframe}"

        try:
            self.logger.info(f"Завантаження ARIMA даних для ETH ({timeframe}) з таблиці {table_name}")
            data = self.db_manager.get_all_eth_arima_data(timeframe)

            if data is None:
                self.logger.warning(f"Не знайдено ARIMA дані для ETH ({timeframe})")
                return pd.DataFrame()

            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)

            # Встановлюємо часовий індекс
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data.set_index('timestamp', inplace=True)

            return data
        except Exception as e:
            self.logger.error(f"Помилка при завантаженні ARIMA даних для ETH ({timeframe}): {str(e)}")
            return pd.DataFrame()

    def load_sol_arima_data(self, timeframe: str) -> pd.DataFrame:
        """Завантажує дані ARIMA для Solana для конкретного часового проміжку"""
        table_name = f"sol_arima_data_{timeframe}"

        try:
            self.logger.info(f"Завантаження ARIMA даних для SOL ({timeframe}) з таблиці {table_name}")
            data = self.db_manager.get_all_sol_arima_data(timeframe)

            if data is None:
                self.logger.warning(f"Не знайдено ARIMA дані для SOL ({timeframe})")
                return pd.DataFrame()

            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)

            # Встановлюємо часовий індекс
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data.set_index('timestamp', inplace=True)

            return data
        except Exception as e:
            self.logger.error(f"Помилка при завантаженні ARIMA даних для SOL ({timeframe}): {str(e)}")
            return pd.DataFrame()

    def load_data(self, data_source: str, symbol: str, timeframe: str,
                  start_date: Optional[Union[str, datetime]] = None,
                  end_date: Optional[Union[str, datetime]] = None,
                  data_type: str = 'candles') -> pd.DataFrame:
        """Завантажує дані з вказаного джерела"""
        start_date_dt = pd.to_datetime(start_date) if start_date else None
        end_date_dt = pd.to_datetime(end_date) if end_date else None

        self.logger.info(f"Завантаження даних з {data_source}: {symbol}, {timeframe}, {data_type}")

        try:
            if data_source != 'database':
                raise ValueError(f"Непідтримуване джерело даних: {data_source}")

            # Отримання даних з бази даних
            if data_type == 'candles':
                data = self.db_manager.get_klines(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_date_dt,
                    end_time=end_date_dt
                )

            elif data_type == 'volume_profile':
                data = self.db_manager.get_volume_profile(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_date_dt,
                    end_time=end_date_dt
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
            self.logger.error(f"Помилка при завантаженні даних: {str(e)}")
            raise