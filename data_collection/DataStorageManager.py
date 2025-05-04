from typing import Optional, Union, Dict, List
import pandas as pd
import numpy as np
from datetime import datetime


class DataStorageManager:
    def __init__(self, db_manager, logger):
        self.db_manager = db_manager
        self.logger = logger

        # Визначення списку підтримуваних криптовалют та моделей
        self.supported_cryptos = ['BTC', 'ETH', 'SOL']
        self.supported_models = ['LSTM', 'ARIMA']
    ''' 
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
                self.logger.error(f" Помилка при збереженні обробленої свічки для {symbol}: {e}")
    '''
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

    # Нові методи для збереження даних моделей для кожної криптовалюти
    def save_lstm_sequence(self, df: pd.DataFrame, symbol: str):
        """Зберігає послідовності LSTM для конкретної криптовалюти"""
        if df.empty:
            self.logger.warning(f"Спроба зберегти порожні LSTM послідовності для {symbol}")
            return False

        # Конвертуємо символ до верхнього регістру для стандартизації
        symbol = symbol.upper()

        if symbol not in self.supported_cryptos:
            self.logger.error(f"Непідтримувана криптовалюта: {symbol}")
            return False

        table_name = f"{symbol.lower()}_lstm_sequence"

        try:
            # Конвертуємо numpy типи перед збереженням
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df[col].astype(float)

            # Підготовка даних для збереження
            processed_data = []
            for idx, row in df.iterrows():
                record = {'timestamp': idx if not isinstance(idx, (int, float)) else pd.to_datetime(idx, unit='ms')}
                for col, val in row.items():
                    # Перетворюємо numpy типи
                    if isinstance(val, np.generic):
                        record[col] = val.item()
                    else:
                        record[col] = val
                processed_data.append(record)

            # Збереження через менеджер БД
            self.db_manager.insert_model_data(table_name, processed_data)
            self.logger.info(f"LSTM послідовності для {symbol} збережено в таблицю {table_name}")
            return True
        except Exception as e:
            self.logger.error(f"Помилка при збереженні LSTM послідовностей для {symbol}: {str(e)}")
            return False

    def save_arima_data(self, df: pd.DataFrame, symbol: str):
        """Зберігає дані ARIMA для конкретної криптовалюти"""
        if df.empty:
            self.logger.warning(f"Спроба зберегти порожні ARIMA дані для {symbol}")
            return False

        # Конвертуємо символ до верхнього регістру для стандартизації
        symbol = symbol.upper()

        if symbol not in self.supported_cryptos:
            self.logger.error(f"Непідтримувана криптовалюта: {symbol}")
            return False

        table_name = f"{symbol.lower()}_arima_data"

        try:
            # Конвертуємо numpy типи перед збереженням
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df[col].astype(float)

            # Підготовка даних для збереження
            processed_data = []
            for idx, row in df.iterrows():
                record = {'timestamp': idx if not isinstance(idx, (int, float)) else pd.to_datetime(idx, unit='ms')}
                for col, val in row.items():
                    # Перетворюємо numpy типи
                    if isinstance(val, np.generic):
                        record[col] = val.item()
                    else:
                        record[col] = val
                processed_data.append(record)

            # Збереження через менеджер БД
            self.db_manager.insert_model_data(table_name, processed_data)
            self.logger.info(f"ARIMA дані для {symbol} збережено в таблицю {table_name}")
            return True
        except Exception as e:
            self.logger.error(f"Помилка при збереженні ARIMA даних для {symbol}: {str(e)}")
            return False

    def save_btc_lstm_sequence(self, df: pd.DataFrame):
        """Зберігає послідовності LSTM для Bitcoin"""
        return self.save_lstm_sequence(df, 'BTC')

    def save_eth_lstm_sequence(self, df: pd.DataFrame):
        """Зберігає послідовності LSTM для Ethereum"""
        return self.save_lstm_sequence(df, 'ETH')

    def save_sol_lstm_sequence(self, df: pd.DataFrame):
        """Зберігає послідовності LSTM для Solana"""
        return self.save_lstm_sequence(df, 'SOL')

    def save_btc_arima_data(self, df: pd.DataFrame):
        """Зберігає дані ARIMA для Bitcoin"""
        return self.save_arima_data(df, 'BTC')

    def save_eth_arima_data(self, df: pd.DataFrame):
        """Зберігає дані ARIMA для Ethereum"""
        return self.save_arima_data(df, 'ETH')

    def save_sol_arima_data(self, df: pd.DataFrame):
        """Зберігає дані ARIMA для Solana"""
        return self.save_arima_data(df, 'SOL')

    # Аналогічні методи для завантаження даних моделей
    def load_lstm_sequence(self, symbol: str) -> pd.DataFrame:
        """Завантажує послідовності LSTM для конкретної криптовалюти"""
        symbol = symbol.upper()

        if symbol not in self.supported_cryptos:
            self.logger.error(f"Непідтримувана криптовалюта: {symbol}")
            return pd.DataFrame()

        table_name = f"{symbol.lower()}_lstm_sequence"

        try:
            self.logger.info(f"Завантаження LSTM послідовностей для {symbol} з таблиці {table_name}")
            data = self.db_manager.get_model_data(table_name)

            if data is None:
                self.logger.warning(f"Не знайдено LSTM послідовності для {symbol}")
                return pd.DataFrame()

            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)

            # Встановлюємо часовий індекс
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data.set_index('timestamp', inplace=True)

            return data
        except Exception as e:
            self.logger.error(f"Помилка при завантаженні LSTM послідовностей для {symbol}: {str(e)}")
            return pd.DataFrame()

    def load_arima_data(self, symbol: str) -> pd.DataFrame:
        """Завантажує дані ARIMA для конкретної криптовалюти"""
        symbol = symbol.upper()

        if symbol not in self.supported_cryptos:
            self.logger.error(f"Непідтримувана криптовалюта: {symbol}")
            return pd.DataFrame()

        table_name = f"{symbol.lower()}_arima_data"

        try:
            self.logger.info(f"Завантаження ARIMA даних для {symbol} з таблиці {table_name}")
            data = self.db_manager.get_model_data(table_name)

            if data is None:
                self.logger.warning(f"Не знайдено ARIMA дані для {symbol}")
                return pd.DataFrame()

            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)

            # Встановлюємо часовий індекс
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data.set_index('timestamp', inplace=True)

            return data
        except Exception as e:
            self.logger.error(f"Помилка при завантаженні ARIMA даних для {symbol}: {str(e)}")
            return pd.DataFrame()

    # Спеціалізовані методи для завантаження даних для кожної криптовалюти і моделі
    def load_btc_lstm_sequence(self) -> pd.DataFrame:
        """Завантажує послідовності LSTM для Bitcoin"""
        return self.load_lstm_sequence('BTC')

    def load_eth_lstm_sequence(self) -> pd.DataFrame:
        """Завантажує послідовності LSTM для Ethereum"""
        return self.load_lstm_sequence('ETH')

    def load_sol_lstm_sequence(self) -> pd.DataFrame:
        """Завантажує послідовності LSTM для Solana"""
        return self.load_lstm_sequence('SOL')

    def load_btc_arima_data(self) -> pd.DataFrame:
        """Завантажує дані ARIMA для Bitcoin"""
        return self.load_arima_data('BTC')

    def load_eth_arima_data(self) -> pd.DataFrame:
        """Завантажує дані ARIMA для Ethereum"""
        return self.load_arima_data('ETH')

    def load_sol_arima_data(self) -> pd.DataFrame:
        """Завантажує дані ARIMA для Solana"""
        return self.load_arima_data('SOL')

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
            elif data_type == 'processed_candles':
                data = self.db_manager.get_processed_klines(
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