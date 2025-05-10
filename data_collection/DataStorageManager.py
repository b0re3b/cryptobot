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

    def save_model_data(self, df: pd.DataFrame, symbol: str, timeframe: str, model_type: str):
        """Універсальний метод для збереження даних моделей (LSTM або ARIMA) для будь-якої криптовалюти"""
        if df.empty:
            self.logger.warning(f"Спроба зберегти порожні дані {model_type} для {symbol} ({timeframe})")
            return False

        # Конвертуємо символ до верхнього регістру для стандартизації
        symbol = symbol.upper()
        model_type = model_type.upper()

        # Перевірка підтримуваних криптовалют та часових проміжків
        if symbol not in self.supported_cryptos:
            self.logger.error(f"Непідтримувана криптовалюта: {symbol}")
            return False

        if timeframe not in self.supported_timeframes:
            self.logger.error(f"Непідтримуваний часовий проміжок: {timeframe}")
            return False

        if model_type not in self.supported_models:
            self.logger.error(f"Непідтримуваний тип моделі: {model_type}")
            return False

        try:
            # Конвертуємо numpy типи перед збереженням
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df[col].astype(float)

            # Підготовка даних для збереження
            processed_data = []
            for idx, row in df.iterrows():
                record = {
                    'timestamp': idx if not isinstance(idx, (int, float)) else pd.to_datetime(idx, unit='ms'),
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'model_type': model_type
                }
                for col, val in row.items():
                    # Перетворюємо numpy типи
                    if isinstance(val, np.generic):
                        record[col] = val.item()
                    else:
                        record[col] = val
                processed_data.append(record)

            # Збереження через менеджер БД
            if model_type == 'LSTM':
                self.db_manager.save_lstm_data(processed_data)
                self.logger.info(f"LSTM дані для {symbol} ({timeframe}) збережено в єдину таблицю")
            elif model_type == 'ARIMA':
                self.db_manager.save_arima_data(processed_data)
                self.logger.info(f"ARIMA дані для {symbol} ({timeframe}) збережено в єдину таблицю")
            else:
                self.logger.error(f"Невідомий тип моделі: {model_type}")
                return False

            return True
        except Exception as e:
            self.logger.error(f"Помилка при збереженні даних {model_type} для {symbol} ({timeframe}): {str(e)}")
            return False

    def save_lstm_sequence(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Зберігає послідовності LSTM для конкретної криптовалюти та часового проміжку"""
        return self.save_model_data(df, symbol, timeframe, 'LSTM')

    def save_arima_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Зберігає дані ARIMA для конкретної криптовалюти та часового проміжку"""
        return self.save_model_data(df, symbol, timeframe, 'ARIMA')

    def load_model_data(self, symbol: str, timeframe: str, model_type: str) -> pd.DataFrame:
        """Універсальний метод для завантаження даних моделей (LSTM або ARIMA) для будь-якої криптовалюти"""
        symbol = symbol.upper()
        model_type = model_type.upper()

        if symbol not in self.supported_cryptos:
            self.logger.error(f"Непідтримувана криптовалюта: {symbol}")
            return pd.DataFrame()

        if timeframe not in self.supported_timeframes:
            self.logger.error(f"Непідтримуваний часовий проміжок: {timeframe}")
            return pd.DataFrame()

        if model_type not in self.supported_models:
            self.logger.error(f"Непідтримуваний тип моделі: {model_type}")
            return pd.DataFrame()

        try:
            self.logger.info(f"Завантаження {model_type} даних для {symbol} ({timeframe}) з єдиної таблиці")

            if model_type == 'LSTM':
                data = self.db_manager.get_lstm_data(symbol, timeframe)
            elif model_type == 'ARIMA':
                data = self.db_manager.get_arima_data(symbol, timeframe)
            else:
                self.logger.error(f"Невідомий тип моделі: {model_type}")
                return pd.DataFrame()

            if data is None:
                self.logger.warning(f"Не знайдено {model_type} дані для {symbol} ({timeframe})")
                return pd.DataFrame()

            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)

            # Видаляємо додаткові системні колонки, якщо вони є
            for col in ['symbol', 'timeframe', 'model_type']:
                if col in data.columns:
                    data = data.drop(columns=[col])

            # Встановлюємо часовий індекс
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data.set_index('timestamp', inplace=True)

            return data
        except Exception as e:
            self.logger.error(f"Помилка при завантаженні {model_type} даних для {symbol} ({timeframe}): {str(e)}")
            return pd.DataFrame()

    def load_lstm_sequence(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Завантажує послідовності LSTM для конкретної криптовалюти та часового проміжку"""
        return self.load_model_data(symbol, timeframe, 'LSTM')

    def load_arima_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Завантажує дані ARIMA для конкретної криптовалюти та часового проміжку"""
        return self.load_model_data(symbol, timeframe, 'ARIMA')

    def load_data(self, data_source: str, symbol: str, timeframe: str,
                  start_date: Optional[Union[str, datetime]] = None,
                  end_date: Optional[Union[str, datetime]] = None,
                  data_type: str = 'candles') -> pd.DataFrame:

        self.logger.info(f"Завантаження даних з {data_source}: {symbol}, {timeframe}, {data_type}")

        try:
            if data_source != 'database':
                raise ValueError(f"Непідтримуване джерело даних: {data_source}")

            # Отримання даних з бази даних
            if data_type == 'candles':
                data = self.db_manager.get_klines(
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
            self.logger.error(f"Помилка при завантаженні даних: {str(e)}")
            raise