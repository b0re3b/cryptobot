from typing import Optional, Union, Dict, List, Any
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
                        'interval': str(timeframe),
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

    # Методи для BTC + LSTM
    def save_btc_lstm_sequence(self, data_points: List[Dict[str, Any]]) -> List[int]:
        """Зберігає послідовності LSTM для BTC"""
        try:
            self.logger.info(f"Збереження LSTM послідовностей для BTC")

            # Конвертуємо numpy типи перед збереженням
            processed_data = []
            for point in data_points:
                processed_point = {}
                for key, val in point.items():
                    if isinstance(val, np.generic):
                        processed_point[key] = val.item()
                    else:
                        processed_point[key] = val
                processed_data.append(processed_point)

            # Виклик відповідного методу з db_manager
            sequence_ids = self.db_manager.save_btc_lstm_sequence(processed_data)
            self.logger.info(f"LSTM послідовності для BTC успішно збережено, IDs: {sequence_ids}")
            return sequence_ids
        except Exception as e:
            self.logger.error(f"Помилка при збереженні LSTM послідовностей для BTC: {str(e)}")
            return []

    def get_btc_lstm_sequence(self, timeframe: str, sequence_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Завантажує послідовності LSTM для BTC"""
        try:
            self.logger.info(
                f"Завантаження LSTM послідовностей для BTC, timeframe: {timeframe}, sequence_id: {sequence_id}")

            if timeframe not in self.supported_timeframes:
                self.logger.error(f"Непідтримуваний часовий проміжок: {timeframe}")
                return []

            # Виклик відповідного методу з db_manager
            data = self.db_manager.get_btc_lstm_sequence(timeframe, sequence_id)

            if not data:
                self.logger.warning(
                    f"Не знайдено LSTM послідовності для BTC (timeframe: {timeframe}, sequence_id: {sequence_id})")
                return []

            return data
        except Exception as e:
            self.logger.error(f"Помилка при завантаженні LSTM послідовностей для BTC: {str(e)}")
            return []

    # Методи для ETH + LSTM
    def save_eth_lstm_sequence(self, data_points: List[Dict[str, Any]]) -> List[int]:
        """Зберігає послідовності LSTM для ETH"""
        try:
            self.logger.info(f"Збереження LSTM послідовностей для ETH")

            # Конвертуємо numpy типи перед збереженням
            processed_data = []
            for point in data_points:
                processed_point = {}
                for key, val in point.items():
                    if isinstance(val, np.generic):
                        processed_point[key] = val.item()
                    else:
                        processed_point[key] = val
                processed_data.append(processed_point)

            # Виклик відповідного методу з db_manager
            sequence_ids = self.db_manager.save_eth_lstm_sequence(processed_data)
            self.logger.info(f"LSTM послідовності для ETH успішно збережено, IDs: {sequence_ids}")
            return sequence_ids
        except Exception as e:
            self.logger.error(f"Помилка при збереженні LSTM послідовностей для ETH: {str(e)}")
            return []

    def get_eth_lstm_sequence(self, timeframe: str, sequence_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Завантажує послідовності LSTM для ETH"""
        try:
            self.logger.info(
                f"Завантаження LSTM послідовностей для ETH, timeframe: {timeframe}, sequence_id: {sequence_id}")

            if timeframe not in self.supported_timeframes:
                self.logger.error(f"Непідтримуваний часовий проміжок: {timeframe}")
                return []

            # Виклик відповідного методу з db_manager
            data = self.db_manager.get_eth_lstm_sequence(timeframe, sequence_id)

            if not data:
                self.logger.warning(
                    f"Не знайдено LSTM послідовності для ETH (timeframe: {timeframe}, sequence_id: {sequence_id})")
                return []

            return data
        except Exception as e:
            self.logger.error(f"Помилка при завантаженні LSTM послідовностей для ETH: {str(e)}")
            return []

    # Методи для SOL + LSTM
    def save_sol_lstm_sequence(self, data_points: List[Dict[str, Any]]) -> List[int]:
        """Зберігає послідовності LSTM для SOL"""
        try:
            self.logger.info(f"Збереження LSTM послідовностей для SOL")

            # Конвертуємо numpy типи перед збереженням
            processed_data = []
            for point in data_points:
                processed_point = {}
                for key, val in point.items():
                    if isinstance(val, np.generic):
                        processed_point[key] = val.item()
                    else:
                        processed_point[key] = val
                processed_data.append(processed_point)

            # Виклик відповідного методу з db_manager
            sequence_ids = self.db_manager.save_sol_lstm_sequence(processed_data)
            self.logger.info(f"LSTM послідовності для SOL успішно збережено, IDs: {sequence_ids}")
            return sequence_ids
        except Exception as e:
            self.logger.error(f"Помилка при збереженні LSTM послідовностей для SOL: {str(e)}")
            return []

    def get_sol_lstm_sequence(self, timeframe: str, sequence_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Завантажує послідовності LSTM для SOL"""
        try:
            self.logger.info(
                f"Завантаження LSTM послідовностей для SOL, timeframe: {timeframe}, sequence_id: {sequence_id}")

            if timeframe not in self.supported_timeframes:
                self.logger.error(f"Непідтримуваний часовий проміжок: {timeframe}")
                return []

            # Виклик відповідного методу з db_manager
            data = self.db_manager.get_sol_lstm_sequence(timeframe, sequence_id)

            if not data:
                self.logger.warning(
                    f"Не знайдено LSTM послідовності для SOL (timeframe: {timeframe}, sequence_id: {sequence_id})")
                return []

            return data
        except Exception as e:
            self.logger.error(f"Помилка при завантаженні LSTM послідовностей для SOL: {str(e)}")
            return []

    # Методи для BTC + ARIMA
    def save_btc_arima_data(self, data_points: List[Dict[str, Any]]) -> List[int]:
        """Зберігає дані ARIMA для BTC"""
        try:
            self.logger.info(f"Збереження ARIMA даних для BTC")

            # Конвертуємо numpy типи перед збереженням
            processed_data = []
            for point in data_points:
                processed_point = {}
                for key, val in point.items():
                    if isinstance(val, np.generic):
                        processed_point[key] = val.item()
                    else:
                        processed_point[key] = val
                processed_data.append(processed_point)

            # Виклик відповідного методу з db_manager
            data_ids = self.db_manager.save_btc_arima_data(processed_data)
            self.logger.info(f"ARIMA дані для BTC успішно збережено, IDs: {data_ids}")
            return data_ids
        except Exception as e:
            self.logger.error(f"Помилка при збереженні ARIMA даних для BTC: {str(e)}")
            return []

    def get_btc_arima_data(self, timeframe: str, data_id: Optional[int] = None) -> list[Any] | dict[Any, Any]:
        """Завантажує дані ARIMA для BTC"""
        try:
            self.logger.info(f"Завантаження ARIMA даних для BTC, timeframe: {timeframe}, data_id: {data_id}")

            if timeframe not in self.supported_timeframes:
                self.logger.error(f"Непідтримуваний часовий проміжок: {timeframe}")
                return []

            # Виклик відповідного методу з db_manager
            data = self.db_manager.get_btc_arima_data(timeframe, data_id)

            if not data:
                self.logger.warning(f"Не знайдено ARIMA дані для BTC (timeframe: {timeframe}, data_id: {data_id})")
                return []

            return data
        except Exception as e:
            self.logger.error(f"Помилка при завантаженні ARIMA даних для BTC: {str(e)}")
            return []

    # Методи для ETH + ARIMA
    def save_eth_arima_data(self, data_points: List[Dict[str, Any]]) -> List[int]:
        """Зберігає дані ARIMA для ETH"""
        try:
            self.logger.info(f"Збереження ARIMA даних для ETH")

            # Конвертуємо numpy типи перед збереженням
            processed_data = []
            for point in data_points:
                processed_point = {}
                for key, val in point.items():
                    if isinstance(val, np.generic):
                        processed_point[key] = val.item()
                    else:
                        processed_point[key] = val
                processed_data.append(processed_point)

            # Виклик відповідного методу з db_manager
            data_ids = self.db_manager.save_eth_arima_data(processed_data)
            self.logger.info(f"ARIMA дані для ETH успішно збережено, IDs: {data_ids}")
            return data_ids
        except Exception as e:
            self.logger.error(f"Помилка при збереженні ARIMA даних для ETH: {str(e)}")
            return []

    def get_eth_arima_data(self, timeframe: str, data_id: Optional[int] = None) -> list[Any] | dict[Any, Any]:
        """Завантажує дані ARIMA для ETH"""
        try:
            self.logger.info(f"Завантаження ARIMA даних для ETH, timeframe: {timeframe}, data_id: {data_id}")

            if timeframe not in self.supported_timeframes:
                self.logger.error(f"Непідтримуваний часовий проміжок: {timeframe}")
                return []

            # Виклик відповідного методу з db_manager
            data = self.db_manager.get_eth_arima_data(timeframe, data_id)

            if not data:
                self.logger.warning(f"Не знайдено ARIMA дані для ETH (timeframe: {timeframe}, data_id: {data_id})")
                return []

            return data
        except Exception as e:
            self.logger.error(f"Помилка при завантаженні ARIMA даних для ETH: {str(e)}")
            return []

    # Методи для SOL + ARIMA
    def save_sol_arima_data(self, data_points: List[Dict[str, Any]]) -> List[int]:
        """Зберігає дані ARIMA для SOL"""
        try:
            self.logger.info(f"Збереження ARIMA даних для SOL")

            # Конвертуємо numpy типи перед збереженням
            processed_data = []
            for point in data_points:
                processed_point = {}
                for key, val in point.items():
                    if isinstance(val, np.generic):
                        processed_point[key] = val.item()
                    else:
                        processed_point[key] = val
                processed_data.append(processed_point)

            # Виклик відповідного методу з db_manager
            data_ids = self.db_manager.save_sol_arima_data(processed_data)
            self.logger.info(f"ARIMA дані для SOL успішно збережено, IDs: {data_ids}")
            return data_ids
        except Exception as e:
            self.logger.error(f"Помилка при збереженні ARIMA даних для SOL: {str(e)}")
            return []

    def get_sol_arima_data(self, timeframe: str, data_id: Optional[int] = None) -> list[Any] | dict[Any, Any]:
        """Завантажує дані ARIMA для SOL"""
        try:
            self.logger.info(f"Завантаження ARIMA даних для SOL, timeframe: {timeframe}, data_id: {data_id}")

            if timeframe not in self.supported_timeframes:
                self.logger.error(f"Непідтримуваний часовий проміжок: {timeframe}")
                return []

            # Виклик відповідного методу з db_manager
            data = self.db_manager.get_sol_arima_data(timeframe, data_id)

            if not data:
                self.logger.warning(f"Не знайдено ARIMA дані для SOL (timeframe: {timeframe}, data_id: {data_id})")
                return []

            return data
        except Exception as e:
            self.logger.error(f"Помилка при завантаженні ARIMA даних для SOL: {str(e)}")
            return []

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