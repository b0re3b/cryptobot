import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Tuple, Union, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import hashlib
import json
from functools import lru_cache
import pytz
from utils.config import db_connection
import data.db as db

class MarketDataProcessor:

    def __init__(self, cache_dir=None, log_level=logging.INFO):
        self.cache_dir = cache_dir
        self.log_level = log_level
        self.db_connection = db_connection

        logging.basicConfig(level=self.log_level)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Ініціалізація класу...")

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.logger.info(f"Директорію для кешу створено: {self.cache_dir}")

        self.cache_index = {}
        self._load_cache_index()
        self.ready = True

    def _get_cache_path(self, cache_key: str) -> str:
        return os.path.join(self.cache_dir, f"{cache_key}.parquet")

    def save_to_cache(self, cache_key: str, data: pd.DataFrame, metadata: Dict = None) -> bool:
        cache_path = self._get_cache_path(cache_key)
        try:
            data.to_parquet(cache_path)

            self.cache_index[cache_key] = {
                "created_at": datetime.now().isoformat(),
                "rows": len(data),
                "columns": list(data.columns),
                **(metadata or {})
            }

            self._save_cache_index()
            return True
        except Exception as e:
            print(f"Помилка збереження в кеш: {e}")
            return False

    def _load_from_database(self, symbol: str, interval: str,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            data_type: str = 'candles') -> pd.DataFrame:

        self.logger.info(f"Завантаження {data_type} даних з бази даних для {symbol} {interval}")

        try:
            if data_type == 'candles':
                data = db.get_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=start_date,
                    end_time=end_date
                )
            elif data_type == 'orderbook':
                data = db.get_orderbook(
                    symbol=symbol,
                    start_time=start_date,
                    end_time=end_date
                )
            else:
                raise ValueError(f"Непідтримуваний тип даних: {data_type}")

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

        cache_key = self.create_cache_key(
            data_source, symbol, interval, start_date, end_date, data_type
        )

        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.parquet")
            if os.path.exists(cache_file):
                self.logger.info(f"Завантаження даних з кешу: {cache_key}")
                return pd.read_parquet(cache_file)

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
                    time_col = next(col for col in ['timestamp', 'date', 'time'] if col in data.columns)
                    data[time_col] = pd.to_datetime(data[time_col])
                    data.set_index(time_col, inplace=True)

                if start_date_dt:
                    data = data[data.index >= start_date_dt]
                if end_date_dt:
                    data = data[data.index <= end_date_dt]

            else:
                raise ValueError(f"Непідтримуване джерело даних: {data_source}")

            if data is None or data.empty:
                self.logger.warning(f"Отримано порожній набір даних від {data_source}")
                return pd.DataFrame()

            if self.cache_dir:
                self.save_to_cache(cache_key, data, metadata={
                    'source': data_source,
                    'symbol': symbol,
                    'interval': interval,
                    'data_type': data_type,
                    'start_date': start_date_dt.isoformat() if start_date_dt else None,
                    'end_date': end_date_dt.isoformat() if end_date_dt else None,
                    'file_path': file_path if data_source == 'csv' else None
                })

            return data

        except Exception as e:
            self.logger.error(f"Помилка при завантаженні даних: {str(e)}")
            raise

    def clean_data(self, data: pd.DataFrame, remove_outliers: bool = True,
                   fill_missing: bool = True) -> pd.DataFrame:

        if data.empty:
            self.logger.warning("Отримано порожній DataFrame для очищення")
            return data

        self.logger.info(f"Початок очищення даних: {data.shape[0]} рядків, {data.shape[1]} стовпців")
        result = data.copy()

        if not isinstance(result.index, pd.DatetimeIndex):
            try:
                time_cols = [col for col in result.columns if
                             any(x in col.lower() for x in ['time', 'date', 'timestamp'])]
                if time_cols:
                    time_col = time_cols[0]
                    self.logger.info(f"Конвертування колонки {time_col} в індекс часу")
                    result[time_col] = pd.to_datetime(result[time_col])
                    result.set_index(time_col, inplace=True)
                else:
                    self.logger.warning("Не знайдено колонку з часом, індекс залишається незмінним")
            except Exception as e:
                self.logger.error(f"Помилка при конвертуванні індексу: {str(e)}")

        if result.index.duplicated().any():
            dup_count = result.index.duplicated().sum()
            self.logger.info(f"Знайдено {dup_count} дублікатів індексу, видалення...")
            result = result[~result.index.duplicated(keep='first')]

        result = result.sort_index()

        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors='coerce')

        if remove_outliers:
            self.logger.info("Видалення аномальних значень...")
            price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in result.columns]
            for col in price_cols:
                Q1 = result[col].quantile(0.25)
                Q3 = result[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR

                outliers = (result[col] < lower_bound) | (result[col] > upper_bound)
                if outliers.any():
                    outlier_count = outliers.sum()
                    self.logger.info(f"Знайдено {outlier_count} аномалій в колонці {col}")
                    result.loc[outliers, col] = np.nan

        if fill_missing and result.isna().any().any():
            self.logger.info("Заповнення відсутніх значень...")
            price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in result.columns]
            if price_cols:
                result[price_cols] = result[price_cols].interpolate(method='time')

            if 'volume' in result.columns and result['volume'].isna().any():
                result['volume'] = result['volume'].fillna(0)

            numeric_cols = result.select_dtypes(include=[np.number]).columns
            other_numeric = [col for col in numeric_cols if col not in price_cols + ['volume']]
            if other_numeric:
                result[other_numeric] = result[other_numeric].interpolate(method='time')

            result = result.fillna(method='ffill').fillna(method='bfill')

        price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in result.columns]
        if len(price_cols) == 4:

            invalid_hl = result['high'] < result['low']
            if invalid_hl.any():
                invalid_count = invalid_hl.sum()
                self.logger.warning(f"Знайдено {invalid_count} рядків, де high < low")

                temp = result.loc[invalid_hl, 'high'].copy()
                result.loc[invalid_hl, 'high'] = result.loc[invalid_hl, 'low']
                result.loc[invalid_hl, 'low'] = temp

        self.logger.info(f"Очищення даних завершено: {result.shape[0]} рядків, {result.shape[1]} стовпців")
        return result

    def resample_data(self, data: pd.DataFrame, target_interval: str) -> pd.DataFrame:
        """
        Перетворює дані з одного часового інтервалу в інший.

        Parameters:
        -----------
        data : pandas.DataFrame
            Вхідний DataFrame
        target_interval : str
            Цільовий інтервал (наприклад, '5m', '1h')

        Returns:
        --------
        pandas.DataFrame
            DataFrame з даними нового інтервалу
        """
        # Перевірити, що дані мають правильну структуру
        # Конвертувати target_interval у відповідний формат для pandas
        # Правильно агрегувати OHLCV дані при ресемплінгу
        # Обробка випадку, коли цільовий інтервал менший за вхідний
        pass

    def detect_outliers(self, data: pd.DataFrame, method: str = 'zscore',
                        threshold: float = 3) -> Tuple[pd.DataFrame, List]:
        """
        Виявляє аномальні значення в даних.

        Parameters:
        -----------
        data : pandas.DataFrame
            Вхідний DataFrame
        method : str
            Метод виявлення аномалій ('zscore', 'iqr', 'isolation_forest')
        threshold : float
            Поріг для визначення аномалії

        Returns:
        --------
        tuple
            (DataFrame з мітками аномалій, список індексів аномальних точок)
        """
        # Вибрати числові стовпці для аналізу
        # Застосувати вибраний метод для виявлення аномалій
        # Створити мітки для кожного стовпця або для всього ряду
        # Повернути мітки і список індексів аномалій
        pass

    def handle_missing_values(self, data: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
        """
        Спеціалізована обробка відсутніх значень.

        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame з відсутніми даними
        method : str
            Метод заповнення ('interpolate', 'ffill', 'mean', 'median')

        Returns:
        --------
        pandas.DataFrame
            DataFrame з обробленими відсутніми значеннями
        """
        # Перевірити наявність відсутніх значень
        # Застосувати вибраний метод заповнення
        # Обробляти кожен стовпець окремо, якщо потрібно
        # Логувати кількість заповнених значень
        pass

    def normalize_data(self, data: pd.DataFrame, method: str = 'z-score') -> Tuple[pd.DataFrame, object]:
        """
        Нормалізує дані для покращення роботи моделей машинного навчання.

        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame для нормалізації
        method : str
            Метод нормалізації ('z-score', 'min-max', 'robust')

        Returns:
        --------
        tuple
            (нормалізований DataFrame, об'єкт скейлера)
        """
        # Вибрати числові стовпці для нормалізації
        # Створити і застосувати відповідний скейлер
        # Повернути трансформовані дані і скейлер для подальшого використання
        pass

    def align_time_series(self, data_list: List[pd.DataFrame],
                          reference_index: int = 0) -> List[pd.DataFrame]:
        """
        Вирівнює кілька часових рядів по спільній часовій шкалі.

        Parameters:
        -----------
        data_list : list of pandas.DataFrame
            Список DataFrames для вирівнювання
        reference_index : int
            Індекс DataFrame, по якому будуть вирівнюватися інші

        Returns:
        --------
        list of pandas.DataFrame
            Список вирівняних DataFrames
        """
        # Перевірити, що всі DataFrames мають часовий індекс
        # Використати reference_index як основу для вирівнювання
        # Обробити випадки з різними частотами даних
        # Повернути список вирівняних DataFrames
        pass

    def validate_data_integrity(self, data: pd.DataFrame) -> Dict[str, List]:
        """
        Перевіряє цілісність даних на наявність невідповідностей.

        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame для перевірки

        Returns:
        --------
        dict
            Словник з проблемами та їх розташуванням
        """
        # Перевірка базових умов для OHLCV даних (high >= low, тощо)
        # Перевірка послідовності часових міток
        # Виявлення різких змін у ціні або об'ємі
        # Повернення інформації про проблеми
        pass

    def aggregate_volume_profile(self, data: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
        """
        Створює профіль об'єму по цінових рівнях.

        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame з даними OHLCV
        bins : int
            Кількість цінових рівнів

        Returns:
        --------
        pandas.DataFrame
            DataFrame з профілем об'єму
        """
        # Визначити ціновий діапазон і розділити на bins
        # Агрегувати об'єм для кожного цінового рівня
        # Створити додаткову інформацію (% від загального об'єму, тощо)
        # Повернути сформований профіль об'єму
        pass

    def add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Додає часові ознаки на основі індексу дати/часу.

        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame з часовим індексом

        Returns:
        --------
        pandas.DataFrame
            DataFrame з доданими часовими ознаками
        """
        # Перевірити, що індекс є DatetimeIndex
        # Додати категоріальні ознаки (день тижня, година, місяць)
        # Додати циклічні ознаки через sin/cos трансформації
        # Додати індикатори торгових сесій, якщо потрібно
        pass

    def remove_duplicate_timestamps(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Спеціальна обробка дублікатів часових міток.

        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame для обробки

        Returns:
        --------
        pandas.DataFrame
            DataFrame без дублікатів
        """
        # Виявити дублікати в індексі
        # Вибрати стратегію обробки (збереження першого, останнього, агрегація)
        # Логування інформації про видалені дублікати
        # Повернення очищеного DataFrame
        pass

    def filter_by_time_range(self, data: pd.DataFrame,
                             start_time: Optional[Union[str, datetime]] = None,
                             end_time: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Фільтрує дані за часовими межами.

        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame для фільтрації
        start_time : str or datetime, optional
            Початковий час для фільтрації
        end_time : str or datetime, optional
            Кінцевий час для фільтрації

        Returns:
        --------
        pandas.DataFrame
            Відфільтрований DataFrame
        """
        # Конвертувати вхідні часи в datetime, якщо вони рядки
        # Перевірити валідність діапазону (start_time < end_time)
        # Фільтрувати дані за індексом
        # Повернути відфільтровані дані
        pass

    def save_processed_data(self, data: pd.DataFrame, filename: str) -> str:
        """
        Зберігає оброблені дані у файл.

        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame для збереження
        filename : str
            Ім'я файлу або шлях для збереження

        Returns:
        --------
        str
            Повний шлях до збереженого файлу
        """
        # Визначити формат на основі розширення або параметра
        # Створити директорію, якщо вона не існує
        # Зберегти дані у вибраному форматі (CSV, HDF5, Parquet)
        # Повернути шлях до збереженого файлу
        pass

    def load_processed_data(self, filename: str) -> pd.DataFrame:
        """
        Завантажує раніше збережені оброблені дані.

        Parameters:
        -----------
        filename : str
            Ім'я файлу або шлях для завантаження

        Returns:
        --------
        pandas.DataFrame
            Завантажений DataFrame
        """
        # Визначити формат на основі розширення
        # Завантажити дані з файлу
        # Переконатися, що індекс є DatetimeIndex
        # Повернути завантажені дані
        pass

    def create_cache_key(self, source: str, symbol: str, interval: str,
                         start_date: Union[str, datetime, None],
                         end_date: Union[str, datetime, None]) -> str:

        params={
            'symbol': symbol,
            'interval': interval,
            'start_date': start_date,
            'end_date': end_date
        }
        cache_dict ={
            'source': source,
            **params
        }
        for key, value in cache_dict.items():
            if isinstance(value, datetime):
                cache_dict[key] = value.strftime('%Y-%m-%d %H:%M:%S')
            elif value is None:
                cache_dict[key] = None

        json_string = json.dumps(cache_dict, sort_keys=True)
        return hashlib.md5(json_string.encode()).hexdigest()

    def merge_datasets(self, datasets: List[pd.DataFrame],
                       merge_on: str = 'timestamp') -> pd.DataFrame:
        """
        Об'єднує кілька наборів даних в один.

        Parameters:
        -----------
        datasets : list of pandas.DataFrame
            Список DataFrames для об'єднання
        merge_on : str
            Стовпець або індекс для об'єднання

        Returns:
        --------
        pandas.DataFrame
            Об'єднаний DataFrame
        """
        # Перевірити, що всі datasets містять merge_on
        # Виконати послідовне об'єднання
        # Обробити випадки з різними назвами стовпців
        # Повернути об'єднаний DataFrame
        pass

    def preprocess_pipeline(self, data: pd.DataFrame, steps: Optional[List[Dict]] = None) -> pd.DataFrame:
        """
        Створює та виконує конвеєр обробки даних.

        Parameters:
        -----------
        data : pandas.DataFrame
            Вхідний DataFrame
        steps : list of dict, optional
            Список кроків обробки з параметрами

        Returns:
        --------
        pandas.DataFrame
            Оброблений DataFrame
        """
        # Якщо steps не вказано, використати стандартний конвеєр
        # Виконати кожен крок з відповідними параметрами
        # Логувати процес та результати
        # Повернути фінальний оброблений DataFrame
        pass

 def main():
     data_source = {
         'csv': {
             'BTC': {
                 '1d': './data/crypto_data/BTCUSDT_1d.csv',
                 '1h': './data/crypto_data/BTCUSDT_1h.csv',
                 '4h': './data/crypto_data/BTCUSDT_4h.csv'
             },
             'ETH': {
                 '1d': './data/crypto_data/ETHUSDT_1d.csv',
                 '1h': './data/crypto_data/ETHUSDT_1h.csv',
                 '4h': './data/crypto_data/ETHUSDT_4h.csv'
             },
             'SOL': {
                 '1d': './data/crypto_data/SOLUSDT_1d.csv',
                 '1h': './data/crypto_data/SOLUSDT_1h.csv',
                 '4h': './data/crypto_data/SOLUSDT_4h.csv'
             }
         }
     }
    symbol = ['BTC','SOL','ETH']
    interval = ['1m','1h','4h','1d']
