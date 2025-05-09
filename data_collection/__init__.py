"""
Модуль data_collection для збору та обробки даних з різних джерел криптовалютного ринку.
"""

# Імпорт класів з відповідних модулів
from .binance_client import BinanceClient
from .market_data_processor import MarketDataProcessor
from .feature_engineering import FeatureEngineering
from .crypto_news_scraper import CryptoNewsScraper
from .DataCleaner import DataCleaner
from .DataStorageManager import DataStorageManager
from .DataResampler import DataResampler
from .AnomalyDetector import AnomalyDetector
from .NewsCollector import NewsCollector
# Версія пакету
__version__ = '0.1.0'

# Публічний API пакету
__all__ = [
    'BinanceClient',
    'MarketDataProcessor',
    'FeatureEngineering',
    'CryptoNewsScraper',
    'DataCleaner',
    'DataStorageManager',
    'DataResampler',
    'AnomalyDetector',
    'NewsCollector'
]