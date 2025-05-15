"""
Модуль data_collection для збору та обробки даних з різних джерел криптовалютного ринку.
"""

# Імпорт класів з відповідних модулів
from .binance_client import BinanceClient
from .feature_engineering import FeatureEngineering
from .crypto_news_scraper import CryptoNewsScraper
from .NewsCollector import NewsCollector
# Версія пакету
__version__ = '0.1.0'

# Публічний API пакету
__all__ = [
    'BinanceClient',
    'FeatureEngineering',
    'CryptoNewsScraper',
    'NewsCollector'
]