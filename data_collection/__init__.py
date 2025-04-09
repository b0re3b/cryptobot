# data_collection/__init__.py

# Імпорт основних класів
from .binance_client import BinanceClient
from .market_data_processor import MarketDataProcessor
from .feature_engineering import FeatureEngineer
from .twitter_scraper import TwitterScraper
from .crypto_news_scraper import CryptoNewsScraper

# Визначення констант, які будуть доступні на рівні пакету
# Ці константи можуть використовуватись в різних модулях пакету
SUPPORTED_EXCHANGES = ['binance', 'kucoin', 'coinbase']
SUPPORTED_TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d', '1w']
DEFAULT_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']


# Функція для зручного створення клієнта з конфігурацією за замовчуванням
def create_binance_client(api_key=None, api_secret=None, use_testnet=False):
    """
    Створює і повертає налаштований екземпляр BinanceClient.

    Args:
        api_key (str, optional): API ключ Binance
        api_secret (str, optional): Секретний ключ Binance
        use_testnet (bool): Використовувати тестову мережу Binance

    Returns:
        BinanceClient: Налаштований клієнт Binance
    """
    client = BinanceClient(api_key=api_key, api_secret=api_secret)
    if use_testnet:
        client.base_url = "https://testnet.binance.vision"
    return client


# Допоміжна функція для налаштування логування
def setup_data_collection_logging(log_level='INFO'):
    """
    Налаштовує логування для модулів збору даних.

    Args:
        log_level (str): Рівень логування ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    import logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=log_format, level=getattr(logging, log_level))

    # Налаштування логерів для пакету
    logger = logging.getLogger('data_collection')
    logger.setLevel(getattr(logging, log_level))

    return logger


# Версія пакету
__version__ = '0.1.0'