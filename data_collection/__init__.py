from utils.config import BINANCE_API_KEY, BINANCE_API_SECRET
from .binance_client import BinanceClient
from .market_data_processor import MarketDataProcessor
from .feature_engineering import FeatureEngineering
from .twitter_scraper import TwitterScraper
from .crypto_news_scraper import CryptoNewsScraper

# Визначення констант, які будуть доступні на рівні пакету
# Ці константи можуть використовуватись в різних модулях пакету
SUPPORTED_EXCHANGES = ['binance', 'kucoin', 'coinbase']
SUPPORTED_TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d', '1w']
DEFAULT_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']


# Функція для зручного створення клієнта з конфігурацією за замовчуванням
def create_binance_client(api_key=None, api_secret=None, use_testnet=False):

    client = BinanceClient(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)
    if use_testnet:
        client.base_url = "https://testnet.binance.vision"
    return client


# Допоміжна функція для налаштування логування
def setup_data_collection_logging(log_level='INFO'):

    import logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=log_format, level=getattr(logging, log_level))

    # Налаштування логерів для пакету
    logger = logging.getLogger('data_collection')
    logger.setLevel(getattr(logging, log_level))

    return logger


# Версія пакету
__version__ = '0.1.0'