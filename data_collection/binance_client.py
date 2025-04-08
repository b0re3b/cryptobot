from binance.client import Client
from utils.config import BINANCE_API_KEY, BINANCE_API_SECRET
import pandas as pd
import yfinance as yf

# ініціалізація клієнта
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=True)
btc = yf.Ticker("BTC-USD")

# Встановлення торгової пари і інтервалу
symbol = 'BTCUSDT'
interval = Client.KLINE_INTERVAL_1MONTH

# Отримання історії свічок (klines) за останній день з проміжком в 1 годину
klines = client.get_historical_klines(symbol, interval)

# Перетворення даних в pandas DataFrame для зручності
data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

# Перетворення timestamp в читабельний формат
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
tickers = client.get_orderbook_tickers()
trades = client.get_historical_trades(symbol='BTCUSDT')
# Виведення цін на Bitcoin за останній день (закриваюча ціна кожної години)

data = btc.history(start="2018-01-01", end="2023-01-01")

print(data.head())

