from binance.client import Client
from utils.config import BINANCE_API_KEY, BINANCE_API_SECRET
import pandas as pd
import yfinance as yf

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=True)

symbol = 'BTCUSDT'
interval = Client.KLINE_INTERVAL_1MONTH

klines = client.get_historical_klines(symbol, interval)
orders = client.ws_get_order_book(symbol="BTCUSDT")

data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
tickers = client.get_orderbook_tickers()
trades = client.get_historical_trades(symbol='BTCUSDT')

print(orders)