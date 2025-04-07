from binance.client import Client
from utils.config import BINANCE_API_KEY, BINANCE_API_SECRET

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
tickers = client.get_all_tickers()
print (tickers)