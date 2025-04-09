import asyncio
import hashlib
import hmac
import json
import time
from datetime import datetime
from urllib.parse import urlencode
from binance.client import Client
import aiohttp
import pandas as pd
import requests
import websocket

from utils.config import BINANCE_API_SECRET, BINANCE_API_KEY


class BinanceClient:
    def __init__(self):
        self.api_key = BINANCE_API_KEY
        self.api_secret = BINANCE_API_SECRET
        self.base_url = "https://api.binance.com"
        self.base_url_v3 = f"{self.base_url}/api/v3"
        self.session = requests.Session()
        self.session.headers.update({
            'X-MBX-APIKEY': self.api_key,
            'Content-Type': 'application/json'
        })

    def _generate_signature(self, params):
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    def get_exchange_info(self):
        endpoint = f"{self.base_url_v3}/exchangeInfo"
        response = self.session.get(endpoint)
        return response.json()

    def get_klines(self, symbol, interval, limit=999, start_time=None, end_time=None):
        endpoint = f"{self.base_url_v3}/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }

        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        response = self.session.get(endpoint, params=params)
        data = response.json()

        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                           'quote_asset_volume', 'taker_buy_base_asset_volume',
                           'taker_buy_quote_asset_volume']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)

        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

        return df

    def get_ticker_price(self, symbol=None):

        endpoint = f"{self.base_url_v3}/ticker/price"
        params = {}
        if symbol:
            params['symbol'] = symbol

        response = self.session.get(endpoint, params=params)
        return response.json()

    def get_order_book(self, symbol, limit=4500):

        endpoint = f"{self.base_url_v3}/depth"
        params = {
            'symbol': symbol,
            'limit': limit
        }

        response = self.session.get(endpoint, params=params)
        data = response.json()

        bids_df = pd.DataFrame(data['bids'], columns=['price', 'quantity'])
        asks_df = pd.DataFrame(data['asks'], columns=['price', 'quantity'])

        bids_df[['price', 'quantity']] = bids_df[['price', 'quantity']].apply(pd.to_numeric)
        asks_df[['price', 'quantity']] = asks_df[['price', 'quantity']].apply(pd.to_numeric)

        return {
            'lastUpdateId': data['lastUpdateId'],
            'bids': bids_df,
            'asks': asks_df
        }

    def get_recent_trades(self, symbol, limit=999):

        endpoint = f"{self.base_url_v3}/trades"
        params = {
            'symbol': symbol,
            'limit': limit
        }

        response = self.session.get(endpoint, params=params)
        data = response.json()

        df = pd.DataFrame(data)

        numeric_columns = ['price', 'qty', 'quoteQty']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)

        df['time'] = pd.to_datetime(df['time'], unit='ms')

        return df

    def get_24hr_ticker_statistics(self, symbol=None):

        endpoint = f"{self.base_url_v3}/ticker/24hr"
        params = {}
        if symbol:
            params['symbol'] = symbol

        response = self.session.get(endpoint, params=params)
        return response.json()

    # ===== Authenticated REST API запити =====

    def get_account_info(self):
        if not self.api_key or not self.api_secret:
            raise ValueError("API key and secret required for authenticated requests")

        endpoint = f"{self.base_url_v3}/account"
        params = {
            'timestamp': int(time.time() * 1000)
        }

        params['signature'] = self._generate_signature(params)
        response = self.session.get(endpoint, params=params)
        return response.json()

    # ===== Асинхронні методи для високої продуктивності =====

    async def get_klines_async(self, symbols, interval, limit=999, start_time=None, end_time=None):

        async def fetch_klines(session, symbol):
            endpoint = f"{self.base_url_v3}/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }

            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time

            async with session.get(endpoint, params=params) as response:
                data = await response.json()

                df = pd.DataFrame(data, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])

                numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                                   'quote_asset_volume', 'taker_buy_base_asset_volume',
                                   'taker_buy_quote_asset_volume']
                df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)

                df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

                return symbol, df

        async with aiohttp.ClientSession() as session:
            tasks = [fetch_klines(session, symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks)

        return {symbol: df for symbol, df in results}

    # ===== WebSocket методи для даних реального часу =====

    def start_kline_socket(self, symbol, interval, callback):

        socket_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@kline_{interval}"

        ws = websocket.WebSocketApp(
            socket_url,
            on_message=callback,
            on_error=lambda ws, error: print(f"WebSocket Error: {error}"),
            on_close=lambda ws: print("WebSocket Connection Closed"),
            on_open=lambda ws: print("WebSocket Connection Opened")
        )

        # Запуск в окремому потоці
        import threading
        ws_thread = threading.Thread(target=ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()

        return ws

    # ===== Методи збереження даних =====

    def save_historical_data(self, symbol, interval, start_date, end_date=None, directory="data/raw/"):

        import os

        # Конвертація дат у мілісекунди
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        if end_date:
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        else:
            end_ts = int(datetime.now().timestamp() * 1000)

        # Створення директорій, якщо не існують
        os.makedirs(f"{directory}/candles/{symbol}", exist_ok=True)

        # Збір даних частинами, враховуючи обмеження API
        all_candles = pd.DataFrame()
        current_start = start_ts

        while current_start < end_ts:
            current_end = min(current_start + (1000 * 60 * 60 * 24), end_ts)  # максимум 1000 свічок

            df = self.get_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=current_end,
                limit=1000
            )

            all_candles = pd.concat([all_candles, df])

            # Переміщення часового вікна
            if len(df) > 0:
                current_start = int(df.iloc[-1]['close_time'].timestamp() * 1000) + 1
            else:
                break

            # Дотримання рейт-лімітів
            time.sleep(0.5)

        # Збереження у Parquet форматі
        filename = f"{directory}/candles/{symbol}/{symbol}_{interval}_{start_date}"
        if end_date:
            filename += f"_to_{end_date}"
        filename += ".parquet"

        all_candles.to_parquet(filename, index=False)
        print(f"Збережено {len(all_candles)} свічок для {symbol} ({interval}) у файл {filename}")

        return filename


# Ініціалізація клієнта
client = BinanceClient(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

# Отримання інформації про біржу
exchange_info = client.get_exchange_info()
print(f"Доступно {len(exchange_info['symbols'])} торгових пар")

# Отримання історичних свічок
btc_candles = client.get_klines(symbol="BTCUSDT", interval="1h", limit=100)
print(f"Отримано {len(btc_candles)} годинних свічок для BTC/USDT")
print(btc_candles.head())

# Отримання поточної ціни
btc_price = client.get_ticker_price(symbol="BTCUSDT")
print(f"Поточна ціна BTC/USDT: {btc_price['price']}")

# Отримання книги ордерів
order_book = client.get_order_book(symbol="BTCUSDT", limit=10)
print("Топ 10 ордерів на купівлю:")
print(order_book['bids'].head(10))
print("Топ 10 ордерів на продаж:")
print(order_book['asks'].head(10))

# Збереження історичних даних
client.save_historical_data(
    symbol="BTCUSDT",
    interval="1d",
    start_date="2023-01-01",
    end_date="2023-12-31"
)


# Приклад обробника WebSocket
def handle_kline_message(ws, message):
    data = json.loads(message)
    candle = data['k']

    print(f"Нова свічка {candle['s']} {candle['i']}:")
    print(f"Час відкриття: {datetime.fromtimestamp(candle['t'] / 1000)}")
    print(f"Ціна відкриття: {candle['o']}")
    print(f"Максимальна ціна: {candle['h']}")
    print(f"Мінімальна ціна: {candle['l']}")
    print(f"Поточна ціна: {candle['c']}")
    print(f"Обсяг: {candle['v']}")
    print(f"Закрита: {candle['x']}")
    print("-----")


# Підключення до WebSocket для отримання свічок у реальному часі
btc_socket = client.start_kline_socket("BTCUSDT", "1m", handle_kline_message)


# Асинхронний приклад (потрібно запускати в асинхронному середовищі)
async def main():
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"]
    klines_data = await client.get_klines_async(symbols, interval="1h", limit=100)

    for symbol, data in klines_data.items():
        print(f"{symbol}: отримано {len(data)} свічок")

# asyncio.run(main())