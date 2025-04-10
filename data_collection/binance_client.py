import asyncio
import hashlib
import hmac
import json
import time
import os
import sys
from datetime import datetime
from logging import exception
from urllib.parse import urlencode
import aiohttp
import pandas as pd
import requests
import websocket
from utils.config import BINANCE_API_SECRET, BINANCE_API_KEY

# Збереження зібраних даних у папку data
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')


os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)


class BinanceClient:
    #Ініціалізація binance client
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
    #Генерація цифрового підпису
    def _generate_signature(self, params):
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    #Отримання даних про ціну у вигляді свічок
    def get_klines(self, symbol, interval, limit=100, start_time=None, end_time=None):
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

        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error getting klines: {e}")
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        if df.empty:
            return df

        numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                           'quote_asset_volume', 'taker_buy_base_asset_volume',
                           'taker_buy_quote_asset_volume']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)

        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

        return df
    #Отримання поточної ціни
    def get_ticker_price(self, symbol=None):
        endpoint = f"{self.base_url_v3}/ticker/price"
        params = {}
        if symbol:
            params['symbol'] = symbol

        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting ticker price: {e}")
            return {}
    #Отримання книги ордерів(поточних пропозицій на купівлю чи продаж)
    def get_order_book(self, symbol, limit=4500):
        endpoint = f"{self.base_url_v3}/depth"
        params = {
            'symbol': symbol,
            'limit': limit
        }

        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error getting order book: {e}")
            return {'bids': pd.DataFrame(), 'asks': pd.DataFrame(), 'lastUpdateId': 0}

        bids_df = pd.DataFrame(data['bids'], columns=['price', 'quantity'])
        asks_df = pd.DataFrame(data['asks'], columns=['price', 'quantity'])

        bids_df[['price', 'quantity']] = bids_df[['price', 'quantity']].apply(pd.to_numeric)
        asks_df[['price', 'quantity']] = asks_df[['price', 'quantity']].apply(pd.to_numeric)

        return {
            'lastUpdateId': data['lastUpdateId'],
            'bids': bids_df,
            'asks': asks_df
        }
    #Отримання останніх угод
    def get_recent_trades(self, symbol, limit=100):
        endpoint = f"{self.base_url_v3}/trades"
        params = {
            'symbol': symbol,
            'limit': limit
        }

        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error getting recent trades: {e}")
            return pd.DataFrame()

        df = pd.DataFrame(data)

        if df.empty:
            return df

        numeric_columns = ['price', 'qty', 'quoteQty']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)

        df['time'] = pd.to_datetime(df['time'], unit='ms')

        return df

    def get_24hr_ticker_statistics(self, symbol=None):
        endpoint = f"{self.base_url_v3}/ticker/24hr"
        params = {}
        if symbol:
            params['symbol'] = symbol

        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting 24hr ticker statistics: {e}")
            return {}

    # ===== Authenticated REST API requests =====

    def get_account_info(self):
        if not self.api_key or not self.api_secret:
            raise ValueError("API key and secret required for authenticated requests")

        endpoint = f"{self.base_url_v3}/account"
        params = {
            'timestamp': int(time.time() * 1000)
        }

        params['signature'] = self._generate_signature(params)

        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting account info: {e}")
            return {}

    # ===== Async methods for high performance =====

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

            try:
                async with session.get(endpoint, params=params) as response:
                    if response.status != 200:
                        text = await response.text()
                        print(f"Error {response.status} for {symbol}: {text}")
                        return symbol, pd.DataFrame()

                    data = await response.json()

                    df = pd.DataFrame(data, columns=[
                        'open_time', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ])

                    if df.empty:
                        return symbol, df

                    numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                                       'quote_asset_volume', 'taker_buy_base_asset_volume',
                                       'taker_buy_quote_asset_volume']
                    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)

                    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

                    return symbol, df
            except Exception as e:
                print(f"Error fetching klines for {symbol}: {e}")
                return symbol, pd.DataFrame()

        async with aiohttp.ClientSession() as session:
            tasks = [fetch_klines(session, symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks)

        return {symbol: df for symbol, df in results}

    # ===== WebSocket methods for real-time data =====

    def start_kline_socket(self, symbol, interval, callback, save_to_file=False, directory=None):
        if directory is None:
            directory = os.path.join(DATA_RAW_DIR, 'candles')

        socket_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@kline_{interval}"

        file_handler = None
        if save_to_file:
            symbol_dir = os.path.join(directory, symbol)
            os.makedirs(symbol_dir, exist_ok=True)
            print(f"Creating directory for candles: {symbol_dir}")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(symbol_dir, f"{symbol}_{interval}_{timestamp}.csv")

            file = open(filename, 'w', newline='')
            import csv
            writer = csv.writer(file)
            writer.writerow([
                'symbol', 'interval', 'open_time', 'open', 'high', 'low',
                'close', 'volume', 'close_time', 'is_closed'
            ])

            file_handler = {
                'file': file,
                'writer': writer,
                'filename': filename
            }

            print(f"Saving kline data to {filename}")

        def callback_wrapper(ws, message):
            callback(ws, message)

            if save_to_file and file_handler:
                try:
                    data = json.loads(message)
                    candle = data['k']

                    file_handler['writer'].writerow([
                        candle['s'],  # symbol
                        candle['i'],  # interval
                        datetime.fromtimestamp(candle['t'] / 1000),  # open_time
                        candle['o'],  # open
                        candle['h'],  # high
                        candle['l'],  # low
                        candle['c'],  # close
                        candle['v'],  # volume
                        datetime.fromtimestamp(candle['T'] / 1000),  # close_time
                        candle['x']  # is_closed
                    ])
                    file_handler['file'].flush()
                except Exception as e:
                    print(f"Error saving kline data to file: {e}")

        ws = websocket.WebSocketApp(
            socket_url,
            on_message=callback_wrapper,
            on_error=lambda ws, error: print(f"WebSocket Error: {error}"),
            on_close=lambda ws, close_status_code, close_msg: print(
                f"WebSocket Connection Closed: {close_msg if close_msg else 'No message'}"
            ),
            on_open=lambda ws: print("WebSocket Connection Opened")
        )

        import threading
        ws_thread = threading.Thread(target=ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()

        ws.file_handler = file_handler

        return ws

    def order_book_socket(self, symbol, callback, save_to_file=False, directory=None):
        if directory is None:
            directory = os.path.join(DATA_RAW_DIR, 'orderbook')

        socket_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@depth20@100ms"

        file_handler = None
        if save_to_file:
            symbol_dir = os.path.join(directory, symbol)
            os.makedirs(symbol_dir, exist_ok=True)
            print(f"Creating directory for orderbook: {symbol_dir}")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(symbol_dir, f"{symbol}_orderbook_{timestamp}.csv")

            file = open(filename, 'w', newline='')
            import csv
            writer = csv.writer(file)
            writer.writerow([
                'timestamp', 'lastUpdateId', 'type', 'price', 'quantity'
            ])

            file_handler = {
                'file': file,
                'writer': writer,
                'filename': filename
            }

            print(f"Saving order book data to {filename}")

        def callback_wrapper(ws, message):
            callback(ws, message)

            if save_to_file and file_handler:
                try:
                    data = json.loads(message)
                    timestamp = datetime.now()

                    if 'bids' in data and data['bids']:
                        for bid in data['bids']:
                            file_handler['writer'].writerow([
                                timestamp,
                                data.get('lastUpdateId', 'N/A'),
                                'bid',
                                bid[0],
                                bid[1]
                            ])

                    if 'asks' in data and data['asks']:
                        for ask in data['asks']:
                            file_handler['writer'].writerow([
                                timestamp,
                                data.get('lastUpdateId', 'N/A'),
                                'ask',
                                ask[0],
                                ask[1]
                            ])

                    file_handler['file'].flush()
                except Exception as e:
                    print(f"Error saving order book data to file: {e}")

        ws = websocket.WebSocketApp(
            socket_url,
            on_message=callback_wrapper,
            on_error=lambda ws, error: print(f"WebSocket Error: {error}"),
            on_close=lambda ws, close_status_code, close_msg: print(
                f"WebSocket Connection Closed: {close_msg if close_msg else 'No message'}"
            ),
            on_open=lambda ws: print("WebSocket Connection Opened")
        )

        import threading
        ws_thread = threading.Thread(target=ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()


        ws.file_handler = file_handler

        return ws

    # Зупинка роботи веб сокета
    def close_websocket(self, ws):
        if ws:
            if hasattr(ws, 'file_handler') and ws.file_handler:
                try:
                    ws.file_handler['file'].close()
                    print(f"Closed data file: {ws.file_handler['filename']}")
                except Exception as e:
                    print(f"Error closing file: {e}")

            ws.close()

    # ===== Збереження даних =====

    def save_historical_data(self, symbol, interval, start_date, end_date=None, directory=None):
        if directory is None:
            directory = os.path.join(DATA_RAW_DIR, 'candles')

        symbol_dir = os.path.join(directory, symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        print(f"Creating directory for historical data: {symbol_dir}")

        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        if end_date:
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        else:
            end_ts = int(datetime.now().timestamp() * 1000)

        all_candles = pd.DataFrame()
        current_start = start_ts

        while current_start < end_ts:
            current_end = min(current_start + (1000 * 60 * 60 * 24), end_ts)

            df = self.get_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=current_end,
                limit=1000
            )

            if df.empty:
                print(
                    f"No data returned for {symbol} from {datetime.fromtimestamp(current_start / 1000)} to {datetime.fromtimestamp(current_end / 1000)}")

                current_start = current_end + 1
                continue

            all_candles = pd.concat([all_candles, df])

            if len(df) > 0:
                current_start = int(df.iloc[-1]['close_time'].timestamp() * 1000) + 1
            else:
                break

            time.sleep(2)

        if all_candles.empty:
            print(f"No data collected for {symbol} for the specified time period")
            return None

        filename_prefix = f"{symbol}_{interval}_{start_date}"
        if end_date:
            filename_prefix += f"_to_{end_date}"

        filename = os.path.join(symbol_dir, f"{filename_prefix}.csv")

        try:
            all_candles.to_csv(filename, index=False)
            print(f"Saved {len(all_candles)} candles for {symbol} ({interval}) to file {filename}")
            return filename
        except Exception as e:
            print(f"Error saving data to file: {e}")
            return None

    def save_processed_data(self, dataframe, symbol, data_type, timestamp=None, directory=None):

        if directory is None:
            directory = DATA_PROCESSED_DIR

        type_dir = os.path.join(directory, data_type)
        symbol_dir = os.path.join(type_dir, symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        print(f"Creating directory for processed data: {symbol_dir}")

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = os.path.join(symbol_dir, f"{symbol}_{data_type}_{timestamp}.csv")

        try:
            dataframe.to_csv(filename, index=False)
            print(f"Saved {len(dataframe)} records of {data_type} data for {symbol} to {filename}")
            return filename
        except Exception as e:
            print(f"Error saving processed data to file: {e}")
            return None


def handle_kline_message(ws, message):
    data = json.loads(message)
    candle = data['k']

    print(f"Нова свічка {candle['s']} {candle['i']}:")
    print(f"Час відкриття: {datetime.fromtimestamp(candle['t'] / 1000)}")
    print(f"Ціна відкриття: {candle['o']}")
    print(f"Найвища ціна: {candle['h']}")
    print(f"Найнижча ціна: {candle['l']}")
    print(f"Поточна ціна: {candle['c']}")
    print(f"Обсяг: {candle['v']}")
    print(f"Ціна закриття: {candle['x']}")
    print("-----")


def handle_order_book_message(ws, message):
    data = json.loads(message)

    print(f"Оновлення книги ордерів:")
    print(f"Час оновлення: {datetime.now()}")
    print(f"Останній ID оновлення: {data.get('lastUpdateId', 'Немає')}")

    if 'bids' in data and data['bids']:
        print("Топ 5 ордерів на купівлю:")
        for i, bid in enumerate(data['bids'][:5]):
            print(f"  {i + 1}. Ціна: {bid[0]}, Обсяг: {bid[1]}")

    if 'asks' in data and data['asks']:
        print("Топ 5 ордерів на продаж:")
        for i, ask in enumerate(data['asks'][:5]):
            print(f"  {i + 1}. Ціна: {ask[0]}, Обсяг: {ask[1]}")

    print("-----")



# Main function
def main():
    client = BinanceClient()

    try:
        btc_price = client.get_ticker_price(symbol="BTCUSDT")
        if btc_price:
            print(f"Поточна ціна BTC/USDT: {btc_price['price']}")
        else:
            print("Не вдалося отримати ціну BTC/USDT")
            return
    except Exception as e:
        print(f"Помилка при отриманні ціни: {e}")
        return

    btc_candles = client.get_klines(symbol="BTCUSDT", interval="1h", limit=10)
    if not btc_candles.empty:
        print(f"Отримано {len(btc_candles)} годинних свічок для BTC/USDT")
        print(btc_candles.head())
    else:
        print("Не вдалося отримати історичні свічки")

    order_book = client.get_order_book(symbol="BTCUSDT", limit=10)
    if not order_book['bids'].empty:
        print("Топ 10 ордерів на купівлю:")
        print(order_book['bids'].head(10))
        print("Топ 10 ордерів на продаж:")
        print(order_book['asks'].head(10))
    else:
        print("Не вдалося отримати книгу ордерів")

    print("Підключення до WebSocket для отримання даних у реальному часі...")

    candles_dir = os.path.join(DATA_RAW_DIR, 'candles')
    kline_socket = client.start_kline_socket("BTCUSDT", "1m", handle_kline_message, save_to_file=True,
                                             directory=candles_dir)

    orderbook_dir = os.path.join(DATA_RAW_DIR, 'orderbook')
    order_book_socket = client.order_book_socket("BTCUSDT", handle_order_book_message, save_to_file=True,
                                                 directory=orderbook_dir)

    try:
        print("Очікування даних ринку в реальному часі. Натисніть Ctrl+C для виходу.")
        print(f"Дані зберігаються в: {DATA_RAW_DIR}")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Завершення роботи...")
        client.close_websocket(kline_socket)
        client.close_websocket(order_book_socket)


# Запуск основної функції
if __name__ == "__main__":
    main()
