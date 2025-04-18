import asyncio
import hashlib
import hmac
import json
import time
import os
from datetime import datetime
from urllib.parse import urlencode
import aiohttp
import pandas as pd
import requests
import websocket
from utils.config import BINANCE_API_SECRET, BINANCE_API_KEY
from data.db import DatabaseManager

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


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
        self.active_websockets = {}
        self.db_manager = DatabaseManager()
        self.reconnect_required = False
        self.supported_symbols = self.db_manager.supported_symbols
        self.cache = {
            'ticker_price': {},
            'order_book': {},
            'klines': {}
        }
        self.cache_ttl = {
            'ticker_price': 5,
            'order_book': 2,
            'klines': 60
        }
        self.last_cache_update = {
            'ticker_price': {},
            'order_book': {},
            'klines': {}
        }

    # Генерація цифрового підпису
    def _generate_signature(self, params):
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    # Перевірка допустимості символу
    def _validate_symbol(self, symbol):
        """Перевіряє, чи є символ допустимим для роботи"""
        base_symbol = None

        # Витягуємо базовий символ (наприклад, BTC з BTCUSDT)
        for s in self.supported_symbols:
            if symbol.startswith(s):
                base_symbol = s
                break

        if not base_symbol:
            self.db_manager.log_event('WARNING', f"Непідтримувана криптовалюта: {symbol}", 'BinanceClient')
            return False, None

        return True, base_symbol

    # Отримання даних про ціну у вигляді свічок
    def get_klines(self, symbol, interval, limit=100, start_time=None, end_time=None, use_cache=True):
        cache_key = f"{symbol}_{interval}_{start_time}_{end_time}_{limit}"

        if use_cache and cache_key in self.cache['klines']:
            cache_time = self.last_cache_update['klines'].get(cache_key, 0)
            if time.time() - cache_time < self.cache_ttl['klines']:
                return self.cache['klines'][cache_key]

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
        except requests.exceptions.RequestException as e:
            self.db_manager.log_event('ERROR', f"Error getting klines: {e}", 'BinanceClient')
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

        # Оновлюємо кеш
        if use_cache:
            self.cache['klines'][cache_key] = df.copy()
            self.last_cache_update['klines'][cache_key] = time.time()

        return df

    # Отримання поточної ціни
    def get_ticker_price(self, symbol=None, use_cache=True):
        # Перевіряємо кеш, якщо дозволено і якщо запитується конкретний символ
        if use_cache and symbol and symbol in self.cache['ticker_price']:
            cache_time = self.last_cache_update['ticker_price'].get(symbol, 0)
            if time.time() - cache_time < self.cache_ttl['ticker_price']:
                return self.cache['ticker_price'][symbol]

        endpoint = f"{self.base_url_v3}/ticker/price"
        params = {}
        if symbol:
            params['symbol'] = symbol

        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            # Оновлюємо кеш
            if use_cache:
                if symbol:
                    self.cache['ticker_price'][symbol] = data
                    self.last_cache_update['ticker_price'][symbol] = time.time()
                else:
                    # Якщо запитуються всі ціни, оновлюємо кеш для кожного символу
                    for item in data:
                        self.cache['ticker_price'][item['symbol']] = {
                            'symbol': item['symbol'],
                            'price': item['price']
                        }
                        self.last_cache_update['ticker_price'][item['symbol']] = time.time()

            return data
        except requests.exceptions.RequestException as e:
            self.db_manager.log_event('ERROR', f"Error getting ticker price: {e}", 'BinanceClient')
            return {}

    # Отримання книги ордерів(поточних пропозицій на купівлю чи продаж)
    def get_order_book(self, symbol, limit=4500, use_cache=True):
        # Перевіряємо кеш, якщо дозволено
        if use_cache and symbol in self.cache['order_book']:
            cache_time = self.last_cache_update['order_book'].get(symbol, 0)
            if time.time() - cache_time < self.cache_ttl['order_book']:
                return self.cache['order_book'][symbol]

        endpoint = f"{self.base_url_v3}/depth"
        params = {
            'symbol': symbol,
            'limit': limit
        }

        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            self.db_manager.log_event('ERROR', f"Error getting order book: {e}", 'BinanceClient')
            return {'bids': pd.DataFrame(), 'asks': pd.DataFrame(), 'lastUpdateId': 0}

        bids_df = pd.DataFrame(data['bids'], columns=['price', 'quantity'])
        asks_df = pd.DataFrame(data['asks'], columns=['price', 'quantity'])

        bids_df[['price', 'quantity']] = bids_df[['price', 'quantity']].apply(pd.to_numeric)
        asks_df[['price', 'quantity']] = asks_df[['price', 'quantity']].apply(pd.to_numeric)

        result = {
            'lastUpdateId': data['lastUpdateId'],
            'bids': bids_df,
            'asks': asks_df
        }

        # Оновлюємо кеш
        if use_cache:
            self.cache['order_book'][symbol] = result
            self.last_cache_update['order_book'][symbol] = time.time()

        return result

    # Отримання останніх угод
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
        except requests.exceptions.RequestException as e:
            self.db_manager.log_event('ERROR', f"Error getting recent trades: {e}", 'BinanceClient')
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
        except requests.exceptions.RequestException as e:
            self.db_manager.log_event('ERROR', f"Error getting 24hr ticker statistics: {e}", 'BinanceClient')
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
        except requests.exceptions.RequestException as e:
            self.db_manager.log_event('ERROR', f"Error getting account info: {e}", 'BinanceClient')
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
                        self.db_manager.log_event('ERROR', f"Error {response.status} for {symbol}: {text}",
                                                  'BinanceClient')
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
                self.db_manager.log_event('ERROR', f"Error fetching klines for {symbol}: {e}", 'BinanceClient')
                return symbol, pd.DataFrame()

        async with aiohttp.ClientSession() as session:
            tasks = [fetch_klines(session, symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks)

        return {symbol: df for symbol, df in results}

    # ===== WebSocket methods for real-time data =====

    def _save_kline_to_db(self, kline_data):
        """Зберігає дані свічки в базу даних"""
        try:
            # Отримуємо базову валюту з символа (напр. BTC з BTCUSDT)
            symbol = kline_data['symbol']
            is_valid, crypto_symbol = self._validate_symbol(symbol)

            if not is_valid:
                return False

            # Перетворюємо дані у формат, який очікує DatabaseManager
            kline_db_data = {
                'interval': kline_data['interval'],
                'open_time': kline_data['open_time'],
                'open': float(kline_data['open']),
                'high': float(kline_data['high']),
                'low': float(kline_data['low']),
                'close': float(kline_data['close']),
                'volume': float(kline_data['volume']),
                'close_time': kline_data['close_time'],
                'quote_asset_volume': float(kline_data['quote_asset_volume']),
                'number_of_trades': int(kline_data['number_of_trades']),
                'taker_buy_base_volume': float(kline_data['taker_buy_base_volume']),
                'taker_buy_quote_volume': float(kline_data['taker_buy_quote_volume']),
                'is_closed': kline_data['is_closed']
            }

            # Вставка даних у відповідну таблицю
            result = self.db_manager.insert_kline(crypto_symbol, kline_db_data)
            return result
        except Exception as e:
            self.db_manager.log_event('ERROR', f"Error saving kline to database: {e}", 'BinanceClient')
            return False

    def _save_orderbook_to_db(self, orderbook_data):
        """Зберігає дані книги ордерів в базу даних"""
        try:
            # Отримуємо базову валюту з символа (напр. BTC з BTCUSDT)
            symbol = orderbook_data['symbol']
            is_valid, crypto_symbol = self._validate_symbol(symbol)

            if not is_valid:
                return False

            timestamp = orderbook_data['timestamp']
            last_update_id = orderbook_data['last_update_id']

            # Обробка ордерів на купівлю (bids)
            if 'bids' in orderbook_data and orderbook_data['bids']:
                for bid in orderbook_data['bids']:
                    orderbook_entry = {
                        'timestamp': timestamp,
                        'last_update_id': last_update_id,
                        'type': 'bid',
                        'price': float(bid[0]),
                        'quantity': float(bid[1])
                    }
                    self.db_manager.insert_orderbook_entry(crypto_symbol, orderbook_entry)

            # Обробка ордерів на продаж (asks)
            if 'asks' in orderbook_data and orderbook_data['asks']:
                for ask in orderbook_data['asks']:
                    orderbook_entry = {
                        'timestamp': timestamp,
                        'last_update_id': last_update_id,
                        'type': 'ask',
                        'price': float(ask[0]),
                        'quantity': float(ask[1])
                    }
                    self.db_manager.insert_orderbook_entry(crypto_symbol, orderbook_entry)

            return True
        except Exception as e:
            self.db_manager.log_event('ERROR', f"Error saving orderbook to database: {e}", 'BinanceClient')
            return False

    def start_kline_socket(self, symbol, interval, callback, save_to_db=True):
        """Запуск WebSocket для отримання даних свічок"""
        # Перевіряємо, чи є базовий символ допустимим
        is_valid, _ = self._validate_symbol(symbol)
        if not is_valid:
            self.db_manager.log_event('ERROR', f"Cannot start kline socket for unsupported symbol: {symbol}",
                                      'BinanceClient')
            return None

        socket_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@kline_{interval}"

        # Додаємо запис про WebSocket-з'єднання в базу даних
        if save_to_db:
            self.db_manager.update_websocket_status(symbol, 'kline', interval, True)

        def on_message(ws, message):
            try:
                callback(ws, message)

                if save_to_db:
                    try:
                        data = json.loads(message)
                        candle = data['k']

                        kline_data = {
                            'symbol': candle['s'],  # symbol
                            'interval': candle['i'],  # interval
                            'open_time': datetime.fromtimestamp(candle['t'] / 1000),  # open_time
                            'open': candle['o'],  # open
                            'high': candle['h'],  # high
                            'low': candle['l'],  # low
                            'close': candle['c'],  # close
                            'volume': candle['v'],  # volume
                            'close_time': datetime.fromtimestamp(candle['T'] / 1000),  # close_time
                            'quote_asset_volume': candle['q'],  # quote asset volume
                            'number_of_trades': candle['n'],  # number of trades
                            'taker_buy_base_volume': candle['V'],  # taker buy base volume
                            'taker_buy_quote_volume': candle['Q'],  # taker buy quote volume
                            'is_closed': candle['x']  # is closed
                        }

                        self._save_kline_to_db(kline_data)
                    except Exception as e:
                        self.db_manager.log_event('ERROR', f"Error saving kline data to database: {e}", 'BinanceClient')
            except Exception as e:
                self.db_manager.log_event('ERROR', f"Error in on_message handler for kline: {e}", 'BinanceClient')

        def on_error(ws, error):
            self.db_manager.log_event('ERROR', f"WebSocket Error: {error}", 'BinanceClient')
            ws.custom_reconnect_info['needs_reconnect'] = True
            if save_to_db:
                self.db_manager.update_websocket_status(symbol, 'kline', interval, False)

        def on_close(ws, close_status_code, close_msg):
            self.db_manager.log_event('INFO',
                                      f"WebSocket Connection Closed: {close_status_code}, {close_msg if close_msg else 'No message'}",
                                      'BinanceClient')
            ws.custom_reconnect_info['needs_reconnect'] = True
            if save_to_db:
                self.db_manager.update_websocket_status(symbol, 'kline', interval, False)
            # Запускаємо пізніше перепідключення
            self.reconnect_required = True

        def on_open(ws):
            self.db_manager.log_event('INFO', f"WebSocket Connection Opened for {symbol} {interval} klines",
                                      'BinanceClient')
            ws.custom_reconnect_info['needs_reconnect'] = False
            if save_to_db:
                self.db_manager.update_websocket_status(symbol, 'kline', interval, True)

        ws = websocket.WebSocketApp(
            socket_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )

        # Додаємо механізм пінгу для підтримки з'єднання
        ws.on_ping = lambda ws, message: ws.send(json.dumps({"type": "pong"}))

        # Додаємо необхідну інформацію для перепідключення
        ws.custom_reconnect_info = {
            'needs_reconnect': False,
            'symbol': symbol,
            'interval': interval,
            'callback': callback,
            'save_to_db': save_to_db,
            'socket_type': 'kline',
            'last_ping_time': time.time()
        }

        import threading
        ws_thread = threading.Thread(target=ws.run_forever, kwargs={'ping_interval': 30, 'ping_timeout': 10})
        ws_thread.daemon = True
        ws_thread.start()

        # Зберігаємо веб-сокет в активних з'єднаннях
        ws_key = f"kline_{symbol}_{interval}"
        self.active_websockets[ws_key] = {
            'ws': ws
        }

        return ws

    def order_book_socket(self, symbol, callback, save_to_db=True):
        """Запуск WebSocket для отримання даних книги ордерів"""
        # Перевіряємо, чи є базовий символ допустимим
        is_valid, _ = self._validate_symbol(symbol)
        if not is_valid:
            self.db_manager.log_event('ERROR', f"Cannot start orderbook socket for unsupported symbol: {symbol}",
                                      'BinanceClient')
            return None

        socket_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@depth20@100ms"

        # Додаємо запис про WebSocket-з'єднання в базу даних
        if save_to_db:
            self.db_manager.update_websocket_status(symbol, 'orderbook', None, True)

        def on_message(ws, message):
            try:
                callback(ws, message)

                if save_to_db:
                    try:
                        data = json.loads(message)
                        timestamp = datetime.now()

                        orderbook_data = {
                            'symbol': symbol,
                            'timestamp': timestamp,
                            'last_update_id': data.get('lastUpdateId', 0),
                            'bids': data.get('bids', []),
                            'asks': data.get('asks', [])
                        }

                        self._save_orderbook_to_db(orderbook_data)
                    except Exception as e:
                        self.db_manager.log_event('ERROR', f"Error saving order book data to database: {e}",
                                                  'BinanceClient')
            except Exception as e:
                self.db_manager.log_event('ERROR', f"Error in on_message handler for orderbook: {e}", 'BinanceClient')

        def on_error(ws, error):
            self.db_manager.log_event('ERROR', f"WebSocket Error: {error}", 'BinanceClient')
            ws.custom_reconnect_info['needs_reconnect'] = True
            if save_to_db:
                self.db_manager.update_websocket_status(symbol, 'orderbook', None, False)

        def on_close(ws, close_status_code, close_msg):
            self.db_manager.log_event('INFO',
                                      f"WebSocket Connection Closed: {close_status_code}, {close_msg if close_msg else 'No message'}",
                                      'BinanceClient')
            ws.custom_reconnect_info['needs_reconnect'] = True
            if save_to_db:
                self.db_manager.update_websocket_status(symbol, 'orderbook', None, False)
            # Запускаємо пізніше перепідключення
            self.reconnect_required = True

        def on_open(ws):
            self.db_manager.log_event('INFO', f"WebSocket Connection Opened for {symbol} orderbook", 'BinanceClient')
            ws.custom_reconnect_info['needs_reconnect'] = False
            if save_to_db:
                self.db_manager.update_websocket_status(symbol, 'orderbook', None, True)

        ws = websocket.WebSocketApp(
            socket_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )

        # Додаємо механізм пінгу для підтримки з'єднання
        ws.on_ping = lambda ws, message: ws.send(json.dumps({"type": "pong"}))

        # Додаємо необхідну інформацію для перепідключення
        ws.custom_reconnect_info = {
            'needs_reconnect': False,
            'symbol': symbol,
            'callback': callback,
            'save_to_db': save_to_db,
            'socket_type': 'orderbook',
            'last_ping_time': time.time()
        }

        import threading
        ws_thread = threading.Thread(target=ws.run_forever, kwargs={'ping_interval': 30, 'ping_timeout': 10})
        ws_thread.daemon = True
        ws_thread.start()

        # Зберігаємо веб-сокет в активних з'єднаннях
        ws_key = f"orderbook_{symbol}"
        self.active_websockets[ws_key] = {
            'ws': ws
        }

        return ws

    # Перевірка та автоматичне перепідключення сокетів
    def check_websocket_connections(self):
        """Перевірка з'єднань WebSocket та перепідключення при необхідності"""
        disconnected_sockets = []

        for key, ws_info in list(self.active_websockets.items()):
            ws = ws_info['ws']
            if not ws.sock or not ws.sock.connected or ws.custom_reconnect_info['needs_reconnect']:
                self.db_manager.log_event('WARNING', f"WebSocket {key} is disconnected or requires reconnection.",
                                          'BinanceClient')
                disconnected_sockets.append(key)
            else:
                # Перевіряємо час останнього пінгу
                if 'last_ping_time' in ws.custom_reconnect_info:
                    last_ping = ws.custom_reconnect_info['last_ping_time']
                    if time.time() - last_ping > 120:  # 2 хвилини без пінгу
                        self.db_manager.log_event('WARNING', f"WebSocket {key} has not received ping for too long.",
                                                  'BinanceClient')
                        disconnected_sockets.append(key)

        # Перепідключення відключених сокетів
        for key in disconnected_sockets:
            self.reconnect_websocket(key)

    def reconnect_websocket(self, ws_key):
        """Перепідключення WebSocket за ключем"""
        if ws_key not in self.active_websockets:
            self.db_manager.log_event('ERROR', f"Cannot reconnect unknown WebSocket: {ws_key}", 'BinanceClient')
            return False

        # Закриваємо старий WebSocket
        old_ws = self.active_websockets[ws_key]['ws']
        reconnect_info = old_ws.custom_reconnect_info

        try:
            old_ws.close()
        except Exception as e:
            print(f"Error closing old WebSocket {ws_key}: {e}")
            self.db_manager.log_event('ERROR', f"Error closing old WebSocket {ws_key}: {e}", 'BinanceClient')

        # Пауза перед повторним підключенням
        time.sleep(2)

        # Перепідключення в залежності від типу сокета
        if reconnect_info['socket_type'] == 'kline':
            print(f"Reconnecting kline WebSocket for {reconnect_info['symbol']} {reconnect_info['interval']}...")
            self.db_manager.log_event('INFO',
                                      f"Reconnecting kline WebSocket for {reconnect_info['symbol']} {reconnect_info['interval']}...",
                                      'BinanceClient')
            self.start_kline_socket(
                reconnect_info['symbol'],
                reconnect_info['interval'],
                reconnect_info['callback'],
                reconnect_info['save_to_db']
            )
        elif reconnect_info['socket_type'] == 'orderbook':
            print(f"Reconnecting orderbook WebSocket for {reconnect_info['symbol']}...")
            self.db_manager.log_event('INFO', f"Reconnecting orderbook WebSocket for {reconnect_info['symbol']}...",
                                      'BinanceClient')
            self.order_book_socket(
                reconnect_info['symbol'],
                reconnect_info['callback'],
                reconnect_info['save_to_db']
            )

        print(f"Successfully reconnected WebSocket: {ws_key}")
        self.db_manager.log_event('INFO', f"Successfully reconnected WebSocket: {ws_key}", 'BinanceClient')
        return True

    # Зупинка роботи веб сокета
    def close_websocket(self, ws_or_key):
        if isinstance(ws_or_key, str):
            if ws_or_key in self.active_websockets:
                ws_info = self.active_websockets[ws_or_key]
                ws = ws_info['ws']

                socket_parts = ws_or_key.split('_')
                if socket_parts[0] == 'kline':
                    symbol = socket_parts[1]
                    interval = socket_parts[2]
                    self.db_manager.update_websocket_status(symbol, 'kline', interval, False)
                elif socket_parts[0] == 'orderbook':
                    symbol = socket_parts[1]
                    self.db_manager.update_websocket_status(symbol, 'orderbook', None, False)

                try:
                    ws.close()
                except Exception as e:
                    print(f"Error closing WebSocket {ws_or_key}: {e}")
                    self.db_manager.log_event('ERROR', f"Error closing WebSocket {ws_or_key}: {e}", 'BinanceClient')
                print(f"Closed WebSocket connection: {ws_or_key}")
                self.db_manager.log_event('INFO', f"Closed WebSocket connection: {ws_or_key}", 'BinanceClient')
                return True
            return False
        else:
            for key, ws_info in self.active_websockets.items():
                if ws_info['ws'] == ws_or_key:
                    try:
                        ws_or_key.close()
                    except Exception as e:
                        print(f"Error closing WebSocket {key}: {e}")
                        self.db_manager.log_event('ERROR', f"Error closing WebSocket {key}: {e}", 'BinanceClient')
                    print(f"Closed WebSocket connection: {key}")
                    self.db_manager.log_event('INFO', f"Closed WebSocket connection: {key}", 'BinanceClient')

                    # Оновлюємо статус у БД
                    socket_parts = key.split('_')
                    if socket_parts[0] == 'kline':
                        symbol = socket_parts[1]
                        interval = socket_parts[2]
                        self.db_manager.update_websocket_status(symbol, 'kline', interval, False)
                    elif socket_parts[0] == 'orderbook':
                        symbol = socket_parts[1]
                        self.db_manager.update_websocket_status(symbol, 'orderbook', None, False)

                    return True
            return False

    def close_all_websockets(self):
        for key, ws_info in list(self.active_websockets.items()):
            try:
                ws_info['ws'].close()
                print(f"Closed WebSocket connection: {key}")
                self.db_manager.log_event('INFO', f"Closed WebSocket connection: {key}", 'BinanceClient')

                # Оновлюємо статус у БД
                socket_parts = key.split('_')
                if socket_parts[0] == 'kline':
                    symbol = socket_parts[1]
                    interval = socket_parts[2]
                    self.db_manager.update_websocket_status(symbol, 'kline', interval, False)
                elif socket_parts[0] == 'orderbook':
                    symbol = socket_parts[1]
                    self.db_manager.update_websocket_status(symbol, 'orderbook', None, False)
            except Exception as e:
                print(f"Error closing WebSocket {key}: {e}")
                self.db_manager.log_event('ERROR', f"Error closing WebSocket {key}: {e}", 'BinanceClient')

        self.active_websockets.clear()

    def cleanup(self):
        """Повне очищення всіх ресурсів"""
        self.close_all_websockets()
        self.db_manager.disconnect()
        print("Cleaned up all resources")

    # ===== Сервісні функції для перепідключення та моніторингу =====

    def check_and_handle_reconnections(self):
        if self.reconnect_required:
            print("Reconnection flag detected, checking WebSocket connections...")
            self.db_manager.log_event('INFO', "Reconnection flag detected, checking WebSocket connections...",
                                      'BinanceClient')
            self.check_websocket_connections()
            self.reconnect_required = False

    # ===== Збереження історичних даних в базу даних =====

    def save_historical_data_to_db(self, symbol, interval, start_date, end_date=None):
        """Зберігає історичні дані свічок в базу даних"""
        print(f"Починаємо зберігання історичних даних для {symbol} з інтервалом {interval}")
        self.db_manager.log_event('INFO', f"Починаємо зберігання історичних даних для {symbol} з інтервалом {interval}",
                                  'BinanceClient')

        base_asset = symbol[:-4]
        quote_asset = symbol[-4:]

        self.db_manager.insert_cryptocurrency(symbol, base_asset, quote_asset)

        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        if end_date:
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        else:
            end_ts = int(datetime.now().timestamp() * 1000)

        current_start = start_ts
        saved_candles_count = 0

        while current_start < end_ts:
            current_end = min(current_start + (1000 * 60 * 60 * 24), end_ts)  # Обмеження на 1 день данних за запит

            df = self.get_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=current_end,
                limit=1000
            )

            if df.empty:
                print(
                    f"Дані відсутні для {symbol} від {datetime.fromtimestamp(current_start / 1000)} до {datetime.fromtimestamp(current_end / 1000)}")
                self.db_manager.log_event('WARNING',
                                          f"Дані відсутні для {symbol} від {datetime.fromtimestamp(current_start / 1000)} до {datetime.fromtimestamp(current_end / 1000)}",
                                          'BinanceClient')
                current_start = current_end + 1
                continue

            for _, row in df.iterrows():
                kline_data = {
                    'symbol': symbol,
                    'interval': interval,
                    'open_time': row['open_time'],
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume']),
                    'close_time': row['close_time'],
                    'quote_asset_volume': float(row['quote_asset_volume']),
                    'number_of_trades': int(row['number_of_trades']),
                    'taker_buy_base_volume': float(row['taker_buy_base_asset_volume']),
                    'taker_buy_quote_volume': float(row['taker_buy_quote_asset_volume']),
                    'is_closed': True
                }

                try:
                    self._save_kline_to_db(kline_data)
                    saved_candles_count += 1
                except Exception as e:
                    print(f"Помилка збереження свічки в базу даних: {e}")
                    self.db_manager.log_event('ERROR', f"Помилка збереження свічки в базу даних: {e}", 'BinanceClient')

            if len(df) > 0:
                current_start = int(df.iloc[-1]['close_time'].timestamp() * 1000) + 1
            else:
                break
            time.sleep(2)

        if saved_candles_count > 0:
            print(f"Збережено {saved_candles_count} свічок для {symbol} ({interval}) в базу даних")
            self.db_manager.log_event('INFO',
                                      f"Збережено {saved_candles_count} свічок для {symbol} ({interval}) в базу даних",
                                      'BinanceClient')
            return saved_candles_count
        else:
            print(f"Не вдалося зберегти жодної свічки для {symbol} за вказаний період")
            self.db_manager.log_event('WARNING', f"Не вдалося зберегти жодної свічки для {symbol} за вказаний період",
                                      'BinanceClient')
            return 0

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

    # Отримуємо символ із URL сокета
    symbol = "Unknown"
    if hasattr(ws, 'url'):
        symbol_part = ws.url.split('@')[0].split('/')[-1]
        symbol = symbol_part.upper()

    print(f"Оновлення книги ордерів {symbol}:")
    print(f"Час оновлення: {datetime.now()}")
    print(f"Останній ID оновлення: {data.get('lastUpdateId', 'Немає')}")

    if 'bids' in data and data['bids']:
        print("Топ 3 ордерів на купівлю:")
        for i, bid in enumerate(data['bids'][:3]):
            print(f"  {i + 1}. Ціна: {bid[0]}, Обсяг: {bid[1]}")

    if 'asks' in data and data['asks']:
        print("Топ 3 ордерів на продаж:")
        for i, ask in enumerate(data['asks'][:3]):
            print(f"  {i + 1}. Ціна: {ask[0]}, Обсяг: {ask[1]}")

    print("-----")


def main():
    client = BinanceClient()
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    try:
        # Отримання поточних цін для всіх символів
        for symbol in symbols:
            try:
                price = client.get_ticker_price(symbol=symbol)
                if price:
                    print(f"Поточна ціна {symbol}: {price['price']}")
                else:
                    print(f"Не вдалося отримати ціну {symbol}")
            except Exception as e:
                print(f"Помилка при отриманні ціни {symbol}: {e}")

        print("Підключення до WebSocket для отримання даних у реальному часі...")

        active_sockets = []
        # Запуск WebSocket для всіх символів (свічки)
        for symbol in symbols:
            try:
                # Створення веб-сокетів для свічок (1-хвилинний інтервал)
                socket1 = client.start_kline_socket(symbol, "1m", handle_kline_message, save_to_db=True)
                if socket1:
                    active_sockets.append(socket1)

                # Створення веб-сокетів для книги ордерів
                socket2 = client.order_book_socket(symbol, handle_order_book_message, save_to_db=True)
                if socket2:
                    active_sockets.append(socket2)
            except Exception as e:
                print(f"Помилка при створенні WebSocket для {symbol}: {e}")

        if not active_sockets:
            print("Не вдалося створити жодного WebSocket з'єднання!")
            return

        print("Очікування даних ринку в реальному часі. Натисніть Ctrl+C для виходу.")

        while True:
            time.sleep(5)
            client.check_and_handle_reconnections()

    except KeyboardInterrupt:
        print("Завершення роботи...")
    except Exception as e:
        print(f"Критична помилка: {e}")
    finally:
        client.cleanup()

if __name__ == "__main__":
    main()