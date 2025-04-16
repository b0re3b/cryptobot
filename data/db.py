import os
import psycopg2
import pandas as pd
from datetime import datetime
from psycopg2.extras import RealDictCursor
from utils.config import *

class DatabaseManager:
    def __init__(self, db_config=None):
        if db_config is None:
            self.db_config = {
                'dbname': DB_NAME,
                'user': USER,
                'password': PASSWORD,
                'host': HOST,
                'port': PORT
            }
        else:
            self.db_config = db_config

        self.conn = None
        self.cursor = None
        self.connect()
        self.create_schema()

        # Підтримувані валюти
        self.supported_symbols = ['BTC', 'ETH', 'SOL']

    def connect(self):
        """Встановлює з'єднання з базою даних"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.conn.autocommit = False
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            print(f"З'єднано з базою даних: {self.db_config['dbname']}")
        except psycopg2.Error as e:
            print(f"Помилка з'єднання з базою даних: {e}")
            raise

    def disconnect(self):
        """Закриває з'єднання з базою даних"""
        if self.conn:
            self.conn.close()
            print("З'єднання з базою даних закрито")

    def create_schema(self):
        """Створює схему бази даних"""
        try:
            schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')

            if os.path.exists(schema_path):
                # Read the script content
                with open(schema_path, 'r') as f:
                    schema_script = f.read()

                # Check if the script contains IF NOT EXISTS clauses
                if "IF NOT EXISTS" in schema_script:
                    self.cursor.execute(schema_script)
                else:
                    print("Warning: schema.sql doesn't contain IF NOT EXISTS clauses. Using built-in table creation.")
                    self._create_tables()
            else:
                self._create_tables()

            self.conn.commit()
            print("Схема бази даних успішно створена")
        except psycopg2.Error as e:
            print(f"Помилка створення схеми бази даних: {e}")
            self.conn.rollback()
            raise

    def _create_tables(self):
        # Таблиці для BTC
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS btc_klines (
            id SERIAL PRIMARY KEY,
            interval TEXT NOT NULL,
            open_time TIMESTAMP NOT NULL,
            open NUMERIC NOT NULL,
            high NUMERIC NOT NULL,
            low NUMERIC NOT NULL,
            close NUMERIC NOT NULL,
            volume NUMERIC NOT NULL,
            close_time TIMESTAMP NOT NULL,
            quote_asset_volume NUMERIC NOT NULL,
            number_of_trades INTEGER NOT NULL,
            taker_buy_base_volume NUMERIC NOT NULL,
            taker_buy_quote_volume NUMERIC NOT NULL,
            is_closed BOOLEAN NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (interval, open_time)
        )
        ''')

        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_btc_klines_time ON btc_klines(interval, open_time)')

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS btc_orderbook (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            last_update_id BIGINT NOT NULL,
            type TEXT NOT NULL,
            price NUMERIC NOT NULL,
            quantity NUMERIC NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_btc_orderbook_time ON btc_orderbook(timestamp)')

        # Таблиці для ETH
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS eth_klines (
            id SERIAL PRIMARY KEY,
            interval TEXT NOT NULL,
            open_time TIMESTAMP NOT NULL,
            open NUMERIC NOT NULL,
            high NUMERIC NOT NULL,
            low NUMERIC NOT NULL,
            close NUMERIC NOT NULL,
            volume NUMERIC NOT NULL,
            close_time TIMESTAMP NOT NULL,
            quote_asset_volume NUMERIC NOT NULL,
            number_of_trades INTEGER NOT NULL,
            taker_buy_base_volume NUMERIC NOT NULL,
            taker_buy_quote_volume NUMERIC NOT NULL,
            is_closed BOOLEAN NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (interval, open_time)
        )
        ''')

        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_eth_klines_time ON eth_klines(interval, open_time)')

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS eth_orderbook (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            last_update_id BIGINT NOT NULL,
            type TEXT NOT NULL,
            price NUMERIC NOT NULL,
            quantity NUMERIC NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_eth_orderbook_time ON eth_orderbook(timestamp)')

        # Таблиці для SOL
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS sol_klines (
            id SERIAL PRIMARY KEY,
            interval TEXT NOT NULL,
            open_time TIMESTAMP NOT NULL,
            open NUMERIC NOT NULL,
            high NUMERIC NOT NULL,
            low NUMERIC NOT NULL,
            close NUMERIC NOT NULL,
            volume NUMERIC NOT NULL,
            close_time TIMESTAMP NOT NULL,
            quote_asset_volume NUMERIC NOT NULL,
            number_of_trades INTEGER NOT NULL,
            taker_buy_base_volume NUMERIC NOT NULL,
            taker_buy_quote_volume NUMERIC NOT NULL,
            is_closed BOOLEAN NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (interval, open_time)
        )
        ''')

        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_sol_klines_time ON sol_klines(interval, open_time)')

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS sol_orderbook (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            last_update_id BIGINT NOT NULL,
            type TEXT NOT NULL,
            price NUMERIC NOT NULL,
            quantity NUMERIC NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_sol_orderbook_time ON sol_orderbook(timestamp)')

        # Таблиця для логів
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id SERIAL PRIMARY KEY,
            log_level TEXT NOT NULL,
            message TEXT NOT NULL,
            component TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

    def insert_kline(self, symbol, kline_data):

        if symbol.upper() not in self.supported_symbols:
            print(f"Валюта {symbol} не підтримується")
            return False

        table_name = f"{symbol.lower()}_klines"

        try:
            self.cursor.execute(f'''
            INSERT INTO {table_name} 
            (interval, open_time, open, high, low, close, volume, close_time, 
            quote_asset_volume, number_of_trades, taker_buy_base_volume, taker_buy_quote_volume, is_closed)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (interval, open_time) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            close_time = EXCLUDED.close_time,
            quote_asset_volume = EXCLUDED.quote_asset_volume,
            number_of_trades = EXCLUDED.number_of_trades,
            taker_buy_base_volume = EXCLUDED.taker_buy_base_volume,
            taker_buy_quote_volume = EXCLUDED.taker_buy_quote_volume,
            is_closed = EXCLUDED.is_closed
            ''', (
                kline_data['interval'],
                kline_data['open_time'],
                kline_data['open'],
                kline_data['high'],
                kline_data['low'],
                kline_data['close'],
                kline_data['volume'],
                kline_data['close_time'],
                kline_data['quote_asset_volume'],
                kline_data['number_of_trades'],
                kline_data['taker_buy_base_volume'],
                kline_data['taker_buy_quote_volume'],
                kline_data['is_closed']
            ))
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            print(f"Помилка додавання свічки для {symbol}: {e}")
            self.conn.rollback()
            return False

    def insert_orderbook_entry(self, symbol, orderbook_data):

        if symbol.upper() not in self.supported_symbols:
            print(f"Валюта {symbol} не підтримується")
            return False

        table_name = f"{symbol.lower()}_orderbook"

        try:
            self.cursor.execute(f'''
            INSERT INTO {table_name} 
            (timestamp, last_update_id, type, price, quantity)
            VALUES (%s, %s, %s, %s, %s)
            ''', (
                orderbook_data['timestamp'],
                orderbook_data['last_update_id'],
                orderbook_data['type'],
                orderbook_data['price'],
                orderbook_data['quantity']
            ))
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            print(f"Помилка додавання запису книги ордерів для {symbol}: {e}")
            self.conn.rollback()
            return False

    def insert_cryptocurrency(self, symbol, base_asset, quote_asset):

        try:
            # Перевіряємо чи існує таблиця cryptocurrencies, якщо ні - створюємо
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS cryptocurrencies (
                id SERIAL PRIMARY KEY,
                symbol TEXT NOT NULL UNIQUE,
                base_asset TEXT NOT NULL,
                quote_asset TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')

            # Перевіряємо, чи існує вже запис з таким символом
            self.cursor.execute('SELECT id FROM cryptocurrencies WHERE symbol = %s', (symbol,))
            result = self.cursor.fetchone()

            if not result:
                # Додаємо новий запис
                self.cursor.execute('''
                INSERT INTO cryptocurrencies (symbol, base_asset, quote_asset)
                VALUES (%s, %s, %s)
                ''', (symbol, base_asset, quote_asset))
                self.conn.commit()
                print(f"Додано запис про криптовалюту: {symbol}")

            return True
        except psycopg2.Error as e:
            print(f"Помилка додавання запису про криптовалюту: {e}")
            self.conn.rollback()
            return False

    def get_klines(self, symbol, interval, start_time=None, end_time=None, limit=100):

        if symbol.upper() not in self.supported_symbols:
            print(f"Валюта {symbol} не підтримується")
            return pd.DataFrame()

        table_name = f"{symbol.lower()}_klines"

        query = f'''
        SELECT * FROM {table_name} 
        WHERE interval = %s
        '''
        params = [interval]

        if start_time:
            query += ' AND open_time >= %s'
            params.append(start_time)
        if end_time:
            query += ' AND open_time <= %s'
            params.append(end_time)

        query += ' ORDER BY open_time DESC LIMIT %s'
        params.append(limit)

        try:
            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            return df
        except psycopg2.Error as e:
            print(f"Помилка отримання свічок для {symbol}: {e}")
            return pd.DataFrame()

    def get_orderbook(self, symbol, timestamp=None, limit=100):

        if symbol.upper() not in self.supported_symbols:
            print(f"Валюта {symbol} не підтримується")
            return pd.DataFrame()

        table_name = f"{symbol.lower()}_orderbook"

        query = f'''
        SELECT * FROM {table_name}
        '''
        params = []

        if timestamp:
            query += ' WHERE timestamp = %s'
            params.append(timestamp)
        else:
            query += f' WHERE timestamp = (SELECT MAX(timestamp) FROM {table_name})'

        query += ' ORDER BY type, price LIMIT %s'
        params.append(limit)

        try:
            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            return df
        except psycopg2.Error as e:
            print(f"Помилка отримання книги ордерів для {symbol}: {e}")
            return pd.DataFrame()

    def log_event(self, log_level, message, component):

        try:
            # Перевіряємо чи існує таблиця логів, якщо ні - створюємо
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id SERIAL PRIMARY KEY,
                log_level TEXT NOT NULL,
                message TEXT NOT NULL,
                component TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')

            self.cursor.execute('''
            INSERT INTO logs (log_level, message, component)
            VALUES (%s, %s, %s)
            ''', (log_level, message, component))
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            print(f"Помилка додавання запису в лог: {e}")
            self.conn.rollback()
            return False

    def update_websocket_status(self, symbol, socket_type, interval, is_active):

        try:
            # Перевіряємо чи існує таблиця websocket_status, якщо ні - створюємо
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS websocket_status (
                id SERIAL PRIMARY KEY,
                symbol TEXT NOT NULL,
                socket_type TEXT NOT NULL,
                interval TEXT,
                is_active BOOLEAN NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')

            # Перевіряємо, чи існує вже запис з такими параметрами
            query = '''
            SELECT id FROM websocket_status 
            WHERE symbol = %s AND socket_type = %s
            '''
            params = [symbol, socket_type]

            if interval:
                query += ' AND interval = %s'
                params.append(interval)
            else:
                query += ' AND interval IS NULL'

            self.cursor.execute(query, params)
            result = self.cursor.fetchone()

            if result:
                # Оновлюємо існуючий запис
                update_query = '''
                UPDATE websocket_status 
                SET is_active = %s, last_updated = CURRENT_TIMESTAMP 
                WHERE id = %s
                '''
                self.cursor.execute(update_query, (is_active, result['id']))
            else:
                # Додаємо новий запис
                insert_query = '''
                INSERT INTO websocket_status (symbol, socket_type, interval, is_active)
                VALUES (%s, %s, %s, %s)
                '''
                self.cursor.execute(insert_query, (symbol, socket_type, interval, is_active))

            self.conn.commit()
            return True
        except psycopg2.Error as e:
            print(f"Помилка оновлення статусу WebSocket: {e}")
            self.conn.rollback()
            return False

    def execute_query(self, query, params=None):

        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)

            if query.strip().upper().startswith(('SELECT', 'SHOW', 'EXPLAIN')):
                rows = self.cursor.fetchall()
                return rows
            else:
                self.conn.commit()
                return True
        except psycopg2.Error as e:
            print(f"Помилка виконання запиту: {e}")
            self.conn.rollback()
            return False