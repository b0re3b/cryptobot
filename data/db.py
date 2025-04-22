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
        # Базові таблиці для BTC, ETH, SOL
        self._create_base_tables()
        # Таблиці для оброблених даних
        self._create_processed_tables()
        # Таблиці для профілів об'єму
        self._create_volume_profile_tables()
        # Таблиця для логування обробки даних
        self._create_data_processing_log_table()

    def _create_base_tables(self):
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

    def _create_processed_tables(self):
        # Таблиці для оброблених даних BTC
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS btc_klines_processed (
            id SERIAL PRIMARY KEY,
            interval TEXT NOT NULL,
            open_time TIMESTAMP NOT NULL,
            open NUMERIC NOT NULL,
            high NUMERIC NOT NULL,
            low NUMERIC NOT NULL,
            close NUMERIC NOT NULL,
            volume NUMERIC NOT NULL,
            price_zscore NUMERIC,
            volume_zscore NUMERIC,
            volatility NUMERIC,
            trend NUMERIC,
            hour INTEGER,
            day_of_week INTEGER,
            is_weekend BOOLEAN,
            session TEXT,
            is_anomaly BOOLEAN DEFAULT FALSE,
            has_missing BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (interval, open_time)
        )
        ''')

        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_btc_klines_processed_time ON btc_klines_processed(interval, open_time)')

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS btc_orderbook_processed (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            spread NUMERIC,
            imbalance NUMERIC,
            bid_volume NUMERIC,
            ask_volume NUMERIC,
            average_bid_price NUMERIC,
            average_ask_price NUMERIC,
            volatility_estimate NUMERIC,
            is_anomaly BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (timestamp)
        )
        ''')

        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_btc_orderbook_processed_time ON btc_orderbook_processed(timestamp)')

        # Таблиці для оброблених даних ETH
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS eth_klines_processed (
            id SERIAL PRIMARY KEY,
            interval TEXT NOT NULL,
            open_time TIMESTAMP NOT NULL,
            open NUMERIC NOT NULL,
            high NUMERIC NOT NULL,
            low NUMERIC NOT NULL,
            close NUMERIC NOT NULL,
            volume NUMERIC NOT NULL,
            price_zscore NUMERIC,
            volume_zscore NUMERIC,
            volatility NUMERIC,
            trend NUMERIC,
            hour INTEGER,
            day_of_week INTEGER,
            is_weekend BOOLEAN,
            session TEXT,
            is_anomaly BOOLEAN DEFAULT FALSE,
            has_missing BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (interval, open_time)
        )
        ''')

        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_eth_klines_processed_time ON eth_klines_processed(interval, open_time)')

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS eth_orderbook_processed (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            spread NUMERIC,
            imbalance NUMERIC,
            bid_volume NUMERIC,
            ask_volume NUMERIC,
            average_bid_price NUMERIC,
            average_ask_price NUMERIC,
            volatility_estimate NUMERIC,
            is_anomaly BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (timestamp)
        )
        ''')

        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_eth_orderbook_processed_time ON eth_orderbook_processed(timestamp)')

        # Таблиці для оброблених даних SOL
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS sol_klines_processed (
            id SERIAL PRIMARY KEY,
            interval TEXT NOT NULL,
            open_time TIMESTAMP NOT NULL,
            open NUMERIC NOT NULL,
            high NUMERIC NOT NULL,
            low NUMERIC NOT NULL,
            close NUMERIC NOT NULL,
            volume NUMERIC NOT NULL,
            price_zscore NUMERIC,
            volume_zscore NUMERIC,
            volatility NUMERIC,
            trend NUMERIC,
            hour INTEGER,
            day_of_week INTEGER,
            is_weekend BOOLEAN,
            session TEXT,
            is_anomaly BOOLEAN DEFAULT FALSE,
            has_missing BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (interval, open_time)
        )
        ''')

        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_sol_klines_processed_time ON sol_klines_processed(interval, open_time)')

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS sol_orderbook_processed (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            spread NUMERIC,
            imbalance NUMERIC,
            bid_volume NUMERIC,
            ask_volume NUMERIC,
            average_bid_price NUMERIC,
            average_ask_price NUMERIC,
            volatility_estimate NUMERIC,
            is_anomaly BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (timestamp)
        )
        ''')

        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_sol_orderbook_processed_time ON sol_orderbook_processed(timestamp)')

    def _create_volume_profile_tables(self):
        # Таблиці для профілів об'єму BTC
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS btc_volume_profile (
            id SERIAL PRIMARY KEY,
            interval TEXT NOT NULL,
            time_bucket TIMESTAMP NOT NULL,
            price_bin_start NUMERIC NOT NULL,
            price_bin_end NUMERIC NOT NULL,
            volume NUMERIC NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (interval, time_bucket, price_bin_start)
        )
        ''')

        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_btc_volume_profile ON btc_volume_profile(interval, time_bucket)')

        # Таблиці для профілів об'єму ETH
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS eth_volume_profile (
            id SERIAL PRIMARY KEY,
            interval TEXT NOT NULL,
            time_bucket TIMESTAMP NOT NULL,
            price_bin_start NUMERIC NOT NULL,
            price_bin_end NUMERIC NOT NULL,
            volume NUMERIC NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (interval, time_bucket, price_bin_start)
        )
        ''')

        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_eth_volume_profile ON eth_volume_profile(interval, time_bucket)')

        # Таблиці для профілів об'єму SOL
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS sol_volume_profile (
            id SERIAL PRIMARY KEY,
            interval TEXT NOT NULL,
            time_bucket TIMESTAMP NOT NULL,
            price_bin_start NUMERIC NOT NULL,
            price_bin_end NUMERIC NOT NULL,
            volume NUMERIC NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (interval, time_bucket, price_bin_start)
        )
        ''')

        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_sol_volume_profile ON sol_volume_profile(interval, time_bucket)')

    def _create_data_processing_log_table(self):
        # Таблиця для логування обробки даних
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS data_processing_log (
            id SERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            data_type TEXT NOT NULL,
            interval TEXT,
            start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP NOT NULL, 
            status TEXT NOT NULL,
            steps TEXT,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

    def insert_kline(self, symbol, kline_data):
        """Додає свічку до відповідної таблиці для валюти"""
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
        """Додає запис книги ордерів до відповідної таблиці для валюти"""
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
        """Додає інформацію про криптовалюту"""
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
        """Отримує свічки для валюти"""
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
        """Отримує книгу ордерів для валюти"""
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
        """Додає запис в лог"""
        try:
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
        """Оновлює статус веб-сокета"""
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
                update_query = '''
                UPDATE websocket_status 
                SET is_active = %s, last_updated = CURRENT_TIMESTAMP 
                WHERE id = %s
                '''
                self.cursor.execute(update_query, (is_active, result['id']))
            else:
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

    # Нові методи для таблиць оброблених даних
    def insert_kline_processed(self, symbol, processed_data):
        """Додає оброблену свічку до відповідної таблиці"""
        if symbol.upper() not in self.supported_symbols:
            print(f"Валюта {symbol} не підтримується")
            return False

        table_name = f"{symbol.lower()}_klines_processed"

        try:
            self.cursor.execute(f'''
            INSERT INTO {table_name} 
            (interval, open_time, open, high, low, close, volume, 
            price_zscore, volume_zscore, volatility, trend, 
            hour, day_of_week, is_weekend, session, is_anomaly, has_missing)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (interval, open_time) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            price_zscore = EXCLUDED.price_zscore,
            volume_zscore = EXCLUDED.volume_zscore,
            volatility = EXCLUDED.volatility,
            trend = EXCLUDED.trend,
            hour = EXCLUDED.hour,
            day_of_week = EXCLUDED.day_of_week,
            is_weekend = EXCLUDED.is_weekend,
            session = EXCLUDED.session,
            is_anomaly = EXCLUDED.is_anomaly,
            has_missing = EXCLUDED.has_missing
            ''', (
                processed_data['interval'],
                processed_data['open_time'],
                processed_data['open'],
                processed_data['high'],
                processed_data['low'],
                processed_data['close'],
                processed_data['volume'],
                processed_data.get('price_zscore'),
                processed_data.get('volume_zscore'),
                processed_data.get('volatility'),
                processed_data.get('trend'),
                processed_data.get('hour'),
                processed_data.get('day_of_week'),
                processed_data.get('is_weekend'),
                processed_data.get('session'),
                processed_data.get('is_anomaly', False),
                processed_data.get('has_missing', False)
            ))
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            print(f"Помилка додавання обробленої свічки для {symbol}: {e}")
            self.conn.rollback()
            return False

    def insert_orderbook_processed(self, symbol, processed_data):
        """Додає оброблений запис книги ордерів до відповідної таблиці"""
        if symbol.upper() not in self.supported_symbols:
            print(f"Валюта {symbol} не підтримується")
            return False

        table_name = f"{symbol.lower()}_orderbook_processed"

        try:
            self.cursor.execute(f'''
            INSERT INTO {table_name} 
            (timestamp, spread, imbalance, bid_volume, ask_volume, 
            average_bid_price, average_ask_price, volatility_estimate, is_anomaly)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (timestamp) DO UPDATE SET
            spread = EXCLUDED.spread,
            imbalance = EXCLUDED.imbalance,
            bid_volume = EXCLUDED.bid_volume,
            ask_volume = EXCLUDED.ask_volume,
            average_bid_price = EXCLUDED.average_bid_price,
            average_ask_price = EXCLUDED.average_ask_price,
            volatility_estimate = EXCLUDED.volatility_estimate,
            is_anomaly = EXCLUDED.is_anomaly
            ''', (
                processed_data['timestamp'],
                processed_data['spread'],
                processed_data['imbalance'],
                processed_data['bid_volume'],
                processed_data['ask_volume'],
                processed_data['average_bid_price'],
                processed_data['average_ask_price'],
                processed_data.get('volatility_estimate'),
                processed_data.get('is_anomaly', False)
            ))
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            print(f"Помилка додавання обробленого запису книги ордерів для {symbol}: {e}")
            self.conn.rollback()
            return False

    def insert_volume_profile(self, symbol, profile_data):
        """Додає запис профілю об'єму до відповідної таблиці"""
        if symbol.upper() not in self.supported_symbols:
            print(f"Валюта {symbol} не підтримується")
            return False

        table_name = f"{symbol.lower()}_volume_profile"

        try:
            self.cursor.execute(f'''
            INSERT INTO {table_name} 
            (interval, time_bucket, price_bin_start, price_bin_end, volume)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (interval, time_bucket, price_bin_start) DO UPDATE SET
            price_bin_end = EXCLUDED.price_bin_end,
            volume = EXCLUDED.volume
            ''', (
                profile_data['interval'],
                profile_data['time_bucket'],
                profile_data['price_bin_start'],
                profile_data['price_bin_end'],
                profile_data['volume']
            ))
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            print(f"Помилка додавання запису профілю об'єму для {symbol}: {e}")
            self.conn.rollback()
            return False

    def log_data_processing(self, symbol, data_type, interval, start_time, end_time, status, steps=None,
                            error_message=None):
        """Додає запис про обробку даних до логу"""
        try:
            self.cursor.execute('''
                                INSERT INTO data_processing_log
                                (symbol, data_type, interval, start_time, end_time, status, steps, error_message)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                                ''', (symbol, data_type, interval, start_time, end_time, status, steps, error_message))
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            print(f"Помилка додавання запису про обробку даних: {e}")
            self.conn.rollback()
            return False

    def get_klines_processed(self, symbol, interval, start_time=None, end_time=None, limit=100):
        """Отримує оброблені свічки для валюти"""
        if symbol.upper() not in self.supported_symbols:
            print(f"Валюта {symbol} не підтримується")
            return pd.DataFrame()

        table_name = f"{symbol.lower()}_klines_processed"

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
            print(f"Помилка отримання оброблених свічок для {symbol}: {e}")
            return pd.DataFrame()

    def get_orderbook_processed(self, symbol, timestamp=None, limit=100):
        """Отримує оброблену книгу ордерів для валюти"""
        if symbol.upper() not in self.supported_symbols:
            print(f"Валюта {symbol} не підтримується")
            return pd.DataFrame()

        table_name = f"{symbol.lower()}_orderbook_processed"

        query = f'''
        SELECT * FROM {table_name}
        '''
        params = []

        if timestamp:
            query += ' WHERE timestamp = %s'
            params.append(timestamp)
        else:
            query += f' WHERE timestamp = (SELECT MAX(timestamp) FROM {table_name})'

        query += ' ORDER BY timestamp DESC LIMIT %s'
        params.append(limit)

        try:
            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            return df
        except psycopg2.Error as e:
            print(f"Помилка отримання обробленої книги ордерів для {symbol}: {e}")
            return pd.DataFrame()

    def get_volume_profile(self, symbol, interval, time_bucket=None, limit=100):
        """Отримує профіль об'єму для валюти"""
        if symbol.upper() not in self.supported_symbols:
            print(f"Валюта {symbol} не підтримується")
            return pd.DataFrame()

        table_name = f"{symbol.lower()}_volume_profile"

        query = f'''
        SELECT * FROM {table_name}
        WHERE interval = %s
        '''
        params = [interval]

        if time_bucket:
            query += ' AND time_bucket = %s'
            params.append(time_bucket)

        query += ' ORDER BY time_bucket DESC, price_bin_start ASC LIMIT %s'
        params.append(limit)

        try:
            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            return df
        except psycopg2.Error as e:
            print(f"Помилка отримання профілю об'єму для {symbol}: {e}")
            return pd.DataFrame()

    def delete_old_data(self, symbol, table_type, days_to_keep=30):
        """Видаляє старі дані з бази даних"""
        if symbol.upper() not in self.supported_symbols:
            print(f"Валюта {symbol} не підтримується")
            return False

        # Визначаємо таблицю та поле з часом в залежності від типу
        if table_type == 'klines':
            table_name = f"{symbol.lower()}_klines"
            time_field = 'open_time'
        elif table_type == 'klines_processed':
            table_name = f"{symbol.lower()}_klines_processed"
            time_field = 'open_time'
        elif table_type == 'orderbook':
            table_name = f"{symbol.lower()}_orderbook"
            time_field = 'timestamp'
        elif table_type == 'orderbook_processed':
            table_name = f"{symbol.lower()}_orderbook_processed"
            time_field = 'timestamp'
        elif table_type == 'volume_profile':
            table_name = f"{symbol.lower()}_volume_profile"
            time_field = 'time_bucket'
        else:
            print(f"Непідтримуваний тип таблиці: {table_type}")
            return False

        try:
            self.cursor.execute(f'''
            DELETE FROM {table_name}
            WHERE {time_field} < NOW() - INTERVAL '{days_to_keep} days'
            ''')
            rows_deleted = self.cursor.rowcount
            self.conn.commit()
            print(f"Видалено {rows_deleted} записів з {table_name}")
            return True
        except psycopg2.Error as e:
            print(f"Помилка видалення даних з {table_name}: {e}")
            self.conn.rollback()
            return False

    def get_anomalies(self, symbol, data_type, start_time=None, end_time=None, limit=100):
        """Отримує аномалії з оброблених даних"""
        if symbol.upper() not in self.supported_symbols:
            print(f"Валюта {symbol} не підтримується")
            return pd.DataFrame()

        if data_type == 'klines':
            table_name = f"{symbol.lower()}_klines_processed"
            time_field = 'open_time'
        elif data_type == 'orderbook':
            table_name = f"{symbol.lower()}_orderbook_processed"
            time_field = 'timestamp'
        else:
            print(f"Непідтримуваний тип даних: {data_type}")
            return pd.DataFrame()

        query = f'''
        SELECT * FROM {table_name}
        WHERE is_anomaly = TRUE
        '''
        params = []

        if start_time:
            query += f' AND {time_field} >= %s'
            params.append(start_time)
        if end_time:
            query += f' AND {time_field} <= %s'
            params.append(end_time)

        query += f' ORDER BY {time_field} DESC LIMIT %s'
        params.append(limit)

        try:
            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            return df
        except psycopg2.Error as e:
            print(f"Помилка отримання аномалій для {symbol}: {e}")
            return pd.DataFrame()

    def execute_raw_query(self, query, params=None):
        """Виконує довільний SQL-запит"""
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)

            # Якщо запит повертає дані
            if self.cursor.description:
                rows = self.cursor.fetchall()
                return pd.DataFrame(rows) if rows else pd.DataFrame()

            self.conn.commit()
            return True
        except psycopg2.Error as e:
            print(f"Помилка виконання запиту: {e}")
            self.conn.rollback()
            return False