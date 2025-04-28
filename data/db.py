import os
import psycopg2
import pandas as pd
from datetime import datetime
from psycopg2.extras import RealDictCursor
from utils.config import *
import json
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
        """Створює схему бази даних без конфліктів тригерів"""
        try:
            schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')

            if os.path.exists(schema_path):
                with open(schema_path, 'r') as f:
                    schema_script = f.read()

                # Спочатку обробимо тригери: знайдемо їх через простий пошук
                import re
                trigger_statements = re.findall(
                    r'CREATE TRIGGER\s+(\w+)\s+BEFORE\s+UPDATE\s+ON\s+(\w+)\s+FOR EACH ROW\s+EXECUTE FUNCTION\s+(\w+\(\));',
                    schema_script,
                    re.IGNORECASE
                )

                for trigger_name, table_name, function_name in trigger_statements:
                    drop_trigger_sql = f"DROP TRIGGER IF EXISTS {trigger_name} ON {table_name};"
                    self.cursor.execute(drop_trigger_sql)

                # Після дропу всіх тригерів виконаємо повністю schema.sql
                self.cursor.execute(schema_script)

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

    def get_klines(self, symbol, interval, start_time=None, end_time=None, limit=None):
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

    def get_orderbook(self, symbol, start_time=None, end_time=None, limit=None):
        """Отримує книгу ордерів для валюти за діапазоном часу"""
        if symbol.upper() not in self.supported_symbols:
            print(f"Валюта {symbol} не підтримується")
            return pd.DataFrame()

        table_name = f"{symbol.lower()}_orderbook"

        query = f'SELECT * FROM {table_name} WHERE TRUE'
        params = []

        if start_time:
            query += ' AND timestamp >= %s'
            params.append(start_time)

        if end_time:
            query += ' AND timestamp <= %s'
            params.append(end_time)

        query += ' ORDER BY timestamp DESC, type, price LIMIT %s'
        params.append(limit)

        try:
            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            return df
        except psycopg2.Error as e:
            print(f"❌ Помилка отримання книги ордерів для {symbol}: {e}")
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

    # Функції для роботи з твітами
    def insert_tweet(self, tweet_data):
        """Додає новий твіт до бази даних"""
        try:
            self.cursor.execute('''
                                INSERT INTO tweets_raw
                                (tweet_id, author_id, author_username, content, created_at, likes_count,
                                 retweets_count, quotes_count, replies_count, language, hashtags, mentioned_cryptos,
                                 tweet_url)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (tweet_id) DO
                                UPDATE SET
                                    likes_count = EXCLUDED.likes_count,
                                    retweets_count = EXCLUDED.retweets_count,
                                    quotes_count = EXCLUDED.quotes_count,
                                    replies_count = EXCLUDED.replies_count,
                                    collected_at = CURRENT_TIMESTAMP
                                ''', (
                                    tweet_data['tweet_id'],
                                    tweet_data['author_id'],
                                    tweet_data['author_username'],
                                    tweet_data['content'],
                                    tweet_data['created_at'],
                                    tweet_data['likes_count'],
                                    tweet_data['retweets_count'],
                                    tweet_data.get('quotes_count'),
                                    tweet_data.get('replies_count'),
                                    tweet_data['language'],
                                    tweet_data.get('hashtags'),
                                    tweet_data.get('mentioned_cryptos'),
                                    tweet_data.get('tweet_url')
                                ))
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            print(f"Помилка додавання твіту: {e}")
            self.conn.rollback()
            return False

    def get_tweets(self, filters=None, limit=100):
        """Отримує твіти з бази даних за заданими фільтрами"""
        query = 'SELECT * FROM tweets_raw WHERE TRUE'
        params = []

        if filters:
            if 'author_username' in filters:
                query += ' AND author_username = %s'
                params.append(filters['author_username'])
            if 'start_date' in filters:
                query += ' AND created_at >= %s'
                params.append(filters['start_date'])
            if 'end_date' in filters:
                query += ' AND created_at <= %s'
                params.append(filters['end_date'])
            if 'crypto' in filters:
                query += ' AND %s = ANY(mentioned_cryptos)'
                params.append(filters['crypto'])
            if 'hashtag' in filters:
                query += ' AND %s = ANY(hashtags)'
                params.append(filters['hashtag'])

        query += ' ORDER BY created_at DESC LIMIT %s'
        params.append(limit)

        try:
            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            return df
        except psycopg2.Error as e:
            print(f"Помилка отримання твітів: {e}")
            return pd.DataFrame()

    # Функції для роботи з настроями твітів
    def insert_tweet_sentiment(self, sentiment_data):
        """Додає аналіз настрою твіту"""
        try:
            self.cursor.execute('''
                                INSERT INTO tweet_sentiments
                                    (tweet_id, sentiment, sentiment_score, confidence, model_used)
                                VALUES (%s, %s, %s, %s, %s) ON CONFLICT (tweet_id) DO
                                UPDATE SET
                                    sentiment = EXCLUDED.sentiment,
                                    sentiment_score = EXCLUDED.sentiment_score,
                                    confidence = EXCLUDED.confidence,
                                    model_used = EXCLUDED.model_used,
                                    analyzed_at = CURRENT_TIMESTAMP
                                ''', (
                                    sentiment_data['tweet_id'],
                                    sentiment_data['sentiment'],
                                    sentiment_data['sentiment_score'],
                                    sentiment_data['confidence'],
                                    sentiment_data['model_used']
                                ))
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            print(f"Помилка додавання аналізу настрою твіту: {e}")
            self.conn.rollback()
            return False

    def get_tweet_sentiments(self, tweet_ids=None, limit=100):
        """Отримує аналіз настроїв твітів"""
        query = '''
                SELECT ts.*, tr.content, tr.author_username, tr.created_at
                FROM tweet_sentiments ts
                         JOIN tweets_raw tr ON ts.tweet_id = tr.tweet_id \
                '''
        params = []

        if tweet_ids:
            if isinstance(tweet_ids, list):
                query += ' WHERE ts.tweet_id = ANY(%s)'
                params.append(tweet_ids)
            else:
                query += ' WHERE ts.tweet_id = %s'
                params.append(tweet_ids)

        query += ' ORDER BY ts.analyzed_at DESC LIMIT %s'
        params.append(limit)

        try:
            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            return df
        except psycopg2.Error as e:
            print(f"Помилка отримання аналізу настроїв твітів: {e}")
            return pd.DataFrame()

    # Функції для роботи з кешем запитів Twitter
    def insert_twitter_query_cache(self, cache_data):
        """Додає кеш запиту Twitter"""
        try:
            self.cursor.execute('''
                                INSERT INTO twitter_query_cache
                                    (query, search_params, cache_expires_at, results_count)
                                VALUES (%s, %s, %s, %s) ON CONFLICT (query, search_params) DO
                                UPDATE SET
                                    cache_expires_at = EXCLUDED.cache_expires_at,
                                    results_count = EXCLUDED.results_count,
                                    created_at = CURRENT_TIMESTAMP
                                ''', (
                                    cache_data['query'],
                                    json.dumps(cache_data['search_params']),
                                    cache_data['cache_expires_at'],
                                    cache_data['results_count']
                                ))
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            print(f"Помилка додавання кешу запиту Twitter: {e}")
            self.conn.rollback()
            return False

    def get_twitter_query_cache(self, query, search_params):
        """Отримує кеш запиту Twitter"""
        try:
            self.cursor.execute('''
                                SELECT *
                                FROM twitter_query_cache
                                WHERE query = %s
                                  AND search_params = %s
                                  AND cache_expires_at > CURRENT_TIMESTAMP
                                ''', (query, json.dumps(search_params)))

            result = self.cursor.fetchone()
            return result
        except psycopg2.Error as e:
            print(f"Помилка отримання кешу запиту Twitter: {e}")
            return None

    def clean_expired_twitter_cache(self):
        """Видаляє застарілі кеші запитів Twitter"""
        try:
            self.cursor.execute('''
                                DELETE
                                FROM twitter_query_cache
                                WHERE cache_expires_at < CURRENT_TIMESTAMP
                                ''')
            deleted_count = self.cursor.rowcount
            self.conn.commit()
            return deleted_count
        except psycopg2.Error as e:
            print(f"Помилка видалення застарілих кешів Twitter: {e}")
            self.conn.rollback()
            return 0

    # Функції для роботи з крипто-інфлюенсерами
    def insert_crypto_influencer(self, influencer_data):
        """Додає або оновлює інформацію про крипто-інфлюенсера"""
        try:
            self.cursor.execute('''
                                INSERT INTO crypto_influencers
                                (username, display_name, description, followers_count, following_count,
                                 tweet_count, verified, influence_score, crypto_topics)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (username) DO
                                UPDATE SET
                                    display_name = EXCLUDED.display_name,
                                    description = EXCLUDED.description,
                                    followers_count = EXCLUDED.followers_count,
                                    following_count = EXCLUDED.following_count,
                                    tweet_count = EXCLUDED.tweet_count,
                                    verified = EXCLUDED.verified,
                                    influence_score = EXCLUDED.influence_score,
                                    crypto_topics = EXCLUDED.crypto_topics,
                                    updated_at = CURRENT_TIMESTAMP
                                ''', (
                                    influencer_data['username'],
                                    influencer_data.get('display_name'),
                                    influencer_data.get('description'),
                                    influencer_data.get('followers_count'),
                                    influencer_data.get('following_count'),
                                    influencer_data.get('tweet_count'),
                                    influencer_data.get('verified', False),
                                    influencer_data.get('influence_score'),
                                    influencer_data.get('crypto_topics')
                                ))
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            print(f"Помилка додавання крипто-інфлюенсера: {e}")
            self.conn.rollback()
            return False

    def get_crypto_influencers(self, min_influence_score=None, topics=None, limit=100):
        """Отримує інформацію про крипто-інфлюенсерів за заданими фільтрами"""
        query = 'SELECT * FROM crypto_influencers WHERE TRUE'
        params = []

        if min_influence_score is not None:
            query += ' AND influence_score >= %s'
            params.append(min_influence_score)

        if topics:
            if isinstance(topics, list):
                query += ' AND crypto_topics && %s'  # Перевірка перетину масивів
                params.append(topics)
            else:
                query += ' AND %s = ANY(crypto_topics)'
                params.append(topics)

        query += ' ORDER BY influence_score DESC LIMIT %s'
        params.append(limit)

        try:
            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            return df
        except psycopg2.Error as e:
            print(f"Помилка отримання крипто-інфлюенсерів: {e}")
            return pd.DataFrame()

    # Функції для роботи з активністю інфлюенсерів
    def insert_influencer_activity(self, activity_data):
        """Додає запис про активність інфлюенсера"""
        try:
            self.cursor.execute('''
                                INSERT INTO influencer_activity
                                    (influencer_id, tweet_id, impact_score)
                                VALUES (%s, %s, %s) ON CONFLICT (influencer_id, tweet_id) DO
                                UPDATE SET
                                    impact_score = EXCLUDED.impact_score
                                ''', (
                                    activity_data['influencer_id'],
                                    activity_data['tweet_id'],
                                    activity_data.get('impact_score')
                                ))
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            print(f"Помилка додавання активності інфлюенсера: {e}")
            self.conn.rollback()
            return False

    def get_influencer_activity(self, influencer_id=None, start_date=None, end_date=None, limit=100):
        """Отримує активність інфлюенсерів"""
        query = '''
                SELECT ia.*, ci.username, tr.content, tr.created_at
                FROM influencer_activity ia
                         JOIN crypto_influencers ci ON ia.influencer_id = ci.id
                         JOIN tweets_raw tr ON ia.tweet_id = tr.tweet_id
                WHERE TRUE \
                '''
        params = []

        if influencer_id:
            query += ' AND ia.influencer_id = %s'
            params.append(influencer_id)

        if start_date:
            query += ' AND tr.created_at >= %s'
            params.append(start_date)

        if end_date:
            query += ' AND tr.created_at <= %s'
            params.append(end_date)

        query += ' ORDER BY tr.created_at DESC LIMIT %s'
        params.append(limit)

        try:
            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            return df
        except psycopg2.Error as e:
            print(f"Помилка отримання активності інфлюенсерів: {e}")
            return pd.DataFrame()

    # Функції для роботи з криптовалютними подіями
    def insert_crypto_event(self, event_data):
        """Додає запис про криптовалютну подію"""
        try:
            self.cursor.execute('''
                                INSERT INTO crypto_events
                                (event_type, crypto_symbol, description, confidence_score, start_time, end_time,
                                 related_tweets)
                                VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id
                                ''', (
                                    event_data['event_type'],
                                    event_data['crypto_symbol'],
                                    event_data['description'],
                                    event_data['confidence_score'],
                                    event_data['start_time'],
                                    event_data.get('end_time'),
                                    event_data.get('related_tweets')
                                ))
            event_id = self.cursor.fetchone()['id']
            self.conn.commit()
            return event_id
        except psycopg2.Error as e:
            print(f"Помилка додавання криптовалютної події: {e}")
            self.conn.rollback()
            return None

    def update_crypto_event(self, event_id, event_data):
        """Оновлює запис про криптовалютну подію"""
        try:
            update_parts = []
            params = []

            for key, value in event_data.items():
                if key in ['event_type', 'crypto_symbol', 'description', 'confidence_score', 'start_time', 'end_time',
                           'related_tweets']:
                    update_parts.append(f"{key} = %s")
                    params.append(value)

            if not update_parts:
                return False

            params.append(event_id)  # Додаємо id в кінець для WHERE

            query = f'''
            UPDATE crypto_events
            SET {', '.join(update_parts)}
            WHERE id = %s
            '''

            self.cursor.execute(query, params)
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            print(f"Помилка оновлення криптовалютної події: {e}")
            self.conn.rollback()
            return False

    def get_crypto_events(self, crypto_symbol=None, event_type=None, start_time=None, end_time=None, limit=100):
        """Отримує записи про криптовалютні події за заданими фільтрами"""
        query = 'SELECT * FROM crypto_events WHERE TRUE'
        params = []

        if crypto_symbol:
            query += ' AND crypto_symbol = %s'
            params.append(crypto_symbol)

        if event_type:
            query += ' AND event_type = %s'
            params.append(event_type)

        if start_time:
            query += ' AND start_time >= %s'
            params.append(start_time)

        if end_time:
            query += ' AND start_time <= %s'
            params.append(end_time)

        query += ' ORDER BY start_time DESC LIMIT %s'
        params.append(limit)

        try:
            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            return df
        except psycopg2.Error as e:
            print(f"Помилка отримання криптовалютних подій: {e}")
            return pd.DataFrame()

    # Функції для роботи з часовими рядами настроїв
    def insert_sentiment_time_series(self, sentiment_data):
        """Додає запис до часового ряду настроїв"""
        try:
            self.cursor.execute('''
                                INSERT INTO sentiment_time_series
                                (crypto_symbol, time_bucket, interval, positive_count, negative_count,
                                 neutral_count, average_sentiment, sentiment_volatility, tweet_volume)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s,
                                        %s) ON CONFLICT (crypto_symbol, time_bucket, interval) DO
                                UPDATE SET
                                    positive_count = EXCLUDED.positive_count,
                                    negative_count = EXCLUDED.negative_count,
                                    neutral_count = EXCLUDED.neutral_count,
                                    average_sentiment = EXCLUDED.average_sentiment,
                                    sentiment_volatility = EXCLUDED.sentiment_volatility,
                                    tweet_volume = EXCLUDED.tweet_volume
                                ''', (
                                    sentiment_data['crypto_symbol'],
                                    sentiment_data['time_bucket'],
                                    sentiment_data['interval'],
                                    sentiment_data['positive_count'],
                                    sentiment_data['negative_count'],
                                    sentiment_data['neutral_count'],
                                    sentiment_data['average_sentiment'],
                                    sentiment_data.get('sentiment_volatility'),
                                    sentiment_data['tweet_volume']
                                ))
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            print(f"Помилка додавання запису до часового ряду настроїв: {e}")
            self.conn.rollback()
            return False

    def get_sentiment_time_series(self, crypto_symbol, interval, start_time=None, end_time=None, limit=100):
        """Отримує часовий ряд настроїв для криптовалюти"""
        query = '''
                SELECT * \
                FROM sentiment_time_series
                WHERE crypto_symbol = %s
                  AND interval = %s \
                '''
        params = [crypto_symbol, interval]

        if start_time:
            query += ' AND time_bucket >= %s'
            params.append(start_time)

        if end_time:
            query += ' AND time_bucket <= %s'
            params.append(end_time)

        query += ' ORDER BY time_bucket ASC LIMIT %s'
        params.append(limit)

        try:
            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            return df
        except psycopg2.Error as e:
            print(f"Помилка отримання часового ряду настроїв: {e}")
            return pd.DataFrame()

    # Функції для логування помилок скрапінгу
    def insert_scraping_error(self, error_data):
        """Додає запис про помилку скрапінгу"""
        try:
            self.cursor.execute('''
                                INSERT INTO scraping_errors
                                    (error_type, error_message, query, retry_count)
                                VALUES (%s, %s, %s, %s)
                                ''', (
                                    error_data['error_type'],
                                    error_data['error_message'],
                                    error_data.get('query'),
                                    error_data.get('retry_count', 0)
                                ))
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            print(f"Помилка додавання запису про помилку скрапінгу: {e}")
            self.conn.rollback()
            return False

    def get_scraping_errors(self, error_type=None, start_time=None, end_time=None, limit=100):
        """Отримує записи про помилки скрапінгу за заданими фільтрами"""
        query = 'SELECT * FROM scraping_errors WHERE TRUE'
        params = []

        if error_type:
            query += ' AND error_type = %s'
            params.append(error_type)

        if start_time:
            query += ' AND occurred_at >= %s'
            params.append(start_time)

        if end_time:
            query += ' AND occurred_at <= %s'
            params.append(end_time)

        query += ' ORDER BY occurred_at DESC LIMIT %s'
        params.append(limit)

        try:
            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            return df
        except psycopg2.Error as e:
            print(f"Помилка отримання записів про помилки скрапінгу: {e}")
            return pd.DataFrame()

    def insert_news_source(self, source_name, base_url, is_active=True):
        try:
            self.cursor.execute('''
                INSERT INTO news_sources (source_name, base_url, is_active)
                VALUES (%s, %s, %s)
                ON CONFLICT (source_name) DO NOTHING
            ''', (source_name, base_url, is_active))
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            print(f"Помилка додавання джерела новин: {e}")
            self.conn.rollback()
            return False

    def get_news_sources(self, only_active=True):
        try:
            query = 'SELECT * FROM news_sources'
            if only_active:
                query += ' WHERE is_active = TRUE'
            self.cursor.execute(query)
            return pd.DataFrame(self.cursor.fetchall())
        except psycopg2.Error as e:
            print(f"Помилка отримання джерел новин: {e}")
            return pd.DataFrame()
    def insert_news_category(self, source_id, category_name, category_url_path=None, is_active=True):
        try:
            self.cursor.execute('''
                INSERT INTO news_categories (source_id, category_name, category_url_path, is_active)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (source_id, category_name) DO NOTHING
            ''', (source_id, category_name, category_url_path, is_active))
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            print(f"Помилка додавання категорії новин: {e}")
            self.conn.rollback()
            return False

    def get_news_categories(self, source_id=None, only_active=True):
        try:
            query = 'SELECT * FROM news_categories WHERE TRUE'
            params = []
            if source_id:
                query += ' AND source_id = %s'
                params.append(source_id)
            if only_active:
                query += ' AND is_active = TRUE'
            self.cursor.execute(query, params)
            return pd.DataFrame(self.cursor.fetchall())
        except psycopg2.Error as e:
            print(f"Помилка отримання категорій новин: {e}")
            return pd.DataFrame()
    def insert_news_article(self, article_data):
        try:
            self.cursor.execute('''
                INSERT INTO news_articles 
                (title, summary, content, link, source_id, category_id, published_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (link) DO NOTHING
            ''', (
                article_data['title'],
                article_data.get('summary'),
                article_data.get('content'),
                article_data['link'],
                article_data['source_id'],
                article_data['category_id'],
                article_data.get('published_at')
            ))
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            print(f"Помилка додавання новинної статті: {e}")
            self.conn.rollback()
            return False
    def insert_news_sentiment(self, sentiment_data):
        try:
            self.cursor.execute('''
                INSERT INTO news_sentiment_analysis 
                (article_id, sentiment_score, sentiment_magnitude, sentiment_label)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (article_id) DO UPDATE SET
                sentiment_score = EXCLUDED.sentiment_score,
                sentiment_magnitude = EXCLUDED.sentiment_magnitude,
                sentiment_label = EXCLUDED.sentiment_label,
                processed_at = CURRENT_TIMESTAMP
            ''', (
                sentiment_data['article_id'],
                sentiment_data['sentiment_score'],
                sentiment_data['sentiment_magnitude'],
                sentiment_data['sentiment_label']
            ))
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            print(f"Помилка додавання настрою новин: {e}")
            self.conn.rollback()
            return False
    def insert_article_mention(self, article_id, crypto_symbol, mention_count=1):
        try:
            self.cursor.execute('''
                INSERT INTO article_mentioned_coins (article_id, crypto_symbol, mention_count)
                VALUES (%s, %s, %s)
                ON CONFLICT (article_id, crypto_symbol) DO UPDATE SET
                mention_count = EXCLUDED.mention_count
            ''', (article_id, crypto_symbol, mention_count))
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            print(f"Помилка згадки криптовалюти в статті: {e}")
            self.conn.rollback()
            return False
    def insert_trending_topic(self, topic_data):
        try:
            self.cursor.execute('''
                INSERT INTO trending_news_topics 
                (topic_name, start_date, end_date, importance_score)
                VALUES (%s, %s, %s, %s)
            ''', (
                topic_data['topic_name'],
                topic_data.get('start_date'),
                topic_data.get('end_date'),
                topic_data.get('importance_score')
            ))
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            print(f"Помилка додавання трендової теми: {e}")
            self.conn.rollback()
            return False

    def link_article_to_topic(self, article_id, topic_id):
        try:
            self.cursor.execute('''
                INSERT INTO article_topics (article_id, topic_id)
                VALUES (%s, %s)
                ON CONFLICT (article_id, topic_id) DO NOTHING
            ''', (article_id, topic_id))
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            print(f"Помилка зв’язку статті з темою: {e}")
            self.conn.rollback()
            return False
    def insert_news_market_correlation(self, correlation_data):
        try:
            self.cursor.execute('''
                INSERT INTO news_market_correlations 
                (topic_id, crypto_symbol, time_period, correlation_coefficient, p_value, start_date, end_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            ''', (
                correlation_data['topic_id'],
                correlation_data['crypto_symbol'],
                correlation_data['time_period'],
                correlation_data['correlation_coefficient'],
                correlation_data['p_value'],
                correlation_data['start_date'],
                correlation_data['end_date']
            ))
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            print(f"Помилка додавання кореляції новин і ринку: {e}")
            self.conn.rollback()
            return False
    def insert_detected_news_event(self, event_data):
        try:
            self.cursor.execute('''
                INSERT INTO news_detected_events
                (event_title, event_description, crypto_symbols, source_articles, confidence_score, detected_at, expected_impact, event_category)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ''', (
                event_data['event_title'],
                event_data.get('event_description'),
                event_data['crypto_symbols'],
                event_data['source_articles'],
                event_data['confidence_score'],
                event_data['detected_at'],
                event_data['expected_impact'],
                event_data['event_category']
            ))
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            print(f"Помилка додавання події з новин: {e}")
            self.conn.rollback()
            return False
    def insert_news_sentiment_series(self, data):
        try:
            self.cursor.execute('''
                INSERT INTO news_sentiment_time_series 
                (crypto_symbol, time_bucket, interval, positive_count, negative_count, neutral_count, average_sentiment, news_volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (crypto_symbol, time_bucket, interval) DO UPDATE SET
                positive_count = EXCLUDED.positive_count,
                negative_count = EXCLUDED.negative_count,
                neutral_count = EXCLUDED.neutral_count,
                average_sentiment = EXCLUDED.average_sentiment,
                news_volume = EXCLUDED.news_volume
            ''', (
                data['crypto_symbol'],
                data['time_bucket'],
                data['interval'],
                data['positive_count'],
                data['negative_count'],
                data['neutral_count'],
                data['average_sentiment'],
                data['news_volume']
            ))
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            print(f"Помилка запису новинного настрою в часовому ряді: {e}")
            self.conn.rollback()
            return False
    def insert_news_scraping_log(self, log_data):
        try:
            self.cursor.execute('''
                INSERT INTO news_scraping_log 
                (source_id, category_id, start_time, end_time, articles_found, articles_processed, status, error_message)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ''', (
                log_data['source_id'],
                log_data['category_id'],
                log_data['start_time'],
                log_data['end_time'],
                log_data['articles_found'],
                log_data['articles_processed'],
                log_data['status'],
                log_data.get('error_message')
            ))
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            print(f"Помилка запису логу скрапінгу новин: {e}")
            self.conn.rollback()
            return False
