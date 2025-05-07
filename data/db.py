import os
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import psycopg2
import pandas as pd
from datetime import datetime
from psycopg2.extras import RealDictCursor, execute_batch
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

    def execute_many(self, query: str, params_list: list) -> None:

        with self.conn.cursor() as cursor:
            # Using psycopg2's execute_batch for efficient batch execution
            execute_batch(cursor, query, params_list)
            self.conn.commit()
    def fetch_one(self, query: str, params: tuple = ()) -> Optional[tuple]:
        """Виконує запит і повертає один результат"""
        with self.conn.cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchone()
    def fetch_all(self, query: str, params: tuple = ()) -> Optional[List[tuple]]:
        """Виконує запит і повертає всі результати"""
        with self.conn.cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()

    def execute_query(self, query: str, params: tuple = ()) -> None:
        """Виконує SQL запит без повернення результату"""
        with self.conn.cursor() as cursor:
            cursor.execute(query, params)
            self.conn.commit()

    def fetch_dict(self, query: str, params: tuple = ()) -> List[Dict]:
        """Виконує запит і повертає результати у вигляді словників"""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()
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
            timeframe TEXT NOT NULL,
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
            UNIQUE (timeframe, open_time)
        )
        ''')

        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_btc_klines_time ON btc_klines(timeframe, open_time)')




        # Таблиці для ETH
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS eth_klines (
            id SERIAL PRIMARY KEY,
            timeframe TEXT NOT NULL,
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
            UNIQUE (timeframe, open_time)
        )
        ''')

        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_eth_klines_time ON eth_klines(timeframe, open_time)')


        # Таблиці для SOL
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS sol_klines (
            id SERIAL PRIMARY KEY,
            timeframe TEXT NOT NULL,
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
            UNIQUE (timeframe, open_time)
        )
        ''')

        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_sol_klines_time ON sol_klines(timeframe, open_time)')



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
            timeframe TEXT NOT NULL,
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
            UNIQUE (timeframe, open_time)
        )
        ''')

        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_btc_klines_processed_time ON btc_klines_processed(timeframe, open_time)')




        # Таблиці для оброблених даних ETH
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS eth_klines_processed (
            id SERIAL PRIMARY KEY,
            timeframe TEXT NOT NULL,
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
            UNIQUE (timeframe, open_time)
        )
        ''')

        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_eth_klines_processed_time ON eth_klines_processed(timefarme, open_time)')



        # Таблиці для оброблених даних SOL
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS sol_klines_processed (
            id SERIAL PRIMARY KEY,
            timeframe TEXT NOT NULL,
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
            UNIQUE (timeframe, open_time)
        )
        ''')

        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_sol_klines_processed_time ON sol_klines_processed(timeframe, open_time)')

    def _create_volume_profile_tables(self):
        # Таблиці для профілів об'єму BTC
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS btc_volume_profile (
            id SERIAL PRIMARY KEY,
            timeframe TEXT NOT NULL,
            time_bucket TIMESTAMP NOT NULL,
            price_bin_start NUMERIC NOT NULL,
            price_bin_end NUMERIC NOT NULL,
            volume NUMERIC NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (timeframe, time_bucket, price_bin_start)
        )
        ''')

        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_btc_volume_profile ON btc_volume_profile(timeframe, time_bucket)')

        # Таблиці для профілів об'єму ETH
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS eth_volume_profile (
            id SERIAL PRIMARY KEY,
            timeframe TEXT NOT NULL,
            time_bucket TIMESTAMP NOT NULL,
            price_bin_start NUMERIC NOT NULL,
            price_bin_end NUMERIC NOT NULL,
            volume NUMERIC NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (timeframe, time_bucket, price_bin_start)
        )
        ''')

        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_eth_volume_profile ON eth_volume_profile(timeframe, time_bucket)')

        # Таблиці для профілів об'єму SOL
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS sol_volume_profile (
            id SERIAL PRIMARY KEY,
            timeframe TEXT NOT NULL,
            time_bucket TIMESTAMP NOT NULL,
            price_bin_start NUMERIC NOT NULL,
            price_bin_end NUMERIC NOT NULL,
            volume NUMERIC NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (timeframe, time_bucket, price_bin_start)
        )
        ''')

        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_sol_volume_profile ON sol_volume_profile(timeframe, time_bucket)')

    def _create_data_processing_log_table(self):
        # Таблиця для логування обробки даних
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS data_processing_log (
            id SERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            data_type TEXT NOT NULL,
            timeframe TEXT,
            start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP NOT NULL, 
            status TEXT NOT NULL,
            steps TEXT,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

    def insert_kline(self, symbol, kline_data,table_name):
        """Додає свічку до відповідної таблиці для валюти"""
        if symbol.upper() not in self.supported_symbols:
            print(f"Валюта {symbol} не підтримується")
            return False

        table_name = f"{symbol.lower()}_klines"

        try:
            self.cursor.execute(f'''
            INSERT INTO {table_name} 
            (timeframe, open_time, open, high, low, close, volume, close_time, 
            quote_asset_volume, number_of_trades, taker_buy_base_volume, taker_buy_quote_volume, is_closed)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (timeframe, open_time) DO UPDATE SET
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
                kline_data['timeframe'],
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

    def get_klines(self, symbol, timeframe, start_time=None, end_time=None, limit=None):
        """Отримує свічки для валюти"""
        if symbol.upper() not in self.supported_symbols:
            print(f"Валюта {symbol} не підтримується")
            return pd.DataFrame()

        table_name = f"{symbol.lower()}_klines"

        query = f'''
        SELECT * FROM {table_name} 
        WHERE timeframe = %s
        '''
        params = [timeframe]

        if start_time:
            query += ' AND open_time >= %s'
            params.append(start_time)
        if end_time:
            query += ' AND open_time <= %s'
            params.append(end_time)

        query += ' ORDER BY open_time DESC'  # ← сортування за часом

        if limit is not None:
            query += ' LIMIT %s'
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

    def update_websocket_status(self, symbol, socket_type, timeframe, is_active):
        """Оновлює статус веб-сокета"""
        try:
            # Перевіряємо чи існує таблиця websocket_status, якщо ні - створюємо
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS websocket_status (
                id SERIAL PRIMARY KEY,
                symbol TEXT NOT NULL,
                socket_type TEXT NOT NULL,
                timeframe TEXT,
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

            if timeframe:
                query += ' AND timeframe = %s'
                params.append(timeframe)
            else:
                query += ' AND timeframe IS NULL'

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
                INSERT INTO websocket_status (symbol, socket_type, timeframe, is_active)
                VALUES (%s, %s, %s, %s)
                '''
                self.cursor.execute(insert_query, (symbol, socket_type, timeframe, is_active))

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
            (timeframe, open_time, open, high, low, close, volume, 
            price_zscore, volume_zscore, volatility, trend, 
            hour, day_of_week, is_weekend, session, is_anomaly, has_missing)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (timeframe, open_time) DO UPDATE SET
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
                processed_data['timeframe'],
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



    def insert_volume_profile(self, symbol, profile_data):
        """Додає запис профілю об'єму до відповідної таблиці"""
        if symbol.upper() not in self.supported_symbols:
            print(f"Валюта {symbol} не підтримується")
            return False

        table_name = f"{symbol.lower()}_volume_profile"

        try:
            self.cursor.execute(f'''
            INSERT INTO {table_name} 
            (timeframe, time_bucket, price_bin_start, price_bin_end, volume)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (timeframe, time_bucket, price_bin_start) DO UPDATE SET
            price_bin_end = EXCLUDED.price_bin_end,
            volume = EXCLUDED.volume
            ''', (
                profile_data['timeframe'],
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

    def log_data_processing(self, symbol, data_type, timeframe, start_time, end_time, status, steps=None,
                            error_message=None):
        """Додає запис про обробку даних до логу"""
        try:
            self.cursor.execute('''
                                INSERT INTO data_processing_log
                                (symbol, data_type, timeframe, start_time, end_time, status, steps, error_message)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                                ''', (symbol, data_type, timeframe, start_time, end_time, status, steps, error_message))
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            print(f"Помилка додавання запису про обробку даних: {e}")
            self.conn.rollback()
            return False

    def get_klines_processed(self, symbol, timeframe, start_time=None, end_time=None, limit=100):
        """Отримує оброблені свічки для валюти"""
        if symbol.upper() not in self.supported_symbols:
            print(f"Валюта {symbol} не підтримується")
            return pd.DataFrame()

        table_name = f"{symbol.lower()}_klines_processed"

        query = f'''
        SELECT * FROM {table_name} 
        WHERE timeframe = %s
        '''
        params = [timeframe]

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



    def get_volume_profile(self, symbol, timeframe, time_bucket=None, limit=100):
        """Отримує профіль об'єму для валюти"""
        if symbol.upper() not in self.supported_symbols:
            print(f"Валюта {symbol} не підтримується")
            return pd.DataFrame()

        table_name = f"{symbol.lower()}_volume_profile"

        query = f'''
        SELECT * FROM {table_name}
        WHERE timeframe = %s
        '''
        params = [timeframe]

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

    def get_anomalies(self, symbol, data_type, start_time=None, end_time=None, limit=None):
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


    # Функції для роботи з часовими рядами настроїв
    def insert_sentiment_time_series(self, sentiment_data):
        """Додає запис до часового ряду настроїв"""
        try:
            self.cursor.execute('''
                                INSERT INTO sentiment_time_series
                                (symbol, time_bucket, timeframe, positive_count, negative_count,
                                 neutral_count, average_sentiment, sentiment_volatility, tweet_volume)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s,
                                        %s) ON CONFLICT (symbol, time_bucket, timeframe) DO
                                UPDATE SET
                                    positive_count = EXCLUDED.positive_count,
                                    negative_count = EXCLUDED.negative_count,
                                    neutral_count = EXCLUDED.neutral_count,
                                    average_sentiment = EXCLUDED.average_sentiment,
                                    sentiment_volatility = EXCLUDED.sentiment_volatility,
                                    tweet_volume = EXCLUDED.tweet_volume
                                ''', (
                                    sentiment_data['symbol'],
                                    sentiment_data['time_bucket'],
                                    sentiment_data['timeframe'],
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

    def get_sentiment_time_series(self, symbol, timeframe, start_time=None, end_time=None, limit=100):
        """Отримує часовий ряд настроїв для криптовалюти"""
        query = '''
                SELECT * \
                FROM sentiment_time_series
                WHERE symbol = %s
                  AND interval = %s \
                '''
        params = [symbol, timeframe]

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
                (topic_id, symbol, timeframe, correlation_coefficient, p_value, start_date, end_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            ''', (
                correlation_data['topic_id'],
                correlation_data['symbol'],
                correlation_data['timeframe'],
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
                (event_title, event_description, symbols, source_articles, confidence_score, detected_at, expected_impact, event_category)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ''', (
                event_data['event_title'],
                event_data.get('event_description'),
                event_data['symbols'],
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
                (symbol, time_bucket, timeframe, positive_count, negative_count, neutral_count, average_sentiment, news_volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (crypto_symbol, time_bucket, interval) DO UPDATE SET
                positive_count = EXCLUDED.positive_count,
                negative_count = EXCLUDED.negative_count,
                neutral_count = EXCLUDED.neutral_count,
                average_sentiment = EXCLUDED.average_sentiment,
                news_volume = EXCLUDED.news_volume
            ''', (
                data['symbol'],
                data['time_bucket'],
                data['timeframe'],
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

    def save_model_metadata(self, model_key: str, symbol: str, model_type: str,
                            timeframe: str, start_date: datetime, end_date: datetime,
                            description: str = None) -> int:

        try:
            conn = self.conn()
            with conn.cursor() as cursor:
                query = """
                        INSERT INTO time_series_models
                        (model_key, symbol, model_type, timeframe, start_date, end_date, description)
                        VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (model_key) 
                DO \
                        UPDATE SET
                            symbol = EXCLUDED.symbol, \
                            model_type = EXCLUDED.model_type, \
                            interval = EXCLUDED.interval, \
                            start_date = EXCLUDED.start_date, \
                            end_date = EXCLUDED.end_date, \
                            description = EXCLUDED.description, \
                            updated_at = CURRENT_TIMESTAMP \
                            RETURNING model_id \
                        """
                cursor.execute(query, (model_key, symbol, model_type, timeframe,
                                       start_date, end_date, description))
                model_id = cursor.fetchone()[0]
                conn.commit()
                return model_id
        except Exception as e:
            raise

    def save_model_parameters(self, model_id: int, parameters: Dict) -> bool:
        """
        Збереження параметрів моделі.

        Args:
            model_id: ID моделі
            parameters: Словник параметрів моделі

        Returns:
            Успішність операції
        """
        try:
            conn = self.conn()
            with conn.cursor() as cursor:
                for param_name, param_value in parameters.items():
                    query = """
                            INSERT INTO model_parameters
                                (model_id, param_name, param_value)
                            VALUES (%s, %s, %s) ON CONFLICT (model_id, param_name) 
                    DO \
                            UPDATE SET param_value = EXCLUDED.param_value \
                            """
                    # Перетворення значення параметра в JSON
                    param_json = json.dumps(param_value)
                    cursor.execute(query, (model_id, param_name, param_json))
                conn.commit()
                return True
        except Exception as e:
            return False

    def save_model_metrics(self, model_id: int, metrics: Dict, test_date: datetime = None) -> bool:
        """
        Збереження метрик ефективності моделі.

        Args:
            model_id: ID моделі
            metrics: Словник з метриками (назва: значення)
            test_date: Дата тестування моделі

        Returns:
            Успішність операції
        """
        try:
            conn = self.conn()
            with conn.cursor() as cursor:
                for metric_name, metric_value in metrics.items():
                    query = """
                            INSERT INTO model_metrics
                                (model_id, metric_name, metric_value, test_date)
                            VALUES (%s, %s, %s, %s) ON CONFLICT (model_id, metric_name, test_date) 
                    DO \
                            UPDATE SET metric_value = EXCLUDED.metric_value \
                            """
                    cursor.execute(query, (model_id, metric_name, float(metric_value), test_date))
                conn.commit()
                return True
        except Exception as e:
            return False

    def save_model_forecasts(self, model_id: int, forecasts: pd.Series,
                             lower_bounds: pd.Series = None, upper_bounds: pd.Series = None,
                             confidence_level: float = 0.95) -> bool:

        try:
            conn = self.conn()
            with conn.cursor() as cursor:
                for date, value in forecasts.items():
                    lower = None if lower_bounds is None else lower_bounds.get(date)
                    upper = None if upper_bounds is None else upper_bounds.get(date)

                    query = """
                            INSERT INTO model_forecasts
                            (model_id, forecast_date, forecast_value, lower_bound, upper_bound, confidence_level)
                            VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (model_id, forecast_date) 
                    DO \
                            UPDATE SET
                                forecast_value = EXCLUDED.forecast_value, \
                                lower_bound = EXCLUDED.lower_bound, \
                                upper_bound = EXCLUDED.upper_bound, \
                                confidence_level = EXCLUDED.confidence_level \
                            """
                    cursor.execute(query, (model_id, date, float(value),
                                           float(lower) if lower is not None else None,
                                           float(upper) if upper is not None else None,
                                           confidence_level))
                conn.commit()
                return True
        except Exception as e:

            return False

    def save_model_binary(self, model_id: int, model_obj: Any) -> bool:

        try:
            conn = self.conn()
            with conn.cursor() as cursor:
                # Серіалізація моделі
                model_binary = pickle.dumps(model_obj)

                query = """
                        INSERT INTO model_binary_data
                            (model_id, model_binary)
                        VALUES (%s, %s) ON CONFLICT (model_id) 
                DO \
                        UPDATE SET model_binary = EXCLUDED.model_binary \
                        """
                cursor.execute(query, (model_id, psycopg2.Binary(model_binary)))
                conn.commit()
                return True
        except Exception as e:
            return False

    def save_data_transformations(self, model_id: int, transformations: List[Dict]) -> bool:
        """
        Збереження інформації про перетворення даних.

        Args:
            model_id: ID моделі
            transformations: Список словників з інформацією про перетворення
                            [{'type': 'log', 'params': {}, 'order': 1}, ...]

        Returns:
            Успішність операції
        """
        try:
            conn = self.conn()
            with conn.cursor() as cursor:
                # Спочатку видаляємо старі записи для цієї моделі
                cursor.execute("DELETE FROM data_transformations WHERE model_id = %s", (model_id,))

                # Додаємо нові записи
                for transform in transformations:
                    query = """
                            INSERT INTO data_transformations
                                (model_id, transform_type, transform_params, transform_order)
                            VALUES (%s, %s, %s, %s) \
                            """
                    transform_type = transform.get('type')
                    transform_params = json.dumps(transform.get('params', {}))
                    transform_order = transform.get('order')

                    cursor.execute(query, (model_id, transform_type, transform_params, transform_order))

                conn.commit()
                return True
        except Exception as e:

            return False

    def get_model_by_key(self, model_key: str) -> Optional[Dict]:
        """
        Отримання інформації про модель за ключем.

        Args:
            model_key: Ключ моделі

        Returns:
            Словник з інформацією про модель або None
        """
        try:
            conn = self.conn()
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                query = "SELECT * FROM time_series_models WHERE model_key = %s"
                cursor.execute(query, (model_key,))
                model_data = cursor.fetchone()

                if model_data:
                    return dict(model_data)
                return None
        except Exception as e:
            return None

    def get_model_parameters(self, model_id: int) -> Dict:
        """
        Отримання параметрів моделі.

        Args:
            model_id: ID моделі

        Returns:
            Словник параметрів моделі
        """
        try:
            conn = self.conn()
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                query = "SELECT param_name, param_value FROM model_parameters WHERE model_id = %s"
                cursor.execute(query, (model_id,))
                parameters = {}

                for row in cursor.fetchall():
                    param_name = row['param_name']
                    param_value = json.loads(row['param_value'])
                    parameters[param_name] = param_value

                return parameters
        except Exception as e:
            return {}

    def get_model_metrics(self, model_id: int, test_date: datetime = None) -> Dict:
        """
        Отримання метрик ефективності моделі.

        Args:
            model_id: ID моделі
            test_date: Дата тестування (якщо None, повертаються всі метрики)

        Returns:
            Словник метрик моделі
        """
        try:
            conn = self.conn()
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                if test_date is None:
                    query = """SELECT metric_name, metric_value, test_date
                               FROM model_metrics
                               WHERE model_id = %s"""
                    cursor.execute(query, (model_id,))
                else:
                    query = """SELECT metric_name, metric_value
                               FROM model_metrics
                               WHERE model_id = %s \
                                 AND test_date = %s"""
                    cursor.execute(query, (model_id, test_date))

                metrics = {}
                for row in cursor.fetchall():
                    metric_name = row['metric_name']
                    metric_value = row['metric_value']

                    if test_date is None:
                        test_date_str = row['test_date'].strftime('%Y-%m-%d') if row['test_date'] else 'unknown'
                        if test_date_str not in metrics:
                            metrics[test_date_str] = {}
                        metrics[test_date_str][metric_name] = metric_value
                    else:
                        metrics[metric_name] = metric_value

                return metrics
        except Exception as e:
            return {}

    def get_model_forecasts(self, model_id: int, start_date: datetime = None,
                            end_date: datetime = None) -> pd.DataFrame:
        """
        Отримання прогнозів моделі.

        Args:
            model_id: ID моделі
            start_date: Початкова дата прогнозів (якщо None, від найранішого)
            end_date: Кінцева дата прогнозів (якщо None, до найпізнішого)

        Returns:
            DataFrame з прогнозами та довірчими інтервалами
        """
        try:
            conn = self.conn()
            query = """SELECT forecast_date, forecast_value, lower_bound, upper_bound, confidence_level
                       FROM model_forecasts
                       WHERE model_id = %s"""
            params = [model_id]

            if start_date:
                query += " AND forecast_date >= %s"
                params.append(start_date)
            if end_date:
                query += " AND forecast_date <= %s"
                params.append(end_date)

            query += " ORDER BY forecast_date"

            forecasts_df = pd.read_sql(query, conn, params=params, parse_dates=['forecast_date'])

            if not forecasts_df.empty:
                forecasts_df.set_index('forecast_date', inplace=True)

            return forecasts_df
        except Exception as e:
            return pd.DataFrame()

    def load_model_binary(self, model_id: int) -> Any:
        """
        Завантаження серіалізованої моделі з бази даних.

        Args:
            model_id: ID моделі

        Returns:
            Десеріалізований об'єкт моделі або None
        """
        try:
            conn = self.conn()
            with conn.cursor() as cursor:
                query = "SELECT model_binary FROM model_binary_data WHERE model_id = %s"
                cursor.execute(query, (model_id,))
                row = cursor.fetchone()

                if row:
                    model_binary = row[0]
                    model_obj = pickle.loads(model_binary)
                    return model_obj

                return None
        except Exception as e:
            return None

    def get_data_transformations(self, model_id: int) -> List[Dict]:
        """
        Отримання інформації про перетворення даних.

        Args:
            model_id: ID моделі

        Returns:
            Список словників з інформацією про перетворення
        """
        try:
            conn = self.conn()
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                query = """SELECT transform_type, transform_params, transform_order
                           FROM data_transformations
                           WHERE model_id = %s
                           ORDER BY transform_order"""
                cursor.execute(query, (model_id,))

                transformations = []
                for row in cursor.fetchall():
                    transform = {
                        'type': row['transform_type'],
                        'params': json.loads(row['transform_params']),
                        'order': row['transform_order']
                    }
                    transformations.append(transform)

                return transformations
        except Exception as e:
            return []

    def delete_model(self, model_id: int) -> bool:

        try:
            conn = self.conn()
            with conn.cursor() as cursor:
                query = "DELETE FROM time_series_models WHERE model_id = %s"
                cursor.execute(query, (model_id,))
                conn.commit()
                return True
        except Exception as e:

            return False

    def get_models_by_symbol(self, symbol: str, timeframe: str = None, active_only: bool = True) -> List[Dict]:

        try:
            conn = self.conn()
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                query = "SELECT * FROM time_series_models WHERE symbol = %s"
                params = [symbol]

                if timeframe:
                    query += " AND timeframe = %s"
                    params.append(timeframe)

                if active_only:
                    query += " AND is_active = TRUE"

                query += " ORDER BY created_at DESC"
                cursor.execute(query, params)

                models = []
                for row in cursor.fetchall():
                    models.append(dict(row))

                return models
        except Exception as e:
            return []

    def get_latest_model_by_symbol(self, symbol: str, model_type: str = None,
                                   timeframe: str = None) -> Optional[Dict]:

        try:
            conn = self.conn()
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                query = "SELECT * FROM time_series_models WHERE symbol = %s AND is_active = TRUE"
                params = [symbol]

                if model_type:
                    query += " AND model_type = %s"
                    params.append(model_type)

                if timeframe:
                    query += " AND timeframe = %s"
                    params.append(timeframe)

                query += " ORDER BY updated_at DESC LIMIT 1"
                cursor.execute(query, params)

                model_data = cursor.fetchone()
                if model_data:
                    return dict(model_data)
                return None
        except Exception as e:
            return None

    def get_model_performance_history(self, model_id: int) -> pd.DataFrame:

        try:
            conn = self.conn()
            query = """SELECT metric_name, metric_value, test_date
                       FROM model_metrics
                       WHERE model_id = %s \
                         AND test_date IS NOT NULL
                       ORDER BY test_date"""

            metrics_df = pd.read_sql(query, conn, params=[model_id], parse_dates=['test_date'])

            if not metrics_df.empty:
                # Перетворення на широкий формат для зручності аналізу
                metrics_pivot = metrics_df.pivot(index='test_date', columns='metric_name', values='metric_value')
                return metrics_pivot

            return pd.DataFrame()
        except Exception as e:
            return pd.DataFrame()


    def update_model_status(self, model_id: int, is_active: bool) -> bool:

        try:
            conn = self.conn()
            with conn.cursor() as cursor:
                query = "UPDATE time_series_models SET is_active = %s WHERE model_id = %s"
                cursor.execute(query, (is_active, model_id))
                conn.commit()
                return True
        except Exception as e:

            return False


    def compare_model_forecasts(self, model_ids: List[int], start_date: datetime = None,
                                end_date: datetime = None) -> pd.DataFrame:

        try:
            results = {}

            for model_id in model_ids:
                # Отримуємо інформацію про модель
                conn = self.connect()
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                    cursor.execute("SELECT model_key FROM time_series_models WHERE model_id = %s", (model_id,))
                    model_info = cursor.fetchone()

                    if model_info:
                        model_key = model_info['model_key']

                        # Отримання прогнозів для цієї моделі
                        forecasts = self.get_model_forecasts(model_id, start_date, end_date)

                        if not forecasts.empty:
                            results[model_key] = forecasts['forecast_value']

            if results:
                # Об'єднуємо всі прогнози в один DataFrame
                combined_df = pd.DataFrame(results)
                return combined_df

            return pd.DataFrame()
        except Exception as e:
            return pd.DataFrame()


    def get_model_forecast_accuracy(self, model_id: int, actual_data: pd.Series) -> Dict:

        try:
            # Отримуємо прогнози моделі
            forecast_df = self.get_model_forecasts(model_id)

            if forecast_df.empty:
                return {"error": "Прогнози не знайдено"}

            forecasts = forecast_df['forecast_value']

            # Фільтруємо фактичні дані для співпадіння з прогнозами
            common_dates = forecasts.index.intersection(actual_data.index)

            if len(common_dates) == 0:
                return {"error": "Немає спільних дат для порівняння прогнозів і фактичних даних"}

            # Підготовка даних для порівняння
            y_true = actual_data.loc[common_dates]
            y_pred = forecasts.loc[common_dates]

            # Розрахунок метрик
            mse = ((y_true - y_pred) ** 2).mean()
            rmse = np.sqrt(mse)
            mae = np.abs(y_true - y_pred).mean()
            mape = np.abs((y_true - y_pred) / y_true).mean() * 100

            # Збереження метрик в базу даних
            metrics = {
                "MSE": mse,
                "RMSE": rmse,
                "MAE": mae,
                "MAPE": mape
            }

            self.save_model_metrics(model_id, metrics, datetime.now())

            return metrics
        except Exception as e:
            return {"error": str(e)}


    def get_available_symbols(self) -> List[str]:

        try:
            conn = self.conn()
            with conn.cursor() as cursor:
                query = "SELECT DISTINCT symbol FROM time_series_models ORDER BY symbol"
                cursor.execute(query)
                symbols = [row[0] for row in cursor.fetchall()]
                return symbols
        except Exception as e:
            return []


    def get_model_transformation_pipeline(self, model_id: int) -> List[Dict]:

        try:
            transformations = self.get_data_transformations(model_id)
            return transformations
        except Exception as e:
            return []


    def save_complete_model(self, model_key: str, symbol: str, model_type: str,
                            timeframe: str, start_date: datetime, end_date: datetime,
                            model_obj: Any, parameters: Dict, metrics: Dict = None,
                            forecasts: pd.Series = None, transformations: List[Dict] = None,
                            lower_bounds: pd.Series = None, upper_bounds: pd.Series = None,
                            description: str = None) -> int:

        try:
            # Зберігаємо метадані моделі
            model_id = self.save_model_metadata(model_key, symbol, model_type, timeframe,
                                                start_date, end_date, description)

            # Зберігаємо параметри моделі
            if parameters:
                self.save_model_parameters(model_id, parameters)

            # Зберігаємо метрики моделі
            if metrics:
                self.save_model_metrics(model_id, metrics)

            # Зберігаємо прогнози
            if forecasts is not None:
                self.save_model_forecasts(model_id, forecasts, lower_bounds, upper_bounds)

            # Зберігаємо перетворення даних
            if transformations:
                self.save_data_transformations(model_id, transformations)

            # Зберігаємо бінарні дані моделі
            if model_obj:
                self.save_model_binary(model_id, model_obj)

            return model_id
        except Exception as e:
            raise


    def load_complete_model(self, model_key: str) -> Dict:

        try:
            # Отримуємо метадані моделі
            model_info = self.get_model_by_key(model_key)

            if not model_info:
                return {"error": f"Модель з ключем {model_key} не знайдена"}

            model_id = model_info['model_id']

            # Отримуємо всі необхідні дані
            parameters = self.get_model_parameters(model_id)
            metrics = self.get_model_metrics(model_id)
            transformations = self.get_data_transformations(model_id)
            forecasts = self.get_model_forecasts(model_id)
            model_obj = self.load_model_binary(model_id)

            # Формуємо повний словник з даними моделі
            result = {
                "model_info": model_info,
                "parameters": parameters,
                "metrics": metrics,
                "forecasts": forecasts,
                "transformations": transformations,
                "model_obj": model_obj
            }

            return result
        except Exception as e:
            return {"error": str(e)}




    def save_correlation_matrix(self, correlation_matrix, symbols_list, correlation_type, timeframe,
                                start_time, end_time, method):

        start_time_str = start_time.isoformat() if isinstance(start_time, datetime) else start_time
        end_time_str = end_time.isoformat() if isinstance(end_time, datetime) else end_time

        # Конвертація матриці в JSON формат
        if hasattr(correlation_matrix, 'to_json'):
            matrix_json = correlation_matrix.to_json()
        else:
            # Якщо це numpy array, перетворюємо в список списків
            matrix_json = json.dumps(correlation_matrix.tolist()
                                     if hasattr(correlation_matrix, 'tolist')
                                     else correlation_matrix)

        # Конвертація списку символів в JSON
        symbols_json = json.dumps(symbols_list)

        query = """
        INSERT OR REPLACE INTO correlation_matrices 
        (correlation_type, timeframe, start_time, end_time, method, matrix_json, symbols_list)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = (correlation_type, timeframe, start_time_str, end_time_str, method, matrix_json, symbols_json)
        return self.execute_query(query, params)


    def get_correlation_matrix(self, correlation_type, timeframe, start_time=None, end_time=None, method=None):

        query = """
                SELECT matrix_json, symbols_list
                FROM correlation_matrices
                WHERE correlation_type = ? \
                  AND timeframe = ? \
                """
        params = [correlation_type, timeframe]

        if start_time:
            start_time_str = start_time.isoformat() if isinstance(start_time, datetime) else start_time
            query += " AND start_time = ?"
            params.append(start_time_str)
        if end_time:
            end_time_str = end_time.isoformat() if isinstance(end_time, datetime) else end_time
            query += " AND end_time = ?"
            params.append(end_time_str)
        if method:
            query += " AND method = ?"
            params.append(method)

        query += " ORDER BY created_at DESC LIMIT 1"

        result = self.fetch_one(query, tuple(params))
        if result:
            matrix_json, symbols_json = result
            matrix = json.loads(matrix_json)
            symbols = json.loads(symbols_json)
            return matrix, symbols
        return None


    def get_correlation_breakdowns(self, symbol1=None, symbol2=None, timeframe=None,
                                   start_time=None, end_time=None, method=None):

        query = """
                SELECT symbol1, \
                       symbol2, \
                       breakdown_time, \
                       correlation_before, \
                       correlation_after,
                       timeframe, \
                       window_size, \
                       threshold, \
                       method
                FROM correlation_breakdowns
                WHERE 1 = 1 \
                """
        params = []

        if symbol1 and symbol2:
            # Переконуємося, що символи впорядковані (це важливо для пошуку)
            if symbol1 > symbol2:
                symbol1, symbol2 = symbol2, symbol1
            query += " AND symbol1 = ? AND symbol2 = ?"
            params.extend([symbol1, symbol2])
        elif symbol1:
            query += " AND (symbol1 = ? OR symbol2 = ?)"
            params.extend([symbol1, symbol1])
        elif symbol2:
            query += " AND (symbol1 = ? OR symbol2 = ?)"
            params.extend([symbol2, symbol2])

        if timeframe:
            query += " AND timeframe = ?"
            params.append(timeframe)

        if start_time:
            start_time_str = start_time.isoformat() if isinstance(start_time, datetime) else start_time
            query += " AND breakdown_time >= ?"
            params.append(start_time_str)

        if end_time:
            end_time_str = end_time.isoformat() if isinstance(end_time, datetime) else end_time
            query += " AND breakdown_time <= ?"
            params.append(end_time_str)

        if method:
            query += " AND method = ?"
            params.append(method)

        query += " ORDER BY breakdown_time DESC"

        return self.fetch_all(query, tuple(params))


    def save_market_beta(self, beta_values, market_symbol, timeframe, start_time, end_time):

        start_time_str = start_time.isoformat() if isinstance(start_time, datetime) else start_time
        end_time_str = end_time.isoformat() if isinstance(end_time, datetime) else end_time

        query = """
        INSERT OR REPLACE INTO market_betas
        (symbol, market_symbol, beta_value, timeframe, start_time, end_time)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        params = [(symbol, market_symbol, beta, timeframe, start_time_str, end_time_str)
                  for symbol, beta in beta_values.items()]
        return self.execute_many(query, params)


    def get_market_beta(self, symbol=None, market_symbol=None, timeframe=None, start_time=None, end_time=None):

        query = """
                SELECT symbol, market_symbol, beta_value, timeframe, start_time, end_time
                FROM market_betas
                WHERE 1 = 1 \
                """
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if market_symbol:
            query += " AND market_symbol = ?"
            params.append(market_symbol)
        if timeframe:
            query += " AND timeframe = ?"
            params.append(timeframe)
        if start_time:
            start_time_str = start_time.isoformat() if isinstance(start_time, datetime) else start_time
            query += " AND start_time = ?"
            params.append(start_time_str)
        if end_time:
            end_time_str = end_time.isoformat() if isinstance(end_time, datetime) else end_time
            query += " AND end_time = ?"
            params.append(end_time_str)

        query += " ORDER BY symbol, start_time DESC"

        return self.fetch_all(query, tuple(params))


    def save_beta_time_series(self, symbol, market_symbol, timestamps, beta_values, timeframe, window_size):

        timestamp_strs = []
        for ts in timestamps:
            if isinstance(ts, datetime):
                timestamp_strs.append(ts.isoformat())
            else:
                timestamp_strs.append(ts)

        query = """
        INSERT OR REPLACE INTO beta_time_series
        (symbol, market_symbol, timestamp, beta_value, timeframe, window_size)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        params = [(symbol, market_symbol, ts, beta, timeframe, window_size)
                  for ts, beta in zip(timestamp_strs, beta_values)]
        return self.execute_many(query, params)


    def get_beta_time_series(self, symbol, market_symbol, timeframe, window_size, start_time=None, end_time=None):

        query = """
                SELECT timestamp, beta_value
                FROM beta_time_series
                WHERE symbol = ? AND market_symbol = ? AND timeframe = ? AND window_size = ? \
                """
        params = [symbol, market_symbol, timeframe, window_size]

        if start_time:
            start_time_str = start_time.isoformat() if isinstance(start_time, datetime) else start_time
            query += " AND timestamp >= ?"
            params.append(start_time_str)
        if end_time:
            end_time_str = end_time.isoformat() if isinstance(end_time, datetime) else end_time
            query += " AND timestamp <= ?"
            params.append(end_time_str)

        query += " ORDER BY timestamp"

        return self.fetch_all(query, tuple(params))


    def save_sector_correlations(self, sector_correlations, correlation_type, timeframe, start_time, end_time, method):

        start_time_str = start_time.isoformat() if isinstance(start_time, datetime) else start_time
        end_time_str = end_time.isoformat() if isinstance(end_time, datetime) else end_time

        query = """
        INSERT OR REPLACE INTO sector_correlations
        (sector1, sector2, correlation_value, correlation_type, timeframe, start_time, end_time, method)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = [(s[0], s[1], s[2], correlation_type, timeframe, start_time_str, end_time_str, method)
                  for s in sector_correlations]
        return self.execute_many(query, params)


    def get_sector_correlations(self, sector=None, correlation_type=None, timeframe=None,
                                min_correlation=0.0, start_time=None, end_time=None, method=None):

        query = """
                SELECT sector1, sector2, correlation_value
                FROM sector_correlations
                WHERE correlation_value >= ? \
                """
        params = [min_correlation]

        if sector:
            query += " AND (sector1 = ? OR sector2 = ?)"
            params.extend([sector, sector])
        if correlation_type:
            query += " AND correlation_type = ?"
            params.append(correlation_type)
        if timeframe:
            query += " AND timeframe = ?"
            params.append(timeframe)
        if start_time:
            start_time_str = start_time.isoformat() if isinstance(start_time, datetime) else start_time
            query += " AND start_time = ?"
            params.append(start_time_str)
        if end_time:
            end_time_str = end_time.isoformat() if isinstance(end_time, datetime) else end_time
            query += " AND end_time = ?"
            params.append(end_time_str)
        if method:
            query += " AND method = ?"
            params.append(method)

        query += " ORDER BY correlation_value DESC"

        return self.fetch_all(query,tuple(params))


    def save_leading_indicators(self, leading_indicators):

        for item in leading_indicators:
            if isinstance(item['start_time'], datetime):
                item['start_time'] = item['start_time'].isoformat()
            if isinstance(item['end_time'], datetime):
                item['end_time'] = item['end_time'].isoformat()

        query = """
        INSERT OR REPLACE INTO leading_indicators
        (target_symbol, indicator_symbol, lag_period, correlation_value, 
         timeframe, start_time, end_time, method)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = [(item['target_symbol'], item['indicator_symbol'], item['lag_period'],
                   item['correlation_value'], item['timeframe'], item['start_time'],
                   item['end_time'], item['method'])
                  for item in leading_indicators]
        return self.execute_many(query, params)


    def get_leading_indicators(self, target_symbol=None, indicator_symbol=None, timeframe=None,
                               min_correlation=0.7, start_time=None, end_time=None, method=None):

        query = """
                SELECT target_symbol, indicator_symbol, lag_period, correlation_value
                FROM leading_indicators
                WHERE correlation_value >= ? \
                """
        params = [min_correlation]

        if target_symbol:
            query += " AND target_symbol = ?"
            params.append(target_symbol)
        if indicator_symbol:
            query += " AND indicator_symbol = ?"
            params.append(indicator_symbol)
        if timeframe:
            query += " AND timeframe = ?"
            params.append(timeframe)
        if start_time:
            start_time_str = start_time.isoformat() if isinstance(start_time, datetime) else start_time
            query += " AND start_time = ?"
            params.append(start_time_str)
        if end_time:
            end_time_str = end_time.isoformat() if isinstance(end_time, datetime) else end_time
            query += " AND end_time = ?"
            params.append(end_time_str)
        if method:
            query += " AND method = ?"
            params.append(method)

        query += " ORDER BY correlation_value DESC"

        return self.fetch_all(query, tuple(params))


    def save_external_asset_correlations(self, correlations, timeframe, start_time, end_time, method):

        start_time_str = start_time.isoformat() if isinstance(start_time, datetime) else start_time
        end_time_str = end_time.isoformat() if isinstance(end_time, datetime) else end_time

        query = """
        INSERT OR REPLACE INTO external_asset_correlations
        (crypto_symbol, external_asset, correlation_value, timeframe, start_time, end_time, method)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = [(c[0], c[1], c[2], timeframe, start_time_str, end_time_str, method)
                  for c in correlations]
        return self.execute_many(query, params)


    def get_external_asset_correlations(self, crypto_symbol=None, external_asset=None, timeframe=None,
                                        min_correlation=-1.0, start_time=None, end_time=None, method=None):

        query = """
                SELECT crypto_symbol, external_asset, correlation_value
                FROM external_asset_correlations
                WHERE correlation_value >= ? \
                """
        params = [min_correlation]

        if crypto_symbol:
            query += " AND crypto_symbol = ?"
            params.append(crypto_symbol)
        if external_asset:
            query += " AND external_asset = ?"
            params.append(external_asset)
        if timeframe:
            query += " AND timeframe = ?"
            params.append(timeframe)
        if start_time:
            start_time_str = start_time.isoformat() if isinstance(start_time, datetime) else start_time
            query += " AND start_time = ?"
            params.append(start_time_str)
        if end_time:
            end_time_str = end_time.isoformat() if isinstance(end_time, datetime) else end_time
            query += " AND end_time = ?"
            params.append(end_time_str)
        if method:
            query += " AND method = ?"
            params.append(method)

        query += " ORDER BY ABS(correlation_value) DESC"

        return self.fetch_all(query, tuple(params))


    def save_market_regime_correlations(self, regime_name, correlation_matrix, symbols_list,
                                        correlation_type, start_time, end_time, method):

        start_time_str = start_time.isoformat() if isinstance(start_time, datetime) else start_time
        end_time_str = end_time.isoformat() if isinstance(end_time, datetime) else end_time

        # Конвертація матриці в JSON формат
        if hasattr(correlation_matrix, 'to_json'):
            matrix_json = correlation_matrix.to_json()
        else:
            # Якщо це numpy array, перетворюємо в список списків
            matrix_json = json.dumps(correlation_matrix.tolist()
                                     if hasattr(correlation_matrix, 'tolist')
                                     else correlation_matrix)

        # Конвертація списку символів в JSON
        symbols_json = json.dumps(symbols_list)

        query = """
        INSERT OR REPLACE INTO market_regime_correlations
        (regime_name, start_time, end_time, correlation_type, matrix_json, symbols_list, method)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = (regime_name, start_time_str, end_time_str, correlation_type, matrix_json, symbols_json, method)
        return self.execute_query(query, params)


    def get_market_regime_correlations(self, regime_name=None, correlation_type=None,
                                       start_time=None, end_time=None, method=None):

        query = """
                SELECT regime_name, start_time, end_time, matrix_json, symbols_list
                FROM market_regime_correlations
                WHERE 1 = 1 \
                """
        params = []

        if regime_name:
            query += " AND regime_name = ?"
            params.append(regime_name)
        if correlation_type:
            query += " AND correlation_type = ?"
            params.append(correlation_type)
        if start_time:
            start_time_str = start_time.isoformat() if isinstance(start_time, datetime) else start_time
            query += " AND start_time = ?"
            params.append(start_time_str)
        if end_time:
            end_time_str = end_time.isoformat() if isinstance(end_time, datetime) else end_time
            query += " AND end_time = ?"
            params.append(end_time_str)
        if method:
            query += " AND method = ?"
            params.append(method)

        query += " ORDER BY regime_name, start_time"

        results = self.fetch_all(query, tuple(params))

        processed_results = []
        for result in results:
            regime_name, start_time, end_time, matrix_json, symbols_json = result
            matrix = json.loads(matrix_json)
            symbols = json.loads(symbols_json)
            processed_results.append((regime_name, start_time, end_time, matrix, symbols))

        return processed_results



    def insert_correlated_pair(self, symbol1, symbol2, correlation_value, correlation_type,
                               timeframe, start_time, end_time, method):

        # Перетворення часу до строкового формату, якщо потрібно
        if isinstance(start_time, datetime):
            start_time = start_time.isoformat()
        if isinstance(end_time, datetime):
            end_time = end_time.isoformat()

        query = """
                INSERT INTO correlated_pairs (symbol1, symbol2, correlation_value, correlation_type,
                                              timeframe, start_time, end_time, method)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (symbol1, symbol2, correlation_type, timeframe, start_time, end_time, method)
            DO NOTHING; \
                """
        params = (symbol1, symbol2, correlation_value, correlation_type,
                  timeframe, start_time, end_time, method)
        self.execute_query(query, params)
        return True


    def get_correlated_pairs(self, symbol=None, correlation_type=None, timeframe=None,
                             min_correlation=None, max_correlation=None, limit=None):

        # Складаємо базовий запит
        query = "SELECT * FROM correlated_pairs WHERE 1=1"
        params = []

        # Додаємо фільтри
        if symbol:
            query += " AND (symbol1 = %s OR symbol2 = %s)"
            params.extend([symbol, symbol])
        if correlation_type:
            query += " AND correlation_type = %s"
            params.append(correlation_type)
        if timeframe:
            query += " AND timeframe = %s"
            params.append(timeframe)
        if min_correlation is not None:
            query += " AND correlation_value >= %s"
            params.append(min_correlation)
        if max_correlation is not None:
            query += " AND correlation_value <= %s"
            params.append(max_correlation)

        # Додаємо сортування та ліміт
        query += " ORDER BY correlation_value DESC"

        if limit is not None:
            query += " LIMIT %s"
            params.append(limit)

        return self.fetch_dict(query, tuple(params))


    def save_correlation_time_series(self, symbol1, symbol2, correlation_type, timeframe,
                                     window_size, timestamp, correlation_value, method):

        # Перетворення часу до строкового формату, якщо потрібно
        if isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()

        query = """
                INSERT INTO correlation_time_series (symbol1, symbol2, correlation_type, timeframe,
                                                     window_size, timestamp, correlation_value, method)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (symbol1, symbol2, correlation_type, timeframe, window_size, timestamp, method) 
            DO \
                UPDATE SET correlation_value = EXCLUDED.correlation_value \
                """
        params = (symbol1, symbol2, correlation_type, timeframe,
                  window_size, timestamp, correlation_value, method)
        self.execute_query(query, params)
        return True

    def get_correlation_time_series(self, symbol1, symbol2, correlation_type, timeframe,
                                    window_size, start_time=None, end_time=None, method=None, limit=None):

        if isinstance(start_time, datetime):
            start_time = start_time.isoformat()
        if isinstance(end_time, datetime):
            end_time = end_time.isoformat()

        query = """
                SELECT timestamp, correlation_value
                FROM correlation_time_series
                WHERE symbol1 = %s
                  AND symbol2 = %s
                  AND correlation_type = %s
                  AND timeframe = %s
                  AND window_size = %s
                """
        params = [symbol1, symbol2, correlation_type, timeframe, window_size]

        if start_time:
            query += " AND timestamp >= %s"
            params.append(start_time)
        if end_time:
            query += " AND timestamp <= %s"
            params.append(end_time)
        if method:
            query += " AND method = %s"
            params.append(method)

        query += " ORDER BY timestamp ASC"

        if limit is not None:
            query += " LIMIT %s"
            params.append(limit)

        return self.fetch_dict(query, tuple(params))

    def get_most_correlated_pairs(self, symbol=None, correlation_type='pearson', timeframe='1h',
                                  limit=10, min_abs_correlation=0.7):

        query = """
                SELECT symbol1,
                       symbol2,
                       correlation_value,
                       correlation_type,
                       timeframe,
                       start_time,
                       end_time,
                       method,
                       ABS(correlation_value) as abs_correlation
                FROM correlated_pairs
                WHERE correlation_type = %s
                  AND timeframe = %s
                  AND ABS(correlation_value) >= %s
                """
        params = [correlation_type, timeframe, min_abs_correlation]

        if symbol:
            query += " AND (symbol1 = %s OR symbol2 = %s)"
            params.extend([symbol, symbol])

        query += " ORDER BY abs_correlation DESC LIMIT %s"
        params.append(limit)

        return self.fetch_dict(query, tuple(params))

    def save_market_cycle(self,
                          symbol: str,
                          cycle_type: str,
                          start_date: datetime,
                          end_date: Optional[datetime] = None,
                          peak_date: Optional[datetime] = None,
                          peak_price: Optional[float] = None,
                          bottom_date: Optional[datetime] = None,
                          bottom_price: Optional[float] = None,
                          max_drawdown: Optional[float] = None,
                          max_roi: Optional[float] = None,
                          cycle_duration_days: Optional[int] = None) -> int:

        if not self.conn:
            self.connect()

        query = """
                INSERT INTO market_cycles
                (symbol, cycle_type, start_date, end_date, peak_date, peak_price,
                 bottom_date, bottom_price, max_drawdown, max_roi, cycle_duration_days)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id \
                """

        try:
            self.cursor.execute(query, (
                symbol, cycle_type, start_date, end_date, peak_date, peak_price,
                bottom_date, bottom_price, max_drawdown, max_roi, cycle_duration_days
            ))
            cycle_id = self.cursor.fetchone()[0]
            self.conn.commit()
            return cycle_id
        except psycopg2.Error as e:
            self.conn.rollback()
            print(f"Помилка збереження ринкового циклу: {e}")
            raise

    def update_market_cycle(self,
                            cycle_id: int,
                            end_date: Optional[datetime] = None,
                            peak_date: Optional[datetime] = None,
                            peak_price: Optional[float] = None,
                            bottom_date: Optional[datetime] = None,
                            bottom_price: Optional[float] = None,
                            max_drawdown: Optional[float] = None,
                            max_roi: Optional[float] = None,
                            cycle_duration_days: Optional[int] = None) -> bool:

        if not self.conn:
            self.connect()

        # Формуємо динамічно частину запиту з полями, які потрібно оновити
        update_fields = []
        params = []

        if end_date is not None:
            update_fields.append("end_date = %s")
            params.append(end_date)

        if peak_date is not None:
            update_fields.append("peak_date = %s")
            params.append(peak_date)

        if peak_price is not None:
            update_fields.append("peak_price = %s")
            params.append(peak_price)

        if bottom_date is not None:
            update_fields.append("bottom_date = %s")
            params.append(bottom_date)

        if bottom_price is not None:
            update_fields.append("bottom_price = %s")
            params.append(bottom_price)

        if max_drawdown is not None:
            update_fields.append("max_drawdown = %s")
            params.append(max_drawdown)

        if max_roi is not None:
            update_fields.append("max_roi = %s")
            params.append(max_roi)

        if cycle_duration_days is not None:
            update_fields.append("cycle_duration_days = %s")
            params.append(cycle_duration_days)

        # Додаємо updated_at
        update_fields.append("updated_at = CURRENT_TIMESTAMP")

        # Якщо немає полів для оновлення, повертаємо True
        if not update_fields:
            return True

        query = f"""
            UPDATE market_cycles
            SET {', '.join(update_fields)}
            WHERE id = %s
        """

        params.append(cycle_id)

        try:
            self.cursor.execute(query, params)
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            self.conn.rollback()
            print(f"Помилка оновлення ринкового циклу: {e}")
            raise

    def get_market_cycle_by_id(self, cycle_id: int) -> Optional[Dict]:

        if not self.conn:
            self.connect()

        query = "SELECT * FROM market_cycles WHERE id = %s"

        try:
            self.cursor.execute(query, (cycle_id,))
            result = self.cursor.fetchone()
            return dict(result) if result else None
        except psycopg2.Error as e:
            print(f"Помилка отримання ринкового циклу: {e}")
            raise

    def get_market_cycles_by_symbol(self, symbol: str, cycle_type: Optional[str] = None) -> List[Dict]:

        if not self.conn:
            self.connect()

        params = [symbol]

        if cycle_type:
            query = "SELECT * FROM market_cycles WHERE symbol = %s AND cycle_type = %s ORDER BY start_date"
            params.append(cycle_type)
        else:
            query = "SELECT * FROM market_cycles WHERE symbol = %s ORDER BY start_date"

        try:
            self.cursor.execute(query, params)
            return [dict(row) for row in self.cursor.fetchall()]
        except psycopg2.Error as e:
            print(f"Помилка отримання ринкових циклів: {e}")
            raise

    def get_active_market_cycles(self, symbol: Optional[str] = None) -> List[Dict]:

        if not self.conn:
            self.connect()

        if symbol:
            query = "SELECT * FROM market_cycles WHERE end_date IS NULL AND symbol = %s ORDER BY start_date"
            params = (symbol,)
        else:
            query = "SELECT * FROM market_cycles WHERE end_date IS NULL ORDER BY symbol, start_date"
            params = ()

        try:
            self.cursor.execute(query, params)
            return [dict(row) for row in self.cursor.fetchall()]
        except psycopg2.Error as e:
            print(f"Помилка отримання активних ринкових циклів: {e}")
            raise

    def delete_market_cycle(self, cycle_id: int) -> bool:

        if not self.conn:
            self.connect()

        query = "DELETE FROM market_cycles WHERE id = %s"

        try:
            self.cursor.execute(query, (cycle_id,))
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            self.conn.rollback()
            print(f"Помилка видалення ринкового циклу: {e}")
            raise

    #
    # МЕТОДИ ДЛЯ РОБОТИ З ТАБЛИЦЕЮ cycle_features
    #

    def save_cycle_feature(self,
                           symbol: str,
                           timestamp: datetime,
                           timeframe: str,
                           days_since_last_halving: int,
                           days_to_next_halving: int,
                           halving_cycle_phase: float,
                           days_since_last_eth_upgrade: int,
                           days_to_next_eth_upgrade: int,
                           eth_upgrade_cycle_phase: float,
                           days_since_last_sol_event: int,
                           sol_network_stability_score: float,
                           weekly_cycle_position: float,
                           monthly_seasonality_factor: float,
                           market_phase: str,
                           optimal_cycle_length: int,
                           btc_correlation: float,
                           eth_correlation: float,
                           sol_correlation: float,
                           volatility_metric: float,
                           is_anomaly: bool = False) -> int:

        if not self.conn:
            self.connect()

        query = """
                INSERT INTO cycle_features
                (symbol, timestamp, timeframe, days_since_last_halving, days_to_next_halving,
                 halving_cycle_phase, days_since_last_eth_upgrade, days_to_next_eth_upgrade,
                 eth_upgrade_cycle_phase, days_since_last_sol_event, sol_network_stability_score,
                 weekly_cycle_position, monthly_seasonality_factor, market_phase,
                 optimal_cycle_length, btc_correlation, eth_correlation, sol_correlation,
                 volatility_metric, is_anomaly)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id \
                """

        try:
            self.cursor.execute(query, (
                symbol, timestamp, timeframe, days_since_last_halving, days_to_next_halving,
                halving_cycle_phase, days_since_last_eth_upgrade, days_to_next_eth_upgrade,
                eth_upgrade_cycle_phase, days_since_last_sol_event, sol_network_stability_score,
                weekly_cycle_position, monthly_seasonality_factor, market_phase,
                optimal_cycle_length, btc_correlation, eth_correlation, sol_correlation,
                volatility_metric, is_anomaly
            ))
            feature_id = self.cursor.fetchone()[0]
            self.conn.commit()
            return feature_id
        except psycopg2.Error as e:
            self.conn.rollback()
            print(f"Помилка збереження функцій циклу: {e}")
            raise

    def get_cycle_features(self,
                           symbol: str,
                           timeframe: str,
                           start_time: datetime,
                           end_time: datetime) -> List[Dict]:

        if not self.conn:
            self.connect()

        query = """
                SELECT * \
                FROM cycle_features
                WHERE symbol = %s \
                  AND timeframe = %s \
                  AND timestamp BETWEEN %s \
                  AND %s
                ORDER BY timestamp \
                """

        try:
            self.cursor.execute(query, (symbol, timeframe, start_time, end_time))
            return [dict(row) for row in self.cursor.fetchall()]
        except psycopg2.Error as e:
            print(f"Помилка отримання функцій циклу: {e}")
            raise

    def get_latest_cycle_features(self, symbol: str, timeframe: str) -> Optional[Dict]:

        if not self.conn:
            self.connect()

        query = """
                SELECT * \
                FROM cycle_features
                WHERE symbol = %s \
                  AND timeframe = %s
                ORDER BY timestamp DESC
                    LIMIT 1 \
                """

        try:
            self.cursor.execute(query, (symbol, timeframe))
            result = self.cursor.fetchone()
            return dict(result) if result else None
        except psycopg2.Error as e:
            print(f"Помилка отримання останніх функцій циклу: {e}")
            raise

    def delete_cycle_feature(self, feature_id: int) -> bool:

        if not self.conn:
            self.connect()

        query = "DELETE FROM cycle_features WHERE id = %s"

        try:
            self.cursor.execute(query, (feature_id,))
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            self.conn.rollback()
            print(f"Помилка видалення функції циклу: {e}")
            raise

    #
    # МЕТОДИ ДЛЯ РОБОТИ З ТАБЛИЦЕЮ cycle_similarity
    #

    def save_cycle_similarity(self,
                              symbol: str,
                              reference_cycle_id: int,
                              compared_cycle_id: int,
                              similarity_score: float,
                              normalized: bool) -> int:

        if not self.conn:
            self.connect()

        query = """
                INSERT INTO cycle_similarity
                (symbol, reference_cycle_id, compared_cycle_id, similarity_score, normalized)
                VALUES (%s, %s, %s, %s, %s) RETURNING id \
                """

        try:
            self.cursor.execute(query, (
                symbol, reference_cycle_id, compared_cycle_id, similarity_score, normalized
            ))
            similarity_id = self.cursor.fetchone()[0]
            self.conn.commit()
            return similarity_id
        except psycopg2.Error as e:
            self.conn.rollback()
            print(f"Помилка збереження схожості циклів: {e}")
            raise

    def get_cycle_similarities_by_reference(self,
                                            reference_cycle_id: int,
                                            min_similarity: Optional[float] = None) -> List[Dict]:

        if not self.conn:
            self.connect()

        query_parts = ["SELECT * FROM cycle_similarity WHERE reference_cycle_id = %s"]
        params = [reference_cycle_id]

        if min_similarity is not None:
            query_parts.append("AND similarity_score >= %s")
            params.append(min_similarity)

        query = " ".join(query_parts) + " ORDER BY similarity_score DESC"

        try:
            self.cursor.execute(query, params)
            return [dict(row) for row in self.cursor.fetchall()]
        except psycopg2.Error as e:
            print(f"Помилка отримання схожості циклів: {e}")
            raise

    def get_most_similar_cycles(self,
                                cycle_id: int,
                                limit: int = 5,
                                normalized: bool = True) -> List[Dict]:

        if not self.conn:
            self.connect()

        query = """
                SELECT cs.*, mc.symbol, mc.cycle_type, mc.start_date, mc.end_date
                FROM cycle_similarity cs
                         JOIN market_cycles mc ON cs.compared_cycle_id = mc.id
                WHERE cs.reference_cycle_id = %s \
                """

        params = [cycle_id]

        if normalized:
            query += " AND cs.normalized = TRUE"

        query += " ORDER BY cs.similarity_score DESC LIMIT %s"
        params.append(limit)

        try:
            self.cursor.execute(query, params)
            return [dict(row) for row in self.cursor.fetchall()]
        except psycopg2.Error as e:
            print(f"Помилка отримання найбільш схожих циклів: {e}")
            raise

    #
    # МЕТОДИ ДЛЯ РОБОТИ З ТАБЛИЦЕЮ predicted_turning_points
    #

    def save_predicted_turning_point(self,
                                     symbol: str,
                                     prediction_date: datetime,
                                     predicted_point_date: datetime,
                                     point_type: str,
                                     confidence: float,
                                     price_prediction: Optional[float] = None) -> int:

        if not self.conn:
            self.connect()

        query = """
                INSERT INTO predicted_turning_points
                (symbol, prediction_date, predicted_point_date, point_type, confidence, price_prediction)
                VALUES (%s, %s, %s, %s, %s, %s) RETURNING id \
                """

        try:
            self.cursor.execute(query, (
                symbol, prediction_date, predicted_point_date, point_type, confidence, price_prediction
            ))
            point_id = self.cursor.fetchone()[0]
            self.conn.commit()
            return point_id
        except psycopg2.Error as e:
            self.conn.rollback()
            print(f"Помилка збереження точки повороту: {e}")
            raise

    def update_turning_point_outcome(self,
                                     point_id: int,
                                     actual_outcome: str,
                                     actual_date: Optional[datetime] = None,
                                     actual_price: Optional[float] = None) -> bool:

        if not self.conn:
            self.connect()

        query = """
                UPDATE predicted_turning_points
                SET actual_outcome = %s, \
                    actual_date    = %s, \
                    actual_price   = %s, \
                    updated_at     = CURRENT_TIMESTAMP
                WHERE id = %s \
                """

        try:
            self.cursor.execute(query, (actual_outcome, actual_date, actual_price, point_id))
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            self.conn.rollback()
            print(f"Помилка оновлення результату точки повороту: {e}")
            raise

    def get_pending_turning_points(self, symbol: Optional[str] = None) -> List[Dict]:

        if not self.conn:
            self.connect()

        if symbol:
            query = """
                    SELECT * \
                    FROM predicted_turning_points
                    WHERE actual_outcome IS NULL \
                      AND symbol = %s
                    ORDER BY predicted_point_date \
                    """
            params = (symbol,)
        else:
            query = """
                    SELECT * \
                    FROM predicted_turning_points
                    WHERE actual_outcome IS NULL
                    ORDER BY symbol, predicted_point_date \
                    """
            params = ()

        try:
            self.cursor.execute(query, params)
            return [dict(row) for row in self.cursor.fetchall()]
        except psycopg2.Error as e:
            print(f"Помилка отримання очікуваних точок повороту: {e}")
            raise

    def get_turning_points_by_date_range(self,
                                         symbol: str,
                                         start_date: datetime,
                                         end_date: datetime) -> List[Dict]:

        if not self.conn:
            self.connect()

        query = """
                SELECT * \
                FROM predicted_turning_points
                WHERE symbol = %s \
                  AND predicted_point_date BETWEEN %s AND %s
                ORDER BY predicted_point_date \
                """

        try:
            self.cursor.execute(query, (symbol, start_date, end_date))
            return [dict(row) for row in self.cursor.fetchall()]
        except psycopg2.Error as e:
            print(f"Помилка отримання точок повороту за діапазоном дат: {e}")
            raise

    def insert_cycle_feature_performance(self, data: dict):

        query = """
                INSERT INTO cycle_feature_performance (model_id, feature_name, feature_importance, \
                                                       correlation_to_target, \
                                                       symbol, timeframe, training_period_start, training_period_end)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id; \
                """
        values = (
            data['model_id'],
            data['feature_name'],
            data['feature_importance'],
            data['correlation_to_target'],
            data['symbol'],
            data['timeframe'],
            data['training_period_start'],
            data['training_period_end'],
        )
        with self.conn.cursor() as cursor:
            cursor.execute(query, values)
            inserted_id = cursor.fetchone()[0]
            return inserted_id

    def get_cycle_feature_performance(self, model_id=None, symbol=None, timeframe=None):

        query = "SELECT * FROM cycle_feature_performance WHERE TRUE"
        params = []

        if model_id:
            query += " AND model_id = %s"
            params.append(model_id)
        if symbol:
            query += " AND symbol = %s"
            params.append(symbol)
        if timeframe:
            query += " AND timeframe = %s"
            params.append(timeframe)

        with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, params)
            results = cursor.fetchall()
            return results

    def save_btc_arima_data(self,
                            timeframe: str,
                            open_time: datetime,
                            original_close: float,
                            close_diff: Optional[float] = None,
                            close_diff2: Optional[float] = None,
                            close_log: Optional[float] = None,
                            close_log_diff: Optional[float] = None,
                            close_pct_change: Optional[float] = None,
                            close_seasonal_diff: Optional[float] = None,
                            close_combo_diff: Optional[float] = None,
                            adf_pvalue: Optional[float] = None,
                            kpss_pvalue: Optional[float] = None,
                            is_stationary: Optional[bool] = None,
                            significant_lags: Optional[List[int]] = None,
                            residual_variance: Optional[float] = None,
                            aic_score: Optional[float] = None,
                            bic_score: Optional[float] = None,
                            id: Optional[int] = None) -> int:

        with self.connect() as conn:
            with conn.cursor() as cursor:
                # Конвертація списку лагів у JSON якщо вони вказані
                significant_lags_json = json.dumps(significant_lags) if significant_lags is not None else None

                if id is None:
                    # Перевіряємо, чи існує запис з такими timeframe і open_time
                    check_query = ("""
                                          SELECT id
                                          FROM btc_arima_data
                                          WHERE timeframe = %s
                                            AND open_time = %s
                                          """)
                    cursor.execute(check_query, (timeframe, open_time))
                    existing_record = cursor.fetchone()

                    if existing_record:
                        # Якщо запис існує, оновлюємо його
                        id = existing_record[0]
                        query = ("""
                                        UPDATE btc_arima_data
                                        SET original_close      = %s,
                                            close_diff          = %s,
                                            close_diff2         = %s,
                                            close_log           = %s,
                                            close_log_diff      = %s,
                                            close_pct_change    = %s,
                                            close_seasonal_diff = %s,
                                            close_combo_diff    = %s,
                                            adf_pvalue          = %s,
                                            kpss_pvalue         = %s,
                                            is_stationary       = %s,
                                            significant_lags    = %s,
                                            residual_variance   = %s,
                                            aic_score           = %s,
                                            bic_score           = %s,
                                            updated_at          = CURRENT_TIMESTAMP
                                        WHERE id = %s RETURNING id
                                        """)
                        cursor.execute(query, (
                            original_close, close_diff, close_diff2, close_log,
                            close_log_diff, close_pct_change, close_seasonal_diff,
                            close_combo_diff, adf_pvalue, kpss_pvalue, is_stationary,
                            significant_lags_json, residual_variance, aic_score, bic_score, id
                        ))
                    else:
                        # Якщо запису немає, створюємо новий
                        query = ("""
                                        INSERT INTO btc_arima_data (timeframe, open_time, original_close, close_diff,
                                                                    close_diff2,
                                                                    close_log, close_log_diff, close_pct_change,
                                                                    close_seasonal_diff,
                                                                    close_combo_diff, adf_pvalue, kpss_pvalue,
                                                                    is_stationary, significant_lags,
                                                                    residual_variance, aic_score, bic_score)
                                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                                %s) RETURNING id
                                        """)
                        cursor.execute(query, (
                            timeframe, open_time, original_close, close_diff, close_diff2,
                            close_log, close_log_diff, close_pct_change, close_seasonal_diff,
                            close_combo_diff, adf_pvalue, kpss_pvalue, is_stationary, significant_lags_json,
                            residual_variance, aic_score, bic_score
                        ))
                else:
                    # Якщо ID вказано, оновлюємо існуючий запис
                    query = ("""
                                    UPDATE btc_arima_data
                                    SET timeframe           = %s,
                                        open_time           = %s,
                                        original_close      = %s,
                                        close_diff          = %s,
                                        close_diff2         = %s,
                                        close_log           = %s,
                                        close_log_diff      = %s,
                                        close_pct_change    = %s,
                                        close_seasonal_diff = %s,
                                        close_combo_diff    = %s,
                                        adf_pvalue          = %s,
                                        kpss_pvalue         = %s,
                                        is_stationary       = %s,
                                        significant_lags    = %s,
                                        residual_variance   = %s,
                                        aic_score           = %s,
                                        bic_score           = %s,
                                        updated_at          = CURRENT_TIMESTAMP
                                    WHERE id = %s RETURNING id
                                    """)
                    cursor.execute(query, (
                        timeframe, open_time, original_close, close_diff, close_diff2,
                        close_log, close_log_diff, close_pct_change, close_seasonal_diff,
                        close_combo_diff, adf_pvalue, kpss_pvalue, is_stationary, significant_lags_json,
                        residual_variance, aic_score, bic_score, id
                    ))

                result = cursor.fetchone()
                conn.commit()
                return result[0]

    def get_btc_arima_data_by_id(self, id: int) -> Optional[Dict[str, Any]]:

        with self.connect() as conn:
            with conn.cursor() as cursor:
                query = ("""
                                SELECT id,
                                       timeframe,
                                       open_time,
                                       original_close,
                                       close_diff,
                                       close_diff2,
                                       close_log,
                                       close_log_diff,
                                       close_pct_change,
                                       close_seasonal_diff,
                                       close_combo_diff,
                                       adf_pvalue,
                                       kpss_pvalue,
                                       is_stationary,
                                       significant_lags,
                                       residual_variance,
                                       aic_score,
                                       bic_score,
                                       created_at,
                                       updated_at
                                FROM btc_arima_data
                                WHERE id = %s
                                """)
                cursor.execute(query, (id,))
                row = cursor.fetchone()

                if row:
                    # Створюємо словник з результатів
                    column_names = [desc[0] for desc in cursor.description]
                    result = dict(zip(column_names, row))

                    # Парсимо JSON для significant_lags
                    if result['significant_lags']:
                        result['significant_lags'] = json.loads(result['significant_lags'])

                    return result
                return None

    def get_btc_arima_data_by_timeframe(self, timeframe: str,
                                        start_time: Optional[datetime] = None,
                                        end_time: Optional[datetime] = None,
                                        limit: int = 1000) -> List[Dict[str, Any]]:

        with self.connect() as conn:
            with conn.cursor() as cursor:
                query_parts = [
                    ("SELECT id, timeframe, open_time, original_close, close_diff, close_diff2, "
                            "close_log, close_log_diff, close_pct_change, close_seasonal_diff, "
                            "close_combo_diff, adf_pvalue, kpss_pvalue, is_stationary, significant_lags, "
                            "residual_variance, aic_score, bic_score, created_at, updated_at "
                            "FROM btc_arima_data "
                            "WHERE timeframe = %s")
                ]
                params = [timeframe]

                if start_time:
                    query_parts.append(("AND open_time >= %s"))
                    params.append(start_time)

                if end_time:
                    query_parts.append(("AND open_time <= %s"))
                    params.append(end_time)

                query_parts.append(("ORDER BY open_time"))
                query_parts.append(("LIMIT %s"))
                params.append(limit)

                query = (" ").join(query_parts)
                cursor.execute(query, params)
                rows = cursor.fetchall()

                # Створюємо словники з результатів
                column_names = [desc[0] for desc in cursor.description]
                results = []

                for row in rows:
                    result = dict(zip(column_names, row))

                    # Парсимо JSON для significant_lags
                    if result['significant_lags']:
                        result['significant_lags'] = json.loads(result['significant_lags'])

                    results.append(result)

                return results

    def delete_btc_arima_data(self, id: int) -> bool:

        with self.connect() as conn:
            with conn.cursor() as cursor:
                query = ("""
                                DELETE
                                FROM btc_arima_data
                                WHERE id = %s RETURNING id
                                """)
                cursor.execute(query, (id,))
                result = cursor.fetchone()
                conn.commit()
                return result is not None

    # --------- ETH ARIMA Data функції ---------

    def save_eth_arima_data(self,
                            timeframe: str,
                            open_time: datetime,
                            original_close: float,
                            close_diff: Optional[float] = None,
                            close_diff2: Optional[float] = None,
                            close_log: Optional[float] = None,
                            close_log_diff: Optional[float] = None,
                            close_pct_change: Optional[float] = None,
                            close_seasonal_diff: Optional[float] = None,
                            close_combo_diff: Optional[float] = None,
                            adf_pvalue: Optional[float] = None,
                            kpss_pvalue: Optional[float] = None,
                            is_stationary: Optional[bool] = None,
                            significant_lags: Optional[List[int]] = None,
                            residual_variance: Optional[float] = None,
                            aic_score: Optional[float] = None,
                            bic_score: Optional[float] = None,
                            id: Optional[int] = None) -> int:

        with self.connect() as conn:
            with conn.cursor() as cursor:
                # Конвертація списку лагів у JSON якщо вони вказані
                significant_lags_json = json.dumps(significant_lags) if significant_lags is not None else None

                if id is None:
                    # Перевіряємо, чи існує запис з такими timeframe і open_time
                    check_query = ("""
                                          SELECT id
                                          FROM eth_arima_data
                                          WHERE timeframe = %s
                                            AND open_time = %s
                                          """)
                    cursor.execute(check_query, (timeframe, open_time))
                    existing_record = cursor.fetchone()

                    if existing_record:
                        # Якщо запис існує, оновлюємо його
                        id = existing_record[0]
                        query = ("""
                                        UPDATE eth_arima_data
                                        SET original_close      = %s,
                                            close_diff          = %s,
                                            close_diff2         = %s,
                                            close_log           = %s,
                                            close_log_diff      = %s,
                                            close_pct_change    = %s,
                                            close_seasonal_diff = %s,
                                            close_combo_diff    = %s,
                                            adf_pvalue          = %s,
                                            kpss_pvalue         = %s,
                                            is_stationary       = %s,
                                            significant_lags    = %s,
                                            residual_variance   = %s,
                                            aic_score           = %s,
                                            bic_score           = %s,
                                            updated_at          = CURRENT_TIMESTAMP
                                        WHERE id = %s RETURNING id
                                        """)
                        cursor.execute(query, (
                            original_close, close_diff, close_diff2, close_log,
                            close_log_diff, close_pct_change, close_seasonal_diff,
                            close_combo_diff, adf_pvalue, kpss_pvalue, is_stationary,
                            significant_lags_json, residual_variance, aic_score, bic_score, id
                        ))
                    else:
                        # Якщо запису немає, створюємо новий
                        query = ("""
                                        INSERT INTO eth_arima_data (timeframe, open_time, original_close, close_diff,
                                                                    close_diff2,
                                                                    close_log, close_log_diff, close_pct_change,
                                                                    close_seasonal_diff,
                                                                    close_combo_diff, adf_pvalue, kpss_pvalue,
                                                                    is_stationary, significant_lags,
                                                                    residual_variance, aic_score, bic_score)
                                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                                %s) RETURNING id
                                        """)
                        cursor.execute(query, (
                            timeframe, open_time, original_close, close_diff, close_diff2,
                            close_log, close_log_diff, close_pct_change, close_seasonal_diff,
                            close_combo_diff, adf_pvalue, kpss_pvalue, is_stationary, significant_lags_json,
                            residual_variance, aic_score, bic_score
                        ))
                else:
                    # Якщо ID вказано, оновлюємо існуючий запис
                    query = ("""
                                    UPDATE eth_arima_data
                                    SET timeframe           = %s,
                                        open_time           = %s,
                                        original_close      = %s,
                                        close_diff          = %s,
                                        close_diff2         = %s,
                                        close_log           = %s,
                                        close_log_diff      = %s,
                                        close_pct_change    = %s,
                                        close_seasonal_diff = %s,
                                        close_combo_diff    = %s,
                                        adf_pvalue          = %s,
                                        kpss_pvalue         = %s,
                                        is_stationary       = %s,
                                        significant_lags    = %s,
                                        residual_variance   = %s,
                                        aic_score           = %s,
                                        bic_score           = %s,
                                        updated_at          = CURRENT_TIMESTAMP
                                    WHERE id = %s RETURNING id
                                    """)
                    cursor.execute(query, (
                        timeframe, open_time, original_close, close_diff, close_diff2,
                        close_log, close_log_diff, close_pct_change, close_seasonal_diff,
                        close_combo_diff, adf_pvalue, kpss_pvalue, is_stationary, significant_lags_json,
                        residual_variance, aic_score, bic_score, id
                    ))

                result = cursor.fetchone()
                conn.commit()
                return result[0]

    def get_eth_arima_data_by_id(self, id: int) -> Optional[Dict[str, Any]]:

        with self.connect() as conn:
            with conn.cursor() as cursor:
                query = ("""
                                SELECT id,
                                       timeframe,
                                       open_time,
                                       original_close,
                                       close_diff,
                                       close_diff2,
                                       close_log,
                                       close_log_diff,
                                       close_pct_change,
                                       close_seasonal_diff,
                                       close_combo_diff,
                                       adf_pvalue,
                                       kpss_pvalue,
                                       is_stationary,
                                       significant_lags,
                                       residual_variance,
                                       aic_score,
                                       bic_score,
                                       created_at,
                                       updated_at
                                FROM eth_arima_data
                                WHERE id = %s
                                """)
                cursor.execute(query, (id,))
                row = cursor.fetchone()

                if row:
                    # Створюємо словник з результатів
                    column_names = [desc[0] for desc in cursor.description]
                    result = dict(zip(column_names, row))

                    # Парсимо JSON для significant_lags
                    if result['significant_lags']:
                        result['significant_lags'] = json.loads(result['significant_lags'])

                    return result
                return None

    def get_eth_arima_data_by_timeframe(self, timeframe: str,
                                        start_time: Optional[datetime] = None,
                                        end_time: Optional[datetime] = None,
                                        limit: int = 1000) -> List[Dict[str, Any]]:

        with self.connect() as conn:
            with conn.cursor() as cursor:
                query_parts = [
                    ("SELECT id, timeframe, open_time, original_close, close_diff, close_diff2, "
                            "close_log, close_log_diff, close_pct_change, close_seasonal_diff, "
                            "close_combo_diff, adf_pvalue, kpss_pvalue, is_stationary, significant_lags, "
                            "residual_variance, aic_score, bic_score, created_at, updated_at "
                            "FROM eth_arima_data "
                            "WHERE timeframe = %s")
                ]
                params = [timeframe]

                if start_time:
                    query_parts.append(("AND open_time >= %s"))
                    params.append(start_time)

                if end_time:
                    query_parts.append(("AND open_time <= %s"))
                    params.append(end_time)

                query_parts.append(("ORDER BY open_time"))
                query_parts.append(("LIMIT %s"))
                params.append(limit)

                query = (" ").join(query_parts)
                cursor.execute(query, params)
                rows = cursor.fetchall()

                # Створюємо словники з результатів
                column_names = [desc[0] for desc in cursor.description]
                results = []

                for row in rows:
                    result = dict(zip(column_names, row))

                    # Парсимо JSON для significant_lags
                    if result['significant_lags']:
                        result['significant_lags'] = json.loads(result['significant_lags'])

                    results.append(result)

                return results

    def delete_eth_arima_data(self, id: int) -> bool:

        with self.connect() as conn:
            with conn.cursor() as cursor:
                query = ("""
                                DELETE
                                FROM eth_arima_data
                                WHERE id = %s RETURNING id
                                """)
                cursor.execute(query, (id,))
                result = cursor.fetchone()
                conn.commit()
                return result is not None

    def save_sol_arima_data(self,
                            timeframe: str,
                            open_time: datetime,
                            original_close: float,
                            close_diff: Optional[float] = None,
                            close_diff2: Optional[float] = None,
                            close_log: Optional[float] = None,
                            close_log_diff: Optional[float] = None,
                            close_pct_change: Optional[float] = None,
                            close_seasonal_diff: Optional[float] = None,
                            close_combo_diff: Optional[float] = None,
                            adf_pvalue: Optional[float] = None,
                            kpss_pvalue: Optional[float] = None,
                            is_stationary: Optional[bool] = None,
                            significant_lags: Optional[List[int]] = None,
                            residual_variance: Optional[float] = None,
                            aic_score: Optional[float] = None,
                            bic_score: Optional[float] = None,
                            id: Optional[int] = None) -> int:

        with self.connect() as conn:
            with conn.cursor() as cursor:
                # Конвертація списку лагів у JSON якщо вони вказані
                significant_lags_json = json.dumps(significant_lags) if significant_lags is not None else None

                if id is None:
                    # Перевіряємо, чи існує запис з такими timeframe і open_time
                    check_query = ("""
                                   SELECT id
                                   FROM sol_arima_data
                                   WHERE timeframe = %s
                                     AND open_time = %s
                                   """)
                    cursor.execute(check_query, (timeframe, open_time))
                    existing_record = cursor.fetchone()

                    if existing_record:
                        # Якщо запис існує, оновлюємо його
                        id = existing_record[0]
                        query = ("""
                                 UPDATE sol_arima_data
                                 SET original_close      = %s,
                                     close_diff          = %s,
                                     close_diff2         = %s,
                                     close_log           = %s,
                                     close_log_diff      = %s,
                                     close_pct_change    = %s,
                                     close_seasonal_diff = %s,
                                     close_combo_diff    = %s,
                                     adf_pvalue          = %s,
                                     kpss_pvalue         = %s,
                                     is_stationary       = %s,
                                     significant_lags    = %s,
                                     residual_variance   = %s,
                                     aic_score           = %s,
                                     bic_score           = %s,
                                     updated_at          = CURRENT_TIMESTAMP
                                 WHERE id = %s RETURNING id
                                 """)
                        cursor.execute(query, (
                            original_close, close_diff, close_diff2, close_log,
                            close_log_diff, close_pct_change, close_seasonal_diff,
                            close_combo_diff, adf_pvalue, kpss_pvalue, is_stationary,
                            significant_lags_json, residual_variance, aic_score, bic_score, id
                        ))
                    else:
                        # Якщо запису немає, створюємо новий
                        query = ("""
                                 INSERT INTO sol_arima_data (timeframe, open_time, original_close, close_diff,
                                                             close_diff2,
                                                             close_log, close_log_diff, close_pct_change,
                                                             close_seasonal_diff,
                                                             close_combo_diff, adf_pvalue, kpss_pvalue,
                                                             is_stationary, significant_lags,
                                                             residual_variance, aic_score, bic_score)
                                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                         %s) RETURNING id
                                 """)
                        cursor.execute(query, (
                            timeframe, open_time, original_close, close_diff, close_diff2,
                            close_log, close_log_diff, close_pct_change, close_seasonal_diff,
                            close_combo_diff, adf_pvalue, kpss_pvalue, is_stationary, significant_lags_json,
                            residual_variance, aic_score, bic_score
                        ))
                else:
                    # Якщо ID вказано, оновлюємо існуючий запис
                    query = ("""
                             UPDATE sol_arima_data
                             SET timeframe           = %s,
                                 open_time           = %s,
                                 original_close      = %s,
                                 close_diff          = %s,
                                 close_diff2         = %s,
                                 close_log           = %s,
                                 close_log_diff      = %s,
                                 close_pct_change    = %s,
                                 close_seasonal_diff = %s,
                                 close_combo_diff    = %s,
                                 adf_pvalue          = %s,
                                 kpss_pvalue         = %s,
                                 is_stationary       = %s,
                                 significant_lags    = %s,
                                 residual_variance   = %s,
                                 aic_score           = %s,
                                 bic_score           = %s,
                                 updated_at          = CURRENT_TIMESTAMP
                             WHERE id = %s RETURNING id
                             """)
                    cursor.execute(query, (
                        timeframe, open_time, original_close, close_diff, close_diff2,
                        close_log, close_log_diff, close_pct_change, close_seasonal_diff,
                        close_combo_diff, adf_pvalue, kpss_pvalue, is_stationary, significant_lags_json,
                        residual_variance, aic_score, bic_score, id
                    ))

                result = cursor.fetchone()
                conn.commit()
                return result[0]

    def get_sol_arima_data_by_id(self, id: int) -> Optional[Dict[str, Any]]:

        with self.connect() as conn:
            with conn.cursor() as cursor:
                query = ("""
                         SELECT id,
                                timeframe,
                                open_time,
                                original_close,
                                close_diff,
                                close_diff2,
                                close_log,
                                close_log_diff,
                                close_pct_change,
                                close_seasonal_diff,
                                close_combo_diff,
                                adf_pvalue,
                                kpss_pvalue,
                                is_stationary,
                                significant_lags,
                                residual_variance,
                                aic_score,
                                bic_score,
                                created_at,
                                updated_at
                         FROM sol_arima_data
                         WHERE id = %s
                         """)
                cursor.execute(query, (id,))
                row = cursor.fetchone()

                if row:
                    # Створюємо словник з результатів
                    column_names = [desc[0] for desc in cursor.description]
                    result = dict(zip(column_names, row))

                    # Парсимо JSON для significant_lags
                    if result['significant_lags']:
                        result['significant_lags'] = json.loads(result['significant_lags'])

                    return result
                return None

    def get_sol_arima_data_by_timeframe(self, timeframe: str,
                                        start_time: Optional[datetime] = None,
                                        end_time: Optional[datetime] = None,
                                        limit: int = 1000) -> List[Dict[str, Any]]:

        with self.connect() as conn:
            with conn.cursor() as cursor:
                query_parts = [
                    ("SELECT id, timeframe, open_time, original_close, close_diff, close_diff2, "
                     "close_log, close_log_diff, close_pct_change, close_seasonal_diff, "
                     "close_combo_diff, adf_pvalue, kpss_pvalue, is_stationary, significant_lags, "
                     "residual_variance, aic_score, bic_score, created_at, updated_at "
                     "FROM sol_arima_data "
                     "WHERE timeframe = %s")
                ]
                params = [timeframe]

                if start_time:
                    query_parts.append(("AND open_time >= %s"))
                    params.append(start_time)

                if end_time:
                    query_parts.append(("AND open_time <= %s"))
                    params.append(end_time)

                query_parts.append(("ORDER BY open_time"))
                query_parts.append(("LIMIT %s"))
                params.append(limit)

                query = (" ").join(query_parts)
                cursor.execute(query, params)
                rows = cursor.fetchall()

                # Створюємо словники з результатів
                column_names = [desc[0] for desc in cursor.description]
                results = []

                for row in rows:
                    result = dict(zip(column_names, row))

                    # Парсимо JSON для significant_lags
                    if result['significant_lags']:
                        result['significant_lags'] = json.loads(result['significant_lags'])

                    results.append(result)

                return results

    def delete_sol_arima_data(self, id: int) -> bool:

        with self.connect() as conn:
            with conn.cursor() as cursor:
                query = ("""
                         DELETE
                         FROM sol_arima_data
                         WHERE id = %s RETURNING id
                         """)
                cursor.execute(query, (id,))
                result = cursor.fetchone()
                conn.commit()
                return result is not None

    def save_eth_lstm_sequence(self, data_points: List[Dict[str, Any]]) -> List[int]:

        inserted_ids = []

        with self.conn.cursor() as cursor:
            for point in data_points:
                query = """
                        INSERT INTO eth_lstm_data (timeframe, sequence_id, sequence_position, open_time, \
                                                   open_scaled, high_scaled, low_scaled, close_scaled, volume_scaled, \
                                                   hour_sin, hour_cos, day_of_week_sin, day_of_week_cos, \
                                                   month_sin, month_cos, day_of_month_sin, day_of_month_cos, \
                                                   target_close_1, target_close_5, target_close_10, \
                                                   sequence_length, scaling_metadata) \
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (timeframe, sequence_id, sequence_position) 
                DO \
                        UPDATE SET
                            open_scaled = EXCLUDED.open_scaled, \
                            high_scaled = EXCLUDED.high_scaled, \
                            low_scaled = EXCLUDED.low_scaled, \
                            close_scaled = EXCLUDED.close_scaled, \
                            volume_scaled = EXCLUDED.volume_scaled, \
                            hour_sin = EXCLUDED.hour_sin, \
                            hour_cos = EXCLUDED.hour_cos, \
                            day_of_week_sin = EXCLUDED.day_of_week_sin, \
                            day_of_week_cos = EXCLUDED.day_of_week_cos, \
                            month_sin = EXCLUDED.month_sin, \
                            month_cos = EXCLUDED.month_cos, \
                            day_of_month_sin = EXCLUDED.day_of_month_sin, \
                            day_of_month_cos = EXCLUDED.day_of_month_cos, \
                            target_close_1 = EXCLUDED.target_close_1, \
                            target_close_5 = EXCLUDED.target_close_5, \
                            target_close_10 = EXCLUDED.target_close_10, \
                            sequence_length = EXCLUDED.sequence_length, \
                            scaling_metadata = EXCLUDED.scaling_metadata, \
                            updated_at = CURRENT_TIMESTAMP \
                            RETURNING id \
                        """

                cursor.execute(query, (
                    point['timeframe'],
                    point['sequence_id'],
                    point['sequence_position'],
                    point['open_time'],
                    point['open_scaled'],
                    point['high_scaled'],
                    point['low_scaled'],
                    point['close_scaled'],
                    point['volume_scaled'],
                    point.get('hour_sin'),
                    point.get('hour_cos'),
                    point.get('day_of_week_sin'),
                    point.get('day_of_week_cos'),
                    point.get('month_sin'),
                    point.get('month_cos'),
                    point.get('day_of_month_sin'),
                    point.get('day_of_month_cos'),
                    point.get('target_close_1'),
                    point.get('target_close_5'),
                    point.get('target_close_10'),
                    point.get('sequence_length'),
                    json.dumps(point['scaling_metadata']) if isinstance(point.get('scaling_metadata'),
                                                                        dict) else point.get('scaling_metadata')
                ))

                inserted_id = cursor.fetchone()[0]
                inserted_ids.append(inserted_id)

        self.conn.commit()
        return inserted_ids

    def get_eth_lstm_sequence(self, timeframe: str, sequence_id: int) -> List[Dict[str, Any]]:


        with self.conn.cursor() as cursor:
            query = """
                    SELECT id, \
                           timeframe, \
                           sequence_id, \
                           sequence_position, \
                           open_time, \
                           open_scaled, \
                           high_scaled, \
                           low_scaled, \
                           close_scaled, \
                           volume_scaled, \
                           hour_sin, \
                           hour_cos, \
                           day_of_week_sin, \
                           day_of_week_cos, \
                           month_sin, \
                           month_cos, \
                           day_of_month_sin, \
                           day_of_month_cos, \
                           target_close_1, \
                           target_close_5, \
                           target_close_10, \
                           sequence_length, \
                           scaling_metadata, \
                           created_at, \
                           updated_at
                    FROM eth_lstm_data
                    WHERE timeframe = %s \
                      AND sequence_id = %s
                    ORDER BY sequence_position \
                    """
            cursor.execute(query, (timeframe, sequence_id))
            results = cursor.fetchall()

            sequence_data = []
            for row in results:
                sequence_data.append({
                    'id': row[0],
                    'timeframe': row[1],
                    'sequence_id': row[2],
                    'sequence_position': row[3],
                    'open_time': row[4],
                    'open_scaled': float(row[5]),
                    'high_scaled': float(row[6]),
                    'low_scaled': float(row[7]),
                    'close_scaled': float(row[8]),
                    'volume_scaled': float(row[9]),
                    'hour_sin': float(row[10]) if row[10] is not None else None,
                    'hour_cos': float(row[11]) if row[11] is not None else None,
                    'day_of_week_sin': float(row[12]) if row[12] is not None else None,
                    'day_of_week_cos': float(row[13]) if row[13] is not None else None,
                    'month_sin': float(row[14]) if row[14] is not None else None,
                    'month_cos': float(row[15]) if row[15] is not None else None,
                    'day_of_month_sin': float(row[16]) if row[16] is not None else None,
                    'day_of_month_cos': float(row[17]) if row[17] is not None else None,
                    'target_close_1': float(row[18]) if row[18] is not None else None,
                    'target_close_5': float(row[19]) if row[19] is not None else None,
                    'target_close_10': float(row[20]) if row[20] is not None else None,
                    'sequence_length': row[21],
                    'scaling_metadata': json.loads(row[22]) if row[22] else None,
                    'created_at': row[23],
                    'updated_at': row[24]
                })

        return sequence_data

    def get_eth_lstm_batch(
            self,
            timeframe: str,
            sequence_length: int,
            batch_size: int,
            offset: int = 0,
            include_targets: bool = True
    ) -> Tuple[List[List[Dict[str, Any]]], List[List[float]]]:


        with self.conn.cursor() as cursor:
            sequence_query = """
                             SELECT DISTINCT sequence_id
                             FROM eth_lstm_data
                             WHERE timeframe = %s \
                               AND sequence_length = %s
                                 LIMIT %s \
                             OFFSET %s \
                             """
            cursor.execute(sequence_query, (timeframe, sequence_length, batch_size, offset))
            sequence_ids = [row[0] for row in cursor.fetchall()]

            if not sequence_ids:
                return [], []

            sequences = []
            targets = []

            for seq_id in sequence_ids:
                sequence_data_query = """
                                      SELECT id, \
                                             timeframe, \
                                             sequence_id, \
                                             sequence_position, \
                                             open_time, \
                                             open_scaled, \
                                             high_scaled, \
                                             low_scaled, \
                                             close_scaled, \
                                             volume_scaled, \
                                             hour_sin, \
                                             hour_cos, \
                                             day_of_week_sin, \
                                             day_of_week_cos, \
                                             month_sin, \
                                             month_cos, \
                                             day_of_month_sin, \
                                             day_of_month_cos, \
                                             target_close_1, \
                                             target_close_5, \
                                             target_close_10, \
                                             sequence_length, \
                                             scaling_metadata
                                      FROM eth_lstm_data
                                      WHERE timeframe = %s \
                                        AND sequence_id = %s
                                      ORDER BY sequence_position \
                                      """
                cursor.execute(sequence_data_query, (timeframe, seq_id))
                rows = cursor.fetchall()

                seq = []
                seq_targets = []

                for row in rows:
                    data_point = {
                        'id': row[0],
                        'timeframe': row[1],
                        'sequence_id': row[2],
                        'sequence_position': row[3],
                        'open_time': row[4],
                        'open_scaled': float(row[5]),
                        'high_scaled': float(row[6]),
                        'low_scaled': float(row[7]),
                        'close_scaled': float(row[8]),
                        'volume_scaled': float(row[9]),
                        'hour_sin': float(row[10]) if row[10] is not None else None,
                        'hour_cos': float(row[11]) if row[11] is not None else None,
                        'day_of_week_sin': float(row[12]) if row[12] is not None else None,
                        'day_of_week_cos': float(row[13]) if row[13] is not None else None,
                        'month_sin': float(row[14]) if row[14] is not None else None,
                        'month_cos': float(row[15]) if row[15] is not None else None,
                        'day_of_month_sin': float(row[16]) if row[16] is not None else None,
                        'day_of_month_cos': float(row[17]) if row[17] is not None else None,
                        'scaling_metadata': json.loads(row[21]) if row[21] else None,
                    }

                    seq.append(data_point)

                    if include_targets:
                        target = [
                            float(row[18]) if row[18] is not None else None,  # target_close_1
                            float(row[19]) if row[19] is not None else None,  # target_close_5
                            float(row[20]) if row[20] is not None else None  # target_close_10
                        ]
                        seq_targets.append(target)

                sequences.append(seq)
                if include_targets:
                    targets.append(seq_targets)

        return (sequences, targets) if include_targets else (sequences, [])

    def delete_eth_lstm_sequence(self, timeframe: str, sequence_id: int) -> int:


        with self.conn.cursor() as cursor:
            query = """
                    DELETE \
                    FROM eth_lstm_data
                    WHERE timeframe = %s \
                      AND sequence_id = %s \
                    """
            cursor.execute(query, (timeframe, sequence_id))
            deleted_count = cursor.rowcount

        self.conn.commit()
        return deleted_count

    def get_eth_latest_sequences(
            self,
            timeframe: str,
            num_sequences: int = 10,
            sequence_length: Optional[int] = None
    ) -> List[Dict[str, Any]]:


        with self.conn.cursor() as cursor:
            conditions = ["timeframe = %s"]
            params = [timeframe]

            if sequence_length is not None:
                conditions.append("sequence_length = %s")
                params.append(sequence_length)

            condition_str = " AND ".join(conditions)

            query = f"""
            SELECT 
                sequence_id, 
                MAX(open_time) as latest_time,
                MIN(open_time) as earliest_time,
                COUNT(*) as points_count,
                MAX(sequence_length) as sequence_length
            FROM eth_lstm_data
            WHERE {condition_str}
            GROUP BY sequence_id
            ORDER BY latest_time DESC
            LIMIT %s
            """
            params.append(num_sequences)

            cursor.execute(query, tuple(params))
            results = cursor.fetchall()

            sequences = []
            for row in results:
                sequences.append({
                    'sequence_id': row[0],
                    'latest_time': row[1],
                    'earliest_time': row[2],
                    'points_count': row[3],
                    'sequence_length': row[4]
                })

        return sequences

    def get_eth_scaling_metadata(self, timeframe: str) -> Dict[str, Any]:


        with self.conn.cursor() as cursor:
            query = """
                    SELECT scaling_metadata
                    FROM eth_lstm_data
                    WHERE timeframe = %s \
                      AND scaling_metadata IS NOT NULL LIMIT 1 \
                    """
            cursor.execute(query, (timeframe,))
            result = cursor.fetchone()

            if result and result[0]:
                return json.loads(result[0])

        return {}

    def count_eth_sequences(self, timeframe: str, sequence_length: Optional[int] = None) -> int:


        with self.conn.cursor() as cursor:
            conditions = ["timeframe = %s"]
            params = [timeframe]

            if sequence_length is not None:
                conditions.append("sequence_length = %s")
                params.append(sequence_length)

            condition_str = " AND ".join(conditions)

            query = f"""
            SELECT COUNT(DISTINCT sequence_id)
            FROM eth_lstm_data
            WHERE {condition_str}
            """

            cursor.execute(query, tuple(params))
            result = cursor.fetchone()

            return result[0] if result else 0

    def save_btc_lstm_sequence(self, data_points: List[Dict[str, Any]]) -> List[int]:

            inserted_ids = []

            with self.conn.cursor() as cursor:
                for point in data_points:
                    query = """
                            INSERT INTO btc_lstm_data (timeframe, sequence_id, sequence_position, open_time, \
                                                       open_scaled, high_scaled, low_scaled, close_scaled, \
                                                       volume_scaled, \
                                                       hour_sin, hour_cos, day_of_week_sin, day_of_week_cos, \
                                                       month_sin, month_cos, day_of_month_sin, day_of_month_cos, \
                                                       target_close_1, target_close_5, target_close_10, \
                                                       sequence_length, scaling_metadata) \
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, \
                                    %s) ON CONFLICT (timeframe, sequence_id, sequence_position) 
                    DO \
                            UPDATE SET
                                open_scaled = EXCLUDED.open_scaled, \
                                high_scaled = EXCLUDED.high_scaled, \
                                low_scaled = EXCLUDED.low_scaled, \
                                close_scaled = EXCLUDED.close_scaled, \
                                volume_scaled = EXCLUDED.volume_scaled, \
                                hour_sin = EXCLUDED.hour_sin, \
                                hour_cos = EXCLUDED.hour_cos, \
                                day_of_week_sin = EXCLUDED.day_of_week_sin, \
                                day_of_week_cos = EXCLUDED.day_of_week_cos, \
                                month_sin = EXCLUDED.month_sin, \
                                month_cos = EXCLUDED.month_cos, \
                                day_of_month_sin = EXCLUDED.day_of_month_sin, \
                                day_of_month_cos = EXCLUDED.day_of_month_cos, \
                                target_close_1 = EXCLUDED.target_close_1, \
                                target_close_5 = EXCLUDED.target_close_5, \
                                target_close_10 = EXCLUDED.target_close_10, \
                                sequence_length = EXCLUDED.sequence_length, \
                                scaling_metadata = EXCLUDED.scaling_metadata, \
                                updated_at = CURRENT_TIMESTAMP \
                                RETURNING id \
                            """

                    cursor.execute(query, (
                        point['timeframe'],
                        point['sequence_id'],
                        point['sequence_position'],
                        point['open_time'],
                        point['open_scaled'],
                        point['high_scaled'],
                        point['low_scaled'],
                        point['close_scaled'],
                        point['volume_scaled'],
                        point.get('hour_sin'),
                        point.get('hour_cos'),
                        point.get('day_of_week_sin'),
                        point.get('day_of_week_cos'),
                        point.get('month_sin'),
                        point.get('month_cos'),
                        point.get('day_of_month_sin'),
                        point.get('day_of_month_cos'),
                        point.get('target_close_1'),
                        point.get('target_close_5'),
                        point.get('target_close_10'),
                        point.get('sequence_length'),
                        json.dumps(point['scaling_metadata']) if isinstance(point.get('scaling_metadata'),
                                                                            dict) else point.get('scaling_metadata')
                    ))

                    inserted_id = cursor.fetchone()[0]
                    inserted_ids.append(inserted_id)

            self.conn.commit()
            return inserted_ids

    def get_btc_lstm_sequence(self, timeframe: str, sequence_id: int) -> List[Dict[str, Any]]:
        with self.conn.cursor() as cursor:
            query = """
                    SELECT id, \
                           timeframe, \
                           sequence_id, \
                           sequence_position, \
                           open_time, \
                           open_scaled, \
                           high_scaled, \
                           low_scaled, \
                           close_scaled, \
                           volume_scaled, \
                           hour_sin, \
                           hour_cos, \
                           day_of_week_sin, \
                           day_of_week_cos, \
                           month_sin, \
                           month_cos, \
                           day_of_month_sin, \
                           day_of_month_cos, \
                           target_close_1, \
                           target_close_5, \
                           target_close_10, \
                           sequence_length, \
                           scaling_metadata, \
                           created_at, \
                           updated_at
                    FROM btc_lstm_data
                    WHERE timeframe = %s \
                      AND sequence_id = %s
                    ORDER BY sequence_position \
                    """
            cursor.execute(query, (timeframe, sequence_id))
            results = cursor.fetchall()

            sequence_data = []
            for row in results:
                sequence_data.append({
                    'id': row[0],
                    'timeframe': row[1],
                    'sequence_id': row[2],
                    'sequence_position': row[3],
                    'open_time': row[4],
                    'open_scaled': float(row[5]),
                    'high_scaled': float(row[6]),
                    'low_scaled': float(row[7]),
                    'close_scaled': float(row[8]),
                    'volume_scaled': float(row[9]),
                    'hour_sin': float(row[10]) if row[10] is not None else None,
                    'hour_cos': float(row[11]) if row[11] is not None else None,
                    'day_of_week_sin': float(row[12]) if row[12] is not None else None,
                    'day_of_week_cos': float(row[13]) if row[13] is not None else None,
                    'month_sin': float(row[14]) if row[14] is not None else None,
                    'month_cos': float(row[15]) if row[15] is not None else None,
                    'day_of_month_sin': float(row[16]) if row[16] is not None else None,
                    'day_of_month_cos': float(row[17]) if row[17] is not None else None,
                    'target_close_1': float(row[18]) if row[18] is not None else None,
                    'target_close_5': float(row[19]) if row[19] is not None else None,
                    'target_close_10': float(row[20]) if row[20] is not None else None,
                    'sequence_length': row[21],
                    'scaling_metadata': json.loads(row[22]) if row[22] else None,
                    'created_at': row[23],
                    'updated_at': row[24]
                })

            return sequence_data

    def delete_btc_lstm_sequence(self, timeframe: str, sequence_id: int) -> int:

            with self.conn.cursor() as cursor:
                query = """
                        DELETE \
                        FROM btc_lstm_data
                        WHERE timeframe = %s \
                          AND sequence_id = %s \
                        """
                cursor.execute(query, (timeframe, sequence_id))
                deleted_count = cursor.rowcount

            self.conn.commit()
            return deleted_count

    def get_btc_latest_sequences(
                self,
                timeframe: str,
                num_sequences: int = 10,
                sequence_length: Optional[int] = None
        ) -> List[Dict[str, Any]]:

            with self.conn.cursor() as cursor:
                conditions = ["timeframe = %s"]
                params = [timeframe]

                if sequence_length is not None:
                    conditions.append("sequence_length = %s")
                    params.append(sequence_length)

                condition_str = " AND ".join(conditions)

                query = f"""
                SELECT 
                    sequence_id, 
                    MAX(open_time) as latest_time,
                    MIN(open_time) as earliest_time,
                    COUNT(*) as points_count,
                    MAX(sequence_length) as sequence_length
                FROM btc_lstm_data
                WHERE {condition_str}
                GROUP BY sequence_id
                ORDER BY latest_time DESC
                LIMIT %s
                """
                params.append(num_sequences)

                cursor.execute(query, tuple(params))
                results = cursor.fetchall()

                sequences = []
                for row in results:
                    sequences.append({
                        'sequence_id': row[0],
                        'latest_time': row[1],
                        'earliest_time': row[2],
                        'points_count': row[3],
                        'sequence_length': row[4]
                    })

            return sequences


    def get_btc_scaling_metadata(self, timeframe: str) -> Dict[str, Any]:
        with self.conn.cursor() as cursor:
            query = """
                    SELECT scaling_metadata
                    FROM btc_lstm_data
                    WHERE timeframe = %s \
                      AND scaling_metadata IS NOT NULL LIMIT 1 \
                    """
            cursor.execute(query, (timeframe,))
            result = cursor.fetchone()

            if result and result[0]:
                return json.loads(result[0])

        return {}


    def count_btc_sequences(self, timeframe: str, sequence_length: Optional[int] = None) -> int:

            with self.conn.cursor() as cursor:
                conditions = ["timeframe = %s"]
                params = [timeframe]

                if sequence_length is not None:
                    conditions.append("sequence_length = %s")
                    params.append(sequence_length)

                condition_str = " AND ".join(conditions)

                query = f"""
                SELECT COUNT(DISTINCT sequence_id)
                FROM btc_lstm_data
                WHERE {condition_str}
                """

                cursor.execute(query, tuple(params))
                result = cursor.fetchone()

                return result[0] if result else 0

    def save_sol_lstm_sequence(self, data_points: List[Dict[str, Any]]) -> List[int]:

        inserted_ids = []

        with self.conn.cursor() as cursor:
            for point in data_points:
                query = """
                        INSERT INTO sol_lstm_data (timeframe, sequence_id, sequence_position, open_time, \
                                                   open_scaled, high_scaled, low_scaled, close_scaled, volume_scaled, \
                                                   hour_sin, hour_cos, day_of_week_sin, day_of_week_cos, \
                                                   month_sin, month_cos, day_of_month_sin, day_of_month_cos, \
                                                   target_close_1, target_close_5, target_close_10, \
                                                   sequence_length, scaling_metadata) \
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (timeframe, sequence_id, sequence_position) 
                DO \
                        UPDATE SET
                            open_scaled = EXCLUDED.open_scaled, \
                            high_scaled = EXCLUDED.high_scaled, \
                            low_scaled = EXCLUDED.low_scaled, \
                            close_scaled = EXCLUDED.close_scaled, \
                            volume_scaled = EXCLUDED.volume_scaled, \
                            hour_sin = EXCLUDED.hour_sin, \
                            hour_cos = EXCLUDED.hour_cos, \
                            day_of_week_sin = EXCLUDED.day_of_week_sin, \
                            day_of_week_cos = EXCLUDED.day_of_week_cos, \
                            month_sin = EXCLUDED.month_sin, \
                            month_cos = EXCLUDED.month_cos, \
                            day_of_month_sin = EXCLUDED.day_of_month_sin, \
                            day_of_month_cos = EXCLUDED.day_of_month_cos, \
                            target_close_1 = EXCLUDED.target_close_1, \
                            target_close_5 = EXCLUDED.target_close_5, \
                            target_close_10 = EXCLUDED.target_close_10, \
                            sequence_length = EXCLUDED.sequence_length, \
                            scaling_metadata = EXCLUDED.scaling_metadata, \
                            updated_at = CURRENT_TIMESTAMP \
                            RETURNING id \
                        """

                cursor.execute(query, (
                    point['timeframe'],
                    point['sequence_id'],
                    point['sequence_position'],
                    point['open_time'],
                    point['open_scaled'],
                    point['high_scaled'],
                    point['low_scaled'],
                    point['close_scaled'],
                    point['volume_scaled'],
                    point.get('hour_sin'),
                    point.get('hour_cos'),
                    point.get('day_of_week_sin'),
                    point.get('day_of_week_cos'),
                    point.get('month_sin'),
                    point.get('month_cos'),
                    point.get('day_of_month_sin'),
                    point.get('day_of_month_cos'),
                    point.get('target_close_1'),
                    point.get('target_close_5'),
                    point.get('target_close_10'),
                    point.get('sequence_length'),
                    json.dumps(point['scaling_metadata']) if isinstance(point.get('scaling_metadata'),
                                                                        dict) else point.get('scaling_metadata')
                ))

                inserted_id = cursor.fetchone()[0]
                inserted_ids.append(inserted_id)

        self.conn.commit()
        return inserted_ids

    def get_sol_lstm_sequence(self, timeframe: str, sequence_id: int) -> List[Dict[str, Any]]:

        with self.conn.cursor() as cursor:
            query = """
                    SELECT id, \
                           timeframe, \
                           sequence_id, \
                           sequence_position, \
                           open_time, \
                           open_scaled, \
                           high_scaled, \
                           low_scaled, \
                           close_scaled, \
                           volume_scaled, \
                           hour_sin, \
                           hour_cos, \
                           day_of_week_sin, \
                           day_of_week_cos, \
                           month_sin, \
                           month_cos, \
                           day_of_month_sin, \
                           day_of_month_cos, \
                           target_close_1, \
                           target_close_5, \
                           target_close_10, \
                           sequence_length, \
                           scaling_metadata, \
                           created_at, \
                           updated_at
                    FROM sol_lstm_data
                    WHERE timeframe = %s \
                      AND sequence_id = %s
                    ORDER BY sequence_position \
                    """
            cursor.execute(query, (timeframe, sequence_id))
            results = cursor.fetchall()

            sequence_data = []
            for row in results:
                sequence_data.append({
                    'id': row[0],
                    'timeframe': row[1],
                    'sequence_id': row[2],
                    'sequence_position': row[3],
                    'open_time': row[4],
                    'open_scaled': float(row[5]),
                    'high_scaled': float(row[6]),
                    'low_scaled': float(row[7]),
                    'close_scaled': float(row[8]),
                    'volume_scaled': float(row[9]),
                    'hour_sin': float(row[10]) if row[10] is not None else None,
                    'hour_cos': float(row[11]) if row[11] is not None else None,
                    'day_of_week_sin': float(row[12]) if row[12] is not None else None,
                    'day_of_week_cos': float(row[13]) if row[13] is not None else None,
                    'month_sin': float(row[14]) if row[14] is not None else None,
                    'month_cos': float(row[15]) if row[15] is not None else None,
                    'day_of_month_sin': float(row[16]) if row[16] is not None else None,
                    'day_of_month_cos': float(row[17]) if row[17] is not None else None,
                    'target_close_1': float(row[18]) if row[18] is not None else None,
                    'target_close_5': float(row[19]) if row[19] is not None else None,
                    'target_close_10': float(row[20]) if row[20] is not None else None,
                    'sequence_length': row[21],
                    'scaling_metadata': json.loads(row[22]) if row[22] else None,
                    'created_at': row[23],
                    'updated_at': row[24]
                })

        return sequence_data

    def get_sol_lstm_batch(
            self,
            timeframe: str,
            sequence_length: int,
            batch_size: int,
            offset: int = 0,
            include_targets: bool = True
    ) -> Tuple[List[List[Dict[str, Any]]], List[List[float]]]:

        with self.conn.cursor() as cursor:
            sequence_query = """
                             SELECT DISTINCT sequence_id
                             FROM sol_lstm_data
                             WHERE timeframe = %s \
                               AND sequence_length = %s
                                 LIMIT %s \
                             OFFSET %s \
                             """
            cursor.execute(sequence_query, (timeframe, sequence_length, batch_size, offset))
            sequence_ids = [row[0] for row in cursor.fetchall()]

            if not sequence_ids:
                return [], []

            sequences = []
            targets = []

            for seq_id in sequence_ids:
                sequence_data_query = """
                                      SELECT id, \
                                             timeframe, \
                                             sequence_id, \
                                             sequence_position, \
                                             open_time, \
                                             open_scaled, \
                                             high_scaled, \
                                             low_scaled, \
                                             close_scaled, \
                                             volume_scaled, \
                                             hour_sin, \
                                             hour_cos, \
                                             day_of_week_sin, \
                                             day_of_week_cos, \
                                             month_sin, \
                                             month_cos, \
                                             day_of_month_sin, \
                                             day_of_month_cos, \
                                             target_close_1, \
                                             target_close_5, \
                                             target_close_10, \
                                             sequence_length, \
                                             scaling_metadata
                                      FROM sol_lstm_data
                                      WHERE timeframe = %s \
                                        AND sequence_id = %s
                                      ORDER BY sequence_position \
                                      """
                cursor.execute(sequence_data_query, (timeframe, seq_id))
                rows = cursor.fetchall()

                seq = []
                seq_targets = []

                for row in rows:
                    data_point = {
                        'id': row[0],
                        'timeframe': row[1],
                        'sequence_id': row[2],
                        'sequence_position': row[3],
                        'open_time': row[4],
                        'open_scaled': float(row[5]),
                        'high_scaled': float(row[6]),
                        'low_scaled': float(row[7]),
                        'close_scaled': float(row[8]),
                        'volume_scaled': float(row[9]),
                        'hour_sin': float(row[10]) if row[10] is not None else None,
                        'hour_cos': float(row[11]) if row[11] is not None else None,
                        'day_of_week_sin': float(row[12]) if row[12] is not None else None,
                        'day_of_week_cos': float(row[13]) if row[13] is not None else None,
                        'month_sin': float(row[14]) if row[14] is not None else None,
                        'month_cos': float(row[15]) if row[15] is not None else None,
                        'day_of_month_sin': float(row[16]) if row[16] is not None else None,
                        'day_of_month_cos': float(row[17]) if row[17] is not None else None,
                        'scaling_metadata': json.loads(row[21]) if row[21] else None,
                    }

                    seq.append(data_point)

                    if include_targets:
                        target = [
                            float(row[18]) if row[18] is not None else None,  # target_close_1
                            float(row[19]) if row[19] is not None else None,  # target_close_5
                            float(row[20]) if row[20] is not None else None  # target_close_10
                        ]
                        seq_targets.append(target)

                sequences.append(seq)
                if include_targets:
                    targets.append(seq_targets)

        return (sequences, targets) if include_targets else (sequences, [])

    def delete_sol_lstm_sequence(self, timeframe: str, sequence_id: int) -> int:

        with self.conn.cursor() as cursor:
            query = """
                    DELETE \
                    FROM sol_lstm_data
                    WHERE timeframe = %s \
                      AND sequence_id = %s \
                    """
            cursor.execute(query, (timeframe, sequence_id))
            deleted_count = cursor.rowcount

        self.conn.commit()
        return deleted_count

    def get_sol_latest_sequences(
            self,
            timeframe: str,
            num_sequences: int = 10,
            sequence_length: Optional[int] = None
    ) -> List[Dict[str, Any]]:

        with self.conn.cursor() as cursor:
            conditions = ["timeframe = %s"]
            params = [timeframe]

            if sequence_length is not None:
                conditions.append("sequence_length = %s")
                params.append(sequence_length)

            condition_str = " AND ".join(conditions)

            query = f"""
            SELECT 
                sequence_id, 
                MAX(open_time) as latest_time,
                MIN(open_time) as earliest_time,
                COUNT(*) as points_count,
                MAX(sequence_length) as sequence_length
            FROM sol_lstm_data
            WHERE {condition_str}
            GROUP BY sequence_id
            ORDER BY latest_time DESC
            LIMIT %s
            """
            params.append(num_sequences)

            cursor.execute(query, tuple(params))
            results = cursor.fetchall()

            sequences = []
            for row in results:
                sequences.append({
                    'sequence_id': row[0],
                    'latest_time': row[1],
                    'earliest_time': row[2],
                    'points_count': row[3],
                    'sequence_length': row[4]
                })

        return sequences

    def get_sol_scaling_metadata(self, timeframe: str) -> Dict[str, Any]:

        with self.conn.cursor() as cursor:
            query = """
                    SELECT scaling_metadata
                    FROM sol_lstm_data
                    WHERE timeframe = %s \
                      AND scaling_metadata IS NOT NULL LIMIT 1 \
                    """
            cursor.execute(query, (timeframe,))
            result = cursor.fetchone()

            if result and result[0]:
                return json.loads(result[0])

        return {}


    def count_sol_sequences(self, timeframe: str, sequence_length: Optional[int] = None) -> int:
        with self.conn.cursor() as cursor:
            conditions = ["timeframe = %s"]
            params = [timeframe]

            if sequence_length is not None:
                conditions.append("sequence_length = %s")
                params.append(sequence_length)

            condition_str = " AND ".join(conditions)

            query = f"""
                SELECT COUNT(DISTINCT sequence_id)
                FROM sol_lstm_data
                WHERE {condition_str}
                """

            cursor.execute(query, tuple(params))
            result = cursor.fetchone()

            return result[0] if result else 0

    def get_btc_arima_data(self, timeframe: str = None, open_time: datetime = None, id: int = None) -> dict:

        with self.connect() as conn:
            with conn.cursor() as cursor:
                query_parts = ["SELECT * FROM btc_arima_data WHERE 1=1"]
                params = []

                if id is not None:
                    query_parts.append("AND id = %s")
                    params.append(id)
                if timeframe is not None:
                    query_parts.append("AND timeframe = %s")
                    params.append(timeframe)
                if open_time is not None:
                    query_parts.append("AND open_time = %s")
                    params.append(open_time)

                query = " ".join(query_parts)
                cursor.execute(query, params)

                columns = [desc[0] for desc in cursor.description]
                result = cursor.fetchone()

                if result:
                    record = dict(zip(columns, result))

                    # Перетворення JSON рядка у список для significant_lags
                    if record.get('significant_lags'):
                        record['significant_lags'] = json.loads(record['significant_lags'])

                    return record
                return None

    def get_all_btc_arima_data(self) -> list:

        with self.connect() as conn:
            with conn.cursor() as cursor:
                query = "SELECT * FROM btc_arima_data ORDER BY open_time DESC"
                cursor.execute(query)

                columns = [desc[0] for desc in cursor.description]
                results = cursor.fetchall()

                records = []
                for result in results:
                    record = dict(zip(columns, result))

                    # Перетворення JSON рядка у список для significant_lags
                    if record.get('significant_lags'):
                        record['significant_lags'] = json.loads(record['significant_lags'])

                    records.append(record)

                return records

    def get_sol_arima_data(self, timeframe: str = None, open_time: datetime = None, id: int = None) -> dict:

        with self.connect() as conn:
            with conn.cursor() as cursor:
                query_parts = ["SELECT * FROM sol_arima_data WHERE 1=1"]
                params = []

                if id is not None:
                    query_parts.append("AND id = %s")
                    params.append(id)
                if timeframe is not None:
                    query_parts.append("AND timeframe = %s")
                    params.append(timeframe)
                if open_time is not None:
                    query_parts.append("AND open_time = %s")
                    params.append(open_time)

                query = " ".join(query_parts)
                cursor.execute(query, params)

                columns = [desc[0] for desc in cursor.description]
                result = cursor.fetchone()

                if result:
                    record = dict(zip(columns, result))

                    # Перетворення JSON рядка у список для significant_lags
                    if record.get('significant_lags'):
                        record['significant_lags'] = json.loads(record['significant_lags'])

                    return record
                return None

    def get_all_sol_arima_data(self) -> list:

        with self.connect() as conn:
            with conn.cursor() as cursor:
                query = "SELECT * FROM sol_arima_data ORDER BY open_time DESC"
                cursor.execute(query)

                columns = [desc[0] for desc in cursor.description]
                results = cursor.fetchall()

                records = []
                for result in results:
                    record = dict(zip(columns, result))

                    # Перетворення JSON рядка у список для significant_lags
                    if record.get('significant_lags'):
                        record['significant_lags'] = json.loads(record['significant_lags'])

                    records.append(record)

                return records

    def get_eth_arima_data(self, timeframe: str = None, open_time: datetime = None, id: int = None) -> dict:

        with self.connect() as conn:
            with conn.cursor() as cursor:
                query_parts = ["SELECT * FROM eth_arima_data WHERE 1=1"]
                params = []

                if id is not None:
                    query_parts.append("AND id = %s")
                    params.append(id)
                if timeframe is not None:
                    query_parts.append("AND timeframe = %s")
                    params.append(timeframe)
                if open_time is not None:
                    query_parts.append("AND open_time = %s")
                    params.append(open_time)

                query = " ".join(query_parts)
                cursor.execute(query, params)

                columns = [desc[0] for desc in cursor.description]
                result = cursor.fetchone()

                if result:
                    record = dict(zip(columns, result))

                    # Перетворення JSON рядка у список для significant_lags
                    if record.get('significant_lags'):
                        record['significant_lags'] = json.loads(record['significant_lags'])

                    return record
                return None

    def get_all_eth_arima_data(self) -> list:

        with self.connect() as conn:
            with conn.cursor() as cursor:
                query = "SELECT * FROM eth_arima_data ORDER BY open_time DESC"
                cursor.execute(query)

                columns = [desc[0] for desc in cursor.description]
                results = cursor.fetchall()

                records = []
                for result in results:
                    record = dict(zip(columns, result))

                    # Перетворення JSON рядка у список для significant_lags
                    if record.get('significant_lags'):
                        record['significant_lags'] = json.loads(record['significant_lags'])

                    records.append(record)

                return records

    def get_multiple_btc_arima_data(self, timeframe: str = None, start_time: datetime = None,
                                    end_time: datetime = None, limit: int = None,
                                    is_stationary: bool = None) -> list:

        with self.connect() as conn:
            with conn.cursor() as cursor:
                query_parts = ["SELECT * FROM btc_arima_data WHERE 1=1"]
                params = []

                if timeframe is not None:
                    query_parts.append("AND timeframe = %s")
                    params.append(timeframe)
                if start_time is not None:
                    query_parts.append("AND open_time >= %s")
                    params.append(start_time)
                if end_time is not None:
                    query_parts.append("AND open_time <= %s")
                    params.append(end_time)
                if is_stationary is not None:
                    query_parts.append("AND is_stationary = %s")
                    params.append(is_stationary)

                query_parts.append("ORDER BY open_time DESC")

                if limit is not None:
                    query_parts.append("LIMIT %s")
                    params.append(limit)

                query = " ".join(query_parts)
                cursor.execute(query, params)

                columns = [desc[0] for desc in cursor.description]
                results = cursor.fetchall()

                records = []
                for result in results:
                    record = dict(zip(columns, result))

                    # Перетворення JSON рядка у список для significant_lags
                    if record.get('significant_lags'):
                        record['significant_lags'] = json.loads(record['significant_lags'])

                    records.append(record)

                return records

    def get_multiple_sol_arima_data(self, timeframe: str = None, start_time: datetime = None,
                                    end_time: datetime = None, limit: int = None,
                                    is_stationary: bool = None) -> list:

        with self.connect() as conn:
            with conn.cursor() as cursor:
                query_parts = ["SELECT * FROM sol_arima_data WHERE 1=1"]
                params = []

                if timeframe is not None:
                    query_parts.append("AND timeframe = %s")
                    params.append(timeframe)
                if start_time is not None:
                    query_parts.append("AND open_time >= %s")
                    params.append(start_time)
                if end_time is not None:
                    query_parts.append("AND open_time <= %s")
                    params.append(end_time)
                if is_stationary is not None:
                    query_parts.append("AND is_stationary = %s")
                    params.append(is_stationary)

                query_parts.append("ORDER BY open_time DESC")

                if limit is not None:
                    query_parts.append("LIMIT %s")
                    params.append(limit)

                query = " ".join(query_parts)
                cursor.execute(query, params)

                columns = [desc[0] for desc in cursor.description]
                results = cursor.fetchall()

                records = []
                for result in results:
                    record = dict(zip(columns, result))

                    # Перетворення JSON рядка у список для significant_lags
                    if record.get('significant_lags'):
                        record['significant_lags'] = json.loads(record['significant_lags'])

                    records.append(record)

                return records

    def get_multiple_eth_arima_data(self, timeframe: str = None, start_time: datetime = None,
                                    end_time: datetime = None, limit: int = None,
                                    is_stationary: bool = None) -> list:

        with self.connect() as conn:
            with conn.cursor() as cursor:
                query_parts = ["SELECT * FROM eth_arima_data WHERE 1=1"]
                params = []

                if timeframe is not None:
                    query_parts.append("AND timeframe = %s")
                    params.append(timeframe)
                if start_time is not None:
                    query_parts.append("AND open_time >= %s")
                    params.append(start_time)
                if end_time is not None:
                    query_parts.append("AND open_time <= %s")
                    params.append(end_time)
                if is_stationary is not None:
                    query_parts.append("AND is_stationary = %s")
                    params.append(is_stationary)

                query_parts.append("ORDER BY open_time DESC")

                if limit is not None:
                    query_parts.append("LIMIT %s")
                    params.append(limit)

                query = " ".join(query_parts)
                cursor.execute(query, params)

                columns = [desc[0] for desc in cursor.description]
                results = cursor.fetchall()

                records = []
                for result in results:
                    record = dict(zip(columns, result))

                    # Перетворення JSON рядка у список для significant_lags
                    if record.get('significant_lags'):
                        record['significant_lags'] = json.loads(record['significant_lags'])

                    records.append(record)

                return records

    # --------- VOLATILITY METRICS ---------

    def save_volatility_metrics(self, metrics: Dict[str, Any]) -> int:
        """
        Зберігає метрики волатильності в таблицю volatility_metrics.

        Args:
            metrics: Словник з даними метрик волатильності

        Returns:
            id: Ідентифікатор збереженого запису
        """
        if not self.conn:
            self.connect()

        try:
            query = """
                    INSERT INTO volatility_metrics (symbol, timeframe, timestamp, hist_vol_7d, hist_vol_14d, \
                                                    hist_vol_30d, hist_vol_60d, parkinson_vol, garman_klass_vol, \
                                                    yang_zhang_vol, vol_of_vol, regime_id, is_breakout) \
                    VALUES (%(symbol)s, %(timeframe)s, %(timestamp)s, %(hist_vol_7d)s, %(hist_vol_14d)s, \
                            %(hist_vol_30d)s, %(hist_vol_60d)s, %(parkinson_vol)s, %(garman_klass_vol)s, \
                            %(yang_zhang_vol)s, %(vol_of_vol)s, %(regime_id)s, \
                            %(is_breakout)s) ON CONFLICT (symbol, timeframe, timestamp) DO \
                    UPDATE \
                    SET
                        hist_vol_7d = EXCLUDED.hist_vol_7d, hist_vol_14d = EXCLUDED.hist_vol_14d, hist_vol_30d = EXCLUDED.hist_vol_30d, hist_vol_60d = EXCLUDED.hist_vol_60d, parkinson_vol = EXCLUDED.parkinson_vol, garman_klass_vol = EXCLUDED.garman_klass_vol, yang_zhang_vol = EXCLUDED.yang_zhang_vol, vol_of_vol = EXCLUDED.vol_of_vol, regime_id = EXCLUDED.regime_id, is_breakout = EXCLUDED.is_breakout
                        RETURNING id; \
                    """

            self.cursor.execute(query, metrics)
            record_id = self.cursor.fetchone()[0]
            self.conn.commit()
            return record_id

        except psycopg2.Error as e:
            self.conn.rollback()
            print(f"Помилка збереження метрик волатильності: {e}")
            return -1

    def get_volatility_metrics(self, symbol: str, timeframe: str,
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None,
                               limit: int = 100) -> List[Dict[str, Any]]:
        """
        Отримує метрики волатильності з таблиці volatility_metrics.

        Args:
            symbol: Символ криптовалюти
            timeframe: Часовий інтервал
            start_date: Початкова дата вибірки (опціонально)
            end_date: Кінцева дата вибірки (опціонально)
            limit: Обмеження кількості результатів (за замовчуванням 100)

        Returns:
            List[Dict]: Список словників з метриками волатильності
        """
        if not self.conn:
            self.connect()

        query = """
                SELECT * \
                FROM volatility_metrics
                WHERE symbol = %(symbol)s \
                  AND timeframe = %(timeframe)s \
                """

        params = {"symbol": symbol, "timeframe": timeframe}

        if start_date:
            query += " AND timestamp >= %(start_date)s"
            params["start_date"] = start_date

        if end_date:
            query += " AND timestamp <= %(end_date)s"
            params["end_date"] = end_date

        query += " ORDER BY timestamp DESC LIMIT %(limit)s"
        params["limit"] = limit

        try:
            self.cursor.execute(query, params)
            results = self.cursor.fetchall()
            return [dict(row) for row in results]
        except psycopg2.Error as e:
            print(f"Помилка отримання метрик волатильності: {e}")
            return []

    # --------- VOLATILITY MODELS ---------

    def save_volatility_model(self, model_data: Dict[str, Any]) -> int:
        """
        Зберігає модель волатильності в таблицю volatility_models.

        Args:
            model_data: Словник з даними моделі

        Returns:
            id: Ідентифікатор збереженого запису
        """
        if not self.conn:
            self.connect()

        try:
            # Перетворення параметрів у JSON, якщо вони є словником
            if isinstance(model_data.get("parameters"), dict):
                model_data["parameters"] = json.dumps(model_data["parameters"])

            query = """
                    INSERT INTO volatility_models (symbol, timeframe, model_type, p, q, created_at, updated_at, \
                                                   parameters, aic, bic, log_likelihood, serialized_model) \
                    VALUES (%(symbol)s, %(timeframe)s, %(model_type)s, %(p)s, %(q)s, \
                            COALESCE(%(created_at)s, NOW()), %(updated_at)s, \
                            %(parameters)s, %(aic)s, %(bic)s, %(log_likelihood)s, \
                            %(serialized_model)s) ON CONFLICT (symbol, timeframe, model_type, p, q) DO \
                    UPDATE \
                    SET
                        updated_at = NOW(), parameters = EXCLUDED.parameters, aic = EXCLUDED.aic, bic = EXCLUDED.bic, log_likelihood = EXCLUDED.log_likelihood, serialized_model = EXCLUDED.serialized_model
                        RETURNING id; \
                    """

            self.cursor.execute(query, model_data)
            record_id = self.cursor.fetchone()[0]
            self.conn.commit()
            return record_id

        except psycopg2.Error as e:
            self.conn.rollback()
            print(f"Помилка збереження моделі волатильності: {e}")
            return -1

    def get_volatility_model(self, symbol: str, timeframe: str,
                             model_type: str = None, p: int = None,
                             q: int = None) -> Optional[Dict[str, Any]]:
        """
        Отримує модель волатильності з таблиці volatility_models.

        Args:
            symbol: Символ криптовалюти
            timeframe: Часовий інтервал
            model_type: Тип моделі (опціонально)
            p: Параметр p моделі (опціонально)
            q: Параметр q моделі (опціонально)

        Returns:
            Dict: Словник з даними моделі або None, якщо модель не знайдено
        """
        if not self.conn:
            self.connect()

        query = """
                SELECT * \
                FROM volatility_models
                WHERE symbol = %(symbol)s \
                  AND timeframe = %(timeframe)s \
                """

        params = {"symbol": symbol, "timeframe": timeframe}

        if model_type:
            query += " AND model_type = %(model_type)s"
            params["model_type"] = model_type

        if p is not None:
            query += " AND p = %(p)s"
            params["p"] = p

        if q is not None:
            query += " AND q = %(q)s"
            params["q"] = q

        query += " ORDER BY updated_at DESC LIMIT 1"

        try:
            self.cursor.execute(query, params)
            result = self.cursor.fetchone()

            if result:
                model_data = dict(result)
                # Перетворення JSON в словник
                if model_data.get("parameters"):
                    model_data["parameters"] = json.loads(model_data["parameters"])
                return model_data
            return None
        except psycopg2.Error as e:
            print(f"Помилка отримання моделі волатильності: {e}")
            return None

    # --------- VOLATILITY REGIMES ---------

    def save_volatility_regime(self, regime_data: Dict[str, Any]) -> int:
        """
        Зберігає режим волатильності в таблицю volatility_regimes.

        Args:
            regime_data: Словник з даними режиму волатильності

        Returns:
            id: Ідентифікатор збереженого запису
        """
        if not self.conn:
            self.connect()

        try:
            # Перетворення параметрів у JSON, якщо вони є словником
            if isinstance(regime_data.get("regime_parameters"), dict):
                regime_data["regime_parameters"] = json.dumps(regime_data["regime_parameters"])

            query = """
                    INSERT INTO volatility_regimes (symbol, timeframe, method, n_regimes, created_at, \
                                                    regime_thresholds, regime_centroids, regime_labels, \
                                                    regime_parameters) \
                    VALUES (%(symbol)s, %(timeframe)s, %(method)s, %(n_regimes)s, \
                            COALESCE(%(created_at)s, NOW()), \
                            %(regime_thresholds)s, %(regime_centroids)s, %(regime_labels)s, \
                            %(regime_parameters)s) ON CONFLICT (symbol, timeframe, method, n_regimes) DO \
                    UPDATE \
                    SET
                        regime_thresholds = EXCLUDED.regime_thresholds, regime_centroids = EXCLUDED.regime_centroids, regime_labels = EXCLUDED.regime_labels, regime_parameters = EXCLUDED.regime_parameters
                        RETURNING id; \
                    """

            self.cursor.execute(query, regime_data)
            record_id = self.cursor.fetchone()[0]
            self.conn.commit()
            return record_id

        except psycopg2.Error as e:
            self.conn.rollback()
            print(f"Помилка збереження режиму волатильності: {e}")
            return -1

    def get_volatility_regime(self, symbol: str, timeframe: str,
                              method: str = None,
                              n_regimes: int = None) -> Optional[Dict[str, Any]]:
        """
        Отримує режим волатильності з таблиці volatility_regimes.

        Args:
            symbol: Символ криптовалюти
            timeframe: Часовий інтервал
            method: Метод виявлення режимів (опціонально)
            n_regimes: Кількість режимів (опціонально)

        Returns:
            Dict: Словник з даними режиму або None, якщо режим не знайдено
        """
        if not self.conn:
            self.connect()

        query = """
                SELECT * \
                FROM volatility_regimes
                WHERE symbol = %(symbol)s \
                  AND timeframe = %(timeframe)s \
                """

        params = {"symbol": symbol, "timeframe": timeframe}

        if method:
            query += " AND method = %(method)s"
            params["method"] = method

        if n_regimes is not None:
            query += " AND n_regimes = %(n_regimes)s"
            params["n_regimes"] = n_regimes

        query += " ORDER BY created_at DESC LIMIT 1"

        try:
            self.cursor.execute(query, params)
            result = self.cursor.fetchone()

            if result:
                regime_data = dict(result)
                # Перетворення JSON в словник
                if regime_data.get("regime_parameters"):
                    regime_data["regime_parameters"] = json.loads(regime_data["regime_parameters"])
                return regime_data
            return None
        except psycopg2.Error as e:
            print(f"Помилка отримання режиму волатильності: {e}")
            return None

    # --------- VOLATILITY FEATURES ---------

    def save_volatility_features(self, features_data: Dict[str, Any]) -> int:
        """
        Зберігає ознаки волатильності в таблицю volatility_features.

        Args:
            features_data: Словник з даними ознак

        Returns:
            id: Ідентифікатор збереженого запису
        """
        if not self.conn:
            self.connect()

        try:
            # Перетворення features у JSON, якщо вони є словником
            if isinstance(features_data.get("features"), dict):
                features_data["features"] = json.dumps(features_data["features"])

            query = """
                    INSERT INTO volatility_features (symbol, timeframe, timestamp, features) \
                    VALUES (%(symbol)s, %(timeframe)s, %(timestamp)s, \
                            %(features)s) ON CONFLICT (symbol, timeframe, timestamp) DO \
                    UPDATE \
                    SET
                        features = EXCLUDED.features
                        RETURNING id; \
                    """

            self.cursor.execute(query, features_data)
            record_id = self.cursor.fetchone()[0]
            self.conn.commit()
            return record_id

        except psycopg2.Error as e:
            self.conn.rollback()
            print(f"Помилка збереження ознак волатильності: {e}")
            return -1

    def get_volatility_features(self, symbol: str, timeframe: str,
                                start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None,
                                limit: int = 100) -> List[Dict[str, Any]]:

        if not self.conn:
            self.connect()

        query = """
                SELECT * \
                FROM volatility_features
                WHERE symbol = %(symbol)s \
                  AND timeframe = %(timeframe)s \
                """

        params = {"symbol": symbol, "timeframe": timeframe}

        if start_date:
            query += " AND timestamp >= %(start_date)s"
            params["start_date"] = start_date

        if end_date:
            query += " AND timestamp <= %(end_date)s"
            params["end_date"] = end_date

        query += " ORDER BY timestamp DESC LIMIT %(limit)s"
        params["limit"] = limit

        try:
            self.cursor.execute(query, params)
            results = self.cursor.fetchall()

            feature_list = []
            for row in results:
                feature_data = dict(row)
                # Перетворення JSON в словник
                if feature_data.get("features"):
                    feature_data["features"] = json.loads(feature_data["features"])
                feature_list.append(feature_data)

            return feature_list
        except psycopg2.Error as e:
            print(f"Помилка отримання ознак волатильності: {e}")
            return []

    # --------- CROSS ASSET VOLATILITY ---------

    def save_cross_asset_volatility(self, cross_vol_data: Dict[str, Any]) -> int:

        """
        Зберігає дані про кросс-активну волатильність в таблицю cross_asset_volatility.

        Args:
            cross_vol_data: Словник з даними кросс-активної волатильності

        Returns:
            id: Ідентифікатор збереженого запису
        """
        if not self.conn:
            self.connect()

        try:
            query = """
                    INSERT INTO cross_asset_volatility (base_symbol, compared_symbol, timeframe, timestamp, correlation, \
                                                        lag) \
                    VALUES (%(base_symbol)s, %(compared_symbol)s, %(timeframe)s, %(timestamp)s, \
                            %(correlation)s, \
                            %(lag)s) ON CONFLICT (base_symbol, compared_symbol, timeframe, timestamp, lag) DO \
                    UPDATE \
                    SET
                        correlation = EXCLUDED.correlation
                        RETURNING id; \
                    """

            self.cursor.execute(query, cross_vol_data)
            record_id = self.cursor.fetchone()[0]
            self.conn.commit()
            return record_id

        except psycopg2.Error as e:
            self.conn.rollback()
            print(f"Помилка збереження даних кросс-активної волатильності: {e}")
            return -1

    def get_cross_asset_volatility(self, base_symbol: str, compared_symbol: str,
                                   timeframe: str, lag: Optional[int] = None,
                                   start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None,
                                   limit: int = 100) -> List[Dict[str, Any]]:
        """
        Отримує дані про кросс-активну волатильність з таблиці cross_asset_volatility.

        Args:
            base_symbol: Базовий символ
            compared_symbol: Символ порівняння
            timeframe: Часовий інтервал
            lag: Лаг кореляції (опціонально)
            start_date: Початкова дата вибірки (опціонально)
            end_date: Кінцева дата вибірки (опціонально)
            limit: Обмеження кількості результатів (за замовчуванням 100)

        Returns:
            List[Dict]: Список словників з даними кросс-активної волатильності
        """
        if not self.conn:
            self.connect()

        query = """
                SELECT * \
                FROM cross_asset_volatility
                WHERE base_symbol = %(base_symbol)s \
                  AND compared_symbol = %(compared_symbol)s
                  AND timeframe = %(timeframe)s \
                """

        params = {"base_symbol": base_symbol, "compared_symbol": compared_symbol, "timeframe": timeframe}

        if lag is not None:
            query += " AND lag = %(lag)s"
            params["lag"] = lag

        if start_date:
            query += " AND timestamp >= %(start_date)s"
            params["start_date"] = start_date

        if end_date:
            query += " AND timestamp <= %(end_date)s"
            params["end_date"] = end_date

        query += " ORDER BY timestamp DESC LIMIT %(limit)s"
        params["limit"] = limit

        try:
            self.cursor.execute(query, params)
            results = self.cursor.fetchall()
            return [dict(row) for row in results]
        except psycopg2.Error as e:
            print(f"Помилка отримання даних кросс-активної волатильності: {e}")
            return []

    # --------- VIEWS ---------

    def get_current_volatility_regimes(self) -> List[Dict[str, Any]]:
        """
        Отримує поточні режими волатильності з представлення current_volatility_regimes.

        Returns:
            List[Dict]: Список словників з поточними режимами волатильності
        """
        if not self.conn:
            self.connect()

        try:
            query = "SELECT * FROM current_volatility_regimes"
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            return [dict(row) for row in results]
        except psycopg2.Error as e:
            print(f"Помилка отримання поточних режимів волатильності: {e}")
            return []

    def get_volatility_metrics_comparison(self, symbol: str = None,
                                          timeframe: str = None,
                                          limit: int = 100) -> List[Dict[str, Any]]:
        """
        Отримує порівняння метрик волатильності з представлення volatility_metrics_comparison.

        Args:
            symbol: Символ криптовалюти (опціонально)
            timeframe: Часовий інтервал (опціонально)
            limit: Обмеження кількості результатів (за замовчуванням 100)

        Returns:
            List[Dict]: Список словників з порівнянням метрик волатильності
        """
        if not self.conn:
            self.connect()

        query = "SELECT * FROM volatility_metrics_comparison"
        params = {}

        conditions = []
        if symbol:
            conditions.append("symbol = %(symbol)s")
            params["symbol"] = symbol

        if timeframe:
            conditions.append("timeframe = %(timeframe)s")
            params["timeframe"] = timeframe

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY timestamp DESC LIMIT %(limit)s"
        params["limit"] = limit

        try:
            self.cursor.execute(query, params)
            results = self.cursor.fetchall()
            return [dict(row) for row in results]
        except psycopg2.Error as e:
            print(f"Помилка отримання порівняння метрик волатильності: {e}")
            return []

    # --------- UTILITY METHODS ---------

    def execute_custom_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:

        if not self.conn:
            self.connect()

        try:
            self.cursor.execute(query, params or {})

            if query.strip().upper().startswith("SELECT"):
                results = self.cursor.fetchall()
                return [dict(row) for row in results]
            else:
                self.conn.commit()
                return []
        except psycopg2.Error as e:
            self.conn.rollback()
            print(f"Помилка виконання запиту: {e}")
            return []


    def save_trend_analysis(
            self,
            symbol: str,
            timeframe: str,
            analysis_date: datetime,
            trend_data: Dict[str, Any]
    ) -> bool:
        """Зберігає аналіз тренду в базу даних"""
        try:
            # Перевірка наявності необхідних полів
            required_fields = ['trend_type', 'trend_strength']
            for field in required_fields:
                if field not in trend_data:
                    trend_data[field] = None

            # Підготовка JSONB полів
            support_levels = json.dumps(trend_data.get('support_levels', []))
            resistance_levels = json.dumps(trend_data.get('resistance_levels', []))
            fibonacci_levels = json.dumps(trend_data.get('fibonacci_levels', {}))
            swing_points = json.dumps(trend_data.get('swing_points', {'highs': [], 'lows': []}))
            detected_patterns = json.dumps(trend_data.get('detected_patterns', []))
            additional_metrics = json.dumps(trend_data.get('additional_metrics', {}))

            # SQL запит для вставки або оновлення даних (upsert)
            query = """
                    INSERT INTO trend_analysis (symbol, timeframe, analysis_date, trend_type, trend_strength,
                                                support_levels, resistance_levels, fibonacci_levels, swing_points,
                                                detected_patterns, market_regime, additional_metrics)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, \
                            %s) ON CONFLICT (symbol, timeframe, analysis_date) DO
                    UPDATE
                        SET
                            trend_type = EXCLUDED.trend_type,
                        trend_strength = EXCLUDED.trend_strength,
                        support_levels = EXCLUDED.support_levels,
                        resistance_levels = EXCLUDED.resistance_levels,
                        fibonacci_levels = EXCLUDED.fibonacci_levels,
                        swing_points = EXCLUDED.swing_points,
                        detected_patterns = EXCLUDED.detected_patterns,
                        market_regime = EXCLUDED.market_regime,
                        additional_metrics = EXCLUDED.additional_metrics
                        RETURNING id
                    """

            self.cursor.execute(query, (
                symbol,
                timeframe,
                analysis_date,
                trend_data.get('trend_type'),
                trend_data.get('trend_strength'),
                support_levels,
                resistance_levels,
                fibonacci_levels,
                swing_points,
                detected_patterns,
                trend_data.get('market_regime'),
                additional_metrics
            ))

            result = self.cursor.fetchone()
            self.conn.commit()

            return True

        except psycopg2.Error as e:
            if self.conn:
                self.conn.rollback()
            print(f"Помилка при збереженні аналізу тренду: {e}")
            return False
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            print(f"Загальна помилка при збереженні аналізу тренду: {e}")
            return False


    def get_trend_analysis(
            self,
            symbol: str,
            timeframe: str,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None,
            latest_only: bool = False
    ) -> Union[Dict[str, Any], List[Dict[str, Any]], None]:
        """Отримує аналіз тренду з бази даних"""
        try:
            if latest_only:
                # Отримання тільки останнього аналізу
                query = """
                        SELECT *
                        FROM trend_analysis
                        WHERE symbol = %s
                          AND timeframe = %s
                        ORDER BY analysis_date DESC LIMIT 1
                        """
                params = (symbol, timeframe)
            else:
                # Формування запиту з можливою фільтрацією за датами
                query = """
                        SELECT *
                        FROM trend_analysis
                        WHERE symbol = %s
                          AND timeframe = %s
                        """
                params = [symbol, timeframe]

                if start_date:
                    query += " AND analysis_date >= %s"
                    params.append(start_date)

                if end_date:
                    query += " AND analysis_date <= %s"
                    params.append(end_date)

                query += " ORDER BY analysis_date DESC"

            self.cursor.execute(query, params)
            results = self.cursor.fetchall()

            if not results:
                return None

            # Перетворення результатів в словники з правильними типами даних
            trend_analyses = []
            for row in results:
                row_dict = dict(row)

                # Перетворення JSON рядків у словники/списки Python
                for json_field in ['support_levels', 'resistance_levels', 'fibonacci_levels',
                                   'swing_points', 'detected_patterns', 'additional_metrics']:
                    if row_dict[json_field]:
                        row_dict[json_field] = json.loads(row_dict[json_field])
                    else:
                        # Встановлення значень за замовчуванням для порожніх полів
                        if json_field in ['support_levels', 'resistance_levels', 'detected_patterns']:
                            row_dict[json_field] = []
                        elif json_field == 'swing_points':
                            row_dict[json_field] = {'highs': [], 'lows': []}
                        else:
                            row_dict[json_field] = {}

                trend_analyses.append(row_dict)

            # Повертаємо один елемент або список в залежності від параметра latest_only
            if latest_only and trend_analyses:
                return trend_analyses[0]

            return trend_analyses

        except psycopg2.Error as e:
            print(f"Помилка при отриманні аналізу тренду: {e}")
            return None
        except Exception as e:
            print(f"Загальна помилка при отриманні аналізу тренду: {e}")
            return None




