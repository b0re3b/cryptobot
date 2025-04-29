import os
import pickle
from typing import Dict, List, Any, Optional

import numpy as np
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

    def save_model_metadata(self, model_key: str, symbol: str, model_type: str,
                            interval: str, start_date: datetime, end_date: datetime,
                            description: str = None) -> int:
        """
        Збереження метаданих моделі в базу даних.

        Args:
            model_key: Унікальний ключ моделі
            symbol: Символ криптовалюти
            model_type: Тип моделі ('arima', 'sarima', etc.)
            interval: Інтервал даних ('1m', '5m', '15m', '1h', '4h', '1d')
            start_date: Початкова дата даних, на яких навчалася модель
            end_date: Кінцева дата даних, на яких навчалася модель
            description: Опис моделі

        Returns:
            ID створеного запису моделі
        """
        try:
            conn = self.connect()
            with conn.cursor() as cursor:
                query = """
                        INSERT INTO time_series_models
                        (model_key, symbol, model_type, interval, start_date, end_date, description)
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
                cursor.execute(query, (model_key, symbol, model_type, interval,
                                       start_date, end_date, description))
                model_id = cursor.fetchone()[0]
                conn.commit()
                self.logger.info(f"Збережено метадані моделі {model_key} з ID {model_id}")
                return model_id
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Помилка при збереженні метаданих моделі: {str(e)}")
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
            conn = self.connect()
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
                self.logger.info(f"Збережено параметри для моделі з ID {model_id}")
                return True
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Помилка при збереженні параметрів моделі: {str(e)}")
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
            conn = self.connect()
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
                self.logger.info(f"Збережено метрики для моделі з ID {model_id}")
                return True
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Помилка при збереженні метрик моделі: {str(e)}")
            return False

    def save_model_forecasts(self, model_id: int, forecasts: pd.Series,
                             lower_bounds: pd.Series = None, upper_bounds: pd.Series = None,
                             confidence_level: float = 0.95) -> bool:
        """
        Збереження прогнозів моделі.

        Args:
            model_id: ID моделі
            forecasts: Серія прогнозів з датами як індексом
            lower_bounds: Нижні межі довірчого інтервалу
            upper_bounds: Верхні межі довірчого інтервалу
            confidence_level: Рівень довіри для інтервалів

        Returns:
            Успішність операції
        """
        try:
            conn = self.connect()
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
                self.logger.info(f"Збережено прогнози для моделі з ID {model_id}")
                return True
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Помилка при збереженні прогнозів моделі: {str(e)}")
            return False

    def save_model_binary(self, model_id: int, model_obj: Any) -> bool:
        """
        Збереження серіалізованої моделі в базу даних.

        Args:
            model_id: ID моделі
            model_obj: Об'єкт моделі для серіалізації

        Returns:
            Успішність операції
        """
        try:
            conn = self.connect()
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
                self.logger.info(f"Збережено бінарні дані моделі з ID {model_id}")
                return True
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Помилка при збереженні бінарних даних моделі: {str(e)}")
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
            conn = self.connect()
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
                self.logger.info(f"Збережено перетворення для моделі з ID {model_id}")
                return True
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Помилка при збереженні перетворень даних: {str(e)}")
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
            conn = self.connect()
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                query = "SELECT * FROM time_series_models WHERE model_key = %s"
                cursor.execute(query, (model_key,))
                model_data = cursor.fetchone()

                if model_data:
                    return dict(model_data)
                return None
        except Exception as e:
            self.logger.error(f"Помилка при отриманні моделі за ключем: {str(e)}")
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
            conn = self.connect()
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
            self.logger.error(f"Помилка при отриманні параметрів моделі: {str(e)}")
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
            conn = self.connect()
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
            self.logger.error(f"Помилка при отриманні метрик моделі: {str(e)}")
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
            conn = self.connect()
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
            self.logger.error(f"Помилка при отриманні прогнозів моделі: {str(e)}")
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
            conn = self.connect()
            with conn.cursor() as cursor:
                query = "SELECT model_binary FROM model_binary_data WHERE model_id = %s"
                cursor.execute(query, (model_id,))
                row = cursor.fetchone()

                if row:
                    model_binary = row[0]
                    model_obj = pickle.loads(model_binary)
                    self.logger.info(f"Завантажено бінарні дані моделі з ID {model_id}")
                    return model_obj

                self.logger.warning(f"Бінарні дані моделі з ID {model_id} не знайдено")
                return None
        except Exception as e:
            self.logger.error(f"Помилка при завантаженні бінарних даних моделі: {str(e)}")
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
            conn = self.connect()
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
            self.logger.error(f"Помилка при отриманні перетворень даних: {str(e)}")
            return []

    def delete_model(self, model_id: int) -> bool:
        """
        Видалення моделі та всіх пов'язаних даних.

        Args:
            model_id: ID моделі

        Returns:
            Успішність операції
        """
        try:
            conn = self.connect()
            with conn.cursor() as cursor:
                # Використовуємо каскадне видалення, визначене в схемі БД
                query = "DELETE FROM time_series_models WHERE model_id = %s"
                cursor.execute(query, (model_id,))
                conn.commit()
                self.logger.info(f"Видалено модель з ID {model_id} та пов'язані дані")
                return True
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Помилка при видаленні моделі: {str(e)}")
            return False

    def get_models_by_symbol(self, symbol: str, interval: str = None, active_only: bool = True) -> List[Dict]:
        """
        Отримання всіх моделей для певного символу.

        Args:
            symbol: Символ криптовалюти
            interval: Інтервал даних (якщо None, всі інтервали)
            active_only: Фільтрувати тільки активні моделі

        Returns:
            Список словників з інформацією про моделі
        """
        try:
            conn = self.connect()
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                query = "SELECT * FROM time_series_models WHERE symbol = %s"
                params = [symbol]

                if interval:
                    query += " AND interval = %s"
                    params.append(interval)

                if active_only:
                    query += " AND is_active = TRUE"

                query += " ORDER BY created_at DESC"
                cursor.execute(query, params)

                models = []
                for row in cursor.fetchall():
                    models.append(dict(row))

                return models
        except Exception as e:
            self.logger.error(f"Помилка при отриманні моделей за символом: {str(e)}")
            return []

    def get_latest_model_by_symbol(self, symbol: str, model_type: str = None,
                                   interval: str = None) -> Optional[Dict]:
        """
        Отримання останньої моделі для певного символу.

        Args:
            symbol: Символ криптовалюти
            model_type: Тип моделі (якщо None, будь-який тип)
            interval: Інтервал даних (якщо None, будь-який інтервал)

        Returns:
            Словник з інформацією про модель або None
        """
        try:
            conn = self.connect()
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                query = "SELECT * FROM time_series_models WHERE symbol = %s AND is_active = TRUE"
                params = [symbol]

                if model_type:
                    query += " AND model_type = %s"
                    params.append(model_type)

                if interval:
                    query += " AND interval = %s"
                    params.append(interval)

                query += " ORDER BY updated_at DESC LIMIT 1"
                cursor.execute(query, params)

                model_data = cursor.fetchone()
                if model_data:
                    return dict(model_data)
                return None
        except Exception as e:
            self.logger.error(f"Помилка при отриманні останньої моделі: {str(e)}")
            return None

    def get_model_performance_history(self, model_id: int) -> pd.DataFrame:
        """
        Отримання історії продуктивності моделі по датах тестування.

        Args:
            model_id: ID моделі

        Returns:
            DataFrame з історією метрик моделі
        """
        try:
            conn = self.connect()
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
            self.logger.error(f"Помилка при отриманні історії продуктивності: {str(e)}")
            return pd.DataFrame()


def update_model_status(self, model_id: int, is_active: bool) -> bool:
    """
    Оновлення статусу активності моделі.

    Args:
        model_id: ID моделі
        is_active: Новий статус активності

    Returns:
        Успішність операції
    """
    try:
        conn = self.connect()
        with conn.cursor() as cursor:
            query = "UPDATE time_series_models SET is_active = %s WHERE model_id = %s"
            cursor.execute(query, (is_active, model_id))
            conn.commit()
            self.logger.info(f"Оновлено статус активності для моделі з ID {model_id}: {is_active}")
            return True
    except Exception as e:
        conn.rollback()
        self.logger.error(f"Помилка при оновленні статусу моделі: {str(e)}")
        return False


def compare_model_forecasts(self, model_ids: List[int], start_date: datetime = None,
                            end_date: datetime = None) -> pd.DataFrame:
    """
    Порівняння прогнозів декількох моделей.

    Args:
        model_ids: Список ID моделей для порівняння
        start_date: Початкова дата прогнозів
        end_date: Кінцева дата прогнозів

    Returns:
        DataFrame з прогнозами різних моделей
    """
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
        self.logger.error(f"Помилка при порівнянні прогнозів моделей: {str(e)}")
        return pd.DataFrame()


def get_model_forecast_accuracy(self, model_id: int, actual_data: pd.Series) -> Dict:
    """
    Розрахунок точності прогнозу на основі фактичних даних.

    Args:
        model_id: ID моделі
        actual_data: Серія з фактичними даними (з датою як індексом)

    Returns:
        Словник з метриками точності
    """
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
        self.logger.error(f"Помилка при розрахунку точності прогнозу: {str(e)}")
        return {"error": str(e)}


def get_available_symbols(self) -> List[str]:
    """
    Отримання списку унікальних символів криптовалют з бази даних моделей.

    Returns:
        Список унікальних символів
    """
    try:
        conn = self.connect()
        with conn.cursor() as cursor:
            query = "SELECT DISTINCT symbol FROM time_series_models ORDER BY symbol"
            cursor.execute(query)
            symbols = [row[0] for row in cursor.fetchall()]
            return symbols
    except Exception as e:
        self.logger.error(f"Помилка при отриманні списку символів: {str(e)}")
        return []


def get_model_transformation_pipeline(self, model_id: int) -> List[Dict]:
    """
    Отримання повного ланцюжка перетворень для моделі.

    Args:
        model_id: ID моделі

    Returns:
        Список словників з інформацією про перетворення, впорядкований за порядком виконання
    """
    try:
        transformations = self.get_data_transformations(model_id)
        return transformations
    except Exception as e:
        self.logger.error(f"Помилка при отриманні ланцюжка перетворень: {str(e)}")
        return []


def save_complete_model(self, model_key: str, symbol: str, model_type: str,
                        interval: str, start_date: datetime, end_date: datetime,
                        model_obj: Any, parameters: Dict, metrics: Dict = None,
                        forecasts: pd.Series = None, transformations: List[Dict] = None,
                        lower_bounds: pd.Series = None, upper_bounds: pd.Series = None,
                        description: str = None) -> int:
    """
    Комплексне збереження моделі та всіх пов'язаних даних.

    Args:
        model_key: Унікальний ключ моделі
        symbol: Символ криптовалюти
        model_type: Тип моделі ('arima', 'sarima', etc.)
        interval: Інтервал даних ('1m', '5m', '15m', '1h', '4h', '1d')
        start_date: Початкова дата даних, на яких навчалася модель
        end_date: Кінцева дата даних, на яких навчалася модель
        model_obj: Об'єкт моделі для серіалізації
        parameters: Словник з параметрами моделі
        metrics: Словник з метриками ефективності моделі
        forecasts: Серія з прогнозами
        transformations: Список словників з інформацією про перетворення
        lower_bounds: Нижні межі довірчого інтервалу
        upper_bounds: Верхні межі довірчого інтервалу
        description: Опис моделі

    Returns:
        ID моделі
    """
    try:
        # Зберігаємо метадані моделі
        model_id = self.save_model_metadata(model_key, symbol, model_type, interval,
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

        self.logger.info(f"Комплексне збереження моделі {model_key} з ID {model_id} завершено успішно")
        return model_id
    except Exception as e:
        self.logger.error(f"Помилка при комплексному збереженні моделі: {str(e)}")
        raise


def load_complete_model(self, model_key: str) -> Dict:
    """
    Комплексне завантаження моделі та всіх пов'язаних даних.

    Args:
        model_key: Ключ моделі

    Returns:
        Словник з даними моделі
    """
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

        self.logger.info(f"Комплексне завантаження моделі {model_key} з ID {model_id} завершено успішно")
        return result
    except Exception as e:
        self.logger.error(f"Помилка при комплексному завантаженні моделі: {str(e)}")
        return {"error": str(e)}