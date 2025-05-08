import os
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union

import joblib
import numpy as np
import psycopg2
import pandas as pd
from datetime import datetime
from psycopg2.extras import RealDictCursor, execute_batch
from utils.config import *
import json

class NewsDatabase:
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
    def save_topic_models(self):
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.topic_model_dir, exist_ok=True)

            # Save vectorizer
            if self.vectorizer:
                joblib.dump(self.vectorizer, os.path.join(self.topic_model_dir, 'vectorizer.pkl'))

            # Save LDA model
            if self.lda_model:
                joblib.dump(self.lda_model, os.path.join(self.topic_model_dir, 'lda_model.pkl'))

            # Save NMF model
            if self.nmf_model:
                joblib.dump(self.nmf_model, os.path.join(self.topic_model_dir, 'nmf_model.pkl'))

            # Save KMeans model
            if self.kmeans_model:
                joblib.dump(self.kmeans_model, os.path.join(self.topic_model_dir, 'kmeans_model.pkl'))

            # Save topic words
            if self.topic_words:
                joblib.dump(self.topic_words, os.path.join(self.topic_model_dir, 'topic_words.pkl'))

            self.logger.info("Topic models saved to disk")
        except Exception as e:
            self.logger.error(f"Error saving topic models: {e}")