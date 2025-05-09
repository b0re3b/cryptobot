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
        """Завантажує схему бази даних з файлу schema.sql"""
        try:
            schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')

            if os.path.exists(schema_path):
                with open(schema_path, 'r') as f:
                    schema_script = f.read()

                # Виконуємо SQL-скрипт схеми
                self.cursor.execute(schema_script)
                self.conn.commit()
                print("Схему бази даних успішно завантажено")
            else:
                print("Файл schema.sql не знайдено")
                raise FileNotFoundError("Файл schema.sql не знайдено")

        except psycopg2.Error as e:
            print(f"Помилка завантаження схеми бази даних: {e}")
            self.conn.rollback()
            raise

    def save_news_source(self, source_data: Dict) -> Dict:

        query = """
                INSERT INTO news_sources (source_name, base_url, is_active)
                VALUES (%s, %s, %s)
                RETURNING source_id, source_name, base_url, is_active, created_at, updated_at \
                """

        source_name = source_data.get('source_name')
        base_url = source_data.get('base_url')
        is_active = source_data.get('is_active', True)

        if not source_name or not base_url:
            raise ValueError("Назва джерела та URL є обов'язковими полями")

        self.cursor.execute(query, (source_name, base_url, is_active))
        result = self.cursor.fetchone()
        self.conn.commit()

        return {
            'source_id': result[0],
            'source_name': result[1],
            'base_url': result[2],
            'is_active': result[3],
            'created_at': result[4],
            'updated_at': result[5]
        }

    def get_news_source_by_name(self, source_name: str) -> Optional[Dict]:
        """
        Отримати джерело новин за назвою

        :param source_name: Назва джерела новин
        :return: Словник з даними джерела або None, якщо джерело не знайдено
        """
        query = """
                SELECT source_id, source_name, base_url, is_active, created_at, updated_at
                FROM news_sources
                WHERE source_name = %s \
                """

        self.cursor.execute(query, (source_name,))
        result = self.cursor.fetchone()

        if result:
            return {
                'source_id': result[0],
                'source_name': result[1],
                'base_url': result[2],
                'is_active': result[3],
                'created_at': result[4],
                'updated_at': result[5]
            }
        return None

    def get_all_news_sources(self, active_only: bool = False) -> List[Dict]:
        """
        Отримати всі джерела новин

        :param active_only: Якщо True, повертає тільки активні джерела
        :return: Список словників з даними джерел
        """
        query = """
                SELECT source_id, source_name, base_url, is_active, created_at, updated_at
                FROM news_sources \
                """

        if active_only:
            query += "WHERE is_active = TRUE "

        query += "ORDER BY source_name"

        self.cursor.execute(query)
        results = self.cursor.fetchall()

        sources = []
        for row in results:
            sources.append({
                'source_id': row[0],
                'source_name': row[1],
                'base_url': row[2],
                'is_active': row[3],
                'created_at': row[4],
                'updated_at': row[5]
            })

        return sources

    def save_news_category(self, category_data: Dict) -> Dict:

        if 'category_id' in category_data and category_data['category_id']:
            # Оновлення існуючої категорії
            query = """
                    UPDATE news_categories
                    SET source_id         = %s,
                        category_name     = %s,
                        category_url_path = %s,
                        is_active         = %s,
                        updated_at        = NOW()
                    WHERE category_id = %s
                    RETURNING category_id, source_id, category_name, category_url_path, is_active, created_at, updated_at \
                    """

            params = (
                category_data.get('source_id'),
                category_data.get('category_name'),
                category_data.get('category_url_path'),
                category_data.get('is_active', True),
                category_data.get('category_id')
            )
        else:
            # Створення нової категорії
            query = """
                    INSERT INTO news_categories (source_id, category_name, category_url_path, is_active)
                    VALUES (%s, %s, %s, %s)
                    RETURNING category_id, source_id, category_name, category_url_path, is_active, created_at, updated_at \
                    """

            params = (
                category_data.get('source_id'),
                category_data.get('category_name'),
                category_data.get('category_url_path'),
                category_data.get('is_active', True)
            )

        try:
            # Перевірка обов'язкових полів
            if not category_data.get('source_id'):
                raise ValueError("source_id є обов'язковим полем")
            if not category_data.get('category_name'):
                raise ValueError("category_name є обов'язковим полем")

            self.cursor.execute(query, params)
            result = self.cursor.fetchone()
            self.conn.commit()

            return dict(result)
        except psycopg2.IntegrityError as e:
            self.conn.rollback()
            if "unique" in str(e).lower():
                raise ValueError(f"Категорія з такою назвою вже існує для цього джерела: {e}")
            raise
        except Exception as e:
            self.conn.rollback()
            print(f"Помилка при збереженні категорії: {e}")
            raise

    def get_news_category_by_id(self, category_id: int) -> Optional[Dict]:

        query = """
                SELECT category_id, source_id, category_name, category_url_path, is_active, created_at, updated_at
                FROM news_categories
                WHERE category_id = %s \
                """

        self.cursor.execute(query, (category_id,))
        result = self.cursor.fetchone()

        if result:
            return dict(result)
        return None

    def get_news_categories_by_source(self, source_id: int, active_only: bool = False) -> List[Dict]:

        query = """
                SELECT category_id, source_id, category_name, category_url_path, is_active, created_at, updated_at
                FROM news_categories
                WHERE source_id = %s \
                """

        if active_only:
            query += " AND is_active = TRUE"

        query += " ORDER BY category_name"

        self.cursor.execute(query, (source_id,))
        results = self.cursor.fetchall()

        categories = []
        for row in results:
            categories.append(dict(row))

        return categories

    def get_all_news_categories(self, active_only: bool = False) -> List[Dict]:

        query = """
                SELECT nc.category_id, \
                       nc.source_id, \
                       nc.category_name, \
                       nc.category_url_path,
                       nc.is_active, \
                       nc.created_at, \
                       nc.updated_at, \
                       ns.source_name
                FROM news_categories nc
                         JOIN news_sources ns ON nc.source_id = ns.source_id \
                """

        if active_only:
            query += " WHERE nc.is_active = TRUE"

        query += " ORDER BY ns.source_name, nc.category_name"

        self.cursor.execute(query)
        results = self.cursor.fetchall()

        categories = []
        for row in results:
            categories.append(dict(row))

        return categories

    def get_news_category_by_source_and_name(self, source_id: int, category_name: str) -> Optional[Dict]:

        query = """
                SELECT category_id, source_id, category_name, category_url_path, is_active, created_at, updated_at
                FROM news_categories
                WHERE source_id = %s \
                  AND category_name = %s \
                """

        self.cursor.execute(query, (source_id, category_name))
        result = self.cursor.fetchone()

        if result:
            return dict(result)
        return None

    def save_news_article(self, article_data: Dict) -> Dict:

        if 'article_id' in article_data and article_data['article_id']:
            # Оновлення існуючої статті
            query = """
                    UPDATE news_articles
                    SET title        = %s,
                        summary      = %s,
                        content      = %s,
                        link         = %s,
                        source_id    = %s,
                        category_id  = %s,
                        published_at = %s
                    WHERE article_id = %s
                    RETURNING article_id, title, summary, content, link, source_id, category_id,
                        published_at, scraped_at \
                    """

            params = (
                article_data.get('title'),
                article_data.get('summary'),
                article_data.get('content'),
                article_data.get('link'),
                article_data.get('source_id'),
                article_data.get('category_id'),
                article_data.get('published_at'),
                article_data.get('article_id')
            )
        else:
            # Створення нової статті
            query = """
                    INSERT INTO news_articles
                        (title, summary, content, link, source_id, category_id, published_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING article_id, title, summary, content, link, source_id, category_id,
                        published_at, scraped_at \
                    """

            params = (
                article_data.get('title'),
                article_data.get('summary'),
                article_data.get('content'),
                article_data.get('link'),
                article_data.get('source_id'),
                article_data.get('category_id'),
                article_data.get('published_at', datetime.now())
            )

        try:
            # Перевірка обов'язкових полів
            if not article_data.get('title'):
                raise ValueError("title є обов'язковим полем")
            if not article_data.get('link'):
                raise ValueError("link є обов'язковим полем")
            if not article_data.get('source_id'):
                raise ValueError("source_id є обов'язковим полем")

            self.cursor.execute(query, params)
            result = self.cursor.fetchone()
            self.conn.commit()

            return dict(result)
        except psycopg2.IntegrityError as e:
            self.conn.rollback()
            if "unique" in str(e).lower():
                # Якщо стаття з таким посиланням вже існує, повернемо її
                self.cursor.execute(
                    "SELECT * FROM news_articles WHERE link = %s",
                    (article_data.get('link'),)
                )
                existing_article = self.cursor.fetchone()
                if existing_article:
                    return dict(existing_article)
                raise ValueError(f"Стаття з таким посиланням вже існує: {e}")
            raise
        except Exception as e:
            self.conn.rollback()
            print(f"Помилка при збереженні статті: {e}")
            raise

    def get_all_articles(self,
                source_id: Optional[int] = None,
                category_id: Optional[int] = None,
                start_date: Optional[datetime] = None,
                end_date: Optional[datetime] = None,
                limit: int = 100,
                offset: int = 0) -> List[Dict]:

        query = """
                SELECT a.article_id, \
                       a.title, \
                       a.summary, \
                       a.content, \
                       a.link,
                       a.source_id, \
                       a.category_id, \
                       a.published_at, \
                       a.scraped_at,
                       s.source_name, \
                       c.category_name
                FROM news_articles a
                         JOIN news_sources s ON a.source_id = s.source_id
                         LEFT JOIN news_categories c ON a.category_id = c.category_id
                WHERE 1 = 1 \
                """

        params = []

        if source_id:
            query += " AND a.source_id = %s"
            params.append(source_id)

        if category_id:
            query += " AND a.category_id = %s"
            params.append(category_id)

        if start_date:
            query += " AND a.published_at >= %s"
            params.append(start_date)

        if end_date:
            query += " AND a.published_at <= %s"
            params.append(end_date)

        query += " ORDER BY a.published_at DESC"
        query += " LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        self.cursor.execute(query, params)
        results = self.cursor.fetchall()

        return [dict(row) for row in results]

    def save_news_sentiment(self, sentiment_data: Dict) -> Dict:

        if not sentiment_data.get('article_id'):
            raise ValueError("article_id є обов'язковим полем")

        # Спроба оновити існуючий запис, якщо такий існує
        query = """
                INSERT INTO news_sentiment_analysis
                    (article_id, sentiment_score, sentiment_magnitude, sentiment_label)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (article_id)
                    DO UPDATE SET sentiment_score     = EXCLUDED.sentiment_score,
                                  sentiment_magnitude = EXCLUDED.sentiment_magnitude,
                                  sentiment_label     = EXCLUDED.sentiment_label,
                                  processed_at        = NOW()
                RETURNING
                    sentiment_id, article_id, sentiment_score,
                    sentiment_magnitude, sentiment_label, processed_at \
                """

        params = (
            sentiment_data.get('article_id'),
            sentiment_data.get('sentiment_score'),
            sentiment_data.get('sentiment_magnitude'),
            sentiment_data.get('sentiment_label')
        )

        try:
            self.cursor.execute(query, params)
            result = self.cursor.fetchone()
            self.conn.commit()

            return dict(result)
        except Exception as e:
            self.conn.rollback()
            print(f"Помилка при збереженні аналізу настроїв: {e}")
            raise

    def get_news_sentiment(self,
                article_ids: Optional[List[int]] = None,
                sentiment_label: Optional[str] = None,
                min_score: Optional[float] = None,
                max_score: Optional[float] = None,
                start_date: Optional[datetime] = None,
                end_date: Optional[datetime] = None,
                limit: int = 100,
                offset: int = 0) -> List[Dict]:

        query = """
                SELECT s.sentiment_id, \
                       s.article_id, \
                       s.sentiment_score,
                       s.sentiment_magnitude, \
                       s.sentiment_label, \
                       s.processed_at,
                       a.title, \
                       a.link, \
                       a.published_at
                FROM news_sentiment_analysis s
                         JOIN news_articles a ON s.article_id = a.article_id
                WHERE 1 = 1 \
                """

        params = []

        if article_ids:
            # Формуємо список параметрів для IN
            placeholders = ', '.join(['%s'] * len(article_ids))
            query += f" AND s.article_id IN ({placeholders})"
            params.extend(article_ids)

        if sentiment_label:
            query += " AND s.sentiment_label = %s"
            params.append(sentiment_label)

        if min_score is not None:
            query += " AND s.sentiment_score >= %s"
            params.append(min_score)

        if max_score is not None:
            query += " AND s.sentiment_score <= %s"
            params.append(max_score)

        if start_date:
            query += " AND s.processed_at >= %s"
            params.append(start_date)

        if end_date:
            query += " AND s.processed_at <= %s"
            params.append(end_date)

        query += " ORDER BY s.processed_at DESC"
        query += " LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        self.cursor.execute(query, params)
        results = self.cursor.fetchall()

        return [dict(row) for row in results]

    def save_mentioned_coins(self, mention_data: Dict) -> Dict:

        if not mention_data.get('article_id'):
            raise ValueError("article_id є обов'язковим полем")
        if not mention_data.get('symbol'):
            raise ValueError("symbol є обов'язковим полем")

        # Спроба вставити новий запис або оновити лічильник існуючого
        query = """
                INSERT INTO article_mentioned_coins
                    (article_id, symbol, mention_count)
                VALUES (%s, %s, %s)
                ON CONFLICT (article_id, symbol)
                    DO UPDATE SET mention_count = article_mentioned_coins.mention_count + EXCLUDED.mention_count,
                                  created_at    = NOW()
                RETURNING
                    mention_id, article_id, symbol, mention_count, created_at \
                """

        params = (
            mention_data.get('article_id'),
            mention_data.get('symbol'),
            mention_data.get('mention_count', 1)
        )

        try:
            self.cursor.execute(query, params)
            result = self.cursor.fetchone()
            self.conn.commit()

            return dict(result)
        except Exception as e:
            self.conn.rollback()
            print(f"Помилка при збереженні згадки криптовалюти: {e}")
            raise

    def get_mentioned_coins(self,
                article_id: Optional[int] = None,
                symbol: Optional[str] = None,
                min_mentions: Optional[int] = None,
                start_date: Optional[datetime] = None,
                end_date: Optional[datetime] = None,
                limit: int = 100,
                offset: int = 0) -> List[Dict]:

        query = """
                SELECT mc.mention_id, \
                       mc.article_id, \
                       mc.symbol,
                       mc.mention_count, \
                       mc.created_at,
                       a.title, \
                       a.link, \
                       a.published_at
                FROM article_mentioned_coins mc
                         JOIN news_articles a ON mc.article_id = a.article_id
                WHERE 1 = 1 \
                """

        params = []

        if article_id:
            query += " AND mc.article_id = %s"
            params.append(article_id)

        if symbol:
            query += " AND mc.symbol = %s"
            params.append(symbol)

        if min_mentions is not None:
            query += " AND mc.mention_count >= %s"
            params.append(min_mentions)

        if start_date:
            query += " AND mc.created_at >= %s"
            params.append(start_date)

        if end_date:
            query += " AND mc.created_at <= %s"
            params.append(end_date)

        query += " ORDER BY mc.mention_count DESC, mc.created_at DESC"
        query += " LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        self.cursor.execute(query, params)
        results = self.cursor.fetchall()

        return [dict(row) for row in results]

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