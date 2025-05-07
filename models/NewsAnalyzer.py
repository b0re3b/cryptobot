import csv
import io
import json
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import re
import time
import os
import joblib
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass
from utils.config import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans

try:
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    nltk_available = True
except ImportError:
    nltk_available = False


@dataclass
class NewsItem:
    """Структурований клас для зберігання даних новини"""
    id: str
    title: str
    summary: str
    content: Optional[str] = None
    source: Optional[str] = None
    source_url: Optional[str] = None
    published_at: Optional[datetime] = None
    author: Optional[str] = None
    categories: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    mentioned_cryptos: Optional[Dict[str, int]] = None
    importance_score: Optional[float] = None

    def to_dict(self):
        """Конвертує об'єкт в словник для зберігання в БД"""
        return {
            'id': self.id,
            'title': self.title,
            'summary': self.summary,
            'content': self.content,
            'source': self.source,
            'source_url': self.source_url,
            'published_at': self.published_at,
            'author': self.author,
            'categories': self.categories,
            'tags': self.tags,
            'sentiment_score': self.sentiment_score,
            'sentiment_label': self.sentiment_label,
            'mentioned_cryptos': self.mentioned_cryptos,
            'importance_score': self.importance_score
        }

    @classmethod
    def from_dict(cls, data: Dict):
        """Створює об'єкт з словника"""
        return cls(**data)


class NewsAnalyzer:

    def __init__(self, sentiment_analyzer=None, db_manager=None, logger=None, topic_model_dir='./models/topics'):

        self.sentiment_analyzer = sentiment_analyzer
        self.db_manager = db_manager
        self.logger = logger or logging.getLogger("NewsAnalyzer")
        self.logger.setLevel(logging.INFO)
        self.CRYPTO_KEYWORDS = CRYPTO_KEYWORDS
        self.SENTIMENT_LEXICON = SENTIMENT_LEXICON
        self.CRITICAL_KEYWORDS = CRITICAL_KEYWORDS
        self.MAJOR_ENTITIES = MAJOR_ENTITIES
        self.COMMON_WORDS = COMMON_WORDS
        self.ML_FEATURE_GROUPS = ML_FEATURE_GROUPS
        # Налаштування логування, якщо логгер не передано
        if not logger:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Директорія для зберігання моделей тематичного моделювання
        self.topic_model_dir = topic_model_dir

        # Моделі для тематичного моделювання
        self.vectorizer = None
        self.lda_model = None
        self.nmf_model = None
        self.kmeans_model = None
        self.topic_words = {}

        # Спробувати завантажити існуючі моделі
        try:
            self._load_topic_models()
        except Exception as e:
            self.logger.warning(f"Не вдалося завантажити моделі тематичного моделювання: {e}")

        # Попередня компіляція регулярних виразів
        self.coin_patterns = self._compile_coin_patterns()
        self.category_patterns = self._compile_category_patterns()
        self.ml_feature_patterns = self._compile_ml_feature_patterns()

    def _compile_coin_patterns(self) -> Dict[str, re.Pattern]:

        coin_patterns = {}
        for coin, aliases in self.CRYPTO_KEYWORDS.items():
            # Створюємо шаблон регулярного виразу для кожної монети та її аліасів
            pattern = r'\b(?i)(' + '|'.join(map(re.escape, aliases)) + r')\b'
            coin_patterns[coin] = re.compile(pattern)
        return coin_patterns

    def _compile_category_patterns(self) -> Dict[str, List[re.Pattern]]:

        category_patterns = {}
        for category, keywords in self.CRITICAL_KEYWORDS.items():
            patterns = [re.compile(rf'\b{re.escape(keyword)}\b', re.IGNORECASE) for keyword in keywords]
            category_patterns[category] = patterns
        return category_patterns

    def _compile_ml_feature_patterns(self) -> Dict[str, List[re.Pattern]]:

        feature_patterns = {}
        for feature_group, keywords in self.ML_FEATURE_GROUPS.items():
            patterns = [re.compile(rf'\b{re.escape(keyword)}\b', re.IGNORECASE) for keyword in keywords]
            feature_patterns[feature_group] = patterns
        return feature_patterns

    def _get_text_to_analyze(self, news_item) -> str:

        if isinstance(news_item, NewsItem):
            title = news_item.title or ""
            summary = news_item.summary or ""
            content = news_item.content or ""
            return f"{title} {summary} {content}"
        elif isinstance(news_item, dict):
            title = news_item.get('title', '')
            summary = news_item.get('summary', '')
            content = news_item.get('content', '')
            return f"{title} {summary} {content}"
        else:
            return str(news_item)

    def analyze_news_sentiment(self, news_data: List[Union[Dict[str, Any], NewsItem]]) -> List[
        Union[Dict[str, Any], NewsItem]]:

        self.logger.info(f"Початок аналізу настроїв для {len(news_data)} новин")

        # Якщо нема зовнішнього аналізатора, використовуємо внутрішній лексичний метод
        analyzed_news = []

        for idx, news in enumerate(news_data):
            try:
                # Текст для аналізу
                text_to_analyze = self._get_text_to_analyze(news)

                # Визначаємо, чи використовувати зовнішній аналізатор чи внутрішній
                if self.sentiment_analyzer:
                    # Зовнішній аналізатор
                    sentiment_result = self.sentiment_analyzer.analyze(text_to_analyze)

                    # Форматуємо результат аналізу
                    if isinstance(sentiment_result, dict):
                        sentiment_data = sentiment_result
                    else:
                        # Якщо результат не у вигляді словника, створюємо базову структуру
                        sentiment_data = {
                            'score': getattr(sentiment_result, 'score', 0.0),
                            'label': getattr(sentiment_result, 'label', 'neutral'),
                            'confidence': getattr(sentiment_result, 'confidence', 0.0),
                            'analyzed': True
                        }
                else:
                    # Внутрішній лексичний аналізатор
                    sentiment_data = self._lexicon_based_sentiment(text_to_analyze)

                # Додаємо результат до об'єкта новини
                if isinstance(news, NewsItem):
                    news.sentiment_score = sentiment_data['score']
                    news.sentiment_label = sentiment_data['label']
                    analyzed_news.append(news)
                else:
                    # Копіюємо словник та додаємо результат
                    news_with_sentiment = news.copy()
                    news_with_sentiment['sentiment'] = sentiment_data
                    analyzed_news.append(news_with_sentiment)

                # Логування прогресу (кожні 50 новин)
                if idx > 0 and idx % 50 == 0:
                    self.logger.info(f"Проаналізовано {idx}/{len(news_data)} новин")

            except Exception as e:
                self.logger.error(f"Помилка при аналізі настроїв для новини: {e}")
                # Додаємо новину з нейтральним настроєм у випадку помилки
                if isinstance(news, NewsItem):
                    news.sentiment_score = 0.0
                    news.sentiment_label = 'neutral'
                    analyzed_news.append(news)
                else:
                    # Додаємо новину з нейтральним настроєм у випадку помилки
                    news_copy = news.copy()
                    news_copy['sentiment'] = {
                        'score': 0.0,
                        'label': 'neutral',
                        'confidence': 0.0,
                        'analyzed': False,
                        'error': str(e)
                    }
                    analyzed_news.append(news_copy)

        self.logger.info(f"Аналіз настроїв завершено для {len(analyzed_news)} новин")
        return analyzed_news

    def _lexicon_based_sentiment(self, text: str) -> Dict[str, Any]:

        # Переводимо в нижній регістр і видаляємо пунктуацію
        text = re.sub(r'[^\w\s]', '', text.lower())
        words = text.split()

        # Підрахунок позитивних і негативних слів
        positive_count = sum(1 for word in words if word in self.SENTIMENT_LEXICON['positive'])
        negative_count = sum(1 for word in words if word in self.SENTIMENT_LEXICON['negative'])

        # Визначення загального настрою
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            score = 0.0
            label = 'neutral'
            confidence = 1.0
        else:
            score = (positive_count - negative_count) / total_sentiment_words

            # Обмеження значення між -1 і 1
            score = max(-1.0, min(1.0, score))

            # Визначення мітки
            if score > 0.2:
                label = 'positive'
            elif score < -0.2:
                label = 'negative'
            else:
                label = 'neutral'

            # Розрахунок впевненості (абсолютне значення оцінки)
            confidence = abs(score)

        return {
            'score': score,
            'label': label,
            'confidence': confidence,
            'positive_words': positive_count,
            'negative_words': negative_count,
            'analyzed': True
        }

    def extract_mentioned_coins(self, news_data: List[Union[Dict[str, Any], NewsItem]]) -> List[
        Union[Dict[str, Any], NewsItem]]:

        self.logger.info(f"Початок пошуку згаданих криптовалют у {len(news_data)} новинах")

        for item in news_data:
            try:
                # Текст для аналізу
                text_to_analyze = self._get_text_to_analyze(item)

                # Ініціалізуємо словник для згаданих монет та їх кількості
                mentioned_coins = {}

                # Пошук кожної монети в тексті
                for coin, pattern in self.coin_patterns.items():
                    matches = pattern.findall(text_to_analyze)
                    if matches:
                        # Записуємо кількість згадок
                        mentioned_coins[coin] = len(matches)

                # Сортуємо монети за кількістю згадок (в порядку спадання)
                sorted_mentions = sorted(
                    mentioned_coins.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                # Формуємо структурований результат
                if isinstance(item, NewsItem):
                    item.mentioned_cryptos = mentioned_coins
                else:
                    item['mentioned_coins'] = {
                        'coins': {coin: count for coin, count in sorted_mentions},
                        'top_mentioned': sorted_mentions[0][0] if sorted_mentions else None,
                        'total_coins': len(sorted_mentions)
                    }

            except Exception as e:
                self.logger.error(f"Помилка при пошуку згаданих криптовалют: {e}")
                # Додаємо порожній результат у випадку помилки
                if isinstance(item, NewsItem):
                    item.mentioned_cryptos = {}
                else:
                    item['mentioned_coins'] = {
                        'coins': {},
                        'top_mentioned': None,
                        'total_coins': 0,
                        'error': str(e)
                    }

        self.logger.info("Пошук згаданих криптовалют завершено")
        return news_data

    def filter_by_keywords(self, news_data: List[Union[Dict[str, Any], NewsItem]], keywords: List[str],
                           threshold: int = 1) -> List[Union[Dict[str, Any], NewsItem]]:

        self.logger.info(f"Початок фільтрації новин за {len(keywords)} ключовими словами")

        # Компілюємо регулярні вирази для кожного ключового слова
        keyword_patterns = [re.compile(rf'\b{re.escape(keyword)}\b', re.IGNORECASE) for keyword in keywords]

        filtered_news = []

        for item in news_data:
            try:
                # Отримуємо текст для аналізу
                text_to_analyze = self._get_text_to_analyze(item)

                # Рахуємо кількість знайдених ключових слів
                match_count = 0
                for pattern in keyword_patterns:
                    match_count += len(pattern.findall(text_to_analyze))

                # Додаємо новину, якщо кількість збігів відповідає порогу
                if match_count >= threshold:
                    # Додаємо метадані про ключові слова
                    if isinstance(item, NewsItem):
                        # Для об'єкта NewsItem додаємо інформацію в поле tags, якщо воно існує
                        if item.tags is None:
                            item.tags = []
                        item.tags.extend([kw for kw in keywords if
                                          any(re.compile(rf'\b{re.escape(kw)}\b', re.IGNORECASE).search(
                                              text_to_analyze))])
                        filtered_news.append(item)
                    else:
                        # Для словника, додаємо метадані
                        item_copy = item.copy()
                        if 'keywords_matched' not in item_copy:
                            item_copy['keywords_matched'] = {}

                        item_copy['keywords_matched'] = {
                            'total': match_count,
                            'matched_keywords': [kw for kw in keywords if
                                                 any(re.compile(rf'\b{re.escape(kw)}\b', re.IGNORECASE).search(
                                                     text_to_analyze))]
                        }
                        filtered_news.append(item_copy)

            except Exception as e:
                self.logger.error(f"Помилка при фільтрації новини за ключовими словами: {e}")

        self.logger.info(f"Фільтрація завершена. Знайдено {len(filtered_news)} новин із {len(news_data)}")
        return filtered_news

    def calculate_importance_score(self, news_data: List[Union[Dict[str, Any], NewsItem]]) -> List[
        Union[Dict[str, Any], NewsItem]]:

        self.logger.info(f"Початок розрахунку оцінки важливості для {len(news_data)} новин")

        for item in news_data:
            try:
                importance_score = 0.0
                text_to_analyze = self._get_text_to_analyze(item)

                # 1. Оцінка за наявністю топових криптовалют
                top_cryptos = ['bitcoin', 'ethereum', 'binance coin', 'ripple', 'cardano']
                for crypto in top_cryptos:
                    pattern = self.coin_patterns.get(crypto)
                    if pattern and pattern.search(text_to_analyze):
                        # Більш важливі криптовалюти отримують більшу вагу
                        if crypto == 'bitcoin':
                            importance_score += 0.3
                        elif crypto == 'ethereum':
                            importance_score += 0.25
                        else:
                            importance_score += 0.15

                # 2. Оцінка за категоріями подій
                critical_categories = ['regulation', 'hack', 'market_crash', 'market_boom', 'scandal']
                category_mentions = {}

                for category, patterns in self.category_patterns.items():
                    matches = sum(1 for pattern in patterns if pattern.search(text_to_analyze))
                    if matches > 0:
                        category_mentions[category] = matches
                        # Критичні категорії мають вищу вагу
                        if category in critical_categories:
                            importance_score += 0.25 * min(matches, 3)  # Обмежуємо множник
                        else:
                            importance_score += 0.1 * min(matches, 3)

                # 3. Оцінка за настроєм
                # Сильні настрої (як позитивні, так і негативні) підвищують важливість
                sentiment_score = 0
                sentiment_label = 'neutral'

                if isinstance(item, NewsItem):
                    sentiment_score = item.sentiment_score if hasattr(item, 'sentiment_score') else 0
                    sentiment_label = item.sentiment_label if hasattr(item, 'sentiment_label') else 'neutral'
                else:
                    if 'sentiment' in item and isinstance(item['sentiment'], dict):
                        sentiment_score = item['sentiment'].get('score', 0)
                        sentiment_label = item['sentiment'].get('label', 'neutral')

                # Додаємо вагу за силою настрою
                importance_score += abs(sentiment_score) * 0.2

                # 4. Згадки основних сутностей галузі
                for entity in self.MAJOR_ENTITIES:
                    pattern = re.compile(rf'\b{re.escape(entity)}\b', re.IGNORECASE)
                    if pattern.search(text_to_analyze):
                        importance_score += 0.05  # Невелика вага за згадку важливої сутності

                # 5. Перевірка на ознаки термінової/важливої новини
                urgent_patterns = [
                    re.compile(r'\b(?:urgent|breaking|alert|attention|important|critical)\b', re.IGNORECASE),
                    re.compile(r'\b(?:just|now|latest|update|announcement)\b', re.IGNORECASE),
                    re.compile(r'\b(?:official|exclusive|report|confirms|announces)\b', re.IGNORECASE)
                ]

                for pattern in urgent_patterns:
                    if pattern.search(text_to_analyze):
                        importance_score += 0.1
                        break

                # Обмежуємо фінальну оцінку від 0 до 1
                importance_score = min(max(importance_score, 0.0), 1.0)

                # Збереження результату
                if isinstance(item, NewsItem):
                    item.importance_score = importance_score
                else:
                    item['importance_score'] = importance_score
                    item['importance_factors'] = {
                        'category_mentions': category_mentions,
                        'sentiment_strength': abs(sentiment_score),
                        'sentiment_label': sentiment_label
                    }

            except Exception as e:
                self.logger.error(f"Помилка при розрахунку оцінки важливості: {e}")
                # Встановлюємо значення за замовчуванням у випадку помилки
                if isinstance(item, NewsItem):
                    item.importance_score = 0.3  # Середня оцінка за замовчуванням
                else:
                    item['importance_score'] = 0.3
                    item['importance_factors'] = {'error': str(e)}

        self.logger.info("Розрахунок оцінки важливості завершено")
        return news_data

    def categorize_news(self, news_data: List[Union[Dict[str, Any], NewsItem]]) -> List[
        Union[Dict[str, Any], NewsItem]]:

        self.logger.info(f"Початок категоризації {len(news_data)} новин")

        for item in news_data:
            try:
                text_to_analyze = self._get_text_to_analyze(item)

                # Словник для підрахунку збігів по кожній категорії
                category_scores = {category: 0 for category in self.CRITICAL_KEYWORDS.keys()}

                # Шукаємо збіги за кожною категорією
                for category, patterns in self.category_patterns.items():
                    for pattern in patterns:
                        matches = pattern.findall(text_to_analyze)
                        category_scores[category] += len(matches)

                # Фільтруємо лише релевантні категорії (з принаймні одним збігом)
                relevant_categories = {cat: score for cat, score in category_scores.items() if score > 0}

                # Сортуємо категорії за кількістю збігів
                sorted_categories = sorted(
                    relevant_categories.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                # Визначаємо основні категорії (не більше 3)
                main_categories = [cat for cat, _ in sorted_categories[:3]] if sorted_categories else []

                # Зберігаємо результат
                if isinstance(item, NewsItem):
                    if item.categories is None:
                        item.categories = []
                    item.categories.extend(main_categories)
                    # Видаляємо дублікати, зберігаючи порядок
                    item.categories = list(dict.fromkeys(item.categories))
                else:
                    item['categories'] = {
                        'main': main_categories,
                        'all_matches': dict(sorted_categories),
                        'top_category': sorted_categories[0][0] if sorted_categories else None
                    }

            except Exception as e:
                self.logger.error(f"Помилка при категоризації новини: {e}")
                # Встановлюємо значення за замовчуванням у випадку помилки
                if isinstance(item, NewsItem):
                    item.categories = []
                else:
                    item['categories'] = {
                        'main': [],
                        'all_matches': {},
                        'top_category': None,
                        'error': str(e)
                    }

        self.logger.info("Категоризація новин завершена")
        return news_data

    def _load_topic_models(self):

        try:
            if not os.path.exists(self.topic_model_dir):
                self.logger.info(f"Директорія моделей {self.topic_model_dir} не існує")
                return

            # Завантаження векторизатора
            vectorizer_path = os.path.join(self.topic_model_dir, 'vectorizer.joblib')
            if os.path.exists(vectorizer_path):
                self.vectorizer = joblib.load(vectorizer_path)
                self.logger.info("Векторизатор успішно завантажено")

            # Завантаження LDA моделі
            lda_path = os.path.join(self.topic_model_dir, 'lda_model.joblib')
            if os.path.exists(lda_path):
                self.lda_model = joblib.load(lda_path)
                self.logger.info("LDA модель успішно завантажено")

            # Завантаження NMF моделі
            nmf_path = os.path.join(self.topic_model_dir, 'nmf_model.joblib')
            if os.path.exists(nmf_path):
                self.nmf_model = joblib.load(nmf_path)
                self.logger.info("NMF модель успішно завантажено")

            # Завантаження K-means моделі
            kmeans_path = os.path.join(self.topic_model_dir, 'kmeans_model.joblib')
            if os.path.exists(kmeans_path):
                self.kmeans_model = joblib.load(kmeans_path)
                self.logger.info("K-means модель успішно завантажено")

            # Завантаження словника ключових слів тем
            topic_words_path = os.path.join(self.topic_model_dir, 'topic_words.joblib')
            if os.path.exists(topic_words_path):
                self.topic_words = joblib.load(topic_words_path)
                self.logger.info("Словник ключових слів тем успішно завантажено")

        except Exception as e:
            self.logger.error(f"Помилка при завантаженні моделей тематичного моделювання: {e}")

    def _save_topic_models(self):

        try:
            # Створюємо директорію, якщо вона не існує
            if not os.path.exists(self.topic_model_dir):
                os.makedirs(self.topic_model_dir)
                self.logger.info(f"Створено директорію для моделей: {self.topic_model_dir}")

            # Зберігаємо векторизатор
            if self.vectorizer is not None:
                joblib.dump(self.vectorizer, os.path.join(self.topic_model_dir, 'vectorizer.joblib'))
                self.logger.info("Векторизатор успішно збережено")

            # Зберігаємо LDA модель
            if self.lda_model is not None:
                joblib.dump(self.lda_model, os.path.join(self.topic_model_dir, 'lda_model.joblib'))
                self.logger.info("LDA модель успішно збережено")

            # Зберігаємо NMF модель
            if self.nmf_model is not None:
                joblib.dump(self.nmf_model, os.path.join(self.topic_model_dir, 'nmf_model.joblib'))
                self.logger.info("NMF модель успішно збережено")

            # Зберігаємо K-means модель
            if self.kmeans_model is not None:
                joblib.dump(self.kmeans_model, os.path.join(self.topic_model_dir, 'kmeans_model.joblib'))
                self.logger.info("K-means модель успішно збережено")

            # Зберігаємо словник ключових слів тем
            if self.topic_words:
                joblib.dump(self.topic_words, os.path.join(self.topic_model_dir, 'topic_words.joblib'))
                self.logger.info("Словник ключових слів тем успішно збережено")

        except Exception as e:
            self.logger.error(f"Помилка при збереженні моделей тематичного моделювання: {e}")

    def _preprocess_text(self, text):

        # Переводимо в нижній регістр
        text = text.lower()

        # Видаляємо спеціальні символи та числа
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)

        # Якщо NLTK встановлено, використовуємо додаткову обробку
        if nltk_available:
            try:
                # Токенізація
                tokens = word_tokenize(text)

                # Видалення стоп-слів
                stop_words = set(stopwords.words('english')) | set(self.COMMON_WORDS)
                tokens = [word for word in tokens if word not in stop_words and len(word) > 2]

                # Лематизація
                lemmatizer = WordNetLemmatizer()
                tokens = [lemmatizer.lemmatize(word) for word in tokens]

                return ' '.join(tokens)
            except Exception as e:
                self.logger.warning(f"Помилка при використанні NLTK для обробки тексту: {e}")
                # Запасний варіант - проста фільтрація загальних слів
                words = text.split()
                words = [word for word in words if word not in self.COMMON_WORDS and len(word) > 2]
                return ' '.join(words)
        else:
            # Проста фільтрація загальних слів
            words = text.split()
            words = [word for word in words if word not in self.COMMON_WORDS and len(word) > 2]
            return ' '.join(words)

    def train_topic_models(self, news_data: List[Union[Dict[str, Any], NewsItem]],
                           n_topics: int = 10, method: str = 'lda'):

        self.logger.info(f"Початок навчання моделей тематичного моделювання на {len(news_data)} новинах")

        # Підготовка даних
        texts = []
        for item in news_data:
            text = self._get_text_to_analyze(item)
            processed_text = self._preprocess_text(text)
            if processed_text.strip():  # Перевіряємо, чи не порожній текст
                texts.append(processed_text)

        if not texts:
            self.logger.warning("Немає текстів для тематичного моделювання")
            return {"status": "error", "message": "Немає даних для тренування"}

        # Векторизація
        self.logger.info("Векторизація текстів...")
        self.vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8)
        X = self.vectorizer.fit_transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()

        # Тренування моделей відповідно до обраного методу
        if method in ['lda', 'all']:
            self.logger.info(f"Тренування LDA моделі з {n_topics} темами...")
            self.lda_model = LatentDirichletAllocation(
                n_components=n_topics,
                max_iter=50,
                random_state=42,
                learning_method='online'
            )
            self.lda_model.fit(X)

            # Збереження ключових слів для тем LDA
            lda_topic_words = {}
            for topic_idx, topic in enumerate(self.lda_model.components_):
                top_words_idx = topic.argsort()[:-11:-1]  # Топ-10 слів для кожної теми
                top_words = [feature_names[i] for i in top_words_idx]
                lda_topic_words[f"lda_topic_{topic_idx}"] = top_words

            self.topic_words.update(lda_topic_words)

        if method in ['nmf', 'all']:
            self.logger.info(f"Тренування NMF моделі з {n_topics} темами...")
            self.nmf_model = NMF(
                n_components=n_topics,
                random_state=42,
                max_iter=300
            )
            self.nmf_model.fit(X)

            # Збереження ключових слів для тем NMF
            nmf_topic_words = {}
            for topic_idx, topic in enumerate(self.nmf_model.components_):
                top_words_idx = topic.argsort()[:-11:-1]  # Топ-10 слів для кожної теми
                top_words = [feature_names[i] for i in top_words_idx]
                nmf_topic_words[f"nmf_topic_{topic_idx}"] = top_words

            self.topic_words.update(nmf_topic_words)

        if method in ['kmeans', 'all']:
            self.logger.info(f"Тренування K-means моделі з {n_topics} кластерами...")
            self.kmeans_model = KMeans(
                n_clusters=n_topics,
                random_state=42,
                n_init=10
            )
            self.kmeans_model.fit(X)

            # Знаходження центрів кластерів та ключових слів
            kmeans_topic_words = {}
            order_centroids = self.kmeans_model.cluster_centers_.argsort()[:, ::-1]
            for cluster_idx in range(n_topics):
                top_words_idx = order_centroids[cluster_idx, :10]  # Топ-10 слів для кожного кластера
                top_words = [feature_names[i] for i in top_words_idx]
                kmeans_topic_words[f"kmeans_cluster_{cluster_idx}"] = top_words

            self.topic_words.update(kmeans_topic_words)

        # Зберігаємо моделі
        self._save_topic_models()

        # Повертаємо інформацію про тренування
        return {
            "status": "success",
            "models_trained": [m for m in ['lda', 'nmf', 'kmeans'] if method in [m, 'all']],
            "n_documents": len(texts),
            "n_features": len(feature_names),
            "topics": self.topic_words
        }

    def assign_topics(self, news_data: List[Union[Dict[str, Any], NewsItem]], method: str = 'lda') -> List[
        Union[Dict[str, Any], NewsItem]]:

        # Перевіряємо, що моделі натреновані
        models_available = {
            'lda': self.lda_model is not None,
            'nmf': self.nmf_model is not None,
            'kmeans': self.kmeans_model is not None
        }

        if not models_available[method]:
            self.logger.warning(f"Модель {method} не натренована")
            return news_data

        if self.vectorizer is None:
            self.logger.warning("Векторизатор не знайдено")
            return news_data

        self.logger.info(f"Призначення тем для {len(news_data)} новин за допомогою методу {method}")

        for item in news_data:
            try:
                # Підготовка тексту
                text = self._get_text_to_analyze(item)
                processed_text = self._preprocess_text(text)

                # Векторизація
                X = self.vectorizer.transform([processed_text])

                # Визначення тем залежно від методу
                if method == 'lda':
                    topic_distribution = self.lda_model.transform(X)[0]
                    top_topics = topic_distribution.argsort()[::-1]

                    # Витягуємо ключові слова для топових тем
                    topic_info = []
                    for topic_idx in top_topics[:3]:  # Беремо топ-3 теми
                        topic_prob = topic_distribution[topic_idx]
                        if topic_prob > 0.05:  # Порогове значення для релевантності теми
                            topic_key = f"lda_topic_{topic_idx}"
                            topic_words = self.topic_words.get(topic_key, [])
                            topic_info.append({
                                'id': topic_idx,
                                'probability': float(topic_prob),
                                'keywords': topic_words
                            })

                elif method == 'nmf':
                    topic_distribution = self.nmf_model.transform(X)[0]
                    top_topics = topic_distribution.argsort()[::-1]

                    topic_info = []
                    for topic_idx in top_topics[:3]:
                        topic_prob = topic_distribution[topic_idx]
                        if topic_prob > 0.05:
                            topic_key = f"nmf_topic_{topic_idx}"
                            topic_words = self.topic_words.get(topic_key, [])
                            topic_info.append({
                                'id': topic_idx,
                                'weight': float(topic_prob),
                                'keywords': topic_words
                            })

                elif method == 'kmeans':
                    cluster = self.kmeans_model.predict(X)[0]

                    topic_info = [{
                        'cluster': int(cluster),
                        'keywords': self.topic_words.get(f"kmeans_cluster_{cluster}", [])
                    }]

                # Зберігаємо результати
                if isinstance(item, NewsItem):
                    if not hasattr(item, 'topics') or item.topics is None:
                        item.topics = {}
                    item.topics[method] = topic_info
                else:
                    if 'topics' not in item:
                        item['topics'] = {}
                    item['topics'][method] = topic_info

            except Exception as e:
                self.logger.error(f"Помилка при призначенні тем для новини: {e}")
                # Встановлюємо пусті значення
                if isinstance(item, NewsItem):
                    if not hasattr(item, 'topics') or item.topics is None:
                        item.topics = {}
                    item.topics[method] = []
                else:
                    if 'topics' not in item:
                        item['topics'] = {}
                    item['topics'][method] = []
                    item['topics'][f'{method}_error'] = str(e)

        self.logger.info(f"Призначення тем завершено для методу {method}")
        return news_data

    def detect_trends(self, news_data: List[Union[Dict[str, Any], NewsItem]],
                      window_days: int = 7, min_count: int = 3) -> Dict[str, Any]:

        self.logger.info(f"Початок виявлення трендів у {len(news_data)} новинах за {window_days} днів")

        # Визначаємо поточну дату та граничну дату для аналізу
        current_date = datetime.now()
        cutoff_date = current_date - timedelta(days=window_days)

        # Фільтруємо новини за періодом
        recent_news = []
        for item in news_data:
            try:
                if isinstance(item, NewsItem):
                    item_date = item.published_at if hasattr(item, 'published_at') else current_date
                else:
                    item_date = item.get('published_at', current_date)

                # Конвертуємо дату з рядка, якщо потрібно
                if isinstance(item_date, str):
                    try:
                        item_date = datetime.fromisoformat(item_date.replace('Z', '+00:00'))
                    except (ValueError, TypeError):
                        try:
                            # Спроба альтернативного формату
                            item_date = datetime.strptime(item_date, "%Y-%m-%d %H:%M:%S")
                        except (ValueError, TypeError):
                            # Встановлюємо поточну дату, якщо не вдалося розпарсити
                            item_date = current_date

                # Додаємо до списку, якщо дата у заданому вікні
                if item_date >= cutoff_date:
                    recent_news.append(item)
            except Exception as e:
                self.logger.error(f"Помилка при фільтрації новини за датою: {e}")

        self.logger.info(f"Знайдено {len(recent_news)} новин за останні {window_days} днів")

        # Підготовка даних для аналізу трендів

        # Аналіз згаданих криптовалют
        crypto_mentions = {}
        for item in recent_news:
            if isinstance(item, NewsItem):
                if hasattr(item, 'mentioned_cryptos'):
                    for crypto, count in item.mentioned_cryptos.items():
                        crypto_mentions[crypto] = crypto_mentions.get(crypto, 0) + count
            else:
                if 'mentioned_coins' in item and isinstance(item['mentioned_coins'], dict):
                    coins = item['mentioned_coins'].get('coins', {})
                    for crypto, count in coins.items():
                        crypto_mentions[crypto] = crypto_mentions.get(crypto, 0) + count

        # Аналіз згаданих категорій подій
        category_mentions = {}
        for item in recent_news:
            if isinstance(item, NewsItem):
                if hasattr(item, 'categories'):
                    for category in item.categories:
                        category_mentions[category] = category_mentions.get(category, 0) + 1
            else:
                if 'categories' in item and isinstance(item['categories'], dict):
                    categories = item['categories'].get('main', [])
                    for category in categories:
                        category_mentions[category] = category_mentions.get(category, 0) + 1

        # Аналіз настроїв
        sentiment_distribution = {'positive': 0, 'neutral': 0, 'negative': 0}
        for item in recent_news:
            if isinstance(item, NewsItem):
                sentiment = item.sentiment_label if hasattr(item, 'sentiment_label') else 'neutral'
            else:
                if 'sentiment' in item and isinstance(item['sentiment'], dict):
                    sentiment = item['sentiment'].get('label', 'neutral')
                else:
                    sentiment = 'neutral'

            sentiment_distribution[sentiment] = sentiment_distribution.get(sentiment, 0) + 1

        # Аналіз частоти слів (для виявлення нових трендових слів)
        all_text = " ".join([self._get_text_to_analyze(item) for item in recent_news])
        processed_text = self._preprocess_text(all_text)
        word_frequencies = {}

        for word in processed_text.split():
            if len(word) > 3 and word.lower() not in self.COMMON_WORDS:
                word_frequencies[word.lower()] = word_frequencies.get(word.lower(), 0) + 1

        # Фільтруємо тренди за мінімальною кількістю згадок
        trending_cryptos = {k: v for k, v in crypto_mentions.items() if v >= min_count}
        trending_categories = {k: v for k, v in category_mentions.items() if v >= min_count}
        trending_words = {k: v for k, v in word_frequencies.items() if v >= min_count * 2}  # Вищий поріг для слів

        # Сортуємо тренди за спаданням популярності
        sorted_cryptos = sorted(trending_cryptos.items(), key=lambda x: x[1], reverse=True)
        sorted_categories = sorted(trending_categories.items(), key=lambda x: x[1], reverse=True)
        sorted_words = sorted(trending_words.items(), key=lambda x: x[1], reverse=True)[:30]  # Обмежуємо до топ-30 слів

        # Формуємо результат
        trends_result = {
            'period': {
                'days': window_days,
                'start_date': cutoff_date.isoformat(),
                'end_date': current_date.isoformat()
            },
            'news_count': len(recent_news),
            'trending_cryptos': [{'name': name, 'mentions': count} for name, count in sorted_cryptos],
            'trending_categories': [{'name': name, 'mentions': count} for name, count in sorted_categories],
            'trending_words': [{'word': word, 'frequency': freq} for word, freq in sorted_words],
            'sentiment_distribution': sentiment_distribution
        }

        self.logger.info(
            f"Виявлено {len(trending_cryptos)} трендових криптовалют та {len(trending_categories)} категорій")
        return trends_result

    def export_analysis_results(self, news_data: List[Union[Dict[str, Any], NewsItem]],
                                output_format: str = 'json', filename: str = None) -> str:

        self.logger.info(f"Експорт результатів аналізу для {len(news_data)} новин у форматі {output_format}")

        # Генеруємо ім'я файлу, якщо не вказано
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"news_analysis_{timestamp}.{output_format}"

        # Конвертуємо об'єкти NewsItem у словники, якщо потрібно
        normalized_data = []
        for item in news_data:
            if isinstance(item, NewsItem):
                # Перетворення NewsItem у словник
                item_dict = item.__dict__.copy()
                # Видаляємо службові атрибути
                if '_sa_instance_state' in item_dict:
                    del item_dict['_sa_instance_state']
                normalized_data.append(item_dict)
            else:
                normalized_data.append(item)

        # Експорт у відповідному форматі
        try:
            if output_format == 'json':
                # Конвертуємо дати у рядки для JSON-серіалізації
                for item in normalized_data:
                    for key, value in item.items():
                        if isinstance(value, datetime):
                            item[key] = value.isoformat()

                result = json.dumps(normalized_data, ensure_ascii=False, indent=2)
                if filename:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(result)
                    return filename
                return result

            elif output_format == 'csv':
                # Створюємо плоский набір даних
                flat_data = []
                for item in normalized_data:
                    flat_item = {}
                    for key, value in item.items():
                        if isinstance(value, dict):
                            # Для вкладених словників
                            for sub_key, sub_value in value.items():
                                flat_item[f"{key}_{sub_key}"] = str(sub_value)
                        elif isinstance(value, list):
                            # Для списків
                            flat_item[key] = ",".join(str(x) for x in value)
                        else:
                            flat_item[key] = value
                    flat_data.append(flat_item)

                # Визначаємо всі унікальні ключі
                all_keys = set()
                for item in flat_data:
                    all_keys.update(item.keys())

                # Створюємо CSV
                if filename:
                    with open(filename, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
                        writer.writeheader()
                        writer.writerows(flat_data)
                    return filename
                else:
                    # Повертаємо CSV як рядок
                    output = io.StringIO()
                    writer = csv.DictWriter(output, fieldnames=sorted(all_keys))
                    writer.writeheader()
                    writer.writerows(flat_data)
                    return output.getvalue()

            else:
                raise ValueError(f"Непідтримуваний формат експорту: {output_format}")

        except Exception as e:
            self.logger.error(f"Помилка при експорті результатів: {e}")
            raise

    def import_analysis_results(self, file_path: str) -> List[Dict[str, Any]]:

        self.logger.info(f"Імпорт результатів аналізу з файлу: {file_path}")

        # Визначаємо формат за розширенням файлу
        _, extension = os.path.splitext(file_path)
        extension = extension.lower()

        try:
            if extension == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Конвертуємо рядки дат назад у об'єкти datetime
                for item in data:
                    for key, value in item.items():
                        if isinstance(value, str) and re.match(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', value):
                            try:
                                item[key] = datetime.fromisoformat(value.replace('Z', '+00:00'))
                            except ValueError:
                                # Залишаємо як є, якщо не вдалося розпарсити
                                pass

                return data

            elif extension == '.csv':
                # Зчитуємо CSV
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    data = list(reader)

                # Перетворення вкладених структур, які були розділені
                result = []
                for item in data:
                    # Групування за префіксом
                    nested_data = {}
                    flat_data = {}

                    for key, value in item.items():
                        if '_' in key:
                            prefix, suffix = key.split('_', 1)
                            if prefix not in nested_data:
                                nested_data[prefix] = {}
                            nested_data[prefix][suffix] = value
                        else:
                            flat_data[key] = value

                    # Об'єднання даних
                    for prefix, nested in nested_data.items():
                        flat_data[prefix] = nested

                    result.append(flat_data)

                return result

            elif extension in ['.xlsx', '.xls']:
                try:
                    import pandas as pd

                    # Зчитуємо Excel
                    df = pd.read_excel(file_path)
                    data = df.to_dict('records')

                    # Обробка рядкових значень для вкладених структур
                    for item in data:
                        for key, value in item.items():
                            if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                                try:
                                    item[key] = json.loads(value)
                                except json.JSONDecodeError:
                                    # Залишаємо як є, якщо не вдалося розпарсити
                                    pass

                    return data
                except ImportError:
                    self.logger.error("Для імпорту з Excel потрібен pandas")
                    raise ImportError("Для імпорту з Excel потрібен pandas")
            else:
                raise ValueError(f"Непідтримуваний формат файлу: {extension}")

        except Exception as e:
            self.logger.error(f"Помилка при імпорті результатів: {e}")
            raise

    def generate_summary_report(self, news_data: List[Union[Dict[str, Any], NewsItem]]) -> Dict[str, Any]:

        self.logger.info(f"Генерація підсумкового звіту для {len(news_data)} новин")

        # Загальна статистика
        total_news = len(news_data)
        if total_news == 0:
            return {"error": "Немає даних для аналізу"}

        # Часові рамки
        dates = []
        for item in news_data:
            if isinstance(item, NewsItem):
                if hasattr(item, 'published_at'):
                    dates.append(item.published_at)
            else:
                if 'published_at' in item:
                    date_value = item['published_at']
                    if isinstance(date_value, str):
                        try:
                            dates.append(datetime.fromisoformat(date_value.replace('Z', '+00:00')))
                        except ValueError:
                            pass
                    elif isinstance(date_value, datetime):
                        dates.append(date_value)

        # Визначаємо діапазон дат
        date_range = {}
        if dates:
            min_date = min(dates)
            max_date = max(dates)
            date_range = {
                'start': min_date.isoformat(),
                'end': max_date.isoformat(),
                'days': (max_date - min_date).days + 1
            }

        # Аналіз настроїв
        sentiment_distribution = {'positive': 0, 'neutral': 0, 'negative': 0}
        sentiment_scores = []

        for item in news_data:
            sentiment_label = 'neutral'
            sentiment_score = 0.0

            if isinstance(item, NewsItem):
                if hasattr(item, 'sentiment_label'):
                    sentiment_label = item.sentiment_label
                if hasattr(item, 'sentiment_score'):
                    sentiment_score = item.sentiment_score
            else:
                if 'sentiment' in item and isinstance(item['sentiment'], dict):
                    sentiment_label = item['sentiment'].get('label', 'neutral')
                    sentiment_score = item['sentiment'].get('score', 0.0)

            sentiment_distribution[sentiment_label] = sentiment_distribution.get(sentiment_label, 0) + 1
            sentiment_scores.append(sentiment_score)

        # Розрахунок середнього настрою
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

        # Топ криптовалюти
        coin_mentions = {}

        for item in news_data:
            if isinstance(item, NewsItem):
                if hasattr(item, 'mentioned_cryptos'):
                    for crypto, count in item.mentioned_cryptos.items():
                        coin_mentions[crypto] = coin_mentions.get(crypto, 0) + count
            else:
                if 'mentioned_coins' in item and isinstance(item['mentioned_coins'], dict):
                    coins = item['mentioned_coins'].get('coins', {})
                    for crypto, count in coins.items():
                        coin_mentions[crypto] = coin_mentions.get(crypto, 0) + count

        # Топ-10 криптовалют
        top_coins = sorted(coin_mentions.items(), key=lambda x: x[1], reverse=True)[:10]

        # Популярні категорії
        category_counts = {}

        for item in news_data:
            if isinstance(item, NewsItem):
                if hasattr(item, 'categories'):
                    for category in item.categories:
                        category_counts[category] = category_counts.get(category, 0) + 1
            else:
                if 'categories' in item and isinstance(item['categories'], dict):
                    categories = item['categories'].get('main', [])
                    for category in categories:
                        category_counts[category] = category_counts.get(category, 0) + 1

        # Топ-5 категорій
        top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # Найважливіші новини
        important_news = []

        for item in news_data:
            importance_score = 0.0

            if isinstance(item, NewsItem):
                if hasattr(item, 'importance_score'):
                    importance_score = item.importance_score

                if importance_score >= 0.7:  # Високий поріг важливості
                    important_news.append({
                        'title': item.title if hasattr(item, 'title') else "Без назви",
                        'importance_score': importance_score,
                        'categories': item.categories if hasattr(item, 'categories') else [],
                        'published_at': item.published_at.isoformat() if hasattr(item, 'published_at') else None
                    })
            else:
                if 'importance_score' in item:
                    importance_score = item['importance_score']

                if importance_score >= 0.7:
                    important_news.append({
                        'title': item.get('title', "Без назви"),
                        'importance_score': importance_score,
                        'categories': item.get('categories', {}).get('main', []) if isinstance(item.get('categories'),
                                                                                               dict) else [],
                        'published_at': item.get('published_at')
                    })

        # Обмежуємо список найважливіших новин
        important_news = sorted(important_news, key=lambda x: x['importance_score'], reverse=True)[:10]

        # Формуємо підсумковий звіт
        report = {
            'summary': {
                'total_news': total_news,
                'date_range': date_range,
                'overall_sentiment': {
                    'average_score': avg_sentiment,
                    'distribution': sentiment_distribution
                }
            },
            'top_cryptocurrencies': [{'name': name, 'mentions': count} for name, count in top_coins],
            'top_categories': [{'name': name, 'count': count} for name, count in top_categories],
            'important_news': important_news,
            'generated_at': datetime.now().isoformat()
        }

        self.logger.info("Підсумковий звіт успішно згенеровано")
        return report

