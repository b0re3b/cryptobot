import os
import re
import json
import logging
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Union, Any
import numpy as np
from sklearn.cluster import KMeans
import torch
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from utils.config import *
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    nltk_available = True
except ImportError:
    nltk_available = False



class NewsItem:
    """Клас для представлення елемента новин з усіма атрибутами для аналізу."""

    def __init__(self, title=None, url=None, source=None, published_at=None, author=None,
                 content=None, summary=None, categories=None, tags=None, sentiment_score=None,
                 sentiment_label=None, importance_score=None, mentioned_cryptos=None, topics=None):
        self.title = title
        self.url = url
        self.source = source
        self.published_at = published_at
        self.author = author
        self.content = content
        self.summary = summary
        self.categories = categories or []
        self.tags = tags or []
        self.sentiment_score = sentiment_score
        self.sentiment_label = sentiment_label
        self.importance_score = importance_score
        self.mentioned_cryptos = mentioned_cryptos or {}
        self.topics = topics or {}

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


class BERTNewsAnalyzer:

    def __init__(self, bert_model_name='bert-base-uncased',
                 sentiment_model_name='nlptown/bert-base-multilingual-uncased-sentiment',
                 db_manager=None, logger=None, topic_model_dir='./models/topics', device=None):

        self.logger = logger or logging.getLogger("BERTNewsAnalyzer")
        self.logger.setLevel(logging.INFO)
        self.db_manager = db_manager
        self.CRYPTO_KEYWORDS = CRYPTO_KEYWORDS
        self.SENTIMENT_LEXICON = SENTIMENT_LEXICON
        self.CRITICAL_KEYWORDS = CRITICAL_KEYWORDS
        self.MAJOR_ENTITIES = MAJOR_ENTITIES
        self.COMMON_WORDS = COMMON_WORDS
        self.ML_FEATURE_GROUPS = ML_FEATURE_GROUPS
        self.topic_model_dir = topic_model_dir

        # Налаштування логування, якщо логгер не передано
        if not logger:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Визначення пристрою (CPU або GPU)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Використовується пристрій: {self.device}")

        # Завантаження моделей BERT і токенізаторів
        try:
            self.logger.info(f"Завантаження базової моделі BERT: {bert_model_name}")
            self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
            self.bert_model = BertModel.from_pretrained(bert_model_name).to(self.device)

            self.logger.info(f"Завантаження моделі BERT для аналізу тональності: {sentiment_model_name}")
            # Використовуємо готовий pipeline для аналізу тональності
            self.sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_model_name,
                                               tokenizer=sentiment_model_name,
                                               device=0 if self.device == 'cuda' else -1)

            self.logger.info("Моделі BERT успішно завантажено")
        except Exception as e:
            self.logger.error(f"Помилка при завантаженні моделей BERT: {e}")
            raise

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
        """Компіляція регулярних виразів для криптовалют."""
        coin_patterns = {}
        for coin, aliases in self.CRYPTO_KEYWORDS.items():
            # Створюємо шаблон регулярного виразу для кожної монети та її аліасів
            pattern = r'\b(?i)(' + '|'.join(map(re.escape, aliases)) + r')\b'
            coin_patterns[coin] = re.compile(pattern)
        return coin_patterns

    def _compile_category_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Компіляція регулярних виразів для категорій новин."""
        category_patterns = {}
        for category, keywords in self.CRITICAL_KEYWORDS.items():
            patterns = [re.compile(rf'\b{re.escape(keyword)}\b', re.IGNORECASE) for keyword in keywords]
            category_patterns[category] = patterns
        return category_patterns

    def _compile_ml_feature_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Компіляція регулярних виразів для груп функцій машинного навчання."""
        feature_patterns = {}
        for feature_group, keywords in self.ML_FEATURE_GROUPS.items():
            patterns = [re.compile(rf'\b{re.escape(keyword)}\b', re.IGNORECASE) for keyword in keywords]
            feature_patterns[feature_group] = patterns
        return feature_patterns

    def _get_text_to_analyze(self, news_item) -> str:
        """Отримати текст для аналізу з об'єкта новин."""
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

    def _get_bert_embeddings(self, text):
        """Отримати вектори BERT для тексту."""
        # Токенізація тексту
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
            self.device)

        # Отримання векторів BERT
        with torch.no_grad():  # Відключаємо обчислення градієнтів для прискорення
            outputs = self.bert_model(**inputs)
            # Використовуємо [CLS] токен як вектор усього речення
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embeddings

    def analyze_news_sentiment(self, news_data: List[Union[Dict[str, Any], NewsItem]]) -> List[
        Union[Dict[str, Any], NewsItem]]:
        """Аналіз тональності новин за допомогою BERT."""
        self.logger.info(f"Початок аналізу настроїв для {len(news_data)} новин")

        analyzed_news = []

        for idx, news in enumerate(news_data):
            try:
                # Текст для аналізу
                text_to_analyze = self._get_text_to_analyze(news)

                # Обмеження довжини тексту для BERT
                max_length = 512
                if len(text_to_analyze) > max_length * 3:  # Якщо текст дуже довгий
                    # Аналізуємо заголовок та початок контенту
                    if isinstance(news, NewsItem):
                        title = news.title or ""
                        content_start = (news.content or "")[:max_length] if news.content else ""
                        text_to_analyze = f"{title} {content_start}"
                    else:
                        title = news.get('title', '')
                        content = news.get('content', '')
                        content_start = content[:max_length] if content else ""
                        text_to_analyze = f"{title} {content_start}"

                # Використовуємо BERT для аналізу тональності
                try:
                    sentiment_result = self.sentiment_analyzer(text_to_analyze[:512])  # Обмеження по довжині

                    # Перетворення результату у потрібний формат
                    label = sentiment_result[0]['label']
                    score = sentiment_result[0]['score']

                    # Мапування міток BERT на наші категорії (1-2: негативний, 3: нейтральний, 4-5: позитивний)
                    if '1' in label or '2' in label:
                        sentiment_label = 'negative'
                        normalized_score = -score
                    elif '4' in label or '5' in label:
                        sentiment_label = 'positive'
                        normalized_score = score
                    else:
                        sentiment_label = 'neutral'
                        normalized_score = 0

                    sentiment_data = {
                        'score': normalized_score,
                        'label': sentiment_label,
                        'confidence': score,
                        'analyzed': True,
                        'raw_label': label
                    }
                except Exception as bert_error:
                    self.logger.warning(f"Помилка BERT аналізу, використовуємо лексичний метод: {bert_error}")
                    # Запасний варіант - лексичний аналіз
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
        """Лексичний аналіз тональності як запасний варіант."""
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
            'analyzed': True,
            'method': 'lexicon'
        }

    def extract_mentioned_coins(self, news_data: List[Union[Dict[str, Any], NewsItem]]) -> List[
        Union[Dict[str, Any], NewsItem]]:
        """Витягнення згаданих криптовалют з новин."""
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
        """Фільтрація новин за ключовими словами."""
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
        """Розрахунок оцінки важливості новин з покращеннями BERT."""
        self.logger.info(f"Початок розрахунку оцінки важливості для {len(news_data)} новин")

        for item in news_data:
            try:
                importance_score = 0.0
                bert_importance_score = 0.0
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

                # 6. НОВИЙ: BERT-аналіз для визначення важливості
                try:
                    # Використовуємо лише перші 512 токенів для аналізу BERT
                    truncated_text = text_to_analyze[:1000]  # Обмежуємо довжину тексту
                    embeddings = self._get_bert_embeddings(truncated_text)

                    # Використовуємо інтенсивність BERT-векторів як показник важливості
                    # Нормалізуємо вектор і використовуємо його норму як міру "сили" тексту
                    vector_intensity = np.linalg.norm(embeddings) / np.sqrt(embeddings.shape[1])

                    # Масштабуємо до діапазону [0, 0.3] для BERT-компонента
                    bert_importance_score = min(0.3, vector_intensity * 0.3)

                    # Додаємо BERT-компонент до загальної оцінки важливості
                    importance_score += bert_importance_score

                except Exception as bert_error:
                    self.logger.warning(f"Помилка при BERT-аналізі важливості: {bert_error}")

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
                        'sentiment_label': sentiment_label,
                        'bert_component': bert_importance_score
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
        """Категоризація новин з використанням BERT."""
        self.logger.info(f"Початок категоризації {len(news_data)} новин")

        for item in news_data:
            try:
                text_to_analyze = self._get_text_to_analyze(item)

                # Традиційний підхід - пошук ключових слів
                # Словник для підрахунку збігів по кожній категорії
                category_scores = {category: 0 for category in self.CRITICAL_KEYWORDS.keys()}

                # Шукаємо збіги за кожною категорією
                for category, patterns in self.category_patterns.items():
                    for pattern in patterns:
                        matches = pattern.findall(text_to_analyze)
                        category_scores[category] += len(matches)

                # BERT для покращення категоризації
                try:
                    # Отримуємо вектор BERT
                    truncated_text = text_to_analyze[:1000]  # Обмежуємо для BERT
                    embeddings = self._get_bert_embeddings(truncated_text)

                    # Покращення категоризації на основі BERT-векторів
                    # Цей підхід може бути розширений для більш точної категоризації

                    # Визначаємо найбільш імовірні категорії
                    sorted_categories = sorted(
                        category_scores.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )

                    # Додаємо дані про категорії до об'єкта новини
                    if isinstance(item, NewsItem):
                        item.categories = [cat for cat, score in sorted_categories if score > 0]
                    else:
                        item['categories'] = {
                            'primary': sorted_categories[0][0] if sorted_categories and sorted_categories[0][
                                1] > 0 else None,
                            'all': {cat: score for cat, score in sorted_categories if score > 0}
                        }

                except Exception as bert_error:
                    self.logger.warning(f"Помилка при BERT-категоризації: {bert_error}")
                    # Використовуємо тільки традиційний підхід у випадку помилки
                    if isinstance(item, NewsItem):
                        item.categories = [cat for cat, score in category_scores.items() if score > 0]
                    else:
                        item['categories'] = {
                            'primary': max(category_scores.items(), key=lambda x: x[1])[0] if any(
                                category_scores.values()) else None,
                            'all': {cat: score for cat, score in category_scores.items() if score > 0}
                        }

            except Exception as e:
                self.logger.error(f"Помилка при категоризації новини: {e}")
                # Додаємо порожні категорії у випадку помилки
                if isinstance(item, NewsItem):
                    item.categories = []
                else:
                    item['categories'] = {'primary': None, 'all': {}, 'error': str(e)}

        self.logger.info("Категоризація новин завершена")
        return news_data

    def _load_topic_models(self):
        """Завантажити збережені моделі тематичного моделювання."""
        try:
            os.makedirs(self.topic_model_dir, exist_ok=True)
            vectorizer_path = os.path.join(self.topic_model_dir, 'vectorizer.joblib')
            lda_path = os.path.join(self.topic_model_dir, 'lda_model.joblib')
            nmf_path = os.path.join(self.topic_model_dir, 'nmf_model.joblib')
            kmeans_path = os.path.join(self.topic_model_dir, 'kmeans_model.joblib')
            topics_path = os.path.join(self.topic_model_dir, 'topic_words.json')

            if os.path.exists(vectorizer_path):
                self.logger.info("Завантаження збережених моделей тематичного моделювання")
                self.vectorizer = joblib.load(vectorizer_path)

                if os.path.exists(lda_path):
                    self.lda_model = joblib.load(lda_path)

                if os.path.exists(nmf_path):
                    self.nmf_model = joblib.load(nmf_path)

                if os.path.exists(kmeans_path):
                    self.kmeans_model = joblib.load(kmeans_path)

                if os.path.exists(topics_path):
                    with open(topics_path, 'r', encoding='utf-8') as f:
                        self.topic_words = json.load(f)

                self.logger.info("Моделі тематичного моделювання успішно завантажено")
            else:
                self.logger.info("Збережені моделі не знайдено, моделі будуть створені при першому запуску")
        except Exception as e:
            self.logger.error(f"Помилка при завантаженні моделей тематичного моделювання: {e}")
            # В разі помилки створюємо нові об'єкти
            self.vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=10000, stop_words='english')
            self.lda_model = None
            self.nmf_model = None
            self.kmeans_model = None
            self.topic_words = {}

    def _save_topic_models(self):
        """Зберігти моделі тематичного моделювання на диск."""
        try:
            os.makedirs(self.topic_model_dir, exist_ok=True)
            vectorizer_path = os.path.join(self.topic_model_dir, 'vectorizer.joblib')
            lda_path = os.path.join(self.topic_model_dir, 'lda_model.joblib')
            nmf_path = os.path.join(self.topic_model_dir, 'nmf_model.joblib')
            kmeans_path = os.path.join(self.topic_model_dir, 'kmeans_model.joblib')
            topics_path = os.path.join(self.topic_model_dir, 'topic_words.json')

            if self.vectorizer:
                joblib.dump(self.vectorizer, vectorizer_path)

            if self.lda_model:
                joblib.dump(self.lda_model, lda_path)

            if self.nmf_model:
                joblib.dump(self.nmf_model, nmf_path)

            if self.kmeans_model:
                joblib.dump(self.kmeans_model, kmeans_path)

            if self.topic_words:
                with open(topics_path, 'w', encoding='utf-8') as f:
                    json.dump(self.topic_words, f, ensure_ascii=False, indent=2)

            self.logger.info("Моделі тематичного моделювання збережено на диск")
        except Exception as e:
            self.logger.error(f"Помилка при збереженні моделей: {e}")

    def discover_topics(self, news_data: List[Union[Dict[str, Any], NewsItem]], num_topics=10, force_retrain=False) -> \
    List[Union[Dict[str, Any], NewsItem]]:
        """Виявлення тем у наборі новин та призначення тем новинам."""
        self.logger.info(f"Початок виявлення тем у {len(news_data)} новинах")

        # Отримуємо тексти для аналізу
        texts = [self._get_text_to_analyze(item) for item in news_data]

        # Перевіряємо, чи потрібно створювати нові моделі
        if force_retrain or not self.lda_model or not self.nmf_model or not self.topic_words:
            self.logger.info("Створення нових моделей тематичного моделювання")

            # Створюємо матрицю TF-IDF
            if not self.vectorizer:
                self.vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=10000, stop_words='english')

            try:
                tfidf_matrix = self.vectorizer.fit_transform(texts)

                # Створюємо LDA модель
                self.lda_model = LatentDirichletAllocation(
                    n_components=num_topics,
                    max_iter=10,
                    learning_method='online',
                    random_state=42
                )
                self.lda_model.fit(tfidf_matrix)

                # Створюємо NMF модель
                self.nmf_model = NMF(
                    n_components=num_topics,
                    random_state=42,
                    alpha=.1,
                    l1_ratio=.5
                )
                self.nmf_model.fit(tfidf_matrix)

                # Створюємо K-means модель
                self.kmeans_model = KMeans(
                    n_clusters=num_topics,
                    random_state=42
                )
                self.kmeans_model.fit(tfidf_matrix)

                # Отримуємо слова, що характеризують кожну тему
                feature_names = self.vectorizer.get_feature_names_out()

                # Теми LDA
                lda_topics = {}
                for topic_idx, topic in enumerate(self.lda_model.components_):
                    top_words_idx = topic.argsort()[:-10 - 1:-1]
                    top_words = [feature_names[i] for i in top_words_idx]
                    lda_topics[f'topic_{topic_idx + 1}'] = top_words

                # Теми NMF
                nmf_topics = {}
                for topic_idx, topic in enumerate(self.nmf_model.components_):
                    top_words_idx = topic.argsort()[:-10 - 1:-1]
                    top_words = [feature_names[i] for i in top_words_idx]
                    nmf_topics[f'topic_{topic_idx + 1}'] = top_words

                # Зберігаємо теми
                self.topic_words = {
                    'lda': lda_topics,
                    'nmf': nmf_topics
                }

                # Зберігаємо моделі
                self._save_topic_models()

            except Exception as e:
                self.logger.error(f"Помилка при створенні моделей тематичного моделювання: {e}")
                return news_data

        # Призначаємо теми для кожної новини
        try:
            for idx, item in enumerate(news_data):
                text = self._get_text_to_analyze(item)

                # Перетворюємо текст у вектор TF-IDF
                text_tfidf = self.vectorizer.transform([text])

                # Отримуємо розподіл тем LDA
                lda_distribution = self.lda_model.transform(text_tfidf)[0]
                lda_top_topic = lda_distribution.argmax()

                # Отримуємо розподіл тем NMF
                nmf_distribution = self.nmf_model.transform(text_tfidf)[0]
                nmf_top_topic = nmf_distribution.argmax()

                # Отримуємо кластер K-means
                kmeans_cluster = self.kmeans_model.predict(text_tfidf)[0]

                # Формуємо результат
                topic_data = {
                    'lda': {
                        'top_topic': int(lda_top_topic),
                        'topic_words': self.topic_words['lda'][f'topic_{lda_top_topic + 1}'],
                        'distribution': {f'topic_{i + 1}': float(score) for i, score in enumerate(lda_distribution)}
                    },
                    'nmf': {
                        'top_topic': int(nmf_top_topic),
                        'topic_words': self.topic_words['nmf'][f'topic_{nmf_top_topic + 1}'],
                        'distribution': {f'topic_{i + 1}': float(score) for i, score in enumerate(nmf_distribution)}
                    },
                    'kmeans_cluster': int(kmeans_cluster)
                }

                # Додаємо результат до об'єкта новини
                if isinstance(item, NewsItem):
                    item.topics = topic_data
                else:
                    item['topics'] = topic_data

                # Логування прогресу
                if idx > 0 and idx % 50 == 0:
                    self.logger.info(f"Призначено темам {idx}/{len(news_data)} новин")

        except Exception as e:
            self.logger.error(f"Помилка при призначенні тем новинам: {e}")

        self.logger.info("Виявлення тем завершено")
        return news_data

    def preprocess_text(self, text):
        """Попередня обробка тексту для аналізу."""
        # Перевірка доступності nltk
        if not nltk_available:
            # Базова обробка без nltk
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            words = text.split()
            words = [word for word in words if word not in self.COMMON_WORDS]
            return ' '.join(words)
        else:
            try:
                # Розширена обробка з nltk
                text = text.lower()
                # Видалення спеціальних символів
                text = re.sub(r'[^\w\s]', '', text)
                # Токенізація
                tokens = word_tokenize(text)
                # Видалення стоп-слів
                stop_words = set(stopwords.words('english'))
                tokens = [word for word in tokens if word not in stop_words]
                # Лематизація
                lemmatizer = WordNetLemmatizer()
                tokens = [lemmatizer.lemmatize(word) for word in tokens]
                return ' '.join(tokens)
            except Exception as e:
                self.logger.warning(f"Помилка при обробці тексту з nltk: {e}. Використовуємо базову обробку.")
                # Базова обробка у випадку помилки
                text = text.lower()
                text = re.sub(r'[^\w\s]', '', text)
                words = text.split()
                words = [word for word in words if word not in self.COMMON_WORDS]
                return ' '.join(words)

    def save_analysis_results(self, news_data: List[Union[Dict[str, Any], NewsItem]]):
        """Зберегти результати аналізу в базу даних."""
        if not self.db_manager:
            self.logger.warning("Відсутній менеджер бази даних. Результати не будуть збережені.")
            return

        self.logger.info(f"Збереження результатів аналізу для {len(news_data)} новин")

        # Словники для зведених даних
        sentiment_time_series = {}

        for item in news_data:
            try:
                # Отримуємо ідентифікатор новини
                if isinstance(item, NewsItem):
                    news_id = item.url or f"news_{hash(item.title)}"
                    title = item.title
                    published_at = item.published_at
                    sentiment_score = item.sentiment_score
                    sentiment_label = item.sentiment_label
                    importance_score = item.importance_score
                    mentioned_cryptos = item.mentioned_cryptos
                    categories = item.categories
                    topics = item.topics
                else:
                    news_id = item.get('url', f"news_{hash(item.get('title', ''))}")
                    title = item.get('title', '')
                    published_at = item.get('published_at', datetime.now())
                    sentiment_data = item.get('sentiment', {})
                    sentiment_score = sentiment_data.get('score', 0.0)
                    sentiment_label = sentiment_data.get('label', 'neutral')
                    importance_score = item.get('importance_score', 0.3)
                    mentioned_cryptos = item.get('mentioned_coins', {}).get('coins', {})
                    categories = item.get('categories', {}).get('all', {})
                    topics = item.get('topics', {})

                # Формуємо дату для часового ряду
                date_key = published_at.strftime('%Y-%m-%d') if isinstance(published_at,
                                                                           datetime) else datetime.now().strftime(
                    '%Y-%m-%d')

                # Зберігаємо аналіз настроїв для статті
                try:
                    article = self.db_manager.get_article(news_id)
                    if article:
                        # Формуємо дані для збереження
                        sentiment_data = {
                            'sentiment_score': sentiment_score,
                            'sentiment_label': sentiment_label,
                            'importance_score': importance_score,
                            'mentioned_cryptos': mentioned_cryptos,
                            'categories': categories,
                            'topics': topics,
                            'analyzed_at': datetime.now().isoformat()
                        }

                        # Зберігаємо результати аналізу
                        self.db_manager.save_sentiment_analysis(news_id, sentiment_data)

                        # Додаємо дані до часового ряду
                        if date_key not in sentiment_time_series:
                            sentiment_time_series[date_key] = {
                                'total': 0,
                                'positive': 0,
                                'negative': 0,
                                'neutral': 0,
                                'sentiment_sum': 0.0,
                                'importance_sum': 0.0,
                                'crypto_mentions': {}
                            }

                        # Оновлюємо дані часового ряду
                        sentiment_time_series[date_key]['total'] += 1
                        sentiment_time_series[date_key][sentiment_label] += 1
                        sentiment_time_series[date_key]['sentiment_sum'] += sentiment_score
                        sentiment_time_series[date_key]['importance_sum'] += importance_score

                        # Додаємо згадки криптовалют
                        for crypto, count in mentioned_cryptos.items():
                            if crypto not in sentiment_time_series[date_key]['crypto_mentions']:
                                sentiment_time_series[date_key]['crypto_mentions'][crypto] = 0
                            sentiment_time_series[date_key]['crypto_mentions'][crypto] += count
                    else:
                        self.logger.warning(f"Статтю не знайдено в базі даних: {news_id}")
                except Exception as db_error:
                    self.logger.error(f"Помилка при збереженні аналізу настроїв: {db_error}")

            except Exception as e:
                self.logger.error(f"Помилка при підготовці даних для збереження: {e}")

        # Зберігаємо зведені дані часового ряду
        try:
            for date_key, data in sentiment_time_series.items():
                # Розрахунок середніх значень
                total = data['total']
                if total > 0:
                    data['avg_sentiment'] = data['sentiment_sum'] / total
                    data['avg_importance'] = data['importance_sum'] / total
                else:
                    data['avg_sentiment'] = 0.0
                    data['avg_importance'] = 0.0

                # Видаляємо проміжні суми
                del data['sentiment_sum']
                del data['importance_sum']

                # Зберігаємо дані часового ряду
                self.db_manager.save_news_sentiment_time_series(date_key, data)
        except Exception as ts_error:
            self.logger.error(f"Помилка при збереженні часового ряду настроїв: {ts_error}")

        self.logger.info("Збереження результатів аналізу завершено")

    def analyze_news_batch(self, news_data: List[Union[Dict[str, Any], NewsItem]], save_results=True) -> List[
        Union[Dict[str, Any], NewsItem]]:
        """Виконати комплексний аналіз пакету новин."""
        self.logger.info(f"Початок комплексного аналізу пакету з {len(news_data)} новин")

        try:
            # 1. Аналіз тональності
            news_data = self.analyze_news_sentiment(news_data)

            # 2. Витягнення згаданих криптовалют
            news_data = self.extract_mentioned_coins(news_data)

            # 3. Розрахунок оцінки важливості
            news_data = self.calculate_importance_score(news_data)

            # 4. Категоризація новин
            news_data = self.categorize_news(news_data)

            # 5. Виявлення тем (якщо достатньо новин)
            if len(news_data) >= 10:
                news_data = self.discover_topics(news_data)

            # 6. Збереження результатів аналізу (за потреби)
            if save_results and self.db_manager:
                self.save_analysis_results(news_data)

            self.logger.info("Комплексний аналіз пакету новин завершено успішно")
        except Exception as e:
            self.logger.error(f"Помилка при комплексному аналізі пакету новин: {e}")

        return news_data

    def analyze_single_news(self, news_item: Union[Dict[str, Any], NewsItem], save_result=True) -> Union[
        Dict[str, Any], NewsItem]:
        """Виконати комплексний аналіз однієї новини."""
        self.logger.info("Аналіз однієї новини")

        try:
            # Створюємо список з одного елемента для використання існуючих методів
            news_data = [news_item]

            # Аналіз тональності
            news_data = self.analyze_news_sentiment(news_data)

            # Витягнення згаданих криптовалют
            news_data = self.extract_mentioned_coins(news_data)

            # Розрахунок оцінки важливості
            news_data = self.calculate_importance_score(news_data)

            # Категоризація новини
            news_data = self.categorize_news(news_data)

            # Для одинарної новини не можемо виявити теми, але можемо спробувати
            # використати існуючу модель тематичного моделювання, якщо вона є
            if self.lda_model and self.nmf_model:
                news_data = self.discover_topics(news_data, force_retrain=False)

            # Збереження результату аналізу (за потреби)
            if save_result and self.db_manager:
                self.save_analysis_results(news_data)

            self.logger.info("Аналіз новини завершено успішно")
            return news_data[0]
        except Exception as e:
            self.logger.error(f"Помилка при аналізі новини: {e}")
            return news_item

