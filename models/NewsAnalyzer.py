import os
import re
import json
import logging
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Union, Any, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertModel,
    BertForSequenceClassification,
    pipeline,
    BatchEncoding,
    AutoTokenizer,
    AutoModel
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from utils.config import *
from data.NewsManager import DatabaseManager
# NLTK imports with proper error handling
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    # Ensure necessary NLTK resources are downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)

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


class NewsDataset(Dataset):
    """Dataset для ефективної обробки текстових даних у BERT."""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Remove batch dimension added by tokenizer when return_tensors='pt'
        return {k: v.squeeze(0) for k, v in encoding.items()}


class BERTNewsAnalyzer:

    def __init__(self,
                 bert_model_name='bert-base-uncased',
                 sentiment_model_name='nlptown/bert-base-multilingual-uncased-sentiment',
                 embedding_model_name=None,  # Optional different model for embeddings
                 db_manager=None,
                 logger=None,
                 topic_model_dir='./models/topics',
                 device=None,
                 batch_size=8,
                 cache_dir='./models/cache'):

        self.logger = logger or logging.getLogger("BERTNewsAnalyzer")
        self.logger.setLevel(logging.INFO)
        self.db_manager = DatabaseManager()
        self.CRYPTO_KEYWORDS = CRYPTO_KEYWORDS
        self.SENTIMENT_LEXICON = SENTIMENT_LEXICON
        self.CRITICAL_KEYWORDS = CRITICAL_KEYWORDS
        self.MAJOR_ENTITIES = MAJOR_ENTITIES
        self.COMMON_WORDS = COMMON_WORDS
        self.ML_FEATURE_GROUPS = ML_FEATURE_GROUPS
        self.topic_model_dir = topic_model_dir
        self.batch_size = batch_size
        self.cache_dir = cache_dir

        # Створюємо каталог для кешу моделей, якщо він не існує
        os.makedirs(self.cache_dir, exist_ok=True)

        # Налаштування логування, якщо логгер не передано
        if not logger:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Визначення пристрою (CPU або GPU)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Використовується пристрій: {self.device}")

        # Встановлюємо embedding_model_name на bert_model_name, якщо не вказано
        embedding_model_name = embedding_model_name or bert_model_name

        # Завантаження моделей BERT і токенізаторів
        try:
            # Базова модель та токенізатор для ембеддінгів
            self.logger.info(f"Завантаження моделі для ембеддінгів: {embedding_model_name}")
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(
                embedding_model_name,
                cache_dir=os.path.join(self.cache_dir, 'embedding_tokenizer')
            )
            self.embedding_model = AutoModel.from_pretrained(
                embedding_model_name,
                cache_dir=os.path.join(self.cache_dir, 'embedding_model')
            ).to(self.device)

            self.logger.info(f"Завантаження моделі BERT для аналізу тональності: {sentiment_model_name}")
            # Використовуємо готовий pipeline для аналізу тональності
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=sentiment_model_name,
                tokenizer=sentiment_model_name,
                device=0 if self.device == 'cuda' else -1,
                cache_dir=os.path.join(self.cache_dir, 'sentiment_model')
            )

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

    def _batch_encode_texts(self, texts: List[str], max_length: int = 512) -> BatchEncoding:
        """Ефективне батчове кодування текстів для BERT."""
        return self.embedding_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)

    def _get_bert_embeddings_batch(self, texts: List[str], max_length: int = 512,
                                   pooling_strategy: str = 'cls') -> np.ndarray:

        # Створюємо dataset і dataloader для ефективної обробки
        dataset = NewsDataset(texts, self.embedding_tokenizer, max_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        all_embeddings = []

        # Обробляємо батчами
        with torch.no_grad():
            for batch in dataloader:
                # Переміщуємо весь батч на потрібний пристрій
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Отримання результатів від моделі
                outputs = self.embedding_model(**batch)

                # Стратегія пулінгу
                if pooling_strategy == 'cls':
                    # Використовуємо [CLS] токен
                    batch_embeddings = outputs.last_hidden_state[:, 0, :]
                elif pooling_strategy == 'mean':
                    # Середнє значення по всіх токенах (з урахуванням маски уваги)
                    # Створюємо маску: 1 для справжніх токенів, 0 для padding
                    attention_mask = batch['attention_mask'].unsqueeze(-1)
                    # Множимо на маску і обчислюємо середнє
                    sum_embeddings = torch.sum(outputs.last_hidden_state * attention_mask, dim=1)
                    sum_mask = torch.sum(attention_mask, dim=1)
                    batch_embeddings = sum_embeddings / sum_mask
                elif pooling_strategy == 'max':
                    # Максимальне значення по всіх токенах (з урахуванням маски)
                    # Спочатку замінюємо всі padding токени на великі від'ємні значення
                    attention_mask = batch['attention_mask'].unsqueeze(-1)
                    # Де маска = 0, ставимо -1e9 (дуже мале число)
                    masked_embeddings = outputs.last_hidden_state * attention_mask + (1 - attention_mask) * -1e9
                    batch_embeddings = torch.max(masked_embeddings, dim=1)[0]
                else:
                    raise ValueError(f"Непідтримувана стратегія пулінгу: {pooling_strategy}")

                # Додаємо ембеддінги до списку
                all_embeddings.append(batch_embeddings.cpu().numpy())

        # Об'єднуємо всі батчі в одну матрицю
        return np.vstack(all_embeddings)

    def _get_bert_embeddings(self, text: str, max_length: int = 512,
                             pooling_strategy: str = 'cls') -> np.ndarray:
        """Отримати вектори BERT для одного тексту."""
        # Для одного тексту використовуємо батчовий метод
        embeddings = self._get_bert_embeddings_batch([text], max_length, pooling_strategy)
        return embeddings[0]

    def analyze_news_sentiment(self, news_data: List[Union[Dict[str, Any], NewsItem]]) -> List[
        Union[Dict[str, Any], NewsItem]]:
        """Аналіз тональності новин за допомогою BERT з ефективною батчовою обробкою."""
        self.logger.info(f"Початок аналізу настроїв для {len(news_data)} новин")

        analyzed_news = []

        # Групуємо новини у батчі для ефективної обробки
        batch_size = min(self.batch_size, len(news_data))
        num_batches = (len(news_data) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(news_data))
            batch_news = news_data[start_idx:end_idx]

            # Підготовка текстів для аналізу
            batch_texts = []
            for news in batch_news:
                text_to_analyze = self._get_text_to_analyze(news)

                # Обмеження довжини тексту для BERT
                max_length = 512
                if len(text_to_analyze) > max_length * 3:
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

                # Обмежуємо довжину для токенізатора BERT
                text_to_analyze = text_to_analyze[:1024]  # Обмежуємо до 1024 символів
                batch_texts.append(text_to_analyze)

            try:
                # Батчовий аналіз тональності через pipeline
                sentiment_results = self.sentiment_analyzer(batch_texts, truncation=True, max_length=512)

                # Обробка результатів для кожної новини в батчі
                for idx, (news, sentiment_result) in enumerate(zip(batch_news, sentiment_results)):
                    try:
                        # Перетворення результату у потрібний формат
                        label = sentiment_result['label']
                        score = sentiment_result['score']

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
                        self.logger.warning(
                            f"Помилка BERT аналізу для новини {start_idx + idx}, використовуємо лексичний метод: {bert_error}")
                        # Запасний варіант - лексичний аналіз
                        sentiment_data = self._lexicon_based_sentiment(batch_texts[idx])

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

            except Exception as batch_error:
                self.logger.error(f"Помилка при батчовому аналізі настроїв: {batch_error}")

                # У випадку помилки батчу, обробляємо кожну новину окремо з запасним методом
                for idx, news in enumerate(batch_news):
                    try:
                        text = batch_texts[idx]
                        sentiment_data = self._lexicon_based_sentiment(text)

                        # Додаємо результат
                        if isinstance(news, NewsItem):
                            news.sentiment_score = sentiment_data['score']
                            news.sentiment_label = sentiment_data['label']
                            analyzed_news.append(news)
                        else:
                            news_copy = news.copy()
                            news_copy['sentiment'] = sentiment_data
                            analyzed_news.append(news_copy)
                    except Exception as e:
                        self.logger.error(f"Помилка при аналізі окремої новини: {e}")
                        # Додаємо новину з нейтральним настроєм
                        if isinstance(news, NewsItem):
                            news.sentiment_score = 0.0
                            news.sentiment_label = 'neutral'
                            analyzed_news.append(news)
                        else:
                            news_copy = news.copy()
                            news_copy['sentiment'] = {
                                'score': 0.0,
                                'label': 'neutral',
                                'confidence': 0.0,
                                'analyzed': False,
                                'error': str(e)
                            }
                            analyzed_news.append(news_copy)

            # Логування прогресу
            self.logger.info(f"Проаналізовано {end_idx}/{len(news_data)} новин")

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

        # Групуємо новини у батчі для ефективної обробки векторизації BERT
        batch_size = min(self.batch_size, len(news_data))
        num_batches = (len(news_data) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(news_data))
            batch_news = news_data[start_idx:end_idx]

            # Підготовка текстів для аналізу
            batch_texts = []
            for item in batch_news:
                text_to_analyze = self._get_text_to_analyze(item)
                # Обмежуємо довжину для ефективної обробки
                truncated_text = text_to_analyze[:1000]
                batch_texts.append(truncated_text)

            try:
                # Отримуємо BERT-ембеддінги для всього батчу
                batch_embeddings = self._get_bert_embeddings_batch(batch_texts, max_length=512, pooling_strategy='cls')

                # Обробляємо кожну новину в батчі
                for idx, (item, embeddings) in enumerate(zip(batch_news, batch_embeddings)):
                    try:
                        importance_score = 0.0
                        bert_importance_score = 0.0
                        text_to_analyze = batch_texts[idx]

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
                        critical_categories = ['regulation', 'hack', 'market_crash', 'major_investment']
                        for category in critical_categories:
                            patterns = self.category_patterns.get(category, [])
                            matches = sum(1 for pattern in patterns if pattern.search(text_to_analyze))
                            if matches > 0:
                                # Критичні події мають високий вплив на важливість
                                importance_score += 0.4 * min(matches, 3) / 3  # Максимум 0.4 за категорію

                        # 3. Оцінка за належністю до мейджорів (великих компаній/організацій)
                        for entity in self.MAJOR_ENTITIES:
                            if re.search(rf'\b{re.escape(entity)}\b', text_to_analyze, re.IGNORECASE):
                                importance_score += 0.15

                        # 4. Використання BERT-ембеддінгів для оцінки важливості тексту
                        # Обчислюємо косинусну подібність з ембеддінгами важливих категорій
                        # або використовуємо магнітуду вектору як показник важливості тексту
                        embedding_magnitude = np.linalg.norm(embeddings)
                        # Нормалізація величини до діапазону [0, 0.3]
                        normalized_magnitude = min(0.3, embedding_magnitude / 100)
                        bert_importance_score += normalized_magnitude

                        # 5. Вплив абсолютного значення тональності (важливіше)
                        sentiment_score = 0.0
                        if isinstance(item, NewsItem):
                            sentiment_score = abs(item.sentiment_score) if item.sentiment_score is not None else 0.0
                        elif isinstance(item, dict) and 'sentiment' in item:
                            sentiment_score = abs(item['sentiment'].get('score', 0.0))

                        importance_score += sentiment_score * 0.2  # Коефіцієнт впливу тональності

                        # 6. Об'єднуємо звичайну оцінку та BERT-оцінку
                        final_score = importance_score + bert_importance_score

                        # Обмежуємо значення між 0 і 1
                        final_score = max(0.0, min(1.0, final_score))

                        # Зберігаємо результат
                        if isinstance(item, NewsItem):
                            item.importance_score = final_score
                        else:
                            item['importance'] = {
                                'score': final_score,
                                'bert_component': bert_importance_score,
                                'traditional_component': importance_score,
                                'analyzed': True
                            }

                    except Exception as inner_error:
                        self.logger.error(f"Помилка аналізу важливості для новини {start_idx + idx}: {inner_error}")
                        # Задаємо стандартне значення у випадку помилки
                        if isinstance(item, NewsItem):
                            item.importance_score = 0.5  # Середня важливість за замовчуванням
                        else:
                            item['importance'] = {
                                'score': 0.5,
                                'analyzed': False,
                                'error': str(inner_error)
                            }

            except Exception as batch_error:
                self.logger.error(f"Помилка при батчовому аналізі важливості: {batch_error}")
                # У випадку помилки батчу, обробляємо кожну новину окремо зі спрощеним методом
                for idx, item in enumerate(batch_news):
                    try:
                        # Спрощений розрахунок важливості
                        text_to_analyze = batch_texts[idx]

                        basic_score = 0.5  # Початкова оцінка

                        # Пошук ключових слів для оцінки важливості
                        for keyword in ['important', 'breaking', 'urgent', 'critical', 'major']:
                            if re.search(rf'\b{re.escape(keyword)}\b', text_to_analyze, re.IGNORECASE):
                                basic_score += 0.1

                        basic_score = min(1.0, basic_score)

                        # Зберігаємо результат
                        if isinstance(item, NewsItem):
                            item.importance_score = basic_score
                        else:
                            item['importance'] = {
                                'score': basic_score,
                                'analyzed': False,
                                'method': 'basic'
                            }
                    except Exception as e:
                        self.logger.error(f"Помилка при спрощеному аналізі важливості: {e}")
                        # Задаємо стандартне значення
                        if isinstance(item, NewsItem):
                            item.importance_score = 0.5
                        else:
                            item['importance'] = {
                                'score': 0.5,
                                'analyzed': False,
                                'error': str(e)
                            }

            # Логування прогресу
            self.logger.info(f"Розраховано важливість для {end_idx}/{len(news_data)} новин")

        self.logger.info("Розрахунок оцінки важливості завершено")
        return news_data

    def extract_topics(self, news_data: List[Union[Dict[str, Any], NewsItem]],
                       n_topics: int = 10,
                       retrain: bool = False,
                       min_docs: int = 50) -> List[Union[Dict[str, Any], NewsItem]]:

        self.logger.info(f"Початок витягнення тем з {len(news_data)} новин")

        if len(news_data) < min_docs and self.vectorizer is None:
            self.logger.warning(f"Недостатньо документів для навчання моделей тематичного моделювання. "
                                f"Потрібно: {min_docs}, є: {len(news_data)}. Використовуємо альтернативний метод.")
            return self._extract_topics_basic(news_data)

        # Підготовка текстів для аналізу
        texts = []
        for item in news_data:
            text_to_analyze = self._get_text_to_analyze(item)
            # Очищення та підготовка тексту
            cleaned_text = self._preprocess_text_for_topics(text_to_analyze)
            texts.append(cleaned_text)

        # Перевірка існування моделей або перенавчання за необхідності
        if self.vectorizer is None or retrain:
            self._train_topic_models(texts, n_topics)

        try:
            # Трансформація текстів для всіх моделей
            X = self.vectorizer.transform(texts)

            # Отримання тем для кожної моделі
            lda_topics = self.lda_model.transform(X) if self.lda_model else None
            nmf_topics = self.nmf_model.transform(X) if self.nmf_model else None

            # Отримання BERT ембеддінгів для кластеризації
            batch_size = 32
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self._get_bert_embeddings_batch(batch_texts, max_length=512)
                all_embeddings.append(batch_embeddings)

            embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])

            # K-Means кластеризація на основі ембеддінгів
            if self.kmeans_model is None or retrain:
                if len(embeddings) >= n_topics:
                    self.kmeans_model = KMeans(n_clusters=n_topics, random_state=42)
                    self.kmeans_model.fit(embeddings)
                else:
                    self.kmeans_model = None

            kmeans_clusters = self.kmeans_model.predict(embeddings) if self.kmeans_model else None

            # Обробка результатів для кожної новини
            for idx, item in enumerate(news_data):
                topics_data = {}

                # LDA теми
                if lda_topics is not None:
                    lda_topic_dist = lda_topics[idx]
                    top_lda_topics = sorted(enumerate(lda_topic_dist), key=lambda x: x[1], reverse=True)[:3]
                    topics_data['lda'] = {
                        'top_topics': [{'id': t[0], 'score': float(t[1]),
                                        'keywords': self.topic_words.get(f'lda_{t[0]}', [])}
                                       for t in top_lda_topics if t[1] > 0.1]
                    }

                # NMF теми
                if nmf_topics is not None:
                    nmf_topic_dist = nmf_topics[idx]
                    top_nmf_topics = sorted(enumerate(nmf_topic_dist), key=lambda x: x[1], reverse=True)[:3]
                    topics_data['nmf'] = {
                        'top_topics': [{'id': t[0], 'score': float(t[1]),
                                        'keywords': self.topic_words.get(f'nmf_{t[0]}', [])}
                                       for t in top_nmf_topics if t[1] > 0.1]
                    }

                # K-Means кластер
                if kmeans_clusters is not None:
                    cluster_id = int(kmeans_clusters[idx])
                    topics_data['kmeans'] = {
                        'cluster': cluster_id,
                        'keywords': self.topic_words.get(f'kmeans_{cluster_id}', [])
                    }

                # Запис результатів
                if isinstance(item, NewsItem):
                    item.topics = topics_data
                else:
                    item['topics'] = topics_data

        except Exception as e:
            self.logger.error(f"Помилка при витягненні тем: {e}")
            # У випадку помилки використовуємо спрощений метод
            return self._extract_topics_basic(news_data)

        self.logger.info("Витягнення тем завершено")
        return news_data

    def _extract_topics_basic(self, news_data: List[Union[Dict[str, Any], NewsItem]]) -> List[
        Union[Dict[str, Any], NewsItem]]:
        """Спрощений метод витягнення тем на основі регулярних виразів."""
        self.logger.info("Використовуємо спрощений метод витягнення тем")

        # Підготовка правил для тем
        topic_rules = {
            'price_movement': ['price', 'increase', 'decrease', 'fall', 'rise', 'bull', 'bear', 'market'],
            'regulation': ['regulation', 'law', 'government', 'ban', 'legal', 'sec', 'compliance'],
            'technology': ['blockchain', 'protocol', 'algorithm', 'mining', 'node', 'dapp', 'smart contract'],
            'adoption': ['adoption', 'partnership', 'integration', 'mainstream', 'use case', 'institutional'],
            'security': ['hack', 'scam', 'fraud', 'security', 'theft', 'vulnerability', 'attack', 'breach'],
            'defi': ['defi', 'yield', 'farming', 'liquidity', 'swap', 'amm', 'dex', 'lending', 'borrowing'],
            'nft': ['nft', 'collectible', 'art', 'token', 'marketplace', 'auction', 'unique'],
            'innovation': ['innovation', 'update', 'upgrade', 'fork', 'development', 'research', 'feature']
        }

        # Підготовка патернів
        topic_patterns = {}
        for topic, keywords in topic_rules.items():
            patterns = [re.compile(rf'\b{re.escape(kw)}\b', re.IGNORECASE) for kw in keywords]
            topic_patterns[topic] = patterns

        # Аналіз кожної новини
        for item in news_data:
            text_to_analyze = self._get_text_to_analyze(item)

            # Пошук тем
            detected_topics = {}
            for topic, patterns in topic_patterns.items():
                matches = sum(1 for pattern in patterns if pattern.search(text_to_analyze))
                if matches > 0:
                    score = min(1.0, matches / len(patterns) * 2)  # Нормалізація оцінки
                    detected_topics[topic] = score

            # Сортування за оцінкою
            sorted_topics = sorted(detected_topics.items(), key=lambda x: x[1], reverse=True)

            # Формування результату
            topics_data = {
                'basic': {
                    'top_topics': [{'topic': t[0], 'score': t[1]} for t in sorted_topics[:3]],
                    'method': 'rule_based'
                }
            }

            # Запис результатів
            if isinstance(item, NewsItem):
                item.topics = topics_data
            else:
                item['topics'] = topics_data

        return news_data

    def _train_topic_models(self, texts: List[str], n_topics: int = 10):
        """Навчання моделей тематичного моделювання."""
        self.logger.info(f"Початок навчання моделей тематичного моделювання з {len(texts)} текстів")

        # Перевірка наявності директорії для збереження моделей
        os.makedirs(self.topic_model_dir, exist_ok=True)

        try:
            # 1. Створення векторизатора TF-IDF
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                min_df=3,
                max_df=0.9,
                stop_words='english'
            )
            X = self.vectorizer.fit_transform(texts)

            # 2. Навчання LDA моделі
            self.lda_model = LatentDirichletAllocation(
                n_components=n_topics,
                max_iter=20,
                learning_method='online',
                random_state=42,
                n_jobs=-1
            )
            self.lda_model.fit(X)

            # 3. Навчання NMF моделі
            self.nmf_model = NMF(
                n_components=n_topics,
                random_state=42,
                max_iter=200
            )
            self.nmf_model.fit(X)

            # 4. Отримання ключових слів для тем
            feature_names = self.vectorizer.get_feature_names_out()

            # Зберігаємо ключові слова для кожної теми LDA
            for topic_idx, topic in enumerate(self.lda_model.components_):
                top_words_idx = topic.argsort()[:-11:-1]  # Топ-10 слів
                topic_keywords = [feature_names[i] for i in top_words_idx]
                self.topic_words[f'lda_{topic_idx}'] = topic_keywords

            # Зберігаємо ключові слова для кожної теми NMF
            for topic_idx, topic in enumerate(self.nmf_model.components_):
                top_words_idx = topic.argsort()[:-11:-1]  # Топ-10 слів
                topic_keywords = [feature_names[i] for i in top_words_idx]
                self.topic_words[f'nmf_{topic_idx}'] = topic_keywords

            # 5. Підготовка ембеддінгів для K-Means
            # Ця частина виконується окремо при виконанні extract_topics

            # 6. Збереження моделей
            self._save_topic_models()

            self.logger.info("Моделі тематичного моделювання успішно навчені")

        except Exception as e:
            self.logger.error(f"Помилка при навчанні моделей тематичного моделювання: {e}")
            # Очищаємо моделі у випадку помилки
            self.vectorizer = None
            self.lda_model = None
            self.nmf_model = None
            self.kmeans_model = None

    def _save_topic_models(self):
        """Збереження навчених моделей тематичного моделювання."""
        try:
            if self.vectorizer:
                joblib.dump(self.vectorizer, os.path.join(self.topic_model_dir, 'vectorizer.joblib'))

            if self.lda_model:
                joblib.dump(self.lda_model, os.path.join(self.topic_model_dir, 'lda_model.joblib'))

            if self.nmf_model:
                joblib.dump(self.nmf_model, os.path.join(self.topic_model_dir, 'nmf_model.joblib'))

            if self.kmeans_model:
                joblib.dump(self.kmeans_model, os.path.join(self.topic_model_dir, 'kmeans_model.joblib'))

            # Збереження словника ключових слів тем
            with open(os.path.join(self.topic_model_dir, 'topic_words.json'), 'w') as f:
                json.dump(self.topic_words, f)

            self.logger.info("Моделі тематичного моделювання успішно збережено")

        except Exception as e:
            self.logger.error(f"Помилка при збереженні моделей тематичного моделювання: {e}")

    def _load_topic_models(self):
        """Завантаження збережених моделей тематичного моделювання."""
        try:
            vectorizer_path = os.path.join(self.topic_model_dir, 'vectorizer.joblib')
            if os.path.exists(vectorizer_path):
                self.vectorizer = joblib.load(vectorizer_path)

            lda_model_path = os.path.join(self.topic_model_dir, 'lda_model.joblib')
            if os.path.exists(lda_model_path):
                self.lda_model = joblib.load(lda_model_path)

            nmf_model_path = os.path.join(self.topic_model_dir, 'nmf_model.joblib')
            if os.path.exists(nmf_model_path):
                self.nmf_model = joblib.load(nmf_model_path)

            kmeans_model_path = os.path.join(self.topic_model_dir, 'kmeans_model.joblib')
            if os.path.exists(kmeans_model_path):
                self.kmeans_model = joblib.load(kmeans_model_path)

            topic_words_path = os.path.join(self.topic_model_dir, 'topic_words.json')
            if os.path.exists(topic_words_path):
                with open(topic_words_path, 'r') as f:
                    self.topic_words = json.load(f)

            self.logger.info("Моделі тематичного моделювання успішно завантажено")
            return True

        except Exception as e:
            self.logger.error(f"Помилка при завантаженні моделей тематичного моделювання: {e}")
            return False

    def _preprocess_text_for_topics(self, text: str) -> str:
        """Попередня обробка тексту для тематичного моделювання."""
        if not nltk_available:
            # Якщо NLTK недоступний, виконуємо базове очищення
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            # Видаляємо загальні слова вручну
            for word in self.COMMON_WORDS:
                text = re.sub(rf'\b{re.escape(word)}\b', '', text)
            return text

        # Якщо NLTK доступний, використовуємо повну обробку
        # Нижній регістр
        text = text.lower()

        # Видалення пунктуації та цифр
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)

        # Токенізація
        tokens = word_tokenize(text)

        # Видалення стоп-слів
        stop_words = set(stopwords.words('english'))
        stop_words.update(self.COMMON_WORDS)
        tokens = [token for token in tokens if token not in stop_words]

        # Лематизація
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

        # Видалення коротких слів
        tokens = [token for token in tokens if len(token) > 2]

        # Зіставлення назад у текст
        preprocessed_text = ' '.join(tokens)
        return preprocessed_text

    def analyze_news_batch(self, news_data: List[Union[Dict[str, Any], NewsItem]],
                           extract_sentiment: bool = True,
                           extract_coins: bool = True,
                           calculate_importance: bool = True,
                           extract_topics: bool = True) -> List[Union[Dict[str, Any], NewsItem]]:
        """Комплексний аналіз пакету новин з усіма доступними функціями."""
        self.logger.info(f"Початок комплексного аналізу {len(news_data)} новин")

        result = news_data

        try:
            # 1. Аналіз тональності
            if extract_sentiment:
                self.logger.info("Виконуємо аналіз тональності")
                result = self.analyze_news_sentiment(result)

            # 2. Витягнення згаданих криптовалют
            if extract_coins:
                self.logger.info("Виконуємо пошук згаданих криптовалют")
                result = self.extract_mentioned_coins(result)

            # 3. Розрахунок оцінки важливості
            if calculate_importance:
                self.logger.info("Виконуємо розрахунок оцінки важливості")
                result = self.calculate_importance_score(result)

            # 4. Витягнення тем
            if extract_topics:
                self.logger.info("Виконуємо витягнення тем")
                result = self.extract_topics(result)

        except Exception as e:
            self.logger.error(f"Помилка при комплексному аналізі новин: {e}")

        self.logger.info("Комплексний аналіз новин завершено")
        return result

    def get_news_summary(self, news_item: Union[Dict[str, Any], NewsItem]) -> Dict[str, Any]:
        """Генерація підсумованої інформації про новину."""
        try:
            if isinstance(news_item, NewsItem):
                # Витягуємо інформацію з об'єкта NewsItem
                title = news_item.title or "Без заголовка"
                published_at = news_item.published_at or "Невідома дата"
                source = news_item.source or "Невідоме джерело"
                sentiment = news_item.sentiment_label or "neutral"
                sentiment_score = news_item.sentiment_score or 0.0
                importance = news_item.importance_score or 0.5

                # Отримуємо основні згадані криптовалюти
                mentioned_cryptos = []
                if news_item.mentioned_cryptos:
                    mentioned_cryptos = sorted(
                        news_item.mentioned_cryptos.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]

                # Отримуємо основні теми
                topics = []
                if news_item.topics and 'lda' in news_item.topics:
                    lda_topics = news_item.topics['lda'].get('top_topics', [])
                    for topic in lda_topics:
                        if 'keywords' in topic:
                            topics.append(', '.join(topic['keywords'][:5]))

            else:
                # Витягуємо інформацію зі словника
                title = news_item.get('title', 'Без заголовка')
                published_at = news_item.get('published_at', 'Невідома дата')
                source = news_item.get('source', 'Невідоме джерело')

                # Налаштування змінної sentiment_data (різні варіанти структури)
                if 'sentiment' in news_item:
                    sentiment_data = news_item['sentiment']
                    sentiment = sentiment_data.get('label', 'neutral')
                    sentiment_score = sentiment_data.get('score', 0.0)
                else:
                    sentiment = 'neutral'
                    sentiment_score = 0.0

                # Налаштування змінної importance
                if 'importance' in news_item:
                    importance = news_item['importance'].get('score', 0.5)
                else:
                    importance = 0.5

                # Отримуємо основні згадані криптовалюти
                mentioned_cryptos = []
                if 'mentioned_coins' in news_item and 'coins' in news_item['mentioned_coins']:
                    coins_dict = news_item['mentioned_coins']['coins']
                    mentioned_cryptos = sorted(
                        coins_dict.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]

                # Отримуємо основні теми
                topics = []
                if 'topics' in news_item and 'lda' in news_item['topics']:
                    lda_topics = news_item['topics']['lda'].get('top_topics', [])
                    for topic in lda_topics:
                        if 'keywords' in topic:
                            topics.append(', '.join(topic['keywords'][:5]))

            # Форматуємо дату для кращого відображення
            try:
                if isinstance(published_at, str):
                    if published_at != "Невідома дата":
                        dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                        formatted_date = dt.strftime('%Y-%m-%d %H:%M')
                    else:
                        formatted_date = published_at
                else:
                    formatted_date = published_at.strftime('%Y-%m-%d %H:%M') if published_at else "Невідома дата"
            except Exception:
                formatted_date = str(published_at)

            # Створюємо словник з узагальненою інформацією
            summary = {
                'title': title,
                'published_at': formatted_date,
                'source': source,
                'sentiment': {
                    'label': sentiment,
                    'score': round(sentiment_score, 2)
                },
                'importance_score': round(importance, 2),
                'mentioned_cryptos': [{'name': name, 'count': count} for name, count in mentioned_cryptos],
                'topics': topics
            }

            return summary

        except Exception as e:
            self.logger.error(f"Помилка при створенні підсумку новини: {e}")
            return {
                'title': "Помилка обробки",
                'error': str(e)
            }

    def analyze_news_cluster(self, news_data: List[Union[Dict[str, Any], NewsItem]],
                             time_window_hours: int = 24,
                             min_similarity: float = 0.7) -> Dict[str, Any]:
        """Аналіз кластерів новин за схожістю та часовими періодами."""
        self.logger.info(f"Початок аналізу кластерів новин для {len(news_data)} новин")

        # Сортуємо новини за датою публікації
        sorted_news = sorted(
            news_data,
            key=lambda x: x.published_at if isinstance(x, NewsItem) else x.get('published_at', ''),
            reverse=True
        )

        # Групування новин за часовими вікнами
        time_windows = []
        current_window = []
        reference_date = None

        for news in sorted_news:
            # Отримуємо дату публікації
            if isinstance(news, NewsItem):
                published_at = news.published_at
            else:
                published_at = news.get('published_at')

            # Пропускаємо новини без дати
            if not published_at:
                continue

            # Перетворюємо рядок у datetime, якщо потрібно
            if isinstance(published_at, str):
                try:
                    published_at = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                except ValueError:
                    continue

            # Ініціалізуємо опорну дату, якщо потрібно
            if reference_date is None:
                reference_date = published_at
                current_window.append(news)
                continue

            # Перевіряємо, чи новина входить у поточне часове вікно
            if (reference_date - published_at).total_seconds() <= time_window_hours * 3600:
                current_window.append(news)
            else:
                # Закінчуємо поточне вікно і починаємо нове
                if current_window:
                    time_windows.append(current_window)
                current_window = [news]
                reference_date = published_at

        # Додаємо останнє вікно, якщо воно не порожнє
        if current_window:
            time_windows.append(current_window)

        # Аналіз кожного часового вікна для пошуку кластерів
        clusters_by_window = []

        for window_idx, window in enumerate(time_windows):
            window_start = None
            window_end = None

            # Визначаємо початок і кінець часового вікна
            for news in window:
                if isinstance(news, NewsItem):
                    published_at = news.published_at
                else:
                    published_at = news.get('published_at')

                if not published_at:
                    continue

                if isinstance(published_at, str):
                    published_at = datetime.fromisoformat(published_at.replace('Z', '+00:00'))

                if window_start is None or published_at > window_start:
                    window_start = published_at
                if window_end is None or published_at < window_end:
                    window_end = published_at

            # Якщо у вікні менше двох новин, пропускаємо аналіз кластерів
            if len(window) < 2:
                clusters_by_window.append({
                    'window_idx': window_idx,
                    'window_start': window_start.isoformat() if window_start else None,
                    'window_end': window_end.isoformat() if window_end else None,
                    'news_count': len(window),
                    'clusters': []
                })
                continue

            # Отримуємо тексти новин для аналізу
            texts = []
            for news in window:
                text = self._get_text_to_analyze(news)
                texts.append(text)

            # Отримуємо ембеддінги для текстів
            embeddings = self._get_bert_embeddings_batch(texts, max_length=512)

            # Кластеризація за допомогою DBSCAN або ієрархічної кластеризації
            # (для простоти використовуємо попарну косинусну подібність)
            clusters = []
            processed = set()

            for i in range(len(window)):
                if i in processed:
                    continue

                # Створюємо новий кластер
                current_cluster = [i]
                processed.add(i)

                # Шукаємо схожі новини
                for j in range(i + 1, len(window)):
                    if j in processed:
                        continue

                    # Обчислюємо косинусну подібність
                    similarity = np.dot(embeddings[i], embeddings[j]) / (
                            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))

                    if similarity >= min_similarity:
                        current_cluster.append(j)
                        processed.add(j)

                # Додаємо кластер, якщо він містить більше однієї новини
                if len(current_cluster) > 1:
                    # Знаходимо найважливішу новину в кластері
                    most_important_idx = current_cluster[0]
                    max_importance = 0

                    for idx in current_cluster:
                        news = window[idx]
                        if isinstance(news, NewsItem):
                            importance = news.importance_score or 0
                        else:
                            importance = news.get('importance', {}).get('score', 0)

                        if importance > max_importance:
                            max_importance = importance
                            most_important_idx = idx

                    # Додаємо кластер до списку
                    cluster_info = {
                        'cluster_size': len(current_cluster),
                        'representative_news': self.get_news_summary(window[most_important_idx]),
                        'news_indices': current_cluster
                    }
                    clusters.append(cluster_info)
                elif len(current_cluster) == 1:
                    # Додаємо одиночну новину як окремий "кластер"
                    cluster_info = {
                        'cluster_size': 1,
                        'representative_news': self.get_news_summary(window[current_cluster[0]]),
                        'news_indices': current_cluster
                    }
                    clusters.append(cluster_info)

            # Додаємо інформацію про вікно та кластери
            window_info = {
                'window_idx': window_idx,
                'window_start': window_start.isoformat() if window_start else None,
                'window_end': window_end.isoformat() if window_end else None,
                'news_count': len(window),
                'clusters': sorted(clusters, key=lambda x: x['cluster_size'], reverse=True)
            }
            clusters_by_window.append(window_info)

        # Формуємо підсумок
        summary = {
            'total_news': len(news_data),
            'time_windows': len(time_windows),
            'clusters_by_window': clusters_by_window
        }

        self.logger.info(f"Аналіз кластерів завершено. Знайдено {len(time_windows)} часових вікон.")
        return summary

    def get_trending_topics(self, news_data: List[Union[Dict[str, Any], NewsItem]], top_n: int = 5) -> Dict[str, Any]:
        """Аналіз трендових тем та криптовалют з усіх новин."""
        self.logger.info(f"Початок аналізу трендових тем для {len(news_data)} новин")

        # Словники для підрахунку
        crypto_counts = {}
        topic_counts = {}
        sentiment_by_crypto = {}
        importance_by_crypto = {}

        # Аналіз кожної новини
        for news in news_data:
            # Аналіз згаданих криптовалют
            if isinstance(news, NewsItem):
                crypto_dict = news.mentioned_cryptos or {}
                sentiment = news.sentiment_score or 0
                importance = news.importance_score or 0.5
            else:
                crypto_dict = news.get('mentioned_coins', {}).get('coins', {})
                sentiment = news.get('sentiment', {}).get('score', 0)
                importance = news.get('importance', {}).get('score', 0.5)

            # Підрахунок згадувань криптовалют
            for crypto, count in crypto_dict.items():
                if crypto not in crypto_counts:
                    crypto_counts[crypto] = 0
                    sentiment_by_crypto[crypto] = []
                    importance_by_crypto[crypto] = []

                crypto_counts[crypto] += count
                sentiment_by_crypto[crypto].append(sentiment)
                importance_by_crypto[crypto].append(importance)

            # Аналіз тем
            if isinstance(news, NewsItem):
                topics_data = news.topics or {}
            else:
                topics_data = news.get('topics', {})

            # Підрахунок згадувань тем з LDA
            if 'lda' in topics_data:
                lda_topics = topics_data['lda'].get('top_topics', [])
                for topic in lda_topics:
                    if 'keywords' in topic:
                        topic_key = tuple(topic['keywords'][:3])  # Використовуємо перші 3 ключові слова як ключ теми
                        score = topic.get('score', 1.0)

                        if topic_key not in topic_counts:
                            topic_counts[topic_key] = 0

                        topic_counts[topic_key] += score

        # Отримуємо топ криптовалют
        top_cryptos = sorted(
            crypto_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        # Розрахунок середнього настрою та важливості для кожної криптовалюти
        crypto_trends = []
        for crypto, count in top_cryptos:
            avg_sentiment = sum(sentiment_by_crypto[crypto]) / len(sentiment_by_crypto[crypto]) if sentiment_by_crypto[
                crypto] else 0
            avg_importance = sum(importance_by_crypto[crypto]) / len(importance_by_crypto[crypto]) if \
            importance_by_crypto[crypto] else 0.5

            crypto_trends.append({
                'name': crypto,
                'mentions': count,
                'avg_sentiment': round(avg_sentiment, 2),
                'avg_importance': round(avg_importance, 2),
                'sentiment_direction': 'positive' if avg_sentiment > 0.1 else (
                    'negative' if avg_sentiment < -0.1 else 'neutral')
            })

        # Отримуємо топ тем
        top_topics = sorted(
            topic_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        # Форматуємо теми для виводу
        topic_trends = []
        for topic_key, score in top_topics:
            topic_trends.append({
                'keywords': list(topic_key),
                'score': round(score, 2)
            })

        # Формуємо підсумок
        trends = {
            'trending_cryptos': crypto_trends,
            'trending_topics': topic_trends,
            'total_news_analyzed': len(news_data),
            'analysis_timestamp': datetime.now().isoformat()
        }

        self.logger.info("Аналіз трендових тем завершено")
        return trends

    def identify_market_signals(self, news_data: List[Union[Dict[str, Any], NewsItem]]) -> Dict[str, Any]:
        """Ідентифікація потенційних ринкових сигналів на основі аналізу новин."""
        self.logger.info(f"Початок ідентифікації ринкових сигналів для {len(news_data)} новин")

        # Словники для відстеження сигналів по криптовалютах
        crypto_signals = {}
        important_news = []

        # Порогові значення для важливості та настрою
        IMPORTANCE_THRESHOLD = 0.7
        SENTIMENT_THRESHOLD_POS = 0.5
        SENTIMENT_THRESHOLD_NEG = -0.5

        # Аналіз кожної новини
        for news in news_data:
            # Отримуємо дані для аналізу
            if isinstance(news, NewsItem):
                sentiment = news.sentiment_score or 0
                importance = news.importance_score or 0.5
                mentioned_cryptos = news.mentioned_cryptos or {}
                title = news.title or ""
                published_at = news.published_at
            else:
                sentiment = news.get('sentiment', {}).get('score', 0)
                importance = news.get('importance', {}).get('score', 0.5)
                mentioned_cryptos = news.get('mentioned_coins', {}).get('coins', {})
                title = news.get('title', "")
                published_at = news.get('published_at')

            # Аналізуємо тільки важливі новини
            if importance >= IMPORTANCE_THRESHOLD:
                # Додаємо новину до списку важливих
                news_summary = {
                    'title': title,
                    'published_at': published_at,
                    'importance': round(importance, 2),
                    'sentiment': round(sentiment, 2),
                    'sentiment_direction': 'positive' if sentiment > SENTIMENT_THRESHOLD_POS else
                    ('negative' if sentiment < SENTIMENT_THRESHOLD_NEG else 'neutral')
                }
                important_news.append(news_summary)

                # Аналізуємо згадані криптовалюти
                for crypto, count in mentioned_cryptos.items():
                    if crypto not in crypto_signals:
                        crypto_signals[crypto] = {
                            'positive_count': 0,
                            'negative_count': 0,
                            'neutral_count': 0,
                            'total_importance': 0,
                            'news_count': 0,
                            'recent_news': []
                        }

                    # Оновлюємо статистику для криптовалюти
                    crypto_signals[crypto]['news_count'] += 1
                    crypto_signals[crypto]['total_importance'] += importance

                    # Підраховуємо позитивні/негативні згадування
                    if sentiment > SENTIMENT_THRESHOLD_POS:
                        crypto_signals[crypto]['positive_count'] += 1
                    elif sentiment < SENTIMENT_THRESHOLD_NEG:
                        crypto_signals[crypto]['negative_count'] += 1
                    else:
                        crypto_signals[crypto]['neutral_count'] += 1

                    # Додаємо новину до списку останніх новин для криптовалюти
                    if len(crypto_signals[crypto]['recent_news']) < 3:  # Зберігаємо до 3 останніх новин
                        crypto_signals[crypto]['recent_news'].append(news_summary)

        # Аналіз сигналів для кожної криптовалюти
        market_signals = []
        for crypto, data in crypto_signals.items():
            if data['news_count'] < 2:  # Пропускаємо криптовалюти з малою кількістю новин
                continue

            # Розрахунок відсотка позитивних та негативних новин
            total_news = data['news_count']
            positive_percent = (data['positive_count'] / total_news) * 100 if total_news > 0 else 0
            negative_percent = (data['negative_count'] / total_news) * 100 if total_news > 0 else 0
            avg_importance = data['total_importance'] / total_news if total_news > 0 else 0

            # Визначення сигналу
            signal = 'neutral'
            signal_strength = 0

            if positive_percent >= 60 and data['positive_count'] >= 2:
                signal = 'bullish'
                signal_strength = min(1.0, (positive_percent / 100) * avg_importance)
            elif negative_percent >= 60 and data['negative_count'] >= 2:
                signal = 'bearish'
                signal_strength = min(1.0, (negative_percent / 100) * avg_importance)

            # Додаємо сигнал до списку
            if signal != 'neutral' or data['news_count'] >= 3:
                market_signals.append({
                    'crypto': crypto,
                    'signal': signal,
                    'signal_strength': round(signal_strength, 2),
                    'news_count': total_news,
                    'positive_percent': round(positive_percent, 1),
                    'negative_percent': round(negative_percent, 1),
                    'avg_importance': round(avg_importance, 2),
                    'recent_news': data['recent_news']
                })

        # Сортуємо сигнали за силою
        market_signals.sort(key=lambda x: x['signal_strength'], reverse=True)

        # Формуємо підсумок
        signals_summary = {
            'market_signals': market_signals[:10],  # Топ-10 сигналів
            'important_news_count': len(important_news),
            'important_news': important_news[:5],  # Топ-5 важливих новин
            'total_news_analyzed': len(news_data),
            'analysis_timestamp': datetime.now().isoformat()
        }

        self.logger.info(f"Ідентифікація ринкових сигналів завершена. Знайдено {len(market_signals)} сигналів.")
        return signals_summary

    def save_analysis_result(self, result: Union[Dict[str, Any], List[Dict[str, Any]], List[NewsItem]],
                             filename: str = None) -> bool:
        """Збереження результатів аналізу в JSON-файл."""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"analysis_result_{timestamp}.json"

        try:
            # Підготовка даних до серіалізації
            if isinstance(result, list):
                # Для списку новин
                serializable_data = []
                for item in result:
                    if isinstance(item, NewsItem):
                        # Перетворення NewsItem в словник
                        item_dict = {
                            'title': item.title,
                            'url': item.url,
                            'source': item.source,
                            'published_at': item.published_at.isoformat() if isinstance(item.published_at,
                                                                                        datetime) else item.published_at,
                            'author': item.author,
                            'content': item.content[:500] + '...' if item.content and len(
                                item.content) > 500 else item.content,
                            'summary': item.summary,
                            'categories': item.categories,
                            'tags': item.tags,
                            'sentiment_score': item.sentiment_score,
                            'sentiment_label': item.sentiment_label,
                            'importance_score': item.importance_score,
                            'mentioned_cryptos': item.mentioned_cryptos,
                            'topics': item.topics
                        }
                        serializable_data.append(item_dict)
                    else:
                        # Для словників - копіюємо та обмежуємо довжину контенту
                        item_copy = item.copy()
                        if 'content' in item_copy and item_copy['content'] and len(item_copy['content']) > 500:
                            item_copy['content'] = item_copy['content'][:500] + '...'
                        serializable_data.append(item_copy)
            else:
                # Для окремого словника
                serializable_data = result

            # Створення директорії, якщо потрібно
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)

            # Запис у файл
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, ensure_ascii=False, indent=2, default=str)

            self.logger.info(f"Результати аналізу збережено у файл: {filename}")
            return True

        except Exception as e:
            self.logger.error(f"Помилка при збереженні результатів аналізу: {e}")
            return False

    def analyze_sentiment_trends(self, news_data: List[Union[Dict[str, Any], NewsItem]],
                                 time_window_hours: int = 24) -> Dict[str, Any]:
        """Аналіз трендів настроїв за часовими періодами для криптовалют."""
        self.logger.info(f"Початок аналізу трендів настроїв для {len(news_data)} новин")

        # Словник для відстеження настроїв за часовими вікнами для кожної криптовалюти
        crypto_sentiment_windows = {}
        all_cryptos = set()

        # Сортуємо новини за датою публікації
        sorted_news = []
        for news in news_data:
            # Отримуємо дату публікації
            if isinstance(news, NewsItem):
                published_at = news.published_at
            else:
                published_at = news.get('published_at')

            # Пропускаємо новини без дати
            if not published_at:
                continue

            # Перетворюємо рядок у datetime, якщо потрібно
            if isinstance(published_at, str):
                try:
                    published_at = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                    sorted_news.append((news, published_at))
                except ValueError:
                    continue
            else:
                sorted_news.append((news, published_at))

        # Сортуємо за датою (від найновіших до найстаріших)
        sorted_news.sort(key=lambda x: x[1], reverse=True)

        # Якщо немає новин з датами, повертаємо порожній результат
        if not sorted_news:
            return {
                'error': 'Немає новин з дійсними датами публікації',
                'analysis_timestamp': datetime.now().isoformat()
            }

        # Визначаємо часові вікна
        newest_date = sorted_news[0][1]
        oldest_date = sorted_news[-1][1]
        total_hours = (newest_date - oldest_date).total_seconds() / 3600

        # Визначаємо кількість вікон (мінімум 1)
        num_windows = max(1, int(total_hours / time_window_hours))

        # Створюємо часові вікна
        time_windows = []
        window_size = timedelta(hours=time_window_hours)

        for i in range(num_windows):
            window_end = newest_date - i * window_size
            window_start = window_end - window_size
            time_windows.append((window_start, window_end))

        # Ініціалізуємо структури даних для кожного вікна
        for window_idx, (window_start, window_end) in enumerate(time_windows):
            window_key = f"window_{window_idx}"
            crypto_sentiment_windows[window_key] = {
                'window_start': window_start.isoformat(),
                'window_end': window_end.isoformat(),
                'cryptos': {}
            }

        # Аналіз кожної новини
        for news, published_at in sorted_news:
            # Визначаємо, до якого вікна належить новина
            for window_idx, (window_start, window_end) in enumerate(time_windows):
                if window_start <= published_at <= window_end:
                    window_key = f"window_{window_idx}"
                    break
            else:
                # Новина не потрапляє в жодне вікно (старіша за останнє вікно)
                continue

            # Отримуємо дані для аналізу
            if isinstance(news, NewsItem):
                sentiment = news.sentiment_score or 0
                mentioned_cryptos = news.mentioned_cryptos or {}
            else:
                sentiment = news.get('sentiment', {}).get('score', 0)
                mentioned_cryptos = news.get('mentioned_coins', {}).get('coins', {})

            # Оновлюємо статистику для кожної згаданої криптовалюти
            for crypto, count in mentioned_cryptos.items():
                all_cryptos.add(crypto)

                # Ініціалізуємо дані для криптовалюти, якщо потрібно
                if crypto not in crypto_sentiment_windows[window_key]['cryptos']:
                    crypto_sentiment_windows[window_key]['cryptos'][crypto] = {
                        'total_sentiment': 0,
                        'news_count': 0,
                        'positive_count': 0,
                        'negative_count': 0,
                        'neutral_count': 0
                    }

                # Оновлюємо статистику
                crypto_data = crypto_sentiment_windows[window_key]['cryptos'][crypto]
                crypto_data['total_sentiment'] += sentiment * count
                crypto_data['news_count'] += count

                # Класифікуємо настрій новини
                if sentiment > 0.1:
                    crypto_data['positive_count'] += count
                elif sentiment < -0.1:
                    crypto_data['negative_count'] += count
                else:
                    crypto_data['neutral_count'] += count

        # Обчислюємо середні показники для кожної криптовалюти в кожному вікні
        for window_key, window_data in crypto_sentiment_windows.items():
            for crypto, stats in window_data['cryptos'].items():
                if stats['news_count'] > 0:
                    stats['average_sentiment'] = stats['total_sentiment'] / stats['news_count']
                    stats['sentiment_ratio'] = (stats['positive_count'] - stats['negative_count']) / max(1, stats[
                        'news_count'])
                else:
                    stats['average_sentiment'] = 0
                    stats['sentiment_ratio'] = 0

        # Додаємо загальну статистику
        result = {
            'analysis_timestamp': datetime.now().isoformat(),
            'time_window_hours': time_window_hours,
            'total_news_analyzed': len(sorted_news),
            'total_cryptos_found': len(all_cryptos),
            'windows': crypto_sentiment_windows
        }

        # Додаємо пріоритетні криптовалюти для кожного вікна (топ-5 за кількістю згадувань)
        for window_key, window_data in crypto_sentiment_windows.items():
            cryptos_in_window = [(crypto, data['news_count']) for crypto, data in window_data['cryptos'].items()]
            top_cryptos = sorted(cryptos_in_window, key=lambda x: x[1], reverse=True)[:5]
            window_data['top_cryptos'] = [{'symbol': crypto, 'mentions': count} for crypto, count in top_cryptos]

        # Додаємо аналіз трендів (порівняння між вікнами)
        if len(time_windows) > 1:
            result['trend_analysis'] = self._calculate_sentiment_trends(crypto_sentiment_windows)

        self.logger.info(
            f"Аналіз трендів настроїв завершено. Знайдено {len(all_cryptos)} криптовалют в {len(time_windows)} часових вікнах.")
        return result

    def _calculate_sentiment_trends(self, windows_data: Dict[str, Any]) -> Dict[str, Any]:
        """Розрахунок трендів настроїв між часовими вікнами."""
        trend_analysis = {
            'rising_sentiment': [],
            'falling_sentiment': [],
            'most_volatile': []
        }

        # Сортуємо вікна за часом (від найновіших до найстаріших)
        sorted_windows = sorted(
            [(k, v) for k, v in windows_data.items()],
            key=lambda x: x[1]['window_start'],
            reverse=True
        )

        # Для кожної криптовалюти відстежуємо зміни настрою
        crypto_trends = {}

        for i in range(len(sorted_windows) - 1):
            current_window_key, current_window = sorted_windows[i]
            next_window_key, next_window = sorted_windows[i + 1]

            # Перевіряємо всі криптовалюти, що є в обох вікнах
            for crypto in set(current_window['cryptos'].keys()) & set(next_window['cryptos'].keys()):
                current_sentiment = current_window['cryptos'][crypto].get('average_sentiment', 0)
                next_sentiment = next_window['cryptos'][crypto].get('average_sentiment', 0)

                # Мінімальна кількість згадувань для врахування тренду
                min_mentions = 3
                if (current_window['cryptos'][crypto].get('news_count', 0) < min_mentions or
                        next_window['cryptos'][crypto].get('news_count', 0) < min_mentions):
                    continue

                # Зміна настрою
                sentiment_change = current_sentiment - next_sentiment

                if crypto not in crypto_trends:
                    crypto_trends[crypto] = {
                        'symbol': crypto,
                        'sentiment_changes': [],
                        'volatility': 0
                    }

                crypto_trends[crypto]['sentiment_changes'].append(sentiment_change)
                crypto_trends[crypto]['volatility'] += abs(sentiment_change)

        # Визначаємо криптовалюти з найбільшим зростанням/падінням настроїв
        for crypto, data in crypto_trends.items():
            if not data['sentiment_changes']:
                continue

            # Середня зміна настрою
            avg_change = sum(data['sentiment_changes']) / len(data['sentiment_changes'])
            data['avg_sentiment_change'] = avg_change

            # Середня волатильність
            data['avg_volatility'] = data['volatility'] / len(data['sentiment_changes'])

            # Класифікація за трендом
            if avg_change > 0.05:
                trend_analysis['rising_sentiment'].append({
                    'symbol': crypto,
                    'avg_change': avg_change,
                    'volatility': data['avg_volatility']
                })
            elif avg_change < -0.05:
                trend_analysis['falling_sentiment'].append({
                    'symbol': crypto,
                    'avg_change': avg_change,
                    'volatility': data['avg_volatility']
                })

        # Сортуємо за величиною зміни
        trend_analysis['rising_sentiment'] = sorted(
            trend_analysis['rising_sentiment'],
            key=lambda x: x['avg_change'],
            reverse=True
        )[:10]  # Топ-10

        trend_analysis['falling_sentiment'] = sorted(
            trend_analysis['falling_sentiment'],
            key=lambda x: x['avg_change']
        )[:10]  # Топ-10

        # Криптовалюти з найбільшою волатильністю настроїв
        most_volatile = sorted(
            [data for _, data in crypto_trends.items() if 'avg_volatility' in data],
            key=lambda x: x['avg_volatility'],
            reverse=True
        )[:10]  # Топ-10

        trend_analysis['most_volatile'] = [
            {'symbol': item['symbol'], 'volatility': item['avg_volatility']}
            for item in most_volatile
        ]

        return trend_analysis