import snscrape.modules.twitter as sntwitter
from transformers import pipeline
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional, Union, Tuple
import psycopg2
from psycopg2 import pool
from data.db import DatabaseManager
from utils.config import db_connection


class TwitterScraper:

    def __init__(self, sentiment_model: str = "finiteautomata/bertweet-base-sentiment-analysis",
                 cache_dir: Optional[str] = None,
                 log_level=logging.INFO,
                 db_config: Optional[Dict] = None,
                 cache_expiry: int = 86400):

        self.sentiment_model_name = sentiment_model
        self.cache_dir = cache_dir
        self.log_level = log_level
        self.cache_expiry = cache_expiry

        # Налаштування логування
        logging.basicConfig(level=self.log_level)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Ініціалізація TwitterScraper...")

        # Підключення до бази даних
        self.db_config = db_config if db_config else db_connection
        self.db_manager = DatabaseManager()
        self.supported_symbols = self.db_manager.supported_symbols

        # Ініціалізація моделі sentiment analysis
        try:
            self.logger.info(f"Завантаження моделі аналізу настроїв: {sentiment_model}")
            self.sentiment_analyzer = pipeline("sentiment-analysis",
                                               model=sentiment_model,
                                               cache_dir=cache_dir)
            self.logger.info("Модель успішно завантажена")
        except Exception as e:
            self.logger.error(f"Помилка завантаження моделі: {str(e)}")
            self.sentiment_analyzer = None

        # Встановлення прапорця готовності
        self.ready = bool(self.db_manager and self.sentiment_analyzer)
        self.logger.info(f"TwitterScraper готовий до роботи: {self.ready}")

    def search_tweets(self, query: str, days_back: int = 7,
                      limit: Optional[int] = None, lang: str = "en") -> List[Dict]:
        """
        Пошук твітів за заданим запитом.

        Args:
            query: Пошуковий запит (наприклад, "#bitcoin")
            days_back: Кількість днів для пошуку назад
            limit: Максимальна кількість твітів для збору
            lang: Мова твітів

        Returns:
            Список зібраних твітів у форматі словника
        """
        if not self.ready:
            self.logger.error("TwitterScraper не ініціалізовано належним чином")
            return []

        # Перевірка наявності твітів у кеші
        min_date = datetime.now() - timedelta(days=days_back)
        cached_tweets = self._get_cached_tweets(query, min_date)
        if cached_tweets:
            self.logger.info(f"Знайдено {len(cached_tweets)} твітів у кеші для запиту '{query}'")
            return cached_tweets

        # Формування пошукового запиту
        search_query = f"{query} lang:{lang}" if lang else query
        since_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        search_query += f" since:{since_date}"

        self.logger.info(f"Виконання пошукового запиту: '{search_query}'")

        # Збір твітів
        collected_tweets = []
        try:
            tweet_count = 0
            for tweet in sntwitter.TwitterSearchScraper(search_query).get_items():
                # Конвертація твіту в словник
                tweet_dict = {
                    "id": tweet.id,
                    "date": tweet.date,
                    "content": tweet.rawContent,
                    "username": tweet.user.username,
                    "displayname": tweet.user.displayname,
                    "followers": tweet.user.followersCount,
                    "retweets": tweet.retweetCount,
                    "likes": tweet.likeCount,
                    "query": query,
                    "lang": tweet.lang,
                    "collected_at": datetime.now()
                }

                collected_tweets.append(tweet_dict)
                tweet_count += 1

                # Перевірка ліміту
                if limit and tweet_count >= limit:
                    break

            self.logger.info(f"Зібрано {len(collected_tweets)} твітів для запиту '{query}'")

            # Кешування результатів
            if collected_tweets:
                self._cache_tweets(query, collected_tweets)

            return collected_tweets

        except Exception as e:
            self.logger.error(f"Помилка при пошуку твітів: {str(e)}")
            return []

    def _get_cached_tweets(self, query: str, min_date: datetime) -> Optional[List[Dict]]:
        """
        Отримання кешованих твітів з бази даних.

        Args:
            query: Пошуковий запит
            min_date: Мінімальна дата для пошуку

        Returns:
            Список кешованих твітів або None, якщо кеш застарів
        """
        if not self.db_manager:
            self.logger.warning("DatabaseManager не ініціалізовано, отримання кешу неможливе")
            return None

        try:
            self.logger.info(f"Пошук кешованих твітів для запиту '{query}' з {min_date}")

            # Перевірка терміну дії кешу
            max_cache_age = datetime.now() - timedelta(seconds=self.cache_expiry)
            if min_date < max_cache_age:
                self.logger.info("Запитувані дані виходять за межі терміну дії кешу")
                return None

            # Налаштування фільтрів для пошуку в базі даних
            filters = {
                'start_date': min_date,
                'content': f"%{query}%",  # Використовуємо LIKE для пошуку в тексті
                'query': query  # Точна відповідність запиту
            }

            # Отримання твітів з бази даних
            tweets_df = self.db_manager.get_tweets(filters=filters)

            if tweets_df.empty:
                self.logger.info("Кешованих твітів не знайдено")
                return None

            self.logger.info(f"Знайдено {len(tweets_df)} кешованих твітів")
            return tweets_df.to_dict('records')

        except Exception as e:
            self.logger.error(f"Помилка при отриманні кешованих твітів: {str(e)}")
            return None

    def analyze_sentiment(self, tweets: List[Dict]) -> List[Dict]:
        """
        Аналіз настроїв у зібраних твітах.

        Args:
            tweets: Список твітів для аналізу

        Returns:
            Список твітів із доданим полем sentiment та sentiment_score
        """
        if not tweets:
            self.logger.warning("Порожній список твітів для аналізу настроїв")
            return []

        if not self.sentiment_analyzer:
            self.logger.error("Аналізатор настроїв не ініціалізовано")
            return tweets

        analyzed_tweets = []
        try:
            self.logger.info(f"Аналіз настроїв для {len(tweets)} твітів")

            # Групування твітів для пакетного аналізу (оптимізація)
            batch_size = 32  # Оптимальний розмір для більшості моделей
            for i in range(0, len(tweets), batch_size):
                batch = tweets[i:i + batch_size]
                texts = [tweet['content'] for tweet in batch]

                # Виконання аналізу настроїв
                sentiment_results = self.sentiment_analyzer(texts, truncation=True)

                # Обробка результатів для кожного твіту в пакеті
                for j, result in enumerate(sentiment_results):
                    tweet = batch[j].copy()

                    # Додавання результатів аналізу настроїв
                    label = result['label'].lower()
                    score = result['score']

                    # Стандартизація міток настроїв
                    if label in ['positive', 'pos']:
                        sentiment = 'positive'
                    elif label in ['negative', 'neg']:
                        sentiment = 'negative'
                    else:
                        sentiment = 'neutral'

                    # Нормалізація оцінки для негативних настроїв
                    sentiment_score = score if sentiment == 'positive' else -score if sentiment == 'negative' else 0.0

                    # Додавання результатів до твіту
                    tweet['sentiment'] = sentiment
                    tweet['sentiment_score'] = sentiment_score
                    tweet['sentiment_confidence'] = score
                    tweet['sentiment_analysis_date'] = datetime.now()

                    analyzed_tweets.append(tweet)

                    # Збереження результатів у базі даних
                    if self.db_manager:
                        sentiment_data = {
                            'tweet_id': tweet['id'],
                            'sentiment': sentiment,
                            'sentiment_score': sentiment_score,
                            'confidence': score,
                            'model_used': self.sentiment_model_name,
                            'analysis_date': tweet['sentiment_analysis_date']
                        }
                        self.db_manager.insert_tweet_sentiment(sentiment_data)

            self.logger.info(f"Аналіз настроїв завершено для {len(analyzed_tweets)} твітів")
            return analyzed_tweets

        except Exception as e:
            self.logger.error(f"Помилка при аналізі настроїв: {str(e)}")
            # Повертаємо оригінальні твіти, якщо аналіз не вдався
            return tweets

    def filter_by_keywords(self, tweets: List[Dict], keywords: List[str],
                           case_sensitive: bool = False) -> List[Dict]:

        if not tweets or not keywords:
            self.logger.warning("Порожній список твітів або ключових слів")
            return tweets

        filtered_tweets = []
        try:
            self.logger.info(f"Фільтрація {len(tweets)} твітів за {len(keywords)} ключовими словами")

            # Підготовка ключових слів
            if not case_sensitive:
                keywords = [keyword.lower() for keyword in keywords]

            # Фільтрація твітів
            for tweet in tweets:
                content = tweet.get('content', '')
                if not case_sensitive:
                    content = content.lower()

                # Перевірка наявності будь-якого ключового слова в контенті
                if any(keyword in content for keyword in keywords):
                    # Додаємо інформацію про знайдені ключові слова
                    matched_keywords = [keyword for keyword in keywords if keyword in content]
                    tweet_copy = tweet.copy()
                    tweet_copy['matched_keywords'] = matched_keywords
                    filtered_tweets.append(tweet_copy)

            self.logger.info(f"Відфільтровано {len(filtered_tweets)} твітів")
            return filtered_tweets

        except Exception as e:
            self.logger.error(f"Помилка при фільтрації твітів: {str(e)}")
            return tweets

    def get_trending_crypto_topics(self, top_n: int = 10) -> List[Dict]:

        if not self.db_manager:
            self.logger.error("DatabaseManager не ініціалізовано")
            return []

        try:
            self.logger.info(f"Пошук топ-{top_n} трендових криптовалютних тем")

            # Список криптовалютних хештегів для пошуку
            crypto_base_tags = [
                "#bitcoin", "#btc", "#ethereum", "#eth", "#crypto",
                "#blockchain", "#defi", "#nft", "#altcoin", "#trading"
            ]

            # Отримання даних за останні 24 години
            since_date = datetime.now() - timedelta(days=1)

            # Пошук твітів із криптовалютними хештегами
            all_hashtags = []
            for crypto_tag in crypto_base_tags:
                filters = {
                    'start_date': since_date,
                    'content': f"%{crypto_tag}%"
                }
                tweets_df = self.db_manager.get_tweets(filters=filters)

                if not tweets_df.empty:
                    # Вилучення всіх хештегів з твітів
                    for content in tweets_df['content']:
                        # Знаходження всіх хештегів у твіті
                        hashtags = [
                            tag.lower() for tag in re.findall(r'#\w+', content)
                            if tag.lower() not in crypto_base_tags  # Виключення базових тегів
                        ]
                        all_hashtags.extend(hashtags)

            # Підрахунок частоти використання хештегів
            hashtag_counts = Counter(all_hashtags)

            # Вибір топ-N найпопулярніших хештегів
            trending_topics = [
                {
                    "hashtag": hashtag,
                    "count": count,
                    "percentage": count / len(all_hashtags) * 100 if all_hashtags else 0
                }
                for hashtag, count in hashtag_counts.most_common(top_n)
            ]

            self.logger.info(f"Знайдено {len(trending_topics)} трендових тем")
            return trending_topics

        except Exception as e:
            self.logger.error(f"Помилка при пошуку трендових тем: {str(e)}")
            return []

    def save_to_database(self, tweets: List[Dict], table_name: str = "tweets") -> bool:

        if not self.db_manager:
            self.logger.error("DatabaseManager не ініціалізовано")
            return False

        try:
            self.logger.info(f"Збереження {len(tweets)} твітів у таблицю {table_name}")

            # Перевірка наявності вказаної таблиці
            if table_name != "tweets" and not self.db_manager.table_exists(table_name):
                self.logger.warning(f"Таблиця {table_name} не існує. Створення нової таблиці...")
                # Тут можна додати логіку створення таблиці, якщо потрібно

            # Використовуємо метод кешування або окремий метод в залежності від таблиці
            if table_name == "tweets":
                return self._cache_tweets("", tweets)
            else:
                # Якщо використовується інша таблиця, можна реалізувати спеціальну логіку
                successful = True
                for tweet in tweets:
                    result = self.db_manager.insert_custom_data(table_name, tweet)
                    if not result:
                        self.logger.warning(
                            f"Не вдалося зберегти твіт {tweet.get('id', 'unknown')} у таблицю {table_name}")
                        successful = False

                self.logger.info(f"Збереження у {table_name} завершено {'успішно' if successful else 'з помилками'}")
                return successful

        except Exception as e:
            self.logger.error(f"Помилка при збереженні твітів у базу даних: {str(e)}")
            return False

    def get_user_influence(self, username: str) -> Dict:
        """
        Аналіз впливовості користувача в крипто-спільноті.

        Args:
            username: Ім'я користувача Twitter

        Returns:
            Словник з метриками впливовості
        """
        pass

    def track_influencers(self, influencers: List[str], days_back: int = 30) -> Dict[str, List[Dict]]:
        """
        Використовує DatabaseManager для:
        1. Отримання інформації про інфлюенсерів (get_crypto_influencers)
        2. Збереження їх активності (insert_influencer_activity)
        """        """
        Відстеження активності відомих крипто-інфлюенсерів.

        Args:
            influencers: Список імен користувачів для відстеження
            days_back: Кількість днів для аналізу

        Returns:
            Словник з ім'ям користувача та його твітами/метриками
        """
        pass

    def track_sentiment_over_time(self, query: str, days: int = 30,
                                  interval: str = "day") -> pd.DataFrame:
        """
        Використовує DatabaseManager.get_sentiment_time_series() для отримання історичних даних
        та DatabaseManager.insert_sentiment_time_series() для збереження нових даних
        """
        """
        Відстеження зміни настроїв щодо криптовалюти за період часу.

        Args:
            query: Пошуковий запит (криптовалюта)
            days: Кількість днів для аналізу
            interval: Інтервал групування (day, hour, week)

        Returns:
            DataFrame з даними про зміну настроїв у часі
        """
        pass

    def detect_sentiment_change(self, coin: str, threshold: float = 0.2) -> Dict:
        """
        Виявлення різких змін настроїв щодо криптовалюти.

        Args:
            coin: Назва криптовалюти
            threshold: Поріг зміни настроїв для сповіщення

        Returns:
            Інформація про зміну настроїв
        """
        pass

    def correlate_with_price(self, tweets: List[Dict],
                             price_data: pd.DataFrame) -> Dict:
        """
        Кореляція настроїв твітів з ціновими даними криптовалюти.

        Args:
            tweets: Список твітів з аналізом настроїв
            price_data: DataFrame з ціновими даними

        Returns:
            Дані про кореляцію та статистичну значущість
        """
        pass

    def handle_api_rate_limits(self, retry_count: int = 3, cooldown_period: int = 300) -> None:
        """
        Обробка обмежень швидкості API Twitter.

        Args:
            retry_count: Кількість спроб повторного запиту
            cooldown_period: Період очікування між спробами в секундах
        """
        pass

    def detect_crypto_events(self, tweets: List[Dict], min_mentions: int = 50) -> List[Dict]:
        """
        Використовує DatabaseManager.insert_crypto_event() для збереження виявлених подій
        """        """
        Виявлення важливих подій на криптовалютному ринку на основі твітів.

        Args:
            tweets: Список твітів для аналізу
            min_mentions: Мінімальна кількість згадувань для визначення події

        Returns:
            Список виявлених подій з метриками важливості
        """
        pass

    def get_error_stats(self) -> Dict:
        """
        Використовує DatabaseManager.get_scraping_errors() для отримання статистики помилок
        """        """
        Отримання статистики помилок при зборі даних.

        Returns:
            Словник зі статистикою помилок
        """
        pass

    def cleanup_database(self, older_than_days: int = 90) -> int:
        """
        Тепер не потрібен, оскільки DatabaseManager має власні методи для очищення даних
        Або можна додати відповідний метод до DatabaseManager
        """
        """
        Очищення старих даних з бази даних.

        Args:
            older_than_days: Видалити дані старші вказаної кількості днів

        Returns:
            Кількість видалених записів
        """
        pass