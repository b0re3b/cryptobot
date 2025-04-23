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

        if not username:
            self.logger.error("Не вказано ім'я користувача")
            return {"error": "Username not provided"}

        try:
            self.logger.info(f"Аналіз впливовості користувача @{username}")

            # Отримання останніх твітів користувача
            user_query = f"from:{username}"
            user_tweets = self.search_tweets(user_query, days_back=90, limit=100)

            if not user_tweets:
                self.logger.warning(f"Твіти користувача @{username} не знайдено")
                return {
                    "username": username,
                    "found": False,
                    "error": "No tweets found for this user"
                }

            # Базова інформація про користувача
            user_info = {
                "username": username,
                "found": True,
                "display_name": user_tweets[0].get("displayname", ""),
                "followers_count": user_tweets[0].get("followers", 0),
                "tweets_analyzed": len(user_tweets)
            }

            # Аналіз взаємодії з твітами
            total_likes = sum(tweet.get("likes", 0) for tweet in user_tweets)
            total_retweets = sum(tweet.get("retweets", 0) for tweet in user_tweets)
            avg_likes = total_likes / max(len(user_tweets), 1)
            avg_retweets = total_retweets / max(len(user_tweets), 1)

            # Аналіз настроїв твітів користувача
            analyzed_tweets = self.analyze_sentiment(user_tweets)
            sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}

            for tweet in analyzed_tweets:
                sentiment = tweet.get("sentiment", "neutral")
                sentiment_counts[sentiment] += 1

            dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)

            # Аналіз цитування та згадування користувача іншими
            mentions_query = f"@{username}"
            mention_tweets = self.search_tweets(mentions_query, days_back=30, limit=200)

            # Визначення впливових акаунтів, які взаємодіють з користувачем
            influential_interactions = [
                {"username": tweet.get("username"), "followers": tweet.get("followers", 0)}
                for tweet in mention_tweets
                if tweet.get("followers", 0) > 10000  # Поріг впливовості
            ]

            # Аналіз хештегів, які використовує користувач
            hashtags = []
            for tweet in user_tweets:
                content = tweet.get("content", "")
                found_hashtags = re.findall(r'#\w+', content)
                hashtags.extend([tag.lower() for tag in found_hashtags])

            top_hashtags = Counter(hashtags).most_common(5)

            # Аналіз тем, про які говорить користувач
            crypto_keywords = [
                "bitcoin", "btc", "ethereum", "eth", "crypto", "blockchain",
                "defi", "nft", "token", "coin", "mining", "wallet", "sol"
            ]

            topics = []
            for tweet in user_tweets:
                content = tweet.get("content", "").lower()
                for keyword in crypto_keywords:
                    if keyword in content:
                        topics.append(keyword)

            top_topics = Counter(topics).most_common(5)

            # Розрахунок індексу впливовості
            engagement_rate = (avg_likes + avg_retweets * 3) / max(user_info["followers_count"], 1) * 100
            mention_influence = len(mention_tweets) / 30  # Середня кількість згадувань на день
            topic_diversity = len(set(topics)) / len(crypto_keywords)

            influence_score = min(100, (
                    (engagement_rate * 0.4) +
                    (mention_influence * 5) +
                    (user_info["followers_count"] / 10000 * 20) +
                    (topic_diversity * 10)
            ))

            # Формування результату
            result = {
                **user_info,
                "influence_score": round(influence_score, 2),
                "engagement": {
                    "avg_likes": round(avg_likes, 2),
                    "avg_retweets": round(avg_retweets, 2),
                    "engagement_rate": round(engagement_rate, 4)
                },
                "sentiment_profile": {
                    "counts": sentiment_counts,
                    "dominant": dominant_sentiment
                },
                "community_interaction": {
                    "mentions_count": len(mention_tweets),
                    "influential_interactions": len(influential_interactions)
                },
                "topics": {
                    "top_hashtags": [{"tag": tag, "count": count} for tag, count in top_hashtags],
                    "top_crypto_topics": [{"topic": topic, "count": count} for topic, count in top_topics]
                }
            }

            self.logger.info(f"Аналіз впливовості користувача @{username} завершено")
            return result

        except Exception as e:
            self.logger.error(f"Помилка при аналізі впливовості користувача @{username}: {str(e)}")
            return {
                "username": username,
                "found": False,
                "error": str(e)
            }

    def track_influencers(self, influencers: List[str], days_back: int = 30) -> Dict[str, List[Dict]]:

        if not influencers:
            self.logger.error("Не вказано імена інфлюенсерів")
            return {}

        try:
            self.logger.info(f"Відстеження активності {len(influencers)} крипто-інфлюенсерів за {days_back} днів")

            # Якщо influencers порожній, спробуємо отримати інфлюенсерів з бази даних
            if not influencers and self.db_manager:
                influencers_data = self.db_manager.get_crypto_influencers()
                if not influencers_data.empty:
                    influencers = influencers_data['username'].tolist()
                    self.logger.info(f"Отримано {len(influencers)} інфлюенсерів з бази даних")

            if not influencers:
                self.logger.warning("Список інфлюенсерів порожній")
                return {}

            results = {}
            for username in influencers:
                self.logger.info(f"Аналіз активності інфлюенсера @{username}")

                # Отримання твітів інфлюенсера
                user_query = f"from:{username}"
                user_tweets = self.search_tweets(user_query, days_back=days_back, limit=100)

                if not user_tweets:
                    self.logger.warning(f"Твіти користувача @{username} не знайдено")
                    results[username] = []
                    continue

                # Аналіз настроїв твітів
                analyzed_tweets = self.analyze_sentiment(user_tweets)

                # Збереження активності інфлюенсера в базу даних
                if self.db_manager:
                    # Базова інформація про інфлюенсера
                    influencer_info = {
                        "username": username,
                        "displayname": user_tweets[0].get("displayname", ""),
                        "followers": user_tweets[0].get("followers", 0),
                        "last_updated": datetime.now()
                    }

                    # Оновлення інформації про інфлюенсера
                    self.db_manager.insert_crypto_influencer(influencer_info)

                    # Збереження активності інфлюенсера
                    for tweet in analyzed_tweets:
                        activity_data = {
                            "influencer_username": username,
                            "tweet_id": tweet.get("id"),
                            "content": tweet.get("content"),
                            "date": tweet.get("date"),
                            "likes": tweet.get("likes", 0),
                            "retweets": tweet.get("retweets", 0),
                            "sentiment": tweet.get("sentiment", "neutral"),
                            "sentiment_score": tweet.get("sentiment_score", 0.0)
                        }
                        self.db_manager.insert_influencer_activity(activity_data)

                # Додавання даних у результат
                results[username] = analyzed_tweets

            self.logger.info(f"Відстеження активності інфлюенсерів завершено")
            return results

        except Exception as e:
            self.logger.error(f"Помилка при відстеженні активності інфлюенсерів: {str(e)}")
            return {}

    def track_sentiment_over_time(self, query: str, days: int = 30,
                                  interval: str = "day") -> pd.DataFrame:

        if not query:
            self.logger.error("Не вказано пошуковий запит")
            return pd.DataFrame()

        try:
            self.logger.info(f"Аналіз зміни настроїв для запиту '{query}' за {days} днів з інтервалом '{interval}'")

            # Валідація інтервалу
            valid_intervals = ["hour", "day", "week"]
            if interval not in valid_intervals:
                self.logger.warning(f"Невідомий інтервал: {interval}. Використовується 'day'")
                interval = "day"

            # Перевірка наявності історичних даних у базі даних
            historical_data = None
            if self.db_manager:
                filters = {
                    'query': query,
                    'interval': interval,
                    'start_date': datetime.now() - timedelta(days=days)
                }
                historical_data = self.db_manager.get_sentiment_time_series(filters)

            # Якщо є достатньо історичних даних, повертаємо їх
            if historical_data is not None and not historical_data.empty:
                rows_count = len(historical_data)
                expected_rows = days if interval == "day" else (days * 24 if interval == "hour" else days // 7 + 1)

                if rows_count >= expected_rows * 0.8:  # Якщо є хоча б 80% очікуваних даних
                    self.logger.info(f"Використання {rows_count} записів історичних даних")
                    return historical_data

            # Якщо недостатньо історичних даних, збираємо нові дані
            # Пошук твітів за вказаний період
            tweets = self.search_tweets(query, days_back=days, limit=1000)

            if not tweets:
                self.logger.warning(f"Твіти для запиту '{query}' не знайдено")
                return pd.DataFrame()

            # Аналіз настроїв твітів
            analyzed_tweets = self.analyze_sentiment(tweets)

            # Перетворення у DataFrame для зручності аналізу
            df = pd.DataFrame(analyzed_tweets)

            # Конвертація дати у datetime формат, якщо потрібно
            if "date" in df.columns:
                if isinstance(df["date"].iloc[0], str):
                    df["date"] = pd.to_datetime(df["date"])
            else:
                self.logger.warning("Колонка 'date' відсутня у даних твітів")
                return pd.DataFrame()

            # Форматування часового інтервалу для групування
            if interval == "hour":
                df["interval"] = df["date"].dt.strftime("%Y-%m-%d %H:00:00")
            elif interval == "day":
                df["interval"] = df["date"].dt.strftime("%Y-%m-%d")
            elif interval == "week":
                df["interval"] = df["date"].dt.to_period("W").dt.start_time

            # Групування та агрегація даних
            grouped = df.groupby("interval").agg({
                "sentiment_score": ["mean", "std", "count"],
                "sentiment": lambda x: x.value_counts().to_dict()
            })

            # Рестуктуризація для зручності використання
            result = pd.DataFrame()
            result["date"] = grouped.index
            result["mean_sentiment"] = grouped[("sentiment_score", "mean")]
            result["std_sentiment"] = grouped[("sentiment_score", "std")]
            result["tweet_count"] = grouped[("sentiment_score", "count")]

            # Додавання розподілу настроїв
            sentiment_distributions = grouped[("sentiment", "<lambda>")].tolist()
            result["positive_count"] = [dist.get("positive", 0) for dist in sentiment_distributions]
            result["neutral_count"] = [dist.get("neutral", 0) for dist in sentiment_distributions]
            result["negative_count"] = [dist.get("negative", 0) for dist in sentiment_distributions]

            # Розрахунок відсоткового співвідношення
            result["positive_percent"] = result["positive_count"] / result["tweet_count"] * 100
            result["neutral_percent"] = result["neutral_count"] / result["tweet_count"] * 100
            result["negative_percent"] = result["negative_count"] / result["tweet_count"] * 100

            # Збереження часових рядів у базу даних
            if self.db_manager:
                for _, row in result.iterrows():
                    time_series_data = {
                        "query": query,
                        "date": row["date"],
                        "interval": interval,
                        "mean_sentiment": row["mean_sentiment"],
                        "std_sentiment": row["std_sentiment"],
                        "tweet_count": row["tweet_count"],
                        "positive_count": row["positive_count"],
                        "neutral_count": row["neutral_count"],
                        "negative_count": row["negative_count"],
                    }
                    self.db_manager.insert_sentiment_time_series(time_series_data)

            self.logger.info(f"Аналіз зміни настроїв завершено, отримано {len(result)} часових точок")
            return result

        except Exception as e:
            self.logger.error(f"Помилка при аналізі зміни настроїв: {str(e)}")
            return pd.DataFrame()

    def detect_sentiment_change(self, coin: str, threshold: float = 0.2) -> Dict:
        """
        Виявлення різких змін настроїв щодо криптовалюти.

        Args:
            coin: Назва криптовалюти
            threshold: Поріг зміни настроїв для сповіщення

        Returns:
            Інформація про зміну настроїв
        """
        if not coin:
            self.logger.error("Не вказано назву криптовалюти")
            return {"error": "Coin name not provided"}

        try:
            self.logger.info(f"Аналіз змін настроїв для {coin} з порогом {threshold}")

            # Отримання даних про настрої за останні 7 днів (тиждень)
            sentiment_data = self.track_sentiment_over_time(
                query=f"#{coin} OR ${coin}",
                days=7,
                interval="day"
            )

            if sentiment_data.empty:
                self.logger.warning(f"Недостатньо даних для аналізу настроїв щодо {coin}")
                return {
                    "coin": coin,
                    "detected_change": False,
                    "reason": "Insufficient data"
                }

            # Підготовка даних для аналізу
            sentiment_data = sentiment_data.sort_values("date")
            daily_means = sentiment_data["mean_sentiment"].tolist()
            dates = sentiment_data["date"].tolist()

            # Пошук різких змін у настроях
            changes = []
            for i in range(1, len(daily_means)):
                change = daily_means[i] - daily_means[i - 1]
                changes.append({
                    "date": dates[i],
                    "previous_date": dates[i - 1],
                    "current_sentiment": daily_means[i],
                    "previous_sentiment": daily_means[i - 1],
                    "change": change,
                    "percent_change": change / (abs(daily_means[i - 1]) + 0.0001) * 100,
                    "is_significant": abs(change) >= threshold
                })

            # Фільтрація значимих змін
            significant_changes = [change for change in changes if change["is_significant"]]

            # Визначення найбільш значної зміни
            if significant_changes:
                most_significant = max(significant_changes, key=lambda x: abs(x["change"]))
                direction = "positive" if most_significant["change"] > 0 else "negative"

                # Пошук ймовірних причин зміни настрою
                start_date = most_significant["previous_date"]
                end_date = most_significant["date"]

                # Отримання твітів за цей період
                tweets_query = f"#{coin} OR ${coin}"
                tweets = self.search_tweets(
                    query=tweets_query,
                    days_back=7,
                    limit=200
                )

                # Фільтрація твітів за датами
                relevant_tweets = [
                    tweet for tweet in tweets
                    if start_date <= tweet["date"] <= end_date
                ]

                # Аналіз найпопулярніших твітів
                popular_tweets = sorted(
                    relevant_tweets,
                    key=lambda x: x.get("retweets", 0) + x.get("likes", 0),
                    reverse=True
                )[:5]

                result = {
                    "coin": coin,
                    "detected_change": True,
                    "change_direction": direction,
                    "change_magnitude": abs(most_significant["change"]),
                    "previous_sentiment": most_significant["previous_sentiment"],
                    "current_sentiment": most_significant["current_sentiment"],
                    "change_date": most_significant["date"],
                    "previous_date": most_significant["previous_date"],
                    "percent_change": most_significant["percent_change"],
                    "potential_causes": [
                        {
                            "tweet_id": tweet.get("id"),
                            "content": tweet.get("content"),
                            "username": tweet.get("username"),
                            "engagement": tweet.get("retweets", 0) + tweet.get("likes", 0)
                        }
                        for tweet in popular_tweets
                    ]
                }

                # Збереження інформації про зміну настроїв у базу даних
                if self.db_manager:
                    event_data = {
                        "coin": coin,
                        "event_type": f"sentiment_change_{direction}",
                        "event_date": most_significant["date"],
                        "description": f"Significant {direction} sentiment change detected",
                        "magnitude": abs(most_significant["change"]),
                        "previous_value": most_significant["previous_sentiment"],
                        "current_value": most_significant["current_sentiment"],
                        "detection_date": datetime.now()
                    }
                    self.db_manager.insert_crypto_event(event_data)

                self.logger.info(
                    f"Виявлено значну зміну настроїв для {coin}: {direction} ({most_significant['percent_change']:.2f}%)")
                return result
            else:
                self.logger.info(f"Не виявлено значних змін настроїв для {coin}")
                return {
                    "coin": coin,
                    "detected_change": False,
                    "reason": "No significant changes detected",
                    "max_change": max(abs(change["change"]) for change in changes) if changes else 0,
                    "threshold": threshold
                }

        except Exception as e:
            self.logger.error(f"Помилка при виявленні змін настроїв для {coin}: {str(e)}")
            return {
                "coin": coin,
                "detected_change": False,
                "error": str(e)
            }

    def correlate_with_price(self, tweets: List[Dict], price_data: pd.DataFrame) -> Dict:

        if not tweets or price_data.empty:
            self.logger.warning("Порожні дані для кореляційного аналізу")
            return {"error": "Empty data for correlation analysis"}

        try:
            self.logger.info(f"Кореляційний аналіз {len(tweets)} твітів з ціновими даними")

            # Перевірка наявності аналізу настроїв у твітах
            if "sentiment_score" not in tweets[0]:
                self.logger.info("Твіти потребують аналізу настроїв")
                tweets = self.analyze_sentiment(tweets)

            # Перетворення твітів у DataFrame
            tweets_df = pd.DataFrame(tweets)

            # Перевірка необхідних колонок у DataFrame з цінами
            required_columns = ["date", "close"]
            if not all(col in price_data.columns for col in required_columns):
                self.logger.error("Відсутні необхідні колонки в даних про ціни")
                return {"error": "Missing required columns in price data"}

            # Конвертація дат у datetime формат
            tweets_df["date"] = pd.to_datetime(tweets_df["date"])
            price_data["date"] = pd.to_datetime(price_data["date"])

            # Агрегація настроїв за днями
            tweets_df["date_day"] = tweets_df["date"].dt.date
            daily_sentiment = tweets_df.groupby("date_day").agg({
                "sentiment_score": "mean",
                "id": "count"
            }).rename(columns={"id": "tweet_count"}).reset_index()

            # Підготовка даних про ціни з відповідною датою
            price_data["date_day"] = price_data["date"].dt.date
            price_daily = price_data.groupby("date_day").agg({
                "close": "last",
                "volume": "sum" if "volume" in price_data.columns else None
            }).reset_index()

            # Об'єднання даних для кореляційного аналізу
            merged_data = pd.merge(daily_sentiment, price_daily, on="date_day", how="inner")

            if len(merged_data) < 3:
                self.logger.warning("Недостатньо точок даних для кореляційного аналізу")
                return {"error": "Insufficient data points for correlation analysis"}

            # Розрахунок кореляцій
            correlation_same_day = merged_data["sentiment_score"].corr(merged_data["close"])

            # Кореляція з наступним днем (настрої як предиктор)
            merged_data["next_day_close"] = merged_data["close"].shift(-1)
            correlation_next_day = merged_data["sentiment_score"].corr(merged_data["next_day_close"])

            # Кореляція з попереднім днем (ціна як предиктор)
            merged_data["prev_day_sentiment"] = merged_data["sentiment_score"].shift(1)
            correlation_prev_day = merged_data["close"].corr(merged_data["prev_day_sentiment"])

            # Розрахунок статистичної значущості для основної кореляції
            from scipy import stats
            r = correlation_same_day
            n = len(merged_data)
            t_stat = r * np.sqrt(n - 2) / np.sqrt(1 - r ** 2)
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))

            # Застосування лагів для пошуку оптимального часового зсуву
            max_lag = min(5, len(merged_data) // 3)  # Максимальний лаг для аналізу
            lag_results = []

            for lag in range(-max_lag, max_lag + 1):
                if lag == 0:
                    continue  # Вже розраховано вище

                # Створення зсунутих даних
                if lag > 0:
                    lagged_sentiment = merged_data["sentiment_score"].shift(lag)
                    lagged_corr = merged_data["close"].corr(lagged_sentiment)
                    direction = "sentiment_follows_price"
                else:
                    lagged_price = merged_data["close"].shift(-lag)
                    lagged_corr = merged_data["sentiment_score"].corr(lagged_price)
                    direction = "price_follows_sentiment"

                lag_results.append({
                    "lag": lag,
                    "correlation": lagged_corr,
                    "direction": direction
                })

            # Знаходження найсильнішої кореляції
            strongest_lag = max(lag_results, key=lambda x: abs(x["correlation"]), default=None)

            # Підготовка результатів
            result = {
                "same_day_correlation": correlation_same_day,
                "next_day_correlation": correlation_next_day,
                "previous_day_correlation": correlation_prev_day,
                "data_points": len(merged_data),
                "statistical_significance": {
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "is_significant": p_value < 0.05
                },
                "strongest_correlation": strongest_lag,
                "lag_analysis": lag_results,
                "interpretation": {
                    "same_day": self._interpret_correlation(correlation_same_day),
                    "predictive_power": self._interpret_correlation(correlation_next_day),
                    "reactive_nature": self._interpret_correlation(correlation_prev_day)
                }
            }

            self.logger.info(f"Кореляційний аналіз завершено. Основна кореляція: {correlation_same_day:.4f}")
            return result

        except Exception as e:
            self.logger.error(f"Помилка при кореляційному аналізі: {str(e)}")
            return {"error": str(e)}

    def _interpret_correlation(self, corr_value: float) -> str:
        abs_corr = abs(corr_value)

        if abs_corr < 0.1:
            strength = "відсутній"
        elif abs_corr < 0.3:
            strength = "слабкий"
        elif abs_corr < 0.5:
            strength = "помірний"
        elif abs_corr < 0.7:
            strength = "сильний"
        else:
            strength = "дуже сильний"

        direction = "позитивний" if corr_value > 0 else "негативний"
        return f"{strength} {direction} зв'язок"

    def handle_api_rate_limits(self, retry_count: int = 3, cooldown_period: int = 300) -> None:

        self.logger.info(
            f"Налаштування параметрів обробки обмежень API: повторні спроби={retry_count}, період очікування={cooldown_period}с")

        # Зберігаємо параметри для використання в інших методах
        self.api_retry_count = retry_count
        self.api_cooldown_period = cooldown_period

        # Додаємо лічильник помилок і часову мітку останньої помилки
        self.api_error_count = 0
        self.last_api_error_time = None

        # Створюємо декоратор для повторних спроб при помилках API
        def retry_on_api_limit(func):
            def wrapper(*args, **kwargs):
                attempts = 0
                while attempts < self.api_retry_count:
                    try:
                        # Перевірка затримки після останньої помилки
                        if self.last_api_error_time:
                            time_since_error = (datetime.now() - self.last_api_error_time).total_seconds()
                            if time_since_error < self.api_cooldown_period:
                                wait_time = self.api_cooldown_period - time_since_error
                                self.logger.info(f"Очікування {wait_time:.1f}с перед наступною спробою")
                                time.sleep(wait_time)

                        # Виклик оригінального методу
                        result = func(*args, **kwargs)

                        # Скидання лічильника помилок при успішній спробі
                        if attempts > 0:
                            self.api_error_count = 0
                            self.last_api_error_time = None
                            self.logger.info("Запит успішний після повторних спроб")

                        return result

                    except Exception as e:
                        error_message = str(e).lower()
                        if "rate limit" in error_message or "too many requests" in error_message:
                            attempts += 1
                            self.api_error_count += 1
                            self.last_api_error_time = datetime.now()

                            # Збільшуємо час очікування з кожною спробою
                            wait_time = self.api_cooldown_period * (2 ** attempts)

                            self.logger.warning(f"Досягнуто обмеження API (спроба {attempts}/{self.api_retry_count}). "
                                                f"Очікування {wait_time}с перед наступною спробою")

                            # Реєстрація помилки в базі даних
                            if self.db_manager:
                                error_data = {
                                    "error_type": "rate_limit",
                                    "error_message": str(e),
                                    "function_name": func.__name__,
                                    "timestamp": datetime.now()
                                }
                                self.db_manager.insert_error_log(error_data)

                            time.sleep(wait_time)
                        else:
                            # Якщо помилка не пов'язана з обмеженням швидкості, передаємо її далі
                            self.logger.error(f"Помилка не пов'язана з обмеженням API: {str(e)}")
                            raise

                # Якщо всі спроби вичерпано
                self.logger.error(f"Вичерпано всі {self.api_retry_count} спроб через обмеження API")
                raise Exception(f"API rate limit exceeded after {self.api_retry_count} attempts")

            return wrapper

        # Застосовуємо декоратор до методів, які взаємодіють з API
        self.search_tweets = retry_on_api_limit(self.search_tweets)

        self.logger.info("Обробка обмежень API налаштована успішно")

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