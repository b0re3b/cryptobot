# """""
# import os
# import re
# import ssl
# import certifi
# import pandas as pd
# import asyncio
# import aiohttp
# import snscrape.modules.twitter as sntwitter
# from transformers import pipeline
# from datetime import datetime, timedelta
# import logging
# from typing import List, Dict, Optional, Tuple, Counter, Any
# from data.db import DatabaseManager
# import concurrent.futures
# import numpy as np
# import requests
#
# os.environ['SSL_CERT_FILE'] = certifi.where()
#
# ssl_context = ssl.create_default_context(cafile=certifi.where())
# ssl._create_default_https_context = lambda: ssl_context
#
# class TwitterScraper:
#
#     def __init__(self, sentiment_model: str = "finiteautomata/bertweet-base-sentiment-analysis",
#                  cache_dir: Optional[str] ="./data_collection/cachetweet",
#                  log_level=logging.INFO,
#                  cache_expiry: int = 86400,
#                  max_connections: int = 20):
#
#         self.sentiment_model_name = sentiment_model
#         self.cache_dir = cache_dir
#         self.log_level = log_level
#         self.cache_expiry = cache_expiry
#         self.max_connections = max_connections
#         logging.basicConfig(level=self.log_level)
#         self.logger = logging.getLogger(__name__)
#         self.logger.info("Ініціалізація TwitterScraper...")
#         self.session = None
#
#         # Підключення до бази даних
#         self.db_manager = DatabaseManager()
#         self.supported_symbols = self.db_manager.supported_symbols
#
#         # Ініціалізація моделі sentiment analysis
#         try:
#             self.logger.info(f"Завантаження моделі аналізу настроїв: {sentiment_model}")
#             self.sentiment_analyzer = pipeline("sentiment-analysis",
#                                                model=sentiment_model,
#                                                cache_dir=cache_dir)
#             self.logger.info("Модель успішно завантажена")
#         except Exception as e:
#             self.logger.error(f"Помилка завантаження моделі: {str(e)}")
#             self.sentiment_analyzer = None
#
#         # Встановлення прапорця готовності
#         self.ready = bool(self.db_manager and self.sentiment_analyzer)
#         self.logger.info(f"TwitterScraper готовий до роботи: {self.ready}")
#
#         # Створення пулу для виконання обчислювально складних завдань
#         self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, max_connections // 2))
#
#         # Ініціалізація параметрів для обробки обмежень API
#         self.api_retry_count = 3
#         self.api_cooldown_period = 300
#         self.api_error_count = 0
#         self.last_api_error_time = None
#
#     async def initialize(self):
#
#         if self.session is None:
#             self.session = aiohttp.ClientSession()
#         if not self.db_manager.is_connected:
#             self.db_manager.connect()
#         return self
#
#     async def close(self):
#         if self.session:
#             await self.session.close()
#         if self.db_manager.is_connected:
#             self.db_manager.disconnect()
#         self.thread_pool.shutdown()
#
#     async def __aenter__(self):
#         return await self.initialize()
#
#     async def __aexit__(self, exc_type, exc_val, exc_tb):
#         await self.close()
#
#     def run_in_thread(self, func, *args, **kwargs):
#         """Запускає функцію в окремому потоці для CPU-інтенсивних операцій"""
#         return self.thread_pool.submit(func, *args, **kwargs)
#
#     def search_tweets(self, query: str, days_back: int = 7,
#                             limit: Optional[int] = None, lang: str = "en") -> List[Dict]:
#         """Асинхронний пошук твітів з використанням snscrape"""
#
#         if not self.ready:
#             self.logger.error("TwitterScraper не ініціалізовано належним чином")
#             return []
#
#         # Перевірка наявності твітів у кеші
#         min_date = datetime.now() - timedelta(days=days_back)
#         cached_tweets =  self._get_cached_tweets(query, min_date)
#         if cached_tweets:
#             self.logger.info(f"Знайдено {len(cached_tweets)} твітів у кеші для запиту '{query}'")
#             return cached_tweets
#
#         # Формування пошукового запиту
#         search_query = f"{query} lang:{lang}" if lang else query
#         since_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
#         search_query += f" since:{since_date}"
#
#         self.logger.info(f"Виконання пошукового запиту: '{search_query}'")
#
#         # Оскільки snscrape не є асинхронним, запускаємо його в окремому потоці
#         # щоб не блокувати event loop
#         def scrape_tweets():
#             # Збір твітів
#             collected_tweets = []
#             try:
#                 # Виконання snscrape в окремому потоці через ThreadPoolExecutor
#                 loop = asyncio.get_event_loop()
#
#                 def _fetch_tweets():
#                     tweet_count = 0
#                     tweets = []
#                     for tweet in sntwitter.TwitterSearchScraper(search_query).get_items():
#                         # Конвертація твіту в словник
#                         tweet_dict = {
#                             "id": tweet.id,
#                             "date": tweet.date,
#                             "content": tweet.rawContent,
#                             "author_username": tweet.user.username,
#                             "displayname": tweet.user.displayname,
#                             "followers": tweet.user.followersCount,
#                             "retweets": tweet.retweetCount,
#                             "likes": tweet.likeCount,
#                             "query": query,
#                             "language": tweet.lang,
#                             "collected_at": datetime.now()
#                         }
#
#                         tweets.append(tweet_dict)
#                         tweet_count += 1
#
#                         # Перевірка ліміту
#                         if limit and tweet_count >= limit:
#                             break
#                     return tweets
#
#                 # Виконання у пулі потоків
#                 collected_tweets = loop.run_in_executor(None, _fetch_tweets)
#
#                 self.logger.info(f"Зібрано {len(collected_tweets)} твітів для запиту '{query}'")
#
#                 # Кешування результатів
#                 if collected_tweets:
#                      self._cache_tweets(query, collected_tweets)
#
#                 return collected_tweets
#
#             except Exception as e:
#                 self.logger.error(f"Помилка при пошуку твітів: {str(e)}")
#                 return []
#
#         # Використовуємо семафор для обмеження кількості одночасних пошукових запитів
#         return  scrape_tweets()
#
#     def _get_cached_tweets(self, query: str, min_date: datetime) -> Optional[List[Dict]]:
#         """Асинхронно отримує кешовані твіти з бази даних"""
#
#         if not self.db_manager:
#             self.logger.warning("DatabaseManager не ініціалізовано, отримання кешу неможливе")
#             return None
#
#         try:
#             self.logger.info(f"Пошук кешованих твітів для запиту '{query}' з {min_date}")
#
#             # Перевірка терміну дії кешу
#             max_cache_age = datetime.now() - timedelta(seconds=self.cache_expiry)
#             if min_date < max_cache_age:
#                 self.logger.info("Запитувані дані виходять за межі терміну дії кешу")
#                 return None
#
#             # SQL запит для отримання кешованих твітів
#             query_sql = """
#                         SELECT * \
#                         FROM twitter_query_cache
#                         WHERE created_at >= :start_date
#                           AND (content LIKE :content \
#                            OR query = :query) \
#                         """
#
#             # Параметри запиту
#             params = {
#                 'start_date': min_date,
#                 'content': f"%{query}%",  # Використовуємо LIKE для пошуку в тексті
#                 'query': query  # Точна відповідність запиту
#             }
#
#             # Виконання запиту асинхронно
#             rows =self.db_manager.fetch_all(query=query_sql, values=params)
#
#             if not rows:
#                 self.logger.info("Кешованих твітів не знайдено")
#                 return None
#
#             # Конвертація результатів у список словників
#             tweets = [dict(row) for row in rows]
#
#             self.logger.info(f"Знайдено {len(tweets)} кешованих твітів")
#             return tweets
#
#         except Exception as e:
#             self.logger.error(f"Помилка при отриманні кешованих твітів: {str(e)}")
#             return None
#
#     def _cache_tweets(self, query: str, tweets: List[Dict]) -> bool:
#         """Асинхронно зберігає твіти в кеш (базу даних)"""
#         if not tweets:
#             return False
#
#         try:
#             self.logger.info(f"Кешування {len(tweets)} твітів для запиту '{query}'")
#
#             # Підготовка списку задач для паралельного виконання
#             tasks = []
#             for tweet in tweets:
#                 # SQL запит для вставки твіту
#                 insert_query = """
#                                INSERT INTO tweets_raw (id, author_id, content, author_username, created_at, \
#                                                        replies_count, hashtags, \
#                                                        retweets_count, likes_count, quotes_count, language, \
#                                                        mentioned_cryptos, tweet_url, collected_at) \
#                                VALUES (:id, :date, :content, :username, :displayname, :followers, \
#                                        :retweets, :likes, :query, :lang, :collected_at) \
#                                ON CONFLICT (id) DO NOTHING \
#                                """
#
#                 tasks.append(self.db_manager.execute_query(insert_query, tweet))
#
#             # Виконання всіх запитів одночасно
#             asyncio.gather(*tasks)
#
#             self.logger.info(f"Успішно кешовано {len(tweets)} твітів")
#             return True
#
#         except Exception as e:
#             self.logger.error(f"Помилка при кешуванні твітів: {str(e)}")
#             return False
#
#     def analyze_sentiment(self, tweets: List[Dict]) -> List[Dict]:
#         """Асинхронний аналіз настроїв для твітів з використанням батчів"""
#
#         if not tweets:
#             self.logger.warning("Порожній список твітів для аналізу настроїв")
#             return []
#
#         if not self.sentiment_analyzer:
#             self.logger.error("Аналізатор настроїв не ініціалізовано")
#             return tweets
#
#         analyzed_tweets = []
#         try:
#             self.logger.info(f"Аналіз настроїв для {len(tweets)} твітів")
#
#             # Групування твітів для пакетного аналізу (оптимізація)
#             batch_size = 32  # Оптимальний розмір для більшості моделей
#             batches = [tweets[i:i + batch_size] for i in range(0, len(tweets), batch_size)]
#
#             # Функція для обробки одного пакету
#             async def process_batch(batch):
#                 texts = [tweet['content'] for tweet in batch]
#
#                 # Виконання аналізу настроїв в окремому потоці через ThreadPoolExecutor
#                 # оскільки моделі трансформерів не є асинхронними
#                 def run_sentiment_analysis():
#                     return self.sentiment_analyzer(texts, truncation=True)
#
#                 # Запуск в окремому потоці
#                 sentiment_results = asyncio.get_event_loop().run_in_executor(
#                     self.thread_pool, run_sentiment_analysis
#                 )
#
#                 batch_results = []
#                 for j, result in enumerate(sentiment_results):
#                     tweet = batch[j].copy()
#
#                     # Додавання результатів аналізу настроїв
#                     label = result['label'].lower()
#                     score = result['score']
#
#                     # Стандартизація міток настроїв
#                     if label in ['positive', 'pos']:
#                         sentiment = 'positive'
#                     elif label in ['negative', 'neg']:
#                         sentiment = 'negative'
#                     else:
#                         sentiment = 'neutral'
#
#                     # Нормалізація оцінки для негативних настроїв
#                     sentiment_score = score if sentiment == 'positive' else -score if sentiment == 'negative' else 0.0
#
#                     # Додавання результатів до твіту
#                     tweet['sentiment'] = sentiment
#                     tweet['sentiment_score'] = sentiment_score
#                     tweet['sentiment_confidence'] = score
#                     tweet['sentiment_analysis_date'] = datetime.now()
#
#                     # Асинхронне збереження результатів у базу даних
#                     self._save_sentiment_result(tweet, sentiment, sentiment_score, score)
#
#                     batch_results.append(tweet)
#
#                 return batch_results
#
#             # Обробка всіх батчів паралельно з обмеженням кількості одночасних задач
#             tasks = []
#             for batch in batches:
#                 tasks.append(process_batch(batch))
#
#             # Збір результатів з усіх пакетів
#             batch_results = asyncio.gather(*tasks)
#             for results in batch_results:
#                 analyzed_tweets.extend(results)
#
#             self.logger.info(f"Аналіз настроїв завершено для {len(analyzed_tweets)} твітів")
#             return analyzed_tweets
#
#         except Exception as e:
#             self.logger.error(f"Помилка при аналізі настроїв: {str(e)}")
#             # Повертаємо оригінальні твіти, якщо аналіз не вдався
#             return tweets
#
#     def _save_sentiment_result(self, tweet: Dict, sentiment: str,
#                                      sentiment_score: float, confidence: float) -> bool:
#         """Асинхронно зберігає результат аналізу настроїв у базу даних"""
#         try:
#             # SQL запит для вставки результату аналізу
#             query = """
#                     INSERT INTO tweet_sentiments (tweet_id, sentiment, sentiment_score, confidence, \
#                                                   model_used, analyzed_at) \
#                     VALUES (:tweet_id, :sentiment, :sentiment_score, :confidence, \
#                             :model_used, :analysis_date) ON CONFLICT (tweet_id) DO \
#                     UPDATE \
#                         SET
#                             sentiment = EXCLUDED.sentiment, sentiment_score = EXCLUDED.sentiment_score, confidence = EXCLUDED.confidence, model_used = EXCLUDED.model_used, analyzed_at = EXCLUDED.analyzed_at \
#                     """
#
#             values = {
#                 'tweet_id': tweet['id'],
#                 'sentiment': sentiment,
#                 'sentiment_score': sentiment_score,
#                 'confidence': confidence,
#                 'model_used': self.sentiment_model_name,
#                 'analyzed_at': tweet['sentiment_analysis_date']
#             }
#
#             self.db_manager.execute_query(query, values)
#             return True
#
#         except Exception as e:
#             self.logger.error(f"Помилка при збереженні результату аналізу настроїв: {str(e)}")
#             return False
#
#     def filter_by_keywords(self, tweets: List[Dict], keywords: List[str],
#                                  case_sensitive: bool = False) -> List[Dict]:
#         """Асинхронна фільтрація твітів за ключовими словами"""
#
#         if not tweets or not keywords:
#             self.logger.warning("Порожній список твітів або ключових слів")
#             return tweets
#
#         try:
#             self.logger.info(f"Фільтрація {len(tweets)} твітів за {len(keywords)} ключовими словами")
#
#             # Підготовка ключових слів
#             if not case_sensitive:
#                 keywords = [keyword.lower() for keyword in keywords]
#
#             # Функція для паралельної обробки частини твітів
#             async def process_batch(batch):
#                 filtered_batch = []
#                 for tweet in batch:
#                     content = tweet.get('content', '')
#                     if not case_sensitive:
#                         content = content.lower()
#
#                     # Перевірка наявності будь-якого ключового слова в контенті
#                     matched_keywords = [keyword for keyword in keywords if keyword in content]
#                     if matched_keywords:
#                         tweet_copy = tweet.copy()
#                         tweet_copy['matched_keywords'] = matched_keywords
#                         filtered_batch.append(tweet_copy)
#
#                 return filtered_batch
#
#             # Розділення твітів на пакети для паралельної обробки
#             batch_size = 1000  # Оптимальний розмір для обробки в пам'яті
#             batches = [tweets[i:i + batch_size] for i in range(0, len(tweets), batch_size)]
#
#             # Обробка всіх пакетів паралельно
#             tasks = [process_batch(batch) for batch in batches]
#             results = asyncio.gather(*tasks)
#
#             # Об'єднання результатів
#             filtered_tweets = []
#             for batch_results in results:
#                 filtered_tweets.extend(batch_results)
#
#             self.logger.info(f"Відфільтровано {len(filtered_tweets)} твітів")
#             return filtered_tweets
#
#         except Exception as e:
#             self.logger.error(f"Помилка при фільтрації твітів: {str(e)}")
#             return tweets
#
#     def get_trending_crypto_topics(self, top_n: int = 10) -> List[Dict]:
#         """Асинхронно отримує трендові криптовалютні теми"""
#
#         if not self.db_manager:
#             self.logger.error("DatabaseManager не ініціалізовано")
#             return []
#
#         try:
#             self.logger.info(f"Пошук топ-{top_n} трендових криптовалютних тем")
#
#             # Список криптовалютних хештегів для пошуку
#             crypto_base_tags = [
#                 "#bitcoin", "#btc", "#ethereum", "#eth", "#crypto",
#                 "#blockchain", "#defi", "#nft", "#altcoin", "#trading"
#             ]
#
#             # Отримання даних за останні 24 години
#             since_date = datetime.now() - timedelta(days=1)
# '''
#             # SQL запит для пошуку твітів із криптовалютними хештегами
#             query = """
#                 '''SELECT content FROM tweets_raw
#                 WHERE created_at >= :since_date
#                 AND (
#                     """ + " OR ".join([f"content ILIKE '%{tag}%'" for tag in crypto_base_tags]) + """
#                 )'''
#             """
# '''
#             rows = self.db_manager.fetch_all(query=query, values={'since_date': since_date})
#
#             if not rows:
#                 self.logger.info("Твітів з криптовалютними хештегами не знайдено")
#                 return []
#
#             # Вилучення всіх хештегів з твітів
#             all_hashtags = []
#             for row in rows:
#                 content = row['content']
#                 # Знаходження всіх хештегів у твіті
#                 hashtags = [
#                     tag.lower() for tag in re.findall(r'#\w+', content)
#                     if tag.lower() not in crypto_base_tags  # Виключення базових тегів
#                 ]
#                 all_hashtags.extend(hashtags)
#
#             # Підрахунок частоти використання хештегів (конкурентно)
#             def count_hashtags():
#                 return Counter(all_hashtags)
#
#             hashtag_counts = asyncio.get_event_loop().run_in_executor(
#                 self.thread_pool, count_hashtags
#             )
#
#             # Вибір топ-N найпопулярніших хештегів
#             trending_topics = [
#                 {
#                     "hashtag": hashtag,
#                     "count": count,
#                     "percentage": count / len(all_hashtags) * 100 if all_hashtags else 0
#                 }
#                 for hashtag, count in hashtag_counts.most_common(top_n)
#             ]
#
#             self.logger.info(f"Знайдено {len(trending_topics)} трендових тем")
#             return trending_topics
#
#         except Exception as e:
#             self.logger.error(f"Помилка при пошуку трендових тем: {str(e)}")
#             return []
#
#     def get_user_influence(self, username: str) -> Dict:
#
#         if not username:
#             self.logger.error("Не вказано ім'я користувача")
#             return {"error": "Username not provided"}
#
#         try:
#             self.logger.info(f"Аналіз впливовості користувача @{username}")
#
#             # Отримання останніх твітів користувача
#             user_query = f"from:{username}"
#             user_tweets = self.search_tweets(user_query, days_back=90, limit=100)
#
#             if not user_tweets:
#                 self.logger.warning(f"Твіти користувача @{username} не знайдено")
#                 return {
#                     "username": username,
#                     "found": True,
#                     "error": "No tweets found for this user"
#                 }
#
#             # Базова інформація про користувача
#             user_info = {
#                 "username": username,
#                 "found": True,
#                 "display_name": user_tweets[0].get("displayname", ""),
#                 "followers_count": user_tweets[0].get("followers", 0),
#                 "tweets_analyzed": len(user_tweets)
#             }
#
#             # Запуск паралельних задач: аналіз настроїв і пошук згадувань
#             tasks = []
#
#             # Задача 1: Аналіз настроїв твітів користувача
#             tasks.append(self.analyze_sentiment(user_tweets))
#
#             # Задача 2: Аналіз цитування та згадування користувача іншими
#             mentions_query = f"@{username}"
#             tasks.append(self.search_tweets(mentions_query, days_back=30, limit=200))
#
#             # Очікування завершення обох задач
#             results = asyncio.gather(*tasks)
#             analyzed_tweets, mention_tweets = results
#
#             # Обробка результатів аналізу настроїв
#             sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
#             for tweet in analyzed_tweets:
#                 sentiment = tweet.get("sentiment", "neutral")
#                 sentiment_counts[sentiment] += 1
#
#             dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)
#
#             # Аналіз взаємодії з твітами
#             total_likes = sum(tweet.get("likes", 0) for tweet in user_tweets)
#             total_retweets = sum(tweet.get("retweets", 0) for tweet in user_tweets)
#             avg_likes = total_likes / max(len(user_tweets), 1)
#             avg_retweets = total_retweets / max(len(user_tweets), 1)
#
#             # Визначення впливових акаунтів, які взаємодіють з користувачем
#             influential_interactions = [
#                 {"username": tweet.get("username"), "followers": tweet.get("followers", 0)}
#                 for tweet in mention_tweets
#                 if tweet.get("followers", 0) > 10000  # Поріг впливовості
#             ]
#
#             # Аналіз хештегів, які використовує користувач (конкурентно)
#             def analyze_hashtags_and_topics():
#                 # Вилучення хештегів
#                 hashtags = []
#                 for tweet in user_tweets:
#                     content = tweet.get("content", "")
#                     found_hashtags = re.findall(r'#\w+', content)
#                     hashtags.extend([tag.lower() for tag in found_hashtags])
#
#                 # Підрахунок хештегів
#                 top_hashtags = Counter(hashtags).most_common(5)
#
#                 # Аналіз тем, про які говорить користувач
#                 crypto_keywords = [
#                     "bitcoin", "btc", "ethereum", "eth", "crypto", "blockchain",
#                     "defi", "nft", "token", "coin", "mining", "wallet", "sol"
#                 ]
#
#                 topics = []
#                 for tweet in user_tweets:
#                     content = tweet.get("content", "").lower()
#                     for keyword in crypto_keywords:
#                         if keyword in content:
#                             topics.append(keyword)
#
#                 top_topics = Counter(topics).most_common(5)
#
#                 return {
#                     "top_hashtags": [{"tag": tag, "count": count} for tag, count in top_hashtags],
#                     "top_crypto_topics": [{"topic": topic, "count": count} for topic, count in top_topics]
#                 }
#
#             # Виконання аналізу хештегів та тем в окремому потоці
#             topics_results = asyncio.get_event_loop().run_in_executor(
#                 self.thread_pool, analyze_hashtags_and_topics
#             )
#
#             # Розрахунок індексу впливовості
#             engagement_rate = (avg_likes + avg_retweets * 3) / max(user_info["followers_count"], 1) * 100
#             mention_influence = len(mention_tweets) / 30  # Середня кількість згадувань на день
#             topic_diversity = len(
#                 set([item["topic"] for item in topics_results["top_crypto_topics"]])) / 13  # 13 = len(crypto_keywords)
#
#             influence_score = min(100, (
#                     (engagement_rate * 0.4) +
#                     (mention_influence * 5) +
#                     (user_info["followers_count"] / 10000 * 20) +
#                     (topic_diversity * 10)
#             ))
#
#             # Формування результату
#             result = {
#                 **user_info,
#                 "influence_score": round(influence_score, 2),
#                 "engagement": {
#                     "avg_likes": round(avg_likes, 2),
#                     "avg_retweets": round(avg_retweets, 2),
#                     "engagement_rate": round(engagement_rate, 4)
#                 },
#                 "sentiment_profile": {
#                     "counts": sentiment_counts,
#                     "dominant": dominant_sentiment
#                 },
#                 "community_interaction": {
#                     "mentions_count": len(mention_tweets),
#                     "influential_interactions": len(influential_interactions)
#                 },
#                 "topics": topics_results
#             }
#
#             self.logger.info(f"Аналіз впливовості користувача @{username} завершено")
#             return result
#
#         except Exception as e:
#             self.logger.error(f"Помилка при аналізі впливовості користувача @{username}: {str(e)}")
#             return {
#                 "username": username,
#                 "found": False,
#                 "error": str(e)
#             }
#
#     def track_influencers(self, influencers: List[str], days_back: int = 30) -> Dict[str, List[Dict]]:
#         try:
#             if not influencers and self.db_manager:
#                 influencers_data = self.db_manager.get_crypto_influencers()
#                 if not influencers_data.empty:
#                     influencers = influencers_data['username'].tolist()
#                     self.logger.info(f"Отримано {len(influencers)} інфлюенсерів з бази даних")
#
#             if not influencers:
#                 self.logger.error("Не вказано імена інфлюенсерів")
#                 return {}
#
#             self.logger.info(f"Відстеження активності {len(influencers)} крипто-інфлюенсерів за {days_back} днів")
#
#             # Якщо influencers порожній, спробуємо отримати інфлюенсерів з бази даних
#             if not influencers and self.db_manager:
#                 influencers_data = self.db_manager.get_crypto_influencers()
#                 if not influencers_data.empty:
#                     influencers = influencers_data['username'].tolist()
#                     self.logger.info(f"Отримано {len(influencers)} інфлюенсерів з бази даних")
#
#             if not influencers:
#                 self.logger.warning("Список інфлюенсерів порожній")
#                 return {}
#
#             # Асинхронне отримання даних для кожного інфлюенсера
#             results = {}
#             tasks = []
#
#             for username in influencers:
#                 task = self._process_influencer(username, days_back)
#                 tasks.append(task)
#
#             # Виконуємо всі завдання паралельно
#             influencer_results = asyncio.gather(*tasks)
#
#             # Обробляємо результати
#             for username, analyzed_tweets in influencer_results:
#                 results[username] = analyzed_tweets
#
#             self.logger.info(f"Відстеження активності інфлюенсерів завершено")
#             return results
#
#         except Exception as e:
#             self.logger.error(f"Помилка при відстеженні активності інфлюенсерів: {str(e)}")
#             return {}
#
#     def _process_influencer(self, username: str, days_back: int) -> Tuple[str, List[Dict]]:
#         self.logger.info(f"Аналіз активності інфлюенсера @{username}")
#
#         # Отримання твітів інфлюенсера
#         user_query = f"from:{username}"
#         user_tweets = self.search_tweets(user_query, days_back=days_back, limit=100)
#
#         if not user_tweets:
#             self.logger.warning(f"Твіти користувача @{username} не знайдено")
#             return username, []
#
#         # Аналіз настроїв твітів
#         analyzed_tweets = self.analyze_sentiment(user_tweets)
#
#         # Збереження активності інфлюенсера в базу даних
#         if self.db_manager:
#             # Базова інформація про інфлюенсера
#             influencer_info = {
#                 "username": username,
#                 "displayname": user_tweets[0].get("displayname", ""),
#                 "followers": user_tweets[0].get("followers", 0),
#                 "last_updated": datetime.now()
#             }
#
#             # Оновлення інформації про інфлюенсера (асинхронно)
#             self.db_manager.insert_crypto_influencer(influencer_info)
#
#             # Збереження активності інфлюенсера (асинхронно)
#             tasks = []
#             for tweet in analyzed_tweets:
#                 tweet_data = {
#                     "influencer_username": username,
#                     "tweet_id": tweet.get("id"),
#                     "content": tweet.get("content"),
#                     "date": tweet.get("date"),
#                     "likes": tweet.get("likes", 0),
#                     "retweets": tweet.get("retweets", 0),
#                     "sentiment": tweet.get("sentiment", "neutral"),
#                     "sentiment_score": tweet.get("sentiment_score", 0.0)
#                 }
#                 tasks.append(self.db_manager.insert_influencer_activity(tweet_data))
#
#             # Очікуємо завершення всіх завдань зі збереження
#             if tasks:
#                 asyncio.gather(*tasks)
#
#         return username, analyzed_tweets
#
#     def track_sentiment_over_time(self, query: str, days: int = 30,
#                                         interval: str = "day") -> pd.DataFrame:
#         if not query:
#             self.logger.error("Не вказано пошуковий запит")
#             return pd.DataFrame()
#
#         try:
#             self.logger.info(f"Аналіз зміни настроїв для запиту '{query}' за {days} днів з інтервалом '{interval}'")
#
#             # Валідація інтервалу
#             valid_intervals = ["hour", "day", "week"]
#             if interval not in valid_intervals:
#                 self.logger.warning(f"Невідомий інтервал: {interval}. Використовується 'day'")
#                 interval = "day"
#
#             # Перевірка наявності історичних даних у базі даних
#             historical_data = None
#             if self.db_manager:
#                 filters = {
#                     'query': query,
#                     'interval': interval,
#                     'start_date': datetime.now() - timedelta(days=days)
#                 }
#                 historical_data = self.db_manager.get_sentiment_time_series(filters)
#
#             # Якщо є достатньо історичних даних, повертаємо їх
#             if historical_data is not None and not historical_data.empty:
#                 rows_count = len(historical_data)
#                 expected_rows = days if interval == "day" else (days * 24 if interval == "hour" else days // 7 + 1)
#
#                 if rows_count >= expected_rows * 0.8:  # Якщо є хоча б 80% очікуваних даних
#                     self.logger.info(f"Використання {rows_count} записів історичних даних")
#                     return historical_data
#
#             # Якщо недостатньо історичних даних, збираємо нові дані
#             # Пошук твітів за вказаний період (асинхронно)
#             tweets = self.search_tweets(query, days_back=days, limit=1000)
#
#             if not tweets:
#                 self.logger.warning(f"Твіти для запиту '{query}' не знайдено")
#                 return pd.DataFrame()
#
#             # Аналіз настроїв твітів (асинхронно)
#             analyzed_tweets = self.analyze_sentiment(tweets)
#
#             # Перетворення у DataFrame для зручності аналізу
#             df = pd.DataFrame(analyzed_tweets)
#
#             # Конвертація дати у datetime формат, якщо потрібно
#             if "date" in df.columns:
#                 if isinstance(df["date"].iloc[0], str):
#                     df["date"] = pd.to_datetime(df["date"])
#             else:
#                 self.logger.warning("Колонка 'date' відсутня у даних твітів")
#                 return pd.DataFrame()
#
#             # Форматування часового інтервалу для групування
#             if interval == "hour":
#                 df["interval"] = df["date"].dt.strftime("%Y-%m-%d %H:00:00")
#             elif interval == "day":
#                 df["interval"] = df["date"].dt.strftime("%Y-%m-%d")
#             elif interval == "week":
#                 df["interval"] = df["date"].dt.to_period("W").dt.start_time
#
#             # Групування та агрегація даних
#             grouped = df.groupby("interval").agg({
#                 "sentiment_score": ["mean", "std", "count"],
#                 "sentiment": lambda x: x.value_counts().to_dict()
#             })
#
#             # Реструктуризація для зручності використання
#             result = pd.DataFrame()
#             result["date"] = grouped.index
#             result["mean_sentiment"] = grouped[("sentiment_score", "mean")]
#             result["std_sentiment"] = grouped[("sentiment_score", "std")]
#             result["tweet_count"] = grouped[("sentiment_score", "count")]
#
#             # Додавання розподілу настроїв
#             sentiment_distributions = grouped[("sentiment", "<lambda>")].tolist()
#             result["positive_count"] = [dist.get("positive", 0) for dist in sentiment_distributions]
#             result["neutral_count"] = [dist.get("neutral", 0) for dist in sentiment_distributions]
#             result["negative_count"] = [dist.get("negative", 0) for dist in sentiment_distributions]
#
#             # Розрахунок відсоткового співвідношення
#             result["positive_percent"] = result["positive_count"] / result["tweet_count"] * 100
#             result["neutral_percent"] = result["neutral_count"] / result["tweet_count"] * 100
#             result["negative_percent"] = result["negative_count"] / result["tweet_count"] * 100
#
#             # Збереження часових рядів у базу даних (асинхронно)
#             if self.db_manager:
#                 tasks = []
#                 for _, row in result.iterrows():
#                     time_series_data = {
#                         "query": query,
#                         "date": row["date"],
#                         "interval": interval,
#                         "mean_sentiment": row["mean_sentiment"],
#                         "std_sentiment": row["std_sentiment"],
#                         "tweet_count": row["tweet_count"],
#                         "positive_count": row["positive_count"],
#                         "neutral_count": row["neutral_count"],
#                         "negative_count": row["negative_count"],
#                     }
#                     tasks.append(self.db_manager.insert_sentiment_time_series(time_series_data))
#
#                 if tasks:
#                     asyncio.gather(*tasks)
#             result["query"] = query
#
#             self.logger.info(f"Аналіз зміни настроїв завершено, отримано {len(result)} часових точок")
#             return result
#
#         except Exception as e:
#             self.logger.error(f"Помилка при аналізі зміни настроїв: {str(e)}")
#             return pd.DataFrame()
#
#     def detect_sentiment_change(self, coin: str, threshold: float = 0.2) -> Dict:
#         if not coin:
#             self.logger.error("Не вказано назву криптовалюти")
#             return {"error": "Coin name not provided"}
#
#         try:
#             self.logger.info(f"Аналіз змін настроїв для {coin} з порогом {threshold}")
#
#             # Отримання даних про настрої за останні 7 днів (тиждень) (асинхронно)
#             sentiment_data = self.track_sentiment_over_time(
#                 query=f"#{coin} OR ${coin}",
#                 days=7,
#                 interval="day"
#             )
#
#             if sentiment_data.empty:
#                 self.logger.warning(f"Недостатньо даних для аналізу настроїв щодо {coin}")
#                 return {
#                     "coin": coin,
#                     "detected_change": False,
#                     "reason": "Insufficient data"
#                 }
#
#             # Підготовка даних для аналізу
#             sentiment_data = sentiment_data.sort_values("date")
#             daily_means = sentiment_data["mean_sentiment"].tolist()
#             dates = sentiment_data["date"].tolist()
#
#             # Пошук різких змін у настроях
#             changes = []
#             for i in range(1, len(daily_means)):
#                 change = daily_means[i] - daily_means[i - 1]
#                 changes.append({
#                     "date": dates[i],
#                     "previous_date": dates[i - 1],
#                     "current_sentiment": daily_means[i],
#                     "previous_sentiment": daily_means[i - 1],
#                     "change": change,
#                     "percent_change": change / (abs(daily_means[i - 1]) + 0.0001) * 100,
#                     "is_significant": abs(change) >= threshold
#                 })
#
#             # Фільтрація значимих змін
#             significant_changes = [change for change in changes if change["is_significant"]]
#
#             # Визначення найбільш значної зміни
#             if significant_changes:
#                 most_significant = max(significant_changes, key=lambda x: abs(x["change"]))
#                 direction = "positive" if most_significant["change"] > 0 else "negative"
#
#                 # Пошук ймовірних причин зміни настрою
#                 start_date = most_significant["previous_date"]
#                 end_date = most_significant["date"]
#
#                 # Отримання твітів за цей період (асинхронно)
#                 tweets_query = f"#{coin} OR ${coin}"
#                 tweets = self.search_tweets(
#                     query=tweets_query,
#                     days_back=7,
#                     limit=200
#                 )
#
#                 # Фільтрація твітів за датами
#                 relevant_tweets = [
#                     tweet for tweet in tweets
#                     if "date" in tweet and start_date <= tweet["date"] <= end_date
#                 ]
#
#                 # Аналіз найпопулярніших твітів
#                 popular_tweets = sorted(
#                     relevant_tweets,
#                     key=lambda x: x.get("retweets", 0) + x.get("likes", 0),
#                     reverse=True
#                 )[:5]
#
#                 result = {
#                     "coin": coin,
#                     "detected_change": True,
#                     "change_direction": direction,
#                     "change_magnitude": abs(most_significant["change"]),
#                     "previous_sentiment": most_significant["previous_sentiment"],
#                     "current_sentiment": most_significant["current_sentiment"],
#                     "change_date": most_significant["date"],
#                     "previous_date": most_significant["previous_date"],
#                     "percent_change": most_significant["percent_change"],
#                     "potential_causes": [
#                         {
#                             "tweet_id": tweet.get("id"),
#                             "content": tweet.get("content"),
#                             "username": tweet.get("username"),
#                             "engagement": tweet.get("retweets", 0) + tweet.get("likes", 0)
#                         }
#                         for tweet in popular_tweets
#                     ]
#                 }
#
#                 # Збереження інформації про зміну настроїв у базу даних (асинхронно)
#                 if self.db_manager:
#                     event_data = {
#                         "coin": coin,
#                         "event_type": f"sentiment_change_{direction}",
#                         "event_date": most_significant["date"],
#                         "description": f"Significant {direction} sentiment change detected",
#                         "magnitude": abs(most_significant["change"]),
#                         "previous_value": most_significant["previous_sentiment"],
#                         "current_value": most_significant["current_sentiment"],
#                         "detection_date": datetime.now()
#                     }
#                     self.db_manager.insert_crypto_event(event_data)
#
#                 self.logger.info(
#                     f"Виявлено значну зміну настроїв для {coin}: {direction} ({most_significant['percent_change']:.2f}%)")
#                 return result
#             else:
#                 self.logger.info(f"Не виявлено значних змін настроїв для {coin}")
#                 return {
#                     "coin": coin,
#                     "detected_change": False,
#                     "reason": "No significant changes detected",
#                     "max_change": max(abs(change["change"]) for change in changes) if changes else 0,
#                     "threshold": threshold
#                 }
#
#         except Exception as e:
#             self.logger.error(f"Помилка при виявленні змін настроїв для {coin}: {str(e)}")
#             return {
#                 "coin": coin,
#                 "detected_change": False,
#                 "error": str(e)
#             }
#
#     @staticmethod
#     def _interpret_correlation(correlation_value: float) -> Dict[str, Any]:
#         abs_corr = abs(correlation_value)
#
#         # Визначення сили зв'язку
#         if abs_corr < 0.2:
#             strength = "дуже слабкий"
#             significance_level = "низька"
#         elif abs_corr < 0.4:
#             strength = "слабкий"
#             significance_level = "низька"
#         elif abs_corr < 0.6:
#             strength = "помірний"
#             significance_level = "середня"
#         elif abs_corr < 0.8:
#             strength = "сильний"
#             significance_level = "висока"
#         else:
#             strength = "дуже сильний"
#             significance_level = "висока"
#
#         # Визначення напрямку зв'язку
#         if correlation_value > 0.1:
#             direction = "позитивний"
#             description = f"Виявлено {strength} {direction} зв'язок: зростання одного параметра корелює зі зростанням іншого."
#         elif correlation_value < -0.1:
#             direction = "негативний"
#             description = f"Виявлено {strength} {direction} зв'язок: зростання одного параметра корелює зі зниженням іншого."
#         else:
#             direction = "нейтральний"
#             strength = "відсутній"
#             significance_level = "відсутня"
#             description = "Значущий зв'язок між параметрами не виявлено."
#
#         # Оцінка практичної значущості
#         if abs_corr > 0.5:
#             practical_significance = "висока практична значущість"
#         elif abs_corr > 0.3:
#             practical_significance = "помірна практична значущість"
#         else:
#             practical_significance = "низька практична значущість"
#
#         # Формування відповіді
#         return {
#             "strength": strength,
#             "direction": direction,
#             "description": description,
#             "statistical_significance": significance_level,
#             "practical_significance": practical_significance,
#             "coefficient": float(correlation_value),
#             "interpretation": f"{description} Коефіцієнт кореляції: {correlation_value:.4f} ({practical_significance})."
#         }
#
#     def correlate_with_price(self, tweets: List[Dict], price_data: pd.DataFrame) -> Dict:
#         if not tweets or price_data.empty:
#             self.logger.warning("Порожні дані для кореляційного аналізу")
#             return {"error": "Empty data for correlation analysis"}
#
#         try:
#             self.logger.info(f"Кореляційний аналіз {len(tweets)} твітів з ціновими даними")
#
#             # Перевірка наявності аналізу настроїв у твітах
#             if "sentiment_score" not in tweets[0]:
#                 self.logger.info("Твіти потребують аналізу настроїв")
#                 tweets = self.analyze_sentiment(tweets)
#
#             # Перетворення твітів у DataFrame
#             tweets_df = pd.DataFrame(tweets)
#
#             # Перевірка необхідних колонок у DataFrame з цінами
#             required_columns = ["date", "close"]
#             if not all(col in price_data.columns for col in required_columns):
#                 self.logger.error("Відсутні необхідні колонки в даних про ціни")
#                 return {"error": "Missing required columns in price data"}
#
#             # Конвертація дат у datetime формат
#             tweets_df["date"] = pd.to_datetime(tweets_df["date"])
#             price_data["date"] = pd.to_datetime(price_data["date"])
#
#             # Агрегація настроїв за днями
#             tweets_df["date_day"] = tweets_df["date"].dt.date
#             daily_sentiment = tweets_df.groupby("date_day").agg({
#                 "sentiment_score": "mean",
#                 "id": "count"
#             }).rename(columns={"id": "tweet_count"}).reset_index()
#
#             # Підготовка даних про ціни з відповідною датою
#             price_data["date_day"] = price_data["date"].dt.date
#             price_daily = price_data.groupby("date_day").agg({
#                 "close": "last",
#                 "volume": "sum" if "volume" in price_data.columns else None
#             }).reset_index()
#
#             # Об'єднання даних для кореляційного аналізу
#             merged_data = pd.merge(daily_sentiment, price_daily, on="date_day", how="outer").dropna()
#             if len(merged_data) < 3:
#                 self.logger.warning("Недостатньо точок даних для кореляційного аналізу")
#                 return {"error": "Insufficient data points for correlation analysis"}
#
#             # Розрахунок кореляцій
#             correlation_same_day = merged_data["sentiment_score"].corr(merged_data["close"])
#
#             # Кореляція з наступним днем (настрої як предиктор)
#             merged_data["next_day_close"] = merged_data["close"].shift(-1)
#             correlation_next_day = merged_data["sentiment_score"].corr(merged_data["next_day_close"])
#
#             # Кореляція з попереднім днем (ціна як предиктор)
#             merged_data["prev_day_sentiment"] = merged_data["sentiment_score"].shift(1)
#             correlation_prev_day = merged_data["close"].corr(merged_data["prev_day_sentiment"])
#
#             # Розрахунок статистичної значущості для основної кореляції
#             from scipy import stats
#             r = correlation_same_day
#             n = len(merged_data)
#             t_stat = r * np.sqrt(n - 2) / np.sqrt(1 - r ** 2)
#             p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
#
#             # Застосування лагів для пошуку оптимального часового зсуву
#             max_lag = min(5, len(merged_data) // 3)  # Максимальний лаг для аналізу
#             lag_results = []
#
#             for lag in range(-max_lag, max_lag + 1):
#                 if lag == 0:
#                     continue  # Вже розраховано вище
#
#                 # Створення зсунутих даних
#                 if lag > 0:
#                     lagged_sentiment = merged_data["sentiment_score"].shift(lag)
#                     lagged_corr = merged_data["close"].corr(lagged_sentiment)
#                     direction = "sentiment_follows_price"
#                 else:
#                     lagged_price = merged_data["close"].shift(-lag)
#                     lagged_corr = merged_data["sentiment_score"].corr(lagged_price)
#                     direction = "price_follows_sentiment"
#
#                 lag_results.append({
#                     "lag": lag,
#                     "correlation": lagged_corr,
#                     "direction": direction
#                 })
#
#             # Знаходження найсильнішої кореляції
#             strongest_lag = max(lag_results, key=lambda x: abs(x["correlation"]), default=None)
#
#             # Підготовка результатів
#             result = {
#                 "same_day_correlation": correlation_same_day,
#                 "next_day_correlation": correlation_next_day,
#                 "previous_day_correlation": correlation_prev_day,
#                 "data_points": len(merged_data),
#                 "statistical_significance": {
#                     "t_statistic": t_stat,
#                     "p_value": p_value,
#                     "is_significant": p_value < 0.05
#                 },
#                 "strongest_correlation": strongest_lag,
#                 "lag_analysis": lag_results,
#                 "interpretation": {
#                     "same_day": self._interpret_correlation(correlation_same_day),
#                     "predictive_power": self._interpret_correlation(correlation_next_day),
#                     "reactive_nature": self._interpret_correlation(correlation_prev_day)
#                 }
#             }
#
#             self.logger.info(f"Кореляційний аналіз завершено. Основна кореляція: {correlation_same_day:.4f}")
#             return result
#
#         except Exception as e:
#             self.logger.error(f"Помилка при кореляційному аналізі: {str(e)}")
#             return {"error": str(e)}
#
#     def handle_api_rate_limits(self, retry_count: int = 3, cooldown_period: int = 300) -> None:
#         self.logger.info(
#             f"Налаштування параметрів обробки обмежень API: повторні спроби={retry_count}, період очікування={cooldown_period}с")
#
#         # Зберігаємо параметри для використання в інших методах
#         self.api_retry_count = retry_count
#         self.api_cooldown_period = cooldown_period
#
#         # Додаємо лічильник помилок і часову мітку останньої помилки
#         self.api_error_count = 0
#         self.last_api_error_time = None
#
#         # Створюємо асинхронний декоратор для повторних спроб при помилках API
#         def retry_on_api_limit(func):
#             async def wrapper(*args, **kwargs):
#                 attempts = 0
#                 while attempts < self.api_retry_count:
#                     try:
#                         # Перевірка затримки після останньої помилки
#                         if self.last_api_error_time:
#                             time_since_error = (datetime.now() - self.last_api_error_time).total_seconds()
#                             if time_since_error < self.api_cooldown_period:
#                                 wait_time = self.api_cooldown_period - time_since_error
#                                 self.logger.info(f"Очікування {wait_time:.1f}с перед наступною спробою")
#                                 await asyncio.sleep(wait_time)
#
#                         # Виклик оригінального методу
#                         result = func(*args, **kwargs)
#
#                         # Скидання лічильника помилок при успішній спробі
#                         if attempts > 0:
#                             self.api_error_count = 0
#                             self.last_api_error_time = None
#                             self.logger.info("Запит успішний після повторних спроб")
#
#                         return result
#
#                     except Exception as e:
#                         error_message = str(e).lower()
#                         if "rate limit" in error_message or "too many requests" in error_message:
#                             attempts += 1
#                             self.api_error_count += 1
#                             self.last_api_error_time = datetime.now()
#
#                             # Збільшуємо час очікування з кожною спробою
#                             wait_time = self.api_cooldown_period * (2 ** attempts)
#
#                             self.logger.warning(f"Досягнуто обмеження API (спроба {attempts}/{self.api_retry_count}). "
#                                                 f"Очікування {wait_time}с перед наступною спробою")
#
#                             # Реєстрація помилки в базі даних
#                             if self.db_manager:
#                                 error_data = {
#                                     "error_type": "rate_limit",
#                                     "error_message": str(e),
#                                     "function_name": func.__name__,
#                                     "timestamp": datetime.now()
#                                 }
#                                 self.db_manager.insert_scraping_error(error_data)
#
#                             asyncio.sleep(wait_time)
#                         else:
#                             # Якщо помилка не пов'язана з обмеженням швидкості, передаємо її далі
#                             self.logger.error(f"Помилка не пов'язана з обмеженням API: {str(e)}")
#                             raise
#                 original_search_tweets = self.search_tweets
#                 self.search_tweets = retry_on_api_limit(original_search_tweets)
#                 # Якщо всі спроби вичерпано
#                 self.logger.error(f"Вичерпано всі {self.api_retry_count} спроб через обмеження API")
#                 raise Exception(f"API rate limit exceeded after {self.api_retry_count} attempts")
#
#             return wrapper
#
#         # Застосовуємо декоратор до методів, які взаємодіють з API
#         self.search_tweets = retry_on_api_limit(self.search_tweets)
#
#         self.logger.info("Обробка обмежень API налаштована успішно")
#
#     def detect_crypto_events(self, tweets: List[Dict], min_mentions: int = 50) -> List[Dict]:
#         if not tweets:
#             self.logger.warning("Порожній список твітів для виявлення подій")
#             return []
#
#         try:
#             self.logger.info(f"Аналіз {len(tweets)} твітів для виявлення криптоподій")
#
#             # Аналіз настроїв, якщо не проведено раніше (асинхронно)
#             if "sentiment" not in tweets[0]:
#                 tweets = self.analyze_sentiment(tweets)
#
#             # Перетворення у DataFrame для зручності аналізу
#             tweets_df = pd.DataFrame(tweets)
#             tweets_df["date"] = pd.to_datetime(tweets_df["date"])
#             tweets_df["date_day"] = tweets_df["date"].dt.date
#
#             # Виявлення згадувань криптовалют
#             crypto_mentions = {}
#             crypto_patterns = {
#                 "BTC": r'#?(?:bitcoin|btc|бітко[її]н)\b',
#                 "ETH": r'#?(?:ethereum|eth|ether|етеріум|ефір)\b',
#                 "BNB": r'#?(?:binance\s*coin|bnb)\b',
#                 "SOL": r'#?(?:solana|sol)\b',
#                 "XRP": r'#?(?:ripple|xrp)\b',
#                 "ADA": r'#?(?:cardano|ada)\b',
#                 "DOGE": r'#?(?:dogecoin|doge)\b',
#                 "DOT": r'#?(?:polkadot|dot)\b',
#                 "SHIB": r'#?(?:shiba\s*inu|shib)\b',
#                 "AVAX": r'#?(?:avalanche|avax)\b'
#             }
#
#             # Підрахунок згадувань
#             # Створити всі індекси одразу
#             for symbol, pattern in crypto_patterns.items():
#                 tweets_df[f"mentions_{symbol}"] = tweets_df["content"].str.lower().str.contains(
#                     pattern, regex=True, case=False).astype(int)
#
#             # Потім використовувати їх
#             crypto_mentions = {symbol: tweets_df[f"mentions_{symbol}"].sum() for symbol in crypto_patterns.keys()}
#
#             # Групування за днями
#             daily_activity = tweets_df.groupby("date_day").agg({
#                 "id": "count",
#                 **{f"mentions_{symbol}": "sum" for symbol in crypto_patterns.keys()},
#                 "sentiment_score": "mean"
#             }).reset_index()
#
#             baseline_activity = daily_activity["id"].median()
#             anomaly_threshold = baseline_activity * 1.5
#
#             # Виявлення днів з аномаліями
#             anomaly_days = daily_activity[daily_activity["id"] > anomaly_threshold]
#
#             if anomaly_days.empty:
#                 self.logger.info("Не виявлено днів з аномальною активністю")
#                 top_cryptos = sorted(crypto_mentions.items(), key=lambda x: x[1], reverse=True)
#                 return [{
#                     "event_type": "most_discussed_cryptos",
#                     "date": datetime.now(),
#                     "cryptos": [{"symbol": symbol, "mentions": count} for symbol, count in top_cryptos[:5]],
#                     "total_tweets_analyzed": len(tweets)
#                 }]
#
#             detected_events = []
#
#             def process_anomaly_day(day_data):
#                 date = day_data["date_day"]
#                 day_tweets = tweets_df[tweets_df["date_day"] == date]
#
#                 coins_with_spikes = []
#                 for symbol in crypto_patterns.keys():
#                     mentions = day_data[f"mentions_{symbol}"]
#                     if mentions >= min_mentions:
#                         coin_tweets = day_tweets[day_tweets[f"mentions_{symbol}"] == 1]
#                         if len(coin_tweets) == 0:
#                             continue
#
#                         coin_tweets["popularity"] = coin_tweets["retweets"] + coin_tweets["likes"]
#                         top_tweets = coin_tweets.sort_values("popularity", ascending=False).head(3)
#
#                         avg_sentiment = coin_tweets["sentiment_score"].mean()
#                         sentiment_direction = (
#                             "позитивний" if avg_sentiment > 0.1 else
#                             "негативний" if avg_sentiment < -0.1 else "нейтральний"
#                         )
#
#                         coins_with_spikes.append({
#                             "symbol": symbol,
#                             "mentions": int(mentions),
#                             "sentiment": sentiment_direction,
#                             "sentiment_score": float(avg_sentiment),
#                             "example_tweets": top_tweets[["content", "retweets", "likes"]].to_dict('records')
#                         })
#
#                 if coins_with_spikes:
#                     main_coin = max(coins_with_spikes, key=lambda x: x["mentions"])
#
#                     event = {
#                         "event_type": "activity_spike",
#                         "date": date,
#                         "total_tweets": int(day_data["id"]),
#                         "baseline_tweets": int(baseline_activity),
#                         "primary_coin": main_coin["symbol"],
#                         "involved_coins": [coin["symbol"] for coin in coins_with_spikes],
#                         "details": coins_with_spikes,
#                         "overall_sentiment": float(day_data["sentiment_score"])
#                     }
#
#                     if self.db_manager:
#                         db_tasks = []
#                         for coin_data in coins_with_spikes:
#                             event_data = {
#                                 "coin": coin_data["symbol"],
#                                 "event_type": "mention_spike",
#                                 "event_date": date,
#                                 "description": f"Spike in mentions for {coin_data['symbol']} with {sentiment_direction} sentiment",
#                                 "magnitude": coin_data["mentions"],
#                                 "previous_value": 0,
#                                 "current_value": coin_data["mentions"],
#                                 "detection_date": datetime.now()
#                             }
#                             db_tasks.append(self.db_manager.insert_crypto_event(event_data))
#                         if db_tasks:
#                             asyncio.gather(*db_tasks)
#
#                     return event
#                 return None
#
#             anomaly_tasks = [process_anomaly_day(day_data) for _, day_data in anomaly_days.iterrows()]
#             anomaly_results = asyncio.gather(*anomaly_tasks)
#
#             detected_events.extend([event for event in anomaly_results if event is not None])
#
#             if not detected_events:
#                 max_activity_day = daily_activity.loc[daily_activity["id"].idxmax()]
#                 sentiment_volatility = {}
#                 for symbol in crypto_patterns.keys():
#                     symbol_tweets = tweets_df[tweets_df[f"mentions_{symbol}"] == 1]
#                     if len(symbol_tweets) >= min_mentions // 2:
#                         sentiment_volatility[symbol] = symbol_tweets["sentiment_score"].std()
#
#                 if sentiment_volatility:
#                     volatile_coin = max(sentiment_volatility.items(), key=lambda x: x[1])
#                     detected_events.append({
#                         "event_type": "noteworthy_activity",
#                         "date": max_activity_day["date_day"],
#                         "total_tweets": int(max_activity_day["id"]),
#                         "most_volatile_coin": volatile_coin[0],
#                         "sentiment_volatility": float(volatile_coin[1]),
#                         "coins_mentioned": [symbol for symbol, mentions in crypto_mentions.items() if mentions > 0]
#                     })
#
#             detected_events.append({
#                 "event_type": "period_summary",
#                 "date_range": {
#                     "start": min(tweets_df["date_day"]),
#                     "end": max(tweets_df["date_day"])
#                 },
#                 "total_tweets": len(tweets),
#                 "most_mentioned_coins": [
#                     {"symbol": symbol, "mentions": count}
#                     for symbol, count in sorted(crypto_mentions.items(), key=lambda x: x[1], reverse=True)[:5]
#                     if count > 0
#                 ],
#                 "overall_sentiment": float(tweets_df["sentiment_score"].mean())
#             })
#
#             self.logger.info(f"Виявлено {len(detected_events)} криптоподій")
#             return detected_events
#
#         except Exception as e:
#             self.logger.error(f"Помилка при виявленні криптоподій: {str(e)}")
#             import traceback
#             self.logger.error(traceback.format_exc())
#             return []
#
#
#     async def get_error_stats(self) -> Dict:
#         if not self.db_manager:
#             self.logger.error("DatabaseManager не ініціалізовано, отримання статистики помилок неможливе")
#             return {"error": "Database manager not initialized"}
#
#         try:
#             self.logger.info("Отримання статистики помилок збору даних")
#
#             # Отримання даних про помилки з бази даних (асинхронно)
#             error_data = await self.db_manager.get_scraping_errors()
#
#             if error_data.empty:
#                 self.logger.info("Помилок не знайдено")
#                 return {
#                     "total_errors": 0,
#                     "error_types": {},
#                     "time_distribution": {},
#                     "most_recent": None
#                 }
#
#             # Агрегація даних для статистики
#             total_errors = len(error_data)
#             error_types = error_data["error_type"].value_counts().to_dict()
#
#             # Аналіз розподілу помилок за часом
#             error_data["timestamp"] = pd.to_datetime(error_data["timestamp"])
#             error_data["date"] = error_data["timestamp"].dt.date
#             time_distribution = error_data["date"].value_counts().sort_index().to_dict()
#
#             # Конвертація datetime.date в str для серіалізації JSON
#             time_distribution = {str(date): count for date, count in time_distribution.items()}
#
#             # Інформація про останню помилку
#             most_recent = error_data.sort_values("timestamp", ascending=False).iloc[0].to_dict()
#             if "timestamp" in most_recent:
#                 most_recent["timestamp"] = most_recent["timestamp"].isoformat()
#
#             # Аналіз типових патернів помилок
#             error_patterns = {}
#             if "error_message" in error_data.columns:
#                 # Спрощені повідомлення про помилки для групування схожих випадків
#                 error_data["simplified_message"] = error_data["error_message"].str.extract(r'^([^:]+)')[0]
#                 message_counts = error_data["simplified_message"].value_counts().head(5).to_dict()
#                 error_patterns = message_counts
#
#             # Аналіз функцій, що викликають найбільше помилок
#             function_errors = {}
#             if "function_name" in error_data.columns:
#                 function_counts = error_data["function_name"].value_counts().to_dict()
#                 function_errors = function_counts
#
#             # Тренд помилок за останній тиждень
#             week_ago = datetime.now() - timedelta(days=7)
#             weekly_errors = error_data[error_data["timestamp"] >= week_ago]
#             weekly_count = len(weekly_errors)
#
#             # Порівняння з попереднім тижнем
#             two_weeks_ago = datetime.now() - timedelta(days=14)
#             prev_week_errors = error_data[(error_data["timestamp"] >= two_weeks_ago) &
#                                           (error_data["timestamp"] < week_ago)]
#             prev_week_count = len(prev_week_errors)
#
#             weekly_change = weekly_count - prev_week_count
#             weekly_change_percent = (weekly_change / max(1, prev_week_count)) * 100
#
#             result = {
#                 "total_errors": total_errors,
#                 "error_types": error_types,
#                 "time_distribution": time_distribution,
#                 "most_recent": most_recent,
#                 "error_patterns": error_patterns,
#                 "function_errors": function_errors,
#                 "weekly_stats": {
#                     "current_week": weekly_count,
#                     "previous_week": prev_week_count,
#                     "change": weekly_change,
#                     "change_percent": round(weekly_change_percent, 2)
#                 },
#                 "analysis_date": datetime.now().isoformat()
#             }
#
#             self.logger.info(f"Отримано статистику з {total_errors} помилок")
#             return result
#
#         except Exception as e:
#             self.logger.error(f"Помилка при отриманні статистики помилок: {str(e)}")
#             return {"error": str(e)}
#
# async def main():
#         import logging
#
#         # Налаштування логування
#         logging.basicConfig(level=logging.INFO)
#
#         # Ініціалізація скрапера
#         scraper = TwitterScraper(
#             sentiment_model="finiteautomata/bertweet-base-sentiment-analysis",
#             log_level=logging.INFO
#         )
#
#         # Приклад 1: Аналіз твітів про Bitcoin
#         print("\n=== Аналіз твітів про Bitcoin ===")
#         bitcoin_tweets = await scraper.search_tweets("#bitcoin OR $BTC", days_back=3, limit=50)
#         analyzed_tweets = await scraper.analyze_sentiment(bitcoin_tweets)
#
#         if analyzed_tweets:
#             positive = sum(1 for t in analyzed_tweets if t['sentiment'] == 'positive')
#             negative = sum(1 for t in analyzed_tweets if t['sentiment'] == 'negative')
#             print(f"Результати аналізу: {len(analyzed_tweets)} твітів")
#             print(f"Позитивні: {positive} ({positive / len(analyzed_tweets) * 100:.1f}%)")
#             print(f"Негативні: {negative} ({negative / len(analyzed_tweets) * 100:.1f}%)")
#
#         # Приклад 2: Трендові крипто-теми
#         print("\n=== Трендові крипто-теми ===")
#         trending_topics = scraper.get_trending_crypto_topics(top_n=5)
#         for topic in trending_topics:
#             print(f"{topic['hashtag']}: {topic['count']} згадок ({topic['percentage']:.1f}%)")
#
#         # Приклад 3: Аналіз впливовості користувача
#         print("\n=== Аналіз впливовості користувача ===")
#         influence = scraper.get_user_influence("elonmusk")
#         print(f"Користувач: @{influence['username']}")
#         if 'influence_score' in influence:
#             print(f"Оцінка впливовості: {influence['influence_score']}")
#         else:
#             print("Оцінка впливовості не обчислена")
#         print(f"Середня кількість лайків: {influence['engagement']['avg_likes']}")
#
#         # Приклад 4: Відстеження змін настроїв
#         print("\n=== Відстеження змін настроїв ===")
#         sentiment_trend =  scraper.track_sentiment_over_time("#ethereum OR $ETH", days=7)
#         if not sentiment_trend.empty:
#             print(sentiment_trend[['date', 'mean_sentiment', 'positive_percent', 'negative_percent']].to_string())
#
#         # Приклад 5: Виявлення подій
#         print("\n=== Виявлення крипто-подій ===")
#         events =  scraper.detect_crypto_events(analyzed_tweets)
#         for event in events:
#             print(f"Подія типу '{event['event_type']}' виявлена {event['date']}")
#
# if __name__ == "__main__":
#     asyncio.run(main())
# """