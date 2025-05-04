"""""
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import logging
import re
import time
import praw
from typing import List, Dict, Optional, Union, Tuple
from random import randint
from data.db import DatabaseManager

class CryptoNewsScraper:


    def __init__(self, news_sources: List[str] = None,
                 sentiment_analyzer=None,
                 logger: Optional[logging.Logger] = None):

        self.news_sources = news_sources or [
            'coindesk', 'cointelegraph', 'decrypt', 'cryptoslate',
            'theblock', 'cryptopanic', 'coinmarketcal', 'feedly',
            'newsnow', 'reddit'
        ]
        self.sentiment_analyzer = sentiment_analyzer

        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger('crypto_news_scraper')
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Налаштування для Reddit API
        self.reddit = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def _make_request(self, url, retries=3, backoff_factor=0.3):

        for i in range(retries):
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                if response.status_code == 200:
                    return response
                elif response.status_code in [403, 429]:
                    self.logger.warning(f"Отримано код {response.status_code} при запиті до {url}. Очікування...")
                    time.sleep((backoff_factor * (2 ** i)) + randint(1, 3))
                else:
                    self.logger.error(f"Помилка при запиті до {url}: HTTP {response.status_code}")
                    return None
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Помилка з'єднання при запиті до {url}: {e}")
                time.sleep((backoff_factor * (2 ** i)) + randint(1, 3))

        return None

    def _initialize_reddit(self, client_id, client_secret, user_agent):

        try:
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
            return True
        except Exception as e:
            self.logger.error(f"Помилка при ініціалізації Reddit API: {e}")
            return False

    def scrape_coindesk(self, days_back: int = 1,
                        categories: List[str] = None) -> List[Dict]:

        self.logger.info("Збір новин з CoinDesk...")
        news_data = []

        try:
            # Визначення дати, починаючи з якої збираємо новини
            start_date = datetime.now() - timedelta(days=days_back)

            # Базова URL для CoinDesk
            base_url = "https://www.coindesk.com"

            # Список категорій, якщо не вказано
            if not categories:
                categories = ["markets", "business", "policy", "tech"]

            for category in categories:
                page = 1
                continue_scraping = True

                while continue_scraping:
                    url = f"{base_url}/{category}/?page={page}"
                    response = requests.get(url, headers=self.headers)

                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        articles = soup.select('article.article-cardstyles__CardWrapper-sc-5xitv1-0')

                        if not articles:
                            continue_scraping = False
                            continue

                        for article in articles:
                            try:
                                # Отримання дати публікації
                                date_str = article.select_one('time')['datetime']
                                pub_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))

                                # Перевірка, чи стаття відповідає періоду
                                if pub_date < start_date:
                                    continue_scraping = False
                                    break

                                # Отримання заголовка і посилання
                                title_elem = article.select_one('h6')
                                if not title_elem:
                                    continue

                                title = title_elem.text.strip()
                                link_elem = article.select_one('a')
                                link = base_url + link_elem['href'] if link_elem else None

                                # Отримання короткого опису
                                summary_elem = article.select_one('p.typography__StyledTypography-sc-owin6q-0')
                                summary = summary_elem.text.strip() if summary_elem else ""

                                # Додавання даних до списку
                                news_data.append({
                                    'title': title,
                                    'summary': summary,
                                    'link': link,
                                    'source': 'coindesk',
                                    'category': category,
                                    'published_at': pub_date,
                                    'scraped_at': datetime.now()
                                })
                            except Exception as e:
                                self.logger.error(f"Помилка при обробці статті CoinDesk: {e}")

                    else:
                        self.logger.error(
                            f"Не вдалося завантажити сторінку {url}. Код відповіді: {response.status_code}")
                        continue_scraping = False

                    # Перехід на наступну сторінку
                    page += 1

                    # Затримка для запобігання блокуванню
                    time.sleep(randint(1, 3))

        except Exception as e:
            self.logger.error(f"Загальна помилка при скрапінгу CoinDesk: {e}")

        self.logger.info(f"Зібрано {len(news_data)} новин з CoinDesk")
        return news_data

    def scrape_cointelegraph(self, days_back: int = 1,
                             categories: List[str] = None) -> List[Dict]:

        self.logger.info("Збір новин з Cointelegraph...")
        news_data = []

        try:
            # Визначення дати, починаючи з якої збираємо новини
            start_date = datetime.now() - timedelta(days=days_back)

            # Базова URL для Cointelegraph
            base_url = "https://cointelegraph.com"

            # Список категорій, якщо не вказано
            if not categories:
                categories = ["news", "markets", "features", "analysis"]

            for category in categories:
                page = 1
                continue_scraping = True

                while continue_scraping and page <= 5:  # Обмеження по кількості сторінок
                    url = f"{base_url}/{category}?page={page}"
                    response = requests.get(url, headers=self.headers)

                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        articles = soup.select('article.post-card-inline')

                        if not articles:
                            continue_scraping = False
                            continue

                        for article in articles:
                            try:
                                # Отримання дати публікації
                                date_elem = article.select_one('time.post-card-inline__date')
                                if not date_elem or not date_elem.get('datetime'):
                                    continue

                                date_str = date_elem['datetime']
                                pub_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))

                                # Перевірка, чи стаття відповідає періоду
                                if pub_date < start_date:
                                    continue_scraping = False
                                    break

                                # Отримання заголовка і посилання
                                title_elem = article.select_one('span.post-card-inline__title')
                                if not title_elem:
                                    continue

                                title = title_elem.text.strip()
                                link_elem = article.select_one('a.post-card-inline__title-link')
                                link = base_url + link_elem['href'] if link_elem else None

                                # Отримання короткого опису
                                summary_elem = article.select_one('p.post-card-inline__text')
                                summary = summary_elem.text.strip() if summary_elem else ""

                                # Додавання даних до списку
                                news_data.append({
                                    'title': title,
                                    'summary': summary,
                                    'link': link,
                                    'source': 'cointelegraph',
                                    'category': category,
                                    'published_at': pub_date,
                                    'scraped_at': datetime.now()
                                })
                            except Exception as e:
                                self.logger.error(f"Помилка при обробці статті Cointelegraph: {e}")

                    else:
                        self.logger.error(
                            f"Не вдалося завантажити сторінку {url}. Код відповіді: {response.status_code}")
                        continue_scraping = False

                    # Перехід на наступну сторінку
                    page += 1

                    # Затримка для запобігання блокуванню
                    time.sleep(randint(1, 3))

        except Exception as e:
            self.logger.error(f"Загальна помилка при скрапінгу Cointelegraph: {e}")

        self.logger.info(f"Зібрано {len(news_data)} новин з Cointelegraph")
        return news_data

    def scrape_decrypt(self, days_back: int = 1,
                       categories: List[str] = None) -> List[Dict]:

        self.logger.info("Збір новин з Decrypt...")
        news_data = []

        try:
            # Визначення дати, починаючи з якої збираємо новини
            start_date = datetime.now() - timedelta(days=days_back)

            # Базова URL для Decrypt
            base_url = "https://decrypt.co"

            # Список категорій, якщо не вказано
            if not categories:
                categories = ["news", "analysis", "features", "learn"]

            for category in categories:
                page = 1
                continue_scraping = True

                while continue_scraping and page <= 5:  # Обмеження по кількості сторінок
                    url = f"{base_url}/{category}/page/{page}"
                    response = requests.get(url, headers=self.headers)

                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        articles = soup.select('article.cardV2')

                        if not articles:
                            continue_scraping = False
                            continue

                        for article in articles:
                            try:
                                # Отримання заголовка і посилання
                                title_elem = article.select_one('h3.cardV2__title')
                                if not title_elem:
                                    continue

                                title = title_elem.text.strip()
                                link_elem = article.select_one('a.cardV2__wrap')
                                link = link_elem['href'] if link_elem else None

                                # Повний URL, якщо він відносний
                                if link and not link.startswith('http'):
                                    link = base_url + link

                                # Отримання дати публікації (дата може бути в форматі "X hours ago")
                                date_elem = article.select_one('time.cardV2__date')
                                if not date_elem:
                                    continue

                                date_text = date_elem.text.strip()

                                # Спроба перетворити текст дати у об'єкт datetime
                                pub_date = None

                                if 'hours ago' in date_text or 'hour ago' in date_text:
                                    hours = int(re.search(r'(\d+)', date_text).group(1))
                                    pub_date = datetime.now() - timedelta(hours=hours)
                                elif 'days ago' in date_text or 'day ago' in date_text:
                                    days = int(re.search(r'(\d+)', date_text).group(1))
                                    pub_date = datetime.now() - timedelta(days=days)
                                elif 'minutes ago' in date_text or 'minute ago' in date_text:
                                    minutes = int(re.search(r'(\d+)', date_text).group(1))
                                    pub_date = datetime.now() - timedelta(minutes=minutes)
                                elif date_text and date_text != '':
                                    try:
                                        # Спроба розпізнати дату у різних форматах
                                        pub_date = datetime.strptime(date_text, '%B %d, %Y')
                                    except:
                                        self.logger.warning(f"Не вдалося розпізнати дату '{date_text}'")
                                        pub_date = datetime.now()  # За замовчуванням використовуємо поточну дату
                                else:
                                    pub_date = datetime.now()

                                # Перевірка, чи стаття відповідає періоду
                                if pub_date and pub_date < start_date:
                                    continue

                                # Отримання короткого опису
                                summary_elem = article.select_one('p.cardV2__description')
                                summary = summary_elem.text.strip() if summary_elem else ""

                                # Додавання даних до списку
                                news_data.append({
                                    'title': title,
                                    'summary': summary,
                                    'link': link,
                                    'source': 'decrypt',
                                    'category': category,
                                    'published_at': pub_date,
                                    'scraped_at': datetime.now()
                                })
                            except Exception as e:
                                self.logger.error(f"Помилка при обробці статті Decrypt: {e}")

                    else:
                        self.logger.error(
                            f"Не вдалося завантажити сторінку {url}. Код відповіді: {response.status_code}")
                        continue_scraping = False

                    # Перехід на наступну сторінку
                    page += 1

                    # Затримка для запобігання блокуванню
                    time.sleep(randint(1, 3))

        except Exception as e:
            self.logger.error(f"Загальна помилка при скрапінгу Decrypt: {e}")

        self.logger.info(f"Зібрано {len(news_data)} новин з Decrypt")
        return news_data

    def scrape_cryptoslate(self, days_back: int = 1,
                           categories: List[str] = None) -> List[Dict]:

        self.logger.info("Збір новин з CryptoSlate...")
        news_data = []

        try:
            # Визначення дати, починаючи з якої збираємо новини
            start_date = datetime.now() - timedelta(days=days_back)

            # Базова URL для CryptoSlate
            base_url = "https://cryptoslate.com"

            # Список категорій, якщо не вказано
            if not categories:
                categories = ["news", "bitcoin", "ethereum", "defi"]

            for category in categories:
                page = 1
                continue_scraping = True

                while continue_scraping and page <= 5:  # Обмеження по кількості сторінок
                    url = f"{base_url}/{category}/page/{page}/"
                    response = requests.get(url, headers=self.headers)

                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        articles = soup.select('article.post-card')

                        if not articles:
                            continue_scraping = False
                            continue

                        for article in articles:
                            try:
                                # Отримання заголовка і посилання
                                title_elem = article.select_one('h3.post-card__title')
                                if not title_elem:
                                    continue

                                title = title_elem.text.strip()
                                link_elem = article.select_one('a.post-card__link')
                                link = link_elem['href'] if link_elem else None

                                # Повний URL, якщо він відносний
                                if link and not link.startswith('http'):
                                    link = base_url + link

                                # Отримання дати публікації
                                date_elem = article.select_one('time.post-card__date')
                                if not date_elem or not date_elem.get('datetime'):
                                    continue

                                date_str = date_elem['datetime']
                                pub_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))

                                # Перевірка, чи стаття відповідає періоду
                                if pub_date < start_date:
                                    continue

                                # Отримання короткого опису
                                summary_elem = article.select_one('p.post-card__excerpt')
                                summary = summary_elem.text.strip() if summary_elem else ""

                                # Додавання даних до списку
                                news_data.append({
                                    'title': title,
                                    'summary': summary,
                                    'link': link,
                                    'source': 'cryptoslate',
                                    'category': category,
                                    'published_at': pub_date,
                                    'scraped_at': datetime.now()
                                })
                            except Exception as e:
                                self.logger.error(f"Помилка при обробці статті CryptoSlate: {e}")

                    else:
                        self.logger.error(
                            f"Не вдалося завантажити сторінку {url}. Код відповіді: {response.status_code}")
                        continue_scraping = False

                    # Перехід на наступну сторінку
                    page += 1

                    # Затримка для запобігання блокуванню
                    time.sleep(randint(1, 3))

        except Exception as e:
            self.logger.error(f"Загальна помилка при скрапінгу CryptoSlate: {e}")

        self.logger.info(f"Зібрано {len(news_data)} новин з CryptoSlate")
        return news_data

    def scrape_theblock(self, days_back: int = 1,
                        categories: List[str] = None) -> List[Dict]:

        self.logger.info("Збір новин з The Block...")
        news_data = []

        try:
            # Визначення дати, починаючи з якої збираємо новини
            start_date = datetime.now() - timedelta(days=days_back)

            # Базова URL для The Block
            base_url = "https://www.theblock.co"

            # Список категорій, якщо не вказано
            if not categories:
                categories = ["latest", "policy", "business", "markets"]

            for category in categories:
                page = 1
                continue_scraping = True

                while continue_scraping and page <= 5:  # Обмеження по кількості сторінок
                    url = f"{base_url}/{category}?page={page}"
                    response = requests.get(url, headers=self.headers)

                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        articles = soup.select('div.post-card')

                        if not articles:
                            continue_scraping = False
                            continue

                        for article in articles:
                            try:
                                # Отримання заголовка і посилання
                                title_elem = article.select_one('h2.post-card__headline')
                                if not title_elem:
                                    continue

                                title = title_elem.text.strip()
                                link_elem = article.select_one('a.post-card__inner')
                                link = link_elem['href'] if link_elem else None

                                # Повний URL, якщо він відносний
                                if link and not link.startswith('http'):
                                    link = base_url + link

                                # Отримання дати публікації
                                date_elem = article.select_one('time.post-card__timestamp')
                                if not date_elem:
                                    continue

                                date_text = date_elem.text.strip()

                                # Спроба перетворити текст дати у об'єкт datetime
                                pub_date = None

                                if 'hours ago' in date_text or 'hour ago' in date_text:
                                    hours = int(re.search(r'(\d+)', date_text).group(1))
                                    pub_date = datetime.now() - timedelta(hours=hours)
                                elif 'days ago' in date_text or 'day ago' in date_text:
                                    days = int(re.search(r'(\d+)', date_text).group(1))
                                    pub_date = datetime.now() - timedelta(days=days)
                                elif 'minutes ago' in date_text or 'minute ago' in date_text:
                                    minutes = int(re.search(r'(\d+)', date_text).group(1))
                                    pub_date = datetime.now() - timedelta(minutes=minutes)
                                else:
                                    try:
                                        # Спроба розпізнати дату у різних форматах
                                        pub_date = datetime.strptime(date_text, '%B %d, %Y')
                                    except:
                                        pub_date = datetime.now()  # За замовчуванням використовуємо поточну дату

                                # Перевірка, чи стаття відповідає періоду
                                if pub_date and pub_date < start_date:
                                    continue

                                # Отримання короткого опису
                                summary_elem = article.select_one('p.post-card__description')
                                summary = summary_elem.text.strip() if summary_elem else ""

                                # Додавання даних до списку
                                news_data.append({
                                    'title': title,
                                    'summary': summary,
                                    'link': link,
                                    'source': 'theblock',
                                    'category': category,
                                    'published_at': pub_date,
                                    'scraped_at': datetime.now()
                                })
                            except Exception as e:
                                self.logger.error(f"Помилка при обробці статті The Block: {e}")

                    else:
                        self.logger.error(
                            f"Не вдалося завантажити сторінку {url}. Код відповіді: {response.status_code}")
                        continue_scraping = False

                    # Перехід на наступну сторінку
                    page += 1

                    # Затримка для запобігання блокуванню
                    time.sleep(randint(1, 3))

        except Exception as e:
            self.logger.error(f"Загальна помилка при скрапінгу The Block: {e}")

        self.logger.info(f"Зібрано {len(news_data)} новин з The Block")
        return news_data

    def scrape_all_sources(self, days_back: int = 1,
                           categories: List[str] = None) -> List[Dict]:

        self.logger.info(f"Початок збору новин з усіх доступних джерел за останні {days_back} днів")

        all_news = []

        # Словник з функціями для кожного джерела
        source_functions = {
            'coindesk': self.scrape_coindesk,
            'cointelegraph': self.scrape_cointelegraph,
            'decrypt': self.scrape_decrypt,
            'cryptoslate': self.scrape_cryptoslate,
            'theblock': self.scrape_theblock
        }

        # Визначення, які джерела будуть використовуватися
        sources_to_scrape = [source for source in self.news_sources if source in source_functions]

        for source in sources_to_scrape:
            try:
                self.logger.info(f"Збір новин з джерела {source}")

                # Встановлення категорій для кожного джерела
                source_categories = None
                if categories:
                    # Різні джерела можуть мати різні імена категорій,
                    # тому можна додати спеціальні категорії для кожного джерела
                    source_categories = categories

                # Виклик відповідної функції для джерела
                news_from_source = source_functions[source](days_back=days_back, categories=source_categories)

                if news_from_source:
                    self.logger.info(f"Успішно зібрано {len(news_from_source)} новин з {source}")
                    all_news.extend(news_from_source)
                else:
                    self.logger.warning(f"Не вдалося зібрати новини з {source}")

                # Затримка між запитами до різних джерел
                time.sleep(randint(2, 5))

            except Exception as e:
                self.logger.error(f"Помилка при зборі новин з {source}: {e}")

        # Видалення дублікатів (якщо є)
        unique_news = []
        seen_titles = set()

        for news in all_news:
            if news['title'] not in seen_titles:
                seen_titles.add(news['title'])
                unique_news.append(news)

        self.logger.info(f"Всього зібрано {len(unique_news)} унікальних новин з {len(sources_to_scrape)} джерел")
        return unique_news

    def scrape_reddit(self, days_back: int = 1, subreddits: List[str] = None) -> List[Dict]:

        self.logger.info("Збір новин з Reddit...")

        if not self.reddit:
            self.logger.error("Reddit API не ініціалізовано")
            return []

        news_data = []

        # Список сабреддітів за замовчуванням
        if not subreddits:
            subreddits = ['CryptoCurrency', 'Bitcoin', 'ethereum', 'CryptoMarkets']

        start_date = datetime.now() - timedelta(days=days_back)

        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)

                # Отримання популярних постів
                for post in subreddit.hot(limit=50):
                    try:
                        # Перетворення timestamp у datetime
                        post_date = datetime.fromtimestamp(post.created_utc)

                        # Перевірка дати
                        if post_date < start_date:
                            continue

                        # Додавання в список новин
                        news_data.append({
                            'title': post.title,
                            'summary': post.selftext[:500] if post.selftext else "",
                            'link': f"https://www.reddit.com{post.permalink}",
                            'source': 'reddit',
                            'category': subreddit_name,
                            'published_at': post_date,
                            'scraped_at': datetime.now(),
                            'score': post.score,
                            'upvote_ratio': post.upvote_ratio,
                            'num_comments': post.num_comments
                        })
                    except Exception as e:
                        self.logger.error(f"Помилка при обробці поста Reddit: {e}")

                # Затримка між запитами до різних сабреддітів
                time.sleep(randint(1, 3))

            except Exception as e:
                self.logger.error(f"Помилка при скрапінгу сабреддіта {subreddit_name}: {e}")

        self.logger.info(f"Зібрано {len(news_data)} новин з Reddit")
        return news_data

    def analyze_news_sentiment(self, news_data: List[Dict]) -> List[Dict]:

        self.logger.info(f"Початок аналізу настроїв для {len(news_data)} новин")

        # Перевірка наявності аналізатора настроїв
        if not self.sentiment_analyzer:
            self.logger.error("Аналізатор настроїв не ініціалізований")
            # Додаємо нейтральний настрій за замовчуванням
            for news in news_data:
                news['sentiment'] = {
                    'score': 0.0,  # Нейтральний настрій
                    'label': 'neutral',
                    'confidence': 0.0,
                    'analyzed': False
                }
            return news_data

        analyzed_news = []

        for idx, news in enumerate(news_data):
            try:
                # Текст для аналізу (комбінуємо заголовок та опис)
                text_to_analyze = f"{news['title']} {news.get('summary', '')}"

                # Викликаємо аналізатор настроїв
                sentiment_result = self.sentiment_analyzer.analyze(text_to_analyze)

                # Копіюємо новину та додаємо результат аналізу
                news_with_sentiment = news.copy()

                # Форматуємо результат аналізу
                if isinstance(sentiment_result, dict):
                    news_with_sentiment['sentiment'] = sentiment_result
                else:
                    # Якщо результат не у вигляді словника, створюємо базову структуру
                    news_with_sentiment['sentiment'] = {
                        'score': getattr(sentiment_result, 'score', 0.0),
                        'label': getattr(sentiment_result, 'label', 'neutral'),
                        'confidence': getattr(sentiment_result, 'confidence', 0.0),
                        'analyzed': True
                    }

                analyzed_news.append(news_with_sentiment)

                # Логування прогресу (кожні 50 новин)
                if idx > 0 and idx % 50 == 0:
                    self.logger.info(f"Проаналізовано {idx}/{len(news_data)} новин")

            except Exception as e:
                self.logger.error(f"Помилка при аналізі настроїв для новини '{news.get('title', 'unknown')}': {e}")
                # Додаємо новину з нейтральним настроєм у випадку помилки
                news['sentiment'] = {
                    'score': 0.0,
                    'label': 'neutral',
                    'confidence': 0.0,
                    'analyzed': False,
                    'error': str(e)
                }
                analyzed_news.append(news)

        self.logger.info(f"Аналіз настроїв завершено для {len(analyzed_news)} новин")
        return analyzed_news

    def extract_mentioned_coins(self, news_data: List[Dict]) -> List[Dict]:

        self.logger.info(f"Початок пошуку згаданих криптовалют у {len(news_data)} новинах")

        # Словник з популярними криптовалютами та їх скороченнями/синонімами
        crypto_keywords = {
            'bitcoin': ['btc', 'xbt', 'bitcoin', 'биткоин', 'біткоїн'],
            'ethereum': ['eth', 'ethereum', 'эфириум', 'етеріум', 'ether'],
            'ripple': ['xrp', 'ripple'],
            'litecoin': ['ltc', 'litecoin'],
            'cardano': ['ada', 'cardano'],
            'polkadot': ['dot', 'polkadot'],
            'binance coin': ['bnb', 'binance coin', 'binance'],
            'dogecoin': ['doge', 'dogecoin'],
            'solana': ['sol', 'solana'],
            'tron': ['trx', 'tron'],
            'tether': ['usdt', 'tether'],
            'usd coin': ['usdc', 'usd coin'],
            'avalanche': ['avax', 'avalanche'],
            'chainlink': ['link', 'chainlink'],
            'polygon': ['matic', 'polygon'],
            'stellar': ['xlm', 'stellar'],
            'cosmos': ['atom', 'cosmos'],
            'vechain': ['vet', 'vechain'],
            'algorand': ['algo', 'algorand'],
            'uniswap': ['uni', 'uniswap'],
            'shiba inu': ['shib', 'shiba inu', 'shiba'],
            'filecoin': ['fil', 'filecoin'],
            'monero': ['xmr', 'monero'],
            'aave': ['aave'],
            'maker': ['mkr', 'maker'],
            'compound': ['comp', 'compound'],
            'decentraland': ['mana', 'decentraland']
        }

        # Компілюємо регулярні вирази для ефективного пошуку
        coin_patterns = {}
        for coin, aliases in crypto_keywords.items():
            # Створюємо шаблон регулярного виразу для кожної монети та її аліасів
            # \b забезпечує пошук цілих слів, а (?i) - нечутливість до регістру
            pattern = r'\b(?i)(' + '|'.join(aliases) + r')\b'
            coin_patterns[coin] = re.compile(pattern)

        for news in news_data:
            try:
                # Текст для аналізу (комбінуємо заголовок та опис)
                text_to_analyze = f"{news['title']} {news.get('summary', '')}"

                # Ініціалізуємо словник для згаданих монет та їх кількості
                mentioned_coins = {}

                # Пошук кожної монети в тексті
                for coin, pattern in coin_patterns.items():
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
                news['mentioned_coins'] = {
                    'coins': {coin: count for coin, count in sorted_mentions},
                    'top_mentioned': sorted_mentions[0][0] if sorted_mentions else None,
                    'total_coins': len(sorted_mentions)
                }

            except Exception as e:
                self.logger.error(
                    f"Помилка при пошуку згаданих криптовалют для новини '{news.get('title', 'unknown')}': {e}")
                # Додаємо порожнє поле у випадку помилки
                news['mentioned_coins'] = {
                    'coins': {},
                    'top_mentioned': None,
                    'total_coins': 0,
                    'error': str(e)
                }

        self.logger.info("Пошук згаданих криптовалют завершено")
        return news_data

    def filter_by_keywords(self, news_data: List[Dict],
                           keywords: List[str]) -> List[Dict]:

        self.logger.info(f"Початок фільтрації {len(news_data)} новин за {len(keywords)} ключовими словами")

        if not keywords or not news_data:
            self.logger.warning("Порожній список ключових слів або новин для фільтрації")
            return news_data

        # Підготовка регулярних виразів для пошуку (нечутливість до регістру)
        keyword_patterns = []
        for keyword in keywords:
            # Екрануємо спеціальні символи у ключових словах
            escaped_keyword = re.escape(keyword)
            # Створюємо шаблон для пошуку цілого слова
            pattern = re.compile(rf'\b{escaped_keyword}\b', re.IGNORECASE)
            keyword_patterns.append(pattern)

        filtered_news = []

        for news in news_data:
            try:
                # Текст для аналізу (комбінуємо заголовок та опис)
                text_to_analyze = f"{news['title']} {news.get('summary', '')}"

                # Перевірка на наявність хоча б одного ключового слова
                matches_found = False
                matched_keywords = []

                for i, pattern in enumerate(keyword_patterns):
                    if pattern.search(text_to_analyze):
                        matches_found = True
                        matched_keywords.append(keywords[i])

                if matches_found:
                    # Копіюємо новину та додаємо інформацію про знайдені ключові слова
                    matched_news = news.copy()
                    matched_news['matched_keywords'] = matched_keywords
                    filtered_news.append(matched_news)

            except Exception as e:
                self.logger.error(f"Помилка при фільтрації новини '{news.get('title', 'unknown')}': {e}")

        self.logger.info(f"Відфільтровано {len(filtered_news)} новин з {len(news_data)} за ключовими словами")
        return filtered_news

    def detect_major_events(self, news_data: List[Dict]) -> List[Dict]:

        self.logger.info(f"Початок аналізу {len(news_data)} новин для виявлення важливих подій")

        # Ключові слова, що вказують на потенційно важливі події
        critical_keywords = {
            'regulation': ['regulation', 'регуляція', 'закон', 'заборона', 'легалізація', 'SEC', 'CFTC'],
            'hack': ['hack', 'хакер', 'зламали', 'атака', 'викрадено', 'вкрадено', 'безпека'],
            'market_crash': ['crash', 'collapse', 'обвал', 'крах', 'падіння', 'bear market', 'ведмежий'],
            'market_boom': ['boom', 'rally', 'ріст', 'буйк', 'bull market', 'бичачий', 'ath', 'all-time high'],
            'merge': ['merge', 'злиття', 'acquisition', 'поглинання', 'buyout', 'викуп'],
            'fork': ['fork', 'форк', 'hard fork', 'soft fork', 'chain split', 'розділення'],
            'adoption': ['adoption', 'впровадження', 'integration', 'інтеграція', 'partnership', 'партнерство'],
            'scandal': ['scandal', 'скандал', 'controversy', 'контроверсія', 'fraud', 'шахрайство'],
            'lawsuit': ['lawsuit', 'позов', 'court', 'суд', 'legal action', 'legal', 'investigation'],
            'innovation': ['innovation', 'інновація', 'breakthrough', 'прорив', 'launch', 'запуск']
        }

        # Створення шаблонів регулярних виразів для кожної категорії
        category_patterns = {}
        for category, keywords in critical_keywords.items():
            patterns = [re.compile(rf'\b{re.escape(keyword)}\b', re.IGNORECASE) for keyword in keywords]
            category_patterns[category] = patterns

        major_events = []

        # Аналіз новин
        for news in news_data:
            try:
                # Текст для аналізу (комбінуємо заголовок та опис)
                text_to_analyze = f"{news['title']} {news.get('summary', '')}"

                # Перевірка по категоріях
                event_categories = set()
                matched_keywords = {}

                for category, patterns in category_patterns.items():
                    for pattern in patterns:
                        if pattern.search(text_to_analyze):
                            event_categories.add(category)

                            # Збереження ключових слів, що співпали
                            if category not in matched_keywords:
                                matched_keywords[category] = []
                            keyword = pattern.pattern.replace(r'\b', '')
                            matched_keywords[category].append(re.escape(keyword))

                # Визначення важливості події
                importance_level = len(event_categories)

                # Додаткові фактори для визначення важливості:
                # 1. Перевірка наявності назв великих компаній/проектів
                major_entities = ['bitcoin', 'ethereum', 'binance', 'coinbase', 'ripple', 'tether',
                                  'ftx', 'metamask', 'opensea', 'uniswap', 'solana', 'avalanche']

                entity_matches = []
                for entity in major_entities:
                    if re.search(rf'\b{re.escape(entity)}\b', text_to_analyze, re.IGNORECASE):
                        entity_matches.append(entity)
                        importance_level += 0.5  # Додаємо ваги до важливості

                # 2. Перевірка наявності цифр (суми грошей, відсотки тощо)
                if re.search(r'\$\d+(?:[,.]\d+)?(?:\s*(?:million|billion|m|b|млн|млрд))?|\d+%', text_to_analyze,
                             re.IGNORECASE):
                    importance_level += 1  # Наявність фінансових даних підвищує важливість

                # Якщо знайдено хоча б одну категорію або важливість висока - це важлива подія
                if event_categories or importance_level >= 2:
                    event_data = {
                        'title': news.get('title', ''),
                        'summary': news.get('summary', ''),
                        'source': news.get('source', ''),
                        'link': news.get('link', ''),
                        'published_at': news.get('published_at', datetime.now()),
                        'categories': list(event_categories),
                        'matched_keywords': matched_keywords,
                        'major_entities': entity_matches,
                        'importance_level': importance_level,
                        'original_news': news
                    }
                    major_events.append(event_data)

            except Exception as e:
                self.logger.error(f"Помилка при аналізі новини '{news.get('title', 'unknown')}' на важливі події: {e}")

        # Сортування за важливістю (в порядку спадання)
        major_events.sort(key=lambda x: x['importance_level'], reverse=True)

        self.logger.info(f"Виявлено {len(major_events)} важливих подій")
        return major_events

    def save_to_database(self, news_data: List[Dict], db_manager) -> bool:
        self.logger.info(f"Початок збереження {len(news_data)} новин у базу даних")

        if not news_data:
            self.logger.warning("Порожній список новин для збереження")
            return False

        success_count = 0

        for news in news_data:
            try:
                # Перевірка на існування статті з таким посиланням
                if db_manager.article_exists_by_link(news.get('link', '')):
                    self.logger.info(f"Стаття з посиланням {news.get('link', '')[:50]} вже існує в БД")
                    continue

                # Отримання source_id та category_id з БД або створення нових записів
                source_id = db_manager.get_or_create_source(news.get('source', ''))
                category_id = db_manager.get_or_create_category(news.get('category', ''))

                # Підготовка даних для статті
                article_data = {
                    'title': news.get('title', ''),
                    'summary': news.get('summary', ''),
                    'content': news.get('content', ''),
                    'link': news.get('link', ''),
                    'source_id': source_id,
                    'category_id': category_id,
                    'published_at': news.get('published_at', datetime.now())
                }

                # Вставка новинної статті
                article_id = db_manager.insert_news_article(article_data)

                if not article_id:
                    self.logger.warning(f"Не вдалося додати статтю: {article_data['title'][:50]}")
                    continue

                # Якщо є дані про настрій — вставити
                if 'sentiment' in news and isinstance(news['sentiment'], dict):
                    sentiment_data = {
                        'article_id': article_id,
                        'sentiment_score': news['sentiment'].get('score', 0.0),
                        'sentiment_magnitude': news['sentiment'].get('confidence', 0.0),
                        'sentiment_label': news['sentiment'].get('label', 'neutral')
                    }
                    db_manager.insert_news_sentiment(sentiment_data)

                # Якщо є дані про згадані монети — вставити
                if 'mentioned_coins' in news and isinstance(news['mentioned_coins'], dict):
                    for crypto_symbol, mention_count in news['mentioned_coins'].get('coins', {}).items():
                        db_manager.insert_article_mention(article_id, crypto_symbol, mention_count)

                success_count += 1

            except Exception as e:
                self.logger.error(f"Помилка при збереженні новини '{news.get('title', 'unknown')}': {e}")
                continue

        self.logger.info(f"Успішно збережено {success_count} новин із {len(news_data)}")
        return success_count > 0

    def get_trending_topics(self, news_data: List[Dict], top_n: int = 10) -> List[Dict]:

        self.logger.info(f"Аналіз трендів серед {len(news_data)} новин")

        # Словник для підрахунку частоти ключових слів
        word_frequency = {}

        # Слова, які часто зустрічаються і не несуть специфічного значення
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                        'to', 'for', 'with', 'by', 'about', 'as', 'of', 'from',
                        'that', 'this', 'these', 'those', 'is', 'are', 'was', 'were',
                        'has', 'have', 'had', 'been', 'will', 'would', 'could', 'should'}

        for news in news_data:
            # Об'єднуємо заголовок і короткий опис
            text = f"{news.get('title', '')} {news.get('summary', '')}"

            # Нормалізація тексту: нижній регістр і видалення пунктуації
            text = re.sub(r'[^\w\s]', '', text.lower())

            # Розбиття на слова
            words = text.split()

            # Підрахунок частоти слів (крім поширених)
            for word in words:
                if len(word) > 3 and word not in common_words:
                    word_frequency[word] = word_frequency.get(word, 0) + 1

        # Сортування за частотою
        sorted_words = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)

        # Формування результату
        trends = []
        for word, frequency in sorted_words[:top_n]:
            trends.append({
                'topic': word,
                'frequency': frequency,
                'weight': frequency / len(news_data)
            })

        self.logger.info(f"Знайдено {len(trends)} трендових тем")
        return trends

    def correlate_with_market(self, news_data: List[Dict], market_data: pd.DataFrame) -> Dict:

        self.logger.info("Початок аналізу кореляції новин з ринком")

        if not news_data or market_data.empty:
            self.logger.warning("Недостатньо даних для аналізу кореляції")
            return {'correlation': 0, 'significance': 0, 'valid': False}

        try:
            # Створюємо DataFrame з даними настроїв по датам
            sentiment_data = []

            for news in news_data:
                if 'sentiment' in news and 'published_at' in news:
                    date = news['published_at'].date()
                    score = news['sentiment'].get('score', 0)
                    sentiment_data.append({
                        'date': date,
                        'sentiment_score': score
                    })

            if not sentiment_data:
                self.logger.warning("Відсутні дані про настрої для аналізу")
                return {'correlation': 0, 'significance': 0, 'valid': False}

            sentiment_df = pd.DataFrame(sentiment_data)

            # Агрегація за датою (середній настрій за день)
            daily_sentiment = sentiment_df.groupby('date')['sentiment_score'].mean().reset_index()

            # Підготовка ринкових даних
            market_df = market_data.copy()
            if 'date' not in market_df.columns:
                market_df['date'] = pd.to_datetime(market_df.index).date

            # Злиття даних по даті
            merged_data = pd.merge(daily_sentiment, market_df, on='date', how='inner')

            if len(merged_data) < 3:  # Мінімум для розрахунку кореляції
                self.logger.warning("Недостатньо даних для розрахунку кореляції")
                return {'correlation': 0, 'significance': 0, 'valid': False}

            # Розрахунок кореляції Пірсона з ціною
            price_column = next((col for col in merged_data.columns if 'price' in col.lower()), 'close')
            correlation = merged_data['sentiment_score'].corr(merged_data[price_column])

            # Розрахунок p-value для визначення статистичної значущості
            from scipy import stats
            correlation_significance = stats.pearsonr(
                merged_data['sentiment_score'],
                merged_data[price_column]
            )[1]  # p-value

            result = {
                'correlation': correlation,
                'significance': correlation_significance,
                'sample_size': len(merged_data),
                'valid': True,
                'period_start': merged_data['date'].min(),
                'period_end': merged_data['date'].max()
            }

            self.logger.info(f"Розрахована кореляція: {correlation:.4f} (p={correlation_significance:.4f})")
            return result

        except Exception as e:
            self.logger.error(f"Помилка при аналізі кореляції: {e}")
            return {'correlation': 0, 'significance': 0, 'valid': False, 'error': str(e)}
"""