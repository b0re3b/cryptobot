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


# Збір новин з криптовалютних ресурсів
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
        """
        Збір новин з усіх доступних джерел.

        Args:
            days_back: Кількість днів для збору новин назад
            categories: Категорії новин для фільтрації

        Returns:
            Об'єднаний список словників з даними новин
        """
        pass

    def analyze_news_sentiment(self, news_data: List[Dict]) -> List[Dict]:
        """
        Аналіз настроїв у зібраних новинах.

        Args:
            news_data: Список новин для аналізу

        Returns:
            Список новин із доданим полем sentiment
        """
        pass

    def extract_mentioned_coins(self, news_data: List[Dict]) -> List[Dict]:
        """
        Витягнення згаданих криптовалют із новин.

        Args:
            news_data: Список новин для аналізу

        Returns:
            Список новин із доданим полем mentioned_coins
        """
        pass

    def filter_by_keywords(self, news_data: List[Dict],
                           keywords: List[str]) -> List[Dict]:
        """
        Фільтрація новин за ключовими словами.

        Args:
            news_data: Список новин для фільтрації
            keywords: Список ключових слів для пошуку

        Returns:
            Відфільтрований список новин
        """
        pass

    def detect_major_events(self, news_data: List[Dict]) -> List[Dict]:
        """
        Виявлення важливих подій, які можуть вплинути на ринок.

        Args:
            news_data: Список новин для аналізу

        Returns:
            Список виявлених важливих подій
        """
        pass

    def save_to_database(self, news_data: List[Dict],
                         db_connection) -> bool:
        """
        Збереження зібраних новин у базу даних.

        Args:
            news_data: Список новин для збереження
            db_connection: З'єднання з базою даних

        Returns:
            Булеве значення успішності операції
        """
        pass

    def save_to_csv(self, news_data: List[Dict],
                    filename: str) -> bool:
        """
        Збереження зібраних новин у CSV файл.

        Args:
            news_data: Список новин для збереження
            filename: Ім'я файлу для збереження

        Returns:
            Булеве значення успішності операції
        """
        pass

    def get_trending_topics(self, news_data: List[Dict],
                            top_n: int = 10) -> List[Dict]:
        """
        Отримання трендових тем з новин.

        Args:
            news_data: Список новин для аналізу
            top_n: Кількість тем для виведення

        Returns:
            Список трендових тем та їх важливість
        """
        pass

    def correlate_with_market(self, news_data: List[Dict],
                              market_data: pd.DataFrame) -> Dict:
        """
        Кореляція новин з рухами ринку.

        Args:
            news_data: Список новин з аналізом настроїв
            market_data: DataFrame з ціновими даними ринку

        Returns:
            Дані про кореляцію та статистичну значущість
        """
        pass