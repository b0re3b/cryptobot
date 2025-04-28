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