import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import logging
import re
import time
import praw
import concurrent.futures
from typing import List, Dict, Optional, Union, Tuple, Any, Callable
from random import randint
from dataclasses import dataclass
from data.db import DatabaseManager
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
import os


@dataclass
class NewsItem:
    title: str
    summary: str
    link: str
    source: str
    category: str
    published_at: datetime
    scraped_at: datetime
    score: Optional[int] = None
    upvote_ratio: Optional[float] = None
    num_comments: Optional[int] = None
    sentiment_score: Optional[float] = None
    topics: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'summary': self.summary,
            'link': self.link,
            'source': self.source,
            'category': self.category,
            'published_at': self.published_at,
            'scraped_at': self.scraped_at,
            'score': self.score,
            'upvote_ratio': self.upvote_ratio,
            'num_comments': self.num_comments,
            'sentiment_score': self.sentiment_score,
            'topics': self.topics
        }


class NewsCollector:

    def __init__(self,
                 news_sources: List[str] = None,
                 sentiment_analyzer=None,
                 logger: Optional[logging.Logger] = None,
                 db_manager: Optional[DatabaseManager] = None,
                 max_pages: int = 5,
                 max_workers: int = 5,
                 topic_model_dir: str = './models'):

        self.news_sources = news_sources or [
            'coindesk', 'cointelegraph', 'decrypt', 'cryptoslate',
            'theblock', 'cryptopanic', 'coinmarketcal', 'feedly',
            'newsnow', 'reddit'
        ]
        self.sentiment_analyzer = sentiment_analyzer
        self.db_manager = db_manager
        self.max_pages = max_pages
        self.max_workers = max_workers
        self.topic_model_dir = topic_model_dir

        # Configure logger
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger('crypto_news_scraper')
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Headers for HTTP requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Reddit API client
        self.reddit = None

        # Source configuration
        self.source_config = {
            'coindesk': {
                'base_url': 'https://www.coindesk.com',
                'default_categories': ["markets", "business", "policy", "tech"]
            },
            'cointelegraph': {
                'base_url': 'https://cointelegraph.com',
                'default_categories': ["news", "markets", "features", "analysis"]
            },
            'decrypt': {
                'base_url': 'https://decrypt.co',
                'default_categories': ["news", "analysis", "features", "learn"]
            },
            'cryptoslate': {
                'base_url': 'https://cryptoslate.com',
                'default_categories': ["news", "bitcoin", "ethereum", "defi"]
            },
            'theblock': {
                'base_url': 'https://www.theblock.co',
                'default_categories': ["latest", "policy", "business", "markets"]
            },
            'cryptopanic': {
                'base_url': 'https://cryptopanic.com',
                'default_categories': ["news", "recent", "rising", "hot"]
            },
            'coinmarketcal': {
                'base_url': 'https://coinmarketcal.com',
                'default_categories': ["events", "upcoming", "ongoing", "recent"]
            },
            'feedly': {
                'base_url': 'https://feedly.com',
                'default_categories': ["crypto", "blockchain", "bitcoin", "ethereum"]
            },
            'newsnow': {
                'base_url': 'https://www.newsnow.co.uk',
                'default_categories': ["crypto", "cryptocurrency", "bitcoin", "ethereum"]
            },
            'reddit': {
                'default_subreddits': ['CryptoCurrency', 'Bitcoin', 'ethereum', 'CryptoMarkets']
            }
        }

        # Initialize caches
        self._cache = {}

        # Initialize topic modeling components
        self.vectorizer = None
        self.lda_model = None
        self.nmf_model = None
        self.kmeans_model = None
        self.topic_words = {}

        # Load topic models if they exist
        self._load_topic_models()

        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')

    def _make_request(self, url: str, retries: int = 3, backoff_factor: float = 0.3) -> Optional[requests.Response]:

        for i in range(retries):
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                if response.status_code == 200:
                    return response
                elif response.status_code in [403, 429]:
                    self.logger.warning(f"Received {response.status_code} status code from {url}. Waiting...")
                    time.sleep((backoff_factor * (2 ** i)) + randint(1, 3))
                else:
                    self.logger.error(f"Error requesting {url}: HTTP {response.status_code}")
                    return None
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Connection error requesting {url}: {e}")
                time.sleep((backoff_factor * (2 ** i)) + randint(1, 3))

        return None

    def initialize_reddit(self, client_id: str, client_secret: str, user_agent: str) -> bool:

        try:
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
            return True
        except Exception as e:
            self.logger.error(f"Error initializing Reddit API: {e}")
            return False

    def _parse_relative_date(self, date_text: str) -> Optional[datetime]:

        try:
            if 'hours ago' in date_text or 'hour ago' in date_text:
                hours = int(re.search(r'(\d+)', date_text).group(1))
                return datetime.now() - timedelta(hours=hours)
            elif 'days ago' in date_text or 'day ago' in date_text:
                days = int(re.search(r'(\d+)', date_text).group(1))
                return datetime.now() - timedelta(days=days)
            elif 'minutes ago' in date_text or 'minute ago' in date_text:
                minutes = int(re.search(r'(\d+)', date_text).group(1))
                return datetime.now() - timedelta(minutes=minutes)
            elif 'weeks ago' in date_text or 'week ago' in date_text:
                weeks = int(re.search(r'(\d+)', date_text).group(1))
                return datetime.now() - timedelta(weeks=weeks)
            elif 'months ago' in date_text or 'month ago' in date_text:
                months = int(re.search(r'(\d+)', date_text).group(1))
                # Approximate month as 30 days
                return datetime.now() - timedelta(days=30 * months)
            elif 'yesterday' in date_text.lower():
                return datetime.now() - timedelta(days=1)
            elif 'today' in date_text.lower():
                return datetime.now()
            else:
                # Try various date formats
                for fmt in [
                    '%B %d, %Y', '%Y-%m-%d', '%d %B %Y', '%d/%m/%Y', '%m/%d/%Y',
                    '%b %d, %Y', '%d %b %Y', '%Y/%m/%d', '%d-%m-%Y', '%m-%d-%Y',
                    '%B %d %Y', '%d %B %Y', '%b %d %Y', '%d %b %Y',
                    # Add time formats
                    '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S'
                ]:
                    try:
                        return datetime.strptime(date_text, fmt)
                    except ValueError:
                        continue

                # If all parsing attempts fail, log a warning and return current time
                self.logger.warning(f"Could not parse date string: '{date_text}'")
                return datetime.now()
        except Exception as e:
            self.logger.error(f"Error parsing date string '{date_text}': {e}")
            return datetime.now()

    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of the given text.
        Returns a float value between -1 (negative) and 1 (positive).
        """
        if self.sentiment_analyzer:
            try:
                return self.sentiment_analyzer.analyze(text)
            except Exception as e:
                self.logger.error(f"Error analyzing sentiment: {e}")
                return 0.0
        else:
            # Return neutral sentiment if no analyzer is provided
            return 0.0

    def _create_news_item(self,
                          title: str,
                          summary: str,
                          link: str,
                          source: str,
                          category: str,
                          published_at: datetime,
                          **kwargs) -> NewsItem:

        # Analyze sentiment if analyzer is available
        sentiment_score = None
        if self.sentiment_analyzer:
            text_to_analyze = f"{title} {summary}"
            sentiment_score = self.analyze_sentiment(text_to_analyze)

        # Get topics if topic models are available
        topics = None
        if self.vectorizer and (self.lda_model or self.nmf_model):
            try:
                topics = self.extract_topics(f"{title} {summary}")
            except Exception as e:
                self.logger.error(f"Error extracting topics: {e}")

        return NewsItem(
            title=title,
            summary=summary,
            link=link,
            source=source,
            category=category,
            published_at=published_at,
            scraped_at=datetime.now(),
            sentiment_score=sentiment_score,
            topics=topics,
            **kwargs
        )

    def scrape_coindesk(self, days_back: int = 1, categories: List[str] = None) -> List[NewsItem]:

        self.logger.info("Scraping news from CoinDesk...")
        news_data = []

        try:
            # Determine start date
            start_date = datetime.now() - timedelta(days=days_back)

            # Get base URL and categories
            base_url = self.source_config['coindesk']['base_url']
            if not categories:
                categories = self.source_config['coindesk']['default_categories']

            for category in categories:
                page = 1
                continue_scraping = True

                while continue_scraping and page <= self.max_pages:
                    url = f"{base_url}/{category}/?page={page}"
                    response = self._make_request(url)

                    if not response:
                        continue_scraping = False
                        continue

                    soup = BeautifulSoup(response.text, 'html.parser')
                    articles = soup.select('article.article-cardstyles__CardWrapper-sc-5xitv1-0')

                    if not articles:
                        continue_scraping = False
                        continue

                    for article in articles:
                        try:
                            # Get publication date
                            date_elem = article.select_one('time')
                            if not date_elem or not date_elem.get('datetime'):
                                continue

                            date_str = date_elem['datetime']
                            pub_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))

                            # Check if article is within the time period
                            if pub_date < start_date:
                                continue_scraping = False
                                break

                            # Get title and link
                            title_elem = article.select_one('h6')
                            if not title_elem:
                                continue

                            title = title_elem.text.strip()
                            link_elem = article.select_one('a')
                            link = base_url + link_elem['href'] if link_elem else None

                            # Get summary
                            summary_elem = article.select_one('p.typography__StyledTypography-sc-owin6q-0')
                            summary = summary_elem.text.strip() if summary_elem else ""

                            # Create news item
                            news_item = self._create_news_item(
                                title=title,
                                summary=summary,
                                link=link,
                                source='coindesk',
                                category=category,
                                published_at=pub_date
                            )

                            news_data.append(news_item)
                        except Exception as e:
                            self.logger.error(f"Error processing CoinDesk article: {e}")

                    # Go to next page
                    page += 1

                    # Delay to prevent blocking
                    time.sleep(randint(1, 3))

        except Exception as e:
            self.logger.error(f"General error while scraping CoinDesk: {e}")

        self.logger.info(f"Collected {len(news_data)} news from CoinDesk")
        return news_data

    def scrape_cointelegraph(self, days_back: int = 1, categories: List[str] = None) -> List[NewsItem]:

        self.logger.info("Scraping news from Cointelegraph...")
        news_data = []

        try:
            # Determine start date
            start_date = datetime.now() - timedelta(days=days_back)

            # Get base URL and categories
            base_url = self.source_config['cointelegraph']['base_url']
            if not categories:
                categories = self.source_config['cointelegraph']['default_categories']

            for category in categories:
                page = 1
                continue_scraping = True

                while continue_scraping and page <= self.max_pages:
                    url = f"{base_url}/{category}?page={page}"
                    response = self._make_request(url)

                    if not response:
                        continue_scraping = False
                        continue

                    soup = BeautifulSoup(response.text, 'html.parser')
                    articles = soup.select('article.post-card-inline')

                    if not articles:
                        continue_scraping = False
                        continue

                    for article in articles:
                        try:
                            # Get publication date
                            date_elem = article.select_one('time.post-card-inline__date')
                            if not date_elem or not date_elem.get('datetime'):
                                continue

                            date_str = date_elem['datetime']
                            pub_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))

                            # Check if article is within the time period
                            if pub_date < start_date:
                                continue_scraping = False
                                break

                            # Get title and link
                            title_elem = article.select_one('span.post-card-inline__title')
                            if not title_elem:
                                continue

                            title = title_elem.text.strip()
                            link_elem = article.select_one('a.post-card-inline__title-link')
                            link = base_url + link_elem['href'] if link_elem else None

                            # Get summary
                            summary_elem = article.select_one('p.post-card-inline__text')
                            summary = summary_elem.text.strip() if summary_elem else ""

                            # Create news item
                            news_item = self._create_news_item(
                                title=title,
                                summary=summary,
                                link=link,
                                source='cointelegraph',
                                category=category,
                                published_at=pub_date
                            )

                            news_data.append(news_item)
                        except Exception as e:
                            self.logger.error(f"Error processing Cointelegraph article: {e}")

                    # Go to next page
                    page += 1

                    # Delay to prevent blocking
                    time.sleep(randint(1, 3))

        except Exception as e:
            self.logger.error(f"General error while scraping Cointelegraph: {e}")

        self.logger.info(f"Collected {len(news_data)} news from Cointelegraph")
        return news_data

    def _scrape_source_with_config(self,
                                   source: str,
                                   days_back: int,
                                   categories: List[str],
                                   article_selector: str,
                                   title_selector: str,
                                   link_selector: str,
                                   date_selector: str,
                                   summary_selector: str,
                                   url_formatter: Callable[[str, str, int], str],
                                   date_attribute: Optional[str] = None) -> List[NewsItem]:

        self.logger.info(f"Scraping news from {source}...")
        news_data = []

        try:
            # Determine start date
            start_date = datetime.now() - timedelta(days=days_back)

            # Get base URL
            base_url = self.source_config[source]['base_url']

            for category in categories:
                page = 1
                continue_scraping = True

                while continue_scraping and page <= self.max_pages:
                    url = url_formatter(base_url, category, page)
                    response = self._make_request(url)

                    if not response:
                        continue_scraping = False
                        continue

                    soup = BeautifulSoup(response.text, 'html.parser')
                    articles = soup.select(article_selector)

                    if not articles:
                        continue_scraping = False
                        continue

                    for article in articles:
                        try:
                            # Get publication date
                            date_elem = article.select_one(date_selector)
                            if not date_elem:
                                continue

                            pub_date = None
                            if date_attribute and date_elem.get(date_attribute):
                                date_str = date_elem[date_attribute]
                                try:
                                    pub_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                                except ValueError:
                                    pub_date = self._parse_relative_date(date_str)
                            else:
                                date_text = date_elem.text.strip()
                                pub_date = self._parse_relative_date(date_text)

                            # Check if article is within the time period
                            if pub_date and pub_date < start_date:
                                continue_scraping = False
                                break

                            # Get title and link
                            title_elem = article.select_one(title_selector)
                            if not title_elem:
                                continue

                            title = title_elem.text.strip()
                            link_elem = article.select_one(link_selector)
                            link = link_elem['href'] if link_elem else None

                            # Handle relative URLs
                            if link and not link.startswith('http'):
                                link = base_url + link

                            # Get summary
                            summary_elem = article.select_one(summary_selector)
                            summary = summary_elem.text.strip() if summary_elem else ""

                            # Create news item
                            news_item = self._create_news_item(
                                title=title,
                                summary=summary,
                                link=link,
                                source=source,
                                category=category,
                                published_at=pub_date
                            )

                            news_data.append(news_item)
                        except Exception as e:
                            self.logger.error(f"Error processing {source} article: {e}")

                    # Go to next page
                    page += 1

                    # Delay to prevent blocking
                    time.sleep(randint(1, 3))

        except Exception as e:
            self.logger.error(f"General error while scraping {source}: {e}")

        self.logger.info(f"Collected {len(news_data)} news from {source}")
        return news_data

    def scrape_decrypt(self, days_back: int = 1, categories: List[str] = None) -> List[NewsItem]:

        if not categories:
            categories = self.source_config['decrypt']['default_categories']

        def url_formatter(base_url, category, page):
            return f"{base_url}/{category}/page/{page}"

        return self._scrape_source_with_config(
            source='decrypt',
            days_back=days_back,
            categories=categories,
            article_selector='article.cardV2',
            title_selector='h3.cardV2__title',
            link_selector='a.cardV2__wrap',
            date_selector='time.cardV2__date',
            summary_selector='p.cardV2__description',
            url_formatter=url_formatter
        )

    def scrape_cryptoslate(self, days_back: int = 1, categories: List[str] = None) -> List[NewsItem]:

        if not categories:
            categories = self.source_config['cryptoslate']['default_categories']

        def url_formatter(base_url, category, page):
            return f"{base_url}/{category}/page/{page}/"

        return self._scrape_source_with_config(
            source='cryptoslate',
            days_back=days_back,
            categories=categories,
            article_selector='article.post-card',
            title_selector='h3.post-card__title',
            link_selector='a.post-card__link',
            date_selector='time.post-card__date',
            summary_selector='p.post-card__excerpt',
            url_formatter=url_formatter,
            date_attribute='datetime'
        )

    def scrape_theblock(self, days_back: int = 1, categories: List[str] = None) -> List[NewsItem]:

        if not categories:
            categories = self.source_config['theblock']['default_categories']

        def url_formatter(base_url, category, page):
            return f"{base_url}/{category}?page={page}"

        return self._scrape_source_with_config(
            source='theblock',
            days_back=days_back,
            categories=categories,
            article_selector='div.post-card',
            title_selector='h2.post-card__headline',
            link_selector='a.post-card__inner',
            date_selector='time.post-card__timestamp',
            summary_selector='p.post-card__description',
            url_formatter=url_formatter
        )

    def scrape_reddit(self, days_back: int = 1, subreddits: List[str] = None) -> List[NewsItem]:

        self.logger.info("Scraping news from Reddit...")

        if not self.reddit:
            self.logger.error("Reddit API not initialized")
            return []

        news_data = []

        # Default subreddits
        if not subreddits:
            subreddits = self.source_config['reddit']['default_subreddits']

        start_date = datetime.now() - timedelta(days=days_back)

        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)

                # Get popular posts
                for post in subreddit.hot(limit=50):
                    try:
                        # Convert timestamp to datetime
                        post_date = datetime.fromtimestamp(post.created_utc)

                        # Check date
                        if post_date < start_date:
                            continue

                        # Create news item
                        news_item = self._create_news_item(
                            title=post.title,
                            summary=post.selftext[:500] if post.selftext else "",
                            link=f"https://www.reddit.com{post.permalink}",
                            source='reddit',
                            category=subreddit_name,
                            published_at=post_date,
                            score=post.score,
                            upvote_ratio=post.upvote_ratio,
                            num_comments=post.num_comments
                        )

                        news_data.append(news_item)
                    except Exception as e:
                        self.logger.error(f"Error processing Reddit post: {e}")

                # Delay between requests to different subreddits
                time.sleep(randint(1, 3))

            except Exception as e:
                self.logger.error(f"Error scraping subreddit {subreddit_name}: {e}")

        self.logger.info(f"Collected {len(news_data)} news from Reddit")
        return news_data

    def _scrape_source(self, source: str, days_back: int, categories: List[str] = None) -> List[NewsItem]:

        source_methods = {
            'coindesk': self.scrape_coindesk,
            'cointelegraph': self.scrape_cointelegraph,
            'decrypt': self.scrape_decrypt,
            'cryptoslate': self.scrape_cryptoslate,
            'theblock': self.scrape_theblock,
            'reddit': self.scrape_reddit
        }

        if source in source_methods:
            if source == 'reddit':
                return source_methods[source](days_back=days_back, subreddits=categories)
            else:
                return source_methods[source](days_back=days_back, categories=categories)
        else:
            self.logger.warning(f"Source {source} not implemented yet")
            return []
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

    def scrape_cryptobriefing(self, days_back: int = 1, categories: List[str] = None) -> List[NewsItem]:
        """Збирає новини з Crypto Briefing"""
        if not categories:
            categories = ["news", "analysis", "insights", "reviews"]

        def url_formatter(base_url, category, page):
            return f"{base_url}/{category}/page/{page}/"

        return self._scrape_source_with_config(
            source='cryptobriefing',
            days_back=days_back,
            categories=categories,
            article_selector='article.post',
            title_selector='h2.entry-title',
            link_selector='a.post-link',
            date_selector='time.entry-date',
            summary_selector='div.entry-content p',
            url_formatter=url_formatter,
            date_attribute='datetime'
        )

    def filter_by_keywords(self, news_items: List[NewsItem], keywords: List[str],
                           threshold: int = 1) -> List[NewsItem]:

        filtered_news = []

        for item in news_items:
            text = f"{item.title} {item.summary}".lower()
            matches = sum(1 for keyword in keywords if keyword.lower() in text)

            if matches >= threshold:
                filtered_news.append(item)

        return filtered_news

    def _preprocess_text(self, text: str) -> str:

        try:
            # Tokenize text
            tokens = word_tokenize(text.lower())

            # Remove stopwords and punctuation
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token.isalpha() and token not in stop_words]

            # Lemmatization
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]

            # Join back to string
            return ' '.join(tokens)
        except Exception as e:
            self.logger.error(f"Error preprocessing text: {e}")
            return text

    def _load_topic_models(self):

        try:
            # Create directory if it doesn't exist
            os.makedirs(self.topic_model_dir, exist_ok=True)

            # Try to load vectorizer
            vectorizer_path = os.path.join(self.topic_model_dir, 'vectorizer.pkl')
            if os.path.exists(vectorizer_path):
                self.vectorizer = joblib.load(vectorizer_path)
                self.logger.info("Loaded TF-IDF vectorizer from disk")

            # Try to load LDA model
            lda_path = os.path.join(self.topic_model_dir, 'lda_model.pkl')
            if os.path.exists(lda_path):
                self.lda_model = joblib.load(lda_path)
                self.logger.info("Loaded LDA model from disk")

            # Try to load NMF model
            nmf_path = os.path.join(self.topic_model_dir, 'nmf_model.pkl')
            if os.path.exists(nmf_path):
                self.nmf_model = joblib.load(nmf_path)
                self.logger.info("Loaded NMF model from disk")

            # Try to load KMeans model
            kmeans_path = os.path.join(self.topic_model_dir, 'kmeans_model.pkl')
            if os.path.exists(kmeans_path):
                self.kmeans_model = joblib.load(kmeans_path)
                self.logger.info("Loaded KMeans model from disk")

            # Try to load topic words
            topic_words_path = os.path.join(self.topic_model_dir, 'topic_words.pkl')
            if os.path.exists(topic_words_path):
                self.topic_words = joblib.load(topic_words_path)
                self.logger.info("Loaded topic keywords from disk")

        except Exception as e:
            self.logger.error(f"Error loading topic models: {e}")
            # Initialize empty models if loading fails
            self.vectorizer = None
            self.lda_model = None
            self.nmf_model = None
            self.kmeans_model = None
            self.topic_words = {}

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

    def train_topic_models(self, news_items: List[NewsItem], num_topics: int = 10, method: str = 'both'):

        self.logger.info(f"Training topic models with {len(news_items)} news items...")

        try:
            # Prepare corpus
            corpus = []
            for item in news_items:
                text = f"{item.title} {item.summary}"
                processed_text = self._preprocess_text(text)
                corpus.append(processed_text)

            if not corpus:
                self.logger.warning("Empty corpus for topic modeling")
                return

            # Create vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                min_df=2,
                max_df=0.85,
                stop_words='english'
            )

            # Transform corpus to TF-IDF matrix
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            feature_names = self.vectorizer.get_feature_names_out()

            # Initialize topic words dictionary
            self.topic_words = {}

            # Train LDA model
            if method in ['lda', 'both']:
                self.logger.info("Training LDA model...")
                self.lda_model = LatentDirichletAllocation(
                    n_components=num_topics,
                    max_iter=10,
                    learning_method='online',
                    random_state=42,
                    n_jobs=-1
                )
                self.lda_model.fit(tfidf_matrix)

                # Extract top words for each topic
                self.topic_words['lda'] = {}
                for topic_idx, topic in enumerate(self.lda_model.components_):
                    top_words_idx = topic.argsort()[:-11:-1]  # Get top 10 words
                    top_words = [feature_names[i] for i in top_words_idx]
                    self.topic_words['lda'][topic_idx] = top_words

            # Train NMF model
            if method in ['nmf', 'both']:
                self.logger.info("Training NMF model...")
                self.nmf_model = NMF(
                    n_components=num_topics,
                    random_state=42,
                    max_iter=100
                )
                self.nmf_model.fit(tfidf_matrix)

                # Extract top words for each topic
                self.topic_words['nmf'] = {}
                for topic_idx, topic in enumerate(self.nmf_model.components_):
                    top_words_idx = topic.argsort()[:-11:-1]  # Get top 10 words
                    top_words = [feature_names[i] for i in top_words_idx]
                    self.topic_words['nmf'][topic_idx] = top_words

            # Train KMeans model
            if method in ['kmeans', 'all']:
                self.logger.info("Training KMeans model...")
                self.kmeans_model = KMeans(
                    n_clusters=num_topics,
                    random_state=42,
                    n_init=10
                )
                self.kmeans_model.fit(tfidf_matrix)

                # Get cluster centers
                centers = self.kmeans_model.cluster_centers_

                # Extract top words for each cluster
                self.topic_words['kmeans'] = {}
                for topic_idx, center in enumerate(centers):
                    # Get indices of top words in the cluster center
                    top_words_idx = center.argsort()[:-11:-1]
                    top_words = [feature_names[i] for i in top_words_idx]
                    self.topic_words['kmeans'][topic_idx] = top_words

            # Save models
            self.save_topic_models()
            self.logger.info("Topic models trained successfully")

        except Exception as e:
            self.logger.error(f"Error training topic models: {e}")

    def extract_topics(self, text: str, method: str = 'auto', top_n: int = 3) -> List[str]:

        if not self.vectorizer:
            self.logger.warning("Vectorizer not initialized. Cannot extract topics.")
            return []

        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)

            # Transform text to TF-IDF vector
            text_tfidf = self.vectorizer.transform([processed_text])

            topics = []

            # Select method
            if method == 'auto':
                # Use LDA if available, otherwise NMF, then KMeans
                if self.lda_model:
                    method = 'lda'
                elif self.nmf_model:
                    method = 'nmf'
                elif self.kmeans_model:
                    method = 'kmeans'
                else:
                    self.logger.warning("No topic model available")
                    return []

            # Extract topics using LDA
            if method == 'lda' and self.lda_model:
                topic_distribution = self.lda_model.transform(text_tfidf)[0]
                top_topics = topic_distribution.argsort()[:-top_n - 1:-1]

                for topic_idx in top_topics:
                    if topic_idx in self.topic_words.get('lda', {}):
                        topics.extend(self.topic_words['lda'][topic_idx][:3])  # Take top 3 words per topic

            # Extract topics using NMF
            elif method == 'nmf' and self.nmf_model:
                topic_distribution = self.nmf_model.transform(text_tfidf)[0]
                top_topics = topic_distribution.argsort()[:-top_n - 1:-1]

                for topic_idx in top_topics:
                    if topic_idx in self.topic_words.get('nmf', {}):
                        topics.extend(self.topic_words['nmf'][topic_idx][:3])  # Take top 3 words per topic

            # Extract topics using KMeans
            elif method == 'kmeans' and self.kmeans_model:
                cluster = self.kmeans_model.predict(text_tfidf)[0]
                if cluster in self.topic_words.get('kmeans', {}):
                    topics.extend(self.topic_words['kmeans'][cluster][:5])  # Take top 5 words from cluster

            # Remove duplicates and return
            return list(set(topics))

        except Exception as e:
            self.logger.error(f"Error extracting topics: {e}")
            return []

    def analyze_news_topics(self, news_items: List[NewsItem], method: str = 'auto') -> Dict[str, int]:

        topic_counts = {}

        for item in news_items:
            topics = self.extract_topics(f"{item.title} {item.summary}", method=method)

            for topic in topics:
                if topic in topic_counts:
                    topic_counts[topic] += 1
                else:
                    topic_counts[topic] = 1

        # Sort by frequency
        sorted_topics = dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True))

        return sorted_topics

    def get_most_common_topics(self, news_items: List[NewsItem], limit: int = 10, method: str = 'auto') -> List[
        Tuple[str, int]]:

        topic_counts = self.analyze_news_topics(news_items, method=method)

        # Get top N topics
        top_topics = list(topic_counts.items())[:limit]

        return top_topics