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
        self.reddit = ("Emm1lUwc-LeGEF6UaQ17ag")

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

