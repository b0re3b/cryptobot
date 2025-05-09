import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import logging
import re
import time
import json
from typing import List, Dict, Optional, Any, Callable
from random import randint
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor


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
        """Convert NewsItem to dictionary format"""
        return {
            'title': self.title,
            'summary': self.summary,
            'link': self.link,
            'source': self.source,
            'category': self.category,
            'published_at': self.published_at.isoformat(),
            'scraped_at': self.scraped_at.isoformat(),
            'score': self.score,
            'upvote_ratio': self.upvote_ratio,
            'num_comments': self.num_comments,
            'sentiment_score': self.sentiment_score,
            'topics': self.topics
        }


class NewsCollector:
    """
    A class for collecting cryptocurrency news from various online sources.
    """

    def __init__(self,
                 news_sources: List[str] = None,
                 sentiment_analyzer=None,
                 logger: Optional[logging.Logger] = None,
                 db_manager=None,
                 max_pages: int = 5,
                 max_workers: int = 5,
                 topic_model_dir: str = './models'):

        self.news_sources = news_sources or [
            'coindesk', 'cointelegraph', 'decrypt', 'cryptoslate',
            'theblock', 'cryptopanic', 'coinmarketcal', 'feedly',
            'newsnow', 'cryptobriefing'
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

        # Headers for HTTP requests with rotation capability
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:93.0) Gecko/20100101 Firefox/93.0'
        ]
        self.headers = {'User-Agent': self.user_agents[0]}

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
                'base_url': 'https://feedly.com/i/collection/content/user/411e30a1-1c84-451d-9bb3-d740a96a284a/category/global.all',
                'default_categories': ["crypto", "blockchain", "bitcoin", "ethereum"]
            },
            'newsnow': {
                'base_url': 'https://www.newsnow.co.uk',
                'default_categories': ["crypto", "cryptocurrency", "bitcoin", "ethereum"]
            },
            'cryptobriefing': {
                'base_url': 'https://cryptobriefing.com',
                'default_categories': ["news", "analysis", "insights", "reviews"]
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

        # Proxy configuration (can be extended)
        self.proxies = None

    def _rotate_user_agent(self):
        """Rotate the user agent to avoid detection"""
        self.headers['User-Agent'] = self.user_agents[randint(0, len(self.user_agents) - 1)]

    def _make_request(self, url: str, retries: int = 3, backoff_factor: float = 0.3) -> Optional[requests.Response]:
        """Make an HTTP request with retry logic and user agent rotation"""
        self._rotate_user_agent()

        for i in range(retries):
            try:
                response = requests.get(
                    url,
                    headers=self.headers,
                    timeout=10,
                    proxies=self.proxies
                )
                if response.status_code == 200:
                    return response
                elif response.status_code in [403, 429]:
                    self.logger.warning(f"Received {response.status_code} status code from {url}. Waiting...")
                    time.sleep((backoff_factor * (2 ** i)) + randint(1, 3))
                    self._rotate_user_agent()  # Rotate user agent after each failure
                else:
                    self.logger.error(f"Error requesting {url}: HTTP {response.status_code}")
                    return None
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Connection error requesting {url}: {e}")
                time.sleep((backoff_factor * (2 ** i)) + randint(1, 3))

        return None

    def set_proxies(self, http_proxy: str = None, https_proxy: str = None):
        """Set proxies for making requests"""
        if http_proxy or https_proxy:
            self.proxies = {
                'http': http_proxy,
                'https': https_proxy or http_proxy
            }
            self.logger.info("Proxy configuration set")
        else:
            self.proxies = None
            self.logger.info("Proxy configuration cleared")

    def _parse_relative_date(self, date_text: str) -> Optional[datetime]:
        """Parse relative date strings like '2 hours ago' into datetime objects"""
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
        """Analyze sentiment of text if a sentiment analyzer is available"""
        if self.sentiment_analyzer:
            try:
                return self.sentiment_analyzer.analyze(text)
            except Exception as e:
                self.logger.error(f"Error analyzing sentiment: {e}")
                return 0.0
        return None

    def _create_news_item(self,
                          title: str,
                          summary: str,
                          link: str,
                          source: str,
                          category: str,
                          published_at: datetime,
                          **kwargs) -> NewsItem:
        """Create a NewsItem object with optional sentiment analysis and topic extraction"""
        # Clean title and summary
        title = title.strip()
        summary = summary.strip()

        # Truncate overly long summaries
        if len(summary) > 1000:
            summary = summary[:997] + "..."

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
        """Scrape news from CoinDesk"""
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
        """Scrape news from Cointelegraph"""
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
        """General method for scraping news with configurable CSS selectors"""
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
        """Scrape news from Decrypt"""
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
        """Scrape news from CryptoSlate"""
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
        """Scrape news from The Block"""
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

    def scrape_cryptobriefing(self, days_back: int = 1, categories: List[str] = None) -> List[NewsItem]:
        """Scrape news from Crypto Briefing"""
        if not categories:
            categories = self.source_config['cryptobriefing']['default_categories']

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

    def _scrape_source(self, source: str, days_back: int, categories: List[str] = None) -> List[NewsItem]:
        """Choose appropriate scraping method based on source name"""
        source_methods = {
            'coindesk': self.scrape_coindesk,
            'cointelegraph': self.scrape_cointelegraph,
            'decrypt': self.scrape_decrypt,
            'cryptoslate': self.scrape_cryptoslate,
            'theblock': self.scrape_theblock,
            'cryptobriefing': self.scrape_cryptobriefing,
            'cryptopanic': self.scrape_cryptopanic,
            'coinmarketcal': self.scrape_coinmarketcal,
            'feedly': self.scrape_feedly,
            'newsnow': self.scrape_newsnow
        }

        if source in source_methods:
            return source_methods[source](days_back=days_back, categories=categories)
        else:
            self.logger.warning(f"Source {source} not implemented yet")
            return []

    def scrape_all_sources(self, days_back: int = 1, categories: List[str] = None) -> List[NewsItem]:
        """Collect news from all configured sources in parallel"""
        self.logger.info(f"Starting news collection from all available sources for the last {days_back} days")

        all_news = []

        # Dictionary with functions for each source
        source_functions = {
            'coindesk': self.scrape_coindesk,
            'cointelegraph': self.scrape_cointelegraph,
            'decrypt': self.scrape_decrypt,
            'cryptoslate': self.scrape_cryptoslate,
            'theblock': self.scrape_theblock,
            'cryptobriefing': self.scrape_cryptobriefing,
            'cryptopanic': self.scrape_cryptopanic,
            'coinmarketcal': self.scrape_coinmarketcal,
            'feedly': self.scrape_feedly,
            'newsnow': self.scrape_newsnow
        }

        # Determine which sources to use
        sources_to_scrape = [source for source in self.news_sources if source in source_functions]

        # Define a worker function for ThreadPoolExecutor
        def scrape_source_worker(source):
            try:
                self.logger.info(f"Collecting news from source {source}")

                # Set categories for each source
                source_categories = categories or self.source_config.get(source, {}).get('default_categories')

                # Call the appropriate function for the source
                news_from_source = source_functions[source](days_back=days_back, categories=source_categories)

                if news_from_source:
                    self.logger.info(f"Successfully collected {len(news_from_source)} news items from {source}")
                    return news_from_source
                else:
                    self.logger.warning(f"No news collected from {source}")
                    return []

            except Exception as e:
                self.logger.error(f"Error collecting news from {source}: {e}")
                return []

        # Use ThreadPoolExecutor for parallel scraping
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(sources_to_scrape))) as executor:
            results = list(executor.map(scrape_source_worker, sources_to_scrape))

        # Flatten results
        for result in results:
            all_news.extend(result)

        # Remove duplicates based on title
        unique_news = []
        seen_titles = set()

        for news_item in all_news:
            if news_item.title not in seen_titles:
                seen_titles.add(news_item.title)
                unique_news.append(news_item)

        self.logger.info(f"Total collected {len(unique_news)} unique news items from all sources")
        return unique_news

    ### 1. Реалізації для відсутніх джерел

    def scrape_cryptopanic(self, days_back: int = 1, categories: List[str] = None) -> List[NewsItem]:
        """Scrape news from CryptoPanic"""
        self.logger.info("Scraping news from CryptoPanic...")
        news_data = []

        try:
            # Determine start date
            start_date = datetime.now() - timedelta(days=days_back)

            # Get base URL and categories
            base_url = self.source_config['cryptopanic']['base_url']
            if not categories:
                categories = self.source_config['cryptopanic']['default_categories']

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
                    articles = soup.select('div.news-item')

                    if not articles:
                        continue_scraping = False
                        continue

                    for article in articles:
                        try:
                            # Get publication date
                            date_elem = article.select_one('div.news-item-footer time')
                            if not date_elem:
                                continue

                            date_text = date_elem.text.strip()
                            pub_date = self._parse_relative_date(date_text)

                            # Check if article is within the time period
                            if pub_date < start_date:
                                continue_scraping = False
                                break

                            # Get title and link
                            title_elem = article.select_one('div.news-item-title a')
                            if not title_elem:
                                continue

                            title = title_elem.text.strip()
                            link = title_elem['href']

                            # If link is relative, make it absolute
                            if link.startswith('/'):
                                link = base_url + link

                            # Get source
                            source_elem = article.select_one('div.news-item-source a')
                            source_name = 'cryptopanic'
                            if source_elem:
                                source_name = source_elem.text.strip()

                            # Get summary - CryptoPanic often doesn't have summaries
                            summary = title  # Default to title if no summary
                            summary_elem = article.select_one('div.news-item-text')
                            if summary_elem:
                                summary = summary_elem.text.strip()

                            # Get votes (similar to score)
                            score = None
                            votes_elem = article.select_one('div.votes-count')
                            if votes_elem:
                                try:
                                    score = int(votes_elem.text.strip())
                                except ValueError:
                                    pass

                            # Create news item
                            news_item = self._create_news_item(
                                title=title,
                                summary=summary,
                                link=link,
                                source='cryptopanic',
                                category=category,
                                published_at=pub_date,
                                score=score
                            )

                            news_data.append(news_item)
                        except Exception as e:
                            self.logger.error(f"Error processing CryptoPanic article: {e}")

                    # Go to next page
                    page += 1

                    # Delay to prevent blocking
                    time.sleep(randint(1, 3))

        except Exception as e:
            self.logger.error(f"General error while scraping CryptoPanic: {e}")

        self.logger.info(f"Collected {len(news_data)} news from CryptoPanic")
        return news_data

    def scrape_coinmarketcal(self, days_back: int = 1, categories: List[str] = None) -> List[NewsItem]:
        """Scrape events from CoinMarketCal"""
        self.logger.info("Scraping events from CoinMarketCal...")
        news_data = []

        try:
            # Determine start date
            start_date = datetime.now() - timedelta(days=days_back)

            # Get base URL and categories
            base_url = self.source_config['coinmarketcal']['base_url']
            if not categories:
                categories = self.source_config['coinmarketcal']['default_categories']

            for category in categories:
                page = 1
                continue_scraping = True

                while continue_scraping and page <= self.max_pages:
                    # CoinMarketCal has a different URL structure depending on category
                    if category == 'events':
                        url = f"{base_url}/en/?page={page}"
                    else:
                        url = f"{base_url}/en/{category}/?page={page}"

                    response = self._make_request(url)

                    if not response:
                        continue_scraping = False
                        continue

                    soup = BeautifulSoup(response.text, 'html.parser')
                    events = soup.select('tr.row-event')

                    if not events:
                        continue_scraping = False
                        continue

                    for event in events:
                        try:
                            # Get date
                            date_elem = event.select_one('td.date-event time')
                            if not date_elem or not date_elem.get('datetime'):
                                continue

                            date_str = date_elem['datetime']
                            try:
                                pub_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                            except ValueError:
                                pub_date = self._parse_relative_date(date_str)

                            # Check if event is within the time period
                            if pub_date < start_date:
                                continue_scraping = False
                                break

                            # Get title and link
                            title_elem = event.select_one('td.title-event a')
                            if not title_elem:
                                continue

                            title = title_elem.text.strip()
                            link = title_elem['href']

                            # If link is relative, make it absolute
                            if not link.startswith('http'):
                                link = base_url + link

                            # Get coin name
                            coin_elem = event.select_one('td.coin-event a')
                            coin_name = "General"
                            if coin_elem:
                                coin_name = coin_elem.text.strip()

                            # Build summary from available information
                            summary = f"Event for {coin_name}: {title}"
                            description_elem = event.select_one('td.description-event')
                            if description_elem:
                                description = description_elem.text.strip()
                                if description:
                                    summary = description

                            # Create news item
                            news_item = self._create_news_item(
                                title=title,
                                summary=summary,
                                link=link,
                                source='coinmarketcal',
                                category=category,
                                published_at=pub_date
                            )

                            news_data.append(news_item)
                        except Exception as e:
                            self.logger.error(f"Error processing CoinMarketCal event: {e}")

                    # Go to next page
                    page += 1

                    # Delay to prevent blocking
                    time.sleep(randint(1, 3))

        except Exception as e:
            self.logger.error(f"General error while scraping CoinMarketCal: {e}")

        self.logger.info(f"Collected {len(news_data)} events from CoinMarketCal")
        return news_data

    def scrape_feedly(self, days_back: int = 1, categories: List[str] = None) -> List[NewsItem]:
        """Scrape news from Feedly curated crypto collections

        Note: Feedly is a feed reader, not a news source. This implementation assumes
        we're scraping a publicly accessible Feedly collection. If authentication is required,
        a different approach would be needed.
        """
        self.logger.info("Scraping news from Feedly collections...")
        news_data = []

        try:
            # Determine start date
            start_date = datetime.now() - timedelta(days=days_back)

            # Get base URL - feedly uses a different approach since it's a feed reader
            base_url = self.source_config['feedly']['base_url']

            # For Feedly we need to set a cookie and maybe additional headers
            self._rotate_user_agent()
            self.headers['Accept'] = 'text/html,application/xhtml+xml,application/xml'

            response = self._make_request(base_url)

            if not response:
                self.logger.error("Failed to access Feedly collection")
                return news_data

            soup = BeautifulSoup(response.text, 'html.parser')
            articles = soup.select('article.entry')

            for article in articles:
                try:
                    # Get publication date
                    date_elem = article.select_one('time.ago')
                    if not date_elem:
                        continue

                    date_text = date_elem.text.strip()
                    pub_date = self._parse_relative_date(date_text)

                    # Check if article is within the time period
                    if pub_date < start_date:
                        continue

                    # Get title and link
                    title_elem = article.select_one('h3.title a')
                    if not title_elem:
                        continue

                    title = title_elem.text.strip()
                    link = title_elem['href']

                    # Get source
                    source_elem = article.select_one('span.source')
                    source_name = 'feedly'
                    if source_elem:
                        source_name = source_elem.text.strip()

                    # Get summary
                    summary_elem = article.select_one('div.content')
                    summary = ""
                    if summary_elem:
                        summary = summary_elem.text.strip()

                    # Get category from available tags
                    category_elem = article.select_one('span.category')
                    category = 'crypto'  # Default
                    if category_elem:
                        category = category_elem.text.strip().lower()

                    # Create news item
                    news_item = self._create_news_item(
                        title=title,
                        summary=summary,
                        link=link,
                        source='feedly',
                        category=category,
                        published_at=pub_date
                    )

                    news_data.append(news_item)
                except Exception as e:
                    self.logger.error(f"Error processing Feedly article: {e}")

        except Exception as e:
            self.logger.error(f"General error while scraping Feedly: {e}")

        self.logger.info(f"Collected {len(news_data)} news from Feedly")
        return news_data

    def scrape_newsnow(self, days_back: int = 1, categories: List[str] = None) -> List[NewsItem]:
        """Scrape news from NewsNow cryptocurrency section"""
        self.logger.info("Scraping news from NewsNow...")
        news_data = []

        try:
            # Determine start date
            start_date = datetime.now() - timedelta(days=days_back)

            # Get base URL and categories
            base_url = self.source_config['newsnow']['base_url']
            if not categories:
                categories = self.source_config['newsnow']['default_categories']

            for category in categories:
                # NewsNow uses a different URL structure for categories
                url = f"{base_url}/{category.lower()}"
                response = self._make_request(url)

                if not response:
                    continue

                soup = BeautifulSoup(response.text, 'html.parser')
                articles = soup.select('div.article')

                for article in articles:
                    try:
                        # Get publication date
                        date_elem = article.select_one('span.time')
                        if not date_elem:
                            continue

                        date_text = date_elem.text.strip()

                        # NewsNow uses relative times like "2h ago"
                        pub_date = self._parse_relative_date(date_text)

                        # Check if article is within the time period
                        if pub_date < start_date:
                            continue

                        # Get title and link
                        title_elem = article.select_one('a.article-link span.title')
                        link_elem = article.select_one('a.article-link')

                        if not title_elem or not link_elem:
                            continue

                        title = title_elem.text.strip()
                        link = link_elem['href']

                        # Get source
                        source_elem = article.select_one('span.source')
                        source_name = source_elem.text.strip() if source_elem else "NewsNow"

                        # NewsNow doesn't typically have summaries in the listing
                        summary = title  # Use title as summary

                        # Create news item
                        news_item = self._create_news_item(
                            title=title,
                            summary=summary,
                            link=link,
                            source='newsnow',
                            category=category,
                            published_at=pub_date
                        )

                        news_data.append(news_item)
                    except Exception as e:
                        self.logger.error(f"Error processing NewsNow article: {e}")

        except Exception as e:
            self.logger.error(f"General error while scraping NewsNow: {e}")

        self.logger.info(f"Collected {len(news_data)} news from NewsNow")
        return news_data

    ### 2. Реалізація методу виділення тем

    def load_topic_models(self):
        """Load topic extraction models"""
        try:
            import pickle
            import numpy as np
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import LatentDirichletAllocation, NMF
            from sklearn.cluster import KMeans

            # Check if topic_model_dir exists
            import os
            if not os.path.exists(self.topic_model_dir):
                os.makedirs(self.topic_model_dir)
                self.logger.warning(f"Created topic model directory: {self.topic_model_dir}")
                return False

            # Load vectorizer
            try:
                with open(f"{self.topic_model_dir}/tfidf_vectorizer.pkl", 'rb') as f:
                    self.vectorizer = pickle.load(f)
                    self.logger.info("Loaded TF-IDF vectorizer model")
            except FileNotFoundError:
                self.vectorizer = TfidfVectorizer(
                    max_df=0.95, min_df=2, stop_words='english',
                    max_features=1000, ngram_range=(1, 2)
                )
                self.logger.warning("Created new TF-IDF vectorizer (not trained)")

            # Load LDA model
            try:
                with open(f"{self.topic_model_dir}/lda_model.pkl", 'rb') as f:
                    self.lda_model = pickle.load(f)
                    self.logger.info("Loaded LDA topic model")
            except FileNotFoundError:
                self.lda_model = None
                self.logger.warning("LDA model not found")

            # Load NMF model
            try:
                with open(f"{self.topic_model_dir}/nmf_model.pkl", 'rb') as f:
                    self.nmf_model = pickle.load(f)
                    self.logger.info("Loaded NMF topic model")
            except FileNotFoundError:
                self.nmf_model = None
                self.logger.warning("NMF model not found")

            # Load KMeans model
            try:
                with open(f"{self.topic_model_dir}/kmeans_model.pkl", 'rb') as f:
                    self.kmeans_model = pickle.load(f)
                    self.logger.info("Loaded KMeans model")
            except FileNotFoundError:
                self.kmeans_model = None
                self.logger.warning("KMeans model not found")

            # Load topic words
            try:
                with open(f"{self.topic_model_dir}/topic_words.pkl", 'rb') as f:
                    self.topic_words = pickle.load(f)
                    self.logger.info("Loaded topic words dictionary")
                    return True
            except FileNotFoundError:
                self.topic_words = {
                    0: ["bitcoin", "btc", "price", "market"],
                    1: ["ethereum", "eth", "defi", "smart"],
                    2: ["nft", "art", "collectible", "token"],
                    3: ["regulation", "sec", "compliance", "legal"],
                    4: ["mining", "hash", "pow", "energy"],
                    5: ["blockchain", "technology", "solution", "protocol"],
                    6: ["exchange", "trading", "dex", "liquidity"],
                    7: ["altcoin", "token", "cryptocurrency", "investment"],
                    8: ["wallet", "security", "storage", "private"],
                    9: ["dao", "governance", "vote", "community"]
                }
                self.logger.warning("Created default topic words dictionary")
                return False
        except ImportError as e:
            self.logger.error(f"Could not load topic models: {e}")
            self.logger.warning("Make sure scikit-learn is installed for topic modeling")
            return False
        except Exception as e:
            self.logger.error(f"Error loading topic models: {e}")
            return False

    def train_topic_models(self, texts: List[str], n_topics: int = 10):
        """Train topic models on a corpus of texts"""
        try:
            import numpy as np
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import LatentDirichletAllocation, NMF
            from sklearn.cluster import KMeans
            import pickle

            self.logger.info(f"Training topic models on {len(texts)} documents")

            # Initialize and fit vectorizer
            self.vectorizer = TfidfVectorizer(
                max_df=0.95, min_df=2, stop_words='english',
                max_features=1000, ngram_range=(1, 2)
            )
            X = self.vectorizer.fit_transform(texts)

            # Train LDA model
            self.lda_model = LatentDirichletAllocation(
                n_components=n_topics, max_iter=10, random_state=42,
                learning_method='online', n_jobs=-1
            )
            self.lda_model.fit(X)

            # Train NMF model
            self.nmf_model = NMF(
                n_components=n_topics, random_state=42, max_iter=1000,
                alpha=0.1, l1_ratio=0.5
            )
            self.nmf_model.fit(X)

            # Train KMeans model
            self.kmeans_model = KMeans(
                n_clusters=n_topics, random_state=42, n_init=10
            )
            self.kmeans_model.fit(X)

            # Extract topic words
            self.topic_words = {}
            feature_names = self.vectorizer.get_feature_names_out()

            # Get top words for each topic in LDA
            for topic_idx, topic in enumerate(self.lda_model.components_):
                top_words_idx = topic.argsort()[:-11:-1]  # Get indices of top 10 words
                top_words = [feature_names[i] for i in top_words_idx]
                self.topic_words[topic_idx] = top_words

            # Save models
            import os
            if not os.path.exists(self.topic_model_dir):
                os.makedirs(self.topic_model_dir)

            with open(f"{self.topic_model_dir}/tfidf_vectorizer.pkl", 'wb') as f:
                pickle.dump(self.vectorizer, f)

            with open(f"{self.topic_model_dir}/lda_model.pkl", 'wb') as f:
                pickle.dump(self.lda_model, f)

            with open(f"{self.topic_model_dir}/nmf_model.pkl", 'wb') as f:
                pickle.dump(self.nmf_model, f)

            with open(f"{self.topic_model_dir}/kmeans_model.pkl", 'wb') as f:
                pickle.dump(self.kmeans_model, f)

            with open(f"{self.topic_model_dir}/topic_words.pkl", 'wb') as f:
                pickle.dump(self.topic_words, f)

            self.logger.info("Topic models trained and saved successfully")
            return True
        except ImportError as e:
            self.logger.error(f"Could not train topic models: {e}")
            self.logger.warning("Make sure scikit-learn is installed for topic modeling")
            return False
        except Exception as e:
            self.logger.error(f"Error training topic models: {e}")
            return False

    def extract_topics(self, text: str, top_n: int = 3) -> List[str]:
        """Extract topics from text using trained topic models"""
        if not (self.vectorizer and (self.lda_model or self.nmf_model)):
            # Try to load models first
            models_loaded = self.load_topic_models()
            if not models_loaded:
                self.logger.warning("Topic models not available, returning default topics")
                return ["crypto", "blockchain"]  # Default topics

        try:
            # Vectorize the text
            text_vector = self.vectorizer.transform([text])

            topics = []

            # Use LDA model if available
            if self.lda_model is not None:
                lda_topic_distribution = self.lda_model.transform(text_vector)[0]
                top_lda_topics = lda_topic_distribution.argsort()[:-top_n - 1:-1]

                # Add top words from top topics
                for topic_idx in top_lda_topics:
                    if topic_idx in self.topic_words:
                        topics.extend(self.topic_words[topic_idx][:2])  # Add top 2 words from each topic

            # Use NMF model if available
            if self.nmf_model is not None:
                nmf_topic_distribution = self.nmf_model.transform(text_vector)[0]
                top_nmf_topics = nmf_topic_distribution.argsort()[:-top_n - 1:-1]

                # Add top words from top topics
                for topic_idx in top_nmf_topics:
                    if topic_idx in self.topic_words:
                        topics.extend(self.topic_words[topic_idx][:2])  # Add top 2 words from each topic

            # Use KMeans model if available
            if self.kmeans_model is not None:
                cluster = self.kmeans_model.predict(text_vector)[0]
                if cluster in self.topic_words:
                    topics.extend(self.topic_words[cluster][:3])  # Add top 3 words from the cluster

            # If no models worked, use keyword extraction as fallback
            if not topics:
                topics = self._extract_keywords(text, n=top_n)

            # Remove duplicates and normalize
            unique_topics = []
            for topic in topics:
                topic = topic.lower().strip()
                if topic not in unique_topics:
                    unique_topics.append(topic)

            return unique_topics[:top_n]  # Return top N unique topics

        except Exception as e:
            self.logger.error(f"Error extracting topics: {e}")
            return ["crypto", "blockchain"]  # Default topics on error

    def _extract_keywords(self, text: str, n: int = 5) -> List[str]:
        """Simple keyword extraction using TF-IDF for when models aren't available"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            # Basic crypto-related stopwords to filter out common terms
            crypto_stopwords = [
                'the', 'and', 'to', 'of', 'a', 'in', 'for', 'on', 'is', 'that',
                'said', 'with', 'by', 'as', 'at', 'from', 'it', 'was', 'be', 'this',
                'crypto', 'cryptocurrency', 'cryptocurrencies', 'blockchain'
            ]

            # Create vectorizer for this single document
            vectorizer = TfidfVectorizer(
                max_df=1.0, min_df=1, stop_words=crypto_stopwords,
                max_features=100, ngram_range=(1, 2)
            )

            # Fit on just this document
            X = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()

            # Get top N keywords
            scores = zip(feature_names, X.toarray()[0])
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

            keywords = [word for word, score in sorted_scores[:n]]
            return keywords
        except ImportError:
            # If sklearn is not available, fallback to simple frequency
            words = text.lower().split()
            word_freq = {}

            for word in words:
                if len(word) > 3:  # Ignore very short words
                    word_freq[word] = word_freq.get(word, 0) + 1

            # Sort by frequency
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [word for word, freq in sorted_words[:n]]
        except Exception as e:
            self.logger.error(f"Error in keyword extraction: {e}")
            return ["crypto", "blockchain", "bitcoin"]
