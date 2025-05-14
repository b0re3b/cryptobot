import requests  # Add this import which was missing in your original code
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import logging
import re
import time
import json
from typing import List, Dict, Optional, Any, Callable, Tuple
from random import randint, choice
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor


# Keep all your existing imports, but make sure requests is imported

# Main changes to make the scraper work:

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
    Enhanced crypto news collector from various online sources.
    """

    def __init__(self,
                 news_sources: List[str] = None,
                 sentiment_analyzer=None,
                 logger: Optional[logging.Logger] = None,
                 db_manager=None,
                 max_pages: int = 5,
                 max_workers: int = 5,
                 topic_model_dir: str = './models',
                 anticaptcha_key: str = None,
                 use_headless_browser: bool = False):

        self.news_sources = news_sources or [
            'coindesk', 'cointelegraph', 'decrypt', 'cryptoslate',
            'theblock', 'cryptobriefing'
        ]
        self.sentiment_analyzer = sentiment_analyzer
        self.db_manager = db_manager
        self.max_pages = max_pages
        self.max_workers = max_workers
        self.topic_model_dir = topic_model_dir
        # Initialize NewsAnalyzer only if available
        try:
            from models.NewsAnalyzer import BERTNewsAnalyzer
            self.NewsAnalyzer = BERTNewsAnalyzer()
        except ImportError:
            self.NewsAnalyzer = None
            print("BERTNewsAnalyzer not available - sentiment analysis will be disabled")

        self.anticaptcha_key = anticaptcha_key
        self.use_headless_browser = use_headless_browser
        self.webdriver = None

        # Set up logger
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger('crypto_news_scraper')
            self.logger.setLevel(logging.DEBUG)

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

            # File handler for detailed logging
            try:
                file_handler = logging.FileHandler('crypto_news_scraper_detailed.log')
                file_handler.setLevel(logging.DEBUG)
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s')
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                print(f"Failed to create log file: {e}")

        # User agents for request headers with rotation capability
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0'
        ]

        # Headers for better browser emulation
        self.headers = {
            'User-Agent': self.user_agents[0],
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

        # Updated source configurations with more accurate selectors for 2025
        self.source_config = {
            'coindesk': {
                'base_url': 'https://www.coindesk.com',
                'default_categories': ["markets", "business", "policy", "tech"],
                'selectors': {
                    'article': 'div.article-cardstyle__AcRoot-sc-1naml06-0, article.card-articlestyle__CardArticleContainer-sc-1s5az0h-0',
                    'title': 'h6.typography__StyledTypography-sc-owin6q-0, h4.typography__StyledTypography-sc-owin6q-0',
                    'link': 'a[href*="/markets/"], a[href*="/business/"], a[href*="/policy/"], a[href*="/tech/"]',
                    'date': 'time, span.typography__StyledTypography-sc-owin6q-0[data-testid="published-timestamp"]',
                    'summary': 'p.typography__StyledTypography-sc-owin6q-0, div.card-articlestyle__AcDek-sc-1s5az0h-8'
                },
                'js_rendering_required': False
            },
            'cointelegraph': {
                'base_url': 'https://cointelegraph.com',
                'default_categories': ["news", "bitcoin", "ethereum", "altcoins", "blockchain"],
                'selectors': {
                    'article': 'li.posts-listing__item, article.post-card-inline',
                    'title': 'span.post-card-inline__title, h2.post-card__title',
                    'link': 'a.post-card-inline__link, a.post-card__link',
                    'date': 'time.post-card-inline__date, time.post-card__date',
                    'summary': 'p.post-card-inline__text, p.post-card__text'
                },
                'js_rendering_required': True
            },
            'decrypt': {
                'base_url': 'https://decrypt.co',
                'default_categories': ["news", "learn", "analysis"],
                'selectors': {
                    'article': 'div.sc-2f9dbb1c-0, article.contentListItem',
                    'title': 'h2.sc-4aa4ae33-0, h3.contentTitle',
                    'link': 'a.sc-2f9dbb1c-2, a.contentLink',
                    'date': 'div.sc-2f9dbb1c-3 span, span.contentDate',
                    'summary': 'p.sc-4aa4ae33-1, p.contentSummary'
                },
                'js_rendering_required': True
            },
            'cryptoslate': {
                'base_url': 'https://cryptoslate.com',
                'default_categories': ["news", "bitcoin", "ethereum", "defi"],
                'selectors': {
                    'article': 'div.news-item, article.post',
                    'title': 'h3.title, h2.entry-title',
                    'link': 'a.news-item-link, a.post-link',
                    'date': 'div.news-item-meta span, span.post-date',
                    'summary': 'p.excerpt, div.entry-summary'
                },
                'js_rendering_required': False
            },
            'theblock': {
                'base_url': 'https://www.theblock.co',
                'default_categories': ["news", "analysis", "latest"],
                'selectors': {
                    'article': 'div.border-b, article.post-item',
                    'title': 'h2.text-xl, h3.post-title',
                    'link': 'a[href*="/post/"], a.post-link',
                    'date': 'div.text-gray-400, span.post-date',
                    'summary': 'p.mt-2, div.post-excerpt'
                },
                'js_rendering_required': True
            },
            'cryptobriefing': {
                'base_url': 'https://cryptobriefing.com',
                'default_categories': ["news", "analysis", "reviews"],
                'selectors': {
                    'article': 'article.post, div.article-item',
                    'title': 'h2.entry-title, h3.article-title',
                    'link': 'a.entry-title-link, a.article-link',
                    'date': 'time.entry-date, span.article-date',
                    'summary': 'div.entry-content p, p.article-excerpt'
                },
                'js_rendering_required': False
            }
        }

        # Initialize caches
        self._cache = {}

        # Topic modeling components
        self.vectorizer = None
        self.lda_model = None
        self.nmf_model = None
        self.kmeans_model = None
        self.topic_words = {}

        # Proxy configuration
        self.proxies = None

        # Scraping success tracking
        self.request_stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'status_codes': {},
            'captcha_encountered': 0,
            'js_rendering_required': 0
        }

        # Initialize webdriver if needed
        if use_headless_browser:
            self._init_webdriver()

    def __del__(self):
        """Close webdriver when object is deleted"""
        if hasattr(self, 'webdriver') and self.webdriver:
            try:
                self.webdriver.quit()
            except Exception as e:
                self.logger.warning(f"Error closing webdriver: {e}")

    def _init_webdriver(self):
        """Initialize Selenium webdriver for JavaScript-generated content"""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options

            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument(f'user-agent={choice(self.user_agents)}')
            options.add_argument('--window-size=1920,1080')

            self.webdriver = webdriver.Chrome(options=options)
            self.logger.info("Selenium webdriver successfully initialized")
        except ImportError:
            self.logger.warning("Selenium not installed. JavaScript processing will be unavailable.")
            self.webdriver = None
        except Exception as e:
            self.logger.error(f"Failed to initialize webdriver: {e}")
            self.webdriver = None

    def _rotate_user_agent(self):
        """Rotate user agent to avoid detection"""
        self.headers['User-Agent'] = choice(self.user_agents)
        self.logger.debug(f"User-Agent changed to: {self.headers['User-Agent']}")

    def _make_request(self, url: str, retries: int = 3, backoff_factor: float = 0.3) -> Optional[requests.Response]:
        """Make HTTP request with retry logic and user agent rotation"""
        self._rotate_user_agent()
        self.request_stats['total'] += 1

        self.logger.debug(f"Making request to {url} (attempt 1 of {retries})")

        for i in range(retries):
            try:
                start_time = time.time()
                response = requests.get(
                    url,
                    headers=self.headers,
                    timeout=15,
                    proxies=self.proxies
                )
                elapsed_time = time.time() - start_time

                # Log status code and execution time
                self.logger.debug(
                    f"Response from {url}: HTTP {response.status_code}, time: {elapsed_time:.2f} sec")

                # Update statistics
                status_code = str(response.status_code)
                if status_code in self.request_stats['status_codes']:
                    self.request_stats['status_codes'][status_code] += 1
                else:
                    self.request_stats['status_codes'][status_code] = 1

                if response.status_code == 200:
                    self.request_stats['success'] += 1

                    # DEBUG: Save the received HTML to inspect
                    with open(f"debug_{url.split('/')[-1]}.html", "w", encoding="utf-8") as f:
                        f.write(response.text)
                    self.logger.debug(f"Saved HTML content to debug_{url.split('/')[-1]}.html")

                    return response
                elif response.status_code in [403, 429]:
                    self.logger.warning(f"Received status code {response.status_code} from {url}. Waiting...")
                    delay = (backoff_factor * (2 ** i)) + randint(2, 5)
                    self.logger.debug(f"Waiting {delay:.2f} seconds before next attempt")
                    time.sleep(delay)
                    self._rotate_user_agent()
                elif response.status_code in [500, 502, 503, 504]:
                    self.logger.warning(f"Server error {response.status_code} from {url}. Waiting...")
                    delay = (backoff_factor * (2 ** i)) + randint(3, 8)
                    self.logger.debug(f"Waiting {delay:.2f} seconds before next attempt")
                    time.sleep(delay)
                else:
                    self.logger.error(f"Request error to {url}: HTTP {response.status_code}")
                    self.request_stats['failed'] += 1
                    return None
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Connection error requesting {url}: {str(e)[:200]}")
                delay = (backoff_factor * (2 ** i)) + randint(1, 3)
                self.logger.debug(f"Waiting {delay:.2f} seconds before next attempt")
                time.sleep(delay)
                self.request_stats['failed'] += 1

            if i < retries - 1:
                self.logger.debug(f"Making request to {url} (attempt {i + 2} of {retries})")

        self.request_stats['failed'] += 1
        return None

    def _fetch_page_with_javascript(self, url: str,
                                    article_selector: str,
                                    wait_time: int = 10) -> Optional[str]:
        """Fetch page with JavaScript rendering using Selenium"""
        if not hasattr(self, 'webdriver') or not self.webdriver:
            self.logger.warning("Webdriver not initialized for JavaScript rendering")
            return None

        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from selenium.common.exceptions import TimeoutException

            self.logger.info(f"Loading page with JS rendering: {url}")
            self.webdriver.get(url)

            # Wait for content to load
            try:
                WebDriverWait(self.webdriver, wait_time).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, article_selector))
                )
                self.logger.debug(f"Page loaded, elements {article_selector} found")
            except TimeoutException:
                self.logger.warning(f"Timeout waiting for elements {article_selector} on {url}")

            # Extra delay for full loading
            time.sleep(2)

            # Scroll to load lazy-loaded content
            self.webdriver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
            time.sleep(1)
            self.webdriver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)

            # Get HTML
            page_source = self.webdriver.page_source

            # Save for debugging
            with open(f"debug_js_{url.split('/')[-1]}.html", "w", encoding="utf-8") as f:
                f.write(page_source)
            self.logger.debug(f"Saved JS-rendered HTML to debug_js_{url.split('/')[-1]}.html")

            self.logger.debug(f"Got HTML content with JS rendering, size: {len(page_source)} bytes")
            return page_source

        except Exception as e:
            self.logger.error(f"Error loading {url} with JS rendering: {e}")
            return None

    def _parse_relative_date(self, date_text: str) -> Optional[datetime]:
        """Convert relative date strings like '2 hours ago' to datetime objects"""
        self.logger.debug(f"Parsing relative date: '{date_text}'")
        try:
            date_text = date_text.lower().strip()

            # Simple fallback - if we can't parse, use today's date
            if not date_text:
                self.logger.warning("Empty date text, using current date")
                return datetime.now()

            # Handle specific date formats
            try:
                # Try common formats
                for fmt in ('%B %d, %Y', '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%d %B %Y', '%d %b %Y'):
                    try:
                        return datetime.strptime(date_text, fmt)
                    except ValueError:
                        continue
            except Exception:
                pass  # Continue with relative date parsing

            # Patterns for finding relative dates
            patterns = [
                (r'(\d+)\s*hours?\s*ago', lambda x: datetime.now() - timedelta(hours=int(x))),
                (r'(\d+)\s*hr', lambda x: datetime.now() - timedelta(hours=int(x))),
                (r'(\d+)\s*days?\s*ago', lambda x: datetime.now() - timedelta(days=int(x))),
                (r'(\d+)\s*d\s', lambda x: datetime.now() - timedelta(days=int(x))),
                (r'(\d+)\s*minutes?\s*ago', lambda x: datetime.now() - timedelta(minutes=int(x))),
                (r'(\d+)\s*mins?', lambda x: datetime.now() - timedelta(minutes=int(x))),
                (r'(\d+)\s*m\s', lambda x: datetime.now() - timedelta(minutes=int(x))),
                (r'(\d+)\s*seconds?\s*ago', lambda x: datetime.now() - timedelta(seconds=int(x))),
                (r'(\d+)\s*s\s', lambda x: datetime.now() - timedelta(seconds=int(x))),
                (r'(\d+)\s*weeks?\s*ago', lambda x: datetime.now() - timedelta(weeks=int(x))),
                (r'(\d+)\s*w\s', lambda x: datetime.now() - timedelta(weeks=int(x))),
                (r'(\d+)\s*months?\s*ago', lambda x: datetime.now() - timedelta(days=30 * int(x))),
                (r'(\d+)mo', lambda x: datetime.now() - timedelta(days=30 * int(x)))
            ]

            # Special cases
            if 'yesterday' in date_text:
                return datetime.now() - timedelta(days=1)
            elif any(term in date_text for term in
                     ['today', 'just now', 'moments ago', 'hour ago', 'hours ago', 'min ago', 'mins ago']):
                # For very recent content
                return datetime.now()
            elif 'last week' in date_text:
                return datetime.now() - timedelta(weeks=1)
            elif 'last month' in date_text:
                return datetime.now() - timedelta(days=30)

            # Check patterns
            for pattern, time_func in patterns:
                match = re.search(pattern, date_text)
                if match:
                    value = match.group(1)
                    return time_func(value)

            # If no pattern found, use current date as fallback
            self.logger.warning(f"Could not parse relative date: '{date_text}', using current date")
            return datetime.now()
        except Exception as e:
            self.logger.error(f"Error processing relative date '{date_text}': {e}")
            return datetime.now()  # Fallback to current date

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

        # Default to today if date is None
        if published_at is None:
            published_at = datetime.now()
            self.logger.warning(f"Using current date for article: {title}")

        # Truncate overly long summaries
        if len(summary) > 1000:
            summary = summary[:997] + "..."

        # Analyze sentiment if analyzer is available
        sentiment_score = None
        if self.NewsAnalyzer:
            try:
                text_to_analyze = f"{title} {summary}"
                sentiment_score = self.NewsAnalyzer.analyze_news_sentiment(text_to_analyze)
            except Exception as e:
                self.logger.error(f"Sentiment analysis error: {e}")

        # Get topics if available
        topics = None
        if hasattr(self, 'NewsAnalyzer') and self.NewsAnalyzer:
            try:
                topics = self.NewsAnalyzer.extract_topics(f"{title} {summary}")
            except Exception as e:
                self.logger.error(f"Topic extraction error: {e}")

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

            # Get base URL and categories from source config
            source_config = self.source_config['coindesk']
            base_url = source_config['base_url']
            if not categories:
                categories = source_config['default_categories']

            selectors = source_config['selectors']
            js_rendering_required = source_config['js_rendering_required']

            # Оновлений селектор для посилань
            link_selector = "a.hover\\:underline.text-color-charcoal-900"

            for category in categories:
                page = 1
                continue_scraping = True

                while continue_scraping and page <= self.max_pages:
                    # Construct URL for the category and page
                    url = f"{base_url}/{category}?page={page}"
                    self.logger.debug(f"Fetching CoinDesk URL: {url}")

                    html_content = None

                    # Check if JS rendering is required
                    if js_rendering_required:
                        self.logger.info(f"Using JS rendering for URL: {url}")
                        self.request_stats['js_rendering_required'] += 1
                        html_content = self._fetch_page_with_javascript(url, selectors['article'])

                    # If JS rendering failed or not required, use regular request
                    if not html_content:
                        response = self._make_request(url)
                        if not response:
                            self.logger.warning(f"Failed to get response from {url}")
                            continue_scraping = False
                            continue
                        html_content = response.text

                    # Parse the HTML
                    soup = BeautifulSoup(html_content, 'html.parser')

                    # Use the configured selectors to find articles
                    articles = soup.select(selectors['article'])

                    if not articles:
                        self.logger.warning(f"No articles found at {url} using selector: {selectors['article']}")
                        continue_scraping = False
                        continue

                    self.logger.info(f"Found {len(articles)} articles on page {page} for category {category}")

                    for article in articles:
                        try:
                            # Extract title
                            title_elem = article.select_one(selectors['title'])
                            if not title_elem:
                                continue
                            title = title_elem.text.strip()

                            # Extract link using the updated selector
                            link_elem = article.select_one(link_selector)
                            if not link_elem or not link_elem.has_attr('href'):
                                # Fallback to original selector if new one doesn't work
                                link_elem = article.select_one(selectors['link'])
                                if not link_elem or not link_elem.has_attr('href'):
                                    continue
                            link = link_elem['href']

                            # Handle relative URLs
                            if link and not link.startswith('http'):
                                link = base_url + link

                            # Extract date
                            date_elem = article.select_one(selectors['date'])
                            pub_date = None

                            if date_elem:
                                # First try to get datetime attribute
                                if date_elem.has_attr('datetime'):
                                    try:
                                        pub_date = datetime.fromisoformat(date_elem['datetime'].replace('Z', '+00:00'))
                                    except (ValueError, TypeError):
                                        pass

                                # If that fails, try parsing the text content
                                if not pub_date:
                                    date_text = date_elem.text.strip()
                                    pub_date = self._parse_relative_date(date_text)

                            # Default to current time if no date found
                            if not pub_date:
                                pub_date = datetime.now()

                            # Check if article is within the time period
                            if pub_date < start_date:
                                self.logger.debug(f"Article date {pub_date} is before start date {start_date}")
                                continue_scraping = False
                                break

                            # Extract summary
                            summary_elem = article.select_one(selectors['summary'])
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
                            self.logger.debug(f"Added article: {title}")

                        except Exception as e:
                            self.logger.error(f"Error processing CoinDesk article: {str(e)[:200]}")

                    # Go to next page
                    page += 1

                    # Delay to prevent blocking
                    delay = randint(1, 3)
                    self.logger.debug(f"Waiting {delay} seconds before next request")
                    time.sleep(delay)

        except Exception as e:
            self.logger.error(f"General error while scraping CoinDesk: {str(e)[:200]}")

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
                    # Updated URL format
                    url = f"{base_url}/{category}?page={page}"
                    if category == "news":
                        url = f"{base_url}?page={page}"  # News is at the root URL

                    response = self._make_request(url)

                    if not response:
                        continue_scraping = False
                        continue

                    soup = BeautifulSoup(response.text, 'html.parser')
                    # Updated selector based on current Cointelegraph structure
                    articles = soup.select('article.post-card')

                    if not articles:
                        continue_scraping = False
                        continue

                    for article in articles:
                        try:
                            # Get publication date
                            date_elem = article.select_one('time.post-card__date')
                            if not date_elem:
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
                            title_elem = article.select_one('span.post-card__title, h2.post-card__title')
                            if not title_elem:
                                continue

                            title = title_elem.text.strip()
                            link_elem = article.select_one('a.post-card__title-link, a.post-card__link')
                            link = link_elem['href'] if link_elem else None

                            # Handle relative URLs
                            if link and not link.startswith('http'):
                                link = base_url + link

                            # Get summary
                            summary_elem = article.select_one('p.post-card__text, .post-card__content-text')
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
            # Updated URL format for Decrypt (2025) - Based on actual URL structure
            if category == "news":
                return f"{base_url}/news/cryptocurrencies?page={page}"  # Main crypto news
            elif category == "business":
                return f"{base_url}/news/business?page={page}"  # Business news
            elif category == "technology":
                return f"{base_url}/news/technology?page={page}"  # Technology news
            elif category == "defi":
                return f"{base_url}/news/defi?page={page}"  # DeFi news
            elif category == "artificial-intelligence":
                return f"{base_url}/news/artificial-intelligence?page={page}"  # AI news
            else:
                return f"{base_url}/news/{category}?page={page}"  # Generic subcategory format

        return self._scrape_source_with_config(
            source='decrypt',
            days_back=days_back,
            categories=categories,
            article_selector='article.article-card, div.post-card',
            title_selector='h3.article-title, h2.post-title',
            link_selector='a.article-link, a.post-link',
            date_selector='time.article-date, span.post-date',
            summary_selector='p.article-description, div.post-excerpt',
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
            # Updated URL format for The Block
            if category == "news":
                return f"{base_url}/latest-crypto-news?page={page}"
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

                # Ensure we're returning a list of NewsItem objects
                if not isinstance(news_from_source, list):
                    self.logger.error(
                        f"Expected a list of NewsItem objects from {source}, but got {type(news_from_source)}")
                    return []

                self.logger.info(f"Successfully collected {len(news_from_source)} news items from {source}")
                return news_from_source
            except Exception as e:
                self.logger.error(f"Error collecting news from {source}: {e}")
                return []

        # Use ThreadPoolExecutor for parallel scraping
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(sources_to_scrape))) as executor:
            results = list(executor.map(scrape_source_worker, sources_to_scrape))

        # Flatten results
        for result in results:
            if isinstance(result, list):
                all_news.extend(result)
            else:
                self.logger.error(f"Unexpected result type: {type(result)}. Expected list of NewsItem.")

        # Remove duplicates based on title
        unique_news = []
        seen_titles = set()

        for news_item in all_news:
            if isinstance(news_item, NewsItem) and news_item.title not in seen_titles:
                seen_titles.add(news_item.title)
                unique_news.append(news_item)

        self.logger.info(f"Total collected {len(unique_news)} unique news items from all sources")
        return unique_news


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

