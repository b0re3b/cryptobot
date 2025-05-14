import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
from data.db import DatabaseManager
from collections import defaultdict
import re
from utils.config import CRYPTO_KEYWORDS

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
class NewsStorage:
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

    def __init__(self, db_manager: DatabaseManager, logger: Optional[logging.Logger] = None):
        """
        Initialize the NewsStorage class.

        Args:
            db_manager: An instance of DatabaseManager for database operations
            logger: Optional logger for logging
        """
        self.db_manager = db_manager

        # Configure logger
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger('news_storage')
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Cache for source and category IDs to avoid redundant database queries
        self.source_id_cache = {}
        self.category_id_cache = {}
        self.CRYPTO_KEYWORDS = CRYPTO_KEYWORDS

    def get_or_create_source_id(self, source_name: str) -> int | None | Any:
        """
        Get source ID from cache or create a new source entry in the database.

        Args:
            source_name: Name of the news source

        Returns:
            The source ID
        """
        if source_name in self.source_id_cache:
            return self.source_id_cache[source_name]

        try:
            source_id = self.db_manager.save_source(source_name)
            self.source_id_cache[source_name] = source_id
            return source_id
        except Exception as e:
            self.logger.error(f"Error creating source {source_name}: {e}")
            return None

    def get_or_create_category_id(self, category_name: str) -> int | None | Any:
        """
        Get category ID from cache or create a new category entry in the database.

        Args:
            category_name: Name of the news category

        Returns:
            The category ID
        """
        if category_name in self.category_id_cache:
            return self.category_id_cache[category_name]

        try:
            category_id = self.db_manager.save_category(category_name)
            self.category_id_cache[category_name] = category_id
            return category_id
        except Exception as e:
            self.logger.error(f"Error creating category {category_name}: {e}")
            return None

    def detect_mentioned_coins(self, text: str) -> List[Dict[str, str]]:
        """
        Detect cryptocurrency mentions in the text.

        Args:
            text: Text to analyze for cryptocurrency mentions

        Returns:
            List of dictionaries with coin symbol and name
        """
        mentioned_coins = []
        text = text.upper()

        # First check for symbols
        for symbol, name in self.CRYPTO_KEYWORDS.items():
            # Use word boundary to avoid partial matches
            pattern = r'\b' + re.escape(symbol) + r'\b'
            if re.search(pattern, text):
                mentioned_coins.append({
                    'symbol': symbol,
                    'name': name
                })

        # Then check for full names (case insensitive)
        for symbol, name in self.CRYPTO_KEYWORDS.items():
            if name.upper() in text:
                # Avoid duplicates
                if not any(coin['symbol'] == symbol for coin in mentioned_coins):
                    mentioned_coins.append({
                        'symbol': symbol,
                        'name': name
                    })

        return mentioned_coins

    def process_topics(self, article_id: int, topics: List[str]) -> None:
        """
        Process and save topics for an article.

        Args:
            article_id: ID of the article
            topics: List of topics to save
        """
        if not topics:
            return

        for topic in topics:
            try:
                # Save topic and get its ID
                topic_id = self.db_manager.save_topic(topic)

                # Link topic to article
                self.db_manager.save_article_topic(article_id, topic_id)
            except Exception as e:
                self.logger.error(f"Error saving topic {topic} for article {article_id}: {e}")

    def store_news_item(self, news_item: Dict[str, Any]) -> int | None:
        """
        Store a single news item in the database.

        Args:
            news_item: Dictionary containing news item data

        Returns:
            ID of the created article or None if unsuccessful
        """
        try:
            # Get source and category IDs
            source_id = self.get_or_create_source_id(news_item['source'])
            category_id = self.get_or_create_category_id(news_item['category'])

            if not source_id or not category_id:
                self.logger.error(f"Could not get/create source or category for {news_item['title']}")
                return None

            # Safely extract values from news_item with proper fallbacks
            scraped_at = news_item.get('scraped_at', datetime.now())
            # Check if scraped_at is a string and convert if needed
            if isinstance(scraped_at, str):
                try:
                    # Try to parse the string to datetime
                    scraped_at = datetime.fromisoformat(scraped_at.replace('Z', '+00:00'))
                except ValueError:
                    # Fallback to current time if parsing fails
                    scraped_at = datetime.now()
                    self.logger.warning(f"Failed to parse scraped_at date for {news_item['title']}, using current time")

            # Prepare article data with safe access to optional fields
            article_data = {
                'title': news_item['title'],
                'summary': news_item['summary'],
                'link': news_item['link'],
                'published_at': news_item['published_at'],
                'scraped_at': scraped_at,
                'source_id': source_id,
                'category_id': category_id,
                'score': news_item.get('score'),
                'upvote_ratio': news_item.get('upvote_ratio'),
                'num_comments': news_item.get('num_comments'),
                'sentiment_score': news_item.get('sentiment_score')
            }

            # Save article and get ID
            article_id = self.db_manager.save_article(**article_data)

            if not article_id:
                self.logger.error(f"Failed to save article: {news_item['title']}")
                return None

            # Process topics if available
            if 'topics' in news_item and news_item['topics']:
                self.process_topics(article_id, news_item['topics'])

            # Detect and save mentioned coins
            combined_text = f"{news_item['title']} {news_item['summary']}"
            mentioned_coins = self.detect_mentioned_coins(combined_text)

            for coin in mentioned_coins:
                try:
                    self.db_manager.save_mentioned_coin(
                        article_id=article_id,
                        symbol=coin['symbol'],
                        name=coin['name']
                    )
                except Exception as e:
                    self.logger.error(f"Error saving mentioned coin {coin['symbol']} for article {article_id}: {e}")

            return article_id

        except Exception as e:
            self.logger.error(f"Error storing news item {news_item.get('title', 'unknown')}: {e}")
            return None

    def store_news_batch(self, news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Store a batch of news items in the database.

        Args:
            news_items: List of news item dictionaries

        Returns:
            Dictionary with success and failure counts
        """
        self.logger.info(f"Starting to store {len(news_items)} news items")

        results = {
            'total': len(news_items),
            'success': 0,
            'failure': 0,
            'article_ids': []
        }

        for item in news_items:
            article_id = self.store_news_item(item)
            if article_id:
                results['success'] += 1
                results['article_ids'].append(article_id)
            else:
                results['failure'] += 1

        self.logger.info(f"Storage completed: {results['success']} successful, {results['failure']} failed")
        return results

    def store_news_collector_data(self, news_collector_data: List) -> Dict[str, Any]:
        """
        Process and store data from a NewsCollector instance.

        Args:
            news_collector_data: List of NewsItem objects from NewsCollector

        Returns:
            Dictionary with storage statistics
        """
        # Convert NewsItem objects to dictionaries
        news_dicts = []
        for news_item in news_collector_data:
            if hasattr(news_item, 'to_dict'):
                news_dicts.append(news_item.to_dict())
            else:
                # If it's already a dict
                news_dicts.append(news_item)

        # Store the converted data
        return self.store_news_batch(news_dicts)