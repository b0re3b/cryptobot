import logging
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta
import argparse
import sys
import json
from data_collection.NewsCollector import NewsCollector, NewsItem as CollectorNewsItem
from models.NewsAnalyzer import BERTNewsAnalyzer, NewsItem as AnalyzerNewsItem


class CryptoNewsScraper:
    """Main class for crypto news scraping and analysis pipeline."""

    def __init__(self,
                 news_sources: List[str] = None,
                 bert_model_name: str = 'bert-base-uncased',
                 sentiment_model_name: str = 'nlptown/bert-base-multilingual-uncased-sentiment',
                 db_manager=None,
                 logger: Optional[logging.Logger] = None,
                 max_pages: int = 5,
                 max_workers: int = 5,
                 topic_model_dir: str = './models'):
        """
        Initialize the crypto news scraper with integrated collection and analysis.

        Args:
            news_sources: List of news sources to scrape
            bert_model_name: Name of BERT model for analysis
            sentiment_model_name: Name of sentiment analysis model
            db_manager: Database manager instance
            logger: Custom logger instance
            max_pages: Maximum pages to scrape per source
            max_workers: Maximum concurrent workers
            topic_model_dir: Directory for topic model storage
        """
        # Configure logger
        self.logger = logger or logging.getLogger('CryptoNewsScraper')
        if not logger:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.db_manager = db_manager
        self.topic_model_dir = topic_model_dir

        # Initialize components
        self.logger.info("Initializing News Collector...")
        self.collector = NewsCollector(
            news_sources=news_sources,
            logger=self.logger,
            db_manager=db_manager,
            max_pages=max_pages,
            max_workers=max_workers,
            topic_model_dir=topic_model_dir
        )

        self.logger.info("Initializing News Analyzer...")
        self.analyzer = BERTNewsAnalyzer(
            bert_model_name=bert_model_name,
            sentiment_model_name=sentiment_model_name,
            db_manager=db_manager,
            logger=self.logger,
            topic_model_dir=topic_model_dir
        )

        # Link components
        self.collector.sentiment_analyzer = self.analyzer

    def scrape_and_analyze(self,
                           days_back: int = 1,
                           categories: List[str] = None,
                           save_results: bool = True) -> List[Dict]:
        """
        Full pipeline: scrape news from all sources and analyze them.

        Args:
            days_back: Number of days to look back for news
            categories: List of categories to filter by
            save_results: Whether to save results to database

        Returns:
            List of processed news items with analysis
        """
        self.logger.info(f"Starting full scraping and analysis pipeline for last {days_back} days")

        try:
            # Step 1: Collect news
            self.logger.info("Collecting news from all sources...")
            news_items = self.collector.scrape_all_sources(days_back=days_back, categories=categories)

            # Convert to analyzer format if needed
            analyzer_items = []
            for item in news_items:
                if isinstance(item, CollectorNewsItem):
                    # Convert collector NewsItem to analyzer NewsItem
                    analyzer_item = AnalyzerNewsItem(
                        title=item.title,
                        url=item.link,
                        source=item.source,
                        published_at=item.published_at,
                        content=item.summary,
                        sentiment_score=item.sentiment_score,
                        categories=[item.category] if item.category else []
                    )
                    analyzer_items.append(analyzer_item)
                else:
                    # Assume it's already in dict format compatible with analyzer
                    analyzer_items.append(item)

            # Step 2: Analyze news
            self.logger.info(f"Analyzing {len(analyzer_items)} news items...")
            analyzed_news = self.analyzer.analyze_news_batch(analyzer_items, save_results=save_results)

            self.logger.info("Scraping and analysis completed successfully")
            return analyzed_news

        except Exception as e:
            self.logger.error(f"Error in scraping and analysis pipeline: {e}")
            raise

    def scrape_single_source(self,
                             source: str,
                             days_back: int = 1,
                             categories: List[str] = None,
                             analyze: bool = True,
                             save_results: bool = True) -> List[Dict]:
        """
        Scrape news from a single source and optionally analyze them.

        Args:
            source: News source to scrape
            days_back: Number of days to look back for news
            categories: List of categories to filter by
            analyze: Whether to perform analysis
            save_results: Whether to save results to database

        Returns:
            List of processed news items
        """
        self.logger.info(f"Scraping from single source: {source}")

        try:
            # Scrape from specific source
            news_items = self.collector._scrape_source(source, days_back, categories)

            if not analyze:
                return [item.to_dict() if isinstance(item, CollectorNewsItem) else item for item in news_items]

            # Convert to analyzer format if needed
            analyzer_items = []
            for item in news_items:
                if isinstance(item, CollectorNewsItem):
                    analyzer_item = AnalyzerNewsItem(
                        title=item.title,
                        url=item.link,
                        source=item.source,
                        published_at=item.published_at,
                        content=item.summary,
                        sentiment_score=item.sentiment_score,
                        categories=[item.category] if item.category else []
                    )
                    analyzer_items.append(analyzer_item)
                else:
                    analyzer_items.append(item)

            # Analyze news
            analyzed_news = self.analyzer.analyze_news_batch(analyzer_items, save_results=save_results)
            return analyzed_news

        except Exception as e:
            self.logger.error(f"Error scraping from {source}: {e}")
            raise

    def analyze_existing_news(self,
                              news_data: List[Union[Dict, CollectorNewsItem]],
                              save_results: bool = True) -> List[Dict]:
        """
        Analyze existing news items without scraping.

        Args:
            news_data: List of news items to analyze
            save_results: Whether to save results to database

        Returns:
            List of analyzed news items
        """
        self.logger.info(f"Analyzing {len(news_data)} existing news items")

        try:
            # Convert to analyzer format if needed
            analyzer_items = []
            for item in news_data:
                if isinstance(item, CollectorNewsItem):
                    analyzer_item = AnalyzerNewsItem(
                        title=item.title,
                        url=item.link,
                        source=item.source,
                        published_at=item.published_at,
                        content=item.summary,
                        sentiment_score=item.sentiment_score,
                        categories=[item.category] if item.category else []
                    )
                    analyzer_items.append(analyzer_item)
                else:
                    analyzer_items.append(item)

            # Analyze news
            analyzed_news = self.analyzer.analyze_news_batch(analyzer_items, save_results=save_results)
            return analyzed_news

        except Exception as e:
            self.logger.error(f"Error analyzing existing news: {e}")
            raise

    def get_sentiment_timeseries(self,
                                 days_back: int = 7,
                                 source: Optional[str] = None) -> Dict:
        """
        Get sentiment time series data from database.

        Args:
            days_back: Number of days to look back
            source: Optional source to filter by

        Returns:
            Dictionary of sentiment time series data
        """
        if not self.db_manager:
            self.logger.warning("No database manager configured")
            return {}

        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            return self.db_manager.get_sentiment_timeseries(start_date, end_date, source)

        except Exception as e:
            self.logger.error(f"Error getting sentiment time series: {e}")
            return {}

    def get_top_coins_mentions(self,
                               days_back: int = 7,
                               limit: int = 10) -> List[Dict]:
        """
        Get most mentioned cryptocurrencies.

        Args:
            days_back: Number of days to look back
            limit: Maximum number of coins to return

        Returns:
            List of coins with mention counts
        """
        if not self.db_manager:
            self.logger.warning("No database manager configured")
            return []

        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            return self.db_manager.get_top_mentioned_coins(start_date, end_date, limit)

        except Exception as e:
            self.logger.error(f"Error getting top coins mentions: {e}")
            return []

    def initialize_reddit(self,
                          client_id: str,
                          client_secret: str,
                          user_agent: str) -> bool:
        """
        Initialize Reddit API connection.

        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            user_agent: Reddit API user agent

        Returns:
            True if initialization succeeded
        """
        return self.collector.initialize_reddit(client_id, client_secret, user_agent)

    @staticmethod
    def main():
        """
        Main function to execute when running the scraper from the command line.
        Collects and analyzes news from all configured sources.
        """
        # Set up command-line argument parsing
        parser = argparse.ArgumentParser(description='Crypto News Scraper and Analyzer')
        parser.add_argument('--days', type=int, default=3, help='Number of days to look back for news (default: 3)')
        parser.add_argument('--sources', type=str, help='Comma-separated list of news sources to scrape')
        parser.add_argument('--categories', type=str, help='Comma-separated list of categories to filter by')
        parser.add_argument('--output', type=str, help='Output file path for saving results (JSON format)')
        parser.add_argument('--max-pages', type=int, default=5, help='Maximum pages to scrape per source (default: 5)')
        parser.add_argument('--max-workers', type=int, default=5, help='Maximum concurrent workers (default: 5)')
        parser.add_argument('--no-save', action='store_true', help='Do not save results to database')
        parser.add_argument('--reddit', action='store_true', help='Include Reddit scraping')
        parser.add_argument('--reddit-config', type=str, help='Path to Reddit API config JSON file')
        parser.add_argument('--model-dir', type=str, default='./models', help='Directory for topic model storage')
        parser.add_argument('--log-level', type=str, default='INFO',
                            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                            help='Logging level')

        args = parser.parse_args()

        # Set up logging
        logger = logging.getLogger('CryptoNewsScraper')
        logger.setLevel(getattr(logging, args.log_level))
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Process arguments
        news_sources = args.sources.split(',') if args.sources else None
        categories = args.categories.split(',') if args.categories else None
        save_results = not args.no_save

        try:
            # Initialize scraper
            logger.info("Initializing CryptoNewsScraper")
            scraper = CryptoNewsScraper(
                news_sources=news_sources,
                logger=logger,
                max_pages=args.max_pages,
                max_workers=args.max_workers,
                topic_model_dir=args.model_dir
            )

            # Initialize Reddit if needed
            if args.reddit:
                if args.reddit_config:
                    try:
                        with open(args.reddit_config, 'r') as f:
                            reddit_config = json.load(f)

                        reddit_initialized = scraper.initialize_reddit(
                            client_id=reddit_config.get('client_id'),
                            client_secret=reddit_config.get('client_secret'),
                            user_agent=reddit_config.get('user_agent')
                        )

                        if not reddit_initialized:
                            logger.warning("Failed to initialize Reddit API")
                    except Exception as e:
                        logger.error(f"Error reading Reddit config: {e}")
                else:
                    logger.warning("Reddit scraping requested but no config provided")

            # Execute the scraping and analysis pipeline
            logger.info(f"Starting scraping and analysis for the last {args.days} days")
            results = scraper.scrape_and_analyze(
                days_back=args.days,
                categories=categories,
                save_results=save_results
            )

            # Output results
            if results:
                logger.info(f"Successfully processed {len(results)} news items")

                # Save to file if requested
                if args.output:
                    with open(args.output, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
                    logger.info(f"Results saved to {args.output}")

                # Print summary statistics
                total_sentiment = sum(item.get('sentiment_score', 0) for item in results)
                avg_sentiment = total_sentiment / len(results) if results else 0
                sources_count = {}
                for item in results:
                    source = item.get('source', 'unknown')
                    sources_count[source] = sources_count.get(source, 0) + 1

                logger.info(f"Average sentiment score: {avg_sentiment:.2f}")
                logger.info("Source distribution:")
                for source, count in sources_count.items():
                    logger.info(f"  - {source}: {count} articles")
            else:
                logger.warning("No news items were processed")

            return 0  # Success exit code

        except Exception as e:
            logger.error(f"An error occurred in the main execution: {e}", exc_info=True)
            return 1  # Error exit code


# Execute main function if the script is run directly
if __name__ == "__main__":
    sys.exit(CryptoNewsScraper.main())