import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from data.db import DatabaseManager


class CryptoNewsScraper:
    """
    A comprehensive cryptocurrency news scraper that integrates collection, analysis, and storage functionality.

    This class combines the functionality of NewsCollector, NewsAnalyzer, and NewsStorage
    to provide a complete pipeline for cryptocurrency news processing.
    """

    def __init__(self,
                 db_manager: DatabaseManager,
                 news_sources: List[str] = None,
                 max_pages: int = 5,
                 max_workers: int = 5,
                 sentiment_model_name: str = 'nlptown/bert-base-multilingual-uncased-sentiment',
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the CryptoNewsScraper.

        Args:
            db_manager: DatabaseManager instance for data storage
            news_sources: List of news sources to scrape (default: all available)
            max_pages: Maximum number of pages to scrape per source
            max_workers: Maximum number of concurrent workers for scraping
            sentiment_model_name: Name of the BERT sentiment analysis model
            logger: Optional logger instance
        """
        # Initialize logger
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger('crypto_news_scraper')
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Initialize components
        from models.NewsAnalyzer import BERTNewsAnalyzer
        from data.NewsManager import NewsStorage
        from data_collection.NewsCollector import NewsCollector

        self.news_analyzer = BERTNewsAnalyzer(
            sentiment_model_name=sentiment_model_name,
            db_manager=db_manager,
            logger=self.logger
        )

        self.news_storage = NewsStorage(
            db_manager=db_manager,
            logger=self.logger
        )

        self.news_collector = NewsCollector(
            news_sources=news_sources,
            sentiment_analyzer=self.news_analyzer,
            logger=self.logger,
            db_manager=db_manager,
            max_pages=max_pages,
            max_workers=max_workers
        )

        self.db_manager = db_manager

    def scrape_and_process(self,
                           days_back: int = 1,
                           categories: List[str] = None,
                           analyze_sentiment: bool = True,
                           extract_coins: bool = True,
                           calculate_importance: bool = True,
                           extract_topics: bool = True) -> Dict[str, Any]:
        """
        Complete pipeline: scrape news, analyze them, and store in database.

        Args:
            days_back: Number of days to look back for news
            categories: List of categories to scrape (default: all for each source)
            analyze_sentiment: Whether to perform sentiment analysis
            extract_coins: Whether to extract mentioned cryptocurrencies
            calculate_importance: Whether to calculate importance scores
            extract_topics: Whether to extract topics

        Returns:
            Dictionary with processing statistics
        """
        self.logger.info(f"Starting crypto news scraping and processing for last {days_back} days")

        try:
            # Step 1: Scrape news from all sources
            news_items = self.news_collector.scrape_all_sources(
                days_back=days_back,
                categories=categories
            )

            if not news_items:
                self.logger.warning("No news items were collected")
                return {'status': 'error', 'message': 'No news items collected'}

            self.logger.info(f"Collected {len(news_items)} news items")

            # Step 2: Analyze news
            analyzed_news = self.news_analyzer.analyze_news_batch(
                news_items,
                extract_sentiment=analyze_sentiment,
                extract_coins=extract_coins,
                calculate_importance=calculate_importance,
                extract_topics=extract_topics
            )

            self.logger.info(f"Finished analyzing {len(analyzed_news)} news items")

            # Step 3: Store news in database
            storage_results = self.news_storage.store_news_collector_data(analyzed_news)

            self.logger.info(
                f"Storage completed: {storage_results['success']} successful, {storage_results['failure']} failed")

            # Step 4: Return comprehensive results
            return {
                'status': 'success',
                'scraped_count': len(news_items),
                'analyzed_count': len(analyzed_news),
                'storage_results': storage_results,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error in scrape_and_process: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def get_trending_topics(self, hours_back: int = 24, top_n: int = 5) -> Dict[str, Any]:
        """
        Get trending topics from recently stored news.

        Args:
            hours_back: Look back period in hours
            top_n: Number of top topics to return

        Returns:
            Dictionary with trending topics analysis
        """
        try:
            # Get recent news from database
            recent_news = self.db_manager.get_recent_news(hours=hours_back)

            if not recent_news:
                return {
                    'status': 'success',
                    'message': 'No recent news found',
                    'timestamp': datetime.now().isoformat()
                }

            # Convert to format expected by analyzer
            news_list = []
            for news in recent_news:
                news_dict = {
                    'title': news.title,
                    'summary': news.summary,
                    'content': news.content,
                    'published_at': news.published_at,
                    'source': news.source.name,
                    'sentiment_score': news.sentiment_score,
                    'importance_score': news.importance_score,
                    'mentioned_coins': {coin.symbol: coin.name for coin in news.mentioned_coins},
                    'topics': news.topics
                }
                news_list.append(news_dict)

            # Get trending topics
            trends = self.news_analyzer.get_trending_topics(news_list, top_n=top_n)

            return {
                'status': 'success',
                'trends': trends,
                'news_count': len(news_list),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error in get_trending_topics: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def identify_market_signals(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Identify potential market signals from recent news.

        Args:
            hours_back: Look back period in hours

        Returns:
            Dictionary with market signals analysis
        """
        try:
            # Get recent news from database
            recent_news = self.db_manager.get_recent_news(hours=hours_back)

            if not recent_news:
                return {
                    'status': 'success',
                    'message': 'No recent news found',
                    'timestamp': datetime.now().isoformat()
                }

            # Convert to format expected by analyzer
            news_list = []
            for news in recent_news:
                news_dict = {
                    'title': news.title,
                    'summary': news.summary,
                    'content': news.content,
                    'published_at': news.published_at,
                    'source': news.source.name,
                    'sentiment_score': news.sentiment_score,
                    'importance_score': news.importance_score,
                    'mentioned_coins': {coin.symbol: coin.name for coin in news.mentioned_coins},
                    'topics': news.topics
                }
                news_list.append(news_dict)

            # Get market signals
            signals = self.news_analyzer.identify_market_signals(news_list)

            return {
                'status': 'success',
                'signals': signals,
                'news_count': len(news_list),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error in identify_market_signals: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def analyze_sentiment_trends(self,
                                 hours_back: int = 24,
                                 time_window_hours: int = 6) -> Dict[str, Any]:
        """
        Analyze sentiment trends over time for cryptocurrencies.

        Args:
            hours_back: Total look back period in hours
            time_window_hours: Size of each analysis window in hours

        Returns:
            Dictionary with sentiment trends analysis
        """
        try:
            # Get recent news from database
            recent_news = self.db_manager.get_recent_news(hours=hours_back)

            if not recent_news:
                return {
                    'status': 'success',
                    'message': 'No recent news found',
                    'timestamp': datetime.now().isoformat()
                }

            # Convert to format expected by analyzer
            news_list = []
            for news in recent_news:
                news_dict = {
                    'title': news.title,
                    'summary': news.summary,
                    'content': news.content,
                    'published_at': news.published_at,
                    'source': news.source.name,
                    'sentiment_score': news.sentiment_score,
                    'importance_score': news.importance_score,
                    'mentioned_coins': {coin.symbol: coin.name for coin in news.mentioned_coins},
                    'topics': news.topics
                }
                news_list.append(news_dict)

            # Analyze sentiment trends
            trends = self.news_analyzer.analyze_sentiment_trends(
                news_list,
                time_window_hours=time_window_hours
            )

            return {
                'status': 'success',
                'trends': trends,
                'news_count': len(news_list),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error in analyze_sentiment_trends: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def run_scheduled_scrape(self,
                             interval_hours: int = 6,
                             days_back: int = 1,
                             analyze_sentiment: bool = True,
                             extract_coins: bool = True,
                             calculate_importance: bool = True,
                             extract_topics: bool = True) -> Dict[str, Any]:
        """
        Run a scheduled scrape with the given interval and parameters.

        Args:
            interval_hours: Hours since last scrape to consider
            days_back: Days to look back when scraping
            analyze_sentiment: Whether to perform sentiment analysis
            extract_coins: Whether to extract mentioned cryptocurrencies
            calculate_importance: Whether to calculate importance scores
            extract_topics: Whether to extract topics

        Returns:
            Dictionary with processing statistics
        """
        try:
            # Check when we last scraped
            last_scrape = self.db_manager.get_last_scrape_time()

            if last_scrape and (datetime.now() - last_scrape) < timedelta(hours=interval_hours):
                return {
                    'status': 'skipped',
                    'message': f"Last scrape was at {last_scrape}. Next scrape in {interval_hours} hours.",
                    'timestamp': datetime.now().isoformat()
                }

            # Run the full scrape and process pipeline
            results = self.scrape_and_process(
                days_back=days_back,
                analyze_sentiment=analyze_sentiment,
                extract_coins=extract_coins,
                calculate_importance=calculate_importance,
                extract_topics=extract_topics
            )

            # Update last scrape time
            self.db_manager.update_last_scrape_time(datetime.now())

            return results

        except Exception as e:
            self.logger.error(f"Error in run_scheduled_scrape: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }


def main():
    """
    Main function to demonstrate the usage of CryptoNewsScraper.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('crypto_news_main')

    try:
        db_manager = DatabaseManager()

        # Define news sources to scrape
        news_sources = [
            'coindesk',
            'cointelegraph',
            'cryptonews',
            'bitcoin.com',
            'decrypt'
        ]

        # Initialize the scraper
        scraper = CryptoNewsScraper(
            db_manager=db_manager,
            news_sources=news_sources,
            max_pages=3,
            max_workers=3
        )

        # Option 1: Run an immediate scrape
        logger.info("Running immediate scrape and analysis...")
        results = scraper.scrape_and_process(
            days_back=2,
            analyze_sentiment=True,
            extract_coins=True,
            calculate_importance=True,
            extract_topics=True
        )

        logger.info(f"Scrape results: {results}")

        # Option 2: Get trending topics from recent news
        logger.info("Analyzing trending topics...")
        trending = scraper.get_trending_topics(hours_back=48, top_n=10)
        logger.info(f"Trending topics: {trending}")

        # Option 3: Get market signals
        logger.info("Identifying market signals...")
        signals = scraper.identify_market_signals(hours_back=48)
        logger.info(f"Market signals: {signals}")

        # Option 4: Analyze sentiment trends
        logger.info("Analyzing sentiment trends...")
        sentiment_trends = scraper.analyze_sentiment_trends(
            hours_back=48,
            time_window_hours=6
        )
        logger.info(f"Sentiment trends: {sentiment_trends}")

        # Option 5: Run as a scheduled task
        logger.info("Running scheduled scrape (if needed)...")
        scheduled_results = scraper.run_scheduled_scrape(
            interval_hours=6,
            days_back=1
        )
        logger.info(f"Scheduled scrape results: {scheduled_results}")

        logger.info("CryptoNewsScraper demonstration completed successfully")

    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)


if __name__ == "__main__":
    main()