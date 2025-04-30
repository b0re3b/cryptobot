import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Internal imports
from data.db import DatabaseManager
from data_collection.binance_client import BinanceClient
from data_collection.market_data_processor import MarketDataProcessor
from utils.logger import setup_logger

logger = setup_logger(__name__)


class MarketCorrelation:

    DEFAULT_CONFIG = {
        'correlation_methods': ['pearson', 'kendall', 'spearman'],
        'default_correlation_method': 'pearson',
        'default_timeframe': '1d',
        'default_correlation_window': 30,
        'correlation_threshold': 0.7,
        'anticorrelation_threshold': -0.7,
        'breakdown_threshold': 0.3,
        'default_n_clusters': 3,
        'default_prediction_horizon': 1,
        'default_lookback_days': 365,
        'default_lag_periods': [1, 3, 5, 7],
        'target_portfolio_correlation': 0.3,
        'visualization': {
            'heatmap_colormap': 'coolwarm',
            'network_node_size': 300,
            'default_figsize': (12, 10)
        }
    }

    def __init__(self, db_manager: Optional[DatabaseManager] = None,
                 binance_client: Optional[BinanceClient] = None,
                 custom_config: Optional[Dict] = None):

        self.db_manager = db_manager or DatabaseManager()
        self.binance_client = binance_client or BinanceClient()
        self.data_processor = MarketDataProcessor()

        logger.info("Ініціалізація аналізатора ринкової кореляції")

        # Ініціалізація конфігурації зі значеннями за замовчуванням та перевизначення користувацькими, якщо надані
        self.config = self.DEFAULT_CONFIG.copy()
        if custom_config:
            logger.debug(f"Застосування користувацької конфігурації: {custom_config}")
            self._update_config_recursive(self.config, custom_config)

    def _update_config_recursive(self, target_dict: Dict, source_dict: Dict) -> None:

        for key, value in source_dict.items():
            if key in target_dict and isinstance(target_dict[key], dict) and isinstance(value, dict):
                self._update_config_recursive(target_dict[key], value)
            else:
                logger.debug(f"Оновлення параметра конфігурації: {key} = {value}")
                target_dict[key] = value

    def calculate_price_correlation(self, symbols: List[str],
                                    timeframe: str = None,
                                    start_time: Optional[datetime] = None,
                                    end_time: Optional[datetime] = None,
                                    method: str = None) -> pd.DataFrame:

        # Використання значень за замовчуванням з конфігурації, якщо не вказані
        timeframe = timeframe or self.config['default_timeframe']
        method = method or self.config['default_correlation_method']

        logger.info(f"Розрахунок кореляції цін для {len(symbols)} символів з таймфреймом {timeframe}")

        # Встановлення кінцевого часу як поточний, якщо не вказано
        end_time = end_time or datetime.now()

        # Якщо початковий час не вказано, використовуємо значення за замовчуванням
        if start_time is None:
            lookback_days = self.config['default_lookback_days']
            start_time = end_time - timedelta(days=lookback_days)
            logger.debug(f"Використання періоду за замовчуванням: {lookback_days} днів")

        try:
            # Отримання цінових даних для всіх символів з бази даних замість Binance API
            price_data = {}
            for symbol in symbols:
                logger.debug(f"Отримання даних для {symbol}")
                # Використовуємо db_manager.get_klines замість binance_client.get_historical_prices
                price_data[symbol] = self.db_manager.get_klines(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time
                )

            # Створення датафрейму з цінами закриття для всіх символів
            df = pd.DataFrame()
            for symbol, data in price_data.items():
                df[symbol] = data['close']

            # Перевірка на відсутні дані
            if df.isnull().values.any():
                missing_count = df.isnull().sum().sum()
                logger.warning(f"Виявлено {missing_count} відсутніх значень. Заповнення методом forward fill")
                df = df.fillna(method='ffill')

            # Розрахунок матриці кореляції
            correlation_matrix = df.corr(method=method)

            logger.info(f"Матриця кореляції цін успішно розрахована з використанням методу {method}")

            # Збереження результатів у базу даних
            self.save_correlation_to_db(
                correlation_matrix=correlation_matrix,
                correlation_type='price',
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                method=method
            )

            return correlation_matrix

        except Exception as e:
            logger.error(f"Помилка при розрахунку кореляції цін: {str(e)}")
            raise

    def calculate_volume_correlation(self, symbols: List[str],
                                     timeframe: str = None,
                                     start_time: Optional[datetime] = None,
                                     end_time: Optional[datetime] = None,
                                     method: str = None) -> pd.DataFrame:

        # Використання значень за замовчуванням з конфігурації, якщо не вказані
        timeframe = timeframe or self.config['default_timeframe']
        method = method or self.config['default_correlation_method']

        logger.info(f"Розрахунок кореляції об'ємів торгівлі для {len(symbols)} символів з таймфреймом {timeframe}")

        # Встановлення кінцевого часу як поточний, якщо не вказано
        end_time = end_time or datetime.now()

        # Якщо початковий час не вказано, використовуємо значення за замовчуванням
        if start_time is None:
            lookback_days = self.config['default_lookback_days']
            start_time = end_time - timedelta(days=lookback_days)
            logger.debug(f"Використання періоду за замовчуванням: {lookback_days} днів")

        try:
            # Отримання даних про об'єми торгівлі для всіх символів з бази даних
            volume_data = {}
            for symbol in symbols:
                logger.debug(f"Отримання даних про об'єми для {symbol}")
                # Використовуємо db_manager.get_klines замість binance_client.get_historical_prices
                volume_data[symbol] = self.db_manager.get_klines(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time
                )

            # Створення датафрейму з об'ємами торгівлі для всіх символів
            df = pd.DataFrame()
            for symbol, data in volume_data.items():
                df[symbol] = data['volume']

            # Перевірка на відсутні дані
            if df.isnull().values.any():
                missing_count = df.isnull().sum().sum()
                logger.warning(f"Виявлено {missing_count} відсутніх значень об'єму. Заповнення методом forward fill")
                df = df.fillna(method='ffill')

            # Фільтрація даних для усунення викидів
            for column in df.columns:
                mean_val = df[column].mean()
                std_val = df[column].std()
                upper_limit = mean_val + 3 * std_val

                # Заміна значних викидів середнім значенням
                outliers_mask = df[column] > upper_limit
                if outliers_mask.any():
                    outlier_count = outliers_mask.sum()
                    logger.warning(
                        f"Виявлено {outlier_count} викидів об'єму для {column}. Заміна значеннями за медіаною")
                    df.loc[outliers_mask, column] = df[column].median()

            # Розрахунок матриці кореляції
            correlation_matrix = df.corr(method=method)

            logger.info(f"Матриця кореляції об'ємів торгівлі успішно розрахована з використанням методу {method}")

            # Збереження результатів у базу даних
            self.save_correlation_to_db(
                correlation_matrix=correlation_matrix,
                correlation_type='volume',
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                method=method
            )

            return correlation_matrix

        except Exception as e:
            logger.error(f"Помилка при розрахунку кореляції об'ємів торгівлі: {str(e)}")
            raise

    def calculate_returns_correlation(self, symbols: List[str],
                                      timeframe: str = None,
                                      start_time: Optional[datetime] = None,
                                      end_time: Optional[datetime] = None,
                                      period: int = 1,
                                      method: str = None) -> pd.DataFrame:

        # Використання значень за замовчуванням з конфігурації, якщо не вказані
        timeframe = timeframe or self.config['default_timeframe']
        method = method or self.config['default_correlation_method']

        logger.info(
            f"Розрахунок кореляції доходності для {len(symbols)} символів з таймфреймом {timeframe} та періодом {period}")

        # Встановлення кінцевого часу як поточний, якщо не вказано
        end_time = end_time or datetime.now()

        # Якщо початковий час не вказано, використовуємо значення за замовчуванням
        if start_time is None:
            lookback_days = self.config['default_lookback_days']
            start_time = end_time - timedelta(days=lookback_days)
            logger.debug(f"Використання періоду за замовчуванням: {lookback_days} днів")

        try:
            # Отримання цінових даних для всіх символів з бази даних
            price_data = {}
            for symbol in symbols:
                logger.debug(f"Отримання даних для {symbol}")
                # Використовуємо db_manager.get_klines замість binance_client.get_historical_prices
                price_data[symbol] = self.db_manager.get_klines(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time
                )

            # Створення датафрейму з цінами закриття для всіх символів
            df = pd.DataFrame()
            for symbol, data in price_data.items():
                df[symbol] = data['close']

            # Перевірка на відсутні дані
            if df.isnull().values.any():
                missing_count = df.isnull().sum().sum()
                logger.warning(f"Виявлено {missing_count} відсутніх значень. Заповнення методом forward fill")
                df = df.fillna(method='ffill')

            # Розрахунок доходності
            returns_df = df.pct_change(period)

            # Видалення перших рядків, які містять NaN через обчислення доходності
            returns_df = returns_df.iloc[period:]

            # Розрахунок матриці кореляції доходності
            correlation_matrix = returns_df.corr(method=method)

            logger.info(f"Матриця кореляції доходності успішно розрахована з використанням методу {method}")

            # Збереження результатів у базу даних
            self.save_correlation_to_db(
                correlation_matrix=correlation_matrix,
                correlation_type='returns',
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                method=method
            )

            return correlation_matrix

        except Exception as e:
            logger.error(f"Помилка при розрахунку кореляції доходності: {str(e)}")
            raise

    def calculate_volatility_correlation(self, symbols: List[str],
                                         timeframe: str = None,
                                         start_time: Optional[datetime] = None,
                                         end_time: Optional[datetime] = None,
                                         window: int = None,
                                         method: str = None) -> pd.DataFrame:

        # Використання значень за замовчуванням з конфігурації, якщо не вказані
        timeframe = timeframe or self.config['default_timeframe']
        window = window or self.config['default_correlation_window']
        method = method or self.config['default_correlation_method']

        logger.info(
            f"Розрахунок кореляції волатильності для {len(symbols)} символів з таймфреймом {timeframe} та вікном {window}")

        # Встановлення кінцевого часу як поточний, якщо не вказано
        end_time = end_time or datetime.now()

        # Якщо початковий час не вказано, використовуємо значення за замовчуванням
        if start_time is None:
            lookback_days = self.config['default_lookback_days']
            start_time = end_time - timedelta(days=lookback_days)
            logger.debug(f"Використання періоду за замовчуванням: {lookback_days} днів")

        try:
            # Отримання цінових даних для всіх символів з бази даних
            price_data = {}
            for symbol in symbols:
                logger.debug(f"Отримання даних для {symbol}")
                # Використовуємо db_manager.get_klines замість binance_client.get_historical_prices
                price_data[symbol] = self.db_manager.get_klines(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time
                )

            # Створення датафрейму з цінами закриття для всіх символів
            price_df = pd.DataFrame()
            for symbol, data in price_data.items():
                price_df[symbol] = data['close']

            # Перевірка на відсутні дані
            if price_df.isnull().values.any():
                missing_count = price_df.isnull().sum().sum()
                logger.warning(f"Виявлено {missing_count} відсутніх значень. Заповнення методом forward fill")
                price_df = price_df.fillna(method='ffill')

            # Розрахунок доходності
            returns_df = price_df.pct_change()
            returns_df = returns_df.iloc[1:]  # Видалення першого рядка, який містить NaN

            # Розрахунок волатильності (стандартне відхилення доходності)
            volatility_df = pd.DataFrame()
            for symbol in symbols:
                volatility_df[symbol] = returns_df[symbol].rolling(window=window).std()

            # Видалення початкових рядків, які містять NaN через обчислення ковзного вікна
            volatility_df = volatility_df.iloc[window - 1:]

            # Розрахунок матриці кореляції волатильності
            correlation_matrix = volatility_df.corr(method=method)

            logger.info(f"Матриця кореляції волатильності успішно розрахована з використанням методу {method}")

            # Збереження результатів у базу даних
            self.save_correlation_to_db(
                correlation_matrix=correlation_matrix,
                correlation_type='volatility',
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                method=method
            )

            return correlation_matrix

        except Exception as e:
            logger.error(f"Помилка при розрахунку кореляції волатильності: {str(e)}")
            raise

    def get_correlated_pairs(self, correlation_matrix: pd.DataFrame,
                             threshold: float = None) -> List[Tuple[str, str, float]]:

        # Use default threshold from config if not specified
        threshold = threshold or self.config['correlation_threshold']

        # Initialize results list
        correlated_pairs = []

        # Get the symbols (assuming they are both column and index names)
        symbols = correlation_matrix.columns.tolist()

        # Iterate through upper triangle of correlation matrix
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):  # Start from i+1 to avoid duplicates and self-correlations
                symbol1, symbol2 = symbols[i], symbols[j]
                correlation = correlation_matrix.loc[symbol1, symbol2]

                # Check if correlation is above threshold
                if correlation >= threshold:
                    correlated_pairs.append((symbol1, symbol2, correlation))

        # Sort pairs by correlation value in descending order
        correlated_pairs.sort(key=lambda x: x[2], reverse=True)

        logger.info(f"Found {len(correlated_pairs)} highly correlated pairs with threshold {threshold}")
        return correlated_pairs

    def get_anticorrelated_pairs(self, correlation_matrix: pd.DataFrame,
                                 threshold: float = None) -> List[Tuple[str, str, float]]:

        # Use default threshold from config if not specified
        threshold = threshold or self.config['anticorrelation_threshold']

        # Initialize results list
        anticorrelated_pairs = []

        # Get the symbols (assuming they are both column and index names)
        symbols = correlation_matrix.columns.tolist()

        # Iterate through upper triangle of correlation matrix
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):  # Start from i+1 to avoid duplicates and self-correlations
                symbol1, symbol2 = symbols[i], symbols[j]
                correlation = correlation_matrix.loc[symbol1, symbol2]

                # Check if correlation is below threshold (negative correlation)
                if correlation <= threshold:
                    anticorrelated_pairs.append((symbol1, symbol2, correlation))

        # Sort pairs by correlation value in ascending order (most negative first)
        anticorrelated_pairs.sort(key=lambda x: x[2])

        logger.info(f"Found {len(anticorrelated_pairs)} highly anti-correlated pairs with threshold {threshold}")
        return anticorrelated_pairs

    def calculate_rolling_correlation(self, symbol1: str, symbol2: str,
                                      timeframe: str = None,
                                      start_time: Optional[datetime] = None,
                                      end_time: Optional[datetime] = None,
                                      window: int = None,
                                      method: str = None) -> pd.Series:

        # Use default values from config if not specified
        timeframe = timeframe or self.config['default_timeframe']
        window = window or self.config['default_correlation_window']
        method = method or self.config['default_correlation_method']

        # Set default time range if not specified
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            # Add extra data to accommodate the window size
            start_time = end_time - timedelta(days=window + 30)  # +30 days buffer

        logger.info(f"Calculating rolling correlation between {symbol1} and {symbol2} with window={window}")

        try:
            # Fetch price data for both symbols
            df1 = self.db_manager.get_klines(
                symbol=symbol1,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time
            )

            df2 = self.db_manager.get_klines(
                symbol=symbol2,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time
            )

            # Extract close prices
            price1 = df1['close']
            price2 = df2['close']

            # Align the time series (handle missing data points)
            aligned_prices = pd.DataFrame({
                symbol1: price1,
                symbol2: price2
            })

            # Calculate percent changes (returns)
            returns = aligned_prices.pct_change().dropna()

            # Calculate rolling correlation
            rolling_corr = returns[symbol1].rolling(window=window).corr(returns[symbol2], method=method)

            logger.info(f"Successfully calculated rolling correlation with {len(rolling_corr)} data points")
            return rolling_corr

        except Exception as e:
            logger.error(f"Error calculating rolling correlation: {str(e)}")
            raise
    def detect_correlation_breakdowns(self, symbol1: str, symbol2: str,
                                      timeframe: str = None,
                                      start_time: Optional[datetime] = None,
                                      end_time: Optional[datetime] = None,
                                      window: int = None,
                                      threshold: float = None) -> List[datetime]:
        """
        Detect points where correlation between two normally correlated assets breaks down.

        Args:
            symbol1: First cryptocurrency symbol
            symbol2: Second cryptocurrency symbol
            timeframe: Time interval for the data (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            start_time: Start time for the analysis period
            end_time: End time for the analysis period
            window: Window size for rolling correlation
            threshold: Threshold change in correlation to consider a breakdown

        Returns:
            List of datetime objects where correlation breakdowns occurred
        """
        # Use default values from config if not specified
        timeframe = timeframe or self.config['default_timeframe']
        window = window or self.config['default_correlation_window']
        threshold = threshold or self.config['breakdown_threshold']
        pass

    def identify_market_clusters(self, symbols: List[str],
                                 timeframe: str = None,
                                 start_time: Optional[datetime] = None,
                                 end_time: Optional[datetime] = None,
                                 n_clusters: int = None,
                                 feature_type: str = 'returns') -> Dict[int, List[str]]:
        """
        Identify clusters of cryptocurrencies that move together using clustering algorithms.

        Args:
            symbols: List of cryptocurrency symbols to analyze
            timeframe: Time interval for the data (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            start_time: Start time for the analysis period
            end_time: End time for the analysis period
            n_clusters: Number of clusters to identify
            feature_type: Type of feature to use for clustering ('price', 'returns', 'volatility')

        Returns:
            Dictionary mapping cluster IDs to lists of symbols
        """
        # Use default values from config if not specified
        timeframe = timeframe or self.config['default_timeframe']
        n_clusters = n_clusters or self.config['default_n_clusters']
        pass

    def calculate_market_beta(self, symbol: str, market_symbol: str = 'BTCUSDT',
                              timeframe: str = None,
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None,
                              window: int = None) -> Union[float, pd.Series]:
        """
        Calculate the beta coefficient of a cryptocurrency relative to a market benchmark.

        Beta measures the volatility of a symbol relative to the market:
        - Beta > 1: More volatile than the market
        - Beta = 1: Same volatility as the market
        - Beta < 1: Less volatile than the market

        Args:
            symbol: Cryptocurrency symbol to analyze
            market_symbol: Market benchmark symbol (usually BTC or total market cap)
            timeframe: Time interval for the data (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            start_time: Start time for the analysis period
            end_time: End time for the analysis period
            window: Window size for rolling beta calculation (if not None)

        Returns:
            Either a single beta value or a time series of rolling beta values
        """
        # Use default values from config if not specified
        timeframe = timeframe or self.config['default_timeframe']
        window = window or self.config['default_correlation_window']
        pass

    def correlation_heatmap(self, correlation_matrix: pd.DataFrame,
                            title: str = "Cryptocurrency Correlation Heatmap",
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Generate a correlation heatmap visualization.

        Args:
            correlation_matrix: Correlation matrix DataFrame
            title: Title for the heatmap
            save_path: File path to save the visualization (if not None)

        Returns:
            Matplotlib figure object
        """
        # Use visualization config for styling
        colormap = self.config['visualization']['heatmap_colormap']
        figsize = self.config['visualization']['default_figsize']
        pass

    def network_graph(self, correlation_matrix: pd.DataFrame,
                      threshold: float = None,
                      title: str = "Cryptocurrency Correlation Network",
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Generate a network graph visualization of cryptocurrency correlations.

        Args:
            correlation_matrix: Correlation matrix DataFrame
            threshold: Minimum correlation coefficient to draw an edge between symbols
            title: Title for the network graph
            save_path: File path to save the visualization (if not None)

        Returns:
            Matplotlib figure object
        """
        # Use default threshold and visualization config for styling
        threshold = threshold or self.config['correlation_threshold']
        node_size = self.config['visualization']['network_node_size']
        figsize = self.config['visualization']['default_figsize']
        pass

    def save_correlation_to_db(self, correlation_matrix: pd.DataFrame,
                               correlation_type: str,
                               timeframe: str,
                               start_time: datetime,
                               end_time: datetime,
                               method: str = None) -> bool:
        """
        Save correlation matrix to database for future reference.

        Args:
            correlation_matrix: Correlation matrix DataFrame
            correlation_type: Type of correlation ('price', 'volume', 'returns', 'volatility')
            timeframe: Time interval for the data (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            start_time: Start time for the analysis period
            end_time: End time for the analysis period
            method: Correlation method used ('pearson', 'kendall', or 'spearman')

        Returns:
            True if successful, False otherwise
        """
        # Use default method from config if not specified
        method = method or self.config['default_correlation_method']
        pass

    def load_correlation_from_db(self, correlation_type: str,
                                 timeframe: str,
                                 start_time: datetime,
                                 end_time: datetime,
                                 method: str = None) -> Optional[pd.DataFrame]:
        """
        Load correlation matrix from database.

        Args:
            correlation_type: Type of correlation ('price', 'volume', 'returns', 'volatility')
            timeframe: Time interval for the data (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            start_time: Start time for the analysis period
            end_time: End time for the analysis period
            method: Correlation method used ('pearson', 'kendall', or 'spearman')

        Returns:
            Correlation matrix DataFrame if found, None otherwise
        """
        # Use default method from config if not specified
        method = method or self.config['default_correlation_method']
        pass

    def correlation_time_series(self, symbols_pair: Tuple[str, str],
                                correlation_window: int = None,
                                lookback_days: int = None,
                                timeframe: str = None) -> pd.Series:
        """
        Get time series of correlation between two cryptocurrencies over time.

        Args:
            symbols_pair: Tuple of two cryptocurrency symbols
            correlation_window: Window size for calculating correlation
            lookback_days: Number of days to look back
            timeframe: Time interval for the data (1m, 5m, 15m, 1h, 4h, 1d, etc.)

        Returns:
            Time series of correlation values
        """
        # Use default values from config if not specified
        correlation_window = correlation_window or self.config['default_correlation_window']
        lookback_days = lookback_days or self.config['default_lookback_days']
        timeframe = timeframe or self.config['default_timeframe']
        pass

    def find_leading_indicators(self, target_symbol: str,
                                candidate_symbols: List[str],
                                lag_periods: List[int] = None,
                                timeframe: str = None,
                                start_time: Optional[datetime] = None,
                                end_time: Optional[datetime] = None) -> Dict[str, Dict[int, float]]:
        """
        Find cryptocurrencies that may act as leading indicators for target symbol.

        Calculates lagged correlations to identify assets that tend to move before the target.

        Args:
            target_symbol: Target cryptocurrency symbol
            candidate_symbols: List of potential leading indicator symbols
            lag_periods: List of lag periods to test
            timeframe: Time interval for the data (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            start_time: Start time for the analysis period
            end_time: End time for the analysis period

        Returns:
            Dictionary mapping symbols to dictionaries of lag periods and correlation values
        """
        # Use default values from config if not specified
        lag_periods = lag_periods or self.config['default_lag_periods']
        timeframe = timeframe or self.config['default_timeframe']
        pass

    def sector_correlation_analysis(self, sector_mapping: Dict[str, str],
                                    timeframe: str = None,
                                    start_time: Optional[datetime] = None,
                                    end_time: Optional[datetime] = None) -> Dict[str, Dict[str, float]]:
        """
        Analyze correlations between different cryptocurrency sectors.

        Args:
            sector_mapping: Dictionary mapping symbols to their sectors
            timeframe: Time interval for the data (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            start_time: Start time for the analysis period
            end_time: End time for the analysis period

        Returns:
            Dictionary mapping each sector to its correlation with other sectors
        """
        # Use default timeframe from config if not specified
        timeframe = timeframe or self.config['default_timeframe']
        pass

    def correlated_movement_prediction(self, symbol: str,
                                       correlated_symbols: List[str],
                                       prediction_horizon: int = None,
                                       timeframe: str = None) -> Dict[str, float]:
        """
        Predict potential movement of target symbol based on highly correlated symbols.

        Args:
            symbol: Target cryptocurrency symbol
            correlated_symbols: List of highly correlated symbols
            prediction_horizon: Number of periods to predict ahead
            timeframe: Time interval for the data (1m, 5m, 15m, 1h, 4h, 1d, etc.)

        Returns:
            Dictionary with prediction metrics and potential movement
        """
        # Use default values from config if not specified
        prediction_horizon = prediction_horizon or self.config['default_prediction_horizon']
        timeframe = timeframe or self.config['default_timeframe']
        pass

    def get_decorrelated_portfolio(self, symbols: List[str],
                                   target_correlation: float = None,
                                   timeframe: str = None,
                                   start_time: Optional[datetime] = None,
                                   end_time: Optional[datetime] = None) -> Dict[str, float]:
        """
        Generate weights for a portfolio of cryptocurrencies with low mutual correlation.

        Args:
            symbols: List of cryptocurrency symbols to include in portfolio
            target_correlation: Target maximum average correlation between assets
            timeframe: Time interval for the data (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            start_time: Start time for the analysis period
            end_time: End time for the analysis period

        Returns:
            Dictionary mapping symbols to suggested portfolio weights
        """
        # Use default values from config if not specified
        target_correlation = target_correlation or self.config['target_portfolio_correlation']
        timeframe = timeframe or self.config['default_timeframe']
        pass

    def analyze_market_regime_correlations(self, symbols: List[str],
                                           market_regimes: Dict[Tuple[datetime, datetime], str],
                                           timeframe: str = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Analyze how correlations between cryptocurrencies change in different market regimes.

        Args:
            symbols: List of cryptocurrency symbols to analyze
            market_regimes: Dictionary mapping time periods to regime names
            timeframe: Time interval for the data (1m, 5m, 15m, 1h, 4h, 1d, etc.)

        Returns:
            Dictionary mapping regime names to correlation matrices
        """
        # Use default timeframe from config if not specified
        timeframe = timeframe or self.config['default_timeframe']
        pass

    def correlation_with_external_assets(self, crypto_symbols: List[str],
                                         external_data: Dict[str, pd.Series],
                                         timeframe: str = None,
                                         start_time: Optional[datetime] = None,
                                         end_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Calculate correlations between cryptocurrencies and external financial assets.

        Args:
            crypto_symbols: List of cryptocurrency symbols to analyze
            external_data: Dictionary mapping asset names to price time series
            timeframe: Time interval for the data (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            start_time: Start time for the analysis period
            end_time: End time for the analysis period

        Returns:
            Correlation matrix of cryptocurrencies vs external assets
        """
        # Use default timeframe from config if not specified
        timeframe = timeframe or self.config['default_timeframe']
        pass