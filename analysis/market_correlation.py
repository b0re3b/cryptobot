import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Internal imports
from data.db import DatabaseManager
from data_collection.binance_client import BinanceClient
from data_collection.market_data_processor import MarketDataProcessor
from utils.logger import setup_logger

logger = setup_logger(__name__)
# таблиці які існують correlation_matrices correlated_pairs correlation_time_series market_clusters correlation_breakdowns market_betas
#beta_time_series sector_correlations leading_indicators external_asset_correlations market_regime_correlations

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

        self.db_manager = DatabaseManager()
        self.binance_client = BinanceClient()
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

        logger.info(f"Detecting correlation breakdowns between {symbol1} and {symbol2} with threshold {threshold}")

        try:
            # Calculate rolling correlation between the two symbols
            rolling_corr = self.calculate_rolling_correlation(
                symbol1=symbol1,
                symbol2=symbol2,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                window=window
            )

            # Calculate absolute changes in correlation
            correlation_changes = rolling_corr.diff().abs()

            # Identify points where correlation changes more than the threshold
            breakdown_points = correlation_changes[correlation_changes > threshold].index.tolist()

            logger.info(f"Found {len(breakdown_points)} correlation breakdown points")

            # Save breakdown data to database
            breakdown_data = []
            for point in breakdown_points:
                breakdown_data.append({
                    'timestamp': point,
                    'symbol1': symbol1,
                    'symbol2': symbol2,
                    'correlation_before': rolling_corr.loc[rolling_corr.index < point].iloc[-1] if not rolling_corr.loc[
                        rolling_corr.index < point].empty else None,
                    'correlation_after': rolling_corr.loc[point],
                    'change_magnitude': correlation_changes.loc[point]
                })

            # Save to correlation_breakdowns table
            if breakdown_data:
                self.db_manager.save_correlation_breakdowns(breakdown_data)
                logger.debug(f"Saved {len(breakdown_data)} breakdown points to database")

            return breakdown_points

        except Exception as e:
            logger.error(f"Error detecting correlation breakdowns: {str(e)}")
            raise

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

        logger.info(f"Identifying market clusters for {len(symbols)} symbols using {feature_type} data")

        # Set default time range if not specified
        end_time = end_time or datetime.now()
        if start_time is None:
            lookback_days = self.config['default_lookback_days']
            start_time = end_time - timedelta(days=lookback_days)

        try:
            # Fetch data for all symbols
            data_dict = {}
            for symbol in symbols:
                data_dict[symbol] = self.db_manager.get_klines(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time
                )

            # Create appropriate feature matrix based on feature_type
            feature_matrix = pd.DataFrame()

            if feature_type == 'price':
                # Use normalized price series
                for symbol, data in data_dict.items():
                    prices = data['close']
                    # Normalize to start from 1.0
                    feature_matrix[symbol] = prices / prices.iloc[0] if not prices.empty else pd.Series()

            elif feature_type == 'returns':
                # Use return series
                for symbol, data in data_dict.items():
                    prices = data['close']
                    returns = prices.pct_change().dropna()
                    feature_matrix[symbol] = returns

            elif feature_type == 'volatility':
                window = self.config['default_correlation_window']
                # Calculate rolling volatility
                for symbol, data in data_dict.items():
                    prices = data['close']
                    returns = prices.pct_change().dropna()
                    feature_matrix[symbol] = returns.rolling(window=window).std().dropna()

            # Drop rows with missing values
            feature_matrix = feature_matrix.dropna()

            if feature_matrix.empty:
                logger.warning("Empty feature matrix after preprocessing, cannot perform clustering")
                return {}

            # Apply standardization to features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_matrix)

            # Apply clustering algorithm (KMeans)
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_features)

            # Create result dictionary
            clusters = {}
            for i in range(n_clusters):
                clusters[i] = []

            # Assign symbols to clusters
            for i, symbol in enumerate(feature_matrix.columns):
                cluster_id = kmeans.labels_[i]
                clusters[cluster_id].append(symbol)

            # Save clustering results to database
            cluster_records = []
            for cluster_id, cluster_symbols in clusters.items():
                for symbol in cluster_symbols:
                    cluster_records.append({
                        'cluster_id': cluster_id,
                        'symbol': symbol,
                        'feature_type': feature_type,
                        'timeframe': timeframe,
                        'start_time': start_time,
                        'end_time': end_time,
                        'analysis_time': datetime.now()
                    })

            # Save to market_clusters table
            self.db_manager.save_market_clusters(cluster_records)
            logger.info(f"Saved {len(cluster_records)} cluster records to database")

            return clusters

        except Exception as e:
            logger.error(f"Error identifying market clusters: {str(e)}")
            raise

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

        logger.info(f"Calculating market beta for {symbol} relative to {market_symbol}")

        # Set default time range if not specified
        end_time = end_time or datetime.now()
        if start_time is None:
            lookback_days = self.config['default_lookback_days']
            start_time = end_time - timedelta(days=lookback_days)

        try:
            # Fetch price data for asset and market benchmark
            asset_data = self.db_manager.get_klines(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time
            )

            market_data = self.db_manager.get_klines(
                symbol=market_symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time
            )

            # Calculate returns
            asset_prices = asset_data['close']
            market_prices = market_data['close']

            # Align time series
            df = pd.DataFrame({
                'asset': asset_prices,
                'market': market_prices
            })

            # Calculate returns
            returns = df.pct_change().dropna()

            # If window is specified, calculate rolling beta
            if window:
                # Calculate rolling covariance and market variance
                rolling_cov = returns['asset'].rolling(window=window).cov(returns['market'])
                rolling_market_var = returns['market'].rolling(window=window).var()

                # Calculate rolling beta
                rolling_beta = rolling_cov / rolling_market_var

                # Save beta time series to database
                beta_records = []
                for timestamp, beta_value in rolling_beta.items():
                    if not np.isnan(beta_value) and not np.isinf(beta_value):
                        beta_records.append({
                            'timestamp': timestamp,
                            'symbol': symbol,
                            'market_symbol': market_symbol,
                            'beta': beta_value,
                            'timeframe': timeframe,
                            'window': window
                        })

                # Save to beta_time_series table
                if beta_records:
                    self.db_manager.save_beta_time_series(beta_records)
                    logger.debug(f"Saved {len(beta_records)} beta values to database")

                return rolling_beta
            else:
                # Calculate overall beta
                beta = np.cov(returns['asset'], returns['market'])[0, 1] / np.var(returns['market'])

                # Save to market_betas table
                beta_record = {
                    'symbol': symbol,
                    'market_symbol': market_symbol,
                    'beta': beta,
                    'timeframe': timeframe,
                    'start_time': start_time,
                    'end_time': end_time,
                    'analysis_time': datetime.now()
                }

                self.db_manager.save_market_betas([beta_record])
                logger.info(f"Saved beta value {beta} for {symbol} to database")

                return beta

        except Exception as e:
            logger.error(f"Error calculating market beta: {str(e)}")
            raise

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

        logger.info(f"Generating correlation heatmap with {correlation_matrix.shape[0]} assets")

        try:
            # Create figure and axes
            fig, ax = plt.subplots(figsize=figsize)

            # Generate mask for upper triangle
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

            # Create heatmap
            sns.heatmap(
                correlation_matrix,
                mask=mask,
                cmap=colormap,
                vmin=-1,
                vmax=1,
                center=0,
                square=True,
                linewidths=.5,
                annot=True if correlation_matrix.shape[0] <= 15 else False,  # Only show annotations for small matrices
                fmt=".2f" if correlation_matrix.shape[0] <= 15 else None,
                cbar_kws={"shrink": .8},
                ax=ax
            )

            # Set title and adjust layout
            plt.title(title, fontsize=16)
            plt.tight_layout()

            # Save figure if path is provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.debug(f"Saved heatmap to {save_path}")

            return fig

        except Exception as e:
            logger.error(f"Error generating correlation heatmap: {str(e)}")
            raise

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

        logger.info(f"Generating correlation network graph with threshold {threshold}")

        try:
            import networkx as nx

            # Create figure
            fig, ax = plt.subplots(figsize=figsize)

            # Create network graph
            G = nx.Graph()

            # Add nodes (symbols)
            symbols = correlation_matrix.columns
            for symbol in symbols:
                G.add_node(symbol)

            # Add edges based on correlation threshold
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    symbol1, symbol2 = symbols[i], symbols[j]
                    correlation = correlation_matrix.loc[symbol1, symbol2]

                    # Add edge if correlation exceeds threshold
                    if correlation >= threshold:
                        # Weight is proportional to correlation
                        G.add_edge(symbol1, symbol2, weight=correlation)

            # Calculate node positioning using spring layout
            pos = nx.spring_layout(G, seed=42)

            # Draw nodes
            nx.draw_networkx_nodes(
                G,
                pos,
                node_size=node_size,
                node_color='skyblue',
                alpha=0.8,
                ax=ax
            )

            # Draw edges with width proportional to correlation
            for u, v, data in G.edges(data=True):
                width = data['weight'] * 3  # Scale width by correlation
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=[(u, v)],
                    width=width,
                    alpha=0.5,
                    edge_color='navy',
                    ax=ax
                )

            # Draw node labels
            nx.draw_networkx_labels(
                G,
                pos,
                font_size=10,
                font_family='sans-serif',
                ax=ax
            )

            # Set title and remove axis
            ax.set_title(title, fontsize=16)
            ax.axis('off')

            # Save figure if path is provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.debug(f"Saved network graph to {save_path}")

            return fig

        except Exception as e:
            logger.error(f"Error generating network graph: {str(e)}")
            raise

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

        logger.info(f"Saving {correlation_type} correlation matrix to database")

        try:
            # Create a record for the correlation matrix
            matrix_id = f"{correlation_type}_{timeframe}_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}_{method}"

            # Prepare data for the correlation_matrices table
            matrix_data = {
                'matrix_id': matrix_id,
                'correlation_type': correlation_type,
                'timeframe': timeframe,
                'start_time': start_time,
                'end_time': end_time,
                'method': method,
                'analysis_time': datetime.now(),
                'symbols': correlation_matrix.columns.tolist(),
                'matrix_json': correlation_matrix.to_json()
            }

            # Save to correlation_matrices table
            self.db_manager.save_correlation_matrices([matrix_data])

            # Extract and save highly correlated pairs
            symbols = correlation_matrix.columns.tolist()
            pairs_data = []

            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    symbol1, symbol2 = symbols[i], symbols[j]
                    correlation = correlation_matrix.loc[symbol1, symbol2]

                    if abs(correlation) >= self.config['correlation_threshold']:
                        pair_data = {
                            'matrix_id': matrix_id,
                            'symbol1': symbol1,
                            'symbol2': symbol2,
                            'correlation': correlation,
                            'correlation_type': correlation_type,
                            'timeframe': timeframe,
                            'analysis_time': datetime.now()
                        }
                        pairs_data.append(pair_data)

            # Save to correlated_pairs table
            if pairs_data:
                self.db_manager.insert_correlated_pair(pairs_data)
                logger.debug(f"Saved {len(pairs_data)} correlated pairs to database")

            logger.info(f"Successfully saved correlation matrix with ID {matrix_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving correlation to database: {str(e)}")
            return False

    def load_correlation_from_db(self, correlation_type: str,
                                 timeframe: str,
                                 start_time: datetime,
                                 end_time: datetime,
                                 method: str = None) -> Optional[pd.DataFrame]:

        # Use default method from config if not specified
        method = method or self.config['default_correlation_method']

        logger.info(f"Loading {correlation_type} correlation matrix from database")

        try:
            # Create the matrix ID for lookup
            matrix_id = f"{correlation_type}_{timeframe}_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}_{method}"

            # Load from correlation_matrices table
            matrix_data = self.db_manager.get_correlation_matrix(matrix_id)

            if matrix_data:
                # Convert JSON string back to DataFrame
                matrix_json = matrix_data.get('matrix_json')
                if matrix_json:
                    correlation_matrix = pd.read_json(matrix_json)
                    logger.info(f"Successfully loaded correlation matrix with ID {matrix_id}")
                    return correlation_matrix

            logger.warning(f"No correlation matrix found with ID {matrix_id}")
            return None

        except Exception as e:
            logger.error(f"Error loading correlation from database: {str(e)}")
            return None

    def correlation_time_series(self, symbols_pair: Tuple[str, str],
                                correlation_window: int = None,
                                lookback_days: int = None,
                                timeframe: str = None) -> pd.Series:

        # Use default values from config if not specified
        correlation_window = correlation_window or self.config['default_correlation_window']
        lookback_days = lookback_days or self.config['default_lookback_days']
        timeframe = timeframe or self.config['default_timeframe']

        symbol1, symbol2 = symbols_pair
        logger.info(f"Calculating correlation time series between {symbol1} and {symbol2}")

        # Set time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)

        try:
            # First check if we already have this data in the database
            correlation_series = self.db_manager.get_correlation_time_series(
                symbol1=symbol1,
                symbol2=symbol2,
                start_time=start_time,
                end_time=end_time,
                timeframe=timeframe,
                window=correlation_window
            )

            if correlation_series is not None and not correlation_series.empty:
                logger.info(f"Found existing correlation time series in database")
                return correlation_series

            # If not found in database, calculate it
            logger.info(f"Calculating new correlation time series")

            # Calculate rolling correlation
            rolling_corr = self.calculate_rolling_correlation(
                symbol1=symbol1,
                symbol2=symbol2,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                window=correlation_window
            )

            # Save correlation time series to database
            correlation_data = []
            for timestamp, corr_value in rolling_corr.items():
                if not np.isnan(corr_value):
                    correlation_data.append({
                        'timestamp': timestamp,
                        'symbol1': symbol1,
                        'symbol2': symbol2,
                        'correlation': corr_value,
                        'timeframe': timeframe,
                        'window': correlation_window
                    })

            # Save to correlation_time_series table
            if correlation_data:
                self.db_manager.save_correlation_time_series(correlation_data)
                logger.debug(f"Saved {len(correlation_data)} correlation values to database")

            return rolling_corr

        except Exception as e:
            logger.error(f"Error calculating correlation time series: {str(e)}")
            raise

    def find_leading_indicators(self, target_symbol: str,
                                candidate_symbols: List[str],
                                lag_periods: List[int] = None,
                                timeframe: str = None,
                                start_time: Optional[datetime] = None,
                                end_time: Optional[datetime] = None) -> Dict[str, Dict[int, float]]:

        # Use default values from config if not specified
        lag_periods = lag_periods or self.config['default_lag_periods']
        timeframe = timeframe or self.config['default_timeframe']

        logger.info(f"Finding leading indicators for {target_symbol} among {len(candidate_symbols)} candidates")

        # Set default time range if not specified
        end_time = end_time or datetime.now()
        if start_time is None:
            lookback_days = self.config['default_lookback_days']
            start_time = end_time - timedelta(days=lookback_days)

        try:
            # Fetch target data
            target_data = self.db_manager.get_klines(
                symbol=target_symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time
            )

            # Calculate target returns
            target_prices = target_data['close']
            target_returns = target_prices.pct_change().dropna()

            # Dictionary to store results
            leading_indicators = {}
            leading_indicators_records = []

            # Test each candidate symbol
            for symbol in candidate_symbols:
                if symbol == target_symbol:
                    continue

                logger.debug(f"Testing {symbol} as a leading indicator")

                # Fetch candidate data
                candidate_data = self.db_manager.get_klines(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time
                )

                # Calculate candidate returns
                candidate_prices = candidate_data['close']
                candidate_returns = candidate_prices.pct_change().dropna()

                # Align time series
                common_index = target_returns.index.intersection(candidate_returns.index)
                if len(common_index) < max(lag_periods) + 10:  # Need enough data points
                    logger.warning(f"Insufficient aligned data points for {symbol}")
                    continue

                aligned_target = target_returns.loc[common_index]
                aligned_candidate = candidate_returns.loc[common_index]

                # Calculate lagged correlations
                lag_correlations = {}
                for lag in lag_periods:
                    # Shift candidate data forward to test if it leads target
                    lagged_candidate = aligned_candidate.shift(lag)

                    # Remove NaN values created by shifting
                    valid_indices = aligned_target.index.intersection(lagged_candidate.dropna().index)
                    if len(valid_indices) < 10:  # Need enough data points for correlation
                        continue

                    # Calculate correlation between target and lagged candidate
                    correlation = aligned_target.loc[valid_indices].corr(lagged_candidate.loc[valid_indices])
                    lag_correlations[lag] = correlation

                    # Add to records for database
                    leading_indicators_records.append({
                        'target_symbol': target_symbol,
                        'indicator_symbol': symbol,
                        'lag_periods': lag,
                        'correlation': correlation,
                        'timeframe': timeframe,
                        'start_time': start_time,
                        'end_time': end_time,
                        'analysis_time': datetime.now()
                    })

                # Only add to results if at least one lag period had a correlation
                if lag_correlations:
                    leading_indicators[symbol] = lag_correlations

            # Save to leading_indicators table
            if leading_indicators_records:
                self.db_manager.save_leading_indicators(leading_indicators_records)
                logger.debug(f"Saved {len(leading_indicators_records)} leading indicator records to database")

            return leading_indicators

        except Exception as e:
            logger.error(f"Error finding leading indicators: {str(e)}")
            raise

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