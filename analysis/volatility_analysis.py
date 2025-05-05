"""
Volatility Analysis Module

This module provides functionality for analyzing volatility in cryptocurrency markets.
It supports various volatility metrics, regime detection, and integrates with the prediction models.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import arch
from arch import arch_model

# Import from other project modules
from data.db import DatabaseManager
from models.time_series import extract_volatility, load_crypto_data
from data_collection.market_data_processor import clean_data, detect_outliers
from data_collection.feature_engineering import create_volatility_features
from utils.logger import get_logger
from utils.crypto_helpers import get_market_phases

logger = get_logger(__name__)


class VolatilityAnalysis:
    """Class for analyzing cryptocurrency market volatility patterns"""

    def __init__(self):
        """Initialize the volatility analysis with optional database connection"""
        self.db_connection = DatabaseManager()
        self.volatility_models = {}
        self.regime_models = {}

    def calculate_historical_volatility(self, price_data, window=14, trading_periods=365, annualize=True):
        """
        Calculate historical volatility using rolling standard deviation

        Args:
            price_data (pd.DataFrame/Series): Price data with datetime index
            window (int): Rolling window size for volatility calculation
            trading_periods (int): Number of trading periods in a year (365 for crypto)
            annualize (bool): Whether to annualize the volatility

        Returns:
            pd.Series: Historical volatility series
        """
        # Calculate log returns
        log_returns = np.log(price_data / price_data.shift(1)).dropna()

        # Calculate rolling volatility
        rolling_vol = log_returns.rolling(window=window).std()

        # Annualize if requested
        if annualize:
            rolling_vol = rolling_vol * np.sqrt(trading_periods)

        return rolling_vol

    def calculate_parkinson_volatility(self, ohlc_data, window=14, trading_periods=365):
        """
        Calculate Parkinson volatility using high-low range

        Args:
            ohlc_data (pd.DataFrame): OHLC data with 'high' and 'low' columns
            window (int): Rolling window size
            trading_periods (int): Number of trading periods in a year

        Returns:
            pd.Series: Parkinson volatility series
        """
        # Calculate normalized high-low range
        hl_range = np.log(ohlc_data['high'] / ohlc_data['low'])
        parkinson = pd.Series(hl_range ** 2 / (4 * np.log(2)), index=ohlc_data.index)

        # Calculate rolling volatility and annualize
        rolling_parkinson = np.sqrt(parkinson.rolling(window=window).mean() * trading_periods)

        return rolling_parkinson

    def calculate_garman_klass_volatility(self, ohlc_data, window=14, trading_periods=365):
        """
        Calculate Garman-Klass volatility using OHLC data

        Args:
            ohlc_data (pd.DataFrame): OHLC data
            window (int): Rolling window size
            trading_periods (int): Number of trading periods in a year

        Returns:
            pd.Series: Garman-Klass volatility series
        """
        # Calculate components
        log_hl = np.log(ohlc_data['high'] / ohlc_data['low']) ** 2 * 0.5
        log_co = np.log(ohlc_data['close'] / ohlc_data['open']) ** 2 * (2 * np.log(2) - 1)

        # Combine components
        gk = pd.Series(log_hl - log_co, index=ohlc_data.index)

        # Calculate rolling volatility and annualize
        rolling_gk = np.sqrt(gk.rolling(window=window).mean() * trading_periods)

        return rolling_gk

    def calculate_yang_zhang_volatility(self, ohlc_data, window=14, trading_periods=365):
        """
        Calculate Yang-Zhang volatility using OHLC data (open, high, low, close)

        Args:
            ohlc_data (pd.DataFrame): OHLC data
            window (int): Rolling window size
            trading_periods (int): Number of trading periods in a year

        Returns:
            pd.Series: Yang-Zhang volatility series
        """
        # Calculate overnight volatility (close to open)
        overnight_returns = np.log(ohlc_data['open'] / ohlc_data['close'].shift(1))
        overnight_vol = overnight_returns.rolling(window=window).var()

        # Calculate open-close volatility
        open_close_returns = np.log(ohlc_data['close'] / ohlc_data['open'])
        open_close_vol = open_close_returns.rolling(window=window).var()

        # Calculate Rogers-Satchell volatility
        log_ho = np.log(ohlc_data['high'] / ohlc_data['open'])
        log_lo = np.log(ohlc_data['low'] / ohlc_data['open'])
        log_hc = np.log(ohlc_data['high'] / ohlc_data['close'])
        log_lc = np.log(ohlc_data['low'] / ohlc_data['close'])

        rs_vol = log_ho * (log_ho - log_lo) + log_lc * (log_lc - log_hc)
        rs_vol = rs_vol.rolling(window=window).mean()

        # Calculate Yang-Zhang volatility with k=0.34 (recommended value)
        k = 0.34
        yang_zhang = overnight_vol + k * open_close_vol + (1 - k) * rs_vol

        # Annualize
        yang_zhang = np.sqrt(yang_zhang * trading_periods)

        return yang_zhang

    def fit_garch_model(self, returns, p=1, q=1, model_type='GARCH'):
        """
        Fit GARCH model to returns data

        Args:
            returns (pd.Series): Return series
            p (int): GARCH lag order
            q (int): ARCH lag order
            model_type (str): Model type ('GARCH', 'EGARCH', 'GJR-GARCH', etc.)

        Returns:
            model: Fitted GARCH model
            forecast: Volatility forecast
        """
        try:
            # Clean and prepare data
            clean_returns = clean_data(returns)

            # Set up the model
            if model_type == 'EGARCH':
                model = arch_model(clean_returns, vol='EGARCH', p=p, q=q)
            elif model_type == 'GJR-GARCH':
                model = arch_model(clean_returns, vol='GARCH', p=p, o=1, q=q)
            else:  # Default to GARCH
                model = arch_model(clean_returns, vol='GARCH', p=p, q=q)

            # Fit the model
            fitted_model = model.fit(disp='off')

            # Get forecast
            forecast = fitted_model.forecast(horizon=30)

            # Store model for later use
            self.volatility_models[f"{model_type}_{p}_{q}"] = fitted_model

            return fitted_model, forecast

        except Exception as e:
            logger.error(f"Error fitting GARCH model: {e}")
            return None, None

    def detect_volatility_regimes(self, volatility_series, n_regimes=3, method='kmeans'):
        """
        Detect different volatility regimes using clustering

        Args:
            volatility_series (pd.Series): Volatility time series
            n_regimes (int): Number of regimes to identify
            method (str): Method to use ('kmeans', 'hmm', 'threshold')

        Returns:
            pd.Series: Series with regime labels
        """
        try:
            # Clean data and reshape for clustering
            clean_vol = volatility_series.dropna().values.reshape(-1, 1)

            if method == 'kmeans':
                # Apply KMeans clustering
                kmeans = KMeans(n_clusters=n_regimes, random_state=42)
                regimes = kmeans.fit_predict(clean_vol)

                # Create series with original index
                regime_series = pd.Series(regimes, index=volatility_series.dropna().index)

                # Map to meaningful labels (0 = low, 1 = medium, 2 = high)
                # Sort clusters by their centroids
                centroids = kmeans.cluster_centers_.flatten()
                centroid_mapping = {i: rank for rank, i in enumerate(np.argsort(centroids))}
                regime_series = regime_series.map(centroid_mapping)

            elif method == 'threshold':
                # Use percentile thresholds
                thresholds = [volatility_series.quantile(q) for q in np.linspace(0, 1, n_regimes + 1)[1:-1]]

                # Initialize with lowest regime
                regimes = np.zeros(len(volatility_series))

                # Assign regimes based on thresholds
                for i, threshold in enumerate(thresholds, 1):
                    regimes[volatility_series > threshold] = i

                regime_series = pd.Series(regimes, index=volatility_series.index)

            else:
                # Default basic method
                regime_series = pd.qcut(volatility_series, n_regimes, labels=False)

            # Store the regime model
            self.regime_models[f"{method}_{n_regimes}"] = {
                'model': kmeans if method == 'kmeans' else None,
                'thresholds': thresholds if method == 'threshold' else None
            }

            return regime_series

        except Exception as e:
            logger.error(f"Error detecting volatility regimes: {e}")
            return None

    def analyze_volatility_clustering(self, returns, max_lag=30):
        """
        Analyze volatility clustering through autocorrelation of squared returns

        Args:
            returns (pd.Series): Return series
            max_lag (int): Maximum lag to consider

        Returns:
            pd.DataFrame: DataFrame with lag and autocorrelation values
        """
        # Calculate squared returns
        squared_returns = returns ** 2

        # Calculate autocorrelation
        acf_values = acf(squared_returns.dropna(), nlags=max_lag)

        # Create result DataFrame
        acf_df = pd.DataFrame({
            'lag': range(max_lag + 1),
            'autocorrelation': acf_values
        })

        return acf_df

    def calculate_volatility_risk_metrics(self, returns, volatility):
        """
        Calculate volatility-based risk metrics

        Args:
            returns (pd.Series): Return series
            volatility (pd.Series): Volatility series

        Returns:
            dict: Dictionary with risk metrics
        """
        # Calculate Value at Risk (VaR) at different confidence levels
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)

        # Calculate Conditional VaR (Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()

        # Calculate volatility of volatility
        vol_of_vol = volatility.rolling(window=30).std()

        # Calculate other risk metrics
        sharpe = returns.mean() / returns.std() * np.sqrt(365)  # Annualized Sharpe

        # Return metrics as dictionary
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'vol_of_vol_mean': vol_of_vol.mean(),
            'sharpe_ratio': sharpe,
            'max_drawdown': (returns.cumsum() - returns.cumsum().cummax()).min()
        }

    def compare_volatility_metrics(self, ohlc_data, windows=[14, 30, 60]):
        """
        Compare different volatility metrics for the same data

        Args:
            ohlc_data (pd.DataFrame): OHLC data
            windows (list): List of window sizes to compare

        Returns:
            pd.DataFrame: DataFrame with different volatility metrics
        """
        result = pd.DataFrame(index=ohlc_data.index)

        # Calculate returns
        ohlc_data['returns'] = ohlc_data['close'].pct_change()

        for window in windows:
            # Calculate different volatility metrics
            result[f'historical_{window}d'] = self.calculate_historical_volatility(
                ohlc_data['close'], window=window)

            result[f'parkinson_{window}d'] = self.calculate_parkinson_volatility(
                ohlc_data, window=window)

            result[f'gk_{window}d'] = self.calculate_garman_klass_volatility(
                ohlc_data, window=window)

            result[f'yz_{window}d'] = self.calculate_yang_zhang_volatility(
                ohlc_data, window=window)

        return result

    def identify_volatility_breakouts(self, volatility_series, window=20, std_dev=2):
        """
        Identify periods of volatility breakouts

        Args:
            volatility_series (pd.Series): Volatility time series
            window (int): Lookback window
            std_dev (float): Number of standard deviations for threshold

        Returns:
            pd.Series: Boolean series indicating volatility breakouts
        """
        # Calculate rolling mean and standard deviation
        rolling_mean = volatility_series.rolling(window=window).mean()
        rolling_std = volatility_series.rolling(window=window).std()

        # Calculate upper threshold
        upper_threshold = rolling_mean + std_dev * rolling_std

        # Identify breakouts
        breakouts = volatility_series > upper_threshold

        return breakouts

    def analyze_cross_asset_volatility(self, asset_dict, window=14):
        """
        Analyze volatility correlation between different crypto assets

        Args:
            asset_dict (dict): Dictionary with asset name as key and price series as value
            window (int): Window for volatility calculation

        Returns:
            pd.DataFrame: Correlation matrix of volatilities
        """
        volatility_dict = {}

        # Calculate volatility for each asset
        for asset_name, price_series in asset_dict.items():
            volatility_dict[asset_name] = self.calculate_historical_volatility(
                price_series, window=window)

        # Create DataFrame with all volatilities
        vol_df = pd.DataFrame(volatility_dict)

        # Calculate correlation matrix
        corr_matrix = vol_df.corr()

        return corr_matrix

    def extract_seasonality_in_volatility(self, volatility_series, period=7):
        """
        Extract seasonality patterns in volatility

        Args:
            volatility_series (pd.Series): Volatility time series with datetime index
            period (int): Period to check (7=weekly, 30=monthly)

        Returns:
            pd.Series: Average volatility by time period
        """
        if period == 7:
            # Extract day of week seasonality
            volatility_series.index = pd.to_datetime(volatility_series.index)
            day_of_week = volatility_series.groupby(volatility_series.index.dayofweek).mean()
            day_of_week.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            return day_of_week

        elif period == 24:
            # Extract hour of day seasonality (for intraday data)
            hour_of_day = volatility_series.groupby(volatility_series.index.hour).mean()
            return hour_of_day

        elif period == 30:
            # Extract day of month seasonality
            day_of_month = volatility_series.groupby(volatility_series.index.day).mean()
            return day_of_month

        elif period == 12:
            # Extract month of year seasonality
            month_of_year = volatility_series.groupby(volatility_series.index.month).mean()
            month_of_year.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            return month_of_year

    def analyze_volatility_term_structure(self, symbol, timeframes=['1h', '4h', '1d', '1w']):
        """
        Analyze volatility across different timeframes (term structure)

        Args:
            symbol (str): Cryptocurrency symbol
            timeframes (list): List of timeframes to analyze

        Returns:
            pd.DataFrame: DataFrame with volatility metrics for different timeframes
        """
        results = {}

        for timeframe in timeframes:
            # Load data for this timeframe using project's data loading function
            data = load_crypto_data(symbol, timeframe=timeframe)

            # Calculate volatility
            vol = self.calculate_historical_volatility(data['close'])

            # Store results
            results[timeframe] = {
                'mean_vol': vol.mean(),
                'median_vol': vol.median(),
                'max_vol': vol.max(),
                'min_vol': vol.min(),
                'std_vol': vol.std()
            }

        # Convert to DataFrame
        term_structure = pd.DataFrame(results).T

        return term_structure

    def volatility_impulse_response(self, returns, shock_size=3, days=30):
        """
        Calculate how volatility responds to a shock over time

        Args:
            returns (pd.Series): Return series
            shock_size (float): Size of shock in standard deviations
            days (int): Number of days to forecast response

        Returns:
            pd.Series: Impulse response series
        """
        try:
            # Fit GARCH model
            model, _ = self.fit_garch_model(returns)

            if model is None:
                return None

            # Create baseline forecast
            baseline = model.forecast(horizon=days).variance.values[-1]

            # Add shock to the last return and reforecast
            shocked_returns = returns.copy()
            shock_value = shock_size * shocked_returns.std()
            shocked_returns.iloc[-1] = shock_value

            # Refit model with shocked data
            shocked_model = arch_model(shocked_returns, vol='GARCH', p=1, q=1).fit(disp='off')

            # Forecast after shock
            shocked_forecast = shocked_model.forecast(horizon=days).variance.values[-1]

            # Calculate impulse response (difference from baseline)
            impulse = pd.Series(shocked_forecast - baseline, index=range(1, days + 1))

            return impulse

        except Exception as e:
            logger.error(f"Error calculating volatility impulse response: {e}")
            return None

    def prepare_volatility_features_for_ml(self, ohlc_data, window_sizes=[7, 14, 30], include_regimes=True):
        """
        Prepare volatility-related features for machine learning

        Args:
            ohlc_data (pd.DataFrame): OHLC data
            window_sizes (list): List of window sizes for features
            include_regimes (bool): Whether to include regime features

        Returns:
            pd.DataFrame: DataFrame with volatility features
        """
        # Initialize result DataFrame
        features = pd.DataFrame(index=ohlc_data.index)

        # Calculate returns
        returns = ohlc_data['close'].pct_change()

        # Use feature engineering module to create standard volatility features
        vol_features = create_volatility_features(ohlc_data)
        features = pd.concat([features, vol_features], axis=1)

        # Calculate additional volatility metrics for different windows
        for window in window_sizes:
            # Add historical volatility
            features[f'hist_vol_{window}d'] = self.calculate_historical_volatility(
                ohlc_data['close'], window=window)

            # Add Parkinson volatility
            features[f'park_vol_{window}d'] = self.calculate_parkinson_volatility(
                ohlc_data, window=window)

            # Add relative volatility (compared to moving average)
            moving_avg_vol = features[f'hist_vol_{window}d'].rolling(window=window * 2).mean()
            features[f'rel_vol_{window}d'] = features[f'hist_vol_{window}d'] / moving_avg_vol

            # Add volatility of volatility
            features[f'vol_of_vol_{window}d'] = features[f'hist_vol_{window}d'].rolling(window=window).std()

            # Volatility trend (increasing or decreasing)
            features[f'vol_trend_{window}d'] = features[f'hist_vol_{window}d'].diff(window)

            # Add High-Low range relative to volatility
            hl_range = (ohlc_data['high'] - ohlc_data['low']) / ohlc_data['close']
            features[f'hl_range_to_vol_{window}d'] = hl_range / features[f'hist_vol_{window}d']

        # Add regime identification if requested
        if include_regimes:
            # Use primary volatility for regime detection
            main_vol = features['hist_vol_14d']

            # Detect regimes and add as features
            regimes = self.detect_volatility_regimes(main_vol, n_regimes=3)

            # One-hot encode regimes
            for i in range(3):
                features[f'vol_regime_{i}'] = (regimes == i).astype(int)

        return features

    def save_volatility_analysis_to_db(self, symbol, timeframe, volatility_data):
        """
        Save volatility analysis results to database

        Args:
            symbol (str): Cryptocurrency symbol
            timeframe (str): Timeframe of the analysis
            volatility_data (pd.DataFrame): Volatility metrics and analysis

        Returns:
            bool: Success status
        """
        try:
            # Create table name
            table_name = f"volatility_analysis_{symbol.lower()}_{timeframe}"

            # Save to database
            with self.db_connection.cursor() as cursor:
                # Check if table exists
                cursor.execute(f"SELECT to_regclass('public.{table_name}')")
                table_exists = cursor.fetchone()[0] is not None

                if not table_exists:
                    # Create table if it doesn't exist
                    columns = ', '.join([f"{col} FLOAT" for col in volatility_data.columns])
                    cursor.execute(f"""
                        CREATE TABLE {table_name} (
                            timestamp TIMESTAMP PRIMARY KEY,
                            {columns}
                        )
                    """)

                # Insert data
                for timestamp, row in volatility_data.iterrows():
                    placeholders = ', '.join(['%s'] * (len(row) + 1))
                    columns = 'timestamp, ' + ', '.join(row.index)
                    values = [timestamp] + list(row.values)

                    cursor.execute(f"""
                        INSERT INTO {table_name} ({columns})
                        VALUES ({placeholders})
                        ON CONFLICT (timestamp) DO UPDATE
                        SET {', '.join([f"{col} = EXCLUDED.{col}" for col in row.index])}
                    """, values)

                self.db_connection.commit()

            return True

        except Exception as e:
            logger.error(f"Error saving volatility analysis to database: {e}")
            self.db_connection.rollback()
            return False

    def load_volatility_analysis_from_db(self, symbol, timeframe, start_date=None, end_date=None):
        """
        Load volatility analysis from database

        Args:
            symbol (str): Cryptocurrency symbol
            timeframe (str): Timeframe of the analysis
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)

        Returns:
            pd.DataFrame: Volatility analysis data
        """
        try:
            # Create table name
            table_name = f"volatility_analysis_{symbol.lower()}_{timeframe}"

            # Build query
            query = f"SELECT * FROM {table_name}"
            params = []

            if start_date or end_date:
                query += " WHERE"

                if start_date:
                    query += " timestamp >= %s"
                    params.append(start_date)

                    if end_date:
                        query += " AND"

                if end_date:
                    query += " timestamp <= %s"
                    params.append(end_date)

            query += " ORDER BY timestamp"

            # Execute query
            volatility_data = pd.read_sql(query, self.db_connection, params=params, index_col='timestamp')

            return volatility_data

        except Exception as e:
            logger.error(f"Error loading volatility analysis from database: {e}")
            return None

    def analyze_crypto_market_conditions(self, symbols=['BTC', 'ETH', 'BNB'], timeframe='1d', window=14):
        """
        Analyze overall crypto market volatility conditions

        Args:
            symbols (list): List of cryptocurrency symbols to analyze
            timeframe (str): Timeframe to analyze
            window (int): Window for volatility calculation

        Returns:
            dict: Dictionary with market volatility metrics
        """
        try:
            volatilities = {}

            # Get volatility for each symbol
            for symbol in symbols:
                data = load_crypto_data(f"{symbol}USDT", timeframe=timeframe)
                vol = self.calculate_historical_volatility(data['close'], window=window)
                volatilities[symbol] = vol

            # Convert to DataFrame
            vol_df = pd.DataFrame(volatilities)

            # Calculate market-wide metrics
            market_vol = vol_df.mean(axis=1)  # Average volatility across assets
            vol_dispersion = vol_df.std(axis=1)  # Dispersion in volatility
            vol_correlation = vol_df.corr().mean().mean()  # Average correlation

            # Get market phases using helper function
            market_phases = get_market_phases(vol_df)
            current_phase = market_phases.iloc[-1] if not market_phases.empty else None

            # Determine if in volatility regime shift
            regime_shifts = {}
            for symbol in symbols:
                regimes = self.detect_volatility_regimes(volatilities[symbol])
                # Check if regime changed in last 3 periods
                recent_changes = regimes.diff().iloc[-3:].abs().sum()
                regime_shifts[symbol] = recent_changes > 0

            # Return consolidated market analysis
            return {
                'average_market_vol': market_vol.iloc[-1],
                'vol_trend_30d': (market_vol.iloc[-1] / market_vol.iloc[-30]) - 1 if len(market_vol) >= 30 else None,
                'vol_dispersion': vol_dispersion.iloc[-1],
                'vol_correlation': vol_correlation,
                'market_phase': current_phase,
                'regime_shifts': regime_shifts,
                'high_vol_assets': [s for s, v in vol_df.iloc[-1].items() if v > vol_df.iloc[-1].mean()],
                'low_vol_assets': [s for s, v in vol_df.iloc[-1].items() if v < vol_df.iloc[-1].mean()]
            }

        except Exception as e:
            logger.error(f"Error analyzing crypto market conditions: {e}")
            return None

    def run_full_volatility_analysis(self, symbol, timeframe='1d', save_to_db=True):
        """
        Run complete volatility analysis for a cryptocurrency

        Args:
            symbol (str): Cryptocurrency symbol
            timeframe (str): Timeframe for analysis
            save_to_db (bool): Whether to save results to database

        Returns:
            dict: Dictionary with analysis results
        """
        try:
            # Load data
            data = load_crypto_data(symbol, timeframe=timeframe)

            # Calculate returns
            data['returns'] = data['close'].pct_change()

            # Calculate various volatility metrics
            volatility = {}

            # Historical volatility for different windows
            for window in [7, 14, 30, 60]:
                volatility[f'hist_vol_{window}d'] = self.calculate_historical_volatility(
                    data['close'], window=window)

            # Parkinson and other volatility measures
            volatility['parkinson_vol'] = self.calculate_parkinson_volatility(data)
            volatility['gk_vol'] = self.calculate_garman_klass_volatility(data)
            volatility['yz_vol'] = self.calculate_yang_zhang_volatility(data)

            # Convert to DataFrame
            vol_df = pd.DataFrame(volatility)

            # Detect volatility regimes
            vol_df['regime'] = self.detect_volatility_regimes(vol_df['hist_vol_14d'])

            # Analyze volatility clustering
            acf_data = self.analyze_volatility_clustering(data['returns'])

            # Calculate risk metrics
            risk_metrics = self.calculate_volatility_risk_metrics(
                data['returns'], vol_df['hist_vol_14d'])

            # Identify volatility breakouts
            vol_df['breakout'] =