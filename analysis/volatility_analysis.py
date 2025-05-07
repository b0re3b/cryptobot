import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import arch
from arch import arch_model

# Import from other project modules
from data.db import DatabaseManager
from data_collection import  DataCleaner
from models import TimeSeriesModels
from data_collection import AnomalyDetector
from data_collection import FeatureEngineering
from utils.logger import get_logger

logger = get_logger(__name__)


class VolatilityAnalysis:
    """Class for analyzing cryptocurrency market volatility patterns"""

    def __init__(self):
        """Initialize the volatility analysis with optional database connection"""
        self.db_manager = DatabaseManager()
        self.volatility_models = {}
        self.regime_models = {}
        self.data_cleaner = DataCleaner()
        self.anomaly_detector = AnomalyDetector()
        self.feature_engineer = FeatureEngineering()
        self.time_series = TimeSeriesModels()

    def get_market_phases(self, volatility_data, lookback_window=90, n_regimes=4):

        try:
            # Отримати останні дані в межах вікна аналізу
            recent_data = volatility_data.iloc[-lookback_window:] if len(
                volatility_data) > lookback_window else volatility_data

            # Розрахувати показники волатильності для всього ринку
            market_vol = recent_data.mean(axis=1)  # Середня волатильність по всіх активах
            vol_dispersion = recent_data.std(axis=1)  # Розсіювання (дисперсія) волатильності
            vol_trend = market_vol.diff(5).rolling(window=10).mean()  # Тренд волатильності (різниця + згладжування)

            # Об'єднати показники у матрицю ознак для кластеризації
            features = pd.DataFrame({
                'market_vol': market_vol,
                'vol_dispersion': vol_dispersion,
                'vol_trend': vol_trend
            }).dropna()

            # Якщо даних недостатньо — повертаємо порожню серію
            if len(features) < 10:
                return pd.Series(index=volatility_data.index)

            # Нормалізуємо ознаки для кластеризації
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)

            # Застосовуємо кластеризацію KMeans для виділення фаз ринку
            kmeans = KMeans(n_clusters=n_regimes, random_state=42)
            phases = kmeans.fit_predict(scaled_features)

            # Отримати центроїди кластерів
            centroids = kmeans.cluster_centers_

            # Визначаємо порядок фаз за рівнем волатильності (перша ознака)
            vol_level_order = np.argsort(centroids[:, 0])

            # Формуємо список назв для фаз
            phase_names = ['Low Vol', 'Normal Vol', 'High Vol', 'Extreme Vol']
            if n_regimes < 4:
                phase_names = phase_names[:n_regimes]
            elif n_regimes > 4:
                phase_names.extend([f'Корист. фаза {i + 1}' for i in range(n_regimes - 4)])

            # Створюємо відповідність між кластерами та назвами фаз
            phase_mapping = {vol_level_order[i]: phase_names[i] for i in range(n_regimes)}

            # Присвоюємо фази кожному рядку
            phase_series = pd.Series(phases, index=features.index).map(phase_mapping)

            # Розширюємо серію на повний період вхідних даних, заповнюючи пропуски
            full_phase_series = pd.Series(index=volatility_data.index)
            full_phase_series.loc[phase_series.index] = phase_series
            full_phase_series = full_phase_series.ffill().bfill()  # Заповнення в обидва боки

            # Зберігаємо модель для подальшого використання
            self.regime_models[f"market_phases_{n_regimes}"] = {
                'model': kmeans,
                'scaler': scaler,
                'features': list(features.columns),
                'mapping': phase_mapping
            }

            return full_phase_series

        except Exception as e:
            logger.error(f"Помилка при визначенні фаз ринку: {e}")
            return pd.Series(index=volatility_data.index)

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
            clean_returns = self.data_cleaner.clean_data(returns)

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
            data = self.db_manager.get_klines(symbol, timeframe=timeframe)

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
        vol_features = self.feature_engineer.create_volatility_features(ohlc_data)
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

    def save_volatility_analysis_to_db(self, symbol, timeframe, volatility_data, model_data=None, regime_data=None,
                                       features_data=None, cross_asset_data=None):

        try:
            logger.info(f"Saving volatility analysis for {symbol} on {timeframe} timeframe")
            success = True

            # 1. Save main volatility metrics
            if volatility_data is not None and not volatility_data.empty:
                metrics_success = self.db_manager.save_volatility_metrics(
                    symbol=symbol,
                    timeframe=timeframe,
                    metrics_data=volatility_data
                )
                success = success and metrics_success
                logger.info(f"Saved volatility metrics: {metrics_success}")

            # 2. Save volatility models (GARCH, etc.)
            if model_data is not None:
                model_success = self.db_manager.save_volatility_model(
                    symbol=symbol,
                    timeframe=timeframe,
                    model_name=model_data.get('name', 'garch'),
                    model_params=model_data.get('params', {}),
                    forecast_data=model_data.get('forecast'),
                    model_stats=model_data.get('stats', {})
                )
                success = success and model_success
                logger.info(f"Saved volatility model: {model_success}")

            # 3. Save regime data
            if regime_data is not None:
                regime_success = self.db_manager.save_volatility_regime(
                    symbol=symbol,
                    timeframe=timeframe,
                    regime_data=regime_data.get('regimes'),
                    regime_method=regime_data.get('method', 'kmeans'),
                    regime_params=regime_data.get('params', {})
                )
                success = success and regime_success
                logger.info(f"Saved volatility regimes: {regime_success}")

            # 4. Save ML features
            if features_data is not None and not features_data.empty:
                features_success = self.db_manager.save_volatility_features(
                    symbol=symbol,
                    timeframe=timeframe,
                    features_data=features_data
                )
                success = success and features_success
                logger.info(f"Saved volatility features: {features_success}")

            # 5. Save cross-asset volatility data
            if cross_asset_data is not None and not cross_asset_data.empty:
                cross_asset_success = self.db_manager.save_cross_asset_volatility(
                    base_symbol=symbol,
                    timeframe=timeframe,
                    correlation_data=cross_asset_data
                )
                success = success and cross_asset_success
                logger.info(f"Saved cross-asset volatility: {cross_asset_success}")

            return success

        except Exception as e:
            logger.error(f"Error saving volatility analysis to database: {e}")
            return False

    def load_volatility_analysis_from_db(self, symbol, timeframe, start_date=None, end_date=None):
        """
        Load comprehensive volatility analysis from database using specialized DB manager methods

        Args:
            symbol (str): Cryptocurrency symbol
            timeframe (str): Timeframe of the analysis
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)

        Returns:
            dict: Dictionary containing all volatility analysis components
        """
        try:
            logger.info(f"Loading volatility analysis for {symbol} on {timeframe} timeframe")

            # 1. Load volatility metrics
            metrics_data = self.db_manager.get_volatility_metrics(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )

            # 2. Load volatility model data (default to GARCH)
            model_data = self.db_manager.get_volatility_model(
                symbol=symbol,
                timeframe=timeframe,
                model_name='garch'
            )

            # 3. Load regime data
            regime_data = self.db_manager.get_volatility_regime(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )

            # 4. Load ML features
            features_data = self.db_manager.get_volatility_features(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )

            # 5. Load cross-asset volatility
            cross_asset_data = self.db_manager.get_cross_asset_volatility(
                base_symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )

            # Combine all results into a comprehensive analysis object
            analysis_results = {
                'symbol': symbol,
                'timeframe': timeframe,
                'volatility_metrics': metrics_data,
                'model_data': model_data,
                'regime_data': regime_data,
                'features_data': features_data,
                'cross_asset_data': cross_asset_data,
                'period': {
                    'start_date': start_date,
                    'end_date': end_date
                }
            }

            # Add summary info if metrics data is available
            if metrics_data is not None and not metrics_data.empty:
                main_vol_col = next((col for col in metrics_data.columns if 'hist_vol_14d' in col), None)
                if main_vol_col:
                    analysis_results['summary'] = {
                        'avg_volatility': metrics_data[main_vol_col].mean(),
                        'current_volatility': metrics_data[main_vol_col].iloc[-1] if not metrics_data.empty else None,
                        'volatility_trend': 'increasing' if metrics_data[main_vol_col].iloc[-1] >
                                                            metrics_data[main_vol_col].iloc[-7]
                        else 'decreasing' if len(metrics_data) >= 7 else 'unknown',
                        'max_volatility': metrics_data[main_vol_col].max(),
                        'min_volatility': metrics_data[main_vol_col].min(),
                        'volatility_of_volatility': metrics_data[main_vol_col].std() / metrics_data[main_vol_col].mean()
                        if not metrics_data.empty else None
                    }

            return analysis_results

        except Exception as e:
            logger.error(f"Error loading volatility analysis from database: {e}")
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'error': str(e)
            }

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
                data = self.db_manager.get_klines(f"{symbol}USDT", timeframe=timeframe)
                vol = self.calculate_historical_volatility(data['close'], window=window)
                volatilities[symbol] = vol

            # Convert to DataFrame
            vol_df = pd.DataFrame(volatilities)

            # Calculate market-wide metrics
            market_vol = vol_df.mean(axis=1)  # Average volatility across assets
            vol_dispersion = vol_df.std(axis=1)  # Dispersion in volatility
            vol_correlation = vol_df.corr().mean().mean()  # Average correlation

            # Get market phases using helper function
            market_phases = self.get_market_phases(vol_df)
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

        This method performs a comprehensive volatility analysis including multiple volatility metrics,
        regime detection, seasonality analysis, volatility clustering, risk metrics, and breakout detection.

        Args:
            symbol (str): Cryptocurrency symbol
            timeframe (str): Timeframe for analysis ('1h', '4h', '1d', '1w', etc.)
            save_to_db (bool): Whether to save results to database

        Returns:
            dict: Dictionary with analysis results
        """
        try:
            logger.info(f"Running full volatility analysis for {symbol} on {timeframe} timeframe")

            # Load data
            data = self.db_manager.get_klines(symbol, timeframe=timeframe)

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

            # Create regime data dictionary for database
            regime_data = {
                'regimes': vol_df['regime'],
                'method': 'kmeans',
                'params': {
                    'n_regimes': 3,
                    'feature': 'hist_vol_14d'
                }
            }

            # Analyze volatility clustering
            acf_data = self.analyze_volatility_clustering(data['returns'])

            # Calculate risk metrics
            risk_metrics = self.calculate_volatility_risk_metrics(
                data['returns'], vol_df['hist_vol_14d'])

            # Identify volatility breakouts
            vol_df['breakout'] = self.identify_volatility_breakouts(vol_df['hist_vol_14d'])

            # Get seasonality patterns
            seasonality = {}
            seasonality['dow'] = self.extract_seasonality_in_volatility(vol_df['hist_vol_14d'], period=7)
            seasonality['month'] = self.extract_seasonality_in_volatility(vol_df['hist_vol_14d'], period=12)

            # Fit GARCH model
            garch_model, garch_forecast = self.fit_garch_model(data['returns'])

            # Extract forecast values if model was successfully fit
            forecast_values = None
            if garch_model is not None:
                forecast_values = garch_forecast.variance.iloc[-1].values

            # Create model data dictionary for database
            model_data = {
                'name': 'garch',
                'params': {
                    'p': 1,
                    'q': 1,
                    'mean': 'Zero',
                    'vol': 'GARCH'
                },
                'forecast': forecast_values,
                'stats': {
                    'aic': garch_model.aic if garch_model is not None else None,
                    'bic': garch_model.bic if garch_model is not None else None
                }
            }

            # Calculate volatility impulse response
            impulse_response = self.volatility_impulse_response(data['returns'])

            # Prepare features for ML models
            ml_features = self.prepare_volatility_features_for_ml(data)

            # Get market-wide conditions for context
            market_conditions = self.analyze_crypto_market_conditions(
                symbols=[symbol, 'BTC', 'ETH'], timeframe=timeframe)

            # Get cross-asset correlation data
            cross_asset_symbols = ['BTC', 'ETH', 'BNB', 'XRP']
            asset_dict = {}
            for asset in cross_asset_symbols:
                asset_data = self.db_manager.get_klines(f"{asset}USDT", timeframe=timeframe)
                if asset_data is not None and not asset_data.empty:
                    asset_dict[asset] = asset_data['close']

            cross_asset_vol = self.analyze_cross_asset_volatility(asset_dict) if asset_dict else None

            # Combine all results
            analysis_results = {
                'symbol': symbol,
                'timeframe': timeframe,
                'volatility_data': vol_df,
                'latest_volatility': {
                    'hist_vol_14d': vol_df['hist_vol_14d'].iloc[-1] if not vol_df.empty else None,
                    'parkinson': vol_df['parkinson_vol'].iloc[-1] if not vol_df.empty else None,
                    'garman_klass': vol_df['gk_vol'].iloc[-1] if not vol_df.empty else None,
                    'yang_zhang': vol_df['yz_vol'].iloc[-1] if not vol_df.empty else None
                },
                'current_regime': vol_df['regime'].iloc[-1] if not vol_df.empty else None,
                'volatility_clustering': {
                    'significant_lags': acf_data[acf_data['autocorrelation'] > 0.1]['lag'].tolist(),
                    'max_autocorrelation': acf_data['autocorrelation'].max()
                },
                'risk_metrics': risk_metrics,
                'seasonality': seasonality,
                'recent_breakouts': vol_df['breakout'].iloc[-30:].sum() if len(vol_df) >= 30 else 0,
                'garch_forecast': forecast_values,
                'impulse_response': impulse_response,
                'market_conditions': market_conditions
            }

            # Calculate summary stats
            analysis_results['summary'] = {
                'avg_volatility': vol_df['hist_vol_14d'].mean(),
                'volatility_trend': 'increasing' if vol_df['hist_vol_14d'].iloc[-1] > vol_df['hist_vol_14d'].iloc[
                    -7] else 'decreasing',
                'regime_changes': vol_df['regime'].diff().abs().sum(),
                'volatility_of_volatility': vol_df['hist_vol_14d'].rolling(window=14).std().iloc[-1] if len(
                    vol_df) >= 14 else None,
                'current_vs_historical': vol_df['hist_vol_14d'].iloc[-1] / vol_df[
                    'hist_vol_14d'].mean() if not vol_df.empty else None
            }

            # Save to database if requested
            if save_to_db:
                logger.info(f"Saving volatility analysis for {symbol} to database")
                save_success = self.save_volatility_analysis_to_db(
                    symbol=symbol,
                    timeframe=timeframe,
                    volatility_data=vol_df,
                    model_data=model_data,
                    regime_data=regime_data,
                    features_data=ml_features,
                    cross_asset_data=cross_asset_vol
                )
                analysis_results['saved_to_db'] = save_success

            # Generate reports and visualizations
            self._generate_volatility_report(symbol, timeframe, analysis_results)

            return analysis_results

        except Exception as e:
            logger.error(f"Error in full volatility analysis for {symbol}: {e}")
            # Return partial results if available
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'error': str(e),
                'partial_results': locals().get('vol_df', None)
            }

    def _generate_volatility_report(self, symbol, timeframe, analysis_results):
        """
        Generate volatility report and visualizations

        Args:
            symbol (str): Cryptocurrency symbol
            timeframe (str): Timeframe of analysis
            analysis_results (dict): Analysis results

        Returns:
            None
        """
        try:
            vol_df = analysis_results['volatility_data']

            # Create plots directory if it doesn't exist
            import os
            os.makedirs('reports/volatility', exist_ok=True)

            # Plot volatility metrics
            plt.figure(figsize=(12, 8))
            for col in ['hist_vol_14d', 'parkinson_vol', 'gk_vol', 'yz_vol']:
                if col in vol_df.columns:
                    plt.plot(vol_df.index, vol_df[col], label=col)
            plt.title(f"{symbol} Volatility Metrics - {timeframe}")
            plt.legend()
            plt.savefig(f"reports/volatility/{symbol}_{timeframe}_volatility_metrics.png")

            # Plot volatility regimes
            plt.figure(figsize=(12, 6))
            plt.plot(vol_df.index, vol_df['hist_vol_14d'], label='Historical Vol (14d)')
            plt.scatter(vol_df.index, vol_df['hist_vol_14d'], c=vol_df['regime'], cmap='viridis', label='Regimes')
            plt.title(f"{symbol} Volatility Regimes - {timeframe}")
            plt.colorbar(label='Regime')
            plt.legend()
            plt.savefig(f"reports/volatility/{symbol}_{timeframe}_volatility_regimes.png")

            # Plot seasonality
            if 'seasonality' in analysis_results and 'dow' in analysis_results['seasonality']:
                plt.figure(figsize=(10, 6))
                analysis_results['seasonality']['dow'].plot(kind='bar')
                plt.title(f"{symbol} Day-of-Week Volatility Seasonality")
                plt.savefig(f"reports/volatility/{symbol}_{timeframe}_dow_seasonality.png")

            logger.info(f"Generated volatility report for {symbol}")

        except Exception as e:
            logger.error(f"Error generating volatility report: {e}")