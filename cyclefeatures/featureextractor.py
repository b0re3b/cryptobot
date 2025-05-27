from typing import Tuple, List
import decimal
import numpy as np
import pandas as pd

from cyclefeatures.MarketPhaseFeatureExtractor import MarketPhaseFeatureExtractor
from cyclefeatures.BitcoinCycleFeatureExtractor import BitcoinCycleFeatureExtractor
from cyclefeatures.SolanaCycleFeatureExtractor import SolanaCycleFeatureExtractor
from cyclefeatures.EthereumCycleFeatureExtractor import EthereumCycleFeatureExtractor
from utils.logger import CryptoLogger
from utils.config import *


class FeatureExtractor:
    def __init__(self):
        self.logger = CryptoLogger('FeatureExtractor')
        self.btcycle = BitcoinCycleFeatureExtractor()
        self.ethcycle = EthereumCycleFeatureExtractor()
        self.solanacycle = SolanaCycleFeatureExtractor()
        # Bitcoin halving dates
        self.btc_halving_dates = btc_halving_dates
        # Ethereum significant network upgrades/events
        self.eth_significant_events = eth_significant_events
        # Solana significant events
        self.sol_significant_events = sol_significant_events
        # Dictionary to map symbol to its significant events
        self.symbol_events_map = {
            "BTC": self.btc_halving_dates,
            "ETH": self.eth_significant_events,
            "SOL": self.sol_significant_events
        }
        self.marketplace = MarketPhaseFeatureExtractor()
        self.logger.info("FeatureExtractor initialized")

    def get_significant_events_for_symbol(self, symbol: str) -> List:
        symbol = symbol.upper().replace('USDT', '').replace('USD', '')
        events = self.symbol_events_map.get(symbol, [])
        self.logger.debug(f"Retrieved {len(events)} significant events for {symbol}")
        return events

    def calculate_token_specific_cycle_features(self,
                                                processed_data: pd.DataFrame,
                                                symbol: str) -> pd.DataFrame:
        symbol = symbol.upper().replace('USDT', '').replace('USD', '')
        self.logger.info(f"Calculating token-specific cycle features for {symbol}")

        if symbol == 'BTC':
            return self.btcycle.calculate_btc_halving_cycle_features(processed_data)
        elif symbol == 'ETH':
            return self.ethcycle.calculate_eth_event_cycle_features(processed_data)
        elif symbol == 'SOL':
            return self.solanacycle.calculate_sol_event_cycle_features(processed_data)
        else:
            self.logger.warning(f"No specific cycle features available for {symbol}. Returning original data.")
            return processed_data

    def create_cyclical_features(self, processed_data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        self.logger.info(f"Creating cyclical features for {symbol}")

        # First, add general cyclical features (common to all cryptocurrencies)
        result_df = processed_data.copy()

        # Ensure the DataFrame has a datetime index
        if not isinstance(result_df.index, pd.DatetimeIndex):
            self.logger.error("DataFrame index must be a DatetimeIndex")
            raise ValueError("DataFrame index must be a DatetimeIndex")

        try:
            # Add day of week cyclical features
            result_df['day_of_week'] = result_df.index.dayofweek
            # Create sine and cosine features for day of week (better for machine learning)
            result_df['day_of_week_sin'] = np.sin(result_df['day_of_week'] * (2 * np.pi / 7))
            result_df['day_of_week_cos'] = np.cos(result_df['day_of_week'] * (2 * np.pi / 7))

            # Add month cyclical features
            result_df['month'] = result_df.index.month
            # Create sine and cosine features for month
            result_df['month_sin'] = np.sin(result_df['month'] * (2 * np.pi / 12))
            result_df['month_cos'] = np.cos(result_df['month'] * (2 * np.pi / 12))

            # Add quarter cyclical features
            result_df['quarter'] = result_df.index.quarter
            # Create sine and cosine features for quarter
            result_df['quarter_sin'] = np.sin(result_df['quarter'] * (2 * np.pi / 4))
            result_df['quarter_cos'] = np.cos(result_df['quarter'] * (2 * np.pi / 4))

            # Add day of month cyclical features
            result_df['day_of_month'] = result_df.index.day
            # Create sine and cosine features for day of month
            result_df['day_of_month_sin'] = np.sin(result_df['day_of_month'] * (2 * np.pi / 31))
            result_df['day_of_month_cos'] = np.cos(result_df['day_of_month'] * (2 * np.pi / 31))

            # Add week of year cyclical features
            result_df['week_of_year'] = result_df.index.isocalendar().week
            # Create sine and cosine features for week of year
            result_df['week_of_year_sin'] = np.sin(result_df['week_of_year'] * (2 * np.pi / 52))
            result_df['week_of_year_cos'] = np.cos(result_df['week_of_year'] * (2 * np.pi / 52))

            self.logger.debug("Added basic cyclical time features")

            # Add market phase features
            try:
                market_phase_df = self.marketplace.detect_market_phase(result_df)
                # Get only the market_phase column
                if 'market_phase' in market_phase_df.columns:
                    result_df['market_phase'] = market_phase_df['market_phase']
                    # One-hot encode the market phase
                    phase_dummies = pd.get_dummies(result_df['market_phase'], prefix='phase')
                    result_df = pd.concat([result_df, phase_dummies], axis=1)
                    self.logger.debug("Added market phase features")
            except Exception as e:
                self.logger.error(f"Error adding market phase features: {str(e)}")

            # Identify bull/bear cycles
            try:
                bull_bear_df = self.marketplace.identify_bull_bear_cycles(result_df)
                # Add cycle state and ID
                if 'cycle_state' in bull_bear_df.columns:
                    result_df['cycle_state'] = bull_bear_df['cycle_state']
                    result_df['cycle_id'] = bull_bear_df['cycle_id']
                    # One-hot encode the cycle state
                    state_dummies = pd.get_dummies(result_df['cycle_state'], prefix='state')
                    result_df = pd.concat([result_df, state_dummies], axis=1)
                    self.logger.debug("Added bull/bear cycle features")
            except Exception as e:
                self.logger.error(f"Error detecting bull/bear cycles: {str(e)}")

            # Then add token-specific cycle features
            result_df = self.calculate_token_specific_cycle_features(result_df, symbol)

            # Try to find and add optimal cycle features
            try:
                optimal_cycle_length, cycle_strength = self.find_optimal_cycle_length(result_df)
                self.logger.info(f"Found optimal cycle length: {optimal_cycle_length}, strength: {cycle_strength:.4f}")

                # If a strong cycle is found, add related features
                if cycle_strength > 0.3:  # Only add if moderately strong cycle detected
                    result_df['optimal_cycle_length'] = optimal_cycle_length
                    result_df['optimal_cycle_strength'] = cycle_strength
                    # Calculate days into the optimal cycle
                    result_df['days_into_optimal_cycle'] = result_df.index.dayofyear % optimal_cycle_length
                    # Normalized position in the optimal cycle (0 to 1)
                    result_df['optimal_cycle_phase'] = result_df['days_into_optimal_cycle'] / float(
                        optimal_cycle_length)
                    # Create sine and cosine features for the optimal cycle
                    cycle_phase = result_df['optimal_cycle_phase'] * 2 * np.pi
                    result_df['optimal_cycle_sin'] = np.sin(cycle_phase)
                    result_df['optimal_cycle_cos'] = np.cos(cycle_phase)
                    self.logger.debug("Added optimal cycle features")
            except Exception as e:
                self.logger.error(f"Error finding optimal cycle: {str(e)}")

            # Drop the original categorical columns if one-hot encoding was successful
            if 'phase_accumulation' in result_df.columns:
                result_df = result_df.drop(columns=['market_phase'], errors='ignore')
            if 'state_bull' in result_df.columns:
                result_df = result_df.drop(columns=['cycle_state'], errors='ignore')

            self.logger.info(f"Successfully created cyclical features for {symbol}")
            return result_df

        except Exception as e:
            self.logger.error(f"Error in create_cyclical_features: {str(e)}")
            raise

    def find_optimal_cycle_length(self, processed_data: pd.DataFrame,
                                  min_period: int = 7,
                                  max_period: int = 365) -> Tuple[int, float]:
        self.logger.info(f"Finding optimal cycle length between {min_period} and {max_period} days")

        try:
            # Ensure we have the required columns
            if 'close' not in processed_data.columns:
                self.logger.error("Required column 'close' not found in processed_data")
                raise ValueError("Required column 'close' not found in processed_data")

            # Ensure datetime index and resample to daily frequency if needed
            df = processed_data.copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                self.logger.error("DataFrame index must be a DatetimeIndex")
                raise ValueError("DataFrame index must be a DatetimeIndex")

            # Create a daily price series if data is not already daily
            if df.index.to_series().diff().mean().total_seconds() != 86400:
                self.logger.debug("Resampling data to daily frequency")
                daily_df = df.resample('D').last()
                price_series = daily_df['close'].dropna()
            else:
                price_series = df['close'].dropna()

            # Convert any decimal.Decimal values to float
            if price_series.dtype == object:
                self.logger.debug("Converting price series to float")
                price_series = price_series.astype(float)

            # If data is limited, adjust the maximum period
            if len(price_series) < max_period * 2:
                adjusted_max_period = min(len(price_series) // 2, max_period)
                self.logger.warning(
                    f"Limited data available. Adjusting max_period from {max_period} to {adjusted_max_period}")
                max_period = adjusted_max_period

            # Calculate price returns for better stationarity
            returns = price_series.pct_change().dropna()

            # Detect seasonality using autocorrelation
            correlations = []

            # Calculate autocorrelation for different lags
            for lag in range(min_period, max_period + 1):
                if len(returns) > lag:
                    # Calculate autocorrelation
                    corr = returns.autocorr(lag=lag)
                    # Ensure result is a float
                    corr = float(corr) if isinstance(corr, decimal.Decimal) else corr
                    correlations.append((lag, corr))

            # If no valid correlations were found, return default values
            if not correlations:
                self.logger.warning("No valid correlations found. Returning default values.")
                return (max_period, 0.0)

            # Convert to DataFrame for easier manipulation
            corr_df = pd.DataFrame(correlations, columns=['lag', 'autocorrelation'])

            # Find local maxima in the autocorrelation function
            local_maxima = []
            for i in range(1, len(corr_df) - 1):
                if (corr_df['autocorrelation'].iloc[i] > corr_df['autocorrelation'].iloc[i - 1] and
                        corr_df['autocorrelation'].iloc[i] > corr_df['autocorrelation'].iloc[i + 1]):
                    local_maxima.append((
                        int(corr_df['lag'].iloc[i]),
                        float(corr_df['autocorrelation'].iloc[i])
                    ))

            # If no local maxima found, return the highest correlation
            if not local_maxima:
                self.logger.info("No local maxima found in autocorrelation. Using highest correlation.")
                best_lag_idx = corr_df['autocorrelation'].idxmax()
                best_lag = corr_df.loc[best_lag_idx]
                self.logger.info(
                    f"Optimal cycle: lag={int(best_lag['lag'])}, correlation={float(best_lag['autocorrelation']):.4f}")
                return (int(best_lag['lag']), float(best_lag['autocorrelation']))

            # Sort local maxima by correlation strength
            local_maxima.sort(key=lambda x: x[1], reverse=True)

            optimal_lag = int(local_maxima[0][0])
            optimal_corr = float(local_maxima[0][1])
            self.logger.info(f"Found optimal cycle: lag={optimal_lag}, correlation={optimal_corr:.4f}")

            # Return the highest correlation local maximum
            return (optimal_lag, optimal_corr)

        except Exception as e:
            self.logger.error(f"Error in find_optimal_cycle_length: {str(e)}")
            # Return a default value when an error occurs
            return (max_period, 0.0)

    def calculate_cycle_roi(self, processed_data: pd.DataFrame,
                            symbol: str,
                            cycle_type: str = 'auto',
                            normalized: bool = True) -> pd.DataFrame:
        """
        Calculate ROI (Return on Investment) based on different market cycles.

        Args:
            processed_data: DataFrame with price data (must include 'close' column with datetime index)
            symbol: Trading symbol (e.g. 'BTC', 'ETH', 'SOL')
            cycle_type: Type of cycle analysis ('auto', 'halving', 'network_upgrade', 'ecosystem_event', 'bull_bear', 'custom')
            normalized: Whether to normalize ROI values for comparison

        Returns:
            DataFrame with ROI calculations based on cycle type
        """
        self.logger.info(f"Calculating cycle ROI for {symbol} using cycle type: {cycle_type}")

        try:
            # Create a copy of the input DataFrame
            df = processed_data.copy()

            # Ensure the DataFrame has a datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                self.logger.error("DataFrame index must be a DatetimeIndex")
                raise ValueError("DataFrame index must be a DatetimeIndex")

            # Ensure we have the required columns
            if 'close' not in df.columns:
                self.logger.error("Required column 'close' not found in processed_data")
                raise ValueError("Required column 'close' not found in processed_data")

            # Convert any decimal.Decimal values to float for the close column
            if df['close'].dtype == object:
                self.logger.debug("Converting close prices from Decimal to float")
                df['close'] = df['close'].astype(float)

            # Determine the cycle type to use based on the symbol if set to 'auto'
            if cycle_type == 'auto':
                symbol = symbol.upper().replace('USDT', '').replace('USD', '')
                if symbol == 'BTC':
                    cycle_type = 'halving'
                elif symbol == 'ETH':
                    cycle_type = 'network_upgrade'
                elif symbol == 'SOL':
                    cycle_type = 'ecosystem_event'
                else:
                    cycle_type = 'bull_bear'
                self.logger.info(f"Auto detected cycle type: {cycle_type} for {symbol}")

            # Initialize result DataFrame
            result_df = pd.DataFrame(index=df.index)
            result_df['close'] = df['close']
            result_df['date'] = df.index

            if cycle_type == 'halving':
                self.logger.debug("Processing halving cycle ROI")
                # For BTC halving cycles
                try:
                    halving_df = self.btcycle.calculate_btc_halving_cycle_features(df)

                    # Ensure numeric types are float, not Decimal
                    numeric_cols = halving_df.select_dtypes(include=['number']).columns
                    for col in numeric_cols:
                        if halving_df[col].dtype == object:  # Possible Decimal objects
                            halving_df[col] = halving_df[col].astype(float)

                    # Group by cycle number
                    for cycle_num in halving_df['cycle_number'].unique():
                        if pd.isna(cycle_num) or cycle_num == 0:
                            continue

                        cycle_data = halving_df[halving_df['cycle_number'] == cycle_num]
                        if len(cycle_data) > 0:
                            # Ensure we're working with float
                            cycle_start_price = float(cycle_data['close'].iloc[0])
                            cycle_start_date = cycle_data.index[0]

                            # Calculate ROI for this cycle
                            result_df.loc[cycle_data.index, f'cycle_{int(cycle_num)}_roi'] = (
                                    cycle_data['close'] / cycle_start_price - 1
                            )
                            result_df.loc[cycle_data.index, f'cycle_{int(cycle_num)}_days'] = (
                                (cycle_data.index - cycle_start_date).days
                            )

                    # Add phase-based ROI (early, mid, late cycle)
                    halving_df['cycle_phase_category'] = pd.cut(
                        halving_df['halving_cycle_phase'],
                        bins=[0, 0.33, 0.66, 1],
                        labels=['early', 'mid', 'late']
                    )

                    for phase in ['early', 'mid', 'late']:
                        phase_data = halving_df[halving_df['cycle_phase_category'] == phase]
                        if len(phase_data) > 0:
                            phase_returns = phase_data['close'].pct_change().dropna()
                            result_df.loc[phase_data.index, f'phase_{phase}_daily_return'] = phase_returns
                            result_df.loc[phase_data.index, f'phase_{phase}_volatility'] = (
                                phase_returns.rolling(window=14).std()
                            )

                    self.logger.debug(f"Processed halving cycles. Found {halving_df['cycle_number'].nunique()} cycles.")
                except Exception as e:
                    self.logger.error(f"Error processing halving cycles: {str(e)}", exc_info=True)
                    raise

            elif cycle_type == 'network_upgrade':
                self.logger.debug("Processing network upgrade cycle ROI")
                # For ETH network upgrades
                try:
                    events = self.eth_significant_events
                    events_dates = [pd.Timestamp(event['date']) for event in events]

                    for i, event in enumerate(events):
                        event_date = pd.Timestamp(event['date'])
                        event_name = event['name']

                        # Find the next event date
                        next_event_date = None
                        if i < len(events) - 1:
                            next_event_date = pd.Timestamp(events[i + 1]['date'])

                        # Calculate ROI from this event to the next (or to the end)
                        mask = (df.index >= event_date)
                        if next_event_date:
                            mask &= (df.index < next_event_date)

                        event_data = df.loc[mask]
                        if len(event_data) > 0:
                            # Ensure we're working with float
                            event_start_price = float(event_data['close'].iloc[0])
                            event_start_date = event_data.index[0]

                            # Calculate ROI for this event cycle
                            result_df.loc[event_data.index, f'event_{event_name}_roi'] = (
                                    event_data['close'] / event_start_price - 1
                            )
                            result_df.loc[event_data.index, f'event_{event_name}_days'] = (
                                (event_data.index - event_start_date).days
                            )

                    self.logger.debug(f"Processed {len(events)} network upgrade events")
                except Exception as e:
                    self.logger.error(f"Error processing network upgrade cycles: {str(e)}", exc_info=True)
                    raise

            elif cycle_type == 'ecosystem_event':
                self.logger.debug("Processing ecosystem event cycle ROI")
                # For SOL ecosystem events
                try:
                    events = self.sol_significant_events
                    events_dates = [pd.Timestamp(event['date']) for event in events]

                    for i, event in enumerate(events):
                        event_date = pd.Timestamp(event['date'])
                        event_name = event['name'].replace(' ', '_').lower()

                        # Find the next event date
                        next_event_date = None
                        if i < len(events) - 1:
                            next_event_date = pd.Timestamp(events[i + 1]['date'])

                        # Calculate ROI from this event to the next (or to the end)
                        mask = (df.index >= event_date)
                        if next_event_date:
                            mask &= (df.index < next_event_date)

                        event_data = df.loc[mask]
                        if len(event_data) > 0:
                            # Ensure we're working with float
                            event_start_price = float(event_data['close'].iloc[0])
                            event_start_date = event_data.index[0]

                            # Calculate ROI for this event cycle
                            result_df.loc[event_data.index, f'event_{event_name}_roi'] = (
                                    event_data['close'] / event_start_price - 1
                            )
                            result_df.loc[event_data.index, f'event_{event_name}_days'] = (
                                (event_data.index - event_start_date).days
                            )

                    self.logger.debug(f"Processed {len(events)} ecosystem events")
                except Exception as e:
                    self.logger.error(f"Error processing ecosystem events: {str(e)}", exc_info=True)
                    raise

            elif cycle_type == 'bull_bear':
                self.logger.debug("Processing bull/bear cycle ROI")
                # For general bull/bear cycles
                try:
                    bull_bear_df = self.marketplace.identify_bull_bear_cycles(df)

                    # Ensure all numeric columns are float
                    numeric_cols = bull_bear_df.select_dtypes(include=['number']).columns
                    for col in numeric_cols:
                        if bull_bear_df[col].dtype == object:  # Possible Decimal objects
                            bull_bear_df[col] = bull_bear_df[col].astype(float)

                    # Process cycle summary if available
                    if hasattr(bull_bear_df, 'cycles_summary') and len(bull_bear_df.cycles_summary) > 0:
                        cycles_summary = bull_bear_df.cycles_summary

                        # Convert any Decimal values to float in cycles_summary
                        numeric_cols = cycles_summary.select_dtypes(include=['number']).columns
                        for col in numeric_cols:
                            if cycles_summary[col].dtype == object:
                                cycles_summary[col] = cycles_summary[col].astype(float)

                        # For each cycle, calculate ROI
                        for _, cycle in cycles_summary.iterrows():
                            cycle_id = cycle['cycle_id']
                            cycle_type = cycle['cycle_type']
                            start_date = cycle['start_date']
                            end_date = cycle['end_date']

                            # Get data for this cycle
                            mask = (df.index >= start_date) & (df.index <= end_date)
                            cycle_data = df.loc[mask]

                            if len(cycle_data) > 0:
                                # Ensure we're working with float
                                cycle_start_price = float(cycle_data['close'].iloc[0])

                                # Calculate ROI for this cycle
                                result_df.loc[cycle_data.index, f'cycle_{cycle_id}_{cycle_type}_roi'] = (
                                        cycle_data['close'] / cycle_start_price - 1
                                )
                                result_df.loc[cycle_data.index, f'cycle_{cycle_id}_{cycle_type}_days'] = (
                                    (cycle_data.index - start_date).days
                                )

                                # Add normalized time feature (0 to 1 representing position in cycle)
                                cycle_duration = (end_date - start_date).days
                                if cycle_duration > 0:
                                    result_df.loc[cycle_data.index, f'cycle_{cycle_id}_norm_time'] = (
                                            (cycle_data.index - start_date).days / float(cycle_duration)
                                    )

                    # Add current cycle information
                    current_cycle_mask = bull_bear_df['cycle_id'] == bull_bear_df['cycle_id'].max()
                    if current_cycle_mask.any():
                        current_cycle = bull_bear_df[current_cycle_mask]
                        current_cycle_state = current_cycle['cycle_state'].iloc[0]
                        current_cycle_id = current_cycle['cycle_id'].iloc[0]

                        # Ensure we're working with float
                        current_cycle_start_price = float(current_cycle['cycle_start_price'].iloc[0])
                        current_cycle_max_price = float(current_cycle['cycle_max_price'].max())
                        current_cycle_min_price = float(current_cycle['cycle_min_price'].min())

                        result_df.loc[current_cycle.index, 'current_cycle_state'] = current_cycle_state
                        result_df.loc[current_cycle.index, 'current_cycle_id'] = current_cycle_id
                        result_df.loc[current_cycle.index, 'current_cycle_roi'] = (
                                current_cycle['close'] / current_cycle_start_price - 1
                        )
                        result_df.loc[current_cycle.index, 'current_cycle_drawdown'] = (
                                current_cycle['close'] / current_cycle_max_price - 1
                        )
                        result_df.loc[current_cycle.index, 'current_cycle_recovery'] = (
                                current_cycle['close'] / current_cycle_min_price - 1
                        )

                    self.logger.debug(f"Processed bull/bear cycles successfully")

                except Exception as e:
                    self.logger.warning(f"Error in calculating bull/bear cycle ROI: {str(e)}. Using simple approach.",
                                        exc_info=True)
                    # If the bull/bear cycle detection fails, we can use a simpler approach
                    # Calculate ROI from local peaks and troughs

                    # Simple peak and trough detection using rolling max/min
                    window = 30  # 30-day window for detecting peaks and troughs
                    df['rolling_max'] = df['close'].rolling(window=window, center=True).max()
                    df['rolling_min'] = df['close'].rolling(window=window, center=True).min()

                    # Identify potential peaks and troughs
                    df['is_peak'] = (df['close'] == df['rolling_max'])
                    df['is_trough'] = (df['close'] == df['rolling_min'])

                    # Get peak and trough points
                    peaks = df[df['is_peak']].index
                    troughs = df[df['is_trough']].index

                    # Combine and sort all turning points
                    turning_points = sorted(list(peaks) + list(troughs))
                    self.logger.debug(f"Using simple approach. Found {len(peaks)} peaks and {len(troughs)} troughs")

                    # Calculate ROI between each pair of turning points
                    for i in range(len(turning_points) - 1):
                        start_date = turning_points[i]
                        end_date = turning_points[i + 1]

                        period_mask = (df.index >= start_date) & (df.index <= end_date)
                        period_data = df.loc[period_mask]

                        if len(period_data) > 0:
                            # Ensure we're working with float
                            period_start_price = float(period_data['close'].iloc[0])
                            point_type = 'peak_to_trough' if start_date in peaks else 'trough_to_peak'

                            result_df.loc[period_data.index, f'turning_point_{i}_{point_type}_roi'] = (
                                    period_data['close'] / period_start_price - 1
                            )

            elif cycle_type == 'custom':
                self.logger.debug("Processing custom cycle ROI")
                # For custom defined cycles
                try:
                    # Default to optimal cycles detected by the find_optimal_cycle_length method
                    optimal_cycle_length, cycle_strength = self.find_optimal_cycle_length(df)

                    if cycle_strength > 0.2:  # Only use if cycle has reasonable strength
                        # Calculate day of the cycle (0 to optimal_cycle_length-1)
                        result_df['day_of_cycle'] = (df.index.dayofyear % optimal_cycle_length)

                        # Group by day of cycle and calculate average returns
                        avg_returns_by_cycle_day = {}

                        for day in range(optimal_cycle_length):
                            day_data = result_df[result_df['day_of_cycle'] == day]
                            if len(day_data) > 1:
                                day_returns = day_data['close'].pct_change().dropna()
                                avg_returns_by_cycle_day[day] = float(day_returns.mean())

                        # Apply the average returns for each day of the cycle
                        for day, avg_return in avg_returns_by_cycle_day.items():
                            result_df.loc[result_df['day_of_cycle'] == day, 'cycle_day_avg_return'] = avg_return

                        # Calculate cumulative expected return based on cycle day
                        result_df['cycle_expected_return'] = result_df['cycle_day_avg_return'].cumsum()

                        # Normalize the cycle phase (0 to 1)
                        result_df['cycle_phase'] = result_df['day_of_cycle'] / float(optimal_cycle_length)

                        # Create phase categories
                        result_df['cycle_phase_category'] = pd.cut(
                            result_df['cycle_phase'],
                            bins=[0, 0.25, 0.5, 0.75, 1],
                            labels=['phase1', 'phase2', 'phase3', 'phase4']
                        )

                        # Calculate ROI for each phase category
                        for phase in ['phase1', 'phase2', 'phase3', 'phase4']:
                            phase_data = result_df[result_df['cycle_phase_category'] == phase]
                            if len(phase_data) > 0:
                                phase_returns = phase_data['close'].pct_change().dropna()
                                # Ensure we're working with float
                                result_df.loc[phase_data.index, f'{phase}_avg_return'] = float(phase_returns.mean())
                                result_df.loc[phase_data.index, f'{phase}_volatility'] = float(phase_returns.std())

                        self.logger.debug(
                            f"Processed custom cycle with length {optimal_cycle_length} and strength {cycle_strength:.4f}")

                except Exception as e:
                    self.logger.error(f"Error in calculating custom cycle ROI: {str(e)}", exc_info=True)
                    raise

            # Normalize ROI values if requested
            if normalized:
                try:
                    # Identify ROI columns
                    roi_columns = [col for col in result_df.columns if 'roi' in col.lower()]

                    # Calculate z-score normalization for each ROI column
                    for col in roi_columns:
                        mean_val = float(result_df[col].mean())
                        std_val = float(result_df[col].std())

                        if std_val > 0:  # Avoid division by zero
                            result_df[f'{col}_normalized'] = (result_df[col] - mean_val) / std_val

                    self.logger.debug(f"Normalized {len(roi_columns)} ROI columns")
                except Exception as e:
                    self.logger.warning(f"Error normalizing ROI values: {str(e)}", exc_info=True)

            # Make sure all numeric columns are float before returning
            numeric_cols = result_df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                if result_df[col].dtype == object:  # Possible Decimal objects
                    result_df[col] = result_df[col].astype(float)

            self.logger.info(f"Successfully calculated cycle ROI for {symbol} with {len(result_df)} data points")
            return result_df

        except Exception as e:
            self.logger.error(f"Failed to calculate cycle ROI for {symbol}: {str(e)}", exc_info=True)
            raise

    def detect_cycle_anomalies(self, processed_data: pd.DataFrame,
                               symbol: str,
                               cycle_type: str = 'auto') -> pd.DataFrame:

        import logging
        import decimal

        logger = logging.getLogger(__name__)
        logger.info(f"Starting cycle anomaly detection for {symbol} using {cycle_type} cycle type")

        try:
            # Create a copy of the input DataFrame
            result_df = processed_data.copy()

            # Ensure we have the required columns
            if 'close' not in result_df.columns:
                logger.error("Required column 'close' not found in processed_data")
                raise ValueError("Required column 'close' not found in processed_data")

            # Ensure the DataFrame has a datetime index
            if not isinstance(result_df.index, pd.DatetimeIndex):
                logger.error("DataFrame index must be a DatetimeIndex")
                raise ValueError("DataFrame index must be a DatetimeIndex")

            # Clean symbol format
            clean_symbol = symbol.upper().replace('USDT', '').replace('USD', '')
            logger.debug(f"Cleaned symbol: {clean_symbol}")

            # Determine the cycle type based on the symbol if set to 'auto'
            if cycle_type == 'auto':
                if clean_symbol == 'BTC':
                    cycle_type = 'halving'
                elif clean_symbol == 'ETH':
                    cycle_type = 'network_upgrade'
                elif clean_symbol == 'SOL':
                    cycle_type = 'ecosystem_event'
                else:
                    cycle_type = 'bull_bear'  # Default for other cryptocurrencies
                logger.info(f"Auto detected cycle type: {cycle_type}")

            # Initialize the anomalies DataFrame
            anomalies = pd.DataFrame(index=result_df.index)
            anomalies['date'] = anomalies.index
            anomalies['symbol'] = symbol
            anomalies['anomaly_detected'] = False
            anomalies['anomaly_type'] = None
            anomalies['significance_score'] = 0.0
            anomalies['description'] = None

            # Calculate baseline metrics for anomaly detection
            logger.debug("Calculating baseline metrics")
            result_df['price_change_1d'] = result_df['close'].pct_change(1)
            result_df['price_change_7d'] = result_df['close'].pct_change(7)
            result_df['price_change_30d'] = result_df['close'].pct_change(30)
            result_df['volatility_14d'] = result_df['close'].pct_change().rolling(window=14).std()

            # Convert any Decimal columns to float to avoid type errors
            # Also handle string columns that might contain time intervals
            for col in result_df.columns:
                if result_df[col].dtype == object:
                    # Check if it's actually numeric data stored as objects (like Decimal)
                    try:
                        # Try to convert first non-null value to see if it's numeric
                        first_non_null = result_df[col].dropna().iloc[0] if not result_df[col].dropna().empty else None
                        if first_non_null is not None:
                            if isinstance(first_non_null, decimal.Decimal):
                                logger.debug(f"Converting {col} from Decimal to float")
                                result_df[col] = result_df[col].astype(float)
                            elif isinstance(first_non_null, str):
                                # Check if it's a time interval string like '1h', '5m', etc.
                                if any(char in str(first_non_null) for char in ['h', 'm', 's', 'd']):
                                    logger.debug(f"Skipping conversion of time interval column: {col}")
                                    continue
                                else:
                                    # Try to convert to numeric if possible
                                    try:
                                        result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
                                        logger.debug(f"Converted {col} from string to numeric")
                                    except:
                                        logger.debug(f"Could not convert {col} to numeric, keeping as string")
                            else:
                                # Try general numeric conversion
                                try:
                                    result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
                                    logger.debug(f"Converted {col} to numeric")
                                except:
                                    logger.debug(f"Could not convert {col} to numeric")
                    except (IndexError, TypeError):
                        logger.debug(
                            f"Skipping conversion for column {col} - no valid data or conversion not applicable")

            # Calculate rolling metrics for baseline comparison
            logger.debug("Calculating z-scores")

            # Guard against division by zero or NaN values
            price_1d_std = result_df['price_change_1d'].rolling(window=365).std()
            price_7d_std = result_df['price_change_7d'].rolling(window=365).std()
            volatility_std = result_df['volatility_14d'].rolling(window=365).std()

            # Replace zeros with NaN to avoid division by zero
            price_1d_std = price_1d_std.replace(0, np.nan)
            price_7d_std = price_7d_std.replace(0, np.nan)
            volatility_std = volatility_std.replace(0, np.nan)

            result_df['price_change_1d_zscore'] = (
                    (result_df['price_change_1d'] - result_df['price_change_1d'].rolling(window=365).mean()) /
                    price_1d_std
            )
            result_df['price_change_7d_zscore'] = (
                    (result_df['price_change_7d'] - result_df['price_change_7d'].rolling(window=365).mean()) /
                    price_7d_std
            )
            result_df['volatility_14d_zscore'] = (
                    (result_df['volatility_14d'] - result_df['volatility_14d'].rolling(window=365).mean()) /
                    volatility_std
            )

            # Add cycle-specific features based on cycle_type
            if cycle_type == 'halving':
                logger.info("Processing halving cycle")
                try:
                    # Get halving cycle features
                    halving_df = self.btcycle.calculate_btc_halving_cycle_features(result_df)

                    # Convert any Decimal columns to float to avoid type errors
                    for col in halving_df.columns:
                        if halving_df[col].dtype == object:
                            try:
                                first_non_null = halving_df[col].dropna().iloc[0] if not halving_df[
                                    col].dropna().empty else None
                                if first_non_null is not None and isinstance(first_non_null, decimal.Decimal):
                                    logger.debug(f"Converting {col} from Decimal to float in halving_df")
                                    halving_df[col] = halving_df[col].astype(float)
                            except (IndexError, TypeError, ValueError):
                                logger.debug(f"Skipping conversion for column {col} in halving_df")

                    # Merge the relevant columns with result_df
                    for col in ['halving_cycle_phase', 'days_since_last_halving', 'days_to_next_halving',
                                'cycle_number']:
                        if col in halving_df.columns:
                            result_df[col] = halving_df[col]

                    # Group historical data by cycle number for comparison
                    if 'cycle_number' in result_df.columns and 'halving_cycle_phase' in result_df.columns:
                        cycles_data = {}
                        current_cycle = result_df['cycle_number'].max()
                        logger.debug(f"Current halving cycle: {current_cycle}")

                        for cycle in result_df['cycle_number'].unique():
                            if pd.notna(cycle) and cycle < current_cycle:  # Historical cycles
                                cycle_data = result_df[result_df['cycle_number'] == cycle]
                                cycles_data[cycle] = cycle_data
                                logger.debug(f"Found historical cycle {cycle} with {len(cycle_data)} data points")

                        # Get current cycle data
                        current_cycle_data = result_df[result_df['cycle_number'] == current_cycle]
                        logger.debug(f"Current cycle has {len(current_cycle_data)} data points")

                        # For each point in the current cycle, compare with historical cycles at the same phase
                        for idx, row in current_cycle_data.iterrows():
                            if pd.isna(row['halving_cycle_phase']):
                                continue

                            current_phase = float(row['halving_cycle_phase'])  # Ensure float type
                            phase_margin = 0.05  # 5% phase window for comparison

                            # Collect historical prices at similar cycle phases
                            historical_values = []

                            for cycle, cycle_data in cycles_data.items():
                                similar_phase_data = cycle_data[
                                    (cycle_data['halving_cycle_phase'] >= current_phase - phase_margin) &
                                    (cycle_data['halving_cycle_phase'] <= current_phase + phase_margin)
                                    ]

                                if not similar_phase_data.empty:
                                    # Use normalized price change from cycle start
                                    if 'price_change_since_halving' in similar_phase_data.columns:
                                        # Convert to float to ensure consistent types
                                        values = similar_phase_data['price_change_since_halving'].astype(float).tolist()
                                        historical_values.extend(values)

                            # If we have enough historical data points for comparison
                            if len(historical_values) >= 3:
                                historical_mean = np.mean(historical_values)
                                historical_std = np.std(historical_values)

                                # Calculate z-score compared to historical cycles
                                if 'price_change_since_halving' in current_cycle_data.columns and historical_std > 0:
                                    current_value = float(row.get('price_change_since_halving', 0))  # Ensure float
                                    z_score = (current_value - historical_mean) / historical_std

                                    # Check for significant deviation
                                    if abs(z_score) > 2.0:  # More than 2 standard deviations
                                        anomalies.loc[idx, 'anomaly_detected'] = True
                                        anomalies.loc[idx, 'significance_score'] = float(abs(z_score))  # Ensure float

                                        if z_score > 0:
                                            anomalies.loc[idx, 'anomaly_type'] = 'higher_than_historical'
                                            anomalies.loc[idx, 'description'] = (
                                                f"Price is {z_score:.2f} std devs higher than historical cycles "
                                                f"at similar phase ({current_phase:.2f})"
                                            )
                                            logger.info(
                                                f"Detected 'higher_than_historical' anomaly on {idx.date()} with score {z_score:.2f}")
                                        else:
                                            anomalies.loc[idx, 'anomaly_type'] = 'lower_than_historical'
                                            anomalies.loc[idx, 'description'] = (
                                                f"Price is {abs(z_score):.2f} std devs lower than historical cycles "
                                                f"at similar phase ({current_phase:.2f})"
                                            )
                                            logger.info(
                                                f"Detected 'lower_than_historical' anomaly on {idx.date()} with score {abs(z_score):.2f}")
                except Exception as e:
                    logger.error(f"Error in halving cycle analysis: {str(e)}")
                    logger.debug("Traceback", exc_info=True)

            elif cycle_type == 'bull_bear':
                logger.info("Processing bull/bear cycle")
                try:
                    # Get bull/bear cycle information
                    bull_bear_df = self.marketplace.identify_bull_bear_cycles(result_df)

                    # Convert any Decimal columns to float to avoid type errors
                    for col in bull_bear_df.columns:
                        if bull_bear_df[col].dtype == object:
                            try:
                                first_non_null = bull_bear_df[col].dropna().iloc[0] if not bull_bear_df[
                                    col].dropna().empty else None
                                if first_non_null is not None and isinstance(first_non_null, decimal.Decimal):
                                    logger.debug(f"Converting {col} from Decimal to float in bull_bear_df")
                                    bull_bear_df[col] = bull_bear_df[col].astype(float)
                            except (IndexError, TypeError, ValueError):
                                logger.debug(f"Skipping conversion for column {col} in bull_bear_df")

                    # Merge the relevant columns with result_df
                    for col in ['cycle_state', 'cycle_id', 'days_in_cycle', 'cycle_max_roi', 'cycle_max_drawdown']:
                        if col in bull_bear_df.columns:
                            result_df[col] = bull_bear_df[col]

                    # Check for anomalies in ROI or drawdown compared to typical bull/bear cycles
                    if 'cycle_state' in result_df.columns and 'cycle_id' in result_df.columns:
                        # Group by cycle_state to get typical metrics for bull and bear markets
                        if hasattr(bull_bear_df, 'cycles_summary') and not bull_bear_df.cycles_summary.empty:
                            cycles_summary = bull_bear_df.cycles_summary

                            # Convert any Decimal columns to float in cycles_summary
                            for col in cycles_summary.columns:
                                if cycles_summary[col].dtype == object:
                                    try:
                                        first_non_null = cycles_summary[col].dropna().iloc[0] if not cycles_summary[
                                            col].dropna().empty else None
                                        if first_non_null is not None and isinstance(first_non_null, decimal.Decimal):
                                            logger.debug(f"Converting {col} from Decimal to float in cycles_summary")
                                            cycles_summary[col] = cycles_summary[col].astype(float)
                                    except (IndexError, TypeError, ValueError):
                                        logger.debug(f"Skipping conversion for column {col} in cycles_summary")

                            # Get metrics by cycle type
                            bull_cycles = cycles_summary[cycles_summary['cycle_type'] == 'bull']
                            bear_cycles = cycles_summary[cycles_summary['cycle_type'] == 'bear']

                            logger.debug(f"Found {len(bull_cycles)} bull cycles and {len(bear_cycles)} bear cycles")

                            # Calculate average duration and ROI metrics
                            avg_bull_duration = None
                            avg_bull_roi = None
                            std_bull_roi = None

                            avg_bear_duration = None
                            avg_bear_drawdown = None
                            std_bear_drawdown = None

                            if not bull_cycles.empty:
                                avg_bull_duration = float(bull_cycles['duration_days'].mean())
                                avg_bull_roi = float(bull_cycles['max_roi'].mean())
                                std_bull_roi = float(bull_cycles['max_roi'].std())
                                logger.debug(
                                    f"Bull cycle stats - Avg duration: {avg_bull_duration}, Avg ROI: {avg_bull_roi}, Std ROI: {std_bull_roi}")

                            if not bear_cycles.empty:
                                avg_bear_duration = float(bear_cycles['duration_days'].mean())
                                avg_bear_drawdown = float(bear_cycles['max_drawdown'].mean())
                                std_bear_drawdown = float(bear_cycles['max_drawdown'].std())
                                logger.debug(
                                    f"Bear cycle stats - Avg duration: {avg_bear_duration}, Avg drawdown: {avg_bear_drawdown}, Std drawdown: {std_bear_drawdown}")

                            # Identify current cycle
                            current_cycle_id = result_df['cycle_id'].max()
                            current_cycle_data = result_df[result_df['cycle_id'] == current_cycle_id]

                            if not current_cycle_data.empty:
                                current_state = current_cycle_data['cycle_state'].iloc[0]
                                current_duration = float(current_cycle_data['days_in_cycle'].max())
                                logger.debug(
                                    f"Current cycle: ID {current_cycle_id}, State {current_state}, Duration {current_duration} days")

                                # Check for duration anomalies
                                if current_state == 'bull' and avg_bull_duration is not None:
                                    if current_duration > 1.5 * avg_bull_duration:
                                        # Extended bull market
                                        for idx in current_cycle_data.index[-30:]:  # Last 30 days
                                            ratio = current_duration / avg_bull_duration
                                            anomalies.loc[idx, 'anomaly_detected'] = True
                                            anomalies.loc[idx, 'anomaly_type'] = 'extended_bull_market'
                                            anomalies.loc[idx, 'significance_score'] = float(ratio)  # Ensure float
                                            anomalies.loc[idx, 'description'] = (
                                                f"Extended bull market: {current_duration:.0f} days vs. "
                                                f"typical {avg_bull_duration:.0f} days"
                                            )
                                        logger.info(
                                            f"Detected 'extended_bull_market' anomaly with ratio {current_duration / avg_bull_duration:.2f}")

                                elif current_state == 'bear' and avg_bear_duration is not None:
                                    if current_duration > 1.5 * avg_bear_duration:
                                        # Extended bear market
                                        for idx in current_cycle_data.index[-30:]:  # Last 30 days
                                            ratio = current_duration / avg_bear_duration
                                            anomalies.loc[idx, 'anomaly_detected'] = True
                                            anomalies.loc[idx, 'anomaly_type'] = 'extended_bear_market'
                                            anomalies.loc[idx, 'significance_score'] = float(ratio)  # Ensure float
                                            anomalies.loc[idx, 'description'] = (
                                                f"Extended bear market: {current_duration:.0f} days vs. "
                                                f"typical {avg_bear_duration:.0f} days"
                                            )
                                        logger.info(
                                            f"Detected 'extended_bear_market' anomaly with ratio {current_duration / avg_bear_duration:.2f}")

                                # Check for ROI/drawdown anomalies
                                if current_state == 'bull' and std_bull_roi is not None and std_bull_roi > 0:
                                    current_roi = float(current_cycle_data['cycle_max_roi'].max())
                                    roi_z_score = (current_roi - avg_bull_roi) / std_bull_roi

                                    if abs(roi_z_score) > 2.0:
                                        anomaly_type = 'stronger_bull' if roi_z_score > 0 else 'weaker_bull'
                                        for idx in current_cycle_data.index[-15:]:  # Last 15 days
                                            anomalies.loc[idx, 'anomaly_detected'] = True
                                            anomalies.loc[idx, 'anomaly_type'] = anomaly_type
                                            anomalies.loc[idx, 'significance_score'] = float(
                                                abs(roi_z_score))  # Ensure float
                                            anomalies.loc[idx, 'description'] = (
                                                f"{'Stronger' if roi_z_score > 0 else 'Weaker'} than typical bull market: "
                                                f"{current_roi:.1%} ROI vs. typical {avg_bull_roi:.1%}"
                                            )
                                        logger.info(
                                            f"Detected '{anomaly_type}' anomaly with z-score {abs(roi_z_score):.2f}")

                                elif current_state == 'bear' and std_bear_drawdown is not None and std_bear_drawdown > 0:
                                    current_drawdown = float(current_cycle_data['cycle_max_drawdown'].min())
                                    drawdown_z_score = (current_drawdown - avg_bear_drawdown) / std_bear_drawdown

                                    if abs(drawdown_z_score) > 2.0:
                                        anomaly_type = 'milder_bear' if drawdown_z_score > 0 else 'severe_bear'
                                        for idx in current_cycle_data.index[-15:]:  # Last 15 days
                                            anomalies.loc[idx, 'anomaly_detected'] = True
                                            anomalies.loc[idx, 'anomaly_type'] = anomaly_type
                                            anomalies.loc[idx, 'significance_score'] = float(
                                                abs(drawdown_z_score))  # Ensure float
                                            anomalies.loc[idx, 'description'] = (
                                                f"{'Milder' if drawdown_z_score > 0 else 'More severe'} than typical bear market: "
                                                f"{current_drawdown:.1%} drawdown vs. typical {avg_bear_drawdown:.1%}"
                                            )
                                        logger.info(
                                            f"Detected '{anomaly_type}' anomaly with z-score {abs(drawdown_z_score):.2f}")
                except Exception as e:
                    logger.error(f"Error in bull/bear cycle analysis: {str(e)}")
                    logger.debug("Traceback", exc_info=True)

            elif cycle_type == 'network_upgrade' or cycle_type == 'ecosystem_event':
                logger.info(f"Processing {cycle_type} cycle for {clean_symbol}")
                try:
                    # For ETH or SOL, detect anomalies around significant events
                    events = self.get_significant_events_for_symbol(clean_symbol)
                    logger.debug(f"Found {len(events) if events else 0} significant events for {clean_symbol}")

                    # If we have events data
                    if events:
                        # Analyze volatility and price changes around events
                        for i, event in enumerate(events):
                            # Skip if event data is not a dictionary (for BTC it's just dates)
                            if not isinstance(event, dict):
                                logger.debug(f"Skipping event {i} - not a dictionary")
                                continue

                            event_date = pd.to_datetime(event['date'])
                            event_name = event['name']
                            logger.debug(f"Processing event: {event_name} on {event_date}")

                            # Skip events that are too recent or future events
                            if event_date > result_df.index.max():
                                logger.debug(f"Skipping future event: {event_name} on {event_date}")
                                continue

                            # Define pre and post event windows
                            pre_event_window = 7  # 7 days before
                            post_event_window = 30  # 30 days after

                            # Get data around the event
                            pre_event_mask = (result_df.index >= event_date - pd.Timedelta(days=pre_event_window)) & (
                                    result_df.index < event_date)
                            post_event_mask = (result_df.index >= event_date) & (
                                    result_df.index < event_date + pd.Timedelta(days=post_event_window))

                            pre_event_data = result_df[pre_event_mask]
                            post_event_data = result_df[post_event_mask]

                            # Skip if not enough data
                            if len(pre_event_data) < 3 or len(post_event_data) < 3:
                                logger.debug(
                                    f"Skipping event {event_name} - insufficient data (pre: {len(pre_event_data)}, post: {len(post_event_data)})")
                                continue

                            # Calculate metrics
                            pre_event_volatility = float(pre_event_data[
                                                             'volatility_14d'].mean()) if 'volatility_14d' in pre_event_data.columns else 0
                            post_event_volatility = float(post_event_data[
                                                              'volatility_14d'].mean()) if 'volatility_14d' in post_event_data.columns else 0

                            logger.debug(
                                f"Event {event_name} - pre volatility: {pre_event_volatility:.6f}, post volatility: {post_event_volatility:.6f}")

                            # Price change from event day
                            if event_date in result_df.index:
                                event_price = float(result_df.loc[event_date, 'close'])
                                post_event_data['price_change_from_event'] = post_event_data['close'].astype(
                                    float) / event_price - 1

                                # Check for significant price changes after the event
                                for idx, row in post_event_data.iterrows():
                                    days_after_event = (idx - event_date).days
                                    price_change = float(row['price_change_from_event'])

                                    # If price change is >10% within 7 days or >20% within 30 days of the event
                                    if (days_after_event <= 7 and abs(price_change) > 0.1) or \
                                            (days_after_event > 7 and abs(price_change) > 0.2):
                                        anomalies.loc[idx, 'anomaly_detected'] = True
                                        anomalies.loc[idx, 'anomaly_type'] = 'significant_post_event_move'
                                        anomalies.loc[idx, 'significance_score'] = float(
                                            abs(price_change) * 5)  # Scale for comparison
                                        anomalies.loc[idx, 'description'] = (
                                            f"Significant price change of {price_change:.1%} "
                                            f"{days_after_event} days after {event_name}"
                                        )
                                        logger.info(
                                            f"Detected 'significant_post_event_move' after {event_name} - {price_change:.1%} change after {days_after_event} days")

                            # Check for volatility anomalies
                            if pre_event_volatility > 0 and post_event_volatility > 1.5 * pre_event_volatility:
                                # Increased volatility after event
                                for idx in post_event_data.index[:10]:  # First 10 days after event
                                    vol_ratio = post_event_volatility / pre_event_volatility
                                    anomalies.loc[idx, 'anomaly_detected'] = True
                                    anomalies.loc[idx, 'anomaly_type'] = 'increased_post_event_volatility'
                                    anomalies.loc[idx, 'significance_score'] = float(vol_ratio)  # Ensure float
                                    anomalies.loc[idx, 'description'] = (
                                        f"Increased volatility after {event_name}: "
                                        f"{post_event_volatility:.4f} vs pre-event {pre_event_volatility:.4f}"
                                    )
                                logger.info(
                                    f"Detected 'increased_post_event_volatility' after {event_name} - ratio: {vol_ratio:.2f}")
                except Exception as e:
                    logger.error(f"Error in {cycle_type} analysis: {str(e)}")
                    logger.debug("Traceback", exc_info=True)

            # General price anomalies (applicable to all cycle types)
            logger.info("Processing general price anomalies")
            anomaly_count = 0
            for idx, row in result_df.iterrows():
                # Skip rows with insufficient data for z-scores
                if pd.isna(row.get('price_change_1d_zscore')) or pd.isna(row.get('volatility_14d_zscore')):
                    continue

                # 1. Check for extreme daily price changes
                if abs(row['price_change_1d_zscore']) > 3.0:  # More than 3 standard deviations
                    zscore = float(row['price_change_1d_zscore'])
                    price_change = float(row['price_change_1d'])
                    anomalies.loc[idx, 'anomaly_detected'] = True
                    anomalies.loc[idx, 'anomaly_type'] = 'extreme_daily_move'
                    anomalies.loc[idx, 'significance_score'] = abs(zscore)
                    direction = "up" if price_change > 0 else "down"
                    anomalies.loc[idx, 'description'] = (
                        f"Extreme daily price move {direction} ({price_change:.1%}), "
                        f"z-score: {zscore:.2f}"
                    )
                    anomaly_count += 1
                    if anomaly_count <= 5:  # Log only a few to avoid excessive logging
                        logger.info(
                            f"Detected 'extreme_daily_move' on {idx.date()} - {direction} {price_change:.1%}, z-score: {zscore:.2f}")

                # 2. Check for extreme volatility
                if row['volatility_14d_zscore'] > 3.0:  # More than 3 standard deviations
                    # Only mark as anomaly if not already detected for price change
                    if not anomalies.loc[idx, 'anomaly_detected']:
                        zscore = float(row['volatility_14d_zscore'])
                        volatility = float(row['volatility_14d'])
                        anomalies.loc[idx, 'anomaly_detected'] = True
                        anomalies.loc[idx, 'anomaly_type'] = 'extreme_volatility'
                        anomalies.loc[idx, 'significance_score'] = zscore
                        anomalies.loc[idx, 'description'] = (
                            f"Extreme volatility detected ({volatility:.4f}), "
                            f"z-score: {zscore:.2f}"
                        )
                        anomaly_count += 1
                        if anomaly_count <= 10:  # Log only a few to avoid excessive logging
                            logger.info(
                                f"Detected 'extreme_volatility' on {idx.date()} - {volatility:.4f}, z-score: {zscore:.2f}")

                # 3. Check for volatility collapse (can precede large moves)
                if row['volatility_14d_zscore'] < -2.0:  # More than 2 standard deviations below mean
                    # Only mark as anomaly if not already detected
                    if not anomalies.loc[idx, 'anomaly_detected']:
                        zscore = float(row['volatility_14d_zscore'])
                        volatility = float(row['volatility_14d'])
                        anomalies.loc[idx, 'anomaly_detected'] = True
                        anomalies.loc[idx, 'anomaly_type'] = 'volatility_collapse'
                        anomalies.loc[idx, 'significance_score'] = abs(zscore)
                        anomalies.loc[idx, 'description'] = (
                            f"Unusually low volatility ({volatility:.4f}), "
                            f"z-score: {zscore:.2f}"
                        )
                        anomaly_count += 1
                        if anomaly_count <= 10:  # Log only a few to avoid excessive logging
                            logger.info(
                                f"Detected 'volatility_collapse' on {idx.date()} - {volatility:.4f}, z-score: {zscore:.2f}")

            # Filter to only include detected anomalies
            anomalies = anomalies[anomalies['anomaly_detected']]

            # Sort by significance score
            anomalies = anomalies.sort_values('significance_score', ascending=False)

            logger.info(f"Anomaly detection complete. Found {len(anomalies)} anomalies for {symbol}.")
            return anomalies

        except Exception as e:
            logger.error(f"Error in detect_cycle_anomalies: {str(e)}")
            logger.debug("Detailed traceback:", exc_info=True)
            raise  # Re-raise the exception after logging